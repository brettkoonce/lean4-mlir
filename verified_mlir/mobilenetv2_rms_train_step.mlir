module @m {
  func.func @mobilenetv2_rms_train_step(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x3x3xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4pW: tensor<32x144x1x1xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<192x32x1x1xf32>, %b5eg: tensor<192xf32>, %b5ebt: tensor<192xf32>, %b5dW: tensor<192x1x3x3xf32>, %b5dg: tensor<192xf32>, %b5dbt: tensor<192xf32>, %b5pW: tensor<32x192x1x1xf32>, %b5pg: tensor<32xf32>, %b5pbt: tensor<32xf32>, %b6eW: tensor<192x32x1x1xf32>, %b6eg: tensor<192xf32>, %b6ebt: tensor<192xf32>, %b6dW: tensor<192x1x3x3xf32>, %b6dg: tensor<192xf32>, %b6dbt: tensor<192xf32>, %b6pW: tensor<32x192x1x1xf32>, %b6pg: tensor<32xf32>, %b6pbt: tensor<32xf32>, %b7eW: tensor<192x32x1x1xf32>, %b7eg: tensor<192xf32>, %b7ebt: tensor<192xf32>, %b7dW: tensor<192x1x3x3xf32>, %b7dg: tensor<192xf32>, %b7dbt: tensor<192xf32>, %b7pW: tensor<64x192x1x1xf32>, %b7pg: tensor<64xf32>, %b7pbt: tensor<64xf32>, %b8eW: tensor<384x64x1x1xf32>, %b8eg: tensor<384xf32>, %b8ebt: tensor<384xf32>, %b8dW: tensor<384x1x3x3xf32>, %b8dg: tensor<384xf32>, %b8dbt: tensor<384xf32>, %b8pW: tensor<64x384x1x1xf32>, %b8pg: tensor<64xf32>, %b8pbt: tensor<64xf32>, %b9eW: tensor<384x64x1x1xf32>, %b9eg: tensor<384xf32>, %b9ebt: tensor<384xf32>, %b9dW: tensor<384x1x3x3xf32>, %b9dg: tensor<384xf32>, %b9dbt: tensor<384xf32>, %b9pW: tensor<64x384x1x1xf32>, %b9pg: tensor<64xf32>, %b9pbt: tensor<64xf32>, %b10eW: tensor<384x64x1x1xf32>, %b10eg: tensor<384xf32>, %b10ebt: tensor<384xf32>, %b10dW: tensor<384x1x3x3xf32>, %b10dg: tensor<384xf32>, %b10dbt: tensor<384xf32>, %b10pW: tensor<64x384x1x1xf32>, %b10pg: tensor<64xf32>, %b10pbt: tensor<64xf32>, %b11eW: tensor<384x64x1x1xf32>, %b11eg: tensor<384xf32>, %b11ebt: tensor<384xf32>, %b11dW: tensor<384x1x3x3xf32>, %b11dg: tensor<384xf32>, %b11dbt: tensor<384xf32>, %b11pW: tensor<96x384x1x1xf32>, %b11pg: tensor<96xf32>, %b11pbt: tensor<96xf32>, %b12eW: tensor<576x96x1x1xf32>, %b12eg: tensor<576xf32>, %b12ebt: tensor<576xf32>, %b12dW: tensor<576x1x3x3xf32>, %b12dg: tensor<576xf32>, %b12dbt: tensor<576xf32>, %b12pW: tensor<96x576x1x1xf32>, %b12pg: tensor<96xf32>, %b12pbt: tensor<96xf32>, %b13eW: tensor<576x96x1x1xf32>, %b13eg: tensor<576xf32>, %b13ebt: tensor<576xf32>, %b13dW: tensor<576x1x3x3xf32>, %b13dg: tensor<576xf32>, %b13dbt: tensor<576xf32>, %b13pW: tensor<96x576x1x1xf32>, %b13pg: tensor<96xf32>, %b13pbt: tensor<96xf32>, %b14eW: tensor<576x96x1x1xf32>, %b14eg: tensor<576xf32>, %b14ebt: tensor<576xf32>, %b14dW: tensor<576x1x3x3xf32>, %b14dg: tensor<576xf32>, %b14dbt: tensor<576xf32>, %b14pW: tensor<160x576x1x1xf32>, %b14pg: tensor<160xf32>, %b14pbt: tensor<160xf32>, %b15eW: tensor<960x160x1x1xf32>, %b15eg: tensor<960xf32>, %b15ebt: tensor<960xf32>, %b15dW: tensor<960x1x3x3xf32>, %b15dg: tensor<960xf32>, %b15dbt: tensor<960xf32>, %b15pW: tensor<160x960x1x1xf32>, %b15pg: tensor<160xf32>, %b15pbt: tensor<160xf32>, %b16eW: tensor<960x160x1x1xf32>, %b16eg: tensor<960xf32>, %b16ebt: tensor<960xf32>, %b16dW: tensor<960x1x3x3xf32>, %b16dg: tensor<960xf32>, %b16dbt: tensor<960xf32>, %b16pW: tensor<160x960x1x1xf32>, %b16pg: tensor<160xf32>, %b16pbt: tensor<160xf32>, %b17eW: tensor<960x160x1x1xf32>, %b17eg: tensor<960xf32>, %b17ebt: tensor<960xf32>, %b17dW: tensor<960x1x3x3xf32>, %b17dg: tensor<960xf32>, %b17dbt: tensor<960xf32>, %b17pW: tensor<320x960x1x1xf32>, %b17pg: tensor<320xf32>, %b17pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<32x3x3x3xf32>, %sgm: tensor<32xf32>, %sbtm: tensor<32xf32>, %b1dWm: tensor<32x1x3x3xf32>, %b1dgm: tensor<32xf32>, %b1dbtm: tensor<32xf32>, %b1pWm: tensor<16x32x1x1xf32>, %b1pgm: tensor<16xf32>, %b1pbtm: tensor<16xf32>, %b2eWm: tensor<96x16x1x1xf32>, %b2egm: tensor<96xf32>, %b2ebtm: tensor<96xf32>, %b2dWm: tensor<96x1x3x3xf32>, %b2dgm: tensor<96xf32>, %b2dbtm: tensor<96xf32>, %b2pWm: tensor<24x96x1x1xf32>, %b2pgm: tensor<24xf32>, %b2pbtm: tensor<24xf32>, %b3eWm: tensor<144x24x1x1xf32>, %b3egm: tensor<144xf32>, %b3ebtm: tensor<144xf32>, %b3dWm: tensor<144x1x3x3xf32>, %b3dgm: tensor<144xf32>, %b3dbtm: tensor<144xf32>, %b3pWm: tensor<24x144x1x1xf32>, %b3pgm: tensor<24xf32>, %b3pbtm: tensor<24xf32>, %b4eWm: tensor<144x24x1x1xf32>, %b4egm: tensor<144xf32>, %b4ebtm: tensor<144xf32>, %b4dWm: tensor<144x1x3x3xf32>, %b4dgm: tensor<144xf32>, %b4dbtm: tensor<144xf32>, %b4pWm: tensor<32x144x1x1xf32>, %b4pgm: tensor<32xf32>, %b4pbtm: tensor<32xf32>, %b5eWm: tensor<192x32x1x1xf32>, %b5egm: tensor<192xf32>, %b5ebtm: tensor<192xf32>, %b5dWm: tensor<192x1x3x3xf32>, %b5dgm: tensor<192xf32>, %b5dbtm: tensor<192xf32>, %b5pWm: tensor<32x192x1x1xf32>, %b5pgm: tensor<32xf32>, %b5pbtm: tensor<32xf32>, %b6eWm: tensor<192x32x1x1xf32>, %b6egm: tensor<192xf32>, %b6ebtm: tensor<192xf32>, %b6dWm: tensor<192x1x3x3xf32>, %b6dgm: tensor<192xf32>, %b6dbtm: tensor<192xf32>, %b6pWm: tensor<32x192x1x1xf32>, %b6pgm: tensor<32xf32>, %b6pbtm: tensor<32xf32>, %b7eWm: tensor<192x32x1x1xf32>, %b7egm: tensor<192xf32>, %b7ebtm: tensor<192xf32>, %b7dWm: tensor<192x1x3x3xf32>, %b7dgm: tensor<192xf32>, %b7dbtm: tensor<192xf32>, %b7pWm: tensor<64x192x1x1xf32>, %b7pgm: tensor<64xf32>, %b7pbtm: tensor<64xf32>, %b8eWm: tensor<384x64x1x1xf32>, %b8egm: tensor<384xf32>, %b8ebtm: tensor<384xf32>, %b8dWm: tensor<384x1x3x3xf32>, %b8dgm: tensor<384xf32>, %b8dbtm: tensor<384xf32>, %b8pWm: tensor<64x384x1x1xf32>, %b8pgm: tensor<64xf32>, %b8pbtm: tensor<64xf32>, %b9eWm: tensor<384x64x1x1xf32>, %b9egm: tensor<384xf32>, %b9ebtm: tensor<384xf32>, %b9dWm: tensor<384x1x3x3xf32>, %b9dgm: tensor<384xf32>, %b9dbtm: tensor<384xf32>, %b9pWm: tensor<64x384x1x1xf32>, %b9pgm: tensor<64xf32>, %b9pbtm: tensor<64xf32>, %b10eWm: tensor<384x64x1x1xf32>, %b10egm: tensor<384xf32>, %b10ebtm: tensor<384xf32>, %b10dWm: tensor<384x1x3x3xf32>, %b10dgm: tensor<384xf32>, %b10dbtm: tensor<384xf32>, %b10pWm: tensor<64x384x1x1xf32>, %b10pgm: tensor<64xf32>, %b10pbtm: tensor<64xf32>, %b11eWm: tensor<384x64x1x1xf32>, %b11egm: tensor<384xf32>, %b11ebtm: tensor<384xf32>, %b11dWm: tensor<384x1x3x3xf32>, %b11dgm: tensor<384xf32>, %b11dbtm: tensor<384xf32>, %b11pWm: tensor<96x384x1x1xf32>, %b11pgm: tensor<96xf32>, %b11pbtm: tensor<96xf32>, %b12eWm: tensor<576x96x1x1xf32>, %b12egm: tensor<576xf32>, %b12ebtm: tensor<576xf32>, %b12dWm: tensor<576x1x3x3xf32>, %b12dgm: tensor<576xf32>, %b12dbtm: tensor<576xf32>, %b12pWm: tensor<96x576x1x1xf32>, %b12pgm: tensor<96xf32>, %b12pbtm: tensor<96xf32>, %b13eWm: tensor<576x96x1x1xf32>, %b13egm: tensor<576xf32>, %b13ebtm: tensor<576xf32>, %b13dWm: tensor<576x1x3x3xf32>, %b13dgm: tensor<576xf32>, %b13dbtm: tensor<576xf32>, %b13pWm: tensor<96x576x1x1xf32>, %b13pgm: tensor<96xf32>, %b13pbtm: tensor<96xf32>, %b14eWm: tensor<576x96x1x1xf32>, %b14egm: tensor<576xf32>, %b14ebtm: tensor<576xf32>, %b14dWm: tensor<576x1x3x3xf32>, %b14dgm: tensor<576xf32>, %b14dbtm: tensor<576xf32>, %b14pWm: tensor<160x576x1x1xf32>, %b14pgm: tensor<160xf32>, %b14pbtm: tensor<160xf32>, %b15eWm: tensor<960x160x1x1xf32>, %b15egm: tensor<960xf32>, %b15ebtm: tensor<960xf32>, %b15dWm: tensor<960x1x3x3xf32>, %b15dgm: tensor<960xf32>, %b15dbtm: tensor<960xf32>, %b15pWm: tensor<160x960x1x1xf32>, %b15pgm: tensor<160xf32>, %b15pbtm: tensor<160xf32>, %b16eWm: tensor<960x160x1x1xf32>, %b16egm: tensor<960xf32>, %b16ebtm: tensor<960xf32>, %b16dWm: tensor<960x1x3x3xf32>, %b16dgm: tensor<960xf32>, %b16dbtm: tensor<960xf32>, %b16pWm: tensor<160x960x1x1xf32>, %b16pgm: tensor<160xf32>, %b16pbtm: tensor<160xf32>, %b17eWm: tensor<960x160x1x1xf32>, %b17egm: tensor<960xf32>, %b17ebtm: tensor<960xf32>, %b17dWm: tensor<960x1x3x3xf32>, %b17dgm: tensor<960xf32>, %b17dbtm: tensor<960xf32>, %b17pWm: tensor<320x960x1x1xf32>, %b17pgm: tensor<320xf32>, %b17pbtm: tensor<320xf32>, %hWm: tensor<1280x320x1x1xf32>, %hgm: tensor<1280xf32>, %hbtm: tensor<1280xf32>, %Wdm: tensor<1280x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<32x3x3x3xf32>, %sgv: tensor<32xf32>, %sbtv: tensor<32xf32>, %b1dWv: tensor<32x1x3x3xf32>, %b1dgv: tensor<32xf32>, %b1dbtv: tensor<32xf32>, %b1pWv: tensor<16x32x1x1xf32>, %b1pgv: tensor<16xf32>, %b1pbtv: tensor<16xf32>, %b2eWv: tensor<96x16x1x1xf32>, %b2egv: tensor<96xf32>, %b2ebtv: tensor<96xf32>, %b2dWv: tensor<96x1x3x3xf32>, %b2dgv: tensor<96xf32>, %b2dbtv: tensor<96xf32>, %b2pWv: tensor<24x96x1x1xf32>, %b2pgv: tensor<24xf32>, %b2pbtv: tensor<24xf32>, %b3eWv: tensor<144x24x1x1xf32>, %b3egv: tensor<144xf32>, %b3ebtv: tensor<144xf32>, %b3dWv: tensor<144x1x3x3xf32>, %b3dgv: tensor<144xf32>, %b3dbtv: tensor<144xf32>, %b3pWv: tensor<24x144x1x1xf32>, %b3pgv: tensor<24xf32>, %b3pbtv: tensor<24xf32>, %b4eWv: tensor<144x24x1x1xf32>, %b4egv: tensor<144xf32>, %b4ebtv: tensor<144xf32>, %b4dWv: tensor<144x1x3x3xf32>, %b4dgv: tensor<144xf32>, %b4dbtv: tensor<144xf32>, %b4pWv: tensor<32x144x1x1xf32>, %b4pgv: tensor<32xf32>, %b4pbtv: tensor<32xf32>, %b5eWv: tensor<192x32x1x1xf32>, %b5egv: tensor<192xf32>, %b5ebtv: tensor<192xf32>, %b5dWv: tensor<192x1x3x3xf32>, %b5dgv: tensor<192xf32>, %b5dbtv: tensor<192xf32>, %b5pWv: tensor<32x192x1x1xf32>, %b5pgv: tensor<32xf32>, %b5pbtv: tensor<32xf32>, %b6eWv: tensor<192x32x1x1xf32>, %b6egv: tensor<192xf32>, %b6ebtv: tensor<192xf32>, %b6dWv: tensor<192x1x3x3xf32>, %b6dgv: tensor<192xf32>, %b6dbtv: tensor<192xf32>, %b6pWv: tensor<32x192x1x1xf32>, %b6pgv: tensor<32xf32>, %b6pbtv: tensor<32xf32>, %b7eWv: tensor<192x32x1x1xf32>, %b7egv: tensor<192xf32>, %b7ebtv: tensor<192xf32>, %b7dWv: tensor<192x1x3x3xf32>, %b7dgv: tensor<192xf32>, %b7dbtv: tensor<192xf32>, %b7pWv: tensor<64x192x1x1xf32>, %b7pgv: tensor<64xf32>, %b7pbtv: tensor<64xf32>, %b8eWv: tensor<384x64x1x1xf32>, %b8egv: tensor<384xf32>, %b8ebtv: tensor<384xf32>, %b8dWv: tensor<384x1x3x3xf32>, %b8dgv: tensor<384xf32>, %b8dbtv: tensor<384xf32>, %b8pWv: tensor<64x384x1x1xf32>, %b8pgv: tensor<64xf32>, %b8pbtv: tensor<64xf32>, %b9eWv: tensor<384x64x1x1xf32>, %b9egv: tensor<384xf32>, %b9ebtv: tensor<384xf32>, %b9dWv: tensor<384x1x3x3xf32>, %b9dgv: tensor<384xf32>, %b9dbtv: tensor<384xf32>, %b9pWv: tensor<64x384x1x1xf32>, %b9pgv: tensor<64xf32>, %b9pbtv: tensor<64xf32>, %b10eWv: tensor<384x64x1x1xf32>, %b10egv: tensor<384xf32>, %b10ebtv: tensor<384xf32>, %b10dWv: tensor<384x1x3x3xf32>, %b10dgv: tensor<384xf32>, %b10dbtv: tensor<384xf32>, %b10pWv: tensor<64x384x1x1xf32>, %b10pgv: tensor<64xf32>, %b10pbtv: tensor<64xf32>, %b11eWv: tensor<384x64x1x1xf32>, %b11egv: tensor<384xf32>, %b11ebtv: tensor<384xf32>, %b11dWv: tensor<384x1x3x3xf32>, %b11dgv: tensor<384xf32>, %b11dbtv: tensor<384xf32>, %b11pWv: tensor<96x384x1x1xf32>, %b11pgv: tensor<96xf32>, %b11pbtv: tensor<96xf32>, %b12eWv: tensor<576x96x1x1xf32>, %b12egv: tensor<576xf32>, %b12ebtv: tensor<576xf32>, %b12dWv: tensor<576x1x3x3xf32>, %b12dgv: tensor<576xf32>, %b12dbtv: tensor<576xf32>, %b12pWv: tensor<96x576x1x1xf32>, %b12pgv: tensor<96xf32>, %b12pbtv: tensor<96xf32>, %b13eWv: tensor<576x96x1x1xf32>, %b13egv: tensor<576xf32>, %b13ebtv: tensor<576xf32>, %b13dWv: tensor<576x1x3x3xf32>, %b13dgv: tensor<576xf32>, %b13dbtv: tensor<576xf32>, %b13pWv: tensor<96x576x1x1xf32>, %b13pgv: tensor<96xf32>, %b13pbtv: tensor<96xf32>, %b14eWv: tensor<576x96x1x1xf32>, %b14egv: tensor<576xf32>, %b14ebtv: tensor<576xf32>, %b14dWv: tensor<576x1x3x3xf32>, %b14dgv: tensor<576xf32>, %b14dbtv: tensor<576xf32>, %b14pWv: tensor<160x576x1x1xf32>, %b14pgv: tensor<160xf32>, %b14pbtv: tensor<160xf32>, %b15eWv: tensor<960x160x1x1xf32>, %b15egv: tensor<960xf32>, %b15ebtv: tensor<960xf32>, %b15dWv: tensor<960x1x3x3xf32>, %b15dgv: tensor<960xf32>, %b15dbtv: tensor<960xf32>, %b15pWv: tensor<160x960x1x1xf32>, %b15pgv: tensor<160xf32>, %b15pbtv: tensor<160xf32>, %b16eWv: tensor<960x160x1x1xf32>, %b16egv: tensor<960xf32>, %b16ebtv: tensor<960xf32>, %b16dWv: tensor<960x1x3x3xf32>, %b16dgv: tensor<960xf32>, %b16dbtv: tensor<960xf32>, %b16pWv: tensor<160x960x1x1xf32>, %b16pgv: tensor<160xf32>, %b16pbtv: tensor<160xf32>, %b17eWv: tensor<960x160x1x1xf32>, %b17egv: tensor<960xf32>, %b17ebtv: tensor<960xf32>, %b17dWv: tensor<960x1x3x3xf32>, %b17dgv: tensor<960xf32>, %b17dbtv: tensor<960xf32>, %b17pWv: tensor<320x960x1x1xf32>, %b17pgv: tensor<320xf32>, %b17pbtv: tensor<320xf32>, %hWv: tensor<1280x320x1x1xf32>, %hgv: tensor<1280xf32>, %hbtv: tensor<1280xf32>, %Wdv: tensor<1280x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<32xf32>, %stnvari: tensor<32xf32>, %b1dnmui: tensor<32xf32>, %b1dnvari: tensor<32xf32>, %b1pnmui: tensor<16xf32>, %b1pnvari: tensor<16xf32>, %b2enmui: tensor<96xf32>, %b2envari: tensor<96xf32>, %b2dnmui: tensor<96xf32>, %b2dnvari: tensor<96xf32>, %b2pnmui: tensor<24xf32>, %b2pnvari: tensor<24xf32>, %b3enmui: tensor<144xf32>, %b3envari: tensor<144xf32>, %b3dnmui: tensor<144xf32>, %b3dnvari: tensor<144xf32>, %b3pnmui: tensor<24xf32>, %b3pnvari: tensor<24xf32>, %b4enmui: tensor<144xf32>, %b4envari: tensor<144xf32>, %b4dnmui: tensor<144xf32>, %b4dnvari: tensor<144xf32>, %b4pnmui: tensor<32xf32>, %b4pnvari: tensor<32xf32>, %b5enmui: tensor<192xf32>, %b5envari: tensor<192xf32>, %b5dnmui: tensor<192xf32>, %b5dnvari: tensor<192xf32>, %b5pnmui: tensor<32xf32>, %b5pnvari: tensor<32xf32>, %b6enmui: tensor<192xf32>, %b6envari: tensor<192xf32>, %b6dnmui: tensor<192xf32>, %b6dnvari: tensor<192xf32>, %b6pnmui: tensor<32xf32>, %b6pnvari: tensor<32xf32>, %b7enmui: tensor<192xf32>, %b7envari: tensor<192xf32>, %b7dnmui: tensor<192xf32>, %b7dnvari: tensor<192xf32>, %b7pnmui: tensor<64xf32>, %b7pnvari: tensor<64xf32>, %b8enmui: tensor<384xf32>, %b8envari: tensor<384xf32>, %b8dnmui: tensor<384xf32>, %b8dnvari: tensor<384xf32>, %b8pnmui: tensor<64xf32>, %b8pnvari: tensor<64xf32>, %b9enmui: tensor<384xf32>, %b9envari: tensor<384xf32>, %b9dnmui: tensor<384xf32>, %b9dnvari: tensor<384xf32>, %b9pnmui: tensor<64xf32>, %b9pnvari: tensor<64xf32>, %b10enmui: tensor<384xf32>, %b10envari: tensor<384xf32>, %b10dnmui: tensor<384xf32>, %b10dnvari: tensor<384xf32>, %b10pnmui: tensor<64xf32>, %b10pnvari: tensor<64xf32>, %b11enmui: tensor<384xf32>, %b11envari: tensor<384xf32>, %b11dnmui: tensor<384xf32>, %b11dnvari: tensor<384xf32>, %b11pnmui: tensor<96xf32>, %b11pnvari: tensor<96xf32>, %b12enmui: tensor<576xf32>, %b12envari: tensor<576xf32>, %b12dnmui: tensor<576xf32>, %b12dnvari: tensor<576xf32>, %b12pnmui: tensor<96xf32>, %b12pnvari: tensor<96xf32>, %b13enmui: tensor<576xf32>, %b13envari: tensor<576xf32>, %b13dnmui: tensor<576xf32>, %b13dnvari: tensor<576xf32>, %b13pnmui: tensor<96xf32>, %b13pnvari: tensor<96xf32>, %b14enmui: tensor<576xf32>, %b14envari: tensor<576xf32>, %b14dnmui: tensor<576xf32>, %b14dnvari: tensor<576xf32>, %b14pnmui: tensor<160xf32>, %b14pnvari: tensor<160xf32>, %b15enmui: tensor<960xf32>, %b15envari: tensor<960xf32>, %b15dnmui: tensor<960xf32>, %b15dnvari: tensor<960xf32>, %b15pnmui: tensor<160xf32>, %b15pnvari: tensor<160xf32>, %b16enmui: tensor<960xf32>, %b16envari: tensor<960xf32>, %b16dnmui: tensor<960xf32>, %b16dnvari: tensor<960xf32>, %b16pnmui: tensor<160xf32>, %b16pnvari: tensor<160xf32>, %b17enmui: tensor<960xf32>, %b17envari: tensor<960xf32>, %b17dnmui: tensor<960xf32>, %b17dnvari: tensor<960xf32>, %b17pnmui: tensor<320xf32>, %b17pnvari: tensor<320xf32>, %hnmui: tensor<1280xf32>, %hnvari: tensor<1280xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280xf32>, tensor<1280xf32>) {
    // ── OPTIMIZER: RMSProp + momentum, TENSORFLOW flavour (the MobileNetV2 reference's
    //    own: jax/MainMobilenetV2Imagenet.lean). Per parameter, in this order:
    //      g  <- g + wd*θ        COUPLED L2, BEFORE the accumulator  (momVNextF)
    //      s' <- ρ*s + (1-ρ)*g²                                      (adamVNextF at ρ)
    //      b' <- μ*b + g/sqrt(s' + ε)   ε INSIDE the sqrt            (rmsBufNextF)
    //      θ' <- θ - lr*b'                                           (sgdParamF)
    //    Packed [θ|m|v] is reused with m = momentum buffer, v = mean-square, so the
    //    interface is byte-identical to the AdamW render's apart from the entry name.
    //    %bc1/%bc2 are Adam bias corrections: unused here, passed through unchanged.
    //    The mean-square must be INITIALISED TO 1.0, not 0 — part of the recipe, not
    //    an implementation detail, since this optimizer is not bias-corrected.
    // ── MobileNetV2 batch-BN RMSProp train step: every op is pretty(verified AST node) except the constants, the input passthroughs and the marked report-only %loss ──
    // The conv biases are not in the signature (BN removes them; He et al.'s
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
    %v2346 = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2347 = stablehlo.convolution(%v2345, %v2346)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x7x7xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2350 = stablehlo.reshape %v1187 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2351 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2352 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2353 = stablehlo.compare GT, %v2350, %v2351 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2354 = stablehlo.compare LT, %v2350, %v2352 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2355 = stablehlo.and %v2353, %v2354 : tensor<32x576x14x14xi1>
    %v2356 = stablehlo.select %v2355, %v2349, %v2351 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2358 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2360 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2361 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2362 = stablehlo.reduce(%v2358 init: %v2359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2363 = stablehlo.broadcast_in_dim %v2362, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2364 = stablehlo.divide %v2363, %v2360 : tensor<32x576x14x14xf32>
    %v2365 = stablehlo.subtract %v2358, %v2364 : tensor<32x576x14x14xf32>
    %v2366 = stablehlo.multiply %v2365, %v2365 : tensor<32x576x14x14xf32>
    %v2367 = stablehlo.reduce(%v2366 init: %v2359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2368 = stablehlo.broadcast_in_dim %v2367, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2369 = stablehlo.divide %v2368, %v2360 : tensor<32x576x14x14xf32>
    %v2370 = stablehlo.add %v2369, %v2361 : tensor<32x576x14x14xf32>
    %v2371 = stablehlo.rsqrt %v2370 : tensor<32x576x14x14xf32>
    %v2372 = stablehlo.multiply %v2365, %v2371 : tensor<32x576x14x14xf32>
    %v2373 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2374 = stablehlo.reshape %v2357 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2375 = stablehlo.multiply %v2373, %v2374 : tensor<32x576x14x14xf32>
    %v2376 = stablehlo.reduce(%v2375 init: %v2359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2377 = stablehlo.broadcast_in_dim %v2376, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2378 = stablehlo.multiply %v2372, %v2375 : tensor<32x576x14x14xf32>
    %v2379 = stablehlo.reduce(%v2378 init: %v2359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2380 = stablehlo.broadcast_in_dim %v2379, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2381 = stablehlo.multiply %v2375, %v2360 : tensor<32x576x14x14xf32>
    %v2382 = stablehlo.subtract %v2381, %v2377 : tensor<32x576x14x14xf32>
    %v2383 = stablehlo.multiply %v2372, %v2380 : tensor<32x576x14x14xf32>
    %v2384 = stablehlo.subtract %v2382, %v2383 : tensor<32x576x14x14xf32>
    %v2385 = stablehlo.divide %v2371, %v2360 : tensor<32x576x14x14xf32>
    %v2386 = stablehlo.multiply %v2385, %v2384 : tensor<32x576x14x14xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2388 = stablehlo.reshape %v2387 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2389 = stablehlo.reverse %b14eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2390 = stablehlo.transpose %v2389, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2391 = stablehlo.convolution(%v2388, %v2390)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2393 = stablehlo.reshape %v1162 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2394 = stablehlo.reshape %v2387 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2395 = stablehlo.transpose %v2393, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2396 = stablehlo.transpose %v2394, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2397 = stablehlo.convolution(%v2395, %v2396)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2398 = stablehlo.transpose %v2397, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2399 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2401 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2402 = stablehlo.reduce(%v2399 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2403 = stablehlo.broadcast_in_dim %v2402, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2404 = stablehlo.divide %v2403, %v2401 : tensor<32x576x14x14xf32>
    %v2405 = stablehlo.subtract %v2399, %v2404 : tensor<32x576x14x14xf32>
    %v2406 = stablehlo.multiply %v2405, %v2405 : tensor<32x576x14x14xf32>
    %v2407 = stablehlo.reduce(%v2406 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2408 = stablehlo.broadcast_in_dim %v2407, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2409 = stablehlo.divide %v2408, %v2401 : tensor<32x576x14x14xf32>
    %v2410 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2411 = stablehlo.add %v2409, %v2410 : tensor<32x576x14x14xf32>
    %v2412 = stablehlo.rsqrt %v2411 : tensor<32x576x14x14xf32>
    %v2413 = stablehlo.multiply %v2405, %v2412 : tensor<32x576x14x14xf32>
    %v2414 = stablehlo.reshape %v2357 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2415 = stablehlo.multiply %v2414, %v2413 : tensor<32x576x14x14xf32>
    %v2416 = stablehlo.reduce(%v2415 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2417 = stablehlo.reshape %v2357 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2419 = stablehlo.reduce(%v2417 init: %v2418) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2420 = stablehlo.reshape %v1193 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2421 = stablehlo.reshape %v2344 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2423 = stablehlo.pad %v2421, %v2422, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2424 = stablehlo.transpose %v2420, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2425 = stablehlo.transpose %v2423, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2426 = stablehlo.convolution(%v2424, %v2425)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2428 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2430 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v2431 = stablehlo.reduce(%v2428 init: %v2429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2432 = stablehlo.broadcast_in_dim %v2431, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2433 = stablehlo.divide %v2432, %v2430 : tensor<32x576x7x7xf32>
    %v2434 = stablehlo.subtract %v2428, %v2433 : tensor<32x576x7x7xf32>
    %v2435 = stablehlo.multiply %v2434, %v2434 : tensor<32x576x7x7xf32>
    %v2436 = stablehlo.reduce(%v2435 init: %v2429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2437 = stablehlo.broadcast_in_dim %v2436, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2438 = stablehlo.divide %v2437, %v2430 : tensor<32x576x7x7xf32>
    %v2439 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2440 = stablehlo.add %v2438, %v2439 : tensor<32x576x7x7xf32>
    %v2441 = stablehlo.rsqrt %v2440 : tensor<32x576x7x7xf32>
    %v2442 = stablehlo.multiply %v2434, %v2441 : tensor<32x576x7x7xf32>
    %v2443 = stablehlo.reshape %v2314 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2444 = stablehlo.multiply %v2443, %v2442 : tensor<32x576x7x7xf32>
    %v2445 = stablehlo.reduce(%v2444 init: %v2429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2446 = stablehlo.reshape %v2314 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2448 = stablehlo.reduce(%v2446 init: %v2447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2449 = stablehlo.reshape %v1224 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2450 = stablehlo.reshape %v2300 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2451 = stablehlo.transpose %v2449, dims = [1, 0, 2, 3] : (tensor<32x576x7x7xf32>) -> tensor<576x32x7x7xf32>
    %v2452 = stablehlo.transpose %v2450, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2453 = stablehlo.convolution(%v2451, %v2452)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<576x160x1x1xf32>
    %v2454 = stablehlo.transpose %v2453, dims = [1, 0, 2, 3] : (tensor<576x160x1x1xf32>) -> tensor<160x576x1x1xf32>
    %v2455 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2457 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2458 = stablehlo.reduce(%v2455 init: %v2456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2459 = stablehlo.broadcast_in_dim %v2458, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2460 = stablehlo.divide %v2459, %v2457 : tensor<32x160x7x7xf32>
    %v2461 = stablehlo.subtract %v2455, %v2460 : tensor<32x160x7x7xf32>
    %v2462 = stablehlo.multiply %v2461, %v2461 : tensor<32x160x7x7xf32>
    %v2463 = stablehlo.reduce(%v2462 init: %v2456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2464 = stablehlo.broadcast_in_dim %v2463, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2465 = stablehlo.divide %v2464, %v2457 : tensor<32x160x7x7xf32>
    %v2466 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2467 = stablehlo.add %v2465, %v2466 : tensor<32x160x7x7xf32>
    %v2468 = stablehlo.rsqrt %v2467 : tensor<32x160x7x7xf32>
    %v2469 = stablehlo.multiply %v2461, %v2468 : tensor<32x160x7x7xf32>
    %v2470 = stablehlo.reshape %v2189 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2471 = stablehlo.multiply %v2470, %v2469 : tensor<32x160x7x7xf32>
    %v2472 = stablehlo.reduce(%v2471 init: %v2456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2473 = stablehlo.reshape %v2189 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2475 = stablehlo.reduce(%v2473 init: %v2474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2476 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2478 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2479 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2480 = stablehlo.reduce(%v2476 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2481 = stablehlo.broadcast_in_dim %v2480, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2482 = stablehlo.divide %v2481, %v2478 : tensor<32x96x14x14xf32>
    %v2483 = stablehlo.subtract %v2476, %v2482 : tensor<32x96x14x14xf32>
    %v2484 = stablehlo.multiply %v2483, %v2483 : tensor<32x96x14x14xf32>
    %v2485 = stablehlo.reduce(%v2484 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2486 = stablehlo.broadcast_in_dim %v2485, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2487 = stablehlo.divide %v2486, %v2478 : tensor<32x96x14x14xf32>
    %v2488 = stablehlo.add %v2487, %v2479 : tensor<32x96x14x14xf32>
    %v2489 = stablehlo.rsqrt %v2488 : tensor<32x96x14x14xf32>
    %v2490 = stablehlo.multiply %v2483, %v2489 : tensor<32x96x14x14xf32>
    %v2491 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2492 = stablehlo.reshape %v2392 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2493 = stablehlo.multiply %v2491, %v2492 : tensor<32x96x14x14xf32>
    %v2494 = stablehlo.reduce(%v2493 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2495 = stablehlo.broadcast_in_dim %v2494, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2496 = stablehlo.multiply %v2490, %v2493 : tensor<32x96x14x14xf32>
    %v2497 = stablehlo.reduce(%v2496 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2498 = stablehlo.broadcast_in_dim %v2497, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2499 = stablehlo.multiply %v2493, %v2478 : tensor<32x96x14x14xf32>
    %v2500 = stablehlo.subtract %v2499, %v2495 : tensor<32x96x14x14xf32>
    %v2501 = stablehlo.multiply %v2490, %v2498 : tensor<32x96x14x14xf32>
    %v2502 = stablehlo.subtract %v2500, %v2501 : tensor<32x96x14x14xf32>
    %v2503 = stablehlo.divide %v2489, %v2478 : tensor<32x96x14x14xf32>
    %v2504 = stablehlo.multiply %v2503, %v2502 : tensor<32x96x14x14xf32>
    %v2505 = stablehlo.reshape %v2504 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2506 = stablehlo.reshape %v2505 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2507 = stablehlo.reverse %b13pW, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2508 = stablehlo.transpose %v2507, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2509 = stablehlo.convolution(%v2506, %v2508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2511 = stablehlo.reshape %v2510 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2512 = stablehlo.reshape %v1127 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2513 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2514 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2515 = stablehlo.compare GT, %v2512, %v2513 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2516 = stablehlo.compare LT, %v2512, %v2514 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2517 = stablehlo.and %v2515, %v2516 : tensor<32x576x14x14xi1>
    %v2518 = stablehlo.select %v2517, %v2511, %v2513 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2520 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2522 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2523 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2524 = stablehlo.reduce(%v2520 init: %v2521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2525 = stablehlo.broadcast_in_dim %v2524, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2526 = stablehlo.divide %v2525, %v2522 : tensor<32x576x14x14xf32>
    %v2527 = stablehlo.subtract %v2520, %v2526 : tensor<32x576x14x14xf32>
    %v2528 = stablehlo.multiply %v2527, %v2527 : tensor<32x576x14x14xf32>
    %v2529 = stablehlo.reduce(%v2528 init: %v2521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2530 = stablehlo.broadcast_in_dim %v2529, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2531 = stablehlo.divide %v2530, %v2522 : tensor<32x576x14x14xf32>
    %v2532 = stablehlo.add %v2531, %v2523 : tensor<32x576x14x14xf32>
    %v2533 = stablehlo.rsqrt %v2532 : tensor<32x576x14x14xf32>
    %v2534 = stablehlo.multiply %v2527, %v2533 : tensor<32x576x14x14xf32>
    %v2535 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2536 = stablehlo.reshape %v2519 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2537 = stablehlo.multiply %v2535, %v2536 : tensor<32x576x14x14xf32>
    %v2538 = stablehlo.reduce(%v2537 init: %v2521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2539 = stablehlo.broadcast_in_dim %v2538, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2540 = stablehlo.multiply %v2534, %v2537 : tensor<32x576x14x14xf32>
    %v2541 = stablehlo.reduce(%v2540 init: %v2521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2542 = stablehlo.broadcast_in_dim %v2541, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2543 = stablehlo.multiply %v2537, %v2522 : tensor<32x576x14x14xf32>
    %v2544 = stablehlo.subtract %v2543, %v2539 : tensor<32x576x14x14xf32>
    %v2545 = stablehlo.multiply %v2534, %v2542 : tensor<32x576x14x14xf32>
    %v2546 = stablehlo.subtract %v2544, %v2545 : tensor<32x576x14x14xf32>
    %v2547 = stablehlo.divide %v2533, %v2522 : tensor<32x576x14x14xf32>
    %v2548 = stablehlo.multiply %v2547, %v2546 : tensor<32x576x14x14xf32>
    %v2549 = stablehlo.reshape %v2548 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2550 = stablehlo.reshape %v2549 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2551 = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2552 = stablehlo.convolution(%v2550, %v2551)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2553 = stablehlo.reshape %v2552 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2554 = stablehlo.reshape %v2553 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2555 = stablehlo.reshape %v1096 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2556 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2557 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2558 = stablehlo.compare GT, %v2555, %v2556 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2559 = stablehlo.compare LT, %v2555, %v2557 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2560 = stablehlo.and %v2558, %v2559 : tensor<32x576x14x14xi1>
    %v2561 = stablehlo.select %v2560, %v2554, %v2556 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2563 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2565 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2566 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2567 = stablehlo.reduce(%v2563 init: %v2564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2568 = stablehlo.broadcast_in_dim %v2567, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2569 = stablehlo.divide %v2568, %v2565 : tensor<32x576x14x14xf32>
    %v2570 = stablehlo.subtract %v2563, %v2569 : tensor<32x576x14x14xf32>
    %v2571 = stablehlo.multiply %v2570, %v2570 : tensor<32x576x14x14xf32>
    %v2572 = stablehlo.reduce(%v2571 init: %v2564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2573 = stablehlo.broadcast_in_dim %v2572, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2574 = stablehlo.divide %v2573, %v2565 : tensor<32x576x14x14xf32>
    %v2575 = stablehlo.add %v2574, %v2566 : tensor<32x576x14x14xf32>
    %v2576 = stablehlo.rsqrt %v2575 : tensor<32x576x14x14xf32>
    %v2577 = stablehlo.multiply %v2570, %v2576 : tensor<32x576x14x14xf32>
    %v2578 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2579 = stablehlo.reshape %v2562 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2580 = stablehlo.multiply %v2578, %v2579 : tensor<32x576x14x14xf32>
    %v2581 = stablehlo.reduce(%v2580 init: %v2564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2582 = stablehlo.broadcast_in_dim %v2581, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2583 = stablehlo.multiply %v2577, %v2580 : tensor<32x576x14x14xf32>
    %v2584 = stablehlo.reduce(%v2583 init: %v2564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2585 = stablehlo.broadcast_in_dim %v2584, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2586 = stablehlo.multiply %v2580, %v2565 : tensor<32x576x14x14xf32>
    %v2587 = stablehlo.subtract %v2586, %v2582 : tensor<32x576x14x14xf32>
    %v2588 = stablehlo.multiply %v2577, %v2585 : tensor<32x576x14x14xf32>
    %v2589 = stablehlo.subtract %v2587, %v2588 : tensor<32x576x14x14xf32>
    %v2590 = stablehlo.divide %v2576, %v2565 : tensor<32x576x14x14xf32>
    %v2591 = stablehlo.multiply %v2590, %v2589 : tensor<32x576x14x14xf32>
    %v2592 = stablehlo.reshape %v2591 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2593 = stablehlo.reshape %v2592 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2594 = stablehlo.reverse %b13eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2595 = stablehlo.transpose %v2594, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2596 = stablehlo.convolution(%v2593, %v2595)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2597 = stablehlo.reshape %v2596 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2599 = stablehlo.reshape %v2392 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2600 = stablehlo.add %v2598, %v2599 : tensor<32x96x14x14xf32>
    %v2601 = stablehlo.reshape %v2600 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2602 = stablehlo.reshape %v1071 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2603 = stablehlo.reshape %v2592 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2604 = stablehlo.transpose %v2602, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2605 = stablehlo.transpose %v2603, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2606 = stablehlo.convolution(%v2604, %v2605)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2607 = stablehlo.transpose %v2606, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2608 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2610 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2611 = stablehlo.reduce(%v2608 init: %v2609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2612 = stablehlo.broadcast_in_dim %v2611, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2613 = stablehlo.divide %v2612, %v2610 : tensor<32x576x14x14xf32>
    %v2614 = stablehlo.subtract %v2608, %v2613 : tensor<32x576x14x14xf32>
    %v2615 = stablehlo.multiply %v2614, %v2614 : tensor<32x576x14x14xf32>
    %v2616 = stablehlo.reduce(%v2615 init: %v2609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2617 = stablehlo.broadcast_in_dim %v2616, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2618 = stablehlo.divide %v2617, %v2610 : tensor<32x576x14x14xf32>
    %v2619 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2620 = stablehlo.add %v2618, %v2619 : tensor<32x576x14x14xf32>
    %v2621 = stablehlo.rsqrt %v2620 : tensor<32x576x14x14xf32>
    %v2622 = stablehlo.multiply %v2614, %v2621 : tensor<32x576x14x14xf32>
    %v2623 = stablehlo.reshape %v2562 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2624 = stablehlo.multiply %v2623, %v2622 : tensor<32x576x14x14xf32>
    %v2625 = stablehlo.reduce(%v2624 init: %v2609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2626 = stablehlo.reshape %v2562 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2628 = stablehlo.reduce(%v2626 init: %v2627) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2629 = stablehlo.reshape %v1102 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2630 = stablehlo.reshape %v2549 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2631 = stablehlo.transpose %v2629, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2632 = stablehlo.transpose %v2630, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2633 = stablehlo.convolution(%v2631, %v2632)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2634 = stablehlo.reshape %v2633 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2635 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2637 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2638 = stablehlo.reduce(%v2635 init: %v2636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2639 = stablehlo.broadcast_in_dim %v2638, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2640 = stablehlo.divide %v2639, %v2637 : tensor<32x576x14x14xf32>
    %v2641 = stablehlo.subtract %v2635, %v2640 : tensor<32x576x14x14xf32>
    %v2642 = stablehlo.multiply %v2641, %v2641 : tensor<32x576x14x14xf32>
    %v2643 = stablehlo.reduce(%v2642 init: %v2636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2644 = stablehlo.broadcast_in_dim %v2643, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2645 = stablehlo.divide %v2644, %v2637 : tensor<32x576x14x14xf32>
    %v2646 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2647 = stablehlo.add %v2645, %v2646 : tensor<32x576x14x14xf32>
    %v2648 = stablehlo.rsqrt %v2647 : tensor<32x576x14x14xf32>
    %v2649 = stablehlo.multiply %v2641, %v2648 : tensor<32x576x14x14xf32>
    %v2650 = stablehlo.reshape %v2519 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2651 = stablehlo.multiply %v2650, %v2649 : tensor<32x576x14x14xf32>
    %v2652 = stablehlo.reduce(%v2651 init: %v2636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2653 = stablehlo.reshape %v2519 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2655 = stablehlo.reduce(%v2653 init: %v2654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2656 = stablehlo.reshape %v1133 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2657 = stablehlo.reshape %v2505 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2658 = stablehlo.transpose %v2656, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2659 = stablehlo.transpose %v2657, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2660 = stablehlo.convolution(%v2658, %v2659)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2661 = stablehlo.transpose %v2660, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2662 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2664 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2665 = stablehlo.reduce(%v2662 init: %v2663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2666 = stablehlo.broadcast_in_dim %v2665, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2667 = stablehlo.divide %v2666, %v2664 : tensor<32x96x14x14xf32>
    %v2668 = stablehlo.subtract %v2662, %v2667 : tensor<32x96x14x14xf32>
    %v2669 = stablehlo.multiply %v2668, %v2668 : tensor<32x96x14x14xf32>
    %v2670 = stablehlo.reduce(%v2669 init: %v2663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2671 = stablehlo.broadcast_in_dim %v2670, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2672 = stablehlo.divide %v2671, %v2664 : tensor<32x96x14x14xf32>
    %v2673 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2674 = stablehlo.add %v2672, %v2673 : tensor<32x96x14x14xf32>
    %v2675 = stablehlo.rsqrt %v2674 : tensor<32x96x14x14xf32>
    %v2676 = stablehlo.multiply %v2668, %v2675 : tensor<32x96x14x14xf32>
    %v2677 = stablehlo.reshape %v2392 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2678 = stablehlo.multiply %v2677, %v2676 : tensor<32x96x14x14xf32>
    %v2679 = stablehlo.reduce(%v2678 init: %v2663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2680 = stablehlo.reshape %v2392 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2682 = stablehlo.reduce(%v2680 init: %v2681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2683 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2685 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2686 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2687 = stablehlo.reduce(%v2683 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2688 = stablehlo.broadcast_in_dim %v2687, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2689 = stablehlo.divide %v2688, %v2685 : tensor<32x96x14x14xf32>
    %v2690 = stablehlo.subtract %v2683, %v2689 : tensor<32x96x14x14xf32>
    %v2691 = stablehlo.multiply %v2690, %v2690 : tensor<32x96x14x14xf32>
    %v2692 = stablehlo.reduce(%v2691 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2693 = stablehlo.broadcast_in_dim %v2692, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2694 = stablehlo.divide %v2693, %v2685 : tensor<32x96x14x14xf32>
    %v2695 = stablehlo.add %v2694, %v2686 : tensor<32x96x14x14xf32>
    %v2696 = stablehlo.rsqrt %v2695 : tensor<32x96x14x14xf32>
    %v2697 = stablehlo.multiply %v2690, %v2696 : tensor<32x96x14x14xf32>
    %v2698 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2699 = stablehlo.reshape %v2601 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2700 = stablehlo.multiply %v2698, %v2699 : tensor<32x96x14x14xf32>
    %v2701 = stablehlo.reduce(%v2700 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2702 = stablehlo.broadcast_in_dim %v2701, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2703 = stablehlo.multiply %v2697, %v2700 : tensor<32x96x14x14xf32>
    %v2704 = stablehlo.reduce(%v2703 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2705 = stablehlo.broadcast_in_dim %v2704, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2706 = stablehlo.multiply %v2700, %v2685 : tensor<32x96x14x14xf32>
    %v2707 = stablehlo.subtract %v2706, %v2702 : tensor<32x96x14x14xf32>
    %v2708 = stablehlo.multiply %v2697, %v2705 : tensor<32x96x14x14xf32>
    %v2709 = stablehlo.subtract %v2707, %v2708 : tensor<32x96x14x14xf32>
    %v2710 = stablehlo.divide %v2696, %v2685 : tensor<32x96x14x14xf32>
    %v2711 = stablehlo.multiply %v2710, %v2709 : tensor<32x96x14x14xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2713 = stablehlo.reshape %v2712 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2714 = stablehlo.reverse %b12pW, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2715 = stablehlo.transpose %v2714, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2716 = stablehlo.convolution(%v2713, %v2715)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2717 = stablehlo.reshape %v2716 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2718 = stablehlo.reshape %v2717 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2719 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2720 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2721 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2722 = stablehlo.compare GT, %v2719, %v2720 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2723 = stablehlo.compare LT, %v2719, %v2721 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2724 = stablehlo.and %v2722, %v2723 : tensor<32x576x14x14xi1>
    %v2725 = stablehlo.select %v2724, %v2718, %v2720 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2726 = stablehlo.reshape %v2725 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2727 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2729 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2730 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2731 = stablehlo.reduce(%v2727 init: %v2728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2732 = stablehlo.broadcast_in_dim %v2731, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2733 = stablehlo.divide %v2732, %v2729 : tensor<32x576x14x14xf32>
    %v2734 = stablehlo.subtract %v2727, %v2733 : tensor<32x576x14x14xf32>
    %v2735 = stablehlo.multiply %v2734, %v2734 : tensor<32x576x14x14xf32>
    %v2736 = stablehlo.reduce(%v2735 init: %v2728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2737 = stablehlo.broadcast_in_dim %v2736, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2738 = stablehlo.divide %v2737, %v2729 : tensor<32x576x14x14xf32>
    %v2739 = stablehlo.add %v2738, %v2730 : tensor<32x576x14x14xf32>
    %v2740 = stablehlo.rsqrt %v2739 : tensor<32x576x14x14xf32>
    %v2741 = stablehlo.multiply %v2734, %v2740 : tensor<32x576x14x14xf32>
    %v2742 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2743 = stablehlo.reshape %v2726 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2744 = stablehlo.multiply %v2742, %v2743 : tensor<32x576x14x14xf32>
    %v2745 = stablehlo.reduce(%v2744 init: %v2728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2746 = stablehlo.broadcast_in_dim %v2745, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2747 = stablehlo.multiply %v2741, %v2744 : tensor<32x576x14x14xf32>
    %v2748 = stablehlo.reduce(%v2747 init: %v2728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2749 = stablehlo.broadcast_in_dim %v2748, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2750 = stablehlo.multiply %v2744, %v2729 : tensor<32x576x14x14xf32>
    %v2751 = stablehlo.subtract %v2750, %v2746 : tensor<32x576x14x14xf32>
    %v2752 = stablehlo.multiply %v2741, %v2749 : tensor<32x576x14x14xf32>
    %v2753 = stablehlo.subtract %v2751, %v2752 : tensor<32x576x14x14xf32>
    %v2754 = stablehlo.divide %v2740, %v2729 : tensor<32x576x14x14xf32>
    %v2755 = stablehlo.multiply %v2754, %v2753 : tensor<32x576x14x14xf32>
    %v2756 = stablehlo.reshape %v2755 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2758 = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2759 = stablehlo.convolution(%v2757, %v2758)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2760 = stablehlo.reshape %v2759 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2762 = stablehlo.reshape %v1005 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2763 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2764 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2765 = stablehlo.compare GT, %v2762, %v2763 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2766 = stablehlo.compare LT, %v2762, %v2764 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2767 = stablehlo.and %v2765, %v2766 : tensor<32x576x14x14xi1>
    %v2768 = stablehlo.select %v2767, %v2761, %v2763 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2769 = stablehlo.reshape %v2768 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2770 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2772 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2773 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2774 = stablehlo.reduce(%v2770 init: %v2771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2775 = stablehlo.broadcast_in_dim %v2774, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2776 = stablehlo.divide %v2775, %v2772 : tensor<32x576x14x14xf32>
    %v2777 = stablehlo.subtract %v2770, %v2776 : tensor<32x576x14x14xf32>
    %v2778 = stablehlo.multiply %v2777, %v2777 : tensor<32x576x14x14xf32>
    %v2779 = stablehlo.reduce(%v2778 init: %v2771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2780 = stablehlo.broadcast_in_dim %v2779, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2781 = stablehlo.divide %v2780, %v2772 : tensor<32x576x14x14xf32>
    %v2782 = stablehlo.add %v2781, %v2773 : tensor<32x576x14x14xf32>
    %v2783 = stablehlo.rsqrt %v2782 : tensor<32x576x14x14xf32>
    %v2784 = stablehlo.multiply %v2777, %v2783 : tensor<32x576x14x14xf32>
    %v2785 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2786 = stablehlo.reshape %v2769 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2787 = stablehlo.multiply %v2785, %v2786 : tensor<32x576x14x14xf32>
    %v2788 = stablehlo.reduce(%v2787 init: %v2771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2790 = stablehlo.multiply %v2784, %v2787 : tensor<32x576x14x14xf32>
    %v2791 = stablehlo.reduce(%v2790 init: %v2771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2792 = stablehlo.broadcast_in_dim %v2791, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2793 = stablehlo.multiply %v2787, %v2772 : tensor<32x576x14x14xf32>
    %v2794 = stablehlo.subtract %v2793, %v2789 : tensor<32x576x14x14xf32>
    %v2795 = stablehlo.multiply %v2784, %v2792 : tensor<32x576x14x14xf32>
    %v2796 = stablehlo.subtract %v2794, %v2795 : tensor<32x576x14x14xf32>
    %v2797 = stablehlo.divide %v2783, %v2772 : tensor<32x576x14x14xf32>
    %v2798 = stablehlo.multiply %v2797, %v2796 : tensor<32x576x14x14xf32>
    %v2799 = stablehlo.reshape %v2798 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2800 = stablehlo.reshape %v2799 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2801 = stablehlo.reverse %b12eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2802 = stablehlo.transpose %v2801, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2803 = stablehlo.convolution(%v2800, %v2802)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2804 = stablehlo.reshape %v2803 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2805 = stablehlo.reshape %v2804 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2806 = stablehlo.reshape %v2601 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2807 = stablehlo.add %v2805, %v2806 : tensor<32x96x14x14xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2809 = stablehlo.reshape %v980 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2810 = stablehlo.reshape %v2799 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2811 = stablehlo.transpose %v2809, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2812 = stablehlo.transpose %v2810, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2813 = stablehlo.convolution(%v2811, %v2812)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2814 = stablehlo.transpose %v2813, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2815 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2817 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2818 = stablehlo.reduce(%v2815 init: %v2816) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2819 = stablehlo.broadcast_in_dim %v2818, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2820 = stablehlo.divide %v2819, %v2817 : tensor<32x576x14x14xf32>
    %v2821 = stablehlo.subtract %v2815, %v2820 : tensor<32x576x14x14xf32>
    %v2822 = stablehlo.multiply %v2821, %v2821 : tensor<32x576x14x14xf32>
    %v2823 = stablehlo.reduce(%v2822 init: %v2816) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2824 = stablehlo.broadcast_in_dim %v2823, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2825 = stablehlo.divide %v2824, %v2817 : tensor<32x576x14x14xf32>
    %v2826 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2827 = stablehlo.add %v2825, %v2826 : tensor<32x576x14x14xf32>
    %v2828 = stablehlo.rsqrt %v2827 : tensor<32x576x14x14xf32>
    %v2829 = stablehlo.multiply %v2821, %v2828 : tensor<32x576x14x14xf32>
    %v2830 = stablehlo.reshape %v2769 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2831 = stablehlo.multiply %v2830, %v2829 : tensor<32x576x14x14xf32>
    %v2832 = stablehlo.reduce(%v2831 init: %v2816) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2833 = stablehlo.reshape %v2769 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2835 = stablehlo.reduce(%v2833 init: %v2834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2836 = stablehlo.reshape %v1011 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2837 = stablehlo.reshape %v2756 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2838 = stablehlo.transpose %v2836, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2839 = stablehlo.transpose %v2837, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2840 = stablehlo.convolution(%v2838, %v2839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2842 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2844 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2845 = stablehlo.reduce(%v2842 init: %v2843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2846 = stablehlo.broadcast_in_dim %v2845, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2847 = stablehlo.divide %v2846, %v2844 : tensor<32x576x14x14xf32>
    %v2848 = stablehlo.subtract %v2842, %v2847 : tensor<32x576x14x14xf32>
    %v2849 = stablehlo.multiply %v2848, %v2848 : tensor<32x576x14x14xf32>
    %v2850 = stablehlo.reduce(%v2849 init: %v2843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2851 = stablehlo.broadcast_in_dim %v2850, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2852 = stablehlo.divide %v2851, %v2844 : tensor<32x576x14x14xf32>
    %v2853 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2854 = stablehlo.add %v2852, %v2853 : tensor<32x576x14x14xf32>
    %v2855 = stablehlo.rsqrt %v2854 : tensor<32x576x14x14xf32>
    %v2856 = stablehlo.multiply %v2848, %v2855 : tensor<32x576x14x14xf32>
    %v2857 = stablehlo.reshape %v2726 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2858 = stablehlo.multiply %v2857, %v2856 : tensor<32x576x14x14xf32>
    %v2859 = stablehlo.reduce(%v2858 init: %v2843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2860 = stablehlo.reshape %v2726 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2862 = stablehlo.reduce(%v2860 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2863 = stablehlo.reshape %v1042 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2864 = stablehlo.reshape %v2712 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2865 = stablehlo.transpose %v2863, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2866 = stablehlo.transpose %v2864, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2867 = stablehlo.convolution(%v2865, %v2866)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2868 = stablehlo.transpose %v2867, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2869 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2871 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2872 = stablehlo.reduce(%v2869 init: %v2870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2873 = stablehlo.broadcast_in_dim %v2872, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2874 = stablehlo.divide %v2873, %v2871 : tensor<32x96x14x14xf32>
    %v2875 = stablehlo.subtract %v2869, %v2874 : tensor<32x96x14x14xf32>
    %v2876 = stablehlo.multiply %v2875, %v2875 : tensor<32x96x14x14xf32>
    %v2877 = stablehlo.reduce(%v2876 init: %v2870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2878 = stablehlo.broadcast_in_dim %v2877, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2879 = stablehlo.divide %v2878, %v2871 : tensor<32x96x14x14xf32>
    %v2880 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2881 = stablehlo.add %v2879, %v2880 : tensor<32x96x14x14xf32>
    %v2882 = stablehlo.rsqrt %v2881 : tensor<32x96x14x14xf32>
    %v2883 = stablehlo.multiply %v2875, %v2882 : tensor<32x96x14x14xf32>
    %v2884 = stablehlo.reshape %v2601 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2885 = stablehlo.multiply %v2884, %v2883 : tensor<32x96x14x14xf32>
    %v2886 = stablehlo.reduce(%v2885 init: %v2870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2887 = stablehlo.reshape %v2601 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2889 = stablehlo.reduce(%v2887 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2890 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2892 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2893 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2894 = stablehlo.reduce(%v2890 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2895 = stablehlo.broadcast_in_dim %v2894, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2896 = stablehlo.divide %v2895, %v2892 : tensor<32x96x14x14xf32>
    %v2897 = stablehlo.subtract %v2890, %v2896 : tensor<32x96x14x14xf32>
    %v2898 = stablehlo.multiply %v2897, %v2897 : tensor<32x96x14x14xf32>
    %v2899 = stablehlo.reduce(%v2898 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2900 = stablehlo.broadcast_in_dim %v2899, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2901 = stablehlo.divide %v2900, %v2892 : tensor<32x96x14x14xf32>
    %v2902 = stablehlo.add %v2901, %v2893 : tensor<32x96x14x14xf32>
    %v2903 = stablehlo.rsqrt %v2902 : tensor<32x96x14x14xf32>
    %v2904 = stablehlo.multiply %v2897, %v2903 : tensor<32x96x14x14xf32>
    %v2905 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2906 = stablehlo.reshape %v2808 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2907 = stablehlo.multiply %v2905, %v2906 : tensor<32x96x14x14xf32>
    %v2908 = stablehlo.reduce(%v2907 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2909 = stablehlo.broadcast_in_dim %v2908, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2910 = stablehlo.multiply %v2904, %v2907 : tensor<32x96x14x14xf32>
    %v2911 = stablehlo.reduce(%v2910 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2912 = stablehlo.broadcast_in_dim %v2911, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2913 = stablehlo.multiply %v2907, %v2892 : tensor<32x96x14x14xf32>
    %v2914 = stablehlo.subtract %v2913, %v2909 : tensor<32x96x14x14xf32>
    %v2915 = stablehlo.multiply %v2904, %v2912 : tensor<32x96x14x14xf32>
    %v2916 = stablehlo.subtract %v2914, %v2915 : tensor<32x96x14x14xf32>
    %v2917 = stablehlo.divide %v2903, %v2892 : tensor<32x96x14x14xf32>
    %v2918 = stablehlo.multiply %v2917, %v2916 : tensor<32x96x14x14xf32>
    %v2919 = stablehlo.reshape %v2918 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2920 = stablehlo.reshape %v2919 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2921 = stablehlo.reverse %b11pW, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v2922 = stablehlo.transpose %v2921, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v2923 = stablehlo.convolution(%v2920, %v2922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2924 = stablehlo.reshape %v2923 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2925 = stablehlo.reshape %v2924 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2926 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2927 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v2928 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v2929 = stablehlo.compare GT, %v2926, %v2927 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v2930 = stablehlo.compare LT, %v2926, %v2928 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v2931 = stablehlo.and %v2929, %v2930 : tensor<32x384x14x14xi1>
    %v2932 = stablehlo.select %v2931, %v2925, %v2927 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v2933 = stablehlo.reshape %v2932 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2934 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2936 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v2937 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2938 = stablehlo.reduce(%v2934 init: %v2935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2939 = stablehlo.broadcast_in_dim %v2938, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2940 = stablehlo.divide %v2939, %v2936 : tensor<32x384x14x14xf32>
    %v2941 = stablehlo.subtract %v2934, %v2940 : tensor<32x384x14x14xf32>
    %v2942 = stablehlo.multiply %v2941, %v2941 : tensor<32x384x14x14xf32>
    %v2943 = stablehlo.reduce(%v2942 init: %v2935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2944 = stablehlo.broadcast_in_dim %v2943, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2945 = stablehlo.divide %v2944, %v2936 : tensor<32x384x14x14xf32>
    %v2946 = stablehlo.add %v2945, %v2937 : tensor<32x384x14x14xf32>
    %v2947 = stablehlo.rsqrt %v2946 : tensor<32x384x14x14xf32>
    %v2948 = stablehlo.multiply %v2941, %v2947 : tensor<32x384x14x14xf32>
    %v2949 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2950 = stablehlo.reshape %v2933 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2951 = stablehlo.multiply %v2949, %v2950 : tensor<32x384x14x14xf32>
    %v2952 = stablehlo.reduce(%v2951 init: %v2935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2953 = stablehlo.broadcast_in_dim %v2952, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2954 = stablehlo.multiply %v2948, %v2951 : tensor<32x384x14x14xf32>
    %v2955 = stablehlo.reduce(%v2954 init: %v2935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2956 = stablehlo.broadcast_in_dim %v2955, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2957 = stablehlo.multiply %v2951, %v2936 : tensor<32x384x14x14xf32>
    %v2958 = stablehlo.subtract %v2957, %v2953 : tensor<32x384x14x14xf32>
    %v2959 = stablehlo.multiply %v2948, %v2956 : tensor<32x384x14x14xf32>
    %v2960 = stablehlo.subtract %v2958, %v2959 : tensor<32x384x14x14xf32>
    %v2961 = stablehlo.divide %v2947, %v2936 : tensor<32x384x14x14xf32>
    %v2962 = stablehlo.multiply %v2961, %v2960 : tensor<32x384x14x14xf32>
    %v2963 = stablehlo.reshape %v2962 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2964 = stablehlo.reshape %v2963 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2965 = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v2966 = stablehlo.convolution(%v2964, %v2965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v2967 = stablehlo.reshape %v2966 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2968 = stablehlo.reshape %v2967 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2969 = stablehlo.reshape %v918 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2970 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v2971 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v2972 = stablehlo.compare GT, %v2969, %v2970 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v2973 = stablehlo.compare LT, %v2969, %v2971 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v2974 = stablehlo.and %v2972, %v2973 : tensor<32x384x14x14xi1>
    %v2975 = stablehlo.select %v2974, %v2968, %v2970 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v2976 = stablehlo.reshape %v2975 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2977 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2979 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v2980 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2981 = stablehlo.reduce(%v2977 init: %v2978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2981, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2983 = stablehlo.divide %v2982, %v2979 : tensor<32x384x14x14xf32>
    %v2984 = stablehlo.subtract %v2977, %v2983 : tensor<32x384x14x14xf32>
    %v2985 = stablehlo.multiply %v2984, %v2984 : tensor<32x384x14x14xf32>
    %v2986 = stablehlo.reduce(%v2985 init: %v2978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2987 = stablehlo.broadcast_in_dim %v2986, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2988 = stablehlo.divide %v2987, %v2979 : tensor<32x384x14x14xf32>
    %v2989 = stablehlo.add %v2988, %v2980 : tensor<32x384x14x14xf32>
    %v2990 = stablehlo.rsqrt %v2989 : tensor<32x384x14x14xf32>
    %v2991 = stablehlo.multiply %v2984, %v2990 : tensor<32x384x14x14xf32>
    %v2992 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2993 = stablehlo.reshape %v2976 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2994 = stablehlo.multiply %v2992, %v2993 : tensor<32x384x14x14xf32>
    %v2995 = stablehlo.reduce(%v2994 init: %v2978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2996 = stablehlo.broadcast_in_dim %v2995, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2997 = stablehlo.multiply %v2991, %v2994 : tensor<32x384x14x14xf32>
    %v2998 = stablehlo.reduce(%v2997 init: %v2978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2999 = stablehlo.broadcast_in_dim %v2998, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3000 = stablehlo.multiply %v2994, %v2979 : tensor<32x384x14x14xf32>
    %v3001 = stablehlo.subtract %v3000, %v2996 : tensor<32x384x14x14xf32>
    %v3002 = stablehlo.multiply %v2991, %v2999 : tensor<32x384x14x14xf32>
    %v3003 = stablehlo.subtract %v3001, %v3002 : tensor<32x384x14x14xf32>
    %v3004 = stablehlo.divide %v2990, %v2979 : tensor<32x384x14x14xf32>
    %v3005 = stablehlo.multiply %v3004, %v3003 : tensor<32x384x14x14xf32>
    %v3006 = stablehlo.reshape %v3005 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3007 = stablehlo.reshape %v3006 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3008 = stablehlo.reverse %b11eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3009 = stablehlo.transpose %v3008, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3010 = stablehlo.convolution(%v3007, %v3009)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3011 = stablehlo.reshape %v3010 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3012 = stablehlo.reshape %v893 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3013 = stablehlo.reshape %v3006 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3014 = stablehlo.transpose %v3012, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3015 = stablehlo.transpose %v3013, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3016 = stablehlo.convolution(%v3014, %v3015)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3017 = stablehlo.transpose %v3016, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3018 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3020 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3021 = stablehlo.reduce(%v3018 init: %v3019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3022 = stablehlo.broadcast_in_dim %v3021, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3023 = stablehlo.divide %v3022, %v3020 : tensor<32x384x14x14xf32>
    %v3024 = stablehlo.subtract %v3018, %v3023 : tensor<32x384x14x14xf32>
    %v3025 = stablehlo.multiply %v3024, %v3024 : tensor<32x384x14x14xf32>
    %v3026 = stablehlo.reduce(%v3025 init: %v3019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3027 = stablehlo.broadcast_in_dim %v3026, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3028 = stablehlo.divide %v3027, %v3020 : tensor<32x384x14x14xf32>
    %v3029 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3030 = stablehlo.add %v3028, %v3029 : tensor<32x384x14x14xf32>
    %v3031 = stablehlo.rsqrt %v3030 : tensor<32x384x14x14xf32>
    %v3032 = stablehlo.multiply %v3024, %v3031 : tensor<32x384x14x14xf32>
    %v3033 = stablehlo.reshape %v2976 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3034 = stablehlo.multiply %v3033, %v3032 : tensor<32x384x14x14xf32>
    %v3035 = stablehlo.reduce(%v3034 init: %v3019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3036 = stablehlo.reshape %v2976 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3038 = stablehlo.reduce(%v3036 init: %v3037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3039 = stablehlo.reshape %v924 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3040 = stablehlo.reshape %v2963 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3041 = stablehlo.transpose %v3039, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3042 = stablehlo.transpose %v3040, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3043 = stablehlo.convolution(%v3041, %v3042)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3044 = stablehlo.reshape %v3043 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3045 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3046 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3047 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3048 = stablehlo.reduce(%v3045 init: %v3046) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3049 = stablehlo.broadcast_in_dim %v3048, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3050 = stablehlo.divide %v3049, %v3047 : tensor<32x384x14x14xf32>
    %v3051 = stablehlo.subtract %v3045, %v3050 : tensor<32x384x14x14xf32>
    %v3052 = stablehlo.multiply %v3051, %v3051 : tensor<32x384x14x14xf32>
    %v3053 = stablehlo.reduce(%v3052 init: %v3046) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3054 = stablehlo.broadcast_in_dim %v3053, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3055 = stablehlo.divide %v3054, %v3047 : tensor<32x384x14x14xf32>
    %v3056 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3057 = stablehlo.add %v3055, %v3056 : tensor<32x384x14x14xf32>
    %v3058 = stablehlo.rsqrt %v3057 : tensor<32x384x14x14xf32>
    %v3059 = stablehlo.multiply %v3051, %v3058 : tensor<32x384x14x14xf32>
    %v3060 = stablehlo.reshape %v2933 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3061 = stablehlo.multiply %v3060, %v3059 : tensor<32x384x14x14xf32>
    %v3062 = stablehlo.reduce(%v3061 init: %v3046) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3063 = stablehlo.reshape %v2933 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3065 = stablehlo.reduce(%v3063 init: %v3064) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3066 = stablehlo.reshape %v955 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3067 = stablehlo.reshape %v2919 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3068 = stablehlo.transpose %v3066, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3069 = stablehlo.transpose %v3067, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v3070 = stablehlo.convolution(%v3068, %v3069)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<384x96x1x1xf32>
    %v3071 = stablehlo.transpose %v3070, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3072 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3074 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v3075 = stablehlo.reduce(%v3072 init: %v3073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v3077 = stablehlo.divide %v3076, %v3074 : tensor<32x96x14x14xf32>
    %v3078 = stablehlo.subtract %v3072, %v3077 : tensor<32x96x14x14xf32>
    %v3079 = stablehlo.multiply %v3078, %v3078 : tensor<32x96x14x14xf32>
    %v3080 = stablehlo.reduce(%v3079 init: %v3073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3081 = stablehlo.broadcast_in_dim %v3080, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v3082 = stablehlo.divide %v3081, %v3074 : tensor<32x96x14x14xf32>
    %v3083 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v3084 = stablehlo.add %v3082, %v3083 : tensor<32x96x14x14xf32>
    %v3085 = stablehlo.rsqrt %v3084 : tensor<32x96x14x14xf32>
    %v3086 = stablehlo.multiply %v3078, %v3085 : tensor<32x96x14x14xf32>
    %v3087 = stablehlo.reshape %v2808 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3088 = stablehlo.multiply %v3087, %v3086 : tensor<32x96x14x14xf32>
    %v3089 = stablehlo.reduce(%v3088 init: %v3073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3090 = stablehlo.reshape %v2808 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3092 = stablehlo.reduce(%v3090 init: %v3091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3093 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3095 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3096 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3097 = stablehlo.reduce(%v3093 init: %v3094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3098 = stablehlo.broadcast_in_dim %v3097, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3099 = stablehlo.divide %v3098, %v3095 : tensor<32x64x14x14xf32>
    %v3100 = stablehlo.subtract %v3093, %v3099 : tensor<32x64x14x14xf32>
    %v3101 = stablehlo.multiply %v3100, %v3100 : tensor<32x64x14x14xf32>
    %v3102 = stablehlo.reduce(%v3101 init: %v3094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3103 = stablehlo.broadcast_in_dim %v3102, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3104 = stablehlo.divide %v3103, %v3095 : tensor<32x64x14x14xf32>
    %v3105 = stablehlo.add %v3104, %v3096 : tensor<32x64x14x14xf32>
    %v3106 = stablehlo.rsqrt %v3105 : tensor<32x64x14x14xf32>
    %v3107 = stablehlo.multiply %v3100, %v3106 : tensor<32x64x14x14xf32>
    %v3108 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3109 = stablehlo.reshape %v3011 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3110 = stablehlo.multiply %v3108, %v3109 : tensor<32x64x14x14xf32>
    %v3111 = stablehlo.reduce(%v3110 init: %v3094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3112 = stablehlo.broadcast_in_dim %v3111, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3113 = stablehlo.multiply %v3107, %v3110 : tensor<32x64x14x14xf32>
    %v3114 = stablehlo.reduce(%v3113 init: %v3094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3115 = stablehlo.broadcast_in_dim %v3114, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3116 = stablehlo.multiply %v3110, %v3095 : tensor<32x64x14x14xf32>
    %v3117 = stablehlo.subtract %v3116, %v3112 : tensor<32x64x14x14xf32>
    %v3118 = stablehlo.multiply %v3107, %v3115 : tensor<32x64x14x14xf32>
    %v3119 = stablehlo.subtract %v3117, %v3118 : tensor<32x64x14x14xf32>
    %v3120 = stablehlo.divide %v3106, %v3095 : tensor<32x64x14x14xf32>
    %v3121 = stablehlo.multiply %v3120, %v3119 : tensor<32x64x14x14xf32>
    %v3122 = stablehlo.reshape %v3121 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3123 = stablehlo.reshape %v3122 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3124 = stablehlo.reverse %b10pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3125 = stablehlo.transpose %v3124, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3126 = stablehlo.convolution(%v3123, %v3125)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3127 = stablehlo.reshape %v3126 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3128 = stablehlo.reshape %v3127 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3129 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3130 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3131 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3132 = stablehlo.compare GT, %v3129, %v3130 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3133 = stablehlo.compare LT, %v3129, %v3131 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3134 = stablehlo.and %v3132, %v3133 : tensor<32x384x14x14xi1>
    %v3135 = stablehlo.select %v3134, %v3128, %v3130 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3136 = stablehlo.reshape %v3135 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3137 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3139 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3140 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3141 = stablehlo.reduce(%v3137 init: %v3138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3142 = stablehlo.broadcast_in_dim %v3141, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3143 = stablehlo.divide %v3142, %v3139 : tensor<32x384x14x14xf32>
    %v3144 = stablehlo.subtract %v3137, %v3143 : tensor<32x384x14x14xf32>
    %v3145 = stablehlo.multiply %v3144, %v3144 : tensor<32x384x14x14xf32>
    %v3146 = stablehlo.reduce(%v3145 init: %v3138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3147 = stablehlo.broadcast_in_dim %v3146, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3148 = stablehlo.divide %v3147, %v3139 : tensor<32x384x14x14xf32>
    %v3149 = stablehlo.add %v3148, %v3140 : tensor<32x384x14x14xf32>
    %v3150 = stablehlo.rsqrt %v3149 : tensor<32x384x14x14xf32>
    %v3151 = stablehlo.multiply %v3144, %v3150 : tensor<32x384x14x14xf32>
    %v3152 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3153 = stablehlo.reshape %v3136 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3154 = stablehlo.multiply %v3152, %v3153 : tensor<32x384x14x14xf32>
    %v3155 = stablehlo.reduce(%v3154 init: %v3138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3156 = stablehlo.broadcast_in_dim %v3155, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3157 = stablehlo.multiply %v3151, %v3154 : tensor<32x384x14x14xf32>
    %v3158 = stablehlo.reduce(%v3157 init: %v3138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3159 = stablehlo.broadcast_in_dim %v3158, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3160 = stablehlo.multiply %v3154, %v3139 : tensor<32x384x14x14xf32>
    %v3161 = stablehlo.subtract %v3160, %v3156 : tensor<32x384x14x14xf32>
    %v3162 = stablehlo.multiply %v3151, %v3159 : tensor<32x384x14x14xf32>
    %v3163 = stablehlo.subtract %v3161, %v3162 : tensor<32x384x14x14xf32>
    %v3164 = stablehlo.divide %v3150, %v3139 : tensor<32x384x14x14xf32>
    %v3165 = stablehlo.multiply %v3164, %v3163 : tensor<32x384x14x14xf32>
    %v3166 = stablehlo.reshape %v3165 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3167 = stablehlo.reshape %v3166 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3168 = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3169 = stablehlo.convolution(%v3167, %v3168)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3170 = stablehlo.reshape %v3169 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3171 = stablehlo.reshape %v3170 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3172 = stablehlo.reshape %v827 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3173 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3174 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3175 = stablehlo.compare GT, %v3172, %v3173 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3176 = stablehlo.compare LT, %v3172, %v3174 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3177 = stablehlo.and %v3175, %v3176 : tensor<32x384x14x14xi1>
    %v3178 = stablehlo.select %v3177, %v3171, %v3173 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3179 = stablehlo.reshape %v3178 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3180 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3182 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3183 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3184 = stablehlo.reduce(%v3180 init: %v3181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3185 = stablehlo.broadcast_in_dim %v3184, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3186 = stablehlo.divide %v3185, %v3182 : tensor<32x384x14x14xf32>
    %v3187 = stablehlo.subtract %v3180, %v3186 : tensor<32x384x14x14xf32>
    %v3188 = stablehlo.multiply %v3187, %v3187 : tensor<32x384x14x14xf32>
    %v3189 = stablehlo.reduce(%v3188 init: %v3181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3190 = stablehlo.broadcast_in_dim %v3189, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3191 = stablehlo.divide %v3190, %v3182 : tensor<32x384x14x14xf32>
    %v3192 = stablehlo.add %v3191, %v3183 : tensor<32x384x14x14xf32>
    %v3193 = stablehlo.rsqrt %v3192 : tensor<32x384x14x14xf32>
    %v3194 = stablehlo.multiply %v3187, %v3193 : tensor<32x384x14x14xf32>
    %v3195 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3196 = stablehlo.reshape %v3179 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3197 = stablehlo.multiply %v3195, %v3196 : tensor<32x384x14x14xf32>
    %v3198 = stablehlo.reduce(%v3197 init: %v3181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3199 = stablehlo.broadcast_in_dim %v3198, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3200 = stablehlo.multiply %v3194, %v3197 : tensor<32x384x14x14xf32>
    %v3201 = stablehlo.reduce(%v3200 init: %v3181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3203 = stablehlo.multiply %v3197, %v3182 : tensor<32x384x14x14xf32>
    %v3204 = stablehlo.subtract %v3203, %v3199 : tensor<32x384x14x14xf32>
    %v3205 = stablehlo.multiply %v3194, %v3202 : tensor<32x384x14x14xf32>
    %v3206 = stablehlo.subtract %v3204, %v3205 : tensor<32x384x14x14xf32>
    %v3207 = stablehlo.divide %v3193, %v3182 : tensor<32x384x14x14xf32>
    %v3208 = stablehlo.multiply %v3207, %v3206 : tensor<32x384x14x14xf32>
    %v3209 = stablehlo.reshape %v3208 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3210 = stablehlo.reshape %v3209 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3211 = stablehlo.reverse %b10eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3212 = stablehlo.transpose %v3211, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3213 = stablehlo.convolution(%v3210, %v3212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3214 = stablehlo.reshape %v3213 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3215 = stablehlo.reshape %v3214 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3216 = stablehlo.reshape %v3011 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3217 = stablehlo.add %v3215, %v3216 : tensor<32x64x14x14xf32>
    %v3218 = stablehlo.reshape %v3217 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3219 = stablehlo.reshape %v802 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3220 = stablehlo.reshape %v3209 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3221 = stablehlo.transpose %v3219, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3222 = stablehlo.transpose %v3220, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3223 = stablehlo.convolution(%v3221, %v3222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3224 = stablehlo.transpose %v3223, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3225 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3227 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3228 = stablehlo.reduce(%v3225 init: %v3226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3229 = stablehlo.broadcast_in_dim %v3228, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3230 = stablehlo.divide %v3229, %v3227 : tensor<32x384x14x14xf32>
    %v3231 = stablehlo.subtract %v3225, %v3230 : tensor<32x384x14x14xf32>
    %v3232 = stablehlo.multiply %v3231, %v3231 : tensor<32x384x14x14xf32>
    %v3233 = stablehlo.reduce(%v3232 init: %v3226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3234 = stablehlo.broadcast_in_dim %v3233, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3235 = stablehlo.divide %v3234, %v3227 : tensor<32x384x14x14xf32>
    %v3236 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3237 = stablehlo.add %v3235, %v3236 : tensor<32x384x14x14xf32>
    %v3238 = stablehlo.rsqrt %v3237 : tensor<32x384x14x14xf32>
    %v3239 = stablehlo.multiply %v3231, %v3238 : tensor<32x384x14x14xf32>
    %v3240 = stablehlo.reshape %v3179 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3241 = stablehlo.multiply %v3240, %v3239 : tensor<32x384x14x14xf32>
    %v3242 = stablehlo.reduce(%v3241 init: %v3226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3243 = stablehlo.reshape %v3179 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3245 = stablehlo.reduce(%v3243 init: %v3244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3246 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3247 = stablehlo.reshape %v3166 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3248 = stablehlo.transpose %v3246, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3249 = stablehlo.transpose %v3247, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3250 = stablehlo.convolution(%v3248, %v3249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3251 = stablehlo.reshape %v3250 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3252 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3254 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3255 = stablehlo.reduce(%v3252 init: %v3253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3256 = stablehlo.broadcast_in_dim %v3255, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3257 = stablehlo.divide %v3256, %v3254 : tensor<32x384x14x14xf32>
    %v3258 = stablehlo.subtract %v3252, %v3257 : tensor<32x384x14x14xf32>
    %v3259 = stablehlo.multiply %v3258, %v3258 : tensor<32x384x14x14xf32>
    %v3260 = stablehlo.reduce(%v3259 init: %v3253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3261 = stablehlo.broadcast_in_dim %v3260, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3262 = stablehlo.divide %v3261, %v3254 : tensor<32x384x14x14xf32>
    %v3263 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3264 = stablehlo.add %v3262, %v3263 : tensor<32x384x14x14xf32>
    %v3265 = stablehlo.rsqrt %v3264 : tensor<32x384x14x14xf32>
    %v3266 = stablehlo.multiply %v3258, %v3265 : tensor<32x384x14x14xf32>
    %v3267 = stablehlo.reshape %v3136 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3268 = stablehlo.multiply %v3267, %v3266 : tensor<32x384x14x14xf32>
    %v3269 = stablehlo.reduce(%v3268 init: %v3253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3270 = stablehlo.reshape %v3136 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3272 = stablehlo.reduce(%v3270 init: %v3271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3273 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3274 = stablehlo.reshape %v3122 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3275 = stablehlo.transpose %v3273, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3276 = stablehlo.transpose %v3274, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3277 = stablehlo.convolution(%v3275, %v3276)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3278 = stablehlo.transpose %v3277, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3279 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3281 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3282 = stablehlo.reduce(%v3279 init: %v3280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3283 = stablehlo.broadcast_in_dim %v3282, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3284 = stablehlo.divide %v3283, %v3281 : tensor<32x64x14x14xf32>
    %v3285 = stablehlo.subtract %v3279, %v3284 : tensor<32x64x14x14xf32>
    %v3286 = stablehlo.multiply %v3285, %v3285 : tensor<32x64x14x14xf32>
    %v3287 = stablehlo.reduce(%v3286 init: %v3280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3288 = stablehlo.broadcast_in_dim %v3287, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3289 = stablehlo.divide %v3288, %v3281 : tensor<32x64x14x14xf32>
    %v3290 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3291 = stablehlo.add %v3289, %v3290 : tensor<32x64x14x14xf32>
    %v3292 = stablehlo.rsqrt %v3291 : tensor<32x64x14x14xf32>
    %v3293 = stablehlo.multiply %v3285, %v3292 : tensor<32x64x14x14xf32>
    %v3294 = stablehlo.reshape %v3011 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3295 = stablehlo.multiply %v3294, %v3293 : tensor<32x64x14x14xf32>
    %v3296 = stablehlo.reduce(%v3295 init: %v3280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3297 = stablehlo.reshape %v3011 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3299 = stablehlo.reduce(%v3297 init: %v3298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3300 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3302 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3303 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3304 = stablehlo.reduce(%v3300 init: %v3301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3305 = stablehlo.broadcast_in_dim %v3304, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3306 = stablehlo.divide %v3305, %v3302 : tensor<32x64x14x14xf32>
    %v3307 = stablehlo.subtract %v3300, %v3306 : tensor<32x64x14x14xf32>
    %v3308 = stablehlo.multiply %v3307, %v3307 : tensor<32x64x14x14xf32>
    %v3309 = stablehlo.reduce(%v3308 init: %v3301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3310 = stablehlo.broadcast_in_dim %v3309, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3311 = stablehlo.divide %v3310, %v3302 : tensor<32x64x14x14xf32>
    %v3312 = stablehlo.add %v3311, %v3303 : tensor<32x64x14x14xf32>
    %v3313 = stablehlo.rsqrt %v3312 : tensor<32x64x14x14xf32>
    %v3314 = stablehlo.multiply %v3307, %v3313 : tensor<32x64x14x14xf32>
    %v3315 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3316 = stablehlo.reshape %v3218 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3317 = stablehlo.multiply %v3315, %v3316 : tensor<32x64x14x14xf32>
    %v3318 = stablehlo.reduce(%v3317 init: %v3301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3319 = stablehlo.broadcast_in_dim %v3318, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3320 = stablehlo.multiply %v3314, %v3317 : tensor<32x64x14x14xf32>
    %v3321 = stablehlo.reduce(%v3320 init: %v3301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3322 = stablehlo.broadcast_in_dim %v3321, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3323 = stablehlo.multiply %v3317, %v3302 : tensor<32x64x14x14xf32>
    %v3324 = stablehlo.subtract %v3323, %v3319 : tensor<32x64x14x14xf32>
    %v3325 = stablehlo.multiply %v3314, %v3322 : tensor<32x64x14x14xf32>
    %v3326 = stablehlo.subtract %v3324, %v3325 : tensor<32x64x14x14xf32>
    %v3327 = stablehlo.divide %v3313, %v3302 : tensor<32x64x14x14xf32>
    %v3328 = stablehlo.multiply %v3327, %v3326 : tensor<32x64x14x14xf32>
    %v3329 = stablehlo.reshape %v3328 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3331 = stablehlo.reverse %b9pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3332 = stablehlo.transpose %v3331, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3333 = stablehlo.convolution(%v3330, %v3332)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3334 = stablehlo.reshape %v3333 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3335 = stablehlo.reshape %v3334 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3336 = stablehlo.reshape %v767 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3337 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3338 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3339 = stablehlo.compare GT, %v3336, %v3337 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3340 = stablehlo.compare LT, %v3336, %v3338 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3341 = stablehlo.and %v3339, %v3340 : tensor<32x384x14x14xi1>
    %v3342 = stablehlo.select %v3341, %v3335, %v3337 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3343 = stablehlo.reshape %v3342 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3344 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3346 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3347 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3348 = stablehlo.reduce(%v3344 init: %v3345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3349 = stablehlo.broadcast_in_dim %v3348, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3350 = stablehlo.divide %v3349, %v3346 : tensor<32x384x14x14xf32>
    %v3351 = stablehlo.subtract %v3344, %v3350 : tensor<32x384x14x14xf32>
    %v3352 = stablehlo.multiply %v3351, %v3351 : tensor<32x384x14x14xf32>
    %v3353 = stablehlo.reduce(%v3352 init: %v3345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3354 = stablehlo.broadcast_in_dim %v3353, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3355 = stablehlo.divide %v3354, %v3346 : tensor<32x384x14x14xf32>
    %v3356 = stablehlo.add %v3355, %v3347 : tensor<32x384x14x14xf32>
    %v3357 = stablehlo.rsqrt %v3356 : tensor<32x384x14x14xf32>
    %v3358 = stablehlo.multiply %v3351, %v3357 : tensor<32x384x14x14xf32>
    %v3359 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3360 = stablehlo.reshape %v3343 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3361 = stablehlo.multiply %v3359, %v3360 : tensor<32x384x14x14xf32>
    %v3362 = stablehlo.reduce(%v3361 init: %v3345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3363 = stablehlo.broadcast_in_dim %v3362, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3364 = stablehlo.multiply %v3358, %v3361 : tensor<32x384x14x14xf32>
    %v3365 = stablehlo.reduce(%v3364 init: %v3345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3366 = stablehlo.broadcast_in_dim %v3365, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3367 = stablehlo.multiply %v3361, %v3346 : tensor<32x384x14x14xf32>
    %v3368 = stablehlo.subtract %v3367, %v3363 : tensor<32x384x14x14xf32>
    %v3369 = stablehlo.multiply %v3358, %v3366 : tensor<32x384x14x14xf32>
    %v3370 = stablehlo.subtract %v3368, %v3369 : tensor<32x384x14x14xf32>
    %v3371 = stablehlo.divide %v3357, %v3346 : tensor<32x384x14x14xf32>
    %v3372 = stablehlo.multiply %v3371, %v3370 : tensor<32x384x14x14xf32>
    %v3373 = stablehlo.reshape %v3372 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3374 = stablehlo.reshape %v3373 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3375 = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3376 = stablehlo.convolution(%v3374, %v3375)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3377 = stablehlo.reshape %v3376 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3378 = stablehlo.reshape %v3377 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3379 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3380 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3381 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3382 = stablehlo.compare GT, %v3379, %v3380 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3383 = stablehlo.compare LT, %v3379, %v3381 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3384 = stablehlo.and %v3382, %v3383 : tensor<32x384x14x14xi1>
    %v3385 = stablehlo.select %v3384, %v3378, %v3380 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3386 = stablehlo.reshape %v3385 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3387 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3389 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3390 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3391 = stablehlo.reduce(%v3387 init: %v3388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3392 = stablehlo.broadcast_in_dim %v3391, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3393 = stablehlo.divide %v3392, %v3389 : tensor<32x384x14x14xf32>
    %v3394 = stablehlo.subtract %v3387, %v3393 : tensor<32x384x14x14xf32>
    %v3395 = stablehlo.multiply %v3394, %v3394 : tensor<32x384x14x14xf32>
    %v3396 = stablehlo.reduce(%v3395 init: %v3388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3397 = stablehlo.broadcast_in_dim %v3396, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3398 = stablehlo.divide %v3397, %v3389 : tensor<32x384x14x14xf32>
    %v3399 = stablehlo.add %v3398, %v3390 : tensor<32x384x14x14xf32>
    %v3400 = stablehlo.rsqrt %v3399 : tensor<32x384x14x14xf32>
    %v3401 = stablehlo.multiply %v3394, %v3400 : tensor<32x384x14x14xf32>
    %v3402 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3403 = stablehlo.reshape %v3386 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3404 = stablehlo.multiply %v3402, %v3403 : tensor<32x384x14x14xf32>
    %v3405 = stablehlo.reduce(%v3404 init: %v3388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3406 = stablehlo.broadcast_in_dim %v3405, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3407 = stablehlo.multiply %v3401, %v3404 : tensor<32x384x14x14xf32>
    %v3408 = stablehlo.reduce(%v3407 init: %v3388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3409 = stablehlo.broadcast_in_dim %v3408, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3410 = stablehlo.multiply %v3404, %v3389 : tensor<32x384x14x14xf32>
    %v3411 = stablehlo.subtract %v3410, %v3406 : tensor<32x384x14x14xf32>
    %v3412 = stablehlo.multiply %v3401, %v3409 : tensor<32x384x14x14xf32>
    %v3413 = stablehlo.subtract %v3411, %v3412 : tensor<32x384x14x14xf32>
    %v3414 = stablehlo.divide %v3400, %v3389 : tensor<32x384x14x14xf32>
    %v3415 = stablehlo.multiply %v3414, %v3413 : tensor<32x384x14x14xf32>
    %v3416 = stablehlo.reshape %v3415 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3417 = stablehlo.reshape %v3416 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3418 = stablehlo.reverse %b9eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3419 = stablehlo.transpose %v3418, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3420 = stablehlo.convolution(%v3417, %v3419)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3421 = stablehlo.reshape %v3420 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3422 = stablehlo.reshape %v3421 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3423 = stablehlo.reshape %v3218 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3424 = stablehlo.add %v3422, %v3423 : tensor<32x64x14x14xf32>
    %v3425 = stablehlo.reshape %v3424 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3426 = stablehlo.reshape %v711 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3427 = stablehlo.reshape %v3416 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3428 = stablehlo.transpose %v3426, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3429 = stablehlo.transpose %v3427, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3430 = stablehlo.convolution(%v3428, %v3429)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3431 = stablehlo.transpose %v3430, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3432 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3434 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3435 = stablehlo.reduce(%v3432 init: %v3433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3436 = stablehlo.broadcast_in_dim %v3435, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3437 = stablehlo.divide %v3436, %v3434 : tensor<32x384x14x14xf32>
    %v3438 = stablehlo.subtract %v3432, %v3437 : tensor<32x384x14x14xf32>
    %v3439 = stablehlo.multiply %v3438, %v3438 : tensor<32x384x14x14xf32>
    %v3440 = stablehlo.reduce(%v3439 init: %v3433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3441 = stablehlo.broadcast_in_dim %v3440, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3442 = stablehlo.divide %v3441, %v3434 : tensor<32x384x14x14xf32>
    %v3443 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3444 = stablehlo.add %v3442, %v3443 : tensor<32x384x14x14xf32>
    %v3445 = stablehlo.rsqrt %v3444 : tensor<32x384x14x14xf32>
    %v3446 = stablehlo.multiply %v3438, %v3445 : tensor<32x384x14x14xf32>
    %v3447 = stablehlo.reshape %v3386 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3448 = stablehlo.multiply %v3447, %v3446 : tensor<32x384x14x14xf32>
    %v3449 = stablehlo.reduce(%v3448 init: %v3433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3450 = stablehlo.reshape %v3386 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3452 = stablehlo.reduce(%v3450 init: %v3451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3453 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3454 = stablehlo.reshape %v3373 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3455 = stablehlo.transpose %v3453, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3456 = stablehlo.transpose %v3454, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3457 = stablehlo.convolution(%v3455, %v3456)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3458 = stablehlo.reshape %v3457 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3459 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3461 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3462 = stablehlo.reduce(%v3459 init: %v3460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3463 = stablehlo.broadcast_in_dim %v3462, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3464 = stablehlo.divide %v3463, %v3461 : tensor<32x384x14x14xf32>
    %v3465 = stablehlo.subtract %v3459, %v3464 : tensor<32x384x14x14xf32>
    %v3466 = stablehlo.multiply %v3465, %v3465 : tensor<32x384x14x14xf32>
    %v3467 = stablehlo.reduce(%v3466 init: %v3460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3468 = stablehlo.broadcast_in_dim %v3467, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3469 = stablehlo.divide %v3468, %v3461 : tensor<32x384x14x14xf32>
    %v3470 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3471 = stablehlo.add %v3469, %v3470 : tensor<32x384x14x14xf32>
    %v3472 = stablehlo.rsqrt %v3471 : tensor<32x384x14x14xf32>
    %v3473 = stablehlo.multiply %v3465, %v3472 : tensor<32x384x14x14xf32>
    %v3474 = stablehlo.reshape %v3343 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3475 = stablehlo.multiply %v3474, %v3473 : tensor<32x384x14x14xf32>
    %v3476 = stablehlo.reduce(%v3475 init: %v3460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3477 = stablehlo.reshape %v3343 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3479 = stablehlo.reduce(%v3477 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3480 = stablehlo.reshape %v773 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3481 = stablehlo.reshape %v3329 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3482 = stablehlo.transpose %v3480, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3483 = stablehlo.transpose %v3481, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3484 = stablehlo.convolution(%v3482, %v3483)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3485 = stablehlo.transpose %v3484, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3486 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3488 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3489 = stablehlo.reduce(%v3486 init: %v3487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3490 = stablehlo.broadcast_in_dim %v3489, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3491 = stablehlo.divide %v3490, %v3488 : tensor<32x64x14x14xf32>
    %v3492 = stablehlo.subtract %v3486, %v3491 : tensor<32x64x14x14xf32>
    %v3493 = stablehlo.multiply %v3492, %v3492 : tensor<32x64x14x14xf32>
    %v3494 = stablehlo.reduce(%v3493 init: %v3487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3495 = stablehlo.broadcast_in_dim %v3494, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3496 = stablehlo.divide %v3495, %v3488 : tensor<32x64x14x14xf32>
    %v3497 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3498 = stablehlo.add %v3496, %v3497 : tensor<32x64x14x14xf32>
    %v3499 = stablehlo.rsqrt %v3498 : tensor<32x64x14x14xf32>
    %v3500 = stablehlo.multiply %v3492, %v3499 : tensor<32x64x14x14xf32>
    %v3501 = stablehlo.reshape %v3218 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3502 = stablehlo.multiply %v3501, %v3500 : tensor<32x64x14x14xf32>
    %v3503 = stablehlo.reduce(%v3502 init: %v3487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3504 = stablehlo.reshape %v3218 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3506 = stablehlo.reduce(%v3504 init: %v3505) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3507 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3509 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3510 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3511 = stablehlo.reduce(%v3507 init: %v3508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3512 = stablehlo.broadcast_in_dim %v3511, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3513 = stablehlo.divide %v3512, %v3509 : tensor<32x64x14x14xf32>
    %v3514 = stablehlo.subtract %v3507, %v3513 : tensor<32x64x14x14xf32>
    %v3515 = stablehlo.multiply %v3514, %v3514 : tensor<32x64x14x14xf32>
    %v3516 = stablehlo.reduce(%v3515 init: %v3508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3517 = stablehlo.broadcast_in_dim %v3516, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3518 = stablehlo.divide %v3517, %v3509 : tensor<32x64x14x14xf32>
    %v3519 = stablehlo.add %v3518, %v3510 : tensor<32x64x14x14xf32>
    %v3520 = stablehlo.rsqrt %v3519 : tensor<32x64x14x14xf32>
    %v3521 = stablehlo.multiply %v3514, %v3520 : tensor<32x64x14x14xf32>
    %v3522 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3523 = stablehlo.reshape %v3425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3524 = stablehlo.multiply %v3522, %v3523 : tensor<32x64x14x14xf32>
    %v3525 = stablehlo.reduce(%v3524 init: %v3508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3526 = stablehlo.broadcast_in_dim %v3525, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3527 = stablehlo.multiply %v3521, %v3524 : tensor<32x64x14x14xf32>
    %v3528 = stablehlo.reduce(%v3527 init: %v3508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3529 = stablehlo.broadcast_in_dim %v3528, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3530 = stablehlo.multiply %v3524, %v3509 : tensor<32x64x14x14xf32>
    %v3531 = stablehlo.subtract %v3530, %v3526 : tensor<32x64x14x14xf32>
    %v3532 = stablehlo.multiply %v3521, %v3529 : tensor<32x64x14x14xf32>
    %v3533 = stablehlo.subtract %v3531, %v3532 : tensor<32x64x14x14xf32>
    %v3534 = stablehlo.divide %v3520, %v3509 : tensor<32x64x14x14xf32>
    %v3535 = stablehlo.multiply %v3534, %v3533 : tensor<32x64x14x14xf32>
    %v3536 = stablehlo.reshape %v3535 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3537 = stablehlo.reshape %v3536 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3538 = stablehlo.reverse %b8pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3539 = stablehlo.transpose %v3538, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3540 = stablehlo.convolution(%v3537, %v3539)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3541 = stablehlo.reshape %v3540 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3542 = stablehlo.reshape %v3541 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3543 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3544 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3545 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3546 = stablehlo.compare GT, %v3543, %v3544 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3547 = stablehlo.compare LT, %v3543, %v3545 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3548 = stablehlo.and %v3546, %v3547 : tensor<32x384x14x14xi1>
    %v3549 = stablehlo.select %v3548, %v3542, %v3544 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3550 = stablehlo.reshape %v3549 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3551 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3553 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3554 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3555 = stablehlo.reduce(%v3551 init: %v3552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3556 = stablehlo.broadcast_in_dim %v3555, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3557 = stablehlo.divide %v3556, %v3553 : tensor<32x384x14x14xf32>
    %v3558 = stablehlo.subtract %v3551, %v3557 : tensor<32x384x14x14xf32>
    %v3559 = stablehlo.multiply %v3558, %v3558 : tensor<32x384x14x14xf32>
    %v3560 = stablehlo.reduce(%v3559 init: %v3552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3561 = stablehlo.broadcast_in_dim %v3560, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3562 = stablehlo.divide %v3561, %v3553 : tensor<32x384x14x14xf32>
    %v3563 = stablehlo.add %v3562, %v3554 : tensor<32x384x14x14xf32>
    %v3564 = stablehlo.rsqrt %v3563 : tensor<32x384x14x14xf32>
    %v3565 = stablehlo.multiply %v3558, %v3564 : tensor<32x384x14x14xf32>
    %v3566 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3567 = stablehlo.reshape %v3550 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3568 = stablehlo.multiply %v3566, %v3567 : tensor<32x384x14x14xf32>
    %v3569 = stablehlo.reduce(%v3568 init: %v3552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3570 = stablehlo.broadcast_in_dim %v3569, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3571 = stablehlo.multiply %v3565, %v3568 : tensor<32x384x14x14xf32>
    %v3572 = stablehlo.reduce(%v3571 init: %v3552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3573 = stablehlo.broadcast_in_dim %v3572, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3574 = stablehlo.multiply %v3568, %v3553 : tensor<32x384x14x14xf32>
    %v3575 = stablehlo.subtract %v3574, %v3570 : tensor<32x384x14x14xf32>
    %v3576 = stablehlo.multiply %v3565, %v3573 : tensor<32x384x14x14xf32>
    %v3577 = stablehlo.subtract %v3575, %v3576 : tensor<32x384x14x14xf32>
    %v3578 = stablehlo.divide %v3564, %v3553 : tensor<32x384x14x14xf32>
    %v3579 = stablehlo.multiply %v3578, %v3577 : tensor<32x384x14x14xf32>
    %v3580 = stablehlo.reshape %v3579 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3581 = stablehlo.reshape %v3580 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3582 = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3583 = stablehlo.convolution(%v3581, %v3582)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3584 = stablehlo.reshape %v3583 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3585 = stablehlo.reshape %v3584 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3586 = stablehlo.reshape %v645 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3587 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3588 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3589 = stablehlo.compare GT, %v3586, %v3587 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3590 = stablehlo.compare LT, %v3586, %v3588 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3591 = stablehlo.and %v3589, %v3590 : tensor<32x384x14x14xi1>
    %v3592 = stablehlo.select %v3591, %v3585, %v3587 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3594 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3596 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3597 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3598 = stablehlo.reduce(%v3594 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3599 = stablehlo.broadcast_in_dim %v3598, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3600 = stablehlo.divide %v3599, %v3596 : tensor<32x384x14x14xf32>
    %v3601 = stablehlo.subtract %v3594, %v3600 : tensor<32x384x14x14xf32>
    %v3602 = stablehlo.multiply %v3601, %v3601 : tensor<32x384x14x14xf32>
    %v3603 = stablehlo.reduce(%v3602 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3604 = stablehlo.broadcast_in_dim %v3603, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3605 = stablehlo.divide %v3604, %v3596 : tensor<32x384x14x14xf32>
    %v3606 = stablehlo.add %v3605, %v3597 : tensor<32x384x14x14xf32>
    %v3607 = stablehlo.rsqrt %v3606 : tensor<32x384x14x14xf32>
    %v3608 = stablehlo.multiply %v3601, %v3607 : tensor<32x384x14x14xf32>
    %v3609 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3610 = stablehlo.reshape %v3593 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3611 = stablehlo.multiply %v3609, %v3610 : tensor<32x384x14x14xf32>
    %v3612 = stablehlo.reduce(%v3611 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3613 = stablehlo.broadcast_in_dim %v3612, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3614 = stablehlo.multiply %v3608, %v3611 : tensor<32x384x14x14xf32>
    %v3615 = stablehlo.reduce(%v3614 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3616 = stablehlo.broadcast_in_dim %v3615, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3617 = stablehlo.multiply %v3611, %v3596 : tensor<32x384x14x14xf32>
    %v3618 = stablehlo.subtract %v3617, %v3613 : tensor<32x384x14x14xf32>
    %v3619 = stablehlo.multiply %v3608, %v3616 : tensor<32x384x14x14xf32>
    %v3620 = stablehlo.subtract %v3618, %v3619 : tensor<32x384x14x14xf32>
    %v3621 = stablehlo.divide %v3607, %v3596 : tensor<32x384x14x14xf32>
    %v3622 = stablehlo.multiply %v3621, %v3620 : tensor<32x384x14x14xf32>
    %v3623 = stablehlo.reshape %v3622 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3624 = stablehlo.reshape %v3623 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3625 = stablehlo.reverse %b8eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3626 = stablehlo.transpose %v3625, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3627 = stablehlo.convolution(%v3624, %v3626)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3628 = stablehlo.reshape %v3627 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3629 = stablehlo.reshape %v3628 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3630 = stablehlo.reshape %v3425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3631 = stablehlo.add %v3629, %v3630 : tensor<32x64x14x14xf32>
    %v3632 = stablehlo.reshape %v3631 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3633 = stablehlo.reshape %v620 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3634 = stablehlo.reshape %v3623 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3635 = stablehlo.transpose %v3633, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3636 = stablehlo.transpose %v3634, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3637 = stablehlo.convolution(%v3635, %v3636)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3638 = stablehlo.transpose %v3637, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3639 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3641 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3642 = stablehlo.reduce(%v3639 init: %v3640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3643 = stablehlo.broadcast_in_dim %v3642, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3644 = stablehlo.divide %v3643, %v3641 : tensor<32x384x14x14xf32>
    %v3645 = stablehlo.subtract %v3639, %v3644 : tensor<32x384x14x14xf32>
    %v3646 = stablehlo.multiply %v3645, %v3645 : tensor<32x384x14x14xf32>
    %v3647 = stablehlo.reduce(%v3646 init: %v3640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3648 = stablehlo.broadcast_in_dim %v3647, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3649 = stablehlo.divide %v3648, %v3641 : tensor<32x384x14x14xf32>
    %v3650 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3651 = stablehlo.add %v3649, %v3650 : tensor<32x384x14x14xf32>
    %v3652 = stablehlo.rsqrt %v3651 : tensor<32x384x14x14xf32>
    %v3653 = stablehlo.multiply %v3645, %v3652 : tensor<32x384x14x14xf32>
    %v3654 = stablehlo.reshape %v3593 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3655 = stablehlo.multiply %v3654, %v3653 : tensor<32x384x14x14xf32>
    %v3656 = stablehlo.reduce(%v3655 init: %v3640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3657 = stablehlo.reshape %v3593 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3659 = stablehlo.reduce(%v3657 init: %v3658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3660 = stablehlo.reshape %v651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3661 = stablehlo.reshape %v3580 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3662 = stablehlo.transpose %v3660, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3663 = stablehlo.transpose %v3661, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3664 = stablehlo.convolution(%v3662, %v3663)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3665 = stablehlo.reshape %v3664 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3666 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3668 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3669 = stablehlo.reduce(%v3666 init: %v3667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3670 = stablehlo.broadcast_in_dim %v3669, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3671 = stablehlo.divide %v3670, %v3668 : tensor<32x384x14x14xf32>
    %v3672 = stablehlo.subtract %v3666, %v3671 : tensor<32x384x14x14xf32>
    %v3673 = stablehlo.multiply %v3672, %v3672 : tensor<32x384x14x14xf32>
    %v3674 = stablehlo.reduce(%v3673 init: %v3667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3675 = stablehlo.broadcast_in_dim %v3674, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3676 = stablehlo.divide %v3675, %v3668 : tensor<32x384x14x14xf32>
    %v3677 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3678 = stablehlo.add %v3676, %v3677 : tensor<32x384x14x14xf32>
    %v3679 = stablehlo.rsqrt %v3678 : tensor<32x384x14x14xf32>
    %v3680 = stablehlo.multiply %v3672, %v3679 : tensor<32x384x14x14xf32>
    %v3681 = stablehlo.reshape %v3550 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3682 = stablehlo.multiply %v3681, %v3680 : tensor<32x384x14x14xf32>
    %v3683 = stablehlo.reduce(%v3682 init: %v3667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3684 = stablehlo.reshape %v3550 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3686 = stablehlo.reduce(%v3684 init: %v3685) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3687 = stablehlo.reshape %v682 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3688 = stablehlo.reshape %v3536 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3689 = stablehlo.transpose %v3687, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3690 = stablehlo.transpose %v3688, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3691 = stablehlo.convolution(%v3689, %v3690)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3692 = stablehlo.transpose %v3691, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3693 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3695 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3696 = stablehlo.reduce(%v3693 init: %v3694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3697 = stablehlo.broadcast_in_dim %v3696, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3698 = stablehlo.divide %v3697, %v3695 : tensor<32x64x14x14xf32>
    %v3699 = stablehlo.subtract %v3693, %v3698 : tensor<32x64x14x14xf32>
    %v3700 = stablehlo.multiply %v3699, %v3699 : tensor<32x64x14x14xf32>
    %v3701 = stablehlo.reduce(%v3700 init: %v3694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3702 = stablehlo.broadcast_in_dim %v3701, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3703 = stablehlo.divide %v3702, %v3695 : tensor<32x64x14x14xf32>
    %v3704 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3705 = stablehlo.add %v3703, %v3704 : tensor<32x64x14x14xf32>
    %v3706 = stablehlo.rsqrt %v3705 : tensor<32x64x14x14xf32>
    %v3707 = stablehlo.multiply %v3699, %v3706 : tensor<32x64x14x14xf32>
    %v3708 = stablehlo.reshape %v3425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3709 = stablehlo.multiply %v3708, %v3707 : tensor<32x64x14x14xf32>
    %v3710 = stablehlo.reduce(%v3709 init: %v3694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3711 = stablehlo.reshape %v3425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3713 = stablehlo.reduce(%v3711 init: %v3712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3714 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3716 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3717 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3718 = stablehlo.reduce(%v3714 init: %v3715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3719 = stablehlo.broadcast_in_dim %v3718, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3720 = stablehlo.divide %v3719, %v3716 : tensor<32x64x14x14xf32>
    %v3721 = stablehlo.subtract %v3714, %v3720 : tensor<32x64x14x14xf32>
    %v3722 = stablehlo.multiply %v3721, %v3721 : tensor<32x64x14x14xf32>
    %v3723 = stablehlo.reduce(%v3722 init: %v3715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3724 = stablehlo.broadcast_in_dim %v3723, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3725 = stablehlo.divide %v3724, %v3716 : tensor<32x64x14x14xf32>
    %v3726 = stablehlo.add %v3725, %v3717 : tensor<32x64x14x14xf32>
    %v3727 = stablehlo.rsqrt %v3726 : tensor<32x64x14x14xf32>
    %v3728 = stablehlo.multiply %v3721, %v3727 : tensor<32x64x14x14xf32>
    %v3729 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3730 = stablehlo.reshape %v3632 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3731 = stablehlo.multiply %v3729, %v3730 : tensor<32x64x14x14xf32>
    %v3732 = stablehlo.reduce(%v3731 init: %v3715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3733 = stablehlo.broadcast_in_dim %v3732, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3734 = stablehlo.multiply %v3728, %v3731 : tensor<32x64x14x14xf32>
    %v3735 = stablehlo.reduce(%v3734 init: %v3715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3736 = stablehlo.broadcast_in_dim %v3735, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3737 = stablehlo.multiply %v3731, %v3716 : tensor<32x64x14x14xf32>
    %v3738 = stablehlo.subtract %v3737, %v3733 : tensor<32x64x14x14xf32>
    %v3739 = stablehlo.multiply %v3728, %v3736 : tensor<32x64x14x14xf32>
    %v3740 = stablehlo.subtract %v3738, %v3739 : tensor<32x64x14x14xf32>
    %v3741 = stablehlo.divide %v3727, %v3716 : tensor<32x64x14x14xf32>
    %v3742 = stablehlo.multiply %v3741, %v3740 : tensor<32x64x14x14xf32>
    %v3743 = stablehlo.reshape %v3742 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3744 = stablehlo.reshape %v3743 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3745 = stablehlo.reverse %b7pW, dims = [2, 3] : tensor<64x192x1x1xf32>
    %v3746 = stablehlo.transpose %v3745, dims = [1, 0, 2, 3] : (tensor<64x192x1x1xf32>) -> tensor<192x64x1x1xf32>
    %v3747 = stablehlo.convolution(%v3744, %v3746)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<192x64x1x1xf32>) -> tensor<32x192x14x14xf32>
    %v3748 = stablehlo.reshape %v3747 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3749 = stablehlo.reshape %v3748 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3750 = stablehlo.reshape %v589 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3751 = stablehlo.constant dense<0.0> : tensor<32x192x14x14xf32>
    %v3752 = stablehlo.constant dense<6.0> : tensor<32x192x14x14xf32>
    %v3753 = stablehlo.compare GT, %v3750, %v3751 : (tensor<32x192x14x14xf32>, tensor<32x192x14x14xf32>) -> tensor<32x192x14x14xi1>
    %v3754 = stablehlo.compare LT, %v3750, %v3752 : (tensor<32x192x14x14xf32>, tensor<32x192x14x14xf32>) -> tensor<32x192x14x14xi1>
    %v3755 = stablehlo.and %v3753, %v3754 : tensor<32x192x14x14xi1>
    %v3756 = stablehlo.select %v3755, %v3749, %v3751 : tensor<32x192x14x14xi1>, tensor<32x192x14x14xf32>
    %v3757 = stablehlo.reshape %v3756 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3758 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3760 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v3761 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v3762 = stablehlo.reduce(%v3758 init: %v3759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3763 = stablehlo.broadcast_in_dim %v3762, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3764 = stablehlo.divide %v3763, %v3760 : tensor<32x192x14x14xf32>
    %v3765 = stablehlo.subtract %v3758, %v3764 : tensor<32x192x14x14xf32>
    %v3766 = stablehlo.multiply %v3765, %v3765 : tensor<32x192x14x14xf32>
    %v3767 = stablehlo.reduce(%v3766 init: %v3759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3768 = stablehlo.broadcast_in_dim %v3767, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3769 = stablehlo.divide %v3768, %v3760 : tensor<32x192x14x14xf32>
    %v3770 = stablehlo.add %v3769, %v3761 : tensor<32x192x14x14xf32>
    %v3771 = stablehlo.rsqrt %v3770 : tensor<32x192x14x14xf32>
    %v3772 = stablehlo.multiply %v3765, %v3771 : tensor<32x192x14x14xf32>
    %v3773 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3774 = stablehlo.reshape %v3757 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3775 = stablehlo.multiply %v3773, %v3774 : tensor<32x192x14x14xf32>
    %v3776 = stablehlo.reduce(%v3775 init: %v3759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3777 = stablehlo.broadcast_in_dim %v3776, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3778 = stablehlo.multiply %v3772, %v3775 : tensor<32x192x14x14xf32>
    %v3779 = stablehlo.reduce(%v3778 init: %v3759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3780 = stablehlo.broadcast_in_dim %v3779, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3781 = stablehlo.multiply %v3775, %v3760 : tensor<32x192x14x14xf32>
    %v3782 = stablehlo.subtract %v3781, %v3777 : tensor<32x192x14x14xf32>
    %v3783 = stablehlo.multiply %v3772, %v3780 : tensor<32x192x14x14xf32>
    %v3784 = stablehlo.subtract %v3782, %v3783 : tensor<32x192x14x14xf32>
    %v3785 = stablehlo.divide %v3771, %v3760 : tensor<32x192x14x14xf32>
    %v3786 = stablehlo.multiply %v3785, %v3784 : tensor<32x192x14x14xf32>
    %v3787 = stablehlo.reshape %v3786 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3788 = stablehlo.reshape %v3787 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3789 = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v3790 = stablehlo.convolution(%v3788, %v3789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x14x14xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v3791 = stablehlo.reshape %v3790 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3792 = stablehlo.reshape %v3791 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3793 = stablehlo.reshape %v558 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3794 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v3795 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v3796 = stablehlo.compare GT, %v3793, %v3794 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v3797 = stablehlo.compare LT, %v3793, %v3795 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v3798 = stablehlo.and %v3796, %v3797 : tensor<32x192x28x28xi1>
    %v3799 = stablehlo.select %v3798, %v3792, %v3794 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v3800 = stablehlo.reshape %v3799 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3801 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3803 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3804 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3805 = stablehlo.reduce(%v3801 init: %v3802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3806 = stablehlo.broadcast_in_dim %v3805, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3807 = stablehlo.divide %v3806, %v3803 : tensor<32x192x28x28xf32>
    %v3808 = stablehlo.subtract %v3801, %v3807 : tensor<32x192x28x28xf32>
    %v3809 = stablehlo.multiply %v3808, %v3808 : tensor<32x192x28x28xf32>
    %v3810 = stablehlo.reduce(%v3809 init: %v3802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3811 = stablehlo.broadcast_in_dim %v3810, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3812 = stablehlo.divide %v3811, %v3803 : tensor<32x192x28x28xf32>
    %v3813 = stablehlo.add %v3812, %v3804 : tensor<32x192x28x28xf32>
    %v3814 = stablehlo.rsqrt %v3813 : tensor<32x192x28x28xf32>
    %v3815 = stablehlo.multiply %v3808, %v3814 : tensor<32x192x28x28xf32>
    %v3816 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3817 = stablehlo.reshape %v3800 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3818 = stablehlo.multiply %v3816, %v3817 : tensor<32x192x28x28xf32>
    %v3819 = stablehlo.reduce(%v3818 init: %v3802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3820 = stablehlo.broadcast_in_dim %v3819, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3821 = stablehlo.multiply %v3815, %v3818 : tensor<32x192x28x28xf32>
    %v3822 = stablehlo.reduce(%v3821 init: %v3802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3823 = stablehlo.broadcast_in_dim %v3822, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3824 = stablehlo.multiply %v3818, %v3803 : tensor<32x192x28x28xf32>
    %v3825 = stablehlo.subtract %v3824, %v3820 : tensor<32x192x28x28xf32>
    %v3826 = stablehlo.multiply %v3815, %v3823 : tensor<32x192x28x28xf32>
    %v3827 = stablehlo.subtract %v3825, %v3826 : tensor<32x192x28x28xf32>
    %v3828 = stablehlo.divide %v3814, %v3803 : tensor<32x192x28x28xf32>
    %v3829 = stablehlo.multiply %v3828, %v3827 : tensor<32x192x28x28xf32>
    %v3830 = stablehlo.reshape %v3829 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3831 = stablehlo.reshape %v3830 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3832 = stablehlo.reverse %b7eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v3833 = stablehlo.transpose %v3832, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v3834 = stablehlo.convolution(%v3831, %v3833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v3835 = stablehlo.reshape %v3834 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3836 = stablehlo.reshape %v533 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3837 = stablehlo.reshape %v3830 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3838 = stablehlo.transpose %v3836, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v3839 = stablehlo.transpose %v3837, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3840 = stablehlo.convolution(%v3838, %v3839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v3841 = stablehlo.transpose %v3840, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3842 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3844 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3845 = stablehlo.reduce(%v3842 init: %v3843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3846 = stablehlo.broadcast_in_dim %v3845, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3847 = stablehlo.divide %v3846, %v3844 : tensor<32x192x28x28xf32>
    %v3848 = stablehlo.subtract %v3842, %v3847 : tensor<32x192x28x28xf32>
    %v3849 = stablehlo.multiply %v3848, %v3848 : tensor<32x192x28x28xf32>
    %v3850 = stablehlo.reduce(%v3849 init: %v3843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3851 = stablehlo.broadcast_in_dim %v3850, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3852 = stablehlo.divide %v3851, %v3844 : tensor<32x192x28x28xf32>
    %v3853 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3854 = stablehlo.add %v3852, %v3853 : tensor<32x192x28x28xf32>
    %v3855 = stablehlo.rsqrt %v3854 : tensor<32x192x28x28xf32>
    %v3856 = stablehlo.multiply %v3848, %v3855 : tensor<32x192x28x28xf32>
    %v3857 = stablehlo.reshape %v3800 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3858 = stablehlo.multiply %v3857, %v3856 : tensor<32x192x28x28xf32>
    %v3859 = stablehlo.reduce(%v3858 init: %v3843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3860 = stablehlo.reshape %v3800 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3862 = stablehlo.reduce(%v3860 init: %v3861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3863 = stablehlo.reshape %v564 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3864 = stablehlo.reshape %v3787 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3866 = stablehlo.pad %v3864, %v3865, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v3867 = stablehlo.transpose %v3863, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3868 = stablehlo.transpose %v3866, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3869 = stablehlo.convolution(%v3867, %v3868)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v3870 = stablehlo.reshape %v3869 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v3871 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3873 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v3874 = stablehlo.reduce(%v3871 init: %v3872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3875 = stablehlo.broadcast_in_dim %v3874, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3876 = stablehlo.divide %v3875, %v3873 : tensor<32x192x14x14xf32>
    %v3877 = stablehlo.subtract %v3871, %v3876 : tensor<32x192x14x14xf32>
    %v3878 = stablehlo.multiply %v3877, %v3877 : tensor<32x192x14x14xf32>
    %v3879 = stablehlo.reduce(%v3878 init: %v3872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3880 = stablehlo.broadcast_in_dim %v3879, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3881 = stablehlo.divide %v3880, %v3873 : tensor<32x192x14x14xf32>
    %v3882 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v3883 = stablehlo.add %v3881, %v3882 : tensor<32x192x14x14xf32>
    %v3884 = stablehlo.rsqrt %v3883 : tensor<32x192x14x14xf32>
    %v3885 = stablehlo.multiply %v3877, %v3884 : tensor<32x192x14x14xf32>
    %v3886 = stablehlo.reshape %v3757 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3887 = stablehlo.multiply %v3886, %v3885 : tensor<32x192x14x14xf32>
    %v3888 = stablehlo.reduce(%v3887 init: %v3872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3889 = stablehlo.reshape %v3757 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3891 = stablehlo.reduce(%v3889 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3892 = stablehlo.reshape %v595 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3893 = stablehlo.reshape %v3743 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3894 = stablehlo.transpose %v3892, dims = [1, 0, 2, 3] : (tensor<32x192x14x14xf32>) -> tensor<192x32x14x14xf32>
    %v3895 = stablehlo.transpose %v3893, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3896 = stablehlo.convolution(%v3894, %v3895)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<192x64x1x1xf32>
    %v3897 = stablehlo.transpose %v3896, dims = [1, 0, 2, 3] : (tensor<192x64x1x1xf32>) -> tensor<64x192x1x1xf32>
    %v3898 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3900 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3901 = stablehlo.reduce(%v3898 init: %v3899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3902 = stablehlo.broadcast_in_dim %v3901, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3903 = stablehlo.divide %v3902, %v3900 : tensor<32x64x14x14xf32>
    %v3904 = stablehlo.subtract %v3898, %v3903 : tensor<32x64x14x14xf32>
    %v3905 = stablehlo.multiply %v3904, %v3904 : tensor<32x64x14x14xf32>
    %v3906 = stablehlo.reduce(%v3905 init: %v3899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3907 = stablehlo.broadcast_in_dim %v3906, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3908 = stablehlo.divide %v3907, %v3900 : tensor<32x64x14x14xf32>
    %v3909 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3910 = stablehlo.add %v3908, %v3909 : tensor<32x64x14x14xf32>
    %v3911 = stablehlo.rsqrt %v3910 : tensor<32x64x14x14xf32>
    %v3912 = stablehlo.multiply %v3904, %v3911 : tensor<32x64x14x14xf32>
    %v3913 = stablehlo.reshape %v3632 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3914 = stablehlo.multiply %v3913, %v3912 : tensor<32x64x14x14xf32>
    %v3915 = stablehlo.reduce(%v3914 init: %v3899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3916 = stablehlo.reshape %v3632 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3918 = stablehlo.reduce(%v3916 init: %v3917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3919 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3921 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v3922 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v3923 = stablehlo.reduce(%v3919 init: %v3920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3924 = stablehlo.broadcast_in_dim %v3923, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3925 = stablehlo.divide %v3924, %v3921 : tensor<32x32x28x28xf32>
    %v3926 = stablehlo.subtract %v3919, %v3925 : tensor<32x32x28x28xf32>
    %v3927 = stablehlo.multiply %v3926, %v3926 : tensor<32x32x28x28xf32>
    %v3928 = stablehlo.reduce(%v3927 init: %v3920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3929 = stablehlo.broadcast_in_dim %v3928, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3930 = stablehlo.divide %v3929, %v3921 : tensor<32x32x28x28xf32>
    %v3931 = stablehlo.add %v3930, %v3922 : tensor<32x32x28x28xf32>
    %v3932 = stablehlo.rsqrt %v3931 : tensor<32x32x28x28xf32>
    %v3933 = stablehlo.multiply %v3926, %v3932 : tensor<32x32x28x28xf32>
    %v3934 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3935 = stablehlo.reshape %v3835 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3936 = stablehlo.multiply %v3934, %v3935 : tensor<32x32x28x28xf32>
    %v3937 = stablehlo.reduce(%v3936 init: %v3920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3938 = stablehlo.broadcast_in_dim %v3937, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3939 = stablehlo.multiply %v3933, %v3936 : tensor<32x32x28x28xf32>
    %v3940 = stablehlo.reduce(%v3939 init: %v3920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3941 = stablehlo.broadcast_in_dim %v3940, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3942 = stablehlo.multiply %v3936, %v3921 : tensor<32x32x28x28xf32>
    %v3943 = stablehlo.subtract %v3942, %v3938 : tensor<32x32x28x28xf32>
    %v3944 = stablehlo.multiply %v3933, %v3941 : tensor<32x32x28x28xf32>
    %v3945 = stablehlo.subtract %v3943, %v3944 : tensor<32x32x28x28xf32>
    %v3946 = stablehlo.divide %v3932, %v3921 : tensor<32x32x28x28xf32>
    %v3947 = stablehlo.multiply %v3946, %v3945 : tensor<32x32x28x28xf32>
    %v3948 = stablehlo.reshape %v3947 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3949 = stablehlo.reshape %v3948 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3950 = stablehlo.reverse %b6pW, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v3951 = stablehlo.transpose %v3950, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3952 = stablehlo.convolution(%v3949, %v3951)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3953 = stablehlo.reshape %v3952 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3954 = stablehlo.reshape %v3953 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3955 = stablehlo.reshape %v498 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3956 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v3957 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v3958 = stablehlo.compare GT, %v3955, %v3956 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v3959 = stablehlo.compare LT, %v3955, %v3957 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v3960 = stablehlo.and %v3958, %v3959 : tensor<32x192x28x28xi1>
    %v3961 = stablehlo.select %v3960, %v3954, %v3956 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v3962 = stablehlo.reshape %v3961 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3963 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3965 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3966 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3967 = stablehlo.reduce(%v3963 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3968 = stablehlo.broadcast_in_dim %v3967, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3969 = stablehlo.divide %v3968, %v3965 : tensor<32x192x28x28xf32>
    %v3970 = stablehlo.subtract %v3963, %v3969 : tensor<32x192x28x28xf32>
    %v3971 = stablehlo.multiply %v3970, %v3970 : tensor<32x192x28x28xf32>
    %v3972 = stablehlo.reduce(%v3971 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3973 = stablehlo.broadcast_in_dim %v3972, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3974 = stablehlo.divide %v3973, %v3965 : tensor<32x192x28x28xf32>
    %v3975 = stablehlo.add %v3974, %v3966 : tensor<32x192x28x28xf32>
    %v3976 = stablehlo.rsqrt %v3975 : tensor<32x192x28x28xf32>
    %v3977 = stablehlo.multiply %v3970, %v3976 : tensor<32x192x28x28xf32>
    %v3978 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3979 = stablehlo.reshape %v3962 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3980 = stablehlo.multiply %v3978, %v3979 : tensor<32x192x28x28xf32>
    %v3981 = stablehlo.reduce(%v3980 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3982 = stablehlo.broadcast_in_dim %v3981, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3983 = stablehlo.multiply %v3977, %v3980 : tensor<32x192x28x28xf32>
    %v3984 = stablehlo.reduce(%v3983 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3985 = stablehlo.broadcast_in_dim %v3984, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3986 = stablehlo.multiply %v3980, %v3965 : tensor<32x192x28x28xf32>
    %v3987 = stablehlo.subtract %v3986, %v3982 : tensor<32x192x28x28xf32>
    %v3988 = stablehlo.multiply %v3977, %v3985 : tensor<32x192x28x28xf32>
    %v3989 = stablehlo.subtract %v3987, %v3988 : tensor<32x192x28x28xf32>
    %v3990 = stablehlo.divide %v3976, %v3965 : tensor<32x192x28x28xf32>
    %v3991 = stablehlo.multiply %v3990, %v3989 : tensor<32x192x28x28xf32>
    %v3992 = stablehlo.reshape %v3991 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3993 = stablehlo.reshape %v3992 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3994 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v3995 = stablehlo.convolution(%v3993, %v3994)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v3996 = stablehlo.reshape %v3995 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3997 = stablehlo.reshape %v3996 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3998 = stablehlo.reshape %v467 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3999 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4000 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4001 = stablehlo.compare GT, %v3998, %v3999 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4002 = stablehlo.compare LT, %v3998, %v4000 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4003 = stablehlo.and %v4001, %v4002 : tensor<32x192x28x28xi1>
    %v4004 = stablehlo.select %v4003, %v3997, %v3999 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4005 = stablehlo.reshape %v4004 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4006 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4008 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4009 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4010 = stablehlo.reduce(%v4006 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4011 = stablehlo.broadcast_in_dim %v4010, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4012 = stablehlo.divide %v4011, %v4008 : tensor<32x192x28x28xf32>
    %v4013 = stablehlo.subtract %v4006, %v4012 : tensor<32x192x28x28xf32>
    %v4014 = stablehlo.multiply %v4013, %v4013 : tensor<32x192x28x28xf32>
    %v4015 = stablehlo.reduce(%v4014 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4016 = stablehlo.broadcast_in_dim %v4015, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4017 = stablehlo.divide %v4016, %v4008 : tensor<32x192x28x28xf32>
    %v4018 = stablehlo.add %v4017, %v4009 : tensor<32x192x28x28xf32>
    %v4019 = stablehlo.rsqrt %v4018 : tensor<32x192x28x28xf32>
    %v4020 = stablehlo.multiply %v4013, %v4019 : tensor<32x192x28x28xf32>
    %v4021 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4022 = stablehlo.reshape %v4005 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4023 = stablehlo.multiply %v4021, %v4022 : tensor<32x192x28x28xf32>
    %v4024 = stablehlo.reduce(%v4023 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4025 = stablehlo.broadcast_in_dim %v4024, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4026 = stablehlo.multiply %v4020, %v4023 : tensor<32x192x28x28xf32>
    %v4027 = stablehlo.reduce(%v4026 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4028 = stablehlo.broadcast_in_dim %v4027, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4029 = stablehlo.multiply %v4023, %v4008 : tensor<32x192x28x28xf32>
    %v4030 = stablehlo.subtract %v4029, %v4025 : tensor<32x192x28x28xf32>
    %v4031 = stablehlo.multiply %v4020, %v4028 : tensor<32x192x28x28xf32>
    %v4032 = stablehlo.subtract %v4030, %v4031 : tensor<32x192x28x28xf32>
    %v4033 = stablehlo.divide %v4019, %v4008 : tensor<32x192x28x28xf32>
    %v4034 = stablehlo.multiply %v4033, %v4032 : tensor<32x192x28x28xf32>
    %v4035 = stablehlo.reshape %v4034 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4036 = stablehlo.reshape %v4035 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4037 = stablehlo.reverse %b6eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4038 = stablehlo.transpose %v4037, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4039 = stablehlo.convolution(%v4036, %v4038)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4040 = stablehlo.reshape %v4039 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4041 = stablehlo.reshape %v4040 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4042 = stablehlo.reshape %v3835 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4043 = stablehlo.add %v4041, %v4042 : tensor<32x32x28x28xf32>
    %v4044 = stablehlo.reshape %v4043 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4045 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4046 = stablehlo.reshape %v4035 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4047 = stablehlo.transpose %v4045, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4048 = stablehlo.transpose %v4046, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4049 = stablehlo.convolution(%v4047, %v4048)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4050 = stablehlo.transpose %v4049, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4051 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4053 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4054 = stablehlo.reduce(%v4051 init: %v4052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4055 = stablehlo.broadcast_in_dim %v4054, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4056 = stablehlo.divide %v4055, %v4053 : tensor<32x192x28x28xf32>
    %v4057 = stablehlo.subtract %v4051, %v4056 : tensor<32x192x28x28xf32>
    %v4058 = stablehlo.multiply %v4057, %v4057 : tensor<32x192x28x28xf32>
    %v4059 = stablehlo.reduce(%v4058 init: %v4052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4060 = stablehlo.broadcast_in_dim %v4059, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4061 = stablehlo.divide %v4060, %v4053 : tensor<32x192x28x28xf32>
    %v4062 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4063 = stablehlo.add %v4061, %v4062 : tensor<32x192x28x28xf32>
    %v4064 = stablehlo.rsqrt %v4063 : tensor<32x192x28x28xf32>
    %v4065 = stablehlo.multiply %v4057, %v4064 : tensor<32x192x28x28xf32>
    %v4066 = stablehlo.reshape %v4005 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4067 = stablehlo.multiply %v4066, %v4065 : tensor<32x192x28x28xf32>
    %v4068 = stablehlo.reduce(%v4067 init: %v4052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4069 = stablehlo.reshape %v4005 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4071 = stablehlo.reduce(%v4069 init: %v4070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4072 = stablehlo.reshape %v473 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4073 = stablehlo.reshape %v3992 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4074 = stablehlo.transpose %v4072, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4075 = stablehlo.transpose %v4073, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4076 = stablehlo.convolution(%v4074, %v4075)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4077 = stablehlo.reshape %v4076 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4078 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4079 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4080 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4081 = stablehlo.reduce(%v4078 init: %v4079) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4082 = stablehlo.broadcast_in_dim %v4081, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4083 = stablehlo.divide %v4082, %v4080 : tensor<32x192x28x28xf32>
    %v4084 = stablehlo.subtract %v4078, %v4083 : tensor<32x192x28x28xf32>
    %v4085 = stablehlo.multiply %v4084, %v4084 : tensor<32x192x28x28xf32>
    %v4086 = stablehlo.reduce(%v4085 init: %v4079) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4087 = stablehlo.broadcast_in_dim %v4086, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4088 = stablehlo.divide %v4087, %v4080 : tensor<32x192x28x28xf32>
    %v4089 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4090 = stablehlo.add %v4088, %v4089 : tensor<32x192x28x28xf32>
    %v4091 = stablehlo.rsqrt %v4090 : tensor<32x192x28x28xf32>
    %v4092 = stablehlo.multiply %v4084, %v4091 : tensor<32x192x28x28xf32>
    %v4093 = stablehlo.reshape %v3962 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4094 = stablehlo.multiply %v4093, %v4092 : tensor<32x192x28x28xf32>
    %v4095 = stablehlo.reduce(%v4094 init: %v4079) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4096 = stablehlo.reshape %v3962 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4098 = stablehlo.reduce(%v4096 init: %v4097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4099 = stablehlo.reshape %v504 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4100 = stablehlo.reshape %v3948 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4101 = stablehlo.transpose %v4099, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4102 = stablehlo.transpose %v4100, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4103 = stablehlo.convolution(%v4101, %v4102)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4104 = stablehlo.transpose %v4103, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4105 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4107 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4108 = stablehlo.reduce(%v4105 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4109 = stablehlo.broadcast_in_dim %v4108, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4110 = stablehlo.divide %v4109, %v4107 : tensor<32x32x28x28xf32>
    %v4111 = stablehlo.subtract %v4105, %v4110 : tensor<32x32x28x28xf32>
    %v4112 = stablehlo.multiply %v4111, %v4111 : tensor<32x32x28x28xf32>
    %v4113 = stablehlo.reduce(%v4112 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4114 = stablehlo.broadcast_in_dim %v4113, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4115 = stablehlo.divide %v4114, %v4107 : tensor<32x32x28x28xf32>
    %v4116 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4117 = stablehlo.add %v4115, %v4116 : tensor<32x32x28x28xf32>
    %v4118 = stablehlo.rsqrt %v4117 : tensor<32x32x28x28xf32>
    %v4119 = stablehlo.multiply %v4111, %v4118 : tensor<32x32x28x28xf32>
    %v4120 = stablehlo.reshape %v3835 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4121 = stablehlo.multiply %v4120, %v4119 : tensor<32x32x28x28xf32>
    %v4122 = stablehlo.reduce(%v4121 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4123 = stablehlo.reshape %v3835 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4125 = stablehlo.reduce(%v4123 init: %v4124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4126 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4128 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4129 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4130 = stablehlo.reduce(%v4126 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4131 = stablehlo.broadcast_in_dim %v4130, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4132 = stablehlo.divide %v4131, %v4128 : tensor<32x32x28x28xf32>
    %v4133 = stablehlo.subtract %v4126, %v4132 : tensor<32x32x28x28xf32>
    %v4134 = stablehlo.multiply %v4133, %v4133 : tensor<32x32x28x28xf32>
    %v4135 = stablehlo.reduce(%v4134 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4136 = stablehlo.broadcast_in_dim %v4135, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4137 = stablehlo.divide %v4136, %v4128 : tensor<32x32x28x28xf32>
    %v4138 = stablehlo.add %v4137, %v4129 : tensor<32x32x28x28xf32>
    %v4139 = stablehlo.rsqrt %v4138 : tensor<32x32x28x28xf32>
    %v4140 = stablehlo.multiply %v4133, %v4139 : tensor<32x32x28x28xf32>
    %v4141 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4142 = stablehlo.reshape %v4044 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4143 = stablehlo.multiply %v4141, %v4142 : tensor<32x32x28x28xf32>
    %v4144 = stablehlo.reduce(%v4143 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4145 = stablehlo.broadcast_in_dim %v4144, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4146 = stablehlo.multiply %v4140, %v4143 : tensor<32x32x28x28xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4148 = stablehlo.broadcast_in_dim %v4147, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4149 = stablehlo.multiply %v4143, %v4128 : tensor<32x32x28x28xf32>
    %v4150 = stablehlo.subtract %v4149, %v4145 : tensor<32x32x28x28xf32>
    %v4151 = stablehlo.multiply %v4140, %v4148 : tensor<32x32x28x28xf32>
    %v4152 = stablehlo.subtract %v4150, %v4151 : tensor<32x32x28x28xf32>
    %v4153 = stablehlo.divide %v4139, %v4128 : tensor<32x32x28x28xf32>
    %v4154 = stablehlo.multiply %v4153, %v4152 : tensor<32x32x28x28xf32>
    %v4155 = stablehlo.reshape %v4154 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4156 = stablehlo.reshape %v4155 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4157 = stablehlo.reverse %b5pW, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4158 = stablehlo.transpose %v4157, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4159 = stablehlo.convolution(%v4156, %v4158)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4160 = stablehlo.reshape %v4159 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4161 = stablehlo.reshape %v4160 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4162 = stablehlo.reshape %v407 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4163 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4164 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4165 = stablehlo.compare GT, %v4162, %v4163 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4166 = stablehlo.compare LT, %v4162, %v4164 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4167 = stablehlo.and %v4165, %v4166 : tensor<32x192x28x28xi1>
    %v4168 = stablehlo.select %v4167, %v4161, %v4163 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4169 = stablehlo.reshape %v4168 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4170 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4172 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4173 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4174 = stablehlo.reduce(%v4170 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4175 = stablehlo.broadcast_in_dim %v4174, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4176 = stablehlo.divide %v4175, %v4172 : tensor<32x192x28x28xf32>
    %v4177 = stablehlo.subtract %v4170, %v4176 : tensor<32x192x28x28xf32>
    %v4178 = stablehlo.multiply %v4177, %v4177 : tensor<32x192x28x28xf32>
    %v4179 = stablehlo.reduce(%v4178 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4180 = stablehlo.broadcast_in_dim %v4179, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4181 = stablehlo.divide %v4180, %v4172 : tensor<32x192x28x28xf32>
    %v4182 = stablehlo.add %v4181, %v4173 : tensor<32x192x28x28xf32>
    %v4183 = stablehlo.rsqrt %v4182 : tensor<32x192x28x28xf32>
    %v4184 = stablehlo.multiply %v4177, %v4183 : tensor<32x192x28x28xf32>
    %v4185 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4186 = stablehlo.reshape %v4169 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4187 = stablehlo.multiply %v4185, %v4186 : tensor<32x192x28x28xf32>
    %v4188 = stablehlo.reduce(%v4187 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4189 = stablehlo.broadcast_in_dim %v4188, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4190 = stablehlo.multiply %v4184, %v4187 : tensor<32x192x28x28xf32>
    %v4191 = stablehlo.reduce(%v4190 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4192 = stablehlo.broadcast_in_dim %v4191, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4193 = stablehlo.multiply %v4187, %v4172 : tensor<32x192x28x28xf32>
    %v4194 = stablehlo.subtract %v4193, %v4189 : tensor<32x192x28x28xf32>
    %v4195 = stablehlo.multiply %v4184, %v4192 : tensor<32x192x28x28xf32>
    %v4196 = stablehlo.subtract %v4194, %v4195 : tensor<32x192x28x28xf32>
    %v4197 = stablehlo.divide %v4183, %v4172 : tensor<32x192x28x28xf32>
    %v4198 = stablehlo.multiply %v4197, %v4196 : tensor<32x192x28x28xf32>
    %v4199 = stablehlo.reshape %v4198 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4200 = stablehlo.reshape %v4199 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4201 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4202 = stablehlo.convolution(%v4200, %v4201)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4203 = stablehlo.reshape %v4202 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4204 = stablehlo.reshape %v4203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4205 = stablehlo.reshape %v376 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4206 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4207 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4208 = stablehlo.compare GT, %v4205, %v4206 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4209 = stablehlo.compare LT, %v4205, %v4207 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4210 = stablehlo.and %v4208, %v4209 : tensor<32x192x28x28xi1>
    %v4211 = stablehlo.select %v4210, %v4204, %v4206 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4212 = stablehlo.reshape %v4211 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4213 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4215 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4216 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4217 = stablehlo.reduce(%v4213 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4218 = stablehlo.broadcast_in_dim %v4217, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4219 = stablehlo.divide %v4218, %v4215 : tensor<32x192x28x28xf32>
    %v4220 = stablehlo.subtract %v4213, %v4219 : tensor<32x192x28x28xf32>
    %v4221 = stablehlo.multiply %v4220, %v4220 : tensor<32x192x28x28xf32>
    %v4222 = stablehlo.reduce(%v4221 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4223 = stablehlo.broadcast_in_dim %v4222, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4224 = stablehlo.divide %v4223, %v4215 : tensor<32x192x28x28xf32>
    %v4225 = stablehlo.add %v4224, %v4216 : tensor<32x192x28x28xf32>
    %v4226 = stablehlo.rsqrt %v4225 : tensor<32x192x28x28xf32>
    %v4227 = stablehlo.multiply %v4220, %v4226 : tensor<32x192x28x28xf32>
    %v4228 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4229 = stablehlo.reshape %v4212 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4230 = stablehlo.multiply %v4228, %v4229 : tensor<32x192x28x28xf32>
    %v4231 = stablehlo.reduce(%v4230 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4232 = stablehlo.broadcast_in_dim %v4231, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4233 = stablehlo.multiply %v4227, %v4230 : tensor<32x192x28x28xf32>
    %v4234 = stablehlo.reduce(%v4233 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4235 = stablehlo.broadcast_in_dim %v4234, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4236 = stablehlo.multiply %v4230, %v4215 : tensor<32x192x28x28xf32>
    %v4237 = stablehlo.subtract %v4236, %v4232 : tensor<32x192x28x28xf32>
    %v4238 = stablehlo.multiply %v4227, %v4235 : tensor<32x192x28x28xf32>
    %v4239 = stablehlo.subtract %v4237, %v4238 : tensor<32x192x28x28xf32>
    %v4240 = stablehlo.divide %v4226, %v4215 : tensor<32x192x28x28xf32>
    %v4241 = stablehlo.multiply %v4240, %v4239 : tensor<32x192x28x28xf32>
    %v4242 = stablehlo.reshape %v4241 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4243 = stablehlo.reshape %v4242 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4244 = stablehlo.reverse %b5eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4245 = stablehlo.transpose %v4244, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4246 = stablehlo.convolution(%v4243, %v4245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4247 = stablehlo.reshape %v4246 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4248 = stablehlo.reshape %v4247 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4249 = stablehlo.reshape %v4044 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4250 = stablehlo.add %v4248, %v4249 : tensor<32x32x28x28xf32>
    %v4251 = stablehlo.reshape %v4250 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4252 = stablehlo.reshape %v351 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4253 = stablehlo.reshape %v4242 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4254 = stablehlo.transpose %v4252, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4255 = stablehlo.transpose %v4253, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4256 = stablehlo.convolution(%v4254, %v4255)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4257 = stablehlo.transpose %v4256, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4258 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4260 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4261 = stablehlo.reduce(%v4258 init: %v4259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4262 = stablehlo.broadcast_in_dim %v4261, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4263 = stablehlo.divide %v4262, %v4260 : tensor<32x192x28x28xf32>
    %v4264 = stablehlo.subtract %v4258, %v4263 : tensor<32x192x28x28xf32>
    %v4265 = stablehlo.multiply %v4264, %v4264 : tensor<32x192x28x28xf32>
    %v4266 = stablehlo.reduce(%v4265 init: %v4259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4267 = stablehlo.broadcast_in_dim %v4266, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4268 = stablehlo.divide %v4267, %v4260 : tensor<32x192x28x28xf32>
    %v4269 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4270 = stablehlo.add %v4268, %v4269 : tensor<32x192x28x28xf32>
    %v4271 = stablehlo.rsqrt %v4270 : tensor<32x192x28x28xf32>
    %v4272 = stablehlo.multiply %v4264, %v4271 : tensor<32x192x28x28xf32>
    %v4273 = stablehlo.reshape %v4212 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4274 = stablehlo.multiply %v4273, %v4272 : tensor<32x192x28x28xf32>
    %v4275 = stablehlo.reduce(%v4274 init: %v4259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4276 = stablehlo.reshape %v4212 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4278 = stablehlo.reduce(%v4276 init: %v4277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4279 = stablehlo.reshape %v382 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4280 = stablehlo.reshape %v4199 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4281 = stablehlo.transpose %v4279, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4282 = stablehlo.transpose %v4280, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4283 = stablehlo.convolution(%v4281, %v4282)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4284 = stablehlo.reshape %v4283 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4285 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4287 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4288 = stablehlo.reduce(%v4285 init: %v4286) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4289 = stablehlo.broadcast_in_dim %v4288, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4290 = stablehlo.divide %v4289, %v4287 : tensor<32x192x28x28xf32>
    %v4291 = stablehlo.subtract %v4285, %v4290 : tensor<32x192x28x28xf32>
    %v4292 = stablehlo.multiply %v4291, %v4291 : tensor<32x192x28x28xf32>
    %v4293 = stablehlo.reduce(%v4292 init: %v4286) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4294 = stablehlo.broadcast_in_dim %v4293, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4295 = stablehlo.divide %v4294, %v4287 : tensor<32x192x28x28xf32>
    %v4296 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4297 = stablehlo.add %v4295, %v4296 : tensor<32x192x28x28xf32>
    %v4298 = stablehlo.rsqrt %v4297 : tensor<32x192x28x28xf32>
    %v4299 = stablehlo.multiply %v4291, %v4298 : tensor<32x192x28x28xf32>
    %v4300 = stablehlo.reshape %v4169 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4301 = stablehlo.multiply %v4300, %v4299 : tensor<32x192x28x28xf32>
    %v4302 = stablehlo.reduce(%v4301 init: %v4286) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4303 = stablehlo.reshape %v4169 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4305 = stablehlo.reduce(%v4303 init: %v4304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4306 = stablehlo.reshape %v413 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4307 = stablehlo.reshape %v4155 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4308 = stablehlo.transpose %v4306, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4309 = stablehlo.transpose %v4307, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4310 = stablehlo.convolution(%v4308, %v4309)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4311 = stablehlo.transpose %v4310, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4312 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4314 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4315 = stablehlo.reduce(%v4312 init: %v4313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4316 = stablehlo.broadcast_in_dim %v4315, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4317 = stablehlo.divide %v4316, %v4314 : tensor<32x32x28x28xf32>
    %v4318 = stablehlo.subtract %v4312, %v4317 : tensor<32x32x28x28xf32>
    %v4319 = stablehlo.multiply %v4318, %v4318 : tensor<32x32x28x28xf32>
    %v4320 = stablehlo.reduce(%v4319 init: %v4313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4321 = stablehlo.broadcast_in_dim %v4320, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4322 = stablehlo.divide %v4321, %v4314 : tensor<32x32x28x28xf32>
    %v4323 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4324 = stablehlo.add %v4322, %v4323 : tensor<32x32x28x28xf32>
    %v4325 = stablehlo.rsqrt %v4324 : tensor<32x32x28x28xf32>
    %v4326 = stablehlo.multiply %v4318, %v4325 : tensor<32x32x28x28xf32>
    %v4327 = stablehlo.reshape %v4044 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4328 = stablehlo.multiply %v4327, %v4326 : tensor<32x32x28x28xf32>
    %v4329 = stablehlo.reduce(%v4328 init: %v4313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4330 = stablehlo.reshape %v4044 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4332 = stablehlo.reduce(%v4330 init: %v4331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4333 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4335 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4336 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4337 = stablehlo.reduce(%v4333 init: %v4334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4338 = stablehlo.broadcast_in_dim %v4337, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4339 = stablehlo.divide %v4338, %v4335 : tensor<32x32x28x28xf32>
    %v4340 = stablehlo.subtract %v4333, %v4339 : tensor<32x32x28x28xf32>
    %v4341 = stablehlo.multiply %v4340, %v4340 : tensor<32x32x28x28xf32>
    %v4342 = stablehlo.reduce(%v4341 init: %v4334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4343 = stablehlo.broadcast_in_dim %v4342, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4344 = stablehlo.divide %v4343, %v4335 : tensor<32x32x28x28xf32>
    %v4345 = stablehlo.add %v4344, %v4336 : tensor<32x32x28x28xf32>
    %v4346 = stablehlo.rsqrt %v4345 : tensor<32x32x28x28xf32>
    %v4347 = stablehlo.multiply %v4340, %v4346 : tensor<32x32x28x28xf32>
    %v4348 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4349 = stablehlo.reshape %v4251 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4350 = stablehlo.multiply %v4348, %v4349 : tensor<32x32x28x28xf32>
    %v4351 = stablehlo.reduce(%v4350 init: %v4334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4352 = stablehlo.broadcast_in_dim %v4351, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4353 = stablehlo.multiply %v4347, %v4350 : tensor<32x32x28x28xf32>
    %v4354 = stablehlo.reduce(%v4353 init: %v4334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4355 = stablehlo.broadcast_in_dim %v4354, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4356 = stablehlo.multiply %v4350, %v4335 : tensor<32x32x28x28xf32>
    %v4357 = stablehlo.subtract %v4356, %v4352 : tensor<32x32x28x28xf32>
    %v4358 = stablehlo.multiply %v4347, %v4355 : tensor<32x32x28x28xf32>
    %v4359 = stablehlo.subtract %v4357, %v4358 : tensor<32x32x28x28xf32>
    %v4360 = stablehlo.divide %v4346, %v4335 : tensor<32x32x28x28xf32>
    %v4361 = stablehlo.multiply %v4360, %v4359 : tensor<32x32x28x28xf32>
    %v4362 = stablehlo.reshape %v4361 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4363 = stablehlo.reshape %v4362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4364 = stablehlo.reverse %b4pW, dims = [2, 3] : tensor<32x144x1x1xf32>
    %v4365 = stablehlo.transpose %v4364, dims = [1, 0, 2, 3] : (tensor<32x144x1x1xf32>) -> tensor<144x32x1x1xf32>
    %v4366 = stablehlo.convolution(%v4363, %v4365)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<144x32x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v4367 = stablehlo.reshape %v4366 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4368 = stablehlo.reshape %v4367 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4369 = stablehlo.reshape %v320 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4370 = stablehlo.constant dense<0.0> : tensor<32x144x28x28xf32>
    %v4371 = stablehlo.constant dense<6.0> : tensor<32x144x28x28xf32>
    %v4372 = stablehlo.compare GT, %v4369, %v4370 : (tensor<32x144x28x28xf32>, tensor<32x144x28x28xf32>) -> tensor<32x144x28x28xi1>
    %v4373 = stablehlo.compare LT, %v4369, %v4371 : (tensor<32x144x28x28xf32>, tensor<32x144x28x28xf32>) -> tensor<32x144x28x28xi1>
    %v4374 = stablehlo.and %v4372, %v4373 : tensor<32x144x28x28xi1>
    %v4375 = stablehlo.select %v4374, %v4368, %v4370 : tensor<32x144x28x28xi1>, tensor<32x144x28x28xf32>
    %v4376 = stablehlo.reshape %v4375 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4377 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4379 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v4380 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4381 = stablehlo.reduce(%v4377 init: %v4378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4382 = stablehlo.broadcast_in_dim %v4381, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4383 = stablehlo.divide %v4382, %v4379 : tensor<32x144x28x28xf32>
    %v4384 = stablehlo.subtract %v4377, %v4383 : tensor<32x144x28x28xf32>
    %v4385 = stablehlo.multiply %v4384, %v4384 : tensor<32x144x28x28xf32>
    %v4386 = stablehlo.reduce(%v4385 init: %v4378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4387 = stablehlo.broadcast_in_dim %v4386, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4388 = stablehlo.divide %v4387, %v4379 : tensor<32x144x28x28xf32>
    %v4389 = stablehlo.add %v4388, %v4380 : tensor<32x144x28x28xf32>
    %v4390 = stablehlo.rsqrt %v4389 : tensor<32x144x28x28xf32>
    %v4391 = stablehlo.multiply %v4384, %v4390 : tensor<32x144x28x28xf32>
    %v4392 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4393 = stablehlo.reshape %v4376 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4394 = stablehlo.multiply %v4392, %v4393 : tensor<32x144x28x28xf32>
    %v4395 = stablehlo.reduce(%v4394 init: %v4378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4396 = stablehlo.broadcast_in_dim %v4395, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4397 = stablehlo.multiply %v4391, %v4394 : tensor<32x144x28x28xf32>
    %v4398 = stablehlo.reduce(%v4397 init: %v4378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4399 = stablehlo.broadcast_in_dim %v4398, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4400 = stablehlo.multiply %v4394, %v4379 : tensor<32x144x28x28xf32>
    %v4401 = stablehlo.subtract %v4400, %v4396 : tensor<32x144x28x28xf32>
    %v4402 = stablehlo.multiply %v4391, %v4399 : tensor<32x144x28x28xf32>
    %v4403 = stablehlo.subtract %v4401, %v4402 : tensor<32x144x28x28xf32>
    %v4404 = stablehlo.divide %v4390, %v4379 : tensor<32x144x28x28xf32>
    %v4405 = stablehlo.multiply %v4404, %v4403 : tensor<32x144x28x28xf32>
    %v4406 = stablehlo.reshape %v4405 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4407 = stablehlo.reshape %v4406 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4408 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4409 = stablehlo.convolution(%v4407, %v4408)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x28x28xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4410 = stablehlo.reshape %v4409 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4411 = stablehlo.reshape %v4410 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4412 = stablehlo.reshape %v289 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4413 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v4414 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v4415 = stablehlo.compare GT, %v4412, %v4413 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4416 = stablehlo.compare LT, %v4412, %v4414 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4417 = stablehlo.and %v4415, %v4416 : tensor<32x144x56x56xi1>
    %v4418 = stablehlo.select %v4417, %v4411, %v4413 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v4419 = stablehlo.reshape %v4418 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4420 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4422 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4423 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4424 = stablehlo.reduce(%v4420 init: %v4421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4425 = stablehlo.broadcast_in_dim %v4424, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4426 = stablehlo.divide %v4425, %v4422 : tensor<32x144x56x56xf32>
    %v4427 = stablehlo.subtract %v4420, %v4426 : tensor<32x144x56x56xf32>
    %v4428 = stablehlo.multiply %v4427, %v4427 : tensor<32x144x56x56xf32>
    %v4429 = stablehlo.reduce(%v4428 init: %v4421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4430 = stablehlo.broadcast_in_dim %v4429, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4431 = stablehlo.divide %v4430, %v4422 : tensor<32x144x56x56xf32>
    %v4432 = stablehlo.add %v4431, %v4423 : tensor<32x144x56x56xf32>
    %v4433 = stablehlo.rsqrt %v4432 : tensor<32x144x56x56xf32>
    %v4434 = stablehlo.multiply %v4427, %v4433 : tensor<32x144x56x56xf32>
    %v4435 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4436 = stablehlo.reshape %v4419 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4437 = stablehlo.multiply %v4435, %v4436 : tensor<32x144x56x56xf32>
    %v4438 = stablehlo.reduce(%v4437 init: %v4421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4439 = stablehlo.broadcast_in_dim %v4438, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4440 = stablehlo.multiply %v4434, %v4437 : tensor<32x144x56x56xf32>
    %v4441 = stablehlo.reduce(%v4440 init: %v4421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4442 = stablehlo.broadcast_in_dim %v4441, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4443 = stablehlo.multiply %v4437, %v4422 : tensor<32x144x56x56xf32>
    %v4444 = stablehlo.subtract %v4443, %v4439 : tensor<32x144x56x56xf32>
    %v4445 = stablehlo.multiply %v4434, %v4442 : tensor<32x144x56x56xf32>
    %v4446 = stablehlo.subtract %v4444, %v4445 : tensor<32x144x56x56xf32>
    %v4447 = stablehlo.divide %v4433, %v4422 : tensor<32x144x56x56xf32>
    %v4448 = stablehlo.multiply %v4447, %v4446 : tensor<32x144x56x56xf32>
    %v4449 = stablehlo.reshape %v4448 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4450 = stablehlo.reshape %v4449 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4451 = stablehlo.reverse %b4eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v4452 = stablehlo.transpose %v4451, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4453 = stablehlo.convolution(%v4450, %v4452)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4454 = stablehlo.reshape %v4453 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4455 = stablehlo.reshape %v264 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4456 = stablehlo.reshape %v4449 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4457 = stablehlo.transpose %v4455, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4458 = stablehlo.transpose %v4456, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4459 = stablehlo.convolution(%v4457, %v4458)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4460 = stablehlo.transpose %v4459, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4461 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4463 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4464 = stablehlo.reduce(%v4461 init: %v4462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4465 = stablehlo.broadcast_in_dim %v4464, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4466 = stablehlo.divide %v4465, %v4463 : tensor<32x144x56x56xf32>
    %v4467 = stablehlo.subtract %v4461, %v4466 : tensor<32x144x56x56xf32>
    %v4468 = stablehlo.multiply %v4467, %v4467 : tensor<32x144x56x56xf32>
    %v4469 = stablehlo.reduce(%v4468 init: %v4462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4470 = stablehlo.broadcast_in_dim %v4469, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4471 = stablehlo.divide %v4470, %v4463 : tensor<32x144x56x56xf32>
    %v4472 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4473 = stablehlo.add %v4471, %v4472 : tensor<32x144x56x56xf32>
    %v4474 = stablehlo.rsqrt %v4473 : tensor<32x144x56x56xf32>
    %v4475 = stablehlo.multiply %v4467, %v4474 : tensor<32x144x56x56xf32>
    %v4476 = stablehlo.reshape %v4419 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4477 = stablehlo.multiply %v4476, %v4475 : tensor<32x144x56x56xf32>
    %v4478 = stablehlo.reduce(%v4477 init: %v4462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4479 = stablehlo.reshape %v4419 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4481 = stablehlo.reduce(%v4479 init: %v4480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4482 = stablehlo.reshape %v295 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4483 = stablehlo.reshape %v4406 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4485 = stablehlo.pad %v4483, %v4484, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4486 = stablehlo.transpose %v4482, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4487 = stablehlo.transpose %v4485, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4488 = stablehlo.convolution(%v4486, %v4487)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4489 = stablehlo.reshape %v4488 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4490 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4492 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v4493 = stablehlo.reduce(%v4490 init: %v4491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4494 = stablehlo.broadcast_in_dim %v4493, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4495 = stablehlo.divide %v4494, %v4492 : tensor<32x144x28x28xf32>
    %v4496 = stablehlo.subtract %v4490, %v4495 : tensor<32x144x28x28xf32>
    %v4497 = stablehlo.multiply %v4496, %v4496 : tensor<32x144x28x28xf32>
    %v4498 = stablehlo.reduce(%v4497 init: %v4491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4499 = stablehlo.broadcast_in_dim %v4498, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4500 = stablehlo.divide %v4499, %v4492 : tensor<32x144x28x28xf32>
    %v4501 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4502 = stablehlo.add %v4500, %v4501 : tensor<32x144x28x28xf32>
    %v4503 = stablehlo.rsqrt %v4502 : tensor<32x144x28x28xf32>
    %v4504 = stablehlo.multiply %v4496, %v4503 : tensor<32x144x28x28xf32>
    %v4505 = stablehlo.reshape %v4376 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4506 = stablehlo.multiply %v4505, %v4504 : tensor<32x144x28x28xf32>
    %v4507 = stablehlo.reduce(%v4506 init: %v4491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4508 = stablehlo.reshape %v4376 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4509 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4510 = stablehlo.reduce(%v4508 init: %v4509) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4511 = stablehlo.reshape %v326 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4512 = stablehlo.reshape %v4362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4513 = stablehlo.transpose %v4511, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v4514 = stablehlo.transpose %v4512, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4515 = stablehlo.convolution(%v4513, %v4514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<144x32x1x1xf32>
    %v4516 = stablehlo.transpose %v4515, dims = [1, 0, 2, 3] : (tensor<144x32x1x1xf32>) -> tensor<32x144x1x1xf32>
    %v4517 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4519 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4520 = stablehlo.reduce(%v4517 init: %v4518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4521 = stablehlo.broadcast_in_dim %v4520, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4522 = stablehlo.divide %v4521, %v4519 : tensor<32x32x28x28xf32>
    %v4523 = stablehlo.subtract %v4517, %v4522 : tensor<32x32x28x28xf32>
    %v4524 = stablehlo.multiply %v4523, %v4523 : tensor<32x32x28x28xf32>
    %v4525 = stablehlo.reduce(%v4524 init: %v4518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4526 = stablehlo.broadcast_in_dim %v4525, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4527 = stablehlo.divide %v4526, %v4519 : tensor<32x32x28x28xf32>
    %v4528 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4529 = stablehlo.add %v4527, %v4528 : tensor<32x32x28x28xf32>
    %v4530 = stablehlo.rsqrt %v4529 : tensor<32x32x28x28xf32>
    %v4531 = stablehlo.multiply %v4523, %v4530 : tensor<32x32x28x28xf32>
    %v4532 = stablehlo.reshape %v4251 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4533 = stablehlo.multiply %v4532, %v4531 : tensor<32x32x28x28xf32>
    %v4534 = stablehlo.reduce(%v4533 init: %v4518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4535 = stablehlo.reshape %v4251 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4537 = stablehlo.reduce(%v4535 init: %v4536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4538 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4540 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4541 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4542 = stablehlo.reduce(%v4538 init: %v4539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4543 = stablehlo.broadcast_in_dim %v4542, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4544 = stablehlo.divide %v4543, %v4540 : tensor<32x24x56x56xf32>
    %v4545 = stablehlo.subtract %v4538, %v4544 : tensor<32x24x56x56xf32>
    %v4546 = stablehlo.multiply %v4545, %v4545 : tensor<32x24x56x56xf32>
    %v4547 = stablehlo.reduce(%v4546 init: %v4539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4548 = stablehlo.broadcast_in_dim %v4547, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4549 = stablehlo.divide %v4548, %v4540 : tensor<32x24x56x56xf32>
    %v4550 = stablehlo.add %v4549, %v4541 : tensor<32x24x56x56xf32>
    %v4551 = stablehlo.rsqrt %v4550 : tensor<32x24x56x56xf32>
    %v4552 = stablehlo.multiply %v4545, %v4551 : tensor<32x24x56x56xf32>
    %v4553 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4554 = stablehlo.reshape %v4454 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4555 = stablehlo.multiply %v4553, %v4554 : tensor<32x24x56x56xf32>
    %v4556 = stablehlo.reduce(%v4555 init: %v4539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4557 = stablehlo.broadcast_in_dim %v4556, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4558 = stablehlo.multiply %v4552, %v4555 : tensor<32x24x56x56xf32>
    %v4559 = stablehlo.reduce(%v4558 init: %v4539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4560 = stablehlo.broadcast_in_dim %v4559, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4561 = stablehlo.multiply %v4555, %v4540 : tensor<32x24x56x56xf32>
    %v4562 = stablehlo.subtract %v4561, %v4557 : tensor<32x24x56x56xf32>
    %v4563 = stablehlo.multiply %v4552, %v4560 : tensor<32x24x56x56xf32>
    %v4564 = stablehlo.subtract %v4562, %v4563 : tensor<32x24x56x56xf32>
    %v4565 = stablehlo.divide %v4551, %v4540 : tensor<32x24x56x56xf32>
    %v4566 = stablehlo.multiply %v4565, %v4564 : tensor<32x24x56x56xf32>
    %v4567 = stablehlo.reshape %v4566 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4568 = stablehlo.reshape %v4567 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4569 = stablehlo.reverse %b3pW, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v4570 = stablehlo.transpose %v4569, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4571 = stablehlo.convolution(%v4568, %v4570)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v4572 = stablehlo.reshape %v4571 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4573 = stablehlo.reshape %v4572 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4574 = stablehlo.reshape %v229 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4575 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v4576 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v4577 = stablehlo.compare GT, %v4574, %v4575 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4578 = stablehlo.compare LT, %v4574, %v4576 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4579 = stablehlo.and %v4577, %v4578 : tensor<32x144x56x56xi1>
    %v4580 = stablehlo.select %v4579, %v4573, %v4575 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v4581 = stablehlo.reshape %v4580 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4582 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4584 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4585 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4586 = stablehlo.reduce(%v4582 init: %v4583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4587 = stablehlo.broadcast_in_dim %v4586, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4588 = stablehlo.divide %v4587, %v4584 : tensor<32x144x56x56xf32>
    %v4589 = stablehlo.subtract %v4582, %v4588 : tensor<32x144x56x56xf32>
    %v4590 = stablehlo.multiply %v4589, %v4589 : tensor<32x144x56x56xf32>
    %v4591 = stablehlo.reduce(%v4590 init: %v4583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4592 = stablehlo.broadcast_in_dim %v4591, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4593 = stablehlo.divide %v4592, %v4584 : tensor<32x144x56x56xf32>
    %v4594 = stablehlo.add %v4593, %v4585 : tensor<32x144x56x56xf32>
    %v4595 = stablehlo.rsqrt %v4594 : tensor<32x144x56x56xf32>
    %v4596 = stablehlo.multiply %v4589, %v4595 : tensor<32x144x56x56xf32>
    %v4597 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4598 = stablehlo.reshape %v4581 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4599 = stablehlo.multiply %v4597, %v4598 : tensor<32x144x56x56xf32>
    %v4600 = stablehlo.reduce(%v4599 init: %v4583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4601 = stablehlo.broadcast_in_dim %v4600, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4602 = stablehlo.multiply %v4596, %v4599 : tensor<32x144x56x56xf32>
    %v4603 = stablehlo.reduce(%v4602 init: %v4583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4604 = stablehlo.broadcast_in_dim %v4603, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4605 = stablehlo.multiply %v4599, %v4584 : tensor<32x144x56x56xf32>
    %v4606 = stablehlo.subtract %v4605, %v4601 : tensor<32x144x56x56xf32>
    %v4607 = stablehlo.multiply %v4596, %v4604 : tensor<32x144x56x56xf32>
    %v4608 = stablehlo.subtract %v4606, %v4607 : tensor<32x144x56x56xf32>
    %v4609 = stablehlo.divide %v4595, %v4584 : tensor<32x144x56x56xf32>
    %v4610 = stablehlo.multiply %v4609, %v4608 : tensor<32x144x56x56xf32>
    %v4611 = stablehlo.reshape %v4610 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4612 = stablehlo.reshape %v4611 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4613 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4614 = stablehlo.convolution(%v4612, %v4613)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4615 = stablehlo.reshape %v4614 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4616 = stablehlo.reshape %v4615 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4617 = stablehlo.reshape %v198 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4618 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v4619 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v4620 = stablehlo.compare GT, %v4617, %v4618 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4621 = stablehlo.compare LT, %v4617, %v4619 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4622 = stablehlo.and %v4620, %v4621 : tensor<32x144x56x56xi1>
    %v4623 = stablehlo.select %v4622, %v4616, %v4618 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v4624 = stablehlo.reshape %v4623 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4625 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4627 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4628 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4629 = stablehlo.reduce(%v4625 init: %v4626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4630 = stablehlo.broadcast_in_dim %v4629, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4631 = stablehlo.divide %v4630, %v4627 : tensor<32x144x56x56xf32>
    %v4632 = stablehlo.subtract %v4625, %v4631 : tensor<32x144x56x56xf32>
    %v4633 = stablehlo.multiply %v4632, %v4632 : tensor<32x144x56x56xf32>
    %v4634 = stablehlo.reduce(%v4633 init: %v4626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4635 = stablehlo.broadcast_in_dim %v4634, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4636 = stablehlo.divide %v4635, %v4627 : tensor<32x144x56x56xf32>
    %v4637 = stablehlo.add %v4636, %v4628 : tensor<32x144x56x56xf32>
    %v4638 = stablehlo.rsqrt %v4637 : tensor<32x144x56x56xf32>
    %v4639 = stablehlo.multiply %v4632, %v4638 : tensor<32x144x56x56xf32>
    %v4640 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4641 = stablehlo.reshape %v4624 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4642 = stablehlo.multiply %v4640, %v4641 : tensor<32x144x56x56xf32>
    %v4643 = stablehlo.reduce(%v4642 init: %v4626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4644 = stablehlo.broadcast_in_dim %v4643, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4645 = stablehlo.multiply %v4639, %v4642 : tensor<32x144x56x56xf32>
    %v4646 = stablehlo.reduce(%v4645 init: %v4626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4647 = stablehlo.broadcast_in_dim %v4646, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4648 = stablehlo.multiply %v4642, %v4627 : tensor<32x144x56x56xf32>
    %v4649 = stablehlo.subtract %v4648, %v4644 : tensor<32x144x56x56xf32>
    %v4650 = stablehlo.multiply %v4639, %v4647 : tensor<32x144x56x56xf32>
    %v4651 = stablehlo.subtract %v4649, %v4650 : tensor<32x144x56x56xf32>
    %v4652 = stablehlo.divide %v4638, %v4627 : tensor<32x144x56x56xf32>
    %v4653 = stablehlo.multiply %v4652, %v4651 : tensor<32x144x56x56xf32>
    %v4654 = stablehlo.reshape %v4653 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4655 = stablehlo.reshape %v4654 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4656 = stablehlo.reverse %b3eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v4657 = stablehlo.transpose %v4656, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4658 = stablehlo.convolution(%v4655, %v4657)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4659 = stablehlo.reshape %v4658 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4660 = stablehlo.reshape %v4659 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4661 = stablehlo.reshape %v4454 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4662 = stablehlo.add %v4660, %v4661 : tensor<32x24x56x56xf32>
    %v4663 = stablehlo.reshape %v4662 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4664 = stablehlo.reshape %v173 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4665 = stablehlo.reshape %v4654 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4666 = stablehlo.transpose %v4664, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4667 = stablehlo.transpose %v4665, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4668 = stablehlo.convolution(%v4666, %v4667)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4669 = stablehlo.transpose %v4668, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4670 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4672 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4673 = stablehlo.reduce(%v4670 init: %v4671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4674 = stablehlo.broadcast_in_dim %v4673, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4675 = stablehlo.divide %v4674, %v4672 : tensor<32x144x56x56xf32>
    %v4676 = stablehlo.subtract %v4670, %v4675 : tensor<32x144x56x56xf32>
    %v4677 = stablehlo.multiply %v4676, %v4676 : tensor<32x144x56x56xf32>
    %v4678 = stablehlo.reduce(%v4677 init: %v4671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4679 = stablehlo.broadcast_in_dim %v4678, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4680 = stablehlo.divide %v4679, %v4672 : tensor<32x144x56x56xf32>
    %v4681 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4682 = stablehlo.add %v4680, %v4681 : tensor<32x144x56x56xf32>
    %v4683 = stablehlo.rsqrt %v4682 : tensor<32x144x56x56xf32>
    %v4684 = stablehlo.multiply %v4676, %v4683 : tensor<32x144x56x56xf32>
    %v4685 = stablehlo.reshape %v4624 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4686 = stablehlo.multiply %v4685, %v4684 : tensor<32x144x56x56xf32>
    %v4687 = stablehlo.reduce(%v4686 init: %v4671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4688 = stablehlo.reshape %v4624 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4690 = stablehlo.reduce(%v4688 init: %v4689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4691 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4692 = stablehlo.reshape %v4611 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4693 = stablehlo.transpose %v4691, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4694 = stablehlo.transpose %v4692, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4695 = stablehlo.convolution(%v4693, %v4694)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4696 = stablehlo.reshape %v4695 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4697 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4699 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4700 = stablehlo.reduce(%v4697 init: %v4698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4701 = stablehlo.broadcast_in_dim %v4700, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4702 = stablehlo.divide %v4701, %v4699 : tensor<32x144x56x56xf32>
    %v4703 = stablehlo.subtract %v4697, %v4702 : tensor<32x144x56x56xf32>
    %v4704 = stablehlo.multiply %v4703, %v4703 : tensor<32x144x56x56xf32>
    %v4705 = stablehlo.reduce(%v4704 init: %v4698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4706 = stablehlo.broadcast_in_dim %v4705, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4707 = stablehlo.divide %v4706, %v4699 : tensor<32x144x56x56xf32>
    %v4708 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4709 = stablehlo.add %v4707, %v4708 : tensor<32x144x56x56xf32>
    %v4710 = stablehlo.rsqrt %v4709 : tensor<32x144x56x56xf32>
    %v4711 = stablehlo.multiply %v4703, %v4710 : tensor<32x144x56x56xf32>
    %v4712 = stablehlo.reshape %v4581 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4713 = stablehlo.multiply %v4712, %v4711 : tensor<32x144x56x56xf32>
    %v4714 = stablehlo.reduce(%v4713 init: %v4698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4715 = stablehlo.reshape %v4581 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4717 = stablehlo.reduce(%v4715 init: %v4716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4718 = stablehlo.reshape %v235 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4719 = stablehlo.reshape %v4567 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4720 = stablehlo.transpose %v4718, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4721 = stablehlo.transpose %v4719, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4722 = stablehlo.convolution(%v4720, %v4721)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v4723 = stablehlo.transpose %v4722, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4724 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4726 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4727 = stablehlo.reduce(%v4724 init: %v4725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4728 = stablehlo.broadcast_in_dim %v4727, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4729 = stablehlo.divide %v4728, %v4726 : tensor<32x24x56x56xf32>
    %v4730 = stablehlo.subtract %v4724, %v4729 : tensor<32x24x56x56xf32>
    %v4731 = stablehlo.multiply %v4730, %v4730 : tensor<32x24x56x56xf32>
    %v4732 = stablehlo.reduce(%v4731 init: %v4725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4733 = stablehlo.broadcast_in_dim %v4732, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4734 = stablehlo.divide %v4733, %v4726 : tensor<32x24x56x56xf32>
    %v4735 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4736 = stablehlo.add %v4734, %v4735 : tensor<32x24x56x56xf32>
    %v4737 = stablehlo.rsqrt %v4736 : tensor<32x24x56x56xf32>
    %v4738 = stablehlo.multiply %v4730, %v4737 : tensor<32x24x56x56xf32>
    %v4739 = stablehlo.reshape %v4454 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4740 = stablehlo.multiply %v4739, %v4738 : tensor<32x24x56x56xf32>
    %v4741 = stablehlo.reduce(%v4740 init: %v4725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4742 = stablehlo.reshape %v4454 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4744 = stablehlo.reduce(%v4742 init: %v4743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4745 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4747 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4748 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4749 = stablehlo.reduce(%v4745 init: %v4746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4750 = stablehlo.broadcast_in_dim %v4749, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4751 = stablehlo.divide %v4750, %v4747 : tensor<32x24x56x56xf32>
    %v4752 = stablehlo.subtract %v4745, %v4751 : tensor<32x24x56x56xf32>
    %v4753 = stablehlo.multiply %v4752, %v4752 : tensor<32x24x56x56xf32>
    %v4754 = stablehlo.reduce(%v4753 init: %v4746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4755 = stablehlo.broadcast_in_dim %v4754, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4756 = stablehlo.divide %v4755, %v4747 : tensor<32x24x56x56xf32>
    %v4757 = stablehlo.add %v4756, %v4748 : tensor<32x24x56x56xf32>
    %v4758 = stablehlo.rsqrt %v4757 : tensor<32x24x56x56xf32>
    %v4759 = stablehlo.multiply %v4752, %v4758 : tensor<32x24x56x56xf32>
    %v4760 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4761 = stablehlo.reshape %v4663 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4762 = stablehlo.multiply %v4760, %v4761 : tensor<32x24x56x56xf32>
    %v4763 = stablehlo.reduce(%v4762 init: %v4746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4764 = stablehlo.broadcast_in_dim %v4763, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4765 = stablehlo.multiply %v4759, %v4762 : tensor<32x24x56x56xf32>
    %v4766 = stablehlo.reduce(%v4765 init: %v4746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4767 = stablehlo.broadcast_in_dim %v4766, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4768 = stablehlo.multiply %v4762, %v4747 : tensor<32x24x56x56xf32>
    %v4769 = stablehlo.subtract %v4768, %v4764 : tensor<32x24x56x56xf32>
    %v4770 = stablehlo.multiply %v4759, %v4767 : tensor<32x24x56x56xf32>
    %v4771 = stablehlo.subtract %v4769, %v4770 : tensor<32x24x56x56xf32>
    %v4772 = stablehlo.divide %v4758, %v4747 : tensor<32x24x56x56xf32>
    %v4773 = stablehlo.multiply %v4772, %v4771 : tensor<32x24x56x56xf32>
    %v4774 = stablehlo.reshape %v4773 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4775 = stablehlo.reshape %v4774 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4776 = stablehlo.reverse %b2pW, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v4777 = stablehlo.transpose %v4776, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v4778 = stablehlo.convolution(%v4775, %v4777)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4779 = stablehlo.reshape %v4778 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4780 = stablehlo.reshape %v4779 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4781 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4782 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v4783 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v4784 = stablehlo.compare GT, %v4781, %v4782 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v4785 = stablehlo.compare LT, %v4781, %v4783 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v4786 = stablehlo.and %v4784, %v4785 : tensor<32x96x56x56xi1>
    %v4787 = stablehlo.select %v4786, %v4780, %v4782 : tensor<32x96x56x56xi1>, tensor<32x96x56x56xf32>
    %v4788 = stablehlo.reshape %v4787 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4789 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4791 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v4792 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v4793 = stablehlo.reduce(%v4789 init: %v4790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4794 = stablehlo.broadcast_in_dim %v4793, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4795 = stablehlo.divide %v4794, %v4791 : tensor<32x96x56x56xf32>
    %v4796 = stablehlo.subtract %v4789, %v4795 : tensor<32x96x56x56xf32>
    %v4797 = stablehlo.multiply %v4796, %v4796 : tensor<32x96x56x56xf32>
    %v4798 = stablehlo.reduce(%v4797 init: %v4790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4799 = stablehlo.broadcast_in_dim %v4798, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4800 = stablehlo.divide %v4799, %v4791 : tensor<32x96x56x56xf32>
    %v4801 = stablehlo.add %v4800, %v4792 : tensor<32x96x56x56xf32>
    %v4802 = stablehlo.rsqrt %v4801 : tensor<32x96x56x56xf32>
    %v4803 = stablehlo.multiply %v4796, %v4802 : tensor<32x96x56x56xf32>
    %v4804 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4805 = stablehlo.reshape %v4788 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4806 = stablehlo.multiply %v4804, %v4805 : tensor<32x96x56x56xf32>
    %v4807 = stablehlo.reduce(%v4806 init: %v4790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4808 = stablehlo.broadcast_in_dim %v4807, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4809 = stablehlo.multiply %v4803, %v4806 : tensor<32x96x56x56xf32>
    %v4810 = stablehlo.reduce(%v4809 init: %v4790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4811 = stablehlo.broadcast_in_dim %v4810, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4812 = stablehlo.multiply %v4806, %v4791 : tensor<32x96x56x56xf32>
    %v4813 = stablehlo.subtract %v4812, %v4808 : tensor<32x96x56x56xf32>
    %v4814 = stablehlo.multiply %v4803, %v4811 : tensor<32x96x56x56xf32>
    %v4815 = stablehlo.subtract %v4813, %v4814 : tensor<32x96x56x56xf32>
    %v4816 = stablehlo.divide %v4802, %v4791 : tensor<32x96x56x56xf32>
    %v4817 = stablehlo.multiply %v4816, %v4815 : tensor<32x96x56x56xf32>
    %v4818 = stablehlo.reshape %v4817 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4819 = stablehlo.reshape %v4818 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4820 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v4821 = stablehlo.convolution(%v4819, %v4820)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v4822 = stablehlo.reshape %v4821 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v4823 = stablehlo.reshape %v4822 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4824 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4825 = stablehlo.constant dense<0.0> : tensor<32x96x112x112xf32>
    %v4826 = stablehlo.constant dense<6.0> : tensor<32x96x112x112xf32>
    %v4827 = stablehlo.compare GT, %v4824, %v4825 : (tensor<32x96x112x112xf32>, tensor<32x96x112x112xf32>) -> tensor<32x96x112x112xi1>
    %v4828 = stablehlo.compare LT, %v4824, %v4826 : (tensor<32x96x112x112xf32>, tensor<32x96x112x112xf32>) -> tensor<32x96x112x112xi1>
    %v4829 = stablehlo.and %v4827, %v4828 : tensor<32x96x112x112xi1>
    %v4830 = stablehlo.select %v4829, %v4823, %v4825 : tensor<32x96x112x112xi1>, tensor<32x96x112x112xf32>
    %v4831 = stablehlo.reshape %v4830 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v4832 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4834 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v4835 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v4836 = stablehlo.reduce(%v4832 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4837 = stablehlo.broadcast_in_dim %v4836, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4838 = stablehlo.divide %v4837, %v4834 : tensor<32x96x112x112xf32>
    %v4839 = stablehlo.subtract %v4832, %v4838 : tensor<32x96x112x112xf32>
    %v4840 = stablehlo.multiply %v4839, %v4839 : tensor<32x96x112x112xf32>
    %v4841 = stablehlo.reduce(%v4840 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4842 = stablehlo.broadcast_in_dim %v4841, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4843 = stablehlo.divide %v4842, %v4834 : tensor<32x96x112x112xf32>
    %v4844 = stablehlo.add %v4843, %v4835 : tensor<32x96x112x112xf32>
    %v4845 = stablehlo.rsqrt %v4844 : tensor<32x96x112x112xf32>
    %v4846 = stablehlo.multiply %v4839, %v4845 : tensor<32x96x112x112xf32>
    %v4847 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4848 = stablehlo.reshape %v4831 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4849 = stablehlo.multiply %v4847, %v4848 : tensor<32x96x112x112xf32>
    %v4850 = stablehlo.reduce(%v4849 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4851 = stablehlo.broadcast_in_dim %v4850, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4852 = stablehlo.multiply %v4846, %v4849 : tensor<32x96x112x112xf32>
    %v4853 = stablehlo.reduce(%v4852 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4854 = stablehlo.broadcast_in_dim %v4853, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4855 = stablehlo.multiply %v4849, %v4834 : tensor<32x96x112x112xf32>
    %v4856 = stablehlo.subtract %v4855, %v4851 : tensor<32x96x112x112xf32>
    %v4857 = stablehlo.multiply %v4846, %v4854 : tensor<32x96x112x112xf32>
    %v4858 = stablehlo.subtract %v4856, %v4857 : tensor<32x96x112x112xf32>
    %v4859 = stablehlo.divide %v4845, %v4834 : tensor<32x96x112x112xf32>
    %v4860 = stablehlo.multiply %v4859, %v4858 : tensor<32x96x112x112xf32>
    %v4861 = stablehlo.reshape %v4860 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v4862 = stablehlo.reshape %v4861 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4863 = stablehlo.reverse %b2eW, dims = [2, 3] : tensor<96x16x1x1xf32>
    %v4864 = stablehlo.transpose %v4863, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v4865 = stablehlo.convolution(%v4862, %v4864)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v4866 = stablehlo.reshape %v4865 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v4867 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4868 = stablehlo.reshape %v4861 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4869 = stablehlo.transpose %v4867, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v4870 = stablehlo.transpose %v4868, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4871 = stablehlo.convolution(%v4869, %v4870)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v4872 = stablehlo.transpose %v4871, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v4873 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4875 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v4876 = stablehlo.reduce(%v4873 init: %v4874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4877 = stablehlo.broadcast_in_dim %v4876, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4878 = stablehlo.divide %v4877, %v4875 : tensor<32x96x112x112xf32>
    %v4879 = stablehlo.subtract %v4873, %v4878 : tensor<32x96x112x112xf32>
    %v4880 = stablehlo.multiply %v4879, %v4879 : tensor<32x96x112x112xf32>
    %v4881 = stablehlo.reduce(%v4880 init: %v4874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4882 = stablehlo.broadcast_in_dim %v4881, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4883 = stablehlo.divide %v4882, %v4875 : tensor<32x96x112x112xf32>
    %v4884 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v4885 = stablehlo.add %v4883, %v4884 : tensor<32x96x112x112xf32>
    %v4886 = stablehlo.rsqrt %v4885 : tensor<32x96x112x112xf32>
    %v4887 = stablehlo.multiply %v4879, %v4886 : tensor<32x96x112x112xf32>
    %v4888 = stablehlo.reshape %v4831 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4889 = stablehlo.multiply %v4888, %v4887 : tensor<32x96x112x112xf32>
    %v4890 = stablehlo.reduce(%v4889 init: %v4874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4891 = stablehlo.reshape %v4831 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4893 = stablehlo.reduce(%v4891 init: %v4892) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4894 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4895 = stablehlo.reshape %v4818 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4897 = stablehlo.pad %v4895, %v4896, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v4898 = stablehlo.transpose %v4894, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4899 = stablehlo.transpose %v4897, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4900 = stablehlo.convolution(%v4898, %v4899)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v4901 = stablehlo.reshape %v4900 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v4902 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4904 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v4905 = stablehlo.reduce(%v4902 init: %v4903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4906 = stablehlo.broadcast_in_dim %v4905, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4907 = stablehlo.divide %v4906, %v4904 : tensor<32x96x56x56xf32>
    %v4908 = stablehlo.subtract %v4902, %v4907 : tensor<32x96x56x56xf32>
    %v4909 = stablehlo.multiply %v4908, %v4908 : tensor<32x96x56x56xf32>
    %v4910 = stablehlo.reduce(%v4909 init: %v4903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4911 = stablehlo.broadcast_in_dim %v4910, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4912 = stablehlo.divide %v4911, %v4904 : tensor<32x96x56x56xf32>
    %v4913 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v4914 = stablehlo.add %v4912, %v4913 : tensor<32x96x56x56xf32>
    %v4915 = stablehlo.rsqrt %v4914 : tensor<32x96x56x56xf32>
    %v4916 = stablehlo.multiply %v4908, %v4915 : tensor<32x96x56x56xf32>
    %v4917 = stablehlo.reshape %v4788 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4918 = stablehlo.multiply %v4917, %v4916 : tensor<32x96x56x56xf32>
    %v4919 = stablehlo.reduce(%v4918 init: %v4903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4920 = stablehlo.reshape %v4788 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4922 = stablehlo.reduce(%v4920 init: %v4921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4923 = stablehlo.reshape %v148 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4924 = stablehlo.reshape %v4774 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4925 = stablehlo.transpose %v4923, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4926 = stablehlo.transpose %v4924, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4927 = stablehlo.convolution(%v4925, %v4926)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v4928 = stablehlo.transpose %v4927, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v4929 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4931 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4932 = stablehlo.reduce(%v4929 init: %v4930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4933 = stablehlo.broadcast_in_dim %v4932, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4934 = stablehlo.divide %v4933, %v4931 : tensor<32x24x56x56xf32>
    %v4935 = stablehlo.subtract %v4929, %v4934 : tensor<32x24x56x56xf32>
    %v4936 = stablehlo.multiply %v4935, %v4935 : tensor<32x24x56x56xf32>
    %v4937 = stablehlo.reduce(%v4936 init: %v4930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4938 = stablehlo.broadcast_in_dim %v4937, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4939 = stablehlo.divide %v4938, %v4931 : tensor<32x24x56x56xf32>
    %v4940 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4941 = stablehlo.add %v4939, %v4940 : tensor<32x24x56x56xf32>
    %v4942 = stablehlo.rsqrt %v4941 : tensor<32x24x56x56xf32>
    %v4943 = stablehlo.multiply %v4935, %v4942 : tensor<32x24x56x56xf32>
    %v4944 = stablehlo.reshape %v4663 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4945 = stablehlo.multiply %v4944, %v4943 : tensor<32x24x56x56xf32>
    %v4946 = stablehlo.reduce(%v4945 init: %v4930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4947 = stablehlo.reshape %v4663 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4949 = stablehlo.reduce(%v4947 init: %v4948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4950 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4952 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v4953 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v4954 = stablehlo.reduce(%v4950 init: %v4951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4955 = stablehlo.broadcast_in_dim %v4954, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4956 = stablehlo.divide %v4955, %v4952 : tensor<32x16x112x112xf32>
    %v4957 = stablehlo.subtract %v4950, %v4956 : tensor<32x16x112x112xf32>
    %v4958 = stablehlo.multiply %v4957, %v4957 : tensor<32x16x112x112xf32>
    %v4959 = stablehlo.reduce(%v4958 init: %v4951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4960 = stablehlo.broadcast_in_dim %v4959, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4961 = stablehlo.divide %v4960, %v4952 : tensor<32x16x112x112xf32>
    %v4962 = stablehlo.add %v4961, %v4953 : tensor<32x16x112x112xf32>
    %v4963 = stablehlo.rsqrt %v4962 : tensor<32x16x112x112xf32>
    %v4964 = stablehlo.multiply %v4957, %v4963 : tensor<32x16x112x112xf32>
    %v4965 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4966 = stablehlo.reshape %v4866 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4967 = stablehlo.multiply %v4965, %v4966 : tensor<32x16x112x112xf32>
    %v4968 = stablehlo.reduce(%v4967 init: %v4951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4969 = stablehlo.broadcast_in_dim %v4968, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4970 = stablehlo.multiply %v4964, %v4967 : tensor<32x16x112x112xf32>
    %v4971 = stablehlo.reduce(%v4970 init: %v4951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4972 = stablehlo.broadcast_in_dim %v4971, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4973 = stablehlo.multiply %v4967, %v4952 : tensor<32x16x112x112xf32>
    %v4974 = stablehlo.subtract %v4973, %v4969 : tensor<32x16x112x112xf32>
    %v4975 = stablehlo.multiply %v4964, %v4972 : tensor<32x16x112x112xf32>
    %v4976 = stablehlo.subtract %v4974, %v4975 : tensor<32x16x112x112xf32>
    %v4977 = stablehlo.divide %v4963, %v4952 : tensor<32x16x112x112xf32>
    %v4978 = stablehlo.multiply %v4977, %v4976 : tensor<32x16x112x112xf32>
    %v4979 = stablehlo.reshape %v4978 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v4980 = stablehlo.reshape %v4979 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4981 = stablehlo.reverse %b1pW, dims = [2, 3] : tensor<16x32x1x1xf32>
    %v4982 = stablehlo.transpose %v4981, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v4983 = stablehlo.convolution(%v4980, %v4982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v4984 = stablehlo.reshape %v4983 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v4985 = stablehlo.reshape %v4984 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4986 = stablehlo.reshape %v55 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4987 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v4988 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v4989 = stablehlo.compare GT, %v4986, %v4987 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v4990 = stablehlo.compare LT, %v4986, %v4988 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v4991 = stablehlo.and %v4989, %v4990 : tensor<32x32x112x112xi1>
    %v4992 = stablehlo.select %v4991, %v4985, %v4987 : tensor<32x32x112x112xi1>, tensor<32x32x112x112xf32>
    %v4993 = stablehlo.reshape %v4992 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v4994 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4996 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v4997 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v4998 = stablehlo.reduce(%v4994 init: %v4995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4999 = stablehlo.broadcast_in_dim %v4998, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5000 = stablehlo.divide %v4999, %v4996 : tensor<32x32x112x112xf32>
    %v5001 = stablehlo.subtract %v4994, %v5000 : tensor<32x32x112x112xf32>
    %v5002 = stablehlo.multiply %v5001, %v5001 : tensor<32x32x112x112xf32>
    %v5003 = stablehlo.reduce(%v5002 init: %v4995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5004 = stablehlo.broadcast_in_dim %v5003, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5005 = stablehlo.divide %v5004, %v4996 : tensor<32x32x112x112xf32>
    %v5006 = stablehlo.add %v5005, %v4997 : tensor<32x32x112x112xf32>
    %v5007 = stablehlo.rsqrt %v5006 : tensor<32x32x112x112xf32>
    %v5008 = stablehlo.multiply %v5001, %v5007 : tensor<32x32x112x112xf32>
    %v5009 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5010 = stablehlo.reshape %v4993 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5011 = stablehlo.multiply %v5009, %v5010 : tensor<32x32x112x112xf32>
    %v5012 = stablehlo.reduce(%v5011 init: %v4995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5013 = stablehlo.broadcast_in_dim %v5012, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5014 = stablehlo.multiply %v5008, %v5011 : tensor<32x32x112x112xf32>
    %v5015 = stablehlo.reduce(%v5014 init: %v4995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5016 = stablehlo.broadcast_in_dim %v5015, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5017 = stablehlo.multiply %v5011, %v4996 : tensor<32x32x112x112xf32>
    %v5018 = stablehlo.subtract %v5017, %v5013 : tensor<32x32x112x112xf32>
    %v5019 = stablehlo.multiply %v5008, %v5016 : tensor<32x32x112x112xf32>
    %v5020 = stablehlo.subtract %v5018, %v5019 : tensor<32x32x112x112xf32>
    %v5021 = stablehlo.divide %v5007, %v4996 : tensor<32x32x112x112xf32>
    %v5022 = stablehlo.multiply %v5021, %v5020 : tensor<32x32x112x112xf32>
    %v5023 = stablehlo.reshape %v5022 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5024 = stablehlo.reshape %v5023 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5025 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v5026 = stablehlo.convolution(%v5024, %v5025)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v5027 = stablehlo.reshape %v5026 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5028 = stablehlo.reshape %v30 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5029 = stablehlo.reshape %v5023 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5030 = stablehlo.transpose %v5028, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5031 = stablehlo.transpose %v5029, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5032 = stablehlo.convolution(%v5030, %v5031)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v5033 = stablehlo.reshape %v5032 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v5034 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5036 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5037 = stablehlo.reduce(%v5034 init: %v5035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5038 = stablehlo.broadcast_in_dim %v5037, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5039 = stablehlo.divide %v5038, %v5036 : tensor<32x32x112x112xf32>
    %v5040 = stablehlo.subtract %v5034, %v5039 : tensor<32x32x112x112xf32>
    %v5041 = stablehlo.multiply %v5040, %v5040 : tensor<32x32x112x112xf32>
    %v5042 = stablehlo.reduce(%v5041 init: %v5035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5043 = stablehlo.broadcast_in_dim %v5042, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5044 = stablehlo.divide %v5043, %v5036 : tensor<32x32x112x112xf32>
    %v5045 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5046 = stablehlo.add %v5044, %v5045 : tensor<32x32x112x112xf32>
    %v5047 = stablehlo.rsqrt %v5046 : tensor<32x32x112x112xf32>
    %v5048 = stablehlo.multiply %v5040, %v5047 : tensor<32x32x112x112xf32>
    %v5049 = stablehlo.reshape %v4993 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5050 = stablehlo.multiply %v5049, %v5048 : tensor<32x32x112x112xf32>
    %v5051 = stablehlo.reduce(%v5050 init: %v5035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5052 = stablehlo.reshape %v4993 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5054 = stablehlo.reduce(%v5052 init: %v5053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5055 = stablehlo.reshape %v61 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5056 = stablehlo.reshape %v4979 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5057 = stablehlo.transpose %v5055, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5058 = stablehlo.transpose %v5056, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v5059 = stablehlo.convolution(%v5057, %v5058)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v5060 = stablehlo.transpose %v5059, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v5061 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5063 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v5064 = stablehlo.reduce(%v5061 init: %v5062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5065 = stablehlo.broadcast_in_dim %v5064, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5066 = stablehlo.divide %v5065, %v5063 : tensor<32x16x112x112xf32>
    %v5067 = stablehlo.subtract %v5061, %v5066 : tensor<32x16x112x112xf32>
    %v5068 = stablehlo.multiply %v5067, %v5067 : tensor<32x16x112x112xf32>
    %v5069 = stablehlo.reduce(%v5068 init: %v5062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5070 = stablehlo.broadcast_in_dim %v5069, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5071 = stablehlo.divide %v5070, %v5063 : tensor<32x16x112x112xf32>
    %v5072 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v5073 = stablehlo.add %v5071, %v5072 : tensor<32x16x112x112xf32>
    %v5074 = stablehlo.rsqrt %v5073 : tensor<32x16x112x112xf32>
    %v5075 = stablehlo.multiply %v5067, %v5074 : tensor<32x16x112x112xf32>
    %v5076 = stablehlo.reshape %v4866 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5077 = stablehlo.multiply %v5076, %v5075 : tensor<32x16x112x112xf32>
    %v5078 = stablehlo.reduce(%v5077 init: %v5062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5079 = stablehlo.reshape %v4866 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5081 = stablehlo.reduce(%v5079 init: %v5080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5082 = stablehlo.reshape %v5027 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5083 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5084 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v5085 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v5086 = stablehlo.compare GT, %v5083, %v5084 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v5087 = stablehlo.compare LT, %v5083, %v5085 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v5088 = stablehlo.and %v5086, %v5087 : tensor<32x32x112x112xi1>
    %v5089 = stablehlo.select %v5088, %v5082, %v5084 : tensor<32x32x112x112xi1>, tensor<32x32x112x112xf32>
    %v5090 = stablehlo.reshape %v5089 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5091 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5093 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5094 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5095 = stablehlo.reduce(%v5091 init: %v5092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5096 = stablehlo.broadcast_in_dim %v5095, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5097 = stablehlo.divide %v5096, %v5093 : tensor<32x32x112x112xf32>
    %v5098 = stablehlo.subtract %v5091, %v5097 : tensor<32x32x112x112xf32>
    %v5099 = stablehlo.multiply %v5098, %v5098 : tensor<32x32x112x112xf32>
    %v5100 = stablehlo.reduce(%v5099 init: %v5092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5101 = stablehlo.broadcast_in_dim %v5100, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5102 = stablehlo.divide %v5101, %v5093 : tensor<32x32x112x112xf32>
    %v5103 = stablehlo.add %v5102, %v5094 : tensor<32x32x112x112xf32>
    %v5104 = stablehlo.rsqrt %v5103 : tensor<32x32x112x112xf32>
    %v5105 = stablehlo.multiply %v5098, %v5104 : tensor<32x32x112x112xf32>
    %v5106 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5107 = stablehlo.reshape %v5090 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5108 = stablehlo.multiply %v5106, %v5107 : tensor<32x32x112x112xf32>
    %v5109 = stablehlo.reduce(%v5108 init: %v5092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5110 = stablehlo.broadcast_in_dim %v5109, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5111 = stablehlo.multiply %v5105, %v5108 : tensor<32x32x112x112xf32>
    %v5112 = stablehlo.reduce(%v5111 init: %v5092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5113 = stablehlo.broadcast_in_dim %v5112, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5114 = stablehlo.multiply %v5108, %v5093 : tensor<32x32x112x112xf32>
    %v5115 = stablehlo.subtract %v5114, %v5110 : tensor<32x32x112x112xf32>
    %v5116 = stablehlo.multiply %v5105, %v5113 : tensor<32x32x112x112xf32>
    %v5117 = stablehlo.subtract %v5115, %v5116 : tensor<32x32x112x112xf32>
    %v5118 = stablehlo.divide %v5104, %v5093 : tensor<32x32x112x112xf32>
    %v5119 = stablehlo.multiply %v5118, %v5117 : tensor<32x32x112x112xf32>
    %v5120 = stablehlo.reshape %v5119 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5121 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v5122 = stablehlo.reshape %v5120 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5123 = stablehlo.transpose %v5121, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v5125 = stablehlo.transpose %v5122, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5126 = stablehlo.convolution(%v5123, %v5125)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x112x112xf32>) -> tensor<3x32x3x3xf32>
    %v5127 = stablehlo.transpose %v5126, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v5128 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5130 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5131 = stablehlo.reduce(%v5128 init: %v5129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5132 = stablehlo.broadcast_in_dim %v5131, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5133 = stablehlo.divide %v5132, %v5130 : tensor<32x32x112x112xf32>
    %v5134 = stablehlo.subtract %v5128, %v5133 : tensor<32x32x112x112xf32>
    %v5135 = stablehlo.multiply %v5134, %v5134 : tensor<32x32x112x112xf32>
    %v5136 = stablehlo.reduce(%v5135 init: %v5129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5137 = stablehlo.broadcast_in_dim %v5136, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5138 = stablehlo.divide %v5137, %v5130 : tensor<32x32x112x112xf32>
    %v5139 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5140 = stablehlo.add %v5138, %v5139 : tensor<32x32x112x112xf32>
    %v5141 = stablehlo.rsqrt %v5140 : tensor<32x32x112x112xf32>
    %v5142 = stablehlo.multiply %v5134, %v5141 : tensor<32x32x112x112xf32>
    %v5143 = stablehlo.reshape %v5090 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5144 = stablehlo.multiply %v5143, %v5142 : tensor<32x32x112x112xf32>
    %v5145 = stablehlo.reduce(%v5144 init: %v5129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5146 = stablehlo.reshape %v5090 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5148 = stablehlo.reduce(%v5146 init: %v5147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5149 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5151 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v5152 = stablehlo.reduce(%v5149 init: %v5150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5153 = stablehlo.divide %v5152, %v5151 : tensor<32xf32>
    %v5154 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5156 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5157 = stablehlo.reduce(%v5154 init: %v5155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5158 = stablehlo.broadcast_in_dim %v5157, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5159 = stablehlo.divide %v5158, %v5156 : tensor<32x32x112x112xf32>
    %v5160 = stablehlo.subtract %v5154, %v5159 : tensor<32x32x112x112xf32>
    %v5161 = stablehlo.multiply %v5160, %v5160 : tensor<32x32x112x112xf32>
    %v5162 = stablehlo.reduce(%v5161 init: %v5155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5163 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v5164 = stablehlo.divide %v5162, %v5163 : tensor<32xf32>
    %v5165 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5167 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v5168 = stablehlo.reduce(%v5165 init: %v5166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5169 = stablehlo.divide %v5168, %v5167 : tensor<32xf32>
    %v5170 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5172 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5173 = stablehlo.reduce(%v5170 init: %v5171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5174 = stablehlo.broadcast_in_dim %v5173, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5175 = stablehlo.divide %v5174, %v5172 : tensor<32x32x112x112xf32>
    %v5176 = stablehlo.subtract %v5170, %v5175 : tensor<32x32x112x112xf32>
    %v5177 = stablehlo.multiply %v5176, %v5176 : tensor<32x32x112x112xf32>
    %v5178 = stablehlo.reduce(%v5177 init: %v5171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5179 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v5180 = stablehlo.divide %v5178, %v5179 : tensor<32xf32>
    %v5181 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5183 = stablehlo.constant dense<401408.0> : tensor<16xf32>
    %v5184 = stablehlo.reduce(%v5181 init: %v5182) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5185 = stablehlo.divide %v5184, %v5183 : tensor<16xf32>
    %v5186 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5188 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v5189 = stablehlo.reduce(%v5186 init: %v5187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5190 = stablehlo.broadcast_in_dim %v5189, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5191 = stablehlo.divide %v5190, %v5188 : tensor<32x16x112x112xf32>
    %v5192 = stablehlo.subtract %v5186, %v5191 : tensor<32x16x112x112xf32>
    %v5193 = stablehlo.multiply %v5192, %v5192 : tensor<32x16x112x112xf32>
    %v5194 = stablehlo.reduce(%v5193 init: %v5187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5195 = stablehlo.constant dense<401408.0> : tensor<16xf32>
    %v5196 = stablehlo.divide %v5194, %v5195 : tensor<16xf32>
    %v5197 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5199 = stablehlo.constant dense<401408.0> : tensor<96xf32>
    %v5200 = stablehlo.reduce(%v5197 init: %v5198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5201 = stablehlo.divide %v5200, %v5199 : tensor<96xf32>
    %v5202 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5204 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v5205 = stablehlo.reduce(%v5202 init: %v5203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5206 = stablehlo.broadcast_in_dim %v5205, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v5207 = stablehlo.divide %v5206, %v5204 : tensor<32x96x112x112xf32>
    %v5208 = stablehlo.subtract %v5202, %v5207 : tensor<32x96x112x112xf32>
    %v5209 = stablehlo.multiply %v5208, %v5208 : tensor<32x96x112x112xf32>
    %v5210 = stablehlo.reduce(%v5209 init: %v5203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5211 = stablehlo.constant dense<401408.0> : tensor<96xf32>
    %v5212 = stablehlo.divide %v5210, %v5211 : tensor<96xf32>
    %v5213 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5215 = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %v5216 = stablehlo.reduce(%v5213 init: %v5214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5217 = stablehlo.divide %v5216, %v5215 : tensor<96xf32>
    %v5218 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5220 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v5221 = stablehlo.reduce(%v5218 init: %v5219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5222 = stablehlo.broadcast_in_dim %v5221, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v5223 = stablehlo.divide %v5222, %v5220 : tensor<32x96x56x56xf32>
    %v5224 = stablehlo.subtract %v5218, %v5223 : tensor<32x96x56x56xf32>
    %v5225 = stablehlo.multiply %v5224, %v5224 : tensor<32x96x56x56xf32>
    %v5226 = stablehlo.reduce(%v5225 init: %v5219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5227 = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %v5228 = stablehlo.divide %v5226, %v5227 : tensor<96xf32>
    %v5229 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5231 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5232 = stablehlo.reduce(%v5229 init: %v5230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5233 = stablehlo.divide %v5232, %v5231 : tensor<24xf32>
    %v5234 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5236 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5237 = stablehlo.reduce(%v5234 init: %v5235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5238 = stablehlo.broadcast_in_dim %v5237, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5239 = stablehlo.divide %v5238, %v5236 : tensor<32x24x56x56xf32>
    %v5240 = stablehlo.subtract %v5234, %v5239 : tensor<32x24x56x56xf32>
    %v5241 = stablehlo.multiply %v5240, %v5240 : tensor<32x24x56x56xf32>
    %v5242 = stablehlo.reduce(%v5241 init: %v5235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5243 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5244 = stablehlo.divide %v5242, %v5243 : tensor<24xf32>
    %v5245 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5247 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5248 = stablehlo.reduce(%v5245 init: %v5246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5249 = stablehlo.divide %v5248, %v5247 : tensor<144xf32>
    %v5250 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5252 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5253 = stablehlo.reduce(%v5250 init: %v5251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5254 = stablehlo.broadcast_in_dim %v5253, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5255 = stablehlo.divide %v5254, %v5252 : tensor<32x144x56x56xf32>
    %v5256 = stablehlo.subtract %v5250, %v5255 : tensor<32x144x56x56xf32>
    %v5257 = stablehlo.multiply %v5256, %v5256 : tensor<32x144x56x56xf32>
    %v5258 = stablehlo.reduce(%v5257 init: %v5251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5259 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5260 = stablehlo.divide %v5258, %v5259 : tensor<144xf32>
    %v5261 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5263 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5264 = stablehlo.reduce(%v5261 init: %v5262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5265 = stablehlo.divide %v5264, %v5263 : tensor<144xf32>
    %v5266 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5268 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5269 = stablehlo.reduce(%v5266 init: %v5267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5270 = stablehlo.broadcast_in_dim %v5269, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5271 = stablehlo.divide %v5270, %v5268 : tensor<32x144x56x56xf32>
    %v5272 = stablehlo.subtract %v5266, %v5271 : tensor<32x144x56x56xf32>
    %v5273 = stablehlo.multiply %v5272, %v5272 : tensor<32x144x56x56xf32>
    %v5274 = stablehlo.reduce(%v5273 init: %v5267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5275 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5276 = stablehlo.divide %v5274, %v5275 : tensor<144xf32>
    %v5277 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5279 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5280 = stablehlo.reduce(%v5277 init: %v5278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5281 = stablehlo.divide %v5280, %v5279 : tensor<24xf32>
    %v5282 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5284 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5285 = stablehlo.reduce(%v5282 init: %v5283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5286 = stablehlo.broadcast_in_dim %v5285, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5287 = stablehlo.divide %v5286, %v5284 : tensor<32x24x56x56xf32>
    %v5288 = stablehlo.subtract %v5282, %v5287 : tensor<32x24x56x56xf32>
    %v5289 = stablehlo.multiply %v5288, %v5288 : tensor<32x24x56x56xf32>
    %v5290 = stablehlo.reduce(%v5289 init: %v5283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5291 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5292 = stablehlo.divide %v5290, %v5291 : tensor<24xf32>
    %v5293 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5295 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5296 = stablehlo.reduce(%v5293 init: %v5294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5297 = stablehlo.divide %v5296, %v5295 : tensor<144xf32>
    %v5298 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5300 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5301 = stablehlo.reduce(%v5298 init: %v5299) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5302 = stablehlo.broadcast_in_dim %v5301, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5303 = stablehlo.divide %v5302, %v5300 : tensor<32x144x56x56xf32>
    %v5304 = stablehlo.subtract %v5298, %v5303 : tensor<32x144x56x56xf32>
    %v5305 = stablehlo.multiply %v5304, %v5304 : tensor<32x144x56x56xf32>
    %v5306 = stablehlo.reduce(%v5305 init: %v5299) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5307 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5308 = stablehlo.divide %v5306, %v5307 : tensor<144xf32>
    %v5309 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5311 = stablehlo.constant dense<25088.0> : tensor<144xf32>
    %v5312 = stablehlo.reduce(%v5309 init: %v5310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5313 = stablehlo.divide %v5312, %v5311 : tensor<144xf32>
    %v5314 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5316 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5317 = stablehlo.reduce(%v5314 init: %v5315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5318 = stablehlo.broadcast_in_dim %v5317, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5319 = stablehlo.divide %v5318, %v5316 : tensor<32x144x28x28xf32>
    %v5320 = stablehlo.subtract %v5314, %v5319 : tensor<32x144x28x28xf32>
    %v5321 = stablehlo.multiply %v5320, %v5320 : tensor<32x144x28x28xf32>
    %v5322 = stablehlo.reduce(%v5321 init: %v5315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5323 = stablehlo.constant dense<25088.0> : tensor<144xf32>
    %v5324 = stablehlo.divide %v5322, %v5323 : tensor<144xf32>
    %v5325 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5327 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5328 = stablehlo.reduce(%v5325 init: %v5326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5329 = stablehlo.divide %v5328, %v5327 : tensor<32xf32>
    %v5330 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5332 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5333 = stablehlo.reduce(%v5330 init: %v5331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5334 = stablehlo.broadcast_in_dim %v5333, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5335 = stablehlo.divide %v5334, %v5332 : tensor<32x32x28x28xf32>
    %v5336 = stablehlo.subtract %v5330, %v5335 : tensor<32x32x28x28xf32>
    %v5337 = stablehlo.multiply %v5336, %v5336 : tensor<32x32x28x28xf32>
    %v5338 = stablehlo.reduce(%v5337 init: %v5331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5339 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5340 = stablehlo.divide %v5338, %v5339 : tensor<32xf32>
    %v5341 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5343 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5344 = stablehlo.reduce(%v5341 init: %v5342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5345 = stablehlo.divide %v5344, %v5343 : tensor<192xf32>
    %v5346 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5348 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5349 = stablehlo.reduce(%v5346 init: %v5347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5350 = stablehlo.broadcast_in_dim %v5349, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5351 = stablehlo.divide %v5350, %v5348 : tensor<32x192x28x28xf32>
    %v5352 = stablehlo.subtract %v5346, %v5351 : tensor<32x192x28x28xf32>
    %v5353 = stablehlo.multiply %v5352, %v5352 : tensor<32x192x28x28xf32>
    %v5354 = stablehlo.reduce(%v5353 init: %v5347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5355 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5356 = stablehlo.divide %v5354, %v5355 : tensor<192xf32>
    %v5357 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5359 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5360 = stablehlo.reduce(%v5357 init: %v5358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5361 = stablehlo.divide %v5360, %v5359 : tensor<192xf32>
    %v5362 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5364 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5365 = stablehlo.reduce(%v5362 init: %v5363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5366 = stablehlo.broadcast_in_dim %v5365, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5367 = stablehlo.divide %v5366, %v5364 : tensor<32x192x28x28xf32>
    %v5368 = stablehlo.subtract %v5362, %v5367 : tensor<32x192x28x28xf32>
    %v5369 = stablehlo.multiply %v5368, %v5368 : tensor<32x192x28x28xf32>
    %v5370 = stablehlo.reduce(%v5369 init: %v5363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5371 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5372 = stablehlo.divide %v5370, %v5371 : tensor<192xf32>
    %v5373 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5375 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5376 = stablehlo.reduce(%v5373 init: %v5374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5377 = stablehlo.divide %v5376, %v5375 : tensor<32xf32>
    %v5378 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5380 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5381 = stablehlo.reduce(%v5378 init: %v5379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5382 = stablehlo.broadcast_in_dim %v5381, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5383 = stablehlo.divide %v5382, %v5380 : tensor<32x32x28x28xf32>
    %v5384 = stablehlo.subtract %v5378, %v5383 : tensor<32x32x28x28xf32>
    %v5385 = stablehlo.multiply %v5384, %v5384 : tensor<32x32x28x28xf32>
    %v5386 = stablehlo.reduce(%v5385 init: %v5379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5387 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5388 = stablehlo.divide %v5386, %v5387 : tensor<32xf32>
    %v5389 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5391 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5392 = stablehlo.reduce(%v5389 init: %v5390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5393 = stablehlo.divide %v5392, %v5391 : tensor<192xf32>
    %v5394 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5396 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5397 = stablehlo.reduce(%v5394 init: %v5395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5398 = stablehlo.broadcast_in_dim %v5397, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5399 = stablehlo.divide %v5398, %v5396 : tensor<32x192x28x28xf32>
    %v5400 = stablehlo.subtract %v5394, %v5399 : tensor<32x192x28x28xf32>
    %v5401 = stablehlo.multiply %v5400, %v5400 : tensor<32x192x28x28xf32>
    %v5402 = stablehlo.reduce(%v5401 init: %v5395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5403 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5404 = stablehlo.divide %v5402, %v5403 : tensor<192xf32>
    %v5405 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5407 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5408 = stablehlo.reduce(%v5405 init: %v5406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5409 = stablehlo.divide %v5408, %v5407 : tensor<192xf32>
    %v5410 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5412 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5413 = stablehlo.reduce(%v5410 init: %v5411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5414 = stablehlo.broadcast_in_dim %v5413, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5415 = stablehlo.divide %v5414, %v5412 : tensor<32x192x28x28xf32>
    %v5416 = stablehlo.subtract %v5410, %v5415 : tensor<32x192x28x28xf32>
    %v5417 = stablehlo.multiply %v5416, %v5416 : tensor<32x192x28x28xf32>
    %v5418 = stablehlo.reduce(%v5417 init: %v5411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5419 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5420 = stablehlo.divide %v5418, %v5419 : tensor<192xf32>
    %v5421 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5423 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5424 = stablehlo.reduce(%v5421 init: %v5422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5425 = stablehlo.divide %v5424, %v5423 : tensor<32xf32>
    %v5426 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5428 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5429 = stablehlo.reduce(%v5426 init: %v5427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5430 = stablehlo.broadcast_in_dim %v5429, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5431 = stablehlo.divide %v5430, %v5428 : tensor<32x32x28x28xf32>
    %v5432 = stablehlo.subtract %v5426, %v5431 : tensor<32x32x28x28xf32>
    %v5433 = stablehlo.multiply %v5432, %v5432 : tensor<32x32x28x28xf32>
    %v5434 = stablehlo.reduce(%v5433 init: %v5427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5435 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5436 = stablehlo.divide %v5434, %v5435 : tensor<32xf32>
    %v5437 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5439 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5440 = stablehlo.reduce(%v5437 init: %v5438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5441 = stablehlo.divide %v5440, %v5439 : tensor<192xf32>
    %v5442 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5444 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5445 = stablehlo.reduce(%v5442 init: %v5443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5446 = stablehlo.broadcast_in_dim %v5445, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5447 = stablehlo.divide %v5446, %v5444 : tensor<32x192x28x28xf32>
    %v5448 = stablehlo.subtract %v5442, %v5447 : tensor<32x192x28x28xf32>
    %v5449 = stablehlo.multiply %v5448, %v5448 : tensor<32x192x28x28xf32>
    %v5450 = stablehlo.reduce(%v5449 init: %v5443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5451 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5452 = stablehlo.divide %v5450, %v5451 : tensor<192xf32>
    %v5453 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v5454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5455 = stablehlo.constant dense<6272.0> : tensor<192xf32>
    %v5456 = stablehlo.reduce(%v5453 init: %v5454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5457 = stablehlo.divide %v5456, %v5455 : tensor<192xf32>
    %v5458 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v5459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5460 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v5461 = stablehlo.reduce(%v5458 init: %v5459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5462 = stablehlo.broadcast_in_dim %v5461, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v5463 = stablehlo.divide %v5462, %v5460 : tensor<32x192x14x14xf32>
    %v5464 = stablehlo.subtract %v5458, %v5463 : tensor<32x192x14x14xf32>
    %v5465 = stablehlo.multiply %v5464, %v5464 : tensor<32x192x14x14xf32>
    %v5466 = stablehlo.reduce(%v5465 init: %v5459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5467 = stablehlo.constant dense<6272.0> : tensor<192xf32>
    %v5468 = stablehlo.divide %v5466, %v5467 : tensor<192xf32>
    %v5469 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5471 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5472 = stablehlo.reduce(%v5469 init: %v5470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5473 = stablehlo.divide %v5472, %v5471 : tensor<64xf32>
    %v5474 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5476 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5477 = stablehlo.reduce(%v5474 init: %v5475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5478 = stablehlo.broadcast_in_dim %v5477, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5479 = stablehlo.divide %v5478, %v5476 : tensor<32x64x14x14xf32>
    %v5480 = stablehlo.subtract %v5474, %v5479 : tensor<32x64x14x14xf32>
    %v5481 = stablehlo.multiply %v5480, %v5480 : tensor<32x64x14x14xf32>
    %v5482 = stablehlo.reduce(%v5481 init: %v5475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5483 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5484 = stablehlo.divide %v5482, %v5483 : tensor<64xf32>
    %v5485 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5487 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5488 = stablehlo.reduce(%v5485 init: %v5486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5489 = stablehlo.divide %v5488, %v5487 : tensor<384xf32>
    %v5490 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5492 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5493 = stablehlo.reduce(%v5490 init: %v5491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5494 = stablehlo.broadcast_in_dim %v5493, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5495 = stablehlo.divide %v5494, %v5492 : tensor<32x384x14x14xf32>
    %v5496 = stablehlo.subtract %v5490, %v5495 : tensor<32x384x14x14xf32>
    %v5497 = stablehlo.multiply %v5496, %v5496 : tensor<32x384x14x14xf32>
    %v5498 = stablehlo.reduce(%v5497 init: %v5491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5499 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5500 = stablehlo.divide %v5498, %v5499 : tensor<384xf32>
    %v5501 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5503 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5504 = stablehlo.reduce(%v5501 init: %v5502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5505 = stablehlo.divide %v5504, %v5503 : tensor<384xf32>
    %v5506 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5508 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5509 = stablehlo.reduce(%v5506 init: %v5507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5510 = stablehlo.broadcast_in_dim %v5509, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5511 = stablehlo.divide %v5510, %v5508 : tensor<32x384x14x14xf32>
    %v5512 = stablehlo.subtract %v5506, %v5511 : tensor<32x384x14x14xf32>
    %v5513 = stablehlo.multiply %v5512, %v5512 : tensor<32x384x14x14xf32>
    %v5514 = stablehlo.reduce(%v5513 init: %v5507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5515 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5516 = stablehlo.divide %v5514, %v5515 : tensor<384xf32>
    %v5517 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5519 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5520 = stablehlo.reduce(%v5517 init: %v5518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5521 = stablehlo.divide %v5520, %v5519 : tensor<64xf32>
    %v5522 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5524 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5525 = stablehlo.reduce(%v5522 init: %v5523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5526 = stablehlo.broadcast_in_dim %v5525, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5527 = stablehlo.divide %v5526, %v5524 : tensor<32x64x14x14xf32>
    %v5528 = stablehlo.subtract %v5522, %v5527 : tensor<32x64x14x14xf32>
    %v5529 = stablehlo.multiply %v5528, %v5528 : tensor<32x64x14x14xf32>
    %v5530 = stablehlo.reduce(%v5529 init: %v5523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5531 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5532 = stablehlo.divide %v5530, %v5531 : tensor<64xf32>
    %v5533 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5535 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5536 = stablehlo.reduce(%v5533 init: %v5534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5537 = stablehlo.divide %v5536, %v5535 : tensor<384xf32>
    %v5538 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5540 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5541 = stablehlo.reduce(%v5538 init: %v5539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5542 = stablehlo.broadcast_in_dim %v5541, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5543 = stablehlo.divide %v5542, %v5540 : tensor<32x384x14x14xf32>
    %v5544 = stablehlo.subtract %v5538, %v5543 : tensor<32x384x14x14xf32>
    %v5545 = stablehlo.multiply %v5544, %v5544 : tensor<32x384x14x14xf32>
    %v5546 = stablehlo.reduce(%v5545 init: %v5539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5547 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5548 = stablehlo.divide %v5546, %v5547 : tensor<384xf32>
    %v5549 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5551 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5552 = stablehlo.reduce(%v5549 init: %v5550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5553 = stablehlo.divide %v5552, %v5551 : tensor<384xf32>
    %v5554 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5556 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5557 = stablehlo.reduce(%v5554 init: %v5555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5558 = stablehlo.broadcast_in_dim %v5557, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5559 = stablehlo.divide %v5558, %v5556 : tensor<32x384x14x14xf32>
    %v5560 = stablehlo.subtract %v5554, %v5559 : tensor<32x384x14x14xf32>
    %v5561 = stablehlo.multiply %v5560, %v5560 : tensor<32x384x14x14xf32>
    %v5562 = stablehlo.reduce(%v5561 init: %v5555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5563 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5564 = stablehlo.divide %v5562, %v5563 : tensor<384xf32>
    %v5565 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5567 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5568 = stablehlo.reduce(%v5565 init: %v5566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5569 = stablehlo.divide %v5568, %v5567 : tensor<64xf32>
    %v5570 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5572 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5573 = stablehlo.reduce(%v5570 init: %v5571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5574 = stablehlo.broadcast_in_dim %v5573, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5575 = stablehlo.divide %v5574, %v5572 : tensor<32x64x14x14xf32>
    %v5576 = stablehlo.subtract %v5570, %v5575 : tensor<32x64x14x14xf32>
    %v5577 = stablehlo.multiply %v5576, %v5576 : tensor<32x64x14x14xf32>
    %v5578 = stablehlo.reduce(%v5577 init: %v5571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5579 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5580 = stablehlo.divide %v5578, %v5579 : tensor<64xf32>
    %v5581 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5583 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5584 = stablehlo.reduce(%v5581 init: %v5582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5585 = stablehlo.divide %v5584, %v5583 : tensor<384xf32>
    %v5586 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5588 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5589 = stablehlo.reduce(%v5586 init: %v5587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5590 = stablehlo.broadcast_in_dim %v5589, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5591 = stablehlo.divide %v5590, %v5588 : tensor<32x384x14x14xf32>
    %v5592 = stablehlo.subtract %v5586, %v5591 : tensor<32x384x14x14xf32>
    %v5593 = stablehlo.multiply %v5592, %v5592 : tensor<32x384x14x14xf32>
    %v5594 = stablehlo.reduce(%v5593 init: %v5587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5595 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5596 = stablehlo.divide %v5594, %v5595 : tensor<384xf32>
    %v5597 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5599 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5600 = stablehlo.reduce(%v5597 init: %v5598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5601 = stablehlo.divide %v5600, %v5599 : tensor<384xf32>
    %v5602 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5604 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5605 = stablehlo.reduce(%v5602 init: %v5603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5606 = stablehlo.broadcast_in_dim %v5605, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5607 = stablehlo.divide %v5606, %v5604 : tensor<32x384x14x14xf32>
    %v5608 = stablehlo.subtract %v5602, %v5607 : tensor<32x384x14x14xf32>
    %v5609 = stablehlo.multiply %v5608, %v5608 : tensor<32x384x14x14xf32>
    %v5610 = stablehlo.reduce(%v5609 init: %v5603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5611 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5612 = stablehlo.divide %v5610, %v5611 : tensor<384xf32>
    %v5613 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5615 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5616 = stablehlo.reduce(%v5613 init: %v5614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5617 = stablehlo.divide %v5616, %v5615 : tensor<64xf32>
    %v5618 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5620 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5621 = stablehlo.reduce(%v5618 init: %v5619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5622 = stablehlo.broadcast_in_dim %v5621, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5623 = stablehlo.divide %v5622, %v5620 : tensor<32x64x14x14xf32>
    %v5624 = stablehlo.subtract %v5618, %v5623 : tensor<32x64x14x14xf32>
    %v5625 = stablehlo.multiply %v5624, %v5624 : tensor<32x64x14x14xf32>
    %v5626 = stablehlo.reduce(%v5625 init: %v5619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5627 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5628 = stablehlo.divide %v5626, %v5627 : tensor<64xf32>
    %v5629 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5631 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5632 = stablehlo.reduce(%v5629 init: %v5630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5633 = stablehlo.divide %v5632, %v5631 : tensor<384xf32>
    %v5634 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5636 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5637 = stablehlo.reduce(%v5634 init: %v5635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5638 = stablehlo.broadcast_in_dim %v5637, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5639 = stablehlo.divide %v5638, %v5636 : tensor<32x384x14x14xf32>
    %v5640 = stablehlo.subtract %v5634, %v5639 : tensor<32x384x14x14xf32>
    %v5641 = stablehlo.multiply %v5640, %v5640 : tensor<32x384x14x14xf32>
    %v5642 = stablehlo.reduce(%v5641 init: %v5635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5643 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5644 = stablehlo.divide %v5642, %v5643 : tensor<384xf32>
    %v5645 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5646 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5647 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5648 = stablehlo.reduce(%v5645 init: %v5646) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5649 = stablehlo.divide %v5648, %v5647 : tensor<384xf32>
    %v5650 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5652 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5653 = stablehlo.reduce(%v5650 init: %v5651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5654 = stablehlo.broadcast_in_dim %v5653, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5655 = stablehlo.divide %v5654, %v5652 : tensor<32x384x14x14xf32>
    %v5656 = stablehlo.subtract %v5650, %v5655 : tensor<32x384x14x14xf32>
    %v5657 = stablehlo.multiply %v5656, %v5656 : tensor<32x384x14x14xf32>
    %v5658 = stablehlo.reduce(%v5657 init: %v5651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5659 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5660 = stablehlo.divide %v5658, %v5659 : tensor<384xf32>
    %v5661 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5663 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5664 = stablehlo.reduce(%v5661 init: %v5662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5665 = stablehlo.divide %v5664, %v5663 : tensor<96xf32>
    %v5666 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5668 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5669 = stablehlo.reduce(%v5666 init: %v5667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5670 = stablehlo.broadcast_in_dim %v5669, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5671 = stablehlo.divide %v5670, %v5668 : tensor<32x96x14x14xf32>
    %v5672 = stablehlo.subtract %v5666, %v5671 : tensor<32x96x14x14xf32>
    %v5673 = stablehlo.multiply %v5672, %v5672 : tensor<32x96x14x14xf32>
    %v5674 = stablehlo.reduce(%v5673 init: %v5667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5675 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5676 = stablehlo.divide %v5674, %v5675 : tensor<96xf32>
    %v5677 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5679 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5680 = stablehlo.reduce(%v5677 init: %v5678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5681 = stablehlo.divide %v5680, %v5679 : tensor<576xf32>
    %v5682 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5684 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5685 = stablehlo.reduce(%v5682 init: %v5683) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5686 = stablehlo.broadcast_in_dim %v5685, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5687 = stablehlo.divide %v5686, %v5684 : tensor<32x576x14x14xf32>
    %v5688 = stablehlo.subtract %v5682, %v5687 : tensor<32x576x14x14xf32>
    %v5689 = stablehlo.multiply %v5688, %v5688 : tensor<32x576x14x14xf32>
    %v5690 = stablehlo.reduce(%v5689 init: %v5683) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5691 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5692 = stablehlo.divide %v5690, %v5691 : tensor<576xf32>
    %v5693 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5695 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5696 = stablehlo.reduce(%v5693 init: %v5694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5697 = stablehlo.divide %v5696, %v5695 : tensor<576xf32>
    %v5698 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5700 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5701 = stablehlo.reduce(%v5698 init: %v5699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5702 = stablehlo.broadcast_in_dim %v5701, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5703 = stablehlo.divide %v5702, %v5700 : tensor<32x576x14x14xf32>
    %v5704 = stablehlo.subtract %v5698, %v5703 : tensor<32x576x14x14xf32>
    %v5705 = stablehlo.multiply %v5704, %v5704 : tensor<32x576x14x14xf32>
    %v5706 = stablehlo.reduce(%v5705 init: %v5699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5707 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5708 = stablehlo.divide %v5706, %v5707 : tensor<576xf32>
    %v5709 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5711 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5712 = stablehlo.reduce(%v5709 init: %v5710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5713 = stablehlo.divide %v5712, %v5711 : tensor<96xf32>
    %v5714 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5716 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5717 = stablehlo.reduce(%v5714 init: %v5715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5718 = stablehlo.broadcast_in_dim %v5717, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5719 = stablehlo.divide %v5718, %v5716 : tensor<32x96x14x14xf32>
    %v5720 = stablehlo.subtract %v5714, %v5719 : tensor<32x96x14x14xf32>
    %v5721 = stablehlo.multiply %v5720, %v5720 : tensor<32x96x14x14xf32>
    %v5722 = stablehlo.reduce(%v5721 init: %v5715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5723 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5724 = stablehlo.divide %v5722, %v5723 : tensor<96xf32>
    %v5725 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5727 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5728 = stablehlo.reduce(%v5725 init: %v5726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5729 = stablehlo.divide %v5728, %v5727 : tensor<576xf32>
    %v5730 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5732 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5733 = stablehlo.reduce(%v5730 init: %v5731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5734 = stablehlo.broadcast_in_dim %v5733, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5735 = stablehlo.divide %v5734, %v5732 : tensor<32x576x14x14xf32>
    %v5736 = stablehlo.subtract %v5730, %v5735 : tensor<32x576x14x14xf32>
    %v5737 = stablehlo.multiply %v5736, %v5736 : tensor<32x576x14x14xf32>
    %v5738 = stablehlo.reduce(%v5737 init: %v5731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5739 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5740 = stablehlo.divide %v5738, %v5739 : tensor<576xf32>
    %v5741 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5743 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5744 = stablehlo.reduce(%v5741 init: %v5742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5745 = stablehlo.divide %v5744, %v5743 : tensor<576xf32>
    %v5746 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5748 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5749 = stablehlo.reduce(%v5746 init: %v5747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5750 = stablehlo.broadcast_in_dim %v5749, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5751 = stablehlo.divide %v5750, %v5748 : tensor<32x576x14x14xf32>
    %v5752 = stablehlo.subtract %v5746, %v5751 : tensor<32x576x14x14xf32>
    %v5753 = stablehlo.multiply %v5752, %v5752 : tensor<32x576x14x14xf32>
    %v5754 = stablehlo.reduce(%v5753 init: %v5747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5755 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5756 = stablehlo.divide %v5754, %v5755 : tensor<576xf32>
    %v5757 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5759 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5760 = stablehlo.reduce(%v5757 init: %v5758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5761 = stablehlo.divide %v5760, %v5759 : tensor<96xf32>
    %v5762 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5763 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5764 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5765 = stablehlo.reduce(%v5762 init: %v5763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5766 = stablehlo.broadcast_in_dim %v5765, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5767 = stablehlo.divide %v5766, %v5764 : tensor<32x96x14x14xf32>
    %v5768 = stablehlo.subtract %v5762, %v5767 : tensor<32x96x14x14xf32>
    %v5769 = stablehlo.multiply %v5768, %v5768 : tensor<32x96x14x14xf32>
    %v5770 = stablehlo.reduce(%v5769 init: %v5763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5771 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5772 = stablehlo.divide %v5770, %v5771 : tensor<96xf32>
    %v5773 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5775 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5776 = stablehlo.reduce(%v5773 init: %v5774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5777 = stablehlo.divide %v5776, %v5775 : tensor<576xf32>
    %v5778 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5780 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5781 = stablehlo.reduce(%v5778 init: %v5779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5782 = stablehlo.broadcast_in_dim %v5781, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5783 = stablehlo.divide %v5782, %v5780 : tensor<32x576x14x14xf32>
    %v5784 = stablehlo.subtract %v5778, %v5783 : tensor<32x576x14x14xf32>
    %v5785 = stablehlo.multiply %v5784, %v5784 : tensor<32x576x14x14xf32>
    %v5786 = stablehlo.reduce(%v5785 init: %v5779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5787 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5788 = stablehlo.divide %v5786, %v5787 : tensor<576xf32>
    %v5789 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v5790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5791 = stablehlo.constant dense<1568.0> : tensor<576xf32>
    %v5792 = stablehlo.reduce(%v5789 init: %v5790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5793 = stablehlo.divide %v5792, %v5791 : tensor<576xf32>
    %v5794 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v5795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5796 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v5797 = stablehlo.reduce(%v5794 init: %v5795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5798 = stablehlo.broadcast_in_dim %v5797, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v5799 = stablehlo.divide %v5798, %v5796 : tensor<32x576x7x7xf32>
    %v5800 = stablehlo.subtract %v5794, %v5799 : tensor<32x576x7x7xf32>
    %v5801 = stablehlo.multiply %v5800, %v5800 : tensor<32x576x7x7xf32>
    %v5802 = stablehlo.reduce(%v5801 init: %v5795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5803 = stablehlo.constant dense<1568.0> : tensor<576xf32>
    %v5804 = stablehlo.divide %v5802, %v5803 : tensor<576xf32>
    %v5805 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5807 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5808 = stablehlo.reduce(%v5805 init: %v5806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5809 = stablehlo.divide %v5808, %v5807 : tensor<160xf32>
    %v5810 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5812 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5813 = stablehlo.reduce(%v5810 init: %v5811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5814 = stablehlo.broadcast_in_dim %v5813, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5815 = stablehlo.divide %v5814, %v5812 : tensor<32x160x7x7xf32>
    %v5816 = stablehlo.subtract %v5810, %v5815 : tensor<32x160x7x7xf32>
    %v5817 = stablehlo.multiply %v5816, %v5816 : tensor<32x160x7x7xf32>
    %v5818 = stablehlo.reduce(%v5817 init: %v5811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5819 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5820 = stablehlo.divide %v5818, %v5819 : tensor<160xf32>
    %v5821 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5823 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5824 = stablehlo.reduce(%v5821 init: %v5822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5825 = stablehlo.divide %v5824, %v5823 : tensor<960xf32>
    %v5826 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5828 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5829 = stablehlo.reduce(%v5826 init: %v5827) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5830 = stablehlo.broadcast_in_dim %v5829, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5831 = stablehlo.divide %v5830, %v5828 : tensor<32x960x7x7xf32>
    %v5832 = stablehlo.subtract %v5826, %v5831 : tensor<32x960x7x7xf32>
    %v5833 = stablehlo.multiply %v5832, %v5832 : tensor<32x960x7x7xf32>
    %v5834 = stablehlo.reduce(%v5833 init: %v5827) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5835 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5836 = stablehlo.divide %v5834, %v5835 : tensor<960xf32>
    %v5837 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5839 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5840 = stablehlo.reduce(%v5837 init: %v5838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5841 = stablehlo.divide %v5840, %v5839 : tensor<960xf32>
    %v5842 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5844 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5845 = stablehlo.reduce(%v5842 init: %v5843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5846 = stablehlo.broadcast_in_dim %v5845, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5847 = stablehlo.divide %v5846, %v5844 : tensor<32x960x7x7xf32>
    %v5848 = stablehlo.subtract %v5842, %v5847 : tensor<32x960x7x7xf32>
    %v5849 = stablehlo.multiply %v5848, %v5848 : tensor<32x960x7x7xf32>
    %v5850 = stablehlo.reduce(%v5849 init: %v5843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5851 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5852 = stablehlo.divide %v5850, %v5851 : tensor<960xf32>
    %v5853 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5855 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5856 = stablehlo.reduce(%v5853 init: %v5854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5857 = stablehlo.divide %v5856, %v5855 : tensor<160xf32>
    %v5858 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5859 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5860 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5861 = stablehlo.reduce(%v5858 init: %v5859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5862 = stablehlo.broadcast_in_dim %v5861, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5863 = stablehlo.divide %v5862, %v5860 : tensor<32x160x7x7xf32>
    %v5864 = stablehlo.subtract %v5858, %v5863 : tensor<32x160x7x7xf32>
    %v5865 = stablehlo.multiply %v5864, %v5864 : tensor<32x160x7x7xf32>
    %v5866 = stablehlo.reduce(%v5865 init: %v5859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5867 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5868 = stablehlo.divide %v5866, %v5867 : tensor<160xf32>
    %v5869 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5871 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5872 = stablehlo.reduce(%v5869 init: %v5870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5873 = stablehlo.divide %v5872, %v5871 : tensor<960xf32>
    %v5874 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5876 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5877 = stablehlo.reduce(%v5874 init: %v5875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5878 = stablehlo.broadcast_in_dim %v5877, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5879 = stablehlo.divide %v5878, %v5876 : tensor<32x960x7x7xf32>
    %v5880 = stablehlo.subtract %v5874, %v5879 : tensor<32x960x7x7xf32>
    %v5881 = stablehlo.multiply %v5880, %v5880 : tensor<32x960x7x7xf32>
    %v5882 = stablehlo.reduce(%v5881 init: %v5875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5883 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5884 = stablehlo.divide %v5882, %v5883 : tensor<960xf32>
    %v5885 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5887 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5888 = stablehlo.reduce(%v5885 init: %v5886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5889 = stablehlo.divide %v5888, %v5887 : tensor<960xf32>
    %v5890 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5892 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5893 = stablehlo.reduce(%v5890 init: %v5891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5894 = stablehlo.broadcast_in_dim %v5893, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5895 = stablehlo.divide %v5894, %v5892 : tensor<32x960x7x7xf32>
    %v5896 = stablehlo.subtract %v5890, %v5895 : tensor<32x960x7x7xf32>
    %v5897 = stablehlo.multiply %v5896, %v5896 : tensor<32x960x7x7xf32>
    %v5898 = stablehlo.reduce(%v5897 init: %v5891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5899 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5900 = stablehlo.divide %v5898, %v5899 : tensor<960xf32>
    %v5901 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5903 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5904 = stablehlo.reduce(%v5901 init: %v5902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5905 = stablehlo.divide %v5904, %v5903 : tensor<160xf32>
    %v5906 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5908 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5909 = stablehlo.reduce(%v5906 init: %v5907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5910 = stablehlo.broadcast_in_dim %v5909, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5911 = stablehlo.divide %v5910, %v5908 : tensor<32x160x7x7xf32>
    %v5912 = stablehlo.subtract %v5906, %v5911 : tensor<32x160x7x7xf32>
    %v5913 = stablehlo.multiply %v5912, %v5912 : tensor<32x160x7x7xf32>
    %v5914 = stablehlo.reduce(%v5913 init: %v5907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5915 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5916 = stablehlo.divide %v5914, %v5915 : tensor<160xf32>
    %v5917 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5919 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5920 = stablehlo.reduce(%v5917 init: %v5918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5921 = stablehlo.divide %v5920, %v5919 : tensor<960xf32>
    %v5922 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5924 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5925 = stablehlo.reduce(%v5922 init: %v5923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5926 = stablehlo.broadcast_in_dim %v5925, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5927 = stablehlo.divide %v5926, %v5924 : tensor<32x960x7x7xf32>
    %v5928 = stablehlo.subtract %v5922, %v5927 : tensor<32x960x7x7xf32>
    %v5929 = stablehlo.multiply %v5928, %v5928 : tensor<32x960x7x7xf32>
    %v5930 = stablehlo.reduce(%v5929 init: %v5923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5931 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5932 = stablehlo.divide %v5930, %v5931 : tensor<960xf32>
    %v5933 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5935 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5936 = stablehlo.reduce(%v5933 init: %v5934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5937 = stablehlo.divide %v5936, %v5935 : tensor<960xf32>
    %v5938 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5940 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5941 = stablehlo.reduce(%v5938 init: %v5939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5942 = stablehlo.broadcast_in_dim %v5941, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5943 = stablehlo.divide %v5942, %v5940 : tensor<32x960x7x7xf32>
    %v5944 = stablehlo.subtract %v5938, %v5943 : tensor<32x960x7x7xf32>
    %v5945 = stablehlo.multiply %v5944, %v5944 : tensor<32x960x7x7xf32>
    %v5946 = stablehlo.reduce(%v5945 init: %v5939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5947 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5948 = stablehlo.divide %v5946, %v5947 : tensor<960xf32>
    %v5949 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v5950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5951 = stablehlo.constant dense<1568.0> : tensor<320xf32>
    %v5952 = stablehlo.reduce(%v5949 init: %v5950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5953 = stablehlo.divide %v5952, %v5951 : tensor<320xf32>
    %v5954 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v5955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5956 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v5957 = stablehlo.reduce(%v5954 init: %v5955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5958 = stablehlo.broadcast_in_dim %v5957, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v5959 = stablehlo.divide %v5958, %v5956 : tensor<32x320x7x7xf32>
    %v5960 = stablehlo.subtract %v5954, %v5959 : tensor<32x320x7x7xf32>
    %v5961 = stablehlo.multiply %v5960, %v5960 : tensor<32x320x7x7xf32>
    %v5962 = stablehlo.reduce(%v5961 init: %v5955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5963 = stablehlo.constant dense<1568.0> : tensor<320xf32>
    %v5964 = stablehlo.divide %v5962, %v5963 : tensor<320xf32>
    %v5965 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v5966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5967 = stablehlo.constant dense<1568.0> : tensor<1280xf32>
    %v5968 = stablehlo.reduce(%v5965 init: %v5966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5969 = stablehlo.divide %v5968, %v5967 : tensor<1280xf32>
    %v5970 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v5971 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5972 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v5973 = stablehlo.reduce(%v5970 init: %v5971) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5974 = stablehlo.broadcast_in_dim %v5973, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v5975 = stablehlo.divide %v5974, %v5972 : tensor<32x1280x7x7xf32>
    %v5976 = stablehlo.subtract %v5970, %v5975 : tensor<32x1280x7x7xf32>
    %v5977 = stablehlo.multiply %v5976, %v5976 : tensor<32x1280x7x7xf32>
    %v5978 = stablehlo.reduce(%v5977 init: %v5971) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5979 = stablehlo.constant dense<1568.0> : tensor<1280xf32>
    %v5980 = stablehlo.divide %v5978, %v5979 : tensor<1280xf32>
    %rho = stablehlo.constant dense<0.900000> : tensor<f32>
    %orho = stablehlo.constant dense<0.100000> : tensor<f32>
    %mu = stablehlo.constant dense<0.900000> : tensor<f32>
    %eps = stablehlo.constant dense<1.000000> : tensor<f32>
    %wd = stablehlo.constant dense<0.000040> : tensor<f32>
    %v5981 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5982 = stablehlo.multiply %v5981, %sW : tensor<32x3x3x3xf32>
    %v5983 = stablehlo.add %v5982, %v5127 : tensor<32x3x3x3xf32>
    %v5984 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5985 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5986 = stablehlo.multiply %v5984, %sWv : tensor<32x3x3x3xf32>
    %v5987 = stablehlo.multiply %v5983, %v5983 : tensor<32x3x3x3xf32>
    %v5988 = stablehlo.multiply %v5985, %v5987 : tensor<32x3x3x3xf32>
    %v5989 = stablehlo.add %v5986, %v5988 : tensor<32x3x3x3xf32>
    %v5990 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5991 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5992 = stablehlo.multiply %v5990, %sWv : tensor<32x3x3x3xf32>
    %v5993 = stablehlo.multiply %v5983, %v5983 : tensor<32x3x3x3xf32>
    %v5994 = stablehlo.multiply %v5991, %v5993 : tensor<32x3x3x3xf32>
    %v5995 = stablehlo.add %v5992, %v5994 : tensor<32x3x3x3xf32>
    %v5996 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5997 = stablehlo.add %v5995, %v5996 : tensor<32x3x3x3xf32>
    %v5998 = stablehlo.sqrt %v5997 : tensor<32x3x3x3xf32>
    %v5999 = stablehlo.divide %v5983, %v5998 : tensor<32x3x3x3xf32>
    %v6000 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v6001 = stablehlo.multiply %v6000, %sWm : tensor<32x3x3x3xf32>
    %v6002 = stablehlo.add %v6001, %v5999 : tensor<32x3x3x3xf32>
    %v6003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v6004 = stablehlo.multiply %v6003, %v6002 : tensor<32x3x3x3xf32>
    %v6005 = stablehlo.subtract %sW, %v6004 : tensor<32x3x3x3xf32>
    %v6006 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6007 = stablehlo.multiply %v6006, %sg : tensor<32xf32>
    %v6008 = stablehlo.add %v6007, %v5145 : tensor<32xf32>
    %v6009 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6010 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6011 = stablehlo.multiply %v6009, %sgv : tensor<32xf32>
    %v6012 = stablehlo.multiply %v6008, %v6008 : tensor<32xf32>
    %v6013 = stablehlo.multiply %v6010, %v6012 : tensor<32xf32>
    %v6014 = stablehlo.add %v6011, %v6013 : tensor<32xf32>
    %v6015 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6016 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6017 = stablehlo.multiply %v6015, %sgv : tensor<32xf32>
    %v6018 = stablehlo.multiply %v6008, %v6008 : tensor<32xf32>
    %v6019 = stablehlo.multiply %v6016, %v6018 : tensor<32xf32>
    %v6020 = stablehlo.add %v6017, %v6019 : tensor<32xf32>
    %v6021 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6022 = stablehlo.add %v6020, %v6021 : tensor<32xf32>
    %v6023 = stablehlo.sqrt %v6022 : tensor<32xf32>
    %v6024 = stablehlo.divide %v6008, %v6023 : tensor<32xf32>
    %v6025 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6026 = stablehlo.multiply %v6025, %sgm : tensor<32xf32>
    %v6027 = stablehlo.add %v6026, %v6024 : tensor<32xf32>
    %v6028 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6029 = stablehlo.multiply %v6028, %v6027 : tensor<32xf32>
    %v6030 = stablehlo.subtract %sg, %v6029 : tensor<32xf32>
    %v6031 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6032 = stablehlo.multiply %v6031, %sbt : tensor<32xf32>
    %v6033 = stablehlo.add %v6032, %v5148 : tensor<32xf32>
    %v6034 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6035 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6036 = stablehlo.multiply %v6034, %sbtv : tensor<32xf32>
    %v6037 = stablehlo.multiply %v6033, %v6033 : tensor<32xf32>
    %v6038 = stablehlo.multiply %v6035, %v6037 : tensor<32xf32>
    %v6039 = stablehlo.add %v6036, %v6038 : tensor<32xf32>
    %v6040 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6041 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6042 = stablehlo.multiply %v6040, %sbtv : tensor<32xf32>
    %v6043 = stablehlo.multiply %v6033, %v6033 : tensor<32xf32>
    %v6044 = stablehlo.multiply %v6041, %v6043 : tensor<32xf32>
    %v6045 = stablehlo.add %v6042, %v6044 : tensor<32xf32>
    %v6046 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6047 = stablehlo.add %v6045, %v6046 : tensor<32xf32>
    %v6048 = stablehlo.sqrt %v6047 : tensor<32xf32>
    %v6049 = stablehlo.divide %v6033, %v6048 : tensor<32xf32>
    %v6050 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6051 = stablehlo.multiply %v6050, %sbtm : tensor<32xf32>
    %v6052 = stablehlo.add %v6051, %v6049 : tensor<32xf32>
    %v6053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6054 = stablehlo.multiply %v6053, %v6052 : tensor<32xf32>
    %v6055 = stablehlo.subtract %sbt, %v6054 : tensor<32xf32>
    %v6056 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6057 = stablehlo.multiply %v6056, %b1dW : tensor<32x1x3x3xf32>
    %v6058 = stablehlo.add %v6057, %v5033 : tensor<32x1x3x3xf32>
    %v6059 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6060 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6061 = stablehlo.multiply %v6059, %b1dWv : tensor<32x1x3x3xf32>
    %v6062 = stablehlo.multiply %v6058, %v6058 : tensor<32x1x3x3xf32>
    %v6063 = stablehlo.multiply %v6060, %v6062 : tensor<32x1x3x3xf32>
    %v6064 = stablehlo.add %v6061, %v6063 : tensor<32x1x3x3xf32>
    %v6065 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6066 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6067 = stablehlo.multiply %v6065, %b1dWv : tensor<32x1x3x3xf32>
    %v6068 = stablehlo.multiply %v6058, %v6058 : tensor<32x1x3x3xf32>
    %v6069 = stablehlo.multiply %v6066, %v6068 : tensor<32x1x3x3xf32>
    %v6070 = stablehlo.add %v6067, %v6069 : tensor<32x1x3x3xf32>
    %v6071 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6072 = stablehlo.add %v6070, %v6071 : tensor<32x1x3x3xf32>
    %v6073 = stablehlo.sqrt %v6072 : tensor<32x1x3x3xf32>
    %v6074 = stablehlo.divide %v6058, %v6073 : tensor<32x1x3x3xf32>
    %v6075 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6076 = stablehlo.multiply %v6075, %b1dWm : tensor<32x1x3x3xf32>
    %v6077 = stablehlo.add %v6076, %v6074 : tensor<32x1x3x3xf32>
    %v6078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6079 = stablehlo.multiply %v6078, %v6077 : tensor<32x1x3x3xf32>
    %v6080 = stablehlo.subtract %b1dW, %v6079 : tensor<32x1x3x3xf32>
    %v6081 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6082 = stablehlo.multiply %v6081, %b1dg : tensor<32xf32>
    %v6083 = stablehlo.add %v6082, %v5051 : tensor<32xf32>
    %v6084 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6085 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6086 = stablehlo.multiply %v6084, %b1dgv : tensor<32xf32>
    %v6087 = stablehlo.multiply %v6083, %v6083 : tensor<32xf32>
    %v6088 = stablehlo.multiply %v6085, %v6087 : tensor<32xf32>
    %v6089 = stablehlo.add %v6086, %v6088 : tensor<32xf32>
    %v6090 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6091 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6092 = stablehlo.multiply %v6090, %b1dgv : tensor<32xf32>
    %v6093 = stablehlo.multiply %v6083, %v6083 : tensor<32xf32>
    %v6094 = stablehlo.multiply %v6091, %v6093 : tensor<32xf32>
    %v6095 = stablehlo.add %v6092, %v6094 : tensor<32xf32>
    %v6096 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6097 = stablehlo.add %v6095, %v6096 : tensor<32xf32>
    %v6098 = stablehlo.sqrt %v6097 : tensor<32xf32>
    %v6099 = stablehlo.divide %v6083, %v6098 : tensor<32xf32>
    %v6100 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6101 = stablehlo.multiply %v6100, %b1dgm : tensor<32xf32>
    %v6102 = stablehlo.add %v6101, %v6099 : tensor<32xf32>
    %v6103 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6104 = stablehlo.multiply %v6103, %v6102 : tensor<32xf32>
    %v6105 = stablehlo.subtract %b1dg, %v6104 : tensor<32xf32>
    %v6106 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6107 = stablehlo.multiply %v6106, %b1dbt : tensor<32xf32>
    %v6108 = stablehlo.add %v6107, %v5054 : tensor<32xf32>
    %v6109 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6110 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6111 = stablehlo.multiply %v6109, %b1dbtv : tensor<32xf32>
    %v6112 = stablehlo.multiply %v6108, %v6108 : tensor<32xf32>
    %v6113 = stablehlo.multiply %v6110, %v6112 : tensor<32xf32>
    %v6114 = stablehlo.add %v6111, %v6113 : tensor<32xf32>
    %v6115 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6116 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6117 = stablehlo.multiply %v6115, %b1dbtv : tensor<32xf32>
    %v6118 = stablehlo.multiply %v6108, %v6108 : tensor<32xf32>
    %v6119 = stablehlo.multiply %v6116, %v6118 : tensor<32xf32>
    %v6120 = stablehlo.add %v6117, %v6119 : tensor<32xf32>
    %v6121 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6122 = stablehlo.add %v6120, %v6121 : tensor<32xf32>
    %v6123 = stablehlo.sqrt %v6122 : tensor<32xf32>
    %v6124 = stablehlo.divide %v6108, %v6123 : tensor<32xf32>
    %v6125 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6126 = stablehlo.multiply %v6125, %b1dbtm : tensor<32xf32>
    %v6127 = stablehlo.add %v6126, %v6124 : tensor<32xf32>
    %v6128 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6129 = stablehlo.multiply %v6128, %v6127 : tensor<32xf32>
    %v6130 = stablehlo.subtract %b1dbt, %v6129 : tensor<32xf32>
    %v6131 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6132 = stablehlo.multiply %v6131, %b1pW : tensor<16x32x1x1xf32>
    %v6133 = stablehlo.add %v6132, %v5060 : tensor<16x32x1x1xf32>
    %v6134 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6135 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6136 = stablehlo.multiply %v6134, %b1pWv : tensor<16x32x1x1xf32>
    %v6137 = stablehlo.multiply %v6133, %v6133 : tensor<16x32x1x1xf32>
    %v6138 = stablehlo.multiply %v6135, %v6137 : tensor<16x32x1x1xf32>
    %v6139 = stablehlo.add %v6136, %v6138 : tensor<16x32x1x1xf32>
    %v6140 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6141 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6142 = stablehlo.multiply %v6140, %b1pWv : tensor<16x32x1x1xf32>
    %v6143 = stablehlo.multiply %v6133, %v6133 : tensor<16x32x1x1xf32>
    %v6144 = stablehlo.multiply %v6141, %v6143 : tensor<16x32x1x1xf32>
    %v6145 = stablehlo.add %v6142, %v6144 : tensor<16x32x1x1xf32>
    %v6146 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6147 = stablehlo.add %v6145, %v6146 : tensor<16x32x1x1xf32>
    %v6148 = stablehlo.sqrt %v6147 : tensor<16x32x1x1xf32>
    %v6149 = stablehlo.divide %v6133, %v6148 : tensor<16x32x1x1xf32>
    %v6150 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6151 = stablehlo.multiply %v6150, %b1pWm : tensor<16x32x1x1xf32>
    %v6152 = stablehlo.add %v6151, %v6149 : tensor<16x32x1x1xf32>
    %v6153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6154 = stablehlo.multiply %v6153, %v6152 : tensor<16x32x1x1xf32>
    %v6155 = stablehlo.subtract %b1pW, %v6154 : tensor<16x32x1x1xf32>
    %v6156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6157 = stablehlo.multiply %v6156, %b1pg : tensor<16xf32>
    %v6158 = stablehlo.add %v6157, %v5078 : tensor<16xf32>
    %v6159 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6160 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6161 = stablehlo.multiply %v6159, %b1pgv : tensor<16xf32>
    %v6162 = stablehlo.multiply %v6158, %v6158 : tensor<16xf32>
    %v6163 = stablehlo.multiply %v6160, %v6162 : tensor<16xf32>
    %v6164 = stablehlo.add %v6161, %v6163 : tensor<16xf32>
    %v6165 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6166 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6167 = stablehlo.multiply %v6165, %b1pgv : tensor<16xf32>
    %v6168 = stablehlo.multiply %v6158, %v6158 : tensor<16xf32>
    %v6169 = stablehlo.multiply %v6166, %v6168 : tensor<16xf32>
    %v6170 = stablehlo.add %v6167, %v6169 : tensor<16xf32>
    %v6171 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6172 = stablehlo.add %v6170, %v6171 : tensor<16xf32>
    %v6173 = stablehlo.sqrt %v6172 : tensor<16xf32>
    %v6174 = stablehlo.divide %v6158, %v6173 : tensor<16xf32>
    %v6175 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6176 = stablehlo.multiply %v6175, %b1pgm : tensor<16xf32>
    %v6177 = stablehlo.add %v6176, %v6174 : tensor<16xf32>
    %v6178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6179 = stablehlo.multiply %v6178, %v6177 : tensor<16xf32>
    %v6180 = stablehlo.subtract %b1pg, %v6179 : tensor<16xf32>
    %v6181 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6182 = stablehlo.multiply %v6181, %b1pbt : tensor<16xf32>
    %v6183 = stablehlo.add %v6182, %v5081 : tensor<16xf32>
    %v6184 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6185 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6186 = stablehlo.multiply %v6184, %b1pbtv : tensor<16xf32>
    %v6187 = stablehlo.multiply %v6183, %v6183 : tensor<16xf32>
    %v6188 = stablehlo.multiply %v6185, %v6187 : tensor<16xf32>
    %v6189 = stablehlo.add %v6186, %v6188 : tensor<16xf32>
    %v6190 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6191 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6192 = stablehlo.multiply %v6190, %b1pbtv : tensor<16xf32>
    %v6193 = stablehlo.multiply %v6183, %v6183 : tensor<16xf32>
    %v6194 = stablehlo.multiply %v6191, %v6193 : tensor<16xf32>
    %v6195 = stablehlo.add %v6192, %v6194 : tensor<16xf32>
    %v6196 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6197 = stablehlo.add %v6195, %v6196 : tensor<16xf32>
    %v6198 = stablehlo.sqrt %v6197 : tensor<16xf32>
    %v6199 = stablehlo.divide %v6183, %v6198 : tensor<16xf32>
    %v6200 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6201 = stablehlo.multiply %v6200, %b1pbtm : tensor<16xf32>
    %v6202 = stablehlo.add %v6201, %v6199 : tensor<16xf32>
    %v6203 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6204 = stablehlo.multiply %v6203, %v6202 : tensor<16xf32>
    %v6205 = stablehlo.subtract %b1pbt, %v6204 : tensor<16xf32>
    %v6206 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6207 = stablehlo.multiply %v6206, %b2eW : tensor<96x16x1x1xf32>
    %v6208 = stablehlo.add %v6207, %v4872 : tensor<96x16x1x1xf32>
    %v6209 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6210 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6211 = stablehlo.multiply %v6209, %b2eWv : tensor<96x16x1x1xf32>
    %v6212 = stablehlo.multiply %v6208, %v6208 : tensor<96x16x1x1xf32>
    %v6213 = stablehlo.multiply %v6210, %v6212 : tensor<96x16x1x1xf32>
    %v6214 = stablehlo.add %v6211, %v6213 : tensor<96x16x1x1xf32>
    %v6215 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6216 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6217 = stablehlo.multiply %v6215, %b2eWv : tensor<96x16x1x1xf32>
    %v6218 = stablehlo.multiply %v6208, %v6208 : tensor<96x16x1x1xf32>
    %v6219 = stablehlo.multiply %v6216, %v6218 : tensor<96x16x1x1xf32>
    %v6220 = stablehlo.add %v6217, %v6219 : tensor<96x16x1x1xf32>
    %v6221 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6222 = stablehlo.add %v6220, %v6221 : tensor<96x16x1x1xf32>
    %v6223 = stablehlo.sqrt %v6222 : tensor<96x16x1x1xf32>
    %v6224 = stablehlo.divide %v6208, %v6223 : tensor<96x16x1x1xf32>
    %v6225 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6226 = stablehlo.multiply %v6225, %b2eWm : tensor<96x16x1x1xf32>
    %v6227 = stablehlo.add %v6226, %v6224 : tensor<96x16x1x1xf32>
    %v6228 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6229 = stablehlo.multiply %v6228, %v6227 : tensor<96x16x1x1xf32>
    %v6230 = stablehlo.subtract %b2eW, %v6229 : tensor<96x16x1x1xf32>
    %v6231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6232 = stablehlo.multiply %v6231, %b2eg : tensor<96xf32>
    %v6233 = stablehlo.add %v6232, %v4890 : tensor<96xf32>
    %v6234 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6235 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6236 = stablehlo.multiply %v6234, %b2egv : tensor<96xf32>
    %v6237 = stablehlo.multiply %v6233, %v6233 : tensor<96xf32>
    %v6238 = stablehlo.multiply %v6235, %v6237 : tensor<96xf32>
    %v6239 = stablehlo.add %v6236, %v6238 : tensor<96xf32>
    %v6240 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6241 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6242 = stablehlo.multiply %v6240, %b2egv : tensor<96xf32>
    %v6243 = stablehlo.multiply %v6233, %v6233 : tensor<96xf32>
    %v6244 = stablehlo.multiply %v6241, %v6243 : tensor<96xf32>
    %v6245 = stablehlo.add %v6242, %v6244 : tensor<96xf32>
    %v6246 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6247 = stablehlo.add %v6245, %v6246 : tensor<96xf32>
    %v6248 = stablehlo.sqrt %v6247 : tensor<96xf32>
    %v6249 = stablehlo.divide %v6233, %v6248 : tensor<96xf32>
    %v6250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6251 = stablehlo.multiply %v6250, %b2egm : tensor<96xf32>
    %v6252 = stablehlo.add %v6251, %v6249 : tensor<96xf32>
    %v6253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6254 = stablehlo.multiply %v6253, %v6252 : tensor<96xf32>
    %v6255 = stablehlo.subtract %b2eg, %v6254 : tensor<96xf32>
    %v6256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6257 = stablehlo.multiply %v6256, %b2ebt : tensor<96xf32>
    %v6258 = stablehlo.add %v6257, %v4893 : tensor<96xf32>
    %v6259 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6260 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6261 = stablehlo.multiply %v6259, %b2ebtv : tensor<96xf32>
    %v6262 = stablehlo.multiply %v6258, %v6258 : tensor<96xf32>
    %v6263 = stablehlo.multiply %v6260, %v6262 : tensor<96xf32>
    %v6264 = stablehlo.add %v6261, %v6263 : tensor<96xf32>
    %v6265 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6266 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6267 = stablehlo.multiply %v6265, %b2ebtv : tensor<96xf32>
    %v6268 = stablehlo.multiply %v6258, %v6258 : tensor<96xf32>
    %v6269 = stablehlo.multiply %v6266, %v6268 : tensor<96xf32>
    %v6270 = stablehlo.add %v6267, %v6269 : tensor<96xf32>
    %v6271 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6272 = stablehlo.add %v6270, %v6271 : tensor<96xf32>
    %v6273 = stablehlo.sqrt %v6272 : tensor<96xf32>
    %v6274 = stablehlo.divide %v6258, %v6273 : tensor<96xf32>
    %v6275 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6276 = stablehlo.multiply %v6275, %b2ebtm : tensor<96xf32>
    %v6277 = stablehlo.add %v6276, %v6274 : tensor<96xf32>
    %v6278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6279 = stablehlo.multiply %v6278, %v6277 : tensor<96xf32>
    %v6280 = stablehlo.subtract %b2ebt, %v6279 : tensor<96xf32>
    %v6281 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6282 = stablehlo.multiply %v6281, %b2dW : tensor<96x1x3x3xf32>
    %v6283 = stablehlo.add %v6282, %v4901 : tensor<96x1x3x3xf32>
    %v6284 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6285 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6286 = stablehlo.multiply %v6284, %b2dWv : tensor<96x1x3x3xf32>
    %v6287 = stablehlo.multiply %v6283, %v6283 : tensor<96x1x3x3xf32>
    %v6288 = stablehlo.multiply %v6285, %v6287 : tensor<96x1x3x3xf32>
    %v6289 = stablehlo.add %v6286, %v6288 : tensor<96x1x3x3xf32>
    %v6290 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6291 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6292 = stablehlo.multiply %v6290, %b2dWv : tensor<96x1x3x3xf32>
    %v6293 = stablehlo.multiply %v6283, %v6283 : tensor<96x1x3x3xf32>
    %v6294 = stablehlo.multiply %v6291, %v6293 : tensor<96x1x3x3xf32>
    %v6295 = stablehlo.add %v6292, %v6294 : tensor<96x1x3x3xf32>
    %v6296 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6297 = stablehlo.add %v6295, %v6296 : tensor<96x1x3x3xf32>
    %v6298 = stablehlo.sqrt %v6297 : tensor<96x1x3x3xf32>
    %v6299 = stablehlo.divide %v6283, %v6298 : tensor<96x1x3x3xf32>
    %v6300 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6301 = stablehlo.multiply %v6300, %b2dWm : tensor<96x1x3x3xf32>
    %v6302 = stablehlo.add %v6301, %v6299 : tensor<96x1x3x3xf32>
    %v6303 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6304 = stablehlo.multiply %v6303, %v6302 : tensor<96x1x3x3xf32>
    %v6305 = stablehlo.subtract %b2dW, %v6304 : tensor<96x1x3x3xf32>
    %v6306 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6307 = stablehlo.multiply %v6306, %b2dg : tensor<96xf32>
    %v6308 = stablehlo.add %v6307, %v4919 : tensor<96xf32>
    %v6309 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6310 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6311 = stablehlo.multiply %v6309, %b2dgv : tensor<96xf32>
    %v6312 = stablehlo.multiply %v6308, %v6308 : tensor<96xf32>
    %v6313 = stablehlo.multiply %v6310, %v6312 : tensor<96xf32>
    %v6314 = stablehlo.add %v6311, %v6313 : tensor<96xf32>
    %v6315 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6316 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6317 = stablehlo.multiply %v6315, %b2dgv : tensor<96xf32>
    %v6318 = stablehlo.multiply %v6308, %v6308 : tensor<96xf32>
    %v6319 = stablehlo.multiply %v6316, %v6318 : tensor<96xf32>
    %v6320 = stablehlo.add %v6317, %v6319 : tensor<96xf32>
    %v6321 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6322 = stablehlo.add %v6320, %v6321 : tensor<96xf32>
    %v6323 = stablehlo.sqrt %v6322 : tensor<96xf32>
    %v6324 = stablehlo.divide %v6308, %v6323 : tensor<96xf32>
    %v6325 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6326 = stablehlo.multiply %v6325, %b2dgm : tensor<96xf32>
    %v6327 = stablehlo.add %v6326, %v6324 : tensor<96xf32>
    %v6328 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6329 = stablehlo.multiply %v6328, %v6327 : tensor<96xf32>
    %v6330 = stablehlo.subtract %b2dg, %v6329 : tensor<96xf32>
    %v6331 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6332 = stablehlo.multiply %v6331, %b2dbt : tensor<96xf32>
    %v6333 = stablehlo.add %v6332, %v4922 : tensor<96xf32>
    %v6334 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6335 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6336 = stablehlo.multiply %v6334, %b2dbtv : tensor<96xf32>
    %v6337 = stablehlo.multiply %v6333, %v6333 : tensor<96xf32>
    %v6338 = stablehlo.multiply %v6335, %v6337 : tensor<96xf32>
    %v6339 = stablehlo.add %v6336, %v6338 : tensor<96xf32>
    %v6340 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6341 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6342 = stablehlo.multiply %v6340, %b2dbtv : tensor<96xf32>
    %v6343 = stablehlo.multiply %v6333, %v6333 : tensor<96xf32>
    %v6344 = stablehlo.multiply %v6341, %v6343 : tensor<96xf32>
    %v6345 = stablehlo.add %v6342, %v6344 : tensor<96xf32>
    %v6346 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6347 = stablehlo.add %v6345, %v6346 : tensor<96xf32>
    %v6348 = stablehlo.sqrt %v6347 : tensor<96xf32>
    %v6349 = stablehlo.divide %v6333, %v6348 : tensor<96xf32>
    %v6350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6351 = stablehlo.multiply %v6350, %b2dbtm : tensor<96xf32>
    %v6352 = stablehlo.add %v6351, %v6349 : tensor<96xf32>
    %v6353 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6354 = stablehlo.multiply %v6353, %v6352 : tensor<96xf32>
    %v6355 = stablehlo.subtract %b2dbt, %v6354 : tensor<96xf32>
    %v6356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6357 = stablehlo.multiply %v6356, %b2pW : tensor<24x96x1x1xf32>
    %v6358 = stablehlo.add %v6357, %v4928 : tensor<24x96x1x1xf32>
    %v6359 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6360 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6361 = stablehlo.multiply %v6359, %b2pWv : tensor<24x96x1x1xf32>
    %v6362 = stablehlo.multiply %v6358, %v6358 : tensor<24x96x1x1xf32>
    %v6363 = stablehlo.multiply %v6360, %v6362 : tensor<24x96x1x1xf32>
    %v6364 = stablehlo.add %v6361, %v6363 : tensor<24x96x1x1xf32>
    %v6365 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6366 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6367 = stablehlo.multiply %v6365, %b2pWv : tensor<24x96x1x1xf32>
    %v6368 = stablehlo.multiply %v6358, %v6358 : tensor<24x96x1x1xf32>
    %v6369 = stablehlo.multiply %v6366, %v6368 : tensor<24x96x1x1xf32>
    %v6370 = stablehlo.add %v6367, %v6369 : tensor<24x96x1x1xf32>
    %v6371 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6372 = stablehlo.add %v6370, %v6371 : tensor<24x96x1x1xf32>
    %v6373 = stablehlo.sqrt %v6372 : tensor<24x96x1x1xf32>
    %v6374 = stablehlo.divide %v6358, %v6373 : tensor<24x96x1x1xf32>
    %v6375 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6376 = stablehlo.multiply %v6375, %b2pWm : tensor<24x96x1x1xf32>
    %v6377 = stablehlo.add %v6376, %v6374 : tensor<24x96x1x1xf32>
    %v6378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6379 = stablehlo.multiply %v6378, %v6377 : tensor<24x96x1x1xf32>
    %v6380 = stablehlo.subtract %b2pW, %v6379 : tensor<24x96x1x1xf32>
    %v6381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6382 = stablehlo.multiply %v6381, %b2pg : tensor<24xf32>
    %v6383 = stablehlo.add %v6382, %v4946 : tensor<24xf32>
    %v6384 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6385 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6386 = stablehlo.multiply %v6384, %b2pgv : tensor<24xf32>
    %v6387 = stablehlo.multiply %v6383, %v6383 : tensor<24xf32>
    %v6388 = stablehlo.multiply %v6385, %v6387 : tensor<24xf32>
    %v6389 = stablehlo.add %v6386, %v6388 : tensor<24xf32>
    %v6390 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6391 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6392 = stablehlo.multiply %v6390, %b2pgv : tensor<24xf32>
    %v6393 = stablehlo.multiply %v6383, %v6383 : tensor<24xf32>
    %v6394 = stablehlo.multiply %v6391, %v6393 : tensor<24xf32>
    %v6395 = stablehlo.add %v6392, %v6394 : tensor<24xf32>
    %v6396 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6397 = stablehlo.add %v6395, %v6396 : tensor<24xf32>
    %v6398 = stablehlo.sqrt %v6397 : tensor<24xf32>
    %v6399 = stablehlo.divide %v6383, %v6398 : tensor<24xf32>
    %v6400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6401 = stablehlo.multiply %v6400, %b2pgm : tensor<24xf32>
    %v6402 = stablehlo.add %v6401, %v6399 : tensor<24xf32>
    %v6403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6404 = stablehlo.multiply %v6403, %v6402 : tensor<24xf32>
    %v6405 = stablehlo.subtract %b2pg, %v6404 : tensor<24xf32>
    %v6406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6407 = stablehlo.multiply %v6406, %b2pbt : tensor<24xf32>
    %v6408 = stablehlo.add %v6407, %v4949 : tensor<24xf32>
    %v6409 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6410 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6411 = stablehlo.multiply %v6409, %b2pbtv : tensor<24xf32>
    %v6412 = stablehlo.multiply %v6408, %v6408 : tensor<24xf32>
    %v6413 = stablehlo.multiply %v6410, %v6412 : tensor<24xf32>
    %v6414 = stablehlo.add %v6411, %v6413 : tensor<24xf32>
    %v6415 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6416 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6417 = stablehlo.multiply %v6415, %b2pbtv : tensor<24xf32>
    %v6418 = stablehlo.multiply %v6408, %v6408 : tensor<24xf32>
    %v6419 = stablehlo.multiply %v6416, %v6418 : tensor<24xf32>
    %v6420 = stablehlo.add %v6417, %v6419 : tensor<24xf32>
    %v6421 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6422 = stablehlo.add %v6420, %v6421 : tensor<24xf32>
    %v6423 = stablehlo.sqrt %v6422 : tensor<24xf32>
    %v6424 = stablehlo.divide %v6408, %v6423 : tensor<24xf32>
    %v6425 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6426 = stablehlo.multiply %v6425, %b2pbtm : tensor<24xf32>
    %v6427 = stablehlo.add %v6426, %v6424 : tensor<24xf32>
    %v6428 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6429 = stablehlo.multiply %v6428, %v6427 : tensor<24xf32>
    %v6430 = stablehlo.subtract %b2pbt, %v6429 : tensor<24xf32>
    %v6431 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6432 = stablehlo.multiply %v6431, %b3eW : tensor<144x24x1x1xf32>
    %v6433 = stablehlo.add %v6432, %v4669 : tensor<144x24x1x1xf32>
    %v6434 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6435 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6436 = stablehlo.multiply %v6434, %b3eWv : tensor<144x24x1x1xf32>
    %v6437 = stablehlo.multiply %v6433, %v6433 : tensor<144x24x1x1xf32>
    %v6438 = stablehlo.multiply %v6435, %v6437 : tensor<144x24x1x1xf32>
    %v6439 = stablehlo.add %v6436, %v6438 : tensor<144x24x1x1xf32>
    %v6440 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6441 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6442 = stablehlo.multiply %v6440, %b3eWv : tensor<144x24x1x1xf32>
    %v6443 = stablehlo.multiply %v6433, %v6433 : tensor<144x24x1x1xf32>
    %v6444 = stablehlo.multiply %v6441, %v6443 : tensor<144x24x1x1xf32>
    %v6445 = stablehlo.add %v6442, %v6444 : tensor<144x24x1x1xf32>
    %v6446 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6447 = stablehlo.add %v6445, %v6446 : tensor<144x24x1x1xf32>
    %v6448 = stablehlo.sqrt %v6447 : tensor<144x24x1x1xf32>
    %v6449 = stablehlo.divide %v6433, %v6448 : tensor<144x24x1x1xf32>
    %v6450 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6451 = stablehlo.multiply %v6450, %b3eWm : tensor<144x24x1x1xf32>
    %v6452 = stablehlo.add %v6451, %v6449 : tensor<144x24x1x1xf32>
    %v6453 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6454 = stablehlo.multiply %v6453, %v6452 : tensor<144x24x1x1xf32>
    %v6455 = stablehlo.subtract %b3eW, %v6454 : tensor<144x24x1x1xf32>
    %v6456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6457 = stablehlo.multiply %v6456, %b3eg : tensor<144xf32>
    %v6458 = stablehlo.add %v6457, %v4687 : tensor<144xf32>
    %v6459 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6460 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6461 = stablehlo.multiply %v6459, %b3egv : tensor<144xf32>
    %v6462 = stablehlo.multiply %v6458, %v6458 : tensor<144xf32>
    %v6463 = stablehlo.multiply %v6460, %v6462 : tensor<144xf32>
    %v6464 = stablehlo.add %v6461, %v6463 : tensor<144xf32>
    %v6465 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6466 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6467 = stablehlo.multiply %v6465, %b3egv : tensor<144xf32>
    %v6468 = stablehlo.multiply %v6458, %v6458 : tensor<144xf32>
    %v6469 = stablehlo.multiply %v6466, %v6468 : tensor<144xf32>
    %v6470 = stablehlo.add %v6467, %v6469 : tensor<144xf32>
    %v6471 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6472 = stablehlo.add %v6470, %v6471 : tensor<144xf32>
    %v6473 = stablehlo.sqrt %v6472 : tensor<144xf32>
    %v6474 = stablehlo.divide %v6458, %v6473 : tensor<144xf32>
    %v6475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6476 = stablehlo.multiply %v6475, %b3egm : tensor<144xf32>
    %v6477 = stablehlo.add %v6476, %v6474 : tensor<144xf32>
    %v6478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6479 = stablehlo.multiply %v6478, %v6477 : tensor<144xf32>
    %v6480 = stablehlo.subtract %b3eg, %v6479 : tensor<144xf32>
    %v6481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6482 = stablehlo.multiply %v6481, %b3ebt : tensor<144xf32>
    %v6483 = stablehlo.add %v6482, %v4690 : tensor<144xf32>
    %v6484 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6485 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6486 = stablehlo.multiply %v6484, %b3ebtv : tensor<144xf32>
    %v6487 = stablehlo.multiply %v6483, %v6483 : tensor<144xf32>
    %v6488 = stablehlo.multiply %v6485, %v6487 : tensor<144xf32>
    %v6489 = stablehlo.add %v6486, %v6488 : tensor<144xf32>
    %v6490 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6491 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6492 = stablehlo.multiply %v6490, %b3ebtv : tensor<144xf32>
    %v6493 = stablehlo.multiply %v6483, %v6483 : tensor<144xf32>
    %v6494 = stablehlo.multiply %v6491, %v6493 : tensor<144xf32>
    %v6495 = stablehlo.add %v6492, %v6494 : tensor<144xf32>
    %v6496 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6497 = stablehlo.add %v6495, %v6496 : tensor<144xf32>
    %v6498 = stablehlo.sqrt %v6497 : tensor<144xf32>
    %v6499 = stablehlo.divide %v6483, %v6498 : tensor<144xf32>
    %v6500 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6501 = stablehlo.multiply %v6500, %b3ebtm : tensor<144xf32>
    %v6502 = stablehlo.add %v6501, %v6499 : tensor<144xf32>
    %v6503 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6504 = stablehlo.multiply %v6503, %v6502 : tensor<144xf32>
    %v6505 = stablehlo.subtract %b3ebt, %v6504 : tensor<144xf32>
    %v6506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6507 = stablehlo.multiply %v6506, %b3dW : tensor<144x1x3x3xf32>
    %v6508 = stablehlo.add %v6507, %v4696 : tensor<144x1x3x3xf32>
    %v6509 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6510 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6511 = stablehlo.multiply %v6509, %b3dWv : tensor<144x1x3x3xf32>
    %v6512 = stablehlo.multiply %v6508, %v6508 : tensor<144x1x3x3xf32>
    %v6513 = stablehlo.multiply %v6510, %v6512 : tensor<144x1x3x3xf32>
    %v6514 = stablehlo.add %v6511, %v6513 : tensor<144x1x3x3xf32>
    %v6515 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6516 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6517 = stablehlo.multiply %v6515, %b3dWv : tensor<144x1x3x3xf32>
    %v6518 = stablehlo.multiply %v6508, %v6508 : tensor<144x1x3x3xf32>
    %v6519 = stablehlo.multiply %v6516, %v6518 : tensor<144x1x3x3xf32>
    %v6520 = stablehlo.add %v6517, %v6519 : tensor<144x1x3x3xf32>
    %v6521 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6522 = stablehlo.add %v6520, %v6521 : tensor<144x1x3x3xf32>
    %v6523 = stablehlo.sqrt %v6522 : tensor<144x1x3x3xf32>
    %v6524 = stablehlo.divide %v6508, %v6523 : tensor<144x1x3x3xf32>
    %v6525 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6526 = stablehlo.multiply %v6525, %b3dWm : tensor<144x1x3x3xf32>
    %v6527 = stablehlo.add %v6526, %v6524 : tensor<144x1x3x3xf32>
    %v6528 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6529 = stablehlo.multiply %v6528, %v6527 : tensor<144x1x3x3xf32>
    %v6530 = stablehlo.subtract %b3dW, %v6529 : tensor<144x1x3x3xf32>
    %v6531 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6532 = stablehlo.multiply %v6531, %b3dg : tensor<144xf32>
    %v6533 = stablehlo.add %v6532, %v4714 : tensor<144xf32>
    %v6534 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6535 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6536 = stablehlo.multiply %v6534, %b3dgv : tensor<144xf32>
    %v6537 = stablehlo.multiply %v6533, %v6533 : tensor<144xf32>
    %v6538 = stablehlo.multiply %v6535, %v6537 : tensor<144xf32>
    %v6539 = stablehlo.add %v6536, %v6538 : tensor<144xf32>
    %v6540 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6541 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6542 = stablehlo.multiply %v6540, %b3dgv : tensor<144xf32>
    %v6543 = stablehlo.multiply %v6533, %v6533 : tensor<144xf32>
    %v6544 = stablehlo.multiply %v6541, %v6543 : tensor<144xf32>
    %v6545 = stablehlo.add %v6542, %v6544 : tensor<144xf32>
    %v6546 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6547 = stablehlo.add %v6545, %v6546 : tensor<144xf32>
    %v6548 = stablehlo.sqrt %v6547 : tensor<144xf32>
    %v6549 = stablehlo.divide %v6533, %v6548 : tensor<144xf32>
    %v6550 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6551 = stablehlo.multiply %v6550, %b3dgm : tensor<144xf32>
    %v6552 = stablehlo.add %v6551, %v6549 : tensor<144xf32>
    %v6553 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6554 = stablehlo.multiply %v6553, %v6552 : tensor<144xf32>
    %v6555 = stablehlo.subtract %b3dg, %v6554 : tensor<144xf32>
    %v6556 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6557 = stablehlo.multiply %v6556, %b3dbt : tensor<144xf32>
    %v6558 = stablehlo.add %v6557, %v4717 : tensor<144xf32>
    %v6559 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6560 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6561 = stablehlo.multiply %v6559, %b3dbtv : tensor<144xf32>
    %v6562 = stablehlo.multiply %v6558, %v6558 : tensor<144xf32>
    %v6563 = stablehlo.multiply %v6560, %v6562 : tensor<144xf32>
    %v6564 = stablehlo.add %v6561, %v6563 : tensor<144xf32>
    %v6565 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6566 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6567 = stablehlo.multiply %v6565, %b3dbtv : tensor<144xf32>
    %v6568 = stablehlo.multiply %v6558, %v6558 : tensor<144xf32>
    %v6569 = stablehlo.multiply %v6566, %v6568 : tensor<144xf32>
    %v6570 = stablehlo.add %v6567, %v6569 : tensor<144xf32>
    %v6571 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6572 = stablehlo.add %v6570, %v6571 : tensor<144xf32>
    %v6573 = stablehlo.sqrt %v6572 : tensor<144xf32>
    %v6574 = stablehlo.divide %v6558, %v6573 : tensor<144xf32>
    %v6575 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6576 = stablehlo.multiply %v6575, %b3dbtm : tensor<144xf32>
    %v6577 = stablehlo.add %v6576, %v6574 : tensor<144xf32>
    %v6578 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6579 = stablehlo.multiply %v6578, %v6577 : tensor<144xf32>
    %v6580 = stablehlo.subtract %b3dbt, %v6579 : tensor<144xf32>
    %v6581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6582 = stablehlo.multiply %v6581, %b3pW : tensor<24x144x1x1xf32>
    %v6583 = stablehlo.add %v6582, %v4723 : tensor<24x144x1x1xf32>
    %v6584 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6585 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6586 = stablehlo.multiply %v6584, %b3pWv : tensor<24x144x1x1xf32>
    %v6587 = stablehlo.multiply %v6583, %v6583 : tensor<24x144x1x1xf32>
    %v6588 = stablehlo.multiply %v6585, %v6587 : tensor<24x144x1x1xf32>
    %v6589 = stablehlo.add %v6586, %v6588 : tensor<24x144x1x1xf32>
    %v6590 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6591 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6592 = stablehlo.multiply %v6590, %b3pWv : tensor<24x144x1x1xf32>
    %v6593 = stablehlo.multiply %v6583, %v6583 : tensor<24x144x1x1xf32>
    %v6594 = stablehlo.multiply %v6591, %v6593 : tensor<24x144x1x1xf32>
    %v6595 = stablehlo.add %v6592, %v6594 : tensor<24x144x1x1xf32>
    %v6596 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6597 = stablehlo.add %v6595, %v6596 : tensor<24x144x1x1xf32>
    %v6598 = stablehlo.sqrt %v6597 : tensor<24x144x1x1xf32>
    %v6599 = stablehlo.divide %v6583, %v6598 : tensor<24x144x1x1xf32>
    %v6600 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6601 = stablehlo.multiply %v6600, %b3pWm : tensor<24x144x1x1xf32>
    %v6602 = stablehlo.add %v6601, %v6599 : tensor<24x144x1x1xf32>
    %v6603 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6604 = stablehlo.multiply %v6603, %v6602 : tensor<24x144x1x1xf32>
    %v6605 = stablehlo.subtract %b3pW, %v6604 : tensor<24x144x1x1xf32>
    %v6606 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6607 = stablehlo.multiply %v6606, %b3pg : tensor<24xf32>
    %v6608 = stablehlo.add %v6607, %v4741 : tensor<24xf32>
    %v6609 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6610 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6611 = stablehlo.multiply %v6609, %b3pgv : tensor<24xf32>
    %v6612 = stablehlo.multiply %v6608, %v6608 : tensor<24xf32>
    %v6613 = stablehlo.multiply %v6610, %v6612 : tensor<24xf32>
    %v6614 = stablehlo.add %v6611, %v6613 : tensor<24xf32>
    %v6615 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6616 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6617 = stablehlo.multiply %v6615, %b3pgv : tensor<24xf32>
    %v6618 = stablehlo.multiply %v6608, %v6608 : tensor<24xf32>
    %v6619 = stablehlo.multiply %v6616, %v6618 : tensor<24xf32>
    %v6620 = stablehlo.add %v6617, %v6619 : tensor<24xf32>
    %v6621 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6622 = stablehlo.add %v6620, %v6621 : tensor<24xf32>
    %v6623 = stablehlo.sqrt %v6622 : tensor<24xf32>
    %v6624 = stablehlo.divide %v6608, %v6623 : tensor<24xf32>
    %v6625 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6626 = stablehlo.multiply %v6625, %b3pgm : tensor<24xf32>
    %v6627 = stablehlo.add %v6626, %v6624 : tensor<24xf32>
    %v6628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6629 = stablehlo.multiply %v6628, %v6627 : tensor<24xf32>
    %v6630 = stablehlo.subtract %b3pg, %v6629 : tensor<24xf32>
    %v6631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6632 = stablehlo.multiply %v6631, %b3pbt : tensor<24xf32>
    %v6633 = stablehlo.add %v6632, %v4744 : tensor<24xf32>
    %v6634 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6635 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6636 = stablehlo.multiply %v6634, %b3pbtv : tensor<24xf32>
    %v6637 = stablehlo.multiply %v6633, %v6633 : tensor<24xf32>
    %v6638 = stablehlo.multiply %v6635, %v6637 : tensor<24xf32>
    %v6639 = stablehlo.add %v6636, %v6638 : tensor<24xf32>
    %v6640 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6641 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6642 = stablehlo.multiply %v6640, %b3pbtv : tensor<24xf32>
    %v6643 = stablehlo.multiply %v6633, %v6633 : tensor<24xf32>
    %v6644 = stablehlo.multiply %v6641, %v6643 : tensor<24xf32>
    %v6645 = stablehlo.add %v6642, %v6644 : tensor<24xf32>
    %v6646 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6647 = stablehlo.add %v6645, %v6646 : tensor<24xf32>
    %v6648 = stablehlo.sqrt %v6647 : tensor<24xf32>
    %v6649 = stablehlo.divide %v6633, %v6648 : tensor<24xf32>
    %v6650 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6651 = stablehlo.multiply %v6650, %b3pbtm : tensor<24xf32>
    %v6652 = stablehlo.add %v6651, %v6649 : tensor<24xf32>
    %v6653 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6654 = stablehlo.multiply %v6653, %v6652 : tensor<24xf32>
    %v6655 = stablehlo.subtract %b3pbt, %v6654 : tensor<24xf32>
    %v6656 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6657 = stablehlo.multiply %v6656, %b4eW : tensor<144x24x1x1xf32>
    %v6658 = stablehlo.add %v6657, %v4460 : tensor<144x24x1x1xf32>
    %v6659 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6660 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6661 = stablehlo.multiply %v6659, %b4eWv : tensor<144x24x1x1xf32>
    %v6662 = stablehlo.multiply %v6658, %v6658 : tensor<144x24x1x1xf32>
    %v6663 = stablehlo.multiply %v6660, %v6662 : tensor<144x24x1x1xf32>
    %v6664 = stablehlo.add %v6661, %v6663 : tensor<144x24x1x1xf32>
    %v6665 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6666 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6667 = stablehlo.multiply %v6665, %b4eWv : tensor<144x24x1x1xf32>
    %v6668 = stablehlo.multiply %v6658, %v6658 : tensor<144x24x1x1xf32>
    %v6669 = stablehlo.multiply %v6666, %v6668 : tensor<144x24x1x1xf32>
    %v6670 = stablehlo.add %v6667, %v6669 : tensor<144x24x1x1xf32>
    %v6671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6672 = stablehlo.add %v6670, %v6671 : tensor<144x24x1x1xf32>
    %v6673 = stablehlo.sqrt %v6672 : tensor<144x24x1x1xf32>
    %v6674 = stablehlo.divide %v6658, %v6673 : tensor<144x24x1x1xf32>
    %v6675 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6676 = stablehlo.multiply %v6675, %b4eWm : tensor<144x24x1x1xf32>
    %v6677 = stablehlo.add %v6676, %v6674 : tensor<144x24x1x1xf32>
    %v6678 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6679 = stablehlo.multiply %v6678, %v6677 : tensor<144x24x1x1xf32>
    %v6680 = stablehlo.subtract %b4eW, %v6679 : tensor<144x24x1x1xf32>
    %v6681 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6682 = stablehlo.multiply %v6681, %b4eg : tensor<144xf32>
    %v6683 = stablehlo.add %v6682, %v4478 : tensor<144xf32>
    %v6684 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6685 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6686 = stablehlo.multiply %v6684, %b4egv : tensor<144xf32>
    %v6687 = stablehlo.multiply %v6683, %v6683 : tensor<144xf32>
    %v6688 = stablehlo.multiply %v6685, %v6687 : tensor<144xf32>
    %v6689 = stablehlo.add %v6686, %v6688 : tensor<144xf32>
    %v6690 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6691 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6692 = stablehlo.multiply %v6690, %b4egv : tensor<144xf32>
    %v6693 = stablehlo.multiply %v6683, %v6683 : tensor<144xf32>
    %v6694 = stablehlo.multiply %v6691, %v6693 : tensor<144xf32>
    %v6695 = stablehlo.add %v6692, %v6694 : tensor<144xf32>
    %v6696 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6697 = stablehlo.add %v6695, %v6696 : tensor<144xf32>
    %v6698 = stablehlo.sqrt %v6697 : tensor<144xf32>
    %v6699 = stablehlo.divide %v6683, %v6698 : tensor<144xf32>
    %v6700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6701 = stablehlo.multiply %v6700, %b4egm : tensor<144xf32>
    %v6702 = stablehlo.add %v6701, %v6699 : tensor<144xf32>
    %v6703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6704 = stablehlo.multiply %v6703, %v6702 : tensor<144xf32>
    %v6705 = stablehlo.subtract %b4eg, %v6704 : tensor<144xf32>
    %v6706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6707 = stablehlo.multiply %v6706, %b4ebt : tensor<144xf32>
    %v6708 = stablehlo.add %v6707, %v4481 : tensor<144xf32>
    %v6709 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6710 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6711 = stablehlo.multiply %v6709, %b4ebtv : tensor<144xf32>
    %v6712 = stablehlo.multiply %v6708, %v6708 : tensor<144xf32>
    %v6713 = stablehlo.multiply %v6710, %v6712 : tensor<144xf32>
    %v6714 = stablehlo.add %v6711, %v6713 : tensor<144xf32>
    %v6715 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6716 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6717 = stablehlo.multiply %v6715, %b4ebtv : tensor<144xf32>
    %v6718 = stablehlo.multiply %v6708, %v6708 : tensor<144xf32>
    %v6719 = stablehlo.multiply %v6716, %v6718 : tensor<144xf32>
    %v6720 = stablehlo.add %v6717, %v6719 : tensor<144xf32>
    %v6721 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6722 = stablehlo.add %v6720, %v6721 : tensor<144xf32>
    %v6723 = stablehlo.sqrt %v6722 : tensor<144xf32>
    %v6724 = stablehlo.divide %v6708, %v6723 : tensor<144xf32>
    %v6725 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6726 = stablehlo.multiply %v6725, %b4ebtm : tensor<144xf32>
    %v6727 = stablehlo.add %v6726, %v6724 : tensor<144xf32>
    %v6728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6729 = stablehlo.multiply %v6728, %v6727 : tensor<144xf32>
    %v6730 = stablehlo.subtract %b4ebt, %v6729 : tensor<144xf32>
    %v6731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6732 = stablehlo.multiply %v6731, %b4dW : tensor<144x1x3x3xf32>
    %v6733 = stablehlo.add %v6732, %v4489 : tensor<144x1x3x3xf32>
    %v6734 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6735 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6736 = stablehlo.multiply %v6734, %b4dWv : tensor<144x1x3x3xf32>
    %v6737 = stablehlo.multiply %v6733, %v6733 : tensor<144x1x3x3xf32>
    %v6738 = stablehlo.multiply %v6735, %v6737 : tensor<144x1x3x3xf32>
    %v6739 = stablehlo.add %v6736, %v6738 : tensor<144x1x3x3xf32>
    %v6740 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6741 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6742 = stablehlo.multiply %v6740, %b4dWv : tensor<144x1x3x3xf32>
    %v6743 = stablehlo.multiply %v6733, %v6733 : tensor<144x1x3x3xf32>
    %v6744 = stablehlo.multiply %v6741, %v6743 : tensor<144x1x3x3xf32>
    %v6745 = stablehlo.add %v6742, %v6744 : tensor<144x1x3x3xf32>
    %v6746 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6747 = stablehlo.add %v6745, %v6746 : tensor<144x1x3x3xf32>
    %v6748 = stablehlo.sqrt %v6747 : tensor<144x1x3x3xf32>
    %v6749 = stablehlo.divide %v6733, %v6748 : tensor<144x1x3x3xf32>
    %v6750 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6751 = stablehlo.multiply %v6750, %b4dWm : tensor<144x1x3x3xf32>
    %v6752 = stablehlo.add %v6751, %v6749 : tensor<144x1x3x3xf32>
    %v6753 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6754 = stablehlo.multiply %v6753, %v6752 : tensor<144x1x3x3xf32>
    %v6755 = stablehlo.subtract %b4dW, %v6754 : tensor<144x1x3x3xf32>
    %v6756 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6757 = stablehlo.multiply %v6756, %b4dg : tensor<144xf32>
    %v6758 = stablehlo.add %v6757, %v4507 : tensor<144xf32>
    %v6759 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6760 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6761 = stablehlo.multiply %v6759, %b4dgv : tensor<144xf32>
    %v6762 = stablehlo.multiply %v6758, %v6758 : tensor<144xf32>
    %v6763 = stablehlo.multiply %v6760, %v6762 : tensor<144xf32>
    %v6764 = stablehlo.add %v6761, %v6763 : tensor<144xf32>
    %v6765 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6766 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6767 = stablehlo.multiply %v6765, %b4dgv : tensor<144xf32>
    %v6768 = stablehlo.multiply %v6758, %v6758 : tensor<144xf32>
    %v6769 = stablehlo.multiply %v6766, %v6768 : tensor<144xf32>
    %v6770 = stablehlo.add %v6767, %v6769 : tensor<144xf32>
    %v6771 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6772 = stablehlo.add %v6770, %v6771 : tensor<144xf32>
    %v6773 = stablehlo.sqrt %v6772 : tensor<144xf32>
    %v6774 = stablehlo.divide %v6758, %v6773 : tensor<144xf32>
    %v6775 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6776 = stablehlo.multiply %v6775, %b4dgm : tensor<144xf32>
    %v6777 = stablehlo.add %v6776, %v6774 : tensor<144xf32>
    %v6778 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6779 = stablehlo.multiply %v6778, %v6777 : tensor<144xf32>
    %v6780 = stablehlo.subtract %b4dg, %v6779 : tensor<144xf32>
    %v6781 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6782 = stablehlo.multiply %v6781, %b4dbt : tensor<144xf32>
    %v6783 = stablehlo.add %v6782, %v4510 : tensor<144xf32>
    %v6784 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6785 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6786 = stablehlo.multiply %v6784, %b4dbtv : tensor<144xf32>
    %v6787 = stablehlo.multiply %v6783, %v6783 : tensor<144xf32>
    %v6788 = stablehlo.multiply %v6785, %v6787 : tensor<144xf32>
    %v6789 = stablehlo.add %v6786, %v6788 : tensor<144xf32>
    %v6790 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6791 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6792 = stablehlo.multiply %v6790, %b4dbtv : tensor<144xf32>
    %v6793 = stablehlo.multiply %v6783, %v6783 : tensor<144xf32>
    %v6794 = stablehlo.multiply %v6791, %v6793 : tensor<144xf32>
    %v6795 = stablehlo.add %v6792, %v6794 : tensor<144xf32>
    %v6796 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6797 = stablehlo.add %v6795, %v6796 : tensor<144xf32>
    %v6798 = stablehlo.sqrt %v6797 : tensor<144xf32>
    %v6799 = stablehlo.divide %v6783, %v6798 : tensor<144xf32>
    %v6800 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6801 = stablehlo.multiply %v6800, %b4dbtm : tensor<144xf32>
    %v6802 = stablehlo.add %v6801, %v6799 : tensor<144xf32>
    %v6803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6804 = stablehlo.multiply %v6803, %v6802 : tensor<144xf32>
    %v6805 = stablehlo.subtract %b4dbt, %v6804 : tensor<144xf32>
    %v6806 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6807 = stablehlo.multiply %v6806, %b4pW : tensor<32x144x1x1xf32>
    %v6808 = stablehlo.add %v6807, %v4516 : tensor<32x144x1x1xf32>
    %v6809 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6810 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6811 = stablehlo.multiply %v6809, %b4pWv : tensor<32x144x1x1xf32>
    %v6812 = stablehlo.multiply %v6808, %v6808 : tensor<32x144x1x1xf32>
    %v6813 = stablehlo.multiply %v6810, %v6812 : tensor<32x144x1x1xf32>
    %v6814 = stablehlo.add %v6811, %v6813 : tensor<32x144x1x1xf32>
    %v6815 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6816 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6817 = stablehlo.multiply %v6815, %b4pWv : tensor<32x144x1x1xf32>
    %v6818 = stablehlo.multiply %v6808, %v6808 : tensor<32x144x1x1xf32>
    %v6819 = stablehlo.multiply %v6816, %v6818 : tensor<32x144x1x1xf32>
    %v6820 = stablehlo.add %v6817, %v6819 : tensor<32x144x1x1xf32>
    %v6821 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6822 = stablehlo.add %v6820, %v6821 : tensor<32x144x1x1xf32>
    %v6823 = stablehlo.sqrt %v6822 : tensor<32x144x1x1xf32>
    %v6824 = stablehlo.divide %v6808, %v6823 : tensor<32x144x1x1xf32>
    %v6825 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6826 = stablehlo.multiply %v6825, %b4pWm : tensor<32x144x1x1xf32>
    %v6827 = stablehlo.add %v6826, %v6824 : tensor<32x144x1x1xf32>
    %v6828 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6829 = stablehlo.multiply %v6828, %v6827 : tensor<32x144x1x1xf32>
    %v6830 = stablehlo.subtract %b4pW, %v6829 : tensor<32x144x1x1xf32>
    %v6831 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6832 = stablehlo.multiply %v6831, %b4pg : tensor<32xf32>
    %v6833 = stablehlo.add %v6832, %v4534 : tensor<32xf32>
    %v6834 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6835 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6836 = stablehlo.multiply %v6834, %b4pgv : tensor<32xf32>
    %v6837 = stablehlo.multiply %v6833, %v6833 : tensor<32xf32>
    %v6838 = stablehlo.multiply %v6835, %v6837 : tensor<32xf32>
    %v6839 = stablehlo.add %v6836, %v6838 : tensor<32xf32>
    %v6840 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6841 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6842 = stablehlo.multiply %v6840, %b4pgv : tensor<32xf32>
    %v6843 = stablehlo.multiply %v6833, %v6833 : tensor<32xf32>
    %v6844 = stablehlo.multiply %v6841, %v6843 : tensor<32xf32>
    %v6845 = stablehlo.add %v6842, %v6844 : tensor<32xf32>
    %v6846 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6847 = stablehlo.add %v6845, %v6846 : tensor<32xf32>
    %v6848 = stablehlo.sqrt %v6847 : tensor<32xf32>
    %v6849 = stablehlo.divide %v6833, %v6848 : tensor<32xf32>
    %v6850 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6851 = stablehlo.multiply %v6850, %b4pgm : tensor<32xf32>
    %v6852 = stablehlo.add %v6851, %v6849 : tensor<32xf32>
    %v6853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6854 = stablehlo.multiply %v6853, %v6852 : tensor<32xf32>
    %v6855 = stablehlo.subtract %b4pg, %v6854 : tensor<32xf32>
    %v6856 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6857 = stablehlo.multiply %v6856, %b4pbt : tensor<32xf32>
    %v6858 = stablehlo.add %v6857, %v4537 : tensor<32xf32>
    %v6859 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6860 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6861 = stablehlo.multiply %v6859, %b4pbtv : tensor<32xf32>
    %v6862 = stablehlo.multiply %v6858, %v6858 : tensor<32xf32>
    %v6863 = stablehlo.multiply %v6860, %v6862 : tensor<32xf32>
    %v6864 = stablehlo.add %v6861, %v6863 : tensor<32xf32>
    %v6865 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6866 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6867 = stablehlo.multiply %v6865, %b4pbtv : tensor<32xf32>
    %v6868 = stablehlo.multiply %v6858, %v6858 : tensor<32xf32>
    %v6869 = stablehlo.multiply %v6866, %v6868 : tensor<32xf32>
    %v6870 = stablehlo.add %v6867, %v6869 : tensor<32xf32>
    %v6871 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6872 = stablehlo.add %v6870, %v6871 : tensor<32xf32>
    %v6873 = stablehlo.sqrt %v6872 : tensor<32xf32>
    %v6874 = stablehlo.divide %v6858, %v6873 : tensor<32xf32>
    %v6875 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6876 = stablehlo.multiply %v6875, %b4pbtm : tensor<32xf32>
    %v6877 = stablehlo.add %v6876, %v6874 : tensor<32xf32>
    %v6878 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6879 = stablehlo.multiply %v6878, %v6877 : tensor<32xf32>
    %v6880 = stablehlo.subtract %b4pbt, %v6879 : tensor<32xf32>
    %v6881 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6882 = stablehlo.multiply %v6881, %b5eW : tensor<192x32x1x1xf32>
    %v6883 = stablehlo.add %v6882, %v4257 : tensor<192x32x1x1xf32>
    %v6884 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6885 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6886 = stablehlo.multiply %v6884, %b5eWv : tensor<192x32x1x1xf32>
    %v6887 = stablehlo.multiply %v6883, %v6883 : tensor<192x32x1x1xf32>
    %v6888 = stablehlo.multiply %v6885, %v6887 : tensor<192x32x1x1xf32>
    %v6889 = stablehlo.add %v6886, %v6888 : tensor<192x32x1x1xf32>
    %v6890 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6891 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6892 = stablehlo.multiply %v6890, %b5eWv : tensor<192x32x1x1xf32>
    %v6893 = stablehlo.multiply %v6883, %v6883 : tensor<192x32x1x1xf32>
    %v6894 = stablehlo.multiply %v6891, %v6893 : tensor<192x32x1x1xf32>
    %v6895 = stablehlo.add %v6892, %v6894 : tensor<192x32x1x1xf32>
    %v6896 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6897 = stablehlo.add %v6895, %v6896 : tensor<192x32x1x1xf32>
    %v6898 = stablehlo.sqrt %v6897 : tensor<192x32x1x1xf32>
    %v6899 = stablehlo.divide %v6883, %v6898 : tensor<192x32x1x1xf32>
    %v6900 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6901 = stablehlo.multiply %v6900, %b5eWm : tensor<192x32x1x1xf32>
    %v6902 = stablehlo.add %v6901, %v6899 : tensor<192x32x1x1xf32>
    %v6903 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6904 = stablehlo.multiply %v6903, %v6902 : tensor<192x32x1x1xf32>
    %v6905 = stablehlo.subtract %b5eW, %v6904 : tensor<192x32x1x1xf32>
    %v6906 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6907 = stablehlo.multiply %v6906, %b5eg : tensor<192xf32>
    %v6908 = stablehlo.add %v6907, %v4275 : tensor<192xf32>
    %v6909 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6910 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6911 = stablehlo.multiply %v6909, %b5egv : tensor<192xf32>
    %v6912 = stablehlo.multiply %v6908, %v6908 : tensor<192xf32>
    %v6913 = stablehlo.multiply %v6910, %v6912 : tensor<192xf32>
    %v6914 = stablehlo.add %v6911, %v6913 : tensor<192xf32>
    %v6915 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6916 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6917 = stablehlo.multiply %v6915, %b5egv : tensor<192xf32>
    %v6918 = stablehlo.multiply %v6908, %v6908 : tensor<192xf32>
    %v6919 = stablehlo.multiply %v6916, %v6918 : tensor<192xf32>
    %v6920 = stablehlo.add %v6917, %v6919 : tensor<192xf32>
    %v6921 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6922 = stablehlo.add %v6920, %v6921 : tensor<192xf32>
    %v6923 = stablehlo.sqrt %v6922 : tensor<192xf32>
    %v6924 = stablehlo.divide %v6908, %v6923 : tensor<192xf32>
    %v6925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6926 = stablehlo.multiply %v6925, %b5egm : tensor<192xf32>
    %v6927 = stablehlo.add %v6926, %v6924 : tensor<192xf32>
    %v6928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6929 = stablehlo.multiply %v6928, %v6927 : tensor<192xf32>
    %v6930 = stablehlo.subtract %b5eg, %v6929 : tensor<192xf32>
    %v6931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6932 = stablehlo.multiply %v6931, %b5ebt : tensor<192xf32>
    %v6933 = stablehlo.add %v6932, %v4278 : tensor<192xf32>
    %v6934 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6935 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6936 = stablehlo.multiply %v6934, %b5ebtv : tensor<192xf32>
    %v6937 = stablehlo.multiply %v6933, %v6933 : tensor<192xf32>
    %v6938 = stablehlo.multiply %v6935, %v6937 : tensor<192xf32>
    %v6939 = stablehlo.add %v6936, %v6938 : tensor<192xf32>
    %v6940 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6941 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6942 = stablehlo.multiply %v6940, %b5ebtv : tensor<192xf32>
    %v6943 = stablehlo.multiply %v6933, %v6933 : tensor<192xf32>
    %v6944 = stablehlo.multiply %v6941, %v6943 : tensor<192xf32>
    %v6945 = stablehlo.add %v6942, %v6944 : tensor<192xf32>
    %v6946 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6947 = stablehlo.add %v6945, %v6946 : tensor<192xf32>
    %v6948 = stablehlo.sqrt %v6947 : tensor<192xf32>
    %v6949 = stablehlo.divide %v6933, %v6948 : tensor<192xf32>
    %v6950 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6951 = stablehlo.multiply %v6950, %b5ebtm : tensor<192xf32>
    %v6952 = stablehlo.add %v6951, %v6949 : tensor<192xf32>
    %v6953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6954 = stablehlo.multiply %v6953, %v6952 : tensor<192xf32>
    %v6955 = stablehlo.subtract %b5ebt, %v6954 : tensor<192xf32>
    %v6956 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6957 = stablehlo.multiply %v6956, %b5dW : tensor<192x1x3x3xf32>
    %v6958 = stablehlo.add %v6957, %v4284 : tensor<192x1x3x3xf32>
    %v6959 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6960 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6961 = stablehlo.multiply %v6959, %b5dWv : tensor<192x1x3x3xf32>
    %v6962 = stablehlo.multiply %v6958, %v6958 : tensor<192x1x3x3xf32>
    %v6963 = stablehlo.multiply %v6960, %v6962 : tensor<192x1x3x3xf32>
    %v6964 = stablehlo.add %v6961, %v6963 : tensor<192x1x3x3xf32>
    %v6965 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6966 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6967 = stablehlo.multiply %v6965, %b5dWv : tensor<192x1x3x3xf32>
    %v6968 = stablehlo.multiply %v6958, %v6958 : tensor<192x1x3x3xf32>
    %v6969 = stablehlo.multiply %v6966, %v6968 : tensor<192x1x3x3xf32>
    %v6970 = stablehlo.add %v6967, %v6969 : tensor<192x1x3x3xf32>
    %v6971 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6972 = stablehlo.add %v6970, %v6971 : tensor<192x1x3x3xf32>
    %v6973 = stablehlo.sqrt %v6972 : tensor<192x1x3x3xf32>
    %v6974 = stablehlo.divide %v6958, %v6973 : tensor<192x1x3x3xf32>
    %v6975 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6976 = stablehlo.multiply %v6975, %b5dWm : tensor<192x1x3x3xf32>
    %v6977 = stablehlo.add %v6976, %v6974 : tensor<192x1x3x3xf32>
    %v6978 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6979 = stablehlo.multiply %v6978, %v6977 : tensor<192x1x3x3xf32>
    %v6980 = stablehlo.subtract %b5dW, %v6979 : tensor<192x1x3x3xf32>
    %v6981 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6982 = stablehlo.multiply %v6981, %b5dg : tensor<192xf32>
    %v6983 = stablehlo.add %v6982, %v4302 : tensor<192xf32>
    %v6984 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6985 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6986 = stablehlo.multiply %v6984, %b5dgv : tensor<192xf32>
    %v6987 = stablehlo.multiply %v6983, %v6983 : tensor<192xf32>
    %v6988 = stablehlo.multiply %v6985, %v6987 : tensor<192xf32>
    %v6989 = stablehlo.add %v6986, %v6988 : tensor<192xf32>
    %v6990 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6991 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6992 = stablehlo.multiply %v6990, %b5dgv : tensor<192xf32>
    %v6993 = stablehlo.multiply %v6983, %v6983 : tensor<192xf32>
    %v6994 = stablehlo.multiply %v6991, %v6993 : tensor<192xf32>
    %v6995 = stablehlo.add %v6992, %v6994 : tensor<192xf32>
    %v6996 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6997 = stablehlo.add %v6995, %v6996 : tensor<192xf32>
    %v6998 = stablehlo.sqrt %v6997 : tensor<192xf32>
    %v6999 = stablehlo.divide %v6983, %v6998 : tensor<192xf32>
    %v7000 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7001 = stablehlo.multiply %v7000, %b5dgm : tensor<192xf32>
    %v7002 = stablehlo.add %v7001, %v6999 : tensor<192xf32>
    %v7003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7004 = stablehlo.multiply %v7003, %v7002 : tensor<192xf32>
    %v7005 = stablehlo.subtract %b5dg, %v7004 : tensor<192xf32>
    %v7006 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7007 = stablehlo.multiply %v7006, %b5dbt : tensor<192xf32>
    %v7008 = stablehlo.add %v7007, %v4305 : tensor<192xf32>
    %v7009 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7010 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7011 = stablehlo.multiply %v7009, %b5dbtv : tensor<192xf32>
    %v7012 = stablehlo.multiply %v7008, %v7008 : tensor<192xf32>
    %v7013 = stablehlo.multiply %v7010, %v7012 : tensor<192xf32>
    %v7014 = stablehlo.add %v7011, %v7013 : tensor<192xf32>
    %v7015 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7016 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7017 = stablehlo.multiply %v7015, %b5dbtv : tensor<192xf32>
    %v7018 = stablehlo.multiply %v7008, %v7008 : tensor<192xf32>
    %v7019 = stablehlo.multiply %v7016, %v7018 : tensor<192xf32>
    %v7020 = stablehlo.add %v7017, %v7019 : tensor<192xf32>
    %v7021 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7022 = stablehlo.add %v7020, %v7021 : tensor<192xf32>
    %v7023 = stablehlo.sqrt %v7022 : tensor<192xf32>
    %v7024 = stablehlo.divide %v7008, %v7023 : tensor<192xf32>
    %v7025 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7026 = stablehlo.multiply %v7025, %b5dbtm : tensor<192xf32>
    %v7027 = stablehlo.add %v7026, %v7024 : tensor<192xf32>
    %v7028 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7029 = stablehlo.multiply %v7028, %v7027 : tensor<192xf32>
    %v7030 = stablehlo.subtract %b5dbt, %v7029 : tensor<192xf32>
    %v7031 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7032 = stablehlo.multiply %v7031, %b5pW : tensor<32x192x1x1xf32>
    %v7033 = stablehlo.add %v7032, %v4311 : tensor<32x192x1x1xf32>
    %v7034 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7035 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7036 = stablehlo.multiply %v7034, %b5pWv : tensor<32x192x1x1xf32>
    %v7037 = stablehlo.multiply %v7033, %v7033 : tensor<32x192x1x1xf32>
    %v7038 = stablehlo.multiply %v7035, %v7037 : tensor<32x192x1x1xf32>
    %v7039 = stablehlo.add %v7036, %v7038 : tensor<32x192x1x1xf32>
    %v7040 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7041 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7042 = stablehlo.multiply %v7040, %b5pWv : tensor<32x192x1x1xf32>
    %v7043 = stablehlo.multiply %v7033, %v7033 : tensor<32x192x1x1xf32>
    %v7044 = stablehlo.multiply %v7041, %v7043 : tensor<32x192x1x1xf32>
    %v7045 = stablehlo.add %v7042, %v7044 : tensor<32x192x1x1xf32>
    %v7046 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7047 = stablehlo.add %v7045, %v7046 : tensor<32x192x1x1xf32>
    %v7048 = stablehlo.sqrt %v7047 : tensor<32x192x1x1xf32>
    %v7049 = stablehlo.divide %v7033, %v7048 : tensor<32x192x1x1xf32>
    %v7050 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7051 = stablehlo.multiply %v7050, %b5pWm : tensor<32x192x1x1xf32>
    %v7052 = stablehlo.add %v7051, %v7049 : tensor<32x192x1x1xf32>
    %v7053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7054 = stablehlo.multiply %v7053, %v7052 : tensor<32x192x1x1xf32>
    %v7055 = stablehlo.subtract %b5pW, %v7054 : tensor<32x192x1x1xf32>
    %v7056 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7057 = stablehlo.multiply %v7056, %b5pg : tensor<32xf32>
    %v7058 = stablehlo.add %v7057, %v4329 : tensor<32xf32>
    %v7059 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7060 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7061 = stablehlo.multiply %v7059, %b5pgv : tensor<32xf32>
    %v7062 = stablehlo.multiply %v7058, %v7058 : tensor<32xf32>
    %v7063 = stablehlo.multiply %v7060, %v7062 : tensor<32xf32>
    %v7064 = stablehlo.add %v7061, %v7063 : tensor<32xf32>
    %v7065 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7066 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7067 = stablehlo.multiply %v7065, %b5pgv : tensor<32xf32>
    %v7068 = stablehlo.multiply %v7058, %v7058 : tensor<32xf32>
    %v7069 = stablehlo.multiply %v7066, %v7068 : tensor<32xf32>
    %v7070 = stablehlo.add %v7067, %v7069 : tensor<32xf32>
    %v7071 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7072 = stablehlo.add %v7070, %v7071 : tensor<32xf32>
    %v7073 = stablehlo.sqrt %v7072 : tensor<32xf32>
    %v7074 = stablehlo.divide %v7058, %v7073 : tensor<32xf32>
    %v7075 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7076 = stablehlo.multiply %v7075, %b5pgm : tensor<32xf32>
    %v7077 = stablehlo.add %v7076, %v7074 : tensor<32xf32>
    %v7078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7079 = stablehlo.multiply %v7078, %v7077 : tensor<32xf32>
    %v7080 = stablehlo.subtract %b5pg, %v7079 : tensor<32xf32>
    %v7081 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7082 = stablehlo.multiply %v7081, %b5pbt : tensor<32xf32>
    %v7083 = stablehlo.add %v7082, %v4332 : tensor<32xf32>
    %v7084 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7085 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7086 = stablehlo.multiply %v7084, %b5pbtv : tensor<32xf32>
    %v7087 = stablehlo.multiply %v7083, %v7083 : tensor<32xf32>
    %v7088 = stablehlo.multiply %v7085, %v7087 : tensor<32xf32>
    %v7089 = stablehlo.add %v7086, %v7088 : tensor<32xf32>
    %v7090 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7091 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7092 = stablehlo.multiply %v7090, %b5pbtv : tensor<32xf32>
    %v7093 = stablehlo.multiply %v7083, %v7083 : tensor<32xf32>
    %v7094 = stablehlo.multiply %v7091, %v7093 : tensor<32xf32>
    %v7095 = stablehlo.add %v7092, %v7094 : tensor<32xf32>
    %v7096 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7097 = stablehlo.add %v7095, %v7096 : tensor<32xf32>
    %v7098 = stablehlo.sqrt %v7097 : tensor<32xf32>
    %v7099 = stablehlo.divide %v7083, %v7098 : tensor<32xf32>
    %v7100 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7101 = stablehlo.multiply %v7100, %b5pbtm : tensor<32xf32>
    %v7102 = stablehlo.add %v7101, %v7099 : tensor<32xf32>
    %v7103 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7104 = stablehlo.multiply %v7103, %v7102 : tensor<32xf32>
    %v7105 = stablehlo.subtract %b5pbt, %v7104 : tensor<32xf32>
    %v7106 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7107 = stablehlo.multiply %v7106, %b6eW : tensor<192x32x1x1xf32>
    %v7108 = stablehlo.add %v7107, %v4050 : tensor<192x32x1x1xf32>
    %v7109 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7110 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7111 = stablehlo.multiply %v7109, %b6eWv : tensor<192x32x1x1xf32>
    %v7112 = stablehlo.multiply %v7108, %v7108 : tensor<192x32x1x1xf32>
    %v7113 = stablehlo.multiply %v7110, %v7112 : tensor<192x32x1x1xf32>
    %v7114 = stablehlo.add %v7111, %v7113 : tensor<192x32x1x1xf32>
    %v7115 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7116 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7117 = stablehlo.multiply %v7115, %b6eWv : tensor<192x32x1x1xf32>
    %v7118 = stablehlo.multiply %v7108, %v7108 : tensor<192x32x1x1xf32>
    %v7119 = stablehlo.multiply %v7116, %v7118 : tensor<192x32x1x1xf32>
    %v7120 = stablehlo.add %v7117, %v7119 : tensor<192x32x1x1xf32>
    %v7121 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7122 = stablehlo.add %v7120, %v7121 : tensor<192x32x1x1xf32>
    %v7123 = stablehlo.sqrt %v7122 : tensor<192x32x1x1xf32>
    %v7124 = stablehlo.divide %v7108, %v7123 : tensor<192x32x1x1xf32>
    %v7125 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7126 = stablehlo.multiply %v7125, %b6eWm : tensor<192x32x1x1xf32>
    %v7127 = stablehlo.add %v7126, %v7124 : tensor<192x32x1x1xf32>
    %v7128 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7129 = stablehlo.multiply %v7128, %v7127 : tensor<192x32x1x1xf32>
    %v7130 = stablehlo.subtract %b6eW, %v7129 : tensor<192x32x1x1xf32>
    %v7131 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7132 = stablehlo.multiply %v7131, %b6eg : tensor<192xf32>
    %v7133 = stablehlo.add %v7132, %v4068 : tensor<192xf32>
    %v7134 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7135 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7136 = stablehlo.multiply %v7134, %b6egv : tensor<192xf32>
    %v7137 = stablehlo.multiply %v7133, %v7133 : tensor<192xf32>
    %v7138 = stablehlo.multiply %v7135, %v7137 : tensor<192xf32>
    %v7139 = stablehlo.add %v7136, %v7138 : tensor<192xf32>
    %v7140 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7141 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7142 = stablehlo.multiply %v7140, %b6egv : tensor<192xf32>
    %v7143 = stablehlo.multiply %v7133, %v7133 : tensor<192xf32>
    %v7144 = stablehlo.multiply %v7141, %v7143 : tensor<192xf32>
    %v7145 = stablehlo.add %v7142, %v7144 : tensor<192xf32>
    %v7146 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7147 = stablehlo.add %v7145, %v7146 : tensor<192xf32>
    %v7148 = stablehlo.sqrt %v7147 : tensor<192xf32>
    %v7149 = stablehlo.divide %v7133, %v7148 : tensor<192xf32>
    %v7150 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7151 = stablehlo.multiply %v7150, %b6egm : tensor<192xf32>
    %v7152 = stablehlo.add %v7151, %v7149 : tensor<192xf32>
    %v7153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7154 = stablehlo.multiply %v7153, %v7152 : tensor<192xf32>
    %v7155 = stablehlo.subtract %b6eg, %v7154 : tensor<192xf32>
    %v7156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7157 = stablehlo.multiply %v7156, %b6ebt : tensor<192xf32>
    %v7158 = stablehlo.add %v7157, %v4071 : tensor<192xf32>
    %v7159 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7160 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7161 = stablehlo.multiply %v7159, %b6ebtv : tensor<192xf32>
    %v7162 = stablehlo.multiply %v7158, %v7158 : tensor<192xf32>
    %v7163 = stablehlo.multiply %v7160, %v7162 : tensor<192xf32>
    %v7164 = stablehlo.add %v7161, %v7163 : tensor<192xf32>
    %v7165 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7166 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7167 = stablehlo.multiply %v7165, %b6ebtv : tensor<192xf32>
    %v7168 = stablehlo.multiply %v7158, %v7158 : tensor<192xf32>
    %v7169 = stablehlo.multiply %v7166, %v7168 : tensor<192xf32>
    %v7170 = stablehlo.add %v7167, %v7169 : tensor<192xf32>
    %v7171 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7172 = stablehlo.add %v7170, %v7171 : tensor<192xf32>
    %v7173 = stablehlo.sqrt %v7172 : tensor<192xf32>
    %v7174 = stablehlo.divide %v7158, %v7173 : tensor<192xf32>
    %v7175 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7176 = stablehlo.multiply %v7175, %b6ebtm : tensor<192xf32>
    %v7177 = stablehlo.add %v7176, %v7174 : tensor<192xf32>
    %v7178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7179 = stablehlo.multiply %v7178, %v7177 : tensor<192xf32>
    %v7180 = stablehlo.subtract %b6ebt, %v7179 : tensor<192xf32>
    %v7181 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7182 = stablehlo.multiply %v7181, %b6dW : tensor<192x1x3x3xf32>
    %v7183 = stablehlo.add %v7182, %v4077 : tensor<192x1x3x3xf32>
    %v7184 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7185 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7186 = stablehlo.multiply %v7184, %b6dWv : tensor<192x1x3x3xf32>
    %v7187 = stablehlo.multiply %v7183, %v7183 : tensor<192x1x3x3xf32>
    %v7188 = stablehlo.multiply %v7185, %v7187 : tensor<192x1x3x3xf32>
    %v7189 = stablehlo.add %v7186, %v7188 : tensor<192x1x3x3xf32>
    %v7190 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7191 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7192 = stablehlo.multiply %v7190, %b6dWv : tensor<192x1x3x3xf32>
    %v7193 = stablehlo.multiply %v7183, %v7183 : tensor<192x1x3x3xf32>
    %v7194 = stablehlo.multiply %v7191, %v7193 : tensor<192x1x3x3xf32>
    %v7195 = stablehlo.add %v7192, %v7194 : tensor<192x1x3x3xf32>
    %v7196 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7197 = stablehlo.add %v7195, %v7196 : tensor<192x1x3x3xf32>
    %v7198 = stablehlo.sqrt %v7197 : tensor<192x1x3x3xf32>
    %v7199 = stablehlo.divide %v7183, %v7198 : tensor<192x1x3x3xf32>
    %v7200 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7201 = stablehlo.multiply %v7200, %b6dWm : tensor<192x1x3x3xf32>
    %v7202 = stablehlo.add %v7201, %v7199 : tensor<192x1x3x3xf32>
    %v7203 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7204 = stablehlo.multiply %v7203, %v7202 : tensor<192x1x3x3xf32>
    %v7205 = stablehlo.subtract %b6dW, %v7204 : tensor<192x1x3x3xf32>
    %v7206 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7207 = stablehlo.multiply %v7206, %b6dg : tensor<192xf32>
    %v7208 = stablehlo.add %v7207, %v4095 : tensor<192xf32>
    %v7209 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7210 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7211 = stablehlo.multiply %v7209, %b6dgv : tensor<192xf32>
    %v7212 = stablehlo.multiply %v7208, %v7208 : tensor<192xf32>
    %v7213 = stablehlo.multiply %v7210, %v7212 : tensor<192xf32>
    %v7214 = stablehlo.add %v7211, %v7213 : tensor<192xf32>
    %v7215 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7216 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7217 = stablehlo.multiply %v7215, %b6dgv : tensor<192xf32>
    %v7218 = stablehlo.multiply %v7208, %v7208 : tensor<192xf32>
    %v7219 = stablehlo.multiply %v7216, %v7218 : tensor<192xf32>
    %v7220 = stablehlo.add %v7217, %v7219 : tensor<192xf32>
    %v7221 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7222 = stablehlo.add %v7220, %v7221 : tensor<192xf32>
    %v7223 = stablehlo.sqrt %v7222 : tensor<192xf32>
    %v7224 = stablehlo.divide %v7208, %v7223 : tensor<192xf32>
    %v7225 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7226 = stablehlo.multiply %v7225, %b6dgm : tensor<192xf32>
    %v7227 = stablehlo.add %v7226, %v7224 : tensor<192xf32>
    %v7228 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7229 = stablehlo.multiply %v7228, %v7227 : tensor<192xf32>
    %v7230 = stablehlo.subtract %b6dg, %v7229 : tensor<192xf32>
    %v7231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7232 = stablehlo.multiply %v7231, %b6dbt : tensor<192xf32>
    %v7233 = stablehlo.add %v7232, %v4098 : tensor<192xf32>
    %v7234 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7235 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7236 = stablehlo.multiply %v7234, %b6dbtv : tensor<192xf32>
    %v7237 = stablehlo.multiply %v7233, %v7233 : tensor<192xf32>
    %v7238 = stablehlo.multiply %v7235, %v7237 : tensor<192xf32>
    %v7239 = stablehlo.add %v7236, %v7238 : tensor<192xf32>
    %v7240 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7241 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7242 = stablehlo.multiply %v7240, %b6dbtv : tensor<192xf32>
    %v7243 = stablehlo.multiply %v7233, %v7233 : tensor<192xf32>
    %v7244 = stablehlo.multiply %v7241, %v7243 : tensor<192xf32>
    %v7245 = stablehlo.add %v7242, %v7244 : tensor<192xf32>
    %v7246 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7247 = stablehlo.add %v7245, %v7246 : tensor<192xf32>
    %v7248 = stablehlo.sqrt %v7247 : tensor<192xf32>
    %v7249 = stablehlo.divide %v7233, %v7248 : tensor<192xf32>
    %v7250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7251 = stablehlo.multiply %v7250, %b6dbtm : tensor<192xf32>
    %v7252 = stablehlo.add %v7251, %v7249 : tensor<192xf32>
    %v7253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7254 = stablehlo.multiply %v7253, %v7252 : tensor<192xf32>
    %v7255 = stablehlo.subtract %b6dbt, %v7254 : tensor<192xf32>
    %v7256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7257 = stablehlo.multiply %v7256, %b6pW : tensor<32x192x1x1xf32>
    %v7258 = stablehlo.add %v7257, %v4104 : tensor<32x192x1x1xf32>
    %v7259 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7260 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7261 = stablehlo.multiply %v7259, %b6pWv : tensor<32x192x1x1xf32>
    %v7262 = stablehlo.multiply %v7258, %v7258 : tensor<32x192x1x1xf32>
    %v7263 = stablehlo.multiply %v7260, %v7262 : tensor<32x192x1x1xf32>
    %v7264 = stablehlo.add %v7261, %v7263 : tensor<32x192x1x1xf32>
    %v7265 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7266 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7267 = stablehlo.multiply %v7265, %b6pWv : tensor<32x192x1x1xf32>
    %v7268 = stablehlo.multiply %v7258, %v7258 : tensor<32x192x1x1xf32>
    %v7269 = stablehlo.multiply %v7266, %v7268 : tensor<32x192x1x1xf32>
    %v7270 = stablehlo.add %v7267, %v7269 : tensor<32x192x1x1xf32>
    %v7271 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7272 = stablehlo.add %v7270, %v7271 : tensor<32x192x1x1xf32>
    %v7273 = stablehlo.sqrt %v7272 : tensor<32x192x1x1xf32>
    %v7274 = stablehlo.divide %v7258, %v7273 : tensor<32x192x1x1xf32>
    %v7275 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7276 = stablehlo.multiply %v7275, %b6pWm : tensor<32x192x1x1xf32>
    %v7277 = stablehlo.add %v7276, %v7274 : tensor<32x192x1x1xf32>
    %v7278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7279 = stablehlo.multiply %v7278, %v7277 : tensor<32x192x1x1xf32>
    %v7280 = stablehlo.subtract %b6pW, %v7279 : tensor<32x192x1x1xf32>
    %v7281 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7282 = stablehlo.multiply %v7281, %b6pg : tensor<32xf32>
    %v7283 = stablehlo.add %v7282, %v4122 : tensor<32xf32>
    %v7284 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7285 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7286 = stablehlo.multiply %v7284, %b6pgv : tensor<32xf32>
    %v7287 = stablehlo.multiply %v7283, %v7283 : tensor<32xf32>
    %v7288 = stablehlo.multiply %v7285, %v7287 : tensor<32xf32>
    %v7289 = stablehlo.add %v7286, %v7288 : tensor<32xf32>
    %v7290 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7291 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7292 = stablehlo.multiply %v7290, %b6pgv : tensor<32xf32>
    %v7293 = stablehlo.multiply %v7283, %v7283 : tensor<32xf32>
    %v7294 = stablehlo.multiply %v7291, %v7293 : tensor<32xf32>
    %v7295 = stablehlo.add %v7292, %v7294 : tensor<32xf32>
    %v7296 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7297 = stablehlo.add %v7295, %v7296 : tensor<32xf32>
    %v7298 = stablehlo.sqrt %v7297 : tensor<32xf32>
    %v7299 = stablehlo.divide %v7283, %v7298 : tensor<32xf32>
    %v7300 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7301 = stablehlo.multiply %v7300, %b6pgm : tensor<32xf32>
    %v7302 = stablehlo.add %v7301, %v7299 : tensor<32xf32>
    %v7303 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7304 = stablehlo.multiply %v7303, %v7302 : tensor<32xf32>
    %v7305 = stablehlo.subtract %b6pg, %v7304 : tensor<32xf32>
    %v7306 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7307 = stablehlo.multiply %v7306, %b6pbt : tensor<32xf32>
    %v7308 = stablehlo.add %v7307, %v4125 : tensor<32xf32>
    %v7309 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7310 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7311 = stablehlo.multiply %v7309, %b6pbtv : tensor<32xf32>
    %v7312 = stablehlo.multiply %v7308, %v7308 : tensor<32xf32>
    %v7313 = stablehlo.multiply %v7310, %v7312 : tensor<32xf32>
    %v7314 = stablehlo.add %v7311, %v7313 : tensor<32xf32>
    %v7315 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7316 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7317 = stablehlo.multiply %v7315, %b6pbtv : tensor<32xf32>
    %v7318 = stablehlo.multiply %v7308, %v7308 : tensor<32xf32>
    %v7319 = stablehlo.multiply %v7316, %v7318 : tensor<32xf32>
    %v7320 = stablehlo.add %v7317, %v7319 : tensor<32xf32>
    %v7321 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7322 = stablehlo.add %v7320, %v7321 : tensor<32xf32>
    %v7323 = stablehlo.sqrt %v7322 : tensor<32xf32>
    %v7324 = stablehlo.divide %v7308, %v7323 : tensor<32xf32>
    %v7325 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7326 = stablehlo.multiply %v7325, %b6pbtm : tensor<32xf32>
    %v7327 = stablehlo.add %v7326, %v7324 : tensor<32xf32>
    %v7328 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7329 = stablehlo.multiply %v7328, %v7327 : tensor<32xf32>
    %v7330 = stablehlo.subtract %b6pbt, %v7329 : tensor<32xf32>
    %v7331 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7332 = stablehlo.multiply %v7331, %b7eW : tensor<192x32x1x1xf32>
    %v7333 = stablehlo.add %v7332, %v3841 : tensor<192x32x1x1xf32>
    %v7334 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7335 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7336 = stablehlo.multiply %v7334, %b7eWv : tensor<192x32x1x1xf32>
    %v7337 = stablehlo.multiply %v7333, %v7333 : tensor<192x32x1x1xf32>
    %v7338 = stablehlo.multiply %v7335, %v7337 : tensor<192x32x1x1xf32>
    %v7339 = stablehlo.add %v7336, %v7338 : tensor<192x32x1x1xf32>
    %v7340 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7341 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7342 = stablehlo.multiply %v7340, %b7eWv : tensor<192x32x1x1xf32>
    %v7343 = stablehlo.multiply %v7333, %v7333 : tensor<192x32x1x1xf32>
    %v7344 = stablehlo.multiply %v7341, %v7343 : tensor<192x32x1x1xf32>
    %v7345 = stablehlo.add %v7342, %v7344 : tensor<192x32x1x1xf32>
    %v7346 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7347 = stablehlo.add %v7345, %v7346 : tensor<192x32x1x1xf32>
    %v7348 = stablehlo.sqrt %v7347 : tensor<192x32x1x1xf32>
    %v7349 = stablehlo.divide %v7333, %v7348 : tensor<192x32x1x1xf32>
    %v7350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7351 = stablehlo.multiply %v7350, %b7eWm : tensor<192x32x1x1xf32>
    %v7352 = stablehlo.add %v7351, %v7349 : tensor<192x32x1x1xf32>
    %v7353 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7354 = stablehlo.multiply %v7353, %v7352 : tensor<192x32x1x1xf32>
    %v7355 = stablehlo.subtract %b7eW, %v7354 : tensor<192x32x1x1xf32>
    %v7356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7357 = stablehlo.multiply %v7356, %b7eg : tensor<192xf32>
    %v7358 = stablehlo.add %v7357, %v3859 : tensor<192xf32>
    %v7359 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7360 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7361 = stablehlo.multiply %v7359, %b7egv : tensor<192xf32>
    %v7362 = stablehlo.multiply %v7358, %v7358 : tensor<192xf32>
    %v7363 = stablehlo.multiply %v7360, %v7362 : tensor<192xf32>
    %v7364 = stablehlo.add %v7361, %v7363 : tensor<192xf32>
    %v7365 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7366 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7367 = stablehlo.multiply %v7365, %b7egv : tensor<192xf32>
    %v7368 = stablehlo.multiply %v7358, %v7358 : tensor<192xf32>
    %v7369 = stablehlo.multiply %v7366, %v7368 : tensor<192xf32>
    %v7370 = stablehlo.add %v7367, %v7369 : tensor<192xf32>
    %v7371 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7372 = stablehlo.add %v7370, %v7371 : tensor<192xf32>
    %v7373 = stablehlo.sqrt %v7372 : tensor<192xf32>
    %v7374 = stablehlo.divide %v7358, %v7373 : tensor<192xf32>
    %v7375 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7376 = stablehlo.multiply %v7375, %b7egm : tensor<192xf32>
    %v7377 = stablehlo.add %v7376, %v7374 : tensor<192xf32>
    %v7378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7379 = stablehlo.multiply %v7378, %v7377 : tensor<192xf32>
    %v7380 = stablehlo.subtract %b7eg, %v7379 : tensor<192xf32>
    %v7381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7382 = stablehlo.multiply %v7381, %b7ebt : tensor<192xf32>
    %v7383 = stablehlo.add %v7382, %v3862 : tensor<192xf32>
    %v7384 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7385 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7386 = stablehlo.multiply %v7384, %b7ebtv : tensor<192xf32>
    %v7387 = stablehlo.multiply %v7383, %v7383 : tensor<192xf32>
    %v7388 = stablehlo.multiply %v7385, %v7387 : tensor<192xf32>
    %v7389 = stablehlo.add %v7386, %v7388 : tensor<192xf32>
    %v7390 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7391 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7392 = stablehlo.multiply %v7390, %b7ebtv : tensor<192xf32>
    %v7393 = stablehlo.multiply %v7383, %v7383 : tensor<192xf32>
    %v7394 = stablehlo.multiply %v7391, %v7393 : tensor<192xf32>
    %v7395 = stablehlo.add %v7392, %v7394 : tensor<192xf32>
    %v7396 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7397 = stablehlo.add %v7395, %v7396 : tensor<192xf32>
    %v7398 = stablehlo.sqrt %v7397 : tensor<192xf32>
    %v7399 = stablehlo.divide %v7383, %v7398 : tensor<192xf32>
    %v7400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7401 = stablehlo.multiply %v7400, %b7ebtm : tensor<192xf32>
    %v7402 = stablehlo.add %v7401, %v7399 : tensor<192xf32>
    %v7403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7404 = stablehlo.multiply %v7403, %v7402 : tensor<192xf32>
    %v7405 = stablehlo.subtract %b7ebt, %v7404 : tensor<192xf32>
    %v7406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7407 = stablehlo.multiply %v7406, %b7dW : tensor<192x1x3x3xf32>
    %v7408 = stablehlo.add %v7407, %v3870 : tensor<192x1x3x3xf32>
    %v7409 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7410 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7411 = stablehlo.multiply %v7409, %b7dWv : tensor<192x1x3x3xf32>
    %v7412 = stablehlo.multiply %v7408, %v7408 : tensor<192x1x3x3xf32>
    %v7413 = stablehlo.multiply %v7410, %v7412 : tensor<192x1x3x3xf32>
    %v7414 = stablehlo.add %v7411, %v7413 : tensor<192x1x3x3xf32>
    %v7415 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7416 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7417 = stablehlo.multiply %v7415, %b7dWv : tensor<192x1x3x3xf32>
    %v7418 = stablehlo.multiply %v7408, %v7408 : tensor<192x1x3x3xf32>
    %v7419 = stablehlo.multiply %v7416, %v7418 : tensor<192x1x3x3xf32>
    %v7420 = stablehlo.add %v7417, %v7419 : tensor<192x1x3x3xf32>
    %v7421 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7422 = stablehlo.add %v7420, %v7421 : tensor<192x1x3x3xf32>
    %v7423 = stablehlo.sqrt %v7422 : tensor<192x1x3x3xf32>
    %v7424 = stablehlo.divide %v7408, %v7423 : tensor<192x1x3x3xf32>
    %v7425 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7426 = stablehlo.multiply %v7425, %b7dWm : tensor<192x1x3x3xf32>
    %v7427 = stablehlo.add %v7426, %v7424 : tensor<192x1x3x3xf32>
    %v7428 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7429 = stablehlo.multiply %v7428, %v7427 : tensor<192x1x3x3xf32>
    %v7430 = stablehlo.subtract %b7dW, %v7429 : tensor<192x1x3x3xf32>
    %v7431 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7432 = stablehlo.multiply %v7431, %b7dg : tensor<192xf32>
    %v7433 = stablehlo.add %v7432, %v3888 : tensor<192xf32>
    %v7434 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7435 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7436 = stablehlo.multiply %v7434, %b7dgv : tensor<192xf32>
    %v7437 = stablehlo.multiply %v7433, %v7433 : tensor<192xf32>
    %v7438 = stablehlo.multiply %v7435, %v7437 : tensor<192xf32>
    %v7439 = stablehlo.add %v7436, %v7438 : tensor<192xf32>
    %v7440 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7441 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7442 = stablehlo.multiply %v7440, %b7dgv : tensor<192xf32>
    %v7443 = stablehlo.multiply %v7433, %v7433 : tensor<192xf32>
    %v7444 = stablehlo.multiply %v7441, %v7443 : tensor<192xf32>
    %v7445 = stablehlo.add %v7442, %v7444 : tensor<192xf32>
    %v7446 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7447 = stablehlo.add %v7445, %v7446 : tensor<192xf32>
    %v7448 = stablehlo.sqrt %v7447 : tensor<192xf32>
    %v7449 = stablehlo.divide %v7433, %v7448 : tensor<192xf32>
    %v7450 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7451 = stablehlo.multiply %v7450, %b7dgm : tensor<192xf32>
    %v7452 = stablehlo.add %v7451, %v7449 : tensor<192xf32>
    %v7453 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7454 = stablehlo.multiply %v7453, %v7452 : tensor<192xf32>
    %v7455 = stablehlo.subtract %b7dg, %v7454 : tensor<192xf32>
    %v7456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7457 = stablehlo.multiply %v7456, %b7dbt : tensor<192xf32>
    %v7458 = stablehlo.add %v7457, %v3891 : tensor<192xf32>
    %v7459 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7460 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7461 = stablehlo.multiply %v7459, %b7dbtv : tensor<192xf32>
    %v7462 = stablehlo.multiply %v7458, %v7458 : tensor<192xf32>
    %v7463 = stablehlo.multiply %v7460, %v7462 : tensor<192xf32>
    %v7464 = stablehlo.add %v7461, %v7463 : tensor<192xf32>
    %v7465 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7466 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7467 = stablehlo.multiply %v7465, %b7dbtv : tensor<192xf32>
    %v7468 = stablehlo.multiply %v7458, %v7458 : tensor<192xf32>
    %v7469 = stablehlo.multiply %v7466, %v7468 : tensor<192xf32>
    %v7470 = stablehlo.add %v7467, %v7469 : tensor<192xf32>
    %v7471 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7472 = stablehlo.add %v7470, %v7471 : tensor<192xf32>
    %v7473 = stablehlo.sqrt %v7472 : tensor<192xf32>
    %v7474 = stablehlo.divide %v7458, %v7473 : tensor<192xf32>
    %v7475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7476 = stablehlo.multiply %v7475, %b7dbtm : tensor<192xf32>
    %v7477 = stablehlo.add %v7476, %v7474 : tensor<192xf32>
    %v7478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7479 = stablehlo.multiply %v7478, %v7477 : tensor<192xf32>
    %v7480 = stablehlo.subtract %b7dbt, %v7479 : tensor<192xf32>
    %v7481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7482 = stablehlo.multiply %v7481, %b7pW : tensor<64x192x1x1xf32>
    %v7483 = stablehlo.add %v7482, %v3897 : tensor<64x192x1x1xf32>
    %v7484 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7485 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7486 = stablehlo.multiply %v7484, %b7pWv : tensor<64x192x1x1xf32>
    %v7487 = stablehlo.multiply %v7483, %v7483 : tensor<64x192x1x1xf32>
    %v7488 = stablehlo.multiply %v7485, %v7487 : tensor<64x192x1x1xf32>
    %v7489 = stablehlo.add %v7486, %v7488 : tensor<64x192x1x1xf32>
    %v7490 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7491 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7492 = stablehlo.multiply %v7490, %b7pWv : tensor<64x192x1x1xf32>
    %v7493 = stablehlo.multiply %v7483, %v7483 : tensor<64x192x1x1xf32>
    %v7494 = stablehlo.multiply %v7491, %v7493 : tensor<64x192x1x1xf32>
    %v7495 = stablehlo.add %v7492, %v7494 : tensor<64x192x1x1xf32>
    %v7496 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7497 = stablehlo.add %v7495, %v7496 : tensor<64x192x1x1xf32>
    %v7498 = stablehlo.sqrt %v7497 : tensor<64x192x1x1xf32>
    %v7499 = stablehlo.divide %v7483, %v7498 : tensor<64x192x1x1xf32>
    %v7500 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7501 = stablehlo.multiply %v7500, %b7pWm : tensor<64x192x1x1xf32>
    %v7502 = stablehlo.add %v7501, %v7499 : tensor<64x192x1x1xf32>
    %v7503 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7504 = stablehlo.multiply %v7503, %v7502 : tensor<64x192x1x1xf32>
    %v7505 = stablehlo.subtract %b7pW, %v7504 : tensor<64x192x1x1xf32>
    %v7506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7507 = stablehlo.multiply %v7506, %b7pg : tensor<64xf32>
    %v7508 = stablehlo.add %v7507, %v3915 : tensor<64xf32>
    %v7509 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7510 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7511 = stablehlo.multiply %v7509, %b7pgv : tensor<64xf32>
    %v7512 = stablehlo.multiply %v7508, %v7508 : tensor<64xf32>
    %v7513 = stablehlo.multiply %v7510, %v7512 : tensor<64xf32>
    %v7514 = stablehlo.add %v7511, %v7513 : tensor<64xf32>
    %v7515 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7516 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7517 = stablehlo.multiply %v7515, %b7pgv : tensor<64xf32>
    %v7518 = stablehlo.multiply %v7508, %v7508 : tensor<64xf32>
    %v7519 = stablehlo.multiply %v7516, %v7518 : tensor<64xf32>
    %v7520 = stablehlo.add %v7517, %v7519 : tensor<64xf32>
    %v7521 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7522 = stablehlo.add %v7520, %v7521 : tensor<64xf32>
    %v7523 = stablehlo.sqrt %v7522 : tensor<64xf32>
    %v7524 = stablehlo.divide %v7508, %v7523 : tensor<64xf32>
    %v7525 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7526 = stablehlo.multiply %v7525, %b7pgm : tensor<64xf32>
    %v7527 = stablehlo.add %v7526, %v7524 : tensor<64xf32>
    %v7528 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7529 = stablehlo.multiply %v7528, %v7527 : tensor<64xf32>
    %v7530 = stablehlo.subtract %b7pg, %v7529 : tensor<64xf32>
    %v7531 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7532 = stablehlo.multiply %v7531, %b7pbt : tensor<64xf32>
    %v7533 = stablehlo.add %v7532, %v3918 : tensor<64xf32>
    %v7534 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7535 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7536 = stablehlo.multiply %v7534, %b7pbtv : tensor<64xf32>
    %v7537 = stablehlo.multiply %v7533, %v7533 : tensor<64xf32>
    %v7538 = stablehlo.multiply %v7535, %v7537 : tensor<64xf32>
    %v7539 = stablehlo.add %v7536, %v7538 : tensor<64xf32>
    %v7540 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7541 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7542 = stablehlo.multiply %v7540, %b7pbtv : tensor<64xf32>
    %v7543 = stablehlo.multiply %v7533, %v7533 : tensor<64xf32>
    %v7544 = stablehlo.multiply %v7541, %v7543 : tensor<64xf32>
    %v7545 = stablehlo.add %v7542, %v7544 : tensor<64xf32>
    %v7546 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7547 = stablehlo.add %v7545, %v7546 : tensor<64xf32>
    %v7548 = stablehlo.sqrt %v7547 : tensor<64xf32>
    %v7549 = stablehlo.divide %v7533, %v7548 : tensor<64xf32>
    %v7550 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7551 = stablehlo.multiply %v7550, %b7pbtm : tensor<64xf32>
    %v7552 = stablehlo.add %v7551, %v7549 : tensor<64xf32>
    %v7553 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7554 = stablehlo.multiply %v7553, %v7552 : tensor<64xf32>
    %v7555 = stablehlo.subtract %b7pbt, %v7554 : tensor<64xf32>
    %v7556 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7557 = stablehlo.multiply %v7556, %b8eW : tensor<384x64x1x1xf32>
    %v7558 = stablehlo.add %v7557, %v3638 : tensor<384x64x1x1xf32>
    %v7559 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7560 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7561 = stablehlo.multiply %v7559, %b8eWv : tensor<384x64x1x1xf32>
    %v7562 = stablehlo.multiply %v7558, %v7558 : tensor<384x64x1x1xf32>
    %v7563 = stablehlo.multiply %v7560, %v7562 : tensor<384x64x1x1xf32>
    %v7564 = stablehlo.add %v7561, %v7563 : tensor<384x64x1x1xf32>
    %v7565 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7566 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7567 = stablehlo.multiply %v7565, %b8eWv : tensor<384x64x1x1xf32>
    %v7568 = stablehlo.multiply %v7558, %v7558 : tensor<384x64x1x1xf32>
    %v7569 = stablehlo.multiply %v7566, %v7568 : tensor<384x64x1x1xf32>
    %v7570 = stablehlo.add %v7567, %v7569 : tensor<384x64x1x1xf32>
    %v7571 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7572 = stablehlo.add %v7570, %v7571 : tensor<384x64x1x1xf32>
    %v7573 = stablehlo.sqrt %v7572 : tensor<384x64x1x1xf32>
    %v7574 = stablehlo.divide %v7558, %v7573 : tensor<384x64x1x1xf32>
    %v7575 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7576 = stablehlo.multiply %v7575, %b8eWm : tensor<384x64x1x1xf32>
    %v7577 = stablehlo.add %v7576, %v7574 : tensor<384x64x1x1xf32>
    %v7578 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7579 = stablehlo.multiply %v7578, %v7577 : tensor<384x64x1x1xf32>
    %v7580 = stablehlo.subtract %b8eW, %v7579 : tensor<384x64x1x1xf32>
    %v7581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7582 = stablehlo.multiply %v7581, %b8eg : tensor<384xf32>
    %v7583 = stablehlo.add %v7582, %v3656 : tensor<384xf32>
    %v7584 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7585 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7586 = stablehlo.multiply %v7584, %b8egv : tensor<384xf32>
    %v7587 = stablehlo.multiply %v7583, %v7583 : tensor<384xf32>
    %v7588 = stablehlo.multiply %v7585, %v7587 : tensor<384xf32>
    %v7589 = stablehlo.add %v7586, %v7588 : tensor<384xf32>
    %v7590 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7591 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7592 = stablehlo.multiply %v7590, %b8egv : tensor<384xf32>
    %v7593 = stablehlo.multiply %v7583, %v7583 : tensor<384xf32>
    %v7594 = stablehlo.multiply %v7591, %v7593 : tensor<384xf32>
    %v7595 = stablehlo.add %v7592, %v7594 : tensor<384xf32>
    %v7596 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7597 = stablehlo.add %v7595, %v7596 : tensor<384xf32>
    %v7598 = stablehlo.sqrt %v7597 : tensor<384xf32>
    %v7599 = stablehlo.divide %v7583, %v7598 : tensor<384xf32>
    %v7600 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7601 = stablehlo.multiply %v7600, %b8egm : tensor<384xf32>
    %v7602 = stablehlo.add %v7601, %v7599 : tensor<384xf32>
    %v7603 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7604 = stablehlo.multiply %v7603, %v7602 : tensor<384xf32>
    %v7605 = stablehlo.subtract %b8eg, %v7604 : tensor<384xf32>
    %v7606 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7607 = stablehlo.multiply %v7606, %b8ebt : tensor<384xf32>
    %v7608 = stablehlo.add %v7607, %v3659 : tensor<384xf32>
    %v7609 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7610 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7611 = stablehlo.multiply %v7609, %b8ebtv : tensor<384xf32>
    %v7612 = stablehlo.multiply %v7608, %v7608 : tensor<384xf32>
    %v7613 = stablehlo.multiply %v7610, %v7612 : tensor<384xf32>
    %v7614 = stablehlo.add %v7611, %v7613 : tensor<384xf32>
    %v7615 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7616 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7617 = stablehlo.multiply %v7615, %b8ebtv : tensor<384xf32>
    %v7618 = stablehlo.multiply %v7608, %v7608 : tensor<384xf32>
    %v7619 = stablehlo.multiply %v7616, %v7618 : tensor<384xf32>
    %v7620 = stablehlo.add %v7617, %v7619 : tensor<384xf32>
    %v7621 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7622 = stablehlo.add %v7620, %v7621 : tensor<384xf32>
    %v7623 = stablehlo.sqrt %v7622 : tensor<384xf32>
    %v7624 = stablehlo.divide %v7608, %v7623 : tensor<384xf32>
    %v7625 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7626 = stablehlo.multiply %v7625, %b8ebtm : tensor<384xf32>
    %v7627 = stablehlo.add %v7626, %v7624 : tensor<384xf32>
    %v7628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7629 = stablehlo.multiply %v7628, %v7627 : tensor<384xf32>
    %v7630 = stablehlo.subtract %b8ebt, %v7629 : tensor<384xf32>
    %v7631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7632 = stablehlo.multiply %v7631, %b8dW : tensor<384x1x3x3xf32>
    %v7633 = stablehlo.add %v7632, %v3665 : tensor<384x1x3x3xf32>
    %v7634 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7635 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7636 = stablehlo.multiply %v7634, %b8dWv : tensor<384x1x3x3xf32>
    %v7637 = stablehlo.multiply %v7633, %v7633 : tensor<384x1x3x3xf32>
    %v7638 = stablehlo.multiply %v7635, %v7637 : tensor<384x1x3x3xf32>
    %v7639 = stablehlo.add %v7636, %v7638 : tensor<384x1x3x3xf32>
    %v7640 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7641 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7642 = stablehlo.multiply %v7640, %b8dWv : tensor<384x1x3x3xf32>
    %v7643 = stablehlo.multiply %v7633, %v7633 : tensor<384x1x3x3xf32>
    %v7644 = stablehlo.multiply %v7641, %v7643 : tensor<384x1x3x3xf32>
    %v7645 = stablehlo.add %v7642, %v7644 : tensor<384x1x3x3xf32>
    %v7646 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7647 = stablehlo.add %v7645, %v7646 : tensor<384x1x3x3xf32>
    %v7648 = stablehlo.sqrt %v7647 : tensor<384x1x3x3xf32>
    %v7649 = stablehlo.divide %v7633, %v7648 : tensor<384x1x3x3xf32>
    %v7650 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7651 = stablehlo.multiply %v7650, %b8dWm : tensor<384x1x3x3xf32>
    %v7652 = stablehlo.add %v7651, %v7649 : tensor<384x1x3x3xf32>
    %v7653 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7654 = stablehlo.multiply %v7653, %v7652 : tensor<384x1x3x3xf32>
    %v7655 = stablehlo.subtract %b8dW, %v7654 : tensor<384x1x3x3xf32>
    %v7656 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7657 = stablehlo.multiply %v7656, %b8dg : tensor<384xf32>
    %v7658 = stablehlo.add %v7657, %v3683 : tensor<384xf32>
    %v7659 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7660 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7661 = stablehlo.multiply %v7659, %b8dgv : tensor<384xf32>
    %v7662 = stablehlo.multiply %v7658, %v7658 : tensor<384xf32>
    %v7663 = stablehlo.multiply %v7660, %v7662 : tensor<384xf32>
    %v7664 = stablehlo.add %v7661, %v7663 : tensor<384xf32>
    %v7665 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7666 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7667 = stablehlo.multiply %v7665, %b8dgv : tensor<384xf32>
    %v7668 = stablehlo.multiply %v7658, %v7658 : tensor<384xf32>
    %v7669 = stablehlo.multiply %v7666, %v7668 : tensor<384xf32>
    %v7670 = stablehlo.add %v7667, %v7669 : tensor<384xf32>
    %v7671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7672 = stablehlo.add %v7670, %v7671 : tensor<384xf32>
    %v7673 = stablehlo.sqrt %v7672 : tensor<384xf32>
    %v7674 = stablehlo.divide %v7658, %v7673 : tensor<384xf32>
    %v7675 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7676 = stablehlo.multiply %v7675, %b8dgm : tensor<384xf32>
    %v7677 = stablehlo.add %v7676, %v7674 : tensor<384xf32>
    %v7678 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7679 = stablehlo.multiply %v7678, %v7677 : tensor<384xf32>
    %v7680 = stablehlo.subtract %b8dg, %v7679 : tensor<384xf32>
    %v7681 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7682 = stablehlo.multiply %v7681, %b8dbt : tensor<384xf32>
    %v7683 = stablehlo.add %v7682, %v3686 : tensor<384xf32>
    %v7684 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7685 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7686 = stablehlo.multiply %v7684, %b8dbtv : tensor<384xf32>
    %v7687 = stablehlo.multiply %v7683, %v7683 : tensor<384xf32>
    %v7688 = stablehlo.multiply %v7685, %v7687 : tensor<384xf32>
    %v7689 = stablehlo.add %v7686, %v7688 : tensor<384xf32>
    %v7690 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7691 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7692 = stablehlo.multiply %v7690, %b8dbtv : tensor<384xf32>
    %v7693 = stablehlo.multiply %v7683, %v7683 : tensor<384xf32>
    %v7694 = stablehlo.multiply %v7691, %v7693 : tensor<384xf32>
    %v7695 = stablehlo.add %v7692, %v7694 : tensor<384xf32>
    %v7696 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7697 = stablehlo.add %v7695, %v7696 : tensor<384xf32>
    %v7698 = stablehlo.sqrt %v7697 : tensor<384xf32>
    %v7699 = stablehlo.divide %v7683, %v7698 : tensor<384xf32>
    %v7700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7701 = stablehlo.multiply %v7700, %b8dbtm : tensor<384xf32>
    %v7702 = stablehlo.add %v7701, %v7699 : tensor<384xf32>
    %v7703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7704 = stablehlo.multiply %v7703, %v7702 : tensor<384xf32>
    %v7705 = stablehlo.subtract %b8dbt, %v7704 : tensor<384xf32>
    %v7706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7707 = stablehlo.multiply %v7706, %b8pW : tensor<64x384x1x1xf32>
    %v7708 = stablehlo.add %v7707, %v3692 : tensor<64x384x1x1xf32>
    %v7709 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7710 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7711 = stablehlo.multiply %v7709, %b8pWv : tensor<64x384x1x1xf32>
    %v7712 = stablehlo.multiply %v7708, %v7708 : tensor<64x384x1x1xf32>
    %v7713 = stablehlo.multiply %v7710, %v7712 : tensor<64x384x1x1xf32>
    %v7714 = stablehlo.add %v7711, %v7713 : tensor<64x384x1x1xf32>
    %v7715 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7716 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7717 = stablehlo.multiply %v7715, %b8pWv : tensor<64x384x1x1xf32>
    %v7718 = stablehlo.multiply %v7708, %v7708 : tensor<64x384x1x1xf32>
    %v7719 = stablehlo.multiply %v7716, %v7718 : tensor<64x384x1x1xf32>
    %v7720 = stablehlo.add %v7717, %v7719 : tensor<64x384x1x1xf32>
    %v7721 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7722 = stablehlo.add %v7720, %v7721 : tensor<64x384x1x1xf32>
    %v7723 = stablehlo.sqrt %v7722 : tensor<64x384x1x1xf32>
    %v7724 = stablehlo.divide %v7708, %v7723 : tensor<64x384x1x1xf32>
    %v7725 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7726 = stablehlo.multiply %v7725, %b8pWm : tensor<64x384x1x1xf32>
    %v7727 = stablehlo.add %v7726, %v7724 : tensor<64x384x1x1xf32>
    %v7728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7729 = stablehlo.multiply %v7728, %v7727 : tensor<64x384x1x1xf32>
    %v7730 = stablehlo.subtract %b8pW, %v7729 : tensor<64x384x1x1xf32>
    %v7731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7732 = stablehlo.multiply %v7731, %b8pg : tensor<64xf32>
    %v7733 = stablehlo.add %v7732, %v3710 : tensor<64xf32>
    %v7734 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7735 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7736 = stablehlo.multiply %v7734, %b8pgv : tensor<64xf32>
    %v7737 = stablehlo.multiply %v7733, %v7733 : tensor<64xf32>
    %v7738 = stablehlo.multiply %v7735, %v7737 : tensor<64xf32>
    %v7739 = stablehlo.add %v7736, %v7738 : tensor<64xf32>
    %v7740 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7741 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7742 = stablehlo.multiply %v7740, %b8pgv : tensor<64xf32>
    %v7743 = stablehlo.multiply %v7733, %v7733 : tensor<64xf32>
    %v7744 = stablehlo.multiply %v7741, %v7743 : tensor<64xf32>
    %v7745 = stablehlo.add %v7742, %v7744 : tensor<64xf32>
    %v7746 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7747 = stablehlo.add %v7745, %v7746 : tensor<64xf32>
    %v7748 = stablehlo.sqrt %v7747 : tensor<64xf32>
    %v7749 = stablehlo.divide %v7733, %v7748 : tensor<64xf32>
    %v7750 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7751 = stablehlo.multiply %v7750, %b8pgm : tensor<64xf32>
    %v7752 = stablehlo.add %v7751, %v7749 : tensor<64xf32>
    %v7753 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7754 = stablehlo.multiply %v7753, %v7752 : tensor<64xf32>
    %v7755 = stablehlo.subtract %b8pg, %v7754 : tensor<64xf32>
    %v7756 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7757 = stablehlo.multiply %v7756, %b8pbt : tensor<64xf32>
    %v7758 = stablehlo.add %v7757, %v3713 : tensor<64xf32>
    %v7759 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7760 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7761 = stablehlo.multiply %v7759, %b8pbtv : tensor<64xf32>
    %v7762 = stablehlo.multiply %v7758, %v7758 : tensor<64xf32>
    %v7763 = stablehlo.multiply %v7760, %v7762 : tensor<64xf32>
    %v7764 = stablehlo.add %v7761, %v7763 : tensor<64xf32>
    %v7765 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7766 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7767 = stablehlo.multiply %v7765, %b8pbtv : tensor<64xf32>
    %v7768 = stablehlo.multiply %v7758, %v7758 : tensor<64xf32>
    %v7769 = stablehlo.multiply %v7766, %v7768 : tensor<64xf32>
    %v7770 = stablehlo.add %v7767, %v7769 : tensor<64xf32>
    %v7771 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7772 = stablehlo.add %v7770, %v7771 : tensor<64xf32>
    %v7773 = stablehlo.sqrt %v7772 : tensor<64xf32>
    %v7774 = stablehlo.divide %v7758, %v7773 : tensor<64xf32>
    %v7775 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7776 = stablehlo.multiply %v7775, %b8pbtm : tensor<64xf32>
    %v7777 = stablehlo.add %v7776, %v7774 : tensor<64xf32>
    %v7778 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7779 = stablehlo.multiply %v7778, %v7777 : tensor<64xf32>
    %v7780 = stablehlo.subtract %b8pbt, %v7779 : tensor<64xf32>
    %v7781 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7782 = stablehlo.multiply %v7781, %b9eW : tensor<384x64x1x1xf32>
    %v7783 = stablehlo.add %v7782, %v3431 : tensor<384x64x1x1xf32>
    %v7784 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7785 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7786 = stablehlo.multiply %v7784, %b9eWv : tensor<384x64x1x1xf32>
    %v7787 = stablehlo.multiply %v7783, %v7783 : tensor<384x64x1x1xf32>
    %v7788 = stablehlo.multiply %v7785, %v7787 : tensor<384x64x1x1xf32>
    %v7789 = stablehlo.add %v7786, %v7788 : tensor<384x64x1x1xf32>
    %v7790 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7791 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7792 = stablehlo.multiply %v7790, %b9eWv : tensor<384x64x1x1xf32>
    %v7793 = stablehlo.multiply %v7783, %v7783 : tensor<384x64x1x1xf32>
    %v7794 = stablehlo.multiply %v7791, %v7793 : tensor<384x64x1x1xf32>
    %v7795 = stablehlo.add %v7792, %v7794 : tensor<384x64x1x1xf32>
    %v7796 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7797 = stablehlo.add %v7795, %v7796 : tensor<384x64x1x1xf32>
    %v7798 = stablehlo.sqrt %v7797 : tensor<384x64x1x1xf32>
    %v7799 = stablehlo.divide %v7783, %v7798 : tensor<384x64x1x1xf32>
    %v7800 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7801 = stablehlo.multiply %v7800, %b9eWm : tensor<384x64x1x1xf32>
    %v7802 = stablehlo.add %v7801, %v7799 : tensor<384x64x1x1xf32>
    %v7803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7804 = stablehlo.multiply %v7803, %v7802 : tensor<384x64x1x1xf32>
    %v7805 = stablehlo.subtract %b9eW, %v7804 : tensor<384x64x1x1xf32>
    %v7806 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7807 = stablehlo.multiply %v7806, %b9eg : tensor<384xf32>
    %v7808 = stablehlo.add %v7807, %v3449 : tensor<384xf32>
    %v7809 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7810 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7811 = stablehlo.multiply %v7809, %b9egv : tensor<384xf32>
    %v7812 = stablehlo.multiply %v7808, %v7808 : tensor<384xf32>
    %v7813 = stablehlo.multiply %v7810, %v7812 : tensor<384xf32>
    %v7814 = stablehlo.add %v7811, %v7813 : tensor<384xf32>
    %v7815 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7816 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7817 = stablehlo.multiply %v7815, %b9egv : tensor<384xf32>
    %v7818 = stablehlo.multiply %v7808, %v7808 : tensor<384xf32>
    %v7819 = stablehlo.multiply %v7816, %v7818 : tensor<384xf32>
    %v7820 = stablehlo.add %v7817, %v7819 : tensor<384xf32>
    %v7821 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7822 = stablehlo.add %v7820, %v7821 : tensor<384xf32>
    %v7823 = stablehlo.sqrt %v7822 : tensor<384xf32>
    %v7824 = stablehlo.divide %v7808, %v7823 : tensor<384xf32>
    %v7825 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7826 = stablehlo.multiply %v7825, %b9egm : tensor<384xf32>
    %v7827 = stablehlo.add %v7826, %v7824 : tensor<384xf32>
    %v7828 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7829 = stablehlo.multiply %v7828, %v7827 : tensor<384xf32>
    %v7830 = stablehlo.subtract %b9eg, %v7829 : tensor<384xf32>
    %v7831 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7832 = stablehlo.multiply %v7831, %b9ebt : tensor<384xf32>
    %v7833 = stablehlo.add %v7832, %v3452 : tensor<384xf32>
    %v7834 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7835 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7836 = stablehlo.multiply %v7834, %b9ebtv : tensor<384xf32>
    %v7837 = stablehlo.multiply %v7833, %v7833 : tensor<384xf32>
    %v7838 = stablehlo.multiply %v7835, %v7837 : tensor<384xf32>
    %v7839 = stablehlo.add %v7836, %v7838 : tensor<384xf32>
    %v7840 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7841 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7842 = stablehlo.multiply %v7840, %b9ebtv : tensor<384xf32>
    %v7843 = stablehlo.multiply %v7833, %v7833 : tensor<384xf32>
    %v7844 = stablehlo.multiply %v7841, %v7843 : tensor<384xf32>
    %v7845 = stablehlo.add %v7842, %v7844 : tensor<384xf32>
    %v7846 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7847 = stablehlo.add %v7845, %v7846 : tensor<384xf32>
    %v7848 = stablehlo.sqrt %v7847 : tensor<384xf32>
    %v7849 = stablehlo.divide %v7833, %v7848 : tensor<384xf32>
    %v7850 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7851 = stablehlo.multiply %v7850, %b9ebtm : tensor<384xf32>
    %v7852 = stablehlo.add %v7851, %v7849 : tensor<384xf32>
    %v7853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7854 = stablehlo.multiply %v7853, %v7852 : tensor<384xf32>
    %v7855 = stablehlo.subtract %b9ebt, %v7854 : tensor<384xf32>
    %v7856 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7857 = stablehlo.multiply %v7856, %b9dW : tensor<384x1x3x3xf32>
    %v7858 = stablehlo.add %v7857, %v3458 : tensor<384x1x3x3xf32>
    %v7859 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7860 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7861 = stablehlo.multiply %v7859, %b9dWv : tensor<384x1x3x3xf32>
    %v7862 = stablehlo.multiply %v7858, %v7858 : tensor<384x1x3x3xf32>
    %v7863 = stablehlo.multiply %v7860, %v7862 : tensor<384x1x3x3xf32>
    %v7864 = stablehlo.add %v7861, %v7863 : tensor<384x1x3x3xf32>
    %v7865 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7866 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7867 = stablehlo.multiply %v7865, %b9dWv : tensor<384x1x3x3xf32>
    %v7868 = stablehlo.multiply %v7858, %v7858 : tensor<384x1x3x3xf32>
    %v7869 = stablehlo.multiply %v7866, %v7868 : tensor<384x1x3x3xf32>
    %v7870 = stablehlo.add %v7867, %v7869 : tensor<384x1x3x3xf32>
    %v7871 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7872 = stablehlo.add %v7870, %v7871 : tensor<384x1x3x3xf32>
    %v7873 = stablehlo.sqrt %v7872 : tensor<384x1x3x3xf32>
    %v7874 = stablehlo.divide %v7858, %v7873 : tensor<384x1x3x3xf32>
    %v7875 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7876 = stablehlo.multiply %v7875, %b9dWm : tensor<384x1x3x3xf32>
    %v7877 = stablehlo.add %v7876, %v7874 : tensor<384x1x3x3xf32>
    %v7878 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7879 = stablehlo.multiply %v7878, %v7877 : tensor<384x1x3x3xf32>
    %v7880 = stablehlo.subtract %b9dW, %v7879 : tensor<384x1x3x3xf32>
    %v7881 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7882 = stablehlo.multiply %v7881, %b9dg : tensor<384xf32>
    %v7883 = stablehlo.add %v7882, %v3476 : tensor<384xf32>
    %v7884 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7885 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7886 = stablehlo.multiply %v7884, %b9dgv : tensor<384xf32>
    %v7887 = stablehlo.multiply %v7883, %v7883 : tensor<384xf32>
    %v7888 = stablehlo.multiply %v7885, %v7887 : tensor<384xf32>
    %v7889 = stablehlo.add %v7886, %v7888 : tensor<384xf32>
    %v7890 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7891 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7892 = stablehlo.multiply %v7890, %b9dgv : tensor<384xf32>
    %v7893 = stablehlo.multiply %v7883, %v7883 : tensor<384xf32>
    %v7894 = stablehlo.multiply %v7891, %v7893 : tensor<384xf32>
    %v7895 = stablehlo.add %v7892, %v7894 : tensor<384xf32>
    %v7896 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7897 = stablehlo.add %v7895, %v7896 : tensor<384xf32>
    %v7898 = stablehlo.sqrt %v7897 : tensor<384xf32>
    %v7899 = stablehlo.divide %v7883, %v7898 : tensor<384xf32>
    %v7900 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7901 = stablehlo.multiply %v7900, %b9dgm : tensor<384xf32>
    %v7902 = stablehlo.add %v7901, %v7899 : tensor<384xf32>
    %v7903 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7904 = stablehlo.multiply %v7903, %v7902 : tensor<384xf32>
    %v7905 = stablehlo.subtract %b9dg, %v7904 : tensor<384xf32>
    %v7906 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7907 = stablehlo.multiply %v7906, %b9dbt : tensor<384xf32>
    %v7908 = stablehlo.add %v7907, %v3479 : tensor<384xf32>
    %v7909 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7910 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7911 = stablehlo.multiply %v7909, %b9dbtv : tensor<384xf32>
    %v7912 = stablehlo.multiply %v7908, %v7908 : tensor<384xf32>
    %v7913 = stablehlo.multiply %v7910, %v7912 : tensor<384xf32>
    %v7914 = stablehlo.add %v7911, %v7913 : tensor<384xf32>
    %v7915 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7916 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7917 = stablehlo.multiply %v7915, %b9dbtv : tensor<384xf32>
    %v7918 = stablehlo.multiply %v7908, %v7908 : tensor<384xf32>
    %v7919 = stablehlo.multiply %v7916, %v7918 : tensor<384xf32>
    %v7920 = stablehlo.add %v7917, %v7919 : tensor<384xf32>
    %v7921 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7922 = stablehlo.add %v7920, %v7921 : tensor<384xf32>
    %v7923 = stablehlo.sqrt %v7922 : tensor<384xf32>
    %v7924 = stablehlo.divide %v7908, %v7923 : tensor<384xf32>
    %v7925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7926 = stablehlo.multiply %v7925, %b9dbtm : tensor<384xf32>
    %v7927 = stablehlo.add %v7926, %v7924 : tensor<384xf32>
    %v7928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7929 = stablehlo.multiply %v7928, %v7927 : tensor<384xf32>
    %v7930 = stablehlo.subtract %b9dbt, %v7929 : tensor<384xf32>
    %v7931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7932 = stablehlo.multiply %v7931, %b9pW : tensor<64x384x1x1xf32>
    %v7933 = stablehlo.add %v7932, %v3485 : tensor<64x384x1x1xf32>
    %v7934 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7935 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7936 = stablehlo.multiply %v7934, %b9pWv : tensor<64x384x1x1xf32>
    %v7937 = stablehlo.multiply %v7933, %v7933 : tensor<64x384x1x1xf32>
    %v7938 = stablehlo.multiply %v7935, %v7937 : tensor<64x384x1x1xf32>
    %v7939 = stablehlo.add %v7936, %v7938 : tensor<64x384x1x1xf32>
    %v7940 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7941 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7942 = stablehlo.multiply %v7940, %b9pWv : tensor<64x384x1x1xf32>
    %v7943 = stablehlo.multiply %v7933, %v7933 : tensor<64x384x1x1xf32>
    %v7944 = stablehlo.multiply %v7941, %v7943 : tensor<64x384x1x1xf32>
    %v7945 = stablehlo.add %v7942, %v7944 : tensor<64x384x1x1xf32>
    %v7946 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7947 = stablehlo.add %v7945, %v7946 : tensor<64x384x1x1xf32>
    %v7948 = stablehlo.sqrt %v7947 : tensor<64x384x1x1xf32>
    %v7949 = stablehlo.divide %v7933, %v7948 : tensor<64x384x1x1xf32>
    %v7950 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7951 = stablehlo.multiply %v7950, %b9pWm : tensor<64x384x1x1xf32>
    %v7952 = stablehlo.add %v7951, %v7949 : tensor<64x384x1x1xf32>
    %v7953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7954 = stablehlo.multiply %v7953, %v7952 : tensor<64x384x1x1xf32>
    %v7955 = stablehlo.subtract %b9pW, %v7954 : tensor<64x384x1x1xf32>
    %v7956 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7957 = stablehlo.multiply %v7956, %b9pg : tensor<64xf32>
    %v7958 = stablehlo.add %v7957, %v3503 : tensor<64xf32>
    %v7959 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7960 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7961 = stablehlo.multiply %v7959, %b9pgv : tensor<64xf32>
    %v7962 = stablehlo.multiply %v7958, %v7958 : tensor<64xf32>
    %v7963 = stablehlo.multiply %v7960, %v7962 : tensor<64xf32>
    %v7964 = stablehlo.add %v7961, %v7963 : tensor<64xf32>
    %v7965 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7966 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7967 = stablehlo.multiply %v7965, %b9pgv : tensor<64xf32>
    %v7968 = stablehlo.multiply %v7958, %v7958 : tensor<64xf32>
    %v7969 = stablehlo.multiply %v7966, %v7968 : tensor<64xf32>
    %v7970 = stablehlo.add %v7967, %v7969 : tensor<64xf32>
    %v7971 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7972 = stablehlo.add %v7970, %v7971 : tensor<64xf32>
    %v7973 = stablehlo.sqrt %v7972 : tensor<64xf32>
    %v7974 = stablehlo.divide %v7958, %v7973 : tensor<64xf32>
    %v7975 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7976 = stablehlo.multiply %v7975, %b9pgm : tensor<64xf32>
    %v7977 = stablehlo.add %v7976, %v7974 : tensor<64xf32>
    %v7978 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7979 = stablehlo.multiply %v7978, %v7977 : tensor<64xf32>
    %v7980 = stablehlo.subtract %b9pg, %v7979 : tensor<64xf32>
    %v7981 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7982 = stablehlo.multiply %v7981, %b9pbt : tensor<64xf32>
    %v7983 = stablehlo.add %v7982, %v3506 : tensor<64xf32>
    %v7984 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7985 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7986 = stablehlo.multiply %v7984, %b9pbtv : tensor<64xf32>
    %v7987 = stablehlo.multiply %v7983, %v7983 : tensor<64xf32>
    %v7988 = stablehlo.multiply %v7985, %v7987 : tensor<64xf32>
    %v7989 = stablehlo.add %v7986, %v7988 : tensor<64xf32>
    %v7990 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7991 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7992 = stablehlo.multiply %v7990, %b9pbtv : tensor<64xf32>
    %v7993 = stablehlo.multiply %v7983, %v7983 : tensor<64xf32>
    %v7994 = stablehlo.multiply %v7991, %v7993 : tensor<64xf32>
    %v7995 = stablehlo.add %v7992, %v7994 : tensor<64xf32>
    %v7996 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7997 = stablehlo.add %v7995, %v7996 : tensor<64xf32>
    %v7998 = stablehlo.sqrt %v7997 : tensor<64xf32>
    %v7999 = stablehlo.divide %v7983, %v7998 : tensor<64xf32>
    %v8000 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8001 = stablehlo.multiply %v8000, %b9pbtm : tensor<64xf32>
    %v8002 = stablehlo.add %v8001, %v7999 : tensor<64xf32>
    %v8003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8004 = stablehlo.multiply %v8003, %v8002 : tensor<64xf32>
    %v8005 = stablehlo.subtract %b9pbt, %v8004 : tensor<64xf32>
    %v8006 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8007 = stablehlo.multiply %v8006, %b10eW : tensor<384x64x1x1xf32>
    %v8008 = stablehlo.add %v8007, %v3224 : tensor<384x64x1x1xf32>
    %v8009 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8010 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8011 = stablehlo.multiply %v8009, %b10eWv : tensor<384x64x1x1xf32>
    %v8012 = stablehlo.multiply %v8008, %v8008 : tensor<384x64x1x1xf32>
    %v8013 = stablehlo.multiply %v8010, %v8012 : tensor<384x64x1x1xf32>
    %v8014 = stablehlo.add %v8011, %v8013 : tensor<384x64x1x1xf32>
    %v8015 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8016 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8017 = stablehlo.multiply %v8015, %b10eWv : tensor<384x64x1x1xf32>
    %v8018 = stablehlo.multiply %v8008, %v8008 : tensor<384x64x1x1xf32>
    %v8019 = stablehlo.multiply %v8016, %v8018 : tensor<384x64x1x1xf32>
    %v8020 = stablehlo.add %v8017, %v8019 : tensor<384x64x1x1xf32>
    %v8021 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8022 = stablehlo.add %v8020, %v8021 : tensor<384x64x1x1xf32>
    %v8023 = stablehlo.sqrt %v8022 : tensor<384x64x1x1xf32>
    %v8024 = stablehlo.divide %v8008, %v8023 : tensor<384x64x1x1xf32>
    %v8025 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8026 = stablehlo.multiply %v8025, %b10eWm : tensor<384x64x1x1xf32>
    %v8027 = stablehlo.add %v8026, %v8024 : tensor<384x64x1x1xf32>
    %v8028 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8029 = stablehlo.multiply %v8028, %v8027 : tensor<384x64x1x1xf32>
    %v8030 = stablehlo.subtract %b10eW, %v8029 : tensor<384x64x1x1xf32>
    %v8031 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8032 = stablehlo.multiply %v8031, %b10eg : tensor<384xf32>
    %v8033 = stablehlo.add %v8032, %v3242 : tensor<384xf32>
    %v8034 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8035 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8036 = stablehlo.multiply %v8034, %b10egv : tensor<384xf32>
    %v8037 = stablehlo.multiply %v8033, %v8033 : tensor<384xf32>
    %v8038 = stablehlo.multiply %v8035, %v8037 : tensor<384xf32>
    %v8039 = stablehlo.add %v8036, %v8038 : tensor<384xf32>
    %v8040 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8041 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8042 = stablehlo.multiply %v8040, %b10egv : tensor<384xf32>
    %v8043 = stablehlo.multiply %v8033, %v8033 : tensor<384xf32>
    %v8044 = stablehlo.multiply %v8041, %v8043 : tensor<384xf32>
    %v8045 = stablehlo.add %v8042, %v8044 : tensor<384xf32>
    %v8046 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8047 = stablehlo.add %v8045, %v8046 : tensor<384xf32>
    %v8048 = stablehlo.sqrt %v8047 : tensor<384xf32>
    %v8049 = stablehlo.divide %v8033, %v8048 : tensor<384xf32>
    %v8050 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8051 = stablehlo.multiply %v8050, %b10egm : tensor<384xf32>
    %v8052 = stablehlo.add %v8051, %v8049 : tensor<384xf32>
    %v8053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8054 = stablehlo.multiply %v8053, %v8052 : tensor<384xf32>
    %v8055 = stablehlo.subtract %b10eg, %v8054 : tensor<384xf32>
    %v8056 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8057 = stablehlo.multiply %v8056, %b10ebt : tensor<384xf32>
    %v8058 = stablehlo.add %v8057, %v3245 : tensor<384xf32>
    %v8059 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8060 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8061 = stablehlo.multiply %v8059, %b10ebtv : tensor<384xf32>
    %v8062 = stablehlo.multiply %v8058, %v8058 : tensor<384xf32>
    %v8063 = stablehlo.multiply %v8060, %v8062 : tensor<384xf32>
    %v8064 = stablehlo.add %v8061, %v8063 : tensor<384xf32>
    %v8065 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8066 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8067 = stablehlo.multiply %v8065, %b10ebtv : tensor<384xf32>
    %v8068 = stablehlo.multiply %v8058, %v8058 : tensor<384xf32>
    %v8069 = stablehlo.multiply %v8066, %v8068 : tensor<384xf32>
    %v8070 = stablehlo.add %v8067, %v8069 : tensor<384xf32>
    %v8071 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8072 = stablehlo.add %v8070, %v8071 : tensor<384xf32>
    %v8073 = stablehlo.sqrt %v8072 : tensor<384xf32>
    %v8074 = stablehlo.divide %v8058, %v8073 : tensor<384xf32>
    %v8075 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8076 = stablehlo.multiply %v8075, %b10ebtm : tensor<384xf32>
    %v8077 = stablehlo.add %v8076, %v8074 : tensor<384xf32>
    %v8078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8079 = stablehlo.multiply %v8078, %v8077 : tensor<384xf32>
    %v8080 = stablehlo.subtract %b10ebt, %v8079 : tensor<384xf32>
    %v8081 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8082 = stablehlo.multiply %v8081, %b10dW : tensor<384x1x3x3xf32>
    %v8083 = stablehlo.add %v8082, %v3251 : tensor<384x1x3x3xf32>
    %v8084 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8085 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8086 = stablehlo.multiply %v8084, %b10dWv : tensor<384x1x3x3xf32>
    %v8087 = stablehlo.multiply %v8083, %v8083 : tensor<384x1x3x3xf32>
    %v8088 = stablehlo.multiply %v8085, %v8087 : tensor<384x1x3x3xf32>
    %v8089 = stablehlo.add %v8086, %v8088 : tensor<384x1x3x3xf32>
    %v8090 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8091 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8092 = stablehlo.multiply %v8090, %b10dWv : tensor<384x1x3x3xf32>
    %v8093 = stablehlo.multiply %v8083, %v8083 : tensor<384x1x3x3xf32>
    %v8094 = stablehlo.multiply %v8091, %v8093 : tensor<384x1x3x3xf32>
    %v8095 = stablehlo.add %v8092, %v8094 : tensor<384x1x3x3xf32>
    %v8096 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8097 = stablehlo.add %v8095, %v8096 : tensor<384x1x3x3xf32>
    %v8098 = stablehlo.sqrt %v8097 : tensor<384x1x3x3xf32>
    %v8099 = stablehlo.divide %v8083, %v8098 : tensor<384x1x3x3xf32>
    %v8100 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8101 = stablehlo.multiply %v8100, %b10dWm : tensor<384x1x3x3xf32>
    %v8102 = stablehlo.add %v8101, %v8099 : tensor<384x1x3x3xf32>
    %v8103 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8104 = stablehlo.multiply %v8103, %v8102 : tensor<384x1x3x3xf32>
    %v8105 = stablehlo.subtract %b10dW, %v8104 : tensor<384x1x3x3xf32>
    %v8106 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8107 = stablehlo.multiply %v8106, %b10dg : tensor<384xf32>
    %v8108 = stablehlo.add %v8107, %v3269 : tensor<384xf32>
    %v8109 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8110 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8111 = stablehlo.multiply %v8109, %b10dgv : tensor<384xf32>
    %v8112 = stablehlo.multiply %v8108, %v8108 : tensor<384xf32>
    %v8113 = stablehlo.multiply %v8110, %v8112 : tensor<384xf32>
    %v8114 = stablehlo.add %v8111, %v8113 : tensor<384xf32>
    %v8115 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8116 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8117 = stablehlo.multiply %v8115, %b10dgv : tensor<384xf32>
    %v8118 = stablehlo.multiply %v8108, %v8108 : tensor<384xf32>
    %v8119 = stablehlo.multiply %v8116, %v8118 : tensor<384xf32>
    %v8120 = stablehlo.add %v8117, %v8119 : tensor<384xf32>
    %v8121 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8122 = stablehlo.add %v8120, %v8121 : tensor<384xf32>
    %v8123 = stablehlo.sqrt %v8122 : tensor<384xf32>
    %v8124 = stablehlo.divide %v8108, %v8123 : tensor<384xf32>
    %v8125 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8126 = stablehlo.multiply %v8125, %b10dgm : tensor<384xf32>
    %v8127 = stablehlo.add %v8126, %v8124 : tensor<384xf32>
    %v8128 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8129 = stablehlo.multiply %v8128, %v8127 : tensor<384xf32>
    %v8130 = stablehlo.subtract %b10dg, %v8129 : tensor<384xf32>
    %v8131 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8132 = stablehlo.multiply %v8131, %b10dbt : tensor<384xf32>
    %v8133 = stablehlo.add %v8132, %v3272 : tensor<384xf32>
    %v8134 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8135 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8136 = stablehlo.multiply %v8134, %b10dbtv : tensor<384xf32>
    %v8137 = stablehlo.multiply %v8133, %v8133 : tensor<384xf32>
    %v8138 = stablehlo.multiply %v8135, %v8137 : tensor<384xf32>
    %v8139 = stablehlo.add %v8136, %v8138 : tensor<384xf32>
    %v8140 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8141 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8142 = stablehlo.multiply %v8140, %b10dbtv : tensor<384xf32>
    %v8143 = stablehlo.multiply %v8133, %v8133 : tensor<384xf32>
    %v8144 = stablehlo.multiply %v8141, %v8143 : tensor<384xf32>
    %v8145 = stablehlo.add %v8142, %v8144 : tensor<384xf32>
    %v8146 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8147 = stablehlo.add %v8145, %v8146 : tensor<384xf32>
    %v8148 = stablehlo.sqrt %v8147 : tensor<384xf32>
    %v8149 = stablehlo.divide %v8133, %v8148 : tensor<384xf32>
    %v8150 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8151 = stablehlo.multiply %v8150, %b10dbtm : tensor<384xf32>
    %v8152 = stablehlo.add %v8151, %v8149 : tensor<384xf32>
    %v8153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8154 = stablehlo.multiply %v8153, %v8152 : tensor<384xf32>
    %v8155 = stablehlo.subtract %b10dbt, %v8154 : tensor<384xf32>
    %v8156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8157 = stablehlo.multiply %v8156, %b10pW : tensor<64x384x1x1xf32>
    %v8158 = stablehlo.add %v8157, %v3278 : tensor<64x384x1x1xf32>
    %v8159 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8160 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8161 = stablehlo.multiply %v8159, %b10pWv : tensor<64x384x1x1xf32>
    %v8162 = stablehlo.multiply %v8158, %v8158 : tensor<64x384x1x1xf32>
    %v8163 = stablehlo.multiply %v8160, %v8162 : tensor<64x384x1x1xf32>
    %v8164 = stablehlo.add %v8161, %v8163 : tensor<64x384x1x1xf32>
    %v8165 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8166 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8167 = stablehlo.multiply %v8165, %b10pWv : tensor<64x384x1x1xf32>
    %v8168 = stablehlo.multiply %v8158, %v8158 : tensor<64x384x1x1xf32>
    %v8169 = stablehlo.multiply %v8166, %v8168 : tensor<64x384x1x1xf32>
    %v8170 = stablehlo.add %v8167, %v8169 : tensor<64x384x1x1xf32>
    %v8171 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8172 = stablehlo.add %v8170, %v8171 : tensor<64x384x1x1xf32>
    %v8173 = stablehlo.sqrt %v8172 : tensor<64x384x1x1xf32>
    %v8174 = stablehlo.divide %v8158, %v8173 : tensor<64x384x1x1xf32>
    %v8175 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8176 = stablehlo.multiply %v8175, %b10pWm : tensor<64x384x1x1xf32>
    %v8177 = stablehlo.add %v8176, %v8174 : tensor<64x384x1x1xf32>
    %v8178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8179 = stablehlo.multiply %v8178, %v8177 : tensor<64x384x1x1xf32>
    %v8180 = stablehlo.subtract %b10pW, %v8179 : tensor<64x384x1x1xf32>
    %v8181 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8182 = stablehlo.multiply %v8181, %b10pg : tensor<64xf32>
    %v8183 = stablehlo.add %v8182, %v3296 : tensor<64xf32>
    %v8184 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8185 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8186 = stablehlo.multiply %v8184, %b10pgv : tensor<64xf32>
    %v8187 = stablehlo.multiply %v8183, %v8183 : tensor<64xf32>
    %v8188 = stablehlo.multiply %v8185, %v8187 : tensor<64xf32>
    %v8189 = stablehlo.add %v8186, %v8188 : tensor<64xf32>
    %v8190 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8191 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8192 = stablehlo.multiply %v8190, %b10pgv : tensor<64xf32>
    %v8193 = stablehlo.multiply %v8183, %v8183 : tensor<64xf32>
    %v8194 = stablehlo.multiply %v8191, %v8193 : tensor<64xf32>
    %v8195 = stablehlo.add %v8192, %v8194 : tensor<64xf32>
    %v8196 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8197 = stablehlo.add %v8195, %v8196 : tensor<64xf32>
    %v8198 = stablehlo.sqrt %v8197 : tensor<64xf32>
    %v8199 = stablehlo.divide %v8183, %v8198 : tensor<64xf32>
    %v8200 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8201 = stablehlo.multiply %v8200, %b10pgm : tensor<64xf32>
    %v8202 = stablehlo.add %v8201, %v8199 : tensor<64xf32>
    %v8203 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8204 = stablehlo.multiply %v8203, %v8202 : tensor<64xf32>
    %v8205 = stablehlo.subtract %b10pg, %v8204 : tensor<64xf32>
    %v8206 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8207 = stablehlo.multiply %v8206, %b10pbt : tensor<64xf32>
    %v8208 = stablehlo.add %v8207, %v3299 : tensor<64xf32>
    %v8209 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8210 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8211 = stablehlo.multiply %v8209, %b10pbtv : tensor<64xf32>
    %v8212 = stablehlo.multiply %v8208, %v8208 : tensor<64xf32>
    %v8213 = stablehlo.multiply %v8210, %v8212 : tensor<64xf32>
    %v8214 = stablehlo.add %v8211, %v8213 : tensor<64xf32>
    %v8215 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8216 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8217 = stablehlo.multiply %v8215, %b10pbtv : tensor<64xf32>
    %v8218 = stablehlo.multiply %v8208, %v8208 : tensor<64xf32>
    %v8219 = stablehlo.multiply %v8216, %v8218 : tensor<64xf32>
    %v8220 = stablehlo.add %v8217, %v8219 : tensor<64xf32>
    %v8221 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8222 = stablehlo.add %v8220, %v8221 : tensor<64xf32>
    %v8223 = stablehlo.sqrt %v8222 : tensor<64xf32>
    %v8224 = stablehlo.divide %v8208, %v8223 : tensor<64xf32>
    %v8225 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8226 = stablehlo.multiply %v8225, %b10pbtm : tensor<64xf32>
    %v8227 = stablehlo.add %v8226, %v8224 : tensor<64xf32>
    %v8228 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8229 = stablehlo.multiply %v8228, %v8227 : tensor<64xf32>
    %v8230 = stablehlo.subtract %b10pbt, %v8229 : tensor<64xf32>
    %v8231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8232 = stablehlo.multiply %v8231, %b11eW : tensor<384x64x1x1xf32>
    %v8233 = stablehlo.add %v8232, %v3017 : tensor<384x64x1x1xf32>
    %v8234 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8235 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8236 = stablehlo.multiply %v8234, %b11eWv : tensor<384x64x1x1xf32>
    %v8237 = stablehlo.multiply %v8233, %v8233 : tensor<384x64x1x1xf32>
    %v8238 = stablehlo.multiply %v8235, %v8237 : tensor<384x64x1x1xf32>
    %v8239 = stablehlo.add %v8236, %v8238 : tensor<384x64x1x1xf32>
    %v8240 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8241 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8242 = stablehlo.multiply %v8240, %b11eWv : tensor<384x64x1x1xf32>
    %v8243 = stablehlo.multiply %v8233, %v8233 : tensor<384x64x1x1xf32>
    %v8244 = stablehlo.multiply %v8241, %v8243 : tensor<384x64x1x1xf32>
    %v8245 = stablehlo.add %v8242, %v8244 : tensor<384x64x1x1xf32>
    %v8246 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8247 = stablehlo.add %v8245, %v8246 : tensor<384x64x1x1xf32>
    %v8248 = stablehlo.sqrt %v8247 : tensor<384x64x1x1xf32>
    %v8249 = stablehlo.divide %v8233, %v8248 : tensor<384x64x1x1xf32>
    %v8250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8251 = stablehlo.multiply %v8250, %b11eWm : tensor<384x64x1x1xf32>
    %v8252 = stablehlo.add %v8251, %v8249 : tensor<384x64x1x1xf32>
    %v8253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8254 = stablehlo.multiply %v8253, %v8252 : tensor<384x64x1x1xf32>
    %v8255 = stablehlo.subtract %b11eW, %v8254 : tensor<384x64x1x1xf32>
    %v8256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8257 = stablehlo.multiply %v8256, %b11eg : tensor<384xf32>
    %v8258 = stablehlo.add %v8257, %v3035 : tensor<384xf32>
    %v8259 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8260 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8261 = stablehlo.multiply %v8259, %b11egv : tensor<384xf32>
    %v8262 = stablehlo.multiply %v8258, %v8258 : tensor<384xf32>
    %v8263 = stablehlo.multiply %v8260, %v8262 : tensor<384xf32>
    %v8264 = stablehlo.add %v8261, %v8263 : tensor<384xf32>
    %v8265 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8266 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8267 = stablehlo.multiply %v8265, %b11egv : tensor<384xf32>
    %v8268 = stablehlo.multiply %v8258, %v8258 : tensor<384xf32>
    %v8269 = stablehlo.multiply %v8266, %v8268 : tensor<384xf32>
    %v8270 = stablehlo.add %v8267, %v8269 : tensor<384xf32>
    %v8271 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8272 = stablehlo.add %v8270, %v8271 : tensor<384xf32>
    %v8273 = stablehlo.sqrt %v8272 : tensor<384xf32>
    %v8274 = stablehlo.divide %v8258, %v8273 : tensor<384xf32>
    %v8275 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8276 = stablehlo.multiply %v8275, %b11egm : tensor<384xf32>
    %v8277 = stablehlo.add %v8276, %v8274 : tensor<384xf32>
    %v8278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8279 = stablehlo.multiply %v8278, %v8277 : tensor<384xf32>
    %v8280 = stablehlo.subtract %b11eg, %v8279 : tensor<384xf32>
    %v8281 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8282 = stablehlo.multiply %v8281, %b11ebt : tensor<384xf32>
    %v8283 = stablehlo.add %v8282, %v3038 : tensor<384xf32>
    %v8284 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8285 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8286 = stablehlo.multiply %v8284, %b11ebtv : tensor<384xf32>
    %v8287 = stablehlo.multiply %v8283, %v8283 : tensor<384xf32>
    %v8288 = stablehlo.multiply %v8285, %v8287 : tensor<384xf32>
    %v8289 = stablehlo.add %v8286, %v8288 : tensor<384xf32>
    %v8290 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8291 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8292 = stablehlo.multiply %v8290, %b11ebtv : tensor<384xf32>
    %v8293 = stablehlo.multiply %v8283, %v8283 : tensor<384xf32>
    %v8294 = stablehlo.multiply %v8291, %v8293 : tensor<384xf32>
    %v8295 = stablehlo.add %v8292, %v8294 : tensor<384xf32>
    %v8296 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8297 = stablehlo.add %v8295, %v8296 : tensor<384xf32>
    %v8298 = stablehlo.sqrt %v8297 : tensor<384xf32>
    %v8299 = stablehlo.divide %v8283, %v8298 : tensor<384xf32>
    %v8300 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8301 = stablehlo.multiply %v8300, %b11ebtm : tensor<384xf32>
    %v8302 = stablehlo.add %v8301, %v8299 : tensor<384xf32>
    %v8303 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8304 = stablehlo.multiply %v8303, %v8302 : tensor<384xf32>
    %v8305 = stablehlo.subtract %b11ebt, %v8304 : tensor<384xf32>
    %v8306 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8307 = stablehlo.multiply %v8306, %b11dW : tensor<384x1x3x3xf32>
    %v8308 = stablehlo.add %v8307, %v3044 : tensor<384x1x3x3xf32>
    %v8309 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8310 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8311 = stablehlo.multiply %v8309, %b11dWv : tensor<384x1x3x3xf32>
    %v8312 = stablehlo.multiply %v8308, %v8308 : tensor<384x1x3x3xf32>
    %v8313 = stablehlo.multiply %v8310, %v8312 : tensor<384x1x3x3xf32>
    %v8314 = stablehlo.add %v8311, %v8313 : tensor<384x1x3x3xf32>
    %v8315 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8316 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8317 = stablehlo.multiply %v8315, %b11dWv : tensor<384x1x3x3xf32>
    %v8318 = stablehlo.multiply %v8308, %v8308 : tensor<384x1x3x3xf32>
    %v8319 = stablehlo.multiply %v8316, %v8318 : tensor<384x1x3x3xf32>
    %v8320 = stablehlo.add %v8317, %v8319 : tensor<384x1x3x3xf32>
    %v8321 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8322 = stablehlo.add %v8320, %v8321 : tensor<384x1x3x3xf32>
    %v8323 = stablehlo.sqrt %v8322 : tensor<384x1x3x3xf32>
    %v8324 = stablehlo.divide %v8308, %v8323 : tensor<384x1x3x3xf32>
    %v8325 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8326 = stablehlo.multiply %v8325, %b11dWm : tensor<384x1x3x3xf32>
    %v8327 = stablehlo.add %v8326, %v8324 : tensor<384x1x3x3xf32>
    %v8328 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8329 = stablehlo.multiply %v8328, %v8327 : tensor<384x1x3x3xf32>
    %v8330 = stablehlo.subtract %b11dW, %v8329 : tensor<384x1x3x3xf32>
    %v8331 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8332 = stablehlo.multiply %v8331, %b11dg : tensor<384xf32>
    %v8333 = stablehlo.add %v8332, %v3062 : tensor<384xf32>
    %v8334 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8335 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8336 = stablehlo.multiply %v8334, %b11dgv : tensor<384xf32>
    %v8337 = stablehlo.multiply %v8333, %v8333 : tensor<384xf32>
    %v8338 = stablehlo.multiply %v8335, %v8337 : tensor<384xf32>
    %v8339 = stablehlo.add %v8336, %v8338 : tensor<384xf32>
    %v8340 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8341 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8342 = stablehlo.multiply %v8340, %b11dgv : tensor<384xf32>
    %v8343 = stablehlo.multiply %v8333, %v8333 : tensor<384xf32>
    %v8344 = stablehlo.multiply %v8341, %v8343 : tensor<384xf32>
    %v8345 = stablehlo.add %v8342, %v8344 : tensor<384xf32>
    %v8346 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8347 = stablehlo.add %v8345, %v8346 : tensor<384xf32>
    %v8348 = stablehlo.sqrt %v8347 : tensor<384xf32>
    %v8349 = stablehlo.divide %v8333, %v8348 : tensor<384xf32>
    %v8350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8351 = stablehlo.multiply %v8350, %b11dgm : tensor<384xf32>
    %v8352 = stablehlo.add %v8351, %v8349 : tensor<384xf32>
    %v8353 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8354 = stablehlo.multiply %v8353, %v8352 : tensor<384xf32>
    %v8355 = stablehlo.subtract %b11dg, %v8354 : tensor<384xf32>
    %v8356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8357 = stablehlo.multiply %v8356, %b11dbt : tensor<384xf32>
    %v8358 = stablehlo.add %v8357, %v3065 : tensor<384xf32>
    %v8359 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8360 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8361 = stablehlo.multiply %v8359, %b11dbtv : tensor<384xf32>
    %v8362 = stablehlo.multiply %v8358, %v8358 : tensor<384xf32>
    %v8363 = stablehlo.multiply %v8360, %v8362 : tensor<384xf32>
    %v8364 = stablehlo.add %v8361, %v8363 : tensor<384xf32>
    %v8365 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8366 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8367 = stablehlo.multiply %v8365, %b11dbtv : tensor<384xf32>
    %v8368 = stablehlo.multiply %v8358, %v8358 : tensor<384xf32>
    %v8369 = stablehlo.multiply %v8366, %v8368 : tensor<384xf32>
    %v8370 = stablehlo.add %v8367, %v8369 : tensor<384xf32>
    %v8371 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8372 = stablehlo.add %v8370, %v8371 : tensor<384xf32>
    %v8373 = stablehlo.sqrt %v8372 : tensor<384xf32>
    %v8374 = stablehlo.divide %v8358, %v8373 : tensor<384xf32>
    %v8375 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8376 = stablehlo.multiply %v8375, %b11dbtm : tensor<384xf32>
    %v8377 = stablehlo.add %v8376, %v8374 : tensor<384xf32>
    %v8378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8379 = stablehlo.multiply %v8378, %v8377 : tensor<384xf32>
    %v8380 = stablehlo.subtract %b11dbt, %v8379 : tensor<384xf32>
    %v8381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8382 = stablehlo.multiply %v8381, %b11pW : tensor<96x384x1x1xf32>
    %v8383 = stablehlo.add %v8382, %v3071 : tensor<96x384x1x1xf32>
    %v8384 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8385 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8386 = stablehlo.multiply %v8384, %b11pWv : tensor<96x384x1x1xf32>
    %v8387 = stablehlo.multiply %v8383, %v8383 : tensor<96x384x1x1xf32>
    %v8388 = stablehlo.multiply %v8385, %v8387 : tensor<96x384x1x1xf32>
    %v8389 = stablehlo.add %v8386, %v8388 : tensor<96x384x1x1xf32>
    %v8390 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8391 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8392 = stablehlo.multiply %v8390, %b11pWv : tensor<96x384x1x1xf32>
    %v8393 = stablehlo.multiply %v8383, %v8383 : tensor<96x384x1x1xf32>
    %v8394 = stablehlo.multiply %v8391, %v8393 : tensor<96x384x1x1xf32>
    %v8395 = stablehlo.add %v8392, %v8394 : tensor<96x384x1x1xf32>
    %v8396 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8397 = stablehlo.add %v8395, %v8396 : tensor<96x384x1x1xf32>
    %v8398 = stablehlo.sqrt %v8397 : tensor<96x384x1x1xf32>
    %v8399 = stablehlo.divide %v8383, %v8398 : tensor<96x384x1x1xf32>
    %v8400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8401 = stablehlo.multiply %v8400, %b11pWm : tensor<96x384x1x1xf32>
    %v8402 = stablehlo.add %v8401, %v8399 : tensor<96x384x1x1xf32>
    %v8403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8404 = stablehlo.multiply %v8403, %v8402 : tensor<96x384x1x1xf32>
    %v8405 = stablehlo.subtract %b11pW, %v8404 : tensor<96x384x1x1xf32>
    %v8406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8407 = stablehlo.multiply %v8406, %b11pg : tensor<96xf32>
    %v8408 = stablehlo.add %v8407, %v3089 : tensor<96xf32>
    %v8409 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8410 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8411 = stablehlo.multiply %v8409, %b11pgv : tensor<96xf32>
    %v8412 = stablehlo.multiply %v8408, %v8408 : tensor<96xf32>
    %v8413 = stablehlo.multiply %v8410, %v8412 : tensor<96xf32>
    %v8414 = stablehlo.add %v8411, %v8413 : tensor<96xf32>
    %v8415 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8416 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8417 = stablehlo.multiply %v8415, %b11pgv : tensor<96xf32>
    %v8418 = stablehlo.multiply %v8408, %v8408 : tensor<96xf32>
    %v8419 = stablehlo.multiply %v8416, %v8418 : tensor<96xf32>
    %v8420 = stablehlo.add %v8417, %v8419 : tensor<96xf32>
    %v8421 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8422 = stablehlo.add %v8420, %v8421 : tensor<96xf32>
    %v8423 = stablehlo.sqrt %v8422 : tensor<96xf32>
    %v8424 = stablehlo.divide %v8408, %v8423 : tensor<96xf32>
    %v8425 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8426 = stablehlo.multiply %v8425, %b11pgm : tensor<96xf32>
    %v8427 = stablehlo.add %v8426, %v8424 : tensor<96xf32>
    %v8428 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8429 = stablehlo.multiply %v8428, %v8427 : tensor<96xf32>
    %v8430 = stablehlo.subtract %b11pg, %v8429 : tensor<96xf32>
    %v8431 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8432 = stablehlo.multiply %v8431, %b11pbt : tensor<96xf32>
    %v8433 = stablehlo.add %v8432, %v3092 : tensor<96xf32>
    %v8434 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8435 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8436 = stablehlo.multiply %v8434, %b11pbtv : tensor<96xf32>
    %v8437 = stablehlo.multiply %v8433, %v8433 : tensor<96xf32>
    %v8438 = stablehlo.multiply %v8435, %v8437 : tensor<96xf32>
    %v8439 = stablehlo.add %v8436, %v8438 : tensor<96xf32>
    %v8440 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8441 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8442 = stablehlo.multiply %v8440, %b11pbtv : tensor<96xf32>
    %v8443 = stablehlo.multiply %v8433, %v8433 : tensor<96xf32>
    %v8444 = stablehlo.multiply %v8441, %v8443 : tensor<96xf32>
    %v8445 = stablehlo.add %v8442, %v8444 : tensor<96xf32>
    %v8446 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8447 = stablehlo.add %v8445, %v8446 : tensor<96xf32>
    %v8448 = stablehlo.sqrt %v8447 : tensor<96xf32>
    %v8449 = stablehlo.divide %v8433, %v8448 : tensor<96xf32>
    %v8450 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8451 = stablehlo.multiply %v8450, %b11pbtm : tensor<96xf32>
    %v8452 = stablehlo.add %v8451, %v8449 : tensor<96xf32>
    %v8453 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8454 = stablehlo.multiply %v8453, %v8452 : tensor<96xf32>
    %v8455 = stablehlo.subtract %b11pbt, %v8454 : tensor<96xf32>
    %v8456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8457 = stablehlo.multiply %v8456, %b12eW : tensor<576x96x1x1xf32>
    %v8458 = stablehlo.add %v8457, %v2814 : tensor<576x96x1x1xf32>
    %v8459 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8460 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8461 = stablehlo.multiply %v8459, %b12eWv : tensor<576x96x1x1xf32>
    %v8462 = stablehlo.multiply %v8458, %v8458 : tensor<576x96x1x1xf32>
    %v8463 = stablehlo.multiply %v8460, %v8462 : tensor<576x96x1x1xf32>
    %v8464 = stablehlo.add %v8461, %v8463 : tensor<576x96x1x1xf32>
    %v8465 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8466 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8467 = stablehlo.multiply %v8465, %b12eWv : tensor<576x96x1x1xf32>
    %v8468 = stablehlo.multiply %v8458, %v8458 : tensor<576x96x1x1xf32>
    %v8469 = stablehlo.multiply %v8466, %v8468 : tensor<576x96x1x1xf32>
    %v8470 = stablehlo.add %v8467, %v8469 : tensor<576x96x1x1xf32>
    %v8471 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8472 = stablehlo.add %v8470, %v8471 : tensor<576x96x1x1xf32>
    %v8473 = stablehlo.sqrt %v8472 : tensor<576x96x1x1xf32>
    %v8474 = stablehlo.divide %v8458, %v8473 : tensor<576x96x1x1xf32>
    %v8475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8476 = stablehlo.multiply %v8475, %b12eWm : tensor<576x96x1x1xf32>
    %v8477 = stablehlo.add %v8476, %v8474 : tensor<576x96x1x1xf32>
    %v8478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8479 = stablehlo.multiply %v8478, %v8477 : tensor<576x96x1x1xf32>
    %v8480 = stablehlo.subtract %b12eW, %v8479 : tensor<576x96x1x1xf32>
    %v8481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8482 = stablehlo.multiply %v8481, %b12eg : tensor<576xf32>
    %v8483 = stablehlo.add %v8482, %v2832 : tensor<576xf32>
    %v8484 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8485 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8486 = stablehlo.multiply %v8484, %b12egv : tensor<576xf32>
    %v8487 = stablehlo.multiply %v8483, %v8483 : tensor<576xf32>
    %v8488 = stablehlo.multiply %v8485, %v8487 : tensor<576xf32>
    %v8489 = stablehlo.add %v8486, %v8488 : tensor<576xf32>
    %v8490 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8491 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8492 = stablehlo.multiply %v8490, %b12egv : tensor<576xf32>
    %v8493 = stablehlo.multiply %v8483, %v8483 : tensor<576xf32>
    %v8494 = stablehlo.multiply %v8491, %v8493 : tensor<576xf32>
    %v8495 = stablehlo.add %v8492, %v8494 : tensor<576xf32>
    %v8496 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8497 = stablehlo.add %v8495, %v8496 : tensor<576xf32>
    %v8498 = stablehlo.sqrt %v8497 : tensor<576xf32>
    %v8499 = stablehlo.divide %v8483, %v8498 : tensor<576xf32>
    %v8500 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8501 = stablehlo.multiply %v8500, %b12egm : tensor<576xf32>
    %v8502 = stablehlo.add %v8501, %v8499 : tensor<576xf32>
    %v8503 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8504 = stablehlo.multiply %v8503, %v8502 : tensor<576xf32>
    %v8505 = stablehlo.subtract %b12eg, %v8504 : tensor<576xf32>
    %v8506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8507 = stablehlo.multiply %v8506, %b12ebt : tensor<576xf32>
    %v8508 = stablehlo.add %v8507, %v2835 : tensor<576xf32>
    %v8509 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8510 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8511 = stablehlo.multiply %v8509, %b12ebtv : tensor<576xf32>
    %v8512 = stablehlo.multiply %v8508, %v8508 : tensor<576xf32>
    %v8513 = stablehlo.multiply %v8510, %v8512 : tensor<576xf32>
    %v8514 = stablehlo.add %v8511, %v8513 : tensor<576xf32>
    %v8515 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8516 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8517 = stablehlo.multiply %v8515, %b12ebtv : tensor<576xf32>
    %v8518 = stablehlo.multiply %v8508, %v8508 : tensor<576xf32>
    %v8519 = stablehlo.multiply %v8516, %v8518 : tensor<576xf32>
    %v8520 = stablehlo.add %v8517, %v8519 : tensor<576xf32>
    %v8521 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8522 = stablehlo.add %v8520, %v8521 : tensor<576xf32>
    %v8523 = stablehlo.sqrt %v8522 : tensor<576xf32>
    %v8524 = stablehlo.divide %v8508, %v8523 : tensor<576xf32>
    %v8525 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8526 = stablehlo.multiply %v8525, %b12ebtm : tensor<576xf32>
    %v8527 = stablehlo.add %v8526, %v8524 : tensor<576xf32>
    %v8528 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8529 = stablehlo.multiply %v8528, %v8527 : tensor<576xf32>
    %v8530 = stablehlo.subtract %b12ebt, %v8529 : tensor<576xf32>
    %v8531 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8532 = stablehlo.multiply %v8531, %b12dW : tensor<576x1x3x3xf32>
    %v8533 = stablehlo.add %v8532, %v2841 : tensor<576x1x3x3xf32>
    %v8534 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8535 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8536 = stablehlo.multiply %v8534, %b12dWv : tensor<576x1x3x3xf32>
    %v8537 = stablehlo.multiply %v8533, %v8533 : tensor<576x1x3x3xf32>
    %v8538 = stablehlo.multiply %v8535, %v8537 : tensor<576x1x3x3xf32>
    %v8539 = stablehlo.add %v8536, %v8538 : tensor<576x1x3x3xf32>
    %v8540 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8541 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8542 = stablehlo.multiply %v8540, %b12dWv : tensor<576x1x3x3xf32>
    %v8543 = stablehlo.multiply %v8533, %v8533 : tensor<576x1x3x3xf32>
    %v8544 = stablehlo.multiply %v8541, %v8543 : tensor<576x1x3x3xf32>
    %v8545 = stablehlo.add %v8542, %v8544 : tensor<576x1x3x3xf32>
    %v8546 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8547 = stablehlo.add %v8545, %v8546 : tensor<576x1x3x3xf32>
    %v8548 = stablehlo.sqrt %v8547 : tensor<576x1x3x3xf32>
    %v8549 = stablehlo.divide %v8533, %v8548 : tensor<576x1x3x3xf32>
    %v8550 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8551 = stablehlo.multiply %v8550, %b12dWm : tensor<576x1x3x3xf32>
    %v8552 = stablehlo.add %v8551, %v8549 : tensor<576x1x3x3xf32>
    %v8553 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8554 = stablehlo.multiply %v8553, %v8552 : tensor<576x1x3x3xf32>
    %v8555 = stablehlo.subtract %b12dW, %v8554 : tensor<576x1x3x3xf32>
    %v8556 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8557 = stablehlo.multiply %v8556, %b12dg : tensor<576xf32>
    %v8558 = stablehlo.add %v8557, %v2859 : tensor<576xf32>
    %v8559 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8560 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8561 = stablehlo.multiply %v8559, %b12dgv : tensor<576xf32>
    %v8562 = stablehlo.multiply %v8558, %v8558 : tensor<576xf32>
    %v8563 = stablehlo.multiply %v8560, %v8562 : tensor<576xf32>
    %v8564 = stablehlo.add %v8561, %v8563 : tensor<576xf32>
    %v8565 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8566 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8567 = stablehlo.multiply %v8565, %b12dgv : tensor<576xf32>
    %v8568 = stablehlo.multiply %v8558, %v8558 : tensor<576xf32>
    %v8569 = stablehlo.multiply %v8566, %v8568 : tensor<576xf32>
    %v8570 = stablehlo.add %v8567, %v8569 : tensor<576xf32>
    %v8571 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8572 = stablehlo.add %v8570, %v8571 : tensor<576xf32>
    %v8573 = stablehlo.sqrt %v8572 : tensor<576xf32>
    %v8574 = stablehlo.divide %v8558, %v8573 : tensor<576xf32>
    %v8575 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8576 = stablehlo.multiply %v8575, %b12dgm : tensor<576xf32>
    %v8577 = stablehlo.add %v8576, %v8574 : tensor<576xf32>
    %v8578 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8579 = stablehlo.multiply %v8578, %v8577 : tensor<576xf32>
    %v8580 = stablehlo.subtract %b12dg, %v8579 : tensor<576xf32>
    %v8581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8582 = stablehlo.multiply %v8581, %b12dbt : tensor<576xf32>
    %v8583 = stablehlo.add %v8582, %v2862 : tensor<576xf32>
    %v8584 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8585 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8586 = stablehlo.multiply %v8584, %b12dbtv : tensor<576xf32>
    %v8587 = stablehlo.multiply %v8583, %v8583 : tensor<576xf32>
    %v8588 = stablehlo.multiply %v8585, %v8587 : tensor<576xf32>
    %v8589 = stablehlo.add %v8586, %v8588 : tensor<576xf32>
    %v8590 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8591 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8592 = stablehlo.multiply %v8590, %b12dbtv : tensor<576xf32>
    %v8593 = stablehlo.multiply %v8583, %v8583 : tensor<576xf32>
    %v8594 = stablehlo.multiply %v8591, %v8593 : tensor<576xf32>
    %v8595 = stablehlo.add %v8592, %v8594 : tensor<576xf32>
    %v8596 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8597 = stablehlo.add %v8595, %v8596 : tensor<576xf32>
    %v8598 = stablehlo.sqrt %v8597 : tensor<576xf32>
    %v8599 = stablehlo.divide %v8583, %v8598 : tensor<576xf32>
    %v8600 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8601 = stablehlo.multiply %v8600, %b12dbtm : tensor<576xf32>
    %v8602 = stablehlo.add %v8601, %v8599 : tensor<576xf32>
    %v8603 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8604 = stablehlo.multiply %v8603, %v8602 : tensor<576xf32>
    %v8605 = stablehlo.subtract %b12dbt, %v8604 : tensor<576xf32>
    %v8606 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8607 = stablehlo.multiply %v8606, %b12pW : tensor<96x576x1x1xf32>
    %v8608 = stablehlo.add %v8607, %v2868 : tensor<96x576x1x1xf32>
    %v8609 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8610 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8611 = stablehlo.multiply %v8609, %b12pWv : tensor<96x576x1x1xf32>
    %v8612 = stablehlo.multiply %v8608, %v8608 : tensor<96x576x1x1xf32>
    %v8613 = stablehlo.multiply %v8610, %v8612 : tensor<96x576x1x1xf32>
    %v8614 = stablehlo.add %v8611, %v8613 : tensor<96x576x1x1xf32>
    %v8615 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8616 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8617 = stablehlo.multiply %v8615, %b12pWv : tensor<96x576x1x1xf32>
    %v8618 = stablehlo.multiply %v8608, %v8608 : tensor<96x576x1x1xf32>
    %v8619 = stablehlo.multiply %v8616, %v8618 : tensor<96x576x1x1xf32>
    %v8620 = stablehlo.add %v8617, %v8619 : tensor<96x576x1x1xf32>
    %v8621 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8622 = stablehlo.add %v8620, %v8621 : tensor<96x576x1x1xf32>
    %v8623 = stablehlo.sqrt %v8622 : tensor<96x576x1x1xf32>
    %v8624 = stablehlo.divide %v8608, %v8623 : tensor<96x576x1x1xf32>
    %v8625 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8626 = stablehlo.multiply %v8625, %b12pWm : tensor<96x576x1x1xf32>
    %v8627 = stablehlo.add %v8626, %v8624 : tensor<96x576x1x1xf32>
    %v8628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8629 = stablehlo.multiply %v8628, %v8627 : tensor<96x576x1x1xf32>
    %v8630 = stablehlo.subtract %b12pW, %v8629 : tensor<96x576x1x1xf32>
    %v8631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8632 = stablehlo.multiply %v8631, %b12pg : tensor<96xf32>
    %v8633 = stablehlo.add %v8632, %v2886 : tensor<96xf32>
    %v8634 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8635 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8636 = stablehlo.multiply %v8634, %b12pgv : tensor<96xf32>
    %v8637 = stablehlo.multiply %v8633, %v8633 : tensor<96xf32>
    %v8638 = stablehlo.multiply %v8635, %v8637 : tensor<96xf32>
    %v8639 = stablehlo.add %v8636, %v8638 : tensor<96xf32>
    %v8640 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8641 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8642 = stablehlo.multiply %v8640, %b12pgv : tensor<96xf32>
    %v8643 = stablehlo.multiply %v8633, %v8633 : tensor<96xf32>
    %v8644 = stablehlo.multiply %v8641, %v8643 : tensor<96xf32>
    %v8645 = stablehlo.add %v8642, %v8644 : tensor<96xf32>
    %v8646 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8647 = stablehlo.add %v8645, %v8646 : tensor<96xf32>
    %v8648 = stablehlo.sqrt %v8647 : tensor<96xf32>
    %v8649 = stablehlo.divide %v8633, %v8648 : tensor<96xf32>
    %v8650 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8651 = stablehlo.multiply %v8650, %b12pgm : tensor<96xf32>
    %v8652 = stablehlo.add %v8651, %v8649 : tensor<96xf32>
    %v8653 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8654 = stablehlo.multiply %v8653, %v8652 : tensor<96xf32>
    %v8655 = stablehlo.subtract %b12pg, %v8654 : tensor<96xf32>
    %v8656 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8657 = stablehlo.multiply %v8656, %b12pbt : tensor<96xf32>
    %v8658 = stablehlo.add %v8657, %v2889 : tensor<96xf32>
    %v8659 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8660 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8661 = stablehlo.multiply %v8659, %b12pbtv : tensor<96xf32>
    %v8662 = stablehlo.multiply %v8658, %v8658 : tensor<96xf32>
    %v8663 = stablehlo.multiply %v8660, %v8662 : tensor<96xf32>
    %v8664 = stablehlo.add %v8661, %v8663 : tensor<96xf32>
    %v8665 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8666 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8667 = stablehlo.multiply %v8665, %b12pbtv : tensor<96xf32>
    %v8668 = stablehlo.multiply %v8658, %v8658 : tensor<96xf32>
    %v8669 = stablehlo.multiply %v8666, %v8668 : tensor<96xf32>
    %v8670 = stablehlo.add %v8667, %v8669 : tensor<96xf32>
    %v8671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8672 = stablehlo.add %v8670, %v8671 : tensor<96xf32>
    %v8673 = stablehlo.sqrt %v8672 : tensor<96xf32>
    %v8674 = stablehlo.divide %v8658, %v8673 : tensor<96xf32>
    %v8675 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8676 = stablehlo.multiply %v8675, %b12pbtm : tensor<96xf32>
    %v8677 = stablehlo.add %v8676, %v8674 : tensor<96xf32>
    %v8678 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8679 = stablehlo.multiply %v8678, %v8677 : tensor<96xf32>
    %v8680 = stablehlo.subtract %b12pbt, %v8679 : tensor<96xf32>
    %v8681 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8682 = stablehlo.multiply %v8681, %b13eW : tensor<576x96x1x1xf32>
    %v8683 = stablehlo.add %v8682, %v2607 : tensor<576x96x1x1xf32>
    %v8684 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8685 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8686 = stablehlo.multiply %v8684, %b13eWv : tensor<576x96x1x1xf32>
    %v8687 = stablehlo.multiply %v8683, %v8683 : tensor<576x96x1x1xf32>
    %v8688 = stablehlo.multiply %v8685, %v8687 : tensor<576x96x1x1xf32>
    %v8689 = stablehlo.add %v8686, %v8688 : tensor<576x96x1x1xf32>
    %v8690 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8691 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8692 = stablehlo.multiply %v8690, %b13eWv : tensor<576x96x1x1xf32>
    %v8693 = stablehlo.multiply %v8683, %v8683 : tensor<576x96x1x1xf32>
    %v8694 = stablehlo.multiply %v8691, %v8693 : tensor<576x96x1x1xf32>
    %v8695 = stablehlo.add %v8692, %v8694 : tensor<576x96x1x1xf32>
    %v8696 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8697 = stablehlo.add %v8695, %v8696 : tensor<576x96x1x1xf32>
    %v8698 = stablehlo.sqrt %v8697 : tensor<576x96x1x1xf32>
    %v8699 = stablehlo.divide %v8683, %v8698 : tensor<576x96x1x1xf32>
    %v8700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8701 = stablehlo.multiply %v8700, %b13eWm : tensor<576x96x1x1xf32>
    %v8702 = stablehlo.add %v8701, %v8699 : tensor<576x96x1x1xf32>
    %v8703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8704 = stablehlo.multiply %v8703, %v8702 : tensor<576x96x1x1xf32>
    %v8705 = stablehlo.subtract %b13eW, %v8704 : tensor<576x96x1x1xf32>
    %v8706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8707 = stablehlo.multiply %v8706, %b13eg : tensor<576xf32>
    %v8708 = stablehlo.add %v8707, %v2625 : tensor<576xf32>
    %v8709 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8710 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8711 = stablehlo.multiply %v8709, %b13egv : tensor<576xf32>
    %v8712 = stablehlo.multiply %v8708, %v8708 : tensor<576xf32>
    %v8713 = stablehlo.multiply %v8710, %v8712 : tensor<576xf32>
    %v8714 = stablehlo.add %v8711, %v8713 : tensor<576xf32>
    %v8715 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8716 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8717 = stablehlo.multiply %v8715, %b13egv : tensor<576xf32>
    %v8718 = stablehlo.multiply %v8708, %v8708 : tensor<576xf32>
    %v8719 = stablehlo.multiply %v8716, %v8718 : tensor<576xf32>
    %v8720 = stablehlo.add %v8717, %v8719 : tensor<576xf32>
    %v8721 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8722 = stablehlo.add %v8720, %v8721 : tensor<576xf32>
    %v8723 = stablehlo.sqrt %v8722 : tensor<576xf32>
    %v8724 = stablehlo.divide %v8708, %v8723 : tensor<576xf32>
    %v8725 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8726 = stablehlo.multiply %v8725, %b13egm : tensor<576xf32>
    %v8727 = stablehlo.add %v8726, %v8724 : tensor<576xf32>
    %v8728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8729 = stablehlo.multiply %v8728, %v8727 : tensor<576xf32>
    %v8730 = stablehlo.subtract %b13eg, %v8729 : tensor<576xf32>
    %v8731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8732 = stablehlo.multiply %v8731, %b13ebt : tensor<576xf32>
    %v8733 = stablehlo.add %v8732, %v2628 : tensor<576xf32>
    %v8734 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8735 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8736 = stablehlo.multiply %v8734, %b13ebtv : tensor<576xf32>
    %v8737 = stablehlo.multiply %v8733, %v8733 : tensor<576xf32>
    %v8738 = stablehlo.multiply %v8735, %v8737 : tensor<576xf32>
    %v8739 = stablehlo.add %v8736, %v8738 : tensor<576xf32>
    %v8740 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8741 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8742 = stablehlo.multiply %v8740, %b13ebtv : tensor<576xf32>
    %v8743 = stablehlo.multiply %v8733, %v8733 : tensor<576xf32>
    %v8744 = stablehlo.multiply %v8741, %v8743 : tensor<576xf32>
    %v8745 = stablehlo.add %v8742, %v8744 : tensor<576xf32>
    %v8746 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8747 = stablehlo.add %v8745, %v8746 : tensor<576xf32>
    %v8748 = stablehlo.sqrt %v8747 : tensor<576xf32>
    %v8749 = stablehlo.divide %v8733, %v8748 : tensor<576xf32>
    %v8750 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8751 = stablehlo.multiply %v8750, %b13ebtm : tensor<576xf32>
    %v8752 = stablehlo.add %v8751, %v8749 : tensor<576xf32>
    %v8753 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8754 = stablehlo.multiply %v8753, %v8752 : tensor<576xf32>
    %v8755 = stablehlo.subtract %b13ebt, %v8754 : tensor<576xf32>
    %v8756 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8757 = stablehlo.multiply %v8756, %b13dW : tensor<576x1x3x3xf32>
    %v8758 = stablehlo.add %v8757, %v2634 : tensor<576x1x3x3xf32>
    %v8759 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8760 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8761 = stablehlo.multiply %v8759, %b13dWv : tensor<576x1x3x3xf32>
    %v8762 = stablehlo.multiply %v8758, %v8758 : tensor<576x1x3x3xf32>
    %v8763 = stablehlo.multiply %v8760, %v8762 : tensor<576x1x3x3xf32>
    %v8764 = stablehlo.add %v8761, %v8763 : tensor<576x1x3x3xf32>
    %v8765 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8766 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8767 = stablehlo.multiply %v8765, %b13dWv : tensor<576x1x3x3xf32>
    %v8768 = stablehlo.multiply %v8758, %v8758 : tensor<576x1x3x3xf32>
    %v8769 = stablehlo.multiply %v8766, %v8768 : tensor<576x1x3x3xf32>
    %v8770 = stablehlo.add %v8767, %v8769 : tensor<576x1x3x3xf32>
    %v8771 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8772 = stablehlo.add %v8770, %v8771 : tensor<576x1x3x3xf32>
    %v8773 = stablehlo.sqrt %v8772 : tensor<576x1x3x3xf32>
    %v8774 = stablehlo.divide %v8758, %v8773 : tensor<576x1x3x3xf32>
    %v8775 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8776 = stablehlo.multiply %v8775, %b13dWm : tensor<576x1x3x3xf32>
    %v8777 = stablehlo.add %v8776, %v8774 : tensor<576x1x3x3xf32>
    %v8778 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8779 = stablehlo.multiply %v8778, %v8777 : tensor<576x1x3x3xf32>
    %v8780 = stablehlo.subtract %b13dW, %v8779 : tensor<576x1x3x3xf32>
    %v8781 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8782 = stablehlo.multiply %v8781, %b13dg : tensor<576xf32>
    %v8783 = stablehlo.add %v8782, %v2652 : tensor<576xf32>
    %v8784 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8785 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8786 = stablehlo.multiply %v8784, %b13dgv : tensor<576xf32>
    %v8787 = stablehlo.multiply %v8783, %v8783 : tensor<576xf32>
    %v8788 = stablehlo.multiply %v8785, %v8787 : tensor<576xf32>
    %v8789 = stablehlo.add %v8786, %v8788 : tensor<576xf32>
    %v8790 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8791 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8792 = stablehlo.multiply %v8790, %b13dgv : tensor<576xf32>
    %v8793 = stablehlo.multiply %v8783, %v8783 : tensor<576xf32>
    %v8794 = stablehlo.multiply %v8791, %v8793 : tensor<576xf32>
    %v8795 = stablehlo.add %v8792, %v8794 : tensor<576xf32>
    %v8796 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8797 = stablehlo.add %v8795, %v8796 : tensor<576xf32>
    %v8798 = stablehlo.sqrt %v8797 : tensor<576xf32>
    %v8799 = stablehlo.divide %v8783, %v8798 : tensor<576xf32>
    %v8800 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8801 = stablehlo.multiply %v8800, %b13dgm : tensor<576xf32>
    %v8802 = stablehlo.add %v8801, %v8799 : tensor<576xf32>
    %v8803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8804 = stablehlo.multiply %v8803, %v8802 : tensor<576xf32>
    %v8805 = stablehlo.subtract %b13dg, %v8804 : tensor<576xf32>
    %v8806 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8807 = stablehlo.multiply %v8806, %b13dbt : tensor<576xf32>
    %v8808 = stablehlo.add %v8807, %v2655 : tensor<576xf32>
    %v8809 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8810 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8811 = stablehlo.multiply %v8809, %b13dbtv : tensor<576xf32>
    %v8812 = stablehlo.multiply %v8808, %v8808 : tensor<576xf32>
    %v8813 = stablehlo.multiply %v8810, %v8812 : tensor<576xf32>
    %v8814 = stablehlo.add %v8811, %v8813 : tensor<576xf32>
    %v8815 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8816 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8817 = stablehlo.multiply %v8815, %b13dbtv : tensor<576xf32>
    %v8818 = stablehlo.multiply %v8808, %v8808 : tensor<576xf32>
    %v8819 = stablehlo.multiply %v8816, %v8818 : tensor<576xf32>
    %v8820 = stablehlo.add %v8817, %v8819 : tensor<576xf32>
    %v8821 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8822 = stablehlo.add %v8820, %v8821 : tensor<576xf32>
    %v8823 = stablehlo.sqrt %v8822 : tensor<576xf32>
    %v8824 = stablehlo.divide %v8808, %v8823 : tensor<576xf32>
    %v8825 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8826 = stablehlo.multiply %v8825, %b13dbtm : tensor<576xf32>
    %v8827 = stablehlo.add %v8826, %v8824 : tensor<576xf32>
    %v8828 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8829 = stablehlo.multiply %v8828, %v8827 : tensor<576xf32>
    %v8830 = stablehlo.subtract %b13dbt, %v8829 : tensor<576xf32>
    %v8831 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8832 = stablehlo.multiply %v8831, %b13pW : tensor<96x576x1x1xf32>
    %v8833 = stablehlo.add %v8832, %v2661 : tensor<96x576x1x1xf32>
    %v8834 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8835 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8836 = stablehlo.multiply %v8834, %b13pWv : tensor<96x576x1x1xf32>
    %v8837 = stablehlo.multiply %v8833, %v8833 : tensor<96x576x1x1xf32>
    %v8838 = stablehlo.multiply %v8835, %v8837 : tensor<96x576x1x1xf32>
    %v8839 = stablehlo.add %v8836, %v8838 : tensor<96x576x1x1xf32>
    %v8840 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8841 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8842 = stablehlo.multiply %v8840, %b13pWv : tensor<96x576x1x1xf32>
    %v8843 = stablehlo.multiply %v8833, %v8833 : tensor<96x576x1x1xf32>
    %v8844 = stablehlo.multiply %v8841, %v8843 : tensor<96x576x1x1xf32>
    %v8845 = stablehlo.add %v8842, %v8844 : tensor<96x576x1x1xf32>
    %v8846 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8847 = stablehlo.add %v8845, %v8846 : tensor<96x576x1x1xf32>
    %v8848 = stablehlo.sqrt %v8847 : tensor<96x576x1x1xf32>
    %v8849 = stablehlo.divide %v8833, %v8848 : tensor<96x576x1x1xf32>
    %v8850 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8851 = stablehlo.multiply %v8850, %b13pWm : tensor<96x576x1x1xf32>
    %v8852 = stablehlo.add %v8851, %v8849 : tensor<96x576x1x1xf32>
    %v8853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8854 = stablehlo.multiply %v8853, %v8852 : tensor<96x576x1x1xf32>
    %v8855 = stablehlo.subtract %b13pW, %v8854 : tensor<96x576x1x1xf32>
    %v8856 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8857 = stablehlo.multiply %v8856, %b13pg : tensor<96xf32>
    %v8858 = stablehlo.add %v8857, %v2679 : tensor<96xf32>
    %v8859 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8860 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8861 = stablehlo.multiply %v8859, %b13pgv : tensor<96xf32>
    %v8862 = stablehlo.multiply %v8858, %v8858 : tensor<96xf32>
    %v8863 = stablehlo.multiply %v8860, %v8862 : tensor<96xf32>
    %v8864 = stablehlo.add %v8861, %v8863 : tensor<96xf32>
    %v8865 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8866 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8867 = stablehlo.multiply %v8865, %b13pgv : tensor<96xf32>
    %v8868 = stablehlo.multiply %v8858, %v8858 : tensor<96xf32>
    %v8869 = stablehlo.multiply %v8866, %v8868 : tensor<96xf32>
    %v8870 = stablehlo.add %v8867, %v8869 : tensor<96xf32>
    %v8871 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8872 = stablehlo.add %v8870, %v8871 : tensor<96xf32>
    %v8873 = stablehlo.sqrt %v8872 : tensor<96xf32>
    %v8874 = stablehlo.divide %v8858, %v8873 : tensor<96xf32>
    %v8875 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8876 = stablehlo.multiply %v8875, %b13pgm : tensor<96xf32>
    %v8877 = stablehlo.add %v8876, %v8874 : tensor<96xf32>
    %v8878 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8879 = stablehlo.multiply %v8878, %v8877 : tensor<96xf32>
    %v8880 = stablehlo.subtract %b13pg, %v8879 : tensor<96xf32>
    %v8881 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8882 = stablehlo.multiply %v8881, %b13pbt : tensor<96xf32>
    %v8883 = stablehlo.add %v8882, %v2682 : tensor<96xf32>
    %v8884 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8885 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8886 = stablehlo.multiply %v8884, %b13pbtv : tensor<96xf32>
    %v8887 = stablehlo.multiply %v8883, %v8883 : tensor<96xf32>
    %v8888 = stablehlo.multiply %v8885, %v8887 : tensor<96xf32>
    %v8889 = stablehlo.add %v8886, %v8888 : tensor<96xf32>
    %v8890 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8891 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8892 = stablehlo.multiply %v8890, %b13pbtv : tensor<96xf32>
    %v8893 = stablehlo.multiply %v8883, %v8883 : tensor<96xf32>
    %v8894 = stablehlo.multiply %v8891, %v8893 : tensor<96xf32>
    %v8895 = stablehlo.add %v8892, %v8894 : tensor<96xf32>
    %v8896 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8897 = stablehlo.add %v8895, %v8896 : tensor<96xf32>
    %v8898 = stablehlo.sqrt %v8897 : tensor<96xf32>
    %v8899 = stablehlo.divide %v8883, %v8898 : tensor<96xf32>
    %v8900 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8901 = stablehlo.multiply %v8900, %b13pbtm : tensor<96xf32>
    %v8902 = stablehlo.add %v8901, %v8899 : tensor<96xf32>
    %v8903 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8904 = stablehlo.multiply %v8903, %v8902 : tensor<96xf32>
    %v8905 = stablehlo.subtract %b13pbt, %v8904 : tensor<96xf32>
    %v8906 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8907 = stablehlo.multiply %v8906, %b14eW : tensor<576x96x1x1xf32>
    %v8908 = stablehlo.add %v8907, %v2398 : tensor<576x96x1x1xf32>
    %v8909 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8910 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8911 = stablehlo.multiply %v8909, %b14eWv : tensor<576x96x1x1xf32>
    %v8912 = stablehlo.multiply %v8908, %v8908 : tensor<576x96x1x1xf32>
    %v8913 = stablehlo.multiply %v8910, %v8912 : tensor<576x96x1x1xf32>
    %v8914 = stablehlo.add %v8911, %v8913 : tensor<576x96x1x1xf32>
    %v8915 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8916 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8917 = stablehlo.multiply %v8915, %b14eWv : tensor<576x96x1x1xf32>
    %v8918 = stablehlo.multiply %v8908, %v8908 : tensor<576x96x1x1xf32>
    %v8919 = stablehlo.multiply %v8916, %v8918 : tensor<576x96x1x1xf32>
    %v8920 = stablehlo.add %v8917, %v8919 : tensor<576x96x1x1xf32>
    %v8921 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8922 = stablehlo.add %v8920, %v8921 : tensor<576x96x1x1xf32>
    %v8923 = stablehlo.sqrt %v8922 : tensor<576x96x1x1xf32>
    %v8924 = stablehlo.divide %v8908, %v8923 : tensor<576x96x1x1xf32>
    %v8925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8926 = stablehlo.multiply %v8925, %b14eWm : tensor<576x96x1x1xf32>
    %v8927 = stablehlo.add %v8926, %v8924 : tensor<576x96x1x1xf32>
    %v8928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8929 = stablehlo.multiply %v8928, %v8927 : tensor<576x96x1x1xf32>
    %v8930 = stablehlo.subtract %b14eW, %v8929 : tensor<576x96x1x1xf32>
    %v8931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8932 = stablehlo.multiply %v8931, %b14eg : tensor<576xf32>
    %v8933 = stablehlo.add %v8932, %v2416 : tensor<576xf32>
    %v8934 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8935 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8936 = stablehlo.multiply %v8934, %b14egv : tensor<576xf32>
    %v8937 = stablehlo.multiply %v8933, %v8933 : tensor<576xf32>
    %v8938 = stablehlo.multiply %v8935, %v8937 : tensor<576xf32>
    %v8939 = stablehlo.add %v8936, %v8938 : tensor<576xf32>
    %v8940 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8941 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8942 = stablehlo.multiply %v8940, %b14egv : tensor<576xf32>
    %v8943 = stablehlo.multiply %v8933, %v8933 : tensor<576xf32>
    %v8944 = stablehlo.multiply %v8941, %v8943 : tensor<576xf32>
    %v8945 = stablehlo.add %v8942, %v8944 : tensor<576xf32>
    %v8946 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8947 = stablehlo.add %v8945, %v8946 : tensor<576xf32>
    %v8948 = stablehlo.sqrt %v8947 : tensor<576xf32>
    %v8949 = stablehlo.divide %v8933, %v8948 : tensor<576xf32>
    %v8950 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8951 = stablehlo.multiply %v8950, %b14egm : tensor<576xf32>
    %v8952 = stablehlo.add %v8951, %v8949 : tensor<576xf32>
    %v8953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8954 = stablehlo.multiply %v8953, %v8952 : tensor<576xf32>
    %v8955 = stablehlo.subtract %b14eg, %v8954 : tensor<576xf32>
    %v8956 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8957 = stablehlo.multiply %v8956, %b14ebt : tensor<576xf32>
    %v8958 = stablehlo.add %v8957, %v2419 : tensor<576xf32>
    %v8959 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8960 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8961 = stablehlo.multiply %v8959, %b14ebtv : tensor<576xf32>
    %v8962 = stablehlo.multiply %v8958, %v8958 : tensor<576xf32>
    %v8963 = stablehlo.multiply %v8960, %v8962 : tensor<576xf32>
    %v8964 = stablehlo.add %v8961, %v8963 : tensor<576xf32>
    %v8965 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8966 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8967 = stablehlo.multiply %v8965, %b14ebtv : tensor<576xf32>
    %v8968 = stablehlo.multiply %v8958, %v8958 : tensor<576xf32>
    %v8969 = stablehlo.multiply %v8966, %v8968 : tensor<576xf32>
    %v8970 = stablehlo.add %v8967, %v8969 : tensor<576xf32>
    %v8971 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8972 = stablehlo.add %v8970, %v8971 : tensor<576xf32>
    %v8973 = stablehlo.sqrt %v8972 : tensor<576xf32>
    %v8974 = stablehlo.divide %v8958, %v8973 : tensor<576xf32>
    %v8975 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8976 = stablehlo.multiply %v8975, %b14ebtm : tensor<576xf32>
    %v8977 = stablehlo.add %v8976, %v8974 : tensor<576xf32>
    %v8978 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8979 = stablehlo.multiply %v8978, %v8977 : tensor<576xf32>
    %v8980 = stablehlo.subtract %b14ebt, %v8979 : tensor<576xf32>
    %v8981 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8982 = stablehlo.multiply %v8981, %b14dW : tensor<576x1x3x3xf32>
    %v8983 = stablehlo.add %v8982, %v2427 : tensor<576x1x3x3xf32>
    %v8984 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8985 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8986 = stablehlo.multiply %v8984, %b14dWv : tensor<576x1x3x3xf32>
    %v8987 = stablehlo.multiply %v8983, %v8983 : tensor<576x1x3x3xf32>
    %v8988 = stablehlo.multiply %v8985, %v8987 : tensor<576x1x3x3xf32>
    %v8989 = stablehlo.add %v8986, %v8988 : tensor<576x1x3x3xf32>
    %v8990 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8991 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8992 = stablehlo.multiply %v8990, %b14dWv : tensor<576x1x3x3xf32>
    %v8993 = stablehlo.multiply %v8983, %v8983 : tensor<576x1x3x3xf32>
    %v8994 = stablehlo.multiply %v8991, %v8993 : tensor<576x1x3x3xf32>
    %v8995 = stablehlo.add %v8992, %v8994 : tensor<576x1x3x3xf32>
    %v8996 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8997 = stablehlo.add %v8995, %v8996 : tensor<576x1x3x3xf32>
    %v8998 = stablehlo.sqrt %v8997 : tensor<576x1x3x3xf32>
    %v8999 = stablehlo.divide %v8983, %v8998 : tensor<576x1x3x3xf32>
    %v9000 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v9001 = stablehlo.multiply %v9000, %b14dWm : tensor<576x1x3x3xf32>
    %v9002 = stablehlo.add %v9001, %v8999 : tensor<576x1x3x3xf32>
    %v9003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v9004 = stablehlo.multiply %v9003, %v9002 : tensor<576x1x3x3xf32>
    %v9005 = stablehlo.subtract %b14dW, %v9004 : tensor<576x1x3x3xf32>
    %v9006 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9007 = stablehlo.multiply %v9006, %b14dg : tensor<576xf32>
    %v9008 = stablehlo.add %v9007, %v2445 : tensor<576xf32>
    %v9009 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9010 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9011 = stablehlo.multiply %v9009, %b14dgv : tensor<576xf32>
    %v9012 = stablehlo.multiply %v9008, %v9008 : tensor<576xf32>
    %v9013 = stablehlo.multiply %v9010, %v9012 : tensor<576xf32>
    %v9014 = stablehlo.add %v9011, %v9013 : tensor<576xf32>
    %v9015 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9016 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9017 = stablehlo.multiply %v9015, %b14dgv : tensor<576xf32>
    %v9018 = stablehlo.multiply %v9008, %v9008 : tensor<576xf32>
    %v9019 = stablehlo.multiply %v9016, %v9018 : tensor<576xf32>
    %v9020 = stablehlo.add %v9017, %v9019 : tensor<576xf32>
    %v9021 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9022 = stablehlo.add %v9020, %v9021 : tensor<576xf32>
    %v9023 = stablehlo.sqrt %v9022 : tensor<576xf32>
    %v9024 = stablehlo.divide %v9008, %v9023 : tensor<576xf32>
    %v9025 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9026 = stablehlo.multiply %v9025, %b14dgm : tensor<576xf32>
    %v9027 = stablehlo.add %v9026, %v9024 : tensor<576xf32>
    %v9028 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9029 = stablehlo.multiply %v9028, %v9027 : tensor<576xf32>
    %v9030 = stablehlo.subtract %b14dg, %v9029 : tensor<576xf32>
    %v9031 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9032 = stablehlo.multiply %v9031, %b14dbt : tensor<576xf32>
    %v9033 = stablehlo.add %v9032, %v2448 : tensor<576xf32>
    %v9034 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9035 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9036 = stablehlo.multiply %v9034, %b14dbtv : tensor<576xf32>
    %v9037 = stablehlo.multiply %v9033, %v9033 : tensor<576xf32>
    %v9038 = stablehlo.multiply %v9035, %v9037 : tensor<576xf32>
    %v9039 = stablehlo.add %v9036, %v9038 : tensor<576xf32>
    %v9040 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9041 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9042 = stablehlo.multiply %v9040, %b14dbtv : tensor<576xf32>
    %v9043 = stablehlo.multiply %v9033, %v9033 : tensor<576xf32>
    %v9044 = stablehlo.multiply %v9041, %v9043 : tensor<576xf32>
    %v9045 = stablehlo.add %v9042, %v9044 : tensor<576xf32>
    %v9046 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9047 = stablehlo.add %v9045, %v9046 : tensor<576xf32>
    %v9048 = stablehlo.sqrt %v9047 : tensor<576xf32>
    %v9049 = stablehlo.divide %v9033, %v9048 : tensor<576xf32>
    %v9050 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9051 = stablehlo.multiply %v9050, %b14dbtm : tensor<576xf32>
    %v9052 = stablehlo.add %v9051, %v9049 : tensor<576xf32>
    %v9053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9054 = stablehlo.multiply %v9053, %v9052 : tensor<576xf32>
    %v9055 = stablehlo.subtract %b14dbt, %v9054 : tensor<576xf32>
    %v9056 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9057 = stablehlo.multiply %v9056, %b14pW : tensor<160x576x1x1xf32>
    %v9058 = stablehlo.add %v9057, %v2454 : tensor<160x576x1x1xf32>
    %v9059 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9060 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9061 = stablehlo.multiply %v9059, %b14pWv : tensor<160x576x1x1xf32>
    %v9062 = stablehlo.multiply %v9058, %v9058 : tensor<160x576x1x1xf32>
    %v9063 = stablehlo.multiply %v9060, %v9062 : tensor<160x576x1x1xf32>
    %v9064 = stablehlo.add %v9061, %v9063 : tensor<160x576x1x1xf32>
    %v9065 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9066 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9067 = stablehlo.multiply %v9065, %b14pWv : tensor<160x576x1x1xf32>
    %v9068 = stablehlo.multiply %v9058, %v9058 : tensor<160x576x1x1xf32>
    %v9069 = stablehlo.multiply %v9066, %v9068 : tensor<160x576x1x1xf32>
    %v9070 = stablehlo.add %v9067, %v9069 : tensor<160x576x1x1xf32>
    %v9071 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9072 = stablehlo.add %v9070, %v9071 : tensor<160x576x1x1xf32>
    %v9073 = stablehlo.sqrt %v9072 : tensor<160x576x1x1xf32>
    %v9074 = stablehlo.divide %v9058, %v9073 : tensor<160x576x1x1xf32>
    %v9075 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9076 = stablehlo.multiply %v9075, %b14pWm : tensor<160x576x1x1xf32>
    %v9077 = stablehlo.add %v9076, %v9074 : tensor<160x576x1x1xf32>
    %v9078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9079 = stablehlo.multiply %v9078, %v9077 : tensor<160x576x1x1xf32>
    %v9080 = stablehlo.subtract %b14pW, %v9079 : tensor<160x576x1x1xf32>
    %v9081 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9082 = stablehlo.multiply %v9081, %b14pg : tensor<160xf32>
    %v9083 = stablehlo.add %v9082, %v2472 : tensor<160xf32>
    %v9084 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9085 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9086 = stablehlo.multiply %v9084, %b14pgv : tensor<160xf32>
    %v9087 = stablehlo.multiply %v9083, %v9083 : tensor<160xf32>
    %v9088 = stablehlo.multiply %v9085, %v9087 : tensor<160xf32>
    %v9089 = stablehlo.add %v9086, %v9088 : tensor<160xf32>
    %v9090 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9091 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9092 = stablehlo.multiply %v9090, %b14pgv : tensor<160xf32>
    %v9093 = stablehlo.multiply %v9083, %v9083 : tensor<160xf32>
    %v9094 = stablehlo.multiply %v9091, %v9093 : tensor<160xf32>
    %v9095 = stablehlo.add %v9092, %v9094 : tensor<160xf32>
    %v9096 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9097 = stablehlo.add %v9095, %v9096 : tensor<160xf32>
    %v9098 = stablehlo.sqrt %v9097 : tensor<160xf32>
    %v9099 = stablehlo.divide %v9083, %v9098 : tensor<160xf32>
    %v9100 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9101 = stablehlo.multiply %v9100, %b14pgm : tensor<160xf32>
    %v9102 = stablehlo.add %v9101, %v9099 : tensor<160xf32>
    %v9103 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9104 = stablehlo.multiply %v9103, %v9102 : tensor<160xf32>
    %v9105 = stablehlo.subtract %b14pg, %v9104 : tensor<160xf32>
    %v9106 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9107 = stablehlo.multiply %v9106, %b14pbt : tensor<160xf32>
    %v9108 = stablehlo.add %v9107, %v2475 : tensor<160xf32>
    %v9109 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9110 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9111 = stablehlo.multiply %v9109, %b14pbtv : tensor<160xf32>
    %v9112 = stablehlo.multiply %v9108, %v9108 : tensor<160xf32>
    %v9113 = stablehlo.multiply %v9110, %v9112 : tensor<160xf32>
    %v9114 = stablehlo.add %v9111, %v9113 : tensor<160xf32>
    %v9115 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9116 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9117 = stablehlo.multiply %v9115, %b14pbtv : tensor<160xf32>
    %v9118 = stablehlo.multiply %v9108, %v9108 : tensor<160xf32>
    %v9119 = stablehlo.multiply %v9116, %v9118 : tensor<160xf32>
    %v9120 = stablehlo.add %v9117, %v9119 : tensor<160xf32>
    %v9121 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9122 = stablehlo.add %v9120, %v9121 : tensor<160xf32>
    %v9123 = stablehlo.sqrt %v9122 : tensor<160xf32>
    %v9124 = stablehlo.divide %v9108, %v9123 : tensor<160xf32>
    %v9125 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9126 = stablehlo.multiply %v9125, %b14pbtm : tensor<160xf32>
    %v9127 = stablehlo.add %v9126, %v9124 : tensor<160xf32>
    %v9128 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9129 = stablehlo.multiply %v9128, %v9127 : tensor<160xf32>
    %v9130 = stablehlo.subtract %b14pbt, %v9129 : tensor<160xf32>
    %v9131 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9132 = stablehlo.multiply %v9131, %b15eW : tensor<960x160x1x1xf32>
    %v9133 = stablehlo.add %v9132, %v2195 : tensor<960x160x1x1xf32>
    %v9134 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9135 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9136 = stablehlo.multiply %v9134, %b15eWv : tensor<960x160x1x1xf32>
    %v9137 = stablehlo.multiply %v9133, %v9133 : tensor<960x160x1x1xf32>
    %v9138 = stablehlo.multiply %v9135, %v9137 : tensor<960x160x1x1xf32>
    %v9139 = stablehlo.add %v9136, %v9138 : tensor<960x160x1x1xf32>
    %v9140 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9141 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9142 = stablehlo.multiply %v9140, %b15eWv : tensor<960x160x1x1xf32>
    %v9143 = stablehlo.multiply %v9133, %v9133 : tensor<960x160x1x1xf32>
    %v9144 = stablehlo.multiply %v9141, %v9143 : tensor<960x160x1x1xf32>
    %v9145 = stablehlo.add %v9142, %v9144 : tensor<960x160x1x1xf32>
    %v9146 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9147 = stablehlo.add %v9145, %v9146 : tensor<960x160x1x1xf32>
    %v9148 = stablehlo.sqrt %v9147 : tensor<960x160x1x1xf32>
    %v9149 = stablehlo.divide %v9133, %v9148 : tensor<960x160x1x1xf32>
    %v9150 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9151 = stablehlo.multiply %v9150, %b15eWm : tensor<960x160x1x1xf32>
    %v9152 = stablehlo.add %v9151, %v9149 : tensor<960x160x1x1xf32>
    %v9153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9154 = stablehlo.multiply %v9153, %v9152 : tensor<960x160x1x1xf32>
    %v9155 = stablehlo.subtract %b15eW, %v9154 : tensor<960x160x1x1xf32>
    %v9156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9157 = stablehlo.multiply %v9156, %b15eg : tensor<960xf32>
    %v9158 = stablehlo.add %v9157, %v2213 : tensor<960xf32>
    %v9159 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9160 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9161 = stablehlo.multiply %v9159, %b15egv : tensor<960xf32>
    %v9162 = stablehlo.multiply %v9158, %v9158 : tensor<960xf32>
    %v9163 = stablehlo.multiply %v9160, %v9162 : tensor<960xf32>
    %v9164 = stablehlo.add %v9161, %v9163 : tensor<960xf32>
    %v9165 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9166 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9167 = stablehlo.multiply %v9165, %b15egv : tensor<960xf32>
    %v9168 = stablehlo.multiply %v9158, %v9158 : tensor<960xf32>
    %v9169 = stablehlo.multiply %v9166, %v9168 : tensor<960xf32>
    %v9170 = stablehlo.add %v9167, %v9169 : tensor<960xf32>
    %v9171 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9172 = stablehlo.add %v9170, %v9171 : tensor<960xf32>
    %v9173 = stablehlo.sqrt %v9172 : tensor<960xf32>
    %v9174 = stablehlo.divide %v9158, %v9173 : tensor<960xf32>
    %v9175 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9176 = stablehlo.multiply %v9175, %b15egm : tensor<960xf32>
    %v9177 = stablehlo.add %v9176, %v9174 : tensor<960xf32>
    %v9178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9179 = stablehlo.multiply %v9178, %v9177 : tensor<960xf32>
    %v9180 = stablehlo.subtract %b15eg, %v9179 : tensor<960xf32>
    %v9181 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9182 = stablehlo.multiply %v9181, %b15ebt : tensor<960xf32>
    %v9183 = stablehlo.add %v9182, %v2216 : tensor<960xf32>
    %v9184 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9185 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9186 = stablehlo.multiply %v9184, %b15ebtv : tensor<960xf32>
    %v9187 = stablehlo.multiply %v9183, %v9183 : tensor<960xf32>
    %v9188 = stablehlo.multiply %v9185, %v9187 : tensor<960xf32>
    %v9189 = stablehlo.add %v9186, %v9188 : tensor<960xf32>
    %v9190 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9191 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9192 = stablehlo.multiply %v9190, %b15ebtv : tensor<960xf32>
    %v9193 = stablehlo.multiply %v9183, %v9183 : tensor<960xf32>
    %v9194 = stablehlo.multiply %v9191, %v9193 : tensor<960xf32>
    %v9195 = stablehlo.add %v9192, %v9194 : tensor<960xf32>
    %v9196 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9197 = stablehlo.add %v9195, %v9196 : tensor<960xf32>
    %v9198 = stablehlo.sqrt %v9197 : tensor<960xf32>
    %v9199 = stablehlo.divide %v9183, %v9198 : tensor<960xf32>
    %v9200 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9201 = stablehlo.multiply %v9200, %b15ebtm : tensor<960xf32>
    %v9202 = stablehlo.add %v9201, %v9199 : tensor<960xf32>
    %v9203 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9204 = stablehlo.multiply %v9203, %v9202 : tensor<960xf32>
    %v9205 = stablehlo.subtract %b15ebt, %v9204 : tensor<960xf32>
    %v9206 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9207 = stablehlo.multiply %v9206, %b15dW : tensor<960x1x3x3xf32>
    %v9208 = stablehlo.add %v9207, %v2222 : tensor<960x1x3x3xf32>
    %v9209 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9210 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9211 = stablehlo.multiply %v9209, %b15dWv : tensor<960x1x3x3xf32>
    %v9212 = stablehlo.multiply %v9208, %v9208 : tensor<960x1x3x3xf32>
    %v9213 = stablehlo.multiply %v9210, %v9212 : tensor<960x1x3x3xf32>
    %v9214 = stablehlo.add %v9211, %v9213 : tensor<960x1x3x3xf32>
    %v9215 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9216 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9217 = stablehlo.multiply %v9215, %b15dWv : tensor<960x1x3x3xf32>
    %v9218 = stablehlo.multiply %v9208, %v9208 : tensor<960x1x3x3xf32>
    %v9219 = stablehlo.multiply %v9216, %v9218 : tensor<960x1x3x3xf32>
    %v9220 = stablehlo.add %v9217, %v9219 : tensor<960x1x3x3xf32>
    %v9221 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9222 = stablehlo.add %v9220, %v9221 : tensor<960x1x3x3xf32>
    %v9223 = stablehlo.sqrt %v9222 : tensor<960x1x3x3xf32>
    %v9224 = stablehlo.divide %v9208, %v9223 : tensor<960x1x3x3xf32>
    %v9225 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9226 = stablehlo.multiply %v9225, %b15dWm : tensor<960x1x3x3xf32>
    %v9227 = stablehlo.add %v9226, %v9224 : tensor<960x1x3x3xf32>
    %v9228 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9229 = stablehlo.multiply %v9228, %v9227 : tensor<960x1x3x3xf32>
    %v9230 = stablehlo.subtract %b15dW, %v9229 : tensor<960x1x3x3xf32>
    %v9231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9232 = stablehlo.multiply %v9231, %b15dg : tensor<960xf32>
    %v9233 = stablehlo.add %v9232, %v2240 : tensor<960xf32>
    %v9234 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9235 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9236 = stablehlo.multiply %v9234, %b15dgv : tensor<960xf32>
    %v9237 = stablehlo.multiply %v9233, %v9233 : tensor<960xf32>
    %v9238 = stablehlo.multiply %v9235, %v9237 : tensor<960xf32>
    %v9239 = stablehlo.add %v9236, %v9238 : tensor<960xf32>
    %v9240 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9241 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9242 = stablehlo.multiply %v9240, %b15dgv : tensor<960xf32>
    %v9243 = stablehlo.multiply %v9233, %v9233 : tensor<960xf32>
    %v9244 = stablehlo.multiply %v9241, %v9243 : tensor<960xf32>
    %v9245 = stablehlo.add %v9242, %v9244 : tensor<960xf32>
    %v9246 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9247 = stablehlo.add %v9245, %v9246 : tensor<960xf32>
    %v9248 = stablehlo.sqrt %v9247 : tensor<960xf32>
    %v9249 = stablehlo.divide %v9233, %v9248 : tensor<960xf32>
    %v9250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9251 = stablehlo.multiply %v9250, %b15dgm : tensor<960xf32>
    %v9252 = stablehlo.add %v9251, %v9249 : tensor<960xf32>
    %v9253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9254 = stablehlo.multiply %v9253, %v9252 : tensor<960xf32>
    %v9255 = stablehlo.subtract %b15dg, %v9254 : tensor<960xf32>
    %v9256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9257 = stablehlo.multiply %v9256, %b15dbt : tensor<960xf32>
    %v9258 = stablehlo.add %v9257, %v2243 : tensor<960xf32>
    %v9259 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9260 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9261 = stablehlo.multiply %v9259, %b15dbtv : tensor<960xf32>
    %v9262 = stablehlo.multiply %v9258, %v9258 : tensor<960xf32>
    %v9263 = stablehlo.multiply %v9260, %v9262 : tensor<960xf32>
    %v9264 = stablehlo.add %v9261, %v9263 : tensor<960xf32>
    %v9265 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9266 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9267 = stablehlo.multiply %v9265, %b15dbtv : tensor<960xf32>
    %v9268 = stablehlo.multiply %v9258, %v9258 : tensor<960xf32>
    %v9269 = stablehlo.multiply %v9266, %v9268 : tensor<960xf32>
    %v9270 = stablehlo.add %v9267, %v9269 : tensor<960xf32>
    %v9271 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9272 = stablehlo.add %v9270, %v9271 : tensor<960xf32>
    %v9273 = stablehlo.sqrt %v9272 : tensor<960xf32>
    %v9274 = stablehlo.divide %v9258, %v9273 : tensor<960xf32>
    %v9275 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9276 = stablehlo.multiply %v9275, %b15dbtm : tensor<960xf32>
    %v9277 = stablehlo.add %v9276, %v9274 : tensor<960xf32>
    %v9278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9279 = stablehlo.multiply %v9278, %v9277 : tensor<960xf32>
    %v9280 = stablehlo.subtract %b15dbt, %v9279 : tensor<960xf32>
    %v9281 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9282 = stablehlo.multiply %v9281, %b15pW : tensor<160x960x1x1xf32>
    %v9283 = stablehlo.add %v9282, %v2249 : tensor<160x960x1x1xf32>
    %v9284 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9285 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9286 = stablehlo.multiply %v9284, %b15pWv : tensor<160x960x1x1xf32>
    %v9287 = stablehlo.multiply %v9283, %v9283 : tensor<160x960x1x1xf32>
    %v9288 = stablehlo.multiply %v9285, %v9287 : tensor<160x960x1x1xf32>
    %v9289 = stablehlo.add %v9286, %v9288 : tensor<160x960x1x1xf32>
    %v9290 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9291 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9292 = stablehlo.multiply %v9290, %b15pWv : tensor<160x960x1x1xf32>
    %v9293 = stablehlo.multiply %v9283, %v9283 : tensor<160x960x1x1xf32>
    %v9294 = stablehlo.multiply %v9291, %v9293 : tensor<160x960x1x1xf32>
    %v9295 = stablehlo.add %v9292, %v9294 : tensor<160x960x1x1xf32>
    %v9296 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9297 = stablehlo.add %v9295, %v9296 : tensor<160x960x1x1xf32>
    %v9298 = stablehlo.sqrt %v9297 : tensor<160x960x1x1xf32>
    %v9299 = stablehlo.divide %v9283, %v9298 : tensor<160x960x1x1xf32>
    %v9300 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9301 = stablehlo.multiply %v9300, %b15pWm : tensor<160x960x1x1xf32>
    %v9302 = stablehlo.add %v9301, %v9299 : tensor<160x960x1x1xf32>
    %v9303 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9304 = stablehlo.multiply %v9303, %v9302 : tensor<160x960x1x1xf32>
    %v9305 = stablehlo.subtract %b15pW, %v9304 : tensor<160x960x1x1xf32>
    %v9306 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9307 = stablehlo.multiply %v9306, %b15pg : tensor<160xf32>
    %v9308 = stablehlo.add %v9307, %v2267 : tensor<160xf32>
    %v9309 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9310 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9311 = stablehlo.multiply %v9309, %b15pgv : tensor<160xf32>
    %v9312 = stablehlo.multiply %v9308, %v9308 : tensor<160xf32>
    %v9313 = stablehlo.multiply %v9310, %v9312 : tensor<160xf32>
    %v9314 = stablehlo.add %v9311, %v9313 : tensor<160xf32>
    %v9315 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9316 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9317 = stablehlo.multiply %v9315, %b15pgv : tensor<160xf32>
    %v9318 = stablehlo.multiply %v9308, %v9308 : tensor<160xf32>
    %v9319 = stablehlo.multiply %v9316, %v9318 : tensor<160xf32>
    %v9320 = stablehlo.add %v9317, %v9319 : tensor<160xf32>
    %v9321 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9322 = stablehlo.add %v9320, %v9321 : tensor<160xf32>
    %v9323 = stablehlo.sqrt %v9322 : tensor<160xf32>
    %v9324 = stablehlo.divide %v9308, %v9323 : tensor<160xf32>
    %v9325 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9326 = stablehlo.multiply %v9325, %b15pgm : tensor<160xf32>
    %v9327 = stablehlo.add %v9326, %v9324 : tensor<160xf32>
    %v9328 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9329 = stablehlo.multiply %v9328, %v9327 : tensor<160xf32>
    %v9330 = stablehlo.subtract %b15pg, %v9329 : tensor<160xf32>
    %v9331 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9332 = stablehlo.multiply %v9331, %b15pbt : tensor<160xf32>
    %v9333 = stablehlo.add %v9332, %v2270 : tensor<160xf32>
    %v9334 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9335 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9336 = stablehlo.multiply %v9334, %b15pbtv : tensor<160xf32>
    %v9337 = stablehlo.multiply %v9333, %v9333 : tensor<160xf32>
    %v9338 = stablehlo.multiply %v9335, %v9337 : tensor<160xf32>
    %v9339 = stablehlo.add %v9336, %v9338 : tensor<160xf32>
    %v9340 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9341 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9342 = stablehlo.multiply %v9340, %b15pbtv : tensor<160xf32>
    %v9343 = stablehlo.multiply %v9333, %v9333 : tensor<160xf32>
    %v9344 = stablehlo.multiply %v9341, %v9343 : tensor<160xf32>
    %v9345 = stablehlo.add %v9342, %v9344 : tensor<160xf32>
    %v9346 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9347 = stablehlo.add %v9345, %v9346 : tensor<160xf32>
    %v9348 = stablehlo.sqrt %v9347 : tensor<160xf32>
    %v9349 = stablehlo.divide %v9333, %v9348 : tensor<160xf32>
    %v9350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9351 = stablehlo.multiply %v9350, %b15pbtm : tensor<160xf32>
    %v9352 = stablehlo.add %v9351, %v9349 : tensor<160xf32>
    %v9353 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9354 = stablehlo.multiply %v9353, %v9352 : tensor<160xf32>
    %v9355 = stablehlo.subtract %b15pbt, %v9354 : tensor<160xf32>
    %v9356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9357 = stablehlo.multiply %v9356, %b16eW : tensor<960x160x1x1xf32>
    %v9358 = stablehlo.add %v9357, %v1988 : tensor<960x160x1x1xf32>
    %v9359 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9360 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9361 = stablehlo.multiply %v9359, %b16eWv : tensor<960x160x1x1xf32>
    %v9362 = stablehlo.multiply %v9358, %v9358 : tensor<960x160x1x1xf32>
    %v9363 = stablehlo.multiply %v9360, %v9362 : tensor<960x160x1x1xf32>
    %v9364 = stablehlo.add %v9361, %v9363 : tensor<960x160x1x1xf32>
    %v9365 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9366 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9367 = stablehlo.multiply %v9365, %b16eWv : tensor<960x160x1x1xf32>
    %v9368 = stablehlo.multiply %v9358, %v9358 : tensor<960x160x1x1xf32>
    %v9369 = stablehlo.multiply %v9366, %v9368 : tensor<960x160x1x1xf32>
    %v9370 = stablehlo.add %v9367, %v9369 : tensor<960x160x1x1xf32>
    %v9371 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9372 = stablehlo.add %v9370, %v9371 : tensor<960x160x1x1xf32>
    %v9373 = stablehlo.sqrt %v9372 : tensor<960x160x1x1xf32>
    %v9374 = stablehlo.divide %v9358, %v9373 : tensor<960x160x1x1xf32>
    %v9375 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9376 = stablehlo.multiply %v9375, %b16eWm : tensor<960x160x1x1xf32>
    %v9377 = stablehlo.add %v9376, %v9374 : tensor<960x160x1x1xf32>
    %v9378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9379 = stablehlo.multiply %v9378, %v9377 : tensor<960x160x1x1xf32>
    %v9380 = stablehlo.subtract %b16eW, %v9379 : tensor<960x160x1x1xf32>
    %v9381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9382 = stablehlo.multiply %v9381, %b16eg : tensor<960xf32>
    %v9383 = stablehlo.add %v9382, %v2006 : tensor<960xf32>
    %v9384 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9385 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9386 = stablehlo.multiply %v9384, %b16egv : tensor<960xf32>
    %v9387 = stablehlo.multiply %v9383, %v9383 : tensor<960xf32>
    %v9388 = stablehlo.multiply %v9385, %v9387 : tensor<960xf32>
    %v9389 = stablehlo.add %v9386, %v9388 : tensor<960xf32>
    %v9390 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9391 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9392 = stablehlo.multiply %v9390, %b16egv : tensor<960xf32>
    %v9393 = stablehlo.multiply %v9383, %v9383 : tensor<960xf32>
    %v9394 = stablehlo.multiply %v9391, %v9393 : tensor<960xf32>
    %v9395 = stablehlo.add %v9392, %v9394 : tensor<960xf32>
    %v9396 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9397 = stablehlo.add %v9395, %v9396 : tensor<960xf32>
    %v9398 = stablehlo.sqrt %v9397 : tensor<960xf32>
    %v9399 = stablehlo.divide %v9383, %v9398 : tensor<960xf32>
    %v9400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9401 = stablehlo.multiply %v9400, %b16egm : tensor<960xf32>
    %v9402 = stablehlo.add %v9401, %v9399 : tensor<960xf32>
    %v9403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9404 = stablehlo.multiply %v9403, %v9402 : tensor<960xf32>
    %v9405 = stablehlo.subtract %b16eg, %v9404 : tensor<960xf32>
    %v9406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9407 = stablehlo.multiply %v9406, %b16ebt : tensor<960xf32>
    %v9408 = stablehlo.add %v9407, %v2009 : tensor<960xf32>
    %v9409 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9410 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9411 = stablehlo.multiply %v9409, %b16ebtv : tensor<960xf32>
    %v9412 = stablehlo.multiply %v9408, %v9408 : tensor<960xf32>
    %v9413 = stablehlo.multiply %v9410, %v9412 : tensor<960xf32>
    %v9414 = stablehlo.add %v9411, %v9413 : tensor<960xf32>
    %v9415 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9416 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9417 = stablehlo.multiply %v9415, %b16ebtv : tensor<960xf32>
    %v9418 = stablehlo.multiply %v9408, %v9408 : tensor<960xf32>
    %v9419 = stablehlo.multiply %v9416, %v9418 : tensor<960xf32>
    %v9420 = stablehlo.add %v9417, %v9419 : tensor<960xf32>
    %v9421 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9422 = stablehlo.add %v9420, %v9421 : tensor<960xf32>
    %v9423 = stablehlo.sqrt %v9422 : tensor<960xf32>
    %v9424 = stablehlo.divide %v9408, %v9423 : tensor<960xf32>
    %v9425 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9426 = stablehlo.multiply %v9425, %b16ebtm : tensor<960xf32>
    %v9427 = stablehlo.add %v9426, %v9424 : tensor<960xf32>
    %v9428 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9429 = stablehlo.multiply %v9428, %v9427 : tensor<960xf32>
    %v9430 = stablehlo.subtract %b16ebt, %v9429 : tensor<960xf32>
    %v9431 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9432 = stablehlo.multiply %v9431, %b16dW : tensor<960x1x3x3xf32>
    %v9433 = stablehlo.add %v9432, %v2015 : tensor<960x1x3x3xf32>
    %v9434 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9435 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9436 = stablehlo.multiply %v9434, %b16dWv : tensor<960x1x3x3xf32>
    %v9437 = stablehlo.multiply %v9433, %v9433 : tensor<960x1x3x3xf32>
    %v9438 = stablehlo.multiply %v9435, %v9437 : tensor<960x1x3x3xf32>
    %v9439 = stablehlo.add %v9436, %v9438 : tensor<960x1x3x3xf32>
    %v9440 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9441 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9442 = stablehlo.multiply %v9440, %b16dWv : tensor<960x1x3x3xf32>
    %v9443 = stablehlo.multiply %v9433, %v9433 : tensor<960x1x3x3xf32>
    %v9444 = stablehlo.multiply %v9441, %v9443 : tensor<960x1x3x3xf32>
    %v9445 = stablehlo.add %v9442, %v9444 : tensor<960x1x3x3xf32>
    %v9446 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9447 = stablehlo.add %v9445, %v9446 : tensor<960x1x3x3xf32>
    %v9448 = stablehlo.sqrt %v9447 : tensor<960x1x3x3xf32>
    %v9449 = stablehlo.divide %v9433, %v9448 : tensor<960x1x3x3xf32>
    %v9450 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9451 = stablehlo.multiply %v9450, %b16dWm : tensor<960x1x3x3xf32>
    %v9452 = stablehlo.add %v9451, %v9449 : tensor<960x1x3x3xf32>
    %v9453 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9454 = stablehlo.multiply %v9453, %v9452 : tensor<960x1x3x3xf32>
    %v9455 = stablehlo.subtract %b16dW, %v9454 : tensor<960x1x3x3xf32>
    %v9456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9457 = stablehlo.multiply %v9456, %b16dg : tensor<960xf32>
    %v9458 = stablehlo.add %v9457, %v2033 : tensor<960xf32>
    %v9459 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9460 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9461 = stablehlo.multiply %v9459, %b16dgv : tensor<960xf32>
    %v9462 = stablehlo.multiply %v9458, %v9458 : tensor<960xf32>
    %v9463 = stablehlo.multiply %v9460, %v9462 : tensor<960xf32>
    %v9464 = stablehlo.add %v9461, %v9463 : tensor<960xf32>
    %v9465 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9466 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9467 = stablehlo.multiply %v9465, %b16dgv : tensor<960xf32>
    %v9468 = stablehlo.multiply %v9458, %v9458 : tensor<960xf32>
    %v9469 = stablehlo.multiply %v9466, %v9468 : tensor<960xf32>
    %v9470 = stablehlo.add %v9467, %v9469 : tensor<960xf32>
    %v9471 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9472 = stablehlo.add %v9470, %v9471 : tensor<960xf32>
    %v9473 = stablehlo.sqrt %v9472 : tensor<960xf32>
    %v9474 = stablehlo.divide %v9458, %v9473 : tensor<960xf32>
    %v9475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9476 = stablehlo.multiply %v9475, %b16dgm : tensor<960xf32>
    %v9477 = stablehlo.add %v9476, %v9474 : tensor<960xf32>
    %v9478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9479 = stablehlo.multiply %v9478, %v9477 : tensor<960xf32>
    %v9480 = stablehlo.subtract %b16dg, %v9479 : tensor<960xf32>
    %v9481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9482 = stablehlo.multiply %v9481, %b16dbt : tensor<960xf32>
    %v9483 = stablehlo.add %v9482, %v2036 : tensor<960xf32>
    %v9484 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9485 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9486 = stablehlo.multiply %v9484, %b16dbtv : tensor<960xf32>
    %v9487 = stablehlo.multiply %v9483, %v9483 : tensor<960xf32>
    %v9488 = stablehlo.multiply %v9485, %v9487 : tensor<960xf32>
    %v9489 = stablehlo.add %v9486, %v9488 : tensor<960xf32>
    %v9490 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9491 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9492 = stablehlo.multiply %v9490, %b16dbtv : tensor<960xf32>
    %v9493 = stablehlo.multiply %v9483, %v9483 : tensor<960xf32>
    %v9494 = stablehlo.multiply %v9491, %v9493 : tensor<960xf32>
    %v9495 = stablehlo.add %v9492, %v9494 : tensor<960xf32>
    %v9496 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9497 = stablehlo.add %v9495, %v9496 : tensor<960xf32>
    %v9498 = stablehlo.sqrt %v9497 : tensor<960xf32>
    %v9499 = stablehlo.divide %v9483, %v9498 : tensor<960xf32>
    %v9500 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9501 = stablehlo.multiply %v9500, %b16dbtm : tensor<960xf32>
    %v9502 = stablehlo.add %v9501, %v9499 : tensor<960xf32>
    %v9503 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9504 = stablehlo.multiply %v9503, %v9502 : tensor<960xf32>
    %v9505 = stablehlo.subtract %b16dbt, %v9504 : tensor<960xf32>
    %v9506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9507 = stablehlo.multiply %v9506, %b16pW : tensor<160x960x1x1xf32>
    %v9508 = stablehlo.add %v9507, %v2042 : tensor<160x960x1x1xf32>
    %v9509 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9510 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9511 = stablehlo.multiply %v9509, %b16pWv : tensor<160x960x1x1xf32>
    %v9512 = stablehlo.multiply %v9508, %v9508 : tensor<160x960x1x1xf32>
    %v9513 = stablehlo.multiply %v9510, %v9512 : tensor<160x960x1x1xf32>
    %v9514 = stablehlo.add %v9511, %v9513 : tensor<160x960x1x1xf32>
    %v9515 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9516 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9517 = stablehlo.multiply %v9515, %b16pWv : tensor<160x960x1x1xf32>
    %v9518 = stablehlo.multiply %v9508, %v9508 : tensor<160x960x1x1xf32>
    %v9519 = stablehlo.multiply %v9516, %v9518 : tensor<160x960x1x1xf32>
    %v9520 = stablehlo.add %v9517, %v9519 : tensor<160x960x1x1xf32>
    %v9521 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9522 = stablehlo.add %v9520, %v9521 : tensor<160x960x1x1xf32>
    %v9523 = stablehlo.sqrt %v9522 : tensor<160x960x1x1xf32>
    %v9524 = stablehlo.divide %v9508, %v9523 : tensor<160x960x1x1xf32>
    %v9525 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9526 = stablehlo.multiply %v9525, %b16pWm : tensor<160x960x1x1xf32>
    %v9527 = stablehlo.add %v9526, %v9524 : tensor<160x960x1x1xf32>
    %v9528 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9529 = stablehlo.multiply %v9528, %v9527 : tensor<160x960x1x1xf32>
    %v9530 = stablehlo.subtract %b16pW, %v9529 : tensor<160x960x1x1xf32>
    %v9531 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9532 = stablehlo.multiply %v9531, %b16pg : tensor<160xf32>
    %v9533 = stablehlo.add %v9532, %v2060 : tensor<160xf32>
    %v9534 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9535 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9536 = stablehlo.multiply %v9534, %b16pgv : tensor<160xf32>
    %v9537 = stablehlo.multiply %v9533, %v9533 : tensor<160xf32>
    %v9538 = stablehlo.multiply %v9535, %v9537 : tensor<160xf32>
    %v9539 = stablehlo.add %v9536, %v9538 : tensor<160xf32>
    %v9540 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9541 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9542 = stablehlo.multiply %v9540, %b16pgv : tensor<160xf32>
    %v9543 = stablehlo.multiply %v9533, %v9533 : tensor<160xf32>
    %v9544 = stablehlo.multiply %v9541, %v9543 : tensor<160xf32>
    %v9545 = stablehlo.add %v9542, %v9544 : tensor<160xf32>
    %v9546 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9547 = stablehlo.add %v9545, %v9546 : tensor<160xf32>
    %v9548 = stablehlo.sqrt %v9547 : tensor<160xf32>
    %v9549 = stablehlo.divide %v9533, %v9548 : tensor<160xf32>
    %v9550 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9551 = stablehlo.multiply %v9550, %b16pgm : tensor<160xf32>
    %v9552 = stablehlo.add %v9551, %v9549 : tensor<160xf32>
    %v9553 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9554 = stablehlo.multiply %v9553, %v9552 : tensor<160xf32>
    %v9555 = stablehlo.subtract %b16pg, %v9554 : tensor<160xf32>
    %v9556 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9557 = stablehlo.multiply %v9556, %b16pbt : tensor<160xf32>
    %v9558 = stablehlo.add %v9557, %v2063 : tensor<160xf32>
    %v9559 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9560 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9561 = stablehlo.multiply %v9559, %b16pbtv : tensor<160xf32>
    %v9562 = stablehlo.multiply %v9558, %v9558 : tensor<160xf32>
    %v9563 = stablehlo.multiply %v9560, %v9562 : tensor<160xf32>
    %v9564 = stablehlo.add %v9561, %v9563 : tensor<160xf32>
    %v9565 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9566 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9567 = stablehlo.multiply %v9565, %b16pbtv : tensor<160xf32>
    %v9568 = stablehlo.multiply %v9558, %v9558 : tensor<160xf32>
    %v9569 = stablehlo.multiply %v9566, %v9568 : tensor<160xf32>
    %v9570 = stablehlo.add %v9567, %v9569 : tensor<160xf32>
    %v9571 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9572 = stablehlo.add %v9570, %v9571 : tensor<160xf32>
    %v9573 = stablehlo.sqrt %v9572 : tensor<160xf32>
    %v9574 = stablehlo.divide %v9558, %v9573 : tensor<160xf32>
    %v9575 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9576 = stablehlo.multiply %v9575, %b16pbtm : tensor<160xf32>
    %v9577 = stablehlo.add %v9576, %v9574 : tensor<160xf32>
    %v9578 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9579 = stablehlo.multiply %v9578, %v9577 : tensor<160xf32>
    %v9580 = stablehlo.subtract %b16pbt, %v9579 : tensor<160xf32>
    %v9581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9582 = stablehlo.multiply %v9581, %b17eW : tensor<960x160x1x1xf32>
    %v9583 = stablehlo.add %v9582, %v1781 : tensor<960x160x1x1xf32>
    %v9584 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9585 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9586 = stablehlo.multiply %v9584, %b17eWv : tensor<960x160x1x1xf32>
    %v9587 = stablehlo.multiply %v9583, %v9583 : tensor<960x160x1x1xf32>
    %v9588 = stablehlo.multiply %v9585, %v9587 : tensor<960x160x1x1xf32>
    %v9589 = stablehlo.add %v9586, %v9588 : tensor<960x160x1x1xf32>
    %v9590 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9591 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9592 = stablehlo.multiply %v9590, %b17eWv : tensor<960x160x1x1xf32>
    %v9593 = stablehlo.multiply %v9583, %v9583 : tensor<960x160x1x1xf32>
    %v9594 = stablehlo.multiply %v9591, %v9593 : tensor<960x160x1x1xf32>
    %v9595 = stablehlo.add %v9592, %v9594 : tensor<960x160x1x1xf32>
    %v9596 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9597 = stablehlo.add %v9595, %v9596 : tensor<960x160x1x1xf32>
    %v9598 = stablehlo.sqrt %v9597 : tensor<960x160x1x1xf32>
    %v9599 = stablehlo.divide %v9583, %v9598 : tensor<960x160x1x1xf32>
    %v9600 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9601 = stablehlo.multiply %v9600, %b17eWm : tensor<960x160x1x1xf32>
    %v9602 = stablehlo.add %v9601, %v9599 : tensor<960x160x1x1xf32>
    %v9603 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9604 = stablehlo.multiply %v9603, %v9602 : tensor<960x160x1x1xf32>
    %v9605 = stablehlo.subtract %b17eW, %v9604 : tensor<960x160x1x1xf32>
    %v9606 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9607 = stablehlo.multiply %v9606, %b17eg : tensor<960xf32>
    %v9608 = stablehlo.add %v9607, %v1799 : tensor<960xf32>
    %v9609 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9610 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9611 = stablehlo.multiply %v9609, %b17egv : tensor<960xf32>
    %v9612 = stablehlo.multiply %v9608, %v9608 : tensor<960xf32>
    %v9613 = stablehlo.multiply %v9610, %v9612 : tensor<960xf32>
    %v9614 = stablehlo.add %v9611, %v9613 : tensor<960xf32>
    %v9615 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9616 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9617 = stablehlo.multiply %v9615, %b17egv : tensor<960xf32>
    %v9618 = stablehlo.multiply %v9608, %v9608 : tensor<960xf32>
    %v9619 = stablehlo.multiply %v9616, %v9618 : tensor<960xf32>
    %v9620 = stablehlo.add %v9617, %v9619 : tensor<960xf32>
    %v9621 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9622 = stablehlo.add %v9620, %v9621 : tensor<960xf32>
    %v9623 = stablehlo.sqrt %v9622 : tensor<960xf32>
    %v9624 = stablehlo.divide %v9608, %v9623 : tensor<960xf32>
    %v9625 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9626 = stablehlo.multiply %v9625, %b17egm : tensor<960xf32>
    %v9627 = stablehlo.add %v9626, %v9624 : tensor<960xf32>
    %v9628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9629 = stablehlo.multiply %v9628, %v9627 : tensor<960xf32>
    %v9630 = stablehlo.subtract %b17eg, %v9629 : tensor<960xf32>
    %v9631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9632 = stablehlo.multiply %v9631, %b17ebt : tensor<960xf32>
    %v9633 = stablehlo.add %v9632, %v1802 : tensor<960xf32>
    %v9634 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9635 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9636 = stablehlo.multiply %v9634, %b17ebtv : tensor<960xf32>
    %v9637 = stablehlo.multiply %v9633, %v9633 : tensor<960xf32>
    %v9638 = stablehlo.multiply %v9635, %v9637 : tensor<960xf32>
    %v9639 = stablehlo.add %v9636, %v9638 : tensor<960xf32>
    %v9640 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9641 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9642 = stablehlo.multiply %v9640, %b17ebtv : tensor<960xf32>
    %v9643 = stablehlo.multiply %v9633, %v9633 : tensor<960xf32>
    %v9644 = stablehlo.multiply %v9641, %v9643 : tensor<960xf32>
    %v9645 = stablehlo.add %v9642, %v9644 : tensor<960xf32>
    %v9646 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9647 = stablehlo.add %v9645, %v9646 : tensor<960xf32>
    %v9648 = stablehlo.sqrt %v9647 : tensor<960xf32>
    %v9649 = stablehlo.divide %v9633, %v9648 : tensor<960xf32>
    %v9650 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9651 = stablehlo.multiply %v9650, %b17ebtm : tensor<960xf32>
    %v9652 = stablehlo.add %v9651, %v9649 : tensor<960xf32>
    %v9653 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9654 = stablehlo.multiply %v9653, %v9652 : tensor<960xf32>
    %v9655 = stablehlo.subtract %b17ebt, %v9654 : tensor<960xf32>
    %v9656 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9657 = stablehlo.multiply %v9656, %b17dW : tensor<960x1x3x3xf32>
    %v9658 = stablehlo.add %v9657, %v1808 : tensor<960x1x3x3xf32>
    %v9659 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9660 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9661 = stablehlo.multiply %v9659, %b17dWv : tensor<960x1x3x3xf32>
    %v9662 = stablehlo.multiply %v9658, %v9658 : tensor<960x1x3x3xf32>
    %v9663 = stablehlo.multiply %v9660, %v9662 : tensor<960x1x3x3xf32>
    %v9664 = stablehlo.add %v9661, %v9663 : tensor<960x1x3x3xf32>
    %v9665 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9666 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9667 = stablehlo.multiply %v9665, %b17dWv : tensor<960x1x3x3xf32>
    %v9668 = stablehlo.multiply %v9658, %v9658 : tensor<960x1x3x3xf32>
    %v9669 = stablehlo.multiply %v9666, %v9668 : tensor<960x1x3x3xf32>
    %v9670 = stablehlo.add %v9667, %v9669 : tensor<960x1x3x3xf32>
    %v9671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9672 = stablehlo.add %v9670, %v9671 : tensor<960x1x3x3xf32>
    %v9673 = stablehlo.sqrt %v9672 : tensor<960x1x3x3xf32>
    %v9674 = stablehlo.divide %v9658, %v9673 : tensor<960x1x3x3xf32>
    %v9675 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9676 = stablehlo.multiply %v9675, %b17dWm : tensor<960x1x3x3xf32>
    %v9677 = stablehlo.add %v9676, %v9674 : tensor<960x1x3x3xf32>
    %v9678 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9679 = stablehlo.multiply %v9678, %v9677 : tensor<960x1x3x3xf32>
    %v9680 = stablehlo.subtract %b17dW, %v9679 : tensor<960x1x3x3xf32>
    %v9681 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9682 = stablehlo.multiply %v9681, %b17dg : tensor<960xf32>
    %v9683 = stablehlo.add %v9682, %v1826 : tensor<960xf32>
    %v9684 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9685 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9686 = stablehlo.multiply %v9684, %b17dgv : tensor<960xf32>
    %v9687 = stablehlo.multiply %v9683, %v9683 : tensor<960xf32>
    %v9688 = stablehlo.multiply %v9685, %v9687 : tensor<960xf32>
    %v9689 = stablehlo.add %v9686, %v9688 : tensor<960xf32>
    %v9690 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9691 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9692 = stablehlo.multiply %v9690, %b17dgv : tensor<960xf32>
    %v9693 = stablehlo.multiply %v9683, %v9683 : tensor<960xf32>
    %v9694 = stablehlo.multiply %v9691, %v9693 : tensor<960xf32>
    %v9695 = stablehlo.add %v9692, %v9694 : tensor<960xf32>
    %v9696 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9697 = stablehlo.add %v9695, %v9696 : tensor<960xf32>
    %v9698 = stablehlo.sqrt %v9697 : tensor<960xf32>
    %v9699 = stablehlo.divide %v9683, %v9698 : tensor<960xf32>
    %v9700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9701 = stablehlo.multiply %v9700, %b17dgm : tensor<960xf32>
    %v9702 = stablehlo.add %v9701, %v9699 : tensor<960xf32>
    %v9703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9704 = stablehlo.multiply %v9703, %v9702 : tensor<960xf32>
    %v9705 = stablehlo.subtract %b17dg, %v9704 : tensor<960xf32>
    %v9706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9707 = stablehlo.multiply %v9706, %b17dbt : tensor<960xf32>
    %v9708 = stablehlo.add %v9707, %v1829 : tensor<960xf32>
    %v9709 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9710 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9711 = stablehlo.multiply %v9709, %b17dbtv : tensor<960xf32>
    %v9712 = stablehlo.multiply %v9708, %v9708 : tensor<960xf32>
    %v9713 = stablehlo.multiply %v9710, %v9712 : tensor<960xf32>
    %v9714 = stablehlo.add %v9711, %v9713 : tensor<960xf32>
    %v9715 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9716 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9717 = stablehlo.multiply %v9715, %b17dbtv : tensor<960xf32>
    %v9718 = stablehlo.multiply %v9708, %v9708 : tensor<960xf32>
    %v9719 = stablehlo.multiply %v9716, %v9718 : tensor<960xf32>
    %v9720 = stablehlo.add %v9717, %v9719 : tensor<960xf32>
    %v9721 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9722 = stablehlo.add %v9720, %v9721 : tensor<960xf32>
    %v9723 = stablehlo.sqrt %v9722 : tensor<960xf32>
    %v9724 = stablehlo.divide %v9708, %v9723 : tensor<960xf32>
    %v9725 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9726 = stablehlo.multiply %v9725, %b17dbtm : tensor<960xf32>
    %v9727 = stablehlo.add %v9726, %v9724 : tensor<960xf32>
    %v9728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9729 = stablehlo.multiply %v9728, %v9727 : tensor<960xf32>
    %v9730 = stablehlo.subtract %b17dbt, %v9729 : tensor<960xf32>
    %v9731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9732 = stablehlo.multiply %v9731, %b17pW : tensor<320x960x1x1xf32>
    %v9733 = stablehlo.add %v9732, %v1835 : tensor<320x960x1x1xf32>
    %v9734 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9735 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9736 = stablehlo.multiply %v9734, %b17pWv : tensor<320x960x1x1xf32>
    %v9737 = stablehlo.multiply %v9733, %v9733 : tensor<320x960x1x1xf32>
    %v9738 = stablehlo.multiply %v9735, %v9737 : tensor<320x960x1x1xf32>
    %v9739 = stablehlo.add %v9736, %v9738 : tensor<320x960x1x1xf32>
    %v9740 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9741 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9742 = stablehlo.multiply %v9740, %b17pWv : tensor<320x960x1x1xf32>
    %v9743 = stablehlo.multiply %v9733, %v9733 : tensor<320x960x1x1xf32>
    %v9744 = stablehlo.multiply %v9741, %v9743 : tensor<320x960x1x1xf32>
    %v9745 = stablehlo.add %v9742, %v9744 : tensor<320x960x1x1xf32>
    %v9746 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9747 = stablehlo.add %v9745, %v9746 : tensor<320x960x1x1xf32>
    %v9748 = stablehlo.sqrt %v9747 : tensor<320x960x1x1xf32>
    %v9749 = stablehlo.divide %v9733, %v9748 : tensor<320x960x1x1xf32>
    %v9750 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9751 = stablehlo.multiply %v9750, %b17pWm : tensor<320x960x1x1xf32>
    %v9752 = stablehlo.add %v9751, %v9749 : tensor<320x960x1x1xf32>
    %v9753 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9754 = stablehlo.multiply %v9753, %v9752 : tensor<320x960x1x1xf32>
    %v9755 = stablehlo.subtract %b17pW, %v9754 : tensor<320x960x1x1xf32>
    %v9756 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9757 = stablehlo.multiply %v9756, %b17pg : tensor<320xf32>
    %v9758 = stablehlo.add %v9757, %v1853 : tensor<320xf32>
    %v9759 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9760 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9761 = stablehlo.multiply %v9759, %b17pgv : tensor<320xf32>
    %v9762 = stablehlo.multiply %v9758, %v9758 : tensor<320xf32>
    %v9763 = stablehlo.multiply %v9760, %v9762 : tensor<320xf32>
    %v9764 = stablehlo.add %v9761, %v9763 : tensor<320xf32>
    %v9765 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9766 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9767 = stablehlo.multiply %v9765, %b17pgv : tensor<320xf32>
    %v9768 = stablehlo.multiply %v9758, %v9758 : tensor<320xf32>
    %v9769 = stablehlo.multiply %v9766, %v9768 : tensor<320xf32>
    %v9770 = stablehlo.add %v9767, %v9769 : tensor<320xf32>
    %v9771 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9772 = stablehlo.add %v9770, %v9771 : tensor<320xf32>
    %v9773 = stablehlo.sqrt %v9772 : tensor<320xf32>
    %v9774 = stablehlo.divide %v9758, %v9773 : tensor<320xf32>
    %v9775 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9776 = stablehlo.multiply %v9775, %b17pgm : tensor<320xf32>
    %v9777 = stablehlo.add %v9776, %v9774 : tensor<320xf32>
    %v9778 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9779 = stablehlo.multiply %v9778, %v9777 : tensor<320xf32>
    %v9780 = stablehlo.subtract %b17pg, %v9779 : tensor<320xf32>
    %v9781 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9782 = stablehlo.multiply %v9781, %b17pbt : tensor<320xf32>
    %v9783 = stablehlo.add %v9782, %v1856 : tensor<320xf32>
    %v9784 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9785 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9786 = stablehlo.multiply %v9784, %b17pbtv : tensor<320xf32>
    %v9787 = stablehlo.multiply %v9783, %v9783 : tensor<320xf32>
    %v9788 = stablehlo.multiply %v9785, %v9787 : tensor<320xf32>
    %v9789 = stablehlo.add %v9786, %v9788 : tensor<320xf32>
    %v9790 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9791 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9792 = stablehlo.multiply %v9790, %b17pbtv : tensor<320xf32>
    %v9793 = stablehlo.multiply %v9783, %v9783 : tensor<320xf32>
    %v9794 = stablehlo.multiply %v9791, %v9793 : tensor<320xf32>
    %v9795 = stablehlo.add %v9792, %v9794 : tensor<320xf32>
    %v9796 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9797 = stablehlo.add %v9795, %v9796 : tensor<320xf32>
    %v9798 = stablehlo.sqrt %v9797 : tensor<320xf32>
    %v9799 = stablehlo.divide %v9783, %v9798 : tensor<320xf32>
    %v9800 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9801 = stablehlo.multiply %v9800, %b17pbtm : tensor<320xf32>
    %v9802 = stablehlo.add %v9801, %v9799 : tensor<320xf32>
    %v9803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9804 = stablehlo.multiply %v9803, %v9802 : tensor<320xf32>
    %v9805 = stablehlo.subtract %b17pbt, %v9804 : tensor<320xf32>
    %v9806 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9807 = stablehlo.multiply %v9806, %hW : tensor<1280x320x1x1xf32>
    %v9808 = stablehlo.add %v9807, %v1632 : tensor<1280x320x1x1xf32>
    %v9809 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9810 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9811 = stablehlo.multiply %v9809, %hWv : tensor<1280x320x1x1xf32>
    %v9812 = stablehlo.multiply %v9808, %v9808 : tensor<1280x320x1x1xf32>
    %v9813 = stablehlo.multiply %v9810, %v9812 : tensor<1280x320x1x1xf32>
    %v9814 = stablehlo.add %v9811, %v9813 : tensor<1280x320x1x1xf32>
    %v9815 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9816 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9817 = stablehlo.multiply %v9815, %hWv : tensor<1280x320x1x1xf32>
    %v9818 = stablehlo.multiply %v9808, %v9808 : tensor<1280x320x1x1xf32>
    %v9819 = stablehlo.multiply %v9816, %v9818 : tensor<1280x320x1x1xf32>
    %v9820 = stablehlo.add %v9817, %v9819 : tensor<1280x320x1x1xf32>
    %v9821 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9822 = stablehlo.add %v9820, %v9821 : tensor<1280x320x1x1xf32>
    %v9823 = stablehlo.sqrt %v9822 : tensor<1280x320x1x1xf32>
    %v9824 = stablehlo.divide %v9808, %v9823 : tensor<1280x320x1x1xf32>
    %v9825 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9826 = stablehlo.multiply %v9825, %hWm : tensor<1280x320x1x1xf32>
    %v9827 = stablehlo.add %v9826, %v9824 : tensor<1280x320x1x1xf32>
    %v9828 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9829 = stablehlo.multiply %v9828, %v9827 : tensor<1280x320x1x1xf32>
    %v9830 = stablehlo.subtract %hW, %v9829 : tensor<1280x320x1x1xf32>
    %v9831 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9832 = stablehlo.multiply %v9831, %hg : tensor<1280xf32>
    %v9833 = stablehlo.add %v9832, %v1650 : tensor<1280xf32>
    %v9834 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9835 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9836 = stablehlo.multiply %v9834, %hgv : tensor<1280xf32>
    %v9837 = stablehlo.multiply %v9833, %v9833 : tensor<1280xf32>
    %v9838 = stablehlo.multiply %v9835, %v9837 : tensor<1280xf32>
    %v9839 = stablehlo.add %v9836, %v9838 : tensor<1280xf32>
    %v9840 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9841 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9842 = stablehlo.multiply %v9840, %hgv : tensor<1280xf32>
    %v9843 = stablehlo.multiply %v9833, %v9833 : tensor<1280xf32>
    %v9844 = stablehlo.multiply %v9841, %v9843 : tensor<1280xf32>
    %v9845 = stablehlo.add %v9842, %v9844 : tensor<1280xf32>
    %v9846 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9847 = stablehlo.add %v9845, %v9846 : tensor<1280xf32>
    %v9848 = stablehlo.sqrt %v9847 : tensor<1280xf32>
    %v9849 = stablehlo.divide %v9833, %v9848 : tensor<1280xf32>
    %v9850 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9851 = stablehlo.multiply %v9850, %hgm : tensor<1280xf32>
    %v9852 = stablehlo.add %v9851, %v9849 : tensor<1280xf32>
    %v9853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9854 = stablehlo.multiply %v9853, %v9852 : tensor<1280xf32>
    %v9855 = stablehlo.subtract %hg, %v9854 : tensor<1280xf32>
    %v9856 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9857 = stablehlo.multiply %v9856, %hbt : tensor<1280xf32>
    %v9858 = stablehlo.add %v9857, %v1653 : tensor<1280xf32>
    %v9859 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9860 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9861 = stablehlo.multiply %v9859, %hbtv : tensor<1280xf32>
    %v9862 = stablehlo.multiply %v9858, %v9858 : tensor<1280xf32>
    %v9863 = stablehlo.multiply %v9860, %v9862 : tensor<1280xf32>
    %v9864 = stablehlo.add %v9861, %v9863 : tensor<1280xf32>
    %v9865 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9866 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9867 = stablehlo.multiply %v9865, %hbtv : tensor<1280xf32>
    %v9868 = stablehlo.multiply %v9858, %v9858 : tensor<1280xf32>
    %v9869 = stablehlo.multiply %v9866, %v9868 : tensor<1280xf32>
    %v9870 = stablehlo.add %v9867, %v9869 : tensor<1280xf32>
    %v9871 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9872 = stablehlo.add %v9870, %v9871 : tensor<1280xf32>
    %v9873 = stablehlo.sqrt %v9872 : tensor<1280xf32>
    %v9874 = stablehlo.divide %v9858, %v9873 : tensor<1280xf32>
    %v9875 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9876 = stablehlo.multiply %v9875, %hbtm : tensor<1280xf32>
    %v9877 = stablehlo.add %v9876, %v9874 : tensor<1280xf32>
    %v9878 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9879 = stablehlo.multiply %v9878, %v9877 : tensor<1280xf32>
    %v9880 = stablehlo.subtract %hbt, %v9879 : tensor<1280xf32>
    %v9881 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9882 = stablehlo.multiply %v9881, %Wd : tensor<1280x10xf32>
    %v9883 = stablehlo.add %v9882, %v1576 : tensor<1280x10xf32>
    %v9884 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9885 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9886 = stablehlo.multiply %v9884, %Wdv : tensor<1280x10xf32>
    %v9887 = stablehlo.multiply %v9883, %v9883 : tensor<1280x10xf32>
    %v9888 = stablehlo.multiply %v9885, %v9887 : tensor<1280x10xf32>
    %v9889 = stablehlo.add %v9886, %v9888 : tensor<1280x10xf32>
    %v9890 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9891 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9892 = stablehlo.multiply %v9890, %Wdv : tensor<1280x10xf32>
    %v9893 = stablehlo.multiply %v9883, %v9883 : tensor<1280x10xf32>
    %v9894 = stablehlo.multiply %v9891, %v9893 : tensor<1280x10xf32>
    %v9895 = stablehlo.add %v9892, %v9894 : tensor<1280x10xf32>
    %v9896 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9897 = stablehlo.add %v9895, %v9896 : tensor<1280x10xf32>
    %v9898 = stablehlo.sqrt %v9897 : tensor<1280x10xf32>
    %v9899 = stablehlo.divide %v9883, %v9898 : tensor<1280x10xf32>
    %v9900 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9901 = stablehlo.multiply %v9900, %Wdm : tensor<1280x10xf32>
    %v9902 = stablehlo.add %v9901, %v9899 : tensor<1280x10xf32>
    %v9903 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9904 = stablehlo.multiply %v9903, %v9902 : tensor<1280x10xf32>
    %v9905 = stablehlo.subtract %Wd, %v9904 : tensor<1280x10xf32>
    %v9906 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9907 = stablehlo.multiply %v9906, %bd : tensor<10xf32>
    %v9908 = stablehlo.add %v9907, %v1578 : tensor<10xf32>
    %v9909 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9910 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9911 = stablehlo.multiply %v9909, %bdv : tensor<10xf32>
    %v9912 = stablehlo.multiply %v9908, %v9908 : tensor<10xf32>
    %v9913 = stablehlo.multiply %v9910, %v9912 : tensor<10xf32>
    %v9914 = stablehlo.add %v9911, %v9913 : tensor<10xf32>
    %v9915 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9916 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9917 = stablehlo.multiply %v9915, %bdv : tensor<10xf32>
    %v9918 = stablehlo.multiply %v9908, %v9908 : tensor<10xf32>
    %v9919 = stablehlo.multiply %v9916, %v9918 : tensor<10xf32>
    %v9920 = stablehlo.add %v9917, %v9919 : tensor<10xf32>
    %v9921 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9922 = stablehlo.add %v9920, %v9921 : tensor<10xf32>
    %v9923 = stablehlo.sqrt %v9922 : tensor<10xf32>
    %v9924 = stablehlo.divide %v9908, %v9923 : tensor<10xf32>
    %v9925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9926 = stablehlo.multiply %v9925, %bdm : tensor<10xf32>
    %v9927 = stablehlo.add %v9926, %v9924 : tensor<10xf32>
    %v9928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9929 = stablehlo.multiply %v9928, %v9927 : tensor<10xf32>
    %v9930 = stablehlo.subtract %bd, %v9929 : tensor<10xf32>
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
    return %v6005, %v6030, %v6055, %v6080, %v6105, %v6130, %v6155, %v6180, %v6205, %v6230, %v6255, %v6280, %v6305, %v6330, %v6355, %v6380, %v6405, %v6430, %v6455, %v6480, %v6505, %v6530, %v6555, %v6580, %v6605, %v6630, %v6655, %v6680, %v6705, %v6730, %v6755, %v6780, %v6805, %v6830, %v6855, %v6880, %v6905, %v6930, %v6955, %v6980, %v7005, %v7030, %v7055, %v7080, %v7105, %v7130, %v7155, %v7180, %v7205, %v7230, %v7255, %v7280, %v7305, %v7330, %v7355, %v7380, %v7405, %v7430, %v7455, %v7480, %v7505, %v7530, %v7555, %v7580, %v7605, %v7630, %v7655, %v7680, %v7705, %v7730, %v7755, %v7780, %v7805, %v7830, %v7855, %v7880, %v7905, %v7930, %v7955, %v7980, %v8005, %v8030, %v8055, %v8080, %v8105, %v8130, %v8155, %v8180, %v8205, %v8230, %v8255, %v8280, %v8305, %v8330, %v8355, %v8380, %v8405, %v8430, %v8455, %v8480, %v8505, %v8530, %v8555, %v8580, %v8605, %v8630, %v8655, %v8680, %v8705, %v8730, %v8755, %v8780, %v8805, %v8830, %v8855, %v8880, %v8905, %v8930, %v8955, %v8980, %v9005, %v9030, %v9055, %v9080, %v9105, %v9130, %v9155, %v9180, %v9205, %v9230, %v9255, %v9280, %v9305, %v9330, %v9355, %v9380, %v9405, %v9430, %v9455, %v9480, %v9505, %v9530, %v9555, %v9580, %v9605, %v9630, %v9655, %v9680, %v9705, %v9730, %v9755, %v9780, %v9805, %v9830, %v9855, %v9880, %v9905, %v9930, %v6002, %v6027, %v6052, %v6077, %v6102, %v6127, %v6152, %v6177, %v6202, %v6227, %v6252, %v6277, %v6302, %v6327, %v6352, %v6377, %v6402, %v6427, %v6452, %v6477, %v6502, %v6527, %v6552, %v6577, %v6602, %v6627, %v6652, %v6677, %v6702, %v6727, %v6752, %v6777, %v6802, %v6827, %v6852, %v6877, %v6902, %v6927, %v6952, %v6977, %v7002, %v7027, %v7052, %v7077, %v7102, %v7127, %v7152, %v7177, %v7202, %v7227, %v7252, %v7277, %v7302, %v7327, %v7352, %v7377, %v7402, %v7427, %v7452, %v7477, %v7502, %v7527, %v7552, %v7577, %v7602, %v7627, %v7652, %v7677, %v7702, %v7727, %v7752, %v7777, %v7802, %v7827, %v7852, %v7877, %v7902, %v7927, %v7952, %v7977, %v8002, %v8027, %v8052, %v8077, %v8102, %v8127, %v8152, %v8177, %v8202, %v8227, %v8252, %v8277, %v8302, %v8327, %v8352, %v8377, %v8402, %v8427, %v8452, %v8477, %v8502, %v8527, %v8552, %v8577, %v8602, %v8627, %v8652, %v8677, %v8702, %v8727, %v8752, %v8777, %v8802, %v8827, %v8852, %v8877, %v8902, %v8927, %v8952, %v8977, %v9002, %v9027, %v9052, %v9077, %v9102, %v9127, %v9152, %v9177, %v9202, %v9227, %v9252, %v9277, %v9302, %v9327, %v9352, %v9377, %v9402, %v9427, %v9452, %v9477, %v9502, %v9527, %v9552, %v9577, %v9602, %v9627, %v9652, %v9677, %v9702, %v9727, %v9752, %v9777, %v9802, %v9827, %v9852, %v9877, %v9902, %v9927, %v5989, %v6014, %v6039, %v6064, %v6089, %v6114, %v6139, %v6164, %v6189, %v6214, %v6239, %v6264, %v6289, %v6314, %v6339, %v6364, %v6389, %v6414, %v6439, %v6464, %v6489, %v6514, %v6539, %v6564, %v6589, %v6614, %v6639, %v6664, %v6689, %v6714, %v6739, %v6764, %v6789, %v6814, %v6839, %v6864, %v6889, %v6914, %v6939, %v6964, %v6989, %v7014, %v7039, %v7064, %v7089, %v7114, %v7139, %v7164, %v7189, %v7214, %v7239, %v7264, %v7289, %v7314, %v7339, %v7364, %v7389, %v7414, %v7439, %v7464, %v7489, %v7514, %v7539, %v7564, %v7589, %v7614, %v7639, %v7664, %v7689, %v7714, %v7739, %v7764, %v7789, %v7814, %v7839, %v7864, %v7889, %v7914, %v7939, %v7964, %v7989, %v8014, %v8039, %v8064, %v8089, %v8114, %v8139, %v8164, %v8189, %v8214, %v8239, %v8264, %v8289, %v8314, %v8339, %v8364, %v8389, %v8414, %v8439, %v8464, %v8489, %v8514, %v8539, %v8564, %v8589, %v8614, %v8639, %v8664, %v8689, %v8714, %v8739, %v8764, %v8789, %v8814, %v8839, %v8864, %v8889, %v8914, %v8939, %v8964, %v8989, %v9014, %v9039, %v9064, %v9089, %v9114, %v9139, %v9164, %v9189, %v9214, %v9239, %v9264, %v9289, %v9314, %v9339, %v9364, %v9389, %v9414, %v9439, %v9464, %v9489, %v9514, %v9539, %v9564, %v9589, %v9614, %v9639, %v9664, %v9689, %v9714, %v9739, %v9764, %v9789, %v9814, %v9839, %v9864, %v9889, %v9914, %loss, %bc1, %bc2, %v5153, %v5164, %v5169, %v5180, %v5185, %v5196, %v5201, %v5212, %v5217, %v5228, %v5233, %v5244, %v5249, %v5260, %v5265, %v5276, %v5281, %v5292, %v5297, %v5308, %v5313, %v5324, %v5329, %v5340, %v5345, %v5356, %v5361, %v5372, %v5377, %v5388, %v5393, %v5404, %v5409, %v5420, %v5425, %v5436, %v5441, %v5452, %v5457, %v5468, %v5473, %v5484, %v5489, %v5500, %v5505, %v5516, %v5521, %v5532, %v5537, %v5548, %v5553, %v5564, %v5569, %v5580, %v5585, %v5596, %v5601, %v5612, %v5617, %v5628, %v5633, %v5644, %v5649, %v5660, %v5665, %v5676, %v5681, %v5692, %v5697, %v5708, %v5713, %v5724, %v5729, %v5740, %v5745, %v5756, %v5761, %v5772, %v5777, %v5788, %v5793, %v5804, %v5809, %v5820, %v5825, %v5836, %v5841, %v5852, %v5857, %v5868, %v5873, %v5884, %v5889, %v5900, %v5905, %v5916, %v5921, %v5932, %v5937, %v5948, %v5953, %v5964, %v5969, %v5980 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280xf32>, tensor<1280xf32>
  }
}

module @m {
  func.func @efficientnet_adam_train_step(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sb: tensor<32xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1db: tensor<32xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pb: tensor<16xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eb: tensor<144xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3db: tensor<144xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pb: tensor<24xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eb: tensor<144xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4db: tensor<144xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pb: tensor<40xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eb: tensor<240xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5db: tensor<240xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pb: tensor<40xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eb: tensor<240xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6db: tensor<240xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pb: tensor<80xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eb: tensor<480xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7db: tensor<480xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pb: tensor<80xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eb: tensor<480xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8db: tensor<480xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pb: tensor<80xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eb: tensor<480xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9db: tensor<480xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pb: tensor<112xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eb: tensor<672xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10db: tensor<672xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pb: tensor<112xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eb: tensor<672xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11db: tensor<672xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pb: tensor<112xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eb: tensor<672xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12db: tensor<672xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pb: tensor<192xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eb: tensor<1152xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13db: tensor<1152xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pb: tensor<192xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eb: tensor<1152xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14db: tensor<1152xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pb: tensor<192xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eb: tensor<1152xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15db: tensor<1152xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pb: tensor<192xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eb: tensor<1152xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16db: tensor<1152xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pb: tensor<320xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hb: tensor<1280xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<32x3x3x3xf32>, %sbm: tensor<32xf32>, %sgm: tensor<32xf32>, %sbtm: tensor<32xf32>, %b1dWm: tensor<32x1x3x3xf32>, %b1dbm: tensor<32xf32>, %b1dgm: tensor<32xf32>, %b1dbtm: tensor<32xf32>, %b1zW1m: tensor<32x8xf32>, %b1zb1m: tensor<8xf32>, %b1zW2m: tensor<8x32xf32>, %b1zb2m: tensor<32xf32>, %b1pWm: tensor<16x32x1x1xf32>, %b1pbm: tensor<16xf32>, %b1pgm: tensor<16xf32>, %b1pbtm: tensor<16xf32>, %b2eWm: tensor<96x16x1x1xf32>, %b2ebm: tensor<96xf32>, %b2egm: tensor<96xf32>, %b2ebtm: tensor<96xf32>, %b2dWm: tensor<96x1x3x3xf32>, %b2dbm: tensor<96xf32>, %b2dgm: tensor<96xf32>, %b2dbtm: tensor<96xf32>, %b2zW1m: tensor<96x4xf32>, %b2zb1m: tensor<4xf32>, %b2zW2m: tensor<4x96xf32>, %b2zb2m: tensor<96xf32>, %b2pWm: tensor<24x96x1x1xf32>, %b2pbm: tensor<24xf32>, %b2pgm: tensor<24xf32>, %b2pbtm: tensor<24xf32>, %b3eWm: tensor<144x24x1x1xf32>, %b3ebm: tensor<144xf32>, %b3egm: tensor<144xf32>, %b3ebtm: tensor<144xf32>, %b3dWm: tensor<144x1x3x3xf32>, %b3dbm: tensor<144xf32>, %b3dgm: tensor<144xf32>, %b3dbtm: tensor<144xf32>, %b3zW1m: tensor<144x6xf32>, %b3zb1m: tensor<6xf32>, %b3zW2m: tensor<6x144xf32>, %b3zb2m: tensor<144xf32>, %b3pWm: tensor<24x144x1x1xf32>, %b3pbm: tensor<24xf32>, %b3pgm: tensor<24xf32>, %b3pbtm: tensor<24xf32>, %b4eWm: tensor<144x24x1x1xf32>, %b4ebm: tensor<144xf32>, %b4egm: tensor<144xf32>, %b4ebtm: tensor<144xf32>, %b4dWm: tensor<144x1x5x5xf32>, %b4dbm: tensor<144xf32>, %b4dgm: tensor<144xf32>, %b4dbtm: tensor<144xf32>, %b4zW1m: tensor<144x6xf32>, %b4zb1m: tensor<6xf32>, %b4zW2m: tensor<6x144xf32>, %b4zb2m: tensor<144xf32>, %b4pWm: tensor<40x144x1x1xf32>, %b4pbm: tensor<40xf32>, %b4pgm: tensor<40xf32>, %b4pbtm: tensor<40xf32>, %b5eWm: tensor<240x40x1x1xf32>, %b5ebm: tensor<240xf32>, %b5egm: tensor<240xf32>, %b5ebtm: tensor<240xf32>, %b5dWm: tensor<240x1x5x5xf32>, %b5dbm: tensor<240xf32>, %b5dgm: tensor<240xf32>, %b5dbtm: tensor<240xf32>, %b5zW1m: tensor<240x10xf32>, %b5zb1m: tensor<10xf32>, %b5zW2m: tensor<10x240xf32>, %b5zb2m: tensor<240xf32>, %b5pWm: tensor<40x240x1x1xf32>, %b5pbm: tensor<40xf32>, %b5pgm: tensor<40xf32>, %b5pbtm: tensor<40xf32>, %b6eWm: tensor<240x40x1x1xf32>, %b6ebm: tensor<240xf32>, %b6egm: tensor<240xf32>, %b6ebtm: tensor<240xf32>, %b6dWm: tensor<240x1x3x3xf32>, %b6dbm: tensor<240xf32>, %b6dgm: tensor<240xf32>, %b6dbtm: tensor<240xf32>, %b6zW1m: tensor<240x10xf32>, %b6zb1m: tensor<10xf32>, %b6zW2m: tensor<10x240xf32>, %b6zb2m: tensor<240xf32>, %b6pWm: tensor<80x240x1x1xf32>, %b6pbm: tensor<80xf32>, %b6pgm: tensor<80xf32>, %b6pbtm: tensor<80xf32>, %b7eWm: tensor<480x80x1x1xf32>, %b7ebm: tensor<480xf32>, %b7egm: tensor<480xf32>, %b7ebtm: tensor<480xf32>, %b7dWm: tensor<480x1x3x3xf32>, %b7dbm: tensor<480xf32>, %b7dgm: tensor<480xf32>, %b7dbtm: tensor<480xf32>, %b7zW1m: tensor<480x20xf32>, %b7zb1m: tensor<20xf32>, %b7zW2m: tensor<20x480xf32>, %b7zb2m: tensor<480xf32>, %b7pWm: tensor<80x480x1x1xf32>, %b7pbm: tensor<80xf32>, %b7pgm: tensor<80xf32>, %b7pbtm: tensor<80xf32>, %b8eWm: tensor<480x80x1x1xf32>, %b8ebm: tensor<480xf32>, %b8egm: tensor<480xf32>, %b8ebtm: tensor<480xf32>, %b8dWm: tensor<480x1x3x3xf32>, %b8dbm: tensor<480xf32>, %b8dgm: tensor<480xf32>, %b8dbtm: tensor<480xf32>, %b8zW1m: tensor<480x20xf32>, %b8zb1m: tensor<20xf32>, %b8zW2m: tensor<20x480xf32>, %b8zb2m: tensor<480xf32>, %b8pWm: tensor<80x480x1x1xf32>, %b8pbm: tensor<80xf32>, %b8pgm: tensor<80xf32>, %b8pbtm: tensor<80xf32>, %b9eWm: tensor<480x80x1x1xf32>, %b9ebm: tensor<480xf32>, %b9egm: tensor<480xf32>, %b9ebtm: tensor<480xf32>, %b9dWm: tensor<480x1x5x5xf32>, %b9dbm: tensor<480xf32>, %b9dgm: tensor<480xf32>, %b9dbtm: tensor<480xf32>, %b9zW1m: tensor<480x20xf32>, %b9zb1m: tensor<20xf32>, %b9zW2m: tensor<20x480xf32>, %b9zb2m: tensor<480xf32>, %b9pWm: tensor<112x480x1x1xf32>, %b9pbm: tensor<112xf32>, %b9pgm: tensor<112xf32>, %b9pbtm: tensor<112xf32>, %b10eWm: tensor<672x112x1x1xf32>, %b10ebm: tensor<672xf32>, %b10egm: tensor<672xf32>, %b10ebtm: tensor<672xf32>, %b10dWm: tensor<672x1x5x5xf32>, %b10dbm: tensor<672xf32>, %b10dgm: tensor<672xf32>, %b10dbtm: tensor<672xf32>, %b10zW1m: tensor<672x28xf32>, %b10zb1m: tensor<28xf32>, %b10zW2m: tensor<28x672xf32>, %b10zb2m: tensor<672xf32>, %b10pWm: tensor<112x672x1x1xf32>, %b10pbm: tensor<112xf32>, %b10pgm: tensor<112xf32>, %b10pbtm: tensor<112xf32>, %b11eWm: tensor<672x112x1x1xf32>, %b11ebm: tensor<672xf32>, %b11egm: tensor<672xf32>, %b11ebtm: tensor<672xf32>, %b11dWm: tensor<672x1x5x5xf32>, %b11dbm: tensor<672xf32>, %b11dgm: tensor<672xf32>, %b11dbtm: tensor<672xf32>, %b11zW1m: tensor<672x28xf32>, %b11zb1m: tensor<28xf32>, %b11zW2m: tensor<28x672xf32>, %b11zb2m: tensor<672xf32>, %b11pWm: tensor<112x672x1x1xf32>, %b11pbm: tensor<112xf32>, %b11pgm: tensor<112xf32>, %b11pbtm: tensor<112xf32>, %b12eWm: tensor<672x112x1x1xf32>, %b12ebm: tensor<672xf32>, %b12egm: tensor<672xf32>, %b12ebtm: tensor<672xf32>, %b12dWm: tensor<672x1x5x5xf32>, %b12dbm: tensor<672xf32>, %b12dgm: tensor<672xf32>, %b12dbtm: tensor<672xf32>, %b12zW1m: tensor<672x28xf32>, %b12zb1m: tensor<28xf32>, %b12zW2m: tensor<28x672xf32>, %b12zb2m: tensor<672xf32>, %b12pWm: tensor<192x672x1x1xf32>, %b12pbm: tensor<192xf32>, %b12pgm: tensor<192xf32>, %b12pbtm: tensor<192xf32>, %b13eWm: tensor<1152x192x1x1xf32>, %b13ebm: tensor<1152xf32>, %b13egm: tensor<1152xf32>, %b13ebtm: tensor<1152xf32>, %b13dWm: tensor<1152x1x5x5xf32>, %b13dbm: tensor<1152xf32>, %b13dgm: tensor<1152xf32>, %b13dbtm: tensor<1152xf32>, %b13zW1m: tensor<1152x48xf32>, %b13zb1m: tensor<48xf32>, %b13zW2m: tensor<48x1152xf32>, %b13zb2m: tensor<1152xf32>, %b13pWm: tensor<192x1152x1x1xf32>, %b13pbm: tensor<192xf32>, %b13pgm: tensor<192xf32>, %b13pbtm: tensor<192xf32>, %b14eWm: tensor<1152x192x1x1xf32>, %b14ebm: tensor<1152xf32>, %b14egm: tensor<1152xf32>, %b14ebtm: tensor<1152xf32>, %b14dWm: tensor<1152x1x5x5xf32>, %b14dbm: tensor<1152xf32>, %b14dgm: tensor<1152xf32>, %b14dbtm: tensor<1152xf32>, %b14zW1m: tensor<1152x48xf32>, %b14zb1m: tensor<48xf32>, %b14zW2m: tensor<48x1152xf32>, %b14zb2m: tensor<1152xf32>, %b14pWm: tensor<192x1152x1x1xf32>, %b14pbm: tensor<192xf32>, %b14pgm: tensor<192xf32>, %b14pbtm: tensor<192xf32>, %b15eWm: tensor<1152x192x1x1xf32>, %b15ebm: tensor<1152xf32>, %b15egm: tensor<1152xf32>, %b15ebtm: tensor<1152xf32>, %b15dWm: tensor<1152x1x5x5xf32>, %b15dbm: tensor<1152xf32>, %b15dgm: tensor<1152xf32>, %b15dbtm: tensor<1152xf32>, %b15zW1m: tensor<1152x48xf32>, %b15zb1m: tensor<48xf32>, %b15zW2m: tensor<48x1152xf32>, %b15zb2m: tensor<1152xf32>, %b15pWm: tensor<192x1152x1x1xf32>, %b15pbm: tensor<192xf32>, %b15pgm: tensor<192xf32>, %b15pbtm: tensor<192xf32>, %b16eWm: tensor<1152x192x1x1xf32>, %b16ebm: tensor<1152xf32>, %b16egm: tensor<1152xf32>, %b16ebtm: tensor<1152xf32>, %b16dWm: tensor<1152x1x3x3xf32>, %b16dbm: tensor<1152xf32>, %b16dgm: tensor<1152xf32>, %b16dbtm: tensor<1152xf32>, %b16zW1m: tensor<1152x48xf32>, %b16zb1m: tensor<48xf32>, %b16zW2m: tensor<48x1152xf32>, %b16zb2m: tensor<1152xf32>, %b16pWm: tensor<320x1152x1x1xf32>, %b16pbm: tensor<320xf32>, %b16pgm: tensor<320xf32>, %b16pbtm: tensor<320xf32>, %hWm: tensor<1280x320x1x1xf32>, %hbm: tensor<1280xf32>, %hgm: tensor<1280xf32>, %hbtm: tensor<1280xf32>, %Wdm: tensor<1280x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<32x3x3x3xf32>, %sbv: tensor<32xf32>, %sgv: tensor<32xf32>, %sbtv: tensor<32xf32>, %b1dWv: tensor<32x1x3x3xf32>, %b1dbv: tensor<32xf32>, %b1dgv: tensor<32xf32>, %b1dbtv: tensor<32xf32>, %b1zW1v: tensor<32x8xf32>, %b1zb1v: tensor<8xf32>, %b1zW2v: tensor<8x32xf32>, %b1zb2v: tensor<32xf32>, %b1pWv: tensor<16x32x1x1xf32>, %b1pbv: tensor<16xf32>, %b1pgv: tensor<16xf32>, %b1pbtv: tensor<16xf32>, %b2eWv: tensor<96x16x1x1xf32>, %b2ebv: tensor<96xf32>, %b2egv: tensor<96xf32>, %b2ebtv: tensor<96xf32>, %b2dWv: tensor<96x1x3x3xf32>, %b2dbv: tensor<96xf32>, %b2dgv: tensor<96xf32>, %b2dbtv: tensor<96xf32>, %b2zW1v: tensor<96x4xf32>, %b2zb1v: tensor<4xf32>, %b2zW2v: tensor<4x96xf32>, %b2zb2v: tensor<96xf32>, %b2pWv: tensor<24x96x1x1xf32>, %b2pbv: tensor<24xf32>, %b2pgv: tensor<24xf32>, %b2pbtv: tensor<24xf32>, %b3eWv: tensor<144x24x1x1xf32>, %b3ebv: tensor<144xf32>, %b3egv: tensor<144xf32>, %b3ebtv: tensor<144xf32>, %b3dWv: tensor<144x1x3x3xf32>, %b3dbv: tensor<144xf32>, %b3dgv: tensor<144xf32>, %b3dbtv: tensor<144xf32>, %b3zW1v: tensor<144x6xf32>, %b3zb1v: tensor<6xf32>, %b3zW2v: tensor<6x144xf32>, %b3zb2v: tensor<144xf32>, %b3pWv: tensor<24x144x1x1xf32>, %b3pbv: tensor<24xf32>, %b3pgv: tensor<24xf32>, %b3pbtv: tensor<24xf32>, %b4eWv: tensor<144x24x1x1xf32>, %b4ebv: tensor<144xf32>, %b4egv: tensor<144xf32>, %b4ebtv: tensor<144xf32>, %b4dWv: tensor<144x1x5x5xf32>, %b4dbv: tensor<144xf32>, %b4dgv: tensor<144xf32>, %b4dbtv: tensor<144xf32>, %b4zW1v: tensor<144x6xf32>, %b4zb1v: tensor<6xf32>, %b4zW2v: tensor<6x144xf32>, %b4zb2v: tensor<144xf32>, %b4pWv: tensor<40x144x1x1xf32>, %b4pbv: tensor<40xf32>, %b4pgv: tensor<40xf32>, %b4pbtv: tensor<40xf32>, %b5eWv: tensor<240x40x1x1xf32>, %b5ebv: tensor<240xf32>, %b5egv: tensor<240xf32>, %b5ebtv: tensor<240xf32>, %b5dWv: tensor<240x1x5x5xf32>, %b5dbv: tensor<240xf32>, %b5dgv: tensor<240xf32>, %b5dbtv: tensor<240xf32>, %b5zW1v: tensor<240x10xf32>, %b5zb1v: tensor<10xf32>, %b5zW2v: tensor<10x240xf32>, %b5zb2v: tensor<240xf32>, %b5pWv: tensor<40x240x1x1xf32>, %b5pbv: tensor<40xf32>, %b5pgv: tensor<40xf32>, %b5pbtv: tensor<40xf32>, %b6eWv: tensor<240x40x1x1xf32>, %b6ebv: tensor<240xf32>, %b6egv: tensor<240xf32>, %b6ebtv: tensor<240xf32>, %b6dWv: tensor<240x1x3x3xf32>, %b6dbv: tensor<240xf32>, %b6dgv: tensor<240xf32>, %b6dbtv: tensor<240xf32>, %b6zW1v: tensor<240x10xf32>, %b6zb1v: tensor<10xf32>, %b6zW2v: tensor<10x240xf32>, %b6zb2v: tensor<240xf32>, %b6pWv: tensor<80x240x1x1xf32>, %b6pbv: tensor<80xf32>, %b6pgv: tensor<80xf32>, %b6pbtv: tensor<80xf32>, %b7eWv: tensor<480x80x1x1xf32>, %b7ebv: tensor<480xf32>, %b7egv: tensor<480xf32>, %b7ebtv: tensor<480xf32>, %b7dWv: tensor<480x1x3x3xf32>, %b7dbv: tensor<480xf32>, %b7dgv: tensor<480xf32>, %b7dbtv: tensor<480xf32>, %b7zW1v: tensor<480x20xf32>, %b7zb1v: tensor<20xf32>, %b7zW2v: tensor<20x480xf32>, %b7zb2v: tensor<480xf32>, %b7pWv: tensor<80x480x1x1xf32>, %b7pbv: tensor<80xf32>, %b7pgv: tensor<80xf32>, %b7pbtv: tensor<80xf32>, %b8eWv: tensor<480x80x1x1xf32>, %b8ebv: tensor<480xf32>, %b8egv: tensor<480xf32>, %b8ebtv: tensor<480xf32>, %b8dWv: tensor<480x1x3x3xf32>, %b8dbv: tensor<480xf32>, %b8dgv: tensor<480xf32>, %b8dbtv: tensor<480xf32>, %b8zW1v: tensor<480x20xf32>, %b8zb1v: tensor<20xf32>, %b8zW2v: tensor<20x480xf32>, %b8zb2v: tensor<480xf32>, %b8pWv: tensor<80x480x1x1xf32>, %b8pbv: tensor<80xf32>, %b8pgv: tensor<80xf32>, %b8pbtv: tensor<80xf32>, %b9eWv: tensor<480x80x1x1xf32>, %b9ebv: tensor<480xf32>, %b9egv: tensor<480xf32>, %b9ebtv: tensor<480xf32>, %b9dWv: tensor<480x1x5x5xf32>, %b9dbv: tensor<480xf32>, %b9dgv: tensor<480xf32>, %b9dbtv: tensor<480xf32>, %b9zW1v: tensor<480x20xf32>, %b9zb1v: tensor<20xf32>, %b9zW2v: tensor<20x480xf32>, %b9zb2v: tensor<480xf32>, %b9pWv: tensor<112x480x1x1xf32>, %b9pbv: tensor<112xf32>, %b9pgv: tensor<112xf32>, %b9pbtv: tensor<112xf32>, %b10eWv: tensor<672x112x1x1xf32>, %b10ebv: tensor<672xf32>, %b10egv: tensor<672xf32>, %b10ebtv: tensor<672xf32>, %b10dWv: tensor<672x1x5x5xf32>, %b10dbv: tensor<672xf32>, %b10dgv: tensor<672xf32>, %b10dbtv: tensor<672xf32>, %b10zW1v: tensor<672x28xf32>, %b10zb1v: tensor<28xf32>, %b10zW2v: tensor<28x672xf32>, %b10zb2v: tensor<672xf32>, %b10pWv: tensor<112x672x1x1xf32>, %b10pbv: tensor<112xf32>, %b10pgv: tensor<112xf32>, %b10pbtv: tensor<112xf32>, %b11eWv: tensor<672x112x1x1xf32>, %b11ebv: tensor<672xf32>, %b11egv: tensor<672xf32>, %b11ebtv: tensor<672xf32>, %b11dWv: tensor<672x1x5x5xf32>, %b11dbv: tensor<672xf32>, %b11dgv: tensor<672xf32>, %b11dbtv: tensor<672xf32>, %b11zW1v: tensor<672x28xf32>, %b11zb1v: tensor<28xf32>, %b11zW2v: tensor<28x672xf32>, %b11zb2v: tensor<672xf32>, %b11pWv: tensor<112x672x1x1xf32>, %b11pbv: tensor<112xf32>, %b11pgv: tensor<112xf32>, %b11pbtv: tensor<112xf32>, %b12eWv: tensor<672x112x1x1xf32>, %b12ebv: tensor<672xf32>, %b12egv: tensor<672xf32>, %b12ebtv: tensor<672xf32>, %b12dWv: tensor<672x1x5x5xf32>, %b12dbv: tensor<672xf32>, %b12dgv: tensor<672xf32>, %b12dbtv: tensor<672xf32>, %b12zW1v: tensor<672x28xf32>, %b12zb1v: tensor<28xf32>, %b12zW2v: tensor<28x672xf32>, %b12zb2v: tensor<672xf32>, %b12pWv: tensor<192x672x1x1xf32>, %b12pbv: tensor<192xf32>, %b12pgv: tensor<192xf32>, %b12pbtv: tensor<192xf32>, %b13eWv: tensor<1152x192x1x1xf32>, %b13ebv: tensor<1152xf32>, %b13egv: tensor<1152xf32>, %b13ebtv: tensor<1152xf32>, %b13dWv: tensor<1152x1x5x5xf32>, %b13dbv: tensor<1152xf32>, %b13dgv: tensor<1152xf32>, %b13dbtv: tensor<1152xf32>, %b13zW1v: tensor<1152x48xf32>, %b13zb1v: tensor<48xf32>, %b13zW2v: tensor<48x1152xf32>, %b13zb2v: tensor<1152xf32>, %b13pWv: tensor<192x1152x1x1xf32>, %b13pbv: tensor<192xf32>, %b13pgv: tensor<192xf32>, %b13pbtv: tensor<192xf32>, %b14eWv: tensor<1152x192x1x1xf32>, %b14ebv: tensor<1152xf32>, %b14egv: tensor<1152xf32>, %b14ebtv: tensor<1152xf32>, %b14dWv: tensor<1152x1x5x5xf32>, %b14dbv: tensor<1152xf32>, %b14dgv: tensor<1152xf32>, %b14dbtv: tensor<1152xf32>, %b14zW1v: tensor<1152x48xf32>, %b14zb1v: tensor<48xf32>, %b14zW2v: tensor<48x1152xf32>, %b14zb2v: tensor<1152xf32>, %b14pWv: tensor<192x1152x1x1xf32>, %b14pbv: tensor<192xf32>, %b14pgv: tensor<192xf32>, %b14pbtv: tensor<192xf32>, %b15eWv: tensor<1152x192x1x1xf32>, %b15ebv: tensor<1152xf32>, %b15egv: tensor<1152xf32>, %b15ebtv: tensor<1152xf32>, %b15dWv: tensor<1152x1x5x5xf32>, %b15dbv: tensor<1152xf32>, %b15dgv: tensor<1152xf32>, %b15dbtv: tensor<1152xf32>, %b15zW1v: tensor<1152x48xf32>, %b15zb1v: tensor<48xf32>, %b15zW2v: tensor<48x1152xf32>, %b15zb2v: tensor<1152xf32>, %b15pWv: tensor<192x1152x1x1xf32>, %b15pbv: tensor<192xf32>, %b15pgv: tensor<192xf32>, %b15pbtv: tensor<192xf32>, %b16eWv: tensor<1152x192x1x1xf32>, %b16ebv: tensor<1152xf32>, %b16egv: tensor<1152xf32>, %b16ebtv: tensor<1152xf32>, %b16dWv: tensor<1152x1x3x3xf32>, %b16dbv: tensor<1152xf32>, %b16dgv: tensor<1152xf32>, %b16dbtv: tensor<1152xf32>, %b16zW1v: tensor<1152x48xf32>, %b16zb1v: tensor<48xf32>, %b16zW2v: tensor<48x1152xf32>, %b16zb2v: tensor<1152xf32>, %b16pWv: tensor<320x1152x1x1xf32>, %b16pbv: tensor<320xf32>, %b16pgv: tensor<320xf32>, %b16pbtv: tensor<320xf32>, %hWv: tensor<1280x320x1x1xf32>, %hbv: tensor<1280xf32>, %hgv: tensor<1280xf32>, %hbtv: tensor<1280xf32>, %Wdv: tensor<1280x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<32xf32>, %stnvari: tensor<32xf32>, %b1dnmui: tensor<32xf32>, %b1dnvari: tensor<32xf32>, %b1pnmui: tensor<16xf32>, %b1pnvari: tensor<16xf32>, %b2enmui: tensor<96xf32>, %b2envari: tensor<96xf32>, %b2dnmui: tensor<96xf32>, %b2dnvari: tensor<96xf32>, %b2pnmui: tensor<24xf32>, %b2pnvari: tensor<24xf32>, %b3enmui: tensor<144xf32>, %b3envari: tensor<144xf32>, %b3dnmui: tensor<144xf32>, %b3dnvari: tensor<144xf32>, %b3pnmui: tensor<24xf32>, %b3pnvari: tensor<24xf32>, %b4enmui: tensor<144xf32>, %b4envari: tensor<144xf32>, %b4dnmui: tensor<144xf32>, %b4dnvari: tensor<144xf32>, %b4pnmui: tensor<40xf32>, %b4pnvari: tensor<40xf32>, %b5enmui: tensor<240xf32>, %b5envari: tensor<240xf32>, %b5dnmui: tensor<240xf32>, %b5dnvari: tensor<240xf32>, %b5pnmui: tensor<40xf32>, %b5pnvari: tensor<40xf32>, %b6enmui: tensor<240xf32>, %b6envari: tensor<240xf32>, %b6dnmui: tensor<240xf32>, %b6dnvari: tensor<240xf32>, %b6pnmui: tensor<80xf32>, %b6pnvari: tensor<80xf32>, %b7enmui: tensor<480xf32>, %b7envari: tensor<480xf32>, %b7dnmui: tensor<480xf32>, %b7dnvari: tensor<480xf32>, %b7pnmui: tensor<80xf32>, %b7pnvari: tensor<80xf32>, %b8enmui: tensor<480xf32>, %b8envari: tensor<480xf32>, %b8dnmui: tensor<480xf32>, %b8dnvari: tensor<480xf32>, %b8pnmui: tensor<80xf32>, %b8pnvari: tensor<80xf32>, %b9enmui: tensor<480xf32>, %b9envari: tensor<480xf32>, %b9dnmui: tensor<480xf32>, %b9dnvari: tensor<480xf32>, %b9pnmui: tensor<112xf32>, %b9pnvari: tensor<112xf32>, %b10enmui: tensor<672xf32>, %b10envari: tensor<672xf32>, %b10dnmui: tensor<672xf32>, %b10dnvari: tensor<672xf32>, %b10pnmui: tensor<112xf32>, %b10pnvari: tensor<112xf32>, %b11enmui: tensor<672xf32>, %b11envari: tensor<672xf32>, %b11dnmui: tensor<672xf32>, %b11dnvari: tensor<672xf32>, %b11pnmui: tensor<112xf32>, %b11pnvari: tensor<112xf32>, %b12enmui: tensor<672xf32>, %b12envari: tensor<672xf32>, %b12dnmui: tensor<672xf32>, %b12dnvari: tensor<672xf32>, %b12pnmui: tensor<192xf32>, %b12pnvari: tensor<192xf32>, %b13enmui: tensor<1152xf32>, %b13envari: tensor<1152xf32>, %b13dnmui: tensor<1152xf32>, %b13dnvari: tensor<1152xf32>, %b13pnmui: tensor<192xf32>, %b13pnvari: tensor<192xf32>, %b14enmui: tensor<1152xf32>, %b14envari: tensor<1152xf32>, %b14dnmui: tensor<1152xf32>, %b14dnvari: tensor<1152xf32>, %b14pnmui: tensor<192xf32>, %b14pnvari: tensor<192xf32>, %b15enmui: tensor<1152xf32>, %b15envari: tensor<1152xf32>, %b15dnmui: tensor<1152xf32>, %b15dnvari: tensor<1152xf32>, %b15pnmui: tensor<192xf32>, %b15pnvari: tensor<192xf32>, %b16enmui: tensor<1152xf32>, %b16envari: tensor<1152xf32>, %b16dnmui: tensor<1152xf32>, %b16dnvari: tensor<1152xf32>, %b16pnmui: tensor<320xf32>, %b16pnvari: tensor<320xf32>, %hnmui: tensor<1280xf32>, %hnvari: tensor<1280xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280xf32>, tensor<1280xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<32x32x112x112xf32>
    %stnnf = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %stnsmr = stablehlo.reduce(%stc init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %stnsm = stablehlo.broadcast_in_dim %stnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stnmu = stablehlo.divide %stnsm, %stnnf : tensor<32x32x112x112xf32>
    %stnxc = stablehlo.subtract %stc, %stnmu : tensor<32x32x112x112xf32>
    %stnsq = stablehlo.multiply %stnxc, %stnxc : tensor<32x32x112x112xf32>
    %stnvsr = stablehlo.reduce(%stnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %stnvs = stablehlo.broadcast_in_dim %stnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stnvr = stablehlo.divide %stnvs, %stnnf : tensor<32x32x112x112xf32>
    %stnve = stablehlo.add %stnvr, %stnep : tensor<32x32x112x112xf32>
    %stnistd = stablehlo.rsqrt %stnve : tensor<32x32x112x112xf32>
    %stnxh = stablehlo.multiply %stnxc, %stnistd : tensor<32x32x112x112xf32>
    %stngb = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stnbtb = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %stngx = stablehlo.multiply %stnxh, %stngb : tensor<32x32x112x112xf32>
    %stn = stablehlo.add %stngx, %stnbtb : tensor<32x32x112x112xf32>
    %strs = stablehlo.logistic %stn : tensor<32x32x112x112xf32>
    %str = stablehlo.multiply %stn, %strs : tensor<32x32x112x112xf32>
    %b1dc = stablehlo.convolution(%str, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %b1dbb = stablehlo.broadcast_in_dim %b1db, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1d = stablehlo.add %b1dc, %b1dbb : tensor<32x32x112x112xf32>
    %b1dnnf = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %b1dnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %b1dnsmr = stablehlo.reduce(%b1d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1dnsm = stablehlo.broadcast_in_dim %b1dnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dnmu = stablehlo.divide %b1dnsm, %b1dnnf : tensor<32x32x112x112xf32>
    %b1dnxc = stablehlo.subtract %b1d, %b1dnmu : tensor<32x32x112x112xf32>
    %b1dnsq = stablehlo.multiply %b1dnxc, %b1dnxc : tensor<32x32x112x112xf32>
    %b1dnvsr = stablehlo.reduce(%b1dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1dnvs = stablehlo.broadcast_in_dim %b1dnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dnvr = stablehlo.divide %b1dnvs, %b1dnnf : tensor<32x32x112x112xf32>
    %b1dnve = stablehlo.add %b1dnvr, %b1dnep : tensor<32x32x112x112xf32>
    %b1dnistd = stablehlo.rsqrt %b1dnve : tensor<32x32x112x112xf32>
    %b1dnxh = stablehlo.multiply %b1dnxc, %b1dnistd : tensor<32x32x112x112xf32>
    %b1dngb = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dnbtb = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1dngx = stablehlo.multiply %b1dnxh, %b1dngb : tensor<32x32x112x112xf32>
    %b1dn = stablehlo.add %b1dngx, %b1dnbtb : tensor<32x32x112x112xf32>
    %b1dss = stablehlo.logistic %b1dn : tensor<32x32x112x112xf32>
    %b1ds = stablehlo.multiply %b1dn, %b1dss : tensor<32x32x112x112xf32>
    %b1zsqs = stablehlo.reduce(%b1ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %b1zsqnf = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %b1zsq = stablehlo.divide %b1zsqs, %b1zsqnf : tensor<32x32xf32>
    %b1zexd = stablehlo.dot_general %b1zsq, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %b1zexbb = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %b1zex = stablehlo.add %b1zexd, %b1zexbb : tensor<32x8xf32>
    %b1za1s = stablehlo.logistic %b1zex : tensor<32x8xf32>
    %b1za1 = stablehlo.multiply %b1zex, %b1za1s : tensor<32x8xf32>
    %b1zh2d = stablehlo.dot_general %b1za1, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %b1zh2bb = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %b1zh2 = stablehlo.add %b1zh2d, %b1zh2bb : tensor<32x32xf32>
    %b1zgate = stablehlo.logistic %b1zh2 : tensor<32x32xf32>
    %b1zgb = stablehlo.broadcast_in_dim %b1zgate, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %b1zse = stablehlo.multiply %b1ds, %b1zgb : tensor<32x32x112x112xf32>
    %b1pc = stablehlo.convolution(%b1zse, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %b1pbb = stablehlo.broadcast_in_dim %b1pb, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1p = stablehlo.add %b1pc, %b1pbb : tensor<32x16x112x112xf32>
    %b1pnnf = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %b1pnep = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %b1pnsmr = stablehlo.reduce(%b1p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1pnsm = stablehlo.broadcast_in_dim %b1pnsmr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pnmu = stablehlo.divide %b1pnsm, %b1pnnf : tensor<32x16x112x112xf32>
    %b1pnxc = stablehlo.subtract %b1p, %b1pnmu : tensor<32x16x112x112xf32>
    %b1pnsq = stablehlo.multiply %b1pnxc, %b1pnxc : tensor<32x16x112x112xf32>
    %b1pnvsr = stablehlo.reduce(%b1pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1pnvs = stablehlo.broadcast_in_dim %b1pnvsr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pnvr = stablehlo.divide %b1pnvs, %b1pnnf : tensor<32x16x112x112xf32>
    %b1pnve = stablehlo.add %b1pnvr, %b1pnep : tensor<32x16x112x112xf32>
    %b1pnistd = stablehlo.rsqrt %b1pnve : tensor<32x16x112x112xf32>
    %b1pnxh = stablehlo.multiply %b1pnxc, %b1pnistd : tensor<32x16x112x112xf32>
    %b1pngb = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pnbtb = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1pngx = stablehlo.multiply %b1pnxh, %b1pngb : tensor<32x16x112x112xf32>
    %b1pn = stablehlo.add %b1pngx, %b1pnbtb : tensor<32x16x112x112xf32>
    %b2ec = stablehlo.convolution(%b1pn, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %b2ebb = stablehlo.broadcast_in_dim %b2eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2e = stablehlo.add %b2ec, %b2ebb : tensor<32x96x112x112xf32>
    %b2ennf = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %b2enep = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %b2ensmr = stablehlo.reduce(%b2e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ensm = stablehlo.broadcast_in_dim %b2ensmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2enmu = stablehlo.divide %b2ensm, %b2ennf : tensor<32x96x112x112xf32>
    %b2enxc = stablehlo.subtract %b2e, %b2enmu : tensor<32x96x112x112xf32>
    %b2ensq = stablehlo.multiply %b2enxc, %b2enxc : tensor<32x96x112x112xf32>
    %b2envsr = stablehlo.reduce(%b2ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2envs = stablehlo.broadcast_in_dim %b2envsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2envr = stablehlo.divide %b2envs, %b2ennf : tensor<32x96x112x112xf32>
    %b2enve = stablehlo.add %b2envr, %b2enep : tensor<32x96x112x112xf32>
    %b2enistd = stablehlo.rsqrt %b2enve : tensor<32x96x112x112xf32>
    %b2enxh = stablehlo.multiply %b2enxc, %b2enistd : tensor<32x96x112x112xf32>
    %b2engb = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2enbtb = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2engx = stablehlo.multiply %b2enxh, %b2engb : tensor<32x96x112x112xf32>
    %b2en = stablehlo.add %b2engx, %b2enbtb : tensor<32x96x112x112xf32>
    %b2ess = stablehlo.logistic %b2en : tensor<32x96x112x112xf32>
    %b2es = stablehlo.multiply %b2en, %b2ess : tensor<32x96x112x112xf32>
    %b2dc = stablehlo.convolution(%b2es, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %b2dbb = stablehlo.broadcast_in_dim %b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2d = stablehlo.add %b2dc, %b2dbb : tensor<32x96x56x56xf32>
    %b2dnnf = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %b2dnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b2dnsmr = stablehlo.reduce(%b2d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dnsm = stablehlo.broadcast_in_dim %b2dnsmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnmu = stablehlo.divide %b2dnsm, %b2dnnf : tensor<32x96x56x56xf32>
    %b2dnxc = stablehlo.subtract %b2d, %b2dnmu : tensor<32x96x56x56xf32>
    %b2dnsq = stablehlo.multiply %b2dnxc, %b2dnxc : tensor<32x96x56x56xf32>
    %b2dnvsr = stablehlo.reduce(%b2dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dnvs = stablehlo.broadcast_in_dim %b2dnvsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnvr = stablehlo.divide %b2dnvs, %b2dnnf : tensor<32x96x56x56xf32>
    %b2dnve = stablehlo.add %b2dnvr, %b2dnep : tensor<32x96x56x56xf32>
    %b2dnistd = stablehlo.rsqrt %b2dnve : tensor<32x96x56x56xf32>
    %b2dnxh = stablehlo.multiply %b2dnxc, %b2dnistd : tensor<32x96x56x56xf32>
    %b2dngb = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnbtb = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dngx = stablehlo.multiply %b2dnxh, %b2dngb : tensor<32x96x56x56xf32>
    %b2dn = stablehlo.add %b2dngx, %b2dnbtb : tensor<32x96x56x56xf32>
    %b2dss = stablehlo.logistic %b2dn : tensor<32x96x56x56xf32>
    %b2ds = stablehlo.multiply %b2dn, %b2dss : tensor<32x96x56x56xf32>
    %b2zsqs = stablehlo.reduce(%b2ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b2zsqnf = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %b2zsq = stablehlo.divide %b2zsqs, %b2zsqnf : tensor<32x96xf32>
    %b2zexd = stablehlo.dot_general %b2zsq, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %b2zexbb = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %b2zex = stablehlo.add %b2zexd, %b2zexbb : tensor<32x4xf32>
    %b2za1s = stablehlo.logistic %b2zex : tensor<32x4xf32>
    %b2za1 = stablehlo.multiply %b2zex, %b2za1s : tensor<32x4xf32>
    %b2zh2d = stablehlo.dot_general %b2za1, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %b2zh2bb = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %b2zh2 = stablehlo.add %b2zh2d, %b2zh2bb : tensor<32x96xf32>
    %b2zgate = stablehlo.logistic %b2zh2 : tensor<32x96xf32>
    %b2zgb = stablehlo.broadcast_in_dim %b2zgate, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2zse = stablehlo.multiply %b2ds, %b2zgb : tensor<32x96x56x56xf32>
    %b2pc = stablehlo.convolution(%b2zse, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b2pbb = stablehlo.broadcast_in_dim %b2pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2p = stablehlo.add %b2pc, %b2pbb : tensor<32x24x56x56xf32>
    %b2pnnf = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %b2pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b2pnsmr = stablehlo.reduce(%b2p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2pnsm = stablehlo.broadcast_in_dim %b2pnsmr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnmu = stablehlo.divide %b2pnsm, %b2pnnf : tensor<32x24x56x56xf32>
    %b2pnxc = stablehlo.subtract %b2p, %b2pnmu : tensor<32x24x56x56xf32>
    %b2pnsq = stablehlo.multiply %b2pnxc, %b2pnxc : tensor<32x24x56x56xf32>
    %b2pnvsr = stablehlo.reduce(%b2pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2pnvs = stablehlo.broadcast_in_dim %b2pnvsr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnvr = stablehlo.divide %b2pnvs, %b2pnnf : tensor<32x24x56x56xf32>
    %b2pnve = stablehlo.add %b2pnvr, %b2pnep : tensor<32x24x56x56xf32>
    %b2pnistd = stablehlo.rsqrt %b2pnve : tensor<32x24x56x56xf32>
    %b2pnxh = stablehlo.multiply %b2pnxc, %b2pnistd : tensor<32x24x56x56xf32>
    %b2pngb = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnbtb = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pngx = stablehlo.multiply %b2pnxh, %b2pngb : tensor<32x24x56x56xf32>
    %b2pn = stablehlo.add %b2pngx, %b2pnbtb : tensor<32x24x56x56xf32>
    %b3ec = stablehlo.convolution(%b2pn, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %b3ebb = stablehlo.broadcast_in_dim %b3eb, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3e = stablehlo.add %b3ec, %b3ebb : tensor<32x144x56x56xf32>
    %b3ennf = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %b3enep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b3ensmr = stablehlo.reduce(%b3e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ensm = stablehlo.broadcast_in_dim %b3ensmr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3enmu = stablehlo.divide %b3ensm, %b3ennf : tensor<32x144x56x56xf32>
    %b3enxc = stablehlo.subtract %b3e, %b3enmu : tensor<32x144x56x56xf32>
    %b3ensq = stablehlo.multiply %b3enxc, %b3enxc : tensor<32x144x56x56xf32>
    %b3envsr = stablehlo.reduce(%b3ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3envs = stablehlo.broadcast_in_dim %b3envsr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3envr = stablehlo.divide %b3envs, %b3ennf : tensor<32x144x56x56xf32>
    %b3enve = stablehlo.add %b3envr, %b3enep : tensor<32x144x56x56xf32>
    %b3enistd = stablehlo.rsqrt %b3enve : tensor<32x144x56x56xf32>
    %b3enxh = stablehlo.multiply %b3enxc, %b3enistd : tensor<32x144x56x56xf32>
    %b3engb = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3enbtb = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3engx = stablehlo.multiply %b3enxh, %b3engb : tensor<32x144x56x56xf32>
    %b3en = stablehlo.add %b3engx, %b3enbtb : tensor<32x144x56x56xf32>
    %b3ess = stablehlo.logistic %b3en : tensor<32x144x56x56xf32>
    %b3es = stablehlo.multiply %b3en, %b3ess : tensor<32x144x56x56xf32>
    %b3dc = stablehlo.convolution(%b3es, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %b3dbb = stablehlo.broadcast_in_dim %b3db, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3d = stablehlo.add %b3dc, %b3dbb : tensor<32x144x56x56xf32>
    %b3dnnf = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %b3dnep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b3dnsmr = stablehlo.reduce(%b3d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dnsm = stablehlo.broadcast_in_dim %b3dnsmr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dnmu = stablehlo.divide %b3dnsm, %b3dnnf : tensor<32x144x56x56xf32>
    %b3dnxc = stablehlo.subtract %b3d, %b3dnmu : tensor<32x144x56x56xf32>
    %b3dnsq = stablehlo.multiply %b3dnxc, %b3dnxc : tensor<32x144x56x56xf32>
    %b3dnvsr = stablehlo.reduce(%b3dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dnvs = stablehlo.broadcast_in_dim %b3dnvsr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dnvr = stablehlo.divide %b3dnvs, %b3dnnf : tensor<32x144x56x56xf32>
    %b3dnve = stablehlo.add %b3dnvr, %b3dnep : tensor<32x144x56x56xf32>
    %b3dnistd = stablehlo.rsqrt %b3dnve : tensor<32x144x56x56xf32>
    %b3dnxh = stablehlo.multiply %b3dnxc, %b3dnistd : tensor<32x144x56x56xf32>
    %b3dngb = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dnbtb = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dngx = stablehlo.multiply %b3dnxh, %b3dngb : tensor<32x144x56x56xf32>
    %b3dn = stablehlo.add %b3dngx, %b3dnbtb : tensor<32x144x56x56xf32>
    %b3dss = stablehlo.logistic %b3dn : tensor<32x144x56x56xf32>
    %b3ds = stablehlo.multiply %b3dn, %b3dss : tensor<32x144x56x56xf32>
    %b3zsqs = stablehlo.reduce(%b3ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %b3zsqnf = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %b3zsq = stablehlo.divide %b3zsqs, %b3zsqnf : tensor<32x144xf32>
    %b3zexd = stablehlo.dot_general %b3zsq, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %b3zexbb = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %b3zex = stablehlo.add %b3zexd, %b3zexbb : tensor<32x6xf32>
    %b3za1s = stablehlo.logistic %b3zex : tensor<32x6xf32>
    %b3za1 = stablehlo.multiply %b3zex, %b3za1s : tensor<32x6xf32>
    %b3zh2d = stablehlo.dot_general %b3za1, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %b3zh2bb = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %b3zh2 = stablehlo.add %b3zh2d, %b3zh2bb : tensor<32x144xf32>
    %b3zgate = stablehlo.logistic %b3zh2 : tensor<32x144xf32>
    %b3zgb = stablehlo.broadcast_in_dim %b3zgate, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %b3zse = stablehlo.multiply %b3ds, %b3zgb : tensor<32x144x56x56xf32>
    %b3pc = stablehlo.convolution(%b3zse, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b3pbb = stablehlo.broadcast_in_dim %b3pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3p = stablehlo.add %b3pc, %b3pbb : tensor<32x24x56x56xf32>
    %b3pnnf = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %b3pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b3pnsmr = stablehlo.reduce(%b3p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3pnsm = stablehlo.broadcast_in_dim %b3pnsmr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pnmu = stablehlo.divide %b3pnsm, %b3pnnf : tensor<32x24x56x56xf32>
    %b3pnxc = stablehlo.subtract %b3p, %b3pnmu : tensor<32x24x56x56xf32>
    %b3pnsq = stablehlo.multiply %b3pnxc, %b3pnxc : tensor<32x24x56x56xf32>
    %b3pnvsr = stablehlo.reduce(%b3pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3pnvs = stablehlo.broadcast_in_dim %b3pnvsr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pnvr = stablehlo.divide %b3pnvs, %b3pnnf : tensor<32x24x56x56xf32>
    %b3pnve = stablehlo.add %b3pnvr, %b3pnep : tensor<32x24x56x56xf32>
    %b3pnistd = stablehlo.rsqrt %b3pnve : tensor<32x24x56x56xf32>
    %b3pnxh = stablehlo.multiply %b3pnxc, %b3pnistd : tensor<32x24x56x56xf32>
    %b3pngb = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pnbtb = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3pngx = stablehlo.multiply %b3pnxh, %b3pngb : tensor<32x24x56x56xf32>
    %b3pn = stablehlo.add %b3pngx, %b3pnbtb : tensor<32x24x56x56xf32>
    %b3o = stablehlo.add %b3pn, %b2pn : tensor<32x24x56x56xf32>
    %b4ec = stablehlo.convolution(%b3o, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %b4ebb = stablehlo.broadcast_in_dim %b4eb, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4e = stablehlo.add %b4ec, %b4ebb : tensor<32x144x56x56xf32>
    %b4ennf = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %b4enep = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %b4ensmr = stablehlo.reduce(%b4e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ensm = stablehlo.broadcast_in_dim %b4ensmr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4enmu = stablehlo.divide %b4ensm, %b4ennf : tensor<32x144x56x56xf32>
    %b4enxc = stablehlo.subtract %b4e, %b4enmu : tensor<32x144x56x56xf32>
    %b4ensq = stablehlo.multiply %b4enxc, %b4enxc : tensor<32x144x56x56xf32>
    %b4envsr = stablehlo.reduce(%b4ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4envs = stablehlo.broadcast_in_dim %b4envsr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4envr = stablehlo.divide %b4envs, %b4ennf : tensor<32x144x56x56xf32>
    %b4enve = stablehlo.add %b4envr, %b4enep : tensor<32x144x56x56xf32>
    %b4enistd = stablehlo.rsqrt %b4enve : tensor<32x144x56x56xf32>
    %b4enxh = stablehlo.multiply %b4enxc, %b4enistd : tensor<32x144x56x56xf32>
    %b4engb = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4enbtb = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4engx = stablehlo.multiply %b4enxh, %b4engb : tensor<32x144x56x56xf32>
    %b4en = stablehlo.add %b4engx, %b4enbtb : tensor<32x144x56x56xf32>
    %b4ess = stablehlo.logistic %b4en : tensor<32x144x56x56xf32>
    %b4es = stablehlo.multiply %b4en, %b4ess : tensor<32x144x56x56xf32>
    %b4dc = stablehlo.convolution(%b4es, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %b4dbb = stablehlo.broadcast_in_dim %b4db, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4d = stablehlo.add %b4dc, %b4dbb : tensor<32x144x28x28xf32>
    %b4dnnf = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %b4dnep = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %b4dnsmr = stablehlo.reduce(%b4d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4dnsm = stablehlo.broadcast_in_dim %b4dnsmr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dnmu = stablehlo.divide %b4dnsm, %b4dnnf : tensor<32x144x28x28xf32>
    %b4dnxc = stablehlo.subtract %b4d, %b4dnmu : tensor<32x144x28x28xf32>
    %b4dnsq = stablehlo.multiply %b4dnxc, %b4dnxc : tensor<32x144x28x28xf32>
    %b4dnvsr = stablehlo.reduce(%b4dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4dnvs = stablehlo.broadcast_in_dim %b4dnvsr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dnvr = stablehlo.divide %b4dnvs, %b4dnnf : tensor<32x144x28x28xf32>
    %b4dnve = stablehlo.add %b4dnvr, %b4dnep : tensor<32x144x28x28xf32>
    %b4dnistd = stablehlo.rsqrt %b4dnve : tensor<32x144x28x28xf32>
    %b4dnxh = stablehlo.multiply %b4dnxc, %b4dnistd : tensor<32x144x28x28xf32>
    %b4dngb = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dnbtb = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4dngx = stablehlo.multiply %b4dnxh, %b4dngb : tensor<32x144x28x28xf32>
    %b4dn = stablehlo.add %b4dngx, %b4dnbtb : tensor<32x144x28x28xf32>
    %b4dss = stablehlo.logistic %b4dn : tensor<32x144x28x28xf32>
    %b4ds = stablehlo.multiply %b4dn, %b4dss : tensor<32x144x28x28xf32>
    %b4zsqs = stablehlo.reduce(%b4ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %b4zsqnf = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %b4zsq = stablehlo.divide %b4zsqs, %b4zsqnf : tensor<32x144xf32>
    %b4zexd = stablehlo.dot_general %b4zsq, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %b4zexbb = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %b4zex = stablehlo.add %b4zexd, %b4zexbb : tensor<32x6xf32>
    %b4za1s = stablehlo.logistic %b4zex : tensor<32x6xf32>
    %b4za1 = stablehlo.multiply %b4zex, %b4za1s : tensor<32x6xf32>
    %b4zh2d = stablehlo.dot_general %b4za1, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %b4zh2bb = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %b4zh2 = stablehlo.add %b4zh2d, %b4zh2bb : tensor<32x144xf32>
    %b4zgate = stablehlo.logistic %b4zh2 : tensor<32x144xf32>
    %b4zgb = stablehlo.broadcast_in_dim %b4zgate, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %b4zse = stablehlo.multiply %b4ds, %b4zgb : tensor<32x144x28x28xf32>
    %b4pc = stablehlo.convolution(%b4zse, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %b4pbb = stablehlo.broadcast_in_dim %b4pb, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4p = stablehlo.add %b4pc, %b4pbb : tensor<32x40x28x28xf32>
    %b4pnnf = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %b4pnep = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %b4pnsmr = stablehlo.reduce(%b4p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4pnsm = stablehlo.broadcast_in_dim %b4pnsmr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4pnmu = stablehlo.divide %b4pnsm, %b4pnnf : tensor<32x40x28x28xf32>
    %b4pnxc = stablehlo.subtract %b4p, %b4pnmu : tensor<32x40x28x28xf32>
    %b4pnsq = stablehlo.multiply %b4pnxc, %b4pnxc : tensor<32x40x28x28xf32>
    %b4pnvsr = stablehlo.reduce(%b4pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4pnvs = stablehlo.broadcast_in_dim %b4pnvsr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4pnvr = stablehlo.divide %b4pnvs, %b4pnnf : tensor<32x40x28x28xf32>
    %b4pnve = stablehlo.add %b4pnvr, %b4pnep : tensor<32x40x28x28xf32>
    %b4pnistd = stablehlo.rsqrt %b4pnve : tensor<32x40x28x28xf32>
    %b4pnxh = stablehlo.multiply %b4pnxc, %b4pnistd : tensor<32x40x28x28xf32>
    %b4pngb = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4pnbtb = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4pngx = stablehlo.multiply %b4pnxh, %b4pngb : tensor<32x40x28x28xf32>
    %b4pn = stablehlo.add %b4pngx, %b4pnbtb : tensor<32x40x28x28xf32>
    %b5ec = stablehlo.convolution(%b4pn, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %b5ebb = stablehlo.broadcast_in_dim %b5eb, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5e = stablehlo.add %b5ec, %b5ebb : tensor<32x240x28x28xf32>
    %b5ennf = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %b5enep = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %b5ensmr = stablehlo.reduce(%b5e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ensm = stablehlo.broadcast_in_dim %b5ensmr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5enmu = stablehlo.divide %b5ensm, %b5ennf : tensor<32x240x28x28xf32>
    %b5enxc = stablehlo.subtract %b5e, %b5enmu : tensor<32x240x28x28xf32>
    %b5ensq = stablehlo.multiply %b5enxc, %b5enxc : tensor<32x240x28x28xf32>
    %b5envsr = stablehlo.reduce(%b5ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5envs = stablehlo.broadcast_in_dim %b5envsr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5envr = stablehlo.divide %b5envs, %b5ennf : tensor<32x240x28x28xf32>
    %b5enve = stablehlo.add %b5envr, %b5enep : tensor<32x240x28x28xf32>
    %b5enistd = stablehlo.rsqrt %b5enve : tensor<32x240x28x28xf32>
    %b5enxh = stablehlo.multiply %b5enxc, %b5enistd : tensor<32x240x28x28xf32>
    %b5engb = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5enbtb = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5engx = stablehlo.multiply %b5enxh, %b5engb : tensor<32x240x28x28xf32>
    %b5en = stablehlo.add %b5engx, %b5enbtb : tensor<32x240x28x28xf32>
    %b5ess = stablehlo.logistic %b5en : tensor<32x240x28x28xf32>
    %b5es = stablehlo.multiply %b5en, %b5ess : tensor<32x240x28x28xf32>
    %b5dc = stablehlo.convolution(%b5es, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %b5dbb = stablehlo.broadcast_in_dim %b5db, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5d = stablehlo.add %b5dc, %b5dbb : tensor<32x240x28x28xf32>
    %b5dnnf = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %b5dnep = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %b5dnsmr = stablehlo.reduce(%b5d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dnsm = stablehlo.broadcast_in_dim %b5dnsmr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dnmu = stablehlo.divide %b5dnsm, %b5dnnf : tensor<32x240x28x28xf32>
    %b5dnxc = stablehlo.subtract %b5d, %b5dnmu : tensor<32x240x28x28xf32>
    %b5dnsq = stablehlo.multiply %b5dnxc, %b5dnxc : tensor<32x240x28x28xf32>
    %b5dnvsr = stablehlo.reduce(%b5dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dnvs = stablehlo.broadcast_in_dim %b5dnvsr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dnvr = stablehlo.divide %b5dnvs, %b5dnnf : tensor<32x240x28x28xf32>
    %b5dnve = stablehlo.add %b5dnvr, %b5dnep : tensor<32x240x28x28xf32>
    %b5dnistd = stablehlo.rsqrt %b5dnve : tensor<32x240x28x28xf32>
    %b5dnxh = stablehlo.multiply %b5dnxc, %b5dnistd : tensor<32x240x28x28xf32>
    %b5dngb = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dnbtb = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dngx = stablehlo.multiply %b5dnxh, %b5dngb : tensor<32x240x28x28xf32>
    %b5dn = stablehlo.add %b5dngx, %b5dnbtb : tensor<32x240x28x28xf32>
    %b5dss = stablehlo.logistic %b5dn : tensor<32x240x28x28xf32>
    %b5ds = stablehlo.multiply %b5dn, %b5dss : tensor<32x240x28x28xf32>
    %b5zsqs = stablehlo.reduce(%b5ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %b5zsqnf = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %b5zsq = stablehlo.divide %b5zsqs, %b5zsqnf : tensor<32x240xf32>
    %b5zexd = stablehlo.dot_general %b5zsq, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %b5zexbb = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %b5zex = stablehlo.add %b5zexd, %b5zexbb : tensor<32x10xf32>
    %b5za1s = stablehlo.logistic %b5zex : tensor<32x10xf32>
    %b5za1 = stablehlo.multiply %b5zex, %b5za1s : tensor<32x10xf32>
    %b5zh2d = stablehlo.dot_general %b5za1, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %b5zh2bb = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %b5zh2 = stablehlo.add %b5zh2d, %b5zh2bb : tensor<32x240xf32>
    %b5zgate = stablehlo.logistic %b5zh2 : tensor<32x240xf32>
    %b5zgb = stablehlo.broadcast_in_dim %b5zgate, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %b5zse = stablehlo.multiply %b5ds, %b5zgb : tensor<32x240x28x28xf32>
    %b5pc = stablehlo.convolution(%b5zse, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %b5pbb = stablehlo.broadcast_in_dim %b5pb, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5p = stablehlo.add %b5pc, %b5pbb : tensor<32x40x28x28xf32>
    %b5pnnf = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %b5pnep = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %b5pnsmr = stablehlo.reduce(%b5p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5pnsm = stablehlo.broadcast_in_dim %b5pnsmr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5pnmu = stablehlo.divide %b5pnsm, %b5pnnf : tensor<32x40x28x28xf32>
    %b5pnxc = stablehlo.subtract %b5p, %b5pnmu : tensor<32x40x28x28xf32>
    %b5pnsq = stablehlo.multiply %b5pnxc, %b5pnxc : tensor<32x40x28x28xf32>
    %b5pnvsr = stablehlo.reduce(%b5pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5pnvs = stablehlo.broadcast_in_dim %b5pnvsr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5pnvr = stablehlo.divide %b5pnvs, %b5pnnf : tensor<32x40x28x28xf32>
    %b5pnve = stablehlo.add %b5pnvr, %b5pnep : tensor<32x40x28x28xf32>
    %b5pnistd = stablehlo.rsqrt %b5pnve : tensor<32x40x28x28xf32>
    %b5pnxh = stablehlo.multiply %b5pnxc, %b5pnistd : tensor<32x40x28x28xf32>
    %b5pngb = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5pnbtb = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5pngx = stablehlo.multiply %b5pnxh, %b5pngb : tensor<32x40x28x28xf32>
    %b5pn = stablehlo.add %b5pngx, %b5pnbtb : tensor<32x40x28x28xf32>
    %b5o = stablehlo.add %b5pn, %b4pn : tensor<32x40x28x28xf32>
    %b6ec = stablehlo.convolution(%b5o, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %b6ebb = stablehlo.broadcast_in_dim %b6eb, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6e = stablehlo.add %b6ec, %b6ebb : tensor<32x240x28x28xf32>
    %b6ennf = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %b6enep = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %b6ensmr = stablehlo.reduce(%b6e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ensm = stablehlo.broadcast_in_dim %b6ensmr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6enmu = stablehlo.divide %b6ensm, %b6ennf : tensor<32x240x28x28xf32>
    %b6enxc = stablehlo.subtract %b6e, %b6enmu : tensor<32x240x28x28xf32>
    %b6ensq = stablehlo.multiply %b6enxc, %b6enxc : tensor<32x240x28x28xf32>
    %b6envsr = stablehlo.reduce(%b6ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6envs = stablehlo.broadcast_in_dim %b6envsr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6envr = stablehlo.divide %b6envs, %b6ennf : tensor<32x240x28x28xf32>
    %b6enve = stablehlo.add %b6envr, %b6enep : tensor<32x240x28x28xf32>
    %b6enistd = stablehlo.rsqrt %b6enve : tensor<32x240x28x28xf32>
    %b6enxh = stablehlo.multiply %b6enxc, %b6enistd : tensor<32x240x28x28xf32>
    %b6engb = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6enbtb = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6engx = stablehlo.multiply %b6enxh, %b6engb : tensor<32x240x28x28xf32>
    %b6en = stablehlo.add %b6engx, %b6enbtb : tensor<32x240x28x28xf32>
    %b6ess = stablehlo.logistic %b6en : tensor<32x240x28x28xf32>
    %b6es = stablehlo.multiply %b6en, %b6ess : tensor<32x240x28x28xf32>
    %b6dc = stablehlo.convolution(%b6es, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %b6dbb = stablehlo.broadcast_in_dim %b6db, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6d = stablehlo.add %b6dc, %b6dbb : tensor<32x240x14x14xf32>
    %b6dnnf = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %b6dnep = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %b6dnsmr = stablehlo.reduce(%b6d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6dnsm = stablehlo.broadcast_in_dim %b6dnsmr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6dnmu = stablehlo.divide %b6dnsm, %b6dnnf : tensor<32x240x14x14xf32>
    %b6dnxc = stablehlo.subtract %b6d, %b6dnmu : tensor<32x240x14x14xf32>
    %b6dnsq = stablehlo.multiply %b6dnxc, %b6dnxc : tensor<32x240x14x14xf32>
    %b6dnvsr = stablehlo.reduce(%b6dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6dnvs = stablehlo.broadcast_in_dim %b6dnvsr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6dnvr = stablehlo.divide %b6dnvs, %b6dnnf : tensor<32x240x14x14xf32>
    %b6dnve = stablehlo.add %b6dnvr, %b6dnep : tensor<32x240x14x14xf32>
    %b6dnistd = stablehlo.rsqrt %b6dnve : tensor<32x240x14x14xf32>
    %b6dnxh = stablehlo.multiply %b6dnxc, %b6dnistd : tensor<32x240x14x14xf32>
    %b6dngb = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6dnbtb = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6dngx = stablehlo.multiply %b6dnxh, %b6dngb : tensor<32x240x14x14xf32>
    %b6dn = stablehlo.add %b6dngx, %b6dnbtb : tensor<32x240x14x14xf32>
    %b6dss = stablehlo.logistic %b6dn : tensor<32x240x14x14xf32>
    %b6ds = stablehlo.multiply %b6dn, %b6dss : tensor<32x240x14x14xf32>
    %b6zsqs = stablehlo.reduce(%b6ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %b6zsqnf = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %b6zsq = stablehlo.divide %b6zsqs, %b6zsqnf : tensor<32x240xf32>
    %b6zexd = stablehlo.dot_general %b6zsq, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %b6zexbb = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %b6zex = stablehlo.add %b6zexd, %b6zexbb : tensor<32x10xf32>
    %b6za1s = stablehlo.logistic %b6zex : tensor<32x10xf32>
    %b6za1 = stablehlo.multiply %b6zex, %b6za1s : tensor<32x10xf32>
    %b6zh2d = stablehlo.dot_general %b6za1, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %b6zh2bb = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %b6zh2 = stablehlo.add %b6zh2d, %b6zh2bb : tensor<32x240xf32>
    %b6zgate = stablehlo.logistic %b6zh2 : tensor<32x240xf32>
    %b6zgb = stablehlo.broadcast_in_dim %b6zgate, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %b6zse = stablehlo.multiply %b6ds, %b6zgb : tensor<32x240x14x14xf32>
    %b6pc = stablehlo.convolution(%b6zse, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b6pbb = stablehlo.broadcast_in_dim %b6pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6p = stablehlo.add %b6pc, %b6pbb : tensor<32x80x14x14xf32>
    %b6pnnf = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %b6pnep = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %b6pnsmr = stablehlo.reduce(%b6p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6pnsm = stablehlo.broadcast_in_dim %b6pnsmr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6pnmu = stablehlo.divide %b6pnsm, %b6pnnf : tensor<32x80x14x14xf32>
    %b6pnxc = stablehlo.subtract %b6p, %b6pnmu : tensor<32x80x14x14xf32>
    %b6pnsq = stablehlo.multiply %b6pnxc, %b6pnxc : tensor<32x80x14x14xf32>
    %b6pnvsr = stablehlo.reduce(%b6pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6pnvs = stablehlo.broadcast_in_dim %b6pnvsr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6pnvr = stablehlo.divide %b6pnvs, %b6pnnf : tensor<32x80x14x14xf32>
    %b6pnve = stablehlo.add %b6pnvr, %b6pnep : tensor<32x80x14x14xf32>
    %b6pnistd = stablehlo.rsqrt %b6pnve : tensor<32x80x14x14xf32>
    %b6pnxh = stablehlo.multiply %b6pnxc, %b6pnistd : tensor<32x80x14x14xf32>
    %b6pngb = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6pnbtb = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6pngx = stablehlo.multiply %b6pnxh, %b6pngb : tensor<32x80x14x14xf32>
    %b6pn = stablehlo.add %b6pngx, %b6pnbtb : tensor<32x80x14x14xf32>
    %b7ec = stablehlo.convolution(%b6pn, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b7ebb = stablehlo.broadcast_in_dim %b7eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7e = stablehlo.add %b7ec, %b7ebb : tensor<32x480x14x14xf32>
    %b7ennf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b7enep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b7ensmr = stablehlo.reduce(%b7e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ensm = stablehlo.broadcast_in_dim %b7ensmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7enmu = stablehlo.divide %b7ensm, %b7ennf : tensor<32x480x14x14xf32>
    %b7enxc = stablehlo.subtract %b7e, %b7enmu : tensor<32x480x14x14xf32>
    %b7ensq = stablehlo.multiply %b7enxc, %b7enxc : tensor<32x480x14x14xf32>
    %b7envsr = stablehlo.reduce(%b7ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7envs = stablehlo.broadcast_in_dim %b7envsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7envr = stablehlo.divide %b7envs, %b7ennf : tensor<32x480x14x14xf32>
    %b7enve = stablehlo.add %b7envr, %b7enep : tensor<32x480x14x14xf32>
    %b7enistd = stablehlo.rsqrt %b7enve : tensor<32x480x14x14xf32>
    %b7enxh = stablehlo.multiply %b7enxc, %b7enistd : tensor<32x480x14x14xf32>
    %b7engb = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7enbtb = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7engx = stablehlo.multiply %b7enxh, %b7engb : tensor<32x480x14x14xf32>
    %b7en = stablehlo.add %b7engx, %b7enbtb : tensor<32x480x14x14xf32>
    %b7ess = stablehlo.logistic %b7en : tensor<32x480x14x14xf32>
    %b7es = stablehlo.multiply %b7en, %b7ess : tensor<32x480x14x14xf32>
    %b7dc = stablehlo.convolution(%b7es, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %b7dbb = stablehlo.broadcast_in_dim %b7db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7d = stablehlo.add %b7dc, %b7dbb : tensor<32x480x14x14xf32>
    %b7dnnf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b7dnep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b7dnsmr = stablehlo.reduce(%b7d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7dnsm = stablehlo.broadcast_in_dim %b7dnsmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dnmu = stablehlo.divide %b7dnsm, %b7dnnf : tensor<32x480x14x14xf32>
    %b7dnxc = stablehlo.subtract %b7d, %b7dnmu : tensor<32x480x14x14xf32>
    %b7dnsq = stablehlo.multiply %b7dnxc, %b7dnxc : tensor<32x480x14x14xf32>
    %b7dnvsr = stablehlo.reduce(%b7dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7dnvs = stablehlo.broadcast_in_dim %b7dnvsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dnvr = stablehlo.divide %b7dnvs, %b7dnnf : tensor<32x480x14x14xf32>
    %b7dnve = stablehlo.add %b7dnvr, %b7dnep : tensor<32x480x14x14xf32>
    %b7dnistd = stablehlo.rsqrt %b7dnve : tensor<32x480x14x14xf32>
    %b7dnxh = stablehlo.multiply %b7dnxc, %b7dnistd : tensor<32x480x14x14xf32>
    %b7dngb = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dnbtb = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dngx = stablehlo.multiply %b7dnxh, %b7dngb : tensor<32x480x14x14xf32>
    %b7dn = stablehlo.add %b7dngx, %b7dnbtb : tensor<32x480x14x14xf32>
    %b7dss = stablehlo.logistic %b7dn : tensor<32x480x14x14xf32>
    %b7ds = stablehlo.multiply %b7dn, %b7dss : tensor<32x480x14x14xf32>
    %b7zsqs = stablehlo.reduce(%b7ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b7zsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b7zsq = stablehlo.divide %b7zsqs, %b7zsqnf : tensor<32x480xf32>
    %b7zexd = stablehlo.dot_general %b7zsq, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %b7zexbb = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %b7zex = stablehlo.add %b7zexd, %b7zexbb : tensor<32x20xf32>
    %b7za1s = stablehlo.logistic %b7zex : tensor<32x20xf32>
    %b7za1 = stablehlo.multiply %b7zex, %b7za1s : tensor<32x20xf32>
    %b7zh2d = stablehlo.dot_general %b7za1, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %b7zh2bb = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %b7zh2 = stablehlo.add %b7zh2d, %b7zh2bb : tensor<32x480xf32>
    %b7zgate = stablehlo.logistic %b7zh2 : tensor<32x480xf32>
    %b7zgb = stablehlo.broadcast_in_dim %b7zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b7zse = stablehlo.multiply %b7ds, %b7zgb : tensor<32x480x14x14xf32>
    %b7pc = stablehlo.convolution(%b7zse, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b7pbb = stablehlo.broadcast_in_dim %b7pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7p = stablehlo.add %b7pc, %b7pbb : tensor<32x80x14x14xf32>
    %b7pnnf = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %b7pnep = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %b7pnsmr = stablehlo.reduce(%b7p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7pnsm = stablehlo.broadcast_in_dim %b7pnsmr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7pnmu = stablehlo.divide %b7pnsm, %b7pnnf : tensor<32x80x14x14xf32>
    %b7pnxc = stablehlo.subtract %b7p, %b7pnmu : tensor<32x80x14x14xf32>
    %b7pnsq = stablehlo.multiply %b7pnxc, %b7pnxc : tensor<32x80x14x14xf32>
    %b7pnvsr = stablehlo.reduce(%b7pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7pnvs = stablehlo.broadcast_in_dim %b7pnvsr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7pnvr = stablehlo.divide %b7pnvs, %b7pnnf : tensor<32x80x14x14xf32>
    %b7pnve = stablehlo.add %b7pnvr, %b7pnep : tensor<32x80x14x14xf32>
    %b7pnistd = stablehlo.rsqrt %b7pnve : tensor<32x80x14x14xf32>
    %b7pnxh = stablehlo.multiply %b7pnxc, %b7pnistd : tensor<32x80x14x14xf32>
    %b7pngb = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7pnbtb = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7pngx = stablehlo.multiply %b7pnxh, %b7pngb : tensor<32x80x14x14xf32>
    %b7pn = stablehlo.add %b7pngx, %b7pnbtb : tensor<32x80x14x14xf32>
    %b7o = stablehlo.add %b7pn, %b6pn : tensor<32x80x14x14xf32>
    %b8ec = stablehlo.convolution(%b7o, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b8ebb = stablehlo.broadcast_in_dim %b8eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8e = stablehlo.add %b8ec, %b8ebb : tensor<32x480x14x14xf32>
    %b8ennf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b8enep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b8ensmr = stablehlo.reduce(%b8e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ensm = stablehlo.broadcast_in_dim %b8ensmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8enmu = stablehlo.divide %b8ensm, %b8ennf : tensor<32x480x14x14xf32>
    %b8enxc = stablehlo.subtract %b8e, %b8enmu : tensor<32x480x14x14xf32>
    %b8ensq = stablehlo.multiply %b8enxc, %b8enxc : tensor<32x480x14x14xf32>
    %b8envsr = stablehlo.reduce(%b8ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8envs = stablehlo.broadcast_in_dim %b8envsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8envr = stablehlo.divide %b8envs, %b8ennf : tensor<32x480x14x14xf32>
    %b8enve = stablehlo.add %b8envr, %b8enep : tensor<32x480x14x14xf32>
    %b8enistd = stablehlo.rsqrt %b8enve : tensor<32x480x14x14xf32>
    %b8enxh = stablehlo.multiply %b8enxc, %b8enistd : tensor<32x480x14x14xf32>
    %b8engb = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8enbtb = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8engx = stablehlo.multiply %b8enxh, %b8engb : tensor<32x480x14x14xf32>
    %b8en = stablehlo.add %b8engx, %b8enbtb : tensor<32x480x14x14xf32>
    %b8ess = stablehlo.logistic %b8en : tensor<32x480x14x14xf32>
    %b8es = stablehlo.multiply %b8en, %b8ess : tensor<32x480x14x14xf32>
    %b8dc = stablehlo.convolution(%b8es, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %b8dbb = stablehlo.broadcast_in_dim %b8db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8d = stablehlo.add %b8dc, %b8dbb : tensor<32x480x14x14xf32>
    %b8dnnf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b8dnep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b8dnsmr = stablehlo.reduce(%b8d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dnsm = stablehlo.broadcast_in_dim %b8dnsmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dnmu = stablehlo.divide %b8dnsm, %b8dnnf : tensor<32x480x14x14xf32>
    %b8dnxc = stablehlo.subtract %b8d, %b8dnmu : tensor<32x480x14x14xf32>
    %b8dnsq = stablehlo.multiply %b8dnxc, %b8dnxc : tensor<32x480x14x14xf32>
    %b8dnvsr = stablehlo.reduce(%b8dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dnvs = stablehlo.broadcast_in_dim %b8dnvsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dnvr = stablehlo.divide %b8dnvs, %b8dnnf : tensor<32x480x14x14xf32>
    %b8dnve = stablehlo.add %b8dnvr, %b8dnep : tensor<32x480x14x14xf32>
    %b8dnistd = stablehlo.rsqrt %b8dnve : tensor<32x480x14x14xf32>
    %b8dnxh = stablehlo.multiply %b8dnxc, %b8dnistd : tensor<32x480x14x14xf32>
    %b8dngb = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dnbtb = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dngx = stablehlo.multiply %b8dnxh, %b8dngb : tensor<32x480x14x14xf32>
    %b8dn = stablehlo.add %b8dngx, %b8dnbtb : tensor<32x480x14x14xf32>
    %b8dss = stablehlo.logistic %b8dn : tensor<32x480x14x14xf32>
    %b8ds = stablehlo.multiply %b8dn, %b8dss : tensor<32x480x14x14xf32>
    %b8zsqs = stablehlo.reduce(%b8ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b8zsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b8zsq = stablehlo.divide %b8zsqs, %b8zsqnf : tensor<32x480xf32>
    %b8zexd = stablehlo.dot_general %b8zsq, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %b8zexbb = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %b8zex = stablehlo.add %b8zexd, %b8zexbb : tensor<32x20xf32>
    %b8za1s = stablehlo.logistic %b8zex : tensor<32x20xf32>
    %b8za1 = stablehlo.multiply %b8zex, %b8za1s : tensor<32x20xf32>
    %b8zh2d = stablehlo.dot_general %b8za1, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %b8zh2bb = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %b8zh2 = stablehlo.add %b8zh2d, %b8zh2bb : tensor<32x480xf32>
    %b8zgate = stablehlo.logistic %b8zh2 : tensor<32x480xf32>
    %b8zgb = stablehlo.broadcast_in_dim %b8zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b8zse = stablehlo.multiply %b8ds, %b8zgb : tensor<32x480x14x14xf32>
    %b8pc = stablehlo.convolution(%b8zse, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b8pbb = stablehlo.broadcast_in_dim %b8pb, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8p = stablehlo.add %b8pc, %b8pbb : tensor<32x80x14x14xf32>
    %b8pnnf = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %b8pnep = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %b8pnsmr = stablehlo.reduce(%b8p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8pnsm = stablehlo.broadcast_in_dim %b8pnsmr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8pnmu = stablehlo.divide %b8pnsm, %b8pnnf : tensor<32x80x14x14xf32>
    %b8pnxc = stablehlo.subtract %b8p, %b8pnmu : tensor<32x80x14x14xf32>
    %b8pnsq = stablehlo.multiply %b8pnxc, %b8pnxc : tensor<32x80x14x14xf32>
    %b8pnvsr = stablehlo.reduce(%b8pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8pnvs = stablehlo.broadcast_in_dim %b8pnvsr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8pnvr = stablehlo.divide %b8pnvs, %b8pnnf : tensor<32x80x14x14xf32>
    %b8pnve = stablehlo.add %b8pnvr, %b8pnep : tensor<32x80x14x14xf32>
    %b8pnistd = stablehlo.rsqrt %b8pnve : tensor<32x80x14x14xf32>
    %b8pnxh = stablehlo.multiply %b8pnxc, %b8pnistd : tensor<32x80x14x14xf32>
    %b8pngb = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8pnbtb = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8pngx = stablehlo.multiply %b8pnxh, %b8pngb : tensor<32x80x14x14xf32>
    %b8pn = stablehlo.add %b8pngx, %b8pnbtb : tensor<32x80x14x14xf32>
    %b8o = stablehlo.add %b8pn, %b7o : tensor<32x80x14x14xf32>
    %b9ec = stablehlo.convolution(%b8o, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b9ebb = stablehlo.broadcast_in_dim %b9eb, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9e = stablehlo.add %b9ec, %b9ebb : tensor<32x480x14x14xf32>
    %b9ennf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b9enep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b9ensmr = stablehlo.reduce(%b9e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ensm = stablehlo.broadcast_in_dim %b9ensmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9enmu = stablehlo.divide %b9ensm, %b9ennf : tensor<32x480x14x14xf32>
    %b9enxc = stablehlo.subtract %b9e, %b9enmu : tensor<32x480x14x14xf32>
    %b9ensq = stablehlo.multiply %b9enxc, %b9enxc : tensor<32x480x14x14xf32>
    %b9envsr = stablehlo.reduce(%b9ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9envs = stablehlo.broadcast_in_dim %b9envsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9envr = stablehlo.divide %b9envs, %b9ennf : tensor<32x480x14x14xf32>
    %b9enve = stablehlo.add %b9envr, %b9enep : tensor<32x480x14x14xf32>
    %b9enistd = stablehlo.rsqrt %b9enve : tensor<32x480x14x14xf32>
    %b9enxh = stablehlo.multiply %b9enxc, %b9enistd : tensor<32x480x14x14xf32>
    %b9engb = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9enbtb = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9engx = stablehlo.multiply %b9enxh, %b9engb : tensor<32x480x14x14xf32>
    %b9en = stablehlo.add %b9engx, %b9enbtb : tensor<32x480x14x14xf32>
    %b9ess = stablehlo.logistic %b9en : tensor<32x480x14x14xf32>
    %b9es = stablehlo.multiply %b9en, %b9ess : tensor<32x480x14x14xf32>
    %b9dc = stablehlo.convolution(%b9es, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %b9dbb = stablehlo.broadcast_in_dim %b9db, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9d = stablehlo.add %b9dc, %b9dbb : tensor<32x480x14x14xf32>
    %b9dnnf = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %b9dnep = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %b9dnsmr = stablehlo.reduce(%b9d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9dnsm = stablehlo.broadcast_in_dim %b9dnsmr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dnmu = stablehlo.divide %b9dnsm, %b9dnnf : tensor<32x480x14x14xf32>
    %b9dnxc = stablehlo.subtract %b9d, %b9dnmu : tensor<32x480x14x14xf32>
    %b9dnsq = stablehlo.multiply %b9dnxc, %b9dnxc : tensor<32x480x14x14xf32>
    %b9dnvsr = stablehlo.reduce(%b9dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9dnvs = stablehlo.broadcast_in_dim %b9dnvsr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dnvr = stablehlo.divide %b9dnvs, %b9dnnf : tensor<32x480x14x14xf32>
    %b9dnve = stablehlo.add %b9dnvr, %b9dnep : tensor<32x480x14x14xf32>
    %b9dnistd = stablehlo.rsqrt %b9dnve : tensor<32x480x14x14xf32>
    %b9dnxh = stablehlo.multiply %b9dnxc, %b9dnistd : tensor<32x480x14x14xf32>
    %b9dngb = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dnbtb = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dngx = stablehlo.multiply %b9dnxh, %b9dngb : tensor<32x480x14x14xf32>
    %b9dn = stablehlo.add %b9dngx, %b9dnbtb : tensor<32x480x14x14xf32>
    %b9dss = stablehlo.logistic %b9dn : tensor<32x480x14x14xf32>
    %b9ds = stablehlo.multiply %b9dn, %b9dss : tensor<32x480x14x14xf32>
    %b9zsqs = stablehlo.reduce(%b9ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b9zsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b9zsq = stablehlo.divide %b9zsqs, %b9zsqnf : tensor<32x480xf32>
    %b9zexd = stablehlo.dot_general %b9zsq, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %b9zexbb = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %b9zex = stablehlo.add %b9zexd, %b9zexbb : tensor<32x20xf32>
    %b9za1s = stablehlo.logistic %b9zex : tensor<32x20xf32>
    %b9za1 = stablehlo.multiply %b9zex, %b9za1s : tensor<32x20xf32>
    %b9zh2d = stablehlo.dot_general %b9za1, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %b9zh2bb = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %b9zh2 = stablehlo.add %b9zh2d, %b9zh2bb : tensor<32x480xf32>
    %b9zgate = stablehlo.logistic %b9zh2 : tensor<32x480xf32>
    %b9zgb = stablehlo.broadcast_in_dim %b9zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b9zse = stablehlo.multiply %b9ds, %b9zgb : tensor<32x480x14x14xf32>
    %b9pc = stablehlo.convolution(%b9zse, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b9pbb = stablehlo.broadcast_in_dim %b9pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9p = stablehlo.add %b9pc, %b9pbb : tensor<32x112x14x14xf32>
    %b9pnnf = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %b9pnep = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %b9pnsmr = stablehlo.reduce(%b9p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9pnsm = stablehlo.broadcast_in_dim %b9pnsmr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9pnmu = stablehlo.divide %b9pnsm, %b9pnnf : tensor<32x112x14x14xf32>
    %b9pnxc = stablehlo.subtract %b9p, %b9pnmu : tensor<32x112x14x14xf32>
    %b9pnsq = stablehlo.multiply %b9pnxc, %b9pnxc : tensor<32x112x14x14xf32>
    %b9pnvsr = stablehlo.reduce(%b9pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9pnvs = stablehlo.broadcast_in_dim %b9pnvsr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9pnvr = stablehlo.divide %b9pnvs, %b9pnnf : tensor<32x112x14x14xf32>
    %b9pnve = stablehlo.add %b9pnvr, %b9pnep : tensor<32x112x14x14xf32>
    %b9pnistd = stablehlo.rsqrt %b9pnve : tensor<32x112x14x14xf32>
    %b9pnxh = stablehlo.multiply %b9pnxc, %b9pnistd : tensor<32x112x14x14xf32>
    %b9pngb = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9pnbtb = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9pngx = stablehlo.multiply %b9pnxh, %b9pngb : tensor<32x112x14x14xf32>
    %b9pn = stablehlo.add %b9pngx, %b9pnbtb : tensor<32x112x14x14xf32>
    %b10ec = stablehlo.convolution(%b9pn, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b10ebb = stablehlo.broadcast_in_dim %b10eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10e = stablehlo.add %b10ec, %b10ebb : tensor<32x672x14x14xf32>
    %b10ennf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b10enep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b10ensmr = stablehlo.reduce(%b10e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ensm = stablehlo.broadcast_in_dim %b10ensmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10enmu = stablehlo.divide %b10ensm, %b10ennf : tensor<32x672x14x14xf32>
    %b10enxc = stablehlo.subtract %b10e, %b10enmu : tensor<32x672x14x14xf32>
    %b10ensq = stablehlo.multiply %b10enxc, %b10enxc : tensor<32x672x14x14xf32>
    %b10envsr = stablehlo.reduce(%b10ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10envs = stablehlo.broadcast_in_dim %b10envsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10envr = stablehlo.divide %b10envs, %b10ennf : tensor<32x672x14x14xf32>
    %b10enve = stablehlo.add %b10envr, %b10enep : tensor<32x672x14x14xf32>
    %b10enistd = stablehlo.rsqrt %b10enve : tensor<32x672x14x14xf32>
    %b10enxh = stablehlo.multiply %b10enxc, %b10enistd : tensor<32x672x14x14xf32>
    %b10engb = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10enbtb = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10engx = stablehlo.multiply %b10enxh, %b10engb : tensor<32x672x14x14xf32>
    %b10en = stablehlo.add %b10engx, %b10enbtb : tensor<32x672x14x14xf32>
    %b10ess = stablehlo.logistic %b10en : tensor<32x672x14x14xf32>
    %b10es = stablehlo.multiply %b10en, %b10ess : tensor<32x672x14x14xf32>
    %b10dc = stablehlo.convolution(%b10es, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b10dbb = stablehlo.broadcast_in_dim %b10db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10d = stablehlo.add %b10dc, %b10dbb : tensor<32x672x14x14xf32>
    %b10dnnf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b10dnep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b10dnsmr = stablehlo.reduce(%b10d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10dnsm = stablehlo.broadcast_in_dim %b10dnsmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dnmu = stablehlo.divide %b10dnsm, %b10dnnf : tensor<32x672x14x14xf32>
    %b10dnxc = stablehlo.subtract %b10d, %b10dnmu : tensor<32x672x14x14xf32>
    %b10dnsq = stablehlo.multiply %b10dnxc, %b10dnxc : tensor<32x672x14x14xf32>
    %b10dnvsr = stablehlo.reduce(%b10dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10dnvs = stablehlo.broadcast_in_dim %b10dnvsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dnvr = stablehlo.divide %b10dnvs, %b10dnnf : tensor<32x672x14x14xf32>
    %b10dnve = stablehlo.add %b10dnvr, %b10dnep : tensor<32x672x14x14xf32>
    %b10dnistd = stablehlo.rsqrt %b10dnve : tensor<32x672x14x14xf32>
    %b10dnxh = stablehlo.multiply %b10dnxc, %b10dnistd : tensor<32x672x14x14xf32>
    %b10dngb = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dnbtb = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dngx = stablehlo.multiply %b10dnxh, %b10dngb : tensor<32x672x14x14xf32>
    %b10dn = stablehlo.add %b10dngx, %b10dnbtb : tensor<32x672x14x14xf32>
    %b10dss = stablehlo.logistic %b10dn : tensor<32x672x14x14xf32>
    %b10ds = stablehlo.multiply %b10dn, %b10dss : tensor<32x672x14x14xf32>
    %b10zsqs = stablehlo.reduce(%b10ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b10zsqnf = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %b10zsq = stablehlo.divide %b10zsqs, %b10zsqnf : tensor<32x672xf32>
    %b10zexd = stablehlo.dot_general %b10zsq, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %b10zexbb = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %b10zex = stablehlo.add %b10zexd, %b10zexbb : tensor<32x28xf32>
    %b10za1s = stablehlo.logistic %b10zex : tensor<32x28xf32>
    %b10za1 = stablehlo.multiply %b10zex, %b10za1s : tensor<32x28xf32>
    %b10zh2d = stablehlo.dot_general %b10za1, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %b10zh2bb = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %b10zh2 = stablehlo.add %b10zh2d, %b10zh2bb : tensor<32x672xf32>
    %b10zgate = stablehlo.logistic %b10zh2 : tensor<32x672xf32>
    %b10zgb = stablehlo.broadcast_in_dim %b10zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b10zse = stablehlo.multiply %b10ds, %b10zgb : tensor<32x672x14x14xf32>
    %b10pc = stablehlo.convolution(%b10zse, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b10pbb = stablehlo.broadcast_in_dim %b10pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10p = stablehlo.add %b10pc, %b10pbb : tensor<32x112x14x14xf32>
    %b10pnnf = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %b10pnep = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %b10pnsmr = stablehlo.reduce(%b10p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10pnsm = stablehlo.broadcast_in_dim %b10pnsmr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10pnmu = stablehlo.divide %b10pnsm, %b10pnnf : tensor<32x112x14x14xf32>
    %b10pnxc = stablehlo.subtract %b10p, %b10pnmu : tensor<32x112x14x14xf32>
    %b10pnsq = stablehlo.multiply %b10pnxc, %b10pnxc : tensor<32x112x14x14xf32>
    %b10pnvsr = stablehlo.reduce(%b10pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10pnvs = stablehlo.broadcast_in_dim %b10pnvsr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10pnvr = stablehlo.divide %b10pnvs, %b10pnnf : tensor<32x112x14x14xf32>
    %b10pnve = stablehlo.add %b10pnvr, %b10pnep : tensor<32x112x14x14xf32>
    %b10pnistd = stablehlo.rsqrt %b10pnve : tensor<32x112x14x14xf32>
    %b10pnxh = stablehlo.multiply %b10pnxc, %b10pnistd : tensor<32x112x14x14xf32>
    %b10pngb = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10pnbtb = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10pngx = stablehlo.multiply %b10pnxh, %b10pngb : tensor<32x112x14x14xf32>
    %b10pn = stablehlo.add %b10pngx, %b10pnbtb : tensor<32x112x14x14xf32>
    %b10o = stablehlo.add %b10pn, %b9pn : tensor<32x112x14x14xf32>
    %b11ec = stablehlo.convolution(%b10o, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b11ebb = stablehlo.broadcast_in_dim %b11eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11e = stablehlo.add %b11ec, %b11ebb : tensor<32x672x14x14xf32>
    %b11ennf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b11enep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b11ensmr = stablehlo.reduce(%b11e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ensm = stablehlo.broadcast_in_dim %b11ensmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11enmu = stablehlo.divide %b11ensm, %b11ennf : tensor<32x672x14x14xf32>
    %b11enxc = stablehlo.subtract %b11e, %b11enmu : tensor<32x672x14x14xf32>
    %b11ensq = stablehlo.multiply %b11enxc, %b11enxc : tensor<32x672x14x14xf32>
    %b11envsr = stablehlo.reduce(%b11ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11envs = stablehlo.broadcast_in_dim %b11envsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11envr = stablehlo.divide %b11envs, %b11ennf : tensor<32x672x14x14xf32>
    %b11enve = stablehlo.add %b11envr, %b11enep : tensor<32x672x14x14xf32>
    %b11enistd = stablehlo.rsqrt %b11enve : tensor<32x672x14x14xf32>
    %b11enxh = stablehlo.multiply %b11enxc, %b11enistd : tensor<32x672x14x14xf32>
    %b11engb = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11enbtb = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11engx = stablehlo.multiply %b11enxh, %b11engb : tensor<32x672x14x14xf32>
    %b11en = stablehlo.add %b11engx, %b11enbtb : tensor<32x672x14x14xf32>
    %b11ess = stablehlo.logistic %b11en : tensor<32x672x14x14xf32>
    %b11es = stablehlo.multiply %b11en, %b11ess : tensor<32x672x14x14xf32>
    %b11dc = stablehlo.convolution(%b11es, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b11dbb = stablehlo.broadcast_in_dim %b11db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11d = stablehlo.add %b11dc, %b11dbb : tensor<32x672x14x14xf32>
    %b11dnnf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b11dnep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b11dnsmr = stablehlo.reduce(%b11d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dnsm = stablehlo.broadcast_in_dim %b11dnsmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dnmu = stablehlo.divide %b11dnsm, %b11dnnf : tensor<32x672x14x14xf32>
    %b11dnxc = stablehlo.subtract %b11d, %b11dnmu : tensor<32x672x14x14xf32>
    %b11dnsq = stablehlo.multiply %b11dnxc, %b11dnxc : tensor<32x672x14x14xf32>
    %b11dnvsr = stablehlo.reduce(%b11dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dnvs = stablehlo.broadcast_in_dim %b11dnvsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dnvr = stablehlo.divide %b11dnvs, %b11dnnf : tensor<32x672x14x14xf32>
    %b11dnve = stablehlo.add %b11dnvr, %b11dnep : tensor<32x672x14x14xf32>
    %b11dnistd = stablehlo.rsqrt %b11dnve : tensor<32x672x14x14xf32>
    %b11dnxh = stablehlo.multiply %b11dnxc, %b11dnistd : tensor<32x672x14x14xf32>
    %b11dngb = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dnbtb = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dngx = stablehlo.multiply %b11dnxh, %b11dngb : tensor<32x672x14x14xf32>
    %b11dn = stablehlo.add %b11dngx, %b11dnbtb : tensor<32x672x14x14xf32>
    %b11dss = stablehlo.logistic %b11dn : tensor<32x672x14x14xf32>
    %b11ds = stablehlo.multiply %b11dn, %b11dss : tensor<32x672x14x14xf32>
    %b11zsqs = stablehlo.reduce(%b11ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b11zsqnf = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %b11zsq = stablehlo.divide %b11zsqs, %b11zsqnf : tensor<32x672xf32>
    %b11zexd = stablehlo.dot_general %b11zsq, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %b11zexbb = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %b11zex = stablehlo.add %b11zexd, %b11zexbb : tensor<32x28xf32>
    %b11za1s = stablehlo.logistic %b11zex : tensor<32x28xf32>
    %b11za1 = stablehlo.multiply %b11zex, %b11za1s : tensor<32x28xf32>
    %b11zh2d = stablehlo.dot_general %b11za1, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %b11zh2bb = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %b11zh2 = stablehlo.add %b11zh2d, %b11zh2bb : tensor<32x672xf32>
    %b11zgate = stablehlo.logistic %b11zh2 : tensor<32x672xf32>
    %b11zgb = stablehlo.broadcast_in_dim %b11zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b11zse = stablehlo.multiply %b11ds, %b11zgb : tensor<32x672x14x14xf32>
    %b11pc = stablehlo.convolution(%b11zse, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b11pbb = stablehlo.broadcast_in_dim %b11pb, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11p = stablehlo.add %b11pc, %b11pbb : tensor<32x112x14x14xf32>
    %b11pnnf = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %b11pnep = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %b11pnsmr = stablehlo.reduce(%b11p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11pnsm = stablehlo.broadcast_in_dim %b11pnsmr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11pnmu = stablehlo.divide %b11pnsm, %b11pnnf : tensor<32x112x14x14xf32>
    %b11pnxc = stablehlo.subtract %b11p, %b11pnmu : tensor<32x112x14x14xf32>
    %b11pnsq = stablehlo.multiply %b11pnxc, %b11pnxc : tensor<32x112x14x14xf32>
    %b11pnvsr = stablehlo.reduce(%b11pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11pnvs = stablehlo.broadcast_in_dim %b11pnvsr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11pnvr = stablehlo.divide %b11pnvs, %b11pnnf : tensor<32x112x14x14xf32>
    %b11pnve = stablehlo.add %b11pnvr, %b11pnep : tensor<32x112x14x14xf32>
    %b11pnistd = stablehlo.rsqrt %b11pnve : tensor<32x112x14x14xf32>
    %b11pnxh = stablehlo.multiply %b11pnxc, %b11pnistd : tensor<32x112x14x14xf32>
    %b11pngb = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11pnbtb = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11pngx = stablehlo.multiply %b11pnxh, %b11pngb : tensor<32x112x14x14xf32>
    %b11pn = stablehlo.add %b11pngx, %b11pnbtb : tensor<32x112x14x14xf32>
    %b11o = stablehlo.add %b11pn, %b10o : tensor<32x112x14x14xf32>
    %b12ec = stablehlo.convolution(%b11o, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b12ebb = stablehlo.broadcast_in_dim %b12eb, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12e = stablehlo.add %b12ec, %b12ebb : tensor<32x672x14x14xf32>
    %b12ennf = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %b12enep = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %b12ensmr = stablehlo.reduce(%b12e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ensm = stablehlo.broadcast_in_dim %b12ensmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12enmu = stablehlo.divide %b12ensm, %b12ennf : tensor<32x672x14x14xf32>
    %b12enxc = stablehlo.subtract %b12e, %b12enmu : tensor<32x672x14x14xf32>
    %b12ensq = stablehlo.multiply %b12enxc, %b12enxc : tensor<32x672x14x14xf32>
    %b12envsr = stablehlo.reduce(%b12ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12envs = stablehlo.broadcast_in_dim %b12envsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12envr = stablehlo.divide %b12envs, %b12ennf : tensor<32x672x14x14xf32>
    %b12enve = stablehlo.add %b12envr, %b12enep : tensor<32x672x14x14xf32>
    %b12enistd = stablehlo.rsqrt %b12enve : tensor<32x672x14x14xf32>
    %b12enxh = stablehlo.multiply %b12enxc, %b12enistd : tensor<32x672x14x14xf32>
    %b12engb = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12enbtb = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12engx = stablehlo.multiply %b12enxh, %b12engb : tensor<32x672x14x14xf32>
    %b12en = stablehlo.add %b12engx, %b12enbtb : tensor<32x672x14x14xf32>
    %b12ess = stablehlo.logistic %b12en : tensor<32x672x14x14xf32>
    %b12es = stablehlo.multiply %b12en, %b12ess : tensor<32x672x14x14xf32>
    %b12dc = stablehlo.convolution(%b12es, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %b12dbb = stablehlo.broadcast_in_dim %b12db, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12d = stablehlo.add %b12dc, %b12dbb : tensor<32x672x7x7xf32>
    %b12dnnf = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %b12dnep = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %b12dnsmr = stablehlo.reduce(%b12d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12dnsm = stablehlo.broadcast_in_dim %b12dnsmr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12dnmu = stablehlo.divide %b12dnsm, %b12dnnf : tensor<32x672x7x7xf32>
    %b12dnxc = stablehlo.subtract %b12d, %b12dnmu : tensor<32x672x7x7xf32>
    %b12dnsq = stablehlo.multiply %b12dnxc, %b12dnxc : tensor<32x672x7x7xf32>
    %b12dnvsr = stablehlo.reduce(%b12dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12dnvs = stablehlo.broadcast_in_dim %b12dnvsr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12dnvr = stablehlo.divide %b12dnvs, %b12dnnf : tensor<32x672x7x7xf32>
    %b12dnve = stablehlo.add %b12dnvr, %b12dnep : tensor<32x672x7x7xf32>
    %b12dnistd = stablehlo.rsqrt %b12dnve : tensor<32x672x7x7xf32>
    %b12dnxh = stablehlo.multiply %b12dnxc, %b12dnistd : tensor<32x672x7x7xf32>
    %b12dngb = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12dnbtb = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12dngx = stablehlo.multiply %b12dnxh, %b12dngb : tensor<32x672x7x7xf32>
    %b12dn = stablehlo.add %b12dngx, %b12dnbtb : tensor<32x672x7x7xf32>
    %b12dss = stablehlo.logistic %b12dn : tensor<32x672x7x7xf32>
    %b12ds = stablehlo.multiply %b12dn, %b12dss : tensor<32x672x7x7xf32>
    %b12zsqs = stablehlo.reduce(%b12ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b12zsqnf = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %b12zsq = stablehlo.divide %b12zsqs, %b12zsqnf : tensor<32x672xf32>
    %b12zexd = stablehlo.dot_general %b12zsq, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %b12zexbb = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %b12zex = stablehlo.add %b12zexd, %b12zexbb : tensor<32x28xf32>
    %b12za1s = stablehlo.logistic %b12zex : tensor<32x28xf32>
    %b12za1 = stablehlo.multiply %b12zex, %b12za1s : tensor<32x28xf32>
    %b12zh2d = stablehlo.dot_general %b12za1, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %b12zh2bb = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %b12zh2 = stablehlo.add %b12zh2d, %b12zh2bb : tensor<32x672xf32>
    %b12zgate = stablehlo.logistic %b12zh2 : tensor<32x672xf32>
    %b12zgb = stablehlo.broadcast_in_dim %b12zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %b12zse = stablehlo.multiply %b12ds, %b12zgb : tensor<32x672x7x7xf32>
    %b12pc = stablehlo.convolution(%b12zse, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b12pbb = stablehlo.broadcast_in_dim %b12pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12p = stablehlo.add %b12pc, %b12pbb : tensor<32x192x7x7xf32>
    %b12pnnf = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %b12pnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %b12pnsmr = stablehlo.reduce(%b12p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12pnsm = stablehlo.broadcast_in_dim %b12pnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12pnmu = stablehlo.divide %b12pnsm, %b12pnnf : tensor<32x192x7x7xf32>
    %b12pnxc = stablehlo.subtract %b12p, %b12pnmu : tensor<32x192x7x7xf32>
    %b12pnsq = stablehlo.multiply %b12pnxc, %b12pnxc : tensor<32x192x7x7xf32>
    %b12pnvsr = stablehlo.reduce(%b12pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12pnvs = stablehlo.broadcast_in_dim %b12pnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12pnvr = stablehlo.divide %b12pnvs, %b12pnnf : tensor<32x192x7x7xf32>
    %b12pnve = stablehlo.add %b12pnvr, %b12pnep : tensor<32x192x7x7xf32>
    %b12pnistd = stablehlo.rsqrt %b12pnve : tensor<32x192x7x7xf32>
    %b12pnxh = stablehlo.multiply %b12pnxc, %b12pnistd : tensor<32x192x7x7xf32>
    %b12pngb = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12pnbtb = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12pngx = stablehlo.multiply %b12pnxh, %b12pngb : tensor<32x192x7x7xf32>
    %b12pn = stablehlo.add %b12pngx, %b12pnbtb : tensor<32x192x7x7xf32>
    %b13ec = stablehlo.convolution(%b12pn, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b13ebb = stablehlo.broadcast_in_dim %b13eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13e = stablehlo.add %b13ec, %b13ebb : tensor<32x1152x7x7xf32>
    %b13ennf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b13enep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b13ensmr = stablehlo.reduce(%b13e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ensm = stablehlo.broadcast_in_dim %b13ensmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13enmu = stablehlo.divide %b13ensm, %b13ennf : tensor<32x1152x7x7xf32>
    %b13enxc = stablehlo.subtract %b13e, %b13enmu : tensor<32x1152x7x7xf32>
    %b13ensq = stablehlo.multiply %b13enxc, %b13enxc : tensor<32x1152x7x7xf32>
    %b13envsr = stablehlo.reduce(%b13ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13envs = stablehlo.broadcast_in_dim %b13envsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13envr = stablehlo.divide %b13envs, %b13ennf : tensor<32x1152x7x7xf32>
    %b13enve = stablehlo.add %b13envr, %b13enep : tensor<32x1152x7x7xf32>
    %b13enistd = stablehlo.rsqrt %b13enve : tensor<32x1152x7x7xf32>
    %b13enxh = stablehlo.multiply %b13enxc, %b13enistd : tensor<32x1152x7x7xf32>
    %b13engb = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13enbtb = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13engx = stablehlo.multiply %b13enxh, %b13engb : tensor<32x1152x7x7xf32>
    %b13en = stablehlo.add %b13engx, %b13enbtb : tensor<32x1152x7x7xf32>
    %b13ess = stablehlo.logistic %b13en : tensor<32x1152x7x7xf32>
    %b13es = stablehlo.multiply %b13en, %b13ess : tensor<32x1152x7x7xf32>
    %b13dc = stablehlo.convolution(%b13es, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b13dbb = stablehlo.broadcast_in_dim %b13db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13d = stablehlo.add %b13dc, %b13dbb : tensor<32x1152x7x7xf32>
    %b13dnnf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b13dnep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b13dnsmr = stablehlo.reduce(%b13d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13dnsm = stablehlo.broadcast_in_dim %b13dnsmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dnmu = stablehlo.divide %b13dnsm, %b13dnnf : tensor<32x1152x7x7xf32>
    %b13dnxc = stablehlo.subtract %b13d, %b13dnmu : tensor<32x1152x7x7xf32>
    %b13dnsq = stablehlo.multiply %b13dnxc, %b13dnxc : tensor<32x1152x7x7xf32>
    %b13dnvsr = stablehlo.reduce(%b13dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13dnvs = stablehlo.broadcast_in_dim %b13dnvsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dnvr = stablehlo.divide %b13dnvs, %b13dnnf : tensor<32x1152x7x7xf32>
    %b13dnve = stablehlo.add %b13dnvr, %b13dnep : tensor<32x1152x7x7xf32>
    %b13dnistd = stablehlo.rsqrt %b13dnve : tensor<32x1152x7x7xf32>
    %b13dnxh = stablehlo.multiply %b13dnxc, %b13dnistd : tensor<32x1152x7x7xf32>
    %b13dngb = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dnbtb = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dngx = stablehlo.multiply %b13dnxh, %b13dngb : tensor<32x1152x7x7xf32>
    %b13dn = stablehlo.add %b13dngx, %b13dnbtb : tensor<32x1152x7x7xf32>
    %b13dss = stablehlo.logistic %b13dn : tensor<32x1152x7x7xf32>
    %b13ds = stablehlo.multiply %b13dn, %b13dss : tensor<32x1152x7x7xf32>
    %b13zsqs = stablehlo.reduce(%b13ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b13zsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b13zsq = stablehlo.divide %b13zsqs, %b13zsqnf : tensor<32x1152xf32>
    %b13zexd = stablehlo.dot_general %b13zsq, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %b13zexbb = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %b13zex = stablehlo.add %b13zexd, %b13zexbb : tensor<32x48xf32>
    %b13za1s = stablehlo.logistic %b13zex : tensor<32x48xf32>
    %b13za1 = stablehlo.multiply %b13zex, %b13za1s : tensor<32x48xf32>
    %b13zh2d = stablehlo.dot_general %b13za1, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %b13zh2bb = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %b13zh2 = stablehlo.add %b13zh2d, %b13zh2bb : tensor<32x1152xf32>
    %b13zgate = stablehlo.logistic %b13zh2 : tensor<32x1152xf32>
    %b13zgb = stablehlo.broadcast_in_dim %b13zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13zse = stablehlo.multiply %b13ds, %b13zgb : tensor<32x1152x7x7xf32>
    %b13pc = stablehlo.convolution(%b13zse, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b13pbb = stablehlo.broadcast_in_dim %b13pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13p = stablehlo.add %b13pc, %b13pbb : tensor<32x192x7x7xf32>
    %b13pnnf = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %b13pnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %b13pnsmr = stablehlo.reduce(%b13p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13pnsm = stablehlo.broadcast_in_dim %b13pnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13pnmu = stablehlo.divide %b13pnsm, %b13pnnf : tensor<32x192x7x7xf32>
    %b13pnxc = stablehlo.subtract %b13p, %b13pnmu : tensor<32x192x7x7xf32>
    %b13pnsq = stablehlo.multiply %b13pnxc, %b13pnxc : tensor<32x192x7x7xf32>
    %b13pnvsr = stablehlo.reduce(%b13pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13pnvs = stablehlo.broadcast_in_dim %b13pnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13pnvr = stablehlo.divide %b13pnvs, %b13pnnf : tensor<32x192x7x7xf32>
    %b13pnve = stablehlo.add %b13pnvr, %b13pnep : tensor<32x192x7x7xf32>
    %b13pnistd = stablehlo.rsqrt %b13pnve : tensor<32x192x7x7xf32>
    %b13pnxh = stablehlo.multiply %b13pnxc, %b13pnistd : tensor<32x192x7x7xf32>
    %b13pngb = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13pnbtb = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13pngx = stablehlo.multiply %b13pnxh, %b13pngb : tensor<32x192x7x7xf32>
    %b13pn = stablehlo.add %b13pngx, %b13pnbtb : tensor<32x192x7x7xf32>
    %b13o = stablehlo.add %b13pn, %b12pn : tensor<32x192x7x7xf32>
    %b14ec = stablehlo.convolution(%b13o, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b14ebb = stablehlo.broadcast_in_dim %b14eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14e = stablehlo.add %b14ec, %b14ebb : tensor<32x1152x7x7xf32>
    %b14ennf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b14enep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b14ensmr = stablehlo.reduce(%b14e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ensm = stablehlo.broadcast_in_dim %b14ensmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14enmu = stablehlo.divide %b14ensm, %b14ennf : tensor<32x1152x7x7xf32>
    %b14enxc = stablehlo.subtract %b14e, %b14enmu : tensor<32x1152x7x7xf32>
    %b14ensq = stablehlo.multiply %b14enxc, %b14enxc : tensor<32x1152x7x7xf32>
    %b14envsr = stablehlo.reduce(%b14ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14envs = stablehlo.broadcast_in_dim %b14envsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14envr = stablehlo.divide %b14envs, %b14ennf : tensor<32x1152x7x7xf32>
    %b14enve = stablehlo.add %b14envr, %b14enep : tensor<32x1152x7x7xf32>
    %b14enistd = stablehlo.rsqrt %b14enve : tensor<32x1152x7x7xf32>
    %b14enxh = stablehlo.multiply %b14enxc, %b14enistd : tensor<32x1152x7x7xf32>
    %b14engb = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14enbtb = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14engx = stablehlo.multiply %b14enxh, %b14engb : tensor<32x1152x7x7xf32>
    %b14en = stablehlo.add %b14engx, %b14enbtb : tensor<32x1152x7x7xf32>
    %b14ess = stablehlo.logistic %b14en : tensor<32x1152x7x7xf32>
    %b14es = stablehlo.multiply %b14en, %b14ess : tensor<32x1152x7x7xf32>
    %b14dc = stablehlo.convolution(%b14es, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b14dbb = stablehlo.broadcast_in_dim %b14db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14d = stablehlo.add %b14dc, %b14dbb : tensor<32x1152x7x7xf32>
    %b14dnnf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b14dnep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b14dnsmr = stablehlo.reduce(%b14d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14dnsm = stablehlo.broadcast_in_dim %b14dnsmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dnmu = stablehlo.divide %b14dnsm, %b14dnnf : tensor<32x1152x7x7xf32>
    %b14dnxc = stablehlo.subtract %b14d, %b14dnmu : tensor<32x1152x7x7xf32>
    %b14dnsq = stablehlo.multiply %b14dnxc, %b14dnxc : tensor<32x1152x7x7xf32>
    %b14dnvsr = stablehlo.reduce(%b14dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14dnvs = stablehlo.broadcast_in_dim %b14dnvsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dnvr = stablehlo.divide %b14dnvs, %b14dnnf : tensor<32x1152x7x7xf32>
    %b14dnve = stablehlo.add %b14dnvr, %b14dnep : tensor<32x1152x7x7xf32>
    %b14dnistd = stablehlo.rsqrt %b14dnve : tensor<32x1152x7x7xf32>
    %b14dnxh = stablehlo.multiply %b14dnxc, %b14dnistd : tensor<32x1152x7x7xf32>
    %b14dngb = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dnbtb = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dngx = stablehlo.multiply %b14dnxh, %b14dngb : tensor<32x1152x7x7xf32>
    %b14dn = stablehlo.add %b14dngx, %b14dnbtb : tensor<32x1152x7x7xf32>
    %b14dss = stablehlo.logistic %b14dn : tensor<32x1152x7x7xf32>
    %b14ds = stablehlo.multiply %b14dn, %b14dss : tensor<32x1152x7x7xf32>
    %b14zsqs = stablehlo.reduce(%b14ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b14zsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b14zsq = stablehlo.divide %b14zsqs, %b14zsqnf : tensor<32x1152xf32>
    %b14zexd = stablehlo.dot_general %b14zsq, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %b14zexbb = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %b14zex = stablehlo.add %b14zexd, %b14zexbb : tensor<32x48xf32>
    %b14za1s = stablehlo.logistic %b14zex : tensor<32x48xf32>
    %b14za1 = stablehlo.multiply %b14zex, %b14za1s : tensor<32x48xf32>
    %b14zh2d = stablehlo.dot_general %b14za1, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %b14zh2bb = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %b14zh2 = stablehlo.add %b14zh2d, %b14zh2bb : tensor<32x1152xf32>
    %b14zgate = stablehlo.logistic %b14zh2 : tensor<32x1152xf32>
    %b14zgb = stablehlo.broadcast_in_dim %b14zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14zse = stablehlo.multiply %b14ds, %b14zgb : tensor<32x1152x7x7xf32>
    %b14pc = stablehlo.convolution(%b14zse, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b14pbb = stablehlo.broadcast_in_dim %b14pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14p = stablehlo.add %b14pc, %b14pbb : tensor<32x192x7x7xf32>
    %b14pnnf = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %b14pnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %b14pnsmr = stablehlo.reduce(%b14p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14pnsm = stablehlo.broadcast_in_dim %b14pnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14pnmu = stablehlo.divide %b14pnsm, %b14pnnf : tensor<32x192x7x7xf32>
    %b14pnxc = stablehlo.subtract %b14p, %b14pnmu : tensor<32x192x7x7xf32>
    %b14pnsq = stablehlo.multiply %b14pnxc, %b14pnxc : tensor<32x192x7x7xf32>
    %b14pnvsr = stablehlo.reduce(%b14pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14pnvs = stablehlo.broadcast_in_dim %b14pnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14pnvr = stablehlo.divide %b14pnvs, %b14pnnf : tensor<32x192x7x7xf32>
    %b14pnve = stablehlo.add %b14pnvr, %b14pnep : tensor<32x192x7x7xf32>
    %b14pnistd = stablehlo.rsqrt %b14pnve : tensor<32x192x7x7xf32>
    %b14pnxh = stablehlo.multiply %b14pnxc, %b14pnistd : tensor<32x192x7x7xf32>
    %b14pngb = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14pnbtb = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14pngx = stablehlo.multiply %b14pnxh, %b14pngb : tensor<32x192x7x7xf32>
    %b14pn = stablehlo.add %b14pngx, %b14pnbtb : tensor<32x192x7x7xf32>
    %b14o = stablehlo.add %b14pn, %b13o : tensor<32x192x7x7xf32>
    %b15ec = stablehlo.convolution(%b14o, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b15ebb = stablehlo.broadcast_in_dim %b15eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15e = stablehlo.add %b15ec, %b15ebb : tensor<32x1152x7x7xf32>
    %b15ennf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b15enep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b15ensmr = stablehlo.reduce(%b15e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ensm = stablehlo.broadcast_in_dim %b15ensmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15enmu = stablehlo.divide %b15ensm, %b15ennf : tensor<32x1152x7x7xf32>
    %b15enxc = stablehlo.subtract %b15e, %b15enmu : tensor<32x1152x7x7xf32>
    %b15ensq = stablehlo.multiply %b15enxc, %b15enxc : tensor<32x1152x7x7xf32>
    %b15envsr = stablehlo.reduce(%b15ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15envs = stablehlo.broadcast_in_dim %b15envsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15envr = stablehlo.divide %b15envs, %b15ennf : tensor<32x1152x7x7xf32>
    %b15enve = stablehlo.add %b15envr, %b15enep : tensor<32x1152x7x7xf32>
    %b15enistd = stablehlo.rsqrt %b15enve : tensor<32x1152x7x7xf32>
    %b15enxh = stablehlo.multiply %b15enxc, %b15enistd : tensor<32x1152x7x7xf32>
    %b15engb = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15enbtb = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15engx = stablehlo.multiply %b15enxh, %b15engb : tensor<32x1152x7x7xf32>
    %b15en = stablehlo.add %b15engx, %b15enbtb : tensor<32x1152x7x7xf32>
    %b15ess = stablehlo.logistic %b15en : tensor<32x1152x7x7xf32>
    %b15es = stablehlo.multiply %b15en, %b15ess : tensor<32x1152x7x7xf32>
    %b15dc = stablehlo.convolution(%b15es, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b15dbb = stablehlo.broadcast_in_dim %b15db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15d = stablehlo.add %b15dc, %b15dbb : tensor<32x1152x7x7xf32>
    %b15dnnf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b15dnep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b15dnsmr = stablehlo.reduce(%b15d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dnsm = stablehlo.broadcast_in_dim %b15dnsmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dnmu = stablehlo.divide %b15dnsm, %b15dnnf : tensor<32x1152x7x7xf32>
    %b15dnxc = stablehlo.subtract %b15d, %b15dnmu : tensor<32x1152x7x7xf32>
    %b15dnsq = stablehlo.multiply %b15dnxc, %b15dnxc : tensor<32x1152x7x7xf32>
    %b15dnvsr = stablehlo.reduce(%b15dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dnvs = stablehlo.broadcast_in_dim %b15dnvsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dnvr = stablehlo.divide %b15dnvs, %b15dnnf : tensor<32x1152x7x7xf32>
    %b15dnve = stablehlo.add %b15dnvr, %b15dnep : tensor<32x1152x7x7xf32>
    %b15dnistd = stablehlo.rsqrt %b15dnve : tensor<32x1152x7x7xf32>
    %b15dnxh = stablehlo.multiply %b15dnxc, %b15dnistd : tensor<32x1152x7x7xf32>
    %b15dngb = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dnbtb = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dngx = stablehlo.multiply %b15dnxh, %b15dngb : tensor<32x1152x7x7xf32>
    %b15dn = stablehlo.add %b15dngx, %b15dnbtb : tensor<32x1152x7x7xf32>
    %b15dss = stablehlo.logistic %b15dn : tensor<32x1152x7x7xf32>
    %b15ds = stablehlo.multiply %b15dn, %b15dss : tensor<32x1152x7x7xf32>
    %b15zsqs = stablehlo.reduce(%b15ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b15zsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b15zsq = stablehlo.divide %b15zsqs, %b15zsqnf : tensor<32x1152xf32>
    %b15zexd = stablehlo.dot_general %b15zsq, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %b15zexbb = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %b15zex = stablehlo.add %b15zexd, %b15zexbb : tensor<32x48xf32>
    %b15za1s = stablehlo.logistic %b15zex : tensor<32x48xf32>
    %b15za1 = stablehlo.multiply %b15zex, %b15za1s : tensor<32x48xf32>
    %b15zh2d = stablehlo.dot_general %b15za1, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %b15zh2bb = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %b15zh2 = stablehlo.add %b15zh2d, %b15zh2bb : tensor<32x1152xf32>
    %b15zgate = stablehlo.logistic %b15zh2 : tensor<32x1152xf32>
    %b15zgb = stablehlo.broadcast_in_dim %b15zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15zse = stablehlo.multiply %b15ds, %b15zgb : tensor<32x1152x7x7xf32>
    %b15pc = stablehlo.convolution(%b15zse, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b15pbb = stablehlo.broadcast_in_dim %b15pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15p = stablehlo.add %b15pc, %b15pbb : tensor<32x192x7x7xf32>
    %b15pnnf = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %b15pnep = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %b15pnsmr = stablehlo.reduce(%b15p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15pnsm = stablehlo.broadcast_in_dim %b15pnsmr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15pnmu = stablehlo.divide %b15pnsm, %b15pnnf : tensor<32x192x7x7xf32>
    %b15pnxc = stablehlo.subtract %b15p, %b15pnmu : tensor<32x192x7x7xf32>
    %b15pnsq = stablehlo.multiply %b15pnxc, %b15pnxc : tensor<32x192x7x7xf32>
    %b15pnvsr = stablehlo.reduce(%b15pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15pnvs = stablehlo.broadcast_in_dim %b15pnvsr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15pnvr = stablehlo.divide %b15pnvs, %b15pnnf : tensor<32x192x7x7xf32>
    %b15pnve = stablehlo.add %b15pnvr, %b15pnep : tensor<32x192x7x7xf32>
    %b15pnistd = stablehlo.rsqrt %b15pnve : tensor<32x192x7x7xf32>
    %b15pnxh = stablehlo.multiply %b15pnxc, %b15pnistd : tensor<32x192x7x7xf32>
    %b15pngb = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15pnbtb = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15pngx = stablehlo.multiply %b15pnxh, %b15pngb : tensor<32x192x7x7xf32>
    %b15pn = stablehlo.add %b15pngx, %b15pnbtb : tensor<32x192x7x7xf32>
    %b15o = stablehlo.add %b15pn, %b14o : tensor<32x192x7x7xf32>
    %b16ec = stablehlo.convolution(%b15o, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b16ebb = stablehlo.broadcast_in_dim %b16eb, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16e = stablehlo.add %b16ec, %b16ebb : tensor<32x1152x7x7xf32>
    %b16ennf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b16enep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b16ensmr = stablehlo.reduce(%b16e init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ensm = stablehlo.broadcast_in_dim %b16ensmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16enmu = stablehlo.divide %b16ensm, %b16ennf : tensor<32x1152x7x7xf32>
    %b16enxc = stablehlo.subtract %b16e, %b16enmu : tensor<32x1152x7x7xf32>
    %b16ensq = stablehlo.multiply %b16enxc, %b16enxc : tensor<32x1152x7x7xf32>
    %b16envsr = stablehlo.reduce(%b16ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16envs = stablehlo.broadcast_in_dim %b16envsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16envr = stablehlo.divide %b16envs, %b16ennf : tensor<32x1152x7x7xf32>
    %b16enve = stablehlo.add %b16envr, %b16enep : tensor<32x1152x7x7xf32>
    %b16enistd = stablehlo.rsqrt %b16enve : tensor<32x1152x7x7xf32>
    %b16enxh = stablehlo.multiply %b16enxc, %b16enistd : tensor<32x1152x7x7xf32>
    %b16engb = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16enbtb = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16engx = stablehlo.multiply %b16enxh, %b16engb : tensor<32x1152x7x7xf32>
    %b16en = stablehlo.add %b16engx, %b16enbtb : tensor<32x1152x7x7xf32>
    %b16ess = stablehlo.logistic %b16en : tensor<32x1152x7x7xf32>
    %b16es = stablehlo.multiply %b16en, %b16ess : tensor<32x1152x7x7xf32>
    %b16dc = stablehlo.convolution(%b16es, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %b16dbb = stablehlo.broadcast_in_dim %b16db, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16d = stablehlo.add %b16dc, %b16dbb : tensor<32x1152x7x7xf32>
    %b16dnnf = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %b16dnep = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %b16dnsmr = stablehlo.reduce(%b16d init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16dnsm = stablehlo.broadcast_in_dim %b16dnsmr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dnmu = stablehlo.divide %b16dnsm, %b16dnnf : tensor<32x1152x7x7xf32>
    %b16dnxc = stablehlo.subtract %b16d, %b16dnmu : tensor<32x1152x7x7xf32>
    %b16dnsq = stablehlo.multiply %b16dnxc, %b16dnxc : tensor<32x1152x7x7xf32>
    %b16dnvsr = stablehlo.reduce(%b16dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16dnvs = stablehlo.broadcast_in_dim %b16dnvsr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dnvr = stablehlo.divide %b16dnvs, %b16dnnf : tensor<32x1152x7x7xf32>
    %b16dnve = stablehlo.add %b16dnvr, %b16dnep : tensor<32x1152x7x7xf32>
    %b16dnistd = stablehlo.rsqrt %b16dnve : tensor<32x1152x7x7xf32>
    %b16dnxh = stablehlo.multiply %b16dnxc, %b16dnistd : tensor<32x1152x7x7xf32>
    %b16dngb = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dnbtb = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dngx = stablehlo.multiply %b16dnxh, %b16dngb : tensor<32x1152x7x7xf32>
    %b16dn = stablehlo.add %b16dngx, %b16dnbtb : tensor<32x1152x7x7xf32>
    %b16dss = stablehlo.logistic %b16dn : tensor<32x1152x7x7xf32>
    %b16ds = stablehlo.multiply %b16dn, %b16dss : tensor<32x1152x7x7xf32>
    %b16zsqs = stablehlo.reduce(%b16ds init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b16zsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b16zsq = stablehlo.divide %b16zsqs, %b16zsqnf : tensor<32x1152xf32>
    %b16zexd = stablehlo.dot_general %b16zsq, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %b16zexbb = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %b16zex = stablehlo.add %b16zexd, %b16zexbb : tensor<32x48xf32>
    %b16za1s = stablehlo.logistic %b16zex : tensor<32x48xf32>
    %b16za1 = stablehlo.multiply %b16zex, %b16za1s : tensor<32x48xf32>
    %b16zh2d = stablehlo.dot_general %b16za1, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %b16zh2bb = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %b16zh2 = stablehlo.add %b16zh2d, %b16zh2bb : tensor<32x1152xf32>
    %b16zgate = stablehlo.logistic %b16zh2 : tensor<32x1152xf32>
    %b16zgb = stablehlo.broadcast_in_dim %b16zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16zse = stablehlo.multiply %b16ds, %b16zgb : tensor<32x1152x7x7xf32>
    %b16pc = stablehlo.convolution(%b16zse, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %b16pbb = stablehlo.broadcast_in_dim %b16pb, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16p = stablehlo.add %b16pc, %b16pbb : tensor<32x320x7x7xf32>
    %b16pnnf = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %b16pnep = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %b16pnsmr = stablehlo.reduce(%b16p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16pnsm = stablehlo.broadcast_in_dim %b16pnsmr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16pnmu = stablehlo.divide %b16pnsm, %b16pnnf : tensor<32x320x7x7xf32>
    %b16pnxc = stablehlo.subtract %b16p, %b16pnmu : tensor<32x320x7x7xf32>
    %b16pnsq = stablehlo.multiply %b16pnxc, %b16pnxc : tensor<32x320x7x7xf32>
    %b16pnvsr = stablehlo.reduce(%b16pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16pnvs = stablehlo.broadcast_in_dim %b16pnvsr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16pnvr = stablehlo.divide %b16pnvs, %b16pnnf : tensor<32x320x7x7xf32>
    %b16pnve = stablehlo.add %b16pnvr, %b16pnep : tensor<32x320x7x7xf32>
    %b16pnistd = stablehlo.rsqrt %b16pnve : tensor<32x320x7x7xf32>
    %b16pnxh = stablehlo.multiply %b16pnxc, %b16pnistd : tensor<32x320x7x7xf32>
    %b16pngb = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16pnbtb = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16pngx = stablehlo.multiply %b16pnxh, %b16pngb : tensor<32x320x7x7xf32>
    %b16pn = stablehlo.add %b16pngx, %b16pnbtb : tensor<32x320x7x7xf32>
    %hc = stablehlo.convolution(%b16pn, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %hbb = stablehlo.broadcast_in_dim %hb, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %h = stablehlo.add %hc, %hbb : tensor<32x1280x7x7xf32>
    %hnnf = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %hnep = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %hnsmr = stablehlo.reduce(%h init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %hnsm = stablehlo.broadcast_in_dim %hnsmr, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hnmu = stablehlo.divide %hnsm, %hnnf : tensor<32x1280x7x7xf32>
    %hnxc = stablehlo.subtract %h, %hnmu : tensor<32x1280x7x7xf32>
    %hnsq = stablehlo.multiply %hnxc, %hnxc : tensor<32x1280x7x7xf32>
    %hnvsr = stablehlo.reduce(%hnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %hnvs = stablehlo.broadcast_in_dim %hnvsr, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hnvr = stablehlo.divide %hnvs, %hnnf : tensor<32x1280x7x7xf32>
    %hnve = stablehlo.add %hnvr, %hnep : tensor<32x1280x7x7xf32>
    %hnistd = stablehlo.rsqrt %hnve : tensor<32x1280x7x7xf32>
    %hnxh = stablehlo.multiply %hnxc, %hnistd : tensor<32x1280x7x7xf32>
    %hngb = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hnbtb = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %hngx = stablehlo.multiply %hnxh, %hngb : tensor<32x1280x7x7xf32>
    %hn = stablehlo.add %hngx, %hnbtb : tensor<32x1280x7x7xf32>
    %hrs = stablehlo.logistic %hn : tensor<32x1280x7x7xf32>
    %hr = stablehlo.multiply %hn, %hrs : tensor<32x1280x7x7xf32>
    %gaps = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %gapnf = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %gap = stablehlo.divide %gaps, %gapnf : tensor<32x1280xf32>
    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %logits = stablehlo.add %ld, %ldb : tensor<32x10xf32>
    %le = stablehlo.exponential %logits : tensor<32x10xf32>
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
    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<1280x10xf32>) -> tensor<32x1280xf32>
    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dgnf = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %dgs = stablehlo.divide %dgap, %dgnf : tensor<32x1280xf32>
    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %dhrs = stablehlo.logistic %hn : tensor<32x1280x7x7xf32>
    %dhrone = stablehlo.constant dense<1.0> : tensor<32x1280x7x7xf32>
    %dhrom = stablehlo.subtract %dhrone, %dhrs : tensor<32x1280x7x7xf32>
    %dhrxom = stablehlo.multiply %hn, %dhrom : tensor<32x1280x7x7xf32>
    %dhrin = stablehlo.add %dhrone, %dhrxom : tensor<32x1280x7x7xf32>
    %dhrsp = stablehlo.multiply %dhrs, %dhrin : tensor<32x1280x7x7xf32>
    %dhr = stablehlo.multiply %dgapin, %dhrsp : tensor<32x1280x7x7xf32>
    %dhndxh = stablehlo.multiply %hngb, %dhr : tensor<32x1280x7x7xf32>
    %dhnsdxr = stablehlo.reduce(%dhndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %dhnsdx = stablehlo.broadcast_in_dim %dhnsdxr, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %dhnxd = stablehlo.multiply %hnxh, %dhndxh : tensor<32x1280x7x7xf32>
    %dhnsxdr = stablehlo.reduce(%dhnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %dhnsxd = stablehlo.broadcast_in_dim %dhnsxdr, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %dhnt1 = stablehlo.multiply %dhndxh, %hnnf : tensor<32x1280x7x7xf32>
    %dhni1 = stablehlo.subtract %dhnt1, %dhnsdx : tensor<32x1280x7x7xf32>
    %dhnxs = stablehlo.multiply %hnxh, %dhnsxd : tensor<32x1280x7x7xf32>
    %dhni2 = stablehlo.subtract %dhni1, %dhnxs : tensor<32x1280x7x7xf32>
    %dhnsN = stablehlo.divide %hnistd, %hnnf : tensor<32x1280x7x7xf32>
    %dhn = stablehlo.multiply %dhnsN, %dhni2 : tensor<32x1280x7x7xf32>
    %dhndgp = stablehlo.multiply %dhr, %hnxh : tensor<32x1280x7x7xf32>
    %dhndg = stablehlo.reduce(%dhndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %dhndb = stablehlo.reduce(%dhr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %dht = stablehlo.transpose %hW, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %dh = stablehlo.convolution(%dhn, %dht)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %dhWxt = stablehlo.transpose %b16pn, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %dhWdt = stablehlo.transpose %dhn, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %dhWraw = stablehlo.convolution(%dhWxt, %dhWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %dhW = stablehlo.transpose %dhWraw, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %dhb = stablehlo.reduce(%dhn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %b16dpndxh = stablehlo.multiply %b16pngb, %dh : tensor<32x320x7x7xf32>
    %b16dpnsdxr = stablehlo.reduce(%b16dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16dpnsdx = stablehlo.broadcast_in_dim %b16dpnsdxr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16dpnxd = stablehlo.multiply %b16pnxh, %b16dpndxh : tensor<32x320x7x7xf32>
    %b16dpnsxdr = stablehlo.reduce(%b16dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16dpnsxd = stablehlo.broadcast_in_dim %b16dpnsxdr, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %b16dpnt1 = stablehlo.multiply %b16dpndxh, %b16pnnf : tensor<32x320x7x7xf32>
    %b16dpni1 = stablehlo.subtract %b16dpnt1, %b16dpnsdx : tensor<32x320x7x7xf32>
    %b16dpnxs = stablehlo.multiply %b16pnxh, %b16dpnsxd : tensor<32x320x7x7xf32>
    %b16dpni2 = stablehlo.subtract %b16dpni1, %b16dpnxs : tensor<32x320x7x7xf32>
    %b16dpnsN = stablehlo.divide %b16pnistd, %b16pnnf : tensor<32x320x7x7xf32>
    %b16dpn = stablehlo.multiply %b16dpnsN, %b16dpni2 : tensor<32x320x7x7xf32>
    %b16dpndgp = stablehlo.multiply %dh, %b16pnxh : tensor<32x320x7x7xf32>
    %b16dpndg = stablehlo.reduce(%b16dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16dpndb = stablehlo.reduce(%dh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16dpt = stablehlo.transpose %b16pW, dims = [1, 0, 2, 3] : (tensor<320x1152x1x1xf32>) -> tensor<1152x320x1x1xf32>
    %b16dp = stablehlo.convolution(%b16dpn, %b16dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1152x320x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b16dpWxt = stablehlo.transpose %b16zse, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b16dpWdt = stablehlo.transpose %b16dpn, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %b16dpWraw = stablehlo.convolution(%b16dpWxt, %b16dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<1152x320x1x1xf32>
    %b16dpW = stablehlo.transpose %b16dpWraw, dims = [1, 0, 2, 3] : (tensor<1152x320x1x1xf32>) -> tensor<320x1152x1x1xf32>
    %b16dpb = stablehlo.reduce(%b16dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %b16zgb2 = stablehlo.broadcast_in_dim %b16zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16zdleft = stablehlo.multiply %b16zgb2, %b16dp : tensor<32x1152x7x7xf32>
    %b16zxdse = stablehlo.multiply %b16ds, %b16dp : tensor<32x1152x7x7xf32>
    %b16zdgate = stablehlo.reduce(%b16zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b16zone = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %b16zomg = stablehlo.subtract %b16zone, %b16zgate : tensor<32x1152xf32>
    %b16zsg = stablehlo.multiply %b16zgate, %b16zomg : tensor<32x1152xf32>
    %b16zdh2 = stablehlo.multiply %b16zdgate, %b16zsg : tensor<32x1152xf32>
    %b16zda1 = stablehlo.dot_general %b16zdh2, %b16zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %b16zdWs2 = stablehlo.dot_general %b16za1, %b16zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %b16zdbs2 = stablehlo.reduce(%b16zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16zdexs = stablehlo.logistic %b16zex : tensor<32x48xf32>
    %b16zdexone = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %b16zdexom = stablehlo.subtract %b16zdexone, %b16zdexs : tensor<32x48xf32>
    %b16zdexxom = stablehlo.multiply %b16zex, %b16zdexom : tensor<32x48xf32>
    %b16zdexin = stablehlo.add %b16zdexone, %b16zdexxom : tensor<32x48xf32>
    %b16zdexsp = stablehlo.multiply %b16zdexs, %b16zdexin : tensor<32x48xf32>
    %b16zdex = stablehlo.multiply %b16zda1, %b16zdexsp : tensor<32x48xf32>
    %b16zdsq = stablehlo.dot_general %b16zdex, %b16zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %b16zdWs1 = stablehlo.dot_general %b16zsq, %b16zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %b16zdbs1 = stablehlo.reduce(%b16zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %b16zdsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b16zdsqd = stablehlo.divide %b16zdsq, %b16zdsqnf : tensor<32x1152xf32>
    %b16zdgsp = stablehlo.broadcast_in_dim %b16zdsqd, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16zdds = stablehlo.add %b16zdleft, %b16zdgsp : tensor<32x1152x7x7xf32>
    %b16ddrs = stablehlo.logistic %b16dn : tensor<32x1152x7x7xf32>
    %b16ddrone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b16ddrom = stablehlo.subtract %b16ddrone, %b16ddrs : tensor<32x1152x7x7xf32>
    %b16ddrxom = stablehlo.multiply %b16dn, %b16ddrom : tensor<32x1152x7x7xf32>
    %b16ddrin = stablehlo.add %b16ddrone, %b16ddrxom : tensor<32x1152x7x7xf32>
    %b16ddrsp = stablehlo.multiply %b16ddrs, %b16ddrin : tensor<32x1152x7x7xf32>
    %b16ddr = stablehlo.multiply %b16zdds, %b16ddrsp : tensor<32x1152x7x7xf32>
    %b16ddndxh = stablehlo.multiply %b16dngb, %b16ddr : tensor<32x1152x7x7xf32>
    %b16ddnsdxr = stablehlo.reduce(%b16ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ddnsdx = stablehlo.broadcast_in_dim %b16ddnsdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16ddnxd = stablehlo.multiply %b16dnxh, %b16ddndxh : tensor<32x1152x7x7xf32>
    %b16ddnsxdr = stablehlo.reduce(%b16ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ddnsxd = stablehlo.broadcast_in_dim %b16ddnsxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16ddnt1 = stablehlo.multiply %b16ddndxh, %b16dnnf : tensor<32x1152x7x7xf32>
    %b16ddni1 = stablehlo.subtract %b16ddnt1, %b16ddnsdx : tensor<32x1152x7x7xf32>
    %b16ddnxs = stablehlo.multiply %b16dnxh, %b16ddnsxd : tensor<32x1152x7x7xf32>
    %b16ddni2 = stablehlo.subtract %b16ddni1, %b16ddnxs : tensor<32x1152x7x7xf32>
    %b16ddnsN = stablehlo.divide %b16dnistd, %b16dnnf : tensor<32x1152x7x7xf32>
    %b16ddn = stablehlo.multiply %b16ddnsN, %b16ddni2 : tensor<32x1152x7x7xf32>
    %b16ddndgp = stablehlo.multiply %b16ddr, %b16dnxh : tensor<32x1152x7x7xf32>
    %b16ddndg = stablehlo.reduce(%b16ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ddndb = stablehlo.reduce(%b16ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ddrev = stablehlo.reverse %b16dW, dims = [2, 3] : tensor<1152x1x3x3xf32>
    %b16dd = stablehlo.convolution(%b16ddn, %b16ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %b16ddWxt = stablehlo.transpose %b16es, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b16ddWdt = stablehlo.transpose %b16ddn, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b16ddWraw = stablehlo.convolution(%b16ddWxt, %b16ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x3x3xf32>
    %b16ddW = stablehlo.reshape %b16ddWraw : (tensor<1x1152x3x3xf32>) -> tensor<1152x1x3x3xf32>
    %b16ddb = stablehlo.reduce(%b16ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16ders = stablehlo.logistic %b16en : tensor<32x1152x7x7xf32>
    %b16derone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b16derom = stablehlo.subtract %b16derone, %b16ders : tensor<32x1152x7x7xf32>
    %b16derxom = stablehlo.multiply %b16en, %b16derom : tensor<32x1152x7x7xf32>
    %b16derin = stablehlo.add %b16derone, %b16derxom : tensor<32x1152x7x7xf32>
    %b16dersp = stablehlo.multiply %b16ders, %b16derin : tensor<32x1152x7x7xf32>
    %b16der = stablehlo.multiply %b16dd, %b16dersp : tensor<32x1152x7x7xf32>
    %b16dendxh = stablehlo.multiply %b16engb, %b16der : tensor<32x1152x7x7xf32>
    %b16densdxr = stablehlo.reduce(%b16dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16densdx = stablehlo.broadcast_in_dim %b16densdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16denxd = stablehlo.multiply %b16enxh, %b16dendxh : tensor<32x1152x7x7xf32>
    %b16densxdr = stablehlo.reduce(%b16denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16densxd = stablehlo.broadcast_in_dim %b16densxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b16dent1 = stablehlo.multiply %b16dendxh, %b16ennf : tensor<32x1152x7x7xf32>
    %b16deni1 = stablehlo.subtract %b16dent1, %b16densdx : tensor<32x1152x7x7xf32>
    %b16denxs = stablehlo.multiply %b16enxh, %b16densxd : tensor<32x1152x7x7xf32>
    %b16deni2 = stablehlo.subtract %b16deni1, %b16denxs : tensor<32x1152x7x7xf32>
    %b16densN = stablehlo.divide %b16enistd, %b16ennf : tensor<32x1152x7x7xf32>
    %b16den = stablehlo.multiply %b16densN, %b16deni2 : tensor<32x1152x7x7xf32>
    %b16dendgp = stablehlo.multiply %b16der, %b16enxh : tensor<32x1152x7x7xf32>
    %b16dendg = stablehlo.reduce(%b16dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16dendb = stablehlo.reduce(%b16der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b16det = stablehlo.transpose %b16eW, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b16de = stablehlo.convolution(%b16den, %b16det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b16deWxt = stablehlo.transpose %b15o, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b16deWdt = stablehlo.transpose %b16den, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b16deWraw = stablehlo.convolution(%b16deWxt, %b16deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %b16deW = stablehlo.transpose %b16deWraw, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b16deb = stablehlo.reduce(%b16den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dpndxh = stablehlo.multiply %b15pngb, %b16de : tensor<32x192x7x7xf32>
    %b15dpnsdxr = stablehlo.reduce(%b15dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15dpnsdx = stablehlo.broadcast_in_dim %b15dpnsdxr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15dpnxd = stablehlo.multiply %b15pnxh, %b15dpndxh : tensor<32x192x7x7xf32>
    %b15dpnsxdr = stablehlo.reduce(%b15dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15dpnsxd = stablehlo.broadcast_in_dim %b15dpnsxdr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b15dpnt1 = stablehlo.multiply %b15dpndxh, %b15pnnf : tensor<32x192x7x7xf32>
    %b15dpni1 = stablehlo.subtract %b15dpnt1, %b15dpnsdx : tensor<32x192x7x7xf32>
    %b15dpnxs = stablehlo.multiply %b15pnxh, %b15dpnsxd : tensor<32x192x7x7xf32>
    %b15dpni2 = stablehlo.subtract %b15dpni1, %b15dpnxs : tensor<32x192x7x7xf32>
    %b15dpnsN = stablehlo.divide %b15pnistd, %b15pnnf : tensor<32x192x7x7xf32>
    %b15dpn = stablehlo.multiply %b15dpnsN, %b15dpni2 : tensor<32x192x7x7xf32>
    %b15dpndgp = stablehlo.multiply %b16de, %b15pnxh : tensor<32x192x7x7xf32>
    %b15dpndg = stablehlo.reduce(%b15dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15dpndb = stablehlo.reduce(%b16de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15dpt = stablehlo.transpose %b15pW, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b15dp = stablehlo.convolution(%b15dpn, %b15dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b15dpWxt = stablehlo.transpose %b15zse, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b15dpWdt = stablehlo.transpose %b15dpn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b15dpWraw = stablehlo.convolution(%b15dpWxt, %b15dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %b15dpW = stablehlo.transpose %b15dpWraw, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b15dpb = stablehlo.reduce(%b15dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b15zgb2 = stablehlo.broadcast_in_dim %b15zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15zdleft = stablehlo.multiply %b15zgb2, %b15dp : tensor<32x1152x7x7xf32>
    %b15zxdse = stablehlo.multiply %b15ds, %b15dp : tensor<32x1152x7x7xf32>
    %b15zdgate = stablehlo.reduce(%b15zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b15zone = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %b15zomg = stablehlo.subtract %b15zone, %b15zgate : tensor<32x1152xf32>
    %b15zsg = stablehlo.multiply %b15zgate, %b15zomg : tensor<32x1152xf32>
    %b15zdh2 = stablehlo.multiply %b15zdgate, %b15zsg : tensor<32x1152xf32>
    %b15zda1 = stablehlo.dot_general %b15zdh2, %b15zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %b15zdWs2 = stablehlo.dot_general %b15za1, %b15zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %b15zdbs2 = stablehlo.reduce(%b15zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15zdexs = stablehlo.logistic %b15zex : tensor<32x48xf32>
    %b15zdexone = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %b15zdexom = stablehlo.subtract %b15zdexone, %b15zdexs : tensor<32x48xf32>
    %b15zdexxom = stablehlo.multiply %b15zex, %b15zdexom : tensor<32x48xf32>
    %b15zdexin = stablehlo.add %b15zdexone, %b15zdexxom : tensor<32x48xf32>
    %b15zdexsp = stablehlo.multiply %b15zdexs, %b15zdexin : tensor<32x48xf32>
    %b15zdex = stablehlo.multiply %b15zda1, %b15zdexsp : tensor<32x48xf32>
    %b15zdsq = stablehlo.dot_general %b15zdex, %b15zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %b15zdWs1 = stablehlo.dot_general %b15zsq, %b15zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %b15zdbs1 = stablehlo.reduce(%b15zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %b15zdsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b15zdsqd = stablehlo.divide %b15zdsq, %b15zdsqnf : tensor<32x1152xf32>
    %b15zdgsp = stablehlo.broadcast_in_dim %b15zdsqd, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15zdds = stablehlo.add %b15zdleft, %b15zdgsp : tensor<32x1152x7x7xf32>
    %b15ddrs = stablehlo.logistic %b15dn : tensor<32x1152x7x7xf32>
    %b15ddrone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b15ddrom = stablehlo.subtract %b15ddrone, %b15ddrs : tensor<32x1152x7x7xf32>
    %b15ddrxom = stablehlo.multiply %b15dn, %b15ddrom : tensor<32x1152x7x7xf32>
    %b15ddrin = stablehlo.add %b15ddrone, %b15ddrxom : tensor<32x1152x7x7xf32>
    %b15ddrsp = stablehlo.multiply %b15ddrs, %b15ddrin : tensor<32x1152x7x7xf32>
    %b15ddr = stablehlo.multiply %b15zdds, %b15ddrsp : tensor<32x1152x7x7xf32>
    %b15ddndxh = stablehlo.multiply %b15dngb, %b15ddr : tensor<32x1152x7x7xf32>
    %b15ddnsdxr = stablehlo.reduce(%b15ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ddnsdx = stablehlo.broadcast_in_dim %b15ddnsdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15ddnxd = stablehlo.multiply %b15dnxh, %b15ddndxh : tensor<32x1152x7x7xf32>
    %b15ddnsxdr = stablehlo.reduce(%b15ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ddnsxd = stablehlo.broadcast_in_dim %b15ddnsxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15ddnt1 = stablehlo.multiply %b15ddndxh, %b15dnnf : tensor<32x1152x7x7xf32>
    %b15ddni1 = stablehlo.subtract %b15ddnt1, %b15ddnsdx : tensor<32x1152x7x7xf32>
    %b15ddnxs = stablehlo.multiply %b15dnxh, %b15ddnsxd : tensor<32x1152x7x7xf32>
    %b15ddni2 = stablehlo.subtract %b15ddni1, %b15ddnxs : tensor<32x1152x7x7xf32>
    %b15ddnsN = stablehlo.divide %b15dnistd, %b15dnnf : tensor<32x1152x7x7xf32>
    %b15ddn = stablehlo.multiply %b15ddnsN, %b15ddni2 : tensor<32x1152x7x7xf32>
    %b15ddndgp = stablehlo.multiply %b15ddr, %b15dnxh : tensor<32x1152x7x7xf32>
    %b15ddndg = stablehlo.reduce(%b15ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ddndb = stablehlo.reduce(%b15ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ddrev = stablehlo.reverse %b15dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %b15dd = stablehlo.convolution(%b15ddn, %b15ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b15ddWxt = stablehlo.transpose %b15es, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b15ddWdt = stablehlo.transpose %b15ddn, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b15ddWraw = stablehlo.convolution(%b15ddWxt, %b15ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %b15ddW = stablehlo.reshape %b15ddWraw : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %b15ddb = stablehlo.reduce(%b15ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15ders = stablehlo.logistic %b15en : tensor<32x1152x7x7xf32>
    %b15derone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b15derom = stablehlo.subtract %b15derone, %b15ders : tensor<32x1152x7x7xf32>
    %b15derxom = stablehlo.multiply %b15en, %b15derom : tensor<32x1152x7x7xf32>
    %b15derin = stablehlo.add %b15derone, %b15derxom : tensor<32x1152x7x7xf32>
    %b15dersp = stablehlo.multiply %b15ders, %b15derin : tensor<32x1152x7x7xf32>
    %b15der = stablehlo.multiply %b15dd, %b15dersp : tensor<32x1152x7x7xf32>
    %b15dendxh = stablehlo.multiply %b15engb, %b15der : tensor<32x1152x7x7xf32>
    %b15densdxr = stablehlo.reduce(%b15dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15densdx = stablehlo.broadcast_in_dim %b15densdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15denxd = stablehlo.multiply %b15enxh, %b15dendxh : tensor<32x1152x7x7xf32>
    %b15densxdr = stablehlo.reduce(%b15denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15densxd = stablehlo.broadcast_in_dim %b15densxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b15dent1 = stablehlo.multiply %b15dendxh, %b15ennf : tensor<32x1152x7x7xf32>
    %b15deni1 = stablehlo.subtract %b15dent1, %b15densdx : tensor<32x1152x7x7xf32>
    %b15denxs = stablehlo.multiply %b15enxh, %b15densxd : tensor<32x1152x7x7xf32>
    %b15deni2 = stablehlo.subtract %b15deni1, %b15denxs : tensor<32x1152x7x7xf32>
    %b15densN = stablehlo.divide %b15enistd, %b15ennf : tensor<32x1152x7x7xf32>
    %b15den = stablehlo.multiply %b15densN, %b15deni2 : tensor<32x1152x7x7xf32>
    %b15dendgp = stablehlo.multiply %b15der, %b15enxh : tensor<32x1152x7x7xf32>
    %b15dendg = stablehlo.reduce(%b15dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dendb = stablehlo.reduce(%b15der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15det = stablehlo.transpose %b15eW, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b15de = stablehlo.convolution(%b15den, %b15det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b15deWxt = stablehlo.transpose %b14o, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b15deWdt = stablehlo.transpose %b15den, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b15deWraw = stablehlo.convolution(%b15deWxt, %b15deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %b15deW = stablehlo.transpose %b15deWraw, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b15deb = stablehlo.reduce(%b15den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b15dx = stablehlo.add %b15de, %b16de : tensor<32x192x7x7xf32>
    %b14dpndxh = stablehlo.multiply %b14pngb, %b15dx : tensor<32x192x7x7xf32>
    %b14dpnsdxr = stablehlo.reduce(%b14dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14dpnsdx = stablehlo.broadcast_in_dim %b14dpnsdxr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14dpnxd = stablehlo.multiply %b14pnxh, %b14dpndxh : tensor<32x192x7x7xf32>
    %b14dpnsxdr = stablehlo.reduce(%b14dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14dpnsxd = stablehlo.broadcast_in_dim %b14dpnsxdr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b14dpnt1 = stablehlo.multiply %b14dpndxh, %b14pnnf : tensor<32x192x7x7xf32>
    %b14dpni1 = stablehlo.subtract %b14dpnt1, %b14dpnsdx : tensor<32x192x7x7xf32>
    %b14dpnxs = stablehlo.multiply %b14pnxh, %b14dpnsxd : tensor<32x192x7x7xf32>
    %b14dpni2 = stablehlo.subtract %b14dpni1, %b14dpnxs : tensor<32x192x7x7xf32>
    %b14dpnsN = stablehlo.divide %b14pnistd, %b14pnnf : tensor<32x192x7x7xf32>
    %b14dpn = stablehlo.multiply %b14dpnsN, %b14dpni2 : tensor<32x192x7x7xf32>
    %b14dpndgp = stablehlo.multiply %b15dx, %b14pnxh : tensor<32x192x7x7xf32>
    %b14dpndg = stablehlo.reduce(%b14dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14dpndb = stablehlo.reduce(%b15dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14dpt = stablehlo.transpose %b14pW, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b14dp = stablehlo.convolution(%b14dpn, %b14dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b14dpWxt = stablehlo.transpose %b14zse, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b14dpWdt = stablehlo.transpose %b14dpn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b14dpWraw = stablehlo.convolution(%b14dpWxt, %b14dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %b14dpW = stablehlo.transpose %b14dpWraw, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b14dpb = stablehlo.reduce(%b14dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b14zgb2 = stablehlo.broadcast_in_dim %b14zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14zdleft = stablehlo.multiply %b14zgb2, %b14dp : tensor<32x1152x7x7xf32>
    %b14zxdse = stablehlo.multiply %b14ds, %b14dp : tensor<32x1152x7x7xf32>
    %b14zdgate = stablehlo.reduce(%b14zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b14zone = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %b14zomg = stablehlo.subtract %b14zone, %b14zgate : tensor<32x1152xf32>
    %b14zsg = stablehlo.multiply %b14zgate, %b14zomg : tensor<32x1152xf32>
    %b14zdh2 = stablehlo.multiply %b14zdgate, %b14zsg : tensor<32x1152xf32>
    %b14zda1 = stablehlo.dot_general %b14zdh2, %b14zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %b14zdWs2 = stablehlo.dot_general %b14za1, %b14zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %b14zdbs2 = stablehlo.reduce(%b14zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14zdexs = stablehlo.logistic %b14zex : tensor<32x48xf32>
    %b14zdexone = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %b14zdexom = stablehlo.subtract %b14zdexone, %b14zdexs : tensor<32x48xf32>
    %b14zdexxom = stablehlo.multiply %b14zex, %b14zdexom : tensor<32x48xf32>
    %b14zdexin = stablehlo.add %b14zdexone, %b14zdexxom : tensor<32x48xf32>
    %b14zdexsp = stablehlo.multiply %b14zdexs, %b14zdexin : tensor<32x48xf32>
    %b14zdex = stablehlo.multiply %b14zda1, %b14zdexsp : tensor<32x48xf32>
    %b14zdsq = stablehlo.dot_general %b14zdex, %b14zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %b14zdWs1 = stablehlo.dot_general %b14zsq, %b14zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %b14zdbs1 = stablehlo.reduce(%b14zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %b14zdsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b14zdsqd = stablehlo.divide %b14zdsq, %b14zdsqnf : tensor<32x1152xf32>
    %b14zdgsp = stablehlo.broadcast_in_dim %b14zdsqd, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14zdds = stablehlo.add %b14zdleft, %b14zdgsp : tensor<32x1152x7x7xf32>
    %b14ddrs = stablehlo.logistic %b14dn : tensor<32x1152x7x7xf32>
    %b14ddrone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b14ddrom = stablehlo.subtract %b14ddrone, %b14ddrs : tensor<32x1152x7x7xf32>
    %b14ddrxom = stablehlo.multiply %b14dn, %b14ddrom : tensor<32x1152x7x7xf32>
    %b14ddrin = stablehlo.add %b14ddrone, %b14ddrxom : tensor<32x1152x7x7xf32>
    %b14ddrsp = stablehlo.multiply %b14ddrs, %b14ddrin : tensor<32x1152x7x7xf32>
    %b14ddr = stablehlo.multiply %b14zdds, %b14ddrsp : tensor<32x1152x7x7xf32>
    %b14ddndxh = stablehlo.multiply %b14dngb, %b14ddr : tensor<32x1152x7x7xf32>
    %b14ddnsdxr = stablehlo.reduce(%b14ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ddnsdx = stablehlo.broadcast_in_dim %b14ddnsdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14ddnxd = stablehlo.multiply %b14dnxh, %b14ddndxh : tensor<32x1152x7x7xf32>
    %b14ddnsxdr = stablehlo.reduce(%b14ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ddnsxd = stablehlo.broadcast_in_dim %b14ddnsxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14ddnt1 = stablehlo.multiply %b14ddndxh, %b14dnnf : tensor<32x1152x7x7xf32>
    %b14ddni1 = stablehlo.subtract %b14ddnt1, %b14ddnsdx : tensor<32x1152x7x7xf32>
    %b14ddnxs = stablehlo.multiply %b14dnxh, %b14ddnsxd : tensor<32x1152x7x7xf32>
    %b14ddni2 = stablehlo.subtract %b14ddni1, %b14ddnxs : tensor<32x1152x7x7xf32>
    %b14ddnsN = stablehlo.divide %b14dnistd, %b14dnnf : tensor<32x1152x7x7xf32>
    %b14ddn = stablehlo.multiply %b14ddnsN, %b14ddni2 : tensor<32x1152x7x7xf32>
    %b14ddndgp = stablehlo.multiply %b14ddr, %b14dnxh : tensor<32x1152x7x7xf32>
    %b14ddndg = stablehlo.reduce(%b14ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ddndb = stablehlo.reduce(%b14ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ddrev = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %b14dd = stablehlo.convolution(%b14ddn, %b14ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b14ddWxt = stablehlo.transpose %b14es, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b14ddWdt = stablehlo.transpose %b14ddn, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b14ddWraw = stablehlo.convolution(%b14ddWxt, %b14ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %b14ddW = stablehlo.reshape %b14ddWraw : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %b14ddb = stablehlo.reduce(%b14ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14ders = stablehlo.logistic %b14en : tensor<32x1152x7x7xf32>
    %b14derone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b14derom = stablehlo.subtract %b14derone, %b14ders : tensor<32x1152x7x7xf32>
    %b14derxom = stablehlo.multiply %b14en, %b14derom : tensor<32x1152x7x7xf32>
    %b14derin = stablehlo.add %b14derone, %b14derxom : tensor<32x1152x7x7xf32>
    %b14dersp = stablehlo.multiply %b14ders, %b14derin : tensor<32x1152x7x7xf32>
    %b14der = stablehlo.multiply %b14dd, %b14dersp : tensor<32x1152x7x7xf32>
    %b14dendxh = stablehlo.multiply %b14engb, %b14der : tensor<32x1152x7x7xf32>
    %b14densdxr = stablehlo.reduce(%b14dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14densdx = stablehlo.broadcast_in_dim %b14densdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14denxd = stablehlo.multiply %b14enxh, %b14dendxh : tensor<32x1152x7x7xf32>
    %b14densxdr = stablehlo.reduce(%b14denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14densxd = stablehlo.broadcast_in_dim %b14densxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b14dent1 = stablehlo.multiply %b14dendxh, %b14ennf : tensor<32x1152x7x7xf32>
    %b14deni1 = stablehlo.subtract %b14dent1, %b14densdx : tensor<32x1152x7x7xf32>
    %b14denxs = stablehlo.multiply %b14enxh, %b14densxd : tensor<32x1152x7x7xf32>
    %b14deni2 = stablehlo.subtract %b14deni1, %b14denxs : tensor<32x1152x7x7xf32>
    %b14densN = stablehlo.divide %b14enistd, %b14ennf : tensor<32x1152x7x7xf32>
    %b14den = stablehlo.multiply %b14densN, %b14deni2 : tensor<32x1152x7x7xf32>
    %b14dendgp = stablehlo.multiply %b14der, %b14enxh : tensor<32x1152x7x7xf32>
    %b14dendg = stablehlo.reduce(%b14dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14dendb = stablehlo.reduce(%b14der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14det = stablehlo.transpose %b14eW, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b14de = stablehlo.convolution(%b14den, %b14det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b14deWxt = stablehlo.transpose %b13o, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b14deWdt = stablehlo.transpose %b14den, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b14deWraw = stablehlo.convolution(%b14deWxt, %b14deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %b14deW = stablehlo.transpose %b14deWraw, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b14deb = stablehlo.reduce(%b14den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b14dx = stablehlo.add %b14de, %b15dx : tensor<32x192x7x7xf32>
    %b13dpndxh = stablehlo.multiply %b13pngb, %b14dx : tensor<32x192x7x7xf32>
    %b13dpnsdxr = stablehlo.reduce(%b13dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13dpnsdx = stablehlo.broadcast_in_dim %b13dpnsdxr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13dpnxd = stablehlo.multiply %b13pnxh, %b13dpndxh : tensor<32x192x7x7xf32>
    %b13dpnsxdr = stablehlo.reduce(%b13dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13dpnsxd = stablehlo.broadcast_in_dim %b13dpnsxdr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b13dpnt1 = stablehlo.multiply %b13dpndxh, %b13pnnf : tensor<32x192x7x7xf32>
    %b13dpni1 = stablehlo.subtract %b13dpnt1, %b13dpnsdx : tensor<32x192x7x7xf32>
    %b13dpnxs = stablehlo.multiply %b13pnxh, %b13dpnsxd : tensor<32x192x7x7xf32>
    %b13dpni2 = stablehlo.subtract %b13dpni1, %b13dpnxs : tensor<32x192x7x7xf32>
    %b13dpnsN = stablehlo.divide %b13pnistd, %b13pnnf : tensor<32x192x7x7xf32>
    %b13dpn = stablehlo.multiply %b13dpnsN, %b13dpni2 : tensor<32x192x7x7xf32>
    %b13dpndgp = stablehlo.multiply %b14dx, %b13pnxh : tensor<32x192x7x7xf32>
    %b13dpndg = stablehlo.reduce(%b13dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13dpndb = stablehlo.reduce(%b14dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13dpt = stablehlo.transpose %b13pW, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b13dp = stablehlo.convolution(%b13dpn, %b13dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %b13dpWxt = stablehlo.transpose %b13zse, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b13dpWdt = stablehlo.transpose %b13dpn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b13dpWraw = stablehlo.convolution(%b13dpWxt, %b13dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %b13dpW = stablehlo.transpose %b13dpWraw, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b13dpb = stablehlo.reduce(%b13dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b13zgb2 = stablehlo.broadcast_in_dim %b13zgate, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13zdleft = stablehlo.multiply %b13zgb2, %b13dp : tensor<32x1152x7x7xf32>
    %b13zxdse = stablehlo.multiply %b13ds, %b13dp : tensor<32x1152x7x7xf32>
    %b13zdgate = stablehlo.reduce(%b13zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %b13zone = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %b13zomg = stablehlo.subtract %b13zone, %b13zgate : tensor<32x1152xf32>
    %b13zsg = stablehlo.multiply %b13zgate, %b13zomg : tensor<32x1152xf32>
    %b13zdh2 = stablehlo.multiply %b13zdgate, %b13zsg : tensor<32x1152xf32>
    %b13zda1 = stablehlo.dot_general %b13zdh2, %b13zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %b13zdWs2 = stablehlo.dot_general %b13za1, %b13zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %b13zdbs2 = stablehlo.reduce(%b13zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13zdexs = stablehlo.logistic %b13zex : tensor<32x48xf32>
    %b13zdexone = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %b13zdexom = stablehlo.subtract %b13zdexone, %b13zdexs : tensor<32x48xf32>
    %b13zdexxom = stablehlo.multiply %b13zex, %b13zdexom : tensor<32x48xf32>
    %b13zdexin = stablehlo.add %b13zdexone, %b13zdexxom : tensor<32x48xf32>
    %b13zdexsp = stablehlo.multiply %b13zdexs, %b13zdexin : tensor<32x48xf32>
    %b13zdex = stablehlo.multiply %b13zda1, %b13zdexsp : tensor<32x48xf32>
    %b13zdsq = stablehlo.dot_general %b13zdex, %b13zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %b13zdWs1 = stablehlo.dot_general %b13zsq, %b13zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %b13zdbs1 = stablehlo.reduce(%b13zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %b13zdsqnf = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %b13zdsqd = stablehlo.divide %b13zdsq, %b13zdsqnf : tensor<32x1152xf32>
    %b13zdgsp = stablehlo.broadcast_in_dim %b13zdsqd, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13zdds = stablehlo.add %b13zdleft, %b13zdgsp : tensor<32x1152x7x7xf32>
    %b13ddrs = stablehlo.logistic %b13dn : tensor<32x1152x7x7xf32>
    %b13ddrone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b13ddrom = stablehlo.subtract %b13ddrone, %b13ddrs : tensor<32x1152x7x7xf32>
    %b13ddrxom = stablehlo.multiply %b13dn, %b13ddrom : tensor<32x1152x7x7xf32>
    %b13ddrin = stablehlo.add %b13ddrone, %b13ddrxom : tensor<32x1152x7x7xf32>
    %b13ddrsp = stablehlo.multiply %b13ddrs, %b13ddrin : tensor<32x1152x7x7xf32>
    %b13ddr = stablehlo.multiply %b13zdds, %b13ddrsp : tensor<32x1152x7x7xf32>
    %b13ddndxh = stablehlo.multiply %b13dngb, %b13ddr : tensor<32x1152x7x7xf32>
    %b13ddnsdxr = stablehlo.reduce(%b13ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ddnsdx = stablehlo.broadcast_in_dim %b13ddnsdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13ddnxd = stablehlo.multiply %b13dnxh, %b13ddndxh : tensor<32x1152x7x7xf32>
    %b13ddnsxdr = stablehlo.reduce(%b13ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ddnsxd = stablehlo.broadcast_in_dim %b13ddnsxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13ddnt1 = stablehlo.multiply %b13ddndxh, %b13dnnf : tensor<32x1152x7x7xf32>
    %b13ddni1 = stablehlo.subtract %b13ddnt1, %b13ddnsdx : tensor<32x1152x7x7xf32>
    %b13ddnxs = stablehlo.multiply %b13dnxh, %b13ddnsxd : tensor<32x1152x7x7xf32>
    %b13ddni2 = stablehlo.subtract %b13ddni1, %b13ddnxs : tensor<32x1152x7x7xf32>
    %b13ddnsN = stablehlo.divide %b13dnistd, %b13dnnf : tensor<32x1152x7x7xf32>
    %b13ddn = stablehlo.multiply %b13ddnsN, %b13ddni2 : tensor<32x1152x7x7xf32>
    %b13ddndgp = stablehlo.multiply %b13ddr, %b13dnxh : tensor<32x1152x7x7xf32>
    %b13ddndg = stablehlo.reduce(%b13ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ddndb = stablehlo.reduce(%b13ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ddrev = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %b13dd = stablehlo.convolution(%b13ddn, %b13ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %b13ddWxt = stablehlo.transpose %b13es, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b13ddWdt = stablehlo.transpose %b13ddn, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b13ddWraw = stablehlo.convolution(%b13ddWxt, %b13ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %b13ddW = stablehlo.reshape %b13ddWraw : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %b13ddb = stablehlo.reduce(%b13ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13ders = stablehlo.logistic %b13en : tensor<32x1152x7x7xf32>
    %b13derone = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %b13derom = stablehlo.subtract %b13derone, %b13ders : tensor<32x1152x7x7xf32>
    %b13derxom = stablehlo.multiply %b13en, %b13derom : tensor<32x1152x7x7xf32>
    %b13derin = stablehlo.add %b13derone, %b13derxom : tensor<32x1152x7x7xf32>
    %b13dersp = stablehlo.multiply %b13ders, %b13derin : tensor<32x1152x7x7xf32>
    %b13der = stablehlo.multiply %b13dd, %b13dersp : tensor<32x1152x7x7xf32>
    %b13dendxh = stablehlo.multiply %b13engb, %b13der : tensor<32x1152x7x7xf32>
    %b13densdxr = stablehlo.reduce(%b13dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13densdx = stablehlo.broadcast_in_dim %b13densdxr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13denxd = stablehlo.multiply %b13enxh, %b13dendxh : tensor<32x1152x7x7xf32>
    %b13densxdr = stablehlo.reduce(%b13denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13densxd = stablehlo.broadcast_in_dim %b13densxdr, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %b13dent1 = stablehlo.multiply %b13dendxh, %b13ennf : tensor<32x1152x7x7xf32>
    %b13deni1 = stablehlo.subtract %b13dent1, %b13densdx : tensor<32x1152x7x7xf32>
    %b13denxs = stablehlo.multiply %b13enxh, %b13densxd : tensor<32x1152x7x7xf32>
    %b13deni2 = stablehlo.subtract %b13deni1, %b13denxs : tensor<32x1152x7x7xf32>
    %b13densN = stablehlo.divide %b13enistd, %b13ennf : tensor<32x1152x7x7xf32>
    %b13den = stablehlo.multiply %b13densN, %b13deni2 : tensor<32x1152x7x7xf32>
    %b13dendgp = stablehlo.multiply %b13der, %b13enxh : tensor<32x1152x7x7xf32>
    %b13dendg = stablehlo.reduce(%b13dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13dendb = stablehlo.reduce(%b13der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13det = stablehlo.transpose %b13eW, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %b13de = stablehlo.convolution(%b13den, %b13det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %b13deWxt = stablehlo.transpose %b12pn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b13deWdt = stablehlo.transpose %b13den, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %b13deWraw = stablehlo.convolution(%b13deWxt, %b13deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %b13deW = stablehlo.transpose %b13deWraw, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %b13deb = stablehlo.reduce(%b13den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %b13dx = stablehlo.add %b13de, %b14dx : tensor<32x192x7x7xf32>
    %b12dpndxh = stablehlo.multiply %b12pngb, %b13dx : tensor<32x192x7x7xf32>
    %b12dpnsdxr = stablehlo.reduce(%b12dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12dpnsdx = stablehlo.broadcast_in_dim %b12dpnsdxr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12dpnxd = stablehlo.multiply %b12pnxh, %b12dpndxh : tensor<32x192x7x7xf32>
    %b12dpnsxdr = stablehlo.reduce(%b12dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12dpnsxd = stablehlo.broadcast_in_dim %b12dpnsxdr, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %b12dpnt1 = stablehlo.multiply %b12dpndxh, %b12pnnf : tensor<32x192x7x7xf32>
    %b12dpni1 = stablehlo.subtract %b12dpnt1, %b12dpnsdx : tensor<32x192x7x7xf32>
    %b12dpnxs = stablehlo.multiply %b12pnxh, %b12dpnsxd : tensor<32x192x7x7xf32>
    %b12dpni2 = stablehlo.subtract %b12dpni1, %b12dpnxs : tensor<32x192x7x7xf32>
    %b12dpnsN = stablehlo.divide %b12pnistd, %b12pnnf : tensor<32x192x7x7xf32>
    %b12dpn = stablehlo.multiply %b12dpnsN, %b12dpni2 : tensor<32x192x7x7xf32>
    %b12dpndgp = stablehlo.multiply %b13dx, %b12pnxh : tensor<32x192x7x7xf32>
    %b12dpndg = stablehlo.reduce(%b12dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12dpndb = stablehlo.reduce(%b13dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12dpt = stablehlo.transpose %b12pW, dims = [1, 0, 2, 3] : (tensor<192x672x1x1xf32>) -> tensor<672x192x1x1xf32>
    %b12dp = stablehlo.convolution(%b12dpn, %b12dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<672x192x1x1xf32>) -> tensor<32x672x7x7xf32>
    %b12dpWxt = stablehlo.transpose %b12zse, dims = [1, 0, 2, 3] : (tensor<32x672x7x7xf32>) -> tensor<672x32x7x7xf32>
    %b12dpWdt = stablehlo.transpose %b12dpn, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %b12dpWraw = stablehlo.convolution(%b12dpWxt, %b12dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<672x192x1x1xf32>
    %b12dpW = stablehlo.transpose %b12dpWraw, dims = [1, 0, 2, 3] : (tensor<672x192x1x1xf32>) -> tensor<192x672x1x1xf32>
    %b12dpb = stablehlo.reduce(%b12dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %b12zgb2 = stablehlo.broadcast_in_dim %b12zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %b12zdleft = stablehlo.multiply %b12zgb2, %b12dp : tensor<32x672x7x7xf32>
    %b12zxdse = stablehlo.multiply %b12ds, %b12dp : tensor<32x672x7x7xf32>
    %b12zdgate = stablehlo.reduce(%b12zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b12zone = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %b12zomg = stablehlo.subtract %b12zone, %b12zgate : tensor<32x672xf32>
    %b12zsg = stablehlo.multiply %b12zgate, %b12zomg : tensor<32x672xf32>
    %b12zdh2 = stablehlo.multiply %b12zdgate, %b12zsg : tensor<32x672xf32>
    %b12zda1 = stablehlo.dot_general %b12zdh2, %b12zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %b12zdWs2 = stablehlo.dot_general %b12za1, %b12zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %b12zdbs2 = stablehlo.reduce(%b12zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %b12zdexs = stablehlo.logistic %b12zex : tensor<32x28xf32>
    %b12zdexone = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %b12zdexom = stablehlo.subtract %b12zdexone, %b12zdexs : tensor<32x28xf32>
    %b12zdexxom = stablehlo.multiply %b12zex, %b12zdexom : tensor<32x28xf32>
    %b12zdexin = stablehlo.add %b12zdexone, %b12zdexxom : tensor<32x28xf32>
    %b12zdexsp = stablehlo.multiply %b12zdexs, %b12zdexin : tensor<32x28xf32>
    %b12zdex = stablehlo.multiply %b12zda1, %b12zdexsp : tensor<32x28xf32>
    %b12zdsq = stablehlo.dot_general %b12zdex, %b12zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %b12zdWs1 = stablehlo.dot_general %b12zsq, %b12zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %b12zdbs1 = stablehlo.reduce(%b12zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %b12zdsqnf = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %b12zdsqd = stablehlo.divide %b12zdsq, %b12zdsqnf : tensor<32x672xf32>
    %b12zdgsp = stablehlo.broadcast_in_dim %b12zdsqd, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %b12zdds = stablehlo.add %b12zdleft, %b12zdgsp : tensor<32x672x7x7xf32>
    %b12ddrs = stablehlo.logistic %b12dn : tensor<32x672x7x7xf32>
    %b12ddrone = stablehlo.constant dense<1.0> : tensor<32x672x7x7xf32>
    %b12ddrom = stablehlo.subtract %b12ddrone, %b12ddrs : tensor<32x672x7x7xf32>
    %b12ddrxom = stablehlo.multiply %b12dn, %b12ddrom : tensor<32x672x7x7xf32>
    %b12ddrin = stablehlo.add %b12ddrone, %b12ddrxom : tensor<32x672x7x7xf32>
    %b12ddrsp = stablehlo.multiply %b12ddrs, %b12ddrin : tensor<32x672x7x7xf32>
    %b12ddr = stablehlo.multiply %b12zdds, %b12ddrsp : tensor<32x672x7x7xf32>
    %b12ddndxh = stablehlo.multiply %b12dngb, %b12ddr : tensor<32x672x7x7xf32>
    %b12ddnsdxr = stablehlo.reduce(%b12ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ddnsdx = stablehlo.broadcast_in_dim %b12ddnsdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12ddnxd = stablehlo.multiply %b12dnxh, %b12ddndxh : tensor<32x672x7x7xf32>
    %b12ddnsxdr = stablehlo.reduce(%b12ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ddnsxd = stablehlo.broadcast_in_dim %b12ddnsxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %b12ddnt1 = stablehlo.multiply %b12ddndxh, %b12dnnf : tensor<32x672x7x7xf32>
    %b12ddni1 = stablehlo.subtract %b12ddnt1, %b12ddnsdx : tensor<32x672x7x7xf32>
    %b12ddnxs = stablehlo.multiply %b12dnxh, %b12ddnsxd : tensor<32x672x7x7xf32>
    %b12ddni2 = stablehlo.subtract %b12ddni1, %b12ddnxs : tensor<32x672x7x7xf32>
    %b12ddnsN = stablehlo.divide %b12dnistd, %b12dnnf : tensor<32x672x7x7xf32>
    %b12ddn = stablehlo.multiply %b12ddnsN, %b12ddni2 : tensor<32x672x7x7xf32>
    %b12ddndgp = stablehlo.multiply %b12ddr, %b12dnxh : tensor<32x672x7x7xf32>
    %b12ddndg = stablehlo.reduce(%b12ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ddndb = stablehlo.reduce(%b12ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ddu = stablehlo.pad %b12ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %b12ddrev = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %b12dd = stablehlo.convolution(%b12ddu, %b12ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b12ddWu = stablehlo.pad %b12ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %b12ddWxt = stablehlo.transpose %b12es, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b12ddWdt = stablehlo.transpose %b12ddWu, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b12ddWraw = stablehlo.convolution(%b12ddWxt, %b12ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %b12ddW = stablehlo.reshape %b12ddWraw : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %b12ddb = stablehlo.reduce(%b12ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %b12ders = stablehlo.logistic %b12en : tensor<32x672x14x14xf32>
    %b12derone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b12derom = stablehlo.subtract %b12derone, %b12ders : tensor<32x672x14x14xf32>
    %b12derxom = stablehlo.multiply %b12en, %b12derom : tensor<32x672x14x14xf32>
    %b12derin = stablehlo.add %b12derone, %b12derxom : tensor<32x672x14x14xf32>
    %b12dersp = stablehlo.multiply %b12ders, %b12derin : tensor<32x672x14x14xf32>
    %b12der = stablehlo.multiply %b12dd, %b12dersp : tensor<32x672x14x14xf32>
    %b12dendxh = stablehlo.multiply %b12engb, %b12der : tensor<32x672x14x14xf32>
    %b12densdxr = stablehlo.reduce(%b12dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12densdx = stablehlo.broadcast_in_dim %b12densdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12denxd = stablehlo.multiply %b12enxh, %b12dendxh : tensor<32x672x14x14xf32>
    %b12densxdr = stablehlo.reduce(%b12denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12densxd = stablehlo.broadcast_in_dim %b12densxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b12dent1 = stablehlo.multiply %b12dendxh, %b12ennf : tensor<32x672x14x14xf32>
    %b12deni1 = stablehlo.subtract %b12dent1, %b12densdx : tensor<32x672x14x14xf32>
    %b12denxs = stablehlo.multiply %b12enxh, %b12densxd : tensor<32x672x14x14xf32>
    %b12deni2 = stablehlo.subtract %b12deni1, %b12denxs : tensor<32x672x14x14xf32>
    %b12densN = stablehlo.divide %b12enistd, %b12ennf : tensor<32x672x14x14xf32>
    %b12den = stablehlo.multiply %b12densN, %b12deni2 : tensor<32x672x14x14xf32>
    %b12dendgp = stablehlo.multiply %b12der, %b12enxh : tensor<32x672x14x14xf32>
    %b12dendg = stablehlo.reduce(%b12dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12dendb = stablehlo.reduce(%b12der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b12det = stablehlo.transpose %b12eW, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b12de = stablehlo.convolution(%b12den, %b12det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b12deWxt = stablehlo.transpose %b11o, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b12deWdt = stablehlo.transpose %b12den, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b12deWraw = stablehlo.convolution(%b12deWxt, %b12deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %b12deW = stablehlo.transpose %b12deWraw, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b12deb = stablehlo.reduce(%b12den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dpndxh = stablehlo.multiply %b11pngb, %b12de : tensor<32x112x14x14xf32>
    %b11dpnsdxr = stablehlo.reduce(%b11dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11dpnsdx = stablehlo.broadcast_in_dim %b11dpnsdxr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11dpnxd = stablehlo.multiply %b11pnxh, %b11dpndxh : tensor<32x112x14x14xf32>
    %b11dpnsxdr = stablehlo.reduce(%b11dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11dpnsxd = stablehlo.broadcast_in_dim %b11dpnsxdr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b11dpnt1 = stablehlo.multiply %b11dpndxh, %b11pnnf : tensor<32x112x14x14xf32>
    %b11dpni1 = stablehlo.subtract %b11dpnt1, %b11dpnsdx : tensor<32x112x14x14xf32>
    %b11dpnxs = stablehlo.multiply %b11pnxh, %b11dpnsxd : tensor<32x112x14x14xf32>
    %b11dpni2 = stablehlo.subtract %b11dpni1, %b11dpnxs : tensor<32x112x14x14xf32>
    %b11dpnsN = stablehlo.divide %b11pnistd, %b11pnnf : tensor<32x112x14x14xf32>
    %b11dpn = stablehlo.multiply %b11dpnsN, %b11dpni2 : tensor<32x112x14x14xf32>
    %b11dpndgp = stablehlo.multiply %b12de, %b11pnxh : tensor<32x112x14x14xf32>
    %b11dpndg = stablehlo.reduce(%b11dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11dpndb = stablehlo.reduce(%b12de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11dpt = stablehlo.transpose %b11pW, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b11dp = stablehlo.convolution(%b11dpn, %b11dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b11dpWxt = stablehlo.transpose %b11zse, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b11dpWdt = stablehlo.transpose %b11dpn, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b11dpWraw = stablehlo.convolution(%b11dpWxt, %b11dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %b11dpW = stablehlo.transpose %b11dpWraw, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b11dpb = stablehlo.reduce(%b11dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b11zgb2 = stablehlo.broadcast_in_dim %b11zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b11zdleft = stablehlo.multiply %b11zgb2, %b11dp : tensor<32x672x14x14xf32>
    %b11zxdse = stablehlo.multiply %b11ds, %b11dp : tensor<32x672x14x14xf32>
    %b11zdgate = stablehlo.reduce(%b11zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b11zone = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %b11zomg = stablehlo.subtract %b11zone, %b11zgate : tensor<32x672xf32>
    %b11zsg = stablehlo.multiply %b11zgate, %b11zomg : tensor<32x672xf32>
    %b11zdh2 = stablehlo.multiply %b11zdgate, %b11zsg : tensor<32x672xf32>
    %b11zda1 = stablehlo.dot_general %b11zdh2, %b11zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %b11zdWs2 = stablehlo.dot_general %b11za1, %b11zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %b11zdbs2 = stablehlo.reduce(%b11zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %b11zdexs = stablehlo.logistic %b11zex : tensor<32x28xf32>
    %b11zdexone = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %b11zdexom = stablehlo.subtract %b11zdexone, %b11zdexs : tensor<32x28xf32>
    %b11zdexxom = stablehlo.multiply %b11zex, %b11zdexom : tensor<32x28xf32>
    %b11zdexin = stablehlo.add %b11zdexone, %b11zdexxom : tensor<32x28xf32>
    %b11zdexsp = stablehlo.multiply %b11zdexs, %b11zdexin : tensor<32x28xf32>
    %b11zdex = stablehlo.multiply %b11zda1, %b11zdexsp : tensor<32x28xf32>
    %b11zdsq = stablehlo.dot_general %b11zdex, %b11zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %b11zdWs1 = stablehlo.dot_general %b11zsq, %b11zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %b11zdbs1 = stablehlo.reduce(%b11zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %b11zdsqnf = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %b11zdsqd = stablehlo.divide %b11zdsq, %b11zdsqnf : tensor<32x672xf32>
    %b11zdgsp = stablehlo.broadcast_in_dim %b11zdsqd, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b11zdds = stablehlo.add %b11zdleft, %b11zdgsp : tensor<32x672x14x14xf32>
    %b11ddrs = stablehlo.logistic %b11dn : tensor<32x672x14x14xf32>
    %b11ddrone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b11ddrom = stablehlo.subtract %b11ddrone, %b11ddrs : tensor<32x672x14x14xf32>
    %b11ddrxom = stablehlo.multiply %b11dn, %b11ddrom : tensor<32x672x14x14xf32>
    %b11ddrin = stablehlo.add %b11ddrone, %b11ddrxom : tensor<32x672x14x14xf32>
    %b11ddrsp = stablehlo.multiply %b11ddrs, %b11ddrin : tensor<32x672x14x14xf32>
    %b11ddr = stablehlo.multiply %b11zdds, %b11ddrsp : tensor<32x672x14x14xf32>
    %b11ddndxh = stablehlo.multiply %b11dngb, %b11ddr : tensor<32x672x14x14xf32>
    %b11ddnsdxr = stablehlo.reduce(%b11ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ddnsdx = stablehlo.broadcast_in_dim %b11ddnsdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11ddnxd = stablehlo.multiply %b11dnxh, %b11ddndxh : tensor<32x672x14x14xf32>
    %b11ddnsxdr = stablehlo.reduce(%b11ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ddnsxd = stablehlo.broadcast_in_dim %b11ddnsxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11ddnt1 = stablehlo.multiply %b11ddndxh, %b11dnnf : tensor<32x672x14x14xf32>
    %b11ddni1 = stablehlo.subtract %b11ddnt1, %b11ddnsdx : tensor<32x672x14x14xf32>
    %b11ddnxs = stablehlo.multiply %b11dnxh, %b11ddnsxd : tensor<32x672x14x14xf32>
    %b11ddni2 = stablehlo.subtract %b11ddni1, %b11ddnxs : tensor<32x672x14x14xf32>
    %b11ddnsN = stablehlo.divide %b11dnistd, %b11dnnf : tensor<32x672x14x14xf32>
    %b11ddn = stablehlo.multiply %b11ddnsN, %b11ddni2 : tensor<32x672x14x14xf32>
    %b11ddndgp = stablehlo.multiply %b11ddr, %b11dnxh : tensor<32x672x14x14xf32>
    %b11ddndg = stablehlo.reduce(%b11ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ddndb = stablehlo.reduce(%b11ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ddrev = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %b11dd = stablehlo.convolution(%b11ddn, %b11ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b11ddWxt = stablehlo.transpose %b11es, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b11ddWdt = stablehlo.transpose %b11ddn, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b11ddWraw = stablehlo.convolution(%b11ddWxt, %b11ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %b11ddW = stablehlo.reshape %b11ddWraw : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %b11ddb = stablehlo.reduce(%b11ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11ders = stablehlo.logistic %b11en : tensor<32x672x14x14xf32>
    %b11derone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b11derom = stablehlo.subtract %b11derone, %b11ders : tensor<32x672x14x14xf32>
    %b11derxom = stablehlo.multiply %b11en, %b11derom : tensor<32x672x14x14xf32>
    %b11derin = stablehlo.add %b11derone, %b11derxom : tensor<32x672x14x14xf32>
    %b11dersp = stablehlo.multiply %b11ders, %b11derin : tensor<32x672x14x14xf32>
    %b11der = stablehlo.multiply %b11dd, %b11dersp : tensor<32x672x14x14xf32>
    %b11dendxh = stablehlo.multiply %b11engb, %b11der : tensor<32x672x14x14xf32>
    %b11densdxr = stablehlo.reduce(%b11dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11densdx = stablehlo.broadcast_in_dim %b11densdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11denxd = stablehlo.multiply %b11enxh, %b11dendxh : tensor<32x672x14x14xf32>
    %b11densxdr = stablehlo.reduce(%b11denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11densxd = stablehlo.broadcast_in_dim %b11densxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b11dent1 = stablehlo.multiply %b11dendxh, %b11ennf : tensor<32x672x14x14xf32>
    %b11deni1 = stablehlo.subtract %b11dent1, %b11densdx : tensor<32x672x14x14xf32>
    %b11denxs = stablehlo.multiply %b11enxh, %b11densxd : tensor<32x672x14x14xf32>
    %b11deni2 = stablehlo.subtract %b11deni1, %b11denxs : tensor<32x672x14x14xf32>
    %b11densN = stablehlo.divide %b11enistd, %b11ennf : tensor<32x672x14x14xf32>
    %b11den = stablehlo.multiply %b11densN, %b11deni2 : tensor<32x672x14x14xf32>
    %b11dendgp = stablehlo.multiply %b11der, %b11enxh : tensor<32x672x14x14xf32>
    %b11dendg = stablehlo.reduce(%b11dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dendb = stablehlo.reduce(%b11der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11det = stablehlo.transpose %b11eW, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b11de = stablehlo.convolution(%b11den, %b11det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b11deWxt = stablehlo.transpose %b10o, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b11deWdt = stablehlo.transpose %b11den, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b11deWraw = stablehlo.convolution(%b11deWxt, %b11deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %b11deW = stablehlo.transpose %b11deWraw, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b11deb = stablehlo.reduce(%b11den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b11dx = stablehlo.add %b11de, %b12de : tensor<32x112x14x14xf32>
    %b10dpndxh = stablehlo.multiply %b10pngb, %b11dx : tensor<32x112x14x14xf32>
    %b10dpnsdxr = stablehlo.reduce(%b10dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10dpnsdx = stablehlo.broadcast_in_dim %b10dpnsdxr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10dpnxd = stablehlo.multiply %b10pnxh, %b10dpndxh : tensor<32x112x14x14xf32>
    %b10dpnsxdr = stablehlo.reduce(%b10dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10dpnsxd = stablehlo.broadcast_in_dim %b10dpnsxdr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b10dpnt1 = stablehlo.multiply %b10dpndxh, %b10pnnf : tensor<32x112x14x14xf32>
    %b10dpni1 = stablehlo.subtract %b10dpnt1, %b10dpnsdx : tensor<32x112x14x14xf32>
    %b10dpnxs = stablehlo.multiply %b10pnxh, %b10dpnsxd : tensor<32x112x14x14xf32>
    %b10dpni2 = stablehlo.subtract %b10dpni1, %b10dpnxs : tensor<32x112x14x14xf32>
    %b10dpnsN = stablehlo.divide %b10pnistd, %b10pnnf : tensor<32x112x14x14xf32>
    %b10dpn = stablehlo.multiply %b10dpnsN, %b10dpni2 : tensor<32x112x14x14xf32>
    %b10dpndgp = stablehlo.multiply %b11dx, %b10pnxh : tensor<32x112x14x14xf32>
    %b10dpndg = stablehlo.reduce(%b10dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10dpndb = stablehlo.reduce(%b11dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10dpt = stablehlo.transpose %b10pW, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b10dp = stablehlo.convolution(%b10dpn, %b10dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %b10dpWxt = stablehlo.transpose %b10zse, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b10dpWdt = stablehlo.transpose %b10dpn, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b10dpWraw = stablehlo.convolution(%b10dpWxt, %b10dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %b10dpW = stablehlo.transpose %b10dpWraw, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b10dpb = stablehlo.reduce(%b10dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b10zgb2 = stablehlo.broadcast_in_dim %b10zgate, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b10zdleft = stablehlo.multiply %b10zgb2, %b10dp : tensor<32x672x14x14xf32>
    %b10zxdse = stablehlo.multiply %b10ds, %b10dp : tensor<32x672x14x14xf32>
    %b10zdgate = stablehlo.reduce(%b10zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %b10zone = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %b10zomg = stablehlo.subtract %b10zone, %b10zgate : tensor<32x672xf32>
    %b10zsg = stablehlo.multiply %b10zgate, %b10zomg : tensor<32x672xf32>
    %b10zdh2 = stablehlo.multiply %b10zdgate, %b10zsg : tensor<32x672xf32>
    %b10zda1 = stablehlo.dot_general %b10zdh2, %b10zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %b10zdWs2 = stablehlo.dot_general %b10za1, %b10zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %b10zdbs2 = stablehlo.reduce(%b10zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %b10zdexs = stablehlo.logistic %b10zex : tensor<32x28xf32>
    %b10zdexone = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %b10zdexom = stablehlo.subtract %b10zdexone, %b10zdexs : tensor<32x28xf32>
    %b10zdexxom = stablehlo.multiply %b10zex, %b10zdexom : tensor<32x28xf32>
    %b10zdexin = stablehlo.add %b10zdexone, %b10zdexxom : tensor<32x28xf32>
    %b10zdexsp = stablehlo.multiply %b10zdexs, %b10zdexin : tensor<32x28xf32>
    %b10zdex = stablehlo.multiply %b10zda1, %b10zdexsp : tensor<32x28xf32>
    %b10zdsq = stablehlo.dot_general %b10zdex, %b10zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %b10zdWs1 = stablehlo.dot_general %b10zsq, %b10zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %b10zdbs1 = stablehlo.reduce(%b10zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %b10zdsqnf = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %b10zdsqd = stablehlo.divide %b10zdsq, %b10zdsqnf : tensor<32x672xf32>
    %b10zdgsp = stablehlo.broadcast_in_dim %b10zdsqd, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %b10zdds = stablehlo.add %b10zdleft, %b10zdgsp : tensor<32x672x14x14xf32>
    %b10ddrs = stablehlo.logistic %b10dn : tensor<32x672x14x14xf32>
    %b10ddrone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b10ddrom = stablehlo.subtract %b10ddrone, %b10ddrs : tensor<32x672x14x14xf32>
    %b10ddrxom = stablehlo.multiply %b10dn, %b10ddrom : tensor<32x672x14x14xf32>
    %b10ddrin = stablehlo.add %b10ddrone, %b10ddrxom : tensor<32x672x14x14xf32>
    %b10ddrsp = stablehlo.multiply %b10ddrs, %b10ddrin : tensor<32x672x14x14xf32>
    %b10ddr = stablehlo.multiply %b10zdds, %b10ddrsp : tensor<32x672x14x14xf32>
    %b10ddndxh = stablehlo.multiply %b10dngb, %b10ddr : tensor<32x672x14x14xf32>
    %b10ddnsdxr = stablehlo.reduce(%b10ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ddnsdx = stablehlo.broadcast_in_dim %b10ddnsdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10ddnxd = stablehlo.multiply %b10dnxh, %b10ddndxh : tensor<32x672x14x14xf32>
    %b10ddnsxdr = stablehlo.reduce(%b10ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ddnsxd = stablehlo.broadcast_in_dim %b10ddnsxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10ddnt1 = stablehlo.multiply %b10ddndxh, %b10dnnf : tensor<32x672x14x14xf32>
    %b10ddni1 = stablehlo.subtract %b10ddnt1, %b10ddnsdx : tensor<32x672x14x14xf32>
    %b10ddnxs = stablehlo.multiply %b10dnxh, %b10ddnsxd : tensor<32x672x14x14xf32>
    %b10ddni2 = stablehlo.subtract %b10ddni1, %b10ddnxs : tensor<32x672x14x14xf32>
    %b10ddnsN = stablehlo.divide %b10dnistd, %b10dnnf : tensor<32x672x14x14xf32>
    %b10ddn = stablehlo.multiply %b10ddnsN, %b10ddni2 : tensor<32x672x14x14xf32>
    %b10ddndgp = stablehlo.multiply %b10ddr, %b10dnxh : tensor<32x672x14x14xf32>
    %b10ddndg = stablehlo.reduce(%b10ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ddndb = stablehlo.reduce(%b10ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ddrev = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %b10dd = stablehlo.convolution(%b10ddn, %b10ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %b10ddWxt = stablehlo.transpose %b10es, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b10ddWdt = stablehlo.transpose %b10ddn, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b10ddWraw = stablehlo.convolution(%b10ddWxt, %b10ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %b10ddW = stablehlo.reshape %b10ddWraw : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %b10ddb = stablehlo.reduce(%b10ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10ders = stablehlo.logistic %b10en : tensor<32x672x14x14xf32>
    %b10derone = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %b10derom = stablehlo.subtract %b10derone, %b10ders : tensor<32x672x14x14xf32>
    %b10derxom = stablehlo.multiply %b10en, %b10derom : tensor<32x672x14x14xf32>
    %b10derin = stablehlo.add %b10derone, %b10derxom : tensor<32x672x14x14xf32>
    %b10dersp = stablehlo.multiply %b10ders, %b10derin : tensor<32x672x14x14xf32>
    %b10der = stablehlo.multiply %b10dd, %b10dersp : tensor<32x672x14x14xf32>
    %b10dendxh = stablehlo.multiply %b10engb, %b10der : tensor<32x672x14x14xf32>
    %b10densdxr = stablehlo.reduce(%b10dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10densdx = stablehlo.broadcast_in_dim %b10densdxr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10denxd = stablehlo.multiply %b10enxh, %b10dendxh : tensor<32x672x14x14xf32>
    %b10densxdr = stablehlo.reduce(%b10denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10densxd = stablehlo.broadcast_in_dim %b10densxdr, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %b10dent1 = stablehlo.multiply %b10dendxh, %b10ennf : tensor<32x672x14x14xf32>
    %b10deni1 = stablehlo.subtract %b10dent1, %b10densdx : tensor<32x672x14x14xf32>
    %b10denxs = stablehlo.multiply %b10enxh, %b10densxd : tensor<32x672x14x14xf32>
    %b10deni2 = stablehlo.subtract %b10deni1, %b10denxs : tensor<32x672x14x14xf32>
    %b10densN = stablehlo.divide %b10enistd, %b10ennf : tensor<32x672x14x14xf32>
    %b10den = stablehlo.multiply %b10densN, %b10deni2 : tensor<32x672x14x14xf32>
    %b10dendgp = stablehlo.multiply %b10der, %b10enxh : tensor<32x672x14x14xf32>
    %b10dendg = stablehlo.reduce(%b10dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10dendb = stablehlo.reduce(%b10der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10det = stablehlo.transpose %b10eW, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %b10de = stablehlo.convolution(%b10den, %b10det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %b10deWxt = stablehlo.transpose %b9pn, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b10deWdt = stablehlo.transpose %b10den, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %b10deWraw = stablehlo.convolution(%b10deWxt, %b10deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %b10deW = stablehlo.transpose %b10deWraw, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %b10deb = stablehlo.reduce(%b10den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %b10dx = stablehlo.add %b10de, %b11dx : tensor<32x112x14x14xf32>
    %b9dpndxh = stablehlo.multiply %b9pngb, %b10dx : tensor<32x112x14x14xf32>
    %b9dpnsdxr = stablehlo.reduce(%b9dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9dpnsdx = stablehlo.broadcast_in_dim %b9dpnsdxr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9dpnxd = stablehlo.multiply %b9pnxh, %b9dpndxh : tensor<32x112x14x14xf32>
    %b9dpnsxdr = stablehlo.reduce(%b9dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9dpnsxd = stablehlo.broadcast_in_dim %b9dpnsxdr, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %b9dpnt1 = stablehlo.multiply %b9dpndxh, %b9pnnf : tensor<32x112x14x14xf32>
    %b9dpni1 = stablehlo.subtract %b9dpnt1, %b9dpnsdx : tensor<32x112x14x14xf32>
    %b9dpnxs = stablehlo.multiply %b9pnxh, %b9dpnsxd : tensor<32x112x14x14xf32>
    %b9dpni2 = stablehlo.subtract %b9dpni1, %b9dpnxs : tensor<32x112x14x14xf32>
    %b9dpnsN = stablehlo.divide %b9pnistd, %b9pnnf : tensor<32x112x14x14xf32>
    %b9dpn = stablehlo.multiply %b9dpnsN, %b9dpni2 : tensor<32x112x14x14xf32>
    %b9dpndgp = stablehlo.multiply %b10dx, %b9pnxh : tensor<32x112x14x14xf32>
    %b9dpndg = stablehlo.reduce(%b9dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9dpndb = stablehlo.reduce(%b10dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9dpt = stablehlo.transpose %b9pW, dims = [1, 0, 2, 3] : (tensor<112x480x1x1xf32>) -> tensor<480x112x1x1xf32>
    %b9dp = stablehlo.convolution(%b9dpn, %b9dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<480x112x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b9dpWxt = stablehlo.transpose %b9zse, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b9dpWdt = stablehlo.transpose %b9dpn, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %b9dpWraw = stablehlo.convolution(%b9dpWxt, %b9dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<480x112x1x1xf32>
    %b9dpW = stablehlo.transpose %b9dpWraw, dims = [1, 0, 2, 3] : (tensor<480x112x1x1xf32>) -> tensor<112x480x1x1xf32>
    %b9dpb = stablehlo.reduce(%b9dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %b9zgb2 = stablehlo.broadcast_in_dim %b9zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b9zdleft = stablehlo.multiply %b9zgb2, %b9dp : tensor<32x480x14x14xf32>
    %b9zxdse = stablehlo.multiply %b9ds, %b9dp : tensor<32x480x14x14xf32>
    %b9zdgate = stablehlo.reduce(%b9zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b9zone = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %b9zomg = stablehlo.subtract %b9zone, %b9zgate : tensor<32x480xf32>
    %b9zsg = stablehlo.multiply %b9zgate, %b9zomg : tensor<32x480xf32>
    %b9zdh2 = stablehlo.multiply %b9zdgate, %b9zsg : tensor<32x480xf32>
    %b9zda1 = stablehlo.dot_general %b9zdh2, %b9zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %b9zdWs2 = stablehlo.dot_general %b9za1, %b9zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %b9zdbs2 = stablehlo.reduce(%b9zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %b9zdexs = stablehlo.logistic %b9zex : tensor<32x20xf32>
    %b9zdexone = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %b9zdexom = stablehlo.subtract %b9zdexone, %b9zdexs : tensor<32x20xf32>
    %b9zdexxom = stablehlo.multiply %b9zex, %b9zdexom : tensor<32x20xf32>
    %b9zdexin = stablehlo.add %b9zdexone, %b9zdexxom : tensor<32x20xf32>
    %b9zdexsp = stablehlo.multiply %b9zdexs, %b9zdexin : tensor<32x20xf32>
    %b9zdex = stablehlo.multiply %b9zda1, %b9zdexsp : tensor<32x20xf32>
    %b9zdsq = stablehlo.dot_general %b9zdex, %b9zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %b9zdWs1 = stablehlo.dot_general %b9zsq, %b9zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %b9zdbs1 = stablehlo.reduce(%b9zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %b9zdsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b9zdsqd = stablehlo.divide %b9zdsq, %b9zdsqnf : tensor<32x480xf32>
    %b9zdgsp = stablehlo.broadcast_in_dim %b9zdsqd, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b9zdds = stablehlo.add %b9zdleft, %b9zdgsp : tensor<32x480x14x14xf32>
    %b9ddrs = stablehlo.logistic %b9dn : tensor<32x480x14x14xf32>
    %b9ddrone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b9ddrom = stablehlo.subtract %b9ddrone, %b9ddrs : tensor<32x480x14x14xf32>
    %b9ddrxom = stablehlo.multiply %b9dn, %b9ddrom : tensor<32x480x14x14xf32>
    %b9ddrin = stablehlo.add %b9ddrone, %b9ddrxom : tensor<32x480x14x14xf32>
    %b9ddrsp = stablehlo.multiply %b9ddrs, %b9ddrin : tensor<32x480x14x14xf32>
    %b9ddr = stablehlo.multiply %b9zdds, %b9ddrsp : tensor<32x480x14x14xf32>
    %b9ddndxh = stablehlo.multiply %b9dngb, %b9ddr : tensor<32x480x14x14xf32>
    %b9ddnsdxr = stablehlo.reduce(%b9ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ddnsdx = stablehlo.broadcast_in_dim %b9ddnsdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9ddnxd = stablehlo.multiply %b9dnxh, %b9ddndxh : tensor<32x480x14x14xf32>
    %b9ddnsxdr = stablehlo.reduce(%b9ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ddnsxd = stablehlo.broadcast_in_dim %b9ddnsxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9ddnt1 = stablehlo.multiply %b9ddndxh, %b9dnnf : tensor<32x480x14x14xf32>
    %b9ddni1 = stablehlo.subtract %b9ddnt1, %b9ddnsdx : tensor<32x480x14x14xf32>
    %b9ddnxs = stablehlo.multiply %b9dnxh, %b9ddnsxd : tensor<32x480x14x14xf32>
    %b9ddni2 = stablehlo.subtract %b9ddni1, %b9ddnxs : tensor<32x480x14x14xf32>
    %b9ddnsN = stablehlo.divide %b9dnistd, %b9dnnf : tensor<32x480x14x14xf32>
    %b9ddn = stablehlo.multiply %b9ddnsN, %b9ddni2 : tensor<32x480x14x14xf32>
    %b9ddndgp = stablehlo.multiply %b9ddr, %b9dnxh : tensor<32x480x14x14xf32>
    %b9ddndg = stablehlo.reduce(%b9ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ddndb = stablehlo.reduce(%b9ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ddrev = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<480x1x5x5xf32>
    %b9dd = stablehlo.convolution(%b9ddn, %b9ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %b9ddWxt = stablehlo.transpose %b9es, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b9ddWdt = stablehlo.transpose %b9ddn, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b9ddWraw = stablehlo.convolution(%b9ddWxt, %b9ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x5x5xf32>
    %b9ddW = stablehlo.reshape %b9ddWraw : (tensor<1x480x5x5xf32>) -> tensor<480x1x5x5xf32>
    %b9ddb = stablehlo.reduce(%b9ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9ders = stablehlo.logistic %b9en : tensor<32x480x14x14xf32>
    %b9derone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b9derom = stablehlo.subtract %b9derone, %b9ders : tensor<32x480x14x14xf32>
    %b9derxom = stablehlo.multiply %b9en, %b9derom : tensor<32x480x14x14xf32>
    %b9derin = stablehlo.add %b9derone, %b9derxom : tensor<32x480x14x14xf32>
    %b9dersp = stablehlo.multiply %b9ders, %b9derin : tensor<32x480x14x14xf32>
    %b9der = stablehlo.multiply %b9dd, %b9dersp : tensor<32x480x14x14xf32>
    %b9dendxh = stablehlo.multiply %b9engb, %b9der : tensor<32x480x14x14xf32>
    %b9densdxr = stablehlo.reduce(%b9dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9densdx = stablehlo.broadcast_in_dim %b9densdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9denxd = stablehlo.multiply %b9enxh, %b9dendxh : tensor<32x480x14x14xf32>
    %b9densxdr = stablehlo.reduce(%b9denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9densxd = stablehlo.broadcast_in_dim %b9densxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b9dent1 = stablehlo.multiply %b9dendxh, %b9ennf : tensor<32x480x14x14xf32>
    %b9deni1 = stablehlo.subtract %b9dent1, %b9densdx : tensor<32x480x14x14xf32>
    %b9denxs = stablehlo.multiply %b9enxh, %b9densxd : tensor<32x480x14x14xf32>
    %b9deni2 = stablehlo.subtract %b9deni1, %b9denxs : tensor<32x480x14x14xf32>
    %b9densN = stablehlo.divide %b9enistd, %b9ennf : tensor<32x480x14x14xf32>
    %b9den = stablehlo.multiply %b9densN, %b9deni2 : tensor<32x480x14x14xf32>
    %b9dendgp = stablehlo.multiply %b9der, %b9enxh : tensor<32x480x14x14xf32>
    %b9dendg = stablehlo.reduce(%b9dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9dendb = stablehlo.reduce(%b9der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b9det = stablehlo.transpose %b9eW, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b9de = stablehlo.convolution(%b9den, %b9det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b9deWxt = stablehlo.transpose %b8o, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b9deWdt = stablehlo.transpose %b9den, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b9deWraw = stablehlo.convolution(%b9deWxt, %b9deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %b9deW = stablehlo.transpose %b9deWraw, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b9deb = stablehlo.reduce(%b9den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dpndxh = stablehlo.multiply %b8pngb, %b9de : tensor<32x80x14x14xf32>
    %b8dpnsdxr = stablehlo.reduce(%b8dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8dpnsdx = stablehlo.broadcast_in_dim %b8dpnsdxr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8dpnxd = stablehlo.multiply %b8pnxh, %b8dpndxh : tensor<32x80x14x14xf32>
    %b8dpnsxdr = stablehlo.reduce(%b8dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8dpnsxd = stablehlo.broadcast_in_dim %b8dpnsxdr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b8dpnt1 = stablehlo.multiply %b8dpndxh, %b8pnnf : tensor<32x80x14x14xf32>
    %b8dpni1 = stablehlo.subtract %b8dpnt1, %b8dpnsdx : tensor<32x80x14x14xf32>
    %b8dpnxs = stablehlo.multiply %b8pnxh, %b8dpnsxd : tensor<32x80x14x14xf32>
    %b8dpni2 = stablehlo.subtract %b8dpni1, %b8dpnxs : tensor<32x80x14x14xf32>
    %b8dpnsN = stablehlo.divide %b8pnistd, %b8pnnf : tensor<32x80x14x14xf32>
    %b8dpn = stablehlo.multiply %b8dpnsN, %b8dpni2 : tensor<32x80x14x14xf32>
    %b8dpndgp = stablehlo.multiply %b9de, %b8pnxh : tensor<32x80x14x14xf32>
    %b8dpndg = stablehlo.reduce(%b8dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8dpndb = stablehlo.reduce(%b9de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8dpt = stablehlo.transpose %b8pW, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b8dp = stablehlo.convolution(%b8dpn, %b8dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b8dpWxt = stablehlo.transpose %b8zse, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b8dpWdt = stablehlo.transpose %b8dpn, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b8dpWraw = stablehlo.convolution(%b8dpWxt, %b8dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %b8dpW = stablehlo.transpose %b8dpWraw, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b8dpb = stablehlo.reduce(%b8dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b8zgb2 = stablehlo.broadcast_in_dim %b8zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b8zdleft = stablehlo.multiply %b8zgb2, %b8dp : tensor<32x480x14x14xf32>
    %b8zxdse = stablehlo.multiply %b8ds, %b8dp : tensor<32x480x14x14xf32>
    %b8zdgate = stablehlo.reduce(%b8zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b8zone = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %b8zomg = stablehlo.subtract %b8zone, %b8zgate : tensor<32x480xf32>
    %b8zsg = stablehlo.multiply %b8zgate, %b8zomg : tensor<32x480xf32>
    %b8zdh2 = stablehlo.multiply %b8zdgate, %b8zsg : tensor<32x480xf32>
    %b8zda1 = stablehlo.dot_general %b8zdh2, %b8zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %b8zdWs2 = stablehlo.dot_general %b8za1, %b8zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %b8zdbs2 = stablehlo.reduce(%b8zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %b8zdexs = stablehlo.logistic %b8zex : tensor<32x20xf32>
    %b8zdexone = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %b8zdexom = stablehlo.subtract %b8zdexone, %b8zdexs : tensor<32x20xf32>
    %b8zdexxom = stablehlo.multiply %b8zex, %b8zdexom : tensor<32x20xf32>
    %b8zdexin = stablehlo.add %b8zdexone, %b8zdexxom : tensor<32x20xf32>
    %b8zdexsp = stablehlo.multiply %b8zdexs, %b8zdexin : tensor<32x20xf32>
    %b8zdex = stablehlo.multiply %b8zda1, %b8zdexsp : tensor<32x20xf32>
    %b8zdsq = stablehlo.dot_general %b8zdex, %b8zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %b8zdWs1 = stablehlo.dot_general %b8zsq, %b8zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %b8zdbs1 = stablehlo.reduce(%b8zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %b8zdsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b8zdsqd = stablehlo.divide %b8zdsq, %b8zdsqnf : tensor<32x480xf32>
    %b8zdgsp = stablehlo.broadcast_in_dim %b8zdsqd, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b8zdds = stablehlo.add %b8zdleft, %b8zdgsp : tensor<32x480x14x14xf32>
    %b8ddrs = stablehlo.logistic %b8dn : tensor<32x480x14x14xf32>
    %b8ddrone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b8ddrom = stablehlo.subtract %b8ddrone, %b8ddrs : tensor<32x480x14x14xf32>
    %b8ddrxom = stablehlo.multiply %b8dn, %b8ddrom : tensor<32x480x14x14xf32>
    %b8ddrin = stablehlo.add %b8ddrone, %b8ddrxom : tensor<32x480x14x14xf32>
    %b8ddrsp = stablehlo.multiply %b8ddrs, %b8ddrin : tensor<32x480x14x14xf32>
    %b8ddr = stablehlo.multiply %b8zdds, %b8ddrsp : tensor<32x480x14x14xf32>
    %b8ddndxh = stablehlo.multiply %b8dngb, %b8ddr : tensor<32x480x14x14xf32>
    %b8ddnsdxr = stablehlo.reduce(%b8ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ddnsdx = stablehlo.broadcast_in_dim %b8ddnsdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8ddnxd = stablehlo.multiply %b8dnxh, %b8ddndxh : tensor<32x480x14x14xf32>
    %b8ddnsxdr = stablehlo.reduce(%b8ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ddnsxd = stablehlo.broadcast_in_dim %b8ddnsxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8ddnt1 = stablehlo.multiply %b8ddndxh, %b8dnnf : tensor<32x480x14x14xf32>
    %b8ddni1 = stablehlo.subtract %b8ddnt1, %b8ddnsdx : tensor<32x480x14x14xf32>
    %b8ddnxs = stablehlo.multiply %b8dnxh, %b8ddnsxd : tensor<32x480x14x14xf32>
    %b8ddni2 = stablehlo.subtract %b8ddni1, %b8ddnxs : tensor<32x480x14x14xf32>
    %b8ddnsN = stablehlo.divide %b8dnistd, %b8dnnf : tensor<32x480x14x14xf32>
    %b8ddn = stablehlo.multiply %b8ddnsN, %b8ddni2 : tensor<32x480x14x14xf32>
    %b8ddndgp = stablehlo.multiply %b8ddr, %b8dnxh : tensor<32x480x14x14xf32>
    %b8ddndg = stablehlo.reduce(%b8ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ddndb = stablehlo.reduce(%b8ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ddrev = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %b8dd = stablehlo.convolution(%b8ddn, %b8ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %b8ddWxt = stablehlo.transpose %b8es, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b8ddWdt = stablehlo.transpose %b8ddn, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b8ddWraw = stablehlo.convolution(%b8ddWxt, %b8ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %b8ddW = stablehlo.reshape %b8ddWraw : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %b8ddb = stablehlo.reduce(%b8ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8ders = stablehlo.logistic %b8en : tensor<32x480x14x14xf32>
    %b8derone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b8derom = stablehlo.subtract %b8derone, %b8ders : tensor<32x480x14x14xf32>
    %b8derxom = stablehlo.multiply %b8en, %b8derom : tensor<32x480x14x14xf32>
    %b8derin = stablehlo.add %b8derone, %b8derxom : tensor<32x480x14x14xf32>
    %b8dersp = stablehlo.multiply %b8ders, %b8derin : tensor<32x480x14x14xf32>
    %b8der = stablehlo.multiply %b8dd, %b8dersp : tensor<32x480x14x14xf32>
    %b8dendxh = stablehlo.multiply %b8engb, %b8der : tensor<32x480x14x14xf32>
    %b8densdxr = stablehlo.reduce(%b8dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8densdx = stablehlo.broadcast_in_dim %b8densdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8denxd = stablehlo.multiply %b8enxh, %b8dendxh : tensor<32x480x14x14xf32>
    %b8densxdr = stablehlo.reduce(%b8denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8densxd = stablehlo.broadcast_in_dim %b8densxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b8dent1 = stablehlo.multiply %b8dendxh, %b8ennf : tensor<32x480x14x14xf32>
    %b8deni1 = stablehlo.subtract %b8dent1, %b8densdx : tensor<32x480x14x14xf32>
    %b8denxs = stablehlo.multiply %b8enxh, %b8densxd : tensor<32x480x14x14xf32>
    %b8deni2 = stablehlo.subtract %b8deni1, %b8denxs : tensor<32x480x14x14xf32>
    %b8densN = stablehlo.divide %b8enistd, %b8ennf : tensor<32x480x14x14xf32>
    %b8den = stablehlo.multiply %b8densN, %b8deni2 : tensor<32x480x14x14xf32>
    %b8dendgp = stablehlo.multiply %b8der, %b8enxh : tensor<32x480x14x14xf32>
    %b8dendg = stablehlo.reduce(%b8dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dendb = stablehlo.reduce(%b8der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8det = stablehlo.transpose %b8eW, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b8de = stablehlo.convolution(%b8den, %b8det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b8deWxt = stablehlo.transpose %b7o, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b8deWdt = stablehlo.transpose %b8den, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b8deWraw = stablehlo.convolution(%b8deWxt, %b8deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %b8deW = stablehlo.transpose %b8deWraw, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b8deb = stablehlo.reduce(%b8den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b8dx = stablehlo.add %b8de, %b9de : tensor<32x80x14x14xf32>
    %b7dpndxh = stablehlo.multiply %b7pngb, %b8dx : tensor<32x80x14x14xf32>
    %b7dpnsdxr = stablehlo.reduce(%b7dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7dpnsdx = stablehlo.broadcast_in_dim %b7dpnsdxr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7dpnxd = stablehlo.multiply %b7pnxh, %b7dpndxh : tensor<32x80x14x14xf32>
    %b7dpnsxdr = stablehlo.reduce(%b7dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7dpnsxd = stablehlo.broadcast_in_dim %b7dpnsxdr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b7dpnt1 = stablehlo.multiply %b7dpndxh, %b7pnnf : tensor<32x80x14x14xf32>
    %b7dpni1 = stablehlo.subtract %b7dpnt1, %b7dpnsdx : tensor<32x80x14x14xf32>
    %b7dpnxs = stablehlo.multiply %b7pnxh, %b7dpnsxd : tensor<32x80x14x14xf32>
    %b7dpni2 = stablehlo.subtract %b7dpni1, %b7dpnxs : tensor<32x80x14x14xf32>
    %b7dpnsN = stablehlo.divide %b7pnistd, %b7pnnf : tensor<32x80x14x14xf32>
    %b7dpn = stablehlo.multiply %b7dpnsN, %b7dpni2 : tensor<32x80x14x14xf32>
    %b7dpndgp = stablehlo.multiply %b8dx, %b7pnxh : tensor<32x80x14x14xf32>
    %b7dpndg = stablehlo.reduce(%b7dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7dpndb = stablehlo.reduce(%b8dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7dpt = stablehlo.transpose %b7pW, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b7dp = stablehlo.convolution(%b7dpn, %b7dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %b7dpWxt = stablehlo.transpose %b7zse, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b7dpWdt = stablehlo.transpose %b7dpn, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b7dpWraw = stablehlo.convolution(%b7dpWxt, %b7dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %b7dpW = stablehlo.transpose %b7dpWraw, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b7dpb = stablehlo.reduce(%b7dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b7zgb2 = stablehlo.broadcast_in_dim %b7zgate, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b7zdleft = stablehlo.multiply %b7zgb2, %b7dp : tensor<32x480x14x14xf32>
    %b7zxdse = stablehlo.multiply %b7ds, %b7dp : tensor<32x480x14x14xf32>
    %b7zdgate = stablehlo.reduce(%b7zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %b7zone = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %b7zomg = stablehlo.subtract %b7zone, %b7zgate : tensor<32x480xf32>
    %b7zsg = stablehlo.multiply %b7zgate, %b7zomg : tensor<32x480xf32>
    %b7zdh2 = stablehlo.multiply %b7zdgate, %b7zsg : tensor<32x480xf32>
    %b7zda1 = stablehlo.dot_general %b7zdh2, %b7zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %b7zdWs2 = stablehlo.dot_general %b7za1, %b7zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %b7zdbs2 = stablehlo.reduce(%b7zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %b7zdexs = stablehlo.logistic %b7zex : tensor<32x20xf32>
    %b7zdexone = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %b7zdexom = stablehlo.subtract %b7zdexone, %b7zdexs : tensor<32x20xf32>
    %b7zdexxom = stablehlo.multiply %b7zex, %b7zdexom : tensor<32x20xf32>
    %b7zdexin = stablehlo.add %b7zdexone, %b7zdexxom : tensor<32x20xf32>
    %b7zdexsp = stablehlo.multiply %b7zdexs, %b7zdexin : tensor<32x20xf32>
    %b7zdex = stablehlo.multiply %b7zda1, %b7zdexsp : tensor<32x20xf32>
    %b7zdsq = stablehlo.dot_general %b7zdex, %b7zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %b7zdWs1 = stablehlo.dot_general %b7zsq, %b7zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %b7zdbs1 = stablehlo.reduce(%b7zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %b7zdsqnf = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %b7zdsqd = stablehlo.divide %b7zdsq, %b7zdsqnf : tensor<32x480xf32>
    %b7zdgsp = stablehlo.broadcast_in_dim %b7zdsqd, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %b7zdds = stablehlo.add %b7zdleft, %b7zdgsp : tensor<32x480x14x14xf32>
    %b7ddrs = stablehlo.logistic %b7dn : tensor<32x480x14x14xf32>
    %b7ddrone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b7ddrom = stablehlo.subtract %b7ddrone, %b7ddrs : tensor<32x480x14x14xf32>
    %b7ddrxom = stablehlo.multiply %b7dn, %b7ddrom : tensor<32x480x14x14xf32>
    %b7ddrin = stablehlo.add %b7ddrone, %b7ddrxom : tensor<32x480x14x14xf32>
    %b7ddrsp = stablehlo.multiply %b7ddrs, %b7ddrin : tensor<32x480x14x14xf32>
    %b7ddr = stablehlo.multiply %b7zdds, %b7ddrsp : tensor<32x480x14x14xf32>
    %b7ddndxh = stablehlo.multiply %b7dngb, %b7ddr : tensor<32x480x14x14xf32>
    %b7ddnsdxr = stablehlo.reduce(%b7ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ddnsdx = stablehlo.broadcast_in_dim %b7ddnsdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7ddnxd = stablehlo.multiply %b7dnxh, %b7ddndxh : tensor<32x480x14x14xf32>
    %b7ddnsxdr = stablehlo.reduce(%b7ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ddnsxd = stablehlo.broadcast_in_dim %b7ddnsxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7ddnt1 = stablehlo.multiply %b7ddndxh, %b7dnnf : tensor<32x480x14x14xf32>
    %b7ddni1 = stablehlo.subtract %b7ddnt1, %b7ddnsdx : tensor<32x480x14x14xf32>
    %b7ddnxs = stablehlo.multiply %b7dnxh, %b7ddnsxd : tensor<32x480x14x14xf32>
    %b7ddni2 = stablehlo.subtract %b7ddni1, %b7ddnxs : tensor<32x480x14x14xf32>
    %b7ddnsN = stablehlo.divide %b7dnistd, %b7dnnf : tensor<32x480x14x14xf32>
    %b7ddn = stablehlo.multiply %b7ddnsN, %b7ddni2 : tensor<32x480x14x14xf32>
    %b7ddndgp = stablehlo.multiply %b7ddr, %b7dnxh : tensor<32x480x14x14xf32>
    %b7ddndg = stablehlo.reduce(%b7ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ddndb = stablehlo.reduce(%b7ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ddrev = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %b7dd = stablehlo.convolution(%b7ddn, %b7ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %b7ddWxt = stablehlo.transpose %b7es, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b7ddWdt = stablehlo.transpose %b7ddn, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b7ddWraw = stablehlo.convolution(%b7ddWxt, %b7ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %b7ddW = stablehlo.reshape %b7ddWraw : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %b7ddb = stablehlo.reduce(%b7ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7ders = stablehlo.logistic %b7en : tensor<32x480x14x14xf32>
    %b7derone = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %b7derom = stablehlo.subtract %b7derone, %b7ders : tensor<32x480x14x14xf32>
    %b7derxom = stablehlo.multiply %b7en, %b7derom : tensor<32x480x14x14xf32>
    %b7derin = stablehlo.add %b7derone, %b7derxom : tensor<32x480x14x14xf32>
    %b7dersp = stablehlo.multiply %b7ders, %b7derin : tensor<32x480x14x14xf32>
    %b7der = stablehlo.multiply %b7dd, %b7dersp : tensor<32x480x14x14xf32>
    %b7dendxh = stablehlo.multiply %b7engb, %b7der : tensor<32x480x14x14xf32>
    %b7densdxr = stablehlo.reduce(%b7dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7densdx = stablehlo.broadcast_in_dim %b7densdxr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7denxd = stablehlo.multiply %b7enxh, %b7dendxh : tensor<32x480x14x14xf32>
    %b7densxdr = stablehlo.reduce(%b7denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7densxd = stablehlo.broadcast_in_dim %b7densxdr, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %b7dent1 = stablehlo.multiply %b7dendxh, %b7ennf : tensor<32x480x14x14xf32>
    %b7deni1 = stablehlo.subtract %b7dent1, %b7densdx : tensor<32x480x14x14xf32>
    %b7denxs = stablehlo.multiply %b7enxh, %b7densxd : tensor<32x480x14x14xf32>
    %b7deni2 = stablehlo.subtract %b7deni1, %b7denxs : tensor<32x480x14x14xf32>
    %b7densN = stablehlo.divide %b7enistd, %b7ennf : tensor<32x480x14x14xf32>
    %b7den = stablehlo.multiply %b7densN, %b7deni2 : tensor<32x480x14x14xf32>
    %b7dendgp = stablehlo.multiply %b7der, %b7enxh : tensor<32x480x14x14xf32>
    %b7dendg = stablehlo.reduce(%b7dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7dendb = stablehlo.reduce(%b7der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7det = stablehlo.transpose %b7eW, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %b7de = stablehlo.convolution(%b7den, %b7det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %b7deWxt = stablehlo.transpose %b6pn, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b7deWdt = stablehlo.transpose %b7den, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %b7deWraw = stablehlo.convolution(%b7deWxt, %b7deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %b7deW = stablehlo.transpose %b7deWraw, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %b7deb = stablehlo.reduce(%b7den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %b7dx = stablehlo.add %b7de, %b8dx : tensor<32x80x14x14xf32>
    %b6dpndxh = stablehlo.multiply %b6pngb, %b7dx : tensor<32x80x14x14xf32>
    %b6dpnsdxr = stablehlo.reduce(%b6dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6dpnsdx = stablehlo.broadcast_in_dim %b6dpnsdxr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6dpnxd = stablehlo.multiply %b6pnxh, %b6dpndxh : tensor<32x80x14x14xf32>
    %b6dpnsxdr = stablehlo.reduce(%b6dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6dpnsxd = stablehlo.broadcast_in_dim %b6dpnsxdr, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %b6dpnt1 = stablehlo.multiply %b6dpndxh, %b6pnnf : tensor<32x80x14x14xf32>
    %b6dpni1 = stablehlo.subtract %b6dpnt1, %b6dpnsdx : tensor<32x80x14x14xf32>
    %b6dpnxs = stablehlo.multiply %b6pnxh, %b6dpnsxd : tensor<32x80x14x14xf32>
    %b6dpni2 = stablehlo.subtract %b6dpni1, %b6dpnxs : tensor<32x80x14x14xf32>
    %b6dpnsN = stablehlo.divide %b6pnistd, %b6pnnf : tensor<32x80x14x14xf32>
    %b6dpn = stablehlo.multiply %b6dpnsN, %b6dpni2 : tensor<32x80x14x14xf32>
    %b6dpndgp = stablehlo.multiply %b7dx, %b6pnxh : tensor<32x80x14x14xf32>
    %b6dpndg = stablehlo.reduce(%b6dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6dpndb = stablehlo.reduce(%b7dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6dpt = stablehlo.transpose %b6pW, dims = [1, 0, 2, 3] : (tensor<80x240x1x1xf32>) -> tensor<240x80x1x1xf32>
    %b6dp = stablehlo.convolution(%b6dpn, %b6dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<240x80x1x1xf32>) -> tensor<32x240x14x14xf32>
    %b6dpWxt = stablehlo.transpose %b6zse, dims = [1, 0, 2, 3] : (tensor<32x240x14x14xf32>) -> tensor<240x32x14x14xf32>
    %b6dpWdt = stablehlo.transpose %b6dpn, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %b6dpWraw = stablehlo.convolution(%b6dpWxt, %b6dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<240x80x1x1xf32>
    %b6dpW = stablehlo.transpose %b6dpWraw, dims = [1, 0, 2, 3] : (tensor<240x80x1x1xf32>) -> tensor<80x240x1x1xf32>
    %b6dpb = stablehlo.reduce(%b6dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %b6zgb2 = stablehlo.broadcast_in_dim %b6zgate, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %b6zdleft = stablehlo.multiply %b6zgb2, %b6dp : tensor<32x240x14x14xf32>
    %b6zxdse = stablehlo.multiply %b6ds, %b6dp : tensor<32x240x14x14xf32>
    %b6zdgate = stablehlo.reduce(%b6zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %b6zone = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %b6zomg = stablehlo.subtract %b6zone, %b6zgate : tensor<32x240xf32>
    %b6zsg = stablehlo.multiply %b6zgate, %b6zomg : tensor<32x240xf32>
    %b6zdh2 = stablehlo.multiply %b6zdgate, %b6zsg : tensor<32x240xf32>
    %b6zda1 = stablehlo.dot_general %b6zdh2, %b6zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %b6zdWs2 = stablehlo.dot_general %b6za1, %b6zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %b6zdbs2 = stablehlo.reduce(%b6zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %b6zdexs = stablehlo.logistic %b6zex : tensor<32x10xf32>
    %b6zdexone = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %b6zdexom = stablehlo.subtract %b6zdexone, %b6zdexs : tensor<32x10xf32>
    %b6zdexxom = stablehlo.multiply %b6zex, %b6zdexom : tensor<32x10xf32>
    %b6zdexin = stablehlo.add %b6zdexone, %b6zdexxom : tensor<32x10xf32>
    %b6zdexsp = stablehlo.multiply %b6zdexs, %b6zdexin : tensor<32x10xf32>
    %b6zdex = stablehlo.multiply %b6zda1, %b6zdexsp : tensor<32x10xf32>
    %b6zdsq = stablehlo.dot_general %b6zdex, %b6zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %b6zdWs1 = stablehlo.dot_general %b6zsq, %b6zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %b6zdbs1 = stablehlo.reduce(%b6zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %b6zdsqnf = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %b6zdsqd = stablehlo.divide %b6zdsq, %b6zdsqnf : tensor<32x240xf32>
    %b6zdgsp = stablehlo.broadcast_in_dim %b6zdsqd, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %b6zdds = stablehlo.add %b6zdleft, %b6zdgsp : tensor<32x240x14x14xf32>
    %b6ddrs = stablehlo.logistic %b6dn : tensor<32x240x14x14xf32>
    %b6ddrone = stablehlo.constant dense<1.0> : tensor<32x240x14x14xf32>
    %b6ddrom = stablehlo.subtract %b6ddrone, %b6ddrs : tensor<32x240x14x14xf32>
    %b6ddrxom = stablehlo.multiply %b6dn, %b6ddrom : tensor<32x240x14x14xf32>
    %b6ddrin = stablehlo.add %b6ddrone, %b6ddrxom : tensor<32x240x14x14xf32>
    %b6ddrsp = stablehlo.multiply %b6ddrs, %b6ddrin : tensor<32x240x14x14xf32>
    %b6ddr = stablehlo.multiply %b6zdds, %b6ddrsp : tensor<32x240x14x14xf32>
    %b6ddndxh = stablehlo.multiply %b6dngb, %b6ddr : tensor<32x240x14x14xf32>
    %b6ddnsdxr = stablehlo.reduce(%b6ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ddnsdx = stablehlo.broadcast_in_dim %b6ddnsdxr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6ddnxd = stablehlo.multiply %b6dnxh, %b6ddndxh : tensor<32x240x14x14xf32>
    %b6ddnsxdr = stablehlo.reduce(%b6ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ddnsxd = stablehlo.broadcast_in_dim %b6ddnsxdr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %b6ddnt1 = stablehlo.multiply %b6ddndxh, %b6dnnf : tensor<32x240x14x14xf32>
    %b6ddni1 = stablehlo.subtract %b6ddnt1, %b6ddnsdx : tensor<32x240x14x14xf32>
    %b6ddnxs = stablehlo.multiply %b6dnxh, %b6ddnsxd : tensor<32x240x14x14xf32>
    %b6ddni2 = stablehlo.subtract %b6ddni1, %b6ddnxs : tensor<32x240x14x14xf32>
    %b6ddnsN = stablehlo.divide %b6dnistd, %b6dnnf : tensor<32x240x14x14xf32>
    %b6ddn = stablehlo.multiply %b6ddnsN, %b6ddni2 : tensor<32x240x14x14xf32>
    %b6ddndgp = stablehlo.multiply %b6ddr, %b6dnxh : tensor<32x240x14x14xf32>
    %b6ddndg = stablehlo.reduce(%b6ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ddndb = stablehlo.reduce(%b6ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ddu = stablehlo.pad %b6ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %b6ddrev = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<240x1x3x3xf32>
    %b6dd = stablehlo.convolution(%b6ddu, %b6ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x28x28xf32>
    %b6ddWu = stablehlo.pad %b6ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %b6ddWxt = stablehlo.transpose %b6es, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b6ddWdt = stablehlo.transpose %b6ddWu, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b6ddWraw = stablehlo.convolution(%b6ddWxt, %b6ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x3x3xf32>
    %b6ddW = stablehlo.reshape %b6ddWraw : (tensor<1x240x3x3xf32>) -> tensor<240x1x3x3xf32>
    %b6ddb = stablehlo.reduce(%b6ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %b6ders = stablehlo.logistic %b6en : tensor<32x240x28x28xf32>
    %b6derone = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %b6derom = stablehlo.subtract %b6derone, %b6ders : tensor<32x240x28x28xf32>
    %b6derxom = stablehlo.multiply %b6en, %b6derom : tensor<32x240x28x28xf32>
    %b6derin = stablehlo.add %b6derone, %b6derxom : tensor<32x240x28x28xf32>
    %b6dersp = stablehlo.multiply %b6ders, %b6derin : tensor<32x240x28x28xf32>
    %b6der = stablehlo.multiply %b6dd, %b6dersp : tensor<32x240x28x28xf32>
    %b6dendxh = stablehlo.multiply %b6engb, %b6der : tensor<32x240x28x28xf32>
    %b6densdxr = stablehlo.reduce(%b6dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6densdx = stablehlo.broadcast_in_dim %b6densdxr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6denxd = stablehlo.multiply %b6enxh, %b6dendxh : tensor<32x240x28x28xf32>
    %b6densxdr = stablehlo.reduce(%b6denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6densxd = stablehlo.broadcast_in_dim %b6densxdr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b6dent1 = stablehlo.multiply %b6dendxh, %b6ennf : tensor<32x240x28x28xf32>
    %b6deni1 = stablehlo.subtract %b6dent1, %b6densdx : tensor<32x240x28x28xf32>
    %b6denxs = stablehlo.multiply %b6enxh, %b6densxd : tensor<32x240x28x28xf32>
    %b6deni2 = stablehlo.subtract %b6deni1, %b6denxs : tensor<32x240x28x28xf32>
    %b6densN = stablehlo.divide %b6enistd, %b6ennf : tensor<32x240x28x28xf32>
    %b6den = stablehlo.multiply %b6densN, %b6deni2 : tensor<32x240x28x28xf32>
    %b6dendgp = stablehlo.multiply %b6der, %b6enxh : tensor<32x240x28x28xf32>
    %b6dendg = stablehlo.reduce(%b6dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6dendb = stablehlo.reduce(%b6der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b6det = stablehlo.transpose %b6eW, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %b6de = stablehlo.convolution(%b6den, %b6det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %b6deWxt = stablehlo.transpose %b5o, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %b6deWdt = stablehlo.transpose %b6den, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b6deWraw = stablehlo.convolution(%b6deWxt, %b6deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %b6deW = stablehlo.transpose %b6deWraw, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %b6deb = stablehlo.reduce(%b6den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dpndxh = stablehlo.multiply %b5pngb, %b6de : tensor<32x40x28x28xf32>
    %b5dpnsdxr = stablehlo.reduce(%b5dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5dpnsdx = stablehlo.broadcast_in_dim %b5dpnsdxr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5dpnxd = stablehlo.multiply %b5pnxh, %b5dpndxh : tensor<32x40x28x28xf32>
    %b5dpnsxdr = stablehlo.reduce(%b5dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5dpnsxd = stablehlo.broadcast_in_dim %b5dpnsxdr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b5dpnt1 = stablehlo.multiply %b5dpndxh, %b5pnnf : tensor<32x40x28x28xf32>
    %b5dpni1 = stablehlo.subtract %b5dpnt1, %b5dpnsdx : tensor<32x40x28x28xf32>
    %b5dpnxs = stablehlo.multiply %b5pnxh, %b5dpnsxd : tensor<32x40x28x28xf32>
    %b5dpni2 = stablehlo.subtract %b5dpni1, %b5dpnxs : tensor<32x40x28x28xf32>
    %b5dpnsN = stablehlo.divide %b5pnistd, %b5pnnf : tensor<32x40x28x28xf32>
    %b5dpn = stablehlo.multiply %b5dpnsN, %b5dpni2 : tensor<32x40x28x28xf32>
    %b5dpndgp = stablehlo.multiply %b6de, %b5pnxh : tensor<32x40x28x28xf32>
    %b5dpndg = stablehlo.reduce(%b5dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5dpndb = stablehlo.reduce(%b6de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5dpt = stablehlo.transpose %b5pW, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %b5dp = stablehlo.convolution(%b5dpn, %b5dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %b5dpWxt = stablehlo.transpose %b5zse, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b5dpWdt = stablehlo.transpose %b5dpn, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %b5dpWraw = stablehlo.convolution(%b5dpWxt, %b5dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<240x40x1x1xf32>
    %b5dpW = stablehlo.transpose %b5dpWraw, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %b5dpb = stablehlo.reduce(%b5dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b5zgb2 = stablehlo.broadcast_in_dim %b5zgate, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %b5zdleft = stablehlo.multiply %b5zgb2, %b5dp : tensor<32x240x28x28xf32>
    %b5zxdse = stablehlo.multiply %b5ds, %b5dp : tensor<32x240x28x28xf32>
    %b5zdgate = stablehlo.reduce(%b5zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %b5zone = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %b5zomg = stablehlo.subtract %b5zone, %b5zgate : tensor<32x240xf32>
    %b5zsg = stablehlo.multiply %b5zgate, %b5zomg : tensor<32x240xf32>
    %b5zdh2 = stablehlo.multiply %b5zdgate, %b5zsg : tensor<32x240xf32>
    %b5zda1 = stablehlo.dot_general %b5zdh2, %b5zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %b5zdWs2 = stablehlo.dot_general %b5za1, %b5zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %b5zdbs2 = stablehlo.reduce(%b5zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %b5zdexs = stablehlo.logistic %b5zex : tensor<32x10xf32>
    %b5zdexone = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %b5zdexom = stablehlo.subtract %b5zdexone, %b5zdexs : tensor<32x10xf32>
    %b5zdexxom = stablehlo.multiply %b5zex, %b5zdexom : tensor<32x10xf32>
    %b5zdexin = stablehlo.add %b5zdexone, %b5zdexxom : tensor<32x10xf32>
    %b5zdexsp = stablehlo.multiply %b5zdexs, %b5zdexin : tensor<32x10xf32>
    %b5zdex = stablehlo.multiply %b5zda1, %b5zdexsp : tensor<32x10xf32>
    %b5zdsq = stablehlo.dot_general %b5zdex, %b5zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %b5zdWs1 = stablehlo.dot_general %b5zsq, %b5zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %b5zdbs1 = stablehlo.reduce(%b5zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %b5zdsqnf = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %b5zdsqd = stablehlo.divide %b5zdsq, %b5zdsqnf : tensor<32x240xf32>
    %b5zdgsp = stablehlo.broadcast_in_dim %b5zdsqd, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %b5zdds = stablehlo.add %b5zdleft, %b5zdgsp : tensor<32x240x28x28xf32>
    %b5ddrs = stablehlo.logistic %b5dn : tensor<32x240x28x28xf32>
    %b5ddrone = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %b5ddrom = stablehlo.subtract %b5ddrone, %b5ddrs : tensor<32x240x28x28xf32>
    %b5ddrxom = stablehlo.multiply %b5dn, %b5ddrom : tensor<32x240x28x28xf32>
    %b5ddrin = stablehlo.add %b5ddrone, %b5ddrxom : tensor<32x240x28x28xf32>
    %b5ddrsp = stablehlo.multiply %b5ddrs, %b5ddrin : tensor<32x240x28x28xf32>
    %b5ddr = stablehlo.multiply %b5zdds, %b5ddrsp : tensor<32x240x28x28xf32>
    %b5ddndxh = stablehlo.multiply %b5dngb, %b5ddr : tensor<32x240x28x28xf32>
    %b5ddnsdxr = stablehlo.reduce(%b5ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ddnsdx = stablehlo.broadcast_in_dim %b5ddnsdxr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5ddnxd = stablehlo.multiply %b5dnxh, %b5ddndxh : tensor<32x240x28x28xf32>
    %b5ddnsxdr = stablehlo.reduce(%b5ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ddnsxd = stablehlo.broadcast_in_dim %b5ddnsxdr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5ddnt1 = stablehlo.multiply %b5ddndxh, %b5dnnf : tensor<32x240x28x28xf32>
    %b5ddni1 = stablehlo.subtract %b5ddnt1, %b5ddnsdx : tensor<32x240x28x28xf32>
    %b5ddnxs = stablehlo.multiply %b5dnxh, %b5ddnsxd : tensor<32x240x28x28xf32>
    %b5ddni2 = stablehlo.subtract %b5ddni1, %b5ddnxs : tensor<32x240x28x28xf32>
    %b5ddnsN = stablehlo.divide %b5dnistd, %b5dnnf : tensor<32x240x28x28xf32>
    %b5ddn = stablehlo.multiply %b5ddnsN, %b5ddni2 : tensor<32x240x28x28xf32>
    %b5ddndgp = stablehlo.multiply %b5ddr, %b5dnxh : tensor<32x240x28x28xf32>
    %b5ddndg = stablehlo.reduce(%b5ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ddndb = stablehlo.reduce(%b5ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ddrev = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<240x1x5x5xf32>
    %b5dd = stablehlo.convolution(%b5ddn, %b5ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %b5ddWxt = stablehlo.transpose %b5es, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b5ddWdt = stablehlo.transpose %b5ddn, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b5ddWraw = stablehlo.convolution(%b5ddWxt, %b5ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x5x5xf32>
    %b5ddW = stablehlo.reshape %b5ddWraw : (tensor<1x240x5x5xf32>) -> tensor<240x1x5x5xf32>
    %b5ddb = stablehlo.reduce(%b5ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5ders = stablehlo.logistic %b5en : tensor<32x240x28x28xf32>
    %b5derone = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %b5derom = stablehlo.subtract %b5derone, %b5ders : tensor<32x240x28x28xf32>
    %b5derxom = stablehlo.multiply %b5en, %b5derom : tensor<32x240x28x28xf32>
    %b5derin = stablehlo.add %b5derone, %b5derxom : tensor<32x240x28x28xf32>
    %b5dersp = stablehlo.multiply %b5ders, %b5derin : tensor<32x240x28x28xf32>
    %b5der = stablehlo.multiply %b5dd, %b5dersp : tensor<32x240x28x28xf32>
    %b5dendxh = stablehlo.multiply %b5engb, %b5der : tensor<32x240x28x28xf32>
    %b5densdxr = stablehlo.reduce(%b5dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5densdx = stablehlo.broadcast_in_dim %b5densdxr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5denxd = stablehlo.multiply %b5enxh, %b5dendxh : tensor<32x240x28x28xf32>
    %b5densxdr = stablehlo.reduce(%b5denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5densxd = stablehlo.broadcast_in_dim %b5densxdr, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %b5dent1 = stablehlo.multiply %b5dendxh, %b5ennf : tensor<32x240x28x28xf32>
    %b5deni1 = stablehlo.subtract %b5dent1, %b5densdx : tensor<32x240x28x28xf32>
    %b5denxs = stablehlo.multiply %b5enxh, %b5densxd : tensor<32x240x28x28xf32>
    %b5deni2 = stablehlo.subtract %b5deni1, %b5denxs : tensor<32x240x28x28xf32>
    %b5densN = stablehlo.divide %b5enistd, %b5ennf : tensor<32x240x28x28xf32>
    %b5den = stablehlo.multiply %b5densN, %b5deni2 : tensor<32x240x28x28xf32>
    %b5dendgp = stablehlo.multiply %b5der, %b5enxh : tensor<32x240x28x28xf32>
    %b5dendg = stablehlo.reduce(%b5dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dendb = stablehlo.reduce(%b5der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5det = stablehlo.transpose %b5eW, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %b5de = stablehlo.convolution(%b5den, %b5det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %b5deWxt = stablehlo.transpose %b4pn, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %b5deWdt = stablehlo.transpose %b5den, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %b5deWraw = stablehlo.convolution(%b5deWxt, %b5deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %b5deW = stablehlo.transpose %b5deWraw, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %b5deb = stablehlo.reduce(%b5den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %b5dx = stablehlo.add %b5de, %b6de : tensor<32x40x28x28xf32>
    %b4dpndxh = stablehlo.multiply %b4pngb, %b5dx : tensor<32x40x28x28xf32>
    %b4dpnsdxr = stablehlo.reduce(%b4dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4dpnsdx = stablehlo.broadcast_in_dim %b4dpnsdxr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4dpnxd = stablehlo.multiply %b4pnxh, %b4dpndxh : tensor<32x40x28x28xf32>
    %b4dpnsxdr = stablehlo.reduce(%b4dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4dpnsxd = stablehlo.broadcast_in_dim %b4dpnsxdr, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %b4dpnt1 = stablehlo.multiply %b4dpndxh, %b4pnnf : tensor<32x40x28x28xf32>
    %b4dpni1 = stablehlo.subtract %b4dpnt1, %b4dpnsdx : tensor<32x40x28x28xf32>
    %b4dpnxs = stablehlo.multiply %b4pnxh, %b4dpnsxd : tensor<32x40x28x28xf32>
    %b4dpni2 = stablehlo.subtract %b4dpni1, %b4dpnxs : tensor<32x40x28x28xf32>
    %b4dpnsN = stablehlo.divide %b4pnistd, %b4pnnf : tensor<32x40x28x28xf32>
    %b4dpn = stablehlo.multiply %b4dpnsN, %b4dpni2 : tensor<32x40x28x28xf32>
    %b4dpndgp = stablehlo.multiply %b5dx, %b4pnxh : tensor<32x40x28x28xf32>
    %b4dpndg = stablehlo.reduce(%b4dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4dpndb = stablehlo.reduce(%b5dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4dpt = stablehlo.transpose %b4pW, dims = [1, 0, 2, 3] : (tensor<40x144x1x1xf32>) -> tensor<144x40x1x1xf32>
    %b4dp = stablehlo.convolution(%b4dpn, %b4dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<144x40x1x1xf32>) -> tensor<32x144x28x28xf32>
    %b4dpWxt = stablehlo.transpose %b4zse, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %b4dpWdt = stablehlo.transpose %b4dpn, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %b4dpWraw = stablehlo.convolution(%b4dpWxt, %b4dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<144x40x1x1xf32>
    %b4dpW = stablehlo.transpose %b4dpWraw, dims = [1, 0, 2, 3] : (tensor<144x40x1x1xf32>) -> tensor<40x144x1x1xf32>
    %b4dpb = stablehlo.reduce(%b4dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %b4zgb2 = stablehlo.broadcast_in_dim %b4zgate, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %b4zdleft = stablehlo.multiply %b4zgb2, %b4dp : tensor<32x144x28x28xf32>
    %b4zxdse = stablehlo.multiply %b4ds, %b4dp : tensor<32x144x28x28xf32>
    %b4zdgate = stablehlo.reduce(%b4zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %b4zone = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %b4zomg = stablehlo.subtract %b4zone, %b4zgate : tensor<32x144xf32>
    %b4zsg = stablehlo.multiply %b4zgate, %b4zomg : tensor<32x144xf32>
    %b4zdh2 = stablehlo.multiply %b4zdgate, %b4zsg : tensor<32x144xf32>
    %b4zda1 = stablehlo.dot_general %b4zdh2, %b4zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %b4zdWs2 = stablehlo.dot_general %b4za1, %b4zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %b4zdbs2 = stablehlo.reduce(%b4zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %b4zdexs = stablehlo.logistic %b4zex : tensor<32x6xf32>
    %b4zdexone = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %b4zdexom = stablehlo.subtract %b4zdexone, %b4zdexs : tensor<32x6xf32>
    %b4zdexxom = stablehlo.multiply %b4zex, %b4zdexom : tensor<32x6xf32>
    %b4zdexin = stablehlo.add %b4zdexone, %b4zdexxom : tensor<32x6xf32>
    %b4zdexsp = stablehlo.multiply %b4zdexs, %b4zdexin : tensor<32x6xf32>
    %b4zdex = stablehlo.multiply %b4zda1, %b4zdexsp : tensor<32x6xf32>
    %b4zdsq = stablehlo.dot_general %b4zdex, %b4zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %b4zdWs1 = stablehlo.dot_general %b4zsq, %b4zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %b4zdbs1 = stablehlo.reduce(%b4zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %b4zdsqnf = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %b4zdsqd = stablehlo.divide %b4zdsq, %b4zdsqnf : tensor<32x144xf32>
    %b4zdgsp = stablehlo.broadcast_in_dim %b4zdsqd, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %b4zdds = stablehlo.add %b4zdleft, %b4zdgsp : tensor<32x144x28x28xf32>
    %b4ddrs = stablehlo.logistic %b4dn : tensor<32x144x28x28xf32>
    %b4ddrone = stablehlo.constant dense<1.0> : tensor<32x144x28x28xf32>
    %b4ddrom = stablehlo.subtract %b4ddrone, %b4ddrs : tensor<32x144x28x28xf32>
    %b4ddrxom = stablehlo.multiply %b4dn, %b4ddrom : tensor<32x144x28x28xf32>
    %b4ddrin = stablehlo.add %b4ddrone, %b4ddrxom : tensor<32x144x28x28xf32>
    %b4ddrsp = stablehlo.multiply %b4ddrs, %b4ddrin : tensor<32x144x28x28xf32>
    %b4ddr = stablehlo.multiply %b4zdds, %b4ddrsp : tensor<32x144x28x28xf32>
    %b4ddndxh = stablehlo.multiply %b4dngb, %b4ddr : tensor<32x144x28x28xf32>
    %b4ddnsdxr = stablehlo.reduce(%b4ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ddnsdx = stablehlo.broadcast_in_dim %b4ddnsdxr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4ddnxd = stablehlo.multiply %b4dnxh, %b4ddndxh : tensor<32x144x28x28xf32>
    %b4ddnsxdr = stablehlo.reduce(%b4ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ddnsxd = stablehlo.broadcast_in_dim %b4ddnsxdr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %b4ddnt1 = stablehlo.multiply %b4ddndxh, %b4dnnf : tensor<32x144x28x28xf32>
    %b4ddni1 = stablehlo.subtract %b4ddnt1, %b4ddnsdx : tensor<32x144x28x28xf32>
    %b4ddnxs = stablehlo.multiply %b4dnxh, %b4ddnsxd : tensor<32x144x28x28xf32>
    %b4ddni2 = stablehlo.subtract %b4ddni1, %b4ddnxs : tensor<32x144x28x28xf32>
    %b4ddnsN = stablehlo.divide %b4dnistd, %b4dnnf : tensor<32x144x28x28xf32>
    %b4ddn = stablehlo.multiply %b4ddnsN, %b4ddni2 : tensor<32x144x28x28xf32>
    %b4ddndgp = stablehlo.multiply %b4ddr, %b4dnxh : tensor<32x144x28x28xf32>
    %b4ddndg = stablehlo.reduce(%b4ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ddndb = stablehlo.reduce(%b4ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ddu = stablehlo.pad %b4ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %b4ddrev = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x5x5xf32>
    %b4dd = stablehlo.convolution(%b4ddu, %b4ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x56x56xf32>
    %b4ddWu = stablehlo.pad %b4ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %b4ddWxt = stablehlo.transpose %b4es, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b4ddWdt = stablehlo.transpose %b4ddWu, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b4ddWraw = stablehlo.convolution(%b4ddWxt, %b4ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x5x5xf32>
    %b4ddW = stablehlo.reshape %b4ddWraw : (tensor<1x144x5x5xf32>) -> tensor<144x1x5x5xf32>
    %b4ddb = stablehlo.reduce(%b4ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %b4ders = stablehlo.logistic %b4en : tensor<32x144x56x56xf32>
    %b4derone = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %b4derom = stablehlo.subtract %b4derone, %b4ders : tensor<32x144x56x56xf32>
    %b4derxom = stablehlo.multiply %b4en, %b4derom : tensor<32x144x56x56xf32>
    %b4derin = stablehlo.add %b4derone, %b4derxom : tensor<32x144x56x56xf32>
    %b4dersp = stablehlo.multiply %b4ders, %b4derin : tensor<32x144x56x56xf32>
    %b4der = stablehlo.multiply %b4dd, %b4dersp : tensor<32x144x56x56xf32>
    %b4dendxh = stablehlo.multiply %b4engb, %b4der : tensor<32x144x56x56xf32>
    %b4densdxr = stablehlo.reduce(%b4dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4densdx = stablehlo.broadcast_in_dim %b4densdxr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4denxd = stablehlo.multiply %b4enxh, %b4dendxh : tensor<32x144x56x56xf32>
    %b4densxdr = stablehlo.reduce(%b4denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4densxd = stablehlo.broadcast_in_dim %b4densxdr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b4dent1 = stablehlo.multiply %b4dendxh, %b4ennf : tensor<32x144x56x56xf32>
    %b4deni1 = stablehlo.subtract %b4dent1, %b4densdx : tensor<32x144x56x56xf32>
    %b4denxs = stablehlo.multiply %b4enxh, %b4densxd : tensor<32x144x56x56xf32>
    %b4deni2 = stablehlo.subtract %b4deni1, %b4denxs : tensor<32x144x56x56xf32>
    %b4densN = stablehlo.divide %b4enistd, %b4ennf : tensor<32x144x56x56xf32>
    %b4den = stablehlo.multiply %b4densN, %b4deni2 : tensor<32x144x56x56xf32>
    %b4dendgp = stablehlo.multiply %b4der, %b4enxh : tensor<32x144x56x56xf32>
    %b4dendg = stablehlo.reduce(%b4dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4dendb = stablehlo.reduce(%b4der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b4det = stablehlo.transpose %b4eW, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %b4de = stablehlo.convolution(%b4den, %b4det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b4deWxt = stablehlo.transpose %b3o, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b4deWdt = stablehlo.transpose %b4den, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b4deWraw = stablehlo.convolution(%b4deWxt, %b4deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %b4deW = stablehlo.transpose %b4deWraw, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %b4deb = stablehlo.reduce(%b4den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dpndxh = stablehlo.multiply %b3pngb, %b4de : tensor<32x24x56x56xf32>
    %b3dpnsdxr = stablehlo.reduce(%b3dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3dpnsdx = stablehlo.broadcast_in_dim %b3dpnsdxr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3dpnxd = stablehlo.multiply %b3pnxh, %b3dpndxh : tensor<32x24x56x56xf32>
    %b3dpnsxdr = stablehlo.reduce(%b3dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3dpnsxd = stablehlo.broadcast_in_dim %b3dpnsxdr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b3dpnt1 = stablehlo.multiply %b3dpndxh, %b3pnnf : tensor<32x24x56x56xf32>
    %b3dpni1 = stablehlo.subtract %b3dpnt1, %b3dpnsdx : tensor<32x24x56x56xf32>
    %b3dpnxs = stablehlo.multiply %b3pnxh, %b3dpnsxd : tensor<32x24x56x56xf32>
    %b3dpni2 = stablehlo.subtract %b3dpni1, %b3dpnxs : tensor<32x24x56x56xf32>
    %b3dpnsN = stablehlo.divide %b3pnistd, %b3pnnf : tensor<32x24x56x56xf32>
    %b3dpn = stablehlo.multiply %b3dpnsN, %b3dpni2 : tensor<32x24x56x56xf32>
    %b3dpndgp = stablehlo.multiply %b4de, %b3pnxh : tensor<32x24x56x56xf32>
    %b3dpndg = stablehlo.reduce(%b3dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3dpndb = stablehlo.reduce(%b4de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3dpt = stablehlo.transpose %b3pW, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %b3dp = stablehlo.convolution(%b3dpn, %b3dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %b3dpWxt = stablehlo.transpose %b3zse, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b3dpWdt = stablehlo.transpose %b3dpn, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b3dpWraw = stablehlo.convolution(%b3dpWxt, %b3dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %b3dpW = stablehlo.transpose %b3dpWraw, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %b3dpb = stablehlo.reduce(%b3dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b3zgb2 = stablehlo.broadcast_in_dim %b3zgate, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %b3zdleft = stablehlo.multiply %b3zgb2, %b3dp : tensor<32x144x56x56xf32>
    %b3zxdse = stablehlo.multiply %b3ds, %b3dp : tensor<32x144x56x56xf32>
    %b3zdgate = stablehlo.reduce(%b3zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %b3zone = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %b3zomg = stablehlo.subtract %b3zone, %b3zgate : tensor<32x144xf32>
    %b3zsg = stablehlo.multiply %b3zgate, %b3zomg : tensor<32x144xf32>
    %b3zdh2 = stablehlo.multiply %b3zdgate, %b3zsg : tensor<32x144xf32>
    %b3zda1 = stablehlo.dot_general %b3zdh2, %b3zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %b3zdWs2 = stablehlo.dot_general %b3za1, %b3zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %b3zdbs2 = stablehlo.reduce(%b3zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %b3zdexs = stablehlo.logistic %b3zex : tensor<32x6xf32>
    %b3zdexone = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %b3zdexom = stablehlo.subtract %b3zdexone, %b3zdexs : tensor<32x6xf32>
    %b3zdexxom = stablehlo.multiply %b3zex, %b3zdexom : tensor<32x6xf32>
    %b3zdexin = stablehlo.add %b3zdexone, %b3zdexxom : tensor<32x6xf32>
    %b3zdexsp = stablehlo.multiply %b3zdexs, %b3zdexin : tensor<32x6xf32>
    %b3zdex = stablehlo.multiply %b3zda1, %b3zdexsp : tensor<32x6xf32>
    %b3zdsq = stablehlo.dot_general %b3zdex, %b3zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %b3zdWs1 = stablehlo.dot_general %b3zsq, %b3zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %b3zdbs1 = stablehlo.reduce(%b3zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %b3zdsqnf = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %b3zdsqd = stablehlo.divide %b3zdsq, %b3zdsqnf : tensor<32x144xf32>
    %b3zdgsp = stablehlo.broadcast_in_dim %b3zdsqd, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %b3zdds = stablehlo.add %b3zdleft, %b3zdgsp : tensor<32x144x56x56xf32>
    %b3ddrs = stablehlo.logistic %b3dn : tensor<32x144x56x56xf32>
    %b3ddrone = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %b3ddrom = stablehlo.subtract %b3ddrone, %b3ddrs : tensor<32x144x56x56xf32>
    %b3ddrxom = stablehlo.multiply %b3dn, %b3ddrom : tensor<32x144x56x56xf32>
    %b3ddrin = stablehlo.add %b3ddrone, %b3ddrxom : tensor<32x144x56x56xf32>
    %b3ddrsp = stablehlo.multiply %b3ddrs, %b3ddrin : tensor<32x144x56x56xf32>
    %b3ddr = stablehlo.multiply %b3zdds, %b3ddrsp : tensor<32x144x56x56xf32>
    %b3ddndxh = stablehlo.multiply %b3dngb, %b3ddr : tensor<32x144x56x56xf32>
    %b3ddnsdxr = stablehlo.reduce(%b3ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ddnsdx = stablehlo.broadcast_in_dim %b3ddnsdxr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3ddnxd = stablehlo.multiply %b3dnxh, %b3ddndxh : tensor<32x144x56x56xf32>
    %b3ddnsxdr = stablehlo.reduce(%b3ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ddnsxd = stablehlo.broadcast_in_dim %b3ddnsxdr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3ddnt1 = stablehlo.multiply %b3ddndxh, %b3dnnf : tensor<32x144x56x56xf32>
    %b3ddni1 = stablehlo.subtract %b3ddnt1, %b3ddnsdx : tensor<32x144x56x56xf32>
    %b3ddnxs = stablehlo.multiply %b3dnxh, %b3ddnsxd : tensor<32x144x56x56xf32>
    %b3ddni2 = stablehlo.subtract %b3ddni1, %b3ddnxs : tensor<32x144x56x56xf32>
    %b3ddnsN = stablehlo.divide %b3dnistd, %b3dnnf : tensor<32x144x56x56xf32>
    %b3ddn = stablehlo.multiply %b3ddnsN, %b3ddni2 : tensor<32x144x56x56xf32>
    %b3ddndgp = stablehlo.multiply %b3ddr, %b3dnxh : tensor<32x144x56x56xf32>
    %b3ddndg = stablehlo.reduce(%b3ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ddndb = stablehlo.reduce(%b3ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ddrev = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %b3dd = stablehlo.convolution(%b3ddn, %b3ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %b3ddWxt = stablehlo.transpose %b3es, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b3ddWdt = stablehlo.transpose %b3ddn, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b3ddWraw = stablehlo.convolution(%b3ddWxt, %b3ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %b3ddW = stablehlo.reshape %b3ddWraw : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %b3ddb = stablehlo.reduce(%b3ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3ders = stablehlo.logistic %b3en : tensor<32x144x56x56xf32>
    %b3derone = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %b3derom = stablehlo.subtract %b3derone, %b3ders : tensor<32x144x56x56xf32>
    %b3derxom = stablehlo.multiply %b3en, %b3derom : tensor<32x144x56x56xf32>
    %b3derin = stablehlo.add %b3derone, %b3derxom : tensor<32x144x56x56xf32>
    %b3dersp = stablehlo.multiply %b3ders, %b3derin : tensor<32x144x56x56xf32>
    %b3der = stablehlo.multiply %b3dd, %b3dersp : tensor<32x144x56x56xf32>
    %b3dendxh = stablehlo.multiply %b3engb, %b3der : tensor<32x144x56x56xf32>
    %b3densdxr = stablehlo.reduce(%b3dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3densdx = stablehlo.broadcast_in_dim %b3densdxr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3denxd = stablehlo.multiply %b3enxh, %b3dendxh : tensor<32x144x56x56xf32>
    %b3densxdr = stablehlo.reduce(%b3denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3densxd = stablehlo.broadcast_in_dim %b3densxdr, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %b3dent1 = stablehlo.multiply %b3dendxh, %b3ennf : tensor<32x144x56x56xf32>
    %b3deni1 = stablehlo.subtract %b3dent1, %b3densdx : tensor<32x144x56x56xf32>
    %b3denxs = stablehlo.multiply %b3enxh, %b3densxd : tensor<32x144x56x56xf32>
    %b3deni2 = stablehlo.subtract %b3deni1, %b3denxs : tensor<32x144x56x56xf32>
    %b3densN = stablehlo.divide %b3enistd, %b3ennf : tensor<32x144x56x56xf32>
    %b3den = stablehlo.multiply %b3densN, %b3deni2 : tensor<32x144x56x56xf32>
    %b3dendgp = stablehlo.multiply %b3der, %b3enxh : tensor<32x144x56x56xf32>
    %b3dendg = stablehlo.reduce(%b3dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dendb = stablehlo.reduce(%b3der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3det = stablehlo.transpose %b3eW, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %b3de = stablehlo.convolution(%b3den, %b3det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %b3deWxt = stablehlo.transpose %b2pn, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b3deWdt = stablehlo.transpose %b3den, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %b3deWraw = stablehlo.convolution(%b3deWxt, %b3deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %b3deW = stablehlo.transpose %b3deWraw, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %b3deb = stablehlo.reduce(%b3den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %b3dx = stablehlo.add %b3de, %b4de : tensor<32x24x56x56xf32>
    %b2dpndxh = stablehlo.multiply %b2pngb, %b3dx : tensor<32x24x56x56xf32>
    %b2dpnsdxr = stablehlo.reduce(%b2dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpnsdx = stablehlo.broadcast_in_dim %b2dpnsdxr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2dpnxd = stablehlo.multiply %b2pnxh, %b2dpndxh : tensor<32x24x56x56xf32>
    %b2dpnsxdr = stablehlo.reduce(%b2dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpnsxd = stablehlo.broadcast_in_dim %b2dpnsxdr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2dpnt1 = stablehlo.multiply %b2dpndxh, %b2pnnf : tensor<32x24x56x56xf32>
    %b2dpni1 = stablehlo.subtract %b2dpnt1, %b2dpnsdx : tensor<32x24x56x56xf32>
    %b2dpnxs = stablehlo.multiply %b2pnxh, %b2dpnsxd : tensor<32x24x56x56xf32>
    %b2dpni2 = stablehlo.subtract %b2dpni1, %b2dpnxs : tensor<32x24x56x56xf32>
    %b2dpnsN = stablehlo.divide %b2pnistd, %b2pnnf : tensor<32x24x56x56xf32>
    %b2dpn = stablehlo.multiply %b2dpnsN, %b2dpni2 : tensor<32x24x56x56xf32>
    %b2dpndgp = stablehlo.multiply %b3dx, %b2pnxh : tensor<32x24x56x56xf32>
    %b2dpndg = stablehlo.reduce(%b2dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpndb = stablehlo.reduce(%b3dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpt = stablehlo.transpose %b2pW, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b2dp = stablehlo.convolution(%b2dpn, %b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %b2dpWxt = stablehlo.transpose %b2zse, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2dpWdt = stablehlo.transpose %b2dpn, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b2dpWraw = stablehlo.convolution(%b2dpWxt, %b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %b2dpW = stablehlo.transpose %b2dpWraw, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %b2dpb = stablehlo.reduce(%b2dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2zgb2 = stablehlo.broadcast_in_dim %b2zgate, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2zdleft = stablehlo.multiply %b2zgb2, %b2dp : tensor<32x96x56x56xf32>
    %b2zxdse = stablehlo.multiply %b2ds, %b2dp : tensor<32x96x56x56xf32>
    %b2zdgate = stablehlo.reduce(%b2zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b2zone = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %b2zomg = stablehlo.subtract %b2zone, %b2zgate : tensor<32x96xf32>
    %b2zsg = stablehlo.multiply %b2zgate, %b2zomg : tensor<32x96xf32>
    %b2zdh2 = stablehlo.multiply %b2zdgate, %b2zsg : tensor<32x96xf32>
    %b2zda1 = stablehlo.dot_general %b2zdh2, %b2zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<4x96xf32>) -> tensor<32x4xf32>
    %b2zdWs2 = stablehlo.dot_general %b2za1, %b2zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<32x96xf32>) -> tensor<4x96xf32>
    %b2zdbs2 = stablehlo.reduce(%b2zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x96xf32>, tensor<f32>) -> tensor<96xf32>
    %b2zdexs = stablehlo.logistic %b2zex : tensor<32x4xf32>
    %b2zdexone = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %b2zdexom = stablehlo.subtract %b2zdexone, %b2zdexs : tensor<32x4xf32>
    %b2zdexxom = stablehlo.multiply %b2zex, %b2zdexom : tensor<32x4xf32>
    %b2zdexin = stablehlo.add %b2zdexone, %b2zdexxom : tensor<32x4xf32>
    %b2zdexsp = stablehlo.multiply %b2zdexs, %b2zdexin : tensor<32x4xf32>
    %b2zdex = stablehlo.multiply %b2zda1, %b2zdexsp : tensor<32x4xf32>
    %b2zdsq = stablehlo.dot_general %b2zdex, %b2zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<96x4xf32>) -> tensor<32x96xf32>
    %b2zdWs1 = stablehlo.dot_general %b2zsq, %b2zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<32x4xf32>) -> tensor<96x4xf32>
    %b2zdbs1 = stablehlo.reduce(%b2zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x4xf32>, tensor<f32>) -> tensor<4xf32>
    %b2zdsqnf = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %b2zdsqd = stablehlo.divide %b2zdsq, %b2zdsqnf : tensor<32x96xf32>
    %b2zdgsp = stablehlo.broadcast_in_dim %b2zdsqd, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2zdds = stablehlo.add %b2zdleft, %b2zdgsp : tensor<32x96x56x56xf32>
    %b2ddrs = stablehlo.logistic %b2dn : tensor<32x96x56x56xf32>
    %b2ddrone = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %b2ddrom = stablehlo.subtract %b2ddrone, %b2ddrs : tensor<32x96x56x56xf32>
    %b2ddrxom = stablehlo.multiply %b2dn, %b2ddrom : tensor<32x96x56x56xf32>
    %b2ddrin = stablehlo.add %b2ddrone, %b2ddrxom : tensor<32x96x56x56xf32>
    %b2ddrsp = stablehlo.multiply %b2ddrs, %b2ddrin : tensor<32x96x56x56xf32>
    %b2ddr = stablehlo.multiply %b2zdds, %b2ddrsp : tensor<32x96x56x56xf32>
    %b2ddndxh = stablehlo.multiply %b2dngb, %b2ddr : tensor<32x96x56x56xf32>
    %b2ddnsdxr = stablehlo.reduce(%b2ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddnsdx = stablehlo.broadcast_in_dim %b2ddnsdxr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2ddnxd = stablehlo.multiply %b2dnxh, %b2ddndxh : tensor<32x96x56x56xf32>
    %b2ddnsxdr = stablehlo.reduce(%b2ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddnsxd = stablehlo.broadcast_in_dim %b2ddnsxdr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2ddnt1 = stablehlo.multiply %b2ddndxh, %b2dnnf : tensor<32x96x56x56xf32>
    %b2ddni1 = stablehlo.subtract %b2ddnt1, %b2ddnsdx : tensor<32x96x56x56xf32>
    %b2ddnxs = stablehlo.multiply %b2dnxh, %b2ddnsxd : tensor<32x96x56x56xf32>
    %b2ddni2 = stablehlo.subtract %b2ddni1, %b2ddnxs : tensor<32x96x56x56xf32>
    %b2ddnsN = stablehlo.divide %b2dnistd, %b2dnnf : tensor<32x96x56x56xf32>
    %b2ddn = stablehlo.multiply %b2ddnsN, %b2ddni2 : tensor<32x96x56x56xf32>
    %b2ddndgp = stablehlo.multiply %b2ddr, %b2dnxh : tensor<32x96x56x56xf32>
    %b2ddndg = stablehlo.reduce(%b2ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddndb = stablehlo.reduce(%b2ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddu = stablehlo.pad %b2ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %b2ddrev = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %b2dd = stablehlo.convolution(%b2ddu, %b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %b2ddWu = stablehlo.pad %b2ddn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %b2ddWxt = stablehlo.transpose %b2es, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %b2ddWdt = stablehlo.transpose %b2ddWu, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %b2ddWraw = stablehlo.convolution(%b2ddWxt, %b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %b2ddW = stablehlo.reshape %b2ddWraw : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %b2ddb = stablehlo.reduce(%b2ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ders = stablehlo.logistic %b2en : tensor<32x96x112x112xf32>
    %b2derone = stablehlo.constant dense<1.0> : tensor<32x96x112x112xf32>
    %b2derom = stablehlo.subtract %b2derone, %b2ders : tensor<32x96x112x112xf32>
    %b2derxom = stablehlo.multiply %b2en, %b2derom : tensor<32x96x112x112xf32>
    %b2derin = stablehlo.add %b2derone, %b2derxom : tensor<32x96x112x112xf32>
    %b2dersp = stablehlo.multiply %b2ders, %b2derin : tensor<32x96x112x112xf32>
    %b2der = stablehlo.multiply %b2dd, %b2dersp : tensor<32x96x112x112xf32>
    %b2dendxh = stablehlo.multiply %b2engb, %b2der : tensor<32x96x112x112xf32>
    %b2densdxr = stablehlo.reduce(%b2dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2densdx = stablehlo.broadcast_in_dim %b2densdxr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2denxd = stablehlo.multiply %b2enxh, %b2dendxh : tensor<32x96x112x112xf32>
    %b2densxdr = stablehlo.reduce(%b2denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2densxd = stablehlo.broadcast_in_dim %b2densxdr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %b2dent1 = stablehlo.multiply %b2dendxh, %b2ennf : tensor<32x96x112x112xf32>
    %b2deni1 = stablehlo.subtract %b2dent1, %b2densdx : tensor<32x96x112x112xf32>
    %b2denxs = stablehlo.multiply %b2enxh, %b2densxd : tensor<32x96x112x112xf32>
    %b2deni2 = stablehlo.subtract %b2deni1, %b2denxs : tensor<32x96x112x112xf32>
    %b2densN = stablehlo.divide %b2enistd, %b2ennf : tensor<32x96x112x112xf32>
    %b2den = stablehlo.multiply %b2densN, %b2deni2 : tensor<32x96x112x112xf32>
    %b2dendgp = stablehlo.multiply %b2der, %b2enxh : tensor<32x96x112x112xf32>
    %b2dendg = stablehlo.reduce(%b2dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dendb = stablehlo.reduce(%b2der init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b2det = stablehlo.transpose %b2eW, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %b2de = stablehlo.convolution(%b2den, %b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %b2deWxt = stablehlo.transpose %b1pn, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %b2deWdt = stablehlo.transpose %b2den, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %b2deWraw = stablehlo.convolution(%b2deWxt, %b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %b2deW = stablehlo.transpose %b2deWraw, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %b2deb = stablehlo.reduce(%b2den init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %b1dpndxh = stablehlo.multiply %b1pngb, %b2de : tensor<32x16x112x112xf32>
    %b1dpnsdxr = stablehlo.reduce(%b1dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1dpnsdx = stablehlo.broadcast_in_dim %b1dpnsdxr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1dpnxd = stablehlo.multiply %b1pnxh, %b1dpndxh : tensor<32x16x112x112xf32>
    %b1dpnsxdr = stablehlo.reduce(%b1dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1dpnsxd = stablehlo.broadcast_in_dim %b1dpnsxdr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %b1dpnt1 = stablehlo.multiply %b1dpndxh, %b1pnnf : tensor<32x16x112x112xf32>
    %b1dpni1 = stablehlo.subtract %b1dpnt1, %b1dpnsdx : tensor<32x16x112x112xf32>
    %b1dpnxs = stablehlo.multiply %b1pnxh, %b1dpnsxd : tensor<32x16x112x112xf32>
    %b1dpni2 = stablehlo.subtract %b1dpni1, %b1dpnxs : tensor<32x16x112x112xf32>
    %b1dpnsN = stablehlo.divide %b1pnistd, %b1pnnf : tensor<32x16x112x112xf32>
    %b1dpn = stablehlo.multiply %b1dpnsN, %b1dpni2 : tensor<32x16x112x112xf32>
    %b1dpndgp = stablehlo.multiply %b2de, %b1pnxh : tensor<32x16x112x112xf32>
    %b1dpndg = stablehlo.reduce(%b1dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1dpndb = stablehlo.reduce(%b2de init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1dpt = stablehlo.transpose %b1pW, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %b1dp = stablehlo.convolution(%b1dpn, %b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %b1dpWxt = stablehlo.transpose %b1zse, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %b1dpWdt = stablehlo.transpose %b1dpn, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %b1dpWraw = stablehlo.convolution(%b1dpWxt, %b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %b1dpW = stablehlo.transpose %b1dpWraw, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %b1dpb = stablehlo.reduce(%b1dpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b1zgb2 = stablehlo.broadcast_in_dim %b1zgate, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %b1zdleft = stablehlo.multiply %b1zgb2, %b1dp : tensor<32x32x112x112xf32>
    %b1zxdse = stablehlo.multiply %b1ds, %b1dp : tensor<32x32x112x112xf32>
    %b1zdgate = stablehlo.reduce(%b1zxdse init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %b1zone = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %b1zomg = stablehlo.subtract %b1zone, %b1zgate : tensor<32x32xf32>
    %b1zsg = stablehlo.multiply %b1zgate, %b1zomg : tensor<32x32xf32>
    %b1zdh2 = stablehlo.multiply %b1zdgate, %b1zsg : tensor<32x32xf32>
    %b1zda1 = stablehlo.dot_general %b1zdh2, %b1zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<8x32xf32>) -> tensor<32x8xf32>
    %b1zdWs2 = stablehlo.dot_general %b1za1, %b1zdh2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x32xf32>) -> tensor<8x32xf32>
    %b1zdbs2 = stablehlo.reduce(%b1zdh2 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %b1zdexs = stablehlo.logistic %b1zex : tensor<32x8xf32>
    %b1zdexone = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %b1zdexom = stablehlo.subtract %b1zdexone, %b1zdexs : tensor<32x8xf32>
    %b1zdexxom = stablehlo.multiply %b1zex, %b1zdexom : tensor<32x8xf32>
    %b1zdexin = stablehlo.add %b1zdexone, %b1zdexxom : tensor<32x8xf32>
    %b1zdexsp = stablehlo.multiply %b1zdexs, %b1zdexin : tensor<32x8xf32>
    %b1zdex = stablehlo.multiply %b1zda1, %b1zdexsp : tensor<32x8xf32>
    %b1zdsq = stablehlo.dot_general %b1zdex, %b1zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x32xf32>
    %b1zdWs1 = stablehlo.dot_general %b1zsq, %b1zdex, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %b1zdbs1 = stablehlo.reduce(%b1zdex init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x8xf32>, tensor<f32>) -> tensor<8xf32>
    %b1zdsqnf = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %b1zdsqd = stablehlo.divide %b1zdsq, %b1zdsqnf : tensor<32x32xf32>
    %b1zdgsp = stablehlo.broadcast_in_dim %b1zdsqd, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %b1zdds = stablehlo.add %b1zdleft, %b1zdgsp : tensor<32x32x112x112xf32>
    %b1ddrs = stablehlo.logistic %b1dn : tensor<32x32x112x112xf32>
    %b1ddrone = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %b1ddrom = stablehlo.subtract %b1ddrone, %b1ddrs : tensor<32x32x112x112xf32>
    %b1ddrxom = stablehlo.multiply %b1dn, %b1ddrom : tensor<32x32x112x112xf32>
    %b1ddrin = stablehlo.add %b1ddrone, %b1ddrxom : tensor<32x32x112x112xf32>
    %b1ddrsp = stablehlo.multiply %b1ddrs, %b1ddrin : tensor<32x32x112x112xf32>
    %b1ddr = stablehlo.multiply %b1zdds, %b1ddrsp : tensor<32x32x112x112xf32>
    %b1ddndxh = stablehlo.multiply %b1dngb, %b1ddr : tensor<32x32x112x112xf32>
    %b1ddnsdxr = stablehlo.reduce(%b1ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ddnsdx = stablehlo.broadcast_in_dim %b1ddnsdxr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1ddnxd = stablehlo.multiply %b1dnxh, %b1ddndxh : tensor<32x32x112x112xf32>
    %b1ddnsxdr = stablehlo.reduce(%b1ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ddnsxd = stablehlo.broadcast_in_dim %b1ddnsxdr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %b1ddnt1 = stablehlo.multiply %b1ddndxh, %b1dnnf : tensor<32x32x112x112xf32>
    %b1ddni1 = stablehlo.subtract %b1ddnt1, %b1ddnsdx : tensor<32x32x112x112xf32>
    %b1ddnxs = stablehlo.multiply %b1dnxh, %b1ddnsxd : tensor<32x32x112x112xf32>
    %b1ddni2 = stablehlo.subtract %b1ddni1, %b1ddnxs : tensor<32x32x112x112xf32>
    %b1ddnsN = stablehlo.divide %b1dnistd, %b1dnnf : tensor<32x32x112x112xf32>
    %b1ddn = stablehlo.multiply %b1ddnsN, %b1ddni2 : tensor<32x32x112x112xf32>
    %b1ddndgp = stablehlo.multiply %b1ddr, %b1dnxh : tensor<32x32x112x112xf32>
    %b1ddndg = stablehlo.reduce(%b1ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ddndb = stablehlo.reduce(%b1ddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %b1ddrev = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %b1dd = stablehlo.convolution(%b1ddn, %b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %b1ddWxt = stablehlo.transpose %str, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %b1ddWdt = stablehlo.transpose %b1ddn, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %b1ddWraw = stablehlo.convolution(%b1ddWxt, %b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %b1ddW = stablehlo.reshape %b1ddWraw : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %b1ddb = stablehlo.reduce(%b1ddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dstrs = stablehlo.logistic %stn : tensor<32x32x112x112xf32>
    %dstrone = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %dstrom = stablehlo.subtract %dstrone, %dstrs : tensor<32x32x112x112xf32>
    %dstrxom = stablehlo.multiply %stn, %dstrom : tensor<32x32x112x112xf32>
    %dstrin = stablehlo.add %dstrone, %dstrxom : tensor<32x32x112x112xf32>
    %dstrsp = stablehlo.multiply %dstrs, %dstrin : tensor<32x32x112x112xf32>
    %dstr = stablehlo.multiply %b1dd, %dstrsp : tensor<32x32x112x112xf32>
    %dstndxh = stablehlo.multiply %stngb, %dstr : tensor<32x32x112x112xf32>
    %dstnsdxr = stablehlo.reduce(%dstndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dstnsdx = stablehlo.broadcast_in_dim %dstnsdxr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %dstnxd = stablehlo.multiply %stnxh, %dstndxh : tensor<32x32x112x112xf32>
    %dstnsxdr = stablehlo.reduce(%dstnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dstnsxd = stablehlo.broadcast_in_dim %dstnsxdr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %dstnt1 = stablehlo.multiply %dstndxh, %stnnf : tensor<32x32x112x112xf32>
    %dstni1 = stablehlo.subtract %dstnt1, %dstnsdx : tensor<32x32x112x112xf32>
    %dstnxs = stablehlo.multiply %stnxh, %dstnsxd : tensor<32x32x112x112xf32>
    %dstni2 = stablehlo.subtract %dstni1, %dstnxs : tensor<32x32x112x112xf32>
    %dstnsN = stablehlo.divide %stnistd, %stnnf : tensor<32x32x112x112xf32>
    %dstn = stablehlo.multiply %dstnsN, %dstni2 : tensor<32x32x112x112xf32>
    %dstndgp = stablehlo.multiply %dstr, %stnxh : tensor<32x32x112x112xf32>
    %dstndg = stablehlo.reduce(%dstndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dstndb = stablehlo.reduce(%dstr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dsb = stablehlo.reduce(%dstn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %dsWu = stablehlo.pad %dstn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %dsWxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %dsWdt = stablehlo.transpose %dsWu, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %dsWraw = stablehlo.convolution(%dsWxt, %dsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %dsW = stablehlo.transpose %dsWraw, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %adb1sW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %adob1sW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %admssW = stablehlo.multiply %adb1sW, %sWm : tensor<32x3x3x3xf32>
    %admgsW = stablehlo.multiply %adob1sW, %dsW : tensor<32x3x3x3xf32>
    %admnsW = stablehlo.add %admssW, %admgsW : tensor<32x3x3x3xf32>
    %adb2sW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %adob2sW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %advssW = stablehlo.multiply %adb2sW, %sWv : tensor<32x3x3x3xf32>
    %adg2sW = stablehlo.multiply %dsW, %dsW : tensor<32x3x3x3xf32>
    %advgsW = stablehlo.multiply %adob2sW, %adg2sW : tensor<32x3x3x3xf32>
    %advnsW = stablehlo.add %advssW, %advgsW : tensor<32x3x3x3xf32>
    %adbc1sW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %adbc2sW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %admhsW = stablehlo.divide %admnsW, %adbc1sW : tensor<32x3x3x3xf32>
    %advhsW = stablehlo.divide %advnsW, %adbc2sW : tensor<32x3x3x3xf32>
    %adlrsW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %adepssW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %adsqsW = stablehlo.sqrt %advhsW : tensor<32x3x3x3xf32>
    %addensW = stablehlo.add %adsqsW, %adepssW : tensor<32x3x3x3xf32>
    %adratsW = stablehlo.divide %admhsW, %addensW : tensor<32x3x3x3xf32>
    %adstsW = stablehlo.multiply %adlrsW, %adratsW : tensor<32x3x3x3xf32>
    %adsubsW = stablehlo.subtract %sW, %adstsW : tensor<32x3x3x3xf32>
    %adwdsW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %adwdlrsW = stablehlo.multiply %adwdsW, %adlrsW : tensor<32x3x3x3xf32>
    %adwdpsW = stablehlo.multiply %adwdlrsW, %sW : tensor<32x3x3x3xf32>
    %adnewsW = stablehlo.subtract %adsubsW, %adwdpsW : tensor<32x3x3x3xf32>
    %adb1sb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1sb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admssb = stablehlo.multiply %adb1sb, %sbm : tensor<32xf32>
    %admgsb = stablehlo.multiply %adob1sb, %dsb : tensor<32xf32>
    %admnsb = stablehlo.add %admssb, %admgsb : tensor<32xf32>
    %adb2sb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2sb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advssb = stablehlo.multiply %adb2sb, %sbv : tensor<32xf32>
    %adg2sb = stablehlo.multiply %dsb, %dsb : tensor<32xf32>
    %advgsb = stablehlo.multiply %adob2sb, %adg2sb : tensor<32xf32>
    %advnsb = stablehlo.add %advssb, %advgsb : tensor<32xf32>
    %adbc1sb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2sb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhsb = stablehlo.divide %admnsb, %adbc1sb : tensor<32xf32>
    %advhsb = stablehlo.divide %advnsb, %adbc2sb : tensor<32xf32>
    %adlrsb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepssb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqsb = stablehlo.sqrt %advhsb : tensor<32xf32>
    %addensb = stablehlo.add %adsqsb, %adepssb : tensor<32xf32>
    %adratsb = stablehlo.divide %admhsb, %addensb : tensor<32xf32>
    %adstsb = stablehlo.multiply %adlrsb, %adratsb : tensor<32xf32>
    %adsubsb = stablehlo.subtract %sb, %adstsb : tensor<32xf32>
    %adwdsb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrsb = stablehlo.multiply %adwdsb, %adlrsb : tensor<32xf32>
    %adwdpsb = stablehlo.multiply %adwdlrsb, %sb : tensor<32xf32>
    %adnewsb = stablehlo.subtract %adsubsb, %adwdpsb : tensor<32xf32>
    %adb1sg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1sg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admssg = stablehlo.multiply %adb1sg, %sgm : tensor<32xf32>
    %admgsg = stablehlo.multiply %adob1sg, %dstndg : tensor<32xf32>
    %admnsg = stablehlo.add %admssg, %admgsg : tensor<32xf32>
    %adb2sg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2sg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advssg = stablehlo.multiply %adb2sg, %sgv : tensor<32xf32>
    %adg2sg = stablehlo.multiply %dstndg, %dstndg : tensor<32xf32>
    %advgsg = stablehlo.multiply %adob2sg, %adg2sg : tensor<32xf32>
    %advnsg = stablehlo.add %advssg, %advgsg : tensor<32xf32>
    %adbc1sg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2sg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhsg = stablehlo.divide %admnsg, %adbc1sg : tensor<32xf32>
    %advhsg = stablehlo.divide %advnsg, %adbc2sg : tensor<32xf32>
    %adlrsg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepssg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqsg = stablehlo.sqrt %advhsg : tensor<32xf32>
    %addensg = stablehlo.add %adsqsg, %adepssg : tensor<32xf32>
    %adratsg = stablehlo.divide %admhsg, %addensg : tensor<32xf32>
    %adstsg = stablehlo.multiply %adlrsg, %adratsg : tensor<32xf32>
    %adsubsg = stablehlo.subtract %sg, %adstsg : tensor<32xf32>
    %adwdsg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrsg = stablehlo.multiply %adwdsg, %adlrsg : tensor<32xf32>
    %adwdpsg = stablehlo.multiply %adwdlrsg, %sg : tensor<32xf32>
    %adnewsg = stablehlo.subtract %adsubsg, %adwdpsg : tensor<32xf32>
    %adb1sbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1sbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admssbt = stablehlo.multiply %adb1sbt, %sbtm : tensor<32xf32>
    %admgsbt = stablehlo.multiply %adob1sbt, %dstndb : tensor<32xf32>
    %admnsbt = stablehlo.add %admssbt, %admgsbt : tensor<32xf32>
    %adb2sbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2sbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advssbt = stablehlo.multiply %adb2sbt, %sbtv : tensor<32xf32>
    %adg2sbt = stablehlo.multiply %dstndb, %dstndb : tensor<32xf32>
    %advgsbt = stablehlo.multiply %adob2sbt, %adg2sbt : tensor<32xf32>
    %advnsbt = stablehlo.add %advssbt, %advgsbt : tensor<32xf32>
    %adbc1sbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2sbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhsbt = stablehlo.divide %admnsbt, %adbc1sbt : tensor<32xf32>
    %advhsbt = stablehlo.divide %advnsbt, %adbc2sbt : tensor<32xf32>
    %adlrsbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepssbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqsbt = stablehlo.sqrt %advhsbt : tensor<32xf32>
    %addensbt = stablehlo.add %adsqsbt, %adepssbt : tensor<32xf32>
    %adratsbt = stablehlo.divide %admhsbt, %addensbt : tensor<32xf32>
    %adstsbt = stablehlo.multiply %adlrsbt, %adratsbt : tensor<32xf32>
    %adsubsbt = stablehlo.subtract %sbt, %adstsbt : tensor<32xf32>
    %adwdsbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrsbt = stablehlo.multiply %adwdsbt, %adlrsbt : tensor<32xf32>
    %adwdpsbt = stablehlo.multiply %adwdlrsbt, %sbt : tensor<32xf32>
    %adnewsbt = stablehlo.subtract %adsubsbt, %adwdpsbt : tensor<32xf32>
    %adb1b1dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %adob1b1dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %admsb1dW = stablehlo.multiply %adb1b1dW, %b1dWm : tensor<32x1x3x3xf32>
    %admgb1dW = stablehlo.multiply %adob1b1dW, %b1ddW : tensor<32x1x3x3xf32>
    %admnb1dW = stablehlo.add %admsb1dW, %admgb1dW : tensor<32x1x3x3xf32>
    %adb2b1dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %adob2b1dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %advsb1dW = stablehlo.multiply %adb2b1dW, %b1dWv : tensor<32x1x3x3xf32>
    %adg2b1dW = stablehlo.multiply %b1ddW, %b1ddW : tensor<32x1x3x3xf32>
    %advgb1dW = stablehlo.multiply %adob2b1dW, %adg2b1dW : tensor<32x1x3x3xf32>
    %advnb1dW = stablehlo.add %advsb1dW, %advgb1dW : tensor<32x1x3x3xf32>
    %adbc1b1dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %adbc2b1dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %admhb1dW = stablehlo.divide %admnb1dW, %adbc1b1dW : tensor<32x1x3x3xf32>
    %advhb1dW = stablehlo.divide %advnb1dW, %adbc2b1dW : tensor<32x1x3x3xf32>
    %adlrb1dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %adepsb1dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %adsqb1dW = stablehlo.sqrt %advhb1dW : tensor<32x1x3x3xf32>
    %addenb1dW = stablehlo.add %adsqb1dW, %adepsb1dW : tensor<32x1x3x3xf32>
    %adratb1dW = stablehlo.divide %admhb1dW, %addenb1dW : tensor<32x1x3x3xf32>
    %adstb1dW = stablehlo.multiply %adlrb1dW, %adratb1dW : tensor<32x1x3x3xf32>
    %adsubb1dW = stablehlo.subtract %b1dW, %adstb1dW : tensor<32x1x3x3xf32>
    %adwdb1dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %adwdlrb1dW = stablehlo.multiply %adwdb1dW, %adlrb1dW : tensor<32x1x3x3xf32>
    %adwdpb1dW = stablehlo.multiply %adwdlrb1dW, %b1dW : tensor<32x1x3x3xf32>
    %adnewb1dW = stablehlo.subtract %adsubb1dW, %adwdpb1dW : tensor<32x1x3x3xf32>
    %adb1b1db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b1db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb1db = stablehlo.multiply %adb1b1db, %b1dbm : tensor<32xf32>
    %admgb1db = stablehlo.multiply %adob1b1db, %b1ddb : tensor<32xf32>
    %admnb1db = stablehlo.add %admsb1db, %admgb1db : tensor<32xf32>
    %adb2b1db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b1db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb1db = stablehlo.multiply %adb2b1db, %b1dbv : tensor<32xf32>
    %adg2b1db = stablehlo.multiply %b1ddb, %b1ddb : tensor<32xf32>
    %advgb1db = stablehlo.multiply %adob2b1db, %adg2b1db : tensor<32xf32>
    %advnb1db = stablehlo.add %advsb1db, %advgb1db : tensor<32xf32>
    %adbc1b1db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b1db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb1db = stablehlo.divide %admnb1db, %adbc1b1db : tensor<32xf32>
    %advhb1db = stablehlo.divide %advnb1db, %adbc2b1db : tensor<32xf32>
    %adlrb1db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb1db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb1db = stablehlo.sqrt %advhb1db : tensor<32xf32>
    %addenb1db = stablehlo.add %adsqb1db, %adepsb1db : tensor<32xf32>
    %adratb1db = stablehlo.divide %admhb1db, %addenb1db : tensor<32xf32>
    %adstb1db = stablehlo.multiply %adlrb1db, %adratb1db : tensor<32xf32>
    %adsubb1db = stablehlo.subtract %b1db, %adstb1db : tensor<32xf32>
    %adwdb1db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb1db = stablehlo.multiply %adwdb1db, %adlrb1db : tensor<32xf32>
    %adwdpb1db = stablehlo.multiply %adwdlrb1db, %b1db : tensor<32xf32>
    %adnewb1db = stablehlo.subtract %adsubb1db, %adwdpb1db : tensor<32xf32>
    %adb1b1dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b1dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb1dg = stablehlo.multiply %adb1b1dg, %b1dgm : tensor<32xf32>
    %admgb1dg = stablehlo.multiply %adob1b1dg, %b1ddndg : tensor<32xf32>
    %admnb1dg = stablehlo.add %admsb1dg, %admgb1dg : tensor<32xf32>
    %adb2b1dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b1dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb1dg = stablehlo.multiply %adb2b1dg, %b1dgv : tensor<32xf32>
    %adg2b1dg = stablehlo.multiply %b1ddndg, %b1ddndg : tensor<32xf32>
    %advgb1dg = stablehlo.multiply %adob2b1dg, %adg2b1dg : tensor<32xf32>
    %advnb1dg = stablehlo.add %advsb1dg, %advgb1dg : tensor<32xf32>
    %adbc1b1dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b1dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb1dg = stablehlo.divide %admnb1dg, %adbc1b1dg : tensor<32xf32>
    %advhb1dg = stablehlo.divide %advnb1dg, %adbc2b1dg : tensor<32xf32>
    %adlrb1dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb1dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb1dg = stablehlo.sqrt %advhb1dg : tensor<32xf32>
    %addenb1dg = stablehlo.add %adsqb1dg, %adepsb1dg : tensor<32xf32>
    %adratb1dg = stablehlo.divide %admhb1dg, %addenb1dg : tensor<32xf32>
    %adstb1dg = stablehlo.multiply %adlrb1dg, %adratb1dg : tensor<32xf32>
    %adsubb1dg = stablehlo.subtract %b1dg, %adstb1dg : tensor<32xf32>
    %adwdb1dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb1dg = stablehlo.multiply %adwdb1dg, %adlrb1dg : tensor<32xf32>
    %adwdpb1dg = stablehlo.multiply %adwdlrb1dg, %b1dg : tensor<32xf32>
    %adnewb1dg = stablehlo.subtract %adsubb1dg, %adwdpb1dg : tensor<32xf32>
    %adb1b1dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b1dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb1dbt = stablehlo.multiply %adb1b1dbt, %b1dbtm : tensor<32xf32>
    %admgb1dbt = stablehlo.multiply %adob1b1dbt, %b1ddndb : tensor<32xf32>
    %admnb1dbt = stablehlo.add %admsb1dbt, %admgb1dbt : tensor<32xf32>
    %adb2b1dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b1dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb1dbt = stablehlo.multiply %adb2b1dbt, %b1dbtv : tensor<32xf32>
    %adg2b1dbt = stablehlo.multiply %b1ddndb, %b1ddndb : tensor<32xf32>
    %advgb1dbt = stablehlo.multiply %adob2b1dbt, %adg2b1dbt : tensor<32xf32>
    %advnb1dbt = stablehlo.add %advsb1dbt, %advgb1dbt : tensor<32xf32>
    %adbc1b1dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b1dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb1dbt = stablehlo.divide %admnb1dbt, %adbc1b1dbt : tensor<32xf32>
    %advhb1dbt = stablehlo.divide %advnb1dbt, %adbc2b1dbt : tensor<32xf32>
    %adlrb1dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb1dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb1dbt = stablehlo.sqrt %advhb1dbt : tensor<32xf32>
    %addenb1dbt = stablehlo.add %adsqb1dbt, %adepsb1dbt : tensor<32xf32>
    %adratb1dbt = stablehlo.divide %admhb1dbt, %addenb1dbt : tensor<32xf32>
    %adstb1dbt = stablehlo.multiply %adlrb1dbt, %adratb1dbt : tensor<32xf32>
    %adsubb1dbt = stablehlo.subtract %b1dbt, %adstb1dbt : tensor<32xf32>
    %adwdb1dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb1dbt = stablehlo.multiply %adwdb1dbt, %adlrb1dbt : tensor<32xf32>
    %adwdpb1dbt = stablehlo.multiply %adwdlrb1dbt, %b1dbt : tensor<32xf32>
    %adnewb1dbt = stablehlo.subtract %adsubb1dbt, %adwdpb1dbt : tensor<32xf32>
    %adb1b1zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %adob1b1zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %admsb1zW1 = stablehlo.multiply %adb1b1zW1, %b1zW1m : tensor<32x8xf32>
    %admgb1zW1 = stablehlo.multiply %adob1b1zW1, %b1zdWs1 : tensor<32x8xf32>
    %admnb1zW1 = stablehlo.add %admsb1zW1, %admgb1zW1 : tensor<32x8xf32>
    %adb2b1zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %adob2b1zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %advsb1zW1 = stablehlo.multiply %adb2b1zW1, %b1zW1v : tensor<32x8xf32>
    %adg2b1zW1 = stablehlo.multiply %b1zdWs1, %b1zdWs1 : tensor<32x8xf32>
    %advgb1zW1 = stablehlo.multiply %adob2b1zW1, %adg2b1zW1 : tensor<32x8xf32>
    %advnb1zW1 = stablehlo.add %advsb1zW1, %advgb1zW1 : tensor<32x8xf32>
    %adbc1b1zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %adbc2b1zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %admhb1zW1 = stablehlo.divide %admnb1zW1, %adbc1b1zW1 : tensor<32x8xf32>
    %advhb1zW1 = stablehlo.divide %advnb1zW1, %adbc2b1zW1 : tensor<32x8xf32>
    %adlrb1zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %adepsb1zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %adsqb1zW1 = stablehlo.sqrt %advhb1zW1 : tensor<32x8xf32>
    %addenb1zW1 = stablehlo.add %adsqb1zW1, %adepsb1zW1 : tensor<32x8xf32>
    %adratb1zW1 = stablehlo.divide %admhb1zW1, %addenb1zW1 : tensor<32x8xf32>
    %adstb1zW1 = stablehlo.multiply %adlrb1zW1, %adratb1zW1 : tensor<32x8xf32>
    %adsubb1zW1 = stablehlo.subtract %b1zW1, %adstb1zW1 : tensor<32x8xf32>
    %adwdb1zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %adwdlrb1zW1 = stablehlo.multiply %adwdb1zW1, %adlrb1zW1 : tensor<32x8xf32>
    %adwdpb1zW1 = stablehlo.multiply %adwdlrb1zW1, %b1zW1 : tensor<32x8xf32>
    %adnewb1zW1 = stablehlo.subtract %adsubb1zW1, %adwdpb1zW1 : tensor<32x8xf32>
    %adb1b1zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %adob1b1zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %admsb1zb1 = stablehlo.multiply %adb1b1zb1, %b1zb1m : tensor<8xf32>
    %admgb1zb1 = stablehlo.multiply %adob1b1zb1, %b1zdbs1 : tensor<8xf32>
    %admnb1zb1 = stablehlo.add %admsb1zb1, %admgb1zb1 : tensor<8xf32>
    %adb2b1zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %adob2b1zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %advsb1zb1 = stablehlo.multiply %adb2b1zb1, %b1zb1v : tensor<8xf32>
    %adg2b1zb1 = stablehlo.multiply %b1zdbs1, %b1zdbs1 : tensor<8xf32>
    %advgb1zb1 = stablehlo.multiply %adob2b1zb1, %adg2b1zb1 : tensor<8xf32>
    %advnb1zb1 = stablehlo.add %advsb1zb1, %advgb1zb1 : tensor<8xf32>
    %adbc1b1zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %adbc2b1zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %admhb1zb1 = stablehlo.divide %admnb1zb1, %adbc1b1zb1 : tensor<8xf32>
    %advhb1zb1 = stablehlo.divide %advnb1zb1, %adbc2b1zb1 : tensor<8xf32>
    %adlrb1zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %adepsb1zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %adsqb1zb1 = stablehlo.sqrt %advhb1zb1 : tensor<8xf32>
    %addenb1zb1 = stablehlo.add %adsqb1zb1, %adepsb1zb1 : tensor<8xf32>
    %adratb1zb1 = stablehlo.divide %admhb1zb1, %addenb1zb1 : tensor<8xf32>
    %adstb1zb1 = stablehlo.multiply %adlrb1zb1, %adratb1zb1 : tensor<8xf32>
    %adsubb1zb1 = stablehlo.subtract %b1zb1, %adstb1zb1 : tensor<8xf32>
    %adwdb1zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %adwdlrb1zb1 = stablehlo.multiply %adwdb1zb1, %adlrb1zb1 : tensor<8xf32>
    %adwdpb1zb1 = stablehlo.multiply %adwdlrb1zb1, %b1zb1 : tensor<8xf32>
    %adnewb1zb1 = stablehlo.subtract %adsubb1zb1, %adwdpb1zb1 : tensor<8xf32>
    %adb1b1zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %adob1b1zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %admsb1zW2 = stablehlo.multiply %adb1b1zW2, %b1zW2m : tensor<8x32xf32>
    %admgb1zW2 = stablehlo.multiply %adob1b1zW2, %b1zdWs2 : tensor<8x32xf32>
    %admnb1zW2 = stablehlo.add %admsb1zW2, %admgb1zW2 : tensor<8x32xf32>
    %adb2b1zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %adob2b1zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %advsb1zW2 = stablehlo.multiply %adb2b1zW2, %b1zW2v : tensor<8x32xf32>
    %adg2b1zW2 = stablehlo.multiply %b1zdWs2, %b1zdWs2 : tensor<8x32xf32>
    %advgb1zW2 = stablehlo.multiply %adob2b1zW2, %adg2b1zW2 : tensor<8x32xf32>
    %advnb1zW2 = stablehlo.add %advsb1zW2, %advgb1zW2 : tensor<8x32xf32>
    %adbc1b1zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %adbc2b1zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %admhb1zW2 = stablehlo.divide %admnb1zW2, %adbc1b1zW2 : tensor<8x32xf32>
    %advhb1zW2 = stablehlo.divide %advnb1zW2, %adbc2b1zW2 : tensor<8x32xf32>
    %adlrb1zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %adepsb1zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %adsqb1zW2 = stablehlo.sqrt %advhb1zW2 : tensor<8x32xf32>
    %addenb1zW2 = stablehlo.add %adsqb1zW2, %adepsb1zW2 : tensor<8x32xf32>
    %adratb1zW2 = stablehlo.divide %admhb1zW2, %addenb1zW2 : tensor<8x32xf32>
    %adstb1zW2 = stablehlo.multiply %adlrb1zW2, %adratb1zW2 : tensor<8x32xf32>
    %adsubb1zW2 = stablehlo.subtract %b1zW2, %adstb1zW2 : tensor<8x32xf32>
    %adwdb1zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<8x32xf32>
    %adwdlrb1zW2 = stablehlo.multiply %adwdb1zW2, %adlrb1zW2 : tensor<8x32xf32>
    %adwdpb1zW2 = stablehlo.multiply %adwdlrb1zW2, %b1zW2 : tensor<8x32xf32>
    %adnewb1zW2 = stablehlo.subtract %adsubb1zW2, %adwdpb1zW2 : tensor<8x32xf32>
    %adb1b1zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b1zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb1zb2 = stablehlo.multiply %adb1b1zb2, %b1zb2m : tensor<32xf32>
    %admgb1zb2 = stablehlo.multiply %adob1b1zb2, %b1zdbs2 : tensor<32xf32>
    %admnb1zb2 = stablehlo.add %admsb1zb2, %admgb1zb2 : tensor<32xf32>
    %adb2b1zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b1zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb1zb2 = stablehlo.multiply %adb2b1zb2, %b1zb2v : tensor<32xf32>
    %adg2b1zb2 = stablehlo.multiply %b1zdbs2, %b1zdbs2 : tensor<32xf32>
    %advgb1zb2 = stablehlo.multiply %adob2b1zb2, %adg2b1zb2 : tensor<32xf32>
    %advnb1zb2 = stablehlo.add %advsb1zb2, %advgb1zb2 : tensor<32xf32>
    %adbc1b1zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b1zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb1zb2 = stablehlo.divide %admnb1zb2, %adbc1b1zb2 : tensor<32xf32>
    %advhb1zb2 = stablehlo.divide %advnb1zb2, %adbc2b1zb2 : tensor<32xf32>
    %adlrb1zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb1zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb1zb2 = stablehlo.sqrt %advhb1zb2 : tensor<32xf32>
    %addenb1zb2 = stablehlo.add %adsqb1zb2, %adepsb1zb2 : tensor<32xf32>
    %adratb1zb2 = stablehlo.divide %admhb1zb2, %addenb1zb2 : tensor<32xf32>
    %adstb1zb2 = stablehlo.multiply %adlrb1zb2, %adratb1zb2 : tensor<32xf32>
    %adsubb1zb2 = stablehlo.subtract %b1zb2, %adstb1zb2 : tensor<32xf32>
    %adwdb1zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb1zb2 = stablehlo.multiply %adwdb1zb2, %adlrb1zb2 : tensor<32xf32>
    %adwdpb1zb2 = stablehlo.multiply %adwdlrb1zb2, %b1zb2 : tensor<32xf32>
    %adnewb1zb2 = stablehlo.subtract %adsubb1zb2, %adwdpb1zb2 : tensor<32xf32>
    %adb1b1pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %adob1b1pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %admsb1pW = stablehlo.multiply %adb1b1pW, %b1pWm : tensor<16x32x1x1xf32>
    %admgb1pW = stablehlo.multiply %adob1b1pW, %b1dpW : tensor<16x32x1x1xf32>
    %admnb1pW = stablehlo.add %admsb1pW, %admgb1pW : tensor<16x32x1x1xf32>
    %adb2b1pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %adob2b1pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %advsb1pW = stablehlo.multiply %adb2b1pW, %b1pWv : tensor<16x32x1x1xf32>
    %adg2b1pW = stablehlo.multiply %b1dpW, %b1dpW : tensor<16x32x1x1xf32>
    %advgb1pW = stablehlo.multiply %adob2b1pW, %adg2b1pW : tensor<16x32x1x1xf32>
    %advnb1pW = stablehlo.add %advsb1pW, %advgb1pW : tensor<16x32x1x1xf32>
    %adbc1b1pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %adbc2b1pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %admhb1pW = stablehlo.divide %admnb1pW, %adbc1b1pW : tensor<16x32x1x1xf32>
    %advhb1pW = stablehlo.divide %advnb1pW, %adbc2b1pW : tensor<16x32x1x1xf32>
    %adlrb1pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %adepsb1pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %adsqb1pW = stablehlo.sqrt %advhb1pW : tensor<16x32x1x1xf32>
    %addenb1pW = stablehlo.add %adsqb1pW, %adepsb1pW : tensor<16x32x1x1xf32>
    %adratb1pW = stablehlo.divide %admhb1pW, %addenb1pW : tensor<16x32x1x1xf32>
    %adstb1pW = stablehlo.multiply %adlrb1pW, %adratb1pW : tensor<16x32x1x1xf32>
    %adsubb1pW = stablehlo.subtract %b1pW, %adstb1pW : tensor<16x32x1x1xf32>
    %adwdb1pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %adwdlrb1pW = stablehlo.multiply %adwdb1pW, %adlrb1pW : tensor<16x32x1x1xf32>
    %adwdpb1pW = stablehlo.multiply %adwdlrb1pW, %b1pW : tensor<16x32x1x1xf32>
    %adnewb1pW = stablehlo.subtract %adsubb1pW, %adwdpb1pW : tensor<16x32x1x1xf32>
    %adb1b1pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1b1pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsb1pb = stablehlo.multiply %adb1b1pb, %b1pbm : tensor<16xf32>
    %admgb1pb = stablehlo.multiply %adob1b1pb, %b1dpb : tensor<16xf32>
    %admnb1pb = stablehlo.add %admsb1pb, %admgb1pb : tensor<16xf32>
    %adb2b1pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2b1pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsb1pb = stablehlo.multiply %adb2b1pb, %b1pbv : tensor<16xf32>
    %adg2b1pb = stablehlo.multiply %b1dpb, %b1dpb : tensor<16xf32>
    %advgb1pb = stablehlo.multiply %adob2b1pb, %adg2b1pb : tensor<16xf32>
    %advnb1pb = stablehlo.add %advsb1pb, %advgb1pb : tensor<16xf32>
    %adbc1b1pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2b1pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhb1pb = stablehlo.divide %admnb1pb, %adbc1b1pb : tensor<16xf32>
    %advhb1pb = stablehlo.divide %advnb1pb, %adbc2b1pb : tensor<16xf32>
    %adlrb1pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsb1pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqb1pb = stablehlo.sqrt %advhb1pb : tensor<16xf32>
    %addenb1pb = stablehlo.add %adsqb1pb, %adepsb1pb : tensor<16xf32>
    %adratb1pb = stablehlo.divide %admhb1pb, %addenb1pb : tensor<16xf32>
    %adstb1pb = stablehlo.multiply %adlrb1pb, %adratb1pb : tensor<16xf32>
    %adsubb1pb = stablehlo.subtract %b1pb, %adstb1pb : tensor<16xf32>
    %adwdb1pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrb1pb = stablehlo.multiply %adwdb1pb, %adlrb1pb : tensor<16xf32>
    %adwdpb1pb = stablehlo.multiply %adwdlrb1pb, %b1pb : tensor<16xf32>
    %adnewb1pb = stablehlo.subtract %adsubb1pb, %adwdpb1pb : tensor<16xf32>
    %adb1b1pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1b1pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsb1pg = stablehlo.multiply %adb1b1pg, %b1pgm : tensor<16xf32>
    %admgb1pg = stablehlo.multiply %adob1b1pg, %b1dpndg : tensor<16xf32>
    %admnb1pg = stablehlo.add %admsb1pg, %admgb1pg : tensor<16xf32>
    %adb2b1pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2b1pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsb1pg = stablehlo.multiply %adb2b1pg, %b1pgv : tensor<16xf32>
    %adg2b1pg = stablehlo.multiply %b1dpndg, %b1dpndg : tensor<16xf32>
    %advgb1pg = stablehlo.multiply %adob2b1pg, %adg2b1pg : tensor<16xf32>
    %advnb1pg = stablehlo.add %advsb1pg, %advgb1pg : tensor<16xf32>
    %adbc1b1pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2b1pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhb1pg = stablehlo.divide %admnb1pg, %adbc1b1pg : tensor<16xf32>
    %advhb1pg = stablehlo.divide %advnb1pg, %adbc2b1pg : tensor<16xf32>
    %adlrb1pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsb1pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqb1pg = stablehlo.sqrt %advhb1pg : tensor<16xf32>
    %addenb1pg = stablehlo.add %adsqb1pg, %adepsb1pg : tensor<16xf32>
    %adratb1pg = stablehlo.divide %admhb1pg, %addenb1pg : tensor<16xf32>
    %adstb1pg = stablehlo.multiply %adlrb1pg, %adratb1pg : tensor<16xf32>
    %adsubb1pg = stablehlo.subtract %b1pg, %adstb1pg : tensor<16xf32>
    %adwdb1pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrb1pg = stablehlo.multiply %adwdb1pg, %adlrb1pg : tensor<16xf32>
    %adwdpb1pg = stablehlo.multiply %adwdlrb1pg, %b1pg : tensor<16xf32>
    %adnewb1pg = stablehlo.subtract %adsubb1pg, %adwdpb1pg : tensor<16xf32>
    %adb1b1pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1b1pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsb1pbt = stablehlo.multiply %adb1b1pbt, %b1pbtm : tensor<16xf32>
    %admgb1pbt = stablehlo.multiply %adob1b1pbt, %b1dpndb : tensor<16xf32>
    %admnb1pbt = stablehlo.add %admsb1pbt, %admgb1pbt : tensor<16xf32>
    %adb2b1pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2b1pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsb1pbt = stablehlo.multiply %adb2b1pbt, %b1pbtv : tensor<16xf32>
    %adg2b1pbt = stablehlo.multiply %b1dpndb, %b1dpndb : tensor<16xf32>
    %advgb1pbt = stablehlo.multiply %adob2b1pbt, %adg2b1pbt : tensor<16xf32>
    %advnb1pbt = stablehlo.add %advsb1pbt, %advgb1pbt : tensor<16xf32>
    %adbc1b1pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2b1pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhb1pbt = stablehlo.divide %admnb1pbt, %adbc1b1pbt : tensor<16xf32>
    %advhb1pbt = stablehlo.divide %advnb1pbt, %adbc2b1pbt : tensor<16xf32>
    %adlrb1pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsb1pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqb1pbt = stablehlo.sqrt %advhb1pbt : tensor<16xf32>
    %addenb1pbt = stablehlo.add %adsqb1pbt, %adepsb1pbt : tensor<16xf32>
    %adratb1pbt = stablehlo.divide %admhb1pbt, %addenb1pbt : tensor<16xf32>
    %adstb1pbt = stablehlo.multiply %adlrb1pbt, %adratb1pbt : tensor<16xf32>
    %adsubb1pbt = stablehlo.subtract %b1pbt, %adstb1pbt : tensor<16xf32>
    %adwdb1pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrb1pbt = stablehlo.multiply %adwdb1pbt, %adlrb1pbt : tensor<16xf32>
    %adwdpb1pbt = stablehlo.multiply %adwdlrb1pbt, %b1pbt : tensor<16xf32>
    %adnewb1pbt = stablehlo.subtract %adsubb1pbt, %adwdpb1pbt : tensor<16xf32>
    %adb1b2eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %adob1b2eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %admsb2eW = stablehlo.multiply %adb1b2eW, %b2eWm : tensor<96x16x1x1xf32>
    %admgb2eW = stablehlo.multiply %adob1b2eW, %b2deW : tensor<96x16x1x1xf32>
    %admnb2eW = stablehlo.add %admsb2eW, %admgb2eW : tensor<96x16x1x1xf32>
    %adb2b2eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %adob2b2eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %advsb2eW = stablehlo.multiply %adb2b2eW, %b2eWv : tensor<96x16x1x1xf32>
    %adg2b2eW = stablehlo.multiply %b2deW, %b2deW : tensor<96x16x1x1xf32>
    %advgb2eW = stablehlo.multiply %adob2b2eW, %adg2b2eW : tensor<96x16x1x1xf32>
    %advnb2eW = stablehlo.add %advsb2eW, %advgb2eW : tensor<96x16x1x1xf32>
    %adbc1b2eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %adbc2b2eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %admhb2eW = stablehlo.divide %admnb2eW, %adbc1b2eW : tensor<96x16x1x1xf32>
    %advhb2eW = stablehlo.divide %advnb2eW, %adbc2b2eW : tensor<96x16x1x1xf32>
    %adlrb2eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %adepsb2eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %adsqb2eW = stablehlo.sqrt %advhb2eW : tensor<96x16x1x1xf32>
    %addenb2eW = stablehlo.add %adsqb2eW, %adepsb2eW : tensor<96x16x1x1xf32>
    %adratb2eW = stablehlo.divide %admhb2eW, %addenb2eW : tensor<96x16x1x1xf32>
    %adstb2eW = stablehlo.multiply %adlrb2eW, %adratb2eW : tensor<96x16x1x1xf32>
    %adsubb2eW = stablehlo.subtract %b2eW, %adstb2eW : tensor<96x16x1x1xf32>
    %adwdb2eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %adwdlrb2eW = stablehlo.multiply %adwdb2eW, %adlrb2eW : tensor<96x16x1x1xf32>
    %adwdpb2eW = stablehlo.multiply %adwdlrb2eW, %b2eW : tensor<96x16x1x1xf32>
    %adnewb2eW = stablehlo.subtract %adsubb2eW, %adwdpb2eW : tensor<96x16x1x1xf32>
    %adb1b2eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2eb = stablehlo.multiply %adb1b2eb, %b2ebm : tensor<96xf32>
    %admgb2eb = stablehlo.multiply %adob1b2eb, %b2deb : tensor<96xf32>
    %admnb2eb = stablehlo.add %admsb2eb, %admgb2eb : tensor<96xf32>
    %adb2b2eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2eb = stablehlo.multiply %adb2b2eb, %b2ebv : tensor<96xf32>
    %adg2b2eb = stablehlo.multiply %b2deb, %b2deb : tensor<96xf32>
    %advgb2eb = stablehlo.multiply %adob2b2eb, %adg2b2eb : tensor<96xf32>
    %advnb2eb = stablehlo.add %advsb2eb, %advgb2eb : tensor<96xf32>
    %adbc1b2eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2eb = stablehlo.divide %admnb2eb, %adbc1b2eb : tensor<96xf32>
    %advhb2eb = stablehlo.divide %advnb2eb, %adbc2b2eb : tensor<96xf32>
    %adlrb2eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2eb = stablehlo.sqrt %advhb2eb : tensor<96xf32>
    %addenb2eb = stablehlo.add %adsqb2eb, %adepsb2eb : tensor<96xf32>
    %adratb2eb = stablehlo.divide %admhb2eb, %addenb2eb : tensor<96xf32>
    %adstb2eb = stablehlo.multiply %adlrb2eb, %adratb2eb : tensor<96xf32>
    %adsubb2eb = stablehlo.subtract %b2eb, %adstb2eb : tensor<96xf32>
    %adwdb2eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2eb = stablehlo.multiply %adwdb2eb, %adlrb2eb : tensor<96xf32>
    %adwdpb2eb = stablehlo.multiply %adwdlrb2eb, %b2eb : tensor<96xf32>
    %adnewb2eb = stablehlo.subtract %adsubb2eb, %adwdpb2eb : tensor<96xf32>
    %adb1b2eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2eg = stablehlo.multiply %adb1b2eg, %b2egm : tensor<96xf32>
    %admgb2eg = stablehlo.multiply %adob1b2eg, %b2dendg : tensor<96xf32>
    %admnb2eg = stablehlo.add %admsb2eg, %admgb2eg : tensor<96xf32>
    %adb2b2eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2eg = stablehlo.multiply %adb2b2eg, %b2egv : tensor<96xf32>
    %adg2b2eg = stablehlo.multiply %b2dendg, %b2dendg : tensor<96xf32>
    %advgb2eg = stablehlo.multiply %adob2b2eg, %adg2b2eg : tensor<96xf32>
    %advnb2eg = stablehlo.add %advsb2eg, %advgb2eg : tensor<96xf32>
    %adbc1b2eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2eg = stablehlo.divide %admnb2eg, %adbc1b2eg : tensor<96xf32>
    %advhb2eg = stablehlo.divide %advnb2eg, %adbc2b2eg : tensor<96xf32>
    %adlrb2eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2eg = stablehlo.sqrt %advhb2eg : tensor<96xf32>
    %addenb2eg = stablehlo.add %adsqb2eg, %adepsb2eg : tensor<96xf32>
    %adratb2eg = stablehlo.divide %admhb2eg, %addenb2eg : tensor<96xf32>
    %adstb2eg = stablehlo.multiply %adlrb2eg, %adratb2eg : tensor<96xf32>
    %adsubb2eg = stablehlo.subtract %b2eg, %adstb2eg : tensor<96xf32>
    %adwdb2eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2eg = stablehlo.multiply %adwdb2eg, %adlrb2eg : tensor<96xf32>
    %adwdpb2eg = stablehlo.multiply %adwdlrb2eg, %b2eg : tensor<96xf32>
    %adnewb2eg = stablehlo.subtract %adsubb2eg, %adwdpb2eg : tensor<96xf32>
    %adb1b2ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2ebt = stablehlo.multiply %adb1b2ebt, %b2ebtm : tensor<96xf32>
    %admgb2ebt = stablehlo.multiply %adob1b2ebt, %b2dendb : tensor<96xf32>
    %admnb2ebt = stablehlo.add %admsb2ebt, %admgb2ebt : tensor<96xf32>
    %adb2b2ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2ebt = stablehlo.multiply %adb2b2ebt, %b2ebtv : tensor<96xf32>
    %adg2b2ebt = stablehlo.multiply %b2dendb, %b2dendb : tensor<96xf32>
    %advgb2ebt = stablehlo.multiply %adob2b2ebt, %adg2b2ebt : tensor<96xf32>
    %advnb2ebt = stablehlo.add %advsb2ebt, %advgb2ebt : tensor<96xf32>
    %adbc1b2ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2ebt = stablehlo.divide %admnb2ebt, %adbc1b2ebt : tensor<96xf32>
    %advhb2ebt = stablehlo.divide %advnb2ebt, %adbc2b2ebt : tensor<96xf32>
    %adlrb2ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2ebt = stablehlo.sqrt %advhb2ebt : tensor<96xf32>
    %addenb2ebt = stablehlo.add %adsqb2ebt, %adepsb2ebt : tensor<96xf32>
    %adratb2ebt = stablehlo.divide %admhb2ebt, %addenb2ebt : tensor<96xf32>
    %adstb2ebt = stablehlo.multiply %adlrb2ebt, %adratb2ebt : tensor<96xf32>
    %adsubb2ebt = stablehlo.subtract %b2ebt, %adstb2ebt : tensor<96xf32>
    %adwdb2ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2ebt = stablehlo.multiply %adwdb2ebt, %adlrb2ebt : tensor<96xf32>
    %adwdpb2ebt = stablehlo.multiply %adwdlrb2ebt, %b2ebt : tensor<96xf32>
    %adnewb2ebt = stablehlo.subtract %adsubb2ebt, %adwdpb2ebt : tensor<96xf32>
    %adb1b2dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adob1b2dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %admsb2dW = stablehlo.multiply %adb1b2dW, %b2dWm : tensor<96x1x3x3xf32>
    %admgb2dW = stablehlo.multiply %adob1b2dW, %b2ddW : tensor<96x1x3x3xf32>
    %admnb2dW = stablehlo.add %admsb2dW, %admgb2dW : tensor<96x1x3x3xf32>
    %adb2b2dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adob2b2dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %advsb2dW = stablehlo.multiply %adb2b2dW, %b2dWv : tensor<96x1x3x3xf32>
    %adg2b2dW = stablehlo.multiply %b2ddW, %b2ddW : tensor<96x1x3x3xf32>
    %advgb2dW = stablehlo.multiply %adob2b2dW, %adg2b2dW : tensor<96x1x3x3xf32>
    %advnb2dW = stablehlo.add %advsb2dW, %advgb2dW : tensor<96x1x3x3xf32>
    %adbc1b2dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adbc2b2dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %admhb2dW = stablehlo.divide %admnb2dW, %adbc1b2dW : tensor<96x1x3x3xf32>
    %advhb2dW = stablehlo.divide %advnb2dW, %adbc2b2dW : tensor<96x1x3x3xf32>
    %adlrb2dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adepsb2dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adsqb2dW = stablehlo.sqrt %advhb2dW : tensor<96x1x3x3xf32>
    %addenb2dW = stablehlo.add %adsqb2dW, %adepsb2dW : tensor<96x1x3x3xf32>
    %adratb2dW = stablehlo.divide %admhb2dW, %addenb2dW : tensor<96x1x3x3xf32>
    %adstb2dW = stablehlo.multiply %adlrb2dW, %adratb2dW : tensor<96x1x3x3xf32>
    %adsubb2dW = stablehlo.subtract %b2dW, %adstb2dW : tensor<96x1x3x3xf32>
    %adwdb2dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adwdlrb2dW = stablehlo.multiply %adwdb2dW, %adlrb2dW : tensor<96x1x3x3xf32>
    %adwdpb2dW = stablehlo.multiply %adwdlrb2dW, %b2dW : tensor<96x1x3x3xf32>
    %adnewb2dW = stablehlo.subtract %adsubb2dW, %adwdpb2dW : tensor<96x1x3x3xf32>
    %adb1b2db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2db = stablehlo.multiply %adb1b2db, %b2dbm : tensor<96xf32>
    %admgb2db = stablehlo.multiply %adob1b2db, %b2ddb : tensor<96xf32>
    %admnb2db = stablehlo.add %admsb2db, %admgb2db : tensor<96xf32>
    %adb2b2db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2db = stablehlo.multiply %adb2b2db, %b2dbv : tensor<96xf32>
    %adg2b2db = stablehlo.multiply %b2ddb, %b2ddb : tensor<96xf32>
    %advgb2db = stablehlo.multiply %adob2b2db, %adg2b2db : tensor<96xf32>
    %advnb2db = stablehlo.add %advsb2db, %advgb2db : tensor<96xf32>
    %adbc1b2db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2db = stablehlo.divide %admnb2db, %adbc1b2db : tensor<96xf32>
    %advhb2db = stablehlo.divide %advnb2db, %adbc2b2db : tensor<96xf32>
    %adlrb2db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2db = stablehlo.sqrt %advhb2db : tensor<96xf32>
    %addenb2db = stablehlo.add %adsqb2db, %adepsb2db : tensor<96xf32>
    %adratb2db = stablehlo.divide %admhb2db, %addenb2db : tensor<96xf32>
    %adstb2db = stablehlo.multiply %adlrb2db, %adratb2db : tensor<96xf32>
    %adsubb2db = stablehlo.subtract %b2db, %adstb2db : tensor<96xf32>
    %adwdb2db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2db = stablehlo.multiply %adwdb2db, %adlrb2db : tensor<96xf32>
    %adwdpb2db = stablehlo.multiply %adwdlrb2db, %b2db : tensor<96xf32>
    %adnewb2db = stablehlo.subtract %adsubb2db, %adwdpb2db : tensor<96xf32>
    %adb1b2dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2dg = stablehlo.multiply %adb1b2dg, %b2dgm : tensor<96xf32>
    %admgb2dg = stablehlo.multiply %adob1b2dg, %b2ddndg : tensor<96xf32>
    %admnb2dg = stablehlo.add %admsb2dg, %admgb2dg : tensor<96xf32>
    %adb2b2dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2dg = stablehlo.multiply %adb2b2dg, %b2dgv : tensor<96xf32>
    %adg2b2dg = stablehlo.multiply %b2ddndg, %b2ddndg : tensor<96xf32>
    %advgb2dg = stablehlo.multiply %adob2b2dg, %adg2b2dg : tensor<96xf32>
    %advnb2dg = stablehlo.add %advsb2dg, %advgb2dg : tensor<96xf32>
    %adbc1b2dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2dg = stablehlo.divide %admnb2dg, %adbc1b2dg : tensor<96xf32>
    %advhb2dg = stablehlo.divide %advnb2dg, %adbc2b2dg : tensor<96xf32>
    %adlrb2dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2dg = stablehlo.sqrt %advhb2dg : tensor<96xf32>
    %addenb2dg = stablehlo.add %adsqb2dg, %adepsb2dg : tensor<96xf32>
    %adratb2dg = stablehlo.divide %admhb2dg, %addenb2dg : tensor<96xf32>
    %adstb2dg = stablehlo.multiply %adlrb2dg, %adratb2dg : tensor<96xf32>
    %adsubb2dg = stablehlo.subtract %b2dg, %adstb2dg : tensor<96xf32>
    %adwdb2dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2dg = stablehlo.multiply %adwdb2dg, %adlrb2dg : tensor<96xf32>
    %adwdpb2dg = stablehlo.multiply %adwdlrb2dg, %b2dg : tensor<96xf32>
    %adnewb2dg = stablehlo.subtract %adsubb2dg, %adwdpb2dg : tensor<96xf32>
    %adb1b2dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2dbt = stablehlo.multiply %adb1b2dbt, %b2dbtm : tensor<96xf32>
    %admgb2dbt = stablehlo.multiply %adob1b2dbt, %b2ddndb : tensor<96xf32>
    %admnb2dbt = stablehlo.add %admsb2dbt, %admgb2dbt : tensor<96xf32>
    %adb2b2dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2dbt = stablehlo.multiply %adb2b2dbt, %b2dbtv : tensor<96xf32>
    %adg2b2dbt = stablehlo.multiply %b2ddndb, %b2ddndb : tensor<96xf32>
    %advgb2dbt = stablehlo.multiply %adob2b2dbt, %adg2b2dbt : tensor<96xf32>
    %advnb2dbt = stablehlo.add %advsb2dbt, %advgb2dbt : tensor<96xf32>
    %adbc1b2dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2dbt = stablehlo.divide %admnb2dbt, %adbc1b2dbt : tensor<96xf32>
    %advhb2dbt = stablehlo.divide %advnb2dbt, %adbc2b2dbt : tensor<96xf32>
    %adlrb2dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2dbt = stablehlo.sqrt %advhb2dbt : tensor<96xf32>
    %addenb2dbt = stablehlo.add %adsqb2dbt, %adepsb2dbt : tensor<96xf32>
    %adratb2dbt = stablehlo.divide %admhb2dbt, %addenb2dbt : tensor<96xf32>
    %adstb2dbt = stablehlo.multiply %adlrb2dbt, %adratb2dbt : tensor<96xf32>
    %adsubb2dbt = stablehlo.subtract %b2dbt, %adstb2dbt : tensor<96xf32>
    %adwdb2dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2dbt = stablehlo.multiply %adwdb2dbt, %adlrb2dbt : tensor<96xf32>
    %adwdpb2dbt = stablehlo.multiply %adwdlrb2dbt, %b2dbt : tensor<96xf32>
    %adnewb2dbt = stablehlo.subtract %adsubb2dbt, %adwdpb2dbt : tensor<96xf32>
    %adb1b2zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %adob1b2zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %admsb2zW1 = stablehlo.multiply %adb1b2zW1, %b2zW1m : tensor<96x4xf32>
    %admgb2zW1 = stablehlo.multiply %adob1b2zW1, %b2zdWs1 : tensor<96x4xf32>
    %admnb2zW1 = stablehlo.add %admsb2zW1, %admgb2zW1 : tensor<96x4xf32>
    %adb2b2zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %adob2b2zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %advsb2zW1 = stablehlo.multiply %adb2b2zW1, %b2zW1v : tensor<96x4xf32>
    %adg2b2zW1 = stablehlo.multiply %b2zdWs1, %b2zdWs1 : tensor<96x4xf32>
    %advgb2zW1 = stablehlo.multiply %adob2b2zW1, %adg2b2zW1 : tensor<96x4xf32>
    %advnb2zW1 = stablehlo.add %advsb2zW1, %advgb2zW1 : tensor<96x4xf32>
    %adbc1b2zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %adbc2b2zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %admhb2zW1 = stablehlo.divide %admnb2zW1, %adbc1b2zW1 : tensor<96x4xf32>
    %advhb2zW1 = stablehlo.divide %advnb2zW1, %adbc2b2zW1 : tensor<96x4xf32>
    %adlrb2zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %adepsb2zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %adsqb2zW1 = stablehlo.sqrt %advhb2zW1 : tensor<96x4xf32>
    %addenb2zW1 = stablehlo.add %adsqb2zW1, %adepsb2zW1 : tensor<96x4xf32>
    %adratb2zW1 = stablehlo.divide %admhb2zW1, %addenb2zW1 : tensor<96x4xf32>
    %adstb2zW1 = stablehlo.multiply %adlrb2zW1, %adratb2zW1 : tensor<96x4xf32>
    %adsubb2zW1 = stablehlo.subtract %b2zW1, %adstb2zW1 : tensor<96x4xf32>
    %adwdb2zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x4xf32>
    %adwdlrb2zW1 = stablehlo.multiply %adwdb2zW1, %adlrb2zW1 : tensor<96x4xf32>
    %adwdpb2zW1 = stablehlo.multiply %adwdlrb2zW1, %b2zW1 : tensor<96x4xf32>
    %adnewb2zW1 = stablehlo.subtract %adsubb2zW1, %adwdpb2zW1 : tensor<96x4xf32>
    %adb1b2zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %adob1b2zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %admsb2zb1 = stablehlo.multiply %adb1b2zb1, %b2zb1m : tensor<4xf32>
    %admgb2zb1 = stablehlo.multiply %adob1b2zb1, %b2zdbs1 : tensor<4xf32>
    %admnb2zb1 = stablehlo.add %admsb2zb1, %admgb2zb1 : tensor<4xf32>
    %adb2b2zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %adob2b2zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %advsb2zb1 = stablehlo.multiply %adb2b2zb1, %b2zb1v : tensor<4xf32>
    %adg2b2zb1 = stablehlo.multiply %b2zdbs1, %b2zdbs1 : tensor<4xf32>
    %advgb2zb1 = stablehlo.multiply %adob2b2zb1, %adg2b2zb1 : tensor<4xf32>
    %advnb2zb1 = stablehlo.add %advsb2zb1, %advgb2zb1 : tensor<4xf32>
    %adbc1b2zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %adbc2b2zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %admhb2zb1 = stablehlo.divide %admnb2zb1, %adbc1b2zb1 : tensor<4xf32>
    %advhb2zb1 = stablehlo.divide %advnb2zb1, %adbc2b2zb1 : tensor<4xf32>
    %adlrb2zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %adepsb2zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %adsqb2zb1 = stablehlo.sqrt %advhb2zb1 : tensor<4xf32>
    %addenb2zb1 = stablehlo.add %adsqb2zb1, %adepsb2zb1 : tensor<4xf32>
    %adratb2zb1 = stablehlo.divide %admhb2zb1, %addenb2zb1 : tensor<4xf32>
    %adstb2zb1 = stablehlo.multiply %adlrb2zb1, %adratb2zb1 : tensor<4xf32>
    %adsubb2zb1 = stablehlo.subtract %b2zb1, %adstb2zb1 : tensor<4xf32>
    %adwdb2zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %adwdlrb2zb1 = stablehlo.multiply %adwdb2zb1, %adlrb2zb1 : tensor<4xf32>
    %adwdpb2zb1 = stablehlo.multiply %adwdlrb2zb1, %b2zb1 : tensor<4xf32>
    %adnewb2zb1 = stablehlo.subtract %adsubb2zb1, %adwdpb2zb1 : tensor<4xf32>
    %adb1b2zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %adob1b2zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %admsb2zW2 = stablehlo.multiply %adb1b2zW2, %b2zW2m : tensor<4x96xf32>
    %admgb2zW2 = stablehlo.multiply %adob1b2zW2, %b2zdWs2 : tensor<4x96xf32>
    %admnb2zW2 = stablehlo.add %admsb2zW2, %admgb2zW2 : tensor<4x96xf32>
    %adb2b2zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %adob2b2zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %advsb2zW2 = stablehlo.multiply %adb2b2zW2, %b2zW2v : tensor<4x96xf32>
    %adg2b2zW2 = stablehlo.multiply %b2zdWs2, %b2zdWs2 : tensor<4x96xf32>
    %advgb2zW2 = stablehlo.multiply %adob2b2zW2, %adg2b2zW2 : tensor<4x96xf32>
    %advnb2zW2 = stablehlo.add %advsb2zW2, %advgb2zW2 : tensor<4x96xf32>
    %adbc1b2zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %adbc2b2zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %admhb2zW2 = stablehlo.divide %admnb2zW2, %adbc1b2zW2 : tensor<4x96xf32>
    %advhb2zW2 = stablehlo.divide %advnb2zW2, %adbc2b2zW2 : tensor<4x96xf32>
    %adlrb2zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %adepsb2zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %adsqb2zW2 = stablehlo.sqrt %advhb2zW2 : tensor<4x96xf32>
    %addenb2zW2 = stablehlo.add %adsqb2zW2, %adepsb2zW2 : tensor<4x96xf32>
    %adratb2zW2 = stablehlo.divide %admhb2zW2, %addenb2zW2 : tensor<4x96xf32>
    %adstb2zW2 = stablehlo.multiply %adlrb2zW2, %adratb2zW2 : tensor<4x96xf32>
    %adsubb2zW2 = stablehlo.subtract %b2zW2, %adstb2zW2 : tensor<4x96xf32>
    %adwdb2zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<4x96xf32>
    %adwdlrb2zW2 = stablehlo.multiply %adwdb2zW2, %adlrb2zW2 : tensor<4x96xf32>
    %adwdpb2zW2 = stablehlo.multiply %adwdlrb2zW2, %b2zW2 : tensor<4x96xf32>
    %adnewb2zW2 = stablehlo.subtract %adsubb2zW2, %adwdpb2zW2 : tensor<4x96xf32>
    %adb1b2zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2zb2 = stablehlo.multiply %adb1b2zb2, %b2zb2m : tensor<96xf32>
    %admgb2zb2 = stablehlo.multiply %adob1b2zb2, %b2zdbs2 : tensor<96xf32>
    %admnb2zb2 = stablehlo.add %admsb2zb2, %admgb2zb2 : tensor<96xf32>
    %adb2b2zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2zb2 = stablehlo.multiply %adb2b2zb2, %b2zb2v : tensor<96xf32>
    %adg2b2zb2 = stablehlo.multiply %b2zdbs2, %b2zdbs2 : tensor<96xf32>
    %advgb2zb2 = stablehlo.multiply %adob2b2zb2, %adg2b2zb2 : tensor<96xf32>
    %advnb2zb2 = stablehlo.add %advsb2zb2, %advgb2zb2 : tensor<96xf32>
    %adbc1b2zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2zb2 = stablehlo.divide %admnb2zb2, %adbc1b2zb2 : tensor<96xf32>
    %advhb2zb2 = stablehlo.divide %advnb2zb2, %adbc2b2zb2 : tensor<96xf32>
    %adlrb2zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2zb2 = stablehlo.sqrt %advhb2zb2 : tensor<96xf32>
    %addenb2zb2 = stablehlo.add %adsqb2zb2, %adepsb2zb2 : tensor<96xf32>
    %adratb2zb2 = stablehlo.divide %admhb2zb2, %addenb2zb2 : tensor<96xf32>
    %adstb2zb2 = stablehlo.multiply %adlrb2zb2, %adratb2zb2 : tensor<96xf32>
    %adsubb2zb2 = stablehlo.subtract %b2zb2, %adstb2zb2 : tensor<96xf32>
    %adwdb2zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2zb2 = stablehlo.multiply %adwdb2zb2, %adlrb2zb2 : tensor<96xf32>
    %adwdpb2zb2 = stablehlo.multiply %adwdlrb2zb2, %b2zb2 : tensor<96xf32>
    %adnewb2zb2 = stablehlo.subtract %adsubb2zb2, %adwdpb2zb2 : tensor<96xf32>
    %adb1b2pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adob1b2pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %admsb2pW = stablehlo.multiply %adb1b2pW, %b2pWm : tensor<24x96x1x1xf32>
    %admgb2pW = stablehlo.multiply %adob1b2pW, %b2dpW : tensor<24x96x1x1xf32>
    %admnb2pW = stablehlo.add %admsb2pW, %admgb2pW : tensor<24x96x1x1xf32>
    %adb2b2pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adob2b2pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %advsb2pW = stablehlo.multiply %adb2b2pW, %b2pWv : tensor<24x96x1x1xf32>
    %adg2b2pW = stablehlo.multiply %b2dpW, %b2dpW : tensor<24x96x1x1xf32>
    %advgb2pW = stablehlo.multiply %adob2b2pW, %adg2b2pW : tensor<24x96x1x1xf32>
    %advnb2pW = stablehlo.add %advsb2pW, %advgb2pW : tensor<24x96x1x1xf32>
    %adbc1b2pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adbc2b2pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %admhb2pW = stablehlo.divide %admnb2pW, %adbc1b2pW : tensor<24x96x1x1xf32>
    %advhb2pW = stablehlo.divide %advnb2pW, %adbc2b2pW : tensor<24x96x1x1xf32>
    %adlrb2pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adepsb2pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adsqb2pW = stablehlo.sqrt %advhb2pW : tensor<24x96x1x1xf32>
    %addenb2pW = stablehlo.add %adsqb2pW, %adepsb2pW : tensor<24x96x1x1xf32>
    %adratb2pW = stablehlo.divide %admhb2pW, %addenb2pW : tensor<24x96x1x1xf32>
    %adstb2pW = stablehlo.multiply %adlrb2pW, %adratb2pW : tensor<24x96x1x1xf32>
    %adsubb2pW = stablehlo.subtract %b2pW, %adstb2pW : tensor<24x96x1x1xf32>
    %adwdb2pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adwdlrb2pW = stablehlo.multiply %adwdb2pW, %adlrb2pW : tensor<24x96x1x1xf32>
    %adwdpb2pW = stablehlo.multiply %adwdlrb2pW, %b2pW : tensor<24x96x1x1xf32>
    %adnewb2pW = stablehlo.subtract %adsubb2pW, %adwdpb2pW : tensor<24x96x1x1xf32>
    %adb1b2pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b2pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb2pb = stablehlo.multiply %adb1b2pb, %b2pbm : tensor<24xf32>
    %admgb2pb = stablehlo.multiply %adob1b2pb, %b2dpb : tensor<24xf32>
    %admnb2pb = stablehlo.add %admsb2pb, %admgb2pb : tensor<24xf32>
    %adb2b2pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b2pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb2pb = stablehlo.multiply %adb2b2pb, %b2pbv : tensor<24xf32>
    %adg2b2pb = stablehlo.multiply %b2dpb, %b2dpb : tensor<24xf32>
    %advgb2pb = stablehlo.multiply %adob2b2pb, %adg2b2pb : tensor<24xf32>
    %advnb2pb = stablehlo.add %advsb2pb, %advgb2pb : tensor<24xf32>
    %adbc1b2pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b2pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb2pb = stablehlo.divide %admnb2pb, %adbc1b2pb : tensor<24xf32>
    %advhb2pb = stablehlo.divide %advnb2pb, %adbc2b2pb : tensor<24xf32>
    %adlrb2pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb2pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb2pb = stablehlo.sqrt %advhb2pb : tensor<24xf32>
    %addenb2pb = stablehlo.add %adsqb2pb, %adepsb2pb : tensor<24xf32>
    %adratb2pb = stablehlo.divide %admhb2pb, %addenb2pb : tensor<24xf32>
    %adstb2pb = stablehlo.multiply %adlrb2pb, %adratb2pb : tensor<24xf32>
    %adsubb2pb = stablehlo.subtract %b2pb, %adstb2pb : tensor<24xf32>
    %adwdb2pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb2pb = stablehlo.multiply %adwdb2pb, %adlrb2pb : tensor<24xf32>
    %adwdpb2pb = stablehlo.multiply %adwdlrb2pb, %b2pb : tensor<24xf32>
    %adnewb2pb = stablehlo.subtract %adsubb2pb, %adwdpb2pb : tensor<24xf32>
    %adb1b2pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b2pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb2pg = stablehlo.multiply %adb1b2pg, %b2pgm : tensor<24xf32>
    %admgb2pg = stablehlo.multiply %adob1b2pg, %b2dpndg : tensor<24xf32>
    %admnb2pg = stablehlo.add %admsb2pg, %admgb2pg : tensor<24xf32>
    %adb2b2pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b2pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb2pg = stablehlo.multiply %adb2b2pg, %b2pgv : tensor<24xf32>
    %adg2b2pg = stablehlo.multiply %b2dpndg, %b2dpndg : tensor<24xf32>
    %advgb2pg = stablehlo.multiply %adob2b2pg, %adg2b2pg : tensor<24xf32>
    %advnb2pg = stablehlo.add %advsb2pg, %advgb2pg : tensor<24xf32>
    %adbc1b2pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b2pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb2pg = stablehlo.divide %admnb2pg, %adbc1b2pg : tensor<24xf32>
    %advhb2pg = stablehlo.divide %advnb2pg, %adbc2b2pg : tensor<24xf32>
    %adlrb2pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb2pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb2pg = stablehlo.sqrt %advhb2pg : tensor<24xf32>
    %addenb2pg = stablehlo.add %adsqb2pg, %adepsb2pg : tensor<24xf32>
    %adratb2pg = stablehlo.divide %admhb2pg, %addenb2pg : tensor<24xf32>
    %adstb2pg = stablehlo.multiply %adlrb2pg, %adratb2pg : tensor<24xf32>
    %adsubb2pg = stablehlo.subtract %b2pg, %adstb2pg : tensor<24xf32>
    %adwdb2pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb2pg = stablehlo.multiply %adwdb2pg, %adlrb2pg : tensor<24xf32>
    %adwdpb2pg = stablehlo.multiply %adwdlrb2pg, %b2pg : tensor<24xf32>
    %adnewb2pg = stablehlo.subtract %adsubb2pg, %adwdpb2pg : tensor<24xf32>
    %adb1b2pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b2pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb2pbt = stablehlo.multiply %adb1b2pbt, %b2pbtm : tensor<24xf32>
    %admgb2pbt = stablehlo.multiply %adob1b2pbt, %b2dpndb : tensor<24xf32>
    %admnb2pbt = stablehlo.add %admsb2pbt, %admgb2pbt : tensor<24xf32>
    %adb2b2pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b2pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb2pbt = stablehlo.multiply %adb2b2pbt, %b2pbtv : tensor<24xf32>
    %adg2b2pbt = stablehlo.multiply %b2dpndb, %b2dpndb : tensor<24xf32>
    %advgb2pbt = stablehlo.multiply %adob2b2pbt, %adg2b2pbt : tensor<24xf32>
    %advnb2pbt = stablehlo.add %advsb2pbt, %advgb2pbt : tensor<24xf32>
    %adbc1b2pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b2pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb2pbt = stablehlo.divide %admnb2pbt, %adbc1b2pbt : tensor<24xf32>
    %advhb2pbt = stablehlo.divide %advnb2pbt, %adbc2b2pbt : tensor<24xf32>
    %adlrb2pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb2pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb2pbt = stablehlo.sqrt %advhb2pbt : tensor<24xf32>
    %addenb2pbt = stablehlo.add %adsqb2pbt, %adepsb2pbt : tensor<24xf32>
    %adratb2pbt = stablehlo.divide %admhb2pbt, %addenb2pbt : tensor<24xf32>
    %adstb2pbt = stablehlo.multiply %adlrb2pbt, %adratb2pbt : tensor<24xf32>
    %adsubb2pbt = stablehlo.subtract %b2pbt, %adstb2pbt : tensor<24xf32>
    %adwdb2pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb2pbt = stablehlo.multiply %adwdb2pbt, %adlrb2pbt : tensor<24xf32>
    %adwdpb2pbt = stablehlo.multiply %adwdlrb2pbt, %b2pbt : tensor<24xf32>
    %adnewb2pbt = stablehlo.subtract %adsubb2pbt, %adwdpb2pbt : tensor<24xf32>
    %adb1b3eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adob1b3eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %admsb3eW = stablehlo.multiply %adb1b3eW, %b3eWm : tensor<144x24x1x1xf32>
    %admgb3eW = stablehlo.multiply %adob1b3eW, %b3deW : tensor<144x24x1x1xf32>
    %admnb3eW = stablehlo.add %admsb3eW, %admgb3eW : tensor<144x24x1x1xf32>
    %adb2b3eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adob2b3eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %advsb3eW = stablehlo.multiply %adb2b3eW, %b3eWv : tensor<144x24x1x1xf32>
    %adg2b3eW = stablehlo.multiply %b3deW, %b3deW : tensor<144x24x1x1xf32>
    %advgb3eW = stablehlo.multiply %adob2b3eW, %adg2b3eW : tensor<144x24x1x1xf32>
    %advnb3eW = stablehlo.add %advsb3eW, %advgb3eW : tensor<144x24x1x1xf32>
    %adbc1b3eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adbc2b3eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %admhb3eW = stablehlo.divide %admnb3eW, %adbc1b3eW : tensor<144x24x1x1xf32>
    %advhb3eW = stablehlo.divide %advnb3eW, %adbc2b3eW : tensor<144x24x1x1xf32>
    %adlrb3eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adepsb3eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adsqb3eW = stablehlo.sqrt %advhb3eW : tensor<144x24x1x1xf32>
    %addenb3eW = stablehlo.add %adsqb3eW, %adepsb3eW : tensor<144x24x1x1xf32>
    %adratb3eW = stablehlo.divide %admhb3eW, %addenb3eW : tensor<144x24x1x1xf32>
    %adstb3eW = stablehlo.multiply %adlrb3eW, %adratb3eW : tensor<144x24x1x1xf32>
    %adsubb3eW = stablehlo.subtract %b3eW, %adstb3eW : tensor<144x24x1x1xf32>
    %adwdb3eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adwdlrb3eW = stablehlo.multiply %adwdb3eW, %adlrb3eW : tensor<144x24x1x1xf32>
    %adwdpb3eW = stablehlo.multiply %adwdlrb3eW, %b3eW : tensor<144x24x1x1xf32>
    %adnewb3eW = stablehlo.subtract %adsubb3eW, %adwdpb3eW : tensor<144x24x1x1xf32>
    %adb1b3eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b3eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb3eb = stablehlo.multiply %adb1b3eb, %b3ebm : tensor<144xf32>
    %admgb3eb = stablehlo.multiply %adob1b3eb, %b3deb : tensor<144xf32>
    %admnb3eb = stablehlo.add %admsb3eb, %admgb3eb : tensor<144xf32>
    %adb2b3eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b3eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb3eb = stablehlo.multiply %adb2b3eb, %b3ebv : tensor<144xf32>
    %adg2b3eb = stablehlo.multiply %b3deb, %b3deb : tensor<144xf32>
    %advgb3eb = stablehlo.multiply %adob2b3eb, %adg2b3eb : tensor<144xf32>
    %advnb3eb = stablehlo.add %advsb3eb, %advgb3eb : tensor<144xf32>
    %adbc1b3eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b3eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb3eb = stablehlo.divide %admnb3eb, %adbc1b3eb : tensor<144xf32>
    %advhb3eb = stablehlo.divide %advnb3eb, %adbc2b3eb : tensor<144xf32>
    %adlrb3eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb3eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb3eb = stablehlo.sqrt %advhb3eb : tensor<144xf32>
    %addenb3eb = stablehlo.add %adsqb3eb, %adepsb3eb : tensor<144xf32>
    %adratb3eb = stablehlo.divide %admhb3eb, %addenb3eb : tensor<144xf32>
    %adstb3eb = stablehlo.multiply %adlrb3eb, %adratb3eb : tensor<144xf32>
    %adsubb3eb = stablehlo.subtract %b3eb, %adstb3eb : tensor<144xf32>
    %adwdb3eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb3eb = stablehlo.multiply %adwdb3eb, %adlrb3eb : tensor<144xf32>
    %adwdpb3eb = stablehlo.multiply %adwdlrb3eb, %b3eb : tensor<144xf32>
    %adnewb3eb = stablehlo.subtract %adsubb3eb, %adwdpb3eb : tensor<144xf32>
    %adb1b3eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b3eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb3eg = stablehlo.multiply %adb1b3eg, %b3egm : tensor<144xf32>
    %admgb3eg = stablehlo.multiply %adob1b3eg, %b3dendg : tensor<144xf32>
    %admnb3eg = stablehlo.add %admsb3eg, %admgb3eg : tensor<144xf32>
    %adb2b3eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b3eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb3eg = stablehlo.multiply %adb2b3eg, %b3egv : tensor<144xf32>
    %adg2b3eg = stablehlo.multiply %b3dendg, %b3dendg : tensor<144xf32>
    %advgb3eg = stablehlo.multiply %adob2b3eg, %adg2b3eg : tensor<144xf32>
    %advnb3eg = stablehlo.add %advsb3eg, %advgb3eg : tensor<144xf32>
    %adbc1b3eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b3eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb3eg = stablehlo.divide %admnb3eg, %adbc1b3eg : tensor<144xf32>
    %advhb3eg = stablehlo.divide %advnb3eg, %adbc2b3eg : tensor<144xf32>
    %adlrb3eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb3eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb3eg = stablehlo.sqrt %advhb3eg : tensor<144xf32>
    %addenb3eg = stablehlo.add %adsqb3eg, %adepsb3eg : tensor<144xf32>
    %adratb3eg = stablehlo.divide %admhb3eg, %addenb3eg : tensor<144xf32>
    %adstb3eg = stablehlo.multiply %adlrb3eg, %adratb3eg : tensor<144xf32>
    %adsubb3eg = stablehlo.subtract %b3eg, %adstb3eg : tensor<144xf32>
    %adwdb3eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb3eg = stablehlo.multiply %adwdb3eg, %adlrb3eg : tensor<144xf32>
    %adwdpb3eg = stablehlo.multiply %adwdlrb3eg, %b3eg : tensor<144xf32>
    %adnewb3eg = stablehlo.subtract %adsubb3eg, %adwdpb3eg : tensor<144xf32>
    %adb1b3ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b3ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb3ebt = stablehlo.multiply %adb1b3ebt, %b3ebtm : tensor<144xf32>
    %admgb3ebt = stablehlo.multiply %adob1b3ebt, %b3dendb : tensor<144xf32>
    %admnb3ebt = stablehlo.add %admsb3ebt, %admgb3ebt : tensor<144xf32>
    %adb2b3ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b3ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb3ebt = stablehlo.multiply %adb2b3ebt, %b3ebtv : tensor<144xf32>
    %adg2b3ebt = stablehlo.multiply %b3dendb, %b3dendb : tensor<144xf32>
    %advgb3ebt = stablehlo.multiply %adob2b3ebt, %adg2b3ebt : tensor<144xf32>
    %advnb3ebt = stablehlo.add %advsb3ebt, %advgb3ebt : tensor<144xf32>
    %adbc1b3ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b3ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb3ebt = stablehlo.divide %admnb3ebt, %adbc1b3ebt : tensor<144xf32>
    %advhb3ebt = stablehlo.divide %advnb3ebt, %adbc2b3ebt : tensor<144xf32>
    %adlrb3ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb3ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb3ebt = stablehlo.sqrt %advhb3ebt : tensor<144xf32>
    %addenb3ebt = stablehlo.add %adsqb3ebt, %adepsb3ebt : tensor<144xf32>
    %adratb3ebt = stablehlo.divide %admhb3ebt, %addenb3ebt : tensor<144xf32>
    %adstb3ebt = stablehlo.multiply %adlrb3ebt, %adratb3ebt : tensor<144xf32>
    %adsubb3ebt = stablehlo.subtract %b3ebt, %adstb3ebt : tensor<144xf32>
    %adwdb3ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb3ebt = stablehlo.multiply %adwdb3ebt, %adlrb3ebt : tensor<144xf32>
    %adwdpb3ebt = stablehlo.multiply %adwdlrb3ebt, %b3ebt : tensor<144xf32>
    %adnewb3ebt = stablehlo.subtract %adsubb3ebt, %adwdpb3ebt : tensor<144xf32>
    %adb1b3dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %adob1b3dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %admsb3dW = stablehlo.multiply %adb1b3dW, %b3dWm : tensor<144x1x3x3xf32>
    %admgb3dW = stablehlo.multiply %adob1b3dW, %b3ddW : tensor<144x1x3x3xf32>
    %admnb3dW = stablehlo.add %admsb3dW, %admgb3dW : tensor<144x1x3x3xf32>
    %adb2b3dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %adob2b3dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %advsb3dW = stablehlo.multiply %adb2b3dW, %b3dWv : tensor<144x1x3x3xf32>
    %adg2b3dW = stablehlo.multiply %b3ddW, %b3ddW : tensor<144x1x3x3xf32>
    %advgb3dW = stablehlo.multiply %adob2b3dW, %adg2b3dW : tensor<144x1x3x3xf32>
    %advnb3dW = stablehlo.add %advsb3dW, %advgb3dW : tensor<144x1x3x3xf32>
    %adbc1b3dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %adbc2b3dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %admhb3dW = stablehlo.divide %admnb3dW, %adbc1b3dW : tensor<144x1x3x3xf32>
    %advhb3dW = stablehlo.divide %advnb3dW, %adbc2b3dW : tensor<144x1x3x3xf32>
    %adlrb3dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %adepsb3dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %adsqb3dW = stablehlo.sqrt %advhb3dW : tensor<144x1x3x3xf32>
    %addenb3dW = stablehlo.add %adsqb3dW, %adepsb3dW : tensor<144x1x3x3xf32>
    %adratb3dW = stablehlo.divide %admhb3dW, %addenb3dW : tensor<144x1x3x3xf32>
    %adstb3dW = stablehlo.multiply %adlrb3dW, %adratb3dW : tensor<144x1x3x3xf32>
    %adsubb3dW = stablehlo.subtract %b3dW, %adstb3dW : tensor<144x1x3x3xf32>
    %adwdb3dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %adwdlrb3dW = stablehlo.multiply %adwdb3dW, %adlrb3dW : tensor<144x1x3x3xf32>
    %adwdpb3dW = stablehlo.multiply %adwdlrb3dW, %b3dW : tensor<144x1x3x3xf32>
    %adnewb3dW = stablehlo.subtract %adsubb3dW, %adwdpb3dW : tensor<144x1x3x3xf32>
    %adb1b3db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b3db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb3db = stablehlo.multiply %adb1b3db, %b3dbm : tensor<144xf32>
    %admgb3db = stablehlo.multiply %adob1b3db, %b3ddb : tensor<144xf32>
    %admnb3db = stablehlo.add %admsb3db, %admgb3db : tensor<144xf32>
    %adb2b3db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b3db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb3db = stablehlo.multiply %adb2b3db, %b3dbv : tensor<144xf32>
    %adg2b3db = stablehlo.multiply %b3ddb, %b3ddb : tensor<144xf32>
    %advgb3db = stablehlo.multiply %adob2b3db, %adg2b3db : tensor<144xf32>
    %advnb3db = stablehlo.add %advsb3db, %advgb3db : tensor<144xf32>
    %adbc1b3db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b3db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb3db = stablehlo.divide %admnb3db, %adbc1b3db : tensor<144xf32>
    %advhb3db = stablehlo.divide %advnb3db, %adbc2b3db : tensor<144xf32>
    %adlrb3db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb3db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb3db = stablehlo.sqrt %advhb3db : tensor<144xf32>
    %addenb3db = stablehlo.add %adsqb3db, %adepsb3db : tensor<144xf32>
    %adratb3db = stablehlo.divide %admhb3db, %addenb3db : tensor<144xf32>
    %adstb3db = stablehlo.multiply %adlrb3db, %adratb3db : tensor<144xf32>
    %adsubb3db = stablehlo.subtract %b3db, %adstb3db : tensor<144xf32>
    %adwdb3db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb3db = stablehlo.multiply %adwdb3db, %adlrb3db : tensor<144xf32>
    %adwdpb3db = stablehlo.multiply %adwdlrb3db, %b3db : tensor<144xf32>
    %adnewb3db = stablehlo.subtract %adsubb3db, %adwdpb3db : tensor<144xf32>
    %adb1b3dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b3dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb3dg = stablehlo.multiply %adb1b3dg, %b3dgm : tensor<144xf32>
    %admgb3dg = stablehlo.multiply %adob1b3dg, %b3ddndg : tensor<144xf32>
    %admnb3dg = stablehlo.add %admsb3dg, %admgb3dg : tensor<144xf32>
    %adb2b3dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b3dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb3dg = stablehlo.multiply %adb2b3dg, %b3dgv : tensor<144xf32>
    %adg2b3dg = stablehlo.multiply %b3ddndg, %b3ddndg : tensor<144xf32>
    %advgb3dg = stablehlo.multiply %adob2b3dg, %adg2b3dg : tensor<144xf32>
    %advnb3dg = stablehlo.add %advsb3dg, %advgb3dg : tensor<144xf32>
    %adbc1b3dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b3dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb3dg = stablehlo.divide %admnb3dg, %adbc1b3dg : tensor<144xf32>
    %advhb3dg = stablehlo.divide %advnb3dg, %adbc2b3dg : tensor<144xf32>
    %adlrb3dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb3dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb3dg = stablehlo.sqrt %advhb3dg : tensor<144xf32>
    %addenb3dg = stablehlo.add %adsqb3dg, %adepsb3dg : tensor<144xf32>
    %adratb3dg = stablehlo.divide %admhb3dg, %addenb3dg : tensor<144xf32>
    %adstb3dg = stablehlo.multiply %adlrb3dg, %adratb3dg : tensor<144xf32>
    %adsubb3dg = stablehlo.subtract %b3dg, %adstb3dg : tensor<144xf32>
    %adwdb3dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb3dg = stablehlo.multiply %adwdb3dg, %adlrb3dg : tensor<144xf32>
    %adwdpb3dg = stablehlo.multiply %adwdlrb3dg, %b3dg : tensor<144xf32>
    %adnewb3dg = stablehlo.subtract %adsubb3dg, %adwdpb3dg : tensor<144xf32>
    %adb1b3dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b3dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb3dbt = stablehlo.multiply %adb1b3dbt, %b3dbtm : tensor<144xf32>
    %admgb3dbt = stablehlo.multiply %adob1b3dbt, %b3ddndb : tensor<144xf32>
    %admnb3dbt = stablehlo.add %admsb3dbt, %admgb3dbt : tensor<144xf32>
    %adb2b3dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b3dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb3dbt = stablehlo.multiply %adb2b3dbt, %b3dbtv : tensor<144xf32>
    %adg2b3dbt = stablehlo.multiply %b3ddndb, %b3ddndb : tensor<144xf32>
    %advgb3dbt = stablehlo.multiply %adob2b3dbt, %adg2b3dbt : tensor<144xf32>
    %advnb3dbt = stablehlo.add %advsb3dbt, %advgb3dbt : tensor<144xf32>
    %adbc1b3dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b3dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb3dbt = stablehlo.divide %admnb3dbt, %adbc1b3dbt : tensor<144xf32>
    %advhb3dbt = stablehlo.divide %advnb3dbt, %adbc2b3dbt : tensor<144xf32>
    %adlrb3dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb3dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb3dbt = stablehlo.sqrt %advhb3dbt : tensor<144xf32>
    %addenb3dbt = stablehlo.add %adsqb3dbt, %adepsb3dbt : tensor<144xf32>
    %adratb3dbt = stablehlo.divide %admhb3dbt, %addenb3dbt : tensor<144xf32>
    %adstb3dbt = stablehlo.multiply %adlrb3dbt, %adratb3dbt : tensor<144xf32>
    %adsubb3dbt = stablehlo.subtract %b3dbt, %adstb3dbt : tensor<144xf32>
    %adwdb3dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb3dbt = stablehlo.multiply %adwdb3dbt, %adlrb3dbt : tensor<144xf32>
    %adwdpb3dbt = stablehlo.multiply %adwdlrb3dbt, %b3dbt : tensor<144xf32>
    %adnewb3dbt = stablehlo.subtract %adsubb3dbt, %adwdpb3dbt : tensor<144xf32>
    %adb1b3zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adob1b3zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %admsb3zW1 = stablehlo.multiply %adb1b3zW1, %b3zW1m : tensor<144x6xf32>
    %admgb3zW1 = stablehlo.multiply %adob1b3zW1, %b3zdWs1 : tensor<144x6xf32>
    %admnb3zW1 = stablehlo.add %admsb3zW1, %admgb3zW1 : tensor<144x6xf32>
    %adb2b3zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adob2b3zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %advsb3zW1 = stablehlo.multiply %adb2b3zW1, %b3zW1v : tensor<144x6xf32>
    %adg2b3zW1 = stablehlo.multiply %b3zdWs1, %b3zdWs1 : tensor<144x6xf32>
    %advgb3zW1 = stablehlo.multiply %adob2b3zW1, %adg2b3zW1 : tensor<144x6xf32>
    %advnb3zW1 = stablehlo.add %advsb3zW1, %advgb3zW1 : tensor<144x6xf32>
    %adbc1b3zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adbc2b3zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %admhb3zW1 = stablehlo.divide %admnb3zW1, %adbc1b3zW1 : tensor<144x6xf32>
    %advhb3zW1 = stablehlo.divide %advnb3zW1, %adbc2b3zW1 : tensor<144x6xf32>
    %adlrb3zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adepsb3zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adsqb3zW1 = stablehlo.sqrt %advhb3zW1 : tensor<144x6xf32>
    %addenb3zW1 = stablehlo.add %adsqb3zW1, %adepsb3zW1 : tensor<144x6xf32>
    %adratb3zW1 = stablehlo.divide %admhb3zW1, %addenb3zW1 : tensor<144x6xf32>
    %adstb3zW1 = stablehlo.multiply %adlrb3zW1, %adratb3zW1 : tensor<144x6xf32>
    %adsubb3zW1 = stablehlo.subtract %b3zW1, %adstb3zW1 : tensor<144x6xf32>
    %adwdb3zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adwdlrb3zW1 = stablehlo.multiply %adwdb3zW1, %adlrb3zW1 : tensor<144x6xf32>
    %adwdpb3zW1 = stablehlo.multiply %adwdlrb3zW1, %b3zW1 : tensor<144x6xf32>
    %adnewb3zW1 = stablehlo.subtract %adsubb3zW1, %adwdpb3zW1 : tensor<144x6xf32>
    %adb1b3zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adob1b3zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %admsb3zb1 = stablehlo.multiply %adb1b3zb1, %b3zb1m : tensor<6xf32>
    %admgb3zb1 = stablehlo.multiply %adob1b3zb1, %b3zdbs1 : tensor<6xf32>
    %admnb3zb1 = stablehlo.add %admsb3zb1, %admgb3zb1 : tensor<6xf32>
    %adb2b3zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adob2b3zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %advsb3zb1 = stablehlo.multiply %adb2b3zb1, %b3zb1v : tensor<6xf32>
    %adg2b3zb1 = stablehlo.multiply %b3zdbs1, %b3zdbs1 : tensor<6xf32>
    %advgb3zb1 = stablehlo.multiply %adob2b3zb1, %adg2b3zb1 : tensor<6xf32>
    %advnb3zb1 = stablehlo.add %advsb3zb1, %advgb3zb1 : tensor<6xf32>
    %adbc1b3zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adbc2b3zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %admhb3zb1 = stablehlo.divide %admnb3zb1, %adbc1b3zb1 : tensor<6xf32>
    %advhb3zb1 = stablehlo.divide %advnb3zb1, %adbc2b3zb1 : tensor<6xf32>
    %adlrb3zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adepsb3zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adsqb3zb1 = stablehlo.sqrt %advhb3zb1 : tensor<6xf32>
    %addenb3zb1 = stablehlo.add %adsqb3zb1, %adepsb3zb1 : tensor<6xf32>
    %adratb3zb1 = stablehlo.divide %admhb3zb1, %addenb3zb1 : tensor<6xf32>
    %adstb3zb1 = stablehlo.multiply %adlrb3zb1, %adratb3zb1 : tensor<6xf32>
    %adsubb3zb1 = stablehlo.subtract %b3zb1, %adstb3zb1 : tensor<6xf32>
    %adwdb3zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adwdlrb3zb1 = stablehlo.multiply %adwdb3zb1, %adlrb3zb1 : tensor<6xf32>
    %adwdpb3zb1 = stablehlo.multiply %adwdlrb3zb1, %b3zb1 : tensor<6xf32>
    %adnewb3zb1 = stablehlo.subtract %adsubb3zb1, %adwdpb3zb1 : tensor<6xf32>
    %adb1b3zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adob1b3zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %admsb3zW2 = stablehlo.multiply %adb1b3zW2, %b3zW2m : tensor<6x144xf32>
    %admgb3zW2 = stablehlo.multiply %adob1b3zW2, %b3zdWs2 : tensor<6x144xf32>
    %admnb3zW2 = stablehlo.add %admsb3zW2, %admgb3zW2 : tensor<6x144xf32>
    %adb2b3zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adob2b3zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %advsb3zW2 = stablehlo.multiply %adb2b3zW2, %b3zW2v : tensor<6x144xf32>
    %adg2b3zW2 = stablehlo.multiply %b3zdWs2, %b3zdWs2 : tensor<6x144xf32>
    %advgb3zW2 = stablehlo.multiply %adob2b3zW2, %adg2b3zW2 : tensor<6x144xf32>
    %advnb3zW2 = stablehlo.add %advsb3zW2, %advgb3zW2 : tensor<6x144xf32>
    %adbc1b3zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adbc2b3zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %admhb3zW2 = stablehlo.divide %admnb3zW2, %adbc1b3zW2 : tensor<6x144xf32>
    %advhb3zW2 = stablehlo.divide %advnb3zW2, %adbc2b3zW2 : tensor<6x144xf32>
    %adlrb3zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adepsb3zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adsqb3zW2 = stablehlo.sqrt %advhb3zW2 : tensor<6x144xf32>
    %addenb3zW2 = stablehlo.add %adsqb3zW2, %adepsb3zW2 : tensor<6x144xf32>
    %adratb3zW2 = stablehlo.divide %admhb3zW2, %addenb3zW2 : tensor<6x144xf32>
    %adstb3zW2 = stablehlo.multiply %adlrb3zW2, %adratb3zW2 : tensor<6x144xf32>
    %adsubb3zW2 = stablehlo.subtract %b3zW2, %adstb3zW2 : tensor<6x144xf32>
    %adwdb3zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adwdlrb3zW2 = stablehlo.multiply %adwdb3zW2, %adlrb3zW2 : tensor<6x144xf32>
    %adwdpb3zW2 = stablehlo.multiply %adwdlrb3zW2, %b3zW2 : tensor<6x144xf32>
    %adnewb3zW2 = stablehlo.subtract %adsubb3zW2, %adwdpb3zW2 : tensor<6x144xf32>
    %adb1b3zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b3zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb3zb2 = stablehlo.multiply %adb1b3zb2, %b3zb2m : tensor<144xf32>
    %admgb3zb2 = stablehlo.multiply %adob1b3zb2, %b3zdbs2 : tensor<144xf32>
    %admnb3zb2 = stablehlo.add %admsb3zb2, %admgb3zb2 : tensor<144xf32>
    %adb2b3zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b3zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb3zb2 = stablehlo.multiply %adb2b3zb2, %b3zb2v : tensor<144xf32>
    %adg2b3zb2 = stablehlo.multiply %b3zdbs2, %b3zdbs2 : tensor<144xf32>
    %advgb3zb2 = stablehlo.multiply %adob2b3zb2, %adg2b3zb2 : tensor<144xf32>
    %advnb3zb2 = stablehlo.add %advsb3zb2, %advgb3zb2 : tensor<144xf32>
    %adbc1b3zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b3zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb3zb2 = stablehlo.divide %admnb3zb2, %adbc1b3zb2 : tensor<144xf32>
    %advhb3zb2 = stablehlo.divide %advnb3zb2, %adbc2b3zb2 : tensor<144xf32>
    %adlrb3zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb3zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb3zb2 = stablehlo.sqrt %advhb3zb2 : tensor<144xf32>
    %addenb3zb2 = stablehlo.add %adsqb3zb2, %adepsb3zb2 : tensor<144xf32>
    %adratb3zb2 = stablehlo.divide %admhb3zb2, %addenb3zb2 : tensor<144xf32>
    %adstb3zb2 = stablehlo.multiply %adlrb3zb2, %adratb3zb2 : tensor<144xf32>
    %adsubb3zb2 = stablehlo.subtract %b3zb2, %adstb3zb2 : tensor<144xf32>
    %adwdb3zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb3zb2 = stablehlo.multiply %adwdb3zb2, %adlrb3zb2 : tensor<144xf32>
    %adwdpb3zb2 = stablehlo.multiply %adwdlrb3zb2, %b3zb2 : tensor<144xf32>
    %adnewb3zb2 = stablehlo.subtract %adsubb3zb2, %adwdpb3zb2 : tensor<144xf32>
    %adb1b3pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %adob1b3pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %admsb3pW = stablehlo.multiply %adb1b3pW, %b3pWm : tensor<24x144x1x1xf32>
    %admgb3pW = stablehlo.multiply %adob1b3pW, %b3dpW : tensor<24x144x1x1xf32>
    %admnb3pW = stablehlo.add %admsb3pW, %admgb3pW : tensor<24x144x1x1xf32>
    %adb2b3pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %adob2b3pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %advsb3pW = stablehlo.multiply %adb2b3pW, %b3pWv : tensor<24x144x1x1xf32>
    %adg2b3pW = stablehlo.multiply %b3dpW, %b3dpW : tensor<24x144x1x1xf32>
    %advgb3pW = stablehlo.multiply %adob2b3pW, %adg2b3pW : tensor<24x144x1x1xf32>
    %advnb3pW = stablehlo.add %advsb3pW, %advgb3pW : tensor<24x144x1x1xf32>
    %adbc1b3pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %adbc2b3pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %admhb3pW = stablehlo.divide %admnb3pW, %adbc1b3pW : tensor<24x144x1x1xf32>
    %advhb3pW = stablehlo.divide %advnb3pW, %adbc2b3pW : tensor<24x144x1x1xf32>
    %adlrb3pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %adepsb3pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %adsqb3pW = stablehlo.sqrt %advhb3pW : tensor<24x144x1x1xf32>
    %addenb3pW = stablehlo.add %adsqb3pW, %adepsb3pW : tensor<24x144x1x1xf32>
    %adratb3pW = stablehlo.divide %admhb3pW, %addenb3pW : tensor<24x144x1x1xf32>
    %adstb3pW = stablehlo.multiply %adlrb3pW, %adratb3pW : tensor<24x144x1x1xf32>
    %adsubb3pW = stablehlo.subtract %b3pW, %adstb3pW : tensor<24x144x1x1xf32>
    %adwdb3pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %adwdlrb3pW = stablehlo.multiply %adwdb3pW, %adlrb3pW : tensor<24x144x1x1xf32>
    %adwdpb3pW = stablehlo.multiply %adwdlrb3pW, %b3pW : tensor<24x144x1x1xf32>
    %adnewb3pW = stablehlo.subtract %adsubb3pW, %adwdpb3pW : tensor<24x144x1x1xf32>
    %adb1b3pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b3pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb3pb = stablehlo.multiply %adb1b3pb, %b3pbm : tensor<24xf32>
    %admgb3pb = stablehlo.multiply %adob1b3pb, %b3dpb : tensor<24xf32>
    %admnb3pb = stablehlo.add %admsb3pb, %admgb3pb : tensor<24xf32>
    %adb2b3pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b3pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb3pb = stablehlo.multiply %adb2b3pb, %b3pbv : tensor<24xf32>
    %adg2b3pb = stablehlo.multiply %b3dpb, %b3dpb : tensor<24xf32>
    %advgb3pb = stablehlo.multiply %adob2b3pb, %adg2b3pb : tensor<24xf32>
    %advnb3pb = stablehlo.add %advsb3pb, %advgb3pb : tensor<24xf32>
    %adbc1b3pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b3pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb3pb = stablehlo.divide %admnb3pb, %adbc1b3pb : tensor<24xf32>
    %advhb3pb = stablehlo.divide %advnb3pb, %adbc2b3pb : tensor<24xf32>
    %adlrb3pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb3pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb3pb = stablehlo.sqrt %advhb3pb : tensor<24xf32>
    %addenb3pb = stablehlo.add %adsqb3pb, %adepsb3pb : tensor<24xf32>
    %adratb3pb = stablehlo.divide %admhb3pb, %addenb3pb : tensor<24xf32>
    %adstb3pb = stablehlo.multiply %adlrb3pb, %adratb3pb : tensor<24xf32>
    %adsubb3pb = stablehlo.subtract %b3pb, %adstb3pb : tensor<24xf32>
    %adwdb3pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb3pb = stablehlo.multiply %adwdb3pb, %adlrb3pb : tensor<24xf32>
    %adwdpb3pb = stablehlo.multiply %adwdlrb3pb, %b3pb : tensor<24xf32>
    %adnewb3pb = stablehlo.subtract %adsubb3pb, %adwdpb3pb : tensor<24xf32>
    %adb1b3pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b3pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb3pg = stablehlo.multiply %adb1b3pg, %b3pgm : tensor<24xf32>
    %admgb3pg = stablehlo.multiply %adob1b3pg, %b3dpndg : tensor<24xf32>
    %admnb3pg = stablehlo.add %admsb3pg, %admgb3pg : tensor<24xf32>
    %adb2b3pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b3pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb3pg = stablehlo.multiply %adb2b3pg, %b3pgv : tensor<24xf32>
    %adg2b3pg = stablehlo.multiply %b3dpndg, %b3dpndg : tensor<24xf32>
    %advgb3pg = stablehlo.multiply %adob2b3pg, %adg2b3pg : tensor<24xf32>
    %advnb3pg = stablehlo.add %advsb3pg, %advgb3pg : tensor<24xf32>
    %adbc1b3pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b3pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb3pg = stablehlo.divide %admnb3pg, %adbc1b3pg : tensor<24xf32>
    %advhb3pg = stablehlo.divide %advnb3pg, %adbc2b3pg : tensor<24xf32>
    %adlrb3pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb3pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb3pg = stablehlo.sqrt %advhb3pg : tensor<24xf32>
    %addenb3pg = stablehlo.add %adsqb3pg, %adepsb3pg : tensor<24xf32>
    %adratb3pg = stablehlo.divide %admhb3pg, %addenb3pg : tensor<24xf32>
    %adstb3pg = stablehlo.multiply %adlrb3pg, %adratb3pg : tensor<24xf32>
    %adsubb3pg = stablehlo.subtract %b3pg, %adstb3pg : tensor<24xf32>
    %adwdb3pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb3pg = stablehlo.multiply %adwdb3pg, %adlrb3pg : tensor<24xf32>
    %adwdpb3pg = stablehlo.multiply %adwdlrb3pg, %b3pg : tensor<24xf32>
    %adnewb3pg = stablehlo.subtract %adsubb3pg, %adwdpb3pg : tensor<24xf32>
    %adb1b3pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b3pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb3pbt = stablehlo.multiply %adb1b3pbt, %b3pbtm : tensor<24xf32>
    %admgb3pbt = stablehlo.multiply %adob1b3pbt, %b3dpndb : tensor<24xf32>
    %admnb3pbt = stablehlo.add %admsb3pbt, %admgb3pbt : tensor<24xf32>
    %adb2b3pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b3pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb3pbt = stablehlo.multiply %adb2b3pbt, %b3pbtv : tensor<24xf32>
    %adg2b3pbt = stablehlo.multiply %b3dpndb, %b3dpndb : tensor<24xf32>
    %advgb3pbt = stablehlo.multiply %adob2b3pbt, %adg2b3pbt : tensor<24xf32>
    %advnb3pbt = stablehlo.add %advsb3pbt, %advgb3pbt : tensor<24xf32>
    %adbc1b3pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b3pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb3pbt = stablehlo.divide %admnb3pbt, %adbc1b3pbt : tensor<24xf32>
    %advhb3pbt = stablehlo.divide %advnb3pbt, %adbc2b3pbt : tensor<24xf32>
    %adlrb3pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb3pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb3pbt = stablehlo.sqrt %advhb3pbt : tensor<24xf32>
    %addenb3pbt = stablehlo.add %adsqb3pbt, %adepsb3pbt : tensor<24xf32>
    %adratb3pbt = stablehlo.divide %admhb3pbt, %addenb3pbt : tensor<24xf32>
    %adstb3pbt = stablehlo.multiply %adlrb3pbt, %adratb3pbt : tensor<24xf32>
    %adsubb3pbt = stablehlo.subtract %b3pbt, %adstb3pbt : tensor<24xf32>
    %adwdb3pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb3pbt = stablehlo.multiply %adwdb3pbt, %adlrb3pbt : tensor<24xf32>
    %adwdpb3pbt = stablehlo.multiply %adwdlrb3pbt, %b3pbt : tensor<24xf32>
    %adnewb3pbt = stablehlo.subtract %adsubb3pbt, %adwdpb3pbt : tensor<24xf32>
    %adb1b4eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adob1b4eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %admsb4eW = stablehlo.multiply %adb1b4eW, %b4eWm : tensor<144x24x1x1xf32>
    %admgb4eW = stablehlo.multiply %adob1b4eW, %b4deW : tensor<144x24x1x1xf32>
    %admnb4eW = stablehlo.add %admsb4eW, %admgb4eW : tensor<144x24x1x1xf32>
    %adb2b4eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adob2b4eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %advsb4eW = stablehlo.multiply %adb2b4eW, %b4eWv : tensor<144x24x1x1xf32>
    %adg2b4eW = stablehlo.multiply %b4deW, %b4deW : tensor<144x24x1x1xf32>
    %advgb4eW = stablehlo.multiply %adob2b4eW, %adg2b4eW : tensor<144x24x1x1xf32>
    %advnb4eW = stablehlo.add %advsb4eW, %advgb4eW : tensor<144x24x1x1xf32>
    %adbc1b4eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adbc2b4eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %admhb4eW = stablehlo.divide %admnb4eW, %adbc1b4eW : tensor<144x24x1x1xf32>
    %advhb4eW = stablehlo.divide %advnb4eW, %adbc2b4eW : tensor<144x24x1x1xf32>
    %adlrb4eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adepsb4eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adsqb4eW = stablehlo.sqrt %advhb4eW : tensor<144x24x1x1xf32>
    %addenb4eW = stablehlo.add %adsqb4eW, %adepsb4eW : tensor<144x24x1x1xf32>
    %adratb4eW = stablehlo.divide %admhb4eW, %addenb4eW : tensor<144x24x1x1xf32>
    %adstb4eW = stablehlo.multiply %adlrb4eW, %adratb4eW : tensor<144x24x1x1xf32>
    %adsubb4eW = stablehlo.subtract %b4eW, %adstb4eW : tensor<144x24x1x1xf32>
    %adwdb4eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %adwdlrb4eW = stablehlo.multiply %adwdb4eW, %adlrb4eW : tensor<144x24x1x1xf32>
    %adwdpb4eW = stablehlo.multiply %adwdlrb4eW, %b4eW : tensor<144x24x1x1xf32>
    %adnewb4eW = stablehlo.subtract %adsubb4eW, %adwdpb4eW : tensor<144x24x1x1xf32>
    %adb1b4eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b4eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb4eb = stablehlo.multiply %adb1b4eb, %b4ebm : tensor<144xf32>
    %admgb4eb = stablehlo.multiply %adob1b4eb, %b4deb : tensor<144xf32>
    %admnb4eb = stablehlo.add %admsb4eb, %admgb4eb : tensor<144xf32>
    %adb2b4eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b4eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb4eb = stablehlo.multiply %adb2b4eb, %b4ebv : tensor<144xf32>
    %adg2b4eb = stablehlo.multiply %b4deb, %b4deb : tensor<144xf32>
    %advgb4eb = stablehlo.multiply %adob2b4eb, %adg2b4eb : tensor<144xf32>
    %advnb4eb = stablehlo.add %advsb4eb, %advgb4eb : tensor<144xf32>
    %adbc1b4eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b4eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb4eb = stablehlo.divide %admnb4eb, %adbc1b4eb : tensor<144xf32>
    %advhb4eb = stablehlo.divide %advnb4eb, %adbc2b4eb : tensor<144xf32>
    %adlrb4eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb4eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb4eb = stablehlo.sqrt %advhb4eb : tensor<144xf32>
    %addenb4eb = stablehlo.add %adsqb4eb, %adepsb4eb : tensor<144xf32>
    %adratb4eb = stablehlo.divide %admhb4eb, %addenb4eb : tensor<144xf32>
    %adstb4eb = stablehlo.multiply %adlrb4eb, %adratb4eb : tensor<144xf32>
    %adsubb4eb = stablehlo.subtract %b4eb, %adstb4eb : tensor<144xf32>
    %adwdb4eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb4eb = stablehlo.multiply %adwdb4eb, %adlrb4eb : tensor<144xf32>
    %adwdpb4eb = stablehlo.multiply %adwdlrb4eb, %b4eb : tensor<144xf32>
    %adnewb4eb = stablehlo.subtract %adsubb4eb, %adwdpb4eb : tensor<144xf32>
    %adb1b4eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b4eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb4eg = stablehlo.multiply %adb1b4eg, %b4egm : tensor<144xf32>
    %admgb4eg = stablehlo.multiply %adob1b4eg, %b4dendg : tensor<144xf32>
    %admnb4eg = stablehlo.add %admsb4eg, %admgb4eg : tensor<144xf32>
    %adb2b4eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b4eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb4eg = stablehlo.multiply %adb2b4eg, %b4egv : tensor<144xf32>
    %adg2b4eg = stablehlo.multiply %b4dendg, %b4dendg : tensor<144xf32>
    %advgb4eg = stablehlo.multiply %adob2b4eg, %adg2b4eg : tensor<144xf32>
    %advnb4eg = stablehlo.add %advsb4eg, %advgb4eg : tensor<144xf32>
    %adbc1b4eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b4eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb4eg = stablehlo.divide %admnb4eg, %adbc1b4eg : tensor<144xf32>
    %advhb4eg = stablehlo.divide %advnb4eg, %adbc2b4eg : tensor<144xf32>
    %adlrb4eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb4eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb4eg = stablehlo.sqrt %advhb4eg : tensor<144xf32>
    %addenb4eg = stablehlo.add %adsqb4eg, %adepsb4eg : tensor<144xf32>
    %adratb4eg = stablehlo.divide %admhb4eg, %addenb4eg : tensor<144xf32>
    %adstb4eg = stablehlo.multiply %adlrb4eg, %adratb4eg : tensor<144xf32>
    %adsubb4eg = stablehlo.subtract %b4eg, %adstb4eg : tensor<144xf32>
    %adwdb4eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb4eg = stablehlo.multiply %adwdb4eg, %adlrb4eg : tensor<144xf32>
    %adwdpb4eg = stablehlo.multiply %adwdlrb4eg, %b4eg : tensor<144xf32>
    %adnewb4eg = stablehlo.subtract %adsubb4eg, %adwdpb4eg : tensor<144xf32>
    %adb1b4ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b4ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb4ebt = stablehlo.multiply %adb1b4ebt, %b4ebtm : tensor<144xf32>
    %admgb4ebt = stablehlo.multiply %adob1b4ebt, %b4dendb : tensor<144xf32>
    %admnb4ebt = stablehlo.add %admsb4ebt, %admgb4ebt : tensor<144xf32>
    %adb2b4ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b4ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb4ebt = stablehlo.multiply %adb2b4ebt, %b4ebtv : tensor<144xf32>
    %adg2b4ebt = stablehlo.multiply %b4dendb, %b4dendb : tensor<144xf32>
    %advgb4ebt = stablehlo.multiply %adob2b4ebt, %adg2b4ebt : tensor<144xf32>
    %advnb4ebt = stablehlo.add %advsb4ebt, %advgb4ebt : tensor<144xf32>
    %adbc1b4ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b4ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb4ebt = stablehlo.divide %admnb4ebt, %adbc1b4ebt : tensor<144xf32>
    %advhb4ebt = stablehlo.divide %advnb4ebt, %adbc2b4ebt : tensor<144xf32>
    %adlrb4ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb4ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb4ebt = stablehlo.sqrt %advhb4ebt : tensor<144xf32>
    %addenb4ebt = stablehlo.add %adsqb4ebt, %adepsb4ebt : tensor<144xf32>
    %adratb4ebt = stablehlo.divide %admhb4ebt, %addenb4ebt : tensor<144xf32>
    %adstb4ebt = stablehlo.multiply %adlrb4ebt, %adratb4ebt : tensor<144xf32>
    %adsubb4ebt = stablehlo.subtract %b4ebt, %adstb4ebt : tensor<144xf32>
    %adwdb4ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb4ebt = stablehlo.multiply %adwdb4ebt, %adlrb4ebt : tensor<144xf32>
    %adwdpb4ebt = stablehlo.multiply %adwdlrb4ebt, %b4ebt : tensor<144xf32>
    %adnewb4ebt = stablehlo.subtract %adsubb4ebt, %adwdpb4ebt : tensor<144xf32>
    %adb1b4dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %adob1b4dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %admsb4dW = stablehlo.multiply %adb1b4dW, %b4dWm : tensor<144x1x5x5xf32>
    %admgb4dW = stablehlo.multiply %adob1b4dW, %b4ddW : tensor<144x1x5x5xf32>
    %admnb4dW = stablehlo.add %admsb4dW, %admgb4dW : tensor<144x1x5x5xf32>
    %adb2b4dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %adob2b4dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %advsb4dW = stablehlo.multiply %adb2b4dW, %b4dWv : tensor<144x1x5x5xf32>
    %adg2b4dW = stablehlo.multiply %b4ddW, %b4ddW : tensor<144x1x5x5xf32>
    %advgb4dW = stablehlo.multiply %adob2b4dW, %adg2b4dW : tensor<144x1x5x5xf32>
    %advnb4dW = stablehlo.add %advsb4dW, %advgb4dW : tensor<144x1x5x5xf32>
    %adbc1b4dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %adbc2b4dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %admhb4dW = stablehlo.divide %admnb4dW, %adbc1b4dW : tensor<144x1x5x5xf32>
    %advhb4dW = stablehlo.divide %advnb4dW, %adbc2b4dW : tensor<144x1x5x5xf32>
    %adlrb4dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %adepsb4dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %adsqb4dW = stablehlo.sqrt %advhb4dW : tensor<144x1x5x5xf32>
    %addenb4dW = stablehlo.add %adsqb4dW, %adepsb4dW : tensor<144x1x5x5xf32>
    %adratb4dW = stablehlo.divide %admhb4dW, %addenb4dW : tensor<144x1x5x5xf32>
    %adstb4dW = stablehlo.multiply %adlrb4dW, %adratb4dW : tensor<144x1x5x5xf32>
    %adsubb4dW = stablehlo.subtract %b4dW, %adstb4dW : tensor<144x1x5x5xf32>
    %adwdb4dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x1x5x5xf32>
    %adwdlrb4dW = stablehlo.multiply %adwdb4dW, %adlrb4dW : tensor<144x1x5x5xf32>
    %adwdpb4dW = stablehlo.multiply %adwdlrb4dW, %b4dW : tensor<144x1x5x5xf32>
    %adnewb4dW = stablehlo.subtract %adsubb4dW, %adwdpb4dW : tensor<144x1x5x5xf32>
    %adb1b4db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b4db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb4db = stablehlo.multiply %adb1b4db, %b4dbm : tensor<144xf32>
    %admgb4db = stablehlo.multiply %adob1b4db, %b4ddb : tensor<144xf32>
    %admnb4db = stablehlo.add %admsb4db, %admgb4db : tensor<144xf32>
    %adb2b4db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b4db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb4db = stablehlo.multiply %adb2b4db, %b4dbv : tensor<144xf32>
    %adg2b4db = stablehlo.multiply %b4ddb, %b4ddb : tensor<144xf32>
    %advgb4db = stablehlo.multiply %adob2b4db, %adg2b4db : tensor<144xf32>
    %advnb4db = stablehlo.add %advsb4db, %advgb4db : tensor<144xf32>
    %adbc1b4db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b4db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb4db = stablehlo.divide %admnb4db, %adbc1b4db : tensor<144xf32>
    %advhb4db = stablehlo.divide %advnb4db, %adbc2b4db : tensor<144xf32>
    %adlrb4db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb4db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb4db = stablehlo.sqrt %advhb4db : tensor<144xf32>
    %addenb4db = stablehlo.add %adsqb4db, %adepsb4db : tensor<144xf32>
    %adratb4db = stablehlo.divide %admhb4db, %addenb4db : tensor<144xf32>
    %adstb4db = stablehlo.multiply %adlrb4db, %adratb4db : tensor<144xf32>
    %adsubb4db = stablehlo.subtract %b4db, %adstb4db : tensor<144xf32>
    %adwdb4db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb4db = stablehlo.multiply %adwdb4db, %adlrb4db : tensor<144xf32>
    %adwdpb4db = stablehlo.multiply %adwdlrb4db, %b4db : tensor<144xf32>
    %adnewb4db = stablehlo.subtract %adsubb4db, %adwdpb4db : tensor<144xf32>
    %adb1b4dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b4dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb4dg = stablehlo.multiply %adb1b4dg, %b4dgm : tensor<144xf32>
    %admgb4dg = stablehlo.multiply %adob1b4dg, %b4ddndg : tensor<144xf32>
    %admnb4dg = stablehlo.add %admsb4dg, %admgb4dg : tensor<144xf32>
    %adb2b4dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b4dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb4dg = stablehlo.multiply %adb2b4dg, %b4dgv : tensor<144xf32>
    %adg2b4dg = stablehlo.multiply %b4ddndg, %b4ddndg : tensor<144xf32>
    %advgb4dg = stablehlo.multiply %adob2b4dg, %adg2b4dg : tensor<144xf32>
    %advnb4dg = stablehlo.add %advsb4dg, %advgb4dg : tensor<144xf32>
    %adbc1b4dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b4dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb4dg = stablehlo.divide %admnb4dg, %adbc1b4dg : tensor<144xf32>
    %advhb4dg = stablehlo.divide %advnb4dg, %adbc2b4dg : tensor<144xf32>
    %adlrb4dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb4dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb4dg = stablehlo.sqrt %advhb4dg : tensor<144xf32>
    %addenb4dg = stablehlo.add %adsqb4dg, %adepsb4dg : tensor<144xf32>
    %adratb4dg = stablehlo.divide %admhb4dg, %addenb4dg : tensor<144xf32>
    %adstb4dg = stablehlo.multiply %adlrb4dg, %adratb4dg : tensor<144xf32>
    %adsubb4dg = stablehlo.subtract %b4dg, %adstb4dg : tensor<144xf32>
    %adwdb4dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb4dg = stablehlo.multiply %adwdb4dg, %adlrb4dg : tensor<144xf32>
    %adwdpb4dg = stablehlo.multiply %adwdlrb4dg, %b4dg : tensor<144xf32>
    %adnewb4dg = stablehlo.subtract %adsubb4dg, %adwdpb4dg : tensor<144xf32>
    %adb1b4dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b4dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb4dbt = stablehlo.multiply %adb1b4dbt, %b4dbtm : tensor<144xf32>
    %admgb4dbt = stablehlo.multiply %adob1b4dbt, %b4ddndb : tensor<144xf32>
    %admnb4dbt = stablehlo.add %admsb4dbt, %admgb4dbt : tensor<144xf32>
    %adb2b4dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b4dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb4dbt = stablehlo.multiply %adb2b4dbt, %b4dbtv : tensor<144xf32>
    %adg2b4dbt = stablehlo.multiply %b4ddndb, %b4ddndb : tensor<144xf32>
    %advgb4dbt = stablehlo.multiply %adob2b4dbt, %adg2b4dbt : tensor<144xf32>
    %advnb4dbt = stablehlo.add %advsb4dbt, %advgb4dbt : tensor<144xf32>
    %adbc1b4dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b4dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb4dbt = stablehlo.divide %admnb4dbt, %adbc1b4dbt : tensor<144xf32>
    %advhb4dbt = stablehlo.divide %advnb4dbt, %adbc2b4dbt : tensor<144xf32>
    %adlrb4dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb4dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb4dbt = stablehlo.sqrt %advhb4dbt : tensor<144xf32>
    %addenb4dbt = stablehlo.add %adsqb4dbt, %adepsb4dbt : tensor<144xf32>
    %adratb4dbt = stablehlo.divide %admhb4dbt, %addenb4dbt : tensor<144xf32>
    %adstb4dbt = stablehlo.multiply %adlrb4dbt, %adratb4dbt : tensor<144xf32>
    %adsubb4dbt = stablehlo.subtract %b4dbt, %adstb4dbt : tensor<144xf32>
    %adwdb4dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb4dbt = stablehlo.multiply %adwdb4dbt, %adlrb4dbt : tensor<144xf32>
    %adwdpb4dbt = stablehlo.multiply %adwdlrb4dbt, %b4dbt : tensor<144xf32>
    %adnewb4dbt = stablehlo.subtract %adsubb4dbt, %adwdpb4dbt : tensor<144xf32>
    %adb1b4zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adob1b4zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %admsb4zW1 = stablehlo.multiply %adb1b4zW1, %b4zW1m : tensor<144x6xf32>
    %admgb4zW1 = stablehlo.multiply %adob1b4zW1, %b4zdWs1 : tensor<144x6xf32>
    %admnb4zW1 = stablehlo.add %admsb4zW1, %admgb4zW1 : tensor<144x6xf32>
    %adb2b4zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adob2b4zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %advsb4zW1 = stablehlo.multiply %adb2b4zW1, %b4zW1v : tensor<144x6xf32>
    %adg2b4zW1 = stablehlo.multiply %b4zdWs1, %b4zdWs1 : tensor<144x6xf32>
    %advgb4zW1 = stablehlo.multiply %adob2b4zW1, %adg2b4zW1 : tensor<144x6xf32>
    %advnb4zW1 = stablehlo.add %advsb4zW1, %advgb4zW1 : tensor<144x6xf32>
    %adbc1b4zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adbc2b4zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %admhb4zW1 = stablehlo.divide %admnb4zW1, %adbc1b4zW1 : tensor<144x6xf32>
    %advhb4zW1 = stablehlo.divide %advnb4zW1, %adbc2b4zW1 : tensor<144x6xf32>
    %adlrb4zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adepsb4zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adsqb4zW1 = stablehlo.sqrt %advhb4zW1 : tensor<144x6xf32>
    %addenb4zW1 = stablehlo.add %adsqb4zW1, %adepsb4zW1 : tensor<144x6xf32>
    %adratb4zW1 = stablehlo.divide %admhb4zW1, %addenb4zW1 : tensor<144x6xf32>
    %adstb4zW1 = stablehlo.multiply %adlrb4zW1, %adratb4zW1 : tensor<144x6xf32>
    %adsubb4zW1 = stablehlo.subtract %b4zW1, %adstb4zW1 : tensor<144x6xf32>
    %adwdb4zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x6xf32>
    %adwdlrb4zW1 = stablehlo.multiply %adwdb4zW1, %adlrb4zW1 : tensor<144x6xf32>
    %adwdpb4zW1 = stablehlo.multiply %adwdlrb4zW1, %b4zW1 : tensor<144x6xf32>
    %adnewb4zW1 = stablehlo.subtract %adsubb4zW1, %adwdpb4zW1 : tensor<144x6xf32>
    %adb1b4zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adob1b4zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %admsb4zb1 = stablehlo.multiply %adb1b4zb1, %b4zb1m : tensor<6xf32>
    %admgb4zb1 = stablehlo.multiply %adob1b4zb1, %b4zdbs1 : tensor<6xf32>
    %admnb4zb1 = stablehlo.add %admsb4zb1, %admgb4zb1 : tensor<6xf32>
    %adb2b4zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adob2b4zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %advsb4zb1 = stablehlo.multiply %adb2b4zb1, %b4zb1v : tensor<6xf32>
    %adg2b4zb1 = stablehlo.multiply %b4zdbs1, %b4zdbs1 : tensor<6xf32>
    %advgb4zb1 = stablehlo.multiply %adob2b4zb1, %adg2b4zb1 : tensor<6xf32>
    %advnb4zb1 = stablehlo.add %advsb4zb1, %advgb4zb1 : tensor<6xf32>
    %adbc1b4zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adbc2b4zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %admhb4zb1 = stablehlo.divide %admnb4zb1, %adbc1b4zb1 : tensor<6xf32>
    %advhb4zb1 = stablehlo.divide %advnb4zb1, %adbc2b4zb1 : tensor<6xf32>
    %adlrb4zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adepsb4zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adsqb4zb1 = stablehlo.sqrt %advhb4zb1 : tensor<6xf32>
    %addenb4zb1 = stablehlo.add %adsqb4zb1, %adepsb4zb1 : tensor<6xf32>
    %adratb4zb1 = stablehlo.divide %admhb4zb1, %addenb4zb1 : tensor<6xf32>
    %adstb4zb1 = stablehlo.multiply %adlrb4zb1, %adratb4zb1 : tensor<6xf32>
    %adsubb4zb1 = stablehlo.subtract %b4zb1, %adstb4zb1 : tensor<6xf32>
    %adwdb4zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<6xf32>
    %adwdlrb4zb1 = stablehlo.multiply %adwdb4zb1, %adlrb4zb1 : tensor<6xf32>
    %adwdpb4zb1 = stablehlo.multiply %adwdlrb4zb1, %b4zb1 : tensor<6xf32>
    %adnewb4zb1 = stablehlo.subtract %adsubb4zb1, %adwdpb4zb1 : tensor<6xf32>
    %adb1b4zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adob1b4zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %admsb4zW2 = stablehlo.multiply %adb1b4zW2, %b4zW2m : tensor<6x144xf32>
    %admgb4zW2 = stablehlo.multiply %adob1b4zW2, %b4zdWs2 : tensor<6x144xf32>
    %admnb4zW2 = stablehlo.add %admsb4zW2, %admgb4zW2 : tensor<6x144xf32>
    %adb2b4zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adob2b4zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %advsb4zW2 = stablehlo.multiply %adb2b4zW2, %b4zW2v : tensor<6x144xf32>
    %adg2b4zW2 = stablehlo.multiply %b4zdWs2, %b4zdWs2 : tensor<6x144xf32>
    %advgb4zW2 = stablehlo.multiply %adob2b4zW2, %adg2b4zW2 : tensor<6x144xf32>
    %advnb4zW2 = stablehlo.add %advsb4zW2, %advgb4zW2 : tensor<6x144xf32>
    %adbc1b4zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adbc2b4zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %admhb4zW2 = stablehlo.divide %admnb4zW2, %adbc1b4zW2 : tensor<6x144xf32>
    %advhb4zW2 = stablehlo.divide %advnb4zW2, %adbc2b4zW2 : tensor<6x144xf32>
    %adlrb4zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adepsb4zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adsqb4zW2 = stablehlo.sqrt %advhb4zW2 : tensor<6x144xf32>
    %addenb4zW2 = stablehlo.add %adsqb4zW2, %adepsb4zW2 : tensor<6x144xf32>
    %adratb4zW2 = stablehlo.divide %admhb4zW2, %addenb4zW2 : tensor<6x144xf32>
    %adstb4zW2 = stablehlo.multiply %adlrb4zW2, %adratb4zW2 : tensor<6x144xf32>
    %adsubb4zW2 = stablehlo.subtract %b4zW2, %adstb4zW2 : tensor<6x144xf32>
    %adwdb4zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<6x144xf32>
    %adwdlrb4zW2 = stablehlo.multiply %adwdb4zW2, %adlrb4zW2 : tensor<6x144xf32>
    %adwdpb4zW2 = stablehlo.multiply %adwdlrb4zW2, %b4zW2 : tensor<6x144xf32>
    %adnewb4zW2 = stablehlo.subtract %adsubb4zW2, %adwdpb4zW2 : tensor<6x144xf32>
    %adb1b4zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob1b4zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admsb4zb2 = stablehlo.multiply %adb1b4zb2, %b4zb2m : tensor<144xf32>
    %admgb4zb2 = stablehlo.multiply %adob1b4zb2, %b4zdbs2 : tensor<144xf32>
    %admnb4zb2 = stablehlo.add %admsb4zb2, %admgb4zb2 : tensor<144xf32>
    %adb2b4zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adob2b4zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %advsb4zb2 = stablehlo.multiply %adb2b4zb2, %b4zb2v : tensor<144xf32>
    %adg2b4zb2 = stablehlo.multiply %b4zdbs2, %b4zdbs2 : tensor<144xf32>
    %advgb4zb2 = stablehlo.multiply %adob2b4zb2, %adg2b4zb2 : tensor<144xf32>
    %advnb4zb2 = stablehlo.add %advsb4zb2, %advgb4zb2 : tensor<144xf32>
    %adbc1b4zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adbc2b4zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %admhb4zb2 = stablehlo.divide %admnb4zb2, %adbc1b4zb2 : tensor<144xf32>
    %advhb4zb2 = stablehlo.divide %advnb4zb2, %adbc2b4zb2 : tensor<144xf32>
    %adlrb4zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adepsb4zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adsqb4zb2 = stablehlo.sqrt %advhb4zb2 : tensor<144xf32>
    %addenb4zb2 = stablehlo.add %adsqb4zb2, %adepsb4zb2 : tensor<144xf32>
    %adratb4zb2 = stablehlo.divide %admhb4zb2, %addenb4zb2 : tensor<144xf32>
    %adstb4zb2 = stablehlo.multiply %adlrb4zb2, %adratb4zb2 : tensor<144xf32>
    %adsubb4zb2 = stablehlo.subtract %b4zb2, %adstb4zb2 : tensor<144xf32>
    %adwdb4zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %adwdlrb4zb2 = stablehlo.multiply %adwdb4zb2, %adlrb4zb2 : tensor<144xf32>
    %adwdpb4zb2 = stablehlo.multiply %adwdlrb4zb2, %b4zb2 : tensor<144xf32>
    %adnewb4zb2 = stablehlo.subtract %adsubb4zb2, %adwdpb4zb2 : tensor<144xf32>
    %adb1b4pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %adob1b4pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %admsb4pW = stablehlo.multiply %adb1b4pW, %b4pWm : tensor<40x144x1x1xf32>
    %admgb4pW = stablehlo.multiply %adob1b4pW, %b4dpW : tensor<40x144x1x1xf32>
    %admnb4pW = stablehlo.add %admsb4pW, %admgb4pW : tensor<40x144x1x1xf32>
    %adb2b4pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %adob2b4pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %advsb4pW = stablehlo.multiply %adb2b4pW, %b4pWv : tensor<40x144x1x1xf32>
    %adg2b4pW = stablehlo.multiply %b4dpW, %b4dpW : tensor<40x144x1x1xf32>
    %advgb4pW = stablehlo.multiply %adob2b4pW, %adg2b4pW : tensor<40x144x1x1xf32>
    %advnb4pW = stablehlo.add %advsb4pW, %advgb4pW : tensor<40x144x1x1xf32>
    %adbc1b4pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %adbc2b4pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %admhb4pW = stablehlo.divide %admnb4pW, %adbc1b4pW : tensor<40x144x1x1xf32>
    %advhb4pW = stablehlo.divide %advnb4pW, %adbc2b4pW : tensor<40x144x1x1xf32>
    %adlrb4pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %adepsb4pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %adsqb4pW = stablehlo.sqrt %advhb4pW : tensor<40x144x1x1xf32>
    %addenb4pW = stablehlo.add %adsqb4pW, %adepsb4pW : tensor<40x144x1x1xf32>
    %adratb4pW = stablehlo.divide %admhb4pW, %addenb4pW : tensor<40x144x1x1xf32>
    %adstb4pW = stablehlo.multiply %adlrb4pW, %adratb4pW : tensor<40x144x1x1xf32>
    %adsubb4pW = stablehlo.subtract %b4pW, %adstb4pW : tensor<40x144x1x1xf32>
    %adwdb4pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<40x144x1x1xf32>
    %adwdlrb4pW = stablehlo.multiply %adwdb4pW, %adlrb4pW : tensor<40x144x1x1xf32>
    %adwdpb4pW = stablehlo.multiply %adwdlrb4pW, %b4pW : tensor<40x144x1x1xf32>
    %adnewb4pW = stablehlo.subtract %adsubb4pW, %adwdpb4pW : tensor<40x144x1x1xf32>
    %adb1b4pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob1b4pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admsb4pb = stablehlo.multiply %adb1b4pb, %b4pbm : tensor<40xf32>
    %admgb4pb = stablehlo.multiply %adob1b4pb, %b4dpb : tensor<40xf32>
    %admnb4pb = stablehlo.add %admsb4pb, %admgb4pb : tensor<40xf32>
    %adb2b4pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob2b4pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %advsb4pb = stablehlo.multiply %adb2b4pb, %b4pbv : tensor<40xf32>
    %adg2b4pb = stablehlo.multiply %b4dpb, %b4dpb : tensor<40xf32>
    %advgb4pb = stablehlo.multiply %adob2b4pb, %adg2b4pb : tensor<40xf32>
    %advnb4pb = stablehlo.add %advsb4pb, %advgb4pb : tensor<40xf32>
    %adbc1b4pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adbc2b4pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admhb4pb = stablehlo.divide %admnb4pb, %adbc1b4pb : tensor<40xf32>
    %advhb4pb = stablehlo.divide %advnb4pb, %adbc2b4pb : tensor<40xf32>
    %adlrb4pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adepsb4pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adsqb4pb = stablehlo.sqrt %advhb4pb : tensor<40xf32>
    %addenb4pb = stablehlo.add %adsqb4pb, %adepsb4pb : tensor<40xf32>
    %adratb4pb = stablehlo.divide %admhb4pb, %addenb4pb : tensor<40xf32>
    %adstb4pb = stablehlo.multiply %adlrb4pb, %adratb4pb : tensor<40xf32>
    %adsubb4pb = stablehlo.subtract %b4pb, %adstb4pb : tensor<40xf32>
    %adwdb4pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adwdlrb4pb = stablehlo.multiply %adwdb4pb, %adlrb4pb : tensor<40xf32>
    %adwdpb4pb = stablehlo.multiply %adwdlrb4pb, %b4pb : tensor<40xf32>
    %adnewb4pb = stablehlo.subtract %adsubb4pb, %adwdpb4pb : tensor<40xf32>
    %adb1b4pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob1b4pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admsb4pg = stablehlo.multiply %adb1b4pg, %b4pgm : tensor<40xf32>
    %admgb4pg = stablehlo.multiply %adob1b4pg, %b4dpndg : tensor<40xf32>
    %admnb4pg = stablehlo.add %admsb4pg, %admgb4pg : tensor<40xf32>
    %adb2b4pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob2b4pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %advsb4pg = stablehlo.multiply %adb2b4pg, %b4pgv : tensor<40xf32>
    %adg2b4pg = stablehlo.multiply %b4dpndg, %b4dpndg : tensor<40xf32>
    %advgb4pg = stablehlo.multiply %adob2b4pg, %adg2b4pg : tensor<40xf32>
    %advnb4pg = stablehlo.add %advsb4pg, %advgb4pg : tensor<40xf32>
    %adbc1b4pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adbc2b4pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admhb4pg = stablehlo.divide %admnb4pg, %adbc1b4pg : tensor<40xf32>
    %advhb4pg = stablehlo.divide %advnb4pg, %adbc2b4pg : tensor<40xf32>
    %adlrb4pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adepsb4pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adsqb4pg = stablehlo.sqrt %advhb4pg : tensor<40xf32>
    %addenb4pg = stablehlo.add %adsqb4pg, %adepsb4pg : tensor<40xf32>
    %adratb4pg = stablehlo.divide %admhb4pg, %addenb4pg : tensor<40xf32>
    %adstb4pg = stablehlo.multiply %adlrb4pg, %adratb4pg : tensor<40xf32>
    %adsubb4pg = stablehlo.subtract %b4pg, %adstb4pg : tensor<40xf32>
    %adwdb4pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adwdlrb4pg = stablehlo.multiply %adwdb4pg, %adlrb4pg : tensor<40xf32>
    %adwdpb4pg = stablehlo.multiply %adwdlrb4pg, %b4pg : tensor<40xf32>
    %adnewb4pg = stablehlo.subtract %adsubb4pg, %adwdpb4pg : tensor<40xf32>
    %adb1b4pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob1b4pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admsb4pbt = stablehlo.multiply %adb1b4pbt, %b4pbtm : tensor<40xf32>
    %admgb4pbt = stablehlo.multiply %adob1b4pbt, %b4dpndb : tensor<40xf32>
    %admnb4pbt = stablehlo.add %admsb4pbt, %admgb4pbt : tensor<40xf32>
    %adb2b4pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob2b4pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %advsb4pbt = stablehlo.multiply %adb2b4pbt, %b4pbtv : tensor<40xf32>
    %adg2b4pbt = stablehlo.multiply %b4dpndb, %b4dpndb : tensor<40xf32>
    %advgb4pbt = stablehlo.multiply %adob2b4pbt, %adg2b4pbt : tensor<40xf32>
    %advnb4pbt = stablehlo.add %advsb4pbt, %advgb4pbt : tensor<40xf32>
    %adbc1b4pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adbc2b4pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admhb4pbt = stablehlo.divide %admnb4pbt, %adbc1b4pbt : tensor<40xf32>
    %advhb4pbt = stablehlo.divide %advnb4pbt, %adbc2b4pbt : tensor<40xf32>
    %adlrb4pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adepsb4pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adsqb4pbt = stablehlo.sqrt %advhb4pbt : tensor<40xf32>
    %addenb4pbt = stablehlo.add %adsqb4pbt, %adepsb4pbt : tensor<40xf32>
    %adratb4pbt = stablehlo.divide %admhb4pbt, %addenb4pbt : tensor<40xf32>
    %adstb4pbt = stablehlo.multiply %adlrb4pbt, %adratb4pbt : tensor<40xf32>
    %adsubb4pbt = stablehlo.subtract %b4pbt, %adstb4pbt : tensor<40xf32>
    %adwdb4pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adwdlrb4pbt = stablehlo.multiply %adwdb4pbt, %adlrb4pbt : tensor<40xf32>
    %adwdpb4pbt = stablehlo.multiply %adwdlrb4pbt, %b4pbt : tensor<40xf32>
    %adnewb4pbt = stablehlo.subtract %adsubb4pbt, %adwdpb4pbt : tensor<40xf32>
    %adb1b5eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adob1b5eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %admsb5eW = stablehlo.multiply %adb1b5eW, %b5eWm : tensor<240x40x1x1xf32>
    %admgb5eW = stablehlo.multiply %adob1b5eW, %b5deW : tensor<240x40x1x1xf32>
    %admnb5eW = stablehlo.add %admsb5eW, %admgb5eW : tensor<240x40x1x1xf32>
    %adb2b5eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adob2b5eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %advsb5eW = stablehlo.multiply %adb2b5eW, %b5eWv : tensor<240x40x1x1xf32>
    %adg2b5eW = stablehlo.multiply %b5deW, %b5deW : tensor<240x40x1x1xf32>
    %advgb5eW = stablehlo.multiply %adob2b5eW, %adg2b5eW : tensor<240x40x1x1xf32>
    %advnb5eW = stablehlo.add %advsb5eW, %advgb5eW : tensor<240x40x1x1xf32>
    %adbc1b5eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adbc2b5eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %admhb5eW = stablehlo.divide %admnb5eW, %adbc1b5eW : tensor<240x40x1x1xf32>
    %advhb5eW = stablehlo.divide %advnb5eW, %adbc2b5eW : tensor<240x40x1x1xf32>
    %adlrb5eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adepsb5eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adsqb5eW = stablehlo.sqrt %advhb5eW : tensor<240x40x1x1xf32>
    %addenb5eW = stablehlo.add %adsqb5eW, %adepsb5eW : tensor<240x40x1x1xf32>
    %adratb5eW = stablehlo.divide %admhb5eW, %addenb5eW : tensor<240x40x1x1xf32>
    %adstb5eW = stablehlo.multiply %adlrb5eW, %adratb5eW : tensor<240x40x1x1xf32>
    %adsubb5eW = stablehlo.subtract %b5eW, %adstb5eW : tensor<240x40x1x1xf32>
    %adwdb5eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adwdlrb5eW = stablehlo.multiply %adwdb5eW, %adlrb5eW : tensor<240x40x1x1xf32>
    %adwdpb5eW = stablehlo.multiply %adwdlrb5eW, %b5eW : tensor<240x40x1x1xf32>
    %adnewb5eW = stablehlo.subtract %adsubb5eW, %adwdpb5eW : tensor<240x40x1x1xf32>
    %adb1b5eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b5eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb5eb = stablehlo.multiply %adb1b5eb, %b5ebm : tensor<240xf32>
    %admgb5eb = stablehlo.multiply %adob1b5eb, %b5deb : tensor<240xf32>
    %admnb5eb = stablehlo.add %admsb5eb, %admgb5eb : tensor<240xf32>
    %adb2b5eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b5eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb5eb = stablehlo.multiply %adb2b5eb, %b5ebv : tensor<240xf32>
    %adg2b5eb = stablehlo.multiply %b5deb, %b5deb : tensor<240xf32>
    %advgb5eb = stablehlo.multiply %adob2b5eb, %adg2b5eb : tensor<240xf32>
    %advnb5eb = stablehlo.add %advsb5eb, %advgb5eb : tensor<240xf32>
    %adbc1b5eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b5eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb5eb = stablehlo.divide %admnb5eb, %adbc1b5eb : tensor<240xf32>
    %advhb5eb = stablehlo.divide %advnb5eb, %adbc2b5eb : tensor<240xf32>
    %adlrb5eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb5eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb5eb = stablehlo.sqrt %advhb5eb : tensor<240xf32>
    %addenb5eb = stablehlo.add %adsqb5eb, %adepsb5eb : tensor<240xf32>
    %adratb5eb = stablehlo.divide %admhb5eb, %addenb5eb : tensor<240xf32>
    %adstb5eb = stablehlo.multiply %adlrb5eb, %adratb5eb : tensor<240xf32>
    %adsubb5eb = stablehlo.subtract %b5eb, %adstb5eb : tensor<240xf32>
    %adwdb5eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb5eb = stablehlo.multiply %adwdb5eb, %adlrb5eb : tensor<240xf32>
    %adwdpb5eb = stablehlo.multiply %adwdlrb5eb, %b5eb : tensor<240xf32>
    %adnewb5eb = stablehlo.subtract %adsubb5eb, %adwdpb5eb : tensor<240xf32>
    %adb1b5eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b5eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb5eg = stablehlo.multiply %adb1b5eg, %b5egm : tensor<240xf32>
    %admgb5eg = stablehlo.multiply %adob1b5eg, %b5dendg : tensor<240xf32>
    %admnb5eg = stablehlo.add %admsb5eg, %admgb5eg : tensor<240xf32>
    %adb2b5eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b5eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb5eg = stablehlo.multiply %adb2b5eg, %b5egv : tensor<240xf32>
    %adg2b5eg = stablehlo.multiply %b5dendg, %b5dendg : tensor<240xf32>
    %advgb5eg = stablehlo.multiply %adob2b5eg, %adg2b5eg : tensor<240xf32>
    %advnb5eg = stablehlo.add %advsb5eg, %advgb5eg : tensor<240xf32>
    %adbc1b5eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b5eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb5eg = stablehlo.divide %admnb5eg, %adbc1b5eg : tensor<240xf32>
    %advhb5eg = stablehlo.divide %advnb5eg, %adbc2b5eg : tensor<240xf32>
    %adlrb5eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb5eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb5eg = stablehlo.sqrt %advhb5eg : tensor<240xf32>
    %addenb5eg = stablehlo.add %adsqb5eg, %adepsb5eg : tensor<240xf32>
    %adratb5eg = stablehlo.divide %admhb5eg, %addenb5eg : tensor<240xf32>
    %adstb5eg = stablehlo.multiply %adlrb5eg, %adratb5eg : tensor<240xf32>
    %adsubb5eg = stablehlo.subtract %b5eg, %adstb5eg : tensor<240xf32>
    %adwdb5eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb5eg = stablehlo.multiply %adwdb5eg, %adlrb5eg : tensor<240xf32>
    %adwdpb5eg = stablehlo.multiply %adwdlrb5eg, %b5eg : tensor<240xf32>
    %adnewb5eg = stablehlo.subtract %adsubb5eg, %adwdpb5eg : tensor<240xf32>
    %adb1b5ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b5ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb5ebt = stablehlo.multiply %adb1b5ebt, %b5ebtm : tensor<240xf32>
    %admgb5ebt = stablehlo.multiply %adob1b5ebt, %b5dendb : tensor<240xf32>
    %admnb5ebt = stablehlo.add %admsb5ebt, %admgb5ebt : tensor<240xf32>
    %adb2b5ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b5ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb5ebt = stablehlo.multiply %adb2b5ebt, %b5ebtv : tensor<240xf32>
    %adg2b5ebt = stablehlo.multiply %b5dendb, %b5dendb : tensor<240xf32>
    %advgb5ebt = stablehlo.multiply %adob2b5ebt, %adg2b5ebt : tensor<240xf32>
    %advnb5ebt = stablehlo.add %advsb5ebt, %advgb5ebt : tensor<240xf32>
    %adbc1b5ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b5ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb5ebt = stablehlo.divide %admnb5ebt, %adbc1b5ebt : tensor<240xf32>
    %advhb5ebt = stablehlo.divide %advnb5ebt, %adbc2b5ebt : tensor<240xf32>
    %adlrb5ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb5ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb5ebt = stablehlo.sqrt %advhb5ebt : tensor<240xf32>
    %addenb5ebt = stablehlo.add %adsqb5ebt, %adepsb5ebt : tensor<240xf32>
    %adratb5ebt = stablehlo.divide %admhb5ebt, %addenb5ebt : tensor<240xf32>
    %adstb5ebt = stablehlo.multiply %adlrb5ebt, %adratb5ebt : tensor<240xf32>
    %adsubb5ebt = stablehlo.subtract %b5ebt, %adstb5ebt : tensor<240xf32>
    %adwdb5ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb5ebt = stablehlo.multiply %adwdb5ebt, %adlrb5ebt : tensor<240xf32>
    %adwdpb5ebt = stablehlo.multiply %adwdlrb5ebt, %b5ebt : tensor<240xf32>
    %adnewb5ebt = stablehlo.subtract %adsubb5ebt, %adwdpb5ebt : tensor<240xf32>
    %adb1b5dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %adob1b5dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %admsb5dW = stablehlo.multiply %adb1b5dW, %b5dWm : tensor<240x1x5x5xf32>
    %admgb5dW = stablehlo.multiply %adob1b5dW, %b5ddW : tensor<240x1x5x5xf32>
    %admnb5dW = stablehlo.add %admsb5dW, %admgb5dW : tensor<240x1x5x5xf32>
    %adb2b5dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %adob2b5dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %advsb5dW = stablehlo.multiply %adb2b5dW, %b5dWv : tensor<240x1x5x5xf32>
    %adg2b5dW = stablehlo.multiply %b5ddW, %b5ddW : tensor<240x1x5x5xf32>
    %advgb5dW = stablehlo.multiply %adob2b5dW, %adg2b5dW : tensor<240x1x5x5xf32>
    %advnb5dW = stablehlo.add %advsb5dW, %advgb5dW : tensor<240x1x5x5xf32>
    %adbc1b5dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %adbc2b5dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %admhb5dW = stablehlo.divide %admnb5dW, %adbc1b5dW : tensor<240x1x5x5xf32>
    %advhb5dW = stablehlo.divide %advnb5dW, %adbc2b5dW : tensor<240x1x5x5xf32>
    %adlrb5dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %adepsb5dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %adsqb5dW = stablehlo.sqrt %advhb5dW : tensor<240x1x5x5xf32>
    %addenb5dW = stablehlo.add %adsqb5dW, %adepsb5dW : tensor<240x1x5x5xf32>
    %adratb5dW = stablehlo.divide %admhb5dW, %addenb5dW : tensor<240x1x5x5xf32>
    %adstb5dW = stablehlo.multiply %adlrb5dW, %adratb5dW : tensor<240x1x5x5xf32>
    %adsubb5dW = stablehlo.subtract %b5dW, %adstb5dW : tensor<240x1x5x5xf32>
    %adwdb5dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240x1x5x5xf32>
    %adwdlrb5dW = stablehlo.multiply %adwdb5dW, %adlrb5dW : tensor<240x1x5x5xf32>
    %adwdpb5dW = stablehlo.multiply %adwdlrb5dW, %b5dW : tensor<240x1x5x5xf32>
    %adnewb5dW = stablehlo.subtract %adsubb5dW, %adwdpb5dW : tensor<240x1x5x5xf32>
    %adb1b5db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b5db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb5db = stablehlo.multiply %adb1b5db, %b5dbm : tensor<240xf32>
    %admgb5db = stablehlo.multiply %adob1b5db, %b5ddb : tensor<240xf32>
    %admnb5db = stablehlo.add %admsb5db, %admgb5db : tensor<240xf32>
    %adb2b5db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b5db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb5db = stablehlo.multiply %adb2b5db, %b5dbv : tensor<240xf32>
    %adg2b5db = stablehlo.multiply %b5ddb, %b5ddb : tensor<240xf32>
    %advgb5db = stablehlo.multiply %adob2b5db, %adg2b5db : tensor<240xf32>
    %advnb5db = stablehlo.add %advsb5db, %advgb5db : tensor<240xf32>
    %adbc1b5db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b5db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb5db = stablehlo.divide %admnb5db, %adbc1b5db : tensor<240xf32>
    %advhb5db = stablehlo.divide %advnb5db, %adbc2b5db : tensor<240xf32>
    %adlrb5db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb5db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb5db = stablehlo.sqrt %advhb5db : tensor<240xf32>
    %addenb5db = stablehlo.add %adsqb5db, %adepsb5db : tensor<240xf32>
    %adratb5db = stablehlo.divide %admhb5db, %addenb5db : tensor<240xf32>
    %adstb5db = stablehlo.multiply %adlrb5db, %adratb5db : tensor<240xf32>
    %adsubb5db = stablehlo.subtract %b5db, %adstb5db : tensor<240xf32>
    %adwdb5db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb5db = stablehlo.multiply %adwdb5db, %adlrb5db : tensor<240xf32>
    %adwdpb5db = stablehlo.multiply %adwdlrb5db, %b5db : tensor<240xf32>
    %adnewb5db = stablehlo.subtract %adsubb5db, %adwdpb5db : tensor<240xf32>
    %adb1b5dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b5dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb5dg = stablehlo.multiply %adb1b5dg, %b5dgm : tensor<240xf32>
    %admgb5dg = stablehlo.multiply %adob1b5dg, %b5ddndg : tensor<240xf32>
    %admnb5dg = stablehlo.add %admsb5dg, %admgb5dg : tensor<240xf32>
    %adb2b5dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b5dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb5dg = stablehlo.multiply %adb2b5dg, %b5dgv : tensor<240xf32>
    %adg2b5dg = stablehlo.multiply %b5ddndg, %b5ddndg : tensor<240xf32>
    %advgb5dg = stablehlo.multiply %adob2b5dg, %adg2b5dg : tensor<240xf32>
    %advnb5dg = stablehlo.add %advsb5dg, %advgb5dg : tensor<240xf32>
    %adbc1b5dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b5dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb5dg = stablehlo.divide %admnb5dg, %adbc1b5dg : tensor<240xf32>
    %advhb5dg = stablehlo.divide %advnb5dg, %adbc2b5dg : tensor<240xf32>
    %adlrb5dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb5dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb5dg = stablehlo.sqrt %advhb5dg : tensor<240xf32>
    %addenb5dg = stablehlo.add %adsqb5dg, %adepsb5dg : tensor<240xf32>
    %adratb5dg = stablehlo.divide %admhb5dg, %addenb5dg : tensor<240xf32>
    %adstb5dg = stablehlo.multiply %adlrb5dg, %adratb5dg : tensor<240xf32>
    %adsubb5dg = stablehlo.subtract %b5dg, %adstb5dg : tensor<240xf32>
    %adwdb5dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb5dg = stablehlo.multiply %adwdb5dg, %adlrb5dg : tensor<240xf32>
    %adwdpb5dg = stablehlo.multiply %adwdlrb5dg, %b5dg : tensor<240xf32>
    %adnewb5dg = stablehlo.subtract %adsubb5dg, %adwdpb5dg : tensor<240xf32>
    %adb1b5dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b5dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb5dbt = stablehlo.multiply %adb1b5dbt, %b5dbtm : tensor<240xf32>
    %admgb5dbt = stablehlo.multiply %adob1b5dbt, %b5ddndb : tensor<240xf32>
    %admnb5dbt = stablehlo.add %admsb5dbt, %admgb5dbt : tensor<240xf32>
    %adb2b5dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b5dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb5dbt = stablehlo.multiply %adb2b5dbt, %b5dbtv : tensor<240xf32>
    %adg2b5dbt = stablehlo.multiply %b5ddndb, %b5ddndb : tensor<240xf32>
    %advgb5dbt = stablehlo.multiply %adob2b5dbt, %adg2b5dbt : tensor<240xf32>
    %advnb5dbt = stablehlo.add %advsb5dbt, %advgb5dbt : tensor<240xf32>
    %adbc1b5dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b5dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb5dbt = stablehlo.divide %admnb5dbt, %adbc1b5dbt : tensor<240xf32>
    %advhb5dbt = stablehlo.divide %advnb5dbt, %adbc2b5dbt : tensor<240xf32>
    %adlrb5dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb5dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb5dbt = stablehlo.sqrt %advhb5dbt : tensor<240xf32>
    %addenb5dbt = stablehlo.add %adsqb5dbt, %adepsb5dbt : tensor<240xf32>
    %adratb5dbt = stablehlo.divide %admhb5dbt, %addenb5dbt : tensor<240xf32>
    %adstb5dbt = stablehlo.multiply %adlrb5dbt, %adratb5dbt : tensor<240xf32>
    %adsubb5dbt = stablehlo.subtract %b5dbt, %adstb5dbt : tensor<240xf32>
    %adwdb5dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb5dbt = stablehlo.multiply %adwdb5dbt, %adlrb5dbt : tensor<240xf32>
    %adwdpb5dbt = stablehlo.multiply %adwdlrb5dbt, %b5dbt : tensor<240xf32>
    %adnewb5dbt = stablehlo.subtract %adsubb5dbt, %adwdpb5dbt : tensor<240xf32>
    %adb1b5zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adob1b5zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %admsb5zW1 = stablehlo.multiply %adb1b5zW1, %b5zW1m : tensor<240x10xf32>
    %admgb5zW1 = stablehlo.multiply %adob1b5zW1, %b5zdWs1 : tensor<240x10xf32>
    %admnb5zW1 = stablehlo.add %admsb5zW1, %admgb5zW1 : tensor<240x10xf32>
    %adb2b5zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adob2b5zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %advsb5zW1 = stablehlo.multiply %adb2b5zW1, %b5zW1v : tensor<240x10xf32>
    %adg2b5zW1 = stablehlo.multiply %b5zdWs1, %b5zdWs1 : tensor<240x10xf32>
    %advgb5zW1 = stablehlo.multiply %adob2b5zW1, %adg2b5zW1 : tensor<240x10xf32>
    %advnb5zW1 = stablehlo.add %advsb5zW1, %advgb5zW1 : tensor<240x10xf32>
    %adbc1b5zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adbc2b5zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %admhb5zW1 = stablehlo.divide %admnb5zW1, %adbc1b5zW1 : tensor<240x10xf32>
    %advhb5zW1 = stablehlo.divide %advnb5zW1, %adbc2b5zW1 : tensor<240x10xf32>
    %adlrb5zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adepsb5zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adsqb5zW1 = stablehlo.sqrt %advhb5zW1 : tensor<240x10xf32>
    %addenb5zW1 = stablehlo.add %adsqb5zW1, %adepsb5zW1 : tensor<240x10xf32>
    %adratb5zW1 = stablehlo.divide %admhb5zW1, %addenb5zW1 : tensor<240x10xf32>
    %adstb5zW1 = stablehlo.multiply %adlrb5zW1, %adratb5zW1 : tensor<240x10xf32>
    %adsubb5zW1 = stablehlo.subtract %b5zW1, %adstb5zW1 : tensor<240x10xf32>
    %adwdb5zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adwdlrb5zW1 = stablehlo.multiply %adwdb5zW1, %adlrb5zW1 : tensor<240x10xf32>
    %adwdpb5zW1 = stablehlo.multiply %adwdlrb5zW1, %b5zW1 : tensor<240x10xf32>
    %adnewb5zW1 = stablehlo.subtract %adsubb5zW1, %adwdpb5zW1 : tensor<240x10xf32>
    %adb1b5zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob1b5zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admsb5zb1 = stablehlo.multiply %adb1b5zb1, %b5zb1m : tensor<10xf32>
    %admgb5zb1 = stablehlo.multiply %adob1b5zb1, %b5zdbs1 : tensor<10xf32>
    %admnb5zb1 = stablehlo.add %admsb5zb1, %admgb5zb1 : tensor<10xf32>
    %adb2b5zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob2b5zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %advsb5zb1 = stablehlo.multiply %adb2b5zb1, %b5zb1v : tensor<10xf32>
    %adg2b5zb1 = stablehlo.multiply %b5zdbs1, %b5zdbs1 : tensor<10xf32>
    %advgb5zb1 = stablehlo.multiply %adob2b5zb1, %adg2b5zb1 : tensor<10xf32>
    %advnb5zb1 = stablehlo.add %advsb5zb1, %advgb5zb1 : tensor<10xf32>
    %adbc1b5zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adbc2b5zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admhb5zb1 = stablehlo.divide %admnb5zb1, %adbc1b5zb1 : tensor<10xf32>
    %advhb5zb1 = stablehlo.divide %advnb5zb1, %adbc2b5zb1 : tensor<10xf32>
    %adlrb5zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adepsb5zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adsqb5zb1 = stablehlo.sqrt %advhb5zb1 : tensor<10xf32>
    %addenb5zb1 = stablehlo.add %adsqb5zb1, %adepsb5zb1 : tensor<10xf32>
    %adratb5zb1 = stablehlo.divide %admhb5zb1, %addenb5zb1 : tensor<10xf32>
    %adstb5zb1 = stablehlo.multiply %adlrb5zb1, %adratb5zb1 : tensor<10xf32>
    %adsubb5zb1 = stablehlo.subtract %b5zb1, %adstb5zb1 : tensor<10xf32>
    %adwdb5zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adwdlrb5zb1 = stablehlo.multiply %adwdb5zb1, %adlrb5zb1 : tensor<10xf32>
    %adwdpb5zb1 = stablehlo.multiply %adwdlrb5zb1, %b5zb1 : tensor<10xf32>
    %adnewb5zb1 = stablehlo.subtract %adsubb5zb1, %adwdpb5zb1 : tensor<10xf32>
    %adb1b5zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adob1b5zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %admsb5zW2 = stablehlo.multiply %adb1b5zW2, %b5zW2m : tensor<10x240xf32>
    %admgb5zW2 = stablehlo.multiply %adob1b5zW2, %b5zdWs2 : tensor<10x240xf32>
    %admnb5zW2 = stablehlo.add %admsb5zW2, %admgb5zW2 : tensor<10x240xf32>
    %adb2b5zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adob2b5zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %advsb5zW2 = stablehlo.multiply %adb2b5zW2, %b5zW2v : tensor<10x240xf32>
    %adg2b5zW2 = stablehlo.multiply %b5zdWs2, %b5zdWs2 : tensor<10x240xf32>
    %advgb5zW2 = stablehlo.multiply %adob2b5zW2, %adg2b5zW2 : tensor<10x240xf32>
    %advnb5zW2 = stablehlo.add %advsb5zW2, %advgb5zW2 : tensor<10x240xf32>
    %adbc1b5zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adbc2b5zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %admhb5zW2 = stablehlo.divide %admnb5zW2, %adbc1b5zW2 : tensor<10x240xf32>
    %advhb5zW2 = stablehlo.divide %advnb5zW2, %adbc2b5zW2 : tensor<10x240xf32>
    %adlrb5zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adepsb5zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adsqb5zW2 = stablehlo.sqrt %advhb5zW2 : tensor<10x240xf32>
    %addenb5zW2 = stablehlo.add %adsqb5zW2, %adepsb5zW2 : tensor<10x240xf32>
    %adratb5zW2 = stablehlo.divide %admhb5zW2, %addenb5zW2 : tensor<10x240xf32>
    %adstb5zW2 = stablehlo.multiply %adlrb5zW2, %adratb5zW2 : tensor<10x240xf32>
    %adsubb5zW2 = stablehlo.subtract %b5zW2, %adstb5zW2 : tensor<10x240xf32>
    %adwdb5zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adwdlrb5zW2 = stablehlo.multiply %adwdb5zW2, %adlrb5zW2 : tensor<10x240xf32>
    %adwdpb5zW2 = stablehlo.multiply %adwdlrb5zW2, %b5zW2 : tensor<10x240xf32>
    %adnewb5zW2 = stablehlo.subtract %adsubb5zW2, %adwdpb5zW2 : tensor<10x240xf32>
    %adb1b5zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b5zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb5zb2 = stablehlo.multiply %adb1b5zb2, %b5zb2m : tensor<240xf32>
    %admgb5zb2 = stablehlo.multiply %adob1b5zb2, %b5zdbs2 : tensor<240xf32>
    %admnb5zb2 = stablehlo.add %admsb5zb2, %admgb5zb2 : tensor<240xf32>
    %adb2b5zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b5zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb5zb2 = stablehlo.multiply %adb2b5zb2, %b5zb2v : tensor<240xf32>
    %adg2b5zb2 = stablehlo.multiply %b5zdbs2, %b5zdbs2 : tensor<240xf32>
    %advgb5zb2 = stablehlo.multiply %adob2b5zb2, %adg2b5zb2 : tensor<240xf32>
    %advnb5zb2 = stablehlo.add %advsb5zb2, %advgb5zb2 : tensor<240xf32>
    %adbc1b5zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b5zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb5zb2 = stablehlo.divide %admnb5zb2, %adbc1b5zb2 : tensor<240xf32>
    %advhb5zb2 = stablehlo.divide %advnb5zb2, %adbc2b5zb2 : tensor<240xf32>
    %adlrb5zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb5zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb5zb2 = stablehlo.sqrt %advhb5zb2 : tensor<240xf32>
    %addenb5zb2 = stablehlo.add %adsqb5zb2, %adepsb5zb2 : tensor<240xf32>
    %adratb5zb2 = stablehlo.divide %admhb5zb2, %addenb5zb2 : tensor<240xf32>
    %adstb5zb2 = stablehlo.multiply %adlrb5zb2, %adratb5zb2 : tensor<240xf32>
    %adsubb5zb2 = stablehlo.subtract %b5zb2, %adstb5zb2 : tensor<240xf32>
    %adwdb5zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb5zb2 = stablehlo.multiply %adwdb5zb2, %adlrb5zb2 : tensor<240xf32>
    %adwdpb5zb2 = stablehlo.multiply %adwdlrb5zb2, %b5zb2 : tensor<240xf32>
    %adnewb5zb2 = stablehlo.subtract %adsubb5zb2, %adwdpb5zb2 : tensor<240xf32>
    %adb1b5pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %adob1b5pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %admsb5pW = stablehlo.multiply %adb1b5pW, %b5pWm : tensor<40x240x1x1xf32>
    %admgb5pW = stablehlo.multiply %adob1b5pW, %b5dpW : tensor<40x240x1x1xf32>
    %admnb5pW = stablehlo.add %admsb5pW, %admgb5pW : tensor<40x240x1x1xf32>
    %adb2b5pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %adob2b5pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %advsb5pW = stablehlo.multiply %adb2b5pW, %b5pWv : tensor<40x240x1x1xf32>
    %adg2b5pW = stablehlo.multiply %b5dpW, %b5dpW : tensor<40x240x1x1xf32>
    %advgb5pW = stablehlo.multiply %adob2b5pW, %adg2b5pW : tensor<40x240x1x1xf32>
    %advnb5pW = stablehlo.add %advsb5pW, %advgb5pW : tensor<40x240x1x1xf32>
    %adbc1b5pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %adbc2b5pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %admhb5pW = stablehlo.divide %admnb5pW, %adbc1b5pW : tensor<40x240x1x1xf32>
    %advhb5pW = stablehlo.divide %advnb5pW, %adbc2b5pW : tensor<40x240x1x1xf32>
    %adlrb5pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %adepsb5pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %adsqb5pW = stablehlo.sqrt %advhb5pW : tensor<40x240x1x1xf32>
    %addenb5pW = stablehlo.add %adsqb5pW, %adepsb5pW : tensor<40x240x1x1xf32>
    %adratb5pW = stablehlo.divide %admhb5pW, %addenb5pW : tensor<40x240x1x1xf32>
    %adstb5pW = stablehlo.multiply %adlrb5pW, %adratb5pW : tensor<40x240x1x1xf32>
    %adsubb5pW = stablehlo.subtract %b5pW, %adstb5pW : tensor<40x240x1x1xf32>
    %adwdb5pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<40x240x1x1xf32>
    %adwdlrb5pW = stablehlo.multiply %adwdb5pW, %adlrb5pW : tensor<40x240x1x1xf32>
    %adwdpb5pW = stablehlo.multiply %adwdlrb5pW, %b5pW : tensor<40x240x1x1xf32>
    %adnewb5pW = stablehlo.subtract %adsubb5pW, %adwdpb5pW : tensor<40x240x1x1xf32>
    %adb1b5pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob1b5pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admsb5pb = stablehlo.multiply %adb1b5pb, %b5pbm : tensor<40xf32>
    %admgb5pb = stablehlo.multiply %adob1b5pb, %b5dpb : tensor<40xf32>
    %admnb5pb = stablehlo.add %admsb5pb, %admgb5pb : tensor<40xf32>
    %adb2b5pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob2b5pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %advsb5pb = stablehlo.multiply %adb2b5pb, %b5pbv : tensor<40xf32>
    %adg2b5pb = stablehlo.multiply %b5dpb, %b5dpb : tensor<40xf32>
    %advgb5pb = stablehlo.multiply %adob2b5pb, %adg2b5pb : tensor<40xf32>
    %advnb5pb = stablehlo.add %advsb5pb, %advgb5pb : tensor<40xf32>
    %adbc1b5pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adbc2b5pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admhb5pb = stablehlo.divide %admnb5pb, %adbc1b5pb : tensor<40xf32>
    %advhb5pb = stablehlo.divide %advnb5pb, %adbc2b5pb : tensor<40xf32>
    %adlrb5pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adepsb5pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adsqb5pb = stablehlo.sqrt %advhb5pb : tensor<40xf32>
    %addenb5pb = stablehlo.add %adsqb5pb, %adepsb5pb : tensor<40xf32>
    %adratb5pb = stablehlo.divide %admhb5pb, %addenb5pb : tensor<40xf32>
    %adstb5pb = stablehlo.multiply %adlrb5pb, %adratb5pb : tensor<40xf32>
    %adsubb5pb = stablehlo.subtract %b5pb, %adstb5pb : tensor<40xf32>
    %adwdb5pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adwdlrb5pb = stablehlo.multiply %adwdb5pb, %adlrb5pb : tensor<40xf32>
    %adwdpb5pb = stablehlo.multiply %adwdlrb5pb, %b5pb : tensor<40xf32>
    %adnewb5pb = stablehlo.subtract %adsubb5pb, %adwdpb5pb : tensor<40xf32>
    %adb1b5pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob1b5pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admsb5pg = stablehlo.multiply %adb1b5pg, %b5pgm : tensor<40xf32>
    %admgb5pg = stablehlo.multiply %adob1b5pg, %b5dpndg : tensor<40xf32>
    %admnb5pg = stablehlo.add %admsb5pg, %admgb5pg : tensor<40xf32>
    %adb2b5pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob2b5pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %advsb5pg = stablehlo.multiply %adb2b5pg, %b5pgv : tensor<40xf32>
    %adg2b5pg = stablehlo.multiply %b5dpndg, %b5dpndg : tensor<40xf32>
    %advgb5pg = stablehlo.multiply %adob2b5pg, %adg2b5pg : tensor<40xf32>
    %advnb5pg = stablehlo.add %advsb5pg, %advgb5pg : tensor<40xf32>
    %adbc1b5pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adbc2b5pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admhb5pg = stablehlo.divide %admnb5pg, %adbc1b5pg : tensor<40xf32>
    %advhb5pg = stablehlo.divide %advnb5pg, %adbc2b5pg : tensor<40xf32>
    %adlrb5pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adepsb5pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adsqb5pg = stablehlo.sqrt %advhb5pg : tensor<40xf32>
    %addenb5pg = stablehlo.add %adsqb5pg, %adepsb5pg : tensor<40xf32>
    %adratb5pg = stablehlo.divide %admhb5pg, %addenb5pg : tensor<40xf32>
    %adstb5pg = stablehlo.multiply %adlrb5pg, %adratb5pg : tensor<40xf32>
    %adsubb5pg = stablehlo.subtract %b5pg, %adstb5pg : tensor<40xf32>
    %adwdb5pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adwdlrb5pg = stablehlo.multiply %adwdb5pg, %adlrb5pg : tensor<40xf32>
    %adwdpb5pg = stablehlo.multiply %adwdlrb5pg, %b5pg : tensor<40xf32>
    %adnewb5pg = stablehlo.subtract %adsubb5pg, %adwdpb5pg : tensor<40xf32>
    %adb1b5pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob1b5pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admsb5pbt = stablehlo.multiply %adb1b5pbt, %b5pbtm : tensor<40xf32>
    %admgb5pbt = stablehlo.multiply %adob1b5pbt, %b5dpndb : tensor<40xf32>
    %admnb5pbt = stablehlo.add %admsb5pbt, %admgb5pbt : tensor<40xf32>
    %adb2b5pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adob2b5pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %advsb5pbt = stablehlo.multiply %adb2b5pbt, %b5pbtv : tensor<40xf32>
    %adg2b5pbt = stablehlo.multiply %b5dpndb, %b5dpndb : tensor<40xf32>
    %advgb5pbt = stablehlo.multiply %adob2b5pbt, %adg2b5pbt : tensor<40xf32>
    %advnb5pbt = stablehlo.add %advsb5pbt, %advgb5pbt : tensor<40xf32>
    %adbc1b5pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adbc2b5pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %admhb5pbt = stablehlo.divide %admnb5pbt, %adbc1b5pbt : tensor<40xf32>
    %advhb5pbt = stablehlo.divide %advnb5pbt, %adbc2b5pbt : tensor<40xf32>
    %adlrb5pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adepsb5pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adsqb5pbt = stablehlo.sqrt %advhb5pbt : tensor<40xf32>
    %addenb5pbt = stablehlo.add %adsqb5pbt, %adepsb5pbt : tensor<40xf32>
    %adratb5pbt = stablehlo.divide %admhb5pbt, %addenb5pbt : tensor<40xf32>
    %adstb5pbt = stablehlo.multiply %adlrb5pbt, %adratb5pbt : tensor<40xf32>
    %adsubb5pbt = stablehlo.subtract %b5pbt, %adstb5pbt : tensor<40xf32>
    %adwdb5pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<40xf32>
    %adwdlrb5pbt = stablehlo.multiply %adwdb5pbt, %adlrb5pbt : tensor<40xf32>
    %adwdpb5pbt = stablehlo.multiply %adwdlrb5pbt, %b5pbt : tensor<40xf32>
    %adnewb5pbt = stablehlo.subtract %adsubb5pbt, %adwdpb5pbt : tensor<40xf32>
    %adb1b6eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adob1b6eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %admsb6eW = stablehlo.multiply %adb1b6eW, %b6eWm : tensor<240x40x1x1xf32>
    %admgb6eW = stablehlo.multiply %adob1b6eW, %b6deW : tensor<240x40x1x1xf32>
    %admnb6eW = stablehlo.add %admsb6eW, %admgb6eW : tensor<240x40x1x1xf32>
    %adb2b6eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adob2b6eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %advsb6eW = stablehlo.multiply %adb2b6eW, %b6eWv : tensor<240x40x1x1xf32>
    %adg2b6eW = stablehlo.multiply %b6deW, %b6deW : tensor<240x40x1x1xf32>
    %advgb6eW = stablehlo.multiply %adob2b6eW, %adg2b6eW : tensor<240x40x1x1xf32>
    %advnb6eW = stablehlo.add %advsb6eW, %advgb6eW : tensor<240x40x1x1xf32>
    %adbc1b6eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adbc2b6eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %admhb6eW = stablehlo.divide %admnb6eW, %adbc1b6eW : tensor<240x40x1x1xf32>
    %advhb6eW = stablehlo.divide %advnb6eW, %adbc2b6eW : tensor<240x40x1x1xf32>
    %adlrb6eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adepsb6eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adsqb6eW = stablehlo.sqrt %advhb6eW : tensor<240x40x1x1xf32>
    %addenb6eW = stablehlo.add %adsqb6eW, %adepsb6eW : tensor<240x40x1x1xf32>
    %adratb6eW = stablehlo.divide %admhb6eW, %addenb6eW : tensor<240x40x1x1xf32>
    %adstb6eW = stablehlo.multiply %adlrb6eW, %adratb6eW : tensor<240x40x1x1xf32>
    %adsubb6eW = stablehlo.subtract %b6eW, %adstb6eW : tensor<240x40x1x1xf32>
    %adwdb6eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240x40x1x1xf32>
    %adwdlrb6eW = stablehlo.multiply %adwdb6eW, %adlrb6eW : tensor<240x40x1x1xf32>
    %adwdpb6eW = stablehlo.multiply %adwdlrb6eW, %b6eW : tensor<240x40x1x1xf32>
    %adnewb6eW = stablehlo.subtract %adsubb6eW, %adwdpb6eW : tensor<240x40x1x1xf32>
    %adb1b6eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b6eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb6eb = stablehlo.multiply %adb1b6eb, %b6ebm : tensor<240xf32>
    %admgb6eb = stablehlo.multiply %adob1b6eb, %b6deb : tensor<240xf32>
    %admnb6eb = stablehlo.add %admsb6eb, %admgb6eb : tensor<240xf32>
    %adb2b6eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b6eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb6eb = stablehlo.multiply %adb2b6eb, %b6ebv : tensor<240xf32>
    %adg2b6eb = stablehlo.multiply %b6deb, %b6deb : tensor<240xf32>
    %advgb6eb = stablehlo.multiply %adob2b6eb, %adg2b6eb : tensor<240xf32>
    %advnb6eb = stablehlo.add %advsb6eb, %advgb6eb : tensor<240xf32>
    %adbc1b6eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b6eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb6eb = stablehlo.divide %admnb6eb, %adbc1b6eb : tensor<240xf32>
    %advhb6eb = stablehlo.divide %advnb6eb, %adbc2b6eb : tensor<240xf32>
    %adlrb6eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb6eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb6eb = stablehlo.sqrt %advhb6eb : tensor<240xf32>
    %addenb6eb = stablehlo.add %adsqb6eb, %adepsb6eb : tensor<240xf32>
    %adratb6eb = stablehlo.divide %admhb6eb, %addenb6eb : tensor<240xf32>
    %adstb6eb = stablehlo.multiply %adlrb6eb, %adratb6eb : tensor<240xf32>
    %adsubb6eb = stablehlo.subtract %b6eb, %adstb6eb : tensor<240xf32>
    %adwdb6eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb6eb = stablehlo.multiply %adwdb6eb, %adlrb6eb : tensor<240xf32>
    %adwdpb6eb = stablehlo.multiply %adwdlrb6eb, %b6eb : tensor<240xf32>
    %adnewb6eb = stablehlo.subtract %adsubb6eb, %adwdpb6eb : tensor<240xf32>
    %adb1b6eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b6eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb6eg = stablehlo.multiply %adb1b6eg, %b6egm : tensor<240xf32>
    %admgb6eg = stablehlo.multiply %adob1b6eg, %b6dendg : tensor<240xf32>
    %admnb6eg = stablehlo.add %admsb6eg, %admgb6eg : tensor<240xf32>
    %adb2b6eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b6eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb6eg = stablehlo.multiply %adb2b6eg, %b6egv : tensor<240xf32>
    %adg2b6eg = stablehlo.multiply %b6dendg, %b6dendg : tensor<240xf32>
    %advgb6eg = stablehlo.multiply %adob2b6eg, %adg2b6eg : tensor<240xf32>
    %advnb6eg = stablehlo.add %advsb6eg, %advgb6eg : tensor<240xf32>
    %adbc1b6eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b6eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb6eg = stablehlo.divide %admnb6eg, %adbc1b6eg : tensor<240xf32>
    %advhb6eg = stablehlo.divide %advnb6eg, %adbc2b6eg : tensor<240xf32>
    %adlrb6eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb6eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb6eg = stablehlo.sqrt %advhb6eg : tensor<240xf32>
    %addenb6eg = stablehlo.add %adsqb6eg, %adepsb6eg : tensor<240xf32>
    %adratb6eg = stablehlo.divide %admhb6eg, %addenb6eg : tensor<240xf32>
    %adstb6eg = stablehlo.multiply %adlrb6eg, %adratb6eg : tensor<240xf32>
    %adsubb6eg = stablehlo.subtract %b6eg, %adstb6eg : tensor<240xf32>
    %adwdb6eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb6eg = stablehlo.multiply %adwdb6eg, %adlrb6eg : tensor<240xf32>
    %adwdpb6eg = stablehlo.multiply %adwdlrb6eg, %b6eg : tensor<240xf32>
    %adnewb6eg = stablehlo.subtract %adsubb6eg, %adwdpb6eg : tensor<240xf32>
    %adb1b6ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b6ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb6ebt = stablehlo.multiply %adb1b6ebt, %b6ebtm : tensor<240xf32>
    %admgb6ebt = stablehlo.multiply %adob1b6ebt, %b6dendb : tensor<240xf32>
    %admnb6ebt = stablehlo.add %admsb6ebt, %admgb6ebt : tensor<240xf32>
    %adb2b6ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b6ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb6ebt = stablehlo.multiply %adb2b6ebt, %b6ebtv : tensor<240xf32>
    %adg2b6ebt = stablehlo.multiply %b6dendb, %b6dendb : tensor<240xf32>
    %advgb6ebt = stablehlo.multiply %adob2b6ebt, %adg2b6ebt : tensor<240xf32>
    %advnb6ebt = stablehlo.add %advsb6ebt, %advgb6ebt : tensor<240xf32>
    %adbc1b6ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b6ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb6ebt = stablehlo.divide %admnb6ebt, %adbc1b6ebt : tensor<240xf32>
    %advhb6ebt = stablehlo.divide %advnb6ebt, %adbc2b6ebt : tensor<240xf32>
    %adlrb6ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb6ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb6ebt = stablehlo.sqrt %advhb6ebt : tensor<240xf32>
    %addenb6ebt = stablehlo.add %adsqb6ebt, %adepsb6ebt : tensor<240xf32>
    %adratb6ebt = stablehlo.divide %admhb6ebt, %addenb6ebt : tensor<240xf32>
    %adstb6ebt = stablehlo.multiply %adlrb6ebt, %adratb6ebt : tensor<240xf32>
    %adsubb6ebt = stablehlo.subtract %b6ebt, %adstb6ebt : tensor<240xf32>
    %adwdb6ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb6ebt = stablehlo.multiply %adwdb6ebt, %adlrb6ebt : tensor<240xf32>
    %adwdpb6ebt = stablehlo.multiply %adwdlrb6ebt, %b6ebt : tensor<240xf32>
    %adnewb6ebt = stablehlo.subtract %adsubb6ebt, %adwdpb6ebt : tensor<240xf32>
    %adb1b6dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %adob1b6dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %admsb6dW = stablehlo.multiply %adb1b6dW, %b6dWm : tensor<240x1x3x3xf32>
    %admgb6dW = stablehlo.multiply %adob1b6dW, %b6ddW : tensor<240x1x3x3xf32>
    %admnb6dW = stablehlo.add %admsb6dW, %admgb6dW : tensor<240x1x3x3xf32>
    %adb2b6dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %adob2b6dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %advsb6dW = stablehlo.multiply %adb2b6dW, %b6dWv : tensor<240x1x3x3xf32>
    %adg2b6dW = stablehlo.multiply %b6ddW, %b6ddW : tensor<240x1x3x3xf32>
    %advgb6dW = stablehlo.multiply %adob2b6dW, %adg2b6dW : tensor<240x1x3x3xf32>
    %advnb6dW = stablehlo.add %advsb6dW, %advgb6dW : tensor<240x1x3x3xf32>
    %adbc1b6dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %adbc2b6dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %admhb6dW = stablehlo.divide %admnb6dW, %adbc1b6dW : tensor<240x1x3x3xf32>
    %advhb6dW = stablehlo.divide %advnb6dW, %adbc2b6dW : tensor<240x1x3x3xf32>
    %adlrb6dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %adepsb6dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %adsqb6dW = stablehlo.sqrt %advhb6dW : tensor<240x1x3x3xf32>
    %addenb6dW = stablehlo.add %adsqb6dW, %adepsb6dW : tensor<240x1x3x3xf32>
    %adratb6dW = stablehlo.divide %admhb6dW, %addenb6dW : tensor<240x1x3x3xf32>
    %adstb6dW = stablehlo.multiply %adlrb6dW, %adratb6dW : tensor<240x1x3x3xf32>
    %adsubb6dW = stablehlo.subtract %b6dW, %adstb6dW : tensor<240x1x3x3xf32>
    %adwdb6dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240x1x3x3xf32>
    %adwdlrb6dW = stablehlo.multiply %adwdb6dW, %adlrb6dW : tensor<240x1x3x3xf32>
    %adwdpb6dW = stablehlo.multiply %adwdlrb6dW, %b6dW : tensor<240x1x3x3xf32>
    %adnewb6dW = stablehlo.subtract %adsubb6dW, %adwdpb6dW : tensor<240x1x3x3xf32>
    %adb1b6db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b6db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb6db = stablehlo.multiply %adb1b6db, %b6dbm : tensor<240xf32>
    %admgb6db = stablehlo.multiply %adob1b6db, %b6ddb : tensor<240xf32>
    %admnb6db = stablehlo.add %admsb6db, %admgb6db : tensor<240xf32>
    %adb2b6db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b6db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb6db = stablehlo.multiply %adb2b6db, %b6dbv : tensor<240xf32>
    %adg2b6db = stablehlo.multiply %b6ddb, %b6ddb : tensor<240xf32>
    %advgb6db = stablehlo.multiply %adob2b6db, %adg2b6db : tensor<240xf32>
    %advnb6db = stablehlo.add %advsb6db, %advgb6db : tensor<240xf32>
    %adbc1b6db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b6db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb6db = stablehlo.divide %admnb6db, %adbc1b6db : tensor<240xf32>
    %advhb6db = stablehlo.divide %advnb6db, %adbc2b6db : tensor<240xf32>
    %adlrb6db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb6db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb6db = stablehlo.sqrt %advhb6db : tensor<240xf32>
    %addenb6db = stablehlo.add %adsqb6db, %adepsb6db : tensor<240xf32>
    %adratb6db = stablehlo.divide %admhb6db, %addenb6db : tensor<240xf32>
    %adstb6db = stablehlo.multiply %adlrb6db, %adratb6db : tensor<240xf32>
    %adsubb6db = stablehlo.subtract %b6db, %adstb6db : tensor<240xf32>
    %adwdb6db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb6db = stablehlo.multiply %adwdb6db, %adlrb6db : tensor<240xf32>
    %adwdpb6db = stablehlo.multiply %adwdlrb6db, %b6db : tensor<240xf32>
    %adnewb6db = stablehlo.subtract %adsubb6db, %adwdpb6db : tensor<240xf32>
    %adb1b6dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b6dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb6dg = stablehlo.multiply %adb1b6dg, %b6dgm : tensor<240xf32>
    %admgb6dg = stablehlo.multiply %adob1b6dg, %b6ddndg : tensor<240xf32>
    %admnb6dg = stablehlo.add %admsb6dg, %admgb6dg : tensor<240xf32>
    %adb2b6dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b6dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb6dg = stablehlo.multiply %adb2b6dg, %b6dgv : tensor<240xf32>
    %adg2b6dg = stablehlo.multiply %b6ddndg, %b6ddndg : tensor<240xf32>
    %advgb6dg = stablehlo.multiply %adob2b6dg, %adg2b6dg : tensor<240xf32>
    %advnb6dg = stablehlo.add %advsb6dg, %advgb6dg : tensor<240xf32>
    %adbc1b6dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b6dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb6dg = stablehlo.divide %admnb6dg, %adbc1b6dg : tensor<240xf32>
    %advhb6dg = stablehlo.divide %advnb6dg, %adbc2b6dg : tensor<240xf32>
    %adlrb6dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb6dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb6dg = stablehlo.sqrt %advhb6dg : tensor<240xf32>
    %addenb6dg = stablehlo.add %adsqb6dg, %adepsb6dg : tensor<240xf32>
    %adratb6dg = stablehlo.divide %admhb6dg, %addenb6dg : tensor<240xf32>
    %adstb6dg = stablehlo.multiply %adlrb6dg, %adratb6dg : tensor<240xf32>
    %adsubb6dg = stablehlo.subtract %b6dg, %adstb6dg : tensor<240xf32>
    %adwdb6dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb6dg = stablehlo.multiply %adwdb6dg, %adlrb6dg : tensor<240xf32>
    %adwdpb6dg = stablehlo.multiply %adwdlrb6dg, %b6dg : tensor<240xf32>
    %adnewb6dg = stablehlo.subtract %adsubb6dg, %adwdpb6dg : tensor<240xf32>
    %adb1b6dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b6dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb6dbt = stablehlo.multiply %adb1b6dbt, %b6dbtm : tensor<240xf32>
    %admgb6dbt = stablehlo.multiply %adob1b6dbt, %b6ddndb : tensor<240xf32>
    %admnb6dbt = stablehlo.add %admsb6dbt, %admgb6dbt : tensor<240xf32>
    %adb2b6dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b6dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb6dbt = stablehlo.multiply %adb2b6dbt, %b6dbtv : tensor<240xf32>
    %adg2b6dbt = stablehlo.multiply %b6ddndb, %b6ddndb : tensor<240xf32>
    %advgb6dbt = stablehlo.multiply %adob2b6dbt, %adg2b6dbt : tensor<240xf32>
    %advnb6dbt = stablehlo.add %advsb6dbt, %advgb6dbt : tensor<240xf32>
    %adbc1b6dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b6dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb6dbt = stablehlo.divide %admnb6dbt, %adbc1b6dbt : tensor<240xf32>
    %advhb6dbt = stablehlo.divide %advnb6dbt, %adbc2b6dbt : tensor<240xf32>
    %adlrb6dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb6dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb6dbt = stablehlo.sqrt %advhb6dbt : tensor<240xf32>
    %addenb6dbt = stablehlo.add %adsqb6dbt, %adepsb6dbt : tensor<240xf32>
    %adratb6dbt = stablehlo.divide %admhb6dbt, %addenb6dbt : tensor<240xf32>
    %adstb6dbt = stablehlo.multiply %adlrb6dbt, %adratb6dbt : tensor<240xf32>
    %adsubb6dbt = stablehlo.subtract %b6dbt, %adstb6dbt : tensor<240xf32>
    %adwdb6dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb6dbt = stablehlo.multiply %adwdb6dbt, %adlrb6dbt : tensor<240xf32>
    %adwdpb6dbt = stablehlo.multiply %adwdlrb6dbt, %b6dbt : tensor<240xf32>
    %adnewb6dbt = stablehlo.subtract %adsubb6dbt, %adwdpb6dbt : tensor<240xf32>
    %adb1b6zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adob1b6zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %admsb6zW1 = stablehlo.multiply %adb1b6zW1, %b6zW1m : tensor<240x10xf32>
    %admgb6zW1 = stablehlo.multiply %adob1b6zW1, %b6zdWs1 : tensor<240x10xf32>
    %admnb6zW1 = stablehlo.add %admsb6zW1, %admgb6zW1 : tensor<240x10xf32>
    %adb2b6zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adob2b6zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %advsb6zW1 = stablehlo.multiply %adb2b6zW1, %b6zW1v : tensor<240x10xf32>
    %adg2b6zW1 = stablehlo.multiply %b6zdWs1, %b6zdWs1 : tensor<240x10xf32>
    %advgb6zW1 = stablehlo.multiply %adob2b6zW1, %adg2b6zW1 : tensor<240x10xf32>
    %advnb6zW1 = stablehlo.add %advsb6zW1, %advgb6zW1 : tensor<240x10xf32>
    %adbc1b6zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adbc2b6zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %admhb6zW1 = stablehlo.divide %admnb6zW1, %adbc1b6zW1 : tensor<240x10xf32>
    %advhb6zW1 = stablehlo.divide %advnb6zW1, %adbc2b6zW1 : tensor<240x10xf32>
    %adlrb6zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adepsb6zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adsqb6zW1 = stablehlo.sqrt %advhb6zW1 : tensor<240x10xf32>
    %addenb6zW1 = stablehlo.add %adsqb6zW1, %adepsb6zW1 : tensor<240x10xf32>
    %adratb6zW1 = stablehlo.divide %admhb6zW1, %addenb6zW1 : tensor<240x10xf32>
    %adstb6zW1 = stablehlo.multiply %adlrb6zW1, %adratb6zW1 : tensor<240x10xf32>
    %adsubb6zW1 = stablehlo.subtract %b6zW1, %adstb6zW1 : tensor<240x10xf32>
    %adwdb6zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240x10xf32>
    %adwdlrb6zW1 = stablehlo.multiply %adwdb6zW1, %adlrb6zW1 : tensor<240x10xf32>
    %adwdpb6zW1 = stablehlo.multiply %adwdlrb6zW1, %b6zW1 : tensor<240x10xf32>
    %adnewb6zW1 = stablehlo.subtract %adsubb6zW1, %adwdpb6zW1 : tensor<240x10xf32>
    %adb1b6zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob1b6zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admsb6zb1 = stablehlo.multiply %adb1b6zb1, %b6zb1m : tensor<10xf32>
    %admgb6zb1 = stablehlo.multiply %adob1b6zb1, %b6zdbs1 : tensor<10xf32>
    %admnb6zb1 = stablehlo.add %admsb6zb1, %admgb6zb1 : tensor<10xf32>
    %adb2b6zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob2b6zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %advsb6zb1 = stablehlo.multiply %adb2b6zb1, %b6zb1v : tensor<10xf32>
    %adg2b6zb1 = stablehlo.multiply %b6zdbs1, %b6zdbs1 : tensor<10xf32>
    %advgb6zb1 = stablehlo.multiply %adob2b6zb1, %adg2b6zb1 : tensor<10xf32>
    %advnb6zb1 = stablehlo.add %advsb6zb1, %advgb6zb1 : tensor<10xf32>
    %adbc1b6zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adbc2b6zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admhb6zb1 = stablehlo.divide %admnb6zb1, %adbc1b6zb1 : tensor<10xf32>
    %advhb6zb1 = stablehlo.divide %advnb6zb1, %adbc2b6zb1 : tensor<10xf32>
    %adlrb6zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adepsb6zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adsqb6zb1 = stablehlo.sqrt %advhb6zb1 : tensor<10xf32>
    %addenb6zb1 = stablehlo.add %adsqb6zb1, %adepsb6zb1 : tensor<10xf32>
    %adratb6zb1 = stablehlo.divide %admhb6zb1, %addenb6zb1 : tensor<10xf32>
    %adstb6zb1 = stablehlo.multiply %adlrb6zb1, %adratb6zb1 : tensor<10xf32>
    %adsubb6zb1 = stablehlo.subtract %b6zb1, %adstb6zb1 : tensor<10xf32>
    %adwdb6zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adwdlrb6zb1 = stablehlo.multiply %adwdb6zb1, %adlrb6zb1 : tensor<10xf32>
    %adwdpb6zb1 = stablehlo.multiply %adwdlrb6zb1, %b6zb1 : tensor<10xf32>
    %adnewb6zb1 = stablehlo.subtract %adsubb6zb1, %adwdpb6zb1 : tensor<10xf32>
    %adb1b6zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adob1b6zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %admsb6zW2 = stablehlo.multiply %adb1b6zW2, %b6zW2m : tensor<10x240xf32>
    %admgb6zW2 = stablehlo.multiply %adob1b6zW2, %b6zdWs2 : tensor<10x240xf32>
    %admnb6zW2 = stablehlo.add %admsb6zW2, %admgb6zW2 : tensor<10x240xf32>
    %adb2b6zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adob2b6zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %advsb6zW2 = stablehlo.multiply %adb2b6zW2, %b6zW2v : tensor<10x240xf32>
    %adg2b6zW2 = stablehlo.multiply %b6zdWs2, %b6zdWs2 : tensor<10x240xf32>
    %advgb6zW2 = stablehlo.multiply %adob2b6zW2, %adg2b6zW2 : tensor<10x240xf32>
    %advnb6zW2 = stablehlo.add %advsb6zW2, %advgb6zW2 : tensor<10x240xf32>
    %adbc1b6zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adbc2b6zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %admhb6zW2 = stablehlo.divide %admnb6zW2, %adbc1b6zW2 : tensor<10x240xf32>
    %advhb6zW2 = stablehlo.divide %advnb6zW2, %adbc2b6zW2 : tensor<10x240xf32>
    %adlrb6zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adepsb6zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adsqb6zW2 = stablehlo.sqrt %advhb6zW2 : tensor<10x240xf32>
    %addenb6zW2 = stablehlo.add %adsqb6zW2, %adepsb6zW2 : tensor<10x240xf32>
    %adratb6zW2 = stablehlo.divide %admhb6zW2, %addenb6zW2 : tensor<10x240xf32>
    %adstb6zW2 = stablehlo.multiply %adlrb6zW2, %adratb6zW2 : tensor<10x240xf32>
    %adsubb6zW2 = stablehlo.subtract %b6zW2, %adstb6zW2 : tensor<10x240xf32>
    %adwdb6zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10x240xf32>
    %adwdlrb6zW2 = stablehlo.multiply %adwdb6zW2, %adlrb6zW2 : tensor<10x240xf32>
    %adwdpb6zW2 = stablehlo.multiply %adwdlrb6zW2, %b6zW2 : tensor<10x240xf32>
    %adnewb6zW2 = stablehlo.subtract %adsubb6zW2, %adwdpb6zW2 : tensor<10x240xf32>
    %adb1b6zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob1b6zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admsb6zb2 = stablehlo.multiply %adb1b6zb2, %b6zb2m : tensor<240xf32>
    %admgb6zb2 = stablehlo.multiply %adob1b6zb2, %b6zdbs2 : tensor<240xf32>
    %admnb6zb2 = stablehlo.add %admsb6zb2, %admgb6zb2 : tensor<240xf32>
    %adb2b6zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adob2b6zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %advsb6zb2 = stablehlo.multiply %adb2b6zb2, %b6zb2v : tensor<240xf32>
    %adg2b6zb2 = stablehlo.multiply %b6zdbs2, %b6zdbs2 : tensor<240xf32>
    %advgb6zb2 = stablehlo.multiply %adob2b6zb2, %adg2b6zb2 : tensor<240xf32>
    %advnb6zb2 = stablehlo.add %advsb6zb2, %advgb6zb2 : tensor<240xf32>
    %adbc1b6zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adbc2b6zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %admhb6zb2 = stablehlo.divide %admnb6zb2, %adbc1b6zb2 : tensor<240xf32>
    %advhb6zb2 = stablehlo.divide %advnb6zb2, %adbc2b6zb2 : tensor<240xf32>
    %adlrb6zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adepsb6zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adsqb6zb2 = stablehlo.sqrt %advhb6zb2 : tensor<240xf32>
    %addenb6zb2 = stablehlo.add %adsqb6zb2, %adepsb6zb2 : tensor<240xf32>
    %adratb6zb2 = stablehlo.divide %admhb6zb2, %addenb6zb2 : tensor<240xf32>
    %adstb6zb2 = stablehlo.multiply %adlrb6zb2, %adratb6zb2 : tensor<240xf32>
    %adsubb6zb2 = stablehlo.subtract %b6zb2, %adstb6zb2 : tensor<240xf32>
    %adwdb6zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<240xf32>
    %adwdlrb6zb2 = stablehlo.multiply %adwdb6zb2, %adlrb6zb2 : tensor<240xf32>
    %adwdpb6zb2 = stablehlo.multiply %adwdlrb6zb2, %b6zb2 : tensor<240xf32>
    %adnewb6zb2 = stablehlo.subtract %adsubb6zb2, %adwdpb6zb2 : tensor<240xf32>
    %adb1b6pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %adob1b6pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %admsb6pW = stablehlo.multiply %adb1b6pW, %b6pWm : tensor<80x240x1x1xf32>
    %admgb6pW = stablehlo.multiply %adob1b6pW, %b6dpW : tensor<80x240x1x1xf32>
    %admnb6pW = stablehlo.add %admsb6pW, %admgb6pW : tensor<80x240x1x1xf32>
    %adb2b6pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %adob2b6pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %advsb6pW = stablehlo.multiply %adb2b6pW, %b6pWv : tensor<80x240x1x1xf32>
    %adg2b6pW = stablehlo.multiply %b6dpW, %b6dpW : tensor<80x240x1x1xf32>
    %advgb6pW = stablehlo.multiply %adob2b6pW, %adg2b6pW : tensor<80x240x1x1xf32>
    %advnb6pW = stablehlo.add %advsb6pW, %advgb6pW : tensor<80x240x1x1xf32>
    %adbc1b6pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %adbc2b6pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %admhb6pW = stablehlo.divide %admnb6pW, %adbc1b6pW : tensor<80x240x1x1xf32>
    %advhb6pW = stablehlo.divide %advnb6pW, %adbc2b6pW : tensor<80x240x1x1xf32>
    %adlrb6pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %adepsb6pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %adsqb6pW = stablehlo.sqrt %advhb6pW : tensor<80x240x1x1xf32>
    %addenb6pW = stablehlo.add %adsqb6pW, %adepsb6pW : tensor<80x240x1x1xf32>
    %adratb6pW = stablehlo.divide %admhb6pW, %addenb6pW : tensor<80x240x1x1xf32>
    %adstb6pW = stablehlo.multiply %adlrb6pW, %adratb6pW : tensor<80x240x1x1xf32>
    %adsubb6pW = stablehlo.subtract %b6pW, %adstb6pW : tensor<80x240x1x1xf32>
    %adwdb6pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80x240x1x1xf32>
    %adwdlrb6pW = stablehlo.multiply %adwdb6pW, %adlrb6pW : tensor<80x240x1x1xf32>
    %adwdpb6pW = stablehlo.multiply %adwdlrb6pW, %b6pW : tensor<80x240x1x1xf32>
    %adnewb6pW = stablehlo.subtract %adsubb6pW, %adwdpb6pW : tensor<80x240x1x1xf32>
    %adb1b6pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b6pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb6pb = stablehlo.multiply %adb1b6pb, %b6pbm : tensor<80xf32>
    %admgb6pb = stablehlo.multiply %adob1b6pb, %b6dpb : tensor<80xf32>
    %admnb6pb = stablehlo.add %admsb6pb, %admgb6pb : tensor<80xf32>
    %adb2b6pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b6pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb6pb = stablehlo.multiply %adb2b6pb, %b6pbv : tensor<80xf32>
    %adg2b6pb = stablehlo.multiply %b6dpb, %b6dpb : tensor<80xf32>
    %advgb6pb = stablehlo.multiply %adob2b6pb, %adg2b6pb : tensor<80xf32>
    %advnb6pb = stablehlo.add %advsb6pb, %advgb6pb : tensor<80xf32>
    %adbc1b6pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b6pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb6pb = stablehlo.divide %admnb6pb, %adbc1b6pb : tensor<80xf32>
    %advhb6pb = stablehlo.divide %advnb6pb, %adbc2b6pb : tensor<80xf32>
    %adlrb6pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb6pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb6pb = stablehlo.sqrt %advhb6pb : tensor<80xf32>
    %addenb6pb = stablehlo.add %adsqb6pb, %adepsb6pb : tensor<80xf32>
    %adratb6pb = stablehlo.divide %admhb6pb, %addenb6pb : tensor<80xf32>
    %adstb6pb = stablehlo.multiply %adlrb6pb, %adratb6pb : tensor<80xf32>
    %adsubb6pb = stablehlo.subtract %b6pb, %adstb6pb : tensor<80xf32>
    %adwdb6pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb6pb = stablehlo.multiply %adwdb6pb, %adlrb6pb : tensor<80xf32>
    %adwdpb6pb = stablehlo.multiply %adwdlrb6pb, %b6pb : tensor<80xf32>
    %adnewb6pb = stablehlo.subtract %adsubb6pb, %adwdpb6pb : tensor<80xf32>
    %adb1b6pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b6pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb6pg = stablehlo.multiply %adb1b6pg, %b6pgm : tensor<80xf32>
    %admgb6pg = stablehlo.multiply %adob1b6pg, %b6dpndg : tensor<80xf32>
    %admnb6pg = stablehlo.add %admsb6pg, %admgb6pg : tensor<80xf32>
    %adb2b6pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b6pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb6pg = stablehlo.multiply %adb2b6pg, %b6pgv : tensor<80xf32>
    %adg2b6pg = stablehlo.multiply %b6dpndg, %b6dpndg : tensor<80xf32>
    %advgb6pg = stablehlo.multiply %adob2b6pg, %adg2b6pg : tensor<80xf32>
    %advnb6pg = stablehlo.add %advsb6pg, %advgb6pg : tensor<80xf32>
    %adbc1b6pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b6pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb6pg = stablehlo.divide %admnb6pg, %adbc1b6pg : tensor<80xf32>
    %advhb6pg = stablehlo.divide %advnb6pg, %adbc2b6pg : tensor<80xf32>
    %adlrb6pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb6pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb6pg = stablehlo.sqrt %advhb6pg : tensor<80xf32>
    %addenb6pg = stablehlo.add %adsqb6pg, %adepsb6pg : tensor<80xf32>
    %adratb6pg = stablehlo.divide %admhb6pg, %addenb6pg : tensor<80xf32>
    %adstb6pg = stablehlo.multiply %adlrb6pg, %adratb6pg : tensor<80xf32>
    %adsubb6pg = stablehlo.subtract %b6pg, %adstb6pg : tensor<80xf32>
    %adwdb6pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb6pg = stablehlo.multiply %adwdb6pg, %adlrb6pg : tensor<80xf32>
    %adwdpb6pg = stablehlo.multiply %adwdlrb6pg, %b6pg : tensor<80xf32>
    %adnewb6pg = stablehlo.subtract %adsubb6pg, %adwdpb6pg : tensor<80xf32>
    %adb1b6pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b6pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb6pbt = stablehlo.multiply %adb1b6pbt, %b6pbtm : tensor<80xf32>
    %admgb6pbt = stablehlo.multiply %adob1b6pbt, %b6dpndb : tensor<80xf32>
    %admnb6pbt = stablehlo.add %admsb6pbt, %admgb6pbt : tensor<80xf32>
    %adb2b6pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b6pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb6pbt = stablehlo.multiply %adb2b6pbt, %b6pbtv : tensor<80xf32>
    %adg2b6pbt = stablehlo.multiply %b6dpndb, %b6dpndb : tensor<80xf32>
    %advgb6pbt = stablehlo.multiply %adob2b6pbt, %adg2b6pbt : tensor<80xf32>
    %advnb6pbt = stablehlo.add %advsb6pbt, %advgb6pbt : tensor<80xf32>
    %adbc1b6pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b6pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb6pbt = stablehlo.divide %admnb6pbt, %adbc1b6pbt : tensor<80xf32>
    %advhb6pbt = stablehlo.divide %advnb6pbt, %adbc2b6pbt : tensor<80xf32>
    %adlrb6pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb6pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb6pbt = stablehlo.sqrt %advhb6pbt : tensor<80xf32>
    %addenb6pbt = stablehlo.add %adsqb6pbt, %adepsb6pbt : tensor<80xf32>
    %adratb6pbt = stablehlo.divide %admhb6pbt, %addenb6pbt : tensor<80xf32>
    %adstb6pbt = stablehlo.multiply %adlrb6pbt, %adratb6pbt : tensor<80xf32>
    %adsubb6pbt = stablehlo.subtract %b6pbt, %adstb6pbt : tensor<80xf32>
    %adwdb6pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb6pbt = stablehlo.multiply %adwdb6pbt, %adlrb6pbt : tensor<80xf32>
    %adwdpb6pbt = stablehlo.multiply %adwdlrb6pbt, %b6pbt : tensor<80xf32>
    %adnewb6pbt = stablehlo.subtract %adsubb6pbt, %adwdpb6pbt : tensor<80xf32>
    %adb1b7eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adob1b7eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %admsb7eW = stablehlo.multiply %adb1b7eW, %b7eWm : tensor<480x80x1x1xf32>
    %admgb7eW = stablehlo.multiply %adob1b7eW, %b7deW : tensor<480x80x1x1xf32>
    %admnb7eW = stablehlo.add %admsb7eW, %admgb7eW : tensor<480x80x1x1xf32>
    %adb2b7eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adob2b7eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %advsb7eW = stablehlo.multiply %adb2b7eW, %b7eWv : tensor<480x80x1x1xf32>
    %adg2b7eW = stablehlo.multiply %b7deW, %b7deW : tensor<480x80x1x1xf32>
    %advgb7eW = stablehlo.multiply %adob2b7eW, %adg2b7eW : tensor<480x80x1x1xf32>
    %advnb7eW = stablehlo.add %advsb7eW, %advgb7eW : tensor<480x80x1x1xf32>
    %adbc1b7eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adbc2b7eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %admhb7eW = stablehlo.divide %admnb7eW, %adbc1b7eW : tensor<480x80x1x1xf32>
    %advhb7eW = stablehlo.divide %advnb7eW, %adbc2b7eW : tensor<480x80x1x1xf32>
    %adlrb7eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adepsb7eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adsqb7eW = stablehlo.sqrt %advhb7eW : tensor<480x80x1x1xf32>
    %addenb7eW = stablehlo.add %adsqb7eW, %adepsb7eW : tensor<480x80x1x1xf32>
    %adratb7eW = stablehlo.divide %admhb7eW, %addenb7eW : tensor<480x80x1x1xf32>
    %adstb7eW = stablehlo.multiply %adlrb7eW, %adratb7eW : tensor<480x80x1x1xf32>
    %adsubb7eW = stablehlo.subtract %b7eW, %adstb7eW : tensor<480x80x1x1xf32>
    %adwdb7eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adwdlrb7eW = stablehlo.multiply %adwdb7eW, %adlrb7eW : tensor<480x80x1x1xf32>
    %adwdpb7eW = stablehlo.multiply %adwdlrb7eW, %b7eW : tensor<480x80x1x1xf32>
    %adnewb7eW = stablehlo.subtract %adsubb7eW, %adwdpb7eW : tensor<480x80x1x1xf32>
    %adb1b7eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b7eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb7eb = stablehlo.multiply %adb1b7eb, %b7ebm : tensor<480xf32>
    %admgb7eb = stablehlo.multiply %adob1b7eb, %b7deb : tensor<480xf32>
    %admnb7eb = stablehlo.add %admsb7eb, %admgb7eb : tensor<480xf32>
    %adb2b7eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b7eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb7eb = stablehlo.multiply %adb2b7eb, %b7ebv : tensor<480xf32>
    %adg2b7eb = stablehlo.multiply %b7deb, %b7deb : tensor<480xf32>
    %advgb7eb = stablehlo.multiply %adob2b7eb, %adg2b7eb : tensor<480xf32>
    %advnb7eb = stablehlo.add %advsb7eb, %advgb7eb : tensor<480xf32>
    %adbc1b7eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b7eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb7eb = stablehlo.divide %admnb7eb, %adbc1b7eb : tensor<480xf32>
    %advhb7eb = stablehlo.divide %advnb7eb, %adbc2b7eb : tensor<480xf32>
    %adlrb7eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb7eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb7eb = stablehlo.sqrt %advhb7eb : tensor<480xf32>
    %addenb7eb = stablehlo.add %adsqb7eb, %adepsb7eb : tensor<480xf32>
    %adratb7eb = stablehlo.divide %admhb7eb, %addenb7eb : tensor<480xf32>
    %adstb7eb = stablehlo.multiply %adlrb7eb, %adratb7eb : tensor<480xf32>
    %adsubb7eb = stablehlo.subtract %b7eb, %adstb7eb : tensor<480xf32>
    %adwdb7eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb7eb = stablehlo.multiply %adwdb7eb, %adlrb7eb : tensor<480xf32>
    %adwdpb7eb = stablehlo.multiply %adwdlrb7eb, %b7eb : tensor<480xf32>
    %adnewb7eb = stablehlo.subtract %adsubb7eb, %adwdpb7eb : tensor<480xf32>
    %adb1b7eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b7eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb7eg = stablehlo.multiply %adb1b7eg, %b7egm : tensor<480xf32>
    %admgb7eg = stablehlo.multiply %adob1b7eg, %b7dendg : tensor<480xf32>
    %admnb7eg = stablehlo.add %admsb7eg, %admgb7eg : tensor<480xf32>
    %adb2b7eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b7eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb7eg = stablehlo.multiply %adb2b7eg, %b7egv : tensor<480xf32>
    %adg2b7eg = stablehlo.multiply %b7dendg, %b7dendg : tensor<480xf32>
    %advgb7eg = stablehlo.multiply %adob2b7eg, %adg2b7eg : tensor<480xf32>
    %advnb7eg = stablehlo.add %advsb7eg, %advgb7eg : tensor<480xf32>
    %adbc1b7eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b7eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb7eg = stablehlo.divide %admnb7eg, %adbc1b7eg : tensor<480xf32>
    %advhb7eg = stablehlo.divide %advnb7eg, %adbc2b7eg : tensor<480xf32>
    %adlrb7eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb7eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb7eg = stablehlo.sqrt %advhb7eg : tensor<480xf32>
    %addenb7eg = stablehlo.add %adsqb7eg, %adepsb7eg : tensor<480xf32>
    %adratb7eg = stablehlo.divide %admhb7eg, %addenb7eg : tensor<480xf32>
    %adstb7eg = stablehlo.multiply %adlrb7eg, %adratb7eg : tensor<480xf32>
    %adsubb7eg = stablehlo.subtract %b7eg, %adstb7eg : tensor<480xf32>
    %adwdb7eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb7eg = stablehlo.multiply %adwdb7eg, %adlrb7eg : tensor<480xf32>
    %adwdpb7eg = stablehlo.multiply %adwdlrb7eg, %b7eg : tensor<480xf32>
    %adnewb7eg = stablehlo.subtract %adsubb7eg, %adwdpb7eg : tensor<480xf32>
    %adb1b7ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b7ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb7ebt = stablehlo.multiply %adb1b7ebt, %b7ebtm : tensor<480xf32>
    %admgb7ebt = stablehlo.multiply %adob1b7ebt, %b7dendb : tensor<480xf32>
    %admnb7ebt = stablehlo.add %admsb7ebt, %admgb7ebt : tensor<480xf32>
    %adb2b7ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b7ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb7ebt = stablehlo.multiply %adb2b7ebt, %b7ebtv : tensor<480xf32>
    %adg2b7ebt = stablehlo.multiply %b7dendb, %b7dendb : tensor<480xf32>
    %advgb7ebt = stablehlo.multiply %adob2b7ebt, %adg2b7ebt : tensor<480xf32>
    %advnb7ebt = stablehlo.add %advsb7ebt, %advgb7ebt : tensor<480xf32>
    %adbc1b7ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b7ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb7ebt = stablehlo.divide %admnb7ebt, %adbc1b7ebt : tensor<480xf32>
    %advhb7ebt = stablehlo.divide %advnb7ebt, %adbc2b7ebt : tensor<480xf32>
    %adlrb7ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb7ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb7ebt = stablehlo.sqrt %advhb7ebt : tensor<480xf32>
    %addenb7ebt = stablehlo.add %adsqb7ebt, %adepsb7ebt : tensor<480xf32>
    %adratb7ebt = stablehlo.divide %admhb7ebt, %addenb7ebt : tensor<480xf32>
    %adstb7ebt = stablehlo.multiply %adlrb7ebt, %adratb7ebt : tensor<480xf32>
    %adsubb7ebt = stablehlo.subtract %b7ebt, %adstb7ebt : tensor<480xf32>
    %adwdb7ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb7ebt = stablehlo.multiply %adwdb7ebt, %adlrb7ebt : tensor<480xf32>
    %adwdpb7ebt = stablehlo.multiply %adwdlrb7ebt, %b7ebt : tensor<480xf32>
    %adnewb7ebt = stablehlo.subtract %adsubb7ebt, %adwdpb7ebt : tensor<480xf32>
    %adb1b7dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adob1b7dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %admsb7dW = stablehlo.multiply %adb1b7dW, %b7dWm : tensor<480x1x3x3xf32>
    %admgb7dW = stablehlo.multiply %adob1b7dW, %b7ddW : tensor<480x1x3x3xf32>
    %admnb7dW = stablehlo.add %admsb7dW, %admgb7dW : tensor<480x1x3x3xf32>
    %adb2b7dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adob2b7dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %advsb7dW = stablehlo.multiply %adb2b7dW, %b7dWv : tensor<480x1x3x3xf32>
    %adg2b7dW = stablehlo.multiply %b7ddW, %b7ddW : tensor<480x1x3x3xf32>
    %advgb7dW = stablehlo.multiply %adob2b7dW, %adg2b7dW : tensor<480x1x3x3xf32>
    %advnb7dW = stablehlo.add %advsb7dW, %advgb7dW : tensor<480x1x3x3xf32>
    %adbc1b7dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adbc2b7dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %admhb7dW = stablehlo.divide %admnb7dW, %adbc1b7dW : tensor<480x1x3x3xf32>
    %advhb7dW = stablehlo.divide %advnb7dW, %adbc2b7dW : tensor<480x1x3x3xf32>
    %adlrb7dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adepsb7dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adsqb7dW = stablehlo.sqrt %advhb7dW : tensor<480x1x3x3xf32>
    %addenb7dW = stablehlo.add %adsqb7dW, %adepsb7dW : tensor<480x1x3x3xf32>
    %adratb7dW = stablehlo.divide %admhb7dW, %addenb7dW : tensor<480x1x3x3xf32>
    %adstb7dW = stablehlo.multiply %adlrb7dW, %adratb7dW : tensor<480x1x3x3xf32>
    %adsubb7dW = stablehlo.subtract %b7dW, %adstb7dW : tensor<480x1x3x3xf32>
    %adwdb7dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adwdlrb7dW = stablehlo.multiply %adwdb7dW, %adlrb7dW : tensor<480x1x3x3xf32>
    %adwdpb7dW = stablehlo.multiply %adwdlrb7dW, %b7dW : tensor<480x1x3x3xf32>
    %adnewb7dW = stablehlo.subtract %adsubb7dW, %adwdpb7dW : tensor<480x1x3x3xf32>
    %adb1b7db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b7db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb7db = stablehlo.multiply %adb1b7db, %b7dbm : tensor<480xf32>
    %admgb7db = stablehlo.multiply %adob1b7db, %b7ddb : tensor<480xf32>
    %admnb7db = stablehlo.add %admsb7db, %admgb7db : tensor<480xf32>
    %adb2b7db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b7db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb7db = stablehlo.multiply %adb2b7db, %b7dbv : tensor<480xf32>
    %adg2b7db = stablehlo.multiply %b7ddb, %b7ddb : tensor<480xf32>
    %advgb7db = stablehlo.multiply %adob2b7db, %adg2b7db : tensor<480xf32>
    %advnb7db = stablehlo.add %advsb7db, %advgb7db : tensor<480xf32>
    %adbc1b7db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b7db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb7db = stablehlo.divide %admnb7db, %adbc1b7db : tensor<480xf32>
    %advhb7db = stablehlo.divide %advnb7db, %adbc2b7db : tensor<480xf32>
    %adlrb7db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb7db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb7db = stablehlo.sqrt %advhb7db : tensor<480xf32>
    %addenb7db = stablehlo.add %adsqb7db, %adepsb7db : tensor<480xf32>
    %adratb7db = stablehlo.divide %admhb7db, %addenb7db : tensor<480xf32>
    %adstb7db = stablehlo.multiply %adlrb7db, %adratb7db : tensor<480xf32>
    %adsubb7db = stablehlo.subtract %b7db, %adstb7db : tensor<480xf32>
    %adwdb7db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb7db = stablehlo.multiply %adwdb7db, %adlrb7db : tensor<480xf32>
    %adwdpb7db = stablehlo.multiply %adwdlrb7db, %b7db : tensor<480xf32>
    %adnewb7db = stablehlo.subtract %adsubb7db, %adwdpb7db : tensor<480xf32>
    %adb1b7dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b7dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb7dg = stablehlo.multiply %adb1b7dg, %b7dgm : tensor<480xf32>
    %admgb7dg = stablehlo.multiply %adob1b7dg, %b7ddndg : tensor<480xf32>
    %admnb7dg = stablehlo.add %admsb7dg, %admgb7dg : tensor<480xf32>
    %adb2b7dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b7dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb7dg = stablehlo.multiply %adb2b7dg, %b7dgv : tensor<480xf32>
    %adg2b7dg = stablehlo.multiply %b7ddndg, %b7ddndg : tensor<480xf32>
    %advgb7dg = stablehlo.multiply %adob2b7dg, %adg2b7dg : tensor<480xf32>
    %advnb7dg = stablehlo.add %advsb7dg, %advgb7dg : tensor<480xf32>
    %adbc1b7dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b7dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb7dg = stablehlo.divide %admnb7dg, %adbc1b7dg : tensor<480xf32>
    %advhb7dg = stablehlo.divide %advnb7dg, %adbc2b7dg : tensor<480xf32>
    %adlrb7dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb7dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb7dg = stablehlo.sqrt %advhb7dg : tensor<480xf32>
    %addenb7dg = stablehlo.add %adsqb7dg, %adepsb7dg : tensor<480xf32>
    %adratb7dg = stablehlo.divide %admhb7dg, %addenb7dg : tensor<480xf32>
    %adstb7dg = stablehlo.multiply %adlrb7dg, %adratb7dg : tensor<480xf32>
    %adsubb7dg = stablehlo.subtract %b7dg, %adstb7dg : tensor<480xf32>
    %adwdb7dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb7dg = stablehlo.multiply %adwdb7dg, %adlrb7dg : tensor<480xf32>
    %adwdpb7dg = stablehlo.multiply %adwdlrb7dg, %b7dg : tensor<480xf32>
    %adnewb7dg = stablehlo.subtract %adsubb7dg, %adwdpb7dg : tensor<480xf32>
    %adb1b7dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b7dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb7dbt = stablehlo.multiply %adb1b7dbt, %b7dbtm : tensor<480xf32>
    %admgb7dbt = stablehlo.multiply %adob1b7dbt, %b7ddndb : tensor<480xf32>
    %admnb7dbt = stablehlo.add %admsb7dbt, %admgb7dbt : tensor<480xf32>
    %adb2b7dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b7dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb7dbt = stablehlo.multiply %adb2b7dbt, %b7dbtv : tensor<480xf32>
    %adg2b7dbt = stablehlo.multiply %b7ddndb, %b7ddndb : tensor<480xf32>
    %advgb7dbt = stablehlo.multiply %adob2b7dbt, %adg2b7dbt : tensor<480xf32>
    %advnb7dbt = stablehlo.add %advsb7dbt, %advgb7dbt : tensor<480xf32>
    %adbc1b7dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b7dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb7dbt = stablehlo.divide %admnb7dbt, %adbc1b7dbt : tensor<480xf32>
    %advhb7dbt = stablehlo.divide %advnb7dbt, %adbc2b7dbt : tensor<480xf32>
    %adlrb7dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb7dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb7dbt = stablehlo.sqrt %advhb7dbt : tensor<480xf32>
    %addenb7dbt = stablehlo.add %adsqb7dbt, %adepsb7dbt : tensor<480xf32>
    %adratb7dbt = stablehlo.divide %admhb7dbt, %addenb7dbt : tensor<480xf32>
    %adstb7dbt = stablehlo.multiply %adlrb7dbt, %adratb7dbt : tensor<480xf32>
    %adsubb7dbt = stablehlo.subtract %b7dbt, %adstb7dbt : tensor<480xf32>
    %adwdb7dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb7dbt = stablehlo.multiply %adwdb7dbt, %adlrb7dbt : tensor<480xf32>
    %adwdpb7dbt = stablehlo.multiply %adwdlrb7dbt, %b7dbt : tensor<480xf32>
    %adnewb7dbt = stablehlo.subtract %adsubb7dbt, %adwdpb7dbt : tensor<480xf32>
    %adb1b7zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adob1b7zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %admsb7zW1 = stablehlo.multiply %adb1b7zW1, %b7zW1m : tensor<480x20xf32>
    %admgb7zW1 = stablehlo.multiply %adob1b7zW1, %b7zdWs1 : tensor<480x20xf32>
    %admnb7zW1 = stablehlo.add %admsb7zW1, %admgb7zW1 : tensor<480x20xf32>
    %adb2b7zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adob2b7zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %advsb7zW1 = stablehlo.multiply %adb2b7zW1, %b7zW1v : tensor<480x20xf32>
    %adg2b7zW1 = stablehlo.multiply %b7zdWs1, %b7zdWs1 : tensor<480x20xf32>
    %advgb7zW1 = stablehlo.multiply %adob2b7zW1, %adg2b7zW1 : tensor<480x20xf32>
    %advnb7zW1 = stablehlo.add %advsb7zW1, %advgb7zW1 : tensor<480x20xf32>
    %adbc1b7zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adbc2b7zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %admhb7zW1 = stablehlo.divide %admnb7zW1, %adbc1b7zW1 : tensor<480x20xf32>
    %advhb7zW1 = stablehlo.divide %advnb7zW1, %adbc2b7zW1 : tensor<480x20xf32>
    %adlrb7zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adepsb7zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adsqb7zW1 = stablehlo.sqrt %advhb7zW1 : tensor<480x20xf32>
    %addenb7zW1 = stablehlo.add %adsqb7zW1, %adepsb7zW1 : tensor<480x20xf32>
    %adratb7zW1 = stablehlo.divide %admhb7zW1, %addenb7zW1 : tensor<480x20xf32>
    %adstb7zW1 = stablehlo.multiply %adlrb7zW1, %adratb7zW1 : tensor<480x20xf32>
    %adsubb7zW1 = stablehlo.subtract %b7zW1, %adstb7zW1 : tensor<480x20xf32>
    %adwdb7zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adwdlrb7zW1 = stablehlo.multiply %adwdb7zW1, %adlrb7zW1 : tensor<480x20xf32>
    %adwdpb7zW1 = stablehlo.multiply %adwdlrb7zW1, %b7zW1 : tensor<480x20xf32>
    %adnewb7zW1 = stablehlo.subtract %adsubb7zW1, %adwdpb7zW1 : tensor<480x20xf32>
    %adb1b7zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adob1b7zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %admsb7zb1 = stablehlo.multiply %adb1b7zb1, %b7zb1m : tensor<20xf32>
    %admgb7zb1 = stablehlo.multiply %adob1b7zb1, %b7zdbs1 : tensor<20xf32>
    %admnb7zb1 = stablehlo.add %admsb7zb1, %admgb7zb1 : tensor<20xf32>
    %adb2b7zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adob2b7zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %advsb7zb1 = stablehlo.multiply %adb2b7zb1, %b7zb1v : tensor<20xf32>
    %adg2b7zb1 = stablehlo.multiply %b7zdbs1, %b7zdbs1 : tensor<20xf32>
    %advgb7zb1 = stablehlo.multiply %adob2b7zb1, %adg2b7zb1 : tensor<20xf32>
    %advnb7zb1 = stablehlo.add %advsb7zb1, %advgb7zb1 : tensor<20xf32>
    %adbc1b7zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adbc2b7zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %admhb7zb1 = stablehlo.divide %admnb7zb1, %adbc1b7zb1 : tensor<20xf32>
    %advhb7zb1 = stablehlo.divide %advnb7zb1, %adbc2b7zb1 : tensor<20xf32>
    %adlrb7zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adepsb7zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adsqb7zb1 = stablehlo.sqrt %advhb7zb1 : tensor<20xf32>
    %addenb7zb1 = stablehlo.add %adsqb7zb1, %adepsb7zb1 : tensor<20xf32>
    %adratb7zb1 = stablehlo.divide %admhb7zb1, %addenb7zb1 : tensor<20xf32>
    %adstb7zb1 = stablehlo.multiply %adlrb7zb1, %adratb7zb1 : tensor<20xf32>
    %adsubb7zb1 = stablehlo.subtract %b7zb1, %adstb7zb1 : tensor<20xf32>
    %adwdb7zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adwdlrb7zb1 = stablehlo.multiply %adwdb7zb1, %adlrb7zb1 : tensor<20xf32>
    %adwdpb7zb1 = stablehlo.multiply %adwdlrb7zb1, %b7zb1 : tensor<20xf32>
    %adnewb7zb1 = stablehlo.subtract %adsubb7zb1, %adwdpb7zb1 : tensor<20xf32>
    %adb1b7zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adob1b7zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %admsb7zW2 = stablehlo.multiply %adb1b7zW2, %b7zW2m : tensor<20x480xf32>
    %admgb7zW2 = stablehlo.multiply %adob1b7zW2, %b7zdWs2 : tensor<20x480xf32>
    %admnb7zW2 = stablehlo.add %admsb7zW2, %admgb7zW2 : tensor<20x480xf32>
    %adb2b7zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adob2b7zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %advsb7zW2 = stablehlo.multiply %adb2b7zW2, %b7zW2v : tensor<20x480xf32>
    %adg2b7zW2 = stablehlo.multiply %b7zdWs2, %b7zdWs2 : tensor<20x480xf32>
    %advgb7zW2 = stablehlo.multiply %adob2b7zW2, %adg2b7zW2 : tensor<20x480xf32>
    %advnb7zW2 = stablehlo.add %advsb7zW2, %advgb7zW2 : tensor<20x480xf32>
    %adbc1b7zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adbc2b7zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %admhb7zW2 = stablehlo.divide %admnb7zW2, %adbc1b7zW2 : tensor<20x480xf32>
    %advhb7zW2 = stablehlo.divide %advnb7zW2, %adbc2b7zW2 : tensor<20x480xf32>
    %adlrb7zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adepsb7zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adsqb7zW2 = stablehlo.sqrt %advhb7zW2 : tensor<20x480xf32>
    %addenb7zW2 = stablehlo.add %adsqb7zW2, %adepsb7zW2 : tensor<20x480xf32>
    %adratb7zW2 = stablehlo.divide %admhb7zW2, %addenb7zW2 : tensor<20x480xf32>
    %adstb7zW2 = stablehlo.multiply %adlrb7zW2, %adratb7zW2 : tensor<20x480xf32>
    %adsubb7zW2 = stablehlo.subtract %b7zW2, %adstb7zW2 : tensor<20x480xf32>
    %adwdb7zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adwdlrb7zW2 = stablehlo.multiply %adwdb7zW2, %adlrb7zW2 : tensor<20x480xf32>
    %adwdpb7zW2 = stablehlo.multiply %adwdlrb7zW2, %b7zW2 : tensor<20x480xf32>
    %adnewb7zW2 = stablehlo.subtract %adsubb7zW2, %adwdpb7zW2 : tensor<20x480xf32>
    %adb1b7zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b7zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb7zb2 = stablehlo.multiply %adb1b7zb2, %b7zb2m : tensor<480xf32>
    %admgb7zb2 = stablehlo.multiply %adob1b7zb2, %b7zdbs2 : tensor<480xf32>
    %admnb7zb2 = stablehlo.add %admsb7zb2, %admgb7zb2 : tensor<480xf32>
    %adb2b7zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b7zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb7zb2 = stablehlo.multiply %adb2b7zb2, %b7zb2v : tensor<480xf32>
    %adg2b7zb2 = stablehlo.multiply %b7zdbs2, %b7zdbs2 : tensor<480xf32>
    %advgb7zb2 = stablehlo.multiply %adob2b7zb2, %adg2b7zb2 : tensor<480xf32>
    %advnb7zb2 = stablehlo.add %advsb7zb2, %advgb7zb2 : tensor<480xf32>
    %adbc1b7zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b7zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb7zb2 = stablehlo.divide %admnb7zb2, %adbc1b7zb2 : tensor<480xf32>
    %advhb7zb2 = stablehlo.divide %advnb7zb2, %adbc2b7zb2 : tensor<480xf32>
    %adlrb7zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb7zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb7zb2 = stablehlo.sqrt %advhb7zb2 : tensor<480xf32>
    %addenb7zb2 = stablehlo.add %adsqb7zb2, %adepsb7zb2 : tensor<480xf32>
    %adratb7zb2 = stablehlo.divide %admhb7zb2, %addenb7zb2 : tensor<480xf32>
    %adstb7zb2 = stablehlo.multiply %adlrb7zb2, %adratb7zb2 : tensor<480xf32>
    %adsubb7zb2 = stablehlo.subtract %b7zb2, %adstb7zb2 : tensor<480xf32>
    %adwdb7zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb7zb2 = stablehlo.multiply %adwdb7zb2, %adlrb7zb2 : tensor<480xf32>
    %adwdpb7zb2 = stablehlo.multiply %adwdlrb7zb2, %b7zb2 : tensor<480xf32>
    %adnewb7zb2 = stablehlo.subtract %adsubb7zb2, %adwdpb7zb2 : tensor<480xf32>
    %adb1b7pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adob1b7pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %admsb7pW = stablehlo.multiply %adb1b7pW, %b7pWm : tensor<80x480x1x1xf32>
    %admgb7pW = stablehlo.multiply %adob1b7pW, %b7dpW : tensor<80x480x1x1xf32>
    %admnb7pW = stablehlo.add %admsb7pW, %admgb7pW : tensor<80x480x1x1xf32>
    %adb2b7pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adob2b7pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %advsb7pW = stablehlo.multiply %adb2b7pW, %b7pWv : tensor<80x480x1x1xf32>
    %adg2b7pW = stablehlo.multiply %b7dpW, %b7dpW : tensor<80x480x1x1xf32>
    %advgb7pW = stablehlo.multiply %adob2b7pW, %adg2b7pW : tensor<80x480x1x1xf32>
    %advnb7pW = stablehlo.add %advsb7pW, %advgb7pW : tensor<80x480x1x1xf32>
    %adbc1b7pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adbc2b7pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %admhb7pW = stablehlo.divide %admnb7pW, %adbc1b7pW : tensor<80x480x1x1xf32>
    %advhb7pW = stablehlo.divide %advnb7pW, %adbc2b7pW : tensor<80x480x1x1xf32>
    %adlrb7pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adepsb7pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adsqb7pW = stablehlo.sqrt %advhb7pW : tensor<80x480x1x1xf32>
    %addenb7pW = stablehlo.add %adsqb7pW, %adepsb7pW : tensor<80x480x1x1xf32>
    %adratb7pW = stablehlo.divide %admhb7pW, %addenb7pW : tensor<80x480x1x1xf32>
    %adstb7pW = stablehlo.multiply %adlrb7pW, %adratb7pW : tensor<80x480x1x1xf32>
    %adsubb7pW = stablehlo.subtract %b7pW, %adstb7pW : tensor<80x480x1x1xf32>
    %adwdb7pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adwdlrb7pW = stablehlo.multiply %adwdb7pW, %adlrb7pW : tensor<80x480x1x1xf32>
    %adwdpb7pW = stablehlo.multiply %adwdlrb7pW, %b7pW : tensor<80x480x1x1xf32>
    %adnewb7pW = stablehlo.subtract %adsubb7pW, %adwdpb7pW : tensor<80x480x1x1xf32>
    %adb1b7pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b7pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb7pb = stablehlo.multiply %adb1b7pb, %b7pbm : tensor<80xf32>
    %admgb7pb = stablehlo.multiply %adob1b7pb, %b7dpb : tensor<80xf32>
    %admnb7pb = stablehlo.add %admsb7pb, %admgb7pb : tensor<80xf32>
    %adb2b7pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b7pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb7pb = stablehlo.multiply %adb2b7pb, %b7pbv : tensor<80xf32>
    %adg2b7pb = stablehlo.multiply %b7dpb, %b7dpb : tensor<80xf32>
    %advgb7pb = stablehlo.multiply %adob2b7pb, %adg2b7pb : tensor<80xf32>
    %advnb7pb = stablehlo.add %advsb7pb, %advgb7pb : tensor<80xf32>
    %adbc1b7pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b7pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb7pb = stablehlo.divide %admnb7pb, %adbc1b7pb : tensor<80xf32>
    %advhb7pb = stablehlo.divide %advnb7pb, %adbc2b7pb : tensor<80xf32>
    %adlrb7pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb7pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb7pb = stablehlo.sqrt %advhb7pb : tensor<80xf32>
    %addenb7pb = stablehlo.add %adsqb7pb, %adepsb7pb : tensor<80xf32>
    %adratb7pb = stablehlo.divide %admhb7pb, %addenb7pb : tensor<80xf32>
    %adstb7pb = stablehlo.multiply %adlrb7pb, %adratb7pb : tensor<80xf32>
    %adsubb7pb = stablehlo.subtract %b7pb, %adstb7pb : tensor<80xf32>
    %adwdb7pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb7pb = stablehlo.multiply %adwdb7pb, %adlrb7pb : tensor<80xf32>
    %adwdpb7pb = stablehlo.multiply %adwdlrb7pb, %b7pb : tensor<80xf32>
    %adnewb7pb = stablehlo.subtract %adsubb7pb, %adwdpb7pb : tensor<80xf32>
    %adb1b7pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b7pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb7pg = stablehlo.multiply %adb1b7pg, %b7pgm : tensor<80xf32>
    %admgb7pg = stablehlo.multiply %adob1b7pg, %b7dpndg : tensor<80xf32>
    %admnb7pg = stablehlo.add %admsb7pg, %admgb7pg : tensor<80xf32>
    %adb2b7pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b7pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb7pg = stablehlo.multiply %adb2b7pg, %b7pgv : tensor<80xf32>
    %adg2b7pg = stablehlo.multiply %b7dpndg, %b7dpndg : tensor<80xf32>
    %advgb7pg = stablehlo.multiply %adob2b7pg, %adg2b7pg : tensor<80xf32>
    %advnb7pg = stablehlo.add %advsb7pg, %advgb7pg : tensor<80xf32>
    %adbc1b7pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b7pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb7pg = stablehlo.divide %admnb7pg, %adbc1b7pg : tensor<80xf32>
    %advhb7pg = stablehlo.divide %advnb7pg, %adbc2b7pg : tensor<80xf32>
    %adlrb7pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb7pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb7pg = stablehlo.sqrt %advhb7pg : tensor<80xf32>
    %addenb7pg = stablehlo.add %adsqb7pg, %adepsb7pg : tensor<80xf32>
    %adratb7pg = stablehlo.divide %admhb7pg, %addenb7pg : tensor<80xf32>
    %adstb7pg = stablehlo.multiply %adlrb7pg, %adratb7pg : tensor<80xf32>
    %adsubb7pg = stablehlo.subtract %b7pg, %adstb7pg : tensor<80xf32>
    %adwdb7pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb7pg = stablehlo.multiply %adwdb7pg, %adlrb7pg : tensor<80xf32>
    %adwdpb7pg = stablehlo.multiply %adwdlrb7pg, %b7pg : tensor<80xf32>
    %adnewb7pg = stablehlo.subtract %adsubb7pg, %adwdpb7pg : tensor<80xf32>
    %adb1b7pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b7pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb7pbt = stablehlo.multiply %adb1b7pbt, %b7pbtm : tensor<80xf32>
    %admgb7pbt = stablehlo.multiply %adob1b7pbt, %b7dpndb : tensor<80xf32>
    %admnb7pbt = stablehlo.add %admsb7pbt, %admgb7pbt : tensor<80xf32>
    %adb2b7pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b7pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb7pbt = stablehlo.multiply %adb2b7pbt, %b7pbtv : tensor<80xf32>
    %adg2b7pbt = stablehlo.multiply %b7dpndb, %b7dpndb : tensor<80xf32>
    %advgb7pbt = stablehlo.multiply %adob2b7pbt, %adg2b7pbt : tensor<80xf32>
    %advnb7pbt = stablehlo.add %advsb7pbt, %advgb7pbt : tensor<80xf32>
    %adbc1b7pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b7pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb7pbt = stablehlo.divide %admnb7pbt, %adbc1b7pbt : tensor<80xf32>
    %advhb7pbt = stablehlo.divide %advnb7pbt, %adbc2b7pbt : tensor<80xf32>
    %adlrb7pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb7pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb7pbt = stablehlo.sqrt %advhb7pbt : tensor<80xf32>
    %addenb7pbt = stablehlo.add %adsqb7pbt, %adepsb7pbt : tensor<80xf32>
    %adratb7pbt = stablehlo.divide %admhb7pbt, %addenb7pbt : tensor<80xf32>
    %adstb7pbt = stablehlo.multiply %adlrb7pbt, %adratb7pbt : tensor<80xf32>
    %adsubb7pbt = stablehlo.subtract %b7pbt, %adstb7pbt : tensor<80xf32>
    %adwdb7pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb7pbt = stablehlo.multiply %adwdb7pbt, %adlrb7pbt : tensor<80xf32>
    %adwdpb7pbt = stablehlo.multiply %adwdlrb7pbt, %b7pbt : tensor<80xf32>
    %adnewb7pbt = stablehlo.subtract %adsubb7pbt, %adwdpb7pbt : tensor<80xf32>
    %adb1b8eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adob1b8eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %admsb8eW = stablehlo.multiply %adb1b8eW, %b8eWm : tensor<480x80x1x1xf32>
    %admgb8eW = stablehlo.multiply %adob1b8eW, %b8deW : tensor<480x80x1x1xf32>
    %admnb8eW = stablehlo.add %admsb8eW, %admgb8eW : tensor<480x80x1x1xf32>
    %adb2b8eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adob2b8eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %advsb8eW = stablehlo.multiply %adb2b8eW, %b8eWv : tensor<480x80x1x1xf32>
    %adg2b8eW = stablehlo.multiply %b8deW, %b8deW : tensor<480x80x1x1xf32>
    %advgb8eW = stablehlo.multiply %adob2b8eW, %adg2b8eW : tensor<480x80x1x1xf32>
    %advnb8eW = stablehlo.add %advsb8eW, %advgb8eW : tensor<480x80x1x1xf32>
    %adbc1b8eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adbc2b8eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %admhb8eW = stablehlo.divide %admnb8eW, %adbc1b8eW : tensor<480x80x1x1xf32>
    %advhb8eW = stablehlo.divide %advnb8eW, %adbc2b8eW : tensor<480x80x1x1xf32>
    %adlrb8eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adepsb8eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adsqb8eW = stablehlo.sqrt %advhb8eW : tensor<480x80x1x1xf32>
    %addenb8eW = stablehlo.add %adsqb8eW, %adepsb8eW : tensor<480x80x1x1xf32>
    %adratb8eW = stablehlo.divide %admhb8eW, %addenb8eW : tensor<480x80x1x1xf32>
    %adstb8eW = stablehlo.multiply %adlrb8eW, %adratb8eW : tensor<480x80x1x1xf32>
    %adsubb8eW = stablehlo.subtract %b8eW, %adstb8eW : tensor<480x80x1x1xf32>
    %adwdb8eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adwdlrb8eW = stablehlo.multiply %adwdb8eW, %adlrb8eW : tensor<480x80x1x1xf32>
    %adwdpb8eW = stablehlo.multiply %adwdlrb8eW, %b8eW : tensor<480x80x1x1xf32>
    %adnewb8eW = stablehlo.subtract %adsubb8eW, %adwdpb8eW : tensor<480x80x1x1xf32>
    %adb1b8eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b8eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb8eb = stablehlo.multiply %adb1b8eb, %b8ebm : tensor<480xf32>
    %admgb8eb = stablehlo.multiply %adob1b8eb, %b8deb : tensor<480xf32>
    %admnb8eb = stablehlo.add %admsb8eb, %admgb8eb : tensor<480xf32>
    %adb2b8eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b8eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb8eb = stablehlo.multiply %adb2b8eb, %b8ebv : tensor<480xf32>
    %adg2b8eb = stablehlo.multiply %b8deb, %b8deb : tensor<480xf32>
    %advgb8eb = stablehlo.multiply %adob2b8eb, %adg2b8eb : tensor<480xf32>
    %advnb8eb = stablehlo.add %advsb8eb, %advgb8eb : tensor<480xf32>
    %adbc1b8eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b8eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb8eb = stablehlo.divide %admnb8eb, %adbc1b8eb : tensor<480xf32>
    %advhb8eb = stablehlo.divide %advnb8eb, %adbc2b8eb : tensor<480xf32>
    %adlrb8eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb8eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb8eb = stablehlo.sqrt %advhb8eb : tensor<480xf32>
    %addenb8eb = stablehlo.add %adsqb8eb, %adepsb8eb : tensor<480xf32>
    %adratb8eb = stablehlo.divide %admhb8eb, %addenb8eb : tensor<480xf32>
    %adstb8eb = stablehlo.multiply %adlrb8eb, %adratb8eb : tensor<480xf32>
    %adsubb8eb = stablehlo.subtract %b8eb, %adstb8eb : tensor<480xf32>
    %adwdb8eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb8eb = stablehlo.multiply %adwdb8eb, %adlrb8eb : tensor<480xf32>
    %adwdpb8eb = stablehlo.multiply %adwdlrb8eb, %b8eb : tensor<480xf32>
    %adnewb8eb = stablehlo.subtract %adsubb8eb, %adwdpb8eb : tensor<480xf32>
    %adb1b8eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b8eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb8eg = stablehlo.multiply %adb1b8eg, %b8egm : tensor<480xf32>
    %admgb8eg = stablehlo.multiply %adob1b8eg, %b8dendg : tensor<480xf32>
    %admnb8eg = stablehlo.add %admsb8eg, %admgb8eg : tensor<480xf32>
    %adb2b8eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b8eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb8eg = stablehlo.multiply %adb2b8eg, %b8egv : tensor<480xf32>
    %adg2b8eg = stablehlo.multiply %b8dendg, %b8dendg : tensor<480xf32>
    %advgb8eg = stablehlo.multiply %adob2b8eg, %adg2b8eg : tensor<480xf32>
    %advnb8eg = stablehlo.add %advsb8eg, %advgb8eg : tensor<480xf32>
    %adbc1b8eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b8eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb8eg = stablehlo.divide %admnb8eg, %adbc1b8eg : tensor<480xf32>
    %advhb8eg = stablehlo.divide %advnb8eg, %adbc2b8eg : tensor<480xf32>
    %adlrb8eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb8eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb8eg = stablehlo.sqrt %advhb8eg : tensor<480xf32>
    %addenb8eg = stablehlo.add %adsqb8eg, %adepsb8eg : tensor<480xf32>
    %adratb8eg = stablehlo.divide %admhb8eg, %addenb8eg : tensor<480xf32>
    %adstb8eg = stablehlo.multiply %adlrb8eg, %adratb8eg : tensor<480xf32>
    %adsubb8eg = stablehlo.subtract %b8eg, %adstb8eg : tensor<480xf32>
    %adwdb8eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb8eg = stablehlo.multiply %adwdb8eg, %adlrb8eg : tensor<480xf32>
    %adwdpb8eg = stablehlo.multiply %adwdlrb8eg, %b8eg : tensor<480xf32>
    %adnewb8eg = stablehlo.subtract %adsubb8eg, %adwdpb8eg : tensor<480xf32>
    %adb1b8ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b8ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb8ebt = stablehlo.multiply %adb1b8ebt, %b8ebtm : tensor<480xf32>
    %admgb8ebt = stablehlo.multiply %adob1b8ebt, %b8dendb : tensor<480xf32>
    %admnb8ebt = stablehlo.add %admsb8ebt, %admgb8ebt : tensor<480xf32>
    %adb2b8ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b8ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb8ebt = stablehlo.multiply %adb2b8ebt, %b8ebtv : tensor<480xf32>
    %adg2b8ebt = stablehlo.multiply %b8dendb, %b8dendb : tensor<480xf32>
    %advgb8ebt = stablehlo.multiply %adob2b8ebt, %adg2b8ebt : tensor<480xf32>
    %advnb8ebt = stablehlo.add %advsb8ebt, %advgb8ebt : tensor<480xf32>
    %adbc1b8ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b8ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb8ebt = stablehlo.divide %admnb8ebt, %adbc1b8ebt : tensor<480xf32>
    %advhb8ebt = stablehlo.divide %advnb8ebt, %adbc2b8ebt : tensor<480xf32>
    %adlrb8ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb8ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb8ebt = stablehlo.sqrt %advhb8ebt : tensor<480xf32>
    %addenb8ebt = stablehlo.add %adsqb8ebt, %adepsb8ebt : tensor<480xf32>
    %adratb8ebt = stablehlo.divide %admhb8ebt, %addenb8ebt : tensor<480xf32>
    %adstb8ebt = stablehlo.multiply %adlrb8ebt, %adratb8ebt : tensor<480xf32>
    %adsubb8ebt = stablehlo.subtract %b8ebt, %adstb8ebt : tensor<480xf32>
    %adwdb8ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb8ebt = stablehlo.multiply %adwdb8ebt, %adlrb8ebt : tensor<480xf32>
    %adwdpb8ebt = stablehlo.multiply %adwdlrb8ebt, %b8ebt : tensor<480xf32>
    %adnewb8ebt = stablehlo.subtract %adsubb8ebt, %adwdpb8ebt : tensor<480xf32>
    %adb1b8dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adob1b8dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %admsb8dW = stablehlo.multiply %adb1b8dW, %b8dWm : tensor<480x1x3x3xf32>
    %admgb8dW = stablehlo.multiply %adob1b8dW, %b8ddW : tensor<480x1x3x3xf32>
    %admnb8dW = stablehlo.add %admsb8dW, %admgb8dW : tensor<480x1x3x3xf32>
    %adb2b8dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adob2b8dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %advsb8dW = stablehlo.multiply %adb2b8dW, %b8dWv : tensor<480x1x3x3xf32>
    %adg2b8dW = stablehlo.multiply %b8ddW, %b8ddW : tensor<480x1x3x3xf32>
    %advgb8dW = stablehlo.multiply %adob2b8dW, %adg2b8dW : tensor<480x1x3x3xf32>
    %advnb8dW = stablehlo.add %advsb8dW, %advgb8dW : tensor<480x1x3x3xf32>
    %adbc1b8dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adbc2b8dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %admhb8dW = stablehlo.divide %admnb8dW, %adbc1b8dW : tensor<480x1x3x3xf32>
    %advhb8dW = stablehlo.divide %advnb8dW, %adbc2b8dW : tensor<480x1x3x3xf32>
    %adlrb8dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adepsb8dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adsqb8dW = stablehlo.sqrt %advhb8dW : tensor<480x1x3x3xf32>
    %addenb8dW = stablehlo.add %adsqb8dW, %adepsb8dW : tensor<480x1x3x3xf32>
    %adratb8dW = stablehlo.divide %admhb8dW, %addenb8dW : tensor<480x1x3x3xf32>
    %adstb8dW = stablehlo.multiply %adlrb8dW, %adratb8dW : tensor<480x1x3x3xf32>
    %adsubb8dW = stablehlo.subtract %b8dW, %adstb8dW : tensor<480x1x3x3xf32>
    %adwdb8dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x1x3x3xf32>
    %adwdlrb8dW = stablehlo.multiply %adwdb8dW, %adlrb8dW : tensor<480x1x3x3xf32>
    %adwdpb8dW = stablehlo.multiply %adwdlrb8dW, %b8dW : tensor<480x1x3x3xf32>
    %adnewb8dW = stablehlo.subtract %adsubb8dW, %adwdpb8dW : tensor<480x1x3x3xf32>
    %adb1b8db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b8db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb8db = stablehlo.multiply %adb1b8db, %b8dbm : tensor<480xf32>
    %admgb8db = stablehlo.multiply %adob1b8db, %b8ddb : tensor<480xf32>
    %admnb8db = stablehlo.add %admsb8db, %admgb8db : tensor<480xf32>
    %adb2b8db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b8db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb8db = stablehlo.multiply %adb2b8db, %b8dbv : tensor<480xf32>
    %adg2b8db = stablehlo.multiply %b8ddb, %b8ddb : tensor<480xf32>
    %advgb8db = stablehlo.multiply %adob2b8db, %adg2b8db : tensor<480xf32>
    %advnb8db = stablehlo.add %advsb8db, %advgb8db : tensor<480xf32>
    %adbc1b8db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b8db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb8db = stablehlo.divide %admnb8db, %adbc1b8db : tensor<480xf32>
    %advhb8db = stablehlo.divide %advnb8db, %adbc2b8db : tensor<480xf32>
    %adlrb8db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb8db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb8db = stablehlo.sqrt %advhb8db : tensor<480xf32>
    %addenb8db = stablehlo.add %adsqb8db, %adepsb8db : tensor<480xf32>
    %adratb8db = stablehlo.divide %admhb8db, %addenb8db : tensor<480xf32>
    %adstb8db = stablehlo.multiply %adlrb8db, %adratb8db : tensor<480xf32>
    %adsubb8db = stablehlo.subtract %b8db, %adstb8db : tensor<480xf32>
    %adwdb8db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb8db = stablehlo.multiply %adwdb8db, %adlrb8db : tensor<480xf32>
    %adwdpb8db = stablehlo.multiply %adwdlrb8db, %b8db : tensor<480xf32>
    %adnewb8db = stablehlo.subtract %adsubb8db, %adwdpb8db : tensor<480xf32>
    %adb1b8dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b8dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb8dg = stablehlo.multiply %adb1b8dg, %b8dgm : tensor<480xf32>
    %admgb8dg = stablehlo.multiply %adob1b8dg, %b8ddndg : tensor<480xf32>
    %admnb8dg = stablehlo.add %admsb8dg, %admgb8dg : tensor<480xf32>
    %adb2b8dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b8dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb8dg = stablehlo.multiply %adb2b8dg, %b8dgv : tensor<480xf32>
    %adg2b8dg = stablehlo.multiply %b8ddndg, %b8ddndg : tensor<480xf32>
    %advgb8dg = stablehlo.multiply %adob2b8dg, %adg2b8dg : tensor<480xf32>
    %advnb8dg = stablehlo.add %advsb8dg, %advgb8dg : tensor<480xf32>
    %adbc1b8dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b8dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb8dg = stablehlo.divide %admnb8dg, %adbc1b8dg : tensor<480xf32>
    %advhb8dg = stablehlo.divide %advnb8dg, %adbc2b8dg : tensor<480xf32>
    %adlrb8dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb8dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb8dg = stablehlo.sqrt %advhb8dg : tensor<480xf32>
    %addenb8dg = stablehlo.add %adsqb8dg, %adepsb8dg : tensor<480xf32>
    %adratb8dg = stablehlo.divide %admhb8dg, %addenb8dg : tensor<480xf32>
    %adstb8dg = stablehlo.multiply %adlrb8dg, %adratb8dg : tensor<480xf32>
    %adsubb8dg = stablehlo.subtract %b8dg, %adstb8dg : tensor<480xf32>
    %adwdb8dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb8dg = stablehlo.multiply %adwdb8dg, %adlrb8dg : tensor<480xf32>
    %adwdpb8dg = stablehlo.multiply %adwdlrb8dg, %b8dg : tensor<480xf32>
    %adnewb8dg = stablehlo.subtract %adsubb8dg, %adwdpb8dg : tensor<480xf32>
    %adb1b8dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b8dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb8dbt = stablehlo.multiply %adb1b8dbt, %b8dbtm : tensor<480xf32>
    %admgb8dbt = stablehlo.multiply %adob1b8dbt, %b8ddndb : tensor<480xf32>
    %admnb8dbt = stablehlo.add %admsb8dbt, %admgb8dbt : tensor<480xf32>
    %adb2b8dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b8dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb8dbt = stablehlo.multiply %adb2b8dbt, %b8dbtv : tensor<480xf32>
    %adg2b8dbt = stablehlo.multiply %b8ddndb, %b8ddndb : tensor<480xf32>
    %advgb8dbt = stablehlo.multiply %adob2b8dbt, %adg2b8dbt : tensor<480xf32>
    %advnb8dbt = stablehlo.add %advsb8dbt, %advgb8dbt : tensor<480xf32>
    %adbc1b8dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b8dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb8dbt = stablehlo.divide %admnb8dbt, %adbc1b8dbt : tensor<480xf32>
    %advhb8dbt = stablehlo.divide %advnb8dbt, %adbc2b8dbt : tensor<480xf32>
    %adlrb8dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb8dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb8dbt = stablehlo.sqrt %advhb8dbt : tensor<480xf32>
    %addenb8dbt = stablehlo.add %adsqb8dbt, %adepsb8dbt : tensor<480xf32>
    %adratb8dbt = stablehlo.divide %admhb8dbt, %addenb8dbt : tensor<480xf32>
    %adstb8dbt = stablehlo.multiply %adlrb8dbt, %adratb8dbt : tensor<480xf32>
    %adsubb8dbt = stablehlo.subtract %b8dbt, %adstb8dbt : tensor<480xf32>
    %adwdb8dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb8dbt = stablehlo.multiply %adwdb8dbt, %adlrb8dbt : tensor<480xf32>
    %adwdpb8dbt = stablehlo.multiply %adwdlrb8dbt, %b8dbt : tensor<480xf32>
    %adnewb8dbt = stablehlo.subtract %adsubb8dbt, %adwdpb8dbt : tensor<480xf32>
    %adb1b8zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adob1b8zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %admsb8zW1 = stablehlo.multiply %adb1b8zW1, %b8zW1m : tensor<480x20xf32>
    %admgb8zW1 = stablehlo.multiply %adob1b8zW1, %b8zdWs1 : tensor<480x20xf32>
    %admnb8zW1 = stablehlo.add %admsb8zW1, %admgb8zW1 : tensor<480x20xf32>
    %adb2b8zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adob2b8zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %advsb8zW1 = stablehlo.multiply %adb2b8zW1, %b8zW1v : tensor<480x20xf32>
    %adg2b8zW1 = stablehlo.multiply %b8zdWs1, %b8zdWs1 : tensor<480x20xf32>
    %advgb8zW1 = stablehlo.multiply %adob2b8zW1, %adg2b8zW1 : tensor<480x20xf32>
    %advnb8zW1 = stablehlo.add %advsb8zW1, %advgb8zW1 : tensor<480x20xf32>
    %adbc1b8zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adbc2b8zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %admhb8zW1 = stablehlo.divide %admnb8zW1, %adbc1b8zW1 : tensor<480x20xf32>
    %advhb8zW1 = stablehlo.divide %advnb8zW1, %adbc2b8zW1 : tensor<480x20xf32>
    %adlrb8zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adepsb8zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adsqb8zW1 = stablehlo.sqrt %advhb8zW1 : tensor<480x20xf32>
    %addenb8zW1 = stablehlo.add %adsqb8zW1, %adepsb8zW1 : tensor<480x20xf32>
    %adratb8zW1 = stablehlo.divide %admhb8zW1, %addenb8zW1 : tensor<480x20xf32>
    %adstb8zW1 = stablehlo.multiply %adlrb8zW1, %adratb8zW1 : tensor<480x20xf32>
    %adsubb8zW1 = stablehlo.subtract %b8zW1, %adstb8zW1 : tensor<480x20xf32>
    %adwdb8zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adwdlrb8zW1 = stablehlo.multiply %adwdb8zW1, %adlrb8zW1 : tensor<480x20xf32>
    %adwdpb8zW1 = stablehlo.multiply %adwdlrb8zW1, %b8zW1 : tensor<480x20xf32>
    %adnewb8zW1 = stablehlo.subtract %adsubb8zW1, %adwdpb8zW1 : tensor<480x20xf32>
    %adb1b8zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adob1b8zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %admsb8zb1 = stablehlo.multiply %adb1b8zb1, %b8zb1m : tensor<20xf32>
    %admgb8zb1 = stablehlo.multiply %adob1b8zb1, %b8zdbs1 : tensor<20xf32>
    %admnb8zb1 = stablehlo.add %admsb8zb1, %admgb8zb1 : tensor<20xf32>
    %adb2b8zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adob2b8zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %advsb8zb1 = stablehlo.multiply %adb2b8zb1, %b8zb1v : tensor<20xf32>
    %adg2b8zb1 = stablehlo.multiply %b8zdbs1, %b8zdbs1 : tensor<20xf32>
    %advgb8zb1 = stablehlo.multiply %adob2b8zb1, %adg2b8zb1 : tensor<20xf32>
    %advnb8zb1 = stablehlo.add %advsb8zb1, %advgb8zb1 : tensor<20xf32>
    %adbc1b8zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adbc2b8zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %admhb8zb1 = stablehlo.divide %admnb8zb1, %adbc1b8zb1 : tensor<20xf32>
    %advhb8zb1 = stablehlo.divide %advnb8zb1, %adbc2b8zb1 : tensor<20xf32>
    %adlrb8zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adepsb8zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adsqb8zb1 = stablehlo.sqrt %advhb8zb1 : tensor<20xf32>
    %addenb8zb1 = stablehlo.add %adsqb8zb1, %adepsb8zb1 : tensor<20xf32>
    %adratb8zb1 = stablehlo.divide %admhb8zb1, %addenb8zb1 : tensor<20xf32>
    %adstb8zb1 = stablehlo.multiply %adlrb8zb1, %adratb8zb1 : tensor<20xf32>
    %adsubb8zb1 = stablehlo.subtract %b8zb1, %adstb8zb1 : tensor<20xf32>
    %adwdb8zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adwdlrb8zb1 = stablehlo.multiply %adwdb8zb1, %adlrb8zb1 : tensor<20xf32>
    %adwdpb8zb1 = stablehlo.multiply %adwdlrb8zb1, %b8zb1 : tensor<20xf32>
    %adnewb8zb1 = stablehlo.subtract %adsubb8zb1, %adwdpb8zb1 : tensor<20xf32>
    %adb1b8zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adob1b8zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %admsb8zW2 = stablehlo.multiply %adb1b8zW2, %b8zW2m : tensor<20x480xf32>
    %admgb8zW2 = stablehlo.multiply %adob1b8zW2, %b8zdWs2 : tensor<20x480xf32>
    %admnb8zW2 = stablehlo.add %admsb8zW2, %admgb8zW2 : tensor<20x480xf32>
    %adb2b8zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adob2b8zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %advsb8zW2 = stablehlo.multiply %adb2b8zW2, %b8zW2v : tensor<20x480xf32>
    %adg2b8zW2 = stablehlo.multiply %b8zdWs2, %b8zdWs2 : tensor<20x480xf32>
    %advgb8zW2 = stablehlo.multiply %adob2b8zW2, %adg2b8zW2 : tensor<20x480xf32>
    %advnb8zW2 = stablehlo.add %advsb8zW2, %advgb8zW2 : tensor<20x480xf32>
    %adbc1b8zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adbc2b8zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %admhb8zW2 = stablehlo.divide %admnb8zW2, %adbc1b8zW2 : tensor<20x480xf32>
    %advhb8zW2 = stablehlo.divide %advnb8zW2, %adbc2b8zW2 : tensor<20x480xf32>
    %adlrb8zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adepsb8zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adsqb8zW2 = stablehlo.sqrt %advhb8zW2 : tensor<20x480xf32>
    %addenb8zW2 = stablehlo.add %adsqb8zW2, %adepsb8zW2 : tensor<20x480xf32>
    %adratb8zW2 = stablehlo.divide %admhb8zW2, %addenb8zW2 : tensor<20x480xf32>
    %adstb8zW2 = stablehlo.multiply %adlrb8zW2, %adratb8zW2 : tensor<20x480xf32>
    %adsubb8zW2 = stablehlo.subtract %b8zW2, %adstb8zW2 : tensor<20x480xf32>
    %adwdb8zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adwdlrb8zW2 = stablehlo.multiply %adwdb8zW2, %adlrb8zW2 : tensor<20x480xf32>
    %adwdpb8zW2 = stablehlo.multiply %adwdlrb8zW2, %b8zW2 : tensor<20x480xf32>
    %adnewb8zW2 = stablehlo.subtract %adsubb8zW2, %adwdpb8zW2 : tensor<20x480xf32>
    %adb1b8zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b8zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb8zb2 = stablehlo.multiply %adb1b8zb2, %b8zb2m : tensor<480xf32>
    %admgb8zb2 = stablehlo.multiply %adob1b8zb2, %b8zdbs2 : tensor<480xf32>
    %admnb8zb2 = stablehlo.add %admsb8zb2, %admgb8zb2 : tensor<480xf32>
    %adb2b8zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b8zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb8zb2 = stablehlo.multiply %adb2b8zb2, %b8zb2v : tensor<480xf32>
    %adg2b8zb2 = stablehlo.multiply %b8zdbs2, %b8zdbs2 : tensor<480xf32>
    %advgb8zb2 = stablehlo.multiply %adob2b8zb2, %adg2b8zb2 : tensor<480xf32>
    %advnb8zb2 = stablehlo.add %advsb8zb2, %advgb8zb2 : tensor<480xf32>
    %adbc1b8zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b8zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb8zb2 = stablehlo.divide %admnb8zb2, %adbc1b8zb2 : tensor<480xf32>
    %advhb8zb2 = stablehlo.divide %advnb8zb2, %adbc2b8zb2 : tensor<480xf32>
    %adlrb8zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb8zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb8zb2 = stablehlo.sqrt %advhb8zb2 : tensor<480xf32>
    %addenb8zb2 = stablehlo.add %adsqb8zb2, %adepsb8zb2 : tensor<480xf32>
    %adratb8zb2 = stablehlo.divide %admhb8zb2, %addenb8zb2 : tensor<480xf32>
    %adstb8zb2 = stablehlo.multiply %adlrb8zb2, %adratb8zb2 : tensor<480xf32>
    %adsubb8zb2 = stablehlo.subtract %b8zb2, %adstb8zb2 : tensor<480xf32>
    %adwdb8zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb8zb2 = stablehlo.multiply %adwdb8zb2, %adlrb8zb2 : tensor<480xf32>
    %adwdpb8zb2 = stablehlo.multiply %adwdlrb8zb2, %b8zb2 : tensor<480xf32>
    %adnewb8zb2 = stablehlo.subtract %adsubb8zb2, %adwdpb8zb2 : tensor<480xf32>
    %adb1b8pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adob1b8pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %admsb8pW = stablehlo.multiply %adb1b8pW, %b8pWm : tensor<80x480x1x1xf32>
    %admgb8pW = stablehlo.multiply %adob1b8pW, %b8dpW : tensor<80x480x1x1xf32>
    %admnb8pW = stablehlo.add %admsb8pW, %admgb8pW : tensor<80x480x1x1xf32>
    %adb2b8pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adob2b8pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %advsb8pW = stablehlo.multiply %adb2b8pW, %b8pWv : tensor<80x480x1x1xf32>
    %adg2b8pW = stablehlo.multiply %b8dpW, %b8dpW : tensor<80x480x1x1xf32>
    %advgb8pW = stablehlo.multiply %adob2b8pW, %adg2b8pW : tensor<80x480x1x1xf32>
    %advnb8pW = stablehlo.add %advsb8pW, %advgb8pW : tensor<80x480x1x1xf32>
    %adbc1b8pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adbc2b8pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %admhb8pW = stablehlo.divide %admnb8pW, %adbc1b8pW : tensor<80x480x1x1xf32>
    %advhb8pW = stablehlo.divide %advnb8pW, %adbc2b8pW : tensor<80x480x1x1xf32>
    %adlrb8pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adepsb8pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adsqb8pW = stablehlo.sqrt %advhb8pW : tensor<80x480x1x1xf32>
    %addenb8pW = stablehlo.add %adsqb8pW, %adepsb8pW : tensor<80x480x1x1xf32>
    %adratb8pW = stablehlo.divide %admhb8pW, %addenb8pW : tensor<80x480x1x1xf32>
    %adstb8pW = stablehlo.multiply %adlrb8pW, %adratb8pW : tensor<80x480x1x1xf32>
    %adsubb8pW = stablehlo.subtract %b8pW, %adstb8pW : tensor<80x480x1x1xf32>
    %adwdb8pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80x480x1x1xf32>
    %adwdlrb8pW = stablehlo.multiply %adwdb8pW, %adlrb8pW : tensor<80x480x1x1xf32>
    %adwdpb8pW = stablehlo.multiply %adwdlrb8pW, %b8pW : tensor<80x480x1x1xf32>
    %adnewb8pW = stablehlo.subtract %adsubb8pW, %adwdpb8pW : tensor<80x480x1x1xf32>
    %adb1b8pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b8pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb8pb = stablehlo.multiply %adb1b8pb, %b8pbm : tensor<80xf32>
    %admgb8pb = stablehlo.multiply %adob1b8pb, %b8dpb : tensor<80xf32>
    %admnb8pb = stablehlo.add %admsb8pb, %admgb8pb : tensor<80xf32>
    %adb2b8pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b8pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb8pb = stablehlo.multiply %adb2b8pb, %b8pbv : tensor<80xf32>
    %adg2b8pb = stablehlo.multiply %b8dpb, %b8dpb : tensor<80xf32>
    %advgb8pb = stablehlo.multiply %adob2b8pb, %adg2b8pb : tensor<80xf32>
    %advnb8pb = stablehlo.add %advsb8pb, %advgb8pb : tensor<80xf32>
    %adbc1b8pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b8pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb8pb = stablehlo.divide %admnb8pb, %adbc1b8pb : tensor<80xf32>
    %advhb8pb = stablehlo.divide %advnb8pb, %adbc2b8pb : tensor<80xf32>
    %adlrb8pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb8pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb8pb = stablehlo.sqrt %advhb8pb : tensor<80xf32>
    %addenb8pb = stablehlo.add %adsqb8pb, %adepsb8pb : tensor<80xf32>
    %adratb8pb = stablehlo.divide %admhb8pb, %addenb8pb : tensor<80xf32>
    %adstb8pb = stablehlo.multiply %adlrb8pb, %adratb8pb : tensor<80xf32>
    %adsubb8pb = stablehlo.subtract %b8pb, %adstb8pb : tensor<80xf32>
    %adwdb8pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb8pb = stablehlo.multiply %adwdb8pb, %adlrb8pb : tensor<80xf32>
    %adwdpb8pb = stablehlo.multiply %adwdlrb8pb, %b8pb : tensor<80xf32>
    %adnewb8pb = stablehlo.subtract %adsubb8pb, %adwdpb8pb : tensor<80xf32>
    %adb1b8pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b8pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb8pg = stablehlo.multiply %adb1b8pg, %b8pgm : tensor<80xf32>
    %admgb8pg = stablehlo.multiply %adob1b8pg, %b8dpndg : tensor<80xf32>
    %admnb8pg = stablehlo.add %admsb8pg, %admgb8pg : tensor<80xf32>
    %adb2b8pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b8pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb8pg = stablehlo.multiply %adb2b8pg, %b8pgv : tensor<80xf32>
    %adg2b8pg = stablehlo.multiply %b8dpndg, %b8dpndg : tensor<80xf32>
    %advgb8pg = stablehlo.multiply %adob2b8pg, %adg2b8pg : tensor<80xf32>
    %advnb8pg = stablehlo.add %advsb8pg, %advgb8pg : tensor<80xf32>
    %adbc1b8pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b8pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb8pg = stablehlo.divide %admnb8pg, %adbc1b8pg : tensor<80xf32>
    %advhb8pg = stablehlo.divide %advnb8pg, %adbc2b8pg : tensor<80xf32>
    %adlrb8pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb8pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb8pg = stablehlo.sqrt %advhb8pg : tensor<80xf32>
    %addenb8pg = stablehlo.add %adsqb8pg, %adepsb8pg : tensor<80xf32>
    %adratb8pg = stablehlo.divide %admhb8pg, %addenb8pg : tensor<80xf32>
    %adstb8pg = stablehlo.multiply %adlrb8pg, %adratb8pg : tensor<80xf32>
    %adsubb8pg = stablehlo.subtract %b8pg, %adstb8pg : tensor<80xf32>
    %adwdb8pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb8pg = stablehlo.multiply %adwdb8pg, %adlrb8pg : tensor<80xf32>
    %adwdpb8pg = stablehlo.multiply %adwdlrb8pg, %b8pg : tensor<80xf32>
    %adnewb8pg = stablehlo.subtract %adsubb8pg, %adwdpb8pg : tensor<80xf32>
    %adb1b8pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob1b8pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admsb8pbt = stablehlo.multiply %adb1b8pbt, %b8pbtm : tensor<80xf32>
    %admgb8pbt = stablehlo.multiply %adob1b8pbt, %b8dpndb : tensor<80xf32>
    %admnb8pbt = stablehlo.add %admsb8pbt, %admgb8pbt : tensor<80xf32>
    %adb2b8pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adob2b8pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %advsb8pbt = stablehlo.multiply %adb2b8pbt, %b8pbtv : tensor<80xf32>
    %adg2b8pbt = stablehlo.multiply %b8dpndb, %b8dpndb : tensor<80xf32>
    %advgb8pbt = stablehlo.multiply %adob2b8pbt, %adg2b8pbt : tensor<80xf32>
    %advnb8pbt = stablehlo.add %advsb8pbt, %advgb8pbt : tensor<80xf32>
    %adbc1b8pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adbc2b8pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %admhb8pbt = stablehlo.divide %admnb8pbt, %adbc1b8pbt : tensor<80xf32>
    %advhb8pbt = stablehlo.divide %advnb8pbt, %adbc2b8pbt : tensor<80xf32>
    %adlrb8pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adepsb8pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adsqb8pbt = stablehlo.sqrt %advhb8pbt : tensor<80xf32>
    %addenb8pbt = stablehlo.add %adsqb8pbt, %adepsb8pbt : tensor<80xf32>
    %adratb8pbt = stablehlo.divide %admhb8pbt, %addenb8pbt : tensor<80xf32>
    %adstb8pbt = stablehlo.multiply %adlrb8pbt, %adratb8pbt : tensor<80xf32>
    %adsubb8pbt = stablehlo.subtract %b8pbt, %adstb8pbt : tensor<80xf32>
    %adwdb8pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<80xf32>
    %adwdlrb8pbt = stablehlo.multiply %adwdb8pbt, %adlrb8pbt : tensor<80xf32>
    %adwdpb8pbt = stablehlo.multiply %adwdlrb8pbt, %b8pbt : tensor<80xf32>
    %adnewb8pbt = stablehlo.subtract %adsubb8pbt, %adwdpb8pbt : tensor<80xf32>
    %adb1b9eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adob1b9eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %admsb9eW = stablehlo.multiply %adb1b9eW, %b9eWm : tensor<480x80x1x1xf32>
    %admgb9eW = stablehlo.multiply %adob1b9eW, %b9deW : tensor<480x80x1x1xf32>
    %admnb9eW = stablehlo.add %admsb9eW, %admgb9eW : tensor<480x80x1x1xf32>
    %adb2b9eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adob2b9eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %advsb9eW = stablehlo.multiply %adb2b9eW, %b9eWv : tensor<480x80x1x1xf32>
    %adg2b9eW = stablehlo.multiply %b9deW, %b9deW : tensor<480x80x1x1xf32>
    %advgb9eW = stablehlo.multiply %adob2b9eW, %adg2b9eW : tensor<480x80x1x1xf32>
    %advnb9eW = stablehlo.add %advsb9eW, %advgb9eW : tensor<480x80x1x1xf32>
    %adbc1b9eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adbc2b9eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %admhb9eW = stablehlo.divide %admnb9eW, %adbc1b9eW : tensor<480x80x1x1xf32>
    %advhb9eW = stablehlo.divide %advnb9eW, %adbc2b9eW : tensor<480x80x1x1xf32>
    %adlrb9eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adepsb9eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adsqb9eW = stablehlo.sqrt %advhb9eW : tensor<480x80x1x1xf32>
    %addenb9eW = stablehlo.add %adsqb9eW, %adepsb9eW : tensor<480x80x1x1xf32>
    %adratb9eW = stablehlo.divide %admhb9eW, %addenb9eW : tensor<480x80x1x1xf32>
    %adstb9eW = stablehlo.multiply %adlrb9eW, %adratb9eW : tensor<480x80x1x1xf32>
    %adsubb9eW = stablehlo.subtract %b9eW, %adstb9eW : tensor<480x80x1x1xf32>
    %adwdb9eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x80x1x1xf32>
    %adwdlrb9eW = stablehlo.multiply %adwdb9eW, %adlrb9eW : tensor<480x80x1x1xf32>
    %adwdpb9eW = stablehlo.multiply %adwdlrb9eW, %b9eW : tensor<480x80x1x1xf32>
    %adnewb9eW = stablehlo.subtract %adsubb9eW, %adwdpb9eW : tensor<480x80x1x1xf32>
    %adb1b9eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b9eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb9eb = stablehlo.multiply %adb1b9eb, %b9ebm : tensor<480xf32>
    %admgb9eb = stablehlo.multiply %adob1b9eb, %b9deb : tensor<480xf32>
    %admnb9eb = stablehlo.add %admsb9eb, %admgb9eb : tensor<480xf32>
    %adb2b9eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b9eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb9eb = stablehlo.multiply %adb2b9eb, %b9ebv : tensor<480xf32>
    %adg2b9eb = stablehlo.multiply %b9deb, %b9deb : tensor<480xf32>
    %advgb9eb = stablehlo.multiply %adob2b9eb, %adg2b9eb : tensor<480xf32>
    %advnb9eb = stablehlo.add %advsb9eb, %advgb9eb : tensor<480xf32>
    %adbc1b9eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b9eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb9eb = stablehlo.divide %admnb9eb, %adbc1b9eb : tensor<480xf32>
    %advhb9eb = stablehlo.divide %advnb9eb, %adbc2b9eb : tensor<480xf32>
    %adlrb9eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb9eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb9eb = stablehlo.sqrt %advhb9eb : tensor<480xf32>
    %addenb9eb = stablehlo.add %adsqb9eb, %adepsb9eb : tensor<480xf32>
    %adratb9eb = stablehlo.divide %admhb9eb, %addenb9eb : tensor<480xf32>
    %adstb9eb = stablehlo.multiply %adlrb9eb, %adratb9eb : tensor<480xf32>
    %adsubb9eb = stablehlo.subtract %b9eb, %adstb9eb : tensor<480xf32>
    %adwdb9eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb9eb = stablehlo.multiply %adwdb9eb, %adlrb9eb : tensor<480xf32>
    %adwdpb9eb = stablehlo.multiply %adwdlrb9eb, %b9eb : tensor<480xf32>
    %adnewb9eb = stablehlo.subtract %adsubb9eb, %adwdpb9eb : tensor<480xf32>
    %adb1b9eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b9eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb9eg = stablehlo.multiply %adb1b9eg, %b9egm : tensor<480xf32>
    %admgb9eg = stablehlo.multiply %adob1b9eg, %b9dendg : tensor<480xf32>
    %admnb9eg = stablehlo.add %admsb9eg, %admgb9eg : tensor<480xf32>
    %adb2b9eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b9eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb9eg = stablehlo.multiply %adb2b9eg, %b9egv : tensor<480xf32>
    %adg2b9eg = stablehlo.multiply %b9dendg, %b9dendg : tensor<480xf32>
    %advgb9eg = stablehlo.multiply %adob2b9eg, %adg2b9eg : tensor<480xf32>
    %advnb9eg = stablehlo.add %advsb9eg, %advgb9eg : tensor<480xf32>
    %adbc1b9eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b9eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb9eg = stablehlo.divide %admnb9eg, %adbc1b9eg : tensor<480xf32>
    %advhb9eg = stablehlo.divide %advnb9eg, %adbc2b9eg : tensor<480xf32>
    %adlrb9eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb9eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb9eg = stablehlo.sqrt %advhb9eg : tensor<480xf32>
    %addenb9eg = stablehlo.add %adsqb9eg, %adepsb9eg : tensor<480xf32>
    %adratb9eg = stablehlo.divide %admhb9eg, %addenb9eg : tensor<480xf32>
    %adstb9eg = stablehlo.multiply %adlrb9eg, %adratb9eg : tensor<480xf32>
    %adsubb9eg = stablehlo.subtract %b9eg, %adstb9eg : tensor<480xf32>
    %adwdb9eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb9eg = stablehlo.multiply %adwdb9eg, %adlrb9eg : tensor<480xf32>
    %adwdpb9eg = stablehlo.multiply %adwdlrb9eg, %b9eg : tensor<480xf32>
    %adnewb9eg = stablehlo.subtract %adsubb9eg, %adwdpb9eg : tensor<480xf32>
    %adb1b9ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b9ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb9ebt = stablehlo.multiply %adb1b9ebt, %b9ebtm : tensor<480xf32>
    %admgb9ebt = stablehlo.multiply %adob1b9ebt, %b9dendb : tensor<480xf32>
    %admnb9ebt = stablehlo.add %admsb9ebt, %admgb9ebt : tensor<480xf32>
    %adb2b9ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b9ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb9ebt = stablehlo.multiply %adb2b9ebt, %b9ebtv : tensor<480xf32>
    %adg2b9ebt = stablehlo.multiply %b9dendb, %b9dendb : tensor<480xf32>
    %advgb9ebt = stablehlo.multiply %adob2b9ebt, %adg2b9ebt : tensor<480xf32>
    %advnb9ebt = stablehlo.add %advsb9ebt, %advgb9ebt : tensor<480xf32>
    %adbc1b9ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b9ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb9ebt = stablehlo.divide %admnb9ebt, %adbc1b9ebt : tensor<480xf32>
    %advhb9ebt = stablehlo.divide %advnb9ebt, %adbc2b9ebt : tensor<480xf32>
    %adlrb9ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb9ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb9ebt = stablehlo.sqrt %advhb9ebt : tensor<480xf32>
    %addenb9ebt = stablehlo.add %adsqb9ebt, %adepsb9ebt : tensor<480xf32>
    %adratb9ebt = stablehlo.divide %admhb9ebt, %addenb9ebt : tensor<480xf32>
    %adstb9ebt = stablehlo.multiply %adlrb9ebt, %adratb9ebt : tensor<480xf32>
    %adsubb9ebt = stablehlo.subtract %b9ebt, %adstb9ebt : tensor<480xf32>
    %adwdb9ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb9ebt = stablehlo.multiply %adwdb9ebt, %adlrb9ebt : tensor<480xf32>
    %adwdpb9ebt = stablehlo.multiply %adwdlrb9ebt, %b9ebt : tensor<480xf32>
    %adnewb9ebt = stablehlo.subtract %adsubb9ebt, %adwdpb9ebt : tensor<480xf32>
    %adb1b9dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %adob1b9dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %admsb9dW = stablehlo.multiply %adb1b9dW, %b9dWm : tensor<480x1x5x5xf32>
    %admgb9dW = stablehlo.multiply %adob1b9dW, %b9ddW : tensor<480x1x5x5xf32>
    %admnb9dW = stablehlo.add %admsb9dW, %admgb9dW : tensor<480x1x5x5xf32>
    %adb2b9dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %adob2b9dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %advsb9dW = stablehlo.multiply %adb2b9dW, %b9dWv : tensor<480x1x5x5xf32>
    %adg2b9dW = stablehlo.multiply %b9ddW, %b9ddW : tensor<480x1x5x5xf32>
    %advgb9dW = stablehlo.multiply %adob2b9dW, %adg2b9dW : tensor<480x1x5x5xf32>
    %advnb9dW = stablehlo.add %advsb9dW, %advgb9dW : tensor<480x1x5x5xf32>
    %adbc1b9dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %adbc2b9dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %admhb9dW = stablehlo.divide %admnb9dW, %adbc1b9dW : tensor<480x1x5x5xf32>
    %advhb9dW = stablehlo.divide %advnb9dW, %adbc2b9dW : tensor<480x1x5x5xf32>
    %adlrb9dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %adepsb9dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %adsqb9dW = stablehlo.sqrt %advhb9dW : tensor<480x1x5x5xf32>
    %addenb9dW = stablehlo.add %adsqb9dW, %adepsb9dW : tensor<480x1x5x5xf32>
    %adratb9dW = stablehlo.divide %admhb9dW, %addenb9dW : tensor<480x1x5x5xf32>
    %adstb9dW = stablehlo.multiply %adlrb9dW, %adratb9dW : tensor<480x1x5x5xf32>
    %adsubb9dW = stablehlo.subtract %b9dW, %adstb9dW : tensor<480x1x5x5xf32>
    %adwdb9dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x1x5x5xf32>
    %adwdlrb9dW = stablehlo.multiply %adwdb9dW, %adlrb9dW : tensor<480x1x5x5xf32>
    %adwdpb9dW = stablehlo.multiply %adwdlrb9dW, %b9dW : tensor<480x1x5x5xf32>
    %adnewb9dW = stablehlo.subtract %adsubb9dW, %adwdpb9dW : tensor<480x1x5x5xf32>
    %adb1b9db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b9db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb9db = stablehlo.multiply %adb1b9db, %b9dbm : tensor<480xf32>
    %admgb9db = stablehlo.multiply %adob1b9db, %b9ddb : tensor<480xf32>
    %admnb9db = stablehlo.add %admsb9db, %admgb9db : tensor<480xf32>
    %adb2b9db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b9db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb9db = stablehlo.multiply %adb2b9db, %b9dbv : tensor<480xf32>
    %adg2b9db = stablehlo.multiply %b9ddb, %b9ddb : tensor<480xf32>
    %advgb9db = stablehlo.multiply %adob2b9db, %adg2b9db : tensor<480xf32>
    %advnb9db = stablehlo.add %advsb9db, %advgb9db : tensor<480xf32>
    %adbc1b9db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b9db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb9db = stablehlo.divide %admnb9db, %adbc1b9db : tensor<480xf32>
    %advhb9db = stablehlo.divide %advnb9db, %adbc2b9db : tensor<480xf32>
    %adlrb9db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb9db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb9db = stablehlo.sqrt %advhb9db : tensor<480xf32>
    %addenb9db = stablehlo.add %adsqb9db, %adepsb9db : tensor<480xf32>
    %adratb9db = stablehlo.divide %admhb9db, %addenb9db : tensor<480xf32>
    %adstb9db = stablehlo.multiply %adlrb9db, %adratb9db : tensor<480xf32>
    %adsubb9db = stablehlo.subtract %b9db, %adstb9db : tensor<480xf32>
    %adwdb9db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb9db = stablehlo.multiply %adwdb9db, %adlrb9db : tensor<480xf32>
    %adwdpb9db = stablehlo.multiply %adwdlrb9db, %b9db : tensor<480xf32>
    %adnewb9db = stablehlo.subtract %adsubb9db, %adwdpb9db : tensor<480xf32>
    %adb1b9dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b9dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb9dg = stablehlo.multiply %adb1b9dg, %b9dgm : tensor<480xf32>
    %admgb9dg = stablehlo.multiply %adob1b9dg, %b9ddndg : tensor<480xf32>
    %admnb9dg = stablehlo.add %admsb9dg, %admgb9dg : tensor<480xf32>
    %adb2b9dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b9dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb9dg = stablehlo.multiply %adb2b9dg, %b9dgv : tensor<480xf32>
    %adg2b9dg = stablehlo.multiply %b9ddndg, %b9ddndg : tensor<480xf32>
    %advgb9dg = stablehlo.multiply %adob2b9dg, %adg2b9dg : tensor<480xf32>
    %advnb9dg = stablehlo.add %advsb9dg, %advgb9dg : tensor<480xf32>
    %adbc1b9dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b9dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb9dg = stablehlo.divide %admnb9dg, %adbc1b9dg : tensor<480xf32>
    %advhb9dg = stablehlo.divide %advnb9dg, %adbc2b9dg : tensor<480xf32>
    %adlrb9dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb9dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb9dg = stablehlo.sqrt %advhb9dg : tensor<480xf32>
    %addenb9dg = stablehlo.add %adsqb9dg, %adepsb9dg : tensor<480xf32>
    %adratb9dg = stablehlo.divide %admhb9dg, %addenb9dg : tensor<480xf32>
    %adstb9dg = stablehlo.multiply %adlrb9dg, %adratb9dg : tensor<480xf32>
    %adsubb9dg = stablehlo.subtract %b9dg, %adstb9dg : tensor<480xf32>
    %adwdb9dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb9dg = stablehlo.multiply %adwdb9dg, %adlrb9dg : tensor<480xf32>
    %adwdpb9dg = stablehlo.multiply %adwdlrb9dg, %b9dg : tensor<480xf32>
    %adnewb9dg = stablehlo.subtract %adsubb9dg, %adwdpb9dg : tensor<480xf32>
    %adb1b9dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b9dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb9dbt = stablehlo.multiply %adb1b9dbt, %b9dbtm : tensor<480xf32>
    %admgb9dbt = stablehlo.multiply %adob1b9dbt, %b9ddndb : tensor<480xf32>
    %admnb9dbt = stablehlo.add %admsb9dbt, %admgb9dbt : tensor<480xf32>
    %adb2b9dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b9dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb9dbt = stablehlo.multiply %adb2b9dbt, %b9dbtv : tensor<480xf32>
    %adg2b9dbt = stablehlo.multiply %b9ddndb, %b9ddndb : tensor<480xf32>
    %advgb9dbt = stablehlo.multiply %adob2b9dbt, %adg2b9dbt : tensor<480xf32>
    %advnb9dbt = stablehlo.add %advsb9dbt, %advgb9dbt : tensor<480xf32>
    %adbc1b9dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b9dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb9dbt = stablehlo.divide %admnb9dbt, %adbc1b9dbt : tensor<480xf32>
    %advhb9dbt = stablehlo.divide %advnb9dbt, %adbc2b9dbt : tensor<480xf32>
    %adlrb9dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb9dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb9dbt = stablehlo.sqrt %advhb9dbt : tensor<480xf32>
    %addenb9dbt = stablehlo.add %adsqb9dbt, %adepsb9dbt : tensor<480xf32>
    %adratb9dbt = stablehlo.divide %admhb9dbt, %addenb9dbt : tensor<480xf32>
    %adstb9dbt = stablehlo.multiply %adlrb9dbt, %adratb9dbt : tensor<480xf32>
    %adsubb9dbt = stablehlo.subtract %b9dbt, %adstb9dbt : tensor<480xf32>
    %adwdb9dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb9dbt = stablehlo.multiply %adwdb9dbt, %adlrb9dbt : tensor<480xf32>
    %adwdpb9dbt = stablehlo.multiply %adwdlrb9dbt, %b9dbt : tensor<480xf32>
    %adnewb9dbt = stablehlo.subtract %adsubb9dbt, %adwdpb9dbt : tensor<480xf32>
    %adb1b9zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adob1b9zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %admsb9zW1 = stablehlo.multiply %adb1b9zW1, %b9zW1m : tensor<480x20xf32>
    %admgb9zW1 = stablehlo.multiply %adob1b9zW1, %b9zdWs1 : tensor<480x20xf32>
    %admnb9zW1 = stablehlo.add %admsb9zW1, %admgb9zW1 : tensor<480x20xf32>
    %adb2b9zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adob2b9zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %advsb9zW1 = stablehlo.multiply %adb2b9zW1, %b9zW1v : tensor<480x20xf32>
    %adg2b9zW1 = stablehlo.multiply %b9zdWs1, %b9zdWs1 : tensor<480x20xf32>
    %advgb9zW1 = stablehlo.multiply %adob2b9zW1, %adg2b9zW1 : tensor<480x20xf32>
    %advnb9zW1 = stablehlo.add %advsb9zW1, %advgb9zW1 : tensor<480x20xf32>
    %adbc1b9zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adbc2b9zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %admhb9zW1 = stablehlo.divide %admnb9zW1, %adbc1b9zW1 : tensor<480x20xf32>
    %advhb9zW1 = stablehlo.divide %advnb9zW1, %adbc2b9zW1 : tensor<480x20xf32>
    %adlrb9zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adepsb9zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adsqb9zW1 = stablehlo.sqrt %advhb9zW1 : tensor<480x20xf32>
    %addenb9zW1 = stablehlo.add %adsqb9zW1, %adepsb9zW1 : tensor<480x20xf32>
    %adratb9zW1 = stablehlo.divide %admhb9zW1, %addenb9zW1 : tensor<480x20xf32>
    %adstb9zW1 = stablehlo.multiply %adlrb9zW1, %adratb9zW1 : tensor<480x20xf32>
    %adsubb9zW1 = stablehlo.subtract %b9zW1, %adstb9zW1 : tensor<480x20xf32>
    %adwdb9zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480x20xf32>
    %adwdlrb9zW1 = stablehlo.multiply %adwdb9zW1, %adlrb9zW1 : tensor<480x20xf32>
    %adwdpb9zW1 = stablehlo.multiply %adwdlrb9zW1, %b9zW1 : tensor<480x20xf32>
    %adnewb9zW1 = stablehlo.subtract %adsubb9zW1, %adwdpb9zW1 : tensor<480x20xf32>
    %adb1b9zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adob1b9zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %admsb9zb1 = stablehlo.multiply %adb1b9zb1, %b9zb1m : tensor<20xf32>
    %admgb9zb1 = stablehlo.multiply %adob1b9zb1, %b9zdbs1 : tensor<20xf32>
    %admnb9zb1 = stablehlo.add %admsb9zb1, %admgb9zb1 : tensor<20xf32>
    %adb2b9zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adob2b9zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %advsb9zb1 = stablehlo.multiply %adb2b9zb1, %b9zb1v : tensor<20xf32>
    %adg2b9zb1 = stablehlo.multiply %b9zdbs1, %b9zdbs1 : tensor<20xf32>
    %advgb9zb1 = stablehlo.multiply %adob2b9zb1, %adg2b9zb1 : tensor<20xf32>
    %advnb9zb1 = stablehlo.add %advsb9zb1, %advgb9zb1 : tensor<20xf32>
    %adbc1b9zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adbc2b9zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %admhb9zb1 = stablehlo.divide %admnb9zb1, %adbc1b9zb1 : tensor<20xf32>
    %advhb9zb1 = stablehlo.divide %advnb9zb1, %adbc2b9zb1 : tensor<20xf32>
    %adlrb9zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adepsb9zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adsqb9zb1 = stablehlo.sqrt %advhb9zb1 : tensor<20xf32>
    %addenb9zb1 = stablehlo.add %adsqb9zb1, %adepsb9zb1 : tensor<20xf32>
    %adratb9zb1 = stablehlo.divide %admhb9zb1, %addenb9zb1 : tensor<20xf32>
    %adstb9zb1 = stablehlo.multiply %adlrb9zb1, %adratb9zb1 : tensor<20xf32>
    %adsubb9zb1 = stablehlo.subtract %b9zb1, %adstb9zb1 : tensor<20xf32>
    %adwdb9zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<20xf32>
    %adwdlrb9zb1 = stablehlo.multiply %adwdb9zb1, %adlrb9zb1 : tensor<20xf32>
    %adwdpb9zb1 = stablehlo.multiply %adwdlrb9zb1, %b9zb1 : tensor<20xf32>
    %adnewb9zb1 = stablehlo.subtract %adsubb9zb1, %adwdpb9zb1 : tensor<20xf32>
    %adb1b9zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adob1b9zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %admsb9zW2 = stablehlo.multiply %adb1b9zW2, %b9zW2m : tensor<20x480xf32>
    %admgb9zW2 = stablehlo.multiply %adob1b9zW2, %b9zdWs2 : tensor<20x480xf32>
    %admnb9zW2 = stablehlo.add %admsb9zW2, %admgb9zW2 : tensor<20x480xf32>
    %adb2b9zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adob2b9zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %advsb9zW2 = stablehlo.multiply %adb2b9zW2, %b9zW2v : tensor<20x480xf32>
    %adg2b9zW2 = stablehlo.multiply %b9zdWs2, %b9zdWs2 : tensor<20x480xf32>
    %advgb9zW2 = stablehlo.multiply %adob2b9zW2, %adg2b9zW2 : tensor<20x480xf32>
    %advnb9zW2 = stablehlo.add %advsb9zW2, %advgb9zW2 : tensor<20x480xf32>
    %adbc1b9zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adbc2b9zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %admhb9zW2 = stablehlo.divide %admnb9zW2, %adbc1b9zW2 : tensor<20x480xf32>
    %advhb9zW2 = stablehlo.divide %advnb9zW2, %adbc2b9zW2 : tensor<20x480xf32>
    %adlrb9zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adepsb9zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adsqb9zW2 = stablehlo.sqrt %advhb9zW2 : tensor<20x480xf32>
    %addenb9zW2 = stablehlo.add %adsqb9zW2, %adepsb9zW2 : tensor<20x480xf32>
    %adratb9zW2 = stablehlo.divide %admhb9zW2, %addenb9zW2 : tensor<20x480xf32>
    %adstb9zW2 = stablehlo.multiply %adlrb9zW2, %adratb9zW2 : tensor<20x480xf32>
    %adsubb9zW2 = stablehlo.subtract %b9zW2, %adstb9zW2 : tensor<20x480xf32>
    %adwdb9zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<20x480xf32>
    %adwdlrb9zW2 = stablehlo.multiply %adwdb9zW2, %adlrb9zW2 : tensor<20x480xf32>
    %adwdpb9zW2 = stablehlo.multiply %adwdlrb9zW2, %b9zW2 : tensor<20x480xf32>
    %adnewb9zW2 = stablehlo.subtract %adsubb9zW2, %adwdpb9zW2 : tensor<20x480xf32>
    %adb1b9zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob1b9zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admsb9zb2 = stablehlo.multiply %adb1b9zb2, %b9zb2m : tensor<480xf32>
    %admgb9zb2 = stablehlo.multiply %adob1b9zb2, %b9zdbs2 : tensor<480xf32>
    %admnb9zb2 = stablehlo.add %admsb9zb2, %admgb9zb2 : tensor<480xf32>
    %adb2b9zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adob2b9zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %advsb9zb2 = stablehlo.multiply %adb2b9zb2, %b9zb2v : tensor<480xf32>
    %adg2b9zb2 = stablehlo.multiply %b9zdbs2, %b9zdbs2 : tensor<480xf32>
    %advgb9zb2 = stablehlo.multiply %adob2b9zb2, %adg2b9zb2 : tensor<480xf32>
    %advnb9zb2 = stablehlo.add %advsb9zb2, %advgb9zb2 : tensor<480xf32>
    %adbc1b9zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adbc2b9zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %admhb9zb2 = stablehlo.divide %admnb9zb2, %adbc1b9zb2 : tensor<480xf32>
    %advhb9zb2 = stablehlo.divide %advnb9zb2, %adbc2b9zb2 : tensor<480xf32>
    %adlrb9zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adepsb9zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adsqb9zb2 = stablehlo.sqrt %advhb9zb2 : tensor<480xf32>
    %addenb9zb2 = stablehlo.add %adsqb9zb2, %adepsb9zb2 : tensor<480xf32>
    %adratb9zb2 = stablehlo.divide %admhb9zb2, %addenb9zb2 : tensor<480xf32>
    %adstb9zb2 = stablehlo.multiply %adlrb9zb2, %adratb9zb2 : tensor<480xf32>
    %adsubb9zb2 = stablehlo.subtract %b9zb2, %adstb9zb2 : tensor<480xf32>
    %adwdb9zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<480xf32>
    %adwdlrb9zb2 = stablehlo.multiply %adwdb9zb2, %adlrb9zb2 : tensor<480xf32>
    %adwdpb9zb2 = stablehlo.multiply %adwdlrb9zb2, %b9zb2 : tensor<480xf32>
    %adnewb9zb2 = stablehlo.subtract %adsubb9zb2, %adwdpb9zb2 : tensor<480xf32>
    %adb1b9pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %adob1b9pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %admsb9pW = stablehlo.multiply %adb1b9pW, %b9pWm : tensor<112x480x1x1xf32>
    %admgb9pW = stablehlo.multiply %adob1b9pW, %b9dpW : tensor<112x480x1x1xf32>
    %admnb9pW = stablehlo.add %admsb9pW, %admgb9pW : tensor<112x480x1x1xf32>
    %adb2b9pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %adob2b9pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %advsb9pW = stablehlo.multiply %adb2b9pW, %b9pWv : tensor<112x480x1x1xf32>
    %adg2b9pW = stablehlo.multiply %b9dpW, %b9dpW : tensor<112x480x1x1xf32>
    %advgb9pW = stablehlo.multiply %adob2b9pW, %adg2b9pW : tensor<112x480x1x1xf32>
    %advnb9pW = stablehlo.add %advsb9pW, %advgb9pW : tensor<112x480x1x1xf32>
    %adbc1b9pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %adbc2b9pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %admhb9pW = stablehlo.divide %admnb9pW, %adbc1b9pW : tensor<112x480x1x1xf32>
    %advhb9pW = stablehlo.divide %advnb9pW, %adbc2b9pW : tensor<112x480x1x1xf32>
    %adlrb9pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %adepsb9pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %adsqb9pW = stablehlo.sqrt %advhb9pW : tensor<112x480x1x1xf32>
    %addenb9pW = stablehlo.add %adsqb9pW, %adepsb9pW : tensor<112x480x1x1xf32>
    %adratb9pW = stablehlo.divide %admhb9pW, %addenb9pW : tensor<112x480x1x1xf32>
    %adstb9pW = stablehlo.multiply %adlrb9pW, %adratb9pW : tensor<112x480x1x1xf32>
    %adsubb9pW = stablehlo.subtract %b9pW, %adstb9pW : tensor<112x480x1x1xf32>
    %adwdb9pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112x480x1x1xf32>
    %adwdlrb9pW = stablehlo.multiply %adwdb9pW, %adlrb9pW : tensor<112x480x1x1xf32>
    %adwdpb9pW = stablehlo.multiply %adwdlrb9pW, %b9pW : tensor<112x480x1x1xf32>
    %adnewb9pW = stablehlo.subtract %adsubb9pW, %adwdpb9pW : tensor<112x480x1x1xf32>
    %adb1b9pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b9pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb9pb = stablehlo.multiply %adb1b9pb, %b9pbm : tensor<112xf32>
    %admgb9pb = stablehlo.multiply %adob1b9pb, %b9dpb : tensor<112xf32>
    %admnb9pb = stablehlo.add %admsb9pb, %admgb9pb : tensor<112xf32>
    %adb2b9pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b9pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb9pb = stablehlo.multiply %adb2b9pb, %b9pbv : tensor<112xf32>
    %adg2b9pb = stablehlo.multiply %b9dpb, %b9dpb : tensor<112xf32>
    %advgb9pb = stablehlo.multiply %adob2b9pb, %adg2b9pb : tensor<112xf32>
    %advnb9pb = stablehlo.add %advsb9pb, %advgb9pb : tensor<112xf32>
    %adbc1b9pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b9pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb9pb = stablehlo.divide %admnb9pb, %adbc1b9pb : tensor<112xf32>
    %advhb9pb = stablehlo.divide %advnb9pb, %adbc2b9pb : tensor<112xf32>
    %adlrb9pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb9pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb9pb = stablehlo.sqrt %advhb9pb : tensor<112xf32>
    %addenb9pb = stablehlo.add %adsqb9pb, %adepsb9pb : tensor<112xf32>
    %adratb9pb = stablehlo.divide %admhb9pb, %addenb9pb : tensor<112xf32>
    %adstb9pb = stablehlo.multiply %adlrb9pb, %adratb9pb : tensor<112xf32>
    %adsubb9pb = stablehlo.subtract %b9pb, %adstb9pb : tensor<112xf32>
    %adwdb9pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb9pb = stablehlo.multiply %adwdb9pb, %adlrb9pb : tensor<112xf32>
    %adwdpb9pb = stablehlo.multiply %adwdlrb9pb, %b9pb : tensor<112xf32>
    %adnewb9pb = stablehlo.subtract %adsubb9pb, %adwdpb9pb : tensor<112xf32>
    %adb1b9pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b9pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb9pg = stablehlo.multiply %adb1b9pg, %b9pgm : tensor<112xf32>
    %admgb9pg = stablehlo.multiply %adob1b9pg, %b9dpndg : tensor<112xf32>
    %admnb9pg = stablehlo.add %admsb9pg, %admgb9pg : tensor<112xf32>
    %adb2b9pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b9pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb9pg = stablehlo.multiply %adb2b9pg, %b9pgv : tensor<112xf32>
    %adg2b9pg = stablehlo.multiply %b9dpndg, %b9dpndg : tensor<112xf32>
    %advgb9pg = stablehlo.multiply %adob2b9pg, %adg2b9pg : tensor<112xf32>
    %advnb9pg = stablehlo.add %advsb9pg, %advgb9pg : tensor<112xf32>
    %adbc1b9pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b9pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb9pg = stablehlo.divide %admnb9pg, %adbc1b9pg : tensor<112xf32>
    %advhb9pg = stablehlo.divide %advnb9pg, %adbc2b9pg : tensor<112xf32>
    %adlrb9pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb9pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb9pg = stablehlo.sqrt %advhb9pg : tensor<112xf32>
    %addenb9pg = stablehlo.add %adsqb9pg, %adepsb9pg : tensor<112xf32>
    %adratb9pg = stablehlo.divide %admhb9pg, %addenb9pg : tensor<112xf32>
    %adstb9pg = stablehlo.multiply %adlrb9pg, %adratb9pg : tensor<112xf32>
    %adsubb9pg = stablehlo.subtract %b9pg, %adstb9pg : tensor<112xf32>
    %adwdb9pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb9pg = stablehlo.multiply %adwdb9pg, %adlrb9pg : tensor<112xf32>
    %adwdpb9pg = stablehlo.multiply %adwdlrb9pg, %b9pg : tensor<112xf32>
    %adnewb9pg = stablehlo.subtract %adsubb9pg, %adwdpb9pg : tensor<112xf32>
    %adb1b9pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b9pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb9pbt = stablehlo.multiply %adb1b9pbt, %b9pbtm : tensor<112xf32>
    %admgb9pbt = stablehlo.multiply %adob1b9pbt, %b9dpndb : tensor<112xf32>
    %admnb9pbt = stablehlo.add %admsb9pbt, %admgb9pbt : tensor<112xf32>
    %adb2b9pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b9pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb9pbt = stablehlo.multiply %adb2b9pbt, %b9pbtv : tensor<112xf32>
    %adg2b9pbt = stablehlo.multiply %b9dpndb, %b9dpndb : tensor<112xf32>
    %advgb9pbt = stablehlo.multiply %adob2b9pbt, %adg2b9pbt : tensor<112xf32>
    %advnb9pbt = stablehlo.add %advsb9pbt, %advgb9pbt : tensor<112xf32>
    %adbc1b9pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b9pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb9pbt = stablehlo.divide %admnb9pbt, %adbc1b9pbt : tensor<112xf32>
    %advhb9pbt = stablehlo.divide %advnb9pbt, %adbc2b9pbt : tensor<112xf32>
    %adlrb9pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb9pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb9pbt = stablehlo.sqrt %advhb9pbt : tensor<112xf32>
    %addenb9pbt = stablehlo.add %adsqb9pbt, %adepsb9pbt : tensor<112xf32>
    %adratb9pbt = stablehlo.divide %admhb9pbt, %addenb9pbt : tensor<112xf32>
    %adstb9pbt = stablehlo.multiply %adlrb9pbt, %adratb9pbt : tensor<112xf32>
    %adsubb9pbt = stablehlo.subtract %b9pbt, %adstb9pbt : tensor<112xf32>
    %adwdb9pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb9pbt = stablehlo.multiply %adwdb9pbt, %adlrb9pbt : tensor<112xf32>
    %adwdpb9pbt = stablehlo.multiply %adwdlrb9pbt, %b9pbt : tensor<112xf32>
    %adnewb9pbt = stablehlo.subtract %adsubb9pbt, %adwdpb9pbt : tensor<112xf32>
    %adb1b10eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adob1b10eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %admsb10eW = stablehlo.multiply %adb1b10eW, %b10eWm : tensor<672x112x1x1xf32>
    %admgb10eW = stablehlo.multiply %adob1b10eW, %b10deW : tensor<672x112x1x1xf32>
    %admnb10eW = stablehlo.add %admsb10eW, %admgb10eW : tensor<672x112x1x1xf32>
    %adb2b10eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adob2b10eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %advsb10eW = stablehlo.multiply %adb2b10eW, %b10eWv : tensor<672x112x1x1xf32>
    %adg2b10eW = stablehlo.multiply %b10deW, %b10deW : tensor<672x112x1x1xf32>
    %advgb10eW = stablehlo.multiply %adob2b10eW, %adg2b10eW : tensor<672x112x1x1xf32>
    %advnb10eW = stablehlo.add %advsb10eW, %advgb10eW : tensor<672x112x1x1xf32>
    %adbc1b10eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adbc2b10eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %admhb10eW = stablehlo.divide %admnb10eW, %adbc1b10eW : tensor<672x112x1x1xf32>
    %advhb10eW = stablehlo.divide %advnb10eW, %adbc2b10eW : tensor<672x112x1x1xf32>
    %adlrb10eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adepsb10eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adsqb10eW = stablehlo.sqrt %advhb10eW : tensor<672x112x1x1xf32>
    %addenb10eW = stablehlo.add %adsqb10eW, %adepsb10eW : tensor<672x112x1x1xf32>
    %adratb10eW = stablehlo.divide %admhb10eW, %addenb10eW : tensor<672x112x1x1xf32>
    %adstb10eW = stablehlo.multiply %adlrb10eW, %adratb10eW : tensor<672x112x1x1xf32>
    %adsubb10eW = stablehlo.subtract %b10eW, %adstb10eW : tensor<672x112x1x1xf32>
    %adwdb10eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adwdlrb10eW = stablehlo.multiply %adwdb10eW, %adlrb10eW : tensor<672x112x1x1xf32>
    %adwdpb10eW = stablehlo.multiply %adwdlrb10eW, %b10eW : tensor<672x112x1x1xf32>
    %adnewb10eW = stablehlo.subtract %adsubb10eW, %adwdpb10eW : tensor<672x112x1x1xf32>
    %adb1b10eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b10eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb10eb = stablehlo.multiply %adb1b10eb, %b10ebm : tensor<672xf32>
    %admgb10eb = stablehlo.multiply %adob1b10eb, %b10deb : tensor<672xf32>
    %admnb10eb = stablehlo.add %admsb10eb, %admgb10eb : tensor<672xf32>
    %adb2b10eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b10eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb10eb = stablehlo.multiply %adb2b10eb, %b10ebv : tensor<672xf32>
    %adg2b10eb = stablehlo.multiply %b10deb, %b10deb : tensor<672xf32>
    %advgb10eb = stablehlo.multiply %adob2b10eb, %adg2b10eb : tensor<672xf32>
    %advnb10eb = stablehlo.add %advsb10eb, %advgb10eb : tensor<672xf32>
    %adbc1b10eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b10eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb10eb = stablehlo.divide %admnb10eb, %adbc1b10eb : tensor<672xf32>
    %advhb10eb = stablehlo.divide %advnb10eb, %adbc2b10eb : tensor<672xf32>
    %adlrb10eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb10eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb10eb = stablehlo.sqrt %advhb10eb : tensor<672xf32>
    %addenb10eb = stablehlo.add %adsqb10eb, %adepsb10eb : tensor<672xf32>
    %adratb10eb = stablehlo.divide %admhb10eb, %addenb10eb : tensor<672xf32>
    %adstb10eb = stablehlo.multiply %adlrb10eb, %adratb10eb : tensor<672xf32>
    %adsubb10eb = stablehlo.subtract %b10eb, %adstb10eb : tensor<672xf32>
    %adwdb10eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb10eb = stablehlo.multiply %adwdb10eb, %adlrb10eb : tensor<672xf32>
    %adwdpb10eb = stablehlo.multiply %adwdlrb10eb, %b10eb : tensor<672xf32>
    %adnewb10eb = stablehlo.subtract %adsubb10eb, %adwdpb10eb : tensor<672xf32>
    %adb1b10eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b10eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb10eg = stablehlo.multiply %adb1b10eg, %b10egm : tensor<672xf32>
    %admgb10eg = stablehlo.multiply %adob1b10eg, %b10dendg : tensor<672xf32>
    %admnb10eg = stablehlo.add %admsb10eg, %admgb10eg : tensor<672xf32>
    %adb2b10eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b10eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb10eg = stablehlo.multiply %adb2b10eg, %b10egv : tensor<672xf32>
    %adg2b10eg = stablehlo.multiply %b10dendg, %b10dendg : tensor<672xf32>
    %advgb10eg = stablehlo.multiply %adob2b10eg, %adg2b10eg : tensor<672xf32>
    %advnb10eg = stablehlo.add %advsb10eg, %advgb10eg : tensor<672xf32>
    %adbc1b10eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b10eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb10eg = stablehlo.divide %admnb10eg, %adbc1b10eg : tensor<672xf32>
    %advhb10eg = stablehlo.divide %advnb10eg, %adbc2b10eg : tensor<672xf32>
    %adlrb10eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb10eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb10eg = stablehlo.sqrt %advhb10eg : tensor<672xf32>
    %addenb10eg = stablehlo.add %adsqb10eg, %adepsb10eg : tensor<672xf32>
    %adratb10eg = stablehlo.divide %admhb10eg, %addenb10eg : tensor<672xf32>
    %adstb10eg = stablehlo.multiply %adlrb10eg, %adratb10eg : tensor<672xf32>
    %adsubb10eg = stablehlo.subtract %b10eg, %adstb10eg : tensor<672xf32>
    %adwdb10eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb10eg = stablehlo.multiply %adwdb10eg, %adlrb10eg : tensor<672xf32>
    %adwdpb10eg = stablehlo.multiply %adwdlrb10eg, %b10eg : tensor<672xf32>
    %adnewb10eg = stablehlo.subtract %adsubb10eg, %adwdpb10eg : tensor<672xf32>
    %adb1b10ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b10ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb10ebt = stablehlo.multiply %adb1b10ebt, %b10ebtm : tensor<672xf32>
    %admgb10ebt = stablehlo.multiply %adob1b10ebt, %b10dendb : tensor<672xf32>
    %admnb10ebt = stablehlo.add %admsb10ebt, %admgb10ebt : tensor<672xf32>
    %adb2b10ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b10ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb10ebt = stablehlo.multiply %adb2b10ebt, %b10ebtv : tensor<672xf32>
    %adg2b10ebt = stablehlo.multiply %b10dendb, %b10dendb : tensor<672xf32>
    %advgb10ebt = stablehlo.multiply %adob2b10ebt, %adg2b10ebt : tensor<672xf32>
    %advnb10ebt = stablehlo.add %advsb10ebt, %advgb10ebt : tensor<672xf32>
    %adbc1b10ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b10ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb10ebt = stablehlo.divide %admnb10ebt, %adbc1b10ebt : tensor<672xf32>
    %advhb10ebt = stablehlo.divide %advnb10ebt, %adbc2b10ebt : tensor<672xf32>
    %adlrb10ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb10ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb10ebt = stablehlo.sqrt %advhb10ebt : tensor<672xf32>
    %addenb10ebt = stablehlo.add %adsqb10ebt, %adepsb10ebt : tensor<672xf32>
    %adratb10ebt = stablehlo.divide %admhb10ebt, %addenb10ebt : tensor<672xf32>
    %adstb10ebt = stablehlo.multiply %adlrb10ebt, %adratb10ebt : tensor<672xf32>
    %adsubb10ebt = stablehlo.subtract %b10ebt, %adstb10ebt : tensor<672xf32>
    %adwdb10ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb10ebt = stablehlo.multiply %adwdb10ebt, %adlrb10ebt : tensor<672xf32>
    %adwdpb10ebt = stablehlo.multiply %adwdlrb10ebt, %b10ebt : tensor<672xf32>
    %adnewb10ebt = stablehlo.subtract %adsubb10ebt, %adwdpb10ebt : tensor<672xf32>
    %adb1b10dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adob1b10dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %admsb10dW = stablehlo.multiply %adb1b10dW, %b10dWm : tensor<672x1x5x5xf32>
    %admgb10dW = stablehlo.multiply %adob1b10dW, %b10ddW : tensor<672x1x5x5xf32>
    %admnb10dW = stablehlo.add %admsb10dW, %admgb10dW : tensor<672x1x5x5xf32>
    %adb2b10dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adob2b10dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %advsb10dW = stablehlo.multiply %adb2b10dW, %b10dWv : tensor<672x1x5x5xf32>
    %adg2b10dW = stablehlo.multiply %b10ddW, %b10ddW : tensor<672x1x5x5xf32>
    %advgb10dW = stablehlo.multiply %adob2b10dW, %adg2b10dW : tensor<672x1x5x5xf32>
    %advnb10dW = stablehlo.add %advsb10dW, %advgb10dW : tensor<672x1x5x5xf32>
    %adbc1b10dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adbc2b10dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %admhb10dW = stablehlo.divide %admnb10dW, %adbc1b10dW : tensor<672x1x5x5xf32>
    %advhb10dW = stablehlo.divide %advnb10dW, %adbc2b10dW : tensor<672x1x5x5xf32>
    %adlrb10dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adepsb10dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adsqb10dW = stablehlo.sqrt %advhb10dW : tensor<672x1x5x5xf32>
    %addenb10dW = stablehlo.add %adsqb10dW, %adepsb10dW : tensor<672x1x5x5xf32>
    %adratb10dW = stablehlo.divide %admhb10dW, %addenb10dW : tensor<672x1x5x5xf32>
    %adstb10dW = stablehlo.multiply %adlrb10dW, %adratb10dW : tensor<672x1x5x5xf32>
    %adsubb10dW = stablehlo.subtract %b10dW, %adstb10dW : tensor<672x1x5x5xf32>
    %adwdb10dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adwdlrb10dW = stablehlo.multiply %adwdb10dW, %adlrb10dW : tensor<672x1x5x5xf32>
    %adwdpb10dW = stablehlo.multiply %adwdlrb10dW, %b10dW : tensor<672x1x5x5xf32>
    %adnewb10dW = stablehlo.subtract %adsubb10dW, %adwdpb10dW : tensor<672x1x5x5xf32>
    %adb1b10db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b10db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb10db = stablehlo.multiply %adb1b10db, %b10dbm : tensor<672xf32>
    %admgb10db = stablehlo.multiply %adob1b10db, %b10ddb : tensor<672xf32>
    %admnb10db = stablehlo.add %admsb10db, %admgb10db : tensor<672xf32>
    %adb2b10db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b10db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb10db = stablehlo.multiply %adb2b10db, %b10dbv : tensor<672xf32>
    %adg2b10db = stablehlo.multiply %b10ddb, %b10ddb : tensor<672xf32>
    %advgb10db = stablehlo.multiply %adob2b10db, %adg2b10db : tensor<672xf32>
    %advnb10db = stablehlo.add %advsb10db, %advgb10db : tensor<672xf32>
    %adbc1b10db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b10db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb10db = stablehlo.divide %admnb10db, %adbc1b10db : tensor<672xf32>
    %advhb10db = stablehlo.divide %advnb10db, %adbc2b10db : tensor<672xf32>
    %adlrb10db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb10db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb10db = stablehlo.sqrt %advhb10db : tensor<672xf32>
    %addenb10db = stablehlo.add %adsqb10db, %adepsb10db : tensor<672xf32>
    %adratb10db = stablehlo.divide %admhb10db, %addenb10db : tensor<672xf32>
    %adstb10db = stablehlo.multiply %adlrb10db, %adratb10db : tensor<672xf32>
    %adsubb10db = stablehlo.subtract %b10db, %adstb10db : tensor<672xf32>
    %adwdb10db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb10db = stablehlo.multiply %adwdb10db, %adlrb10db : tensor<672xf32>
    %adwdpb10db = stablehlo.multiply %adwdlrb10db, %b10db : tensor<672xf32>
    %adnewb10db = stablehlo.subtract %adsubb10db, %adwdpb10db : tensor<672xf32>
    %adb1b10dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b10dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb10dg = stablehlo.multiply %adb1b10dg, %b10dgm : tensor<672xf32>
    %admgb10dg = stablehlo.multiply %adob1b10dg, %b10ddndg : tensor<672xf32>
    %admnb10dg = stablehlo.add %admsb10dg, %admgb10dg : tensor<672xf32>
    %adb2b10dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b10dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb10dg = stablehlo.multiply %adb2b10dg, %b10dgv : tensor<672xf32>
    %adg2b10dg = stablehlo.multiply %b10ddndg, %b10ddndg : tensor<672xf32>
    %advgb10dg = stablehlo.multiply %adob2b10dg, %adg2b10dg : tensor<672xf32>
    %advnb10dg = stablehlo.add %advsb10dg, %advgb10dg : tensor<672xf32>
    %adbc1b10dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b10dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb10dg = stablehlo.divide %admnb10dg, %adbc1b10dg : tensor<672xf32>
    %advhb10dg = stablehlo.divide %advnb10dg, %adbc2b10dg : tensor<672xf32>
    %adlrb10dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb10dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb10dg = stablehlo.sqrt %advhb10dg : tensor<672xf32>
    %addenb10dg = stablehlo.add %adsqb10dg, %adepsb10dg : tensor<672xf32>
    %adratb10dg = stablehlo.divide %admhb10dg, %addenb10dg : tensor<672xf32>
    %adstb10dg = stablehlo.multiply %adlrb10dg, %adratb10dg : tensor<672xf32>
    %adsubb10dg = stablehlo.subtract %b10dg, %adstb10dg : tensor<672xf32>
    %adwdb10dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb10dg = stablehlo.multiply %adwdb10dg, %adlrb10dg : tensor<672xf32>
    %adwdpb10dg = stablehlo.multiply %adwdlrb10dg, %b10dg : tensor<672xf32>
    %adnewb10dg = stablehlo.subtract %adsubb10dg, %adwdpb10dg : tensor<672xf32>
    %adb1b10dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b10dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb10dbt = stablehlo.multiply %adb1b10dbt, %b10dbtm : tensor<672xf32>
    %admgb10dbt = stablehlo.multiply %adob1b10dbt, %b10ddndb : tensor<672xf32>
    %admnb10dbt = stablehlo.add %admsb10dbt, %admgb10dbt : tensor<672xf32>
    %adb2b10dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b10dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb10dbt = stablehlo.multiply %adb2b10dbt, %b10dbtv : tensor<672xf32>
    %adg2b10dbt = stablehlo.multiply %b10ddndb, %b10ddndb : tensor<672xf32>
    %advgb10dbt = stablehlo.multiply %adob2b10dbt, %adg2b10dbt : tensor<672xf32>
    %advnb10dbt = stablehlo.add %advsb10dbt, %advgb10dbt : tensor<672xf32>
    %adbc1b10dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b10dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb10dbt = stablehlo.divide %admnb10dbt, %adbc1b10dbt : tensor<672xf32>
    %advhb10dbt = stablehlo.divide %advnb10dbt, %adbc2b10dbt : tensor<672xf32>
    %adlrb10dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb10dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb10dbt = stablehlo.sqrt %advhb10dbt : tensor<672xf32>
    %addenb10dbt = stablehlo.add %adsqb10dbt, %adepsb10dbt : tensor<672xf32>
    %adratb10dbt = stablehlo.divide %admhb10dbt, %addenb10dbt : tensor<672xf32>
    %adstb10dbt = stablehlo.multiply %adlrb10dbt, %adratb10dbt : tensor<672xf32>
    %adsubb10dbt = stablehlo.subtract %b10dbt, %adstb10dbt : tensor<672xf32>
    %adwdb10dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb10dbt = stablehlo.multiply %adwdb10dbt, %adlrb10dbt : tensor<672xf32>
    %adwdpb10dbt = stablehlo.multiply %adwdlrb10dbt, %b10dbt : tensor<672xf32>
    %adnewb10dbt = stablehlo.subtract %adsubb10dbt, %adwdpb10dbt : tensor<672xf32>
    %adb1b10zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adob1b10zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %admsb10zW1 = stablehlo.multiply %adb1b10zW1, %b10zW1m : tensor<672x28xf32>
    %admgb10zW1 = stablehlo.multiply %adob1b10zW1, %b10zdWs1 : tensor<672x28xf32>
    %admnb10zW1 = stablehlo.add %admsb10zW1, %admgb10zW1 : tensor<672x28xf32>
    %adb2b10zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adob2b10zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %advsb10zW1 = stablehlo.multiply %adb2b10zW1, %b10zW1v : tensor<672x28xf32>
    %adg2b10zW1 = stablehlo.multiply %b10zdWs1, %b10zdWs1 : tensor<672x28xf32>
    %advgb10zW1 = stablehlo.multiply %adob2b10zW1, %adg2b10zW1 : tensor<672x28xf32>
    %advnb10zW1 = stablehlo.add %advsb10zW1, %advgb10zW1 : tensor<672x28xf32>
    %adbc1b10zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adbc2b10zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %admhb10zW1 = stablehlo.divide %admnb10zW1, %adbc1b10zW1 : tensor<672x28xf32>
    %advhb10zW1 = stablehlo.divide %advnb10zW1, %adbc2b10zW1 : tensor<672x28xf32>
    %adlrb10zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adepsb10zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adsqb10zW1 = stablehlo.sqrt %advhb10zW1 : tensor<672x28xf32>
    %addenb10zW1 = stablehlo.add %adsqb10zW1, %adepsb10zW1 : tensor<672x28xf32>
    %adratb10zW1 = stablehlo.divide %admhb10zW1, %addenb10zW1 : tensor<672x28xf32>
    %adstb10zW1 = stablehlo.multiply %adlrb10zW1, %adratb10zW1 : tensor<672x28xf32>
    %adsubb10zW1 = stablehlo.subtract %b10zW1, %adstb10zW1 : tensor<672x28xf32>
    %adwdb10zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adwdlrb10zW1 = stablehlo.multiply %adwdb10zW1, %adlrb10zW1 : tensor<672x28xf32>
    %adwdpb10zW1 = stablehlo.multiply %adwdlrb10zW1, %b10zW1 : tensor<672x28xf32>
    %adnewb10zW1 = stablehlo.subtract %adsubb10zW1, %adwdpb10zW1 : tensor<672x28xf32>
    %adb1b10zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adob1b10zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %admsb10zb1 = stablehlo.multiply %adb1b10zb1, %b10zb1m : tensor<28xf32>
    %admgb10zb1 = stablehlo.multiply %adob1b10zb1, %b10zdbs1 : tensor<28xf32>
    %admnb10zb1 = stablehlo.add %admsb10zb1, %admgb10zb1 : tensor<28xf32>
    %adb2b10zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adob2b10zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %advsb10zb1 = stablehlo.multiply %adb2b10zb1, %b10zb1v : tensor<28xf32>
    %adg2b10zb1 = stablehlo.multiply %b10zdbs1, %b10zdbs1 : tensor<28xf32>
    %advgb10zb1 = stablehlo.multiply %adob2b10zb1, %adg2b10zb1 : tensor<28xf32>
    %advnb10zb1 = stablehlo.add %advsb10zb1, %advgb10zb1 : tensor<28xf32>
    %adbc1b10zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adbc2b10zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %admhb10zb1 = stablehlo.divide %admnb10zb1, %adbc1b10zb1 : tensor<28xf32>
    %advhb10zb1 = stablehlo.divide %advnb10zb1, %adbc2b10zb1 : tensor<28xf32>
    %adlrb10zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adepsb10zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adsqb10zb1 = stablehlo.sqrt %advhb10zb1 : tensor<28xf32>
    %addenb10zb1 = stablehlo.add %adsqb10zb1, %adepsb10zb1 : tensor<28xf32>
    %adratb10zb1 = stablehlo.divide %admhb10zb1, %addenb10zb1 : tensor<28xf32>
    %adstb10zb1 = stablehlo.multiply %adlrb10zb1, %adratb10zb1 : tensor<28xf32>
    %adsubb10zb1 = stablehlo.subtract %b10zb1, %adstb10zb1 : tensor<28xf32>
    %adwdb10zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adwdlrb10zb1 = stablehlo.multiply %adwdb10zb1, %adlrb10zb1 : tensor<28xf32>
    %adwdpb10zb1 = stablehlo.multiply %adwdlrb10zb1, %b10zb1 : tensor<28xf32>
    %adnewb10zb1 = stablehlo.subtract %adsubb10zb1, %adwdpb10zb1 : tensor<28xf32>
    %adb1b10zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adob1b10zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %admsb10zW2 = stablehlo.multiply %adb1b10zW2, %b10zW2m : tensor<28x672xf32>
    %admgb10zW2 = stablehlo.multiply %adob1b10zW2, %b10zdWs2 : tensor<28x672xf32>
    %admnb10zW2 = stablehlo.add %admsb10zW2, %admgb10zW2 : tensor<28x672xf32>
    %adb2b10zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adob2b10zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %advsb10zW2 = stablehlo.multiply %adb2b10zW2, %b10zW2v : tensor<28x672xf32>
    %adg2b10zW2 = stablehlo.multiply %b10zdWs2, %b10zdWs2 : tensor<28x672xf32>
    %advgb10zW2 = stablehlo.multiply %adob2b10zW2, %adg2b10zW2 : tensor<28x672xf32>
    %advnb10zW2 = stablehlo.add %advsb10zW2, %advgb10zW2 : tensor<28x672xf32>
    %adbc1b10zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adbc2b10zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %admhb10zW2 = stablehlo.divide %admnb10zW2, %adbc1b10zW2 : tensor<28x672xf32>
    %advhb10zW2 = stablehlo.divide %advnb10zW2, %adbc2b10zW2 : tensor<28x672xf32>
    %adlrb10zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adepsb10zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adsqb10zW2 = stablehlo.sqrt %advhb10zW2 : tensor<28x672xf32>
    %addenb10zW2 = stablehlo.add %adsqb10zW2, %adepsb10zW2 : tensor<28x672xf32>
    %adratb10zW2 = stablehlo.divide %admhb10zW2, %addenb10zW2 : tensor<28x672xf32>
    %adstb10zW2 = stablehlo.multiply %adlrb10zW2, %adratb10zW2 : tensor<28x672xf32>
    %adsubb10zW2 = stablehlo.subtract %b10zW2, %adstb10zW2 : tensor<28x672xf32>
    %adwdb10zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adwdlrb10zW2 = stablehlo.multiply %adwdb10zW2, %adlrb10zW2 : tensor<28x672xf32>
    %adwdpb10zW2 = stablehlo.multiply %adwdlrb10zW2, %b10zW2 : tensor<28x672xf32>
    %adnewb10zW2 = stablehlo.subtract %adsubb10zW2, %adwdpb10zW2 : tensor<28x672xf32>
    %adb1b10zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b10zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb10zb2 = stablehlo.multiply %adb1b10zb2, %b10zb2m : tensor<672xf32>
    %admgb10zb2 = stablehlo.multiply %adob1b10zb2, %b10zdbs2 : tensor<672xf32>
    %admnb10zb2 = stablehlo.add %admsb10zb2, %admgb10zb2 : tensor<672xf32>
    %adb2b10zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b10zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb10zb2 = stablehlo.multiply %adb2b10zb2, %b10zb2v : tensor<672xf32>
    %adg2b10zb2 = stablehlo.multiply %b10zdbs2, %b10zdbs2 : tensor<672xf32>
    %advgb10zb2 = stablehlo.multiply %adob2b10zb2, %adg2b10zb2 : tensor<672xf32>
    %advnb10zb2 = stablehlo.add %advsb10zb2, %advgb10zb2 : tensor<672xf32>
    %adbc1b10zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b10zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb10zb2 = stablehlo.divide %admnb10zb2, %adbc1b10zb2 : tensor<672xf32>
    %advhb10zb2 = stablehlo.divide %advnb10zb2, %adbc2b10zb2 : tensor<672xf32>
    %adlrb10zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb10zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb10zb2 = stablehlo.sqrt %advhb10zb2 : tensor<672xf32>
    %addenb10zb2 = stablehlo.add %adsqb10zb2, %adepsb10zb2 : tensor<672xf32>
    %adratb10zb2 = stablehlo.divide %admhb10zb2, %addenb10zb2 : tensor<672xf32>
    %adstb10zb2 = stablehlo.multiply %adlrb10zb2, %adratb10zb2 : tensor<672xf32>
    %adsubb10zb2 = stablehlo.subtract %b10zb2, %adstb10zb2 : tensor<672xf32>
    %adwdb10zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb10zb2 = stablehlo.multiply %adwdb10zb2, %adlrb10zb2 : tensor<672xf32>
    %adwdpb10zb2 = stablehlo.multiply %adwdlrb10zb2, %b10zb2 : tensor<672xf32>
    %adnewb10zb2 = stablehlo.subtract %adsubb10zb2, %adwdpb10zb2 : tensor<672xf32>
    %adb1b10pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adob1b10pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %admsb10pW = stablehlo.multiply %adb1b10pW, %b10pWm : tensor<112x672x1x1xf32>
    %admgb10pW = stablehlo.multiply %adob1b10pW, %b10dpW : tensor<112x672x1x1xf32>
    %admnb10pW = stablehlo.add %admsb10pW, %admgb10pW : tensor<112x672x1x1xf32>
    %adb2b10pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adob2b10pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %advsb10pW = stablehlo.multiply %adb2b10pW, %b10pWv : tensor<112x672x1x1xf32>
    %adg2b10pW = stablehlo.multiply %b10dpW, %b10dpW : tensor<112x672x1x1xf32>
    %advgb10pW = stablehlo.multiply %adob2b10pW, %adg2b10pW : tensor<112x672x1x1xf32>
    %advnb10pW = stablehlo.add %advsb10pW, %advgb10pW : tensor<112x672x1x1xf32>
    %adbc1b10pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adbc2b10pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %admhb10pW = stablehlo.divide %admnb10pW, %adbc1b10pW : tensor<112x672x1x1xf32>
    %advhb10pW = stablehlo.divide %advnb10pW, %adbc2b10pW : tensor<112x672x1x1xf32>
    %adlrb10pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adepsb10pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adsqb10pW = stablehlo.sqrt %advhb10pW : tensor<112x672x1x1xf32>
    %addenb10pW = stablehlo.add %adsqb10pW, %adepsb10pW : tensor<112x672x1x1xf32>
    %adratb10pW = stablehlo.divide %admhb10pW, %addenb10pW : tensor<112x672x1x1xf32>
    %adstb10pW = stablehlo.multiply %adlrb10pW, %adratb10pW : tensor<112x672x1x1xf32>
    %adsubb10pW = stablehlo.subtract %b10pW, %adstb10pW : tensor<112x672x1x1xf32>
    %adwdb10pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adwdlrb10pW = stablehlo.multiply %adwdb10pW, %adlrb10pW : tensor<112x672x1x1xf32>
    %adwdpb10pW = stablehlo.multiply %adwdlrb10pW, %b10pW : tensor<112x672x1x1xf32>
    %adnewb10pW = stablehlo.subtract %adsubb10pW, %adwdpb10pW : tensor<112x672x1x1xf32>
    %adb1b10pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b10pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb10pb = stablehlo.multiply %adb1b10pb, %b10pbm : tensor<112xf32>
    %admgb10pb = stablehlo.multiply %adob1b10pb, %b10dpb : tensor<112xf32>
    %admnb10pb = stablehlo.add %admsb10pb, %admgb10pb : tensor<112xf32>
    %adb2b10pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b10pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb10pb = stablehlo.multiply %adb2b10pb, %b10pbv : tensor<112xf32>
    %adg2b10pb = stablehlo.multiply %b10dpb, %b10dpb : tensor<112xf32>
    %advgb10pb = stablehlo.multiply %adob2b10pb, %adg2b10pb : tensor<112xf32>
    %advnb10pb = stablehlo.add %advsb10pb, %advgb10pb : tensor<112xf32>
    %adbc1b10pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b10pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb10pb = stablehlo.divide %admnb10pb, %adbc1b10pb : tensor<112xf32>
    %advhb10pb = stablehlo.divide %advnb10pb, %adbc2b10pb : tensor<112xf32>
    %adlrb10pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb10pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb10pb = stablehlo.sqrt %advhb10pb : tensor<112xf32>
    %addenb10pb = stablehlo.add %adsqb10pb, %adepsb10pb : tensor<112xf32>
    %adratb10pb = stablehlo.divide %admhb10pb, %addenb10pb : tensor<112xf32>
    %adstb10pb = stablehlo.multiply %adlrb10pb, %adratb10pb : tensor<112xf32>
    %adsubb10pb = stablehlo.subtract %b10pb, %adstb10pb : tensor<112xf32>
    %adwdb10pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb10pb = stablehlo.multiply %adwdb10pb, %adlrb10pb : tensor<112xf32>
    %adwdpb10pb = stablehlo.multiply %adwdlrb10pb, %b10pb : tensor<112xf32>
    %adnewb10pb = stablehlo.subtract %adsubb10pb, %adwdpb10pb : tensor<112xf32>
    %adb1b10pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b10pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb10pg = stablehlo.multiply %adb1b10pg, %b10pgm : tensor<112xf32>
    %admgb10pg = stablehlo.multiply %adob1b10pg, %b10dpndg : tensor<112xf32>
    %admnb10pg = stablehlo.add %admsb10pg, %admgb10pg : tensor<112xf32>
    %adb2b10pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b10pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb10pg = stablehlo.multiply %adb2b10pg, %b10pgv : tensor<112xf32>
    %adg2b10pg = stablehlo.multiply %b10dpndg, %b10dpndg : tensor<112xf32>
    %advgb10pg = stablehlo.multiply %adob2b10pg, %adg2b10pg : tensor<112xf32>
    %advnb10pg = stablehlo.add %advsb10pg, %advgb10pg : tensor<112xf32>
    %adbc1b10pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b10pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb10pg = stablehlo.divide %admnb10pg, %adbc1b10pg : tensor<112xf32>
    %advhb10pg = stablehlo.divide %advnb10pg, %adbc2b10pg : tensor<112xf32>
    %adlrb10pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb10pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb10pg = stablehlo.sqrt %advhb10pg : tensor<112xf32>
    %addenb10pg = stablehlo.add %adsqb10pg, %adepsb10pg : tensor<112xf32>
    %adratb10pg = stablehlo.divide %admhb10pg, %addenb10pg : tensor<112xf32>
    %adstb10pg = stablehlo.multiply %adlrb10pg, %adratb10pg : tensor<112xf32>
    %adsubb10pg = stablehlo.subtract %b10pg, %adstb10pg : tensor<112xf32>
    %adwdb10pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb10pg = stablehlo.multiply %adwdb10pg, %adlrb10pg : tensor<112xf32>
    %adwdpb10pg = stablehlo.multiply %adwdlrb10pg, %b10pg : tensor<112xf32>
    %adnewb10pg = stablehlo.subtract %adsubb10pg, %adwdpb10pg : tensor<112xf32>
    %adb1b10pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b10pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb10pbt = stablehlo.multiply %adb1b10pbt, %b10pbtm : tensor<112xf32>
    %admgb10pbt = stablehlo.multiply %adob1b10pbt, %b10dpndb : tensor<112xf32>
    %admnb10pbt = stablehlo.add %admsb10pbt, %admgb10pbt : tensor<112xf32>
    %adb2b10pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b10pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb10pbt = stablehlo.multiply %adb2b10pbt, %b10pbtv : tensor<112xf32>
    %adg2b10pbt = stablehlo.multiply %b10dpndb, %b10dpndb : tensor<112xf32>
    %advgb10pbt = stablehlo.multiply %adob2b10pbt, %adg2b10pbt : tensor<112xf32>
    %advnb10pbt = stablehlo.add %advsb10pbt, %advgb10pbt : tensor<112xf32>
    %adbc1b10pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b10pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb10pbt = stablehlo.divide %admnb10pbt, %adbc1b10pbt : tensor<112xf32>
    %advhb10pbt = stablehlo.divide %advnb10pbt, %adbc2b10pbt : tensor<112xf32>
    %adlrb10pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb10pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb10pbt = stablehlo.sqrt %advhb10pbt : tensor<112xf32>
    %addenb10pbt = stablehlo.add %adsqb10pbt, %adepsb10pbt : tensor<112xf32>
    %adratb10pbt = stablehlo.divide %admhb10pbt, %addenb10pbt : tensor<112xf32>
    %adstb10pbt = stablehlo.multiply %adlrb10pbt, %adratb10pbt : tensor<112xf32>
    %adsubb10pbt = stablehlo.subtract %b10pbt, %adstb10pbt : tensor<112xf32>
    %adwdb10pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb10pbt = stablehlo.multiply %adwdb10pbt, %adlrb10pbt : tensor<112xf32>
    %adwdpb10pbt = stablehlo.multiply %adwdlrb10pbt, %b10pbt : tensor<112xf32>
    %adnewb10pbt = stablehlo.subtract %adsubb10pbt, %adwdpb10pbt : tensor<112xf32>
    %adb1b11eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adob1b11eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %admsb11eW = stablehlo.multiply %adb1b11eW, %b11eWm : tensor<672x112x1x1xf32>
    %admgb11eW = stablehlo.multiply %adob1b11eW, %b11deW : tensor<672x112x1x1xf32>
    %admnb11eW = stablehlo.add %admsb11eW, %admgb11eW : tensor<672x112x1x1xf32>
    %adb2b11eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adob2b11eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %advsb11eW = stablehlo.multiply %adb2b11eW, %b11eWv : tensor<672x112x1x1xf32>
    %adg2b11eW = stablehlo.multiply %b11deW, %b11deW : tensor<672x112x1x1xf32>
    %advgb11eW = stablehlo.multiply %adob2b11eW, %adg2b11eW : tensor<672x112x1x1xf32>
    %advnb11eW = stablehlo.add %advsb11eW, %advgb11eW : tensor<672x112x1x1xf32>
    %adbc1b11eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adbc2b11eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %admhb11eW = stablehlo.divide %admnb11eW, %adbc1b11eW : tensor<672x112x1x1xf32>
    %advhb11eW = stablehlo.divide %advnb11eW, %adbc2b11eW : tensor<672x112x1x1xf32>
    %adlrb11eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adepsb11eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adsqb11eW = stablehlo.sqrt %advhb11eW : tensor<672x112x1x1xf32>
    %addenb11eW = stablehlo.add %adsqb11eW, %adepsb11eW : tensor<672x112x1x1xf32>
    %adratb11eW = stablehlo.divide %admhb11eW, %addenb11eW : tensor<672x112x1x1xf32>
    %adstb11eW = stablehlo.multiply %adlrb11eW, %adratb11eW : tensor<672x112x1x1xf32>
    %adsubb11eW = stablehlo.subtract %b11eW, %adstb11eW : tensor<672x112x1x1xf32>
    %adwdb11eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adwdlrb11eW = stablehlo.multiply %adwdb11eW, %adlrb11eW : tensor<672x112x1x1xf32>
    %adwdpb11eW = stablehlo.multiply %adwdlrb11eW, %b11eW : tensor<672x112x1x1xf32>
    %adnewb11eW = stablehlo.subtract %adsubb11eW, %adwdpb11eW : tensor<672x112x1x1xf32>
    %adb1b11eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b11eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb11eb = stablehlo.multiply %adb1b11eb, %b11ebm : tensor<672xf32>
    %admgb11eb = stablehlo.multiply %adob1b11eb, %b11deb : tensor<672xf32>
    %admnb11eb = stablehlo.add %admsb11eb, %admgb11eb : tensor<672xf32>
    %adb2b11eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b11eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb11eb = stablehlo.multiply %adb2b11eb, %b11ebv : tensor<672xf32>
    %adg2b11eb = stablehlo.multiply %b11deb, %b11deb : tensor<672xf32>
    %advgb11eb = stablehlo.multiply %adob2b11eb, %adg2b11eb : tensor<672xf32>
    %advnb11eb = stablehlo.add %advsb11eb, %advgb11eb : tensor<672xf32>
    %adbc1b11eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b11eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb11eb = stablehlo.divide %admnb11eb, %adbc1b11eb : tensor<672xf32>
    %advhb11eb = stablehlo.divide %advnb11eb, %adbc2b11eb : tensor<672xf32>
    %adlrb11eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb11eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb11eb = stablehlo.sqrt %advhb11eb : tensor<672xf32>
    %addenb11eb = stablehlo.add %adsqb11eb, %adepsb11eb : tensor<672xf32>
    %adratb11eb = stablehlo.divide %admhb11eb, %addenb11eb : tensor<672xf32>
    %adstb11eb = stablehlo.multiply %adlrb11eb, %adratb11eb : tensor<672xf32>
    %adsubb11eb = stablehlo.subtract %b11eb, %adstb11eb : tensor<672xf32>
    %adwdb11eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb11eb = stablehlo.multiply %adwdb11eb, %adlrb11eb : tensor<672xf32>
    %adwdpb11eb = stablehlo.multiply %adwdlrb11eb, %b11eb : tensor<672xf32>
    %adnewb11eb = stablehlo.subtract %adsubb11eb, %adwdpb11eb : tensor<672xf32>
    %adb1b11eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b11eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb11eg = stablehlo.multiply %adb1b11eg, %b11egm : tensor<672xf32>
    %admgb11eg = stablehlo.multiply %adob1b11eg, %b11dendg : tensor<672xf32>
    %admnb11eg = stablehlo.add %admsb11eg, %admgb11eg : tensor<672xf32>
    %adb2b11eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b11eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb11eg = stablehlo.multiply %adb2b11eg, %b11egv : tensor<672xf32>
    %adg2b11eg = stablehlo.multiply %b11dendg, %b11dendg : tensor<672xf32>
    %advgb11eg = stablehlo.multiply %adob2b11eg, %adg2b11eg : tensor<672xf32>
    %advnb11eg = stablehlo.add %advsb11eg, %advgb11eg : tensor<672xf32>
    %adbc1b11eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b11eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb11eg = stablehlo.divide %admnb11eg, %adbc1b11eg : tensor<672xf32>
    %advhb11eg = stablehlo.divide %advnb11eg, %adbc2b11eg : tensor<672xf32>
    %adlrb11eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb11eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb11eg = stablehlo.sqrt %advhb11eg : tensor<672xf32>
    %addenb11eg = stablehlo.add %adsqb11eg, %adepsb11eg : tensor<672xf32>
    %adratb11eg = stablehlo.divide %admhb11eg, %addenb11eg : tensor<672xf32>
    %adstb11eg = stablehlo.multiply %adlrb11eg, %adratb11eg : tensor<672xf32>
    %adsubb11eg = stablehlo.subtract %b11eg, %adstb11eg : tensor<672xf32>
    %adwdb11eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb11eg = stablehlo.multiply %adwdb11eg, %adlrb11eg : tensor<672xf32>
    %adwdpb11eg = stablehlo.multiply %adwdlrb11eg, %b11eg : tensor<672xf32>
    %adnewb11eg = stablehlo.subtract %adsubb11eg, %adwdpb11eg : tensor<672xf32>
    %adb1b11ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b11ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb11ebt = stablehlo.multiply %adb1b11ebt, %b11ebtm : tensor<672xf32>
    %admgb11ebt = stablehlo.multiply %adob1b11ebt, %b11dendb : tensor<672xf32>
    %admnb11ebt = stablehlo.add %admsb11ebt, %admgb11ebt : tensor<672xf32>
    %adb2b11ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b11ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb11ebt = stablehlo.multiply %adb2b11ebt, %b11ebtv : tensor<672xf32>
    %adg2b11ebt = stablehlo.multiply %b11dendb, %b11dendb : tensor<672xf32>
    %advgb11ebt = stablehlo.multiply %adob2b11ebt, %adg2b11ebt : tensor<672xf32>
    %advnb11ebt = stablehlo.add %advsb11ebt, %advgb11ebt : tensor<672xf32>
    %adbc1b11ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b11ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb11ebt = stablehlo.divide %admnb11ebt, %adbc1b11ebt : tensor<672xf32>
    %advhb11ebt = stablehlo.divide %advnb11ebt, %adbc2b11ebt : tensor<672xf32>
    %adlrb11ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb11ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb11ebt = stablehlo.sqrt %advhb11ebt : tensor<672xf32>
    %addenb11ebt = stablehlo.add %adsqb11ebt, %adepsb11ebt : tensor<672xf32>
    %adratb11ebt = stablehlo.divide %admhb11ebt, %addenb11ebt : tensor<672xf32>
    %adstb11ebt = stablehlo.multiply %adlrb11ebt, %adratb11ebt : tensor<672xf32>
    %adsubb11ebt = stablehlo.subtract %b11ebt, %adstb11ebt : tensor<672xf32>
    %adwdb11ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb11ebt = stablehlo.multiply %adwdb11ebt, %adlrb11ebt : tensor<672xf32>
    %adwdpb11ebt = stablehlo.multiply %adwdlrb11ebt, %b11ebt : tensor<672xf32>
    %adnewb11ebt = stablehlo.subtract %adsubb11ebt, %adwdpb11ebt : tensor<672xf32>
    %adb1b11dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adob1b11dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %admsb11dW = stablehlo.multiply %adb1b11dW, %b11dWm : tensor<672x1x5x5xf32>
    %admgb11dW = stablehlo.multiply %adob1b11dW, %b11ddW : tensor<672x1x5x5xf32>
    %admnb11dW = stablehlo.add %admsb11dW, %admgb11dW : tensor<672x1x5x5xf32>
    %adb2b11dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adob2b11dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %advsb11dW = stablehlo.multiply %adb2b11dW, %b11dWv : tensor<672x1x5x5xf32>
    %adg2b11dW = stablehlo.multiply %b11ddW, %b11ddW : tensor<672x1x5x5xf32>
    %advgb11dW = stablehlo.multiply %adob2b11dW, %adg2b11dW : tensor<672x1x5x5xf32>
    %advnb11dW = stablehlo.add %advsb11dW, %advgb11dW : tensor<672x1x5x5xf32>
    %adbc1b11dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adbc2b11dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %admhb11dW = stablehlo.divide %admnb11dW, %adbc1b11dW : tensor<672x1x5x5xf32>
    %advhb11dW = stablehlo.divide %advnb11dW, %adbc2b11dW : tensor<672x1x5x5xf32>
    %adlrb11dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adepsb11dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adsqb11dW = stablehlo.sqrt %advhb11dW : tensor<672x1x5x5xf32>
    %addenb11dW = stablehlo.add %adsqb11dW, %adepsb11dW : tensor<672x1x5x5xf32>
    %adratb11dW = stablehlo.divide %admhb11dW, %addenb11dW : tensor<672x1x5x5xf32>
    %adstb11dW = stablehlo.multiply %adlrb11dW, %adratb11dW : tensor<672x1x5x5xf32>
    %adsubb11dW = stablehlo.subtract %b11dW, %adstb11dW : tensor<672x1x5x5xf32>
    %adwdb11dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adwdlrb11dW = stablehlo.multiply %adwdb11dW, %adlrb11dW : tensor<672x1x5x5xf32>
    %adwdpb11dW = stablehlo.multiply %adwdlrb11dW, %b11dW : tensor<672x1x5x5xf32>
    %adnewb11dW = stablehlo.subtract %adsubb11dW, %adwdpb11dW : tensor<672x1x5x5xf32>
    %adb1b11db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b11db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb11db = stablehlo.multiply %adb1b11db, %b11dbm : tensor<672xf32>
    %admgb11db = stablehlo.multiply %adob1b11db, %b11ddb : tensor<672xf32>
    %admnb11db = stablehlo.add %admsb11db, %admgb11db : tensor<672xf32>
    %adb2b11db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b11db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb11db = stablehlo.multiply %adb2b11db, %b11dbv : tensor<672xf32>
    %adg2b11db = stablehlo.multiply %b11ddb, %b11ddb : tensor<672xf32>
    %advgb11db = stablehlo.multiply %adob2b11db, %adg2b11db : tensor<672xf32>
    %advnb11db = stablehlo.add %advsb11db, %advgb11db : tensor<672xf32>
    %adbc1b11db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b11db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb11db = stablehlo.divide %admnb11db, %adbc1b11db : tensor<672xf32>
    %advhb11db = stablehlo.divide %advnb11db, %adbc2b11db : tensor<672xf32>
    %adlrb11db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb11db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb11db = stablehlo.sqrt %advhb11db : tensor<672xf32>
    %addenb11db = stablehlo.add %adsqb11db, %adepsb11db : tensor<672xf32>
    %adratb11db = stablehlo.divide %admhb11db, %addenb11db : tensor<672xf32>
    %adstb11db = stablehlo.multiply %adlrb11db, %adratb11db : tensor<672xf32>
    %adsubb11db = stablehlo.subtract %b11db, %adstb11db : tensor<672xf32>
    %adwdb11db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb11db = stablehlo.multiply %adwdb11db, %adlrb11db : tensor<672xf32>
    %adwdpb11db = stablehlo.multiply %adwdlrb11db, %b11db : tensor<672xf32>
    %adnewb11db = stablehlo.subtract %adsubb11db, %adwdpb11db : tensor<672xf32>
    %adb1b11dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b11dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb11dg = stablehlo.multiply %adb1b11dg, %b11dgm : tensor<672xf32>
    %admgb11dg = stablehlo.multiply %adob1b11dg, %b11ddndg : tensor<672xf32>
    %admnb11dg = stablehlo.add %admsb11dg, %admgb11dg : tensor<672xf32>
    %adb2b11dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b11dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb11dg = stablehlo.multiply %adb2b11dg, %b11dgv : tensor<672xf32>
    %adg2b11dg = stablehlo.multiply %b11ddndg, %b11ddndg : tensor<672xf32>
    %advgb11dg = stablehlo.multiply %adob2b11dg, %adg2b11dg : tensor<672xf32>
    %advnb11dg = stablehlo.add %advsb11dg, %advgb11dg : tensor<672xf32>
    %adbc1b11dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b11dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb11dg = stablehlo.divide %admnb11dg, %adbc1b11dg : tensor<672xf32>
    %advhb11dg = stablehlo.divide %advnb11dg, %adbc2b11dg : tensor<672xf32>
    %adlrb11dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb11dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb11dg = stablehlo.sqrt %advhb11dg : tensor<672xf32>
    %addenb11dg = stablehlo.add %adsqb11dg, %adepsb11dg : tensor<672xf32>
    %adratb11dg = stablehlo.divide %admhb11dg, %addenb11dg : tensor<672xf32>
    %adstb11dg = stablehlo.multiply %adlrb11dg, %adratb11dg : tensor<672xf32>
    %adsubb11dg = stablehlo.subtract %b11dg, %adstb11dg : tensor<672xf32>
    %adwdb11dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb11dg = stablehlo.multiply %adwdb11dg, %adlrb11dg : tensor<672xf32>
    %adwdpb11dg = stablehlo.multiply %adwdlrb11dg, %b11dg : tensor<672xf32>
    %adnewb11dg = stablehlo.subtract %adsubb11dg, %adwdpb11dg : tensor<672xf32>
    %adb1b11dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b11dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb11dbt = stablehlo.multiply %adb1b11dbt, %b11dbtm : tensor<672xf32>
    %admgb11dbt = stablehlo.multiply %adob1b11dbt, %b11ddndb : tensor<672xf32>
    %admnb11dbt = stablehlo.add %admsb11dbt, %admgb11dbt : tensor<672xf32>
    %adb2b11dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b11dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb11dbt = stablehlo.multiply %adb2b11dbt, %b11dbtv : tensor<672xf32>
    %adg2b11dbt = stablehlo.multiply %b11ddndb, %b11ddndb : tensor<672xf32>
    %advgb11dbt = stablehlo.multiply %adob2b11dbt, %adg2b11dbt : tensor<672xf32>
    %advnb11dbt = stablehlo.add %advsb11dbt, %advgb11dbt : tensor<672xf32>
    %adbc1b11dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b11dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb11dbt = stablehlo.divide %admnb11dbt, %adbc1b11dbt : tensor<672xf32>
    %advhb11dbt = stablehlo.divide %advnb11dbt, %adbc2b11dbt : tensor<672xf32>
    %adlrb11dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb11dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb11dbt = stablehlo.sqrt %advhb11dbt : tensor<672xf32>
    %addenb11dbt = stablehlo.add %adsqb11dbt, %adepsb11dbt : tensor<672xf32>
    %adratb11dbt = stablehlo.divide %admhb11dbt, %addenb11dbt : tensor<672xf32>
    %adstb11dbt = stablehlo.multiply %adlrb11dbt, %adratb11dbt : tensor<672xf32>
    %adsubb11dbt = stablehlo.subtract %b11dbt, %adstb11dbt : tensor<672xf32>
    %adwdb11dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb11dbt = stablehlo.multiply %adwdb11dbt, %adlrb11dbt : tensor<672xf32>
    %adwdpb11dbt = stablehlo.multiply %adwdlrb11dbt, %b11dbt : tensor<672xf32>
    %adnewb11dbt = stablehlo.subtract %adsubb11dbt, %adwdpb11dbt : tensor<672xf32>
    %adb1b11zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adob1b11zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %admsb11zW1 = stablehlo.multiply %adb1b11zW1, %b11zW1m : tensor<672x28xf32>
    %admgb11zW1 = stablehlo.multiply %adob1b11zW1, %b11zdWs1 : tensor<672x28xf32>
    %admnb11zW1 = stablehlo.add %admsb11zW1, %admgb11zW1 : tensor<672x28xf32>
    %adb2b11zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adob2b11zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %advsb11zW1 = stablehlo.multiply %adb2b11zW1, %b11zW1v : tensor<672x28xf32>
    %adg2b11zW1 = stablehlo.multiply %b11zdWs1, %b11zdWs1 : tensor<672x28xf32>
    %advgb11zW1 = stablehlo.multiply %adob2b11zW1, %adg2b11zW1 : tensor<672x28xf32>
    %advnb11zW1 = stablehlo.add %advsb11zW1, %advgb11zW1 : tensor<672x28xf32>
    %adbc1b11zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adbc2b11zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %admhb11zW1 = stablehlo.divide %admnb11zW1, %adbc1b11zW1 : tensor<672x28xf32>
    %advhb11zW1 = stablehlo.divide %advnb11zW1, %adbc2b11zW1 : tensor<672x28xf32>
    %adlrb11zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adepsb11zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adsqb11zW1 = stablehlo.sqrt %advhb11zW1 : tensor<672x28xf32>
    %addenb11zW1 = stablehlo.add %adsqb11zW1, %adepsb11zW1 : tensor<672x28xf32>
    %adratb11zW1 = stablehlo.divide %admhb11zW1, %addenb11zW1 : tensor<672x28xf32>
    %adstb11zW1 = stablehlo.multiply %adlrb11zW1, %adratb11zW1 : tensor<672x28xf32>
    %adsubb11zW1 = stablehlo.subtract %b11zW1, %adstb11zW1 : tensor<672x28xf32>
    %adwdb11zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adwdlrb11zW1 = stablehlo.multiply %adwdb11zW1, %adlrb11zW1 : tensor<672x28xf32>
    %adwdpb11zW1 = stablehlo.multiply %adwdlrb11zW1, %b11zW1 : tensor<672x28xf32>
    %adnewb11zW1 = stablehlo.subtract %adsubb11zW1, %adwdpb11zW1 : tensor<672x28xf32>
    %adb1b11zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adob1b11zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %admsb11zb1 = stablehlo.multiply %adb1b11zb1, %b11zb1m : tensor<28xf32>
    %admgb11zb1 = stablehlo.multiply %adob1b11zb1, %b11zdbs1 : tensor<28xf32>
    %admnb11zb1 = stablehlo.add %admsb11zb1, %admgb11zb1 : tensor<28xf32>
    %adb2b11zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adob2b11zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %advsb11zb1 = stablehlo.multiply %adb2b11zb1, %b11zb1v : tensor<28xf32>
    %adg2b11zb1 = stablehlo.multiply %b11zdbs1, %b11zdbs1 : tensor<28xf32>
    %advgb11zb1 = stablehlo.multiply %adob2b11zb1, %adg2b11zb1 : tensor<28xf32>
    %advnb11zb1 = stablehlo.add %advsb11zb1, %advgb11zb1 : tensor<28xf32>
    %adbc1b11zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adbc2b11zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %admhb11zb1 = stablehlo.divide %admnb11zb1, %adbc1b11zb1 : tensor<28xf32>
    %advhb11zb1 = stablehlo.divide %advnb11zb1, %adbc2b11zb1 : tensor<28xf32>
    %adlrb11zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adepsb11zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adsqb11zb1 = stablehlo.sqrt %advhb11zb1 : tensor<28xf32>
    %addenb11zb1 = stablehlo.add %adsqb11zb1, %adepsb11zb1 : tensor<28xf32>
    %adratb11zb1 = stablehlo.divide %admhb11zb1, %addenb11zb1 : tensor<28xf32>
    %adstb11zb1 = stablehlo.multiply %adlrb11zb1, %adratb11zb1 : tensor<28xf32>
    %adsubb11zb1 = stablehlo.subtract %b11zb1, %adstb11zb1 : tensor<28xf32>
    %adwdb11zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adwdlrb11zb1 = stablehlo.multiply %adwdb11zb1, %adlrb11zb1 : tensor<28xf32>
    %adwdpb11zb1 = stablehlo.multiply %adwdlrb11zb1, %b11zb1 : tensor<28xf32>
    %adnewb11zb1 = stablehlo.subtract %adsubb11zb1, %adwdpb11zb1 : tensor<28xf32>
    %adb1b11zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adob1b11zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %admsb11zW2 = stablehlo.multiply %adb1b11zW2, %b11zW2m : tensor<28x672xf32>
    %admgb11zW2 = stablehlo.multiply %adob1b11zW2, %b11zdWs2 : tensor<28x672xf32>
    %admnb11zW2 = stablehlo.add %admsb11zW2, %admgb11zW2 : tensor<28x672xf32>
    %adb2b11zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adob2b11zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %advsb11zW2 = stablehlo.multiply %adb2b11zW2, %b11zW2v : tensor<28x672xf32>
    %adg2b11zW2 = stablehlo.multiply %b11zdWs2, %b11zdWs2 : tensor<28x672xf32>
    %advgb11zW2 = stablehlo.multiply %adob2b11zW2, %adg2b11zW2 : tensor<28x672xf32>
    %advnb11zW2 = stablehlo.add %advsb11zW2, %advgb11zW2 : tensor<28x672xf32>
    %adbc1b11zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adbc2b11zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %admhb11zW2 = stablehlo.divide %admnb11zW2, %adbc1b11zW2 : tensor<28x672xf32>
    %advhb11zW2 = stablehlo.divide %advnb11zW2, %adbc2b11zW2 : tensor<28x672xf32>
    %adlrb11zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adepsb11zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adsqb11zW2 = stablehlo.sqrt %advhb11zW2 : tensor<28x672xf32>
    %addenb11zW2 = stablehlo.add %adsqb11zW2, %adepsb11zW2 : tensor<28x672xf32>
    %adratb11zW2 = stablehlo.divide %admhb11zW2, %addenb11zW2 : tensor<28x672xf32>
    %adstb11zW2 = stablehlo.multiply %adlrb11zW2, %adratb11zW2 : tensor<28x672xf32>
    %adsubb11zW2 = stablehlo.subtract %b11zW2, %adstb11zW2 : tensor<28x672xf32>
    %adwdb11zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adwdlrb11zW2 = stablehlo.multiply %adwdb11zW2, %adlrb11zW2 : tensor<28x672xf32>
    %adwdpb11zW2 = stablehlo.multiply %adwdlrb11zW2, %b11zW2 : tensor<28x672xf32>
    %adnewb11zW2 = stablehlo.subtract %adsubb11zW2, %adwdpb11zW2 : tensor<28x672xf32>
    %adb1b11zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b11zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb11zb2 = stablehlo.multiply %adb1b11zb2, %b11zb2m : tensor<672xf32>
    %admgb11zb2 = stablehlo.multiply %adob1b11zb2, %b11zdbs2 : tensor<672xf32>
    %admnb11zb2 = stablehlo.add %admsb11zb2, %admgb11zb2 : tensor<672xf32>
    %adb2b11zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b11zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb11zb2 = stablehlo.multiply %adb2b11zb2, %b11zb2v : tensor<672xf32>
    %adg2b11zb2 = stablehlo.multiply %b11zdbs2, %b11zdbs2 : tensor<672xf32>
    %advgb11zb2 = stablehlo.multiply %adob2b11zb2, %adg2b11zb2 : tensor<672xf32>
    %advnb11zb2 = stablehlo.add %advsb11zb2, %advgb11zb2 : tensor<672xf32>
    %adbc1b11zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b11zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb11zb2 = stablehlo.divide %admnb11zb2, %adbc1b11zb2 : tensor<672xf32>
    %advhb11zb2 = stablehlo.divide %advnb11zb2, %adbc2b11zb2 : tensor<672xf32>
    %adlrb11zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb11zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb11zb2 = stablehlo.sqrt %advhb11zb2 : tensor<672xf32>
    %addenb11zb2 = stablehlo.add %adsqb11zb2, %adepsb11zb2 : tensor<672xf32>
    %adratb11zb2 = stablehlo.divide %admhb11zb2, %addenb11zb2 : tensor<672xf32>
    %adstb11zb2 = stablehlo.multiply %adlrb11zb2, %adratb11zb2 : tensor<672xf32>
    %adsubb11zb2 = stablehlo.subtract %b11zb2, %adstb11zb2 : tensor<672xf32>
    %adwdb11zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb11zb2 = stablehlo.multiply %adwdb11zb2, %adlrb11zb2 : tensor<672xf32>
    %adwdpb11zb2 = stablehlo.multiply %adwdlrb11zb2, %b11zb2 : tensor<672xf32>
    %adnewb11zb2 = stablehlo.subtract %adsubb11zb2, %adwdpb11zb2 : tensor<672xf32>
    %adb1b11pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adob1b11pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %admsb11pW = stablehlo.multiply %adb1b11pW, %b11pWm : tensor<112x672x1x1xf32>
    %admgb11pW = stablehlo.multiply %adob1b11pW, %b11dpW : tensor<112x672x1x1xf32>
    %admnb11pW = stablehlo.add %admsb11pW, %admgb11pW : tensor<112x672x1x1xf32>
    %adb2b11pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adob2b11pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %advsb11pW = stablehlo.multiply %adb2b11pW, %b11pWv : tensor<112x672x1x1xf32>
    %adg2b11pW = stablehlo.multiply %b11dpW, %b11dpW : tensor<112x672x1x1xf32>
    %advgb11pW = stablehlo.multiply %adob2b11pW, %adg2b11pW : tensor<112x672x1x1xf32>
    %advnb11pW = stablehlo.add %advsb11pW, %advgb11pW : tensor<112x672x1x1xf32>
    %adbc1b11pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adbc2b11pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %admhb11pW = stablehlo.divide %admnb11pW, %adbc1b11pW : tensor<112x672x1x1xf32>
    %advhb11pW = stablehlo.divide %advnb11pW, %adbc2b11pW : tensor<112x672x1x1xf32>
    %adlrb11pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adepsb11pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adsqb11pW = stablehlo.sqrt %advhb11pW : tensor<112x672x1x1xf32>
    %addenb11pW = stablehlo.add %adsqb11pW, %adepsb11pW : tensor<112x672x1x1xf32>
    %adratb11pW = stablehlo.divide %admhb11pW, %addenb11pW : tensor<112x672x1x1xf32>
    %adstb11pW = stablehlo.multiply %adlrb11pW, %adratb11pW : tensor<112x672x1x1xf32>
    %adsubb11pW = stablehlo.subtract %b11pW, %adstb11pW : tensor<112x672x1x1xf32>
    %adwdb11pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112x672x1x1xf32>
    %adwdlrb11pW = stablehlo.multiply %adwdb11pW, %adlrb11pW : tensor<112x672x1x1xf32>
    %adwdpb11pW = stablehlo.multiply %adwdlrb11pW, %b11pW : tensor<112x672x1x1xf32>
    %adnewb11pW = stablehlo.subtract %adsubb11pW, %adwdpb11pW : tensor<112x672x1x1xf32>
    %adb1b11pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b11pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb11pb = stablehlo.multiply %adb1b11pb, %b11pbm : tensor<112xf32>
    %admgb11pb = stablehlo.multiply %adob1b11pb, %b11dpb : tensor<112xf32>
    %admnb11pb = stablehlo.add %admsb11pb, %admgb11pb : tensor<112xf32>
    %adb2b11pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b11pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb11pb = stablehlo.multiply %adb2b11pb, %b11pbv : tensor<112xf32>
    %adg2b11pb = stablehlo.multiply %b11dpb, %b11dpb : tensor<112xf32>
    %advgb11pb = stablehlo.multiply %adob2b11pb, %adg2b11pb : tensor<112xf32>
    %advnb11pb = stablehlo.add %advsb11pb, %advgb11pb : tensor<112xf32>
    %adbc1b11pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b11pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb11pb = stablehlo.divide %admnb11pb, %adbc1b11pb : tensor<112xf32>
    %advhb11pb = stablehlo.divide %advnb11pb, %adbc2b11pb : tensor<112xf32>
    %adlrb11pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb11pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb11pb = stablehlo.sqrt %advhb11pb : tensor<112xf32>
    %addenb11pb = stablehlo.add %adsqb11pb, %adepsb11pb : tensor<112xf32>
    %adratb11pb = stablehlo.divide %admhb11pb, %addenb11pb : tensor<112xf32>
    %adstb11pb = stablehlo.multiply %adlrb11pb, %adratb11pb : tensor<112xf32>
    %adsubb11pb = stablehlo.subtract %b11pb, %adstb11pb : tensor<112xf32>
    %adwdb11pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb11pb = stablehlo.multiply %adwdb11pb, %adlrb11pb : tensor<112xf32>
    %adwdpb11pb = stablehlo.multiply %adwdlrb11pb, %b11pb : tensor<112xf32>
    %adnewb11pb = stablehlo.subtract %adsubb11pb, %adwdpb11pb : tensor<112xf32>
    %adb1b11pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b11pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb11pg = stablehlo.multiply %adb1b11pg, %b11pgm : tensor<112xf32>
    %admgb11pg = stablehlo.multiply %adob1b11pg, %b11dpndg : tensor<112xf32>
    %admnb11pg = stablehlo.add %admsb11pg, %admgb11pg : tensor<112xf32>
    %adb2b11pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b11pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb11pg = stablehlo.multiply %adb2b11pg, %b11pgv : tensor<112xf32>
    %adg2b11pg = stablehlo.multiply %b11dpndg, %b11dpndg : tensor<112xf32>
    %advgb11pg = stablehlo.multiply %adob2b11pg, %adg2b11pg : tensor<112xf32>
    %advnb11pg = stablehlo.add %advsb11pg, %advgb11pg : tensor<112xf32>
    %adbc1b11pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b11pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb11pg = stablehlo.divide %admnb11pg, %adbc1b11pg : tensor<112xf32>
    %advhb11pg = stablehlo.divide %advnb11pg, %adbc2b11pg : tensor<112xf32>
    %adlrb11pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb11pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb11pg = stablehlo.sqrt %advhb11pg : tensor<112xf32>
    %addenb11pg = stablehlo.add %adsqb11pg, %adepsb11pg : tensor<112xf32>
    %adratb11pg = stablehlo.divide %admhb11pg, %addenb11pg : tensor<112xf32>
    %adstb11pg = stablehlo.multiply %adlrb11pg, %adratb11pg : tensor<112xf32>
    %adsubb11pg = stablehlo.subtract %b11pg, %adstb11pg : tensor<112xf32>
    %adwdb11pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb11pg = stablehlo.multiply %adwdb11pg, %adlrb11pg : tensor<112xf32>
    %adwdpb11pg = stablehlo.multiply %adwdlrb11pg, %b11pg : tensor<112xf32>
    %adnewb11pg = stablehlo.subtract %adsubb11pg, %adwdpb11pg : tensor<112xf32>
    %adb1b11pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob1b11pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admsb11pbt = stablehlo.multiply %adb1b11pbt, %b11pbtm : tensor<112xf32>
    %admgb11pbt = stablehlo.multiply %adob1b11pbt, %b11dpndb : tensor<112xf32>
    %admnb11pbt = stablehlo.add %admsb11pbt, %admgb11pbt : tensor<112xf32>
    %adb2b11pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adob2b11pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %advsb11pbt = stablehlo.multiply %adb2b11pbt, %b11pbtv : tensor<112xf32>
    %adg2b11pbt = stablehlo.multiply %b11dpndb, %b11dpndb : tensor<112xf32>
    %advgb11pbt = stablehlo.multiply %adob2b11pbt, %adg2b11pbt : tensor<112xf32>
    %advnb11pbt = stablehlo.add %advsb11pbt, %advgb11pbt : tensor<112xf32>
    %adbc1b11pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adbc2b11pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %admhb11pbt = stablehlo.divide %admnb11pbt, %adbc1b11pbt : tensor<112xf32>
    %advhb11pbt = stablehlo.divide %advnb11pbt, %adbc2b11pbt : tensor<112xf32>
    %adlrb11pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adepsb11pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adsqb11pbt = stablehlo.sqrt %advhb11pbt : tensor<112xf32>
    %addenb11pbt = stablehlo.add %adsqb11pbt, %adepsb11pbt : tensor<112xf32>
    %adratb11pbt = stablehlo.divide %admhb11pbt, %addenb11pbt : tensor<112xf32>
    %adstb11pbt = stablehlo.multiply %adlrb11pbt, %adratb11pbt : tensor<112xf32>
    %adsubb11pbt = stablehlo.subtract %b11pbt, %adstb11pbt : tensor<112xf32>
    %adwdb11pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<112xf32>
    %adwdlrb11pbt = stablehlo.multiply %adwdb11pbt, %adlrb11pbt : tensor<112xf32>
    %adwdpb11pbt = stablehlo.multiply %adwdlrb11pbt, %b11pbt : tensor<112xf32>
    %adnewb11pbt = stablehlo.subtract %adsubb11pbt, %adwdpb11pbt : tensor<112xf32>
    %adb1b12eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adob1b12eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %admsb12eW = stablehlo.multiply %adb1b12eW, %b12eWm : tensor<672x112x1x1xf32>
    %admgb12eW = stablehlo.multiply %adob1b12eW, %b12deW : tensor<672x112x1x1xf32>
    %admnb12eW = stablehlo.add %admsb12eW, %admgb12eW : tensor<672x112x1x1xf32>
    %adb2b12eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adob2b12eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %advsb12eW = stablehlo.multiply %adb2b12eW, %b12eWv : tensor<672x112x1x1xf32>
    %adg2b12eW = stablehlo.multiply %b12deW, %b12deW : tensor<672x112x1x1xf32>
    %advgb12eW = stablehlo.multiply %adob2b12eW, %adg2b12eW : tensor<672x112x1x1xf32>
    %advnb12eW = stablehlo.add %advsb12eW, %advgb12eW : tensor<672x112x1x1xf32>
    %adbc1b12eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adbc2b12eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %admhb12eW = stablehlo.divide %admnb12eW, %adbc1b12eW : tensor<672x112x1x1xf32>
    %advhb12eW = stablehlo.divide %advnb12eW, %adbc2b12eW : tensor<672x112x1x1xf32>
    %adlrb12eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adepsb12eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adsqb12eW = stablehlo.sqrt %advhb12eW : tensor<672x112x1x1xf32>
    %addenb12eW = stablehlo.add %adsqb12eW, %adepsb12eW : tensor<672x112x1x1xf32>
    %adratb12eW = stablehlo.divide %admhb12eW, %addenb12eW : tensor<672x112x1x1xf32>
    %adstb12eW = stablehlo.multiply %adlrb12eW, %adratb12eW : tensor<672x112x1x1xf32>
    %adsubb12eW = stablehlo.subtract %b12eW, %adstb12eW : tensor<672x112x1x1xf32>
    %adwdb12eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x112x1x1xf32>
    %adwdlrb12eW = stablehlo.multiply %adwdb12eW, %adlrb12eW : tensor<672x112x1x1xf32>
    %adwdpb12eW = stablehlo.multiply %adwdlrb12eW, %b12eW : tensor<672x112x1x1xf32>
    %adnewb12eW = stablehlo.subtract %adsubb12eW, %adwdpb12eW : tensor<672x112x1x1xf32>
    %adb1b12eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b12eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb12eb = stablehlo.multiply %adb1b12eb, %b12ebm : tensor<672xf32>
    %admgb12eb = stablehlo.multiply %adob1b12eb, %b12deb : tensor<672xf32>
    %admnb12eb = stablehlo.add %admsb12eb, %admgb12eb : tensor<672xf32>
    %adb2b12eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b12eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb12eb = stablehlo.multiply %adb2b12eb, %b12ebv : tensor<672xf32>
    %adg2b12eb = stablehlo.multiply %b12deb, %b12deb : tensor<672xf32>
    %advgb12eb = stablehlo.multiply %adob2b12eb, %adg2b12eb : tensor<672xf32>
    %advnb12eb = stablehlo.add %advsb12eb, %advgb12eb : tensor<672xf32>
    %adbc1b12eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b12eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb12eb = stablehlo.divide %admnb12eb, %adbc1b12eb : tensor<672xf32>
    %advhb12eb = stablehlo.divide %advnb12eb, %adbc2b12eb : tensor<672xf32>
    %adlrb12eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb12eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb12eb = stablehlo.sqrt %advhb12eb : tensor<672xf32>
    %addenb12eb = stablehlo.add %adsqb12eb, %adepsb12eb : tensor<672xf32>
    %adratb12eb = stablehlo.divide %admhb12eb, %addenb12eb : tensor<672xf32>
    %adstb12eb = stablehlo.multiply %adlrb12eb, %adratb12eb : tensor<672xf32>
    %adsubb12eb = stablehlo.subtract %b12eb, %adstb12eb : tensor<672xf32>
    %adwdb12eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb12eb = stablehlo.multiply %adwdb12eb, %adlrb12eb : tensor<672xf32>
    %adwdpb12eb = stablehlo.multiply %adwdlrb12eb, %b12eb : tensor<672xf32>
    %adnewb12eb = stablehlo.subtract %adsubb12eb, %adwdpb12eb : tensor<672xf32>
    %adb1b12eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b12eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb12eg = stablehlo.multiply %adb1b12eg, %b12egm : tensor<672xf32>
    %admgb12eg = stablehlo.multiply %adob1b12eg, %b12dendg : tensor<672xf32>
    %admnb12eg = stablehlo.add %admsb12eg, %admgb12eg : tensor<672xf32>
    %adb2b12eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b12eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb12eg = stablehlo.multiply %adb2b12eg, %b12egv : tensor<672xf32>
    %adg2b12eg = stablehlo.multiply %b12dendg, %b12dendg : tensor<672xf32>
    %advgb12eg = stablehlo.multiply %adob2b12eg, %adg2b12eg : tensor<672xf32>
    %advnb12eg = stablehlo.add %advsb12eg, %advgb12eg : tensor<672xf32>
    %adbc1b12eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b12eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb12eg = stablehlo.divide %admnb12eg, %adbc1b12eg : tensor<672xf32>
    %advhb12eg = stablehlo.divide %advnb12eg, %adbc2b12eg : tensor<672xf32>
    %adlrb12eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb12eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb12eg = stablehlo.sqrt %advhb12eg : tensor<672xf32>
    %addenb12eg = stablehlo.add %adsqb12eg, %adepsb12eg : tensor<672xf32>
    %adratb12eg = stablehlo.divide %admhb12eg, %addenb12eg : tensor<672xf32>
    %adstb12eg = stablehlo.multiply %adlrb12eg, %adratb12eg : tensor<672xf32>
    %adsubb12eg = stablehlo.subtract %b12eg, %adstb12eg : tensor<672xf32>
    %adwdb12eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb12eg = stablehlo.multiply %adwdb12eg, %adlrb12eg : tensor<672xf32>
    %adwdpb12eg = stablehlo.multiply %adwdlrb12eg, %b12eg : tensor<672xf32>
    %adnewb12eg = stablehlo.subtract %adsubb12eg, %adwdpb12eg : tensor<672xf32>
    %adb1b12ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b12ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb12ebt = stablehlo.multiply %adb1b12ebt, %b12ebtm : tensor<672xf32>
    %admgb12ebt = stablehlo.multiply %adob1b12ebt, %b12dendb : tensor<672xf32>
    %admnb12ebt = stablehlo.add %admsb12ebt, %admgb12ebt : tensor<672xf32>
    %adb2b12ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b12ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb12ebt = stablehlo.multiply %adb2b12ebt, %b12ebtv : tensor<672xf32>
    %adg2b12ebt = stablehlo.multiply %b12dendb, %b12dendb : tensor<672xf32>
    %advgb12ebt = stablehlo.multiply %adob2b12ebt, %adg2b12ebt : tensor<672xf32>
    %advnb12ebt = stablehlo.add %advsb12ebt, %advgb12ebt : tensor<672xf32>
    %adbc1b12ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b12ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb12ebt = stablehlo.divide %admnb12ebt, %adbc1b12ebt : tensor<672xf32>
    %advhb12ebt = stablehlo.divide %advnb12ebt, %adbc2b12ebt : tensor<672xf32>
    %adlrb12ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb12ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb12ebt = stablehlo.sqrt %advhb12ebt : tensor<672xf32>
    %addenb12ebt = stablehlo.add %adsqb12ebt, %adepsb12ebt : tensor<672xf32>
    %adratb12ebt = stablehlo.divide %admhb12ebt, %addenb12ebt : tensor<672xf32>
    %adstb12ebt = stablehlo.multiply %adlrb12ebt, %adratb12ebt : tensor<672xf32>
    %adsubb12ebt = stablehlo.subtract %b12ebt, %adstb12ebt : tensor<672xf32>
    %adwdb12ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb12ebt = stablehlo.multiply %adwdb12ebt, %adlrb12ebt : tensor<672xf32>
    %adwdpb12ebt = stablehlo.multiply %adwdlrb12ebt, %b12ebt : tensor<672xf32>
    %adnewb12ebt = stablehlo.subtract %adsubb12ebt, %adwdpb12ebt : tensor<672xf32>
    %adb1b12dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adob1b12dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %admsb12dW = stablehlo.multiply %adb1b12dW, %b12dWm : tensor<672x1x5x5xf32>
    %admgb12dW = stablehlo.multiply %adob1b12dW, %b12ddW : tensor<672x1x5x5xf32>
    %admnb12dW = stablehlo.add %admsb12dW, %admgb12dW : tensor<672x1x5x5xf32>
    %adb2b12dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adob2b12dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %advsb12dW = stablehlo.multiply %adb2b12dW, %b12dWv : tensor<672x1x5x5xf32>
    %adg2b12dW = stablehlo.multiply %b12ddW, %b12ddW : tensor<672x1x5x5xf32>
    %advgb12dW = stablehlo.multiply %adob2b12dW, %adg2b12dW : tensor<672x1x5x5xf32>
    %advnb12dW = stablehlo.add %advsb12dW, %advgb12dW : tensor<672x1x5x5xf32>
    %adbc1b12dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adbc2b12dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %admhb12dW = stablehlo.divide %admnb12dW, %adbc1b12dW : tensor<672x1x5x5xf32>
    %advhb12dW = stablehlo.divide %advnb12dW, %adbc2b12dW : tensor<672x1x5x5xf32>
    %adlrb12dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adepsb12dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adsqb12dW = stablehlo.sqrt %advhb12dW : tensor<672x1x5x5xf32>
    %addenb12dW = stablehlo.add %adsqb12dW, %adepsb12dW : tensor<672x1x5x5xf32>
    %adratb12dW = stablehlo.divide %admhb12dW, %addenb12dW : tensor<672x1x5x5xf32>
    %adstb12dW = stablehlo.multiply %adlrb12dW, %adratb12dW : tensor<672x1x5x5xf32>
    %adsubb12dW = stablehlo.subtract %b12dW, %adstb12dW : tensor<672x1x5x5xf32>
    %adwdb12dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x1x5x5xf32>
    %adwdlrb12dW = stablehlo.multiply %adwdb12dW, %adlrb12dW : tensor<672x1x5x5xf32>
    %adwdpb12dW = stablehlo.multiply %adwdlrb12dW, %b12dW : tensor<672x1x5x5xf32>
    %adnewb12dW = stablehlo.subtract %adsubb12dW, %adwdpb12dW : tensor<672x1x5x5xf32>
    %adb1b12db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b12db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb12db = stablehlo.multiply %adb1b12db, %b12dbm : tensor<672xf32>
    %admgb12db = stablehlo.multiply %adob1b12db, %b12ddb : tensor<672xf32>
    %admnb12db = stablehlo.add %admsb12db, %admgb12db : tensor<672xf32>
    %adb2b12db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b12db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb12db = stablehlo.multiply %adb2b12db, %b12dbv : tensor<672xf32>
    %adg2b12db = stablehlo.multiply %b12ddb, %b12ddb : tensor<672xf32>
    %advgb12db = stablehlo.multiply %adob2b12db, %adg2b12db : tensor<672xf32>
    %advnb12db = stablehlo.add %advsb12db, %advgb12db : tensor<672xf32>
    %adbc1b12db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b12db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb12db = stablehlo.divide %admnb12db, %adbc1b12db : tensor<672xf32>
    %advhb12db = stablehlo.divide %advnb12db, %adbc2b12db : tensor<672xf32>
    %adlrb12db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb12db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb12db = stablehlo.sqrt %advhb12db : tensor<672xf32>
    %addenb12db = stablehlo.add %adsqb12db, %adepsb12db : tensor<672xf32>
    %adratb12db = stablehlo.divide %admhb12db, %addenb12db : tensor<672xf32>
    %adstb12db = stablehlo.multiply %adlrb12db, %adratb12db : tensor<672xf32>
    %adsubb12db = stablehlo.subtract %b12db, %adstb12db : tensor<672xf32>
    %adwdb12db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb12db = stablehlo.multiply %adwdb12db, %adlrb12db : tensor<672xf32>
    %adwdpb12db = stablehlo.multiply %adwdlrb12db, %b12db : tensor<672xf32>
    %adnewb12db = stablehlo.subtract %adsubb12db, %adwdpb12db : tensor<672xf32>
    %adb1b12dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b12dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb12dg = stablehlo.multiply %adb1b12dg, %b12dgm : tensor<672xf32>
    %admgb12dg = stablehlo.multiply %adob1b12dg, %b12ddndg : tensor<672xf32>
    %admnb12dg = stablehlo.add %admsb12dg, %admgb12dg : tensor<672xf32>
    %adb2b12dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b12dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb12dg = stablehlo.multiply %adb2b12dg, %b12dgv : tensor<672xf32>
    %adg2b12dg = stablehlo.multiply %b12ddndg, %b12ddndg : tensor<672xf32>
    %advgb12dg = stablehlo.multiply %adob2b12dg, %adg2b12dg : tensor<672xf32>
    %advnb12dg = stablehlo.add %advsb12dg, %advgb12dg : tensor<672xf32>
    %adbc1b12dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b12dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb12dg = stablehlo.divide %admnb12dg, %adbc1b12dg : tensor<672xf32>
    %advhb12dg = stablehlo.divide %advnb12dg, %adbc2b12dg : tensor<672xf32>
    %adlrb12dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb12dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb12dg = stablehlo.sqrt %advhb12dg : tensor<672xf32>
    %addenb12dg = stablehlo.add %adsqb12dg, %adepsb12dg : tensor<672xf32>
    %adratb12dg = stablehlo.divide %admhb12dg, %addenb12dg : tensor<672xf32>
    %adstb12dg = stablehlo.multiply %adlrb12dg, %adratb12dg : tensor<672xf32>
    %adsubb12dg = stablehlo.subtract %b12dg, %adstb12dg : tensor<672xf32>
    %adwdb12dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb12dg = stablehlo.multiply %adwdb12dg, %adlrb12dg : tensor<672xf32>
    %adwdpb12dg = stablehlo.multiply %adwdlrb12dg, %b12dg : tensor<672xf32>
    %adnewb12dg = stablehlo.subtract %adsubb12dg, %adwdpb12dg : tensor<672xf32>
    %adb1b12dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b12dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb12dbt = stablehlo.multiply %adb1b12dbt, %b12dbtm : tensor<672xf32>
    %admgb12dbt = stablehlo.multiply %adob1b12dbt, %b12ddndb : tensor<672xf32>
    %admnb12dbt = stablehlo.add %admsb12dbt, %admgb12dbt : tensor<672xf32>
    %adb2b12dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b12dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb12dbt = stablehlo.multiply %adb2b12dbt, %b12dbtv : tensor<672xf32>
    %adg2b12dbt = stablehlo.multiply %b12ddndb, %b12ddndb : tensor<672xf32>
    %advgb12dbt = stablehlo.multiply %adob2b12dbt, %adg2b12dbt : tensor<672xf32>
    %advnb12dbt = stablehlo.add %advsb12dbt, %advgb12dbt : tensor<672xf32>
    %adbc1b12dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b12dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb12dbt = stablehlo.divide %admnb12dbt, %adbc1b12dbt : tensor<672xf32>
    %advhb12dbt = stablehlo.divide %advnb12dbt, %adbc2b12dbt : tensor<672xf32>
    %adlrb12dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb12dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb12dbt = stablehlo.sqrt %advhb12dbt : tensor<672xf32>
    %addenb12dbt = stablehlo.add %adsqb12dbt, %adepsb12dbt : tensor<672xf32>
    %adratb12dbt = stablehlo.divide %admhb12dbt, %addenb12dbt : tensor<672xf32>
    %adstb12dbt = stablehlo.multiply %adlrb12dbt, %adratb12dbt : tensor<672xf32>
    %adsubb12dbt = stablehlo.subtract %b12dbt, %adstb12dbt : tensor<672xf32>
    %adwdb12dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb12dbt = stablehlo.multiply %adwdb12dbt, %adlrb12dbt : tensor<672xf32>
    %adwdpb12dbt = stablehlo.multiply %adwdlrb12dbt, %b12dbt : tensor<672xf32>
    %adnewb12dbt = stablehlo.subtract %adsubb12dbt, %adwdpb12dbt : tensor<672xf32>
    %adb1b12zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adob1b12zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %admsb12zW1 = stablehlo.multiply %adb1b12zW1, %b12zW1m : tensor<672x28xf32>
    %admgb12zW1 = stablehlo.multiply %adob1b12zW1, %b12zdWs1 : tensor<672x28xf32>
    %admnb12zW1 = stablehlo.add %admsb12zW1, %admgb12zW1 : tensor<672x28xf32>
    %adb2b12zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adob2b12zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %advsb12zW1 = stablehlo.multiply %adb2b12zW1, %b12zW1v : tensor<672x28xf32>
    %adg2b12zW1 = stablehlo.multiply %b12zdWs1, %b12zdWs1 : tensor<672x28xf32>
    %advgb12zW1 = stablehlo.multiply %adob2b12zW1, %adg2b12zW1 : tensor<672x28xf32>
    %advnb12zW1 = stablehlo.add %advsb12zW1, %advgb12zW1 : tensor<672x28xf32>
    %adbc1b12zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adbc2b12zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %admhb12zW1 = stablehlo.divide %admnb12zW1, %adbc1b12zW1 : tensor<672x28xf32>
    %advhb12zW1 = stablehlo.divide %advnb12zW1, %adbc2b12zW1 : tensor<672x28xf32>
    %adlrb12zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adepsb12zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adsqb12zW1 = stablehlo.sqrt %advhb12zW1 : tensor<672x28xf32>
    %addenb12zW1 = stablehlo.add %adsqb12zW1, %adepsb12zW1 : tensor<672x28xf32>
    %adratb12zW1 = stablehlo.divide %admhb12zW1, %addenb12zW1 : tensor<672x28xf32>
    %adstb12zW1 = stablehlo.multiply %adlrb12zW1, %adratb12zW1 : tensor<672x28xf32>
    %adsubb12zW1 = stablehlo.subtract %b12zW1, %adstb12zW1 : tensor<672x28xf32>
    %adwdb12zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672x28xf32>
    %adwdlrb12zW1 = stablehlo.multiply %adwdb12zW1, %adlrb12zW1 : tensor<672x28xf32>
    %adwdpb12zW1 = stablehlo.multiply %adwdlrb12zW1, %b12zW1 : tensor<672x28xf32>
    %adnewb12zW1 = stablehlo.subtract %adsubb12zW1, %adwdpb12zW1 : tensor<672x28xf32>
    %adb1b12zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adob1b12zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %admsb12zb1 = stablehlo.multiply %adb1b12zb1, %b12zb1m : tensor<28xf32>
    %admgb12zb1 = stablehlo.multiply %adob1b12zb1, %b12zdbs1 : tensor<28xf32>
    %admnb12zb1 = stablehlo.add %admsb12zb1, %admgb12zb1 : tensor<28xf32>
    %adb2b12zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adob2b12zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %advsb12zb1 = stablehlo.multiply %adb2b12zb1, %b12zb1v : tensor<28xf32>
    %adg2b12zb1 = stablehlo.multiply %b12zdbs1, %b12zdbs1 : tensor<28xf32>
    %advgb12zb1 = stablehlo.multiply %adob2b12zb1, %adg2b12zb1 : tensor<28xf32>
    %advnb12zb1 = stablehlo.add %advsb12zb1, %advgb12zb1 : tensor<28xf32>
    %adbc1b12zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adbc2b12zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %admhb12zb1 = stablehlo.divide %admnb12zb1, %adbc1b12zb1 : tensor<28xf32>
    %advhb12zb1 = stablehlo.divide %advnb12zb1, %adbc2b12zb1 : tensor<28xf32>
    %adlrb12zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adepsb12zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adsqb12zb1 = stablehlo.sqrt %advhb12zb1 : tensor<28xf32>
    %addenb12zb1 = stablehlo.add %adsqb12zb1, %adepsb12zb1 : tensor<28xf32>
    %adratb12zb1 = stablehlo.divide %admhb12zb1, %addenb12zb1 : tensor<28xf32>
    %adstb12zb1 = stablehlo.multiply %adlrb12zb1, %adratb12zb1 : tensor<28xf32>
    %adsubb12zb1 = stablehlo.subtract %b12zb1, %adstb12zb1 : tensor<28xf32>
    %adwdb12zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<28xf32>
    %adwdlrb12zb1 = stablehlo.multiply %adwdb12zb1, %adlrb12zb1 : tensor<28xf32>
    %adwdpb12zb1 = stablehlo.multiply %adwdlrb12zb1, %b12zb1 : tensor<28xf32>
    %adnewb12zb1 = stablehlo.subtract %adsubb12zb1, %adwdpb12zb1 : tensor<28xf32>
    %adb1b12zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adob1b12zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %admsb12zW2 = stablehlo.multiply %adb1b12zW2, %b12zW2m : tensor<28x672xf32>
    %admgb12zW2 = stablehlo.multiply %adob1b12zW2, %b12zdWs2 : tensor<28x672xf32>
    %admnb12zW2 = stablehlo.add %admsb12zW2, %admgb12zW2 : tensor<28x672xf32>
    %adb2b12zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adob2b12zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %advsb12zW2 = stablehlo.multiply %adb2b12zW2, %b12zW2v : tensor<28x672xf32>
    %adg2b12zW2 = stablehlo.multiply %b12zdWs2, %b12zdWs2 : tensor<28x672xf32>
    %advgb12zW2 = stablehlo.multiply %adob2b12zW2, %adg2b12zW2 : tensor<28x672xf32>
    %advnb12zW2 = stablehlo.add %advsb12zW2, %advgb12zW2 : tensor<28x672xf32>
    %adbc1b12zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adbc2b12zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %admhb12zW2 = stablehlo.divide %admnb12zW2, %adbc1b12zW2 : tensor<28x672xf32>
    %advhb12zW2 = stablehlo.divide %advnb12zW2, %adbc2b12zW2 : tensor<28x672xf32>
    %adlrb12zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adepsb12zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adsqb12zW2 = stablehlo.sqrt %advhb12zW2 : tensor<28x672xf32>
    %addenb12zW2 = stablehlo.add %adsqb12zW2, %adepsb12zW2 : tensor<28x672xf32>
    %adratb12zW2 = stablehlo.divide %admhb12zW2, %addenb12zW2 : tensor<28x672xf32>
    %adstb12zW2 = stablehlo.multiply %adlrb12zW2, %adratb12zW2 : tensor<28x672xf32>
    %adsubb12zW2 = stablehlo.subtract %b12zW2, %adstb12zW2 : tensor<28x672xf32>
    %adwdb12zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<28x672xf32>
    %adwdlrb12zW2 = stablehlo.multiply %adwdb12zW2, %adlrb12zW2 : tensor<28x672xf32>
    %adwdpb12zW2 = stablehlo.multiply %adwdlrb12zW2, %b12zW2 : tensor<28x672xf32>
    %adnewb12zW2 = stablehlo.subtract %adsubb12zW2, %adwdpb12zW2 : tensor<28x672xf32>
    %adb1b12zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob1b12zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admsb12zb2 = stablehlo.multiply %adb1b12zb2, %b12zb2m : tensor<672xf32>
    %admgb12zb2 = stablehlo.multiply %adob1b12zb2, %b12zdbs2 : tensor<672xf32>
    %admnb12zb2 = stablehlo.add %admsb12zb2, %admgb12zb2 : tensor<672xf32>
    %adb2b12zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adob2b12zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %advsb12zb2 = stablehlo.multiply %adb2b12zb2, %b12zb2v : tensor<672xf32>
    %adg2b12zb2 = stablehlo.multiply %b12zdbs2, %b12zdbs2 : tensor<672xf32>
    %advgb12zb2 = stablehlo.multiply %adob2b12zb2, %adg2b12zb2 : tensor<672xf32>
    %advnb12zb2 = stablehlo.add %advsb12zb2, %advgb12zb2 : tensor<672xf32>
    %adbc1b12zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adbc2b12zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %admhb12zb2 = stablehlo.divide %admnb12zb2, %adbc1b12zb2 : tensor<672xf32>
    %advhb12zb2 = stablehlo.divide %advnb12zb2, %adbc2b12zb2 : tensor<672xf32>
    %adlrb12zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adepsb12zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adsqb12zb2 = stablehlo.sqrt %advhb12zb2 : tensor<672xf32>
    %addenb12zb2 = stablehlo.add %adsqb12zb2, %adepsb12zb2 : tensor<672xf32>
    %adratb12zb2 = stablehlo.divide %admhb12zb2, %addenb12zb2 : tensor<672xf32>
    %adstb12zb2 = stablehlo.multiply %adlrb12zb2, %adratb12zb2 : tensor<672xf32>
    %adsubb12zb2 = stablehlo.subtract %b12zb2, %adstb12zb2 : tensor<672xf32>
    %adwdb12zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<672xf32>
    %adwdlrb12zb2 = stablehlo.multiply %adwdb12zb2, %adlrb12zb2 : tensor<672xf32>
    %adwdpb12zb2 = stablehlo.multiply %adwdlrb12zb2, %b12zb2 : tensor<672xf32>
    %adnewb12zb2 = stablehlo.subtract %adsubb12zb2, %adwdpb12zb2 : tensor<672xf32>
    %adb1b12pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %adob1b12pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %admsb12pW = stablehlo.multiply %adb1b12pW, %b12pWm : tensor<192x672x1x1xf32>
    %admgb12pW = stablehlo.multiply %adob1b12pW, %b12dpW : tensor<192x672x1x1xf32>
    %admnb12pW = stablehlo.add %admsb12pW, %admgb12pW : tensor<192x672x1x1xf32>
    %adb2b12pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %adob2b12pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %advsb12pW = stablehlo.multiply %adb2b12pW, %b12pWv : tensor<192x672x1x1xf32>
    %adg2b12pW = stablehlo.multiply %b12dpW, %b12dpW : tensor<192x672x1x1xf32>
    %advgb12pW = stablehlo.multiply %adob2b12pW, %adg2b12pW : tensor<192x672x1x1xf32>
    %advnb12pW = stablehlo.add %advsb12pW, %advgb12pW : tensor<192x672x1x1xf32>
    %adbc1b12pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %adbc2b12pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %admhb12pW = stablehlo.divide %admnb12pW, %adbc1b12pW : tensor<192x672x1x1xf32>
    %advhb12pW = stablehlo.divide %advnb12pW, %adbc2b12pW : tensor<192x672x1x1xf32>
    %adlrb12pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %adepsb12pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %adsqb12pW = stablehlo.sqrt %advhb12pW : tensor<192x672x1x1xf32>
    %addenb12pW = stablehlo.add %adsqb12pW, %adepsb12pW : tensor<192x672x1x1xf32>
    %adratb12pW = stablehlo.divide %admhb12pW, %addenb12pW : tensor<192x672x1x1xf32>
    %adstb12pW = stablehlo.multiply %adlrb12pW, %adratb12pW : tensor<192x672x1x1xf32>
    %adsubb12pW = stablehlo.subtract %b12pW, %adstb12pW : tensor<192x672x1x1xf32>
    %adwdb12pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x672x1x1xf32>
    %adwdlrb12pW = stablehlo.multiply %adwdb12pW, %adlrb12pW : tensor<192x672x1x1xf32>
    %adwdpb12pW = stablehlo.multiply %adwdlrb12pW, %b12pW : tensor<192x672x1x1xf32>
    %adnewb12pW = stablehlo.subtract %adsubb12pW, %adwdpb12pW : tensor<192x672x1x1xf32>
    %adb1b12pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b12pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb12pb = stablehlo.multiply %adb1b12pb, %b12pbm : tensor<192xf32>
    %admgb12pb = stablehlo.multiply %adob1b12pb, %b12dpb : tensor<192xf32>
    %admnb12pb = stablehlo.add %admsb12pb, %admgb12pb : tensor<192xf32>
    %adb2b12pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b12pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb12pb = stablehlo.multiply %adb2b12pb, %b12pbv : tensor<192xf32>
    %adg2b12pb = stablehlo.multiply %b12dpb, %b12dpb : tensor<192xf32>
    %advgb12pb = stablehlo.multiply %adob2b12pb, %adg2b12pb : tensor<192xf32>
    %advnb12pb = stablehlo.add %advsb12pb, %advgb12pb : tensor<192xf32>
    %adbc1b12pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b12pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb12pb = stablehlo.divide %admnb12pb, %adbc1b12pb : tensor<192xf32>
    %advhb12pb = stablehlo.divide %advnb12pb, %adbc2b12pb : tensor<192xf32>
    %adlrb12pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb12pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb12pb = stablehlo.sqrt %advhb12pb : tensor<192xf32>
    %addenb12pb = stablehlo.add %adsqb12pb, %adepsb12pb : tensor<192xf32>
    %adratb12pb = stablehlo.divide %admhb12pb, %addenb12pb : tensor<192xf32>
    %adstb12pb = stablehlo.multiply %adlrb12pb, %adratb12pb : tensor<192xf32>
    %adsubb12pb = stablehlo.subtract %b12pb, %adstb12pb : tensor<192xf32>
    %adwdb12pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb12pb = stablehlo.multiply %adwdb12pb, %adlrb12pb : tensor<192xf32>
    %adwdpb12pb = stablehlo.multiply %adwdlrb12pb, %b12pb : tensor<192xf32>
    %adnewb12pb = stablehlo.subtract %adsubb12pb, %adwdpb12pb : tensor<192xf32>
    %adb1b12pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b12pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb12pg = stablehlo.multiply %adb1b12pg, %b12pgm : tensor<192xf32>
    %admgb12pg = stablehlo.multiply %adob1b12pg, %b12dpndg : tensor<192xf32>
    %admnb12pg = stablehlo.add %admsb12pg, %admgb12pg : tensor<192xf32>
    %adb2b12pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b12pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb12pg = stablehlo.multiply %adb2b12pg, %b12pgv : tensor<192xf32>
    %adg2b12pg = stablehlo.multiply %b12dpndg, %b12dpndg : tensor<192xf32>
    %advgb12pg = stablehlo.multiply %adob2b12pg, %adg2b12pg : tensor<192xf32>
    %advnb12pg = stablehlo.add %advsb12pg, %advgb12pg : tensor<192xf32>
    %adbc1b12pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b12pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb12pg = stablehlo.divide %admnb12pg, %adbc1b12pg : tensor<192xf32>
    %advhb12pg = stablehlo.divide %advnb12pg, %adbc2b12pg : tensor<192xf32>
    %adlrb12pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb12pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb12pg = stablehlo.sqrt %advhb12pg : tensor<192xf32>
    %addenb12pg = stablehlo.add %adsqb12pg, %adepsb12pg : tensor<192xf32>
    %adratb12pg = stablehlo.divide %admhb12pg, %addenb12pg : tensor<192xf32>
    %adstb12pg = stablehlo.multiply %adlrb12pg, %adratb12pg : tensor<192xf32>
    %adsubb12pg = stablehlo.subtract %b12pg, %adstb12pg : tensor<192xf32>
    %adwdb12pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb12pg = stablehlo.multiply %adwdb12pg, %adlrb12pg : tensor<192xf32>
    %adwdpb12pg = stablehlo.multiply %adwdlrb12pg, %b12pg : tensor<192xf32>
    %adnewb12pg = stablehlo.subtract %adsubb12pg, %adwdpb12pg : tensor<192xf32>
    %adb1b12pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b12pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb12pbt = stablehlo.multiply %adb1b12pbt, %b12pbtm : tensor<192xf32>
    %admgb12pbt = stablehlo.multiply %adob1b12pbt, %b12dpndb : tensor<192xf32>
    %admnb12pbt = stablehlo.add %admsb12pbt, %admgb12pbt : tensor<192xf32>
    %adb2b12pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b12pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb12pbt = stablehlo.multiply %adb2b12pbt, %b12pbtv : tensor<192xf32>
    %adg2b12pbt = stablehlo.multiply %b12dpndb, %b12dpndb : tensor<192xf32>
    %advgb12pbt = stablehlo.multiply %adob2b12pbt, %adg2b12pbt : tensor<192xf32>
    %advnb12pbt = stablehlo.add %advsb12pbt, %advgb12pbt : tensor<192xf32>
    %adbc1b12pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b12pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb12pbt = stablehlo.divide %admnb12pbt, %adbc1b12pbt : tensor<192xf32>
    %advhb12pbt = stablehlo.divide %advnb12pbt, %adbc2b12pbt : tensor<192xf32>
    %adlrb12pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb12pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb12pbt = stablehlo.sqrt %advhb12pbt : tensor<192xf32>
    %addenb12pbt = stablehlo.add %adsqb12pbt, %adepsb12pbt : tensor<192xf32>
    %adratb12pbt = stablehlo.divide %admhb12pbt, %addenb12pbt : tensor<192xf32>
    %adstb12pbt = stablehlo.multiply %adlrb12pbt, %adratb12pbt : tensor<192xf32>
    %adsubb12pbt = stablehlo.subtract %b12pbt, %adstb12pbt : tensor<192xf32>
    %adwdb12pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb12pbt = stablehlo.multiply %adwdb12pbt, %adlrb12pbt : tensor<192xf32>
    %adwdpb12pbt = stablehlo.multiply %adwdlrb12pbt, %b12pbt : tensor<192xf32>
    %adnewb12pbt = stablehlo.subtract %adsubb12pbt, %adwdpb12pbt : tensor<192xf32>
    %adb1b13eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adob1b13eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %admsb13eW = stablehlo.multiply %adb1b13eW, %b13eWm : tensor<1152x192x1x1xf32>
    %admgb13eW = stablehlo.multiply %adob1b13eW, %b13deW : tensor<1152x192x1x1xf32>
    %admnb13eW = stablehlo.add %admsb13eW, %admgb13eW : tensor<1152x192x1x1xf32>
    %adb2b13eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adob2b13eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %advsb13eW = stablehlo.multiply %adb2b13eW, %b13eWv : tensor<1152x192x1x1xf32>
    %adg2b13eW = stablehlo.multiply %b13deW, %b13deW : tensor<1152x192x1x1xf32>
    %advgb13eW = stablehlo.multiply %adob2b13eW, %adg2b13eW : tensor<1152x192x1x1xf32>
    %advnb13eW = stablehlo.add %advsb13eW, %advgb13eW : tensor<1152x192x1x1xf32>
    %adbc1b13eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adbc2b13eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %admhb13eW = stablehlo.divide %admnb13eW, %adbc1b13eW : tensor<1152x192x1x1xf32>
    %advhb13eW = stablehlo.divide %advnb13eW, %adbc2b13eW : tensor<1152x192x1x1xf32>
    %adlrb13eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adepsb13eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adsqb13eW = stablehlo.sqrt %advhb13eW : tensor<1152x192x1x1xf32>
    %addenb13eW = stablehlo.add %adsqb13eW, %adepsb13eW : tensor<1152x192x1x1xf32>
    %adratb13eW = stablehlo.divide %admhb13eW, %addenb13eW : tensor<1152x192x1x1xf32>
    %adstb13eW = stablehlo.multiply %adlrb13eW, %adratb13eW : tensor<1152x192x1x1xf32>
    %adsubb13eW = stablehlo.subtract %b13eW, %adstb13eW : tensor<1152x192x1x1xf32>
    %adwdb13eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adwdlrb13eW = stablehlo.multiply %adwdb13eW, %adlrb13eW : tensor<1152x192x1x1xf32>
    %adwdpb13eW = stablehlo.multiply %adwdlrb13eW, %b13eW : tensor<1152x192x1x1xf32>
    %adnewb13eW = stablehlo.subtract %adsubb13eW, %adwdpb13eW : tensor<1152x192x1x1xf32>
    %adb1b13eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b13eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb13eb = stablehlo.multiply %adb1b13eb, %b13ebm : tensor<1152xf32>
    %admgb13eb = stablehlo.multiply %adob1b13eb, %b13deb : tensor<1152xf32>
    %admnb13eb = stablehlo.add %admsb13eb, %admgb13eb : tensor<1152xf32>
    %adb2b13eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b13eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb13eb = stablehlo.multiply %adb2b13eb, %b13ebv : tensor<1152xf32>
    %adg2b13eb = stablehlo.multiply %b13deb, %b13deb : tensor<1152xf32>
    %advgb13eb = stablehlo.multiply %adob2b13eb, %adg2b13eb : tensor<1152xf32>
    %advnb13eb = stablehlo.add %advsb13eb, %advgb13eb : tensor<1152xf32>
    %adbc1b13eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b13eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb13eb = stablehlo.divide %admnb13eb, %adbc1b13eb : tensor<1152xf32>
    %advhb13eb = stablehlo.divide %advnb13eb, %adbc2b13eb : tensor<1152xf32>
    %adlrb13eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb13eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb13eb = stablehlo.sqrt %advhb13eb : tensor<1152xf32>
    %addenb13eb = stablehlo.add %adsqb13eb, %adepsb13eb : tensor<1152xf32>
    %adratb13eb = stablehlo.divide %admhb13eb, %addenb13eb : tensor<1152xf32>
    %adstb13eb = stablehlo.multiply %adlrb13eb, %adratb13eb : tensor<1152xf32>
    %adsubb13eb = stablehlo.subtract %b13eb, %adstb13eb : tensor<1152xf32>
    %adwdb13eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb13eb = stablehlo.multiply %adwdb13eb, %adlrb13eb : tensor<1152xf32>
    %adwdpb13eb = stablehlo.multiply %adwdlrb13eb, %b13eb : tensor<1152xf32>
    %adnewb13eb = stablehlo.subtract %adsubb13eb, %adwdpb13eb : tensor<1152xf32>
    %adb1b13eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b13eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb13eg = stablehlo.multiply %adb1b13eg, %b13egm : tensor<1152xf32>
    %admgb13eg = stablehlo.multiply %adob1b13eg, %b13dendg : tensor<1152xf32>
    %admnb13eg = stablehlo.add %admsb13eg, %admgb13eg : tensor<1152xf32>
    %adb2b13eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b13eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb13eg = stablehlo.multiply %adb2b13eg, %b13egv : tensor<1152xf32>
    %adg2b13eg = stablehlo.multiply %b13dendg, %b13dendg : tensor<1152xf32>
    %advgb13eg = stablehlo.multiply %adob2b13eg, %adg2b13eg : tensor<1152xf32>
    %advnb13eg = stablehlo.add %advsb13eg, %advgb13eg : tensor<1152xf32>
    %adbc1b13eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b13eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb13eg = stablehlo.divide %admnb13eg, %adbc1b13eg : tensor<1152xf32>
    %advhb13eg = stablehlo.divide %advnb13eg, %adbc2b13eg : tensor<1152xf32>
    %adlrb13eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb13eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb13eg = stablehlo.sqrt %advhb13eg : tensor<1152xf32>
    %addenb13eg = stablehlo.add %adsqb13eg, %adepsb13eg : tensor<1152xf32>
    %adratb13eg = stablehlo.divide %admhb13eg, %addenb13eg : tensor<1152xf32>
    %adstb13eg = stablehlo.multiply %adlrb13eg, %adratb13eg : tensor<1152xf32>
    %adsubb13eg = stablehlo.subtract %b13eg, %adstb13eg : tensor<1152xf32>
    %adwdb13eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb13eg = stablehlo.multiply %adwdb13eg, %adlrb13eg : tensor<1152xf32>
    %adwdpb13eg = stablehlo.multiply %adwdlrb13eg, %b13eg : tensor<1152xf32>
    %adnewb13eg = stablehlo.subtract %adsubb13eg, %adwdpb13eg : tensor<1152xf32>
    %adb1b13ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b13ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb13ebt = stablehlo.multiply %adb1b13ebt, %b13ebtm : tensor<1152xf32>
    %admgb13ebt = stablehlo.multiply %adob1b13ebt, %b13dendb : tensor<1152xf32>
    %admnb13ebt = stablehlo.add %admsb13ebt, %admgb13ebt : tensor<1152xf32>
    %adb2b13ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b13ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb13ebt = stablehlo.multiply %adb2b13ebt, %b13ebtv : tensor<1152xf32>
    %adg2b13ebt = stablehlo.multiply %b13dendb, %b13dendb : tensor<1152xf32>
    %advgb13ebt = stablehlo.multiply %adob2b13ebt, %adg2b13ebt : tensor<1152xf32>
    %advnb13ebt = stablehlo.add %advsb13ebt, %advgb13ebt : tensor<1152xf32>
    %adbc1b13ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b13ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb13ebt = stablehlo.divide %admnb13ebt, %adbc1b13ebt : tensor<1152xf32>
    %advhb13ebt = stablehlo.divide %advnb13ebt, %adbc2b13ebt : tensor<1152xf32>
    %adlrb13ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb13ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb13ebt = stablehlo.sqrt %advhb13ebt : tensor<1152xf32>
    %addenb13ebt = stablehlo.add %adsqb13ebt, %adepsb13ebt : tensor<1152xf32>
    %adratb13ebt = stablehlo.divide %admhb13ebt, %addenb13ebt : tensor<1152xf32>
    %adstb13ebt = stablehlo.multiply %adlrb13ebt, %adratb13ebt : tensor<1152xf32>
    %adsubb13ebt = stablehlo.subtract %b13ebt, %adstb13ebt : tensor<1152xf32>
    %adwdb13ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb13ebt = stablehlo.multiply %adwdb13ebt, %adlrb13ebt : tensor<1152xf32>
    %adwdpb13ebt = stablehlo.multiply %adwdlrb13ebt, %b13ebt : tensor<1152xf32>
    %adnewb13ebt = stablehlo.subtract %adsubb13ebt, %adwdpb13ebt : tensor<1152xf32>
    %adb1b13dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adob1b13dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %admsb13dW = stablehlo.multiply %adb1b13dW, %b13dWm : tensor<1152x1x5x5xf32>
    %admgb13dW = stablehlo.multiply %adob1b13dW, %b13ddW : tensor<1152x1x5x5xf32>
    %admnb13dW = stablehlo.add %admsb13dW, %admgb13dW : tensor<1152x1x5x5xf32>
    %adb2b13dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adob2b13dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %advsb13dW = stablehlo.multiply %adb2b13dW, %b13dWv : tensor<1152x1x5x5xf32>
    %adg2b13dW = stablehlo.multiply %b13ddW, %b13ddW : tensor<1152x1x5x5xf32>
    %advgb13dW = stablehlo.multiply %adob2b13dW, %adg2b13dW : tensor<1152x1x5x5xf32>
    %advnb13dW = stablehlo.add %advsb13dW, %advgb13dW : tensor<1152x1x5x5xf32>
    %adbc1b13dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adbc2b13dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %admhb13dW = stablehlo.divide %admnb13dW, %adbc1b13dW : tensor<1152x1x5x5xf32>
    %advhb13dW = stablehlo.divide %advnb13dW, %adbc2b13dW : tensor<1152x1x5x5xf32>
    %adlrb13dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adepsb13dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adsqb13dW = stablehlo.sqrt %advhb13dW : tensor<1152x1x5x5xf32>
    %addenb13dW = stablehlo.add %adsqb13dW, %adepsb13dW : tensor<1152x1x5x5xf32>
    %adratb13dW = stablehlo.divide %admhb13dW, %addenb13dW : tensor<1152x1x5x5xf32>
    %adstb13dW = stablehlo.multiply %adlrb13dW, %adratb13dW : tensor<1152x1x5x5xf32>
    %adsubb13dW = stablehlo.subtract %b13dW, %adstb13dW : tensor<1152x1x5x5xf32>
    %adwdb13dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adwdlrb13dW = stablehlo.multiply %adwdb13dW, %adlrb13dW : tensor<1152x1x5x5xf32>
    %adwdpb13dW = stablehlo.multiply %adwdlrb13dW, %b13dW : tensor<1152x1x5x5xf32>
    %adnewb13dW = stablehlo.subtract %adsubb13dW, %adwdpb13dW : tensor<1152x1x5x5xf32>
    %adb1b13db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b13db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb13db = stablehlo.multiply %adb1b13db, %b13dbm : tensor<1152xf32>
    %admgb13db = stablehlo.multiply %adob1b13db, %b13ddb : tensor<1152xf32>
    %admnb13db = stablehlo.add %admsb13db, %admgb13db : tensor<1152xf32>
    %adb2b13db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b13db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb13db = stablehlo.multiply %adb2b13db, %b13dbv : tensor<1152xf32>
    %adg2b13db = stablehlo.multiply %b13ddb, %b13ddb : tensor<1152xf32>
    %advgb13db = stablehlo.multiply %adob2b13db, %adg2b13db : tensor<1152xf32>
    %advnb13db = stablehlo.add %advsb13db, %advgb13db : tensor<1152xf32>
    %adbc1b13db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b13db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb13db = stablehlo.divide %admnb13db, %adbc1b13db : tensor<1152xf32>
    %advhb13db = stablehlo.divide %advnb13db, %adbc2b13db : tensor<1152xf32>
    %adlrb13db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb13db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb13db = stablehlo.sqrt %advhb13db : tensor<1152xf32>
    %addenb13db = stablehlo.add %adsqb13db, %adepsb13db : tensor<1152xf32>
    %adratb13db = stablehlo.divide %admhb13db, %addenb13db : tensor<1152xf32>
    %adstb13db = stablehlo.multiply %adlrb13db, %adratb13db : tensor<1152xf32>
    %adsubb13db = stablehlo.subtract %b13db, %adstb13db : tensor<1152xf32>
    %adwdb13db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb13db = stablehlo.multiply %adwdb13db, %adlrb13db : tensor<1152xf32>
    %adwdpb13db = stablehlo.multiply %adwdlrb13db, %b13db : tensor<1152xf32>
    %adnewb13db = stablehlo.subtract %adsubb13db, %adwdpb13db : tensor<1152xf32>
    %adb1b13dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b13dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb13dg = stablehlo.multiply %adb1b13dg, %b13dgm : tensor<1152xf32>
    %admgb13dg = stablehlo.multiply %adob1b13dg, %b13ddndg : tensor<1152xf32>
    %admnb13dg = stablehlo.add %admsb13dg, %admgb13dg : tensor<1152xf32>
    %adb2b13dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b13dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb13dg = stablehlo.multiply %adb2b13dg, %b13dgv : tensor<1152xf32>
    %adg2b13dg = stablehlo.multiply %b13ddndg, %b13ddndg : tensor<1152xf32>
    %advgb13dg = stablehlo.multiply %adob2b13dg, %adg2b13dg : tensor<1152xf32>
    %advnb13dg = stablehlo.add %advsb13dg, %advgb13dg : tensor<1152xf32>
    %adbc1b13dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b13dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb13dg = stablehlo.divide %admnb13dg, %adbc1b13dg : tensor<1152xf32>
    %advhb13dg = stablehlo.divide %advnb13dg, %adbc2b13dg : tensor<1152xf32>
    %adlrb13dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb13dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb13dg = stablehlo.sqrt %advhb13dg : tensor<1152xf32>
    %addenb13dg = stablehlo.add %adsqb13dg, %adepsb13dg : tensor<1152xf32>
    %adratb13dg = stablehlo.divide %admhb13dg, %addenb13dg : tensor<1152xf32>
    %adstb13dg = stablehlo.multiply %adlrb13dg, %adratb13dg : tensor<1152xf32>
    %adsubb13dg = stablehlo.subtract %b13dg, %adstb13dg : tensor<1152xf32>
    %adwdb13dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb13dg = stablehlo.multiply %adwdb13dg, %adlrb13dg : tensor<1152xf32>
    %adwdpb13dg = stablehlo.multiply %adwdlrb13dg, %b13dg : tensor<1152xf32>
    %adnewb13dg = stablehlo.subtract %adsubb13dg, %adwdpb13dg : tensor<1152xf32>
    %adb1b13dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b13dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb13dbt = stablehlo.multiply %adb1b13dbt, %b13dbtm : tensor<1152xf32>
    %admgb13dbt = stablehlo.multiply %adob1b13dbt, %b13ddndb : tensor<1152xf32>
    %admnb13dbt = stablehlo.add %admsb13dbt, %admgb13dbt : tensor<1152xf32>
    %adb2b13dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b13dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb13dbt = stablehlo.multiply %adb2b13dbt, %b13dbtv : tensor<1152xf32>
    %adg2b13dbt = stablehlo.multiply %b13ddndb, %b13ddndb : tensor<1152xf32>
    %advgb13dbt = stablehlo.multiply %adob2b13dbt, %adg2b13dbt : tensor<1152xf32>
    %advnb13dbt = stablehlo.add %advsb13dbt, %advgb13dbt : tensor<1152xf32>
    %adbc1b13dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b13dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb13dbt = stablehlo.divide %admnb13dbt, %adbc1b13dbt : tensor<1152xf32>
    %advhb13dbt = stablehlo.divide %advnb13dbt, %adbc2b13dbt : tensor<1152xf32>
    %adlrb13dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb13dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb13dbt = stablehlo.sqrt %advhb13dbt : tensor<1152xf32>
    %addenb13dbt = stablehlo.add %adsqb13dbt, %adepsb13dbt : tensor<1152xf32>
    %adratb13dbt = stablehlo.divide %admhb13dbt, %addenb13dbt : tensor<1152xf32>
    %adstb13dbt = stablehlo.multiply %adlrb13dbt, %adratb13dbt : tensor<1152xf32>
    %adsubb13dbt = stablehlo.subtract %b13dbt, %adstb13dbt : tensor<1152xf32>
    %adwdb13dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb13dbt = stablehlo.multiply %adwdb13dbt, %adlrb13dbt : tensor<1152xf32>
    %adwdpb13dbt = stablehlo.multiply %adwdlrb13dbt, %b13dbt : tensor<1152xf32>
    %adnewb13dbt = stablehlo.subtract %adsubb13dbt, %adwdpb13dbt : tensor<1152xf32>
    %adb1b13zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adob1b13zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %admsb13zW1 = stablehlo.multiply %adb1b13zW1, %b13zW1m : tensor<1152x48xf32>
    %admgb13zW1 = stablehlo.multiply %adob1b13zW1, %b13zdWs1 : tensor<1152x48xf32>
    %admnb13zW1 = stablehlo.add %admsb13zW1, %admgb13zW1 : tensor<1152x48xf32>
    %adb2b13zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adob2b13zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %advsb13zW1 = stablehlo.multiply %adb2b13zW1, %b13zW1v : tensor<1152x48xf32>
    %adg2b13zW1 = stablehlo.multiply %b13zdWs1, %b13zdWs1 : tensor<1152x48xf32>
    %advgb13zW1 = stablehlo.multiply %adob2b13zW1, %adg2b13zW1 : tensor<1152x48xf32>
    %advnb13zW1 = stablehlo.add %advsb13zW1, %advgb13zW1 : tensor<1152x48xf32>
    %adbc1b13zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adbc2b13zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %admhb13zW1 = stablehlo.divide %admnb13zW1, %adbc1b13zW1 : tensor<1152x48xf32>
    %advhb13zW1 = stablehlo.divide %advnb13zW1, %adbc2b13zW1 : tensor<1152x48xf32>
    %adlrb13zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adepsb13zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adsqb13zW1 = stablehlo.sqrt %advhb13zW1 : tensor<1152x48xf32>
    %addenb13zW1 = stablehlo.add %adsqb13zW1, %adepsb13zW1 : tensor<1152x48xf32>
    %adratb13zW1 = stablehlo.divide %admhb13zW1, %addenb13zW1 : tensor<1152x48xf32>
    %adstb13zW1 = stablehlo.multiply %adlrb13zW1, %adratb13zW1 : tensor<1152x48xf32>
    %adsubb13zW1 = stablehlo.subtract %b13zW1, %adstb13zW1 : tensor<1152x48xf32>
    %adwdb13zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adwdlrb13zW1 = stablehlo.multiply %adwdb13zW1, %adlrb13zW1 : tensor<1152x48xf32>
    %adwdpb13zW1 = stablehlo.multiply %adwdlrb13zW1, %b13zW1 : tensor<1152x48xf32>
    %adnewb13zW1 = stablehlo.subtract %adsubb13zW1, %adwdpb13zW1 : tensor<1152x48xf32>
    %adb1b13zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adob1b13zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %admsb13zb1 = stablehlo.multiply %adb1b13zb1, %b13zb1m : tensor<48xf32>
    %admgb13zb1 = stablehlo.multiply %adob1b13zb1, %b13zdbs1 : tensor<48xf32>
    %admnb13zb1 = stablehlo.add %admsb13zb1, %admgb13zb1 : tensor<48xf32>
    %adb2b13zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adob2b13zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %advsb13zb1 = stablehlo.multiply %adb2b13zb1, %b13zb1v : tensor<48xf32>
    %adg2b13zb1 = stablehlo.multiply %b13zdbs1, %b13zdbs1 : tensor<48xf32>
    %advgb13zb1 = stablehlo.multiply %adob2b13zb1, %adg2b13zb1 : tensor<48xf32>
    %advnb13zb1 = stablehlo.add %advsb13zb1, %advgb13zb1 : tensor<48xf32>
    %adbc1b13zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adbc2b13zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %admhb13zb1 = stablehlo.divide %admnb13zb1, %adbc1b13zb1 : tensor<48xf32>
    %advhb13zb1 = stablehlo.divide %advnb13zb1, %adbc2b13zb1 : tensor<48xf32>
    %adlrb13zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adepsb13zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adsqb13zb1 = stablehlo.sqrt %advhb13zb1 : tensor<48xf32>
    %addenb13zb1 = stablehlo.add %adsqb13zb1, %adepsb13zb1 : tensor<48xf32>
    %adratb13zb1 = stablehlo.divide %admhb13zb1, %addenb13zb1 : tensor<48xf32>
    %adstb13zb1 = stablehlo.multiply %adlrb13zb1, %adratb13zb1 : tensor<48xf32>
    %adsubb13zb1 = stablehlo.subtract %b13zb1, %adstb13zb1 : tensor<48xf32>
    %adwdb13zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adwdlrb13zb1 = stablehlo.multiply %adwdb13zb1, %adlrb13zb1 : tensor<48xf32>
    %adwdpb13zb1 = stablehlo.multiply %adwdlrb13zb1, %b13zb1 : tensor<48xf32>
    %adnewb13zb1 = stablehlo.subtract %adsubb13zb1, %adwdpb13zb1 : tensor<48xf32>
    %adb1b13zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adob1b13zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %admsb13zW2 = stablehlo.multiply %adb1b13zW2, %b13zW2m : tensor<48x1152xf32>
    %admgb13zW2 = stablehlo.multiply %adob1b13zW2, %b13zdWs2 : tensor<48x1152xf32>
    %admnb13zW2 = stablehlo.add %admsb13zW2, %admgb13zW2 : tensor<48x1152xf32>
    %adb2b13zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adob2b13zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %advsb13zW2 = stablehlo.multiply %adb2b13zW2, %b13zW2v : tensor<48x1152xf32>
    %adg2b13zW2 = stablehlo.multiply %b13zdWs2, %b13zdWs2 : tensor<48x1152xf32>
    %advgb13zW2 = stablehlo.multiply %adob2b13zW2, %adg2b13zW2 : tensor<48x1152xf32>
    %advnb13zW2 = stablehlo.add %advsb13zW2, %advgb13zW2 : tensor<48x1152xf32>
    %adbc1b13zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adbc2b13zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %admhb13zW2 = stablehlo.divide %admnb13zW2, %adbc1b13zW2 : tensor<48x1152xf32>
    %advhb13zW2 = stablehlo.divide %advnb13zW2, %adbc2b13zW2 : tensor<48x1152xf32>
    %adlrb13zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adepsb13zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adsqb13zW2 = stablehlo.sqrt %advhb13zW2 : tensor<48x1152xf32>
    %addenb13zW2 = stablehlo.add %adsqb13zW2, %adepsb13zW2 : tensor<48x1152xf32>
    %adratb13zW2 = stablehlo.divide %admhb13zW2, %addenb13zW2 : tensor<48x1152xf32>
    %adstb13zW2 = stablehlo.multiply %adlrb13zW2, %adratb13zW2 : tensor<48x1152xf32>
    %adsubb13zW2 = stablehlo.subtract %b13zW2, %adstb13zW2 : tensor<48x1152xf32>
    %adwdb13zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adwdlrb13zW2 = stablehlo.multiply %adwdb13zW2, %adlrb13zW2 : tensor<48x1152xf32>
    %adwdpb13zW2 = stablehlo.multiply %adwdlrb13zW2, %b13zW2 : tensor<48x1152xf32>
    %adnewb13zW2 = stablehlo.subtract %adsubb13zW2, %adwdpb13zW2 : tensor<48x1152xf32>
    %adb1b13zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b13zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb13zb2 = stablehlo.multiply %adb1b13zb2, %b13zb2m : tensor<1152xf32>
    %admgb13zb2 = stablehlo.multiply %adob1b13zb2, %b13zdbs2 : tensor<1152xf32>
    %admnb13zb2 = stablehlo.add %admsb13zb2, %admgb13zb2 : tensor<1152xf32>
    %adb2b13zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b13zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb13zb2 = stablehlo.multiply %adb2b13zb2, %b13zb2v : tensor<1152xf32>
    %adg2b13zb2 = stablehlo.multiply %b13zdbs2, %b13zdbs2 : tensor<1152xf32>
    %advgb13zb2 = stablehlo.multiply %adob2b13zb2, %adg2b13zb2 : tensor<1152xf32>
    %advnb13zb2 = stablehlo.add %advsb13zb2, %advgb13zb2 : tensor<1152xf32>
    %adbc1b13zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b13zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb13zb2 = stablehlo.divide %admnb13zb2, %adbc1b13zb2 : tensor<1152xf32>
    %advhb13zb2 = stablehlo.divide %advnb13zb2, %adbc2b13zb2 : tensor<1152xf32>
    %adlrb13zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb13zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb13zb2 = stablehlo.sqrt %advhb13zb2 : tensor<1152xf32>
    %addenb13zb2 = stablehlo.add %adsqb13zb2, %adepsb13zb2 : tensor<1152xf32>
    %adratb13zb2 = stablehlo.divide %admhb13zb2, %addenb13zb2 : tensor<1152xf32>
    %adstb13zb2 = stablehlo.multiply %adlrb13zb2, %adratb13zb2 : tensor<1152xf32>
    %adsubb13zb2 = stablehlo.subtract %b13zb2, %adstb13zb2 : tensor<1152xf32>
    %adwdb13zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb13zb2 = stablehlo.multiply %adwdb13zb2, %adlrb13zb2 : tensor<1152xf32>
    %adwdpb13zb2 = stablehlo.multiply %adwdlrb13zb2, %b13zb2 : tensor<1152xf32>
    %adnewb13zb2 = stablehlo.subtract %adsubb13zb2, %adwdpb13zb2 : tensor<1152xf32>
    %adb1b13pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adob1b13pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %admsb13pW = stablehlo.multiply %adb1b13pW, %b13pWm : tensor<192x1152x1x1xf32>
    %admgb13pW = stablehlo.multiply %adob1b13pW, %b13dpW : tensor<192x1152x1x1xf32>
    %admnb13pW = stablehlo.add %admsb13pW, %admgb13pW : tensor<192x1152x1x1xf32>
    %adb2b13pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adob2b13pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %advsb13pW = stablehlo.multiply %adb2b13pW, %b13pWv : tensor<192x1152x1x1xf32>
    %adg2b13pW = stablehlo.multiply %b13dpW, %b13dpW : tensor<192x1152x1x1xf32>
    %advgb13pW = stablehlo.multiply %adob2b13pW, %adg2b13pW : tensor<192x1152x1x1xf32>
    %advnb13pW = stablehlo.add %advsb13pW, %advgb13pW : tensor<192x1152x1x1xf32>
    %adbc1b13pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adbc2b13pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %admhb13pW = stablehlo.divide %admnb13pW, %adbc1b13pW : tensor<192x1152x1x1xf32>
    %advhb13pW = stablehlo.divide %advnb13pW, %adbc2b13pW : tensor<192x1152x1x1xf32>
    %adlrb13pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adepsb13pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adsqb13pW = stablehlo.sqrt %advhb13pW : tensor<192x1152x1x1xf32>
    %addenb13pW = stablehlo.add %adsqb13pW, %adepsb13pW : tensor<192x1152x1x1xf32>
    %adratb13pW = stablehlo.divide %admhb13pW, %addenb13pW : tensor<192x1152x1x1xf32>
    %adstb13pW = stablehlo.multiply %adlrb13pW, %adratb13pW : tensor<192x1152x1x1xf32>
    %adsubb13pW = stablehlo.subtract %b13pW, %adstb13pW : tensor<192x1152x1x1xf32>
    %adwdb13pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adwdlrb13pW = stablehlo.multiply %adwdb13pW, %adlrb13pW : tensor<192x1152x1x1xf32>
    %adwdpb13pW = stablehlo.multiply %adwdlrb13pW, %b13pW : tensor<192x1152x1x1xf32>
    %adnewb13pW = stablehlo.subtract %adsubb13pW, %adwdpb13pW : tensor<192x1152x1x1xf32>
    %adb1b13pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b13pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb13pb = stablehlo.multiply %adb1b13pb, %b13pbm : tensor<192xf32>
    %admgb13pb = stablehlo.multiply %adob1b13pb, %b13dpb : tensor<192xf32>
    %admnb13pb = stablehlo.add %admsb13pb, %admgb13pb : tensor<192xf32>
    %adb2b13pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b13pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb13pb = stablehlo.multiply %adb2b13pb, %b13pbv : tensor<192xf32>
    %adg2b13pb = stablehlo.multiply %b13dpb, %b13dpb : tensor<192xf32>
    %advgb13pb = stablehlo.multiply %adob2b13pb, %adg2b13pb : tensor<192xf32>
    %advnb13pb = stablehlo.add %advsb13pb, %advgb13pb : tensor<192xf32>
    %adbc1b13pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b13pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb13pb = stablehlo.divide %admnb13pb, %adbc1b13pb : tensor<192xf32>
    %advhb13pb = stablehlo.divide %advnb13pb, %adbc2b13pb : tensor<192xf32>
    %adlrb13pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb13pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb13pb = stablehlo.sqrt %advhb13pb : tensor<192xf32>
    %addenb13pb = stablehlo.add %adsqb13pb, %adepsb13pb : tensor<192xf32>
    %adratb13pb = stablehlo.divide %admhb13pb, %addenb13pb : tensor<192xf32>
    %adstb13pb = stablehlo.multiply %adlrb13pb, %adratb13pb : tensor<192xf32>
    %adsubb13pb = stablehlo.subtract %b13pb, %adstb13pb : tensor<192xf32>
    %adwdb13pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb13pb = stablehlo.multiply %adwdb13pb, %adlrb13pb : tensor<192xf32>
    %adwdpb13pb = stablehlo.multiply %adwdlrb13pb, %b13pb : tensor<192xf32>
    %adnewb13pb = stablehlo.subtract %adsubb13pb, %adwdpb13pb : tensor<192xf32>
    %adb1b13pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b13pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb13pg = stablehlo.multiply %adb1b13pg, %b13pgm : tensor<192xf32>
    %admgb13pg = stablehlo.multiply %adob1b13pg, %b13dpndg : tensor<192xf32>
    %admnb13pg = stablehlo.add %admsb13pg, %admgb13pg : tensor<192xf32>
    %adb2b13pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b13pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb13pg = stablehlo.multiply %adb2b13pg, %b13pgv : tensor<192xf32>
    %adg2b13pg = stablehlo.multiply %b13dpndg, %b13dpndg : tensor<192xf32>
    %advgb13pg = stablehlo.multiply %adob2b13pg, %adg2b13pg : tensor<192xf32>
    %advnb13pg = stablehlo.add %advsb13pg, %advgb13pg : tensor<192xf32>
    %adbc1b13pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b13pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb13pg = stablehlo.divide %admnb13pg, %adbc1b13pg : tensor<192xf32>
    %advhb13pg = stablehlo.divide %advnb13pg, %adbc2b13pg : tensor<192xf32>
    %adlrb13pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb13pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb13pg = stablehlo.sqrt %advhb13pg : tensor<192xf32>
    %addenb13pg = stablehlo.add %adsqb13pg, %adepsb13pg : tensor<192xf32>
    %adratb13pg = stablehlo.divide %admhb13pg, %addenb13pg : tensor<192xf32>
    %adstb13pg = stablehlo.multiply %adlrb13pg, %adratb13pg : tensor<192xf32>
    %adsubb13pg = stablehlo.subtract %b13pg, %adstb13pg : tensor<192xf32>
    %adwdb13pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb13pg = stablehlo.multiply %adwdb13pg, %adlrb13pg : tensor<192xf32>
    %adwdpb13pg = stablehlo.multiply %adwdlrb13pg, %b13pg : tensor<192xf32>
    %adnewb13pg = stablehlo.subtract %adsubb13pg, %adwdpb13pg : tensor<192xf32>
    %adb1b13pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b13pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb13pbt = stablehlo.multiply %adb1b13pbt, %b13pbtm : tensor<192xf32>
    %admgb13pbt = stablehlo.multiply %adob1b13pbt, %b13dpndb : tensor<192xf32>
    %admnb13pbt = stablehlo.add %admsb13pbt, %admgb13pbt : tensor<192xf32>
    %adb2b13pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b13pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb13pbt = stablehlo.multiply %adb2b13pbt, %b13pbtv : tensor<192xf32>
    %adg2b13pbt = stablehlo.multiply %b13dpndb, %b13dpndb : tensor<192xf32>
    %advgb13pbt = stablehlo.multiply %adob2b13pbt, %adg2b13pbt : tensor<192xf32>
    %advnb13pbt = stablehlo.add %advsb13pbt, %advgb13pbt : tensor<192xf32>
    %adbc1b13pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b13pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb13pbt = stablehlo.divide %admnb13pbt, %adbc1b13pbt : tensor<192xf32>
    %advhb13pbt = stablehlo.divide %advnb13pbt, %adbc2b13pbt : tensor<192xf32>
    %adlrb13pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb13pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb13pbt = stablehlo.sqrt %advhb13pbt : tensor<192xf32>
    %addenb13pbt = stablehlo.add %adsqb13pbt, %adepsb13pbt : tensor<192xf32>
    %adratb13pbt = stablehlo.divide %admhb13pbt, %addenb13pbt : tensor<192xf32>
    %adstb13pbt = stablehlo.multiply %adlrb13pbt, %adratb13pbt : tensor<192xf32>
    %adsubb13pbt = stablehlo.subtract %b13pbt, %adstb13pbt : tensor<192xf32>
    %adwdb13pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb13pbt = stablehlo.multiply %adwdb13pbt, %adlrb13pbt : tensor<192xf32>
    %adwdpb13pbt = stablehlo.multiply %adwdlrb13pbt, %b13pbt : tensor<192xf32>
    %adnewb13pbt = stablehlo.subtract %adsubb13pbt, %adwdpb13pbt : tensor<192xf32>
    %adb1b14eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adob1b14eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %admsb14eW = stablehlo.multiply %adb1b14eW, %b14eWm : tensor<1152x192x1x1xf32>
    %admgb14eW = stablehlo.multiply %adob1b14eW, %b14deW : tensor<1152x192x1x1xf32>
    %admnb14eW = stablehlo.add %admsb14eW, %admgb14eW : tensor<1152x192x1x1xf32>
    %adb2b14eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adob2b14eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %advsb14eW = stablehlo.multiply %adb2b14eW, %b14eWv : tensor<1152x192x1x1xf32>
    %adg2b14eW = stablehlo.multiply %b14deW, %b14deW : tensor<1152x192x1x1xf32>
    %advgb14eW = stablehlo.multiply %adob2b14eW, %adg2b14eW : tensor<1152x192x1x1xf32>
    %advnb14eW = stablehlo.add %advsb14eW, %advgb14eW : tensor<1152x192x1x1xf32>
    %adbc1b14eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adbc2b14eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %admhb14eW = stablehlo.divide %admnb14eW, %adbc1b14eW : tensor<1152x192x1x1xf32>
    %advhb14eW = stablehlo.divide %advnb14eW, %adbc2b14eW : tensor<1152x192x1x1xf32>
    %adlrb14eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adepsb14eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adsqb14eW = stablehlo.sqrt %advhb14eW : tensor<1152x192x1x1xf32>
    %addenb14eW = stablehlo.add %adsqb14eW, %adepsb14eW : tensor<1152x192x1x1xf32>
    %adratb14eW = stablehlo.divide %admhb14eW, %addenb14eW : tensor<1152x192x1x1xf32>
    %adstb14eW = stablehlo.multiply %adlrb14eW, %adratb14eW : tensor<1152x192x1x1xf32>
    %adsubb14eW = stablehlo.subtract %b14eW, %adstb14eW : tensor<1152x192x1x1xf32>
    %adwdb14eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adwdlrb14eW = stablehlo.multiply %adwdb14eW, %adlrb14eW : tensor<1152x192x1x1xf32>
    %adwdpb14eW = stablehlo.multiply %adwdlrb14eW, %b14eW : tensor<1152x192x1x1xf32>
    %adnewb14eW = stablehlo.subtract %adsubb14eW, %adwdpb14eW : tensor<1152x192x1x1xf32>
    %adb1b14eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b14eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb14eb = stablehlo.multiply %adb1b14eb, %b14ebm : tensor<1152xf32>
    %admgb14eb = stablehlo.multiply %adob1b14eb, %b14deb : tensor<1152xf32>
    %admnb14eb = stablehlo.add %admsb14eb, %admgb14eb : tensor<1152xf32>
    %adb2b14eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b14eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb14eb = stablehlo.multiply %adb2b14eb, %b14ebv : tensor<1152xf32>
    %adg2b14eb = stablehlo.multiply %b14deb, %b14deb : tensor<1152xf32>
    %advgb14eb = stablehlo.multiply %adob2b14eb, %adg2b14eb : tensor<1152xf32>
    %advnb14eb = stablehlo.add %advsb14eb, %advgb14eb : tensor<1152xf32>
    %adbc1b14eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b14eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb14eb = stablehlo.divide %admnb14eb, %adbc1b14eb : tensor<1152xf32>
    %advhb14eb = stablehlo.divide %advnb14eb, %adbc2b14eb : tensor<1152xf32>
    %adlrb14eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb14eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb14eb = stablehlo.sqrt %advhb14eb : tensor<1152xf32>
    %addenb14eb = stablehlo.add %adsqb14eb, %adepsb14eb : tensor<1152xf32>
    %adratb14eb = stablehlo.divide %admhb14eb, %addenb14eb : tensor<1152xf32>
    %adstb14eb = stablehlo.multiply %adlrb14eb, %adratb14eb : tensor<1152xf32>
    %adsubb14eb = stablehlo.subtract %b14eb, %adstb14eb : tensor<1152xf32>
    %adwdb14eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb14eb = stablehlo.multiply %adwdb14eb, %adlrb14eb : tensor<1152xf32>
    %adwdpb14eb = stablehlo.multiply %adwdlrb14eb, %b14eb : tensor<1152xf32>
    %adnewb14eb = stablehlo.subtract %adsubb14eb, %adwdpb14eb : tensor<1152xf32>
    %adb1b14eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b14eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb14eg = stablehlo.multiply %adb1b14eg, %b14egm : tensor<1152xf32>
    %admgb14eg = stablehlo.multiply %adob1b14eg, %b14dendg : tensor<1152xf32>
    %admnb14eg = stablehlo.add %admsb14eg, %admgb14eg : tensor<1152xf32>
    %adb2b14eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b14eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb14eg = stablehlo.multiply %adb2b14eg, %b14egv : tensor<1152xf32>
    %adg2b14eg = stablehlo.multiply %b14dendg, %b14dendg : tensor<1152xf32>
    %advgb14eg = stablehlo.multiply %adob2b14eg, %adg2b14eg : tensor<1152xf32>
    %advnb14eg = stablehlo.add %advsb14eg, %advgb14eg : tensor<1152xf32>
    %adbc1b14eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b14eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb14eg = stablehlo.divide %admnb14eg, %adbc1b14eg : tensor<1152xf32>
    %advhb14eg = stablehlo.divide %advnb14eg, %adbc2b14eg : tensor<1152xf32>
    %adlrb14eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb14eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb14eg = stablehlo.sqrt %advhb14eg : tensor<1152xf32>
    %addenb14eg = stablehlo.add %adsqb14eg, %adepsb14eg : tensor<1152xf32>
    %adratb14eg = stablehlo.divide %admhb14eg, %addenb14eg : tensor<1152xf32>
    %adstb14eg = stablehlo.multiply %adlrb14eg, %adratb14eg : tensor<1152xf32>
    %adsubb14eg = stablehlo.subtract %b14eg, %adstb14eg : tensor<1152xf32>
    %adwdb14eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb14eg = stablehlo.multiply %adwdb14eg, %adlrb14eg : tensor<1152xf32>
    %adwdpb14eg = stablehlo.multiply %adwdlrb14eg, %b14eg : tensor<1152xf32>
    %adnewb14eg = stablehlo.subtract %adsubb14eg, %adwdpb14eg : tensor<1152xf32>
    %adb1b14ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b14ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb14ebt = stablehlo.multiply %adb1b14ebt, %b14ebtm : tensor<1152xf32>
    %admgb14ebt = stablehlo.multiply %adob1b14ebt, %b14dendb : tensor<1152xf32>
    %admnb14ebt = stablehlo.add %admsb14ebt, %admgb14ebt : tensor<1152xf32>
    %adb2b14ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b14ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb14ebt = stablehlo.multiply %adb2b14ebt, %b14ebtv : tensor<1152xf32>
    %adg2b14ebt = stablehlo.multiply %b14dendb, %b14dendb : tensor<1152xf32>
    %advgb14ebt = stablehlo.multiply %adob2b14ebt, %adg2b14ebt : tensor<1152xf32>
    %advnb14ebt = stablehlo.add %advsb14ebt, %advgb14ebt : tensor<1152xf32>
    %adbc1b14ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b14ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb14ebt = stablehlo.divide %admnb14ebt, %adbc1b14ebt : tensor<1152xf32>
    %advhb14ebt = stablehlo.divide %advnb14ebt, %adbc2b14ebt : tensor<1152xf32>
    %adlrb14ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb14ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb14ebt = stablehlo.sqrt %advhb14ebt : tensor<1152xf32>
    %addenb14ebt = stablehlo.add %adsqb14ebt, %adepsb14ebt : tensor<1152xf32>
    %adratb14ebt = stablehlo.divide %admhb14ebt, %addenb14ebt : tensor<1152xf32>
    %adstb14ebt = stablehlo.multiply %adlrb14ebt, %adratb14ebt : tensor<1152xf32>
    %adsubb14ebt = stablehlo.subtract %b14ebt, %adstb14ebt : tensor<1152xf32>
    %adwdb14ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb14ebt = stablehlo.multiply %adwdb14ebt, %adlrb14ebt : tensor<1152xf32>
    %adwdpb14ebt = stablehlo.multiply %adwdlrb14ebt, %b14ebt : tensor<1152xf32>
    %adnewb14ebt = stablehlo.subtract %adsubb14ebt, %adwdpb14ebt : tensor<1152xf32>
    %adb1b14dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adob1b14dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %admsb14dW = stablehlo.multiply %adb1b14dW, %b14dWm : tensor<1152x1x5x5xf32>
    %admgb14dW = stablehlo.multiply %adob1b14dW, %b14ddW : tensor<1152x1x5x5xf32>
    %admnb14dW = stablehlo.add %admsb14dW, %admgb14dW : tensor<1152x1x5x5xf32>
    %adb2b14dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adob2b14dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %advsb14dW = stablehlo.multiply %adb2b14dW, %b14dWv : tensor<1152x1x5x5xf32>
    %adg2b14dW = stablehlo.multiply %b14ddW, %b14ddW : tensor<1152x1x5x5xf32>
    %advgb14dW = stablehlo.multiply %adob2b14dW, %adg2b14dW : tensor<1152x1x5x5xf32>
    %advnb14dW = stablehlo.add %advsb14dW, %advgb14dW : tensor<1152x1x5x5xf32>
    %adbc1b14dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adbc2b14dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %admhb14dW = stablehlo.divide %admnb14dW, %adbc1b14dW : tensor<1152x1x5x5xf32>
    %advhb14dW = stablehlo.divide %advnb14dW, %adbc2b14dW : tensor<1152x1x5x5xf32>
    %adlrb14dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adepsb14dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adsqb14dW = stablehlo.sqrt %advhb14dW : tensor<1152x1x5x5xf32>
    %addenb14dW = stablehlo.add %adsqb14dW, %adepsb14dW : tensor<1152x1x5x5xf32>
    %adratb14dW = stablehlo.divide %admhb14dW, %addenb14dW : tensor<1152x1x5x5xf32>
    %adstb14dW = stablehlo.multiply %adlrb14dW, %adratb14dW : tensor<1152x1x5x5xf32>
    %adsubb14dW = stablehlo.subtract %b14dW, %adstb14dW : tensor<1152x1x5x5xf32>
    %adwdb14dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adwdlrb14dW = stablehlo.multiply %adwdb14dW, %adlrb14dW : tensor<1152x1x5x5xf32>
    %adwdpb14dW = stablehlo.multiply %adwdlrb14dW, %b14dW : tensor<1152x1x5x5xf32>
    %adnewb14dW = stablehlo.subtract %adsubb14dW, %adwdpb14dW : tensor<1152x1x5x5xf32>
    %adb1b14db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b14db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb14db = stablehlo.multiply %adb1b14db, %b14dbm : tensor<1152xf32>
    %admgb14db = stablehlo.multiply %adob1b14db, %b14ddb : tensor<1152xf32>
    %admnb14db = stablehlo.add %admsb14db, %admgb14db : tensor<1152xf32>
    %adb2b14db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b14db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb14db = stablehlo.multiply %adb2b14db, %b14dbv : tensor<1152xf32>
    %adg2b14db = stablehlo.multiply %b14ddb, %b14ddb : tensor<1152xf32>
    %advgb14db = stablehlo.multiply %adob2b14db, %adg2b14db : tensor<1152xf32>
    %advnb14db = stablehlo.add %advsb14db, %advgb14db : tensor<1152xf32>
    %adbc1b14db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b14db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb14db = stablehlo.divide %admnb14db, %adbc1b14db : tensor<1152xf32>
    %advhb14db = stablehlo.divide %advnb14db, %adbc2b14db : tensor<1152xf32>
    %adlrb14db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb14db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb14db = stablehlo.sqrt %advhb14db : tensor<1152xf32>
    %addenb14db = stablehlo.add %adsqb14db, %adepsb14db : tensor<1152xf32>
    %adratb14db = stablehlo.divide %admhb14db, %addenb14db : tensor<1152xf32>
    %adstb14db = stablehlo.multiply %adlrb14db, %adratb14db : tensor<1152xf32>
    %adsubb14db = stablehlo.subtract %b14db, %adstb14db : tensor<1152xf32>
    %adwdb14db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb14db = stablehlo.multiply %adwdb14db, %adlrb14db : tensor<1152xf32>
    %adwdpb14db = stablehlo.multiply %adwdlrb14db, %b14db : tensor<1152xf32>
    %adnewb14db = stablehlo.subtract %adsubb14db, %adwdpb14db : tensor<1152xf32>
    %adb1b14dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b14dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb14dg = stablehlo.multiply %adb1b14dg, %b14dgm : tensor<1152xf32>
    %admgb14dg = stablehlo.multiply %adob1b14dg, %b14ddndg : tensor<1152xf32>
    %admnb14dg = stablehlo.add %admsb14dg, %admgb14dg : tensor<1152xf32>
    %adb2b14dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b14dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb14dg = stablehlo.multiply %adb2b14dg, %b14dgv : tensor<1152xf32>
    %adg2b14dg = stablehlo.multiply %b14ddndg, %b14ddndg : tensor<1152xf32>
    %advgb14dg = stablehlo.multiply %adob2b14dg, %adg2b14dg : tensor<1152xf32>
    %advnb14dg = stablehlo.add %advsb14dg, %advgb14dg : tensor<1152xf32>
    %adbc1b14dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b14dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb14dg = stablehlo.divide %admnb14dg, %adbc1b14dg : tensor<1152xf32>
    %advhb14dg = stablehlo.divide %advnb14dg, %adbc2b14dg : tensor<1152xf32>
    %adlrb14dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb14dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb14dg = stablehlo.sqrt %advhb14dg : tensor<1152xf32>
    %addenb14dg = stablehlo.add %adsqb14dg, %adepsb14dg : tensor<1152xf32>
    %adratb14dg = stablehlo.divide %admhb14dg, %addenb14dg : tensor<1152xf32>
    %adstb14dg = stablehlo.multiply %adlrb14dg, %adratb14dg : tensor<1152xf32>
    %adsubb14dg = stablehlo.subtract %b14dg, %adstb14dg : tensor<1152xf32>
    %adwdb14dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb14dg = stablehlo.multiply %adwdb14dg, %adlrb14dg : tensor<1152xf32>
    %adwdpb14dg = stablehlo.multiply %adwdlrb14dg, %b14dg : tensor<1152xf32>
    %adnewb14dg = stablehlo.subtract %adsubb14dg, %adwdpb14dg : tensor<1152xf32>
    %adb1b14dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b14dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb14dbt = stablehlo.multiply %adb1b14dbt, %b14dbtm : tensor<1152xf32>
    %admgb14dbt = stablehlo.multiply %adob1b14dbt, %b14ddndb : tensor<1152xf32>
    %admnb14dbt = stablehlo.add %admsb14dbt, %admgb14dbt : tensor<1152xf32>
    %adb2b14dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b14dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb14dbt = stablehlo.multiply %adb2b14dbt, %b14dbtv : tensor<1152xf32>
    %adg2b14dbt = stablehlo.multiply %b14ddndb, %b14ddndb : tensor<1152xf32>
    %advgb14dbt = stablehlo.multiply %adob2b14dbt, %adg2b14dbt : tensor<1152xf32>
    %advnb14dbt = stablehlo.add %advsb14dbt, %advgb14dbt : tensor<1152xf32>
    %adbc1b14dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b14dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb14dbt = stablehlo.divide %admnb14dbt, %adbc1b14dbt : tensor<1152xf32>
    %advhb14dbt = stablehlo.divide %advnb14dbt, %adbc2b14dbt : tensor<1152xf32>
    %adlrb14dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb14dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb14dbt = stablehlo.sqrt %advhb14dbt : tensor<1152xf32>
    %addenb14dbt = stablehlo.add %adsqb14dbt, %adepsb14dbt : tensor<1152xf32>
    %adratb14dbt = stablehlo.divide %admhb14dbt, %addenb14dbt : tensor<1152xf32>
    %adstb14dbt = stablehlo.multiply %adlrb14dbt, %adratb14dbt : tensor<1152xf32>
    %adsubb14dbt = stablehlo.subtract %b14dbt, %adstb14dbt : tensor<1152xf32>
    %adwdb14dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb14dbt = stablehlo.multiply %adwdb14dbt, %adlrb14dbt : tensor<1152xf32>
    %adwdpb14dbt = stablehlo.multiply %adwdlrb14dbt, %b14dbt : tensor<1152xf32>
    %adnewb14dbt = stablehlo.subtract %adsubb14dbt, %adwdpb14dbt : tensor<1152xf32>
    %adb1b14zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adob1b14zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %admsb14zW1 = stablehlo.multiply %adb1b14zW1, %b14zW1m : tensor<1152x48xf32>
    %admgb14zW1 = stablehlo.multiply %adob1b14zW1, %b14zdWs1 : tensor<1152x48xf32>
    %admnb14zW1 = stablehlo.add %admsb14zW1, %admgb14zW1 : tensor<1152x48xf32>
    %adb2b14zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adob2b14zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %advsb14zW1 = stablehlo.multiply %adb2b14zW1, %b14zW1v : tensor<1152x48xf32>
    %adg2b14zW1 = stablehlo.multiply %b14zdWs1, %b14zdWs1 : tensor<1152x48xf32>
    %advgb14zW1 = stablehlo.multiply %adob2b14zW1, %adg2b14zW1 : tensor<1152x48xf32>
    %advnb14zW1 = stablehlo.add %advsb14zW1, %advgb14zW1 : tensor<1152x48xf32>
    %adbc1b14zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adbc2b14zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %admhb14zW1 = stablehlo.divide %admnb14zW1, %adbc1b14zW1 : tensor<1152x48xf32>
    %advhb14zW1 = stablehlo.divide %advnb14zW1, %adbc2b14zW1 : tensor<1152x48xf32>
    %adlrb14zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adepsb14zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adsqb14zW1 = stablehlo.sqrt %advhb14zW1 : tensor<1152x48xf32>
    %addenb14zW1 = stablehlo.add %adsqb14zW1, %adepsb14zW1 : tensor<1152x48xf32>
    %adratb14zW1 = stablehlo.divide %admhb14zW1, %addenb14zW1 : tensor<1152x48xf32>
    %adstb14zW1 = stablehlo.multiply %adlrb14zW1, %adratb14zW1 : tensor<1152x48xf32>
    %adsubb14zW1 = stablehlo.subtract %b14zW1, %adstb14zW1 : tensor<1152x48xf32>
    %adwdb14zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adwdlrb14zW1 = stablehlo.multiply %adwdb14zW1, %adlrb14zW1 : tensor<1152x48xf32>
    %adwdpb14zW1 = stablehlo.multiply %adwdlrb14zW1, %b14zW1 : tensor<1152x48xf32>
    %adnewb14zW1 = stablehlo.subtract %adsubb14zW1, %adwdpb14zW1 : tensor<1152x48xf32>
    %adb1b14zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adob1b14zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %admsb14zb1 = stablehlo.multiply %adb1b14zb1, %b14zb1m : tensor<48xf32>
    %admgb14zb1 = stablehlo.multiply %adob1b14zb1, %b14zdbs1 : tensor<48xf32>
    %admnb14zb1 = stablehlo.add %admsb14zb1, %admgb14zb1 : tensor<48xf32>
    %adb2b14zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adob2b14zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %advsb14zb1 = stablehlo.multiply %adb2b14zb1, %b14zb1v : tensor<48xf32>
    %adg2b14zb1 = stablehlo.multiply %b14zdbs1, %b14zdbs1 : tensor<48xf32>
    %advgb14zb1 = stablehlo.multiply %adob2b14zb1, %adg2b14zb1 : tensor<48xf32>
    %advnb14zb1 = stablehlo.add %advsb14zb1, %advgb14zb1 : tensor<48xf32>
    %adbc1b14zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adbc2b14zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %admhb14zb1 = stablehlo.divide %admnb14zb1, %adbc1b14zb1 : tensor<48xf32>
    %advhb14zb1 = stablehlo.divide %advnb14zb1, %adbc2b14zb1 : tensor<48xf32>
    %adlrb14zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adepsb14zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adsqb14zb1 = stablehlo.sqrt %advhb14zb1 : tensor<48xf32>
    %addenb14zb1 = stablehlo.add %adsqb14zb1, %adepsb14zb1 : tensor<48xf32>
    %adratb14zb1 = stablehlo.divide %admhb14zb1, %addenb14zb1 : tensor<48xf32>
    %adstb14zb1 = stablehlo.multiply %adlrb14zb1, %adratb14zb1 : tensor<48xf32>
    %adsubb14zb1 = stablehlo.subtract %b14zb1, %adstb14zb1 : tensor<48xf32>
    %adwdb14zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adwdlrb14zb1 = stablehlo.multiply %adwdb14zb1, %adlrb14zb1 : tensor<48xf32>
    %adwdpb14zb1 = stablehlo.multiply %adwdlrb14zb1, %b14zb1 : tensor<48xf32>
    %adnewb14zb1 = stablehlo.subtract %adsubb14zb1, %adwdpb14zb1 : tensor<48xf32>
    %adb1b14zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adob1b14zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %admsb14zW2 = stablehlo.multiply %adb1b14zW2, %b14zW2m : tensor<48x1152xf32>
    %admgb14zW2 = stablehlo.multiply %adob1b14zW2, %b14zdWs2 : tensor<48x1152xf32>
    %admnb14zW2 = stablehlo.add %admsb14zW2, %admgb14zW2 : tensor<48x1152xf32>
    %adb2b14zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adob2b14zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %advsb14zW2 = stablehlo.multiply %adb2b14zW2, %b14zW2v : tensor<48x1152xf32>
    %adg2b14zW2 = stablehlo.multiply %b14zdWs2, %b14zdWs2 : tensor<48x1152xf32>
    %advgb14zW2 = stablehlo.multiply %adob2b14zW2, %adg2b14zW2 : tensor<48x1152xf32>
    %advnb14zW2 = stablehlo.add %advsb14zW2, %advgb14zW2 : tensor<48x1152xf32>
    %adbc1b14zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adbc2b14zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %admhb14zW2 = stablehlo.divide %admnb14zW2, %adbc1b14zW2 : tensor<48x1152xf32>
    %advhb14zW2 = stablehlo.divide %advnb14zW2, %adbc2b14zW2 : tensor<48x1152xf32>
    %adlrb14zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adepsb14zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adsqb14zW2 = stablehlo.sqrt %advhb14zW2 : tensor<48x1152xf32>
    %addenb14zW2 = stablehlo.add %adsqb14zW2, %adepsb14zW2 : tensor<48x1152xf32>
    %adratb14zW2 = stablehlo.divide %admhb14zW2, %addenb14zW2 : tensor<48x1152xf32>
    %adstb14zW2 = stablehlo.multiply %adlrb14zW2, %adratb14zW2 : tensor<48x1152xf32>
    %adsubb14zW2 = stablehlo.subtract %b14zW2, %adstb14zW2 : tensor<48x1152xf32>
    %adwdb14zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adwdlrb14zW2 = stablehlo.multiply %adwdb14zW2, %adlrb14zW2 : tensor<48x1152xf32>
    %adwdpb14zW2 = stablehlo.multiply %adwdlrb14zW2, %b14zW2 : tensor<48x1152xf32>
    %adnewb14zW2 = stablehlo.subtract %adsubb14zW2, %adwdpb14zW2 : tensor<48x1152xf32>
    %adb1b14zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b14zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb14zb2 = stablehlo.multiply %adb1b14zb2, %b14zb2m : tensor<1152xf32>
    %admgb14zb2 = stablehlo.multiply %adob1b14zb2, %b14zdbs2 : tensor<1152xf32>
    %admnb14zb2 = stablehlo.add %admsb14zb2, %admgb14zb2 : tensor<1152xf32>
    %adb2b14zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b14zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb14zb2 = stablehlo.multiply %adb2b14zb2, %b14zb2v : tensor<1152xf32>
    %adg2b14zb2 = stablehlo.multiply %b14zdbs2, %b14zdbs2 : tensor<1152xf32>
    %advgb14zb2 = stablehlo.multiply %adob2b14zb2, %adg2b14zb2 : tensor<1152xf32>
    %advnb14zb2 = stablehlo.add %advsb14zb2, %advgb14zb2 : tensor<1152xf32>
    %adbc1b14zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b14zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb14zb2 = stablehlo.divide %admnb14zb2, %adbc1b14zb2 : tensor<1152xf32>
    %advhb14zb2 = stablehlo.divide %advnb14zb2, %adbc2b14zb2 : tensor<1152xf32>
    %adlrb14zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb14zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb14zb2 = stablehlo.sqrt %advhb14zb2 : tensor<1152xf32>
    %addenb14zb2 = stablehlo.add %adsqb14zb2, %adepsb14zb2 : tensor<1152xf32>
    %adratb14zb2 = stablehlo.divide %admhb14zb2, %addenb14zb2 : tensor<1152xf32>
    %adstb14zb2 = stablehlo.multiply %adlrb14zb2, %adratb14zb2 : tensor<1152xf32>
    %adsubb14zb2 = stablehlo.subtract %b14zb2, %adstb14zb2 : tensor<1152xf32>
    %adwdb14zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb14zb2 = stablehlo.multiply %adwdb14zb2, %adlrb14zb2 : tensor<1152xf32>
    %adwdpb14zb2 = stablehlo.multiply %adwdlrb14zb2, %b14zb2 : tensor<1152xf32>
    %adnewb14zb2 = stablehlo.subtract %adsubb14zb2, %adwdpb14zb2 : tensor<1152xf32>
    %adb1b14pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adob1b14pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %admsb14pW = stablehlo.multiply %adb1b14pW, %b14pWm : tensor<192x1152x1x1xf32>
    %admgb14pW = stablehlo.multiply %adob1b14pW, %b14dpW : tensor<192x1152x1x1xf32>
    %admnb14pW = stablehlo.add %admsb14pW, %admgb14pW : tensor<192x1152x1x1xf32>
    %adb2b14pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adob2b14pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %advsb14pW = stablehlo.multiply %adb2b14pW, %b14pWv : tensor<192x1152x1x1xf32>
    %adg2b14pW = stablehlo.multiply %b14dpW, %b14dpW : tensor<192x1152x1x1xf32>
    %advgb14pW = stablehlo.multiply %adob2b14pW, %adg2b14pW : tensor<192x1152x1x1xf32>
    %advnb14pW = stablehlo.add %advsb14pW, %advgb14pW : tensor<192x1152x1x1xf32>
    %adbc1b14pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adbc2b14pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %admhb14pW = stablehlo.divide %admnb14pW, %adbc1b14pW : tensor<192x1152x1x1xf32>
    %advhb14pW = stablehlo.divide %advnb14pW, %adbc2b14pW : tensor<192x1152x1x1xf32>
    %adlrb14pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adepsb14pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adsqb14pW = stablehlo.sqrt %advhb14pW : tensor<192x1152x1x1xf32>
    %addenb14pW = stablehlo.add %adsqb14pW, %adepsb14pW : tensor<192x1152x1x1xf32>
    %adratb14pW = stablehlo.divide %admhb14pW, %addenb14pW : tensor<192x1152x1x1xf32>
    %adstb14pW = stablehlo.multiply %adlrb14pW, %adratb14pW : tensor<192x1152x1x1xf32>
    %adsubb14pW = stablehlo.subtract %b14pW, %adstb14pW : tensor<192x1152x1x1xf32>
    %adwdb14pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adwdlrb14pW = stablehlo.multiply %adwdb14pW, %adlrb14pW : tensor<192x1152x1x1xf32>
    %adwdpb14pW = stablehlo.multiply %adwdlrb14pW, %b14pW : tensor<192x1152x1x1xf32>
    %adnewb14pW = stablehlo.subtract %adsubb14pW, %adwdpb14pW : tensor<192x1152x1x1xf32>
    %adb1b14pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b14pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb14pb = stablehlo.multiply %adb1b14pb, %b14pbm : tensor<192xf32>
    %admgb14pb = stablehlo.multiply %adob1b14pb, %b14dpb : tensor<192xf32>
    %admnb14pb = stablehlo.add %admsb14pb, %admgb14pb : tensor<192xf32>
    %adb2b14pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b14pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb14pb = stablehlo.multiply %adb2b14pb, %b14pbv : tensor<192xf32>
    %adg2b14pb = stablehlo.multiply %b14dpb, %b14dpb : tensor<192xf32>
    %advgb14pb = stablehlo.multiply %adob2b14pb, %adg2b14pb : tensor<192xf32>
    %advnb14pb = stablehlo.add %advsb14pb, %advgb14pb : tensor<192xf32>
    %adbc1b14pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b14pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb14pb = stablehlo.divide %admnb14pb, %adbc1b14pb : tensor<192xf32>
    %advhb14pb = stablehlo.divide %advnb14pb, %adbc2b14pb : tensor<192xf32>
    %adlrb14pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb14pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb14pb = stablehlo.sqrt %advhb14pb : tensor<192xf32>
    %addenb14pb = stablehlo.add %adsqb14pb, %adepsb14pb : tensor<192xf32>
    %adratb14pb = stablehlo.divide %admhb14pb, %addenb14pb : tensor<192xf32>
    %adstb14pb = stablehlo.multiply %adlrb14pb, %adratb14pb : tensor<192xf32>
    %adsubb14pb = stablehlo.subtract %b14pb, %adstb14pb : tensor<192xf32>
    %adwdb14pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb14pb = stablehlo.multiply %adwdb14pb, %adlrb14pb : tensor<192xf32>
    %adwdpb14pb = stablehlo.multiply %adwdlrb14pb, %b14pb : tensor<192xf32>
    %adnewb14pb = stablehlo.subtract %adsubb14pb, %adwdpb14pb : tensor<192xf32>
    %adb1b14pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b14pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb14pg = stablehlo.multiply %adb1b14pg, %b14pgm : tensor<192xf32>
    %admgb14pg = stablehlo.multiply %adob1b14pg, %b14dpndg : tensor<192xf32>
    %admnb14pg = stablehlo.add %admsb14pg, %admgb14pg : tensor<192xf32>
    %adb2b14pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b14pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb14pg = stablehlo.multiply %adb2b14pg, %b14pgv : tensor<192xf32>
    %adg2b14pg = stablehlo.multiply %b14dpndg, %b14dpndg : tensor<192xf32>
    %advgb14pg = stablehlo.multiply %adob2b14pg, %adg2b14pg : tensor<192xf32>
    %advnb14pg = stablehlo.add %advsb14pg, %advgb14pg : tensor<192xf32>
    %adbc1b14pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b14pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb14pg = stablehlo.divide %admnb14pg, %adbc1b14pg : tensor<192xf32>
    %advhb14pg = stablehlo.divide %advnb14pg, %adbc2b14pg : tensor<192xf32>
    %adlrb14pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb14pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb14pg = stablehlo.sqrt %advhb14pg : tensor<192xf32>
    %addenb14pg = stablehlo.add %adsqb14pg, %adepsb14pg : tensor<192xf32>
    %adratb14pg = stablehlo.divide %admhb14pg, %addenb14pg : tensor<192xf32>
    %adstb14pg = stablehlo.multiply %adlrb14pg, %adratb14pg : tensor<192xf32>
    %adsubb14pg = stablehlo.subtract %b14pg, %adstb14pg : tensor<192xf32>
    %adwdb14pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb14pg = stablehlo.multiply %adwdb14pg, %adlrb14pg : tensor<192xf32>
    %adwdpb14pg = stablehlo.multiply %adwdlrb14pg, %b14pg : tensor<192xf32>
    %adnewb14pg = stablehlo.subtract %adsubb14pg, %adwdpb14pg : tensor<192xf32>
    %adb1b14pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b14pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb14pbt = stablehlo.multiply %adb1b14pbt, %b14pbtm : tensor<192xf32>
    %admgb14pbt = stablehlo.multiply %adob1b14pbt, %b14dpndb : tensor<192xf32>
    %admnb14pbt = stablehlo.add %admsb14pbt, %admgb14pbt : tensor<192xf32>
    %adb2b14pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b14pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb14pbt = stablehlo.multiply %adb2b14pbt, %b14pbtv : tensor<192xf32>
    %adg2b14pbt = stablehlo.multiply %b14dpndb, %b14dpndb : tensor<192xf32>
    %advgb14pbt = stablehlo.multiply %adob2b14pbt, %adg2b14pbt : tensor<192xf32>
    %advnb14pbt = stablehlo.add %advsb14pbt, %advgb14pbt : tensor<192xf32>
    %adbc1b14pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b14pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb14pbt = stablehlo.divide %admnb14pbt, %adbc1b14pbt : tensor<192xf32>
    %advhb14pbt = stablehlo.divide %advnb14pbt, %adbc2b14pbt : tensor<192xf32>
    %adlrb14pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb14pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb14pbt = stablehlo.sqrt %advhb14pbt : tensor<192xf32>
    %addenb14pbt = stablehlo.add %adsqb14pbt, %adepsb14pbt : tensor<192xf32>
    %adratb14pbt = stablehlo.divide %admhb14pbt, %addenb14pbt : tensor<192xf32>
    %adstb14pbt = stablehlo.multiply %adlrb14pbt, %adratb14pbt : tensor<192xf32>
    %adsubb14pbt = stablehlo.subtract %b14pbt, %adstb14pbt : tensor<192xf32>
    %adwdb14pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb14pbt = stablehlo.multiply %adwdb14pbt, %adlrb14pbt : tensor<192xf32>
    %adwdpb14pbt = stablehlo.multiply %adwdlrb14pbt, %b14pbt : tensor<192xf32>
    %adnewb14pbt = stablehlo.subtract %adsubb14pbt, %adwdpb14pbt : tensor<192xf32>
    %adb1b15eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adob1b15eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %admsb15eW = stablehlo.multiply %adb1b15eW, %b15eWm : tensor<1152x192x1x1xf32>
    %admgb15eW = stablehlo.multiply %adob1b15eW, %b15deW : tensor<1152x192x1x1xf32>
    %admnb15eW = stablehlo.add %admsb15eW, %admgb15eW : tensor<1152x192x1x1xf32>
    %adb2b15eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adob2b15eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %advsb15eW = stablehlo.multiply %adb2b15eW, %b15eWv : tensor<1152x192x1x1xf32>
    %adg2b15eW = stablehlo.multiply %b15deW, %b15deW : tensor<1152x192x1x1xf32>
    %advgb15eW = stablehlo.multiply %adob2b15eW, %adg2b15eW : tensor<1152x192x1x1xf32>
    %advnb15eW = stablehlo.add %advsb15eW, %advgb15eW : tensor<1152x192x1x1xf32>
    %adbc1b15eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adbc2b15eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %admhb15eW = stablehlo.divide %admnb15eW, %adbc1b15eW : tensor<1152x192x1x1xf32>
    %advhb15eW = stablehlo.divide %advnb15eW, %adbc2b15eW : tensor<1152x192x1x1xf32>
    %adlrb15eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adepsb15eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adsqb15eW = stablehlo.sqrt %advhb15eW : tensor<1152x192x1x1xf32>
    %addenb15eW = stablehlo.add %adsqb15eW, %adepsb15eW : tensor<1152x192x1x1xf32>
    %adratb15eW = stablehlo.divide %admhb15eW, %addenb15eW : tensor<1152x192x1x1xf32>
    %adstb15eW = stablehlo.multiply %adlrb15eW, %adratb15eW : tensor<1152x192x1x1xf32>
    %adsubb15eW = stablehlo.subtract %b15eW, %adstb15eW : tensor<1152x192x1x1xf32>
    %adwdb15eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adwdlrb15eW = stablehlo.multiply %adwdb15eW, %adlrb15eW : tensor<1152x192x1x1xf32>
    %adwdpb15eW = stablehlo.multiply %adwdlrb15eW, %b15eW : tensor<1152x192x1x1xf32>
    %adnewb15eW = stablehlo.subtract %adsubb15eW, %adwdpb15eW : tensor<1152x192x1x1xf32>
    %adb1b15eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b15eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb15eb = stablehlo.multiply %adb1b15eb, %b15ebm : tensor<1152xf32>
    %admgb15eb = stablehlo.multiply %adob1b15eb, %b15deb : tensor<1152xf32>
    %admnb15eb = stablehlo.add %admsb15eb, %admgb15eb : tensor<1152xf32>
    %adb2b15eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b15eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb15eb = stablehlo.multiply %adb2b15eb, %b15ebv : tensor<1152xf32>
    %adg2b15eb = stablehlo.multiply %b15deb, %b15deb : tensor<1152xf32>
    %advgb15eb = stablehlo.multiply %adob2b15eb, %adg2b15eb : tensor<1152xf32>
    %advnb15eb = stablehlo.add %advsb15eb, %advgb15eb : tensor<1152xf32>
    %adbc1b15eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b15eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb15eb = stablehlo.divide %admnb15eb, %adbc1b15eb : tensor<1152xf32>
    %advhb15eb = stablehlo.divide %advnb15eb, %adbc2b15eb : tensor<1152xf32>
    %adlrb15eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb15eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb15eb = stablehlo.sqrt %advhb15eb : tensor<1152xf32>
    %addenb15eb = stablehlo.add %adsqb15eb, %adepsb15eb : tensor<1152xf32>
    %adratb15eb = stablehlo.divide %admhb15eb, %addenb15eb : tensor<1152xf32>
    %adstb15eb = stablehlo.multiply %adlrb15eb, %adratb15eb : tensor<1152xf32>
    %adsubb15eb = stablehlo.subtract %b15eb, %adstb15eb : tensor<1152xf32>
    %adwdb15eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb15eb = stablehlo.multiply %adwdb15eb, %adlrb15eb : tensor<1152xf32>
    %adwdpb15eb = stablehlo.multiply %adwdlrb15eb, %b15eb : tensor<1152xf32>
    %adnewb15eb = stablehlo.subtract %adsubb15eb, %adwdpb15eb : tensor<1152xf32>
    %adb1b15eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b15eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb15eg = stablehlo.multiply %adb1b15eg, %b15egm : tensor<1152xf32>
    %admgb15eg = stablehlo.multiply %adob1b15eg, %b15dendg : tensor<1152xf32>
    %admnb15eg = stablehlo.add %admsb15eg, %admgb15eg : tensor<1152xf32>
    %adb2b15eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b15eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb15eg = stablehlo.multiply %adb2b15eg, %b15egv : tensor<1152xf32>
    %adg2b15eg = stablehlo.multiply %b15dendg, %b15dendg : tensor<1152xf32>
    %advgb15eg = stablehlo.multiply %adob2b15eg, %adg2b15eg : tensor<1152xf32>
    %advnb15eg = stablehlo.add %advsb15eg, %advgb15eg : tensor<1152xf32>
    %adbc1b15eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b15eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb15eg = stablehlo.divide %admnb15eg, %adbc1b15eg : tensor<1152xf32>
    %advhb15eg = stablehlo.divide %advnb15eg, %adbc2b15eg : tensor<1152xf32>
    %adlrb15eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb15eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb15eg = stablehlo.sqrt %advhb15eg : tensor<1152xf32>
    %addenb15eg = stablehlo.add %adsqb15eg, %adepsb15eg : tensor<1152xf32>
    %adratb15eg = stablehlo.divide %admhb15eg, %addenb15eg : tensor<1152xf32>
    %adstb15eg = stablehlo.multiply %adlrb15eg, %adratb15eg : tensor<1152xf32>
    %adsubb15eg = stablehlo.subtract %b15eg, %adstb15eg : tensor<1152xf32>
    %adwdb15eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb15eg = stablehlo.multiply %adwdb15eg, %adlrb15eg : tensor<1152xf32>
    %adwdpb15eg = stablehlo.multiply %adwdlrb15eg, %b15eg : tensor<1152xf32>
    %adnewb15eg = stablehlo.subtract %adsubb15eg, %adwdpb15eg : tensor<1152xf32>
    %adb1b15ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b15ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb15ebt = stablehlo.multiply %adb1b15ebt, %b15ebtm : tensor<1152xf32>
    %admgb15ebt = stablehlo.multiply %adob1b15ebt, %b15dendb : tensor<1152xf32>
    %admnb15ebt = stablehlo.add %admsb15ebt, %admgb15ebt : tensor<1152xf32>
    %adb2b15ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b15ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb15ebt = stablehlo.multiply %adb2b15ebt, %b15ebtv : tensor<1152xf32>
    %adg2b15ebt = stablehlo.multiply %b15dendb, %b15dendb : tensor<1152xf32>
    %advgb15ebt = stablehlo.multiply %adob2b15ebt, %adg2b15ebt : tensor<1152xf32>
    %advnb15ebt = stablehlo.add %advsb15ebt, %advgb15ebt : tensor<1152xf32>
    %adbc1b15ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b15ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb15ebt = stablehlo.divide %admnb15ebt, %adbc1b15ebt : tensor<1152xf32>
    %advhb15ebt = stablehlo.divide %advnb15ebt, %adbc2b15ebt : tensor<1152xf32>
    %adlrb15ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb15ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb15ebt = stablehlo.sqrt %advhb15ebt : tensor<1152xf32>
    %addenb15ebt = stablehlo.add %adsqb15ebt, %adepsb15ebt : tensor<1152xf32>
    %adratb15ebt = stablehlo.divide %admhb15ebt, %addenb15ebt : tensor<1152xf32>
    %adstb15ebt = stablehlo.multiply %adlrb15ebt, %adratb15ebt : tensor<1152xf32>
    %adsubb15ebt = stablehlo.subtract %b15ebt, %adstb15ebt : tensor<1152xf32>
    %adwdb15ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb15ebt = stablehlo.multiply %adwdb15ebt, %adlrb15ebt : tensor<1152xf32>
    %adwdpb15ebt = stablehlo.multiply %adwdlrb15ebt, %b15ebt : tensor<1152xf32>
    %adnewb15ebt = stablehlo.subtract %adsubb15ebt, %adwdpb15ebt : tensor<1152xf32>
    %adb1b15dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adob1b15dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %admsb15dW = stablehlo.multiply %adb1b15dW, %b15dWm : tensor<1152x1x5x5xf32>
    %admgb15dW = stablehlo.multiply %adob1b15dW, %b15ddW : tensor<1152x1x5x5xf32>
    %admnb15dW = stablehlo.add %admsb15dW, %admgb15dW : tensor<1152x1x5x5xf32>
    %adb2b15dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adob2b15dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %advsb15dW = stablehlo.multiply %adb2b15dW, %b15dWv : tensor<1152x1x5x5xf32>
    %adg2b15dW = stablehlo.multiply %b15ddW, %b15ddW : tensor<1152x1x5x5xf32>
    %advgb15dW = stablehlo.multiply %adob2b15dW, %adg2b15dW : tensor<1152x1x5x5xf32>
    %advnb15dW = stablehlo.add %advsb15dW, %advgb15dW : tensor<1152x1x5x5xf32>
    %adbc1b15dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adbc2b15dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %admhb15dW = stablehlo.divide %admnb15dW, %adbc1b15dW : tensor<1152x1x5x5xf32>
    %advhb15dW = stablehlo.divide %advnb15dW, %adbc2b15dW : tensor<1152x1x5x5xf32>
    %adlrb15dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adepsb15dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adsqb15dW = stablehlo.sqrt %advhb15dW : tensor<1152x1x5x5xf32>
    %addenb15dW = stablehlo.add %adsqb15dW, %adepsb15dW : tensor<1152x1x5x5xf32>
    %adratb15dW = stablehlo.divide %admhb15dW, %addenb15dW : tensor<1152x1x5x5xf32>
    %adstb15dW = stablehlo.multiply %adlrb15dW, %adratb15dW : tensor<1152x1x5x5xf32>
    %adsubb15dW = stablehlo.subtract %b15dW, %adstb15dW : tensor<1152x1x5x5xf32>
    %adwdb15dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x1x5x5xf32>
    %adwdlrb15dW = stablehlo.multiply %adwdb15dW, %adlrb15dW : tensor<1152x1x5x5xf32>
    %adwdpb15dW = stablehlo.multiply %adwdlrb15dW, %b15dW : tensor<1152x1x5x5xf32>
    %adnewb15dW = stablehlo.subtract %adsubb15dW, %adwdpb15dW : tensor<1152x1x5x5xf32>
    %adb1b15db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b15db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb15db = stablehlo.multiply %adb1b15db, %b15dbm : tensor<1152xf32>
    %admgb15db = stablehlo.multiply %adob1b15db, %b15ddb : tensor<1152xf32>
    %admnb15db = stablehlo.add %admsb15db, %admgb15db : tensor<1152xf32>
    %adb2b15db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b15db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb15db = stablehlo.multiply %adb2b15db, %b15dbv : tensor<1152xf32>
    %adg2b15db = stablehlo.multiply %b15ddb, %b15ddb : tensor<1152xf32>
    %advgb15db = stablehlo.multiply %adob2b15db, %adg2b15db : tensor<1152xf32>
    %advnb15db = stablehlo.add %advsb15db, %advgb15db : tensor<1152xf32>
    %adbc1b15db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b15db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb15db = stablehlo.divide %admnb15db, %adbc1b15db : tensor<1152xf32>
    %advhb15db = stablehlo.divide %advnb15db, %adbc2b15db : tensor<1152xf32>
    %adlrb15db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb15db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb15db = stablehlo.sqrt %advhb15db : tensor<1152xf32>
    %addenb15db = stablehlo.add %adsqb15db, %adepsb15db : tensor<1152xf32>
    %adratb15db = stablehlo.divide %admhb15db, %addenb15db : tensor<1152xf32>
    %adstb15db = stablehlo.multiply %adlrb15db, %adratb15db : tensor<1152xf32>
    %adsubb15db = stablehlo.subtract %b15db, %adstb15db : tensor<1152xf32>
    %adwdb15db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb15db = stablehlo.multiply %adwdb15db, %adlrb15db : tensor<1152xf32>
    %adwdpb15db = stablehlo.multiply %adwdlrb15db, %b15db : tensor<1152xf32>
    %adnewb15db = stablehlo.subtract %adsubb15db, %adwdpb15db : tensor<1152xf32>
    %adb1b15dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b15dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb15dg = stablehlo.multiply %adb1b15dg, %b15dgm : tensor<1152xf32>
    %admgb15dg = stablehlo.multiply %adob1b15dg, %b15ddndg : tensor<1152xf32>
    %admnb15dg = stablehlo.add %admsb15dg, %admgb15dg : tensor<1152xf32>
    %adb2b15dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b15dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb15dg = stablehlo.multiply %adb2b15dg, %b15dgv : tensor<1152xf32>
    %adg2b15dg = stablehlo.multiply %b15ddndg, %b15ddndg : tensor<1152xf32>
    %advgb15dg = stablehlo.multiply %adob2b15dg, %adg2b15dg : tensor<1152xf32>
    %advnb15dg = stablehlo.add %advsb15dg, %advgb15dg : tensor<1152xf32>
    %adbc1b15dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b15dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb15dg = stablehlo.divide %admnb15dg, %adbc1b15dg : tensor<1152xf32>
    %advhb15dg = stablehlo.divide %advnb15dg, %adbc2b15dg : tensor<1152xf32>
    %adlrb15dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb15dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb15dg = stablehlo.sqrt %advhb15dg : tensor<1152xf32>
    %addenb15dg = stablehlo.add %adsqb15dg, %adepsb15dg : tensor<1152xf32>
    %adratb15dg = stablehlo.divide %admhb15dg, %addenb15dg : tensor<1152xf32>
    %adstb15dg = stablehlo.multiply %adlrb15dg, %adratb15dg : tensor<1152xf32>
    %adsubb15dg = stablehlo.subtract %b15dg, %adstb15dg : tensor<1152xf32>
    %adwdb15dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb15dg = stablehlo.multiply %adwdb15dg, %adlrb15dg : tensor<1152xf32>
    %adwdpb15dg = stablehlo.multiply %adwdlrb15dg, %b15dg : tensor<1152xf32>
    %adnewb15dg = stablehlo.subtract %adsubb15dg, %adwdpb15dg : tensor<1152xf32>
    %adb1b15dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b15dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb15dbt = stablehlo.multiply %adb1b15dbt, %b15dbtm : tensor<1152xf32>
    %admgb15dbt = stablehlo.multiply %adob1b15dbt, %b15ddndb : tensor<1152xf32>
    %admnb15dbt = stablehlo.add %admsb15dbt, %admgb15dbt : tensor<1152xf32>
    %adb2b15dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b15dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb15dbt = stablehlo.multiply %adb2b15dbt, %b15dbtv : tensor<1152xf32>
    %adg2b15dbt = stablehlo.multiply %b15ddndb, %b15ddndb : tensor<1152xf32>
    %advgb15dbt = stablehlo.multiply %adob2b15dbt, %adg2b15dbt : tensor<1152xf32>
    %advnb15dbt = stablehlo.add %advsb15dbt, %advgb15dbt : tensor<1152xf32>
    %adbc1b15dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b15dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb15dbt = stablehlo.divide %admnb15dbt, %adbc1b15dbt : tensor<1152xf32>
    %advhb15dbt = stablehlo.divide %advnb15dbt, %adbc2b15dbt : tensor<1152xf32>
    %adlrb15dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb15dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb15dbt = stablehlo.sqrt %advhb15dbt : tensor<1152xf32>
    %addenb15dbt = stablehlo.add %adsqb15dbt, %adepsb15dbt : tensor<1152xf32>
    %adratb15dbt = stablehlo.divide %admhb15dbt, %addenb15dbt : tensor<1152xf32>
    %adstb15dbt = stablehlo.multiply %adlrb15dbt, %adratb15dbt : tensor<1152xf32>
    %adsubb15dbt = stablehlo.subtract %b15dbt, %adstb15dbt : tensor<1152xf32>
    %adwdb15dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb15dbt = stablehlo.multiply %adwdb15dbt, %adlrb15dbt : tensor<1152xf32>
    %adwdpb15dbt = stablehlo.multiply %adwdlrb15dbt, %b15dbt : tensor<1152xf32>
    %adnewb15dbt = stablehlo.subtract %adsubb15dbt, %adwdpb15dbt : tensor<1152xf32>
    %adb1b15zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adob1b15zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %admsb15zW1 = stablehlo.multiply %adb1b15zW1, %b15zW1m : tensor<1152x48xf32>
    %admgb15zW1 = stablehlo.multiply %adob1b15zW1, %b15zdWs1 : tensor<1152x48xf32>
    %admnb15zW1 = stablehlo.add %admsb15zW1, %admgb15zW1 : tensor<1152x48xf32>
    %adb2b15zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adob2b15zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %advsb15zW1 = stablehlo.multiply %adb2b15zW1, %b15zW1v : tensor<1152x48xf32>
    %adg2b15zW1 = stablehlo.multiply %b15zdWs1, %b15zdWs1 : tensor<1152x48xf32>
    %advgb15zW1 = stablehlo.multiply %adob2b15zW1, %adg2b15zW1 : tensor<1152x48xf32>
    %advnb15zW1 = stablehlo.add %advsb15zW1, %advgb15zW1 : tensor<1152x48xf32>
    %adbc1b15zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adbc2b15zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %admhb15zW1 = stablehlo.divide %admnb15zW1, %adbc1b15zW1 : tensor<1152x48xf32>
    %advhb15zW1 = stablehlo.divide %advnb15zW1, %adbc2b15zW1 : tensor<1152x48xf32>
    %adlrb15zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adepsb15zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adsqb15zW1 = stablehlo.sqrt %advhb15zW1 : tensor<1152x48xf32>
    %addenb15zW1 = stablehlo.add %adsqb15zW1, %adepsb15zW1 : tensor<1152x48xf32>
    %adratb15zW1 = stablehlo.divide %admhb15zW1, %addenb15zW1 : tensor<1152x48xf32>
    %adstb15zW1 = stablehlo.multiply %adlrb15zW1, %adratb15zW1 : tensor<1152x48xf32>
    %adsubb15zW1 = stablehlo.subtract %b15zW1, %adstb15zW1 : tensor<1152x48xf32>
    %adwdb15zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adwdlrb15zW1 = stablehlo.multiply %adwdb15zW1, %adlrb15zW1 : tensor<1152x48xf32>
    %adwdpb15zW1 = stablehlo.multiply %adwdlrb15zW1, %b15zW1 : tensor<1152x48xf32>
    %adnewb15zW1 = stablehlo.subtract %adsubb15zW1, %adwdpb15zW1 : tensor<1152x48xf32>
    %adb1b15zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adob1b15zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %admsb15zb1 = stablehlo.multiply %adb1b15zb1, %b15zb1m : tensor<48xf32>
    %admgb15zb1 = stablehlo.multiply %adob1b15zb1, %b15zdbs1 : tensor<48xf32>
    %admnb15zb1 = stablehlo.add %admsb15zb1, %admgb15zb1 : tensor<48xf32>
    %adb2b15zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adob2b15zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %advsb15zb1 = stablehlo.multiply %adb2b15zb1, %b15zb1v : tensor<48xf32>
    %adg2b15zb1 = stablehlo.multiply %b15zdbs1, %b15zdbs1 : tensor<48xf32>
    %advgb15zb1 = stablehlo.multiply %adob2b15zb1, %adg2b15zb1 : tensor<48xf32>
    %advnb15zb1 = stablehlo.add %advsb15zb1, %advgb15zb1 : tensor<48xf32>
    %adbc1b15zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adbc2b15zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %admhb15zb1 = stablehlo.divide %admnb15zb1, %adbc1b15zb1 : tensor<48xf32>
    %advhb15zb1 = stablehlo.divide %advnb15zb1, %adbc2b15zb1 : tensor<48xf32>
    %adlrb15zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adepsb15zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adsqb15zb1 = stablehlo.sqrt %advhb15zb1 : tensor<48xf32>
    %addenb15zb1 = stablehlo.add %adsqb15zb1, %adepsb15zb1 : tensor<48xf32>
    %adratb15zb1 = stablehlo.divide %admhb15zb1, %addenb15zb1 : tensor<48xf32>
    %adstb15zb1 = stablehlo.multiply %adlrb15zb1, %adratb15zb1 : tensor<48xf32>
    %adsubb15zb1 = stablehlo.subtract %b15zb1, %adstb15zb1 : tensor<48xf32>
    %adwdb15zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adwdlrb15zb1 = stablehlo.multiply %adwdb15zb1, %adlrb15zb1 : tensor<48xf32>
    %adwdpb15zb1 = stablehlo.multiply %adwdlrb15zb1, %b15zb1 : tensor<48xf32>
    %adnewb15zb1 = stablehlo.subtract %adsubb15zb1, %adwdpb15zb1 : tensor<48xf32>
    %adb1b15zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adob1b15zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %admsb15zW2 = stablehlo.multiply %adb1b15zW2, %b15zW2m : tensor<48x1152xf32>
    %admgb15zW2 = stablehlo.multiply %adob1b15zW2, %b15zdWs2 : tensor<48x1152xf32>
    %admnb15zW2 = stablehlo.add %admsb15zW2, %admgb15zW2 : tensor<48x1152xf32>
    %adb2b15zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adob2b15zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %advsb15zW2 = stablehlo.multiply %adb2b15zW2, %b15zW2v : tensor<48x1152xf32>
    %adg2b15zW2 = stablehlo.multiply %b15zdWs2, %b15zdWs2 : tensor<48x1152xf32>
    %advgb15zW2 = stablehlo.multiply %adob2b15zW2, %adg2b15zW2 : tensor<48x1152xf32>
    %advnb15zW2 = stablehlo.add %advsb15zW2, %advgb15zW2 : tensor<48x1152xf32>
    %adbc1b15zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adbc2b15zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %admhb15zW2 = stablehlo.divide %admnb15zW2, %adbc1b15zW2 : tensor<48x1152xf32>
    %advhb15zW2 = stablehlo.divide %advnb15zW2, %adbc2b15zW2 : tensor<48x1152xf32>
    %adlrb15zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adepsb15zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adsqb15zW2 = stablehlo.sqrt %advhb15zW2 : tensor<48x1152xf32>
    %addenb15zW2 = stablehlo.add %adsqb15zW2, %adepsb15zW2 : tensor<48x1152xf32>
    %adratb15zW2 = stablehlo.divide %admhb15zW2, %addenb15zW2 : tensor<48x1152xf32>
    %adstb15zW2 = stablehlo.multiply %adlrb15zW2, %adratb15zW2 : tensor<48x1152xf32>
    %adsubb15zW2 = stablehlo.subtract %b15zW2, %adstb15zW2 : tensor<48x1152xf32>
    %adwdb15zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adwdlrb15zW2 = stablehlo.multiply %adwdb15zW2, %adlrb15zW2 : tensor<48x1152xf32>
    %adwdpb15zW2 = stablehlo.multiply %adwdlrb15zW2, %b15zW2 : tensor<48x1152xf32>
    %adnewb15zW2 = stablehlo.subtract %adsubb15zW2, %adwdpb15zW2 : tensor<48x1152xf32>
    %adb1b15zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b15zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb15zb2 = stablehlo.multiply %adb1b15zb2, %b15zb2m : tensor<1152xf32>
    %admgb15zb2 = stablehlo.multiply %adob1b15zb2, %b15zdbs2 : tensor<1152xf32>
    %admnb15zb2 = stablehlo.add %admsb15zb2, %admgb15zb2 : tensor<1152xf32>
    %adb2b15zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b15zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb15zb2 = stablehlo.multiply %adb2b15zb2, %b15zb2v : tensor<1152xf32>
    %adg2b15zb2 = stablehlo.multiply %b15zdbs2, %b15zdbs2 : tensor<1152xf32>
    %advgb15zb2 = stablehlo.multiply %adob2b15zb2, %adg2b15zb2 : tensor<1152xf32>
    %advnb15zb2 = stablehlo.add %advsb15zb2, %advgb15zb2 : tensor<1152xf32>
    %adbc1b15zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b15zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb15zb2 = stablehlo.divide %admnb15zb2, %adbc1b15zb2 : tensor<1152xf32>
    %advhb15zb2 = stablehlo.divide %advnb15zb2, %adbc2b15zb2 : tensor<1152xf32>
    %adlrb15zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb15zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb15zb2 = stablehlo.sqrt %advhb15zb2 : tensor<1152xf32>
    %addenb15zb2 = stablehlo.add %adsqb15zb2, %adepsb15zb2 : tensor<1152xf32>
    %adratb15zb2 = stablehlo.divide %admhb15zb2, %addenb15zb2 : tensor<1152xf32>
    %adstb15zb2 = stablehlo.multiply %adlrb15zb2, %adratb15zb2 : tensor<1152xf32>
    %adsubb15zb2 = stablehlo.subtract %b15zb2, %adstb15zb2 : tensor<1152xf32>
    %adwdb15zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb15zb2 = stablehlo.multiply %adwdb15zb2, %adlrb15zb2 : tensor<1152xf32>
    %adwdpb15zb2 = stablehlo.multiply %adwdlrb15zb2, %b15zb2 : tensor<1152xf32>
    %adnewb15zb2 = stablehlo.subtract %adsubb15zb2, %adwdpb15zb2 : tensor<1152xf32>
    %adb1b15pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adob1b15pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %admsb15pW = stablehlo.multiply %adb1b15pW, %b15pWm : tensor<192x1152x1x1xf32>
    %admgb15pW = stablehlo.multiply %adob1b15pW, %b15dpW : tensor<192x1152x1x1xf32>
    %admnb15pW = stablehlo.add %admsb15pW, %admgb15pW : tensor<192x1152x1x1xf32>
    %adb2b15pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adob2b15pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %advsb15pW = stablehlo.multiply %adb2b15pW, %b15pWv : tensor<192x1152x1x1xf32>
    %adg2b15pW = stablehlo.multiply %b15dpW, %b15dpW : tensor<192x1152x1x1xf32>
    %advgb15pW = stablehlo.multiply %adob2b15pW, %adg2b15pW : tensor<192x1152x1x1xf32>
    %advnb15pW = stablehlo.add %advsb15pW, %advgb15pW : tensor<192x1152x1x1xf32>
    %adbc1b15pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adbc2b15pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %admhb15pW = stablehlo.divide %admnb15pW, %adbc1b15pW : tensor<192x1152x1x1xf32>
    %advhb15pW = stablehlo.divide %advnb15pW, %adbc2b15pW : tensor<192x1152x1x1xf32>
    %adlrb15pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adepsb15pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adsqb15pW = stablehlo.sqrt %advhb15pW : tensor<192x1152x1x1xf32>
    %addenb15pW = stablehlo.add %adsqb15pW, %adepsb15pW : tensor<192x1152x1x1xf32>
    %adratb15pW = stablehlo.divide %admhb15pW, %addenb15pW : tensor<192x1152x1x1xf32>
    %adstb15pW = stablehlo.multiply %adlrb15pW, %adratb15pW : tensor<192x1152x1x1xf32>
    %adsubb15pW = stablehlo.subtract %b15pW, %adstb15pW : tensor<192x1152x1x1xf32>
    %adwdb15pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1152x1x1xf32>
    %adwdlrb15pW = stablehlo.multiply %adwdb15pW, %adlrb15pW : tensor<192x1152x1x1xf32>
    %adwdpb15pW = stablehlo.multiply %adwdlrb15pW, %b15pW : tensor<192x1152x1x1xf32>
    %adnewb15pW = stablehlo.subtract %adsubb15pW, %adwdpb15pW : tensor<192x1152x1x1xf32>
    %adb1b15pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b15pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb15pb = stablehlo.multiply %adb1b15pb, %b15pbm : tensor<192xf32>
    %admgb15pb = stablehlo.multiply %adob1b15pb, %b15dpb : tensor<192xf32>
    %admnb15pb = stablehlo.add %admsb15pb, %admgb15pb : tensor<192xf32>
    %adb2b15pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b15pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb15pb = stablehlo.multiply %adb2b15pb, %b15pbv : tensor<192xf32>
    %adg2b15pb = stablehlo.multiply %b15dpb, %b15dpb : tensor<192xf32>
    %advgb15pb = stablehlo.multiply %adob2b15pb, %adg2b15pb : tensor<192xf32>
    %advnb15pb = stablehlo.add %advsb15pb, %advgb15pb : tensor<192xf32>
    %adbc1b15pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b15pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb15pb = stablehlo.divide %admnb15pb, %adbc1b15pb : tensor<192xf32>
    %advhb15pb = stablehlo.divide %advnb15pb, %adbc2b15pb : tensor<192xf32>
    %adlrb15pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb15pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb15pb = stablehlo.sqrt %advhb15pb : tensor<192xf32>
    %addenb15pb = stablehlo.add %adsqb15pb, %adepsb15pb : tensor<192xf32>
    %adratb15pb = stablehlo.divide %admhb15pb, %addenb15pb : tensor<192xf32>
    %adstb15pb = stablehlo.multiply %adlrb15pb, %adratb15pb : tensor<192xf32>
    %adsubb15pb = stablehlo.subtract %b15pb, %adstb15pb : tensor<192xf32>
    %adwdb15pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb15pb = stablehlo.multiply %adwdb15pb, %adlrb15pb : tensor<192xf32>
    %adwdpb15pb = stablehlo.multiply %adwdlrb15pb, %b15pb : tensor<192xf32>
    %adnewb15pb = stablehlo.subtract %adsubb15pb, %adwdpb15pb : tensor<192xf32>
    %adb1b15pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b15pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb15pg = stablehlo.multiply %adb1b15pg, %b15pgm : tensor<192xf32>
    %admgb15pg = stablehlo.multiply %adob1b15pg, %b15dpndg : tensor<192xf32>
    %admnb15pg = stablehlo.add %admsb15pg, %admgb15pg : tensor<192xf32>
    %adb2b15pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b15pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb15pg = stablehlo.multiply %adb2b15pg, %b15pgv : tensor<192xf32>
    %adg2b15pg = stablehlo.multiply %b15dpndg, %b15dpndg : tensor<192xf32>
    %advgb15pg = stablehlo.multiply %adob2b15pg, %adg2b15pg : tensor<192xf32>
    %advnb15pg = stablehlo.add %advsb15pg, %advgb15pg : tensor<192xf32>
    %adbc1b15pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b15pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb15pg = stablehlo.divide %admnb15pg, %adbc1b15pg : tensor<192xf32>
    %advhb15pg = stablehlo.divide %advnb15pg, %adbc2b15pg : tensor<192xf32>
    %adlrb15pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb15pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb15pg = stablehlo.sqrt %advhb15pg : tensor<192xf32>
    %addenb15pg = stablehlo.add %adsqb15pg, %adepsb15pg : tensor<192xf32>
    %adratb15pg = stablehlo.divide %admhb15pg, %addenb15pg : tensor<192xf32>
    %adstb15pg = stablehlo.multiply %adlrb15pg, %adratb15pg : tensor<192xf32>
    %adsubb15pg = stablehlo.subtract %b15pg, %adstb15pg : tensor<192xf32>
    %adwdb15pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb15pg = stablehlo.multiply %adwdb15pg, %adlrb15pg : tensor<192xf32>
    %adwdpb15pg = stablehlo.multiply %adwdlrb15pg, %b15pg : tensor<192xf32>
    %adnewb15pg = stablehlo.subtract %adsubb15pg, %adwdpb15pg : tensor<192xf32>
    %adb1b15pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b15pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb15pbt = stablehlo.multiply %adb1b15pbt, %b15pbtm : tensor<192xf32>
    %admgb15pbt = stablehlo.multiply %adob1b15pbt, %b15dpndb : tensor<192xf32>
    %admnb15pbt = stablehlo.add %admsb15pbt, %admgb15pbt : tensor<192xf32>
    %adb2b15pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b15pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb15pbt = stablehlo.multiply %adb2b15pbt, %b15pbtv : tensor<192xf32>
    %adg2b15pbt = stablehlo.multiply %b15dpndb, %b15dpndb : tensor<192xf32>
    %advgb15pbt = stablehlo.multiply %adob2b15pbt, %adg2b15pbt : tensor<192xf32>
    %advnb15pbt = stablehlo.add %advsb15pbt, %advgb15pbt : tensor<192xf32>
    %adbc1b15pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b15pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb15pbt = stablehlo.divide %admnb15pbt, %adbc1b15pbt : tensor<192xf32>
    %advhb15pbt = stablehlo.divide %advnb15pbt, %adbc2b15pbt : tensor<192xf32>
    %adlrb15pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb15pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb15pbt = stablehlo.sqrt %advhb15pbt : tensor<192xf32>
    %addenb15pbt = stablehlo.add %adsqb15pbt, %adepsb15pbt : tensor<192xf32>
    %adratb15pbt = stablehlo.divide %admhb15pbt, %addenb15pbt : tensor<192xf32>
    %adstb15pbt = stablehlo.multiply %adlrb15pbt, %adratb15pbt : tensor<192xf32>
    %adsubb15pbt = stablehlo.subtract %b15pbt, %adstb15pbt : tensor<192xf32>
    %adwdb15pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb15pbt = stablehlo.multiply %adwdb15pbt, %adlrb15pbt : tensor<192xf32>
    %adwdpb15pbt = stablehlo.multiply %adwdlrb15pbt, %b15pbt : tensor<192xf32>
    %adnewb15pbt = stablehlo.subtract %adsubb15pbt, %adwdpb15pbt : tensor<192xf32>
    %adb1b16eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adob1b16eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %admsb16eW = stablehlo.multiply %adb1b16eW, %b16eWm : tensor<1152x192x1x1xf32>
    %admgb16eW = stablehlo.multiply %adob1b16eW, %b16deW : tensor<1152x192x1x1xf32>
    %admnb16eW = stablehlo.add %admsb16eW, %admgb16eW : tensor<1152x192x1x1xf32>
    %adb2b16eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adob2b16eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %advsb16eW = stablehlo.multiply %adb2b16eW, %b16eWv : tensor<1152x192x1x1xf32>
    %adg2b16eW = stablehlo.multiply %b16deW, %b16deW : tensor<1152x192x1x1xf32>
    %advgb16eW = stablehlo.multiply %adob2b16eW, %adg2b16eW : tensor<1152x192x1x1xf32>
    %advnb16eW = stablehlo.add %advsb16eW, %advgb16eW : tensor<1152x192x1x1xf32>
    %adbc1b16eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adbc2b16eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %admhb16eW = stablehlo.divide %admnb16eW, %adbc1b16eW : tensor<1152x192x1x1xf32>
    %advhb16eW = stablehlo.divide %advnb16eW, %adbc2b16eW : tensor<1152x192x1x1xf32>
    %adlrb16eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adepsb16eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adsqb16eW = stablehlo.sqrt %advhb16eW : tensor<1152x192x1x1xf32>
    %addenb16eW = stablehlo.add %adsqb16eW, %adepsb16eW : tensor<1152x192x1x1xf32>
    %adratb16eW = stablehlo.divide %admhb16eW, %addenb16eW : tensor<1152x192x1x1xf32>
    %adstb16eW = stablehlo.multiply %adlrb16eW, %adratb16eW : tensor<1152x192x1x1xf32>
    %adsubb16eW = stablehlo.subtract %b16eW, %adstb16eW : tensor<1152x192x1x1xf32>
    %adwdb16eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x192x1x1xf32>
    %adwdlrb16eW = stablehlo.multiply %adwdb16eW, %adlrb16eW : tensor<1152x192x1x1xf32>
    %adwdpb16eW = stablehlo.multiply %adwdlrb16eW, %b16eW : tensor<1152x192x1x1xf32>
    %adnewb16eW = stablehlo.subtract %adsubb16eW, %adwdpb16eW : tensor<1152x192x1x1xf32>
    %adb1b16eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b16eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb16eb = stablehlo.multiply %adb1b16eb, %b16ebm : tensor<1152xf32>
    %admgb16eb = stablehlo.multiply %adob1b16eb, %b16deb : tensor<1152xf32>
    %admnb16eb = stablehlo.add %admsb16eb, %admgb16eb : tensor<1152xf32>
    %adb2b16eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b16eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb16eb = stablehlo.multiply %adb2b16eb, %b16ebv : tensor<1152xf32>
    %adg2b16eb = stablehlo.multiply %b16deb, %b16deb : tensor<1152xf32>
    %advgb16eb = stablehlo.multiply %adob2b16eb, %adg2b16eb : tensor<1152xf32>
    %advnb16eb = stablehlo.add %advsb16eb, %advgb16eb : tensor<1152xf32>
    %adbc1b16eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b16eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb16eb = stablehlo.divide %admnb16eb, %adbc1b16eb : tensor<1152xf32>
    %advhb16eb = stablehlo.divide %advnb16eb, %adbc2b16eb : tensor<1152xf32>
    %adlrb16eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb16eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb16eb = stablehlo.sqrt %advhb16eb : tensor<1152xf32>
    %addenb16eb = stablehlo.add %adsqb16eb, %adepsb16eb : tensor<1152xf32>
    %adratb16eb = stablehlo.divide %admhb16eb, %addenb16eb : tensor<1152xf32>
    %adstb16eb = stablehlo.multiply %adlrb16eb, %adratb16eb : tensor<1152xf32>
    %adsubb16eb = stablehlo.subtract %b16eb, %adstb16eb : tensor<1152xf32>
    %adwdb16eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb16eb = stablehlo.multiply %adwdb16eb, %adlrb16eb : tensor<1152xf32>
    %adwdpb16eb = stablehlo.multiply %adwdlrb16eb, %b16eb : tensor<1152xf32>
    %adnewb16eb = stablehlo.subtract %adsubb16eb, %adwdpb16eb : tensor<1152xf32>
    %adb1b16eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b16eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb16eg = stablehlo.multiply %adb1b16eg, %b16egm : tensor<1152xf32>
    %admgb16eg = stablehlo.multiply %adob1b16eg, %b16dendg : tensor<1152xf32>
    %admnb16eg = stablehlo.add %admsb16eg, %admgb16eg : tensor<1152xf32>
    %adb2b16eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b16eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb16eg = stablehlo.multiply %adb2b16eg, %b16egv : tensor<1152xf32>
    %adg2b16eg = stablehlo.multiply %b16dendg, %b16dendg : tensor<1152xf32>
    %advgb16eg = stablehlo.multiply %adob2b16eg, %adg2b16eg : tensor<1152xf32>
    %advnb16eg = stablehlo.add %advsb16eg, %advgb16eg : tensor<1152xf32>
    %adbc1b16eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b16eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb16eg = stablehlo.divide %admnb16eg, %adbc1b16eg : tensor<1152xf32>
    %advhb16eg = stablehlo.divide %advnb16eg, %adbc2b16eg : tensor<1152xf32>
    %adlrb16eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb16eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb16eg = stablehlo.sqrt %advhb16eg : tensor<1152xf32>
    %addenb16eg = stablehlo.add %adsqb16eg, %adepsb16eg : tensor<1152xf32>
    %adratb16eg = stablehlo.divide %admhb16eg, %addenb16eg : tensor<1152xf32>
    %adstb16eg = stablehlo.multiply %adlrb16eg, %adratb16eg : tensor<1152xf32>
    %adsubb16eg = stablehlo.subtract %b16eg, %adstb16eg : tensor<1152xf32>
    %adwdb16eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb16eg = stablehlo.multiply %adwdb16eg, %adlrb16eg : tensor<1152xf32>
    %adwdpb16eg = stablehlo.multiply %adwdlrb16eg, %b16eg : tensor<1152xf32>
    %adnewb16eg = stablehlo.subtract %adsubb16eg, %adwdpb16eg : tensor<1152xf32>
    %adb1b16ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b16ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb16ebt = stablehlo.multiply %adb1b16ebt, %b16ebtm : tensor<1152xf32>
    %admgb16ebt = stablehlo.multiply %adob1b16ebt, %b16dendb : tensor<1152xf32>
    %admnb16ebt = stablehlo.add %admsb16ebt, %admgb16ebt : tensor<1152xf32>
    %adb2b16ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b16ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb16ebt = stablehlo.multiply %adb2b16ebt, %b16ebtv : tensor<1152xf32>
    %adg2b16ebt = stablehlo.multiply %b16dendb, %b16dendb : tensor<1152xf32>
    %advgb16ebt = stablehlo.multiply %adob2b16ebt, %adg2b16ebt : tensor<1152xf32>
    %advnb16ebt = stablehlo.add %advsb16ebt, %advgb16ebt : tensor<1152xf32>
    %adbc1b16ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b16ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb16ebt = stablehlo.divide %admnb16ebt, %adbc1b16ebt : tensor<1152xf32>
    %advhb16ebt = stablehlo.divide %advnb16ebt, %adbc2b16ebt : tensor<1152xf32>
    %adlrb16ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb16ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb16ebt = stablehlo.sqrt %advhb16ebt : tensor<1152xf32>
    %addenb16ebt = stablehlo.add %adsqb16ebt, %adepsb16ebt : tensor<1152xf32>
    %adratb16ebt = stablehlo.divide %admhb16ebt, %addenb16ebt : tensor<1152xf32>
    %adstb16ebt = stablehlo.multiply %adlrb16ebt, %adratb16ebt : tensor<1152xf32>
    %adsubb16ebt = stablehlo.subtract %b16ebt, %adstb16ebt : tensor<1152xf32>
    %adwdb16ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb16ebt = stablehlo.multiply %adwdb16ebt, %adlrb16ebt : tensor<1152xf32>
    %adwdpb16ebt = stablehlo.multiply %adwdlrb16ebt, %b16ebt : tensor<1152xf32>
    %adnewb16ebt = stablehlo.subtract %adsubb16ebt, %adwdpb16ebt : tensor<1152xf32>
    %adb1b16dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %adob1b16dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %admsb16dW = stablehlo.multiply %adb1b16dW, %b16dWm : tensor<1152x1x3x3xf32>
    %admgb16dW = stablehlo.multiply %adob1b16dW, %b16ddW : tensor<1152x1x3x3xf32>
    %admnb16dW = stablehlo.add %admsb16dW, %admgb16dW : tensor<1152x1x3x3xf32>
    %adb2b16dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %adob2b16dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %advsb16dW = stablehlo.multiply %adb2b16dW, %b16dWv : tensor<1152x1x3x3xf32>
    %adg2b16dW = stablehlo.multiply %b16ddW, %b16ddW : tensor<1152x1x3x3xf32>
    %advgb16dW = stablehlo.multiply %adob2b16dW, %adg2b16dW : tensor<1152x1x3x3xf32>
    %advnb16dW = stablehlo.add %advsb16dW, %advgb16dW : tensor<1152x1x3x3xf32>
    %adbc1b16dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %adbc2b16dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %admhb16dW = stablehlo.divide %admnb16dW, %adbc1b16dW : tensor<1152x1x3x3xf32>
    %advhb16dW = stablehlo.divide %advnb16dW, %adbc2b16dW : tensor<1152x1x3x3xf32>
    %adlrb16dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %adepsb16dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %adsqb16dW = stablehlo.sqrt %advhb16dW : tensor<1152x1x3x3xf32>
    %addenb16dW = stablehlo.add %adsqb16dW, %adepsb16dW : tensor<1152x1x3x3xf32>
    %adratb16dW = stablehlo.divide %admhb16dW, %addenb16dW : tensor<1152x1x3x3xf32>
    %adstb16dW = stablehlo.multiply %adlrb16dW, %adratb16dW : tensor<1152x1x3x3xf32>
    %adsubb16dW = stablehlo.subtract %b16dW, %adstb16dW : tensor<1152x1x3x3xf32>
    %adwdb16dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x1x3x3xf32>
    %adwdlrb16dW = stablehlo.multiply %adwdb16dW, %adlrb16dW : tensor<1152x1x3x3xf32>
    %adwdpb16dW = stablehlo.multiply %adwdlrb16dW, %b16dW : tensor<1152x1x3x3xf32>
    %adnewb16dW = stablehlo.subtract %adsubb16dW, %adwdpb16dW : tensor<1152x1x3x3xf32>
    %adb1b16db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b16db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb16db = stablehlo.multiply %adb1b16db, %b16dbm : tensor<1152xf32>
    %admgb16db = stablehlo.multiply %adob1b16db, %b16ddb : tensor<1152xf32>
    %admnb16db = stablehlo.add %admsb16db, %admgb16db : tensor<1152xf32>
    %adb2b16db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b16db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb16db = stablehlo.multiply %adb2b16db, %b16dbv : tensor<1152xf32>
    %adg2b16db = stablehlo.multiply %b16ddb, %b16ddb : tensor<1152xf32>
    %advgb16db = stablehlo.multiply %adob2b16db, %adg2b16db : tensor<1152xf32>
    %advnb16db = stablehlo.add %advsb16db, %advgb16db : tensor<1152xf32>
    %adbc1b16db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b16db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb16db = stablehlo.divide %admnb16db, %adbc1b16db : tensor<1152xf32>
    %advhb16db = stablehlo.divide %advnb16db, %adbc2b16db : tensor<1152xf32>
    %adlrb16db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb16db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb16db = stablehlo.sqrt %advhb16db : tensor<1152xf32>
    %addenb16db = stablehlo.add %adsqb16db, %adepsb16db : tensor<1152xf32>
    %adratb16db = stablehlo.divide %admhb16db, %addenb16db : tensor<1152xf32>
    %adstb16db = stablehlo.multiply %adlrb16db, %adratb16db : tensor<1152xf32>
    %adsubb16db = stablehlo.subtract %b16db, %adstb16db : tensor<1152xf32>
    %adwdb16db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb16db = stablehlo.multiply %adwdb16db, %adlrb16db : tensor<1152xf32>
    %adwdpb16db = stablehlo.multiply %adwdlrb16db, %b16db : tensor<1152xf32>
    %adnewb16db = stablehlo.subtract %adsubb16db, %adwdpb16db : tensor<1152xf32>
    %adb1b16dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b16dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb16dg = stablehlo.multiply %adb1b16dg, %b16dgm : tensor<1152xf32>
    %admgb16dg = stablehlo.multiply %adob1b16dg, %b16ddndg : tensor<1152xf32>
    %admnb16dg = stablehlo.add %admsb16dg, %admgb16dg : tensor<1152xf32>
    %adb2b16dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b16dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb16dg = stablehlo.multiply %adb2b16dg, %b16dgv : tensor<1152xf32>
    %adg2b16dg = stablehlo.multiply %b16ddndg, %b16ddndg : tensor<1152xf32>
    %advgb16dg = stablehlo.multiply %adob2b16dg, %adg2b16dg : tensor<1152xf32>
    %advnb16dg = stablehlo.add %advsb16dg, %advgb16dg : tensor<1152xf32>
    %adbc1b16dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b16dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb16dg = stablehlo.divide %admnb16dg, %adbc1b16dg : tensor<1152xf32>
    %advhb16dg = stablehlo.divide %advnb16dg, %adbc2b16dg : tensor<1152xf32>
    %adlrb16dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb16dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb16dg = stablehlo.sqrt %advhb16dg : tensor<1152xf32>
    %addenb16dg = stablehlo.add %adsqb16dg, %adepsb16dg : tensor<1152xf32>
    %adratb16dg = stablehlo.divide %admhb16dg, %addenb16dg : tensor<1152xf32>
    %adstb16dg = stablehlo.multiply %adlrb16dg, %adratb16dg : tensor<1152xf32>
    %adsubb16dg = stablehlo.subtract %b16dg, %adstb16dg : tensor<1152xf32>
    %adwdb16dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb16dg = stablehlo.multiply %adwdb16dg, %adlrb16dg : tensor<1152xf32>
    %adwdpb16dg = stablehlo.multiply %adwdlrb16dg, %b16dg : tensor<1152xf32>
    %adnewb16dg = stablehlo.subtract %adsubb16dg, %adwdpb16dg : tensor<1152xf32>
    %adb1b16dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b16dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb16dbt = stablehlo.multiply %adb1b16dbt, %b16dbtm : tensor<1152xf32>
    %admgb16dbt = stablehlo.multiply %adob1b16dbt, %b16ddndb : tensor<1152xf32>
    %admnb16dbt = stablehlo.add %admsb16dbt, %admgb16dbt : tensor<1152xf32>
    %adb2b16dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b16dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb16dbt = stablehlo.multiply %adb2b16dbt, %b16dbtv : tensor<1152xf32>
    %adg2b16dbt = stablehlo.multiply %b16ddndb, %b16ddndb : tensor<1152xf32>
    %advgb16dbt = stablehlo.multiply %adob2b16dbt, %adg2b16dbt : tensor<1152xf32>
    %advnb16dbt = stablehlo.add %advsb16dbt, %advgb16dbt : tensor<1152xf32>
    %adbc1b16dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b16dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb16dbt = stablehlo.divide %admnb16dbt, %adbc1b16dbt : tensor<1152xf32>
    %advhb16dbt = stablehlo.divide %advnb16dbt, %adbc2b16dbt : tensor<1152xf32>
    %adlrb16dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb16dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb16dbt = stablehlo.sqrt %advhb16dbt : tensor<1152xf32>
    %addenb16dbt = stablehlo.add %adsqb16dbt, %adepsb16dbt : tensor<1152xf32>
    %adratb16dbt = stablehlo.divide %admhb16dbt, %addenb16dbt : tensor<1152xf32>
    %adstb16dbt = stablehlo.multiply %adlrb16dbt, %adratb16dbt : tensor<1152xf32>
    %adsubb16dbt = stablehlo.subtract %b16dbt, %adstb16dbt : tensor<1152xf32>
    %adwdb16dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb16dbt = stablehlo.multiply %adwdb16dbt, %adlrb16dbt : tensor<1152xf32>
    %adwdpb16dbt = stablehlo.multiply %adwdlrb16dbt, %b16dbt : tensor<1152xf32>
    %adnewb16dbt = stablehlo.subtract %adsubb16dbt, %adwdpb16dbt : tensor<1152xf32>
    %adb1b16zW1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adob1b16zW1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %admsb16zW1 = stablehlo.multiply %adb1b16zW1, %b16zW1m : tensor<1152x48xf32>
    %admgb16zW1 = stablehlo.multiply %adob1b16zW1, %b16zdWs1 : tensor<1152x48xf32>
    %admnb16zW1 = stablehlo.add %admsb16zW1, %admgb16zW1 : tensor<1152x48xf32>
    %adb2b16zW1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adob2b16zW1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %advsb16zW1 = stablehlo.multiply %adb2b16zW1, %b16zW1v : tensor<1152x48xf32>
    %adg2b16zW1 = stablehlo.multiply %b16zdWs1, %b16zdWs1 : tensor<1152x48xf32>
    %advgb16zW1 = stablehlo.multiply %adob2b16zW1, %adg2b16zW1 : tensor<1152x48xf32>
    %advnb16zW1 = stablehlo.add %advsb16zW1, %advgb16zW1 : tensor<1152x48xf32>
    %adbc1b16zW1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adbc2b16zW1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %admhb16zW1 = stablehlo.divide %admnb16zW1, %adbc1b16zW1 : tensor<1152x48xf32>
    %advhb16zW1 = stablehlo.divide %advnb16zW1, %adbc2b16zW1 : tensor<1152x48xf32>
    %adlrb16zW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adepsb16zW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adsqb16zW1 = stablehlo.sqrt %advhb16zW1 : tensor<1152x48xf32>
    %addenb16zW1 = stablehlo.add %adsqb16zW1, %adepsb16zW1 : tensor<1152x48xf32>
    %adratb16zW1 = stablehlo.divide %admhb16zW1, %addenb16zW1 : tensor<1152x48xf32>
    %adstb16zW1 = stablehlo.multiply %adlrb16zW1, %adratb16zW1 : tensor<1152x48xf32>
    %adsubb16zW1 = stablehlo.subtract %b16zW1, %adstb16zW1 : tensor<1152x48xf32>
    %adwdb16zW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152x48xf32>
    %adwdlrb16zW1 = stablehlo.multiply %adwdb16zW1, %adlrb16zW1 : tensor<1152x48xf32>
    %adwdpb16zW1 = stablehlo.multiply %adwdlrb16zW1, %b16zW1 : tensor<1152x48xf32>
    %adnewb16zW1 = stablehlo.subtract %adsubb16zW1, %adwdpb16zW1 : tensor<1152x48xf32>
    %adb1b16zb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adob1b16zb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %admsb16zb1 = stablehlo.multiply %adb1b16zb1, %b16zb1m : tensor<48xf32>
    %admgb16zb1 = stablehlo.multiply %adob1b16zb1, %b16zdbs1 : tensor<48xf32>
    %admnb16zb1 = stablehlo.add %admsb16zb1, %admgb16zb1 : tensor<48xf32>
    %adb2b16zb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adob2b16zb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %advsb16zb1 = stablehlo.multiply %adb2b16zb1, %b16zb1v : tensor<48xf32>
    %adg2b16zb1 = stablehlo.multiply %b16zdbs1, %b16zdbs1 : tensor<48xf32>
    %advgb16zb1 = stablehlo.multiply %adob2b16zb1, %adg2b16zb1 : tensor<48xf32>
    %advnb16zb1 = stablehlo.add %advsb16zb1, %advgb16zb1 : tensor<48xf32>
    %adbc1b16zb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adbc2b16zb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %admhb16zb1 = stablehlo.divide %admnb16zb1, %adbc1b16zb1 : tensor<48xf32>
    %advhb16zb1 = stablehlo.divide %advnb16zb1, %adbc2b16zb1 : tensor<48xf32>
    %adlrb16zb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adepsb16zb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adsqb16zb1 = stablehlo.sqrt %advhb16zb1 : tensor<48xf32>
    %addenb16zb1 = stablehlo.add %adsqb16zb1, %adepsb16zb1 : tensor<48xf32>
    %adratb16zb1 = stablehlo.divide %admhb16zb1, %addenb16zb1 : tensor<48xf32>
    %adstb16zb1 = stablehlo.multiply %adlrb16zb1, %adratb16zb1 : tensor<48xf32>
    %adsubb16zb1 = stablehlo.subtract %b16zb1, %adstb16zb1 : tensor<48xf32>
    %adwdb16zb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<48xf32>
    %adwdlrb16zb1 = stablehlo.multiply %adwdb16zb1, %adlrb16zb1 : tensor<48xf32>
    %adwdpb16zb1 = stablehlo.multiply %adwdlrb16zb1, %b16zb1 : tensor<48xf32>
    %adnewb16zb1 = stablehlo.subtract %adsubb16zb1, %adwdpb16zb1 : tensor<48xf32>
    %adb1b16zW2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adob1b16zW2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %admsb16zW2 = stablehlo.multiply %adb1b16zW2, %b16zW2m : tensor<48x1152xf32>
    %admgb16zW2 = stablehlo.multiply %adob1b16zW2, %b16zdWs2 : tensor<48x1152xf32>
    %admnb16zW2 = stablehlo.add %admsb16zW2, %admgb16zW2 : tensor<48x1152xf32>
    %adb2b16zW2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adob2b16zW2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %advsb16zW2 = stablehlo.multiply %adb2b16zW2, %b16zW2v : tensor<48x1152xf32>
    %adg2b16zW2 = stablehlo.multiply %b16zdWs2, %b16zdWs2 : tensor<48x1152xf32>
    %advgb16zW2 = stablehlo.multiply %adob2b16zW2, %adg2b16zW2 : tensor<48x1152xf32>
    %advnb16zW2 = stablehlo.add %advsb16zW2, %advgb16zW2 : tensor<48x1152xf32>
    %adbc1b16zW2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adbc2b16zW2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %admhb16zW2 = stablehlo.divide %admnb16zW2, %adbc1b16zW2 : tensor<48x1152xf32>
    %advhb16zW2 = stablehlo.divide %advnb16zW2, %adbc2b16zW2 : tensor<48x1152xf32>
    %adlrb16zW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adepsb16zW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adsqb16zW2 = stablehlo.sqrt %advhb16zW2 : tensor<48x1152xf32>
    %addenb16zW2 = stablehlo.add %adsqb16zW2, %adepsb16zW2 : tensor<48x1152xf32>
    %adratb16zW2 = stablehlo.divide %admhb16zW2, %addenb16zW2 : tensor<48x1152xf32>
    %adstb16zW2 = stablehlo.multiply %adlrb16zW2, %adratb16zW2 : tensor<48x1152xf32>
    %adsubb16zW2 = stablehlo.subtract %b16zW2, %adstb16zW2 : tensor<48x1152xf32>
    %adwdb16zW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<48x1152xf32>
    %adwdlrb16zW2 = stablehlo.multiply %adwdb16zW2, %adlrb16zW2 : tensor<48x1152xf32>
    %adwdpb16zW2 = stablehlo.multiply %adwdlrb16zW2, %b16zW2 : tensor<48x1152xf32>
    %adnewb16zW2 = stablehlo.subtract %adsubb16zW2, %adwdpb16zW2 : tensor<48x1152xf32>
    %adb1b16zb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob1b16zb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admsb16zb2 = stablehlo.multiply %adb1b16zb2, %b16zb2m : tensor<1152xf32>
    %admgb16zb2 = stablehlo.multiply %adob1b16zb2, %b16zdbs2 : tensor<1152xf32>
    %admnb16zb2 = stablehlo.add %admsb16zb2, %admgb16zb2 : tensor<1152xf32>
    %adb2b16zb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adob2b16zb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %advsb16zb2 = stablehlo.multiply %adb2b16zb2, %b16zb2v : tensor<1152xf32>
    %adg2b16zb2 = stablehlo.multiply %b16zdbs2, %b16zdbs2 : tensor<1152xf32>
    %advgb16zb2 = stablehlo.multiply %adob2b16zb2, %adg2b16zb2 : tensor<1152xf32>
    %advnb16zb2 = stablehlo.add %advsb16zb2, %advgb16zb2 : tensor<1152xf32>
    %adbc1b16zb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adbc2b16zb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %admhb16zb2 = stablehlo.divide %admnb16zb2, %adbc1b16zb2 : tensor<1152xf32>
    %advhb16zb2 = stablehlo.divide %advnb16zb2, %adbc2b16zb2 : tensor<1152xf32>
    %adlrb16zb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adepsb16zb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adsqb16zb2 = stablehlo.sqrt %advhb16zb2 : tensor<1152xf32>
    %addenb16zb2 = stablehlo.add %adsqb16zb2, %adepsb16zb2 : tensor<1152xf32>
    %adratb16zb2 = stablehlo.divide %admhb16zb2, %addenb16zb2 : tensor<1152xf32>
    %adstb16zb2 = stablehlo.multiply %adlrb16zb2, %adratb16zb2 : tensor<1152xf32>
    %adsubb16zb2 = stablehlo.subtract %b16zb2, %adstb16zb2 : tensor<1152xf32>
    %adwdb16zb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1152xf32>
    %adwdlrb16zb2 = stablehlo.multiply %adwdb16zb2, %adlrb16zb2 : tensor<1152xf32>
    %adwdpb16zb2 = stablehlo.multiply %adwdlrb16zb2, %b16zb2 : tensor<1152xf32>
    %adnewb16zb2 = stablehlo.subtract %adsubb16zb2, %adwdpb16zb2 : tensor<1152xf32>
    %adb1b16pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %adob1b16pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %admsb16pW = stablehlo.multiply %adb1b16pW, %b16pWm : tensor<320x1152x1x1xf32>
    %admgb16pW = stablehlo.multiply %adob1b16pW, %b16dpW : tensor<320x1152x1x1xf32>
    %admnb16pW = stablehlo.add %admsb16pW, %admgb16pW : tensor<320x1152x1x1xf32>
    %adb2b16pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %adob2b16pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %advsb16pW = stablehlo.multiply %adb2b16pW, %b16pWv : tensor<320x1152x1x1xf32>
    %adg2b16pW = stablehlo.multiply %b16dpW, %b16dpW : tensor<320x1152x1x1xf32>
    %advgb16pW = stablehlo.multiply %adob2b16pW, %adg2b16pW : tensor<320x1152x1x1xf32>
    %advnb16pW = stablehlo.add %advsb16pW, %advgb16pW : tensor<320x1152x1x1xf32>
    %adbc1b16pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %adbc2b16pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %admhb16pW = stablehlo.divide %admnb16pW, %adbc1b16pW : tensor<320x1152x1x1xf32>
    %advhb16pW = stablehlo.divide %advnb16pW, %adbc2b16pW : tensor<320x1152x1x1xf32>
    %adlrb16pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %adepsb16pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %adsqb16pW = stablehlo.sqrt %advhb16pW : tensor<320x1152x1x1xf32>
    %addenb16pW = stablehlo.add %adsqb16pW, %adepsb16pW : tensor<320x1152x1x1xf32>
    %adratb16pW = stablehlo.divide %admhb16pW, %addenb16pW : tensor<320x1152x1x1xf32>
    %adstb16pW = stablehlo.multiply %adlrb16pW, %adratb16pW : tensor<320x1152x1x1xf32>
    %adsubb16pW = stablehlo.subtract %b16pW, %adstb16pW : tensor<320x1152x1x1xf32>
    %adwdb16pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320x1152x1x1xf32>
    %adwdlrb16pW = stablehlo.multiply %adwdb16pW, %adlrb16pW : tensor<320x1152x1x1xf32>
    %adwdpb16pW = stablehlo.multiply %adwdlrb16pW, %b16pW : tensor<320x1152x1x1xf32>
    %adnewb16pW = stablehlo.subtract %adsubb16pW, %adwdpb16pW : tensor<320x1152x1x1xf32>
    %adb1b16pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adob1b16pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %admsb16pb = stablehlo.multiply %adb1b16pb, %b16pbm : tensor<320xf32>
    %admgb16pb = stablehlo.multiply %adob1b16pb, %b16dpb : tensor<320xf32>
    %admnb16pb = stablehlo.add %admsb16pb, %admgb16pb : tensor<320xf32>
    %adb2b16pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adob2b16pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %advsb16pb = stablehlo.multiply %adb2b16pb, %b16pbv : tensor<320xf32>
    %adg2b16pb = stablehlo.multiply %b16dpb, %b16dpb : tensor<320xf32>
    %advgb16pb = stablehlo.multiply %adob2b16pb, %adg2b16pb : tensor<320xf32>
    %advnb16pb = stablehlo.add %advsb16pb, %advgb16pb : tensor<320xf32>
    %adbc1b16pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adbc2b16pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %admhb16pb = stablehlo.divide %admnb16pb, %adbc1b16pb : tensor<320xf32>
    %advhb16pb = stablehlo.divide %advnb16pb, %adbc2b16pb : tensor<320xf32>
    %adlrb16pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adepsb16pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adsqb16pb = stablehlo.sqrt %advhb16pb : tensor<320xf32>
    %addenb16pb = stablehlo.add %adsqb16pb, %adepsb16pb : tensor<320xf32>
    %adratb16pb = stablehlo.divide %admhb16pb, %addenb16pb : tensor<320xf32>
    %adstb16pb = stablehlo.multiply %adlrb16pb, %adratb16pb : tensor<320xf32>
    %adsubb16pb = stablehlo.subtract %b16pb, %adstb16pb : tensor<320xf32>
    %adwdb16pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adwdlrb16pb = stablehlo.multiply %adwdb16pb, %adlrb16pb : tensor<320xf32>
    %adwdpb16pb = stablehlo.multiply %adwdlrb16pb, %b16pb : tensor<320xf32>
    %adnewb16pb = stablehlo.subtract %adsubb16pb, %adwdpb16pb : tensor<320xf32>
    %adb1b16pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adob1b16pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %admsb16pg = stablehlo.multiply %adb1b16pg, %b16pgm : tensor<320xf32>
    %admgb16pg = stablehlo.multiply %adob1b16pg, %b16dpndg : tensor<320xf32>
    %admnb16pg = stablehlo.add %admsb16pg, %admgb16pg : tensor<320xf32>
    %adb2b16pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adob2b16pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %advsb16pg = stablehlo.multiply %adb2b16pg, %b16pgv : tensor<320xf32>
    %adg2b16pg = stablehlo.multiply %b16dpndg, %b16dpndg : tensor<320xf32>
    %advgb16pg = stablehlo.multiply %adob2b16pg, %adg2b16pg : tensor<320xf32>
    %advnb16pg = stablehlo.add %advsb16pg, %advgb16pg : tensor<320xf32>
    %adbc1b16pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adbc2b16pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %admhb16pg = stablehlo.divide %admnb16pg, %adbc1b16pg : tensor<320xf32>
    %advhb16pg = stablehlo.divide %advnb16pg, %adbc2b16pg : tensor<320xf32>
    %adlrb16pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adepsb16pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adsqb16pg = stablehlo.sqrt %advhb16pg : tensor<320xf32>
    %addenb16pg = stablehlo.add %adsqb16pg, %adepsb16pg : tensor<320xf32>
    %adratb16pg = stablehlo.divide %admhb16pg, %addenb16pg : tensor<320xf32>
    %adstb16pg = stablehlo.multiply %adlrb16pg, %adratb16pg : tensor<320xf32>
    %adsubb16pg = stablehlo.subtract %b16pg, %adstb16pg : tensor<320xf32>
    %adwdb16pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adwdlrb16pg = stablehlo.multiply %adwdb16pg, %adlrb16pg : tensor<320xf32>
    %adwdpb16pg = stablehlo.multiply %adwdlrb16pg, %b16pg : tensor<320xf32>
    %adnewb16pg = stablehlo.subtract %adsubb16pg, %adwdpb16pg : tensor<320xf32>
    %adb1b16pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adob1b16pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %admsb16pbt = stablehlo.multiply %adb1b16pbt, %b16pbtm : tensor<320xf32>
    %admgb16pbt = stablehlo.multiply %adob1b16pbt, %b16dpndb : tensor<320xf32>
    %admnb16pbt = stablehlo.add %admsb16pbt, %admgb16pbt : tensor<320xf32>
    %adb2b16pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adob2b16pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %advsb16pbt = stablehlo.multiply %adb2b16pbt, %b16pbtv : tensor<320xf32>
    %adg2b16pbt = stablehlo.multiply %b16dpndb, %b16dpndb : tensor<320xf32>
    %advgb16pbt = stablehlo.multiply %adob2b16pbt, %adg2b16pbt : tensor<320xf32>
    %advnb16pbt = stablehlo.add %advsb16pbt, %advgb16pbt : tensor<320xf32>
    %adbc1b16pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adbc2b16pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %admhb16pbt = stablehlo.divide %admnb16pbt, %adbc1b16pbt : tensor<320xf32>
    %advhb16pbt = stablehlo.divide %advnb16pbt, %adbc2b16pbt : tensor<320xf32>
    %adlrb16pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adepsb16pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adsqb16pbt = stablehlo.sqrt %advhb16pbt : tensor<320xf32>
    %addenb16pbt = stablehlo.add %adsqb16pbt, %adepsb16pbt : tensor<320xf32>
    %adratb16pbt = stablehlo.divide %admhb16pbt, %addenb16pbt : tensor<320xf32>
    %adstb16pbt = stablehlo.multiply %adlrb16pbt, %adratb16pbt : tensor<320xf32>
    %adsubb16pbt = stablehlo.subtract %b16pbt, %adstb16pbt : tensor<320xf32>
    %adwdb16pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %adwdlrb16pbt = stablehlo.multiply %adwdb16pbt, %adlrb16pbt : tensor<320xf32>
    %adwdpb16pbt = stablehlo.multiply %adwdlrb16pbt, %b16pbt : tensor<320xf32>
    %adnewb16pbt = stablehlo.subtract %adsubb16pbt, %adwdpb16pbt : tensor<320xf32>
    %adb1hW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %adob1hW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %admshW = stablehlo.multiply %adb1hW, %hWm : tensor<1280x320x1x1xf32>
    %admghW = stablehlo.multiply %adob1hW, %dhW : tensor<1280x320x1x1xf32>
    %admnhW = stablehlo.add %admshW, %admghW : tensor<1280x320x1x1xf32>
    %adb2hW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %adob2hW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %advshW = stablehlo.multiply %adb2hW, %hWv : tensor<1280x320x1x1xf32>
    %adg2hW = stablehlo.multiply %dhW, %dhW : tensor<1280x320x1x1xf32>
    %advghW = stablehlo.multiply %adob2hW, %adg2hW : tensor<1280x320x1x1xf32>
    %advnhW = stablehlo.add %advshW, %advghW : tensor<1280x320x1x1xf32>
    %adbc1hW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %adbc2hW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %admhhW = stablehlo.divide %admnhW, %adbc1hW : tensor<1280x320x1x1xf32>
    %advhhW = stablehlo.divide %advnhW, %adbc2hW : tensor<1280x320x1x1xf32>
    %adlrhW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %adepshW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %adsqhW = stablehlo.sqrt %advhhW : tensor<1280x320x1x1xf32>
    %addenhW = stablehlo.add %adsqhW, %adepshW : tensor<1280x320x1x1xf32>
    %adrathW = stablehlo.divide %admhhW, %addenhW : tensor<1280x320x1x1xf32>
    %adsthW = stablehlo.multiply %adlrhW, %adrathW : tensor<1280x320x1x1xf32>
    %adsubhW = stablehlo.subtract %hW, %adsthW : tensor<1280x320x1x1xf32>
    %adwdhW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %adwdlrhW = stablehlo.multiply %adwdhW, %adlrhW : tensor<1280x320x1x1xf32>
    %adwdphW = stablehlo.multiply %adwdlrhW, %hW : tensor<1280x320x1x1xf32>
    %adnewhW = stablehlo.subtract %adsubhW, %adwdphW : tensor<1280x320x1x1xf32>
    %adb1hb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adob1hb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %admshb = stablehlo.multiply %adb1hb, %hbm : tensor<1280xf32>
    %admghb = stablehlo.multiply %adob1hb, %dhb : tensor<1280xf32>
    %admnhb = stablehlo.add %admshb, %admghb : tensor<1280xf32>
    %adb2hb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adob2hb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %advshb = stablehlo.multiply %adb2hb, %hbv : tensor<1280xf32>
    %adg2hb = stablehlo.multiply %dhb, %dhb : tensor<1280xf32>
    %advghb = stablehlo.multiply %adob2hb, %adg2hb : tensor<1280xf32>
    %advnhb = stablehlo.add %advshb, %advghb : tensor<1280xf32>
    %adbc1hb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adbc2hb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %admhhb = stablehlo.divide %admnhb, %adbc1hb : tensor<1280xf32>
    %advhhb = stablehlo.divide %advnhb, %adbc2hb : tensor<1280xf32>
    %adlrhb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adepshb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adsqhb = stablehlo.sqrt %advhhb : tensor<1280xf32>
    %addenhb = stablehlo.add %adsqhb, %adepshb : tensor<1280xf32>
    %adrathb = stablehlo.divide %admhhb, %addenhb : tensor<1280xf32>
    %adsthb = stablehlo.multiply %adlrhb, %adrathb : tensor<1280xf32>
    %adsubhb = stablehlo.subtract %hb, %adsthb : tensor<1280xf32>
    %adwdhb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adwdlrhb = stablehlo.multiply %adwdhb, %adlrhb : tensor<1280xf32>
    %adwdphb = stablehlo.multiply %adwdlrhb, %hb : tensor<1280xf32>
    %adnewhb = stablehlo.subtract %adsubhb, %adwdphb : tensor<1280xf32>
    %adb1hg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adob1hg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %admshg = stablehlo.multiply %adb1hg, %hgm : tensor<1280xf32>
    %admghg = stablehlo.multiply %adob1hg, %dhndg : tensor<1280xf32>
    %admnhg = stablehlo.add %admshg, %admghg : tensor<1280xf32>
    %adb2hg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adob2hg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %advshg = stablehlo.multiply %adb2hg, %hgv : tensor<1280xf32>
    %adg2hg = stablehlo.multiply %dhndg, %dhndg : tensor<1280xf32>
    %advghg = stablehlo.multiply %adob2hg, %adg2hg : tensor<1280xf32>
    %advnhg = stablehlo.add %advshg, %advghg : tensor<1280xf32>
    %adbc1hg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adbc2hg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %admhhg = stablehlo.divide %admnhg, %adbc1hg : tensor<1280xf32>
    %advhhg = stablehlo.divide %advnhg, %adbc2hg : tensor<1280xf32>
    %adlrhg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adepshg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adsqhg = stablehlo.sqrt %advhhg : tensor<1280xf32>
    %addenhg = stablehlo.add %adsqhg, %adepshg : tensor<1280xf32>
    %adrathg = stablehlo.divide %admhhg, %addenhg : tensor<1280xf32>
    %adsthg = stablehlo.multiply %adlrhg, %adrathg : tensor<1280xf32>
    %adsubhg = stablehlo.subtract %hg, %adsthg : tensor<1280xf32>
    %adwdhg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adwdlrhg = stablehlo.multiply %adwdhg, %adlrhg : tensor<1280xf32>
    %adwdphg = stablehlo.multiply %adwdlrhg, %hg : tensor<1280xf32>
    %adnewhg = stablehlo.subtract %adsubhg, %adwdphg : tensor<1280xf32>
    %adb1hbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adob1hbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %admshbt = stablehlo.multiply %adb1hbt, %hbtm : tensor<1280xf32>
    %admghbt = stablehlo.multiply %adob1hbt, %dhndb : tensor<1280xf32>
    %admnhbt = stablehlo.add %admshbt, %admghbt : tensor<1280xf32>
    %adb2hbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adob2hbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %advshbt = stablehlo.multiply %adb2hbt, %hbtv : tensor<1280xf32>
    %adg2hbt = stablehlo.multiply %dhndb, %dhndb : tensor<1280xf32>
    %advghbt = stablehlo.multiply %adob2hbt, %adg2hbt : tensor<1280xf32>
    %advnhbt = stablehlo.add %advshbt, %advghbt : tensor<1280xf32>
    %adbc1hbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adbc2hbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %admhhbt = stablehlo.divide %admnhbt, %adbc1hbt : tensor<1280xf32>
    %advhhbt = stablehlo.divide %advnhbt, %adbc2hbt : tensor<1280xf32>
    %adlrhbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adepshbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adsqhbt = stablehlo.sqrt %advhhbt : tensor<1280xf32>
    %addenhbt = stablehlo.add %adsqhbt, %adepshbt : tensor<1280xf32>
    %adrathbt = stablehlo.divide %admhhbt, %addenhbt : tensor<1280xf32>
    %adsthbt = stablehlo.multiply %adlrhbt, %adrathbt : tensor<1280xf32>
    %adsubhbt = stablehlo.subtract %hbt, %adsthbt : tensor<1280xf32>
    %adwdhbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %adwdlrhbt = stablehlo.multiply %adwdhbt, %adlrhbt : tensor<1280xf32>
    %adwdphbt = stablehlo.multiply %adwdlrhbt, %hbt : tensor<1280xf32>
    %adnewhbt = stablehlo.subtract %adsubhbt, %adwdphbt : tensor<1280xf32>
    %adb1Wd = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %adob1Wd = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %admsWd = stablehlo.multiply %adb1Wd, %Wdm : tensor<1280x10xf32>
    %admgWd = stablehlo.multiply %adob1Wd, %dWd : tensor<1280x10xf32>
    %admnWd = stablehlo.add %admsWd, %admgWd : tensor<1280x10xf32>
    %adb2Wd = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %adob2Wd = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %advsWd = stablehlo.multiply %adb2Wd, %Wdv : tensor<1280x10xf32>
    %adg2Wd = stablehlo.multiply %dWd, %dWd : tensor<1280x10xf32>
    %advgWd = stablehlo.multiply %adob2Wd, %adg2Wd : tensor<1280x10xf32>
    %advnWd = stablehlo.add %advsWd, %advgWd : tensor<1280x10xf32>
    %adbc1Wd = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %adbc2Wd = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %admhWd = stablehlo.divide %admnWd, %adbc1Wd : tensor<1280x10xf32>
    %advhWd = stablehlo.divide %advnWd, %adbc2Wd : tensor<1280x10xf32>
    %adlrWd = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %adepsWd = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %adsqWd = stablehlo.sqrt %advhWd : tensor<1280x10xf32>
    %addenWd = stablehlo.add %adsqWd, %adepsWd : tensor<1280x10xf32>
    %adratWd = stablehlo.divide %admhWd, %addenWd : tensor<1280x10xf32>
    %adstWd = stablehlo.multiply %adlrWd, %adratWd : tensor<1280x10xf32>
    %adsubWd = stablehlo.subtract %Wd, %adstWd : tensor<1280x10xf32>
    %adwdWd = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %adwdlrWd = stablehlo.multiply %adwdWd, %adlrWd : tensor<1280x10xf32>
    %adwdpWd = stablehlo.multiply %adwdlrWd, %Wd : tensor<1280x10xf32>
    %adnewWd = stablehlo.subtract %adsubWd, %adwdpWd : tensor<1280x10xf32>
    %adb1bd = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob1bd = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admsbd = stablehlo.multiply %adb1bd, %bdm : tensor<10xf32>
    %admgbd = stablehlo.multiply %adob1bd, %dbd : tensor<10xf32>
    %admnbd = stablehlo.add %admsbd, %admgbd : tensor<10xf32>
    %adb2bd = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob2bd = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %advsbd = stablehlo.multiply %adb2bd, %bdv : tensor<10xf32>
    %adg2bd = stablehlo.multiply %dbd, %dbd : tensor<10xf32>
    %advgbd = stablehlo.multiply %adob2bd, %adg2bd : tensor<10xf32>
    %advnbd = stablehlo.add %advsbd, %advgbd : tensor<10xf32>
    %adbc1bd = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adbc2bd = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admhbd = stablehlo.divide %admnbd, %adbc1bd : tensor<10xf32>
    %advhbd = stablehlo.divide %advnbd, %adbc2bd : tensor<10xf32>
    %adlrbd = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adepsbd = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adsqbd = stablehlo.sqrt %advhbd : tensor<10xf32>
    %addenbd = stablehlo.add %adsqbd, %adepsbd : tensor<10xf32>
    %adratbd = stablehlo.divide %admhbd, %addenbd : tensor<10xf32>
    %adstbd = stablehlo.multiply %adlrbd, %adratbd : tensor<10xf32>
    %adsubbd = stablehlo.subtract %bd, %adstbd : tensor<10xf32>
    %adwdbd = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adwdlrbd = stablehlo.multiply %adwdbd, %adlrbd : tensor<10xf32>
    %adwdpbd = stablehlo.multiply %adwdlrbd, %bd : tensor<10xf32>
    %adnewbd = stablehlo.subtract %adsubbd, %adwdpbd : tensor<10xf32>
    %stnbnnf = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %stnbnmu = stablehlo.divide %stnsmr, %stnbnnf : tensor<32xf32>
    %stnbnvar = stablehlo.divide %stnvsr, %stnbnnf : tensor<32xf32>
    %b1dnbnnf = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %b1dnbnmu = stablehlo.divide %b1dnsmr, %b1dnbnnf : tensor<32xf32>
    %b1dnbnvar = stablehlo.divide %b1dnvsr, %b1dnbnnf : tensor<32xf32>
    %b1pnbnnf = stablehlo.constant dense<401408.0> : tensor<16xf32>
    %b1pnbnmu = stablehlo.divide %b1pnsmr, %b1pnbnnf : tensor<16xf32>
    %b1pnbnvar = stablehlo.divide %b1pnvsr, %b1pnbnnf : tensor<16xf32>
    %b2enbnnf = stablehlo.constant dense<401408.0> : tensor<96xf32>
    %b2enbnmu = stablehlo.divide %b2ensmr, %b2enbnnf : tensor<96xf32>
    %b2enbnvar = stablehlo.divide %b2envsr, %b2enbnnf : tensor<96xf32>
    %b2dnbnnf = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %b2dnbnmu = stablehlo.divide %b2dnsmr, %b2dnbnnf : tensor<96xf32>
    %b2dnbnvar = stablehlo.divide %b2dnvsr, %b2dnbnnf : tensor<96xf32>
    %b2pnbnnf = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %b2pnbnmu = stablehlo.divide %b2pnsmr, %b2pnbnnf : tensor<24xf32>
    %b2pnbnvar = stablehlo.divide %b2pnvsr, %b2pnbnnf : tensor<24xf32>
    %b3enbnnf = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %b3enbnmu = stablehlo.divide %b3ensmr, %b3enbnnf : tensor<144xf32>
    %b3enbnvar = stablehlo.divide %b3envsr, %b3enbnnf : tensor<144xf32>
    %b3dnbnnf = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %b3dnbnmu = stablehlo.divide %b3dnsmr, %b3dnbnnf : tensor<144xf32>
    %b3dnbnvar = stablehlo.divide %b3dnvsr, %b3dnbnnf : tensor<144xf32>
    %b3pnbnnf = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %b3pnbnmu = stablehlo.divide %b3pnsmr, %b3pnbnnf : tensor<24xf32>
    %b3pnbnvar = stablehlo.divide %b3pnvsr, %b3pnbnnf : tensor<24xf32>
    %b4enbnnf = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %b4enbnmu = stablehlo.divide %b4ensmr, %b4enbnnf : tensor<144xf32>
    %b4enbnvar = stablehlo.divide %b4envsr, %b4enbnnf : tensor<144xf32>
    %b4dnbnnf = stablehlo.constant dense<25088.0> : tensor<144xf32>
    %b4dnbnmu = stablehlo.divide %b4dnsmr, %b4dnbnnf : tensor<144xf32>
    %b4dnbnvar = stablehlo.divide %b4dnvsr, %b4dnbnnf : tensor<144xf32>
    %b4pnbnnf = stablehlo.constant dense<25088.0> : tensor<40xf32>
    %b4pnbnmu = stablehlo.divide %b4pnsmr, %b4pnbnnf : tensor<40xf32>
    %b4pnbnvar = stablehlo.divide %b4pnvsr, %b4pnbnnf : tensor<40xf32>
    %b5enbnnf = stablehlo.constant dense<25088.0> : tensor<240xf32>
    %b5enbnmu = stablehlo.divide %b5ensmr, %b5enbnnf : tensor<240xf32>
    %b5enbnvar = stablehlo.divide %b5envsr, %b5enbnnf : tensor<240xf32>
    %b5dnbnnf = stablehlo.constant dense<25088.0> : tensor<240xf32>
    %b5dnbnmu = stablehlo.divide %b5dnsmr, %b5dnbnnf : tensor<240xf32>
    %b5dnbnvar = stablehlo.divide %b5dnvsr, %b5dnbnnf : tensor<240xf32>
    %b5pnbnnf = stablehlo.constant dense<25088.0> : tensor<40xf32>
    %b5pnbnmu = stablehlo.divide %b5pnsmr, %b5pnbnnf : tensor<40xf32>
    %b5pnbnvar = stablehlo.divide %b5pnvsr, %b5pnbnnf : tensor<40xf32>
    %b6enbnnf = stablehlo.constant dense<25088.0> : tensor<240xf32>
    %b6enbnmu = stablehlo.divide %b6ensmr, %b6enbnnf : tensor<240xf32>
    %b6enbnvar = stablehlo.divide %b6envsr, %b6enbnnf : tensor<240xf32>
    %b6dnbnnf = stablehlo.constant dense<6272.0> : tensor<240xf32>
    %b6dnbnmu = stablehlo.divide %b6dnsmr, %b6dnbnnf : tensor<240xf32>
    %b6dnbnvar = stablehlo.divide %b6dnvsr, %b6dnbnnf : tensor<240xf32>
    %b6pnbnnf = stablehlo.constant dense<6272.0> : tensor<80xf32>
    %b6pnbnmu = stablehlo.divide %b6pnsmr, %b6pnbnnf : tensor<80xf32>
    %b6pnbnvar = stablehlo.divide %b6pnvsr, %b6pnbnnf : tensor<80xf32>
    %b7enbnnf = stablehlo.constant dense<6272.0> : tensor<480xf32>
    %b7enbnmu = stablehlo.divide %b7ensmr, %b7enbnnf : tensor<480xf32>
    %b7enbnvar = stablehlo.divide %b7envsr, %b7enbnnf : tensor<480xf32>
    %b7dnbnnf = stablehlo.constant dense<6272.0> : tensor<480xf32>
    %b7dnbnmu = stablehlo.divide %b7dnsmr, %b7dnbnnf : tensor<480xf32>
    %b7dnbnvar = stablehlo.divide %b7dnvsr, %b7dnbnnf : tensor<480xf32>
    %b7pnbnnf = stablehlo.constant dense<6272.0> : tensor<80xf32>
    %b7pnbnmu = stablehlo.divide %b7pnsmr, %b7pnbnnf : tensor<80xf32>
    %b7pnbnvar = stablehlo.divide %b7pnvsr, %b7pnbnnf : tensor<80xf32>
    %b8enbnnf = stablehlo.constant dense<6272.0> : tensor<480xf32>
    %b8enbnmu = stablehlo.divide %b8ensmr, %b8enbnnf : tensor<480xf32>
    %b8enbnvar = stablehlo.divide %b8envsr, %b8enbnnf : tensor<480xf32>
    %b8dnbnnf = stablehlo.constant dense<6272.0> : tensor<480xf32>
    %b8dnbnmu = stablehlo.divide %b8dnsmr, %b8dnbnnf : tensor<480xf32>
    %b8dnbnvar = stablehlo.divide %b8dnvsr, %b8dnbnnf : tensor<480xf32>
    %b8pnbnnf = stablehlo.constant dense<6272.0> : tensor<80xf32>
    %b8pnbnmu = stablehlo.divide %b8pnsmr, %b8pnbnnf : tensor<80xf32>
    %b8pnbnvar = stablehlo.divide %b8pnvsr, %b8pnbnnf : tensor<80xf32>
    %b9enbnnf = stablehlo.constant dense<6272.0> : tensor<480xf32>
    %b9enbnmu = stablehlo.divide %b9ensmr, %b9enbnnf : tensor<480xf32>
    %b9enbnvar = stablehlo.divide %b9envsr, %b9enbnnf : tensor<480xf32>
    %b9dnbnnf = stablehlo.constant dense<6272.0> : tensor<480xf32>
    %b9dnbnmu = stablehlo.divide %b9dnsmr, %b9dnbnnf : tensor<480xf32>
    %b9dnbnvar = stablehlo.divide %b9dnvsr, %b9dnbnnf : tensor<480xf32>
    %b9pnbnnf = stablehlo.constant dense<6272.0> : tensor<112xf32>
    %b9pnbnmu = stablehlo.divide %b9pnsmr, %b9pnbnnf : tensor<112xf32>
    %b9pnbnvar = stablehlo.divide %b9pnvsr, %b9pnbnnf : tensor<112xf32>
    %b10enbnnf = stablehlo.constant dense<6272.0> : tensor<672xf32>
    %b10enbnmu = stablehlo.divide %b10ensmr, %b10enbnnf : tensor<672xf32>
    %b10enbnvar = stablehlo.divide %b10envsr, %b10enbnnf : tensor<672xf32>
    %b10dnbnnf = stablehlo.constant dense<6272.0> : tensor<672xf32>
    %b10dnbnmu = stablehlo.divide %b10dnsmr, %b10dnbnnf : tensor<672xf32>
    %b10dnbnvar = stablehlo.divide %b10dnvsr, %b10dnbnnf : tensor<672xf32>
    %b10pnbnnf = stablehlo.constant dense<6272.0> : tensor<112xf32>
    %b10pnbnmu = stablehlo.divide %b10pnsmr, %b10pnbnnf : tensor<112xf32>
    %b10pnbnvar = stablehlo.divide %b10pnvsr, %b10pnbnnf : tensor<112xf32>
    %b11enbnnf = stablehlo.constant dense<6272.0> : tensor<672xf32>
    %b11enbnmu = stablehlo.divide %b11ensmr, %b11enbnnf : tensor<672xf32>
    %b11enbnvar = stablehlo.divide %b11envsr, %b11enbnnf : tensor<672xf32>
    %b11dnbnnf = stablehlo.constant dense<6272.0> : tensor<672xf32>
    %b11dnbnmu = stablehlo.divide %b11dnsmr, %b11dnbnnf : tensor<672xf32>
    %b11dnbnvar = stablehlo.divide %b11dnvsr, %b11dnbnnf : tensor<672xf32>
    %b11pnbnnf = stablehlo.constant dense<6272.0> : tensor<112xf32>
    %b11pnbnmu = stablehlo.divide %b11pnsmr, %b11pnbnnf : tensor<112xf32>
    %b11pnbnvar = stablehlo.divide %b11pnvsr, %b11pnbnnf : tensor<112xf32>
    %b12enbnnf = stablehlo.constant dense<6272.0> : tensor<672xf32>
    %b12enbnmu = stablehlo.divide %b12ensmr, %b12enbnnf : tensor<672xf32>
    %b12enbnvar = stablehlo.divide %b12envsr, %b12enbnnf : tensor<672xf32>
    %b12dnbnnf = stablehlo.constant dense<1568.0> : tensor<672xf32>
    %b12dnbnmu = stablehlo.divide %b12dnsmr, %b12dnbnnf : tensor<672xf32>
    %b12dnbnvar = stablehlo.divide %b12dnvsr, %b12dnbnnf : tensor<672xf32>
    %b12pnbnnf = stablehlo.constant dense<1568.0> : tensor<192xf32>
    %b12pnbnmu = stablehlo.divide %b12pnsmr, %b12pnbnnf : tensor<192xf32>
    %b12pnbnvar = stablehlo.divide %b12pnvsr, %b12pnbnnf : tensor<192xf32>
    %b13enbnnf = stablehlo.constant dense<1568.0> : tensor<1152xf32>
    %b13enbnmu = stablehlo.divide %b13ensmr, %b13enbnnf : tensor<1152xf32>
    %b13enbnvar = stablehlo.divide %b13envsr, %b13enbnnf : tensor<1152xf32>
    %b13dnbnnf = stablehlo.constant dense<1568.0> : tensor<1152xf32>
    %b13dnbnmu = stablehlo.divide %b13dnsmr, %b13dnbnnf : tensor<1152xf32>
    %b13dnbnvar = stablehlo.divide %b13dnvsr, %b13dnbnnf : tensor<1152xf32>
    %b13pnbnnf = stablehlo.constant dense<1568.0> : tensor<192xf32>
    %b13pnbnmu = stablehlo.divide %b13pnsmr, %b13pnbnnf : tensor<192xf32>
    %b13pnbnvar = stablehlo.divide %b13pnvsr, %b13pnbnnf : tensor<192xf32>
    %b14enbnnf = stablehlo.constant dense<1568.0> : tensor<1152xf32>
    %b14enbnmu = stablehlo.divide %b14ensmr, %b14enbnnf : tensor<1152xf32>
    %b14enbnvar = stablehlo.divide %b14envsr, %b14enbnnf : tensor<1152xf32>
    %b14dnbnnf = stablehlo.constant dense<1568.0> : tensor<1152xf32>
    %b14dnbnmu = stablehlo.divide %b14dnsmr, %b14dnbnnf : tensor<1152xf32>
    %b14dnbnvar = stablehlo.divide %b14dnvsr, %b14dnbnnf : tensor<1152xf32>
    %b14pnbnnf = stablehlo.constant dense<1568.0> : tensor<192xf32>
    %b14pnbnmu = stablehlo.divide %b14pnsmr, %b14pnbnnf : tensor<192xf32>
    %b14pnbnvar = stablehlo.divide %b14pnvsr, %b14pnbnnf : tensor<192xf32>
    %b15enbnnf = stablehlo.constant dense<1568.0> : tensor<1152xf32>
    %b15enbnmu = stablehlo.divide %b15ensmr, %b15enbnnf : tensor<1152xf32>
    %b15enbnvar = stablehlo.divide %b15envsr, %b15enbnnf : tensor<1152xf32>
    %b15dnbnnf = stablehlo.constant dense<1568.0> : tensor<1152xf32>
    %b15dnbnmu = stablehlo.divide %b15dnsmr, %b15dnbnnf : tensor<1152xf32>
    %b15dnbnvar = stablehlo.divide %b15dnvsr, %b15dnbnnf : tensor<1152xf32>
    %b15pnbnnf = stablehlo.constant dense<1568.0> : tensor<192xf32>
    %b15pnbnmu = stablehlo.divide %b15pnsmr, %b15pnbnnf : tensor<192xf32>
    %b15pnbnvar = stablehlo.divide %b15pnvsr, %b15pnbnnf : tensor<192xf32>
    %b16enbnnf = stablehlo.constant dense<1568.0> : tensor<1152xf32>
    %b16enbnmu = stablehlo.divide %b16ensmr, %b16enbnnf : tensor<1152xf32>
    %b16enbnvar = stablehlo.divide %b16envsr, %b16enbnnf : tensor<1152xf32>
    %b16dnbnnf = stablehlo.constant dense<1568.0> : tensor<1152xf32>
    %b16dnbnmu = stablehlo.divide %b16dnsmr, %b16dnbnnf : tensor<1152xf32>
    %b16dnbnvar = stablehlo.divide %b16dnvsr, %b16dnbnnf : tensor<1152xf32>
    %b16pnbnnf = stablehlo.constant dense<1568.0> : tensor<320xf32>
    %b16pnbnmu = stablehlo.divide %b16pnsmr, %b16pnbnnf : tensor<320xf32>
    %b16pnbnvar = stablehlo.divide %b16pnvsr, %b16pnbnnf : tensor<320xf32>
    %hnbnnf = stablehlo.constant dense<1568.0> : tensor<1280xf32>
    %hnbnmu = stablehlo.divide %hnsmr, %hnbnnf : tensor<1280xf32>
    %hnbnvar = stablehlo.divide %hnvsr, %hnbnnf : tensor<1280xf32>
    return %adnewsW, %adnewsb, %adnewsg, %adnewsbt, %adnewb1dW, %adnewb1db, %adnewb1dg, %adnewb1dbt, %adnewb1zW1, %adnewb1zb1, %adnewb1zW2, %adnewb1zb2, %adnewb1pW, %adnewb1pb, %adnewb1pg, %adnewb1pbt, %adnewb2eW, %adnewb2eb, %adnewb2eg, %adnewb2ebt, %adnewb2dW, %adnewb2db, %adnewb2dg, %adnewb2dbt, %adnewb2zW1, %adnewb2zb1, %adnewb2zW2, %adnewb2zb2, %adnewb2pW, %adnewb2pb, %adnewb2pg, %adnewb2pbt, %adnewb3eW, %adnewb3eb, %adnewb3eg, %adnewb3ebt, %adnewb3dW, %adnewb3db, %adnewb3dg, %adnewb3dbt, %adnewb3zW1, %adnewb3zb1, %adnewb3zW2, %adnewb3zb2, %adnewb3pW, %adnewb3pb, %adnewb3pg, %adnewb3pbt, %adnewb4eW, %adnewb4eb, %adnewb4eg, %adnewb4ebt, %adnewb4dW, %adnewb4db, %adnewb4dg, %adnewb4dbt, %adnewb4zW1, %adnewb4zb1, %adnewb4zW2, %adnewb4zb2, %adnewb4pW, %adnewb4pb, %adnewb4pg, %adnewb4pbt, %adnewb5eW, %adnewb5eb, %adnewb5eg, %adnewb5ebt, %adnewb5dW, %adnewb5db, %adnewb5dg, %adnewb5dbt, %adnewb5zW1, %adnewb5zb1, %adnewb5zW2, %adnewb5zb2, %adnewb5pW, %adnewb5pb, %adnewb5pg, %adnewb5pbt, %adnewb6eW, %adnewb6eb, %adnewb6eg, %adnewb6ebt, %adnewb6dW, %adnewb6db, %adnewb6dg, %adnewb6dbt, %adnewb6zW1, %adnewb6zb1, %adnewb6zW2, %adnewb6zb2, %adnewb6pW, %adnewb6pb, %adnewb6pg, %adnewb6pbt, %adnewb7eW, %adnewb7eb, %adnewb7eg, %adnewb7ebt, %adnewb7dW, %adnewb7db, %adnewb7dg, %adnewb7dbt, %adnewb7zW1, %adnewb7zb1, %adnewb7zW2, %adnewb7zb2, %adnewb7pW, %adnewb7pb, %adnewb7pg, %adnewb7pbt, %adnewb8eW, %adnewb8eb, %adnewb8eg, %adnewb8ebt, %adnewb8dW, %adnewb8db, %adnewb8dg, %adnewb8dbt, %adnewb8zW1, %adnewb8zb1, %adnewb8zW2, %adnewb8zb2, %adnewb8pW, %adnewb8pb, %adnewb8pg, %adnewb8pbt, %adnewb9eW, %adnewb9eb, %adnewb9eg, %adnewb9ebt, %adnewb9dW, %adnewb9db, %adnewb9dg, %adnewb9dbt, %adnewb9zW1, %adnewb9zb1, %adnewb9zW2, %adnewb9zb2, %adnewb9pW, %adnewb9pb, %adnewb9pg, %adnewb9pbt, %adnewb10eW, %adnewb10eb, %adnewb10eg, %adnewb10ebt, %adnewb10dW, %adnewb10db, %adnewb10dg, %adnewb10dbt, %adnewb10zW1, %adnewb10zb1, %adnewb10zW2, %adnewb10zb2, %adnewb10pW, %adnewb10pb, %adnewb10pg, %adnewb10pbt, %adnewb11eW, %adnewb11eb, %adnewb11eg, %adnewb11ebt, %adnewb11dW, %adnewb11db, %adnewb11dg, %adnewb11dbt, %adnewb11zW1, %adnewb11zb1, %adnewb11zW2, %adnewb11zb2, %adnewb11pW, %adnewb11pb, %adnewb11pg, %adnewb11pbt, %adnewb12eW, %adnewb12eb, %adnewb12eg, %adnewb12ebt, %adnewb12dW, %adnewb12db, %adnewb12dg, %adnewb12dbt, %adnewb12zW1, %adnewb12zb1, %adnewb12zW2, %adnewb12zb2, %adnewb12pW, %adnewb12pb, %adnewb12pg, %adnewb12pbt, %adnewb13eW, %adnewb13eb, %adnewb13eg, %adnewb13ebt, %adnewb13dW, %adnewb13db, %adnewb13dg, %adnewb13dbt, %adnewb13zW1, %adnewb13zb1, %adnewb13zW2, %adnewb13zb2, %adnewb13pW, %adnewb13pb, %adnewb13pg, %adnewb13pbt, %adnewb14eW, %adnewb14eb, %adnewb14eg, %adnewb14ebt, %adnewb14dW, %adnewb14db, %adnewb14dg, %adnewb14dbt, %adnewb14zW1, %adnewb14zb1, %adnewb14zW2, %adnewb14zb2, %adnewb14pW, %adnewb14pb, %adnewb14pg, %adnewb14pbt, %adnewb15eW, %adnewb15eb, %adnewb15eg, %adnewb15ebt, %adnewb15dW, %adnewb15db, %adnewb15dg, %adnewb15dbt, %adnewb15zW1, %adnewb15zb1, %adnewb15zW2, %adnewb15zb2, %adnewb15pW, %adnewb15pb, %adnewb15pg, %adnewb15pbt, %adnewb16eW, %adnewb16eb, %adnewb16eg, %adnewb16ebt, %adnewb16dW, %adnewb16db, %adnewb16dg, %adnewb16dbt, %adnewb16zW1, %adnewb16zb1, %adnewb16zW2, %adnewb16zb2, %adnewb16pW, %adnewb16pb, %adnewb16pg, %adnewb16pbt, %adnewhW, %adnewhb, %adnewhg, %adnewhbt, %adnewWd, %adnewbd, %admnsW, %admnsb, %admnsg, %admnsbt, %admnb1dW, %admnb1db, %admnb1dg, %admnb1dbt, %admnb1zW1, %admnb1zb1, %admnb1zW2, %admnb1zb2, %admnb1pW, %admnb1pb, %admnb1pg, %admnb1pbt, %admnb2eW, %admnb2eb, %admnb2eg, %admnb2ebt, %admnb2dW, %admnb2db, %admnb2dg, %admnb2dbt, %admnb2zW1, %admnb2zb1, %admnb2zW2, %admnb2zb2, %admnb2pW, %admnb2pb, %admnb2pg, %admnb2pbt, %admnb3eW, %admnb3eb, %admnb3eg, %admnb3ebt, %admnb3dW, %admnb3db, %admnb3dg, %admnb3dbt, %admnb3zW1, %admnb3zb1, %admnb3zW2, %admnb3zb2, %admnb3pW, %admnb3pb, %admnb3pg, %admnb3pbt, %admnb4eW, %admnb4eb, %admnb4eg, %admnb4ebt, %admnb4dW, %admnb4db, %admnb4dg, %admnb4dbt, %admnb4zW1, %admnb4zb1, %admnb4zW2, %admnb4zb2, %admnb4pW, %admnb4pb, %admnb4pg, %admnb4pbt, %admnb5eW, %admnb5eb, %admnb5eg, %admnb5ebt, %admnb5dW, %admnb5db, %admnb5dg, %admnb5dbt, %admnb5zW1, %admnb5zb1, %admnb5zW2, %admnb5zb2, %admnb5pW, %admnb5pb, %admnb5pg, %admnb5pbt, %admnb6eW, %admnb6eb, %admnb6eg, %admnb6ebt, %admnb6dW, %admnb6db, %admnb6dg, %admnb6dbt, %admnb6zW1, %admnb6zb1, %admnb6zW2, %admnb6zb2, %admnb6pW, %admnb6pb, %admnb6pg, %admnb6pbt, %admnb7eW, %admnb7eb, %admnb7eg, %admnb7ebt, %admnb7dW, %admnb7db, %admnb7dg, %admnb7dbt, %admnb7zW1, %admnb7zb1, %admnb7zW2, %admnb7zb2, %admnb7pW, %admnb7pb, %admnb7pg, %admnb7pbt, %admnb8eW, %admnb8eb, %admnb8eg, %admnb8ebt, %admnb8dW, %admnb8db, %admnb8dg, %admnb8dbt, %admnb8zW1, %admnb8zb1, %admnb8zW2, %admnb8zb2, %admnb8pW, %admnb8pb, %admnb8pg, %admnb8pbt, %admnb9eW, %admnb9eb, %admnb9eg, %admnb9ebt, %admnb9dW, %admnb9db, %admnb9dg, %admnb9dbt, %admnb9zW1, %admnb9zb1, %admnb9zW2, %admnb9zb2, %admnb9pW, %admnb9pb, %admnb9pg, %admnb9pbt, %admnb10eW, %admnb10eb, %admnb10eg, %admnb10ebt, %admnb10dW, %admnb10db, %admnb10dg, %admnb10dbt, %admnb10zW1, %admnb10zb1, %admnb10zW2, %admnb10zb2, %admnb10pW, %admnb10pb, %admnb10pg, %admnb10pbt, %admnb11eW, %admnb11eb, %admnb11eg, %admnb11ebt, %admnb11dW, %admnb11db, %admnb11dg, %admnb11dbt, %admnb11zW1, %admnb11zb1, %admnb11zW2, %admnb11zb2, %admnb11pW, %admnb11pb, %admnb11pg, %admnb11pbt, %admnb12eW, %admnb12eb, %admnb12eg, %admnb12ebt, %admnb12dW, %admnb12db, %admnb12dg, %admnb12dbt, %admnb12zW1, %admnb12zb1, %admnb12zW2, %admnb12zb2, %admnb12pW, %admnb12pb, %admnb12pg, %admnb12pbt, %admnb13eW, %admnb13eb, %admnb13eg, %admnb13ebt, %admnb13dW, %admnb13db, %admnb13dg, %admnb13dbt, %admnb13zW1, %admnb13zb1, %admnb13zW2, %admnb13zb2, %admnb13pW, %admnb13pb, %admnb13pg, %admnb13pbt, %admnb14eW, %admnb14eb, %admnb14eg, %admnb14ebt, %admnb14dW, %admnb14db, %admnb14dg, %admnb14dbt, %admnb14zW1, %admnb14zb1, %admnb14zW2, %admnb14zb2, %admnb14pW, %admnb14pb, %admnb14pg, %admnb14pbt, %admnb15eW, %admnb15eb, %admnb15eg, %admnb15ebt, %admnb15dW, %admnb15db, %admnb15dg, %admnb15dbt, %admnb15zW1, %admnb15zb1, %admnb15zW2, %admnb15zb2, %admnb15pW, %admnb15pb, %admnb15pg, %admnb15pbt, %admnb16eW, %admnb16eb, %admnb16eg, %admnb16ebt, %admnb16dW, %admnb16db, %admnb16dg, %admnb16dbt, %admnb16zW1, %admnb16zb1, %admnb16zW2, %admnb16zb2, %admnb16pW, %admnb16pb, %admnb16pg, %admnb16pbt, %admnhW, %admnhb, %admnhg, %admnhbt, %admnWd, %admnbd, %advnsW, %advnsb, %advnsg, %advnsbt, %advnb1dW, %advnb1db, %advnb1dg, %advnb1dbt, %advnb1zW1, %advnb1zb1, %advnb1zW2, %advnb1zb2, %advnb1pW, %advnb1pb, %advnb1pg, %advnb1pbt, %advnb2eW, %advnb2eb, %advnb2eg, %advnb2ebt, %advnb2dW, %advnb2db, %advnb2dg, %advnb2dbt, %advnb2zW1, %advnb2zb1, %advnb2zW2, %advnb2zb2, %advnb2pW, %advnb2pb, %advnb2pg, %advnb2pbt, %advnb3eW, %advnb3eb, %advnb3eg, %advnb3ebt, %advnb3dW, %advnb3db, %advnb3dg, %advnb3dbt, %advnb3zW1, %advnb3zb1, %advnb3zW2, %advnb3zb2, %advnb3pW, %advnb3pb, %advnb3pg, %advnb3pbt, %advnb4eW, %advnb4eb, %advnb4eg, %advnb4ebt, %advnb4dW, %advnb4db, %advnb4dg, %advnb4dbt, %advnb4zW1, %advnb4zb1, %advnb4zW2, %advnb4zb2, %advnb4pW, %advnb4pb, %advnb4pg, %advnb4pbt, %advnb5eW, %advnb5eb, %advnb5eg, %advnb5ebt, %advnb5dW, %advnb5db, %advnb5dg, %advnb5dbt, %advnb5zW1, %advnb5zb1, %advnb5zW2, %advnb5zb2, %advnb5pW, %advnb5pb, %advnb5pg, %advnb5pbt, %advnb6eW, %advnb6eb, %advnb6eg, %advnb6ebt, %advnb6dW, %advnb6db, %advnb6dg, %advnb6dbt, %advnb6zW1, %advnb6zb1, %advnb6zW2, %advnb6zb2, %advnb6pW, %advnb6pb, %advnb6pg, %advnb6pbt, %advnb7eW, %advnb7eb, %advnb7eg, %advnb7ebt, %advnb7dW, %advnb7db, %advnb7dg, %advnb7dbt, %advnb7zW1, %advnb7zb1, %advnb7zW2, %advnb7zb2, %advnb7pW, %advnb7pb, %advnb7pg, %advnb7pbt, %advnb8eW, %advnb8eb, %advnb8eg, %advnb8ebt, %advnb8dW, %advnb8db, %advnb8dg, %advnb8dbt, %advnb8zW1, %advnb8zb1, %advnb8zW2, %advnb8zb2, %advnb8pW, %advnb8pb, %advnb8pg, %advnb8pbt, %advnb9eW, %advnb9eb, %advnb9eg, %advnb9ebt, %advnb9dW, %advnb9db, %advnb9dg, %advnb9dbt, %advnb9zW1, %advnb9zb1, %advnb9zW2, %advnb9zb2, %advnb9pW, %advnb9pb, %advnb9pg, %advnb9pbt, %advnb10eW, %advnb10eb, %advnb10eg, %advnb10ebt, %advnb10dW, %advnb10db, %advnb10dg, %advnb10dbt, %advnb10zW1, %advnb10zb1, %advnb10zW2, %advnb10zb2, %advnb10pW, %advnb10pb, %advnb10pg, %advnb10pbt, %advnb11eW, %advnb11eb, %advnb11eg, %advnb11ebt, %advnb11dW, %advnb11db, %advnb11dg, %advnb11dbt, %advnb11zW1, %advnb11zb1, %advnb11zW2, %advnb11zb2, %advnb11pW, %advnb11pb, %advnb11pg, %advnb11pbt, %advnb12eW, %advnb12eb, %advnb12eg, %advnb12ebt, %advnb12dW, %advnb12db, %advnb12dg, %advnb12dbt, %advnb12zW1, %advnb12zb1, %advnb12zW2, %advnb12zb2, %advnb12pW, %advnb12pb, %advnb12pg, %advnb12pbt, %advnb13eW, %advnb13eb, %advnb13eg, %advnb13ebt, %advnb13dW, %advnb13db, %advnb13dg, %advnb13dbt, %advnb13zW1, %advnb13zb1, %advnb13zW2, %advnb13zb2, %advnb13pW, %advnb13pb, %advnb13pg, %advnb13pbt, %advnb14eW, %advnb14eb, %advnb14eg, %advnb14ebt, %advnb14dW, %advnb14db, %advnb14dg, %advnb14dbt, %advnb14zW1, %advnb14zb1, %advnb14zW2, %advnb14zb2, %advnb14pW, %advnb14pb, %advnb14pg, %advnb14pbt, %advnb15eW, %advnb15eb, %advnb15eg, %advnb15ebt, %advnb15dW, %advnb15db, %advnb15dg, %advnb15dbt, %advnb15zW1, %advnb15zb1, %advnb15zW2, %advnb15zb2, %advnb15pW, %advnb15pb, %advnb15pg, %advnb15pbt, %advnb16eW, %advnb16eb, %advnb16eg, %advnb16ebt, %advnb16dW, %advnb16db, %advnb16dg, %advnb16dbt, %advnb16zW1, %advnb16zb1, %advnb16zW2, %advnb16zb2, %advnb16pW, %advnb16pb, %advnb16pg, %advnb16pbt, %advnhW, %advnhb, %advnhg, %advnhbt, %advnWd, %advnbd, %loss, %bc1, %bc2, %stnbnmu, %stnbnvar, %b1dnbnmu, %b1dnbnvar, %b1pnbnmu, %b1pnbnvar, %b2enbnmu, %b2enbnvar, %b2dnbnmu, %b2dnbnvar, %b2pnbnmu, %b2pnbnvar, %b3enbnmu, %b3enbnvar, %b3dnbnmu, %b3dnbnvar, %b3pnbnmu, %b3pnbnvar, %b4enbnmu, %b4enbnvar, %b4dnbnmu, %b4dnbnvar, %b4pnbnmu, %b4pnbnvar, %b5enbnmu, %b5enbnvar, %b5dnbnmu, %b5dnbnvar, %b5pnbnmu, %b5pnbnvar, %b6enbnmu, %b6enbnvar, %b6dnbnmu, %b6dnbnvar, %b6pnbnmu, %b6pnbnvar, %b7enbnmu, %b7enbnvar, %b7dnbnmu, %b7dnbnvar, %b7pnbnmu, %b7pnbnvar, %b8enbnmu, %b8enbnvar, %b8dnbnmu, %b8dnbnvar, %b8pnbnmu, %b8pnbnvar, %b9enbnmu, %b9enbnvar, %b9dnbnmu, %b9dnbnvar, %b9pnbnmu, %b9pnbnvar, %b10enbnmu, %b10enbnvar, %b10dnbnmu, %b10dnbnvar, %b10pnbnmu, %b10pnbnvar, %b11enbnmu, %b11enbnvar, %b11dnbnmu, %b11dnbnvar, %b11pnbnmu, %b11pnbnvar, %b12enbnmu, %b12enbnvar, %b12dnbnmu, %b12dnbnvar, %b12pnbnmu, %b12pnbnvar, %b13enbnmu, %b13enbnvar, %b13dnbnmu, %b13dnbnvar, %b13pnbnmu, %b13pnbnvar, %b14enbnmu, %b14enbnvar, %b14dnbnmu, %b14dnbnvar, %b14pnbnmu, %b14pnbnvar, %b15enbnmu, %b15enbnvar, %b15dnbnmu, %b15dnbnvar, %b15pnbnmu, %b15pnbnvar, %b16enbnmu, %b16enbnvar, %b16dnbnmu, %b16dnbnvar, %b16pnbnmu, %b16pnbnvar, %hnbnmu, %hnbnvar : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280xf32>, tensor<1280xf32>
  }
}

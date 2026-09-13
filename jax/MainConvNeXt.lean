import Jax

/-! ConvNeXt-T on Imagenette — the JAX reference for Chapter 8's Imagenette run.

    Same body as `MainConvNeXtImagenet.lean` (patchify stem, (3,3,9,3) blocks at
    (96,192,384,768), depthwise-7×7 + channel-LN + inverted bottleneck + GELU +
    LayerScale, head LN), 10 classes, on the shared Imagenette recipe every Part-I
    net trains on: AdamW 1e-3, cosine with a 3-epoch warmup, 80 epochs, batch 32,
    weight decay 1e-4 on every parameter, label smoothing 0.1, random-crop + hflip,
    fp32. None of the ImageNet recipe's regularizers (RandAugment, Mixup/CutMix,
    EMA, drop-path, grad clip) is on, and `cnxInit` is left off so the convs take
    the same fan-out init the verified path uses.

    Written 2026-09-13 as the control for the layer-scale init question: on the
    verified path the paper's 1e-6 lands 81.78% where ones landed 85.45% on the
    same render (`runs/2026-09-13-convnext-imagenette-ls1e-6/`). This is the
    reference at 1e-6 (`emitLayerScaleInit`) on the same recipe. -/

def convNeXtTiny : NetSpec where
  name := "ConvNeXt-T"
  imageH := 224
  imageW := 224
  layers := [
    .convNextStem 3 96 4,              -- patchify: 4×4 s4 conv → channel-LN
    .convNextStage 96 3 .ln .gelu,     -- stage 1: 3 blocks @ 96   (56×56)
    .convNextDownsample 96 192,        -- 56 → 28
    .convNextStage 192 3 .ln .gelu,    -- stage 2: 3 blocks @ 192
    .convNextDownsample 192 384,       -- 28 → 14
    .convNextStage 384 9 .ln .gelu,    -- stage 3: 9 blocks @ 384
    .convNextDownsample 384 768,       -- 14 → 7
    .convNextStage 768 3 .ln .gelu,    -- stage 4: 3 blocks @ 768
    .globalAvgPool,
    .layerNorm 768,                    -- head LN: GAP → LN → Linear, as the verified render
    .dense 768 10 .identity            -- 10-class head
  ]

/-- The shared Imagenette recipe (Chapters 5–9's Run-it-first), nothing else. -/
def convNeXtTinyConfig : TrainConfig where
  learningRate   := 0.001
  batchSize      := 32
  epochs         := 80
  useAdam        := true
  weightDecay    := 0.0001
  cosineDecay    := true
  warmupEpochs   := 3
  augment        := true
  labelSmoothing := 0.1
  bf16           := false

#eval convNeXtTiny.validate!

def main (args : List String) : IO Unit :=
  runJax convNeXtTiny convNeXtTinyConfig .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_convnext_tiny.py"

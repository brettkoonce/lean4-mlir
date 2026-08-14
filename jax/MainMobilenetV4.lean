import Jax

/-! MobileNetV4-Conv-M on Imagenette
    Universal Inverted Bottleneck blocks with varied DW configurations.
    Conv-only variant (no attention).

    ⚠ Until 2026-08-14 this file's table was Conv-S-SIZED (14 UIB blocks, one 1×1 head conv,
    ~4.1M) while its `name` said "Medium" — the misnomer `VerifiedNets.lean` called out. It is
    now the real Conv-M table, transcribed from `MainMobilenetV4Imagenet.lean` (which was always
    faithful Conv-M) with the 10-class head kept. `RESULTS.md`'s 84.58% belongs to the OLD Conv-S
    table and is tagged there as such; this spec has no Imagenette accuracy run of its own yet.

    The verified peer `mobilenetv4Verified` moved in the same commit, and it has to: the ties
    (`scripts/mnv4_forward_tie.py`, `scripts/grad_tie.py --net mnv4`) read this file's generated
    output, so a divergence here silently compares two different networks. -/

def mobilenetV4Medium : NetSpec where
  name := "MobileNetV4-Conv-M"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,                     -- 224→112  stem
    -- stage 0 (FusedIB): er_r1_k3_s2_e4_c48
    .fusedMbConv 32 48 4 3 2 1 false,           -- 112→56
    -- stage 1: 2× UIB
    .uib  48  80 4 2 3 5,                        -- 56→28   ExtraDW
    .uib  80  80 2 1 3 3,                        --         ExtraDW
    -- stage 2: 8× UIB
    .uib  80 160 6 2 3 5,                        -- 28→14   ExtraDW
    .uib 160 160 4 1 3 3,                        --         ExtraDW
    .uib 160 160 4 1 3 3,                        --         ExtraDW
    .uib 160 160 4 1 3 5,                        --         ExtraDW
    .uib 160 160 4 1 3 3,                        --         ExtraDW
    .uib 160 160 4 1 3 0,                        --         ConvNeXt
    .uib 160 160 2 1 0 0,                        --         FFN
    .uib 160 160 4 1 3 0,                        --         ConvNeXt
    -- stage 3: 11× UIB
    .uib 160 256 6 2 5 5,                        -- 14→7    ExtraDW
    .uib 256 256 4 1 5 5,                        --         ExtraDW
    .uib 256 256 4 1 3 5,                        --         ExtraDW
    .uib 256 256 4 1 3 5,                        --         ExtraDW
    .uib 256 256 4 1 0 0,                        --         FFN
    .uib 256 256 4 1 3 0,                        --         ConvNeXt
    .uib 256 256 2 1 3 5,                        --         ExtraDW
    .uib 256 256 4 1 5 5,                        --         ExtraDW
    .uib 256 256 4 1 0 0,                        --         FFN
    .uib 256 256 4 1 0 0,                        --         FFN
    .uib 256 256 2 1 5 0,                        --         ConvNeXt
    -- head: cn_r1_k1_s1_c960 → conv_head 1280 → GAP → FC
    .convBn 256 960 1 1 .same,                   -- 7  cn to 960
    .convBn 960 1280 1 1 .same,                  -- 7  conv_head (num_features)
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def mobilenetV4Config : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval mobilenetV4Medium.validate!

def main (args : List String) : IO Unit :=
  runJax mobilenetV4Medium mobilenetV4Config .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_mobilenet_v4.py"

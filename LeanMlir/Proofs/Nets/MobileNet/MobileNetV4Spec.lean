/-! # MobileNetV4-Conv-M's block table

`UibSpec` (one Universal Inverted Bottleneck row) and `mnv4Blocks` (the 21 rows of Conv-M), the one
transcription of the net's layout. The renderer (`MobileNetV4RenderB`) folds over it to
emit the artifacts, and the proof chain from `MobileNetV4BackB0.lean` on folds over it to state the
net, so the proofs read the table without importing the renderer. Names are in `Proofs.StableHLO`,
where the renderer first defined them. -/

namespace Proofs.StableHLO

/-- **One row of the MobileNetV4-Conv-M block table.** `h` is the block's OUTPUT spatial size, so
    a `stride2` block reads its input at `2h`. -/
structure UibSpec where
  /-- parameter-name prefix: `"1"` … `"21"`. -/
  p : String
  ic : Nat
  oc : Nat
  expand : Nat
  /-- pre-depthwise kernel, `0` = absent. -/
  preDWk : Nat
  /-- post-depthwise kernel, `0` = absent. -/
  postDWk : Nat
  h : Nat
  stride2 : Bool
deriving Inhabited, DecidableEq

/-- **THE BLOCK TABLE — transcribed once, from [`jax/MainMobilenetV4.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainMobilenetV4.lean).**

    Everything downstream folds over this list: the parameter signature, the BN stat slots, the
    forward chain, the backward chain and the running-statistic recomputes. Two readings of the
    layout that diverged would type-check alike (same ops, same channel counts, same types), so
    the rows are written once.

    Families in order (Conv-**M**): ExtraDW ×7, ConvNeXt, FFN, ConvNeXt, ExtraDW ×4, FFN, ConvNeXt,
    ExtraDW ×2, FFN ×2, ConvNeXt — 13 ExtraDW / 4 ConvNeXt / 4 FFN, and no IB row. Spatial ladder 56 → 28 → 14 → 7.

    Checked against timm 1.0.28 (`mobilenetv4_conv_medium`, walking `model.blocks[1:4]`):
    all 21 rows agree on `(ic, oc, expand, preDWk, postDWk, h, stride2)`. The `#guard`s in
    [`Proofs/Nets/MobileNet/MobileNetV4BackB0.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Nets/MobileNet/MobileNetV4BackB0.lean) pin that reading; they are derived from timm rather
    than re-read off this table, or they would gate nothing. What the table cannot say — which
    depthwise carries a stride (the post-DW when both exist; every strided row has both, and a
    `#guard` in the renderer holds that), and that the pre-DW is BN only — is the renderer's, and
    `scripts/parity/mnv4_timm_parity.py` pins it against timm on shared weights. -/
def mnv4Blocks : List UibSpec :=
  [ ⟨"1",   48,  80, 4, 3, 5, 28, true⟩,   -- ExtraDW  56→28
    ⟨"2",   80,  80, 2, 3, 3, 28, false⟩,  -- ExtraDW  28
    ⟨"3",   80, 160, 6, 3, 5, 14, true⟩,   -- ExtraDW  28→14
    ⟨"4",  160, 160, 4, 3, 3, 14, false⟩,  -- ExtraDW  14
    ⟨"5",  160, 160, 4, 3, 3, 14, false⟩,  -- ExtraDW  14
    ⟨"6",  160, 160, 4, 3, 5, 14, false⟩,  -- ExtraDW  14
    ⟨"7",  160, 160, 4, 3, 3, 14, false⟩,  -- ExtraDW  14
    ⟨"8",  160, 160, 4, 3, 0, 14, false⟩,  -- ConvNeXt 14
    ⟨"9",  160, 160, 2, 0, 0, 14, false⟩,  -- FFN      14
    ⟨"10", 160, 160, 4, 3, 0, 14, false⟩,  -- ConvNeXt 14
    ⟨"11", 160, 256, 6, 5, 5,  7, true⟩,   -- ExtraDW  14→7
    ⟨"12", 256, 256, 4, 5, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"13", 256, 256, 4, 3, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"14", 256, 256, 4, 3, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"15", 256, 256, 4, 0, 0,  7, false⟩,  -- FFN      7
    ⟨"16", 256, 256, 4, 3, 0,  7, false⟩,  -- ConvNeXt 7
    ⟨"17", 256, 256, 2, 3, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"18", 256, 256, 4, 5, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"19", 256, 256, 4, 0, 0,  7, false⟩,  -- FFN      7
    ⟨"20", 256, 256, 4, 0, 0,  7, false⟩,  -- FFN      7
    ⟨"21", 256, 256, 2, 5, 0,  7, false⟩ ] -- ConvNeXt 7

end Proofs.StableHLO

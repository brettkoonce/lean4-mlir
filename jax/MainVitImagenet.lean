import Jax

/-! Vision Transformer (ViT-Tiny) on full 1000-class ImageNet — bf16
    mixed precision, 2-GPU data-parallel (tfds streaming). Same ViT-Tiny
    spec as `MainVit.lean` but with a 1000-class head and the `.imagenet`
    dataset.

    bf16 routes every matmul (patch embed, attention QKV/scores/out, MLP,
    head) through bfloat16 while keeping LayerNorm/softmax/GELU in fp32 —
    measured ~2.7× on the isolated ViT-Ti matmuls on gfx1100, where
    convnets see no bf16 benefit. See reference_bf16_gfx1100_conv_vs_gemm.
    Note: at ViT-Tiny scale the run may be input-bound on tfds aug. -/

def vitTinyImagenet : NetSpec where
  name := "ViT-Tiny (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP 768
    .dense 192 1000 .identity             -- 1000-class head
  ]

/-- DeiT-flavored 80-epoch recipe: AdamW, lr 5e-4 at batch 512 (256/device
    on 2 GPUs), 5-epoch warmup + cosine, weight decay 0.05, label smoothing
    0.1, augmentation on. bf16 matmuls. -/
def vitTinyImagenetConfig : TrainConfig where
  learningRate   := 0.0005
  batchSize      := 512
  epochs         := 80
  useAdam        := true
  weightDecay    := 0.05
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true
  labelSmoothing := 0.1
  bf16           := true

#eval vitTinyImagenet.validate!

def main (args : List String) : IO Unit :=
  runJax vitTinyImagenet vitTinyImagenetConfig .imagenet
    (args.head? |>.getD "data/imagenet")
    "generated_vit_tiny_imagenet.py"

import LeanMlir.LEBytes
/-! # The verified trainers' packed-parameter layouts

One namespace per net family (`MlpLayout`, `CnnLayout`, `CifarLayout`, `ResNet34Layout`,
`MobileNetV2Layout`, `EfficientNetLayout`, `ConvNeXtLayout`, `ViTLayout`): each net's
`(dims, initKind)` list in the train step's argument order, its sizes, and the packed shape
descriptors (`packShapes`, `packXShape`) the runtime passes to the FFI. Pure data — importing only `LEBytes` —
so the spec side (`VerifiedNetsCore`'s `#guard spec.toSpecs == XLayout.specs`) can read the tables
without importing the runtime; `IreeRuntime` re-exports them. -/

/- Sizes for the packed-params layout. -/
namespace MlpLayout
def nW0 : Nat := 784 * 512  -- 401408
def nb0 : Nat := 512
def nW1 : Nat := 512 * 512  -- 262144
def nb1 : Nat := 512
def nW2 : Nat := 512 * 10   -- 5120
def nb2 : Nat := 10
def nParams : Nat := nW0 + nb0 + nW1 + nb1 + nW2 + nb2  -- 669706
def lossIdx : Nat := nParams
end MlpLayout

/-- Pack param shape descriptors: `[nParams, rank0, d0..., rank1, d1..., ...]` as int32 LE. -/
def packShapes (shapes : Array (Array Nat)) : ByteArray := Id.run do
  let mut ba := pushU32LE .empty shapes.size
  for shape in shapes do
    ba := pushU32LE ba shape.size
    for d in shape do ba := pushU32LE ba d
  return ba

/-- Pack a single shape: `[rank, d0, d1, ...]` as int32 LE (for x input). -/
def packXShape (dims : Array Nat) : ByteArray := Id.run do
  let mut ba := pushU32LE .empty dims.size
  for d in dims do ba := pushU32LE ba d
  return ba

namespace CnnLayout
def paramShapes : Array (Array Nat) := #[
  #[32, 1, 3, 3], #[32],          -- conv0
  #[32, 32, 3, 3], #[32],         -- conv1
  #[6272, 512], #[512],           -- dense0
  #[512, 512], #[512],            -- dense1
  #[512, 10], #[10]               -- dense2
]
def nParams : Nat := 32*1*3*3 + 32 + 32*32*3*3 + 32 + 6272*512 + 512 + 512*512 + 512 + 512*10 + 10
def lossIdx : Nat := nParams
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 784]
end CnnLayout

namespace CifarLayout
def paramShapes : Array (Array Nat) := #[
  #[32, 3, 3, 3], #[32],          -- conv0: 3→32
  #[32, 32, 3, 3], #[32],         -- conv1: 32→32
  #[64, 32, 3, 3], #[64],         -- conv2: 32→64
  #[64, 64, 3, 3], #[64],         -- conv3: 64→64
  #[4096, 512], #[512],           -- dense0
  #[512, 512], #[512],            -- dense1
  #[512, 10], #[10]               -- dense2
]
def nParams : Nat :=
  32*3*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 +
  4096*512 + 512 + 512*512 + 512 + 512*10 + 10  -- 2430018
def lossIdx : Nat := nParams
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3072]
end CifarLayout

namespace ResNet34Layout
/-- Chapter-5 **real ResNet-34** params (IMAGENETTE 3×224×224 — paper-native ImageNet
    resolution): **7×7 stride-2 stem** {W=`[64,3,7,7]`,γ,β} (224→112), then the
    16 basic blocks (3 strided downsample {W,γ,β}×2 + proj{W,γ,β}; 13 identity
    {W,γ,β}×2) at channels 64/128/256/512 (spatial 56/28/14/7), then dense {W,b}.
    Per-channel BN ⇒ γ/β are **rank-1 `[c]`** (not rank-0 scalars). **110 params** (no conv
    biases: every conv is BN-followed). The `(dims, initKind)` order MUST match the
    `@resnet34_<variant>_train_step` and `@resnet34_fwd` signatures, rendered by
    Proofs/Codegen/ResNet34RenderB.lean. `initKind`: 0 = random weight (`mkParam`: conv He
    fan-out, dense Glorot), 1 = ones (γ), 2 = zeros (β / bias). -/
-- §2l step B (2026-07-30): the conv BIASES are gone — `{W, γ, β}` per conv, not `{W, b, γ, β}`.
-- Every conv here is BN-followed and BN removes the bias, so it was 8,512 parameters that could
-- not affect the output; He et al.'s `.convBn` has none, and carrying them put this layout
-- 8,512 params away from the ImageNet reference it is supposed to be paired with (§2k).
-- MEASURED before the change, not argued: zeroing all 8,512 in the TRAINED net moves the logits
-- by rel 1e-6 (the same ablation on BN β moves them by 0.79), and the bias-free render ties the
-- biased one with every forward-only output BIT-EXACT. `tests/TestConvBiasZero.lean`.
private def idBlk (c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,c,3,3],0),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],1),(#[c],2)]
private def downBlk (cin c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,cin,3,3],0),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],1),(#[c],2),
    (#[c,cin,1,1],0),(#[c],1),(#[c],2)]   -- §2l step A: option-B 1×1 projection
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[64,3,7,7],0),(#[64],1),(#[64],2)]  -- 7×7-s2 stem
  for _ in [0:3] do a := a ++ idBlk 64                                                     -- stage1
  a := a ++ downBlk 64 128;  for _ in [0:3] do a := a ++ idBlk 128                         -- stage2
  a := a ++ downBlk 128 256; for _ in [0:5] do a := a ++ idBlk 256                         -- stage3
  a := a ++ downBlk 256 512; for _ in [0:2] do a := a ++ idBlk 512                         -- stage4
  a := a ++ #[(#[512,10],0),(#[10],2)]                                                     -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end ResNet34Layout

namespace MobileNetV2Layout
/-- Chapter-6 **MobileNetV2** params (IMAGENETTE 3×224×224 — paper-native ImageNet
    resolution, full-paper downsampling `[t,c,n,s]`): stem {W,γ,β} (3×3 stride-2 conv
    3→32), then 17 inverted-residual blocks — each expand 1×1 {W,γ,β}, depthwise 3×3
    {W,γ,β} (a `[mid,1,3,3]` kernel, feature_group_count = mid; stride-2 for the
    downsampling blocks), project 1×1 {W,γ,β} — then the head 1×1 conv
    {W,γ,β} (320→1280, the MNv2 "features" layer: conv→BN→relu6 before GAP, so the
    pooled tensor isn't the constant β of an instance-normed BN) and dense {W,b}.

    **158 params** (no conv biases: every conv here is BN-followed and BN removes a bias). At
    K = 1000 the count is 3,504,872, the JAX reference's.
    Per-channel BN ⇒ γ/β are **rank-1 `[c]`**. Spatial
    224→112(stem)→56→28→14→7 — the MobileNetV2 /32 flow. The `(dims, initKind)` order MUST match
    `@mobilenetv2_adam_train_step`'s signature (and `@mobilenetv2_fwd`'s) — both rendered from the
    same `Proofs.StableHLO.mnv2FwdChainB` traversal. Strides live only in the renderers (no
    param-shape effect). `initKind`: 0 = random weight (`mkParam`: conv He fan-out, dense Glorot), 1 = ones (γ), 2 = zeros. -/
private def irBlk (ic mid oc : Nat) : Array (Array Nat × Nat) :=
  (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],1),(#[mid],2)] else #[]) ++  -- expand 1×1 (skip if t=1, mid=ic)
  #[(#[mid,1,3,3],0),(#[mid],1),(#[mid],2),                -- depthwise 3×3 (stride 1 or 2)
    (#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)]                 -- project 1×1
/-- (ic, mid, oc) per block — MUST match tests/TestMobilenetV2*.lean `blocks`. Full paper net (17). -/
private def blocks : Array (Nat × Nat × Nat) :=
  #[(32,32,16),
    (16,96,24),(24,144,24),
    (24,144,32),(32,192,32),(32,192,32),
    (32,192,64),(64,384,64),(64,384,64),(64,384,64),
    (64,384,96),(96,576,96),(96,576,96),
    (96,576,160),(160,960,160),(160,960,160),
    (160,960,320)]
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[32,3,3,3],0),(#[32],1),(#[32],2)]             -- stem
  for (ic, mid, oc) in blocks do a := a ++ irBlk ic mid oc                                 -- 17 IR blocks
  a := a ++ #[(#[1280,320,1,1],0),(#[1280],1),(#[1280],2)]                                 -- head 1×1 conv→BN→relu6
  a := a ++ #[(#[1280,10],0),(#[10],2)]                                                    -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end MobileNetV2Layout

namespace EfficientNetLayout
/-- Chapter-7 **EfficientNet-B0** params (Imagenette 3×224×224 — the `[t,c,n,s,k]`
    config, all-swish + batch norm): stem {W,γ,β} (3×3 stride-2 conv 3→32), then 16 MBConv
    layers across 7 stages (channels [16,24,40,80,112,192,320],
    kernels [3,3,5,3,5,5,3], expand [1,6,6,6,6,6,6] — the MBConv1 stage-1 blocks have NO
    expand conv) — each (when expanded) expand 1×1 {W,γ,β}, depthwise k×k {W,γ,β}
    (`[mid,1,k,k]`, feature_group_count = mid), **squeeze-excite** {Ws₁`[mid,r]`,bs₁`[r]`,
    Ws₂`[r,mid]`,bs₂`[mid]`} (r = ic/4), project 1×1 {W,γ,β} — then head 1×1 conv {W,γ,β}
    (320→1280) and dense {W,b}. Batch-norm γ/β rank-1 `[c]`. **213 params** (the 49
    BN-followed convs carry no bias; at K = 1000 this is 5,288,548 — the JAX reference's own
    count). SE's two biases stay: those 1×1 convs are followed by the sigmoid gate, not by
    BN, so nothing absorbs them and the reference carries them. Spatial
    224→112→56→28→14→7 (stride-2 stem, 4 strided stages). The `(dims, initKind)` order MUST match
    `@efficientnet_train_step`'s signature, rendered by Proofs/Codegen/EfficientNetRender.lean.
    `initKind`: 0 = random weight (`mkParam`: conv He fan-out, dense Glorot), 1 = ones (γ), 2 = zeros (β / bias). -/
private def stages : Array (Nat × Nat × Nat × Nat × Nat) :=
  #[(1,16,1,1,3),(6,24,2,2,3),(6,40,2,2,5),(6,80,3,2,3),(6,112,3,1,5),(6,192,4,2,5),(6,320,1,1,3)]
private def mbBlk (ic mid oc r k : Nat) : Array (Array Nat × Nat) :=
  (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],1),(#[mid],2)] else #[]) ++  -- expand (skip if t=1)
  #[(#[mid,1,k,k],0),(#[mid],1),(#[mid],2),               -- depthwise k×k (stride 1 or 2)
    (#[mid,r],0),(#[r],2),(#[r,mid],0),(#[mid],2),        -- squeeze-excite dense₁/dense₂ — biases KEPT
    (#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)]                -- project 1×1
/-- `(dims, initKind)` for every param, in func-arg order — generated from the B0 stage
    spec exactly as tests/TestEfficientNet*.lean `blocks` (stem out 32, prev threading). -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[32,3,3,3],0),(#[32],1),(#[32],2)]             -- stem 3→32
  let mut prev := 32
  for (t, c, n, _s, k) in stages do
    for j in [0:n] do
      let ic := if j == 0 then prev else c
      a := a ++ mbBlk ic (t*ic) c (max 1 (ic/4)) k
    prev := c
  a := a ++ #[(#[1280,320,1,1],0),(#[1280],1),(#[1280],2)]                                 -- head 320→1280
  a := a ++ #[(#[1280,10],0),(#[10],2)]                                                     -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end EfficientNetLayout

namespace ConvNeXtLayout
/-- Chapter-8 **ConvNeXt-T** params (IMAGENETTE 3×224×224 — paper-native resolution):
    4×4/s4 patchify stem {W=`[96,3,4,4]`,b} (224→56), then [3,3,9,3] blocks @ [96,192,
    384,768] (spatial 56/28/14/7) with 3 between-stage LN+2×2/s2 downsamples, then head
    GAP → LN(768) → dense {W,b}. The patchify stem carries a channel LN {γ,β} `[96]`.
    ConvNeXt block (9 params): depthwise 7×7 {W=`[c,1,7,7]`,b}
    → **channel LN** (per-channel γ/β `[c]`) → 1×1 expand {W=`[4c,c,1,1]`,b}
    → GELU → 1×1 project {W=`[c,4c,1,1]`,b} → **layerScale** (per-channel γ=`[c]`). Each
    downsample (4 params): channel LN {γ,β} `[c]` + 2×2 conv {W=`[2c,c,2,2]`,b}. 182 params. The
    `(dims, initKind)` order MUST match `@convnext_train_step`'s signature, rendered by
    Proofs/Codegen/ConvNeXtRender.lean. `initKind`: 0 = random weight (`mkParam`; σ = 0.02
    under its ConvNeXt flag), 1 = ones (LN γ), 2 = zeros (LN β / bias), 3 = 1e-6 (layerScale γ,
    the paper's init). -/
private def depths : Array Nat := #[3, 3, 9, 3]
private def dims   : Array Nat := #[96, 192, 384, 768]
private def blockSpec (c e : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,1,7,7],0),(#[c],2),(#[c],1),(#[c],2),   -- depthwise W,b ; LN γ,β (PER-CHANNEL, §2m)
    (#[e,c,1,1],0),(#[e],2),                      -- expand W,b
    (#[c,e,1,1],0),(#[c],2),                      -- project W,b
    (#[c],3)]                                     -- layerScale γ (per-channel), kind 3 = 1e-6
private def downSpec (ci co : Nat) : Array (Array Nat × Nat) :=
  #[(#[ci],1),(#[ci],2),(#[co,ci,2,2],0),(#[co],2)]  -- LN γ,β at the PRE-conv width ; conv W,b
/-- `(dims, initKind)` for every param, in func-arg order.

    The head is `GAP → LN(768) → dense`, as in both the paper (`self.norm(x.mean([-2,-1]))`,
    `nn.LayerNorm(dims[-1], eps=1e-6)`) and timm (`NormMlpClassifierHead`). **182 param
    tensors**; the floats are 27,827,818 at K = 10, i.e. **28,589,128 at K = 1000** — timm's
    count. Without the head LN the count would be short by exactly `2×768 = 1,536`, so a
    near-matching count is not evidence of the right architecture. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) :=
    #[(#[96,3,4,4],0),(#[96],2),(#[96],1),(#[96],2)]   -- patchify stem + stem LN γ,β
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    for _ in [0:depths[si]!] do a := a ++ blockSpec c e
    if si < 3 then a := a ++ downSpec c dims[si+1]!
  -- head: LN γ,β then dense W,b. ⚠ The LN comes FIRST — that is its order in the layer list and
  -- in `@convnext_train_step`'s signature, and the blob is read positionally.
  a := a ++ #[(#[768],1),(#[768],2),(#[768,10],0),(#[10],2)]
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end ConvNeXtLayout

namespace ViTLayout
/-- Chapter-9 **ViT-Tiny** params (IMAGENETTE 3×224×224, patch-16): a 16×16/s16 conv
    patch embed {W=`[192,3,16,16]`,b} (224→14×14=196 patches), a learned CLS token
    `[192]` (1D, matching the proof-tied render) + positional embed `[197,192]`, then 12 pre-norm transformer blocks
    (dim 192, 3 heads, MLP 768), final LayerNorm γ/β, CLS-slice dense head {W=`[192,10]`,b}.
    LayerNorm γ/β are **per-channel `[192]`** (normalize ∘ per-channel affine, as in
    `Proofs.vitForwardKV`). Each block
    (16 params): LN1 γ/β, Wq/bq/Wk/bk/Wv/bv/Wo/bo `[192,192]`/`[192]`, LN2 γ/β, MLP
    Wfc1`[192,768]`/bfc1/Wfc2`[768,192]`/bfc2. 4+12·16+4 = 200 params. The `(dims,initKind)`
    order MUST match `@vit_train_step`/`@vit_fwd`, whose parameter list is
    `Proofs.StableHLO.vitParamSig` (Proofs/Codegen/ViTRender.lean). `initKind`: 0 = random
    weight (`mkParam`: conv He fan-out, dense Glorot, or timm's σ = 0.02 under its ViT flag),
    1 = ones (LN γ), 2 = zeros (LN β / bias / CLS / pos). -/
private def D : Nat := 192
private def M : Nat := 768
private def S : Nat := 16
private def nTok : Nat := 197    -- 14·14 + 1 (CLS)
private def depth : Nat := 12
private def nCls : Nat := 10
private def blockSpec : Array (Array Nat × Nat) :=
  #[(#[D],1),(#[D],2),                                                       -- LN1 γ,β
    (#[D,D],0),(#[D],2),(#[D,D],0),(#[D],2),(#[D,D],0),(#[D],2),(#[D,D],0),(#[D],2),  -- Wq..bo
    (#[D],1),(#[D],2),                                                       -- LN2 γ,β
    (#[D,M],0),(#[M],2),(#[M,D],0),(#[D],2)]                                 -- MLP
/-- `(dims, initKind)` for every param, in `@vit_train_step` func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) :=
    #[(#[D,3,S,S],0),(#[D],2),(#[D],2),(#[nTok,D],2)]   -- patch W,b ; CLS [192] (1D) ; pos
  for _ in [0:depth] do a := a ++ blockSpec
  a := a ++ #[(#[D],1),(#[D],2),(#[D,nCls],0),(#[nCls],2)]   -- final LN γ,β ; head W,b
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end ViTLayout

def MlpLayout.paramShapes : Array (Array Nat) := #[
  #[784, 512], #[512], #[512, 512], #[512], #[512, 10], #[10]
]
def MlpLayout.shapesBA : ByteArray := packShapes MlpLayout.paramShapes

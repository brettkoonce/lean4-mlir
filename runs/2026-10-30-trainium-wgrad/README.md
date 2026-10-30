# The 15× ViT gap on Trainium is one op: the patch-embed weight gradient

2026-10-30. Found on atlas with no instance: a static profile of the compiled NEFFs that already sat
in the Neuron compile caches from the warms (`neff_id.sh`, `neff_attr.py`), three micro-compiles of
the op in isolation (`micro/`), and a text rewrite of the render checked on XLA:CPU
(`wgrad_rewrite.py`). Then two instance sessions (`trn1/`): **the rewritten render runs the
f32 ViT-Tiny step in 71 ms against the original's 1,281 and stock JAX's 82.5 on the same
NeuronCore; 54 ms with bf16 matmuls; the full 80-epoch Imagenette run takes 34.5 minutes and lands at 69.27 %.** It also fixes a numerics fault: the original's patch-embed
Adam moment was 72 % off the CPU golden on Neuron; the rewrite is within 1e-4.

| f32, batch 32, one NeuronCore | ms/step |
|---|---|
| verified `vit_adam_train_step`, as rendered (control, same session) | 1,281 |
| rewritten, form B (`rhs_dilate` conv) | **71** |
| rewritten, form C (im2col matmul) | **70** |
| rewritten B, `--auto-cast=matmult --auto-cast-type=bf16` | **54** |
| stock JAX ViT-Tiny, f32 / bf16 (`../2026-10-30-trainium-imagenette/`) | 82.5 / 63.8 |
| verified, as rendered, bf16 auto-cast (same) | 733 |
| 4060 Ti, verified, f32 | 61 |

## What the render does

`vit_adam_train_step.mlir` lines 7013–7019 (`Pretty.lean`'s `patchEmbedWeightGrad`, same text in
`patchEmbedWeightSgd` and the bf16 peer): the patch-token cotangent `[32,192,14,14]` is padded
with `interior = [0, 0, 15, 15]` to `[32,192,209,209]`, transposed, and used as the KERNEL of a
valid stride-1 convolution against the image, `[3,32,224,224] × [192,32,209,209] → [3,192,16,16]`.
That is the strided-conv weight gradient written with the dilation materialized. 99.5 % of the
kernel is zeros (196 of 43,681 taps), the padded tensor is 1.07 GB of f32, and the conv is
206.1 GMAC where the gradient needs 0.92.

| step, f32, batch 32 | MAC (G) | of which this conv |
|---|---|---|
| verified `vit_adam_train_step` (`hlo_stats.json`) | 324.6 | 206.1 |
| the same step without it | 118.5 | — |
| stock JAX ViT-Tiny (`jax_bench.py` graph) | 119.4 | 0 (im2col matmul) |

Every other matmul in the step matches stock JAX to within 1 GMAC. The per-head split the
planning doc suspected (`slice`/`pad` per head, 148 + 146 ops) costs no extra MACs and — below —
no extra matrix-engine work.

## Static profile of the NEFFs (`attr/vit_f32_verified_vs_jax.txt`)

`neff_attr.py` walks the debug info inside a NEFF (`debug_info_asm_<engine>.dbg` → backend →
Penguin → HLO) and counts, per HLO op type, the DMA blocks and the instructions on each engine.
The two NEFFs are the f32 ViT-Tiny b32 train steps the instance ran at 1,281 ms (verified) and
82.5 ms (stock JAX).

| f32, batch 32 | DMA blocks | PE | DVE | Pool | Activation | SP |
|---|---|---|---|---|---|---|
| verified, whole step | 341,729 | 808,023 | 424,514 | 67,050 | 160,497 | 563,800 |
| — the weight-grad conv | 123,389 | 266,764 | 170,088 | 63,326 | 64,524 | 179,052 |
| — spills/reloads of its padded operand (`?pad`) | 191,243 | 0 | 0 | 0 | 0 | 0 |
| stock JAX, whole step | 34,323 | 419,219 | 115,164 | 3,642 | 111,046 | 48,867 |

The conv and the spill traffic of its 1 GB operand are 92 % of the verified step's DMA blocks
(314,632 of 341,729). Without them the verified step has 27k DMA blocks against stock JAX's 34k,
and the per-head attention dots take 180k PE instructions against JAX's 191k. Instruction-stream
sizes tell the same story (`attr/cache_entries.txt`): SP (the DMA/sync program) 36.1 MB vs 3.1,
DMA debug 40.2 MB vs 4.0, Pool 4.3 MB vs 0.2 — the 10–12× ratios that match the 15× in time; PE
51.7 vs 26.8 and DVE 27.2 vs 7.4 do not.

The bf16 renders (`MODULE_1171792218952093389`, `MODULE_16104594695929880757`) carry the same conv
in f32 (the bf16 converts sit on the dense matmuls only), which is why bf16 moved the step from
1,281 to 733 ms and no further.

## The op alone (`micro/`)

Three forms of the same gradient, each compiled by `neuronx-cc` from MLIR text on atlas
(`--target trn1 --auto-cast none`, the trainers' flags):

| form | text | MAC | compile | NEFF | DMA blocks | PE | DVE | Pool | Act | SP |
|---|---|---|---|---|---|---|---|---|---|---|
| A, as rendered | pad interior 15 → transpose → conv, kernel 209×209 | 206.1 G | 527 s, 18 GB RSS | 39.4 MB | 266,995 | 591,652 | 164,781 | 63,445 | 172,592 | 397,492 |
| B, dilated kernel | the same conv with `rhs_dilate = [16, 16]`, kernel 14×14 | 0.92 G | 2 s | 93 KB | 184 | 1,564 | 200 | 89 | 318 | 244 |
| C, im2col matmul | reshape x to `[6272,768]`, dy to `[6272,192]`, one `dot_general` | 0.92 G | 2 s | 124 KB | 152 | 2,121 | 235 | 110 | 186 | 192 |

Form A on its own is 78 % of the whole verified step's DMA blocks and 73 % of its PE instructions.
`hlo2penguin` also logs `needsModular_PreSplit? Yes. macCnt 206112817152 threshold 200000000000`
for it: the step crosses the compiler's 200 GMAC modular-flow threshold on this op alone; stock
JAX's 119 GMAC does not. The compiler does not fold the interior pad into kernel dilation, and
neither does XLA:CPU (the optimized HLO keeps `window={size=209x209}`), so the 4060 Ti most likely
runs the 206 GMAC too — unmeasured.

## The rewrite (`wgrad_rewrite.py`)

`wgrad_rewrite.py {B|C} IN OUT` rewrites the three-op tail in a render; `check {B|C} NAME` runs
the original and the rewritten render on XLA:CPU with `tier2.py`'s inputs. On `vit_adam_train_step`
(`attr/xla_cpu_check_*.log`): 600 of 603 outputs bit-identical; the three that differ are the
patch-embed weight and its two Adam moments, max abs diff 2.1e-8 (f32 summation order over the
6,272 patch positions). Form B is the smaller text change and XLA's own gradient of a strided conv;
form C is what stock JAX does and keeps Neuron's conv lowering — which has two open miscompiles in
weight-gradient-form convs (`../2026-10-30-trainium-upstream/`) — out of the path.

The rewritten renders are scratch files, not committed: `verified_mlir/vit_adamwgB_train_step.mlir`
and `verified_mlir/vit_adamwgC_train_step.mlir`, warmed through `neuron_warm.sh` into
`/var/tmp/neuron-compile-cache-wgrad` in 398 s each where the original takes 1,600–2,200 s
(`runs/platform/2026-10-30-atlas-neuron-warm-12/`). Their whole-step profiles
(`attr/vit_f32_rewritten.txt`) land at or under stock JAX in every column: 24.9k / 24.4k DMA blocks
(JAX 34.3k), PE 436k (419k), DVE 74k (115k), Pool 3.6k (3.6k), SP 38k (49k).

## On the instance (`trn1/`)

`wgrad_probe.sh VARIANT` runs the trainer's 160-step probe (60 warm) alone on NeuronCore 0 with
`LEAN_MLIR_VARIANT=adamwg{B,C}`, which picks the scratch render by name; the compile cache shipped
from atlas was hit (6.7 s and 6.2 s "compile" against 47.5 s for the original's cached load; the
bf16 auto-cast variant compiled cold on the instance in 287 s). Numbers in the table at the top;
per-step dumps in `*-probe.tsv`.

**The full 80-epoch run (`trn1/full.log`, `wgrad_full.sh`):** `LEAN_MLIR_VARIANT=adamwgB`, f32,
one NeuronCore, real Imagenette with eval every epoch. Start 17:11:45 UTC, done 17:46:16 — **34.5
minutes wall for 80 epochs, 69.27 % top-1 / 90.17 % top-5 at epoch 80** (95 % CI 67.8–70.7). The
Arc B50's 80-epoch run of the same recipe (`../2026-10-23-vit-verified-adam/`) took 39 min and
landed at 68.46 %; epoch 1 here is 1377/3925 = 35.08 %, the Arc's epoch-1 count to the image, against the 34.62 % the wrong
gradient gave on Neuron. The instance was up 40 minutes for the session, about $0.90.

Numerics (`tier2.log`): the suite's tier-2 check (`tier2.py run`, one NeuronCore against the
XLA:CPU golden, `tolerances.tsv`'s attention/f32 row):

| render on Neuron vs CPU golden | median | p90 | max (output) |
|---|---|---|---|
| `vit_adam_train_step` as rendered | 4.5e-07 | 4.9e-06 | **7.2e-01** (200: patch-embed Adam `m`) |
| form B | 1.3e-06 | 9.1e-06 | 7.4e-05 (166) |
| form C | 5.9e-07 | 5.6e-06 | 6.3e-05 (209) |

The original's 72 % is the same number the 2026-10-29 suite run recorded and passed on (median and
p90 are inside tolerance; the max is reported, not gated). So Neuron was computing the 209×209
conv wrong as well as slowly, and the loss curves show it: at step 100 of the probe the control
sits at 2.264 where both rewrites sit at 2.180 from the same seed. The 34.62 % epoch-1 accuracy in
`../2026-10-30-trainium-imagenette/` was trained with that gradient. The goldens for the scratch
renders were generated on the instance's CPU and deleted again; `tier2_artifacts.tsv` there was
restored.
The real fix is the renderer: the `patchEmbedWeightGrad`/`patchEmbedWeightSgd` templates in
`LeanMlir/Proofs/Codegen/StableHLO/Pretty.lean` (three places, 4190 and 1889 plus the bf16 peer).
`den` is on the AST node, so the proofs do not move; what moves is the pinned text of all 75 ViT
renders (every `vit*_train_step.mlir`), the tier-2 golden `vit_adam_train_step.npz`, and whatever
`book_listings.py --check` pins. The ConvNeXt stem (`interior = [0, 0, 3, 3]`, 65 renders, 16× the
needed MACs, a 600 MB operand at b32) and the ResNets' stride-2 weight grads (`interior = [0, 0, 1, 1]`,
4×) are the same pattern at smaller multipliers.

## The renderer change (the same afternoon)

`LeanMlir/Proofs/Codegen/StableHLO/Pretty.lean`: sixteen templates stop materializing the dilation.
Ten weight gradients of plain strided convs (`convStrided{,Xla}Weight{Grad,Sgd}` batched and
per-example, `convStride4WeightGrad`, `patchEmbedWeight{Grad,Sgd}`) emit the undilated cotangent as
the kernel with `rhs_dilate = [s, s]`; six input gradients (`convStridedBack`, `depthwiseStrided{,Xla}Back`,
batched and per-example) emit the undilated cotangent as the input with `lhs_dilate = [2, 2]` and
one more row of high padding. `sWGradDil` replaces `sWGradGeom` for the weight grads: the padded
kernel's trailing zero row read one input row the dilated kernel does not, so `hi` drops by one —
at a 1×1 stride-2 projection that is −1, emitted as a one-row `stablehlo.slice` of the input instead
of a negative pad. The six grouped (depthwise) weight gradients keep the pad: neuronx-cc 2.28 fails
`rhs_dilate` with `batch_group_count > 1` at three of B0's four shapes (NCC_IBIR243; issue 6 in
`../2026-10-30-trainium-upstream/`), and in isolation they compile and are tiny.

`den` is on the AST, so no proof moved. `scripts/regen_verified_mlir.sh proofs` re-emitted 252
renders (3 min 48 s); the diff is mostly SSA renumbering from the dropped pad and zero-constant
lines. The re-emitted renders equal the scratch rewrites `dilate_rewrite.py` produced, SSA names
stripped, except that the scratch files kept the now-unused zero constants — and the plugin's cache
key does not see that: the re-emitted `vit_adam_train_step.mlir` warms to `MODULE_6062314317683602994`,
the entry the scratch form B produced, so the 71 ms probe and the 69.27 % run above ran the NEFF the
committed render compiles to. `regen_verified_mlir.sh check` and
`book_listings.py --check` pass. `tier2.py goldens --check` (`attr/goldens_check.log`) found 7 of the
16 CPU goldens stale, but not from this change: `resnet34_adam` fails by the same 1.5e-2 against the
pre-change render, and `cifar8wb_bn_bf16adam` is a render this change does not touch — they had
drifted since 2026-10-02. All 16 were regenerated and pass.

Before/after at `ty` level, the stride-2 3×3 weight grad (`sWGradDil 3 = (0, 1, 0)`):

```
  %du = stablehlo.pad %dr, %z, low = [0,0,0,0], high = [0,0,1,1], interior = [0,0,1,1] : [B,oc,h,w] -> [B,oc,2h,2w]
  %dt = stablehlo.transpose %du … [oc,B,2h,2w]
  %g  = stablehlo.convolution(%xt, %dt) … pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]
→
  %dt = stablehlo.transpose %dr … [oc,B,h,w]
  %g  = stablehlo.convolution(%xt, %dt) … pad = [[1, 0], [1, 0]], rhs_dilate = [2, 2]
```

## The convnets on the instance (`trn1/r34-*`, `trn1/r50-*`, `trn1/tier2_cnn.log`, `trn1/*-summary.*`)

Second instance session, 18:13–18:33 UTC, with the six convnet rewrites warmed on atlas
(`runs/platform/2026-10-30-atlas-neuron-warm-13/`, `-14`):

| render, f32 | compiles? | ms/step | numerics vs CPU golden (tier 2, conv-pool/f32 tolerance 3e-2 / 7e-2) |
|---|---|---|---|
| ResNet-34 b16, as rendered (control, same session) | yes | 491 | OK: median 3.3e-03, p90 1.4e-02, max 2.1e-02 |
| ResNet-34 b16, dilated | yes, 495 s on atlas (was 21–26 min) | **478** | OK: median 3.9e-03, p90 1.6e-02, max 2.1e-02 |
| stock JAX ResNet-34 b16 (`../2026-10-30-trainium-imagenette/`) | | 349 | |
| ResNet-50 b32, dilated | **yes — the first b32 f32 ResNet-50 to compile** (1,202 s) | 659 | **FAIL: median 5.1e-02, p90 7.3e-02, max 1.1e-01** |
| ResNet-34 b32, dilated | no: NCC_IBIR243 on the max-pool's `maximum`, as before (issue 3) | | |
| ConvNeXt-T b32, dilated | no: 8.56M instructions, cap 5M (was 9.17M) | | |
| EfficientNet-B0 / MobileNetV2 b32, dilated | no: 5.04M / 5.05M instructions, cap 5M (were 5.12M / 5.08M) | | |

So the dilation was ViT's problem and only ViT's: ResNet-34 moves 3 %, even though its MAC count fell
from 182 to 160 G (stock JAX's number exactly) and its PE stream from 66 to 28 MB. Its numerics do
not move either, and they pass. ResNet-50 at batch 32 now compiles and runs but is off the CPU golden
by a median 5 %, fifteen times ResNet-34's, with no b32 f32 baseline to compare against (the original
never compiled); unexplained. ConvNeXt's instruction count is not about dilation — its stride-1 7×7
depthwise convs and channel LayerNorms at batch 32 are — and B0 and MobileNetV2 sit 1 % over the cap;
a batch-16 render is the obvious next try for all three.

**Where ResNet-34's 1.4× is**, from `neuron-explorer` device profiles of the dilated verified NEFF and
the stock JAX NEFF, each run once alone (`trn1/r34dil-summary.txt`, `trn1/jaxr34-summary.txt`):

| one step, profiled | verified (dilated) | stock JAX |
|---|---|---|
| total active time | 0.312 s | 0.200 s |
| DMA active | **88 %** of the time, 409M cycles | 255M cycles |
| DMA bytes / transfers | 10.79 GB / 45,875 | 10.60 GB / 54,991 |
| spill save / reload | 3.52 GB / 7.17 GB | 3.56 GB / 6.95 GB |
| tensor engine active | 11.9 % | 19.4 % |
| vector engine active | 4.2 % | 19.6 % |
| tensor-engine instructions | 441k | 502k |

Same bytes, same spills, 60 % more DMA time, half the engine utilization: the verified step is
DMA-bound on how its transfers are shaped, not on how many bytes move or how many MACs it does.
The per-op breakdown (`neuron-explorer view --output-format json`, 300 MB `.ntff` captures left in
`~/wgrad/` on the instance) is the next step; the summaries are here.

## ResNet-34 for contrast (`attr/r34_f32_b16_verified_vs_jax.txt`)

Verified 494 ms vs stock JAX 349 (1.4×). Same DMA blocks (54,447 vs 54,991), 2× the PE
instructions (1,032k vs 502k; the convolutions 912k vs 352k) on 1.14× the MACs (182.4 vs 160.1 G).
That gap is conv lowering, not DMA, and the dilated stride-2 weight grads are the first suspect.

## Files

- `neff_id.sh CACHE…` — one line per compile-cache entry: argument count, bf16, conv/dot counts,
  `HloMacCount`, traffic, instruction-stream sizes. `attr/cache_entries.txt` is its output for the
  four caches on atlas.
- `neff_attr.py NEFFDIR…` — the per-HLO-op static profile of an extracted NEFF. A NEFF is a 1 KB
  header then a `.tar.gz`: `dd if=model.neff bs=1024 skip=1 | tar xzf -`.
- `wgrad_rewrite.py` — the rewrite and its XLA:CPU check (`~/lean/venv-cpu/bin/python -I`).
- `micro/` — the three forms as MLIR, their compile logs, `hlo_stats.json` and profiles.

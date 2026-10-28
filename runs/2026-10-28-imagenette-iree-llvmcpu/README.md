# `lake run imagenette-iree` on a CPU — IREE 3.13.0, llvm-cpu, atlas, 2026-10-28

The first time the second lowerer ran the Imagenette tier on a box without a GPU, and the first
time the tier's seven binaries were all pointed at IREE. All seven lower and step — the two
ResNets only after their stem pool's backward was re-emitted the same day, below. `results.tsv`
is the table, the per-net logs are beside it.

**The box.** atlas: 2× Xeon E5-2699 v4 (44 cores, 88 threads), 1 TB, no GPU. The stack is
`deploy/iree_cpu_setup.sh`'s: `iree-base-compiler`/`-runtime` 3.13.0 in `.venv`, the runtime
built at the wheel's commit (`89ece87b`) with the `local-task` and `local-sync` drivers,
`ffi/libiree_ffi.so` compiled `-DUSE_CPU`. Each run: `LEAN_MLIR_LOWERER=iree
IREE_BACKEND=llvm-cpu LEAN_MLIR_G2_STEPS=10 LEAN_MLIR_MAX_EPOCHS=1 LEAN_MLIR_CKPT_TAG=smoke
./run.sh <net> 0 llvm-cpu` — `iree-compile` of the train step and both forwards, ten steps of
batch 32, the full 3925-image eval through the eval forward.

**What passed as rendered.** MobileNetV2, MobileNetV4-Conv-M, EfficientNet-B0, ConvNeXt-T and ViT-Tiny:
losses fall from ln 10 + init over the ten steps, the eval forward scores the val set, the
checkpoint and its BN companion are written. 4–15 minutes each end to end, 12–18 GB resident;
ConvNeXt is the slow one. MobileNetV2 also ran a whole first epoch (295 steps) earlier in the
day: 1052 s with compile and eval, 17.4 % top-1 after one epoch
(`mobilenetv2-verified-adam.full-epoch.log`). At ~3 s per step for MobileNetV2 and ~20–30 s for
MobileNetV4, an 80-epoch run here is days per net; this record is the lowering and the loop, not
a training result.

**What the ResNets needed.** As rendered that morning, `resnet34_adam_train_step.mlir` and
`resnet50_adam_train_step.mlir` failed at `iree-compile` in under a second
(`*.old-select_and_scatter.log`):

    error: failed to legalize operation 'stablehlo.select_and_scatter' that was explicitly marked illegal
    ... window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding [[0,0],[0,0],[1,1],[1,1]]
    (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>

That is the stem's 3×3 stride-2 max pool's backward (`maxPool3s2Back`), whose windows OVERLAP.
No IREE release lowers `select_and_scatter` over overlapping windows — upstream StableHLO's
linalg legalization handles window = stride only — and the same text fails on 3.11.0 (CI's pin),
3.12.0, 3.13.0 and the 3.14 nightly of 2026-10-24. So the ResNets had never trained through IREE
as rendered; `imagenette-iree` listed ResNet-34 anyway, and `LeanMlir/ReferenceNets.lean` had
already given a 2×2 pool to the JAX reference "because IREE had no select_and_scatter for 3/2".
The 2×2 pools of the MNIST and CIFAR nets lower, and the ResNet forwards always did.

The fix is in the emitter, not the proofs. `maxPool3s2BackText` (Pretty.lean) now writes the same
function without the op: pad the input with −inf by one, take the nine stride-2 slices (one per
window offset), one `maximum` over them is the forward's max, a running first-max mask in
row-major order is the `GE` select's choice, `select` routes the cotangent to each offset, nine
interior-1 pads put each back where it was read, and their sum is the accumulation the scatter
region's `add` did. The denotation `maxPool3s2BackFlat` did not move; the emitted text is a
different spelling of it. Before it went into the renderer it was checked standalone at the stem
shape (32×64×112×112) with ties forced, against XLA's own `select_and_scatter` through JAX on
the CPU and against an independent argmax/bincount reference: identical to 5e-7, i.e. fp32
summation order. `scripts/regen_verified_mlir.sh proofs` then rerendered the 61 ResNet-34/50
train steps (`regen-proofs.log`, 3 min 19 s; no forward changed), and every gate that reads
them passed: the writer and prefix audits, the StableHLO parse gate on all 61, the DP render
text tie, and `TestBatchedEmitTie`, whose stem-pool block now pins the nine-slice form and fails
on `select_and_scatter`.

**The gradient, gated through IREE.** `r50-gradcheck` — the two-tier gate on R50's backward,
whose tier-1 identity B at the stem BN crosses `maxPool3s2` and nothing else does — ran on the
rerendered `resnet50in_adam64_train_step.mlir` (ImageNet, batch 64) under `LEAN_MLIR_LOWERER=iree
IREE_BACKEND=llvm-cpu`, the first time it has run anywhere but a GPU: **PASS** in 3050 s and 49
invokes (`r50-gradcheck.log`). Tier 1: 53 conv kernels and 33 BN affines to |cos∠| ≤ 5.8e-5, the
worst at s4b2.bn2, with the 21 control sites separated at 124× (10th percentile) and 330×
(median). Tier 2: all 22 groups to rel ≤ 0.090 at the stem, where the header's GPU measurement
was 0.17, and the doubled-gradient control misses by 11× the tie. Then both Imagenette trainers
stepped: ResNet-34 and ResNet-50, ten steps each plus the full eval, losses falling, checkpoints
written (`results.tsv`). The group is now all seven nets (lakefile.lean).

**Why 3.13 and not CI's 3.11.** On IREE 3.11.0 the CPU task device segfaults inside the first
invoke of every 224² artifact — MobileNetV2's eval forward and train step, reproduced with
`iree-run-module` alone (one worker, 8 MB worker stacks, any group count), while the same `.vmfb`
runs on `local-sync` and the MNIST CNN trains fine on `local-task`. 3.12.0, 3.13.0 and the 3.14
nightly run them. Compiler wheel, runtime wheel and runtime commit move together (a 3.11 `.vmfb`
on a 3.14 runtime: "hal.command_buffer.dispatch signature mismatch"; the reverse: "required
module 'hal' version mismatch; have 6 but require 7"). The setup script pins 3.13.0 and smokes
exactly this case.

**Two checks that came along.** (1) `tests/TestCifar8BbnGradCheck.lean` — the gate that found
3.11's llvm-cpu max-pool-backward miscompile and sent the CPU gate to vmvx — PASSES on 3.13.0
llvm-cpu for both artifacts (`cifar8wb_bbn` worst rel 0.0051, `cifar8wb_bn` 0.0067, tier-1
|cos∠| ≤ 1e-6, 17 s each). (2) `scripts/regen_verified_mlir.sh tests`, CI's compile smoke: all
34 committed artifacts compile on 3.13.0 llvm-cpu and the re-rendered ones match the committed
bytes. CI's pin stays 3.11.0 in this change; moving it is its own run.

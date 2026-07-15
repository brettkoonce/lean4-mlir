# unet3d.md — volumetric segmentation, and what it actually costs

Companion to `planning/brats_demo.md`, which built the 2D demo and
deliberately deferred this. That doc's first pass called 3D a
"research-scale lift... threading NCDHW through ~8.5k lines of codegen."
**That estimate was wrong by roughly 10× and has been corrected there.**
This doc is the scope it should have had, written down before the evidence
goes stale.

Standing recommendation, up front: **do not start this until Gate D in
`brats_demo.md` (the 2.5D number) says through-plane context is where the
2D ceiling actually is.** Everything below argues 3D is *affordable*, not
that it is *warranted*. Those are different claims and the second one is
still unmeasured.

## The one real unknown, now measured

IREE's willingness to lower rank-5 was the thing that could have flipped
this from "weeks of plumbing" to "blocked." The repo has hit IREE op gaps
before and routed around them — `MlirCodegen.lean:6182` documents avoiding
`select_and_scatter` because "IREE doesn't support" it, and there is a whole
`upstream-issues/` tree.

It is not a gap here. `planning/conv3d_spike.mlir` hand-writes the five ops
a 3D UNet needs and compiles all of them clean to gfx1100:

```
.venv/bin/iree-compile planning/conv3d_spike.mlir \
  --iree-hal-target-backends=rocm --iree-rocm-target=gfx1100 \
  --iree-codegen-llvmgpu-use-reduction-vector-distribution=false -o /dev/null
# exit 0
```

| op | form | status |
|---|---|---|
| `conv3d` forward | `stablehlo.convolution`, `input_spatial_dimensions = [2,3,4]` | compiles |
| `conv3d` dx | transpose `[1,0,2,3,4]` + `reverse [2,3,4]` + convolve | compiles |
| `maxPool3d` | `reduce_window`, `window_dimensions = [1,1,2,2,2]` | compiles |
| bias grad | `reduce` across `[0,2,3,4]` | compiles |
| trilinear factor | `dot_general` on a rank-5 operand | compiles |

Caveat worth stating plainly: **compiling is not running.** The spike proves
op support and shape legality, not numerical correctness or speed. A rank-5
conv that lowers to a catastrophically slow loop nest is still a compile
success. Measuring throughput on a real 128³ patch is Phase 0 below, and it
is the next thing to do if this workstream ever opens.

## Why the codegen is ~675 lines, not 8.5k

The rank-4 locking is **incidental, not structural**. Evidence:

- **No `Shape` type exists.** Shapes are plain `List Nat` (92 occurrences),
  and the type printer is already rank-generic:
  `tensorTy (dims : List Nat) : String` (`MlirCodegen.lean:44-45`).
  **Rank is not in any codegen type** — it lives only in pattern matches.
- **Rank > 4 already ships.** The maxPool backward emits rank-6
  intermediates: `tensorTy [b, c, oH, stride, oW, stride]`
  (`MlirCodegen.lean:6191-6196`). `tensorTy`, the shape plumbing, and IREE
  all handle rank-6 in a production path today. NCDHW is not novel ground.
- **`dimension_numbers` is centralized.** Of 37 `stablehlo.convolution`
  emission sites, only **2** write `input_spatial_dimensions` themselves —
  `convDimNumbers` (`:62-70`) and an inline duplicate in `emitConv2d`
  (`:178`). The other ~35 route through `convAttrBlock` /
  `convAttrBlockFull` / `dwConvAttrBlock` / `dwConvAttrBlockFull`
  (`:72-110`), which all call `convDimNumbers`.
- **The conv VJP math is dimension-agnostic.** `MlirCodegen.lean:5991-6021`
  does dW by the batch↔channel transpose trick, dx by reversed+transposed
  kernel. `[1,0,2,3]`→`[1,0,2,3,4]`, `reverse [2,3]`→`[2,3,4]`,
  `reduce [0,2,3]`→`[0,2,3,4]`. There is no 2D-specific reshape to redesign.
- **`bilinearWeights1D` (`:846`) is strictly 1-D** — it takes one length and
  a scale and returns a resampling matrix. Trilinear is that helper called
  three times plus a third `dot_general` factor. The factorized `Wy·X·Wxᵀ`
  structure *helps*; the naive 8-corner 3D gather would have been the hard
  path and the UNet demo didn't take it.
- **Of 111 four-element list patterns, only ~51 are on a 3D UNet's path.**
  The rest live in ViT / MHSA / Mamba / Swin / MobileNet / EfficientNet /
  ConvNeXt / YOLO / DDPM / SE / mbConv code a 3D UNet never touches. There
  are 39 `Layer` constructors; a 3D UNet needs ~5.

The mechanical surface, measured per function:

| function | lines | rank-4 literals |
|---|---|---|
| `emitConvBnBackward` (`:3589-3757`) | 168 | 19 |
| maxPool + unetUp backward (`:6181-6300`) | 119 | 14 |
| `emitConvBnTrain` (`:3221-3290`) | 69 | 3 |
| `emitConvBn` (`:224-291`) | 67 | 3 |
| `emitBatchNormForwardNCHW` (`:1055-1100`) | 45 | 1 |
| `emitPerPixelCEBlock` (`:4303-4348`) | 45 | 2 |
| `emitChannelConcat`/`SplitGrad` (`:809-846`) | 37 | 2 |
| `emitMaxPool` (`:763-794`) | 31 | 2 |
| `emitBilinearUpsample` (`:885-912`) | 27 | 5 |
| attr helpers + `samePad` (`:54-120`) | 66 | 0 |
| **total** | **~675** | **51** |

Plus 17 `convAttrBlock*` call sites (their signatures are positionally
rank-2: `pH0 pH1 pW0 pW1`, `sH sW`, ... — list-ify them), and 6 dispatch
walkers × ~5 layer arms. That 6× walker fan-out (`emitForwardBody` `:1789`,
`emitForwardSig` `:2087`, `emitForwardEvalSig` `:2586`, `emitForwardCamSig`
`:3013`, `emitTrainStepBody` `:4348`, `emitTrainStepSig` `:7483`) is the
real plumbing multiplier — and the reason to **parameterize by rank rather
than add `conv3d`/`maxPool3d`/... constructors**, which would multiply
across all six.

Extending the types breaks nothing: every `Layer` and `NetSpec` field is
already defaulted (`Types.lean:290-291`), so `imageD : Nat := 1` and a
`rank : Nat := 2` arg on `convBn`/`maxPool`/`unetDown`/`unetUp` migrate zero
existing specs. The `SegLoss`/`LossKind.isSeg` refactor from the Dice work
is the template: add the parameter, default it to today's behaviour, verify
the emitted MLIR is byte-identical for existing callers.

## The loader: patches, not streaming

`brats_demo.md`'s original 550 GB objection assumed loading whole volumes as
f32. That is not how 3D UNets train. nnU-Net-style training samples **128³
patches**:

- One 128³×4 f32 patch ≈ 33 MB; batch 2 ≈ 66 MB. Trivial.
- All 484 volumes held as **uint8** ≈ 17 GB — which *fits the existing
  read-everything-into-a-ByteArray pattern* on a 188 GB box.
- So: keep the whole corpus as uint8 (the format `preprocess_brats.py`
  already writes), sample patches host-side, dequantize per batch.

That is a loader change, not a rearchitecture. The `Train.lean:332-337`
ImageNet panic ("needs a C-side streaming reader") does **not** apply —
ImageNet's problem is 1.28M *separate files*; BraTS is 484 volumes that fit
in RAM whole.

Real work here: a patch sampler with foreground oversampling (nnU-Net forces
~33% of patches to contain a tumour voxel — with 0.5% classes, uniform
random patches would be almost all background, which is the same imbalance
disease this demo already documented at the loss layer).

## The genuinely hard part: proofs — and the precedent that skips them

This is a proof repo, and the codegen cites theorems inline (`:5992` cites
`conv2d_has_vjp3` / `conv2d_weight_grad_has_vjp`; `:6187` cites
`maxPool2_has_vjp3`). At the proof layer, **rank is in the type and cannot
be parameterized**:

```lean
abbrev Tensor3 (c h w : Nat) := Fin c → Fin h → Fin w → ℝ     -- Proofs/Tensor.lean:1490
noncomputable def conv2d {ic oc h w kH kW : Nat} ...           -- Proofs/CNN.lean:130-143
```

`Tensor3` is a curried function type with exactly three `Fin` indices;
`conv2d` is a triple sum with a 2-spatial pad predicate. The *sizes* are
already general (`h w kH kW` are variables) — the **arity** is fixed.
Nothing instantiates to 3D. A twin needs `Tensor4`, `Kernel5`, a quadruple
sum, a 6-conjunct pad predicate, and a 3D windowing algebra to replace
`winRow`/`winCol`/`winRowMod`/... (`Proofs/CNN.lean:1857-1930`) and its ~12
supporting theorems. Scale, by the 2D originals:

- `conv2d_has_vjp3` (`:330`) → `..._correct` (`:819`): ~490 lines
- `conv2d_weight_grad_has_vjp` (`:1128`) → `conv2d_weight_grad` (`:1622`): ~494 lines
- `maxPool2` + windowing algebra + `MaxPool2Smooth`/`IsArgmax`/`Argmax`: ~1000 lines

Call it **~2000 lines of new Lean**. It is additive (a 3D twin doesn't
refactor the ~8 downstream consumers of `conv2d_has_vjp3`), but it is
multi-month.

**And it is optional by this repo's own precedent.** `bilinearUpsample` has
no Lean proof. Neither do `unetDown` / `unetUp` — `grep -rln "unet"
LeanMlir/Proofs/` matches only `check_jacobians.py`. **The entire UNet
decoder shipped codegen-backed and FD-validated**, and `brats_demo.md` just
shipped the Dice loss the same way (`scripts/seg_loss_probe_check.py`,
grad-vs-central-FD 1.3e-07). conv3d can follow that road.

So the decision is **editorial, not technical**: what does the demo claim?

| route | cost | claim |
|---|---|---|
| codegen + FD | ~3-5 weeks | "a working 3D UNet", peer of the existing decoder |
| codegen + FD + proofs | + multi-month | "a *verified* 3D UNet", peer of the classifiers |

### DECIDED 2026-07-15: codegen + FD. No proofs.

Brett's call, and the target audience is the reason: **a working demo a
medical student can pick up and run.** That framing settles more than the
proof question — it sets the whole workstream's priorities:

- **Reproducibility outranks verification depth.** The thing that must not
  break is `./download_brats.sh && lake exe unet3d-brats-train`. That is
  already why this demo uses MSD Task01 over the Synapse-gated BraTS 2021,
  why `preprocess_brats.py` has a dependency-free NIfTI reader instead of a
  `pip install nibabel` step that PEP 668 blocks, and why the downloader
  parallelizes (3 h → 12 min). Hold that line.
- **FD validation is the correctness story, and it is a real one.** Every op
  gets a probe module + central-finite-difference check, exactly like
  `seg-loss-probe` and `flash-probe`. "Its gradient matches finite
  differences to 1e-7" is a claim a medical student can check by running one
  script. That's the right rigor for this audience.
- **Phase 5 is struck**, not deferred-with-intent. `Tensor4` and the 3D
  windowing algebra are not happening for this demo. If a future chapter
  wants a *verified* 3D UNet it re-opens as its own project with its own
  budget — and it should know going in that `Tensor3`'s arity is baked into
  the type, so the retrofit is a rewrite.
- Corollary for reviewers: **do not let the codegen cite 3D VJP theorems in
  comments the way `:5992` cites `conv2d_has_vjp3`.** There will not be any.
  Cite the FD probe instead, so the provenance stays honest.

## Phases

```
Phase 0  spike-for-real   compile AND RUN a rank-5 conv on a 128³ patch;
                          measure ms/step vs the 2D baseline.  (Gate 0)
Phase 1  rank-generic     parameterize convDimNumbers + convAttrBlock* by
                          spatial rank; verify existing 2D MLIR is
                          byte-identical (the SegLoss refactor's template)
Phase 2  ops              conv3d fwd/bwd, maxPool3d fwd/bwd, trilinear
                          upsample fwd/bwd; each FD-verified via a probe
                          module (the seg-loss/flash-attn pattern)
Phase 3  loader           uint8 corpus + patch sampler with foreground
                          oversampling; DatasetKind.brats3d
Phase 4  net              unet3dBrats spec + exe; train; per-class Dice
                          vs the 2D and 2.5D rows
Phase 5  (struck)         Tensor4 + conv3d VJP — NOT this workstream, see
                          the decision above. codegen + FD is the claim.
```

**Gate 0 is a real gate.** If rank-5 convolution runs but is (say) 20× off
the 2D conv's throughput per voxel, a 128³ patch step becomes unaffordable
on a box that is already MIOpen-conv-weak, and the whole workstream should
be re-planned around a CUDA box or dropped. Measure before building.

## Out of scope

Anisotropic kernels / spacing-aware resampling (BraTS is 1mm isotropic, so
this is a real nnU-Net concern that Task01 lets us skip), sliding-window
inference stitching (needed for whole-volume eval — its own line item),
deep supervision, 3D augmentation (rotation/elastic), and any other
Decathlon task.

# BraTS: 2.5D first, and let it decide 3D

**Opened 2026-09-09.** Scoped, not started. Companions: `planning/archive/unet3d.md` (the 3D scope,
with the XLA compile spike and the DECIDED-2026-07-15 entry: **codegen + FD, no proofs** — Phase 5
struck, not deferred) and `planning/archive/brats_demo.md` §Dimensionality / Workstream D (Gate D:
2.5D must beat 2D at matched budget before 3D starts). ⚠ `brats_demo.md:315,347` say "3D is not
implied" — those lines are inside the retracted appendix (banner at :270). The live position is
the opposite: affordable, unwarranted until Gate D.

## §0 The demo today

* **Data:** MSD Task01 (BraTS-derived), 484 volumes, modalities FLAIR/T1w/T1gd/T2w, MSD labels
  0/1/2/3 (1↔2 permuted vs raw BraTS; `Train.lean:457` has the WT/TC/ET map). Axial slices,
  tumour-bearing only (`--min-tumor-px 1`), **`--stride 2`**, z-scored over brain voxels then u8
  at ±5σ. 411/73 patients → 14,415 / 2,569 slices. Records: `4·S·S` image bytes + `S·S` mask,
  **no volume id, no z index** (`preprocess_brats.py:8-14, 184-227`).
* **Nets:** `unetBrats` from scratch (7.85M, 240²) and `r34UnetBratsOf true` (24.5M, 224²,
  ImageNet-bootstrapped encoder). The 4-channel stem is FRESH `[64,4,7,7]`; the bootstrap skips
  it by byte range (`stemFloats ic`, `MainUnetBratsR34.lean:71,241-247`) — **channel-count
  agnostic already.**
* ⚠ **Every published number is IREE-era** (0.740/0.910/0.869/0.856 at 10 ep); the only XLA row
  is 3 epochs (0.730). `content.tex` ~12496–12502 labels the **scratch** arm's number as the
  ResNet-34 arm. XLA is ~12× the ROCm box (217 ms/step, ~201 s/epoch from-scratch 240²);
  `unet-brats-r34` has no recorded ms/step at all. `runs/brats_*.log` are pre-shuffle-fix and void.
* Tier: codegen-only, FD-validated (`grep -i unet LeanMlir/Proofs/` → only `check_jacobians.py`).
  bf16: not available on the generic walk.

## §1 Stage A — close the 2D ledger (½ day, ~2 GPU-h)

Both arms at 10 epochs on XLA via `scripts/run_brats_r34_ab.sh 10 data/brats224` (drop the
`IREE_BACKEND` default at :23); record ms/step for `unet-brats-r34`. **Gate:** 0.740 ± 0.01
reproduces. Fix the `content.tex` label while there.

## §2 Stage B — 2.5D, the gate that decides 3D (1–2 days + a regen + ~2 GPU-days)

Neighbours are NOT recoverable from the records (no z, stride 2, non-uniform filter, and the
loader shuffles) — a regen is required.

1. `preprocess_brats.py:184-227` — `--context k`; neighbours from the UNFILTERED volume, kept
   centres still filtered/strided, replicate at volume ends. Regen to `data/brats224c3/`.
2. `ffi/f32_helpers.c:342,354` — `BRATS_CHANNELS` #define → parameter; `F32Array.lean:313` follows.
3. `Types.lean` ~:1084 `DatasetKind.brats224c3`; `Train.lean` ~:483 `brats224c3IO := { bratsIO
   with trainPixels := 12*224*224, channels := 12, labelBytesPerRecord := 224*224 }`; dispatch ~:497.
4. `MainUnetBratsR34.lean:119` `.convBn 12 64 7 2`; from-scratch `.unetDown 12 32`. Bootstrap:
   nothing.
5. `MainBratsPredict.lean` — T1gd backdrop is channel `4k+2`.
6. Run k=1 and the 2D control at **matched epochs, schedule and seed** (the `r34UnetBratsOf` A/B
   discipline). Add k=2 only if k=1 moved.

⚠ Host RAM: the loader holds the split as f32 — 11.6 GB today → **34.7 GB at 12 channels, 57.9
GB at 20**. ⚠ Decide ±1 slice (true adjacency) vs ±stride (matched spacing) deliberately; no doc
ever did. **Gate D:** ≥ +0.02 mIoU or ≥ +0.03 ET → Stage C. Small or zero → a FINDING; write it
into `brats_demo.md` §5 and stop. This is the highest information per GPU-hour on the table.

## §3 Stage C — Gate 0 for real (1–2 days, only if D passed)

Extend `planning/archive/conv3d_spike.mlir` from 16³ toys to a 128³×32ch patch at batch 2
through `ffi/libpjrt_ffi.so`; measure ms/step against Stage A. Add the **rank-8 maxPool backward
tile** (`MlirCodegen.lean:7773` widened; 512 MiB per pool at 128³/B2) — the one op NOT in the
spike and the largest memory risk. Gate: per-voxel throughput within ~3–5× of the 2D conv and no
OOM at 11.68 GiB; else re-plan at 96³ or drop.

## §4 Stages D–F — the 3D UNet (~4–5 weeks total if every gate passes)

* **D, rank-generic refactor (~1 week):** `convDimNumbers` (~:62) is the ONE definition every conv
  attr block calls — parameterize by spatial rank and ~35 of 37 conv sites follow; `samePad`
  (:54) list-ifies; `imageD : Nat := 1` on `NetSpec`; 7 hardcoded `ic*imageH*imageW` sites;
  58 four-element pattern matches; 6 dispatch walkers grow arms (parameterize `Layer` by rank,
  do not add `conv3d`/`maxPool3d` constructors). **Gate: emitted MLIR byte-identical for every
  existing 2D spec.** Make the `conv2d_has_vjp3` / `maxPool2_has_vjp3` citations at :4130-4131 /
  :7769 rank-conditional or the 3D MLIR states theorems that do not exist.
* **E, ops one at a time behind FD probes (~2 weeks):** maxPool3d fwd+VJP first (the unknown),
  conv3d fwd/dW/dx (transpose `[1,0,2,3,4]`, `reverse [2,3,4]`), trilinear fwd/VJP
  (`bilinearWeights1D` is already 1-D — a third `dot_general` factor; the 8-corner gather would
  not have been cheap), BN-3D (`[0,2,3,4]`), concat/split, the two loss blocks (re-verify the
  IREE-miscompile workaround at :4886-4898 on XLA at rank 5). Each op: a
  `tests/vjp_oracle/phase3/MainVjpOracle*3d.lean` clone + a `scripts/*_probe_check.py` clone;
  JAX side is `conv_general_dilated` with `NCDHW`. Bar: ≤ 1e-6 vs central FD. ⚠ Whole-net FD is
  NOT usable here — the family carries a ~15% analytic-vs-FD gap (`MainUnetBratsR34.lean:207-210`).
* **F, loader + net (~1 week + GPU):** whole-volume u8 corpus (~17 GB, fits the read-all pattern),
  patch sampler with ~33% foreground oversampling (the 2D deferral of oversampling does not
  transfer), `DatasetKind.brats3d`, `unet3dBrats` + `lean_exe unet3d-brats-train`. Memory: 96³×B2
  (~2.8 GiB activations) or 128³×B1–2 fits the 11.68 GiB arena; ~23M params ×3 with Adam.

## §5 Stage G — sliding-window whole-volume inference (separate line item)

The only thing that makes ANY BraTS number here comparable to the literature
(`brats_demo.md:110-118`: slice-level over tumour-bearing slices). Do it for the **2D** model
first, where it is cheap and immediately makes the 0.910 WT honest.

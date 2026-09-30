# Slice X — non-proof Lean + cross-cutting scope/attribution: rubric review 2026-09-30

Tree: `rubric-review` at `55ad3a5a`. Files: `LeanMlir.lean`, `LeanMlir/*.lean`, `LeanMlir/Verified/`,
`Bestiary/`, a placement skim of `apps/` + `demos/`, and repo-wide scope + attribution.
⚠ `LeanMlir/LEBytes.lean`, `LeanMlir/TicTacToe.lean`, `demos/MainAlphaZeroTtt.lean`,
`demos/MainTttEnv.lean` are being edited in the main checkout; findings in them are tagged
**(in-flight)** — re-check against the other session's commit.

Angles translated to program code: correctness = the code does what its docstring / README says;
proof-quality is N/A (no proofs in the slice).

## Verdicts

| angle | verdict | findings |
|---|---|---|
| correctness | request_changes | 6 (3 program, 3 Bestiary) |
| reuse | request_changes | 6 |
| scope | request_changes | 3 |
| attribution | request_changes | 6 |
| api-design | approve (one note under placement) | 0 |
| generality | approve | 0 |
| placement | request_changes | 6 |
| naming | request_changes | 1 |
| documentation | request_changes | 6 |
| proof-quality | n/a (no proofs) | 0 |

Bestiary findings (45 zoo specs, not trained by any path) are in each angle's list under a
`Bestiary` sub-heading; they share one gate: `tests/test_bestiary_params.py` against
`tests/bestiary_params.yml` (the golden `totalParams` per variant, which every param number below
is checked against).

## Findings

### correctness

- **X-cor-1** `LeanMlir/MlirCodegen.lean:24` (`MlirCodegen.unsupported` / `checkSupported`, landed
  after 09-26) — the guard's docstring says it refuses "a spec this emitter cannot render
  faithfully", but it checks only `pad := .valid` and `convBnAct ≠ .relu`. The 22 `Layer`
  constructors the module header (:9–15) lists as having **no emitter** (`separableConv`,
  `fireModule`, `mambaBlock`, `swinStage`, `denseBlock`, …) pass it. For them the forward walk
  emits `// UNSUPPORTED LAYER` (:2593) and the train walk `// UNSUPPORTED` (:6786), leaving
  `curSSA`/`curShape` unchanged, and `Layer.paramSlots` is `none` (Spec.lean:69, catch-all), so no
  parameters are declared. A shape-preserving block (`mambaBlock`, `swinStage`, `transformerDecoder`)
  therefore compiles and trains as the identity: the silent-wrong-function outcome `unsupported`
  was written to prevent. **Fix:** add a `| .separableConv .. | .fireModule .. | … => return some
  s!"{layer} has no MlirCodegen emitter"` arm covering the 22 constructors (or make the default arm
  of a `Layer.mlirEmitted : Layer → Bool` total and read it). Then replace the two `UNSUPPORTED`
  comment arms with unreachable-by-guard comments. **Gate:** `tests/TestYolov1Mutex.lean`-style
  expectThrow on a one-layer `mambaBlock` spec; `lake build Reference Apps`; `regen_jax_generated.sh
  check` is unaffected (JAX emitter untouched). **Cost:** 0 pins; +25 lines. **Size:** S.

- **X-cor-2** `LeanMlir/Verified/Attack.lean:16`, `:27` (`oneHotBatch`, `oneHotBatchPad`) — both
  decode the int32 label as byte 0 alone (`(labels.get! (4 * (start + j))).toNat`), which is exactly
  the `label % 256` bug `F32.readLabel`'s docstring (F32Array.lean:210–214) and
  `tests/TestLabelDecode.lean` exist to prevent. Today the PGD apps are MNIST / CIFAR (10 classes), so
  no number is wrong, but the docstring says "from int32-LE labels" and any ≥ 256-class use gets
  wrong one-hots silently. **Fix:** `let lbl := F32.readLabel labels (start + j)` at both sites, and
  define `oneHotBatch labels start bs d1 := oneHotBatchPad labels start bs d1 (start + bs)` (the
  bodies are identical but for the loop bound). **Gate:** `lake build Apps`; `mnist-linear-pgd` smoke
  (clean acc unchanged). **Cost:** 0 pins, −8 lines. **Size:** S.

- **X-cor-3** `LeanMlir/E4M3Quant.lean:8`, `:26` (`roundE4M3`) — the module says it implements "the
  *same* E4M3 … round-to-nearest grid as the numpy oracle" and the function "Mirrors `to_e4m3`". The
  grid matches, the tie rule does not: `Float.round` is C `round` (ties away from zero), the oracle's
  `np.round` (`scripts/demos/mnist_e4m3_demo.py:60`) is ties-to-even, and so is OCP FP8 E4M3. Ties
  (`a/step` exactly `k + ½`) are rare on f32 data but reachable (any value on a midpoint of the scaled
  grid). **Fix:** either round half-to-even (`let r := Float.round y; if (r - y).abs == 0.5 && r % 2
  != 0 then r - sign(y) else r`) or say "ties away from zero, unlike the numpy oracle's
  ties-to-even" in both docstrings. **Gate:** a `#guard roundE4M3 (2.5 * 2^-9)` pin once public, or a
  `#eval` in a scratch file; `mnist-linear-e4m3-verified` smoke. **Cost:** 0 pins. **Size:** S.

### reuse

- **X-reu-1** LE u32 readers hand-rolled beside `F32.readLabel` (F32Array.lean:215), which
  `LEBytes`'s docstring names as THE reader: `LeanMlir/Verified/Train.lean:706` (`rd32` lambda),
  `:723`, `:749`, `:814` (all 4-aligned, so `F32.readLabel pre (off / 4)`),
  `demos/MainBigramShakespeare.lean:69`, `demos/MainBratsEval.lean:59` (`readU32`, byte offset),
  `demos/MainAlphaZeroTtt.lean:73` (**in-flight**). Writers too: `LeanMlir/GradcheckHelpers.lean:45`
  `writeBinF32` is a `pushF32LE` loop spelled out; `demos/MainRsBands.lean:124` packs a u32 with
  `ByteArray.mk #[…]`; `demos/MainNqsIsing.lean:72` `pushF64`, `:187` `pushU64` and
  `demos/MainDiffusion2d.lean:199` `floatsToBytes` are writers `LEBytes` lacks (the in-flight
  session is adding `pushU64LE` to `LEBytes`). **Fix:** move the reader into `LEBytes` as
  `readU32LE (ba) (byteOff)` (keep `F32.readLabel` as `readU32LE lbl (4*i)` or repoint its callers),
  add `pushU64LE`/`pushF64LE`/`ofFloats`, and delete the copies; `GradcheckHelpers` may import
  `LEBytes` (both import-free). **Gate:** `lake build Apps Reference`, `test-dataset-record-sizes`,
  `TestLabelDecode`, a streamed-val smoke (`imagenet` shim preamble path). **Cost:** 0 pins; −60
  lines. **Size:** S–M. (carried in part: audit_v2 §10 LE-writer fold, which landed the writers only)

- **X-reu-2** `LeanMlir/MlirCodegen.lean:861` `emitChannelSplitGrad` (private, **zero callers**) is the
  exact body the UNet concat-split backward inlines at `:7110–7138` (two `stablehlo.slice` ops,
  differing only in SSA names). `dwConvAttrBlock` (:128) and `dwConvAttrBlockFull` (:137) are
  also private with zero callers. **Fix:** call `emitChannelSplitGrad s!"{r.pos}"` from the
  `.unetUp` backward arm (rename its SSAs to the `%uut_a`/`%unet_skip_g{e}` the arm needs, or keep
  the arm and delete the helper); delete the two `dwConvAttrBlock*`. **Gate:** reference-codegen
  harness / `test-unet-forward` plus a byte-diff of `generateTrainStep ReferenceNets.r34UnetBrats`
  before/after. **Cost:** 0 pins, −45 lines. **Size:** S.

- **X-reu-3** Four float parsers re-implement `ViTGradcheck.parseFloat?`
  (`LeanMlir/GradcheckHelpers.lean:25`, exact via `Float.ofScientific`):
  `demos/MainPongDqn.lean:120` ≡ `demos/MainAlphaZeroTtt.lean:78` (**in-flight**, identical 15 lines),
  `demos/MainNqsIsing.lean:625` (no exponent), `demos/MainRsBands.lean:158` (no sign, `toNat!`),
  `demos/MainGwDetect.lean:211,215` (inline `splitOn "."`). `parseArg` is md5-identical ×4
  (`MainAraslSigns:129`, `MainPlantLeaf:208`, `MainGwDetect:197`, `MainRsBands:153`) and the
  `kv`/`natArg`/`floatArg` lambdas repeat at `MainPongDqn:140`, `MainAlphaZeroTtt:376`,
  `MainTttEnv:33` (in-flight). **Fix:** see X-pla-2 (one `LeanMlir/CliArgs.lean`). **Size:** S.

- **X-reu-4** BraTS scoring re-implemented in `demos/MainBratsEval.lean` (new, 145b911b):
  `regionCounts` + `diceOf` (:72–91) redo the Dice loop at `LeanMlir/Train.lean:1069–1086` with the
  same convention; `readI64` (:64) duplicates Train.lean:1027's lambda; `regions` (:55) copies
  `segRegions` (Train.lean:411) — its own docstring says "the same table"; the `net=`/`ctx=` →
  (spec, kind) resolution (:111–123) is the code at `demos/MainBratsPredict.lean:163–176`.
  **Fix:** make `regionCounts`/`regionDice`/`readI64LE` public in `Train.lean` (or a `SegMetrics`
  module) and call them from both; read `(datasetIO kind).segRegions` (expose it) in BratsEval; add
  `ReferenceNets.bratsNetOf (skips : Bool) (ctx : Nat) : NetSpec × DatasetKind`. **Gate:**
  `brats-eval` on the committed anchor checkpoint reproduces its logged pooled number (the tool
  already self-checks this to 4e-4). **Cost:** 0 pins, −60 lines. **Size:** S–M.

- **X-reu-5** Reference `NetSpec`s still spelled in Main files although `ReferenceNets` exists for
  exactly this (audit_v2 d9b15213): `tinyDdpmUnet` md5-identical ×3 (`demos/MainMnistDdpmTrain.lean:21`,
  `MainMnistDdpmSample.lean:25`, `probes/MainMnistDdpmScore.lean:56`) — a trainer/sampler pair whose
  `buildPrefix` must agree, the drift hazard `ReferenceNets`' docstring names; the R34-FPN detector
  ×3 (`MainYolov1VisdroneFpn:100`, `MainYolov1NeuDetFpn:81`, `probes/MainFpnTrainEmit:20`) and
  R50-FPN ×2, differing only in `name`; `cifar8w` ×3 (`MainAraslSigns:37`, `MainGwDetect:119`,
  `MainRsBands:57`) differing in inC/H/W/nOut; `MainPlantLeaf:46` = `ReferenceNets.resnet34` with a
  38-way head. **Fix:** `ReferenceNets.tinyDdpmUnet`, `r34FpnDetOf tower`, `r50FpnDetOf tower`,
  `cifar8wOf inC H W nOut`, and `{ resnet34 with layers := … }` for PlantLeaf. **Gate:** `lake build
  Apps`; byte-diff each demo's `generateTrainStep` output before/after. **Cost:** 0 pins, −150
  lines. **Size:** M.

- **X-reu-6** Private copies of library plumbing in demos: the "compile if IREE" shell-out ×3
  (`MainMnistDdpmTrain:86–97`, `MainMnistDdpmSample:40`, `probes/MainGradCAM:105–120`) while
  `Train.lean:46` `runIree` is private; `probes/MainGradCAM:92` `argmaxN` re-implements
  `F32.argmaxN`; P6 PPM headers written inline at `MainBratsPredict:300`, `MainMnistDdpmSample:205,233`
  instead of `Cam.writePPM`; `NqsIsing:52` `fmt` byte-identical to `FloatFmt.fmt` and `:49` `piF` ≡
  `Ddpm.piF`. **Fix:** expose `runIree` as `NetSpec.compileArtifact` beside `graphArtifact` and call
  it; use `F32.argmaxN`, `Cam.writePPM`, `FloatFmt.fmt`, `Ddpm.piF`. The four classifier-demo `fmt`s
  are a different function (round then `toString`, cut at 10 chars) — folding them changes printed
  text; leave or get sign-off. **Gate:** `lake build Apps`. **Cost:** 0 pins, −60 lines. **Size:** S.
  (carried in part: audit_v2.md:429 "`fmt` ×6 in demos", "8 inline iree-compile")

### scope

Module-level reachability (import closure from every lakefile lib root, every `lean_exe` root,
the `Apps` globs and `tests/AuditAxioms`): 500 of 596 tracked `.lean` modules reached. Of the 96 not
reached, the `jax/` package (its own lakefile), `planning/`, `runs/`, `scripts/` Lean files,
`IRPrint` (deliberate), `AuditAxiomsHeavy` (certs-heavy.yml) and the two `ScorecardSDPFull*`
(`scripts/certs/check_sdpfull.sh`) are accounted for. Nothing in `LeanMlir/` program code is
reachable only through the `LeanMlir.lean` umbrella or the `Reference` lib. The remainder:

- **X-sco-1** `tests/` — 21 `lake env lean` scripts that no workflow, `regen_verified_mlir.sh`, or
  gate script runs (grep of `.github/`, `scripts/`, `run.sh`): (a) the op-level `iree-compile`
  smokes from the archived per-net plans — `TestGelu`, `TestSwish`, `TestSigmoid`, `TestSE`,
  `TestRelu6`, `TestSoftmaxRow`, `TestDepthwise`, `TestDepthwiseStrided`, `TestPerChannelBn`,
  `TestConvNeXtBlock`, `TestConvNeXtStrided`, `TestVeclnGammaSgd` (cited only by
  `planning/archive/verified_*.md`); (b) render/tie scripts `TestAdamOpTie`, `RenderAdamSmoke`,
  `TestRmsPropOpTie`, `TestConvNeXtTrainPC`, `TestConvNeXtTTrainPC`, `TestViTTrainPC` (cited only by
  each other and `tests/ViTRender.lean`); (c) `#guard` gates the lakefile or a script calls
  authoritative — `TestVariantPredicates` ("the AUTHORITY", `scripts/gates/gen_mlir_manifest.py:18`),
  `TestDropPathRamp` (lakefile:1238), `TestBatchedEmitTie` (lakefile:1134), `TestR50Contract`
  (lakefile:1617), `TestXlaPadOps` (`xla_pad_op_check.py`, manual). Group (c) still passes
  (`TestVariantPredicates`: "104 variant spellings, 5 axes, no collision"; `TestDropPathRamp`: all
  guards green, checked 09-30 against main's oleans) — its issue is a gap, below. Groups (a)/(b) are
  satellites whose targets closed (the proven renders and `iree.yml`'s ViT gradchecks superseded
  them). **Fix:** delete (a) and (b) (≈ 2.4k lines) unless one is promoted into `iree.yml`'s smoke
  list — `TestConvNeXtBlock` (283 lines, block + gradcheck) is the only one with a check the CI list
  lacks. **Cost:** 0 pins (none is in AuditAxioms, the yaml, or the book — grep); `tests/README.md`
  rows. **Size:** S.

- **X-sco-2** `LeanMlir/Train.lean:181–184` — `LossKind.floatTargetMse` arm of `compileVmfbs` is
  "currently unused but reserved" (DDPM bypasses `compileVmfbs`). A reserved arm is a stated
  non-roadmap placeholder. **Fix:** make it throw ("DDPM trains through
  `demos/MainMnistDdpmTrain.lean`'s own compile, not `compileVmfbs`"), so a caller that reaches it
  learns why, or route DDPM through `compileVmfbs` (X-reu-6's `compileArtifact`). **Size:** S.

- **X-sco-3** `.github/workflows/certs.yml:15–66` — the Bestiary param-drift guard (:270–289) runs
  on push/PR only when `tests/bestiary_*` changes; its `paths` omit `Bestiary/**` and
  `LeanMlir/Spec.lean` (the `paramSlots` / `nParamsUntrained` it measures), so a PR that edits a spec
  or the counter is caught only by the nightly cron. **Fix:** add both paths to both `paths` blocks.
  **Size:** S. (Coverage otherwise complete: all 45 files are `bestiary-*` exe roots, built by
  certs.yml, and in the golden table; no demo or app imports a Bestiary module.)

### attribution

The full census is under **Attribution census** below. Findings, highest value first:

- **X-att-1** `README.md` (no section), `TRUST.md`, the book front matter — the external stack the repo
  mirrors or runs on is named but never credited: timm (Wightman, `pytorch-image-models`, pinned
  1.0.28 in `requirements-timm-lock.txt`; the verified specs are stated at timm parity, e.g.
  `Verified/NetsCore.lean:1400`) — the RSB paper is cited (content.tex:13667) but the software never
  is; JAX (README.md:109,121 name only); OpenXLA XLA/PJRT/StableHLO and IREE (names only, "OpenXLA"
  0 hits); Mathlib, doc-gen4, leanblueprint/plasTeX (0 hits in the five top-level docs). There is no
  acknowledgements / third-party section anywhere (grep acknowledg|third-party: 0). **Fix:** a
  "Built on" section in README.md listing each with URL (+ Bradbury et al. 2018 for JAX, the
  mathlib CPP 2020 paper), mirrored as one paragraph in the book's introduction. **Size:** S.

- **X-att-2** `demos/README.md:596–597` says the AlphaZero loop "follows" alpha-zero-general; neither the
  README nor `demos/MainAlphaZeroTtt.lean` (**in-flight**) nor the book names its author (Surag Nair)
  or URL — the credit exists only in `planning/alphazero_ttt_demo.md:11`. Following an identifiable
  implementation's loop is exactly what the rubric requires credit for. **Fix:** "Surag Nair's
  alpha-zero-general (github.com/suragnair/alpha-zero-general)" in the demo docstring, the README
  sentence, and the book's RL paragraph. **Size:** S.

- **X-att-3** `LeanMlir/Verified/NetsCore.lean:5` — the single source of every verified net (ResNet-34/50,
  MobileNetV2, MobileNetV4-Conv-M, EfficientNet-B0, ConvNeXt-T/S/B, ViT-Tiny/S/B, drop-path) cites no
  architecture paper; only the three `apps/imagenette/Main{ConvNeXt,EfficientNet,ViT}Verified.lean`
  do. Same for `LeanMlir/ReferenceNets.lean:14` (`resnet34`, `unetBrats`: He et al. 2016 /
  Ronneberger et al. 2015 missing; ConvNeXt is credited). **Fix:** a "Sources" paragraph in NetsCore's
  module docstring (He 2016, Sandler 2018, Qin 2024, Tan & Le 2019, Liu 2022, Dosovitskiy 2021 /
  Touvron 2021, Huang 2016) and the two names in ReferenceNets. **Size:** S.

- **X-att-4** Methods named without their paper in slice modules: `LeanMlir/Ddpm.lean:86–103` flow
  matching, reflow and minibatch-OT pairing (Lipman et al. 2023; Liu et al. 2023; Tong et al. /
  Pooladian et al. 2023) — the file credits Ho, Nichol & Dhariwal and Song for everything else;
  `LeanMlir/Verified/Attack.lean:4` PGD (Madry et al. 2018) and spectral-norm projection (Miyato et
  al. 2018); `LeanMlir/E4M3Quant.lean:4` E4M3 (Micikevicius et al. 2022); `LeanMlir/SyncBnCheck.lean:60`
  "Chan's" variance (Chan, Golub & LeVeque 1983); `LeanMlir/Verified/Smoothing.lean:3` Clopper–Pearson
  (1934, name only). **Fix:** one citation each in the module docstring. **Size:** S.

- **X-att-5** Demos and data outside the slice's program modules but in its census: `demos/MainTinyStories.lean:3`
  + `demos/README.md:685` never credit TinyStories (Eldan & Li 2023; 0 hits in the book, which has an
  appendix dataset table at content.tex:17094–17139 that omits it); `demos/MainNqsIsing.lean:3` lacks
  Carleo & Troyer 2017 (the book has it at :16559); `demos/MainDiffusion2d.lean:3` lacks Ho 2020 /
  Lipman 2023 / Noé 2019 (book :16121); `demos/MainPongDqn.lean` credits Mnih 2015 but not the double-DQN
  target (van Hasselt 2016), `demos/MainBlackjackDqn.lean:4` neither; `ffi/f32_helpers.c:3634` PUCT
  (Silver 2017); FPN/YOLO/DIoU demos (Redmon 2016, Lin 2017, Zheng 2020); `demos/MainUnetBratsTrain.lean:6`
  (Ronneberger 2015). Dataset licences are absent for MNIST, CIFAR-10, Imagenette, VisDrone, MSD/BraTS,
  tiny Shakespeare, TinyStories (present for GWOSC, ArASL, PlantVillage/PlantDoc, EuroSAT, MapBiomas).
  **Fix:** one line per demo docstring; a TinyStories row and a licence column in the appendix table.
  **Size:** S–M.

- **X-att-6** Bestiary entries credited by author-year only, missing title/venue/arXiv:
  `AlphaZero.lean` (also omits AlphaGo Zero, Silver et al. Nature 2017, which its 40-block net
  models), `Evoformer.lean` (Jumper 2021), `Mamba.lean` (arXiv:2312.00752), `ShuffleNet.lean`,
  `SwinT.lean`, `UNet.lean`; `LLaVA.lean` models 1.5 but credits only LLaVA-1; `DeepLabV3Plus.lean`'s
  MobileNetV2 backbone uncredited; `ResNet.lean` names timm's RSB recipe without Wightman et al. 2021
  (the book cites it); YOLO v5/v8/v11 are "Ultralytics 20xx" with no repo/version. **Fix:** one line
  each. **Size:** S.

### placement

- **X-pla-1** `LeanMlir/Pong.lean:1,9`, `LeanMlir/Blackjack.lean:1,10`, `LeanMlir/TicTacToe.lean:2,17`
  (**in-flight**) import `LeanMlir.FloatFmt` and `open FloatFmt` but call `fmt` nowhere (grep: 0 uses in
  each); the import exists only so the demos get `FloatFmt` transitively — a forwarding import, the
  residue of placement §5.2's `export FloatFmt (fmt)` removal. `TicTacToe` also imports
  `LeanMlir.LEBytes` with 0 uses at HEAD (the in-flight edit may start using it). `FloatFmt`'s
  docstring ("so the pure-Lean game modules `Blackjack` and `Pong` share it") is then false.
  **Fix:** drop the import and `open` from the three library modules; add `import LeanMlir.FloatFmt`
  to `MainBlackjackEnv`, `MainBlackjackDqn`, `MainPongEnv`, `MainPongDqn`, `MainTttEnv`,
  `MainAlphaZeroTtt` (they already `open FloatFmt`); reword FloatFmt's docstring to "the RL demos'
  tables". **Gate:** `lake build Apps`, `import_audit.py implied`. **Size:** S.

- **X-pla-2** `LeanMlir/GradcheckHelpers.lean:17–39` — the general float-token parser
  (`parseFloat?`/`parseFloat`) lives in namespace `ViTGradcheck` inside an iree-run-module gradcheck
  harness, and three demos (`MainAraslSigns:156`, `MainPlantLeaf:244`, `probes/MainSegLossProbe:48`)
  import it from there; four more demos re-implement it (X-reu-3). **Fix:** new import-free
  `LeanMlir/CliArgs.lean` with `parseFloat?`, `parseFloat`, `parseArg`, `natArg`/`floatArg`
  (namespace `CliArgs`); `GradcheckHelpers` imports it; repoint the 3 users and delete the 4 copies and
  4 `parseArg`s. **Cost:** `ViTGradcheck.parseFloat` appears in 5 non-planning doc lines and
  `tests/TestSgdRenderTie.lean:71–72`; `docstring_ref_baseline.txt` may need a line. **Size:** S.

- **X-pla-3** `LeanMlir/SyncBnCheck.lean` (307 lines) is imported only by the four
  `tests/*SyncBnCheck.lean` exes (grep), and `LeanMlir/README.md:14` files it as "support for the gates
  in tests/". The repo already has the home for that: the `TestSupport` lean_lib
  (lakefile:346, `tests.ViTRender`). **Fix:** `git mv LeanMlir/SyncBnCheck.lean tests/SyncBnCheck.lean`,
  add it to `TestSupport`'s roots, repoint the four imports and the README row. (`VjpOracleNets` stays:
  the `jax/` package imports it; `GradcheckHelpers` stays until X-pla-2 moves its demo-used half.)
  **Gate:** `lake build TestSupport resnet34-syncbn-check mobilenetv2-syncbn-check …`,
  `check_target_names.sh`. **Size:** S.

- **X-pla-4** Classifier-demo kit copied ×4 (`MainAraslSigns`, `MainPlantLeaf`, `MainGwDetect`,
  `MainRsBands`): `xs` (4 lines), `permutation` (11) md5-identical at Arasl:81/87, Plant:67/73,
  Gw:149/155, Rs:87/93; `scoreSet` (17–20 lines, Arasl:111, Plant:130, Gw:179, Rs:135) differing only
  in the class-count source and image fetch; `gather` (Arasl:100 ≡ Gw:168 up to `nPix`). The
  LM demos likewise: `sampleToken` `MainTinyGptShakespeare:175` ≡ `MainTinyStories:177` (32 lines),
  `loadVocab`/`reverseVocab` Bigram:45/58 ≡ TinyGpt:155/167, `readFloats` ×3, `asSegLabels`
  TinyGpt:218 ≡ TinyStories:97. And the FPN detectors: `inferDump` ×4 (46–53 lines, VisdroneFpn:357,
  NeuDetFpn:206, NeuDet448:95, archive/VisDrone448:63), `tagFromEnv` NeuDetFpn:187 ≡ VisdroneFpn:298,
  VisdroneFpn:230–356's 11 single-knob env readers vs NeuDetFpn:154–172's generic `envNat`/`envFlag`.
  **Fix:** `LeanMlir/SmallClassifier.lean` (`xs`, `permutation`, `scoreSet nClasses`, `gather`),
  `LeanMlir/TextLm.lean` (`sampleToken`, `loadVocab`, `reverseVocab`, `readFloats`, `asSegLabels`,
  `lrAt warmup`), `LeanMlir/Detect.lean` (`inferDump`, `envNat/envFlag/envPct/envStr`). No pair of
  Main files is > 70 % identical, so no whole-driver extraction is proposed. **Gate:** `lake build
  Apps`; one smoke per family. **Cost:** 0 pins; −450 lines. **Size:** M.

- **X-pla-5** The packed `[θ|m|v]` Adam step + three-way unpack
  (`(p.append m).append v` → `trainStepAdamF32*` → 3× `F32.slice`) is written out at 11 sites in 9
  demos (`MainBlackjackDqn:205`, `MainNqsIsing:563,835`, `MainAlphaZeroTtt:541` in-flight,
  `MainMnistDdpmTrain:169`, `MainPlantLeaf:378`, `MainAraslSigns:256`, `MainRsBands:316`,
  `MainGwDetect:314`, `MainDiffusion2d:415`). **Fix:** `structure AdamState (θ m v : ByteArray)` +
  `LowererSession.adamStep` in `LeanMlir/Train.lean`. **Size:** S–M, −70 lines. Also, in-flight:
  `demos/MainAlphaZeroTtt.lean:64,70` bind `lean_ttt_gather_aug` / `lean_ttt_targets` in the Main
  file while every other `lean_ttt_*` extern lives in `LeanMlir/TicTacToe.lean`; the six
  `lean_mcts_*` externs belong there or in a new `LeanMlir/Mcts.lean`.

### naming

- **X-nam-1** `LeanMlir/GradcheckHelpers.lean:17` — namespace `ViTGradcheck` holds a generic adjoint
  gradcheck (`adjointGradcheck*`, also used by `TestSDPA`/`TestMHSA`, no ViT specifics) and the
  float parser (X-pla-2); the module is `GradcheckHelpers`. **Fix:** rename the namespace to
  `Gradcheck` with X-pla-2's move (6 importers, 5 doc lines, no pins). **Size:** S.

### documentation

- **X-doc-1** `lakefile.lean` hand-kept counts, all stale: `Proofs` "23 roots" (:41; actual 24),
  `Certs` "185 roots reaching 235 proof modules" (:73; 194 roots), `Apps` "111 modules" (:329; 88
  tracked `.lean` files under `apps/` + `demos/`), "227 of them" exes (:326; 208 `lean_exe`), the
  umbrella "~90 of the 261 `LeanMlir/` modules" (:26; 292). `.github/workflows/certs.yml:244–259`
  says 111 and then 113 — the 09-29 AlphaZero commit bumped 111 → 113 while the true count is 88.
  `Reference`'s docstring "no CI job uses it" (:314) is false: `blueprint.yml:186` builds it.
  **Fix:** delete the counts (or generate them in a check), fix the Reference sentence. **Size:** S.

- **X-doc-2** `LeanMlir/IreeRuntime.lean:38` `backendName` — "which shim this binary was linked
  against": since the dlopen scheme (lakefile `lowererLink`) nothing is linked; the C side answers
  `lowerer_active_name()` (ffi/iree_lean_ffi.c:22), i.e. which shim `$LEAN_MLIR_LOWERER` loaded.
  `create` (:14) still leads with "Load a `.vmfb` … onto the default CUDA device", the IREE-only path;
  `linearTrainStepV`/`mlpTrainStepV` say "through the generic IREE invoke". **Fix:** reword the four
  docstrings for the lowerer-agnostic dispatch. **Size:** S.

- **X-doc-3** `LeanMlir/MlirCodegen.lean:7128` emits the comment `channelSplitHasVJP ↔ channelConcat`
  into every UNet train step; no declaration `channelSplitHasVJP` exists (grep over `LeanMlir/`).
  `:7496` cites `Pointwise.lean: elemwiseProductHasVJP`; the witness is in
  `Proofs/Foundation/Tensor.lean:394`. (These are the only two stale proof citations among the
  codegen's emitted comments — every other `*HasVJP*`/`_faithful` name cited resolves.) **Fix:** name
  the real declarations or drop the parenthetical. **Size:** S.

- **X-doc-4** `LeanMlir/Cam.lean:21` — "32-stop subsample of matplotlib's viridis … for the full
  256-entry LUT": there are 21 stops plus 11 padding entries that the lerp never reads (`i1 < nReal`),
  no LUT is built, and `let _ := n` (:49) is dead. **Fix:** "21 stops, linearly interpolated", drop
  the padding and `n`. **Size:** S.

- **X-doc-5** `LeanMlir/README.md:10–16` — the module table omits `TicTacToe` (landed 120021f2);
  its row 16 "support for the Chapter 10 demos" should gain it. `LeanMlir/Train.lean:471`'s panic
  still says "not supported by phase 3; use phase 2 (jax/)", the phase jargon the comment above it
  was just cleaned of. **Size:** S.

#### Bestiary

- **X-bes-1 (correctness)** `Bestiary/AlphaZero.lean:24,51–52,97–110` — the chess/shogi net is
  `.residualBlock 256 256 40 1` ("40 for AlphaZero chess/shogi") plus `.dense (73*8*8) (73*8*8)`
  (:110), a 4672×4672 layer of 21.8M parameters that the printed note (:186–188) calls "identity … no
  further projection needed". AlphaZero (Silver et al., Science 2018) used 19 residual blocks and a
  conv-only policy head; 40 blocks is AlphaGo Zero's larger net. The golden count is 69,352,658
  against the docstring's "~46M" (:52). `:35–36,185` "LN γ/β scalar simplification" is stale
  (`slotsLN d` is per-channel `[d]`, Spec.lean:61). **Fix:** 19 blocks, drop the dense, restate the
  count; or relabel the variant "AlphaGo Zero 40-block". Regenerate the golden row. **Size:** S.

- **X-bes-2 (correctness)** Parameter claims that disagree with the file's own spec (golden count in
  parentheses): `YOLO.lean:75` fastYolo "~163M" (66,648,926); `DETR.lean:43` tinyDETR "~3M"
  (314,809); `Pix2Pix.lean:75,200` "~70M" (62,068,099); `NeRF.lean:71,82,156` "~528K" (593,924);
  `WaveNet.lean:66–78` "3 × 10 layers, ~4.5M" for single-stack specs (426,464);
  `ShuffleNet.lean:54–57` 1.0/2.4/5.4M (0.73/1.91/5.64M) and a 1.5× row with no spec;
  `AlphaGo.lean:22,25` ~4.5M/~5M (3.88M/3.98M); `Inception.lean:14` GoogLeNet "5M" (7,005,832) and
  `:61,215` "within ~20%" (v4 is −21.7% per `tests/bestiary_timm_report.md`); `DenseNet.lean:53–58`
  quotes 1000-class paper counts beside 10-class specs (201: 18,125,258 vs "18.6M");
  `SegFormer.lean:44–47,189–194` says spatial-reduction attention leaves the count "unaffected" while
  its own B0/B2/B5 come out 26–29 % under its table (2.64M vs 3.7M …) — undisclosed;
  `WRN.lean:9–12` "WRN-28-10 … a quarter of ResNet-1001's parameters" (36.5M vs ~10.2M);
  `Diffusion.lean:58,100,198` attributes ADM's ~550M ImageNet-256 model to DDPM (DDPM has no ImageNet
  model; Dhariwal & Nichol 2021 do). **Fix:** replace each number with the golden count or say what the
  spec leaves out. **Size:** M (one pass over ~12 files).

- **X-bes-3 (correctness)** Flatten fan-in not what the layer list computes: `AlexNet.lean:73–89`
  (11×11/s4 stem on 227 → 57, three `.maxPool 2 2` → 7×7, but `.dense (6*6*256)`); `YOLO.lean:96–130`
  (stride-1 stem and a "stride-2 last conv" that is stride 1, so 448 → 28×28, but
  `.dense (7*7*1024)`). `NetSpec.validate` stops checking at `.flatten` (Spec.lean:598), so both pass.
  The printed counts are therefore the paper's head on a body that would not produce its input.
  **Fix:** `.maxPool 3 2` and real strides, or a docstring line saying the fan-in is pinned to the
  paper's value. Gap: `validate` could carry the spatial shape through `.flatten`. **Size:** S.
  Smaller statement errors in the same pass: `Highway.lean:37–39` ("ResNet is the special case T ≡ 1"
  — with T ≡ 1 there is no carry; ResNet sums both paths), `Highway.lean:51–54` ("Highway-50" is a
  width-50 single layer, not 50-deep), `GPT.lean:177` ("GPT-1 predates BPE", contradicting `:60`),
  `Xception.lean:43–53` (entry-flow blocks have 2 separable convs, not 3; "linear bottlenecks" are
  MobileNetV2's, not v1's), `Nystromformer.lean:11,22` + README:106 (Nyström 1930, not 1928).

- **X-bes-4 (documentation)** Stale statements about the codebase: `SqueezeNet.lean:191–193` ("no
  maxPool with separate kernel and stride" — `.maxPool 3 2` exists and `ResNet.lean:67` uses it);
  `YOLO.lean:67–68,430–432` (Activation is "{relu, relu6, identity}"; it also has swish/hSwish/gelu,
  and v5+ is SiLU per `:241`); `CLIP.lean:112,251` and `GPT.lean:72–137` (`.transformerEncoder`
  "doesn't distinguish causal" — it has `causalMask`, never set by the GPT decoders);
  `UNet.lean:179–193` ("codegen emits UNSUPPORTED", "follow-up entry" — the UNet trains via
  `ReferenceNets.unetBrats`, `StableDiffusion.lean` exists); `ShuffleNet.lean:131–133` (v2 "worth its
  own entry" — it has one); `Mamba.lean:55,158` ("bundled into one axiom" — the repo has zero
  axioms); placeholder "Chapter-N" at `AlphaZero.lean:13`, `Mamba.lean:51,54`. `Bestiary/README.md`
  :171,179 "41 binaries, 189 variants" (45 exes, 201 golden rows); `:137` tells a new entry to call
  `archStr`/`totalParams`/`validate`, but `tests/test_bestiary_params.py` parses `NetSpec.summarize`'s
  output, so an entry following the README escapes the guard; `:149–151` "any spec is one line from
  training" is false for the 22 no-emitter constructors (X-cor-1); `:90` implies the AlphaZero spec is
  trained by the tic-tac-toe demo (it is AlphaGo's stack). `lakefile.lean:734` still says the demo
  runs "on the bestiary's tinyAlphaZero body" (stale since 120021f2). `tests/bestiary_timm_report.md`
  lists `bestiary-convnext` rows for a deleted exe and stale MobileViT/ShuffleNetV2 counts —
  regenerate. **Size:** S–M.

- **X-bes-5 (placement/reuse)** Token embeddings in `GPT`, `BERT`, `Nystromformer`, `CLIP`,
  `StableDiffusion`, `LLaVA` are `.dense vocab d` (a spurious `d`-bias, no position table) although
  `Layer.tokenPositionEmbed` / `.lmHead` exist; with them GPT-2 small is exactly the reference
  124,439,808 (golden 123,654,144 − 768 + 1024·768). The ResNet-50/101 body is re-spelled in
  `CLIP.lean:97–103`, `DETR.lean:65–70,91–96`, `MaskRCNN.lean:~82`, `DeepLabV3Plus.lean:~88` and
  `ResNet.lean` (each Bestiary module has a `main`, so they cannot import each other). **Fix:** use
  `.tokenPositionEmbed` (+ `causalMask := true` for decoders) and a main-less `Bestiary/Common.lean`
  (or `ReferenceNets`) exporting `resNetBody`. Regenerate golden rows. **Size:** M.

## Attribution census

Method → module docstring credit (full = authors + year / arXiv; name-only = method named, no
source). Non-proof modules; spot-checked by reading each "none" header.

| module | method(s) | credit |
|---|---|---|
| `LeanMlir/Cam.lean` | CAM / Grad-CAM | full (Zhou 2016, Selvaraju) |
| `LeanMlir/Ddpm.lean` | DDPM, cosine schedule, DDIM, score-SDE | full; flow matching / reflow / minibatch OT none (X-att-4) |
| `LeanMlir/Blackjack.lean` | Blackjack-v1 env | full (Sutton & Barto Ex. 5.1, Gymnasium); the published-table arm cites "IEEE 1299399" by number only |
| `LeanMlir/TicTacToe.lean` | minimax, scripted players | routine — no credit needed |
| `LeanMlir/Verified/Smoothing.lean` | randomized smoothing; Clopper–Pearson; Acklam probit | Cohen 2019 full; CP name-only; Acklam named |
| `LeanMlir/Verified/Attack.lean`, `PgdGen.lean` | PGD, spectral norm | name-only |
| `LeanMlir/E4M3Quant.lean` | FP8 E4M3 | name-only |
| `LeanMlir/SyncBnCheck.lean` | sync-BN, Chan variance | name-only |
| `LeanMlir/Verified/NetsCore.lean` | 7 architectures + drop-path | none |
| `LeanMlir/ReferenceNets.lean` | ResNet-34, UNet, ConvNeXt-T | ConvNeXt full; ResNet, UNet none |
| `LeanMlir/Verified/Train.lean` | AdamW, LAMB, RMSProp, EMA, FP8 | none / name-only |
| `LeanMlir/Types.lean`, `Train.lean` | Mixup, CutMix, RandAugment, label smoothing, RSB, EMA | none (the C side credits Zhang 2017, Yun 2019, Cubuk 2019 at `ffi/f32_helpers.c:1738,1743,2241`) |
| `LeanMlir/MlirCodegen.lean` | UNet, DDPM, DIoU, FPN/RetinaNet prior, YOLOv1, SE, LN, GELU | name-only (RetinaNet prior credits Lin et al. in `SpecHelpers.lean:94`) |
| `ffi/f32_helpers.c` | PUCT/MCTS; DQN replay | name-only |
| `demos/MainAlphaZeroTtt.lean` | AlphaZero, PUCT | Silver 2017 full; alpha-zero-general none (X-att-2) |
| `demos/MainPongDqn.lean` / `MainBlackjackDqn.lean` | DQN, Double DQN | Mnih full / none; van Hasselt none |
| `demos/MainNqsIsing.lean` | NQS | none |
| `demos/MainDiffusion2d.lean`, `MainMnistDdpm*` | DDPM, flow matching, Boltzmann generator | name-only |
| `demos/MainUnetBrats*`, `MainBratsPredict` | UNet | name-only |
| `demos/MainYolov1*`, `probes/MainDiouLossProbe` | YOLOv1, FPN, DIoU | name-only |
| `demos/MainTinyGpt*`, `MainTinyStories` | GPT, TinyStories | none |
| `apps/imagenette/Main{ConvNeXt,EfficientNet,ViT}Verified` | architecture | full |
| `apps/*/*Smooth.lean` (4) | randomized smoothing | full (Cohen 2019) |
| other `apps/imagenette/*` (16) | architecture + AdamW / RSB / DeiT / RMSProp | name-only (4 mention timm) |
| `apps/ablation/MainAblation.lean` | Mixup, CutMix, RandAug, ConvNeXt, DeiT | none |
| `jax/MainResnet50Imagenet.lean` | RSB-A2, LAMB | RSB full; LAMB name-only |
| other `jax/Main*.lean` (28), `jax/Jax/Codegen.lean` | architectures; every recipe knob | name-only / none |

**Bestiary (45 files):** 33 fully credited (authors + year + title or arXiv); 6 author-year only
(AlphaZero, Evoformer, Mamba, ShuffleNet, SwinT, UNet); 6 partial (DenseNet, Highway, VGG, WRN
lack titles; Inception v1/v4 lack titles; LLaVA-1.5 uncredited). Named external code: timm RSB
(ResNet), nanoGPT (GPT, Karpathy by name), Ultralytics (YOLO), TF model garden (DeepLab) — names
only, no URLs.

**`LeanMlir/Proofs/` (266 files) — counts only** (files with ≥ 3 hits for the method / of those,
files whose docstring credits the source): ResNet 63/2, BatchNorm 50/0, ConvNeXt 47/0,
EfficientNet 42/0, MobileNetV2 41/0, ViT 35/0, GELU 31/0, LayerNorm 26/0, sync-BN 21/0,
MobileNetV4 17/0, AdamW 17/0, label smoothing 16/0, IBP 14/0 (no Gowal 2018, incl.
`Certificates/IntervalBound.lean`), drop-path 12/0, FP8 9/0, LAMB 6/1, RMSProp 6/0, Adam 5/0,
Higham 4/3 (named in 10 files, no book/year), Nesterov 4/0, PGD 4/0, RSB 4/0, CROWN 3/1, Mixup 3/0,
DeiT 2/0, SE 2/0. 34 of 266 files cite any paper in any comment. The certificate area is the
well-credited one (Cohen–Rosenfeld–Kolter, Tsuzuku–Sato–Sugiyama, Fazlyab–Robey–Hassani); Chan's
variance is name-only in ~10 files (`Foundation/DataParallel/Sync.lean`,
`Architectures/BatchNorm.lean:135`, the `*SyncB` nets); Clopper–Pearson name-only in
`Certificates/Smoothing/CP.lean`. Per-file findings belong to the proof-tree slices.

**External code / data.** timm: pinned, cited as RSB paper (content.tex:13667) and 44 name mentions,
the software never. JAX, IREE, XLA/PJRT/StableHLO, Mathlib, doc-gen4, leanblueprint: names only, no
URL/citation (X-att-1). alpha-zero-general: name only (X-att-2). torchvision / `tf_efficientnet_b0`:
book :13646, :16970, :9365 by name. Datasets: the book's appendix tables (content.tex:16876–16895,
17094–17139) give reference + homepage for MNIST, CIFAR, Imagenette, ImageNet, VisDrone, NEU-DET,
ArASL, PlantVillage, PlantDoc, EuroSAT, Sentinel-2, MapBiomas, MSD/BraTS, tiny Shakespeare, GWOSC —
TinyStories missing (X-att-5). The book has no `\bibitem`/`.bib`; citations are inline (87 arXiv
hrefs), which is a consistent choice, not a finding.

## Checked, not findings

- `LeanMlir.lean` umbrella: no module is reachable only through it or through `Reference`; its "87 of
  the headline theorems" matches `tests/comparator/config*.json` (13 + 35 + 39).
- `LeanMlir/Ddpm.lean`: `betaC` = −d/dt log ᾱ = π tan θ /(1+s), `tOfAbar` inverts `abarC`, `ddimCoefs`
  matches Song–Meng–Ermon eq. 12/16 (a, b, σ) — all checked by hand.
- `LeanMlir/Blackjack.lean`: 4/13 ten draw, `sab=True` settlement, dealer stands on soft 17, the
  DP order (hard 21..11, soft 21..12, hard 10..4) is a valid topological order of `hitValue`'s
  dependencies, `stickValue`'s dealer-natural row — correct against Gymnasium Blackjack-v1.
- `LeanMlir/TicTacToe.lean` (HEAD): `winsThrough`, `result`, `Table.optimal`, `heuristicPlayer`
  correct; `Pos.resultForMover` has zero callers (`dead.py` over the repo) — low, left to the in-flight
  session.
- `LeanMlir/Verified/Smoothing.lean`: selection (n0) and estimation seeds are disjoint per image;
  `pA > 0.5` abstain; one-sided CP at α = 0.001; Lanczos g = 7 and Lentz `betacf` match NR; the
  midpoint-not-conservative caveat is already stated in the docstring.
- `LeanMlir/Verified/NetsCore.lean` stated counts, evaluated 09-30 (`lake env lean` scratch file):
  R34 110 tensors; R50-ImageNet 161 / 25,557,032; MNv2-ImageNet 158 / 3,504,872 (ParamLayouts'
  claim); EfficientNet 213 / 4,020,358; ConvNeXt-T 182; -S 50.2M; -B 88.6M; ViT 200 — all match.
- `LeanMlir/ParamLayouts.lean`: hand lists are the deliberate second route behind
  `#guard toSpecs == XLayout.specs`; not duplication.
- `LeanMlir/SpecHelpers.lean`: prior-bias splices are guarded; `applyHeadPriorBias` assumes a bias-last
  head, true of its one caller (`unetBrats`'s `.conv2d`).
- `LeanMlir/Types.lean` / `Train.lean` post-09-26 `lossKindFor`: one resolution, pinned by the
  `#guard` at Train.lean:476; `.bce`, `.rmsprop`, `.lamb` refused with a reason on the MLIR path.
- Every other proof name cited in MlirCodegen's comments resolves (script over the file).
- `LeanMlir/Spec.lean` `has*` queries: all consumed by `jax/Jax/Codegen.lean`.
- `IreeRuntime` → a lowerer-neutral module name was considered: 3 importers but 59 doc mentions; not
  worth it (the docstring already says "PJRT/XLA by default").
- `LeanMlir/Verified/Train.lean` `VerifiedVariant.*` predicates: `TestVariantPredicates` passes.
- Bestiary: all 45 files have a module docstring and a README row (1:1); every backticked path
  resolves; ResNet-18/50/101/152 and VGG-11/13/16/19 counts equal torchvision's exactly; Swin,
  MobileViT, ShuffleNetV2, DETR, BERT/RoBERTa, GPT-2 M/L/XL, CLIP, SAM, DeepLab, CycleGAN, DCGAN,
  PINN and FNO's 99.5 % spectral share are consistent with their docstrings; LLaVA's and SD's
  undercounts are disclosed.
- apps/: thin configs; best pair 66 % of 27 lines; no Main imports another Main.

## Gaps for the humans

- `tests/TestVariantPredicates.lean`, `TestDropPathRamp.lean`, `TestBatchedEmitTie.lean`,
  `TestR50Contract.lean`, `TestXlaPadOps.lean` are pure `#guard`/render gates that the lakefile or a
  gate script calls authoritative, and no CI job runs them. They pass today; a `lake env lean` loop
  over them in `proofs.yml` (they need no GPU) would keep it so.
- Hand-maintained counts drift (X-doc-1: four wrong in one file, one bumped in the wrong direction the
  day before this audit). A `check_counts.py` (lib roots, exes, Apps modules) in the name-lint step
  would catch it.
- `MlirCodegen.unsupported` vs the header's constructor list (X-cor-1): a `#guard` that every
  constructor with `paramSlots = none` and no `emitForwardBody` arm is refused would pin it.
- **In-flight watch** (not at HEAD, seen in the main checkout's diff of `LeanMlir/TicTacToe.lean`):
  `solverEntry` is a *pure* `@[extern] opaque … (arena : @& ByteArray) … : UInt8` whose C side fills a
  transposition cache inside the borrowed arena ("the cache the call fills is invisible"). Mutating a
  borrowed, possibly shared `ByteArray` from a pure function is outside Lean's FFI contract: the
  compiler may share the array or CSE/hoist the call. The value is a function of the position, so
  results stay right, but the arena can be mutated while another reference reads it. Worth a look
  before that commit lands (make it `IO`, or document why sharing is impossible).

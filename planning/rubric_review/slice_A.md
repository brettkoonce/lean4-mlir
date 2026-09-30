# Slice A — Architectures + Training: rubric review 2026-09-30

Scope: `LeanMlir/Proofs/Architectures/**` (20 files) and `LeanMlir/Proofs/Training/**` except
`SgdDescent/MlpBias.lean`. Generated and skipped: `Training/Trained/CnnSeal.lean`
(`scripts/certs/trained_cnn_seal.py`), `Training/Trained/CnnWitness.lean`
(`scripts/certs/trained_cnn_witness.py`) and `Training/Trained/LinearDescent.lean`
(`scripts/certs/trained_linear_descent.py`). `Training/Trained/MlpWitness.lean` is hand-written
and was audited. Every hand-written file was compiled from the main checkout
(`lake env lean <audit-tree file>`, HEAD 55ad3a5a in both trees): all clean, no linter warnings;
`SgdDescent/Cnn` 19.5 s, everything else under 6 s.

Since 373059db the slice has had: the MaxPool predicate change (37f8ab12, 244c370a), api-design
§2/§6/§7/§9 (simp/ext API, `rmsSqNext` deletion, 939 privatisations), the comment sweep (f48a8533)
and `rowLNBack_affine_eq` / `geluFlat_eq_backward` for the ViT cotangent work (4fcf613f).

## Verdicts
| angle | verdict | findings |
|---|---|---|
| correctness | block | 1 |
| reuse | request_changes | 1 |
| scope | request_changes | 1 |
| attribution | request_changes | 2 |
| api-design | approve | 0 |
| generality | request_changes | 2 |
| placement | request_changes | 3 (2 carried) |
| naming | approve | 0 |
| documentation | request_changes | 2 |
| proof-quality | request_changes | 3 |

## Findings

### correctness

- **A-corr-1** `LeanMlir/Proofs/Architectures/ConvIndex.lean:215` `MaxPool2MarginQ`,
  `Architectures/CNN.lean:893` `MaxPool2Smooth`, as used by `Training/SgdDescent/Cnn.lean`
  (`cnn_conv2_sgd_descends` :2333, `cnn_conv1_sgd_descends` :4313, `cnn_conv2_bias_sgd_descends`
  :4941, `cnn_conv1_bias_sgd_descends` :5413, the four `*_float_sgd_descends`) and
  `SgdDescent/Cifar.lean:103` `cifar8_lastConv_sgd_descends`. **The 09-27 fix (37f8ab12) does not
  make the pool hypothesis hold on MNIST: it still fails on 99.5% of MNIST test images, whatever the
  weights, so the CNN descent rungs never apply to the Chapter-3 net's data.** The fix allows ties
  below a window's max. A window whose four cells are all *equal* still fails `MaxPool2Smooth`, and
  still fails `MaxPool2MarginQ δ` for every δ ≥ 0. In the shipped `cnnVerified` (28×28, conv 1→32 3×3
  SAME, conv 32→32 3×3 SAME, pool 2×2; `SpecVJP.lean:107`, `Verified/NetsCore.lean:129`), each 2×2
  pool window at pooled position (i, j) reads a 6×6 input patch. When that patch is constant and
  clear of the padding, all four conv2 pre-activations are equal in every channel for every
  `W₁ b₁ W₂ b₂`. So all four post-ReLU cells tie at the max, and `hmq` is false. **Evidence:**
  `data/t10k-images-idx3-ubyte` (inputs normalised to [0,1], `F32Array.lean:240`, so the background
  is exactly 0). 9,950 of 10,000 test images have at least one such window, 20.1 of 196 windows per
  image on average. The count is conservative: windows touching padding are excluded (script in the
  scratch dir `A/`). A second, weight-dependent failure: any window whose four pre-activations are
  all < 0 is an all-zero post-ReLU window. `MaxPool2Smooth` rejects it, but the capstones' own relu₂
  margin `hm2` already keeps those cells dead along the step. So `maxPoolFlat ∘ relu` is constant 0
  there and smooth, and the hypothesis is stronger than the theorem needs. The same shape affects
  `h_mp` in `CNN.lean:1604 cnnHasVJPAt`, `MaxPool3s2Smooth` (`MaxPool3s2.lean`) and the
  R34/R50 smooth-point bundles that apply it post-ReLU. The book (content.tex:17990–17996, "the one
  irreducible boundary") and `formalization.yaml:287–289` call the all-zero window irreducible. That
  is false for the composite these theorems state. **Fix:**
  (a) For the whole-net VJPs, add a post-ReLU predicate beside `MaxPool2Smooth`, e.g.
  `MaxPool2SmoothAfterRelu z`: each window either has a unique argmax of `relu z`, or has all four
  `z < 0`. Prove `(maxPoolFlat ∘ relu)` differentiable under it plus the existing `z ≠ 0`, and restate
  `cnnHasVJPAt` / the MaxPool3s2 twins on it. Keep `MaxPool2Smooth` for the bare op, which is
  cited by the comparator (2 files) and the book (2 sites).
  (b) For the descent rungs, weaken `hmq` to: each window has a 2δ max-margin, or all cells dead with
  margin, or every cell within 2δ of the max is equal to it *as a function of the moving parameter*
  on the step segment. Discharge the last case from patch equality (`convPadWin` equal ⇒ equal
  pre-activations for every kernel/bias). Then pool∘relu∘conv is a fixed reindex on the segment
  even though the pool alone is not differentiable there. Fix the book/yaml "irreducible" sentences
  when (a) lands. **Gate:** a lemma that the new predicate holds on constant-patch windows. Then a
  concrete MNIST instance through a generator, as `TrainedLinearDescent` does for the linear rung.
  No CNN descent rung has a satisfiable instance today. **Cost:** `MaxPool2MarginQ` in 3 Lean
  files + 4 AuditAxioms lines; the eight CNN rungs + Cifar are AuditAxioms-only (blueprint cites
  `cnn_conv2_sgd_descends`, `cnn_conv2_float_sgd_descends`, `cifar8_lastConv_sgd_descends` by name,
  statements free to change). (a) +~150 lines, (b) +~300. Risk: the conv1 rung's patch equality is a
  two-layer receptive field. **Size:** (a) M, (b) L.

### reuse

- **A-reuse-1** `.correct`-field restatements with no consumer, the class api-design §2.7 deleted
  (it removed `flatConvStride2XlaHasVJP_correct`, `dropPathHasVJP_correct`, `dropoutHasVJP_correct`,
  `trainedMlpHasVJP_correct` and said "the other ~40 are consumed by `tests/comparator/`". These
  six are not):
  `Architectures/StridedConv.lean:107 flatConvStride2HasVJP_correct`,
  `Architectures/Depthwise.lean:299 depthwiseStride2FlatHasVJP_correct`,
  `Architectures/LayerNorm.lean:346 swishHasVJP_correct`, `:382 layerScaleHasVJP_correct`,
  `Architectures/SE.lean:213 sigmoidHasVJP_correct`,
  `Architectures/ConvGrad.lean depthwise_bias_grad_bridge` (`= (depthwiseBiasGradHasVJP …).correct`).
  **Fix:** delete them; consumers use `(X).correct`. **Evidence:** a word grep over `LeanMlir apps
  demos tests blueprint/src formalization.yaml scripts` finds only the definition and one
  `tests/AuditAxioms.lean` line each. **Cost:** −6 AuditAxioms lines; the witnesses stay audited;
  ~−45 lines. **Size:** S.

### scope

- **A-scope-1** `Training/SgdDescent/Cnn.lean:85–230, :612–765`, the MNIST-CNN float budget cluster:
  `FloatModel.mnistCnnNoBnForwardF`, `FloatModel.cnn_float_close` (127-line proof),
  `FloatModel.cnn_convW_step_float_close`, `FloatModel.cnn_convb_step_float_close`,
  `FloatModel.mnist_cnn_convW_step_float_budget`, `FloatModel.mnist_cnn_convb_step_float_budget`.
  **Nothing reaches these from the roadmap.** No Lean consumer outside the cluster; absent from the
  book, `formalization.yaml`, the comparator, README/TRUST. The descent rungs use `convF_close`
  directly. Their siblings (`linear_float_close`, `cifar_float_close`) went in the 09-08 float chop;
  these survived it. **Fix:** delete the cluster, or give it a roadmap consumer. The comment ("the
  binary32 forward-error bound for the Chapter-3 CNN") only asserts one. **Evidence:** a grep for
  each name returns only the cluster and `tests/AuditAxioms.lean:803,1027,1029` (+
  `cnn_convW/b_step_float_close` used only by the two `mnist_*_budget`). **Cost:** −4 AuditAxioms
  lines, ~−330 lines. Also shrinks the "real + float" second job of this file (A-place-1). **Size:** S.

### attribution

- **A-attr-1** Optimizer and regulariser specs name no source paper. `Training/Optim/AdamStep.lean`
  (Adam: Kingma & Ba 2015; the decoupled weight decay is AdamW: Loshchilov & Hutter 2019. The file
  cites only Reddi et al.). `Optim/RmsPropStep.lean` (Tieleman & Hinton 2012 lecture 6e; the
  ε-inside-√ variant is TensorFlow's, which the file does say). `Optim/SgdMomentumStep.lean` (Polyak
  1964 heavy ball; Nesterov 1983 / Sutskever et al. 2013 for the Nesterov form it implements).
  `Optim/GradClip.lean` (global-norm clipping: Pascanu, Mikolov & Bengio 2013). `DropPath.lean`
  (stochastic depth: Huang et al. 2016; "drop-path": Larsson et al. 2017; dropout: Srivastava et
  al. 2014; the linear depth schedule `keepProb` is Huang et al.'s rule, as timm implements it).
  `SgdDescent/Basic.lean` (the L-smooth descent lemma: Nesterov 2004, *Introductory Lectures*,
  Lemma 1.2.3). Only `Lamb.lean` (You et al. 2019) credits its paper. **Fix:** one reference line in
  each module docstring. **Evidence:** a grep for author names over `Training/` hits only
  `Lamb.lean:7` and `AdamStep.lean:10` (Reddi). **Cost:** docstrings only, 0 pins. **Size:** S.
- **A-attr-2** Architecture ops name no source paper. `Architectures/Residual.lean` (He et al.
  2016), `BatchNorm.lean` (Ioffe & Szegedy 2015; the consolidated three-term backward it derives is
  theirs; `bnVar_shard_chan` :135 says "Chan's parallel variance" without the citation Chan, Golub
  & LeVeque 1979/1983), `LayerNorm.lean` (Ba, Kiros & Hinton 2016; GELU + its tanh approximation:
  Hendrycks & Gimpel 2016; Swish: Ramachandran, Zoph & Le 2017; `layerScale`: Touvron et al. 2021,
  CaiT), `Attention.lean` (Vaswani et al. 2017; patch embedding / CLS token: Dosovitskiy et al.
  2021), `SE.lean` (Hu, Shen & Sun 2018), `Depthwise.lean` (names "Xception, MobileNet": Chollet
  2017, Howard et al. 2017), `ChannelLN.lean` (ConvNeXt: Liu et al. 2022). `MaxPool3s2.lean`
  already cites He et al. **Fix:** add the reference to each module docstring. **Evidence:** the
  same grep over `Architectures/` finds only `MaxPool3s2.lean` and `StridedConv.lean` ("He et al").
  **Cost:** docstrings; `BatchNorm`/`LayerNorm` are high-rdeps files, so batch with the next root
  edit. **Size:** S.

### generality

- **A-gen-1** Unused hypotheses, underscore-silenced, propagated to every caller.
  `Architectures/CNN.lean:1042 maxPool2_flat_hasFDerivAt` and
  `Architectures/MaxPool3s2.lean:299 maxPool3s2_flat_hasFDerivAt` take `(_hc : 0 < c) (_hh : 0 < h)
  (_hw : 0 < w)`, and both docstrings say "The positivity hypotheses are not used". They are
  forwarded by `maxPoolFlat_differentiableAt` (CNN.lean:1441, 87 call sites in 5 files) and
  `maxPool3s2Flat_differentiableAt` (MaxPool3s2.lean:392, 7 sites).
  `Architectures/BatchNorm.lean:87 bnMean_shard` takes `(_hR : R ≠ 0) (_hm : m ≠ 0)` (12 callers).
  The identity holds at R = 0 or m = 0: both sides are 0.
  `Training/SgdDescent/Cnn.lean:988 mask_scalar_close` takes `(_hex : 0 ≤ ex)` (8 callers).
  **Fix:** drop the binders and every argument that feeds them; then recompile and drop any caller
  binder the unused-variables linter newly reports. **Evidence:** the underscore names; the whole
  slice compiles warning-free today only because of them. **Cost:** AuditAxioms lines are name-only
  (unchanged); ~110 call-site edits; statement change for `maxPool3s2Flat_differentiableAt`
  (1 AA pin) and the R34/R50 seal callers. **Size:** S–M.
- **A-gen-2** `Architectures/CNN.lean` (2×2/s2 max-pool, ~:820–1450) and
  `Architectures/MaxPool3s2.lean` (3×3/s2 clamped max-pool, 448 lines) are two full parallel
  developments of one construction. Each has a window index map and `*Smooth`, `*_of_pairwise`, an
  argmax, `*_eq_at_max`, a local reindex, `*_flat_hasFDerivAt`, `pdiv3_*_smooth`, `HasVJPAt3`,
  `_close` and a flat wrapper. MaxPool3s2's docstrings say "Mirrors `maxPool2_flat_hasFDerivAt`".
  **Fix:** one window-max over an index family `win : Fin h → Fin w → Fin k × Fin k → Fin H × Fin W`,
  with smoothness over positions (MaxPool3s2's form, which already covers duplicate positions). Prove
  the local-linearisation, VJP and float `_close` once, and instantiate at 2×2 and 3×3/s2. **Risk:**
  the emitted `maxPoolBack` graph ties match by `rfl` (known trap); keep the two instance names as
  `abbrev`s of the general op and check the T2 ties still close by `rfl` before landing. **Cost:**
  `MaxPool2Smooth` is cited by the comparator (2 files) and the book (2), so keep the instance names.
  ~−400 lines. **Size:** L.

### placement

- **A-place-1** `Training/SgdDescent/Cnn.lean` (6,283 lines) still does two jobs: the ℝ descent
  rungs and the FloatModel rungs. The float half is `cnnConv{1,2}{,Bias}FloatGrad`, `*GradBudget`,
  `*_grad_close`, `*_float_sgd_descends` and `cnn_conv2_cot_close`, ~2.5k lines. **Fix:** split the
  float half into `Training/SgdDescent/CnnFloat.lean`, importing `Cnn`. (carried: audit_v2.md "Two
  jobs per file", still open.) **Cost:** module path only (names unchanged); AuditAxioms + blueprint
  cite names, not paths; add the module to the `Proofs`/`Certs` roots. **Size:** S.
- **A-place-2** General conv/float lemmas are stranded in the descent file. From
  `Training/SgdDescent/Cnn.lean`, these belong in `Architectures/ConvGrad` or `ConvIndex`:
  `conv2d_kernel_sub` :237, `conv2d_kernel_drift{,_total,_sum}` :250–302, `conv2d_weight_pdiv` :539,
  `convWeightGrad_eq_dot` :598, `convBiasGrad_eq_sum` :609, `convTap` + `convTap_abs_le` /
  `abs_convTap_expand` / `convTap_out_l1` :2633–2684, `conv2d_input_pdiv3` :2724,
  `conv2d_flat_input_pdiv` :2750, `conv2d_input_entry_drift` / `conv2d_input_l1_drift` :2793/:2850,
  `conv2d_bias_sub` / `conv2d_bias_pdiv` :4632/:4679. Beside `MaxPool2MarginQ` (ConvIndex):
  `MaxPool2MarginQ.poolBack_close` :322. In `Float/`: `FloatModel.dot_perturbed_close` :1001,
  `FloatModel.sum_perturbed_close` :5516, `mask_scalar_close` :987, `abs_le_of_close` :1472. Each is
  stated over conv/pool/FloatModel only and has nothing to do with descent. **Evidence:** none
  mentions a loss or a descent hypothesis; outside consumers today are AuditAxioms only.
  **Fix:** move them, names unchanged. **Cost:** 0 renames, import edits only; ~300 lines move.
  **Size:** S.
- **A-place-3** Activations have three homes, and `LayerNorm.lean` does five jobs. GELU and Swish
  (+ derivatives, VJPs) live in `Architectures/LayerNorm.lean` beside LN, the vector LN, `layerScale`
  and the ViT LN γ/β bridges. `sigmoid` is in `SE.lean:175`, `relu`/`relu6` in `Foundation/MLP`.
  **Fix:** `Architectures/Activations.lean` for gelu/swish/sigmoid (+ `hasDerivAt_swishScalar`,
  `*ScalarDeriv_eq`); LayerNorm keeps LN, vector LN and `layerScale`. (carried: audit_v2.md §2.3
  "gelu/swish/tanh/sigmoid → one Architectures/Activations.lean". Commit 4530a0df did the other §2.3
  moves, not this one.) **Cost:** `LayerNorm` has 191 rdeps, so batch as a root edit; names
  unchanged. **Size:** S.

### documentation

- **A-doc-1** Public definitions with no docstring. `Architectures/CNN.lean:1604 cnnHasVJPAt` is
  the file's whole-net capstone witness. Also `:1437 maxPoolFlat`, `:1447 maxPoolFlatHasVJPAt`;
  `MaxPool3s2.lean:398 maxPool3s2FlatHasVJPAt`; `SE.lean:175 sigmoidScalar`, `:178 sigmoid`,
  `:181 sigmoidScalarDeriv`, `:207 sigmoidHasVJP`, `:275 seGateHasVJP`, `:316 seBlockFullHasVJP`;
  `Attention.lean:564 sdpaKChain`, `:576 sdpaKChainHasVJP`, `:1058 mhsaQkvB`; `PerChannelBN.lean:274
  reassocBackHasVJP`, `:444 bnchwFwdHasVJP`, `:448 bnchwBackHasVJP`;
  `SgdDescent/Cnn.lean:5837 FloatModel.cnnConv1BiasFloatGrad` (its three siblings have one).
  **Fix:** one-sentence docstrings. `cnnHasVJPAt`'s should list its smoothness binders; `h_mp` sits
  on the post-ReLU stem output (see A-corr-1). **Evidence:** the preceding line is blank or a `--`
  comment (checked each). **Size:** S.
- **A-doc-2** `Training/Trained/MlpWitness.lean:10` link text reads `Proofs/MlpCanonical.lean` but
  the file is `Proofs/Nets/Small/MlpCanonical.lean`, as the URL beside it says. The same string is
  emitted by `scripts/certs/trained_linear_descent.py:112` (→ `Trained/LinearDescent.lean:8`) and
  by four `lipschitz_cert_*.py` generators (Certificates slice). **Fix:** `Nets/Small/MlpCanonical.lean`
  in the hand file and the generators, then regenerate (byte-check before/after). **Size:** S.

### proof-quality

- **A-pq-1** `Training/SgdDescent/Cnn.lean` repeats one descent argument four times.
  `cnn_conv2_sgd_descends` (:2333, 142 lines), `cnn_conv2_bias_sgd_descends` (:4941, 93),
  `cnn_conv1_sgd_descends` (:4313, 144) and `cnn_conv1_bias_sgd_descends` (:5413, 103) are the same
  proof. Each sets `f` and restates `hden`/`hC0` with the fully expanded drift constant. It restates
  the margins at `Kernel4.unflatten (Kernel4.flatten W)` (`rw [Kernel4.unflatten_flatten]`, 18
  sites), then feeds `sgd_descends` the per-rung `*_keeps_offkink` wrappers and
  `*_loss_grad_lipschitz`. The wrappers are 16 thin instances of the `Conv2Slot`/`Conv1Slot` lemmas
  (`cnn_margin{2,3,4}`, `cnnb2_margin{2,3,4}`, `cnn1_margin{1..4}`, `cnnb1_margin{1..4}`). The
  file's own design (`Conv2Slot`, `Conv1Slot` "for any parameter map with per-entry drift ρ·‖e‖₁")
  stops one lemma short. **Fix:** add `Conv2Slot.sgd_descends` / `Conv1Slot.sgd_descends`, generic
  in the parameter map `Z` and `ρ`, taking the slot margins at `Z v`. Each capstone is then one
  application (kernel: ρ = a, bias: ρ = 1) and the 16 wrappers go. This subsumes audit_v2.md §8's
  owner item "SgdDescentCnn's 19 single-use margin instances (14 pins, ~420 lines)". **Cost:** 14
  AuditAxioms lines for the wrappers (drop; the capstones keep the audit transitively); capstone
  statements unchanged; ~−600 lines. **Size:** M.
- **A-pq-2** `cnn_conv1_grad_close` (:4038, 181 lines) and `cnn_conv1_bias_grad_close` (:5999, 165)
  share a ~70-line verbatim prefix: the `hz1…hz4` off-kink discharges from the float margins, the
  `Z1C/Z1CF/X2/X2F/A1/E1/e2/CP/eback` `set` block, their non-negativity and `hZ1close`. Compare
  :4104–4160 with :6060–6120. The conv2 pair already factors its common part as
  `cnn_conv2_cot_close` (:1198). **Fix:** `cnn_conv1_cot_close`, the conv1-output cotangent
  closeness at the float margins, used by both. The `hz*` discharge block (14 `abs_pos.mp
  (lt_of_le_of_lt (layerBudget_nonneg …))` sites at :1552–1556, :4108–4130, :5691–5701, :6064–6084)
  becomes one lemma `FloatModel.offkink_of_margin`. **Cost:** statements unchanged; ~−150 lines.
  **Size:** M.
- **A-pq-3** 13 uncommented `show HasVJP (decimate(Odd)Flat … ∘ f) from vjpComp …` in
  `Architectures/StridedConv.lean` (:102, 142, 183, 286, 373, 389, 406) and
  `Architectures/Depthwise.lean` (:294, 604, 621, 678, 696, 711). Each unfolds the strided op's
  definition by defeq. **Fix:** `HasVJP.decimate` / `HasVJP.decimateOdd` (`(hf : HasVJP f)
  (hd : Differentiable ℝ f) : HasVJP (decimateFlat … ∘ f)`, beside `decimateFlatHasVJP`); each
  witness becomes one application after `unfold`, as `flatConvStride4HasVJP` already does.
  **Risk:** the backward term gains one delta step. Check the `ConvBackCertifiedTie` /
  `DepthwiseBackCertifiedTie` `rfl`s. **Size:** S. (The slice's other 43 uncommented `show`s, in the
  census, are mostly index arithmetic and one-off defeq peels, below the bar individually.)

## Checked, not findings
- 09-27 MaxPool change (37f8ab12, 244c370a): `MaxPool2Smooth` ⇔ unique argmax, `MaxPool3s2Smooth`
  over positions; `_of_pairwise` discharges are correct; `MaxPool2MarginQ.dom_eq`/`smooth_of_close`
  proofs are sound for the new definition. The residual problem is A-corr-1, not the edit.
- Optimizer specs against their JAX references: `adamWParam` (decoupled wd), `lambDir`/`lambTrust`
  (`where(wn>0, where(rn>0, wn/rn, 1), 1)`), `rmsBufNext` (ε inside √, s₀ = 1), `clipFactor`
  (`min 1 (c/(√s+ε))`), `keepProb` (timm linspace). All match.
- `sgd_descends` / `descent_segment` (SgdDescent/Basic): statement correct, the C·D² vs C·D²/2
  constant is documented.
- Prior doc-audit overclaims (find_D): `vitFull` weight-tied/scalar-LN, Residual "floor",
  LayerNorm taxonomy, sync-BN E[x²] wording, `sdpa_back_*`, `adamGraph`, SgdDescent "non-vacuous" /
  "EVERY parameter" / "exactly as the rendered trainer". All fixed at HEAD.
- New since 09-26: `HasVJPMat3` `@[ext]`/`Subsingleton`/`canonical` (consistent with the Tensor
  family), `geluFlat_eq_backward`, `rowLNBack_affine_eq` (consumed by ViT/ConvNeXt ties),
  Optim `_apply`/`_fst`/`_snd` simp API, `dropout` phantom args removed. Clean.
- `sigmoidScalar` vs `Real.sigmoid`: rejected in v1 (bridged by `sigmoidScalar_eq_sigmoid`); not
  re-reported.
- `convTap` (SgdDescent/Cnn, kernel tap) vs `IBP.convTap` (IntervalBoundConv, input tap): different
  functions in different namespaces.
- Naming: descent names read conclusion-first; `cnnb2_`/`cnn1_` prefixes (audit_v2 B9) disappear
  with A-pq-1.
- Unused-variable linter: whole slice compiles warning-free (the four silenced sites are A-gen-1).

## Gaps for the humans
- No check that a smooth-point / margin hypothesis is satisfiable on shipped data. A small
  data-driven gate (count windows/cells violating `MaxPool2Smooth` / relu `≠ 0` on the test set at
  the trained weights) would have caught A-corr-1 when the predicate changed.
- No gate for "public theorem whose only mention is an AuditAxioms line" (A-reuse-1, A-scope-1).
  The census's `AA-only` list could be a CI report.
- Generator byte-reproducibility of `Trained/CnnWitness.lean` after 37f8ab12 was not re-run here
  (it writes into the repo); the commit says regenerated.

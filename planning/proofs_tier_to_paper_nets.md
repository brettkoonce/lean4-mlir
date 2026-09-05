# Bringing every net's Proofs tier to its paper-faithful net

**Scoped 2026-09-05 from the Proofs-tier audit run during the XLA-SAME re-spelling. Nothing
below is started except where a row says so; 3.1 has since landed.** The target is the level
ConvNeXt-T sits at: every certification tier stated at the net the artifact runs, the column
"tiers only at a reduced or representative net" empty for every architecture. ConvNeXt-T had one
hole of its own when this was scoped — its train-step tie was at the retired scalar LN — and
3.1 closed it, so it is now the clean reference the other six are measured against.

The standing conventions are in `planning/xla_same_respell_and_blueprint_audit.md` (padding)
and `planning/float_budget_numbers.md` (the numbers and what they certify). This document adds
the per-net work packages, in the order that pays off soonest, each with an acceptance
criterion, the file to mirror, and the traps that file already hit.

## 1. The tiers, and what "the paper net" means

Seven tiers. A row is green only when the statement is about the net with the real depth, the
real widths, and the shipped conventions: stride-2 padding phase (XLA-SAME for the TF-origin
nets, symmetric for the PyTorch-origin ones), BatchNorm mode, activation, LayerNorm spelling.

| tier | what it is | ConvNeXt-T's instance |
|---|---|---|
| T1 forward + VJP | the ℝ forward and a whole-net `HasVJP` (or `HasVJPAt` with the kink clauses) | `convNextForwardTCh`, `convNextForwardTCh_has_vjp` |
| T2 graph faithfulness | a typed `SHlo` graph with `den graph = forward`, block by block then chained | `ConvNeXtFullT.lean` |
| T3 train-step tie | every emitted param-SGD op `den = certified` (§1 fold), then each pinned to the real backward-chain cotangent (§1a tie) | `ConvNeXtFaithfulPoC.lean`, `ConvNeXtTiePoC.lean` (all 182 params, 3.1) |
| T4 forward budget | `*_float_logits_le`, fold or CAP, kernel-checked numerals from `scripts/float_budget_envelope.py` | `cnx_float_logits_le`, CAP |
| T5 backward budget | `*_grad_float_le` on the input gradient | `cnx_grad_float_le` |
| T6 certified backward tie | the float-tier hand-written backward chain IS the certified VJP, with a `rfl` shape check that the opaque-block chain is the committed forward | `convnextInputGrad_eq_convNextForwardTCh_vjp`, `convNextForwardTCh_eq_chain` |
| T7 witness | non-degeneracy: non-constant forward and a nonzero Jacobian at a point | none for ConvNeXt; optional column |

The trunk folds (`r34Trunk_3463`, `r50Trunk_3463`, `enetTrunk`, `vitTinyTrunk`,
`Foundation/BackNetFolds.lean`) bundle T1 with a *backward-graph* faithfulness (`CertLayer`:
forward, `HasVJPAt`, backward graph, `den graph = vjp.backward`). They are not T2 (no forward
graph) and not T6 (no float-tier chain), and only the ones whose block list is pinned to the
shipped config count as paper-net statements.

## 2. Where each net stands (audit of 2026-09-05)

| net | paper net in Proofs | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|---|
| ResNet-34 | `resnet34Forward_full_pc`, [3,4,6,3], 64 to 512 | ✓ | ✓ | ✓ 146 params | ✓ eval and train BN | ✓ | ✓ | full depth at 2 channels; 224 realistic |
| ConvNeXt-T | `convNextForwardTCh`, [3,3,9,3], 96 to 768 | ✓ | ✓ | ✓ 182 params | ✓ CAP | ✓ | ✓ | none |
| ViT-Tiny | `vitForwardKV` / `vitBodyKVFlat`, depth 12, D 192, 3 heads | ✓ | ✓ `vitFwdGraphKMHV_faithful` | ✓ 200 params | ✓ CAP | ✗ | ✗ block-level only | none |
| EfficientNet-B0 | `EfficientNetFullB0.lean`, 16 MBConv | ✓ | ✓ | ✓ 262 params (stem symmetric) | ✗ 3-block | ✗ 3-block, N=1 | ✗ open at 3-block | none |
| MobileNetV2 | `MobileNetV2FullPaper.lean`, 17 blocks | ✓ `mobilenetv2_full_has_vjp_at` (`MobileNetV2FullVJP.lean`), with shape check `mobilenetv2ForwardPaper_eq_chain`; the yaml headline still points at the 2-block generic | ✓ | ✓ 210 params (symmetric) | ✗ 6-block | ✗ 6-block | ✗ 6-block | 17 blocks at toy dims; 2 blocks at 224 |
| ResNet-50 | none; `r50Trunk_3463` is a backward fold | trunk only | ✗ | ✗ | ✗ | ✗ | ✗ | none |
| MobileNetV4-Conv-M | none; UIB bodies as `CertLayer` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | none |

Two axes cut across the table and are recorded separately from it:

* **Padding phase.** B0's stem and MobileNetV2's five stride-2 sites are XLA-SAME in every
  shipped artifact (MobileNetV2's SGD pair since step 0, 2026-09-05) and symmetric in every
  Proofs definition that names them, including the paper-net T2/T3 files. The re-spelling thread
  fixes this; nothing in section 3 should be built on the symmetric stem.
* **BatchNorm world.** `resnet34Forward_full_pc` and both MobileNetV2 chains are per-example BN,
  the SGD trainers' world. The accuracies the repo quotes for those nets come from the Adam
  trainers at batch BN (`scripts/convention_audit.py` reports `bn-split` for exactly these two).
  EfficientNet is batch BN on both sides (`bnBatchLA`). Section 4 prices the batched forwards;
  it is a larger gap than padding, and a decision, not a task.

## 3. Work packages, in order

Order: finish the padding thread first (packages here that touch B0 or MobileNetV2 must land on
the XLA stem), then the cheap closes on nets that are nearly there, then the two nets with
nothing. Each package names the file to copy the shape from; the repo's lesson is that a
paper-net file is an enumeration of an existing block file, not new mathematics.

### 3.0 Prerequisite: the re-spelling thread, steps 3 to 7

`planning/xla_same_respell_and_blueprint_audit.md`. **Steps 0 to 7 all landed 2026-09-05** (the
per-example XLA tokens, the odd-phase backwards with leaf ties, the float leaves and `Maps`, the
re-spelled forwards/chains/PoC files, and B0's 3-block T6 in `d3b0db6`). Only step 8, the
blueprint audit, is left, and it is a docs commit that gates nothing in Lean. Packages 3.2 and
3.3 extend those to the paper nets; 3.1 touches neither B0 nor MobileNetV2 and never depended on
this thread.

### 3.1 ConvNeXt-T: the train-step tie at channel LN (T3) — **DONE 2026-09-05**

**Gap, as scoped.** `ConvNeXtTiePoC.lean` and the LN half of `ConvNeXtFaithfulPoC.lean` were
stated at the scalar-LN render that `verified_mlir/convnext_train_step.mlir` stopped being on
2026-07-31 (§2m/§2n in `planning/convnext_close.md`). Every theorem was true and none was about
the committed bytes.

**What landed.** The one piece of new mathematics is the channel-LN γ/β parameter certs
(`cnx_render_chln{gamma,beta}_certified`, in `ConvNeXtChannelLN.lean` beside `chanLNTensor3`).
The render's γ/β tails re-emit the `[h·w, c]` transposes and then run ViT's `veclnGammaSgd` /
`rowDenseBiasSgd` on that view, so ViT's certs apply *at the row layout*; the theorem the net
needs is about `chanLNTensor3` at the `c·h·w` activation layout, contracted with the cotangent
the block backward delivers there. The bridge is that `chanLNTensor3`'s pre- and post-
conjugations are inverse permutations (`chanRowsPerm`), so the output-side one moves onto the
cotangent as its inverse — which is exactly the transposed cotangent the op is fed.
`pdiv_reindexOut_contract` states that generically; it is a permutation's adjoint, not new
analysis. `chanLNRows` moved from `Float/ChannelLNFloatBridge.lean` to sit with them.

Everything else was re-threading `ConvNeXtTiePoC` at the shipped spelling: `cnxCotD` becomes
`chanLNTensor3Back` (the certified VJP, by `chanLNTensor3Back_eq_chanLN_vjp`); the head becomes
ViT's vector-LN at `N = 1`, stated at the literal 768 because `1 * m` does not reduce at a
variable `m`; the stem LN joins the thread, with the patchify conv's own gradients now behind its
input-VJP. `cnxCotP`/`cnxCotE`/`cnxCotN` are LN-form-agnostic and were reused verbatim.

**And the scope grew, correctly.** The "four even-kernel weight grads" the old file carried as a
render gap are no longer one: the three downsample 2×2/s2 weights are `convStridedWeightSgd`
(kernel-generic, since `sWGradGeom` split the odd/even padding cases) and the stem 4×4/s4 is
`convStride4WeightGrad`. So `cnx_net_tied_certified` now covers **all 182** parameters — 181 as
`θ − lr·∂Loss/∂θ`, and `psW` at its **gradient**, because that op's SGD wrap is hand-written text
(a declared §5 carve-out, and the reason it cannot take the `den(op) = θ − lr·…` shape).

Gates: `lake build Certs` 3956 green, `lake env lean tests/AuditAxioms.lean` 3-axiom clean,
`lake exe docstring-checkrefs`, `python3 scripts/check_audit_coverage.py`. No renderer or `.mlir`
change — this is a den-level tie, as r34/mnv2 are.

### 3.2 MobileNetV2 at 17 blocks (T1, T4, T5, T6)

Everything here lands on the XLA-SAME spelling from 3.0.

**(a) T1 exists; the headline does not point at it.** `MobileNetV2FullVJP.lean` folds the whole
`[t,c,n,s]` table: `mobilenetv2_full_has_vjp_at` / `_correct` over the `MNV2PaperWeights` bundle,
pointwise (relu6 is kinked, so `_at` is the form), with the shape check
`mobilenetv2ForwardPaper_eq_chain` already audited. `formalization.yaml`'s headline
`mobilenetv2_has_vjp_at_correct` is still the 2-block generic net in `MobileNetV2.lean`. Point
the yaml row (and the blueprint's `\lean{}` tag) at `mobilenetv2_full_has_vjp_at_correct`. A
doc change; done when the row's file is `MobileNetV2FullVJP.lean`.

**(b) T4 and T5, the numbers. Probe first.** Extend `scripts/float_budget_envelope.py`'s
`mnv2_eval_chain` / `mnv2_back_chain` to the `[t,c,n,s]` table (the block list is
`MobileNetV2Render.lean`'s `paperSig`; do not hand-copy it a fourth time, read it from one
place). Expect the eval-BN forward fold to grow about 4.8 orders per BN site (20 sites gave
1.444e96 with the relu6-clamped window 2154), so 52 sites lands near 1e250 with the window
unchanged; expect the backward, 1e152 at 6 blocks, to pass 1e300 at 17. `norm_num` refuses
numerals past ~1e300, so the backward will need `FloatBridgesTo.capped` at the BN sites the way
the LayerNorm nets do, or an operating-point `S` below the ε-floor stated in the hypothesis.
Decide from the probe, then write `MobileNetV2PaperFloatBudget.lean` /
`MobileNetV2PaperBackFloatBudget.lean` beside the 6-block files, not replacing them (the 6-block
numbers stay as the reduced net's; the yaml gets new rows). Mirror: `MobileNetV2FloatBudget.lean`
(the `Env` structure + bottom-up `have` chain; the `verify_*` pass; the inhabitation section).
Done when `verify_mnv2_paper` re-asserts every rounded row and both files compile.

**(c) T6, the certified backward tie.** `MobileNetV2WholeBackCertifiedTie.lean` does the 6-block
net with the blocks opaque; the 17-block tie is the same assembly over 17 opaque block
backwards. Use twelve-plus top-level `def`s for the chain (the `let`-chain and the whole-net
`rfl` both failed on ConvNeXt at depth 12, the top-level defs took 2.4 s), state the
single-level reductions applied, and write the shape check
`mobilenetv2ForwardPaper_eq_chain` as a `rfl` first. Done when the tie compiles, the shape
check is used by it, and the numeral in (b)'s backward file is unchanged or the change is
explained.

**(d) T7, optional.** `MobileNetV2JacobianSealFull.lean` seals 17 blocks at 2-channel toy dims
and `MobileNetV2SealRealistic.lean` seals 2 blocks at 224. The two mechanisms compose (zeroed
skip blocks are the affine shift `a ↦ a + 3`; γ-scaling keeps BN inside `(0,6)` at `n = 2·112²`);
a 17-block 224 seal is their product. Skip unless a witness at the paper net is wanted for the
blueprint.

### 3.3 EfficientNet-B0 at 16 blocks (T4, T5, T6)

3.0 step 7 (the 3-block T6) landed in `d3b0db6`, so 3.3(c) generalises it rather than waiting on it. On the XLA stem.

**(a) T4, and it will not be a fold. Probe first.** Each squeeze-excite site roughly doubles
the budget's exponent (1e25, 1e83, 1e199 across the representative's three blocks:
`planning/archive/float_budget_numbers_log.md` §3.4). Sixteen SE sites cannot fold. Extend
`b0_eval_chain` to the `[t,c,n,s,k]` table and confirm; then state the number the way
ConvNeXt-T and ViT-Tiny state theirs, `FloatBridgesTo.capped` at every SE gate (the gate's own
range `≤ 1` is what the cap uses), and label it CAP in the yaml and the blueprint table. The
window is the honest half; report it. Mirror: `ConvNeXtFloatBudget.lean` for the capped fold,
`EfficientNetFloatBudget.lean` for the record shapes (`EnetWeights`, `EnetMBBlk.maps`). Done when
the file compiles, `verify_b0_full` passes, and the yaml row says CAP.

**(b) T5.** `b0_back_chain` at 16 blocks; the SE backward is linear at a fixed point so the
backward folds (`backward_is_always_a_fold`), but 16 SE sites of `A · Eg` will also pass 1e300.
Same decision as (a); state at `N = 1` as the 3-block number is, and say so.

**(c) T6.** The 3-block tie from 3.0 step 7 generalizes: the three batched block ties at
`bnBatchLA` are already dim-polymorphic; the assembly is 16 opaque blocks. Same elaboration
discipline as 3.2(c). Done when it compiles against `efficientnetForwardB_full_has_vjp` with a
used shape check.

**(d) `enetTrunk` at 16 blocks.** Optional; the `CertLayer` fold is a type-level check that the
block ladder is the shipped one, and the current one is the 3-block ladder.

### 3.4 ViT-Tiny: the backward tiers (T5, T6)

**Gap.** Forward is complete at depth 12, three heads, vector LN. The backward has the float
chain (`vitGradFlat`, `MhsaBackFloatBridge.lean`), the block-level tie
(`vitBlockBackPR_eq_transformerBlock_vjp`, `ViTMhsaBackCertifiedTie.lean`) and the end-to-end
backward-graph fold (`vitTinyTrunk_is_shipped`), but no whole-net T6 and no number.

**(a) T6.** `vitGradFlat_eq_vitBodyKVFlat_vjp`: induction on depth over the block tie, the
`towerBack` fold reconciled with `vitBodyKVFlat`'s head recursion the way
`vit_full_eq_vitForwardFlat` (`ViTWholeFloatBridge.lean`) reconciles the forwards. The block tie
is at heads = 1 per token (`ViTMhsaBackCertifiedTie` says so in its header); the multi-head
cotangents are in `ViTMultiHeadChain.lean`, so the tie either goes through `mhsa_layer_spelled`
at 3 heads or is stated at the single-head representative and says so. Prefer the former; the
forward already is.

**(b) T5.** Write `vit_back_chain` in the probe (there is none; the four backward chains are
r34, mnv2, b0, cnx). All 25 LN sites and 12 attention sites are capped on the forward; the
backward through a softmax Jacobian is `smRho`-conditioned (`vit_float_logits_le` already
needs `smRho u eexp 197 < 1`), so expect a CAP with that side-condition. Then
`ViTBackFloatBudget.lean`, mirror `ConvNeXtBackFloatBudget.lean` (the first LayerNorm-net
backward, and its `|istd| ≤ 16` operating point).

Done when both compile, the yaml gets a `vit_grad_float_le` row, and the tie uses a shape check.

### 3.5 ResNet-50: all six tiers

**What exists.** The three bottleneck forms with VJPs and backward graphs
(`Resnet50BlocksCertified.lean` per example, `ResNet50BackB0.lean` batched), the net-level
backward fold `r50Trunk_3463` (`ResNet50BackNet.lean`), the batched render `ResNet50RenderB.lean`
(`r50FwdChainB`, batch BN, the token order to mirror), and `resnet50_fwd.mlir` as a byte-prefix of
the Adam train step (the pairing R34 lacks). No ℝ-forward of the whole net, no forward graph,
no PoC, no numbers, no tie.

**Order.** (a) T1: `resnet50ForwardB` at the batched index with `bnBatchLA` (the artifact is
batch BN, so state the proof net there; this is the one net where T1 can match the trained
world from the start), bottleneck [3,4,6,3] at widths 64/256, 128/512, 256/1024, 512/2048, the
stride-1 projection at stage 1 block 0 (`bblkPProjPC`'s batched peer; the block file's header
explains why `bblkPC` there is the dangerous mistake). VJP by `vjp_comp_at` with the relu clauses
in a record. Mirror: `EfficientNetFullB0.lean` (batched, record of hypotheses), with
`ResNet34RenderPC.lean` part 2 for the stage enumeration. (b) T2: the typed graph, per-block
`_faithful` then chained; the tokens are `r50FwdChainB`'s. (c) T3: `ResNet50FaithfulPoC` /
`ResNet50TiePoC` from the R34 pair; the bottleneck adds one conv per block and nothing new in
kind. (d) T4/T5: probe `r50_eval_chain` (53 BN sites at eval BN, plain relu, so a fold near
1e250 like r34's 1.548e209 plus 17 sites; the training-BN form is a CAP as r34's is) and
`r50_back_chain`; then the two budget files, mirror `Resnet34FloatBudget.lean` /
`Resnet34BackFloatBudget.lean`. (e) T6: `r50InputGrad` and its tie, mirror
`Resnet34BackCertifiedTie.lean`; the bottleneck's three-conv body needs one new block-level tie
(`r50BottleneckBack_eq_…_vjp`) and the projection forms two more.

Done when the seven yaml rows exist and `scripts/convention_audit.py` still reports r50 clean.
This is the largest package after MNv4; budget it as the R34 close was, one tier per session.

### 3.6 MobileNetV4-Conv-M: all six tiers

**What exists.** `mnv4Blocks` (`MobileNetV4RenderB.lean`), the 21-row block table verified
against timm; the UIB body as one `CertLayer` with the four families as `CertLayer.id'`
substitutions (`MobileNetV4BackB0.lean`); the batched render; forward tie at 1.4e-6 and gradient
tie at 0/147 against JAX. Nothing at the net level in Lean.

**Order.** (a) T1: `mnv4ForwardB` from the table: stem 3x3/s2 at the XLA phase (the one
XLA-SAME site; use `flatConvStride2Xla` from this thread, its VJP and float leaves exist), the
fused-IB stage (3x3/s2 conv + 1x1), 21 UIB blocks with the `id'` slots, the two head convs, GAP,
classifier. Batch BN (`bnBatchLA`), relu. The block table must be READ from one place; the
render already folds over `mnv4Blocks`, so the Proofs forward should take the same list and a
`Fin 21 → UibParams` record (the ConvNeXt `Fin k → CnxBlockParamsCh` shape). (b) T2: graph +
faithfulness per family, then chained over the table. (c) T3: PoC pair; the UIB param ops are
the depthwise/1x1/BN ops the other nets already certify, at the four family wirings. (d) T4/T5:
probe `mnv4_eval_chain` (batch BN at 21 blocks with two depthwises each; expect a fold, relu
has no clamp so the window will be r34-sized). (e) T6: `mnv4InputGrad` and the tie, mirror
`EfficientNetWholeBackFloatBridge.lean` for the depthwise-heavy chain.

Done when the yaml rows exist and `mnv4_fwd.mlir`'s pad profile (1 XLA site) is the one the
forward definition names. Largest package; the family collapse is what keeps it to one block
file rather than four.

### 3.7 ResNet-34: nothing structural

T1 to T6 are at the paper net. The witness is 2 channels wide at the full depth and at 224
resolution; a full-width witness buys nothing the blueprint needs. What remains for R34 is the
BN-world axis (section 4).

## 4. The BatchNorm-world axis, priced and left as a decision

The R34 and MobileNetV2 Proofs tiers are per-example BN (`bnPerChannelTensor3`); the Adam
artifacts that produced the quoted numbers are batch BN; EfficientNet is batch BN on both sides
because its Proofs tier was built at the batched index from the start (`StableHLO.bnBatchLA`,
`batchMap N`). Making R34's and MobileNetV2's Proofs tiers "the net that trained" means the
batched form of every tier: the forward at `N·(c·h·w)` with `bnBatchLA`, the graph at the batched
tokens (`MobileNetV2RenderB` / `ResNet34RenderB` already emit them), the budgets at batch BN
(the training-mode number is a CAP, as `r34_train_float_logits_le` already is; the eval-mode
number is the same either way since frozen statistics reduce nothing), and the T6 tie at
`bnBatchTensor4`, for which no batched BatchNorm backward leaf exists (the reason B0's T6 is
stated at `N = 1`). That leaf is the real cost: the rest is the EfficientNet recipe applied
twice. Cost is comparable to package 3.5. Decide whether the SGD-trainer world is the one the
Proofs tier should describe (it is a real trainer with its own artifacts and prefix audits) or
whether the batched forms are wanted for the two nets whose numbers are quoted from Adam;
either answer should be written into `formalization.yaml` 4d, which today says neither.

## 5. Traps, all previously paid for

* **Probe before Lean.** Every number comes from `scripts/float_budget_envelope.py` first; the
  Lean re-asserts rounded rows. A fold that passes 1e300 is a CAP, decided at the probe, not
  discovered at `norm_num`.
* **Loose leaf bounds block folds.** A whole-net fold that will not state is usually one leaf
  discarding a bound proved one lemma down (relu6's clamp, swish's modulus, seScale's window).
  Ablate in the probe before blaming the depth.
* **Quadratic-in-the-window ops.** Training BN, LN, and SE each square the window; ten or more
  such sites end a fold. Cap at those sites.
* **Elaboration.** `Env` as a structure and a bottom-up `have` chain, or the unifier times out.
  A whole-net `rfl` against a tactic-built `HasVJP` apex does not terminate; a term-mode peer
  of top-level `def`s (not a `let` chain, 200×) plus `HasVJP.backward_unique` is the escape.
  State single-level reductions applied, after `funext`. Transport a witness along an equation
  by rebuilding, never with `▸` (an `Eq.mpr` blocks `.backward` and `.mag` from reducing). A
  computed dimension in an applied position sends the unifier into the net; give the stage a
  `def` with the type ascribed in the chain's spelling. `Elab.async` makes timings
  order-dependent; measure with it off.
* **Conventions are invisible to types.** Padding phase, BN world and activation all preserve
  shapes and arities; `scripts/convention_audit.py` sees the first and third at the artifact
  tier and nothing sees the Proofs tier. When an emitter fix lands, grep every other definition
  that claims to denote the same map (`imagenet_specs_drift_from_twins`; this thread's origin).
  When a docstring justifies a slot with a count or a convention, re-derive it.
* **Two lists for one net.** Block tables and parameter signatures written twice drift
  silently; read them from one place and `#guard` the arity (`paperSig`, `mnv4Blocks`).
* **Build the corpus.** Bare `lake build` is 2251 jobs and skips `Certs`; CI runs
  `lake build Certs` at 3956. `lake env lean tests/AuditAxioms.lean` for the axiom gate,
  `lake exe docstring-checkrefs`, `python3 scripts/check_audit_coverage.py`. A new artifact
  needs `proofs.yml`'s diff list and `check_render_coverage.py`.
* **A theorem should be about the program we run.** Before each package, write the net's
  convention table in the file header: depth, widths, padding phase per stride-2 site, BN mode
  and world, activation, LN spelling, and which artifact those are read from.

## 6. Files

New, by package: 3.1 none; 3.2 `MobileNetV2PaperFloatBudget.lean`,
`MobileNetV2PaperBackFloatBudget.lean`, `MobileNetV2PaperWholeBackCertifiedTie.lean`, a VJP in
`MobileNetV2FullPaper.lean`; 3.3 `EfficientNetFullFloatBudget.lean`,
`EfficientNetFullBackFloatBudget.lean`, `EfficientNetFullWholeBackCertifiedTie.lean`; 3.4
`ViTWholeBackCertifiedTie.lean`, `ViTBackFloatBudget.lean`; 3.5 `Resnet50FullB.lean`,
`Resnet50FaithfulPoC.lean`, `Resnet50TiePoC.lean`, `Resnet50FloatBudget.lean`,
`Resnet50BackFloatBudget.lean`, `Resnet50WholeBackCertifiedTie.lean`; 3.6 the same six for
MobileNetV4. Every new file: a `lakefile.lean` `Certs` root or an import of one, an
`AuditAxioms.lean` block, a `formalization.yaml` row with the convention comment, a
`planning/float_budget_numbers.md` section-1 row for each number.

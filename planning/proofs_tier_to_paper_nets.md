# Bringing every net's Proofs tier to its paper-faithful net

**Scoped 2026-09-05 from the Proofs-tier audit run during the XLA-SAME re-spelling. Nothing
below is started except where a row says so; 3.1, 3.2(a)–(c), (e), 3.3 and 3.4 have since
landed, and section 4's decision was taken 2026-09-06 with its shared foundation.** The target is the
level ConvNeXt-T sits at: every certification tier stated at the net the artifact runs, the column
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
| ViT-Tiny | `vitForwardKV` / `vitBodyKVFlat`, depth 12, D 192, 3 heads | ✓ | ✓ `vitFwdGraphKMHV_faithful` | ✓ 200 params | ✓ CAP | ⛔ 1.703e399, priced and declined | ✓ `vitInputGradK_eq_vitForwardKV_vjp` | none |
| EfficientNet-B0 | `EfficientNetFullB0.lean`, 16 MBConv | ✓ | ✓ train and eval BN (`EfficientNetFullB0Eval.lean`) | ✓ 262 params | ✓ CAP 2.416e287 at the 16 SE sigmoids, window 1.886e279 honest | ⛔ no number at 16 blocks (9.112e2648; statable, declined) | ✓ `efficientnetInputGradB_full_correct`, through `backward_unique` to the concrete witness | none |
| MobileNetV2 | `MobileNetV2FullPaper.lean`, 17 blocks | ✓ `mobilenetv2_full_has_vjp_at` (`MobileNetV2FullVJP.lean`), shape check `mobilenetv2ForwardPaper_eq_chain` | ✓ train and eval BN (`MobileNetV2FullPaperEval.lean`) | ✓ 210 params | ✓ CAP 8.176e16, all 52 BN sites | ⛔ no number at 17 blocks | ✓ `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp` | 17 blocks at toy dims; 2 blocks at 224 |
| ResNet-50 | none; `r50Trunk_3463` is a backward fold | trunk only | ✗ | ✗ | ✗ | ✗ | ✗ | none |
| MobileNetV4-Conv-M | none; UIB bodies as `CertLayer` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | none |

Two axes cut across the table and are recorded separately from it:

* **Padding phase.** B0's stem and MobileNetV2's five stride-2 sites are XLA-SAME in every
  shipped artifact (MobileNetV2's SGD pair since step 0, 2026-09-05) and symmetric in every
  Proofs definition that names them, including the paper-net T2/T3 files. The re-spelling thread
  fixes this; nothing in section 3 should be built on the symmetric stem.
* **BatchNorm world — DECIDED 2026-09-06, port in progress.** `resnet34Forward_full_pc` and both
  MobileNetV2 chains are per-example BN, the SGD trainer's world. The ImageNet accuracies the repo
  quotes for those nets come from the Adam and momentum trainers at batch BN
  (`scripts/convention_audit.py` reports `bn-split` for exactly these two). EfficientNet is batch
  BN on both sides (`bnBatchLA`). Section 4 takes the decision — port both — and its shared
  foundation has landed; the per-net tiers have not. This is a larger gap than padding.
  ⚠ A third, narrower split sits under T4 specifically: every whole-net forward BUDGET is stated
  at INFERENCE BN, because the training-mode modulus is quadratic in the window, and a net's
  eval-BN graph is a separate artifact from its training one. MobileNetV2 has an eval graph at
  six blocks and none at seventeen, which is why 3.2(b)'s number has no whole-net graph tie.

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

### 3.2 MobileNetV2 at 17 blocks (T1, T4, T5, T6) — **(a)–(c), (e) DONE 2026-09-05**

**(a) DONE.** `formalization.yaml`'s headline row now names `mobilenetv2_full_has_vjp_at_correct`
in `MobileNetV2FullVJP.lean` — the 17-block paper net — where it named the 2-block generic in
`MobileNetV2.lean`. The blueprint (`content.tex:6249`) already had the right theorem.

**(b) DONE; the forward has a number, the backward does not.** `scripts/float_budget_envelope.py`
gains `mnv2_paper_{plan,eval_chain,back_chain}` and `verify_mnv2_paper` (354 rounded inequalities
re-asserted). The block table is read from TWO Lean sources rather than a fourth hand-written
copy — kinds and spatial dims from `mobilenetv2ForwardPaper`, widths from `paperSig` — and the
loader asserts they name the same 17 blocks and that "has an identity skip" agrees with
`ic == oc`. It reproduces the shipped 6-block numbers exactly, which is what makes the extension
trustworthy.

| | window | budget | statable |
|---|---|---|---|
| forward, uncapped fold at the ε-floor | 2.152e4 | 2.104e266 | no |
| forward, **capped at the 52 BN sites** | 2.152e4 | **8.176e16** | yes |
| forward, uncapped at `\|istd\| <= 32` | 2.152e4 | 3.228e215 | yes |
| backward, shipped `\|istd\| <= 16` | 1.246e323 | 1.296e322 | no |
| backward, sigma^2 ~ 1 (crudest possible) | 4.901e260 | 2.199e260 | no |

Three things the scoping did not anticipate. The uncapped forward is 16 orders worse than the
"near 1e250" predicted. The window grows 2154 to 2.152e4, and that is entirely the HEAD width
(dense fan-in 1280 against the reduced net's 128) — relu6 still pins the body flat, so the
prediction that the window would not move was right about the part it was about. And **capping is
worth more than the eleven extra blocks cost**: 8.176e16 is 79 orders SMALLER than the shipped
uncapped 6-block number (1.444e96).

⛔ **The backward has no number and no cap rescues it**, because a cap's budget is `2·window` and
the window itself is past the ceiling. There is no loose leaf either — ablated per §5 before
blaming the depth: the BN γ bound 1.69 → 1 buys 12 orders and the conv kernel bound 2.72 → 1 buys
24, both MEASURED bounds rather than bounds discarded one lemma down, and only their simultaneous
fiction gets under. This is EfficientNet-B0's backward situation (1e431) and takes the same
answer.

**What landed for (b).** `MobileNetV2PaperFloatBudget.lean`: `mnv2Paper_float_logits_le`, window
`2152 * 10 ^ 1` and budget `8176 * 10 ^ 13`, over a closed `FloatBridgesTo` for the whole
seventeen-block inference net, 25 `Maps` steps at block granularity, ~90 s to elaborate. The §1
row and the backward finding are in `planning/float_budget_numbers.md`. No
`MobileNetV2PaperBackFloatBudget.lean` — there is nothing to state.

⛔ **The cap is at ALL 52 sites, not 40.** The scoping said the `min` selects the fold at 12 of
them; the Lean caps uniformly instead, and the headline is the same to four figures, because the
last cap discards everything before it. Capping uniformly avoids a per-site case split in the
block combinators (`bodyMapsC` / `stridedMapsC` / `resMapsC` / `MnvBlockNoExp.mapsC` take
`2 * Ā' ≤ Ē'` at every normalisation and no bound on the inherited error at all) and makes the
label unambiguous: at every BatchNorm the claim is the triangle inequality. The probe was moved
to match (`mnv2_paper_eval_chain` now emits `r4(2·Ā')`, rounded AFTER doubling so
`Maps.capped`'s own side condition closes; `verify_mnv2_paper` checks it against the numeral
rather than against itself, and its GAP check was corrected to `Maps.gap`'s shape).

⭐ **Why the file is writable at seventeen blocks at all**: at a capped site the output error is
`2·window` and the incoming error is discarded, so a block's stage numerals depend only on its
`(ic, mid, oc)` — `b8`, `b9` and `b10` are numerically one block, and every relu6 resets the
window to 6. A property of the cap, not of the net.

⚠ **One rung is open and the file says so.** The number is tied to the paper ladder by four
dimension-polymorphic `rfl`s (`MnvBlock.{body,strided,res}Fwd_eq_pcEval`,
`MnvBlockNoExp.fwd_eq_pcEval`: each block IS the abbreviation the committed inference forward is
built from) but NOT to a whole-net graph, because the paper net's *eval* twin has no ℝ-def and no
typed `SHlo` graph in Lean — `MobileNetV2RenderPCEval.lean` covers the six-block net only, while
`MobileNetV2FullPaper.lean` is at training BN. Closing it is the eval twin of that file's graph
section (four block-kind graphs + faithfulness + the chain, mechanical) and a `_committed`
restatement; it is T2-at-eval for the paper net and is scoped nowhere else in this document.

**(c) DONE.** `MobileNetV2PaperWholeBackCertifiedTie.lean`:
`mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp`, the 6-block tie at depth 17. No new mathematics —
the four endpoint leaf ties are reused verbatim at the paper widths (32-channel stem,
1280-channel head) and the seventeen blocks stay opaque, so the composition is checked between
variables and the file elaborates in ~3 s. What depth forced is `mnv2OpaqueA0 … A17`, one prefix
def per slot, because the 6-block statement spells each hypothesis's running activation as a
nested application — unreadable by block 5 and quadratic in the writing, the wall
`MobileNetV2FullVJP.lean` hit and answered the same way.

⛔ **The shape check cannot be a one-step `rfl`.** `mobilenetv2ForwardPaper_eq_slots` goes through
`mobilenetv2ForwardPaper_eq_chain` and then unfolds the prefixes by name; a bare `rfl` takes a
kernel deterministic timeout after three minutes, and adding `Function.comp_assoc` to the `simp`
set reproduces it. Parenthesise the head group the way `mnv2HeadW` associates and the peeled goal
closes with no associativity step at all.

**(d) T7, optional.** `MobileNetV2JacobianSealFull.lean` seals 17 blocks at 2-channel toy dims
and `MobileNetV2SealRealistic.lean` seals 2 blocks at 224. The two mechanisms compose (zeroed
skip blocks are the affine shift `a ↦ a + 3`; γ-scaling keeps BN inside `(0,6)` at `n = 2·112²`);
a 17-block 224 seal is their product. Skip unless a witness at the paper net is wanted for the
blueprint.

**(e) DONE 2026-09-05 — the whole-net eval tie, and it ends at bytes.** Written straight after
3.3(e), whose file it mirrors; ~2 s to elaborate, and the number did not move.

`Architectures/MobileNetV2FullPaperEval.lean` (318 lines): `IVWEval` / `IVWNoExpEval` /
`MNV2PaperWeightsEval nCls` (γ, β and the two frozen statistics per site, **one shared ε** as the
forward's argument — `mobilenetv2Forward_full_pc_eval`'s convention and the render's, where the
training bundle carries one per site), the four block wrappers over
`ivProjectPCEval`/`ivDepthwisePCEval`/`ivDepthwiseStridedPCEval`/`invresBodyPCEval`/
`invresBodyStridedPCEval`, `mobilenetv2ForwardPaperEval` in nested-application form, the four
block-kind graphs with `bnPerChannelEvalF`, their `_faithful` lemmas, then
`mobilenetv2FwdGraphPaperEval` + `_faithful` (one `rw` per block kind, then `rfl`).

In `MobileNetV2PaperFloatBudget.lean`: `MnvPaperWeights.toEval`,
`mnv2PaperEvalForward_eq_paperEval`, `mnv2PaperEvalGraph_faithful`,
`mnv2Paper_float_logits_le_committed`. ⭐ The scoping's first suggestion — define
`mnv2PaperEvalForward` AS `mobilenetv2ForwardPaperEval (W.toEval ε)` outright — was not needed and
not taken: the separate spelling plus the equation is what B0 had just used, and the equation
closes in ~2 s by rewriting with the four `*_eq_pcEval` lemmas and the four eval wrappers. Those
four `rfl`s said each BLOCK is the committed abbreviation; this says the whole LADDER is.

⭐ **Point 2 delivered, and it is the reason to prefer this graph to the two that exist.** The
SSA names are `bnSiteP`'s and `irSig`'s — `%stnmu`/`%stnvar`, `%b{k}enmu`/`%b{k}dnmu`/`%b{k}pnmu`
and their `nvar` peers, `%hnmu`/`%hnvar`, around `%We{k}`/`%ge{k}`/`%bte{k}`/`%Wd{k}`/`%gd{k}`/
`%btd{k}`/`%Wp{k}`/`%gp{k}`/`%btp{k}` — so the typed graph diffs against `mobilenetv2_fwd_eval`
line for line. The six-block eval graph's `%mue1`/`%vare1` matches no artifact and could not; this
net's own TRAINING graph writes `%b17gp` where the render emits `%gp17`.

⭐ **Point 4 delivered, and it closed a real qualification rather than a cosmetic one.** The head
is generic in `nCls` (`MnvPaperWeights nCls w' β' G Bb Mb`), since `Maps.dense`'s envelope depends
on the fan-in `1280` and never on the output count, so both numerals hold verbatim and one theorem
covers `mobilenetv2_fwd_eval` and its 1000-class twin `mobilenetv2in_fwd_eval`. The `|·| ≤ 28/10`
profile was measured on the 1000-class checkpoint, so before this the bound was a measurement on
52 convolutions and 52 BatchNorms and an ASSUMPTION on the `1280 × 10` head; now it is neither.

⛔ **The artifact's input count is 263, not the 315 this section scoped.** `paperSig` at
`convBias := false` is 158 tensors (the render folds each conv bias into the BatchNorm that
follows it), plus 104 statistic slots and `%x`. The 210 in the scoping is the TRAIN step's
SGD-updated parameter count, a different quantity; the graph still carries a bias slot per conv
(`%bs`, `%bd{k}`, …) that the signature has no argument for, which is the one naming difference
left and does not enter `den`.

⚠ **Three lists for one net, as the trap predicted.** `mnv2_paper_plan` now reads the eval file as
a third source and asserts `MNV2PaperWeightsEval`'s widths and `mobilenetv2ForwardPaperEval`'s
kinds and spatial sizes agree with `paperSig` and `mobilenetv2ForwardPaper`, block for block —
the same extension 3.3(e) made for B0.

### 3.3 EfficientNet-B0 at 16 blocks (T4, T5, T6) — **(a), (c), (e) DONE 2026-09-05; (b) probed, declined**

On the XLA stem, as 3.0 required. Probe first, and the probe changed the plan twice.

**(a) DONE — T4 is a CAP, and the cap is on the gate's sigmoid.** `scripts/float_budget_envelope.py`
gains `b0_full_plan` (the `[t,c,n,s,k]` table read from TWO Lean sources — widths, SE reductions
and kernels from `B0Weights`, kinds and spatial sizes from `efficientnetForwardB_full` — with the
loader asserting they name the same 16 blocks, that "has an identity skip" is `ic = oc`, and that
the no-expand form is exactly the `MBWNoExp` record), `b0_full_eval_chain`, `verify_b0_full` (476
inequalities) and `b0_full_back_chain`.

| | window | budget | statable |
|---|---|---|---|
| forward, uncapped fold at the ε-floor | 1.886e279 | 1e1897907 | no |
| forward, **capped at the sigmoid of all 16 SE gates** | 1.886e279 | **2.416e287** | yes, at `exponentiation.threshold 400` |
| forward, capped, operating point `\|istd\| ≤ 16` | 5.490e215 | 6.829e223 | yes, and not taken |
| backward, shipped leaves (`\|swish'\| ≤ 2`, `S = 317`, `Sx` = fwd window) | 9.112e2648 | 6.550e2648 | in principle; declined |
| backward, every measured bound set to 1, `S = 1`, `Sx ≤ 16` (a fiction) | 1.379e344 | 1.178e344 | in principle; declined |

Three things the scoping did not anticipate. **First, where the cap goes.** The scoping said
"at every SE gate (the gate's own range ≤ 1 is what the cap uses)", and the probe made that
precise: capping the SE *rescale* is unwritable, because `Maps.capped` needs the site's window
and the rescale's window comes from the gate's `Maps`, whose error numerals are the quadratic
`A · Eg` themselves. The cap has to sit on the **sigmoid**, the one stage in the gate path whose
window is a constant (`1 + esig`), where `Maps.capped` needs no error numeral at all and the side
condition is `2·(1 + esig) ≤ Eg`. The rescale's modulus then reads `≈ 2A + 3E`, linear in both,
and everything else — 49 BatchNorms, the rescale, every conv — stays the fold. The tell is not
`budget/window ≈ 2` but `≈ 1.3·10⁸`: sixteen gate caps compounding.

**Second, the window is 1.886e279, and that is past the "ceiling".** Swish never resets a window
(relu6 pins MobileNetV2's body flat; nothing pins B0's), so sixteen blocks cost `10¹⁶`–`10¹⁸`
each. Under §5's stated rule the next step was an operating point (`|istd| ≤ 16` lands at
5.490e215). Before paying it the ceiling was tested directly, and ⭐⭐ **it is Lean's
`exponentiation.threshold` option (default 256), not a wall**: in the exact goal shape that was
failing, `10 ^ 256` evaluates and `10 ^ 257` does not, and under `set_option
exponentiation.threshold 400` the same goals close at `10 ^ 290` in the same time. The kernel's
`Nat.pow` is GMP-backed and never had a limit. So the number is stated at the ε-floor with no
operating point. `planning/float_budget_numbers.md` §3 finding 5 carries the correction, §7's
pitfall list the new rule; every "no theorem to state" in the archive now reads "at the default
threshold".

**Third, the representative is a shape cover, not a prefix.** The loader's prefix assertion
failed: the 3-block net's `b3` is a 5×5 depthwise (so the 5×5 shape is exercised) where the
paper's `b3` — stage 2, `k = 3` — is 3×3; and its head runs on 24 channels at 56×56 where the
paper's runs on 320 at 7×7. Recorded in the probe and the file header; nothing downstream
depended on the assumption.

**What landed for (a).** `EfficientNetFullFloatBudget.lean` (1177 lines, ~6.6 min to elaborate):
the capped SE gate (`floatBridgesTo_seGateC` / `seBlockFullC` / `seBC`, `EnetSE.bridgeC`,
`EnetSE.mapsC`), the four block shapes with the gate capped — including the **fourth block
shape** the representative has no instance of, `EnetMBBlk.expFwd` (stride 1, `ic ≠ oc`, no skip;
`b9`, `b16`) — `EnetFullWeights nCls` (generic in the class count, since `Maps.dense`'s envelope
depends on the fan-in only), `b0FullEvalForward`, the closed bridge, `b0FullEvalBridge_maps` under
the raised threshold, and `b0Full_float_logits_le`. The profile is the 3-block file's, and it is
measured on THIS net (5,288,548 f32 is the 16-block count).

**(e) DONE — the whole-net eval tie, 3.2(e)'s twin, in the same session.**
`Architectures/EfficientNetFullB0Eval.lean` (~2 s): `MBWEval`/`MBWNoExpEval`/`B0WeightsEval nCls`
(γ, β and the two frozen statistics per BN site, one shared `ε` as the forward's argument, as the
render and the three-block eval both do), the fourth block shape at inference
(`mbExpFwdBEval`/`mbExpGraphBEval`, which the three-block eval render has no instance of), the
four block wrappers, `efficientnetForwardB_fullEval` in nested-application form, the graph
wrappers, `efficientnetFwdGraphB_fullEval` and its `_faithful` (one `rw` per block, then `rfl`).
In the budget file: `EnetFullWeights.toEval`, `b0FullEvalForward_eq_fullEval` (NOT one `rfl`, the
three-block lesson — it rewrites with the per-stage `*Eval_eq_gen` lemmas, and closes in seconds
at sixteen blocks), `b0FullEvalGraph_faithful`, `b0Full_float_logits_le_committed`. The shipped
artifact it ends at is `efficientnet_fwd_eval.mlir`, which IS this net (312 inputs: `%x`, 213
parameters with the conv biases folded into their BatchNorms, 98 statistic slots) and its
1000-class twin `efficientnetin_fwd_eval.mlir`; the head is generic in `nCls`, so one theorem
covers both. ⚠ The typed graph inherits the three-block eval graph's SSA names, which differ from
the artifact's in four cosmetic ways (bias slots, `mu`/`nmu`, `zWa`/`zW1`, `Wfc`/`Wd`); none
enters `den`, and 3.2(e)'s point 2 applies to `EfficientNetRenderPCEval.lean` as a separate
cosmetic pass. The probe's `b0_full_plan` now reads the eval file as a third source and asserts
its record and ladder agree with the training file's, block for block.

**(b) PROBED, DECLINED — the backward has no number at 16 blocks, and the reason changed.**
`b0_full_back_chain` at the shipped leaves (the global `|swish'| ≤ 2`, the ε-floor `S = 317`, the
SE's saved input from the forward's certified window) is 9.112e2648 / 6.550e2648. No loose leaf:
the fiction that sets every measured bound to 1 with `S = 1` and `Sx ≤ 16` is still 1e344. With
the threshold finding this IS statable — `10 ^ 2645` is a numeral the kernel carries — so the
scoping's "cannot state" is no longer the reason not to. The reason is §2 of
`planning/float_budget_numbers.md`: a number that says nothing at 1e182 says nothing at 1e2648,
and the sixteen-block chain has its certified tie (c) without one. Declined and listed in that
document's §6. The chain itself, `efficientnetInputGradB_full` with its `FloatBridgesTo` thread,
lives in `EfficientNetFullWholeBackFloatBridge.lean` so that (c) is about a named term of the
representative's shape.

**(c) DONE — and one step further than the representative.** `EfficientNetFullWholeBackCertifiedTie.lean`
(~3 s): `b0OpaqueA0 … A16` prefix defs, the generic eighteen-stage apex
`efficientnetB_full_has_vjp` (seventeen `vjp_comp`s), the tie with stem and head concrete and the
sixteen blocks opaque (`unfold`, two `rw`s, `rfl`, exactly the 3-block proof), and then
`efficientnetInputGradB_full_eq_efficientnetForwardB_full_vjp`: the tie instantiated at the
concrete `mbNoExpW`/`mbStridedW`/`mbResidW`/`mbExpW` blocks and carried to
`efficientnetForwardB_full_has_vjp` by `HasVJP.backward_unique` (`ConvNeXtBackCertifiedTie.lean`:
two witnesses for one map have one backward). ⭐ That is the step the 3-block file could not take
— it stopped at a `▸`-transported `_committed` witness the kernel could not reduce through — and
the difference is the lemma, not the depth. `efficientnetInputGradB_full_correct` then reads the
result through `efficientnetForwardB_full_has_vjp_correct`, whose proof IS the shape check
`efficientnetForwardB_full_eq_chain`, so the hand-written chain is stated to be the
`pdiv`-contracted Jacobian of the committed nested-application forward. Every batch size, no
smooth point.

**(d) `enetTrunk` at 16 blocks.** Optional and untouched; the `CertLayer` fold is a type-level
check that the block ladder is the shipped one, and the current one is the 3-block ladder.

Gates: `lake build Certs` green, `lake env lean tests/AuditAxioms.lean` 3-axiom clean,
`lake exe docstring-checkrefs`, `python3 scripts/check_audit_coverage.py`; the probe end to end
with every other net's output unchanged.

### 3.4 ViT-Tiny: the backward tiers (T5, T6) — **(a) DONE 2026-09-05; (b) probed, declined**

**Gap, as scoped.** Forward complete at depth 12, three heads, vector LN. The backward had the
float chain (`vitGradFlat`, `MhsaBackFloatBridge.lean`), a block-level tie
(`vitBlockBackPR_eq_transformerBlock_vjp`) and the end-to-end backward-graph fold
(`vitTinyTrunk_is_shipped`), but no whole-net T6 and no number.

⛔ **The scoping named the wrong gap, twice.** It recorded the block tie as being at `heads = 1`
and proposed routing it through `mhsa_layer_spelled` at 3 heads. `ViTMhsaBackCertifiedTie.lean` is
general in `h` throughout — its own header says *"assembled from the block unfold (general heads)"*
— and `mhsa_backward_collapseMH` is what makes it so. The real gap is the **LayerNorm form**: the
tie is at `γ1 β1 γ2 β2 : ℝ` against `transformerBlock_has_vjp_mat`, and the shipped `vitForwardKV`
runs `transformerBlockV` at `γ β : Vec D`. That is package 3.1's ConvNeXt hole exactly — a tie
true of a net the repo stopped running — and it is the *third* time the ViT float cone has been
caught at the scalar affines (`planning/float_budget_numbers.md` §4, row 3). ⚠ A second, quieter
gap: `vitGradFlat`'s final-LN slot is `perRowFlat` of one shared `Vec D → Vec D`, where the
certified backward is per-token, so the whole-net chain needed the enrichment `vitBlockBackPR`
already had on the block side.

**(a) DONE — T6, in three modules and ~660 lines.**

* `Float/ViTWholeBackFloatBridge.lean` names the chain: `vitBlockBackV` (the vector-LN block
  backward, LN slots as `rowLNVecFlatBack`), `vitBlockBackVAt` (every saved slot pinned to the
  real forward at the block's own input), `vitTowerBackK` (the depth-`k` fold), the two saved
  prefixes `vitSavedPE` / `vitSavedBody`, and `vitInputGradK`. B0's
  `EfficientNetFullWholeBackFloatBridge.lean` role: the tie is about a named term.
* `Architectures/ViTVecLNBackCertifiedTie.lean` re-states the block tie there:
  `vitBlockBackV_eq_transformerBlockV_vjp` and its flat form `vitBlockBackVAt_eq_vjp`.
* `Foundation/ViTWholeBackCertifiedTie.lean` folds the tower and closes the apex:
  `vitTowerBackK_eq_vjp`, `vitInputGradK_eq_vitForwardKV_vjp`, `vitInputGradK_correct`, and
  `vitTinyInputGrad_eq_vitTiny_vjp` at the shipped `3×224×224` / 196+1 tokens / D 192 / 12 blocks
  / 10 classes.

⭐⭐ **There is no new analysis in any of it, because ConvNeXt already built ViT's LayerNorm
backward.** `rowLNVecFlatBack` (`ChannelLNFloatBridge.lean`) is `perRowIdxFlat` of
`bn_grad_input c ε 1 (X r) ∘ diagBack γ`, its header says it is *"literally ViT's per-token LN with
'token' read as 'spatial position'"*, and `rowLNVecFlat_has_vjp_backward_eq` already pins it to
`layerNormVec_per_token_has_vjp_mat`. So the vector-LN seam is ONE lemma
(`rowLNVecFlatBack_eq_vecLN_vjp`, two tactics) and everything else in the block —
`mhsaBackFlat_eq_mhsa_vjp`, `dense_transpose_eq_mulVec`, `diagBack_eq_gelu_vjp`,
`transformerMlp_back_flat_eq_perRowFlatPR`, `perRowFlatPR_residual` — is LayerNorm-agnostic and
reused from the scalar file verbatim. Both sublayer decompositions and the block unfold stay
`rfl` at the vector LN. §7's *"grep the whole cone for a bound before proving one, not the files
named after the net"* paid a second time, on the same lemma.

⛔ **The tower is NOT `towerBack` of a `List`.** `towerBack (f :: fs) = towerBack fs ∘ f` applies
the HEAD first, so a list in block order runs the shallowest block's backward first; the ordering
was never pinned because the only `towerBack` result in the repo is at `List.replicate`
(`towerBack_replicate`), where it cannot matter. `vitTowerBackK` is its own recursion, mirroring
`vitBodyKVFlat`'s head-first fold, which is what makes the saved-activation thread visible — and
the thread is the content: the tail's saved input is block 0's forward OUTPUT.

⚠ **The apex needs the term-mode escape.** `vitForwardKV_has_vjp` is tactic-built and opens with
`unfold vitForwardKV`, so its `.backward` sits behind an `Eq.mpr`. `vitApexVJP` is the same
four-factor `vjp_comp` chain written as a term, and `HasVJP.backward_unique` carries the tie to
the committed witness — B0 3.3(c)'s escape, and §5's `▸`-transport trap.

⭐ The result is `HasVJP`, not `HasVJPAt`: ViT has no kink anywhere, so like ConvNeXt-T and unlike
r34/mnv2/B0 there is no smoothness witness, no operating point and no batch size. The only
hypothesis is `0 < ε`. Three of the four endpoint ties are one tactic each and the patch embed's
is `rfl` — `patchEmbed_flat_has_vjp`'s `backward` field IS
`patchEmbed_input_grad_formula`, the one endpoint in the repo that needed no reconciliation.

Gates: `lake build Certs` 3966 green, `lake env lean tests/AuditAxioms.lean` 3-axiom clean on all
eighteen new declarations, `lake exe docstring-checkrefs`,
`python3 scripts/check_audit_coverage.py`.

**(b) PROBED, DECLINED — and the scoping's predicted CAP could not have been right.**
`scripts/float_budget_envelope.py` gains `vit_back_chain` / `verify_vit_back` (195 stages, 390
rounded inequalities), `vit_qkv_xhat`, and the four sdpa-core / patch-embed-back leaf helpers.

| | window | budget | per block |
|---|---|---|---|
| **shipped: saved Q/K/V from `bnXhat_sq_le`** | **5.686e399** | **1.703e399** | 10^32 |
| saved Q/K/V from the FORWARD's certified window | 5.798e1557 | 1.741e1557 | 10^225 → 10^32 |
| the same, `\|istd\| ≤ 16` | 2.178e367 | 6.741e366 | 10^29 |
| attention cores replaced by the identity | 2.827e251 | 3.140e250 | 10^19 |
| depth 2 / depth 6 | 5.038e76 / 8.399e205 | 2.958e75 / 1.372e205 | 10^32 |

⛔ **It is a FOLD (ratio 0.30), not a cap, and no cap is available.** `Maps.capped` needs a stage
whose window is bounded by a constant; no backward stage has one, because a VJP is linear in the
cotangent and every window is proportional to it. That is `planning/float_budget_numbers.md`
finding 2 read the other way, and it is the same answer MobileNetV2's 17-block backward got.

⭐⭐ **The probe's real product is the leaf, recorded as finding 8.** `floatBridges_mhsaBack` takes
`|Q i k| ≤ qA` as a FREE hypothesis, and the chain had no reason to discharge it from anything but
the forward's certified window — which grows `2·A·S` per LayerNorm site to 1e108 at depth 12.
`bnXhat_sq_le` bounds ViT's LN OUTPUT by the constant `G·√192 + Bl` (ViT's per-token LN is
literally `γ·x̂ + β`), so every saved projection is ≤ 3281 whatever arrives. 1158 orders, and the
per-block multiplier stops growing with depth. ⚠ Arithmetic in the probe only — one line from
`bnXhat_sq_le` (`layerNormVec D ε γ β x k = γ k * bnXhat D ε x k + β k` by delta), but not written
in Lean, because the number it rescues was declined. ⚠ The same reading is available to ViT's committed
FORWARD number and to every other LayerNorm/BatchNorm forward in the table; **not chased** (user
decision, 2026-09-05 — the thread is closed and the numbers are vacuous either way), recorded so
a future forward number is written with the `min` from the start.

**Declined** on B0 3.3(b)'s ground: a number that says nothing at 1e2648 says nothing at 1e399,
and T6 landed without one. ViT's cost is also the highest of the three declined backwards — it is
the one net needing `Maps` leaves nothing else uses (the three sdpa cores, the patch-embed
backward, a per-token `perRowPR` lift), on top of the `floatBridgesTo_` migration
`MhsaBackFloatBridge.lean` still wants. Listed in `planning/float_budget_numbers.md` §6.

⛔ `ViTBackFloatBudget.lean` is CANCELLED. §6's file list scoped one new file for 3.4; three
landed, because the chain needed naming and the block tie needed re-stating before the fold.

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

## 4. The BatchNorm-world axis — DECIDED 2026-09-06, port in progress

**The decision is option 1: port both nets' Proofs tiers to batch BN.** Recorded in
`formalization.yaml` 4e, which is a new section — 4d named neither world, which was the disclosed
defect this axis carried. The shared foundation landed the same day; the per-net tiers have not.

### 4.0 What the split is, in the artifacts

`resnet34_fwd.mlir` and `mobilenetv2_fwd.mlir` reduce over `[2,3]` — 73 and 105 spatial
reductions, i.e. 36 and 52 BN sites ×2 plus the head GAP — so they are per-example. So is the SGD
trainer `resnet34_train_step.mlir`. ⚠ **Its 72 batch-axis reductions are NOT batch statistics**:
they are the γ and β parameter gradients (`Σ_{n,h,w} dy·x̂`, `Σ_{n,h,w} dy`), which reduce across
the batch because the parameter is shared. Every Adam and momentum step — `resnet34_sgd_train_step`
(468 `[0,2,3]`, 1 `[2,3]` for the GAP), `resnet34in_mom256_*`, `mobilenetv2in_adam64_*` and the
rest — is batch BN. The Imagenette SGD trainer is a real artifact and the Proofs tier is correctly
paired with it; the ImageNet accuracies the repo quotes come from the other world.

### 4.1 DONE 2026-09-06 — the shared foundation, both directions

`Float/BnBatchFloatBridge.lean` (≈300 lines, ~4 s to elaborate, `Certs` 3966 → 3967, one new module):
`bnchwEquiv`, `bnBatchTensor4FV` / `floatBridgesTo_bnBatchTensor4` / `..._eps`,
`bnBatchTensor4BackFV` / `floatBridgesTo_bnBatchBack`, and the envelopes
`Maps.bnBatchTensor4`, `Maps.bnBatchTensor4Capped`, `Maps.bnBatchBack`. Ten declarations, all
3-axiom clean; `lake build Certs`, `lake env lean tests/AuditAxioms.lean`,
`lake exe docstring-checkrefs`, `python3 scripts/check_audit_coverage.py` all green.

⛔ **Correction 1 to this section as scoped: the FORWARD leaf was missing too.** §4 priced "no
batched BatchNorm backward leaf exists" as the real cost. In fact `bnBatchTensor4` had no float
leaf in **either** direction. `EfficientNetWholeFloatBridge.lean` takes twenty-odd
`hbn : FloatBridges (StableHLO.bnBatchLA …)` as hypotheses, and a legacy `FloatBridges`
constrains no float implementation at all (4d) — so those hypotheses named nothing and nothing
discharged them. B0's numbers dodge the hole twice over: `b0_float_logits_le` is at inference BN
(`batchMap` of a per-example op) and `b0_grad_float_le` is at `N = 1`, where the batched width
`N·h·w` coincides with `h·w` and the per-example leaf is honest.

⛔ **Correction 2: B0's `N = 1` is T5, not T6.** This section attributed the missing leaf to
"the reason B0's T6 is stated at `N = 1`". `efficientnetInputGradB_full_correct` takes `(N : Nat)`
and §3.3(c) says so — every batch size. What is at `N = 1` is the backward *number*
`b0_grad_float_le`, for the reason its own header gives: `bnGradInputReMag`'s gain carries
`Xh² = n = N·h·w`, so the numeral moves with the batch (7.104e182 at `N=1`, 2.880e194 at `N=256`).
A batched leaf does not remove that; it lets the number be stated at the batch actually trained at.

⭐ **Cheaper than priced, and the reason is the same one this document keeps rediscovering.**
`bnBatchTensor4` IS `bnPerChannelTensor3` at a different width — both are `bnPerChannelFlat oc m`
conjugated by a permutation, with `m = N·(h·w)` rather than `h·w`. The flat leaves were already
generic in `m`, `floatBridgesTo_gather` holds for any `Equiv`, and the round-trip lemmas
(`bnchwFwdIdx_bnchwBackIdx`, `bnchwBackIdx_bnchwFwdIdx`) were already proven. Both bridges
typechecked on the first pass with no new analysis; §5's *"grep the whole cone for a bound before
proving one"* paid again.

⭐ **The `Maps` proofs are layout-free, so they were factored rather than copied.**
`Maps.bnPerChannelTensor3` and `Maps.bnPerChannelBack` never mention `oc`/`h`/`w` except to pick a
channel index for a nonnegativity side condition: they are statements about `bnLeafMag`/`bnLeafMod`
and `bnGradInputReMag`/`bnGradInputBudgetG`, which take a **width** and no indices.
`Maps.bnLeafCore` / `Maps.bnLeafCoreCapped` / `Maps.bnGradLeafCore` extract that, and the three
batched envelopes are corollaries at `rfl`. R50's and MNv4's will be too — this is §5's "two lists
for one net" applied to a `linarith` chain. ⚠ The existing per-example envelopes were left alone
rather than re-based on the cores, to keep the change out of the downstream rebuild.

### 4.1b DONE 2026-09-06 — ResNet-34's T1-forward and T2

`Architectures/ResNet34FullB.lean` (~310 lines, ~1.7 s): the `R34IdW` / `R34DownW` /
`R34BWeights nCls` records, the four batched block forwards (`r34IdB`, `r34DownB`, `r34StemB`,
`r34HeadB`), `resnet34ForwardB_full` in nested-application form, the four block-kind graphs at the
render's own tokens, their `_faithful` lemmas, and `resnet34FwdGraphB_full_faithful` — one `rw` per
block. `Certs` 3968 green, six declarations 3-axiom clean, all four scripts green.

⭐ **It is an enumeration, as predicted.** `ResNet34BackB0.lean` already carried every batched
stage (`cbReluB`, `cbReluStridedB`, `projStridedB`, and `projB` from `EfficientNetRenderPC`) with
its `_at` VJP and backward-graph faithfulness at `bnBatchLA`; `BackNetFolds.lean` already folded
them to `[3,4,6,3]`. The only thing missing was the level above. Every `den` lemma the graphs need
(`den_batchOp_conv`, `_convStrided`, `_relu_eq_reluF`, `_maxPool3s2`, `_gap`, `_dense`,
`den_bnBatchF`, `den_addV`) already existed.

⚠ **Two conventions are stated in the file header because nothing checks them here.** Padding is
symmetric at all seven stride-2 sites (`.convStrided`, **not** `.convStridedXla` — B0's stem is the
XLA-`SAME` one and the two tokens have identical types), and the stem pool is 3×3/s2
(`maxPool3s2Flat`, same type as the 2×2 pool and a different function; the render carried the wrong
one until 2026-08-04). `scripts/convention_audit.py` sees the first at the artifact tier only.

⭐ The head is generic in `nCls`, so one statement covers the 10-class Imagenette artifacts and the
1000-class `resnet34in` ones.

⛔→✅ **The VJP half of T1 was blocked on one missing lemma; it landed the same day (4.1c).** The whole-net
`HasVJPAt` needs the stem pool's VJP lifted through `batchMap`, and `batchMap_has_vjp`
(`EfficientNetChainClose.lean`) is the GLOBAL form only — there is no `batchMap_has_vjp_at`. B0
never needed one because swish is smooth everywhere and its stem has no pool. The per-example
pieces are both there (`maxPool3s2Flat_has_vjp_at_vec`, `MaxPool3s2BackFloatBridge.lean`, is
already the `Vec`-point form a chain needs), so this is the pointwise peer of an existing
construction plus its differentiability companion — write it beside `batchMap_has_vjp`, then the
r34 assembly is `MobileNetV2FullVJP.lean`'s shape: per-block `Pos`/`SmoothAt` bundles, sixteen
prefix defs, and the apex. ⚠ r34 carries **two** relu clauses per block (the body's mid-relu and
the post-residual one) where MobileNetV2 carries two relu6 clauses; same count, same shape.

### 4.1c DONE 2026-09-06 — `batchMap` at a point

`Foundation/BatchMapVJPAt.lean` (~200 lines, ~1.9 s): `pdivMat_rowIndep_at`,
`batchMap_differentiableAt`, `pdiv_batchMap_at`, `batchMap_has_vjp_at`. All four 3-axiom clean;
`Certs` 3969.

⭐ **`pdivMat_rowIndep`'s global differentiability was never actually global.** Its docstring
explains why it asks for `Differentiable ℝ g` — a non-differentiable coordinate makes `fderiv` junk
and breaks the per-row decomposition — but every *use* of the hypothesis in the proof is at a ROW
of the matrix the statement is about. So it weakens to `∀ r, DifferentiableAt ℝ g (A r)` with no
change to the argument at all; the only edit is moving the row-projection equation
`(rowProj k) (Mat.flatten A) = A k` to the top, so the coordinate differentiability can be stated
at the projected point. It compiled on the first pass.

⚠ **`batchMap_has_vjp_at` is built field by field, not transported with `▸`.** `batchMap_has_vjp`
transports along `batchMap_eq_rowwiseFlat`; an `Eq.mpr` blocks `.backward` from reducing, which is
§5's transport trap and which T6 will need. `maxPool3s2Flat_has_vjp_at_vec` was written for exactly
this reason one tier down.

✅ **Checked against the case that motivated it.** The batched pool VJP is
`batchMap_has_vjp_at _ v (fun r => maxPool3s2Flat_has_vjp_at_vec (Mat.unflatten v r) (hs r))
(fun r => maxPool3s2Flat_differentiableAt_vec (Mat.unflatten v r) (hs r) hc hh hw)` — the two
per-example pieces plugged straight in with no glue between them, which is the evidence that the
lemma has the right shape. It is stated in r34's VJP file rather than here, since
`maxPool3s2Flat_has_vjp_at_vec` is in the `Float` tier and this is a `Foundation` file.

### 4.1d DONE 2026-09-06 — ResNet-34's T1 is complete

`Architectures/ResNet34FullBVJP.lean` (514 lines, **2.8 s**): the four hypothesis bundles
(`R34IdPos` / `R34DownPos` / `R34IdSmoothAt` / `R34DownSmoothAt`) plus `R34StemSmoothAt` and
`R34PoolSmoothAt`, the six bundle lemmas, `r34Pre0 … r34Pre16`,
`resnet34ForwardB_full_has_vjp_at`, `resnet34ForwardB_full_eq_chain`, and
`resnet34ForwardB_full_has_vjp_at_correct`. Seven declarations 3-axiom clean, `Certs` 3970.

⭐ **Delegation, as scoped.** `r34BasicBlockB_has_vjp_at` and `r34DownBlockB_has_vjp_at`
(`ResNet34BackB0.lean`) are exactly the two shapes `r34IdB` / `r34DownB` unfold to, so the bundle
lemmas are one line each. The only piece that did not exist is 4.1c's `batchMap_has_vjp_at`.

⛔ **Two kink clauses per block, not one** — the body's mid-relu AND the post-residual **outer**
relu. That outer relu is ResNet's structural difference from MobileNetV2/EfficientNet, whose
residual add IS the block output, and it is why `ResNet34BackB0.lean`'s block VJPs take `h_s1` and
`h_out` separately. Sixteen blocks give 32 clauses, plus the stem's relu and the pool's no-tie
condition, bundled into 18 binders.

⚠ **The pool's condition is PER EXAMPLE** (`∀ r : Fin N, MaxPool3s2Smooth …` on that row): a tie
is a property of one image's 3×3 window, not of the batch. That is exactly the shape
`batchMap_has_vjp_at` consumes, which is the second check that 4.1c has the right statement.

⭐ **The head takes no hypothesis at all.** GAP and dense are smooth and each is `batchMap` of a
per-example op, so `batchMap_has_vjp` (the global one) suffices — the one place in this net where
the pointwise machinery is not needed.

⭐ `_correct` is about `resnet34ForwardB_full` itself — the forward whose typed graph 4.1b
certifies — not about the layered chain the VJP is assembled on; `resnet34ForwardB_full_eq_chain`
is the bridge, peeled one `*_apply` layer at a time per `MobileNetV2FullVJP.lean`'s recipe.

**ResNet-34's T1 and T2 at batch BN are done. T3 is next.**

### 4.2 Still open, per net

For each of ResNet-34 and MobileNetV2, at `bnBatchLA` (r34's T1 and T2 landed, 4.1b–4.1d):

| tier | what it needs | mirror |
|---|---|---|
| T1 | net-level ℝ forward + whole-net `HasVJPAt` (both nets have relu kinks) — ✅ r34 | `EfficientNetFullB0.lean` |
| T2 | typed forward graph, per-block `_faithful` then chained — ✅ r34 | `ResNet34RenderB` / `MobileNetV2RenderB` tokens |
| T3 | FaithfulPoC / TiePoC against the batch-BN train step | the existing per-example pair |
| T4 | training-BN forward budget — a **CAP**, as `r34_train_float_logits_le` already is | `Maps.bnBatchTensor4Capped` |
| T5 | backward budget, one theorem per `N` | `Maps.bnBatchBack` |
| T6 | certified backward tie at `bnBatchTensor4` | `Resnet34BackCertifiedTie.lean` |

⭐ **Two tiers are cheaper than the table suggests.** The block-level batched VJPs and
backward-graph faithfulness already exist for both nets (`ResNet34BackB0.lean`,
`MobileNetV2BackB0.lean`), folded to the paper depth by `BackNetFolds.lean`'s `r34Trunk_3463` —
so T1's hard half is done and T6 composes over it. And the **eval-mode forward budgets are
world-agnostic**: frozen statistics reduce nothing, so `r34_float_logits_le` and
`mnv2_float_logits_le` already hold in both worlds and need only saying so.

⚠ **Every batched number carries the batch size in its statement.** Decide `N` once per net, at
the batch the quoted checkpoint trained at, and put it in the theorem name or the file header —
not in a docstring.

## 5. Traps, all previously paid for

* **Probe before Lean.** Every number comes from `scripts/float_budget_envelope.py` first; the
  Lean re-asserts rounded rows. A fold whose modulus is quadratic in the window is a CAP, decided
  at the probe, not discovered at `norm_num`. ⭐ A numeral past 1e253 is NOT a reason for a cap or
  an operating point: `norm_num`'s ceiling is `exponentiation.threshold` (default 256) and
  `set_option exponentiation.threshold 400 in` lifts it at no cost (3.3(a)). Cap the stage whose
  WINDOW is bounded (a sigmoid, a relu6), not the stage whose error is large.
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
* ⛔ **An audit's named gap can be the wrong one.** §3.4 as scoped said ViT's block tie was at
  `heads = 1`; it is general in `h`, and says so in its own header. The real gap was the LayerNorm
  form, which the audit did not mention — and which the same net had already been caught on twice
  (`planning/float_budget_numbers.md` §4, row 3). Re-read the file before costing the package, and
  cost it against the SHIPPED spelling of every convention, not against the one the row names.
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
`MobileNetV2PaperWholeBackCertifiedTie.lean` and `Architectures/MobileNetV2FullPaperEval.lean`
(3.2(e)) all landed (⛔ `MobileNetV2PaperBackFloatBudget.lean` is CANCELLED, there is no backward number to state, and
the VJP was already in `MobileNetV2FullVJP.lean`); 3.3 `EfficientNetFullFloatBudget.lean`,
`EfficientNetFullWholeBackFloatBridge.lean` and `EfficientNetFullWholeBackCertifiedTie.lean`
landed, as did `Architectures/EfficientNetFullB0Eval.lean` for 3.3(e) (⛔
`EfficientNetFullBackFloatBudget.lean` is DECLINED: statable at 1e2648 since the threshold
finding, and worth nothing); 3.4
`Float/ViTWholeBackFloatBridge.lean`, `Architectures/ViTVecLNBackCertifiedTie.lean` and
`Foundation/ViTWholeBackCertifiedTie.lean` all landed (⛔ `ViTBackFloatBudget.lean` is DECLINED:
a fold at 5.686e399 / 1.703e399, statable at `exponentiation.threshold 500` and worth nothing,
and the most expensive of the three declined backwards); §4 `Float/BnBatchFloatBridge.lean` landed 2026-09-06 (the batched BatchNorm leaves, both
directions, plus the three layout-free `Maps` cores); 3.5 `Resnet50FullB.lean`,
`Resnet50FaithfulPoC.lean`, `Resnet50TiePoC.lean`, `Resnet50FloatBudget.lean`,
`Resnet50BackFloatBudget.lean`, `Resnet50WholeBackCertifiedTie.lean`; 3.6 the same six for
MobileNetV4. Every new file: a `lakefile.lean` `Certs` root or an import of one, an
`AuditAxioms.lean` block, a `formalization.yaml` row with the convention comment, a
`planning/float_budget_numbers.md` section-1 row for each number.

# Bringing every net's Proofs tier to its paper-faithful net

**Scoped 2026-09-05 from the Proofs-tier audit run during the XLA-SAME re-spelling. Nothing
below is started except where a row says so; 3.1, 3.2(a)–(c) and 3.3 have since landed.** The target is the
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
| ViT-Tiny | `vitForwardKV` / `vitBodyKVFlat`, depth 12, D 192, 3 heads | ✓ | ✓ `vitFwdGraphKMHV_faithful` | ✓ 200 params | ✓ CAP | ✗ | ✗ block-level only | none |
| EfficientNet-B0 | `EfficientNetFullB0.lean`, 16 MBConv | ✓ | ✓ train and eval BN (`EfficientNetFullB0Eval.lean`) | ✓ 262 params | ✓ CAP 2.416e287 at the 16 SE sigmoids, window 1.886e279 honest | ⛔ no number at 16 blocks (9.112e2648; statable, declined) | ✓ `efficientnetInputGradB_full_correct`, through `backward_unique` to the concrete witness | none |
| MobileNetV2 | `MobileNetV2FullPaper.lean`, 17 blocks | ✓ `mobilenetv2_full_has_vjp_at` (`MobileNetV2FullVJP.lean`), shape check `mobilenetv2ForwardPaper_eq_chain` | ✓ train BN (eval: 3.2e) | ✓ 210 params | ✓ CAP 8.176e16, all 52 BN sites | ⛔ no number at 17 blocks | ✓ `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp` | 17 blocks at toy dims; 2 blocks at 224 |
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

### 3.2 MobileNetV2 at 17 blocks (T1, T4, T5, T6) — **(a)–(c) DONE 2026-09-05; (e) OPEN**

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

**(e) OPEN — the whole-net eval tie, and the artifact it would end at.** One session.

**Gap.** `mnv2Paper_float_logits_le` is about `mnv2PaperEvalForward`, the record-bundled
composition in `MobileNetV2PaperFloatBudget.lean`. Each of the seventeen blocks is `rfl`-tied to
the abbreviation the committed inference forward is built from
(`MnvBlock.{body,strided,res}Fwd_eq_pcEval`, `MnvBlockNoExp.fwd_eq_pcEval`), but the WHOLE net is
tied to nothing: the paper net's eval twin has no ℝ-def and no typed `SHlo` graph in Lean.
`MobileNetV2RenderPCEval.lean` covers the six-block net; `MobileNetV2FullPaper.lean` is at
training BN, the world the VJP and the graph live in. So `MobileNetV2FloatBudget.lean` has a
`*_committed` restatement and this file does not.

⭐⭐ **And this rung is worth more here than the one it mirrors.** The six-block eval graph
denotes a net with NO committed artifact — `verified_mlir/` has
`mobilenetv2_reduced_train_step.mlir` and no reduced eval forward — so that number's "tie to the
rendered graph" ends at a typed graph. The paper net's eval forward IS shipped, twice:
`mobilenetv2_fwd_eval.mlir` (315 inputs — `%x`, `paperSig`'s 210 params, 104 stat slots) and its
1000-class twin `mobilenetv2in_fwd_eval.mlir`. This tie ends at bytes.

**Order.**

1. `LeanMlir/Proofs/Architectures/MobileNetV2FullPaperEval.lean`, the eval twin of
   `MobileNetV2FullPaper.lean`: records (`IVW`/`IVWNoExp` plus μ/v per site — ⚠ **one shared ε**,
   as `mobilenetv2Forward_full_pc_eval` and the render both do, where the training file carries a
   per-site ε), the four block wrappers in the committed `ivExpandPCEval`/`ivDepthwisePCEval`/
   `ivDepthwiseStridedPCEval`/`ivProjectPCEval` vocabulary, `mobilenetv2ForwardPaperEval`, the four
   block graphs with `bnPerChannelEvalF`, their `_faithful` lemmas, then
   `mobilenetv2FwdGraphPaperEval` + `_faithful`. Two mirrors: that file for the structure,
   `MobileNetV2RenderPCEval.lean` for the eval nodes.
2. ⚠ **SSA names: `bnSiteP`'s, not the six-block file's.** The render names the stat slots
   `%stnmu`/`%stnvar`, `%b{k}{en,dn,pn}{mu,var}`, `%hnmu`/`%hnvar`;
   `mobilenetv2FwdGraphFullPCEval` uses `%mue1`/`%vare1`, which matches no artifact and could not,
   since its net has none. Names are pretty-printing metadata and do not enter `den` — matching
   them is what lets a reader diff the typed graph against the committed text.
3. In `MobileNetV2PaperFloatBudget.lean`: `MnvPaperWeights.toEval`,
   `mnv2PaperEvalForward_eq_paperEval` (`rfl`), `mnv2PaperEvalGraph_faithful`,
   `mnv2Paper_float_logits_le_committed`. ⭐ Try defining `mnv2PaperEvalForward` AS
   `mobilenetv2ForwardPaperEval (W.toEval ε)` outright — one spelling, no `rfl` needed — and fall
   back to the separate spelling plus the `rfl` if the `.comp` chain will not typecheck against a
   17-deep nested def.
4. ⭐ **Make the head generic in `nCls` while there, and the profile becomes exact.**
   `Maps.dense`'s envelope depends on the fan-in `1280` and never on the output count, so
   `MnvHead 1280 nCls` carries the same two numerals verbatim and one theorem covers both shipped
   eval artifacts. It also closes a real qualification in the header: the 3,504,872-entry
   checkpoint the profile is measured on is the **1000-class** net, so at 10 classes
   `|·| ≤ 28/10` is a measurement on all 52 convolutions and 52 BatchNorms and an assumption on
   the `1280 × 10` head.

**Traps.**

* **Two lists for one net.** This adds a third spelling of the ladder (the eval forward and its
  graph) and a fourth of the widths. The forward/graph pair is pinned by its own faithfulness
  theorem and the float file's spelling by the `rfl`; the WIDTHS are pinned only by
  `scripts/float_budget_envelope.py`'s `mnv2_paper_plan`, which today asserts
  `mobilenetv2ForwardPaper` and `paperSig` name the same 17 blocks. Extend that loader to read the
  eval file too, or the new record is the one list nothing checks.
* **Elaboration.** `mobilenetv2FwdGraphPaper_faithful` is `simp only [...]` then `rfl` at 17
  blocks; the eval twin adds 52 `bnPerChannelEvalF_faithful` rewrites and needs `maxRecDepth`
  raised (the six-block eval needed 10000). §5's discipline applies unchanged.
* **The eval graph must be the same net as the train step whose statistics it consumes** —
  `mnv2FwdEvalFaithfulV`'s own header comment, and the reason the padding thread mattered. Both
  are XLA-`SAME` since 2026-09-05, so this is now a check rather than a risk.

**Done when** `den (mobilenetv2FwdGraphPaperEval …) = mobilenetv2ForwardPaperEval …` compiles,
`mnv2Paper_float_logits_le_committed` states the number with that forward on the real side, the
`formalization.yaml` row and `planning/float_budget_numbers.md` §1's "one exception" note both
come out, and the four gates are green. The mathematics is nil; the cost is elaboration.

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

New, by package: 3.1 none; 3.2 `MobileNetV2PaperFloatBudget.lean` and
`MobileNetV2PaperWholeBackCertifiedTie.lean` landed, `MobileNetV2FullPaperEval.lean` is 3.2(e)'s
(⛔ `MobileNetV2PaperBackFloatBudget.lean` is CANCELLED, there is no backward number to state, and
the VJP was already in `MobileNetV2FullVJP.lean`); 3.3 `EfficientNetFullFloatBudget.lean`,
`EfficientNetFullWholeBackFloatBridge.lean` and `EfficientNetFullWholeBackCertifiedTie.lean`
landed, as did `Architectures/EfficientNetFullB0Eval.lean` for 3.3(e) (⛔
`EfficientNetFullBackFloatBudget.lean` is DECLINED: statable at 1e2648 since the threshold
finding, and worth nothing); 3.4
`ViTWholeBackCertifiedTie.lean`, `ViTBackFloatBudget.lean`; 3.5 `Resnet50FullB.lean`,
`Resnet50FaithfulPoC.lean`, `Resnet50TiePoC.lean`, `Resnet50FloatBudget.lean`,
`Resnet50BackFloatBudget.lean`, `Resnet50WholeBackCertifiedTie.lean`; 3.6 the same six for
MobileNetV4. Every new file: a `lakefile.lean` `Certs` root or an import of one, an
`AuditAxioms.lean` block, a `formalization.yaml` row with the convention comment, a
`planning/float_budget_numbers.md` section-1 row for each number.

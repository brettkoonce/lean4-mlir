# Bringing every net's Proofs tier to its paper-faithful net

**Scoped 2026-09-05 from the Proofs-tier audit run during the XLA-SAME re-spelling; current to
2026-09-07. ▶ START at "NEXT SESSION" below — it is three rows and the rest of this document is
the record behind them.** The target is every certification tier stated at the net the artifact
runs, the column "tiers only at a reduced or representative net" empty for every architecture —
and, since the review of 2026-09-06, at the ARTIFACT the quoted number came from, not only the
architecture.

⭐ **Where it stands.** Sections 3.1–3.4 landed 2026-09-05; section 4's BatchNorm-world decision was
taken 2026-09-06 and its port is complete on every statement that says something; the review of
2026-09-06 added three axes the audit had not named — optimizer form (4b), renderer (4c) and data
parallelism (4d) — and 4b is done at the fold, 4c is done for all four nets, 4d for its first
piece. **ResNet-34, MobileNetV2 and ResNet-50 are finished** on every tier that is not a float
budget, and ConvNeXt-T was the reference when this was scoped. What is genuinely open is one large net —
every capstone is landed and 4d's op node too.

**Why the three new axes exist at all.** The suite grew organically: one net at a time, each with
the renderer, optimizer form and batch index that was convenient when it landed. The end state the
user wants (2026-09-06) is homogeneous — every ImageNet-scale net is ONE batched chain, whose
gradient nodes are un-fused and feed one shared optimizer-tail fold, whose loss cotangent is one
shared lemma at a general target, and whose data-parallel mean is one AST node — with the Proofs
tier stated at that shape and the CIFAR chapter keeping its per-example op family as the
pedagogical ladder. 4b, 4c and 4d are that unification, in the order that pays soonest.

**Order of work — where the thread stands after 2026-09-07.** Eleven packages landed on
2026-09-06:
4b (four files, 41 declarations), 4.2a, **4.2b + 4.2c** (MobileNetV2's T1/T2/T3 at batch BN),
**4c legs 1 and 2** (ResNet-34 and MobileNetV2 on one chain, both per-example renderers retired),
the **ImageNet PAIRS extension**, **4b's capstone re-pointing for EfficientNet-B0**,
**4d piece 1** (data parallelism at the ℝ level), **§3.5a** (ResNet-50's LAMB tail and BCE
cotangent) and **§3.5(a)+(b)+(c)** (ResNet-50's T1, T2 and T3). `Certs` 3966 → **3986**. **§4.2d**
— T6 at batch BN for BOTH ResNet-34 and MobileNetV2 — followed just after midnight and is dated
**2026-09-07** to match its commit; `Certs` 3986 → **3990**. **§3.5d** — ResNet-50's T6 — followed
the same day on §4.2d's machinery and cost four declarations; `Certs` 3990 → **3992**. **§4c-ter**
— 4c leg 4, ViT-Tiny's nineteen artifacts onto the batched chain with zero bytes moved — closed the
same day; `Certs` 3992 → **3993**. **§4c-quater** — 4c leg 3, ConvNeXt-T's seventeen writers onto
the batched chain, thirteen train steps moving exactly 78 lines each by user decision, with the
batched fold landed FIRST — closed the same day; `Certs` 3993 → **3994**.

⭐⭐ **What that adds up to.** Both BatchNorm nets have T1, T2, T3 **and T6** stated at the
artifact that trains, and ResNet-50 now has the same four — for all three nets every remaining row
is a float budget, i.e. §4's port and §3.5 are complete on every statement that says something; both per-example renderers are gone; `check_adam_prefix`'s `KNOWN_SPLIT` ratchet is
**EMPTY** and its coverage now reaches the ImageNet tier (20 paired, 0 split, plus a completeness
assertion); ⭐ ViT-Tiny (4c leg 4, nineteen artifacts, zero bytes moved) and ConvNeXt-T (4c leg 3, seventeen
writers, 78 lines moved in each of thirteen train steps) are on ONE chain too — every net is; three of the five T3 capstones are at the
un-fused gradient node and the smoothed loss; and the data-parallel disclaimer every one of those ties carries now has a theorem behind
it — `(1/R) Σ_r g_r` is the gradient of the mean of the per-replica losses, that mean IS the
global-batch loss for a net with no batch coupling, and for a batch-BN net it provably is NOT.

✅ **The one axis that was ORDER-CONSTRAINED is released on both halves.** ConvNeXt-T's and
ViT-Tiny's capstones could not usefully be re-pointed ahead of their renderer legs: their Adam
artifacts came from the PER-EXAMPLE renderers, where the six-op loss chain is emitted at `N := 1`
with the batch in `pretty`'s argument rather than inside `den`, while `SmoothedLossCot` is stated at
the batched index. Writing those capstones then meant writing them twice. ✅ **Both legs landed 2026-09-07**
(§4c-ter, §4c-quater), so both capstones are writable once, at the batched index.

**4c has its own thread and log: `planning/renderer_convergence.md`, and it is CLOSED.** All four
legs are done. 4d's ℝ-level lemma has landed; its op node (`allReduceMeanF`) is no longer gated on
anything. 3.5 and 3.6 unchanged, on the batched chain from the start.

**WHAT HAS LANDED, so a fresh session does not re-read the rows.** 4b (four files, 41 decls),
4.2a–4.2d (r34 + MobileNetV2's T1/T2/T3/T6 at batch BN), 4c legs **1–4**, the ImageNet PAIRS
extension, 4d piece 1, B0's capstone re-pointing, and §3.5a–§3.5d (ResNet-50's LAMB tail, BCE
cotangent, T1, T2, T3, T6). `Certs` 3966 → **3994**. Their write-ups are the numbered sections
below; nothing in them is open.

**NEXT SESSION — one live target: MobileNetV4 (its own planning doc first), then the cleanup/unification session. Both capstones (§4b.6, §4b.7) and 4d piece 2 (§4d.2) landed 2026-09-07.**

| target | cost | what a fresh session needs to know |
|---|---|---|
| ⭐ **3.6 MobileNetV4-Conv-M** — the only open item; ✅ its planning doc is `planning/mnv4_proofs_tier.md` (2026-09-07) — START THERE, not at §3.6 | many sessions; §3.6 says what the FIRST one is | The last net with nothing at the net level, and it skips 4b and 4c entirely (`mnv4FwdChainB` is already the one traversal `@mnv4_fwd`, its eval twin and the train step all use). ⭐ **Session one is NOT T1**: it is the three pieces `MobileNetV4BackB0.lean`'s own header names as missing — the fused stage's VJP and backward graph, the head's, and the strided UIB body assembled from the stages already there. Everything after that is enumeration. ⭐ The four-family collapse, the hard half, is DONE. |

⛔ **Do NOT write, and the reason is a closed thread, not an oversight.** Every remaining row for
ResNet-34, MobileNetV2 and ResNet-50 is a float BUDGET (T4/T5), and
`planning/float_budget_numbers.md` closed that thread as vacuous by user decision on 2026-09-05.
On the statements that say something, §4's port and §3.5 are COMPLETE for all three.

⭐ **Order, set by the user 2026-09-07: the capstones (both done, §4b.6–§4b.7), then 4d piece 2 (done, §4d.2), then MobileNetV4 (which
gets its own planning doc first), then a cleanup/unification session** — the generic constructions
to a `Foundation` leaf, the root-file lemmas, the bf16 `*GradBBf16` lemmas for the other three
folds (§4c-quater), and this document archived behind a short standing one.

⭐ **What §4.2d + §3.5d hand the next kinked net** (which is MNv4, not ViT — ViT has no kink). `r34B_full_has_vjp_at` is a net-agnostic
eighteen-stage apex with two consumers already; `maxPool3s2FlatBackB`, its `rfl` tie and the
`batchMapAux` float lift are ResNet-shaped but generic in every dimension; and the shape of a
kinked net's T6 — generic tie, `pdiv` reading, `*_eq_slots` — is now settled and priced (§5).

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

**Since the 2026-09-06 review, "the net the artifact runs" includes three things the types do not
see and the audit did not name.** (i) The OPTIMIZER FORM: a T3 row is green only when its `den`
lemmas are at the un-fused gradient node every optimizer tail consumes, not at the fused
`θ − lr·g` op that only the SGD-inline render emits (4b). (ii) The RENDERER: the Imagenette and
ImageNet artifacts of a net come from one chain, and the tiers are stated at that chain (4c).
(iii) The BATCH: `N` is a binder in every tier that carries no numeral (T1, T2, T3, T6) — the
batched capstones already take `(N : Nat)`, and the per-example ties never see the batch at all
because it is `pretty`'s argument, outside the AST. It is pinned only where a numeral depends on
the BatchNorm width `N·h·w` (the training-BN cap and the backward budget), and never in a tie.
The user's rule, 2026-09-06: batch size is an INPUT to the proofs; the artifact at 32 or 64 is an
instance.

## 2. Where each net stands (audit of 2026-09-05; ⚠ read §4's state-of-play table with it)

⚠ **This table is the 2026-09-05 audit and is kept as written, because the corrections underneath
it are the record.** Three rows have moved since and each says so below: ResNet-34's and
MobileNetV2's live nets are the batch-BN ones (§4's table is the current one for both), ResNet-50
gained T1/T2/T3/T6, and ViT-Tiny's renderer column closed on 2026-09-07 (§4c-ter). A fresh session
wanting only "what is open" should read the NEXT SESSION table at the top instead.


| net | paper net in Proofs | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|---|
| ResNet-34 | `resnet34Forward_full_pc`, [3,4,6,3], 64 to 512 | ✓ | ✓ | ✓ 110 params ⛔ | ✓ eval and train BN | ✓ | ✓ | full depth at 2 channels; 224 realistic |
| ConvNeXt-T | `convNextForwardTCh`, [3,3,9,3], 96 to 768 | ✓ | ✓ | ✓ 182 params, at the per-example SGD-inline file; ⭐ the FOLD is at the batched nodes too (`ConvNeXtFaithfulPoCGB`, 2026-09-07) and the capstone is the open item | ✓ CAP | ✓ | ✓ | none |
| ViT-Tiny | `vitForwardKV` / `vitBodyKVFlat`, depth 12, D 192, 3 heads | ✓ | ✓ `vitFwdGraphKMHV_faithful` | ✓ 200 params, at the per-example SGD-inline file; ⭐ the FOLD is at the batched nodes too (`ViTFaithfulPoCGB`, 2026-09-07) and the capstone is the open item | ✓ CAP | ⛔ 1.703e399, priced and declined | ✓ `vitInputGradK_eq_vitForwardKV_vjp` | none |
| EfficientNet-B0 | `EfficientNetFullB0.lean`, 16 MBConv | ✓ | ✓ train and eval BN (`EfficientNetFullB0Eval.lean`) | ✓ 262 params | ✓ CAP 2.416e287 at the 16 SE sigmoids, window 1.886e279 honest | ⛔ no number at 16 blocks (9.112e2648; statable, declined) | ✓ `efficientnetInputGradB_full_correct`, through `backward_unique` to the concrete witness | none |
| MobileNetV2 | `MobileNetV2FullPaper.lean`, 17 blocks | ✓ `mobilenetv2_full_has_vjp_at` (`MobileNetV2FullVJP.lean`), shape check `mobilenetv2ForwardPaper_eq_chain` | ✓ train and eval BN (`MobileNetV2FullPaperEval.lean`) | ✓ 158 params ⛔ | ✓ CAP 8.176e16, all 52 BN sites | ⛔ no number at 17 blocks | ✓ `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp` | 17 blocks at toy dims; 2 blocks at 224 |
| ResNet-50 | `resnet50ForwardB_full`, [3,4,6,3] bottlenecks, batch BN, `q` a binder | ✓ 2026-09-06 | ✓ 2026-09-06 | ✓ 161 params, loss a binder | ✗ | ✗ | ✓ 2026-09-07 | none |
| MobileNetV4-Conv-M | none; UIB bodies as `CertLayer` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | none |

⛔ **The T3 counts were the `convBias := true` census and are corrected here (2026-09-06).** Every
render in the suite takes `convBias := false` by default and no writer passes otherwise — each conv
bias is folded into the BatchNorm after it and bound to a `zeroBiasPrelude` zero constant — so the
committed artifacts carry **110** ResNet-34 parameters (not 146) and **158** MobileNetV2 ones (not
210). Three file headers said the larger number, plus `resnet34AdamTrainStepFaithfulB`'s "515
inputs". No theorem moved: every fold is `∀`-quantified over op instances and `bias = 0` is one of
them. ⚠ **Assume any render or tie docstring's census is the `convBias := true` one until the
artifact is counted.**

⚠ **ResNet-34's row is now TWO nets.** The per-example `resnet34Forward_full_pc` above keeps T1,
T2, T4, T5, T6 and T7 — but its train step and renderer were retired by 4c leg 1, so its T3 column
is about bytes that no longer exist. The live ResNet-34 is the batch-BN one
(`resnet34ForwardB_full`), whose T1/T2/T3 landed 2026-09-06 (4.1b–4.1e, 4.2a) and whose **T6
landed 2026-09-07** (4.2d); only T4/T5 — the two vacuous float budgets — are open there. The
same reading applies to MobileNetV2's row. §4's state-of-play table is the one to read for both.

**2b. What each T3 tie is about, against the artifact whose accuracy is quoted (review of
2026-09-06).** Every tie is at the SGD-inline `<net>_train_step.mlir`, rendered at batch 32 with
the fused `*Sgd` ops, from the per-example renderer where one exists. None is at Adam — and the
reference recipes are not all Adam either.

⭐ **Since 4b landed (2026-09-06), the first column has two halves.** The §1 FOLD — every parameter
gradient node `den`s to the certified gradient, `∀ cot` — is now at the un-fused nodes for all five
nets, so it covers the `_adam_`, `_rms_`, `_ema*`, `_dp*` and bf16 artifacts as well. The §1a TIE,
which pins each cotangent to the emitted backward subgraph, is still only at the SGD-inline file.
The table below is the TIE's column; read it that way.

| net | T3 is about | its renderer | quoted ImageNet artifact | its optimizer | layers between them |
|---|---|---|---|---|---|
| ResNet-34 | `resnet34_train_step` (SGD fused, per-example BN) | `ResNet34Render` | `resnet34in_momdp64` | heavy-ball | optimizer form, renderer, BN world, DP, smoothed loss |
| MobileNetV2 | `mobilenetv2_train_step` (SGD fused, per-example BN, 17 blocks) | `MobileNetV2Render` | `mobilenetv2in_rmsdp64` | RMSProp | the same five |
| EfficientNet-B0 | `efficientnet_train_step` (SGD fused `*SgdB`, batch BN) | `EfficientNetRender`, which writes the Adam one too | `efficientnetin_emarmsdp64dropdo` | RMSProp + EMA | optimizer form, DP, smoothed loss; drop/dropout are certified inputs |
| ConvNeXt-T | `convnext_train_step` (SGD fused, per-example index) | ⭐ `ConvNeXtRenderB` for all 35 other artifacts (4c leg 3, 2026-09-07); `ConvNeXtRender` writes this one and nothing else | `convnextin_adamdpwxclipdrop` | AdamW | optimizer form, DP, smoothed loss — the RENDERER axis is closed; clip/wx/drop certified |
| ViT-Tiny | `vit_train_step` (SGD fused, per-example index) | ⭐ `ViTRenderB` for all nineteen other artifacts (4c leg 4, 2026-09-07); `ViTRender` writes this one and nothing else | `vitin_adamdp128x4wxclipdrop` | AdamW, 4× accumulation | optimizer form, DP, smoothed loss — the RENDERER axis is closed |
| ResNet-50 | ✅ `r50_net_tiedB`, all 161 params, `g` a BINDER (§3.5c) | `ResNet50RenderB` only | `resnet50in160_lambaccdp8x64bce` | LAMB, 8× accumulation, BCE, 160 px | DP only — the optimizer, renderer, BN world and loss all met 2026-09-06 |
| MobileNetV4-Conv-M | none | `MobileNetV4RenderB` only | `mnv4in_adamdp64` | AdamW | no tie |

The book names `<net>_adam_train_step.mlir` in every ImageNet chapter and describes the tie
generically (`blueprint/src/content.tex` ≈14579), so nothing it says is false — but the theorems
and the named bytes do not meet. `formalization.yaml`'s headline "every committed train step of
all 12 nets is tied" was an overclaim against ~230 train-step files (corrected 2026-09-06 to "every
net's SGD-inline train step"; 4e carries the disclosure).

Five axes cut across the table and are recorded separately from it:

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
* **Optimizer form — ✅ DONE 2026-09-06 at the FOLD; the ties are not re-pointed.** Section 4b.
* **Renderer — ✅ DONE for all four nets (ResNet-34, MobileNetV2 2026-09-06; ViT-Tiny, ConvNeXt-T
  2026-09-07).** Section 4c, §4c-ter, §4c-quater, and its own thread at
  `planning/renderer_convergence.md`. `KNOWN_SPLIT` is empty; every net is one chain.
* **Data parallelism — ✅ pieces 1 and 2 DONE (2026-09-06, 2026-09-07).** The all-reduce is the
  AST node `allReduceMeanF` in every `*dp*` artifact (byte-identical swap), its `den` is piece 1's
  `dpMean`, and every tie's per-replica statement composes with it (`adamW_at_allReduceMeanF`).
  Piece 3 — the driver — stays prose plus the `*-dp-check` gates. Section 4d.

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

**Order.** ✅ **(a) T1, (b) T2 and (c) T3 all DONE 2026-09-06** — see §3.5b and §3.5c below. (c) T3: `ResNet50FaithfulPoC` /
`ResNet50TiePoC` from the R34 pair; the bottleneck adds one conv per block and nothing new in
kind. (d) T4/T5: probe `r50_eval_chain` (53 BN sites at eval BN, plain relu, so a fold near
1e250 like r34's 1.548e209 plus 17 sites; the training-BN form is a CAP as r34's is) and
`r50_back_chain`; then the two budget files, mirror `Resnet34FloatBudget.lean` /
`Resnet34BackFloatBudget.lean`. ✅ **(e) T6 DONE 2026-09-07 — see §3.5d below.** It mirrored `Resnet34BackCertifiedTieB.lean`, the
BATCHED file (§4.2d), not the per-example `Resnet34BackCertifiedTie.lean` this row originally
named, and cost four declarations.

⭐⭐ **(e) is materially cheaper than this row priced it, as of §4.2d.** R50's stem and head ARE
ResNet-34's (§3.5b), so `cbReluStridedBBack_eq_vjp_backward`, `r34StemBBack_eq_vjp_backward`,
`r34HeadBBack_eq_vjp_backward`, `maxPool3s2FlatBackB_eq_vjp_backward` and the whole
`FloatClose.batchMapAux` lift apply VERBATIM at R50's widths — the same "spent rather than
re-derived" economy T1 got. What is new: three bottleneck backward kinds (`r50IdB` / `r50ProjB` /
`r50DownB`), an eighteen-stage generic apex over `r50Pre0 … r50Pre16`, and
`resnet50ForwardB_full_eq_slots`.

⚠ Two things ResNet-34's file did not face. **`q` is a BINDER**, so every dimension is a `2 * (…)`
nest rather than a literal — `(by norm_num)` closes r34's `0 < 56` and will NOT close `0 < q`, so
thread the hypothesis (§3.5b's trap, on the backward side). And **both projection forms need an
`add_comm`**, as T2's graph faithfulness and T3's tie both did; the identity bottleneck needs none.

⛔ **Do not price B0's extra step** (instantiate the tie at the concrete blocks, then
`backward_unique`): R50's block witnesses are `HasVJPAt`, so §5's new trap applies and the file
stops at opaque blocks plus the shape check. §4.2d measured that wall.

Done when the seven yaml rows exist and `scripts/convention_audit.py` still reports r50 clean.
This is the largest package after MNv4; budget it as the R34 close was, one tier per session.

⚠ Three things the 2026-09-06 review added. The quoted run `resnet50in160_lambaccdp8x64bce` is
LAMB over 8 accumulated micro-batches with BCE-with-logits at 160 px. **The LAMB tail cert and the
BCE cotangent were scoped inside (c) and were instead paid for in advance — both landed 2026-09-06,
§3.5a below**, so (c) is now the tie alone. Accumulation is `momVNextF` at `(μ := akeep)`, already
certified as an op. And run `convention_audit.py` under `.venv/bin/python3` — the system
interpreter has no jaxlib and the script dies importing the reference.

### 3.5a DONE 2026-09-06 — ResNet-50's two prerequisites, paid before the package

Two files, ~2 s each, 15 declarations, all 3-axiom clean, `Certs` 3980 → **3982**. Both are LEAF
modules for 4d.1's reason: their natural homes (`Lamb.lean`, `StableHLO.lean`) carry 315+
downstream modules apiece and a declaration added to either rebuilds the corpus.

**`Codegen/LambTriple.lean` — the LAMB tail.** `lambStep` (the ℝ `(θ', m', v')`, `adamWStep`'s
peer, reusing Adam's two moment recurrences because LAMB's `m` and `v` ARE Adam's),
`lamb_triple_faithful` over the four ops the `.lamb` arm emits, and `lambScale_zero_weight`.

⛔ **The audit's row was wrong about the cause, and this is the correction.** `lambDirF_faithful`
and `lambScaleF_faithful` have said the emitted ops denote `lambDir` and `lambScale` since LAMB
landed — both `rfl`, both at `adamWParamF_faithful`'s bar. `Lamb.lean` carrying only trust-ratio
properties was true and was not the reason. Only the assembly was missing, which made the package
much smaller than §3.5 priced it.

⭐ **The scalar child is a BINDER because the AST makes it one**, and the two shapes the render
emits are corollaries. `lamb_triple_faithful_committed` is at the shipped seed —
`gradSumSqAccF` from `%lzero` over `θ` ALONE, one leaf deep, so the scalar is `gradSumSq θ`. That
single-leaf fold is the entire structural difference from the global-norm clip, whose whole content
is that ONE scalar is shared (`clipFactor_shared` against `lambScale_not_shared`); the two emit
nearly the same lines and differ only in the quantifier.

⭐⭐ **`lamb_triple_faithful_excluded` is the one that says something new.** timm reads
`if weight_decay != 0 or group['always_adapt']:` before computing the trust ratio, so the
`no_weight_decay` group is NOT layer-adapted; the render implements that by SKIPPING the norm op
and passing `%lzero` in. The theorem says the emitted step is then exactly `θ − lr·r` at trust 1.
⚠ `lambTrust_zero_weight` does not already give this at the artifact — it fires at `‖θ‖ = 0`
exactly, i.e. step one, and from step two `‖θ‖/‖r‖` collapses to ~0.01–0.1 against timm's 1.0, so
that lemma could hold while the render was wrong. ✅ Both corollaries were read off the committed
bytes: `resnet50in160_lambaccdp8x64bce_train_step.mlir` has `%v6157 = add %lzero, reduce(sW*sW)`,
and its `wx` twin has `sqrt %lzero` / `compare GT, %lzero` on the BatchNorm γ.

**`Foundation/BceLossCot.lean` — BCE-with-logits' cotangent.** `SmoothedLossCot.lean`'s twin at
RSB-A2/A3's loss: `softplus` and `softplus_hasDerivAt`, `bceLogits`, `bceLogits_grad`, the emitted
three-op graph and its per-row reading.

⭐⭐ **`bceLogits_eq_logSigmoid` is what keeps the gradient theorem from being circular.**
`softplus(z) − t·z` IS `−[t·log σ(z) + (1−t)·log(1 − σ(z))]`, class by class. Without it the file
would define the loss as whatever has the derivative the render emits and then prove it has it —
§5's "a comparison against a re-derivation tests the re-derivation" in its loss-shaped form. The
stable `softplus` spelling is the renderer's own `%loss` block, `max(z,0) + log(1 + e^−|z|)`.

⭐ **No hypothesis on the target**, where `softCE_grad` needs `Σ t = 1` to collapse
`(Σ t)·softmax − t`. BCE is per-class and separable, which is the point under mixup — and the a3
arg string is `ls0.0`, so there is no label smoothing on this path at all, which is why the chain
is three ops against CE's five.

⚠⚠ **The divisor is `B·K`, not `B`, and `bceLossCotGraph_row_committed` pins it** rather than
leaving it a binder nobody instantiated. timm's `BinaryCrossEntropy` is `reduction='mean'` over
`B×C`, not the mean of the per-example sum over classes; at `K = 1000` the two differ by 1000× on
the effective step. ✅ The artifact divides by `dense<64000.0>` = 64 × 1000.

⚠ `pdiv_coordFun` — a scalar function of ONE coordinate lifted to `Vec 1` — is `pdiv_sigmoid`'s
proof at a general `f`, and belongs in `Tensor.lean` for the same reason `pdiv_const_smul` does.
Third leaf-placed foundation lemma in two sessions; they are accumulating and should move together.

**What §3.5 still needs** is unchanged apart from these two: (a) `resnet50ForwardB` at batch BN,
(b) the typed graph, (c) the PoC/Tie pair — now the tie alone — and (d)/(e) the budgets and T6.

### 3.5b DONE 2026-09-06 — ResNet-50's T1 and T2, the net's first net-level tiers

Two files, ~2.5 s to elaborate together, ten declarations, all 3-axiom clean, `Certs` 3982 →
**3984**. `Architectures/ResNet50FullB.lean` (the records, the three batched block forwards and
`resnet50ForwardB_full` in nested-application form) and `ResNet50FullBVJP.lean` (the hypothesis
bundles, the six delegation lemmas, `r50Pre0 … r50Pre16`, `resnet50ForwardB_full_has_vjp_at`,
`_eq_chain` and `_correct`).

⭐⭐ **Pure enumeration, and — unlike ResNet-34's T1 — nothing new one tier down.**
`ResNet50BackB0.lean` already carries all three batched bottleneck forms with their `_at` VJPs and
backward graphs at `bnBatchLA`. And **the stem and the head are ResNet-34's**: `r34StemB` is generic
in `{ic oc}` and `r34HeadB` in `{c nCls}`, so `r34StemB_has_vjp_at` and `r34HeadB_has_vjp` apply
verbatim at R50's widths. r34's own T1 needed a new `Foundation` file (4.1c's `batchMap_has_vjp_at`)
for exactly that stem pool; here the lemma is spent rather than re-derived, and a second `r50StemB`
would have been two writers for one fact.

⭐ **One weight record serves BOTH projection forms.** The stride-1 projection (stage 1 block 0,
64 → 256, the form with no R34 analogue) and the strided one (stages 2/3/4 block 0) have identical
parameter shapes and differ only in which convolutions are strided — a property of the forward, not
of the weights. `R50ProjW` is that record and `r50ProjB` / `r50DownB` are the two forwards over it.
ResNet-34 needed two records for its two block kinds.

⭐⭐ **`q` is a BINDER, and the scoping did not name this.** ResNet-50 ships at TWO resolutions —
`resnet50in_fwd` at 224 = 32·7 and `resnet50in160_fwd` at 160 = 32·5, the second being the net the
quoted 76.66% trains — so one statement covers both. ⚠ Every resolution is written as an explicit
nest of `2 * (…)` rather than `8 * q`: those are equal Nats and NOT definitionally equal terms at a
variable `q`, and each block signature demands its operand at exactly the spelling it names (the
render's own `q1 … q5` comment records the same trap on the emitter side). ✅ Both resolutions are
CHECKED, not asserted: two `example`s at the literal `Vec (N*(3*224*224))` / `(3*160*160)` input
types, plus six `#guard`s on the ladder arithmetic.

⛔ **Three kink clauses per bottleneck, not r34's two** — the two interior relus and the
post-residual outer one; the 1×1 expand has no activation. 48 clauses in 16 bundles.

⛔ **`0 < q` is a real hypothesis, where ResNet-34 needed none.** r34's ladder is at literals so
`0 < 56` closes by `norm_num`; the stem pool's VJP needs its output grid nonempty, and at a variable
`q` that is a binder. At `q = 0` the net is degenerate and the statement says so.

⭐ **The census is 161 updated parameters** and it agrees with `ResNet50RenderB`'s own docstring
("161 θ / 161 m / 161 v"): stem 3 + 12 identity blocks × 9 + 4 projection blocks × 12 + head 2. The
records' conv-bias slots are the `convBias := true` census and are `∀`-quantified, as r34's and
mnv2's are — the trap §5 lists.

⚠ **Two conventions are stated in the file header because nothing checks them here.** v1.5 stride
placement (on the 3×3, not the leading 1×1 — the v1 net compiles, trains, descends and is worth
~0.5 pt of top-1) and symmetric padding at all five stride-2 sites.

✅ **T2 landed in the same session, in the second half of `ResNet50FullB.lean`** — four
per-block-kind graphs at `r50FwdChainB`'s own tokens, their `_faithful` lemmas, and
`resnet50FwdGraphB_full_faithful` by one `rw` each. ⭐ The head graph is ResNet-34's unchanged
(`r34HeadGraphB` is generic in `{c nCls}` and emits the same two tokens).

⚠ **`.addVB`, not `.addV`**, since `ResNet50RenderB` emits the batched add. `den` is identical
(both `fun j => den a j + den b j`, both `rfl`) and `skel` is not.

⚠ **The residual operand order is the RENDER's, and it costs one `add_comm`.** Both projection
blocks emit `addVB(body, projection)` while `residualProj proj body` adds `proj + body`, so
`r50{Proj,Down}GraphB_faithful` close with `congr 1; funext; ring` rather than by `simp` alone —
the commutation sits INSIDE the outer relu's argument, so a top-level `funext j; ring` does not
reach it. Writing the graph in `residualProj`'s order would make `den` close by `rfl` and the
emitted operand order wrong. ⭐ The identity block needs nothing: `addVB(body, x)` IS `residual`'s
order, which is why r34 needed `add_comm` only on its downsample.

⚠ **The bias operands call `biasName false "" c`, the render's own function, not a literal.**
`ResNet50RenderB` has no `convBias` flag at all — its `zb` bakes `false` — so `%zb{c}` is the only
name this net emits, and calling the shared function is what keeps the two from drifting.
⛔ **A finding on r34's side**: `ResNet34FullB.lean` writes `"%sb"` and `"%{p}b1"`, which are the
`convBias := true` names its render does NOT emit by default. That is the graph-operand form of the
census trap — cosmetic (`den` is `∀`-quantified over the bias value) and worth fixing when that
file is next touched. The stem graph is written out here rather than reused for exactly this one
string; the ops, their order and their `den` are `r34StemGraphB`'s.

✅ **T2 was checked against the committed bytes, not asserted.** `verified_mlir/resnet50_fwd.mlir`'s
signature is **162 arguments = `%x` + 161 parameters**, with 12 projection slots — exactly
`R50BWeights`' census — and every SSA name this file writes appears there: `%sW`/`%sg`/`%sbt`,
`%zb64` … `%zb2048`, `%s1b0W1` … `%s4b2bt3`, `%s1b0Wp`/`%gp`/`%btp`, `%Wd`/`%bd`.

### 3.5c DONE 2026-09-06 — ResNet-50's T3, the fold and the tie

`Foundation/ResNet50FaithfulPoCB.lean` (the §1 fold, 6 declarations) and `ResNet50TiePoCB.lean`
(the §1a tie, 894 lines, ~2.6 s), `Certs` 3984 → **3986**, all 3-axiom clean. **ResNet-50 is the
third net whose train-step tie is about the artifact its quoted accuracy comes from.**

⭐⭐ **Zero new op-kind lemmas — 4b's lesson taken to its end.** `ResNet34FaithfulPoCB.lean`'s six
are statements about OP KINDS at full generality in `{N ic oc h w kH kW}`, and the bottleneck's
third convolution is one more instance of the first. So the fold file is an ENUMERATION of the
artifact's op table by block profile: 3 + 12×9 + 4×12 + 2 = **161**, the render's own census. 4b
found five of B0's eight were r34's; here it is six of six.

⛔ **ResNet-50 emits NO conv bias gradient at all**, so every slot this tie states is exercised by
the bytes — **161 of 161**, where r34 states 146 and exercises 110 and MobileNetV2 states 210 and
exercises 158. `ResNet50RenderB` has no `convBias` flag; its `zb` bakes `false`.

⭐⭐ **The loss cotangent is a BINDER, and for this net it HAD to be.** r34's and mnv2's capstones
compute `g` internally from `smoothedLossCotGraph`. ResNet-50 ships BOTH losses — that chain on the
`bce := false` artifacts, and BCE's three-op chain on the `bce := true` ones including
`resnet50in160_lambaccdp8x64bce`. So `r50_net_tiedB` takes `g` as a hypothesis and
`r50_lossCot_is_smoothedCE_grad` / `r50_lossCot_is_bce_grad` instantiate it, neither privileged.
That is 4b's "the head takes `g` as a BINDER" made NECESSARY rather than tidier, and it is what
§3.5a's BCE cotangent was written for. ⭐ The BCE arm needs no hypothesis on the target where the
smoothed one needs its mass to be 1, and its divisor is `N·K`.

⭐ **The stem and head tie bundles are ResNet-34's, reused verbatim** — `r34StemTiedB` is generic in
`{ic oc}` and `r34HeadTiedB` in `{c nCls}`, as the stem and head forwards were in T1. ⚠ r34's stem
bundle carries a conv-bias conjunct this net never emits; one delegation, true at `bias = 0`.

⚠ **Two `add_comm`s, one per projection form**, the same seam T2's graph faithfulness has: the
render emits `addVB(body, projection)` where `residualProj proj body` adds `proj + body`. The
identity block needs none, because `addVB(body, x)` IS `residual`'s order.

⚠ **The cotangent bookkeeping is where this file could be silently wrong.** Each BatchNorm's γ/β
reads the cotangent at THAT BatchNorm's output (`cotN1`/`cotN2`/`cotA`) while its conv reads the one
at the conv's output (`cotC1`/`cotC2`/`cotC3`) — off by one and the gradient type-checks and is
wrong. ⚠⚠ And v1.5 puts `r50DownCotN1`/`r50DownCotC1` at the INPUT grid `2h × 2w`; writing them at
`h × w` typechecks nowhere, which is the one place the shape catches the error for you.

⭐ The capstone carries no smoothness hypothesis and no `0 < ε`; `N`, `q` and `g` are all binders.

### 3.5d DONE 2026-09-07 — ResNet-50's T6; **the net's last statement that says anything**

Two files, `Certs` 3990 → **3992**, four declarations, all 3-axiom clean.
`Float/Resnet50WholeBackFloatBridgeB.lean` (~2 s) names `r50InputGradB` / `r50InputGradBF` and
threads `r50_grad_floatBridgesToB`; `Foundation/Resnet50WholeBackCertifiedTieB.lean` (**~3 s**)
states `r50InputGradB_eq_r34B_full_vjp`, `r50InputGradB_correct` and the shape check
`resnet50ForwardB_full_eq_slots`. Gates as §4.2d's, plus `docstring-checkrefs` at 1668.

⭐⭐ **Four declarations, and the reason is §3.5b's economy collected a second time.**
`resnet50ForwardB_full` is `r34HeadB ∘ [3,4,6,3] bottlenecks ∘ r34StemB` — the stem and head are
literally ResNet-34's functions at R50's widths — so §4.2d supplies **all four endpoint ties**
(`cbReluStridedBBack_eq_vjp_backward`, `r34HeadBBack_eq_vjp_backward`,
`maxPool3s2FlatBackB_eq_vjp_backward` and the `FloatClose.batchMapAux` lift the batched pool
needed) **and the apex itself**. `r34B_full_has_vjp_at` is generic in all nineteen dimensions and
[3,4,6,3] is sixteen blocks for both nets, so ResNet-50 instantiates it directly; a second copy
would have been two writers for one fact. ⚠ It is ResNet-34's only by where it was written — the
third net-agnostic construction now sitting in a net-named file, after `b0OpaqueA*` and
`mnv2OpaqueA*`. They should move to one `Foundation` leaf together.

⚠ **Two things ResNet-34's file did not face, both `q`.** One statement covers `resnet50in_fwd`
(`q = 7`, 224 px) and `resnet50in160_fwd` (`q = 5`, 160 px, the net the quoted 76.66% trains) —
and ⛔ every dimension is therefore an explicit `2 * (…)` nest rather than `8 * q`, since those are
equal Nats and not definitionally equal terms at a variable `q` (§3.5b's trap, on the backward
side now). And ⛔ `0 < q` is a real hypothesis where ResNet-34 needed none: the stem pool's VJP
needs its output grid nonempty, and at a literal 56 that closed by `norm_num`. It also reaches the
float side — `floatBridgesTo_flatConvStride2Back`'s `0 < oc·(2h)·(2w)` is `Nat.mul_pos` twice over
`omega`, not `norm_num`.

⛔ **Three relu clauses per bottleneck** — the two interior ones and the post-residual outer one —
where ResNet-34's basic block has two and EfficientNet's MBConv none. The heaviest kink budget in
the suite, and §3.5b's bundles carry it unchanged; this file adds no hypothesis of its own.

⚠⚠ **~3 s here against ~60 s for ResNet-34's peer, and the 57 s is MEASURED rather than
diagnosed.** `Resnet34BackCertifiedTieB.lean` up to its apex is 2.2 s, and its tie's STATEMENT
elaborates in 2.5 s (replace the proof with `sorry` and the file is 2.5 s), so the whole 57 s is
that file's tie `rfl`. The two ties are the same shape against the same apex; the only structural
difference is that ResNet-34's dimensions are LITERALS the kernel can evaluate and ResNet-50's are
`2 * (…)` nests at a variable `q` that it cannot. Recorded as an observation, not a diagnosis.
▶ If `Resnet34BackCertifiedTieB.lean` ever has to be fast, look there first.

**What ResNet-50 has left: (d)'s two float budgets, and nothing else.** T1, T2, T3 and T6 are all
at the artifact its 76.66% comes from.

### 3.6 MobileNetV4-Conv-M: all six tiers

⛔⛔ **SUPERSEDED 2026-09-07 by `planning/mnv4_proofs_tier.md`, and the "(0) FIRST" row below was
WRONG.** It said the first session must build "the three pieces `MobileNetV4BackB0.lean`'s header
names as missing — the fused stage, the head, the strided body". All three exist in that file
(`mnv4FusedStage_faithful`, `mnv4Head_faithful`, `mnv4UibPreStridedBody_faithful`, all in
`AuditAxioms`); what was stale was the file's OWN `## Scope` paragraph, which this row copied.
Seventh instance of §5's "an audit's named gap can be the wrong one", and the cheapest — the
declaration list was one `grep` away. The rest of this section is kept as the record; the plan,
the order and the traps are in the new doc. ⚠ Two facts the new doc adds that this section did
not know: Conv-M has NO quoted accuracy (Conv-S's 87.36% is a superseded spec; the ImageNet port
has not run), and the Conv-M empirical ties are still OWED (`planning/mnv4_convm_ties_todo.md`).


**What exists — re-read 2026-09-07, and it is more than this row said.** `mnv4Blocks`
(`MobileNetV4RenderB.lean`), the 21-row `UibSpec` table transcribed once from
`jax/MainMobilenetV4.lean` and verified against timm; the batched render, whose `mnv4FwdChainB`
is ALREADY the one traversal `@mnv4_fwd`, its eval twin and the train step all use (so 4c does not
apply to this net and never will); forward tie at 1.4e-6 and gradient tie at 0/147 against JAX; and
`MobileNetV4BackB0.lean` (745 lines), which is more than "the UIB body as one `CertLayer`":

* ⭐⭐ **the four-family collapse, which is the hard half and it is DONE.** ExtraDW / ConvNeXt-like /
  IB / FFN differ only in which of the two depthwise slots is present, and **both slots are
  channel- and shape-preserving**, so an absent one is `CertLayer.id'` in the same slot rather than
  a different composition. One `mnv4UibBody` takes the two slots as arguments. No case split, no
  four proofs — and no dispatch that could silently disagree with the forward's.
* the two depthwise-bn-**relu** stages (stride-1 and strided) — ⚠ MNv4 is plain relu, **not**
  relu6, with MobileNetV2 sitting one file over; `bnReluStage_has_vjp_at` is generic in the inner
  op so `dwbReluB` cost one instantiation and the backward graph's `.selectPos` for mnv2's
  `.selectMid`.
* the four stage `CertLayer`s and the skip block.

⛔ **What `MobileNetV4BackB0.lean`'s own header says is NOT built, and it is T1's critical path:**
the **fused stage** (swish, stage 0), the **head**, and the **strided body assembly** for the three
stride-2 blocks — "the stage is here, the assembly is not". Nothing at the net level.

**So the first session is not "T1" — it is those three pieces**, and they are what makes T1 an
enumeration afterwards. ⭐ Budget them off §4.2d/§3.5d's now-settled shape for a kinked net: MNv4 is
relu throughout, so its whole-net VJP is `HasVJPAt`, its T6 stops at opaque blocks plus an
`*_eq_slots` shape check, and **B0's `backward_unique` step must not be priced** (§5's trap).
⛔ Unlike ResNet-50 there is no free ride on the endpoints: MNv4's stem is a 3×3/s2 conv at the
XLA-`SAME` phase followed by a fused-IB stage, and its head is two convs before GAP — neither is
ResNet-34's, so §4.2d's endpoint ties do not transfer. What DOES transfer is the apex
(`r34B_full_has_vjp_at` is net-agnostic, but ⚠ **at eighteen stages** — MNv4's ladder is stem +
fused + 21 blocks + head, so it needs a wider peer, and that is the fourth consumer arguing for
moving the generic prefixes to one `Foundation` leaf).

⚠ **`mnv4ShapeList` and `VLayer.toSpecs` are two hand-written readings of one layout** and
`mnv4-fwd-smoke` is what pins them — §5's "two lists for one net", already instrumented here.

**Order.** ⭐ **(0) FIRST, and it is not a tier**: the three pieces `MobileNetV4BackB0.lean`'s
header names as missing — the fused stage's VJP and backward graph, the head's, and the strided
UIB body assembled from the stages already there. Everything below is an enumeration once those
exist, and none of it can start without them. Budget one session.

(a) T1: `mnv4ForwardB` from the table: stem 3x3/s2 at the XLA phase (the one
XLA-SAME site; use `flatConvStride2Xla` from this thread, its VJP and float leaves exist), the
fused-IB stage (3x3/s2 conv + 1x1), 21 UIB blocks with the `id'` slots, the two head convs, GAP,
classifier. Batch BN (`bnBatchLA`), relu. The block table must be READ from one place; the
render already folds over `mnv4Blocks`, so the Proofs forward should take the same list and a
`Fin 21 → UibParams` record (the ConvNeXt `Fin k → CnxBlockParamsCh` shape). (b) T2: graph +
faithfulness per family, then chained over the table. (c) T3: PoC pair; the UIB param ops are
the depthwise/1x1/BN ops the other nets already certify, at the four family wirings. (d) T4/T5:
⛔ **do not write these.** They are float budgets and `planning/float_budget_numbers.md` closed
that thread; the probe row below is kept only because a probe is cheap and says what the number
would be (batch BN at 21 blocks with two depthwises each; expect a fold, relu has no clamp so the
window will be r34-sized). (e) T6: `mnv4InputGradB` and the tie — ⭐ **mirror
`Resnet50WholeBackCertifiedTieB.lean`, the newest and smallest of the four** (§3.5d): a float chain
naming the term, then a generic-apex tie with the blocks opaque, its `pdiv` reading, and
`mnv4ForwardB_eq_slots`. `EfficientNetFullWholeBackFloatBridge.lean` is the shape to copy for the
depthwise-heavy float chain specifically.

Done when the yaml rows exist and `mnv4_fwd.mlir`'s pad profile (1 XLA site) is the one the
forward definition names. Largest package remaining; the family collapse is what keeps it to one
block file rather than four, and it is already paid for.

### 3.7 ResNet-34: nothing structural

T1 to T6 are at the paper net. The witness is 2 channels wide at the full depth and at 224
resolution; a full-width witness buys nothing the blueprint needs. What remains for R34 is the
BN-world axis (section 4).

## 4. The BatchNorm-world axis — DECIDED 2026-09-06, port in progress

**The decision is option 1: port both nets' Proofs tiers to batch BN.** Recorded in
`formalization.yaml` 4e, which is a new section — 4d named neither world, which was the disclosed
defect this axis carried.

### State of play (end of 2026-09-06) — START HERE

| | r34 | mnv2 | landed in |
|---|---|---|---|
| shared BN float leaves, both directions | ✅ | ✅ | 4.1 |
| `batchMap` at a point | ✅ | n/a (no stem pool) | 4.1c |
| T1 forward + graph faithfulness (T2) | ✅ | ✅ | 4.1b / **4.2b** |
| T1 whole-net `HasVJPAt` | ✅ | ✅ | 4.1d / **4.2b** |
| T3 §1 fold (`den = certified`, un-fused) | ✅ | ✅ | 4.1e / 4b.4 |
| **T3 §1a tie** | ✅ | ✅ | 4.2a / **4.2c** |
| **T6 certified backward tie** | ✅ | ✅ | **4.2d** |
| T4 / T5 (float budgets, vacuous) | ✗ | ✗ | 4.2 |
| T3 at the un-fused gradient (the optimizer axis) | ✅ (4.1e is already that form) | ✅ | 4b.4 |
| renderer converged (4c) | ✅ leg 1 | ✗ leg 2 | `planning/renderer_convergence.md` |

**BOTH nets' T3 AND T6 at batch BatchNorm are COMPLETE (r34: 4.1b–4.1e + 4.2a + 4.2d;
MobileNetV2: 4.2b + 4b.4 + 4.2c + 4.2d; T3 on 2026-09-06 and T6 on 2026-09-07).** What is left on this axis is T4 and
T5 for both — the two float BUDGETS, which `planning/float_budget_numbers.md` closed as vacuous.
⭐ On the statements that say something, this axis is DONE. ⚠ The tie is at the SINGLE-REPLICA batched step, not at `resnet34in_momdp64`: that
artifact is four replicas with an all-reduce outside the AST (4d). ⭐ The one thing a MobileNetV2
session should read first from 4.2a: the cotangent chain is built from the CERTIFIED block VJPs
(4.1d's `_has_vjp_at` bundles), not derived by hand — `MobileNetV2BackB0.lean` has the same
`*BackBatchedGraph_faithful` lemmas, so the same `rfl` route is available.

⚠ Two standing facts a fresh session should not re-derive. **`N` is the PER-REPLICA batch** — the
data-parallel artifacts all-reduce gradients and no BatchNorm statistic is all-reduced, so
`bnBatchLA`'s reduction width is the per-card batch (64 on the runs with logs, 256 only on the
single-device `mom256` peer, which is a different function and says so in its own header). And
**T1/T2/T3 carry no numerals**, so `N` stays a variable through all of them; it is pinned only at
T4/T5.

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

### 4.1e DONE 2026-09-06 — T3's §1 fold at batch BN; the §1a tie is scoped, not written

`Foundation/ResNet34FaithfulPoCB.lean` (~190 lines, 2.1 s): eight `den = certified` lemmas,
all 3-axiom clean, `Certs` 3971.

⛔ **The finding that shaped the file: r34's batched render emits `*GradB`, not `*SgdB`.** Every
batched ResNet-34 train step — `resnet34_sgd_train_step`, the Adam family, `resnet34in_mom256` and
its data-parallel peers — emits the RAW gradient and hands it to an optimizer tail
(`adamMNextF`/`adamVNextF`, heavy-ball, plain SGD). The fused `θ − lr·∂Loss/∂θ` op only appears
where the optimizer is SGD-inline, which EfficientNet's is and r34's batched one is not. **Every
`den = certified` lemma in the repo before this file is stated at the fused form**, so none of them
applied.

⭐ **The un-fused statement is better, not weaker.** One lemma per op kind certifies every optimizer
variant at once — `sgd`, `mom`, `momdp64`, `adam`, `adamdp128` and the bf16 twins all consume the
same `*GradB` node. It is also the shape ConvNeXt's `psW` carve-out already had to take, there for
a different reason (a hand-written SGD wrap).

⭐ **And it cost nothing.** `StableHLO.lean`'s `*SgdB_eq_grad` family says each fused op IS
`θ − lr·` applied to the un-fused one, all by `rfl`, and its own docstring says it exists to
*"unblock a batched `resnet34_adam_train_step` rendered from `Proofs/` — the blocker was the fusion,
never Adam."* The eight lemmas are `EfficientNetFaithfulPoC.lean`'s proofs with the
`congr 1` / `congrArg (lr * ·)` wrapper peeling dropped.

⚠ **Symmetric padding, and only the certificate says so.** These are `convStridedWeightGradB` /
`convStridedBiasGradB`, whose `den` is `flatConvStride2_*`; B0's peers are the `convStridedXla*`
ops. Identical types, identical emitted shapes.

**§1a — the tie — is scoped and NOT written.** Its inventory, so the next session does not re-derive it:

* ⭐ **Do not mirror `ResNet34ChainClose.lean`.** That file derives per-example block cotangents by
  hand because no whole-block VJP existed when it was written. 4.1d's
  `r34IdB_has_vjp_at` / `r34DownB_has_vjp_at` ARE the certified block backwards, so the batched
  block input-cotangent is `(r34IdB_has_vjp_at …).backward dyOut` — a definition, not a derivation.
  This should make the batched tie materially shorter than the 615-line per-example one.
* The per-parameter cotangents still need spelling, and `ResNet34BackB0.lean` gives the exact chain:
  outer-relu mask (`.selectPos` at `residual (projB ∘ cbReluB) x`), then `projBackBatchedGraph`'s
  BN backward at conv₂'s output, then `cbReluBackBatchedGraph`'s mask + BN backward at conv₁'s.
* Primitives to reuse from `EfficientNetTiePoC.lean` (`Proofs.EnetTiePoC`): `bnBackB`, `cInB`,
  `gapInB`. New: a strided conv input-VJP and the batched 3×3/s2 pool backward.
* Then three tie bundles (identity / downsample / stem), the head's loss fold, and the 146-param
  capstone with the residual fan-in sums at all 16 skip merges.

### 4.2a DONE 2026-09-06 — ResNet-34's T3 §1a tie, and the shared smoothed loss cotangent

Two files, `Certs` 3975 → **3977**, both ~2 s to elaborate, all fifteen declarations 3-axiom clean.
**ResNet-34's T3 at batch BatchNorm is complete.**

* `Foundation/SmoothedLossCot.lean` (196 lines) — the shared, general-target label-smoothed loss
  cotangent: `softCE` (cross-entropy against a target DISTRIBUTION), `softCE_grad`,
  `smoothTarget` / `smoothTarget_sum`, `smoothedCE_grad`, and the emitted six-op chain with its
  `den` and its per-row reading.
* `Foundation/ResNet34TiePoCB.lean` (763 lines) — the tie: the render's cotangent chain node for
  node, the two `_eq_vjp` lemmas, three per-block-type bundles, the head, the capstone
  `r34_net_tiedB`, and the corollary that the threaded loss cotangent is the smoothed CE's gradient.

⭐⭐ **The scoping's biggest call was right, and it paid more than predicted.** "Do not mirror
`ResNet34ChainClose.lean`" — 4.1d's `r34IdB_has_vjp_at` / `r34DownB_has_vjp_at` ARE the certified
block backwards, and `r34{BasicBlock,DownBlock}BackBatchedGraph_faithful` already proves the emitted
seven-node fan-in denotes them. So `r34IdCotIn_eq_vjp` closes by **`rfl`** and `r34DownCotIn_eq_vjp`
by **one `add_comm`**, and the cross-block chain is a composition of certified VJPs rather than a
re-derivation. 763 lines against the per-example file's 615, carrying three more axes (batch BN, the
un-fused gradient, the smoothed general-target loss).

⭐⭐ **The tie has two halves with different hypotheses, and separating them is the structural
find.** The capstone `r34_net_tiedB` carries **no smoothness hypothesis and no `0 < ε`** — the folds
are `∀ cot` statements instantiated at explicitly constructed cotangents, so nothing about relu
kinks or positivity is needed to say every parameter node denotes the certified gradient at the
chain's cotangent. The kink and positivity conditions enter ONLY in the two `_eq_vjp` lemmas, which
say those cotangents ARE the certified whole-net backward. The per-example file conflated the two;
keeping them apart is why the capstone elaborates in 2.4 s.

⭐ **One `rfl`-shaped trick made the chain cheap: write the BN backward as a `den`, not as a
`.backward`.** `bnInB` is `den (bnBatchLABack …)`, and `bnInB_eq_bnBackB` (= `bnBatchLABack_faithful`)
is the **only non-`rfl` step in the whole cotangent chain** — every relu mask, the conv and
strided-conv input-VJPs and the pool backward denote their certified backwards definitionally,
because `den` ignores the name strings. ⭐ `bnInB` takes no `β`, which records that the BatchNorm
input-gradient does not depend on the shift.

⭐ **The head needs nothing.** GAP and dense are smooth and each is `batchMap` of a per-example op,
so `r34HeadB_has_vjp` is GLOBAL — the one place in the net where the certified backward comes with
no hypothesis at all.

⛔ **A third parameter-census correction, same root cause as MobileNetV2's.** `ResNet34TiePoC.lean`
and `ResNet34Render.lean` said 146 parameters, 147 forward inputs and 219 eval inputs. Both r34
renders default to `convBias := false` — the conv biases are gone from the signature and bound to
`zeroBiasPrelude`'s zero constants — so the committed artifacts are **110 / 111 / 183**. 146 is the
`convBias := true` census. Five docstrings corrected; no theorem moves, since every fold is
quantified over op instances and `bias = 0` is one of them. That is now three files caught on this
in one session (`MobileNetV2FaithfulPoCPaper`, `ResNet34TiePoC`, `ResNet34Render`); **assume the
census in any tie header is the `convBias := true` one until the artifact is counted.**

⭐ **`N` stays a binder**, as decided. The capstone takes `(N : Nat)`; batch 32 or 64 is an instance.

⛔ **What the tie does not reach, said in the file header, the yaml (4g) and here.** The all-reduce
in `resnet34in_momdp64` is emitted text outside the AST, so every statement is at the per-replica
gradient node. And the 3×3/s2 pool backward is threaded as the emitted `den` but is NOT identified
with `batchMap_has_vjp_at`'s backward — there is no `maxPool3s2BackB_faithful`, so the stem's
cotangent is the artifact's and not yet provably the certified pool VJP's. That lemma is the one
piece of this package left, and it is small.

⚠ **`unrowB` / `rowB` are a real seam, not a cosmetic one.** The loss chain runs at one ROW per
example (`softmaxRow` needs a row index) and the dense parameter ops at the plain per-example width;
the render writes one SSA name for both because `1 * K = K` as an emitted shape, but `Vec (N*(1*K))`
and `Vec (N*K)` are not definitionally equal at a variable `K`. The casts are explicit and named.

**What the next session gets for free.** The four remaining nets' capstone re-pointing (4b's open
half) now has its prerequisite: `SmoothedLossCot.lean` is shared and general in the target. And
MobileNetV2's batched tie is the same construction — `MobileNetV2BackB0.lean` carries the same
`*BackBatchedGraph_faithful` family, so the `rfl` route to `_eq_vjp` is available there too.

### 4.2b DONE 2026-09-06 — MobileNetV2's T1 and T2 at batch BN

Two files, `Certs` 3977 → **3978**, both ~2 s to elaborate, all sixteen declarations 3-axiom clean.

* `Architectures/MobileNetV2FullB.lean` (362 lines): `MNV2BWeights nCls`, the six batched block
  forwards (`mnv2StemB`, `mnv2NoExpB`, `mnv2ExpOnlyB`, `mnv2ResidB`, `mnv2StridedB`, `mnv2HeadB`),
  `mobilenetv2ForwardB_full` in nested-application form, the six block-kind graphs at the render's
  own tokens, their `_faithful` lemmas, and `mobilenetv2FwdGraphB_full_faithful` — one `rw` per
  block.
* `Architectures/MobileNetV2FullBVJP.lean` (581 lines): the three batched smoothness bundles plus
  the stem's and the head's, the six bundle lemmas, `mnv2PreB0 … mnv2PreB17`,
  `mobilenetv2ForwardB_full_has_vjp_at`, `mobilenetv2ForwardB_full_eq_chain`, and
  `mobilenetv2ForwardB_full_has_vjp_at_correct`.

⭐⭐ **Cheaper than r34's, and the reason is structural.** ResNet-34's 4.1b–4.1d needed a new
`Foundation` file (4.1c's `batchMap_has_vjp_at`) because its stem ends in `batchMap N maxPool3s2Flat`
and a max-pool has no derivative at a tie. **MobileNetV2 has no stem pool** — the stem is
conv-BN-relu6 and downsamples once — and its head's GAP and dense are smooth, so the global
`batchMap_has_vjp` covers every `batchMap` in the net. Nothing new one tier down at all.

⭐ **Three of the six block shapes are one lemma at a different inner op.**
`bnRelu6Stage_has_vjp_at` (`MobileNetV2BackB0.lean`) already takes `(op, hop, hopv)` as parameters,
so the XLA-`SAME` strided stem is the SAME construction as every stride-1 stage. The only
composition written from scratch is b1's `projB ∘ dwbrB`, the `t = 1` block, which has no
`mnv2*BodyB` peer — exactly the per-example file's `ivNoExpW_has_vjp_at` situation.

⭐ **The weight and positivity bundles are reused, not re-declared.** `IVW` / `IVWNoExp` /
`IVPos` / `IVNoExpPos` hold kernels and epsilons and know nothing about which axis the norm
reduces, so the batched net binds the records `MobileNetV2FullPaper.lean` and
`MobileNetV2FullVJP.lean` already define. Only `MNV2BWeights` is new — because it is generic in
`nCls` where `MNV2PaperWeights` is pinned at 10 — and only the smoothness bundles need batched
peers, because a kink condition names its activation and `bnBatchLA` is a different activation from
`bnPerChannelTensor3`. That is §5's "two lists for one net" honoured rather than paid.

⚠ **One existing lemma family had to be generalised, and the scoping did not see it.**
`mnv2BodyB_has_vjp_at`, `mnv2BodyB_differentiableAt`, `mnv2BodyBackBatchedGraph` and its
`_faithful` pinned input and output channels EQUAL — they were written for the residual block,
where they must be. But `b11` (64 → 96) and `b17` (160 → 320) are stride-1 bodies with `ic ≠ oc`,
and the paper ladder has no other home for them. The family now takes `ic`/`oc` separately;
`mnv2DownBodyB` already had that shape, every existing call site (`BackNetFolds.lean`) is at
`ic = oc` and infers it, and **no proof changed**. ⚠ This will matter again at 4.2c, where the
backward graph's `_faithful` is what makes `_eq_vjp` close.

⛔ **Two kink clauses per bottleneck, and they are NOT r34's two.** The expand relu6 and the
depthwise relu6, both INSIDE the body: MobileNetV2's linear bottleneck has no activation after
`project`, so the residual add IS the block output and contributes nothing. ResNet-34's second
clause is the post-residual OUTER relu. 35 relu6 sites (16 blocks × 2, b1's one, the stem's, the
head's), bundled into 19 binders; relu6 is kinked on BOTH sides, so each carries `≠ 0 ∧ ≠ 6`.

⚠ **Unlike r34's, the head is not hypothesis-free.** ResNet-34's head is GAP then dense, both
smooth, so `r34HeadB_has_vjp` is GLOBAL. MobileNetV2 puts a 1×1 conv-BN-relu6 in front of the pool,
so `mnv2HeadB_has_vjp_at` carries the net's 35th kink site.

⚠ **The graph's bias operands are the render's DEFAULT `convBias := false` names** (`%zb{c}`, the
shared zero constant each bias is folded into its BatchNorm and bound to), so the typed graph diffs
against `mobilenetv2_adam_train_step`'s forward half name for name and the census reads 158 rather
than 210. Every graph is `∀`-quantified over the bias VALUE, so it covers the `convBias := true`
render too.

⚠ **`.addVB`, not `.addV`.** `MobileNetV2RenderB` emits the batched add for the identity skip, and
this file's `mnv2ResidGraphB` uses that token. ⛔ `ResNet34FullB.lean`'s `r34IdGraphB` /
`r34DownGraphB` use `.addV` where `ResNet34RenderB` emits `.addVB`; `den` is identical (both are
`fun j => den a j + den b j`, both by `rfl`) so T2 is unaffected, but the two differ under `skel`
and so in the emitted shape annotation. Left alone rather than fixed in this package.

### 4.2c DONE 2026-09-06 — MobileNetV2's T3 §1a tie; **MobileNetV2's T3 is COMPLETE**

`Foundation/MobileNetV2TiePoCB.lean` (1037 lines, ~2.7 s), `Certs` 3978 → **3979**, all twelve
declarations 3-axiom clean. The scoping above was right on every point, and the four bullets it
listed are what the file is.

⭐⭐ **Three of the four `_eq_vjp` lemmas close by `rfl`**, as 4.2a's did — the block cotangents are
`.backward` applications of 4.2b's certified block VJPs, and `MobileNetV2BackB0.lean` already
proves the emitted subgraphs denote them. ⚠ The fourth, the residual one, needs `Eq.trans` rather
than `rw [← h]`: `mnv2ResidB_has_vjp_at` unfolds to `residual_has_vjp_at` at the abbreviation
`mnv2ExpOnlyB`, where `mnv2ResidBlockBackBatchedGraph_faithful` states it at that abbreviation's own
unfolding. Definitionally equal, not syntactically — `refine Eq.trans ?_ h; rfl` is the escape.

⭐⭐ **One tie bundle covers TWELVE of the seventeen blocks**, which the scoping did not predict. A
skip block and a stride-1 widening have the SAME parameter cotangents: the identity skip changes
only the `dx` handed to the previous block, never a parameter's. That is why
`MobileNetV2RenderB.irBackStride1GradB` is one function with a `skip` flag rather than two
near-copies, and `mnv2Stride1TiedB` inherits it — b3, b5, b6, b8–b13, b15, b16 (skip) and b11, b17
(widen). ResNet-34 needed two bundles for sixteen blocks; MobileNetV2 needs three for seventeen.

⭐ **No `add_comm` anywhere.** r34's `r34DownCotIn_eq_vjp` needed one because the render emits
`addVB(body, projection)` and the graph builds `addV(projection, body)`. MobileNetV2's skip emits
`addVB(body, %dy)` and `residualBackGraph` builds the fan-in in the same order.

⭐ **Two helpers are ResNet-34's, imported rather than copied.** `bnInB` (the batched BatchNorm
input-cotangent written as a `den`) and `bnInB_eq_bnBackB` are net-agnostic and live in the file
that first needed them. What this net adds is the TWO-SIDED `relu6MaskB` (r34 threads the
one-sided `reluMaskB`) and `dStridedXlaInB`. ⚠ That last is NOT `EnetTiePoC.dStridedInB`: B0's
strided depthwise is the SYMMETRIC op and MobileNetV2's is the XLA-`SAME` one — identical types,
different certificates, and this is the only place the distinction is recorded on the backward side.

⭐ **The capstone carries no smoothness hypothesis and no `0 < ε`**, exactly as r34's does not: the
folds are `∀ cot` statements at explicitly constructed cotangents. The kink and positivity
conditions live only in the four `*CotIn_eq_vjp` lemmas. `N` is a binder.

⭐ **`Foundation/SmoothedLossCot.lean` applied verbatim** — no loss work at all, which is the return
on having written it at a general target in 4.2a.

⛔ **210 slots, 158 exercised, one replica.** `MobileNetV2RenderB` runs `convBias := false`, so the
52 conv/depthwise/project bias nodes are not emitted; the bias conjuncts are kept because they cost
one delegation each and cover the flag. And every gradient node in `mobilenetv2in_rmsdp64` is
followed by `all_reduce(add)/4` as emitted text outside the AST, so the tie is at the per-replica
node (4d).

⭐ **One thing r34 left open that this net does not have.** 4.2a's residual is the 3×3/s2 pool
backward, threaded as the emitted `den` but not identified with `batchMap_has_vjp_at`'s. MobileNetV2
has no pool, so its stem chain ends at the BatchNorm backward and there is nothing left over.

### 4.2d DONE 2026-09-07 — T6 for BOTH nets; **§4's port is complete on every statement that says something**

Four files, `Certs` 3986 → **3990**, twenty declarations, all 3-axiom clean. Gates: `lake build
Certs`, `lake env lean tests/AuditAxioms.lean`, `lake exe docstring-checkrefs` (1663 citations),
`python3 scripts/check_audit_coverage.py`.

* `Float/Resnet34WholeBackFloatBridgeB.lean` (~1 s) — `r34InputGradB`, `r34InputGradBF` and
  `r34_grad_floatBridgesToB`, plus the two general lifts the batched pool forced.
* `Foundation/Resnet34BackCertifiedTieB.lean` (**~60 s**) — `HasVJPAt.backward_unique`, the four
  endpoint ties, `r34OpaqueA0 … A16` and the generic eighteen-stage apex `r34B_full_has_vjp_at`,
  the tie, its `pdiv` reading, and the shape check `resnet34ForwardB_full_eq_slots`.
* `Float/MobileNetV2WholeBackFloatBridgeB.lean` (~2 s) — the same three for MobileNetV2.
* `Foundation/MobileNetV2WholeBackCertifiedTieB.lean` (~3 s) — two endpoint ties, the tie, its
  `pdiv` reading, and `mobilenetv2ForwardB_full_eq_slots`.

⭐⭐ **MobileNetV2's file defines no apex and no prefix defs at all.**
`mobilenetv2PaperPC_has_vjp_at` — the per-example paper file's twenty-one-stage chain — is
generic in every dimension and every stage, so the batched net instantiates it directly at
`mnv2StemB` / the seventeen batched blocks / `cbrB` / `batchMap gap` / `batchMap dense`, with
`mnv2OpaqueA0 … A17` coming along. ResNet-34 had to write both, because its committed apex bundles
the stem's pool into `stem` and its head into one stage. ⚠ The one cost of reusing a 21-stage apex
for a net whose head is one stage: `mnv2HeadB` unfolds to three, so the shape check meets
`(dns ∘ gap ∘ head) ∘ trunk` against `dns ∘ gap ∘ head ∘ trunk`. ⛔ Letting the kernel discover
that on the concrete net is a deterministic timeout (whnf unfolds the `@[reducible]` block
abbreviations to get there); `comp3_assoc`, proved between VARIABLES, closes it for free. Third
appearance of `Resnet34BackCertifiedTie.lean`'s `chainComp₂_comp` lesson.

⭐ **ResNet-34's batched pool tie is `rfl`, and it closes the one seam §4.2a left open.** That file
threaded the 3×3/s2 pool backward as the emitted `den` and could not identify it with the certified
pool VJP. `maxPool3s2FlatBackB_eq_vjp_backward` does, definitionally — and it is definitional only
because 4.1c built `batchMap_has_vjp_at` field by field rather than transporting it with `▸`, and
`maxPool3s2Flat_has_vjp_at_vec` did the same one tier down. §5's transport trap, paid forward twice
and collected here.

⛔ **The batched pool backward is `batchMapAux`, not `batchMap`, and that is the one new piece of
float machinery.** A pool backward is indexed by the SAVED forward activation and each example has
its own, so a `batchMap` would hand example 0's argmax pattern to the whole batch —
`StableHLO.batchMapAux`'s own header records the same trap on the emitter side. `FloatClose.batchMapAux`
and `FloatBridgesTo.batchMapAux` are the lift; they are `FloatClose.batchMap`'s proof with the
function allowed to depend on the row, and the per-row bridges share one `mag`/`mod` because the
pool's `4` (`maxPool3s2Back_mask_sum_abs_le`) and its rounding width are facts about the WINDOW
GEOMETRY, not about which cell won. ⚠ They belong beside their `batchMap` peers in
`EfficientNetBackFloatBridge.lean` and are stated in the leaf for §5's root-file-lemma reason —
the fourth such lemma parked this way in three sessions.

⛔⛔ **Two walls, both measured, and the second one is the finding.** A `rfl` straight at 4.1d's
tactic-built `resnet34ForwardB_full_has_vjp_at` is a five-minute `(deterministic) timeout at
isDefEq` at four million heartbeats — §5's elaboration trap, and the reason a generic apex exists.
**And instantiating the generic tie at the sixteen CONCRETE blocks is a *kernel* deterministic
timeout at six minutes**, with `backward_unique` or without it. ⭐ **The cause is the KINK, not the
depth and not the net.** B0's generic tie (§3.3(c)) takes GLOBAL `HasVJP` block witnesses, which
carry no point, so B0's extra step — instantiate, then `HasVJP.backward_unique` — costs nothing
there. r34's and mnv2's are `HasVJPAt` at `r34OpaqueA{k-1} … x` while a caller's witnesses are at
`r34Pre{k-1} N w x`: sixteen defeq checks between sixteen-deep nested applications spelled through
two different definition chains. ▶ **So a kinked net's T6 stops where MobileNetV2's per-example T6
stopped — opaque blocks plus a `*_eq_slots` shape check — and B0's "one step further" is not
available to ConvNeXt-T's or ViT-Tiny's peers either, both of which are `HasVJP`.** `HasVJPAt.backward_unique`
is stated anyway: it is `HasVJP.backward_unique`'s pointwise peer, no `HasVJPAt` net in the repo had
it, and a reader instantiating one block at a time needs it.

⚠ **The 60 s is r34's alone and it is the `let` chain, not the net.** MobileNetV2's tie is 3 s over
seventeen blocks where r34's is 60 s over sixteen; the difference is that `r34B_full_has_vjp_at`'s
`rfl` runs against a seventeen-deep `let` chain of `vjp_comp_diff_at`s whose `PProd` projections do
not share, while `mobilenetv2PaperPC_has_vjp_at` was already compiled.

⛔ **No number is stated about either chain, and that is a decision.** §4.2's T4/T5 are float
budgets; `planning/float_budget_numbers.md` closed that thread, and the batched backward is
strictly worse than the per-example `8.857e245` because `bnGradInputReMag`'s gain carries
`Xh² = N·h·w` — one theorem per `N`, and it says nothing. The chains are NAMED so the ties are
about a term, which is `EfficientNetFullWholeBackFloatBridge.lean`'s role for B0.

⚠ **Padding is where the two nets still differ**, and each tie names its own leaf: r34's stem is
`flatConvStride2Back` (symmetric) and MobileNetV2's is `flatConvStride2XlaBack`, whose float
skeleton scatters with `decimateOddBack` rather than `decimateBack`. Identical types, different
certificates — the memory's rule that a float backward net spells its own `decimateBack` and must
move with the bridge.

### 4.2 Still open, per net

For each of ResNet-34 and MobileNetV2, at `bnBatchLA` (r34's T1 and T2 landed, 4.1b–4.1d):

| tier | what it needs | mirror |
|---|---|---|
| T1 | net-level ℝ forward + whole-net `HasVJPAt` (both nets have relu kinks) — ✅ r34 | `EfficientNetFullB0.lean` |
| T2 | typed forward graph, per-block `_faithful` then chained — ✅ r34 | `ResNet34RenderB` / `MobileNetV2RenderB` tokens |
| T3 | FaithfulPoC / TiePoC against the batch-BN train step, at the UN-FUSED gradient (4b's form) — ✅ r34's §1 fold (4.1e); §1a scoped | the existing per-example pair, and `ResNet34FaithfulPoCB.lean` for the gradient form |
| T4 | training-BN forward budget — a **CAP**, as `r34_train_float_logits_le` already is | `Maps.bnBatchTensor4Capped` |
| T5 | backward budget, one theorem per `N` | `Maps.bnBatchBack` |
| T6 | certified backward tie at `bnBatchTensor4` — ✅ **BOTH nets, 4.2d** | `Resnet34BackCertifiedTie.lean` |

⭐ **Two tiers are cheaper than the table suggests.** The block-level batched VJPs and
backward-graph faithfulness already exist for both nets (`ResNet34BackB0.lean`,
`MobileNetV2BackB0.lean`), folded to the paper depth by `BackNetFolds.lean`'s `r34Trunk_3463` —
so T1's hard half is done and T6 composes over it. And the **eval-mode forward budgets are
world-agnostic**: frozen statistics reduce nothing, so `r34_float_logits_le` and
`mnv2_float_logits_le` already hold in both worlds and need only saying so.

⚠ **Every batched number carries the batch size in its statement.** Decide `N` once per net, at
the batch the quoted checkpoint trained at, and put it in the theorem name or the file header —
not in a docstring.

## 4b. The optimizer-form axis — ✅ DONE 2026-09-06: every T3 fold at the un-fused gradient

**The finding.** All five T3 ties are at the fused `*Sgd` / `*SgdB` ops, which only the SGD-inline
render emits. Every other train step in `verified_mlir/` — the `_adam_`, `_mom_`, `_sgd_`, `_rms_`,
`_lamb*`, `_ema*` families and every ImageNet one — emits the RAW gradient (`*Grad` per-example,
`*GradB` batched) and hands it to an optimizer tail. 4.1e found this for r34 and took the un-fused
form; the review found the same fact holds for the other four nets and nobody had re-stated them.
The per-example Adam renders already emit the un-fused constructors (`convWeightGrad`,
`depthwiseWeightGrad`, `veclnGammaGrad`, `layerScaleChGammaGrad` in `ConvNeXtRender`;
`rowDenseWeightGrad`, `patchEmbedWeightGrad`, `posEmbedGrad` in `ViTRender`; the `*GradB` family in
`EfficientNetRender`), so for B0, ConvNeXt and ViT the fold at the gradient IS the fold at
`<net>_adam_train_step.mlir`, today, on the renderer they already have.

**Why it is the right form and not a workaround.** One lemma per op kind certifies every optimizer
variant at once, because they all consume the same gradient node. The fusion is `rfl`: the 29
`*Sgd_eq_grad` / `*SgdB_eq_grad` theorems in `StableHLO.lean` (`weightSgd_eq_grad` …
`posEmbedSgd_eq_grad`, `layerScaleChGammaSgd_eq_grad`, the depthwise, patch-embed, vector-LN,
row-dense and batched conv/BN/dense families) say `den (xSgd …) = θ − lr · den (xGrad …)`
coordinatewise, so each existing fold lemma becomes its gradient peer by dropping the
`congr 1` / `congrArg (lr * ·)` peeling. 4.1e's eight lemmas are the template — they are
`EfficientNetFaithfulPoC.lean`'s proofs with that peeling removed.

**The tails are ALL certified as of 2026-09-06 — the two holes are closed.**
`adamW_triple_faithful` (Adam/AdamW), `mom_pair_faithful` (heavy-ball), `rmsProp_triple_faithful`
(RMSProp), `clipGrad_faithful` / `clipShared_faithful` (global-norm clip; `clipGrad_accum` for the
accumulated form), `dropPathB_faithful` / `dropoutB_faithful`, and now `lamb_triple_faithful`
(§3.5a). EMA is `adamMNextF` at its other reading and gradient accumulation is `momVNextF` at
`(μ := akeep)`, both by the renders' own docstrings. The loss side is `SmoothedLossCot.lean` and,
for RSB-A2/A3, `BceLossCot.lean` (§3.5a).

⛔ **This paragraph said "LAMB has NO faithfulness theorem" and the cause it named was wrong.**
`lambDirF_faithful` and `lambScaleF_faithful` have said the emitted ops denote `lambDir` and
`lambScale` since LAMB landed, both by `rfl` and both at `adamWParamF_faithful`'s bar; `Lamb.lean`
carrying only trust-ratio properties was true and was not the reason. What was missing was the
`(θ', m', v')` ASSEMBLY. ▶ The lesson is §5's own: re-read the file before costing the package. A
row that says "X has no theorem" should name the theorem it looked for.

### 4b.1 – 4b.4 The four folds — ALL LANDED 2026-09-06

| package | file | declarations | elaborates |
|---|---|---|---|
| 4b.1 EfficientNet-B0 | `Architectures/EfficientNetFaithfulPoCG.lean` | 8 | 1.5 s |
| 4b.2 ConvNeXt-T | `Architectures/ConvNeXtFaithfulPoCG.lean` | 14 | 1.6 s |
| 4b.3 ViT-Tiny | `Architectures/ViTFaithfulPoCG.lean` | 10 | 1.6 s |
| 4b.4 MobileNetV2, 17 blocks | `Architectures/MobileNetV2FaithfulPoCPaperG.lean` | 4 op kinds + 5 block-profile capstones | 1.6 s |

Gates: `lake build Certs` 3971 → **3975** green, `lake env lean tests/AuditAxioms.lean` 3-axiom
clean on all 41 declarations, `lake exe docstring-checkrefs` (1585 citations), `python3
scripts/check_audit_coverage.py`. No renderer or `.mlir` change — these are den-level folds.

⭐⭐ **The single biggest finding: op kinds are shared across nets far more than the per-net file
names suggest, and r34's file had already proven most of them.** `ResNet34FaithfulPoCB.lean`'s
eight lemmas are statements about OP KINDS at full generality, not about ResNet-34 — so **five of
B0's eight** (`convWeightGradB`, `bn{Gamma,Beta}GradB`, `dense{Weight,Bias}GradB`) and **eight of
MobileNetV2's twelve** are delegations rather than copies. 4b.1 is three new lemmas, not eight;
4b.4 is four. The per-net files are still the right unit, because each one is the complete op table
for its artifact — but the scoping's "one file each, mechanical" understated how mechanical.

⭐ **ConvNeXt's `psW` was the shape everything else moved towards.** `convStride4WeightGrad` never
had a fused peer, because the SGD path wraps its gradient in hand-written text (the §5 carve-out).
`psWGrad_den` is the one lemma in the four files whose STATEMENT is unchanged from what the SGD
render already needed. The exception became the rule.

⛔ **Correction to 4b.4 as scoped: MobileNetV2's two renderers do not overlap.** The scoping said
"per-example `*Sgd_eq_grad` — or go straight to the batched `*GradB` ops if 4c's MobileNetV2 render
is in hand". The "or" is not optional: `MobileNetV2Render` is SGD-inline only (no `adam` flag
anywhere in it) and `MobileNetV2RenderB` is AdamW-only, so `mobilenetv2_adam_train_step`,
`mobilenetv2_rms_train_step` and every ImageNet artifact have **no fused op to un-fuse**. The file
is at the batched `*GradB` nodes, which makes it also a down-payment on 4c for this net.

⛔ **Two header defects in `MobileNetV2FaithfulPoCPaper.lean`, both fixed there.** The known one:
it named `verified_mlir/mobilenetv2_paper_train_step.mlir`, which is `mnv2TrainStepFaithfulVPaper`'s
`funcName` DEFAULT and no artifact — the one call site (`MobileNetV2Render.lean:788`) passes
`"mobilenetv2_train_step"`. The new one: **the shipped parameter count is 158, not 210.** Both
MobileNetV2 renders default to `convBias := false` (each conv, depthwise and project bias folded
into the BatchNorm after it), and `mobilenetv2_train_step.mlir` returns exactly 158 updated
tensors — stem 3 + b1 6 + 16 × 9 + head 3 + dense 2. 210 is the `convBias := true` census.
Neither touches a theorem: every fold in that file is `∀`-quantified over op instances.

⚠ **Padding is the axis where two nets share a type and not a certificate, and 4b made that
concrete.** r34's and ConvNeXt's strided folds are the symmetric `convStrided*Grad*` ops; B0's stem
and all five of MobileNetV2's stride-2 sites are the XLA-`SAME` `convStridedXla*` /
`depthwiseStridedXla*` ones; and **B0's strided DEPTHWISE is symmetric while MobileNetV2's is not**
(`EfficientNetRender` emits `.depthwiseStrided` on the forward side too). Identical types,
identical emitted shapes, four different certificates.

⚠ **ViT's `clsGrad_den` is at the committed dims, not generic** — the operand's type is
`Vec (1 * D)`, which reduces to `Vec D` only at a literal `D`. That is `ViTTiePoC.vit_cls_den`'s
reason as well, and it is the one place in the four files where a statement is not dimension-
polymorphic.

⚠ **`rowDenseBiasGrad` appears twice in ViT's file against two different certified Jacobians** — a
dense bias and a LayerNorm β are the same reduce. That is not an ambiguity: the tie is what says
which forward a given SSA name's operand came from, and the fused file carries the same pair.

⚠ **MobileNetV2's folds are at BATCH BatchNorm, where every other MobileNetV2 statement in
`Proofs/` is per-example.** Nothing in the file claims otherwise — a `den = certified gradient`
fold is about one op and its FREE cotangent and says nothing about which whole-net forward produced
that cotangent — but a reader will expect the caveat and the file header carries it.

**What was NOT done when this landed, and where it stands now (2026-09-07).** The
`<net>_net_tied_certified` capstones were NOT re-pointed at the gradient nodes when the folds
landed: re-pointing needs the shared smoothed-target loss cotangent (4.2a). Since then: ResNet-34
(4.2a), MobileNetV2 (4.2c) and ResNet-50 (§3.5c) at batch BN, EfficientNet-B0 (4b.5, 2026-09-06)
**ConvNeXt-T (4b.6)** and **ViT-Tiny (4b.7)**, both 2026-09-07, at their batched chains. ⭐ **Five
of five.** Every net's T3 §1a tie is at the un-fused gradient node, the smoothed loss at a general
target and the batched index — the artifact each quoted accuracy comes from, up to one replica.

### 4b.6 DONE 2026-09-07 — ConvNeXt-T's capstone, three axes at once

`Architectures/ConvNeXtTiePoCGB.lean` (~610 lines, **~3 s**, 13 declarations) plus
`smoothedLossCotGraphDiv` / `_den` / `_row` in `Foundation/SmoothedLossCot.lean`; `Certs` 3994 →
**3995**, all 3-axiom clean. `cnx_net_tiedGB (N) {nC} …`: all 182 ConvNeXt-T parameters at the
`*GradB` nodes `convnext_adam_train_step` and every `convnextin_*` artifact emit, at the smoothed
loss at a general target, at the batched index.

⭐ **Three axes, and the third was free.** B0's re-pointing (4b.5) moved two axes because
`EfficientNetTiePoC` was already batched. `ConvNeXtTiePoC` is per-example at a single image, so
the INDEX had to move too — and it cost nothing: every activation is `batchMap N` of the fused
file's per-example prefix (`cnxStemFwdO`, `cnxBlockFwdChO`, `cnxDownFwdChO`, reused verbatim) and
every cotangent is `batchMapAux N` of its chain (`cnxBlockCotInChAt`, `cnxDownCotInChAt`, reused;
three small per-example internal-cotangent helpers added so `batchMapAux` has a function of the
block INPUT to lift). Each conjunct is one `CnxPoCGB.*_den` lemma. ▶ **The lift is the honesty
argument, and it holds for this net and for no BatchNorm net**: every ConvNeXt op is
batch-separable, so the batched op IS the per-example op under `batchMap`, which is what the `*B`
constructors' `den` arms say.

⛔ **The NEXT SESSION row's "ConvNeXt has `*BackBatchedGraph_faithful`, so its `*CotIn_eq_vjp`
closes by `rfl`" was wrong on both counts.** `ConvNeXtBackB0.lean`'s family is per-example
backward-graph faithfulness, and the capstone needed no `_eq_vjp` at all — B0's shape (4b.5)
composes the cotangent chain directly and so does this. Sixth instance of §5's rule, caught by
reading the file rather than by paying for it. ▶ ViT's capstone needs no such family either.

⭐ **The loss chain needed its own spelling.** `smoothedLossCotGraph` is `softmaxRow` at one row per
example, width `N·(1·K)` — hence `rowB`/`unrowB`. ConvNeXt and ViT spell the row softmax as
`batchOp expe` then `batchOp softmaxDiv` at the plain width `N·K`. `smoothedLossCotGraphDiv` is
that AST, `_den` says it denotes the same function, `_row` is `smoothedLossCotGraph_row` at it with
no cast anywhere. Serves ViT unchanged.

⚠ **~3 s against the fused file's 16M-heartbeat budget.** Same 182-parameter thread, same 22-deep
`let` chain; the difference is that every `let` here is a `batchMap`/`batchMapAux` of an
`@[irreducible]` per-example def, so nothing unfolds. Recorded as an observation.

⛔ Stated at the drop-free chain and at ConvNeXt-T's literal widths; one replica (4d).

**Gates.** `lake build Certs` 3995; `lake env lean tests/AuditAxioms.lean` 3-axiom clean on all
seven new prints; `lake exe docstring-checkrefs` 1669; `python3 scripts/check_audit_coverage.py`.
No artifact moved, so no render gate.

**The original scoping, for reference.**

| package | file (mirror `ResNet34FaithfulPoCB.lean`) | ops | `_eq_grad` source |
|---|---|---|---|
| 4b.1 EfficientNet-B0 | `Architectures/EfficientNetFaithfulPoCG.lean` | the eight `*SgdB` kinds `EnetPoC` is at — `conv{Weight,Bias}SgdB`, `convStridedXla{Weight,Bias}SgdB`, `depthwise{,Strided}{Weight,Bias}SgdB`, `bn{Gamma,Beta}SgdB`, `dense{Weight,Bias}SgdB` — to their `*GradB` peers | batched `*SgdB_eq_grad` |
| 4b.2 ConvNeXt-T | `Architectures/ConvNeXtFaithfulPoCG.lean` | `conv{Weight,Bias}Sgd`, `convStrided{Weight,Bias}Sgd`, `depthwise{Weight,Bias}Sgd`, `veclnGammaSgd`, `rowDenseBiasSgd`, `layerScaleChGammaSgd`; `convStride4WeightGrad` is already a gradient (the §5 carve-out becomes the norm) | per-example `*Sgd_eq_grad` |
| 4b.3 ViT-Tiny | `Architectures/ViTFaithfulPoCG.lean` | `rowDense{Weight,Bias}Sgd`, `veclnGammaSgd`, `patchEmbed{Weight,Bias}Sgd`, `posEmbedSgd`, the cls token | per-example `*Sgd_eq_grad` |
| 4b.4 MobileNetV2, 17 blocks | `Architectures/MobileNetV2FaithfulPoCPaperG.lean` | the twelve op types `MobileNetV2FaithfulPoCPaper` tabulates | per-example `*Sgd_eq_grad` — or go straight to the batched `*GradB` ops if 4c's MobileNetV2 render is in hand |

### 4b.7 DONE 2026-09-07 — ViT-Tiny's capstone; the set closes at five of five

`Architectures/ViTTiePoCGB.lean` (~615 lines, **~3 s**, 9 declarations + the `BlkSaves`
packaging); `Certs` 3995 → **3996**, all 3-axiom clean. `vit_net_tiedGB (N) {nC} …`: all 200
ViT-Tiny parameters at the `*GradB` nodes `vit_adam_train_step` and every `vitin_*` artifact emit,
at the smoothed loss at a general target, at the batched index. §4b.6's transformation applied
verbatim — `batchMap N` of `patchEmbed_flat` / `vitBlockFwdOMHV` / the final LN / `clsSliceFlat`,
`batchMapAux N` of `vitCotB2outV` and `vitBlockCotInAtMHV`, conjuncts delegating to
`ViTFaithfulPoCGB`, `g := den (smoothedLossCotGraphDiv …)` reused unchanged (ViT emits the same
chain).

⭐⭐ **The conjunct the per-example capstone could not state is here.** §4c-ter recorded that the
CLS token's gradient sums over the batch INSIDE `den` at the batched node, where the fused file's
`vit_cls_den` is at `denseBiasSgdB (N := 1)`. `vitEmbedTiedGB`'s third conjunct is
`ViTPoCGB.clsGrad_denB` at the real embed-output cotangent — leg 4's one genuinely new statement,
now tied rather than only folded.

⭐ **What it meant to write, and it was the only difference from ConvNeXt.** ViT's per-example
block tie takes its nine saved activations as arguments, and `batchMapAux` lifts a function of one
saved value and one input; so the saves and the eight internal cotangents are repackaged as
functions of the block INPUT (`blkSaves` returning a nine-field structure, `cAtt` … `cM1`) —
`vitBlockTiedAtMHV`'s and `vitBlockCotInAtMHV`'s `let` chains, verbatim. ⚠ One binder trap on the
way: a helper carrying an unused `Wfc2` binder shifted every call's `xin` into its slot; the
elaborator's error named the wrong argument, not the missing use. No `*BackBatchedGraph_faithful`
family, none needed.

⛔ One replica (4d); the 4× accumulation is `momVNextF`'s other reading on top; drop-free chain;
ViT-Tiny's literal dims (S and B are other nets).

**Gates.** `lake build Certs` 3996; `lake env lean tests/AuditAxioms.lean` 3-axiom clean on all
five new prints; `lake exe docstring-checkrefs`; `python3 scripts/check_audit_coverage.py`. No
artifact moved.

## 4c. The renderer axis — ✅ DECIDED and OPENED as its own thread

**The decision (user, 2026-09-06): the Imagenette and ImageNet artifacts of a net come from the
SAME renderer, the batched one, and the tiers are stated there.** The work moved to
**`planning/renderer_convergence.md`** on the day it opened, per this section's own "budget it as a
thread of its own with a planning log". That log carries the per-net legs, the seams outside Lean
and the traps; what follows is the summary this document needs.

**The finding that resized it: ResNet-50 already did this, and it is the template.**
`ResNet50RenderB.r50FwdChainB` exists for exactly this reason — its docstring says *"This exists so
`@resnet50_fwd` and `@resnet50_adam_train_step` cannot be different nets"* — and the fix was ONE
forward traversal with two consumers, not a deletion and not a driver change. Each leg is
"factor the forward out of `<net>RenderB` and render `<net>_fwd` from it", plus the guards.

**The acceptance criterion is mechanical.** `scripts/regen_verified_mlir.sh`'s `check_adam_prefix`
carries a `KNOWN_SPLIT` ratchet that "may shrink, never grow"; a net is done when its entry leaves.
2026-09-06: **5 paired / 2 known-split → 7 paired / 0 known-split** — legs 1 and 2 emptied it.

| net | state |
|---|---|
| **ResNet-34** | ✅ leg 1 DONE 2026-09-06 — `ResNet34Render.lean`, `resnet34_train_step.mlir` and the `resnet34-verified` binary all retired; both forwards on `r34FwdChainB`; ⭐ the ImageNet `resnet34in_fwd` was split too, which `check_adam_prefix`'s Imagenette-only PAIRS list could not see |
| **MobileNetV2** | ✅ leg 2 DONE 2026-09-06 — `MobileNetV2Render.lean`, `mobilenetv2_train_step.mlir`, `mobilenetv2_reduced_train_step.mlir` and the `mobilenetv2-verified` binary all retired; both train forwards on `mnv2FwdChainB`; ⛔ the eval pair deliberately stays on the migrated per-example chain, because a float-budget theorem's provenance names its SSA names; ⚠ §4c's "already has an `sgdParamF` tail" was wrong, and `check_fwd_prefix` lost its mnv2 entry rather than gaining a partner |
| **ConvNeXt-T** | ✅ leg 3 DONE 2026-09-07 — seventeen writers moved to `ConvNeXtRenderB`; four forwards byte-identical, thirteen train steps each moved exactly the 78 conv-VJP lines, by user decision under the keep = 1 numeric licence; `ConvNeXtFaithfulPoCGB` (18 decls, incl. the first bf16-node lemmas) landed FIRST; `convnext_train_step.mlir` stays per-example on purpose, as ViT's does — see §4c-quater |
| **ViT-Tiny** | ✅ leg 4 DONE 2026-09-07 — 19 of 19 drop-free artifacts byte-identical off the batched chain, nineteen writers moved to `ViTRenderB`, ZERO bytes moved and NO file orphaned; ⛔ `vit_train_step.mlir` stays per-example on purpose (no fused-SGD arm, and `ViTTiePoC`'s 200-param tie is at those bytes); ⛔⛔ "a one-line renderer change" was half the price — see §4c-ter |

⚠ **"Every Imagenette number gets re-run" was vacuous for ResNet-34** and must not be assumed for
the rest. `MainResnet34Verified` printed chance (`390/3925`, byte identical every epoch) and said
so in its own header, because running-stat threading lives only in `trainAdamSched`; the real
number came from `resnet34-verified-adam`, already on the batched chain.

⚠ **What retirement costs, per net.** For ResNet-34 it left `ResNet34FaithfulPoC` and
`ResNet34TiePoC` about an artifact that no longer exists — every theorem still true, no committed
bytes exercising it — which is only acceptable because their batched peers (4.1e, 4.2a) landed the
same day. **A leg should not retire a per-example renderer before the batched tier that replaces it
exists.** That ordering is why 4b and 4.2a came first.

### 4c-bis DECIDED 2026-09-07 (leg 4 measured, §4c-ter; leg 3 swapped, §4c-quater) — legs 3 and 4 were not blocked

⛔⛔ **Three statements in this document and one in `planning/renderer_convergence.md` said legs 3–4
were blocked because "the licensing gate is IREE-linked and does not link on this box". All four
were stale.** Measured 2026-09-07:

* **`convnext-adam-tie` builds and links in 2 s** (`lake build convnext-adam-tie`, 23 jobs). Its own
  lakefile docstring records why: it moved to `lowererLink` on 2026-08-12 and **`ireeLink` stopped
  existing on 2026-08-25**. `vit-adam-tie`, `convnext-fwd-b-tie` and `vit-fwd-b-tie` all build too.
* ⭐⭐ **The XLA-side numeric A/B this document says to "build first" ALREADY EXISTS AND ALREADY
  RAN.** `planning/xla_pjrt_handoff.md` §0.10: the keep = 1 stochastic-depth gate compares
  `convnext_adam` (per-example chain) against `convnext_adamdrop` (batched chain, mask ≡ 1.0) and
  measures **0 of 83,478,846 floats differing after 3 AdamW steps**, against a bit-exact floor,
  over a pair that differs by exactly the 78 conv-VJP lines. ⭐ It carries its negative control:
  `scripts/perturb_conv_vjp.py` fires at 0.0343, so the comparison provably reads those lines.
  That file's own verdict is *"That is the §5 license `convnext-adam-tie` could not give"* and
  *"so the blocker is gone"*.

▶ **Third instance of §5's "an audit's named gap can be the wrong one"** — after ViT's `heads = 1`
and LAMB's missing `_faithful` — and the first where the correction was already written down in
another planning file. ▶ The rule earns a clause: **before costing a package on a blocker, grep the
other planning docs for the blocker itself.**

**So what leg 3 actually is: a DECISION, and it is the user's.** `xla_pjrt_handoff.md` §0.10 also
records why the swap was not made: *"Left unswapped deliberately: it would move bytes in the
artifact behind the 84.41% 80-epoch run for no functional gain."* That trade has changed since it
was written, because 4b's last two capstones and 4d piece 2 now sit behind it — the gain is no
longer nothing. The question to put is: **is re-pointing ConvNeXt-T's committed train step at the
batched chain worth moving bytes behind the 84.41% run?** ⚠ Whichever way it goes it should be
recorded here, because the answer also settles `convBack` vs `convBackBatched` — two emitters for
one VJP, never tied to each other, and whichever side moves changes committed artifacts (batched:
R34 / mnv2 / EfficientNet; per-example: ConvNeXt / ViT).

⭐⭐ **And leg 4 may be nearly free, which nothing had noticed.** The 78-line divergence is entirely
`convBack` vs `convBackBatched` — a CONVOLUTION VJP. **`ViTRender.lean` contains zero occurrences
of either** (`ViTRenderB.lean` has one), because ViT's 16×16 patch embed is not a `conv2d`. So
ViT's two chains plausibly differ nowhere at all, and `vit-fwd-b-tie`'s own docstring says as much
for the forward: *"ViT uses one emitter per op, so this is exact byte-identity with no
allowance"*. ▶ **Leg 4's first step is a measurement, not a build**: render `vit_adam_train_step`
off both chains and diff. If it is byte-identical the leg is a one-line renderer change with no
artifact movement and no decision to take — and it unblocks ViT-Tiny's 4b capstone on its own.

### 4c-ter DONE 2026-09-07 — leg 4, ViT-Tiny: nineteen artifacts, zero bytes moved

Two files, `Certs` 3992 → **3993**, ten declarations, all 3-axiom clean.
`Architectures/ViTFaithfulPoCGB.lean` (~2 s) is the §1 fold at the batched constructors; the
nineteen writers moved from `ViTRender.lean` into `ViTRenderB.lean`. Full write-up in
`planning/renderer_convergence.md`; what this document needs is below.

⭐ **The measurement the row asked for, and it is the good answer.** All **19 of 19** drop-free ViT
artifacts — `vit_fwd`, `vitin_fwd` and the seventeen AdamW/EMA train steps — re-render
**byte-identically** off `vitBackAllB`, checked whole-net before a writer moved. `git diff
verified_mlir/` after the swap is empty. The row's reasoning held exactly: the 78-line divergence
that makes ConvNeXt's leg a decision is entirely `convBack` vs `convBackBatched`, and ViT's 16×16
patch embed is not a `conv2d`.

⛔⛔ **But "a one-line renderer change" was HALF the price, and the miss is a new shape of §5's
rule.** **Byte-identity is not tier-identity.** The two traversals emit different CONSTRUCTORS —
`veclnGammaGradB` against `veclnGammaGrad`, `rowDenseWeightGradB` against `rowDenseWeightGrad`, and
so on for all ten — and every `den` lemma is about the AST, not about the bytes. Swapping the
writers alone would have left 4b.3's ten ViT lemmas about an AST that no committed artifact is
`pretty` of: leg 1's orphaning cost, incurred by a change that moves no bytes at all and that no
byte-level gate in the repo can see. So the leg had to land its batched tier FIRST, which is leg 1's
ordering rule arriving from a direction nobody had priced.
▶ **A leg whose artifacts are byte-identical still moves the tier, because a tier is stated about
the graph.** Fourth instance of *"an audit's named gap can be the wrong one"*, and the first where
the gap was invisible to every gate.

⭐⭐ **What the swap BOUGHT, in one parameter.** The CLS token is one shared `[192]` vector, so its
gradient is the sum of every example's CLS-row cotangent. The per-example render emits
`denseBiasGradB (N := 1)` — "sum one thing", correct there because `pretty B` performed the batch
lift OUTSIDE the AST — where the batched one emits `(N := vbB)` and the sum is inside `den`. Same
emitted text either way, so the byte tie provably cannot see it; `den_rowDenseBiasGradB_at_one` is
the general form of the trap, `vit-fwd-b-tie` had been printing it as a standing ⚠ since the batched
chain landed, and `ViTPoCGB.clsGrad_denB` closes it. The gate's text now says so.

⭐ **The fold cost nothing in mathematics.** Each of the ten proofs is `Finset.sum_congr rfl` over
the batch and then the per-example bridge at `batchSlice n`, because every batched `den` arm is
literally the per-example one under a batch sum — the constructors were written that way, and
`ResNet34FaithfulPoCB.denseWGradB_den` is the shape. Ten theorems, ~2 s, one non-obvious step (the
`N = 1` peer's `den` has to be unfolded before it matches the sliced goal).

⛔ **One artifact did NOT move, deliberately, and nothing is orphaned.**
`verified_mlir/vit_train_step.mlir` is the SGD-inline step; `vitBackAllB` has no fused-SGD arm
(`vitBackAll` takes an `adam : Bool`, its batched peer emits the raw gradient only), and ViT's T3
§1a tie — `ViTTiePoC.lean`, all 200 parameters — is stated at exactly those bytes. Its batched peer
is 4b's ViT capstone, which this leg unblocks. So `ViTRender.lean` keeps that one writer and its
traversal. ⭐ **Unlike legs 1 and 2, this leg orphaned no file at all.**

⚠ **The wrappers take the batch in different positions** — `vitAdamTrainStepFaithful fn bStr
replicas bs nClasses …` against `vitAdamTrainStepFaithfulB fn bStr replicas nClasses … (vbB := …)`.
That is the one place a positional slip ships a wrong artifact silently; the per-artifact byte diff
is what catches it, and it was run before the writers moved rather than after.

Gates: `lake build Certs` 3993, `lake env lean tests/AuditAxioms.lean` 3-axiom clean,
`lake exe docstring-checkrefs` (1668), `python3 scripts/check_audit_coverage.py`,
`python3 scripts/check_render_coverage.py`, `bash scripts/regen_verified_mlir.sh` (252 artifacts one
writer each; `check_adam_prefix` 20 paired / 0 known-split / 0 unaccounted; empty artifact diff),
and `.lake/build/bin/vit-fwd-b-tie` byte-identical on all 14,457 lines.


### 4c-quater DONE 2026-09-07 — leg 3, ConvNeXt-T: thirteen artifacts, 78 lines each, by decision

One new Lean file, `Certs` 3993 → **3994**, eighteen declarations, all 3-axiom clean.
`Architectures/ConvNeXtFaithfulPoCGB.lean` (~2 s) is the §1 fold at the batched constructors;
seventeen writers moved from `ConvNeXtRender.lean` into `ConvNeXtRenderB.lean`. Full write-up in
`planning/renderer_convergence.md` (leg 3); what this document needs is below.

⭐ **The measurement, then the decision.** Rendered off the batched chain before a writer moved: the
four drop-free forwards byte-identical, each of the thirteen AdamW/EMA train steps differing on
exactly 78 lines, all the conv-VJP `transpose`/`reverse` pair, line counts equal. The user took the
swap (2026-09-07). After it, `git diff --stat verified_mlir/` is thirteen files at 1014/1014 and
zero non-pair lines. `convnext-adam-tie` on the pre-swap artifact against the committed one: gradient
norm-rel 0.000000 against a 0.000002 reorder control, `%loss` and `v` bit-exact, 0/182 parameters
disagreeing. ⛔ The 84.41% run was not re-run; the function did not move, the bytes did.

⭐⭐ **For this net the fold was OWED before the swap, which §4c's framing missed.** Leg 3 was scoped
as "a decision about the Imagenette pair". But every `convnextin_*` train step, every `*drop*`
variant and the S/B artifacts had rendered from the batched chain since they existed — with a fold
(`ConvNeXtFaithfulPoCG`) at the per-example `convWeightGrad` constructors, where `ConvNeXtRenderB`
emits `convWeightGradB`. So 4b's "one lemma per op kind certifies every optimizer tail" was, for
ConvNeXt, a statement about the Imagenette pair only, and the artifact behind the quoted ImageNet
accuracy had no `den` lemma at its constructors at all. Leg 4's lesson ("byte-identity is not
tier-identity") arrived here as its converse: an artifact that never moved was never covered.
▶ **Fifth instance of §5's rule, and the one where the gap was on the artifact nobody was
looking at.**

⭐ **The bf16 nodes are stated, for the first time.** `convWeightGradBBf16` and its three siblings
are what the bf16 artifacts emit; their `den` is `rnd (Σ_n VJP(rnd x_n, rnd cot_n))`, one rounding
outside the sum. Four lemmas, four lines each. ⚠ The r34, B0 and MobileNetV2 fold headers say their
bf16 twins "consume the same node"; those renders emit the `*Bf16` constructors too, so that
sentence is loose in the same way — batch the same lemmas into the cleanup pass.

⛔ **`convnext_train_step.mlir` stays per-example, for ViT's reason**: the batched traversal has no
fused-SGD arm and `ConvNeXtTiePoC.lean`'s 182-parameter tie is stated at those bytes. Its batched
peer is 4b's ConvNeXt capstone, which this leg unblocks and which is cheaper than ViT's
(`ConvNeXtBackB0.lean` has the `*BackBatchedGraph_faithful` family; ViT does not).

**Gates.** `lake build Certs` 3994; `lake env lean tests/AuditAxioms.lean` 3-axiom clean on all
eighteen; `lake exe docstring-checkrefs` 1668; `check_audit_coverage.py`, `check_render_coverage.py`
(241 files, one writer each); `regen_verified_mlir.sh check` 20 paired / 0 known-split / 0
unaccounted; `convnext-fwd-b-tie` (now per-example against the committed batched bytes) green;
`convnext-adam-tie` old vs new as above.

## 4d. Data parallelism — what is provable, what is calling logic (user question, 2026-09-06)

**As shipped.** Every `*dp*` artifact is ONE program run on `R` replicas. Per parameter, after the
gradient node and before the optimizer tail, `emitGradAllReduce` (`LeanMlir/ViTRender.lean`,
called from every ImageNet renderer) emits `stablehlo.all_reduce(add)` over
`replica_groups = [[0..R-1]]` followed by a divide by `R`. It is emitted TEXT outside the `SHlo`
AST, and `Proofs/Codegen/ViTRender.lean` declares it a trusted carve-out. BatchNorm statistics are
per replica (nothing all-reduces μ/var), which is why `N` in section 4 is the per-replica batch.
The host shards the global batch into `R` slices, broadcasts the initial parameters, and
checkpoints from replica 0 (`VerifiedTrain.lean`, `ffi/pjrt_ffi.c`, `PJRT_REPLICAS`);
`ffi/test_pjrt_allreduce.c` validates the syntax, and the `*-dp-check` gates validate the semantics
numerically — a duplicated batch on `R` replicas must reproduce the single-device step bit-exactly
(ViT's checks all 16.6 M returned floats).

**Provable, in three pieces of increasing cost.**

1. ✅ **The ℝ-level lemma — DONE 2026-09-06.** See §4d.1 below for what landed.
2. **The op-level node (moderate; with 4c).** An `SHlo` constructor `allReduceMeanF R` whose
   `den` is stated over `R` graphs of ONE skeleton — the hypothesis `∀ r, skel (g r) = skel (g 0)`
   IS SPMD, and it is free, because `skel` erases the values the ops carry and `pretty` prints only
   the skeleton — with `den (allReduceMeanF R g) = (1/R) Σ_r den (g r)`. The `pretty` case is
   `emitGradAllReduce`'s text verbatim and the round-trip parser gets one case. That turns the
   carve-out into a faithfulness theorem at the artifact and composes with 4b's gradient folds:
   `den (tail (allReduceMeanF R (convWeightGradB …))) = adamW ((1/R) Σ_r certifiedGrad_r)`.
   ⚠ It needs the values on the `R` graphs to be the replica slices of one host batch, which is
   piece 3.
3. **The calling logic (a page, not a theorem).** Which examples land on which replica, that every
   replica sees the same parameters at step 0, that the checkpoint is read from one replica, and
   that `replica_groups` names all `R` devices — these are `VerifiedTrain.lean` and the FFI, and no
   theorem here reaches them. The user's "worst case" is the right floor and should be written
   regardless: one page in the ImageNet-trainer chapter that says what SPMD data parallelism is in
   this system (one graph, `R` copies, per-replica BN, gradient mean, lockstep), which of that is a
   theorem (pieces 1 and 2) and which is the driver. The `*-dp-check` gate is the empirical
   evidence for piece 3 and belongs on that page.

**Recommendation.** ✅ Piece 1 landed 2026-09-06, piece 2 on 2026-09-07 (§4d.2); the page (piece
3) can be written any time and is not written yet. Every tie against a `*dp*` artifact is still
STATED at the per-replica gradient node, and now composes with the node one `rw` away.

### 4d.1 DONE 2026-09-06 — the ℝ-level lemma, and the negative half as a theorem

`Foundation/DataParallel.lean` (~340 lines, ~2 s to elaborate, 15 declarations, all 3-axiom clean,
`Certs` 3979 → **3980**). Gates: `lake build Certs`, `lake env lean tests/AuditAxioms.lean`,
`lake exe docstring-checkrefs` (1613 citations), `python3 scripts/check_audit_coverage.py`.

What landed, in the order the file reads:

* `lossGrad` / `meanLoss` / `dpMean` — the scalar gradient in the `Vec 1`-lifted spelling `pdiv`
  reads, a mean of losses, and the all-reduced gradient. ⭐ `meanLoss` is ONE definition used at
  two index meanings: over replicas it is the function DP minimises, over examples it is the batch
  mean a single device minimises. That those coincide under no batch coupling is the content.
* `lossGrad_meanLoss` and `dpMeanGrad_eq_grad_meanLoss` — the gradient of a mean is the mean of the
  gradients, hence the collective computes `∇((1/R) Σ_r L_r)`. The only analysis in the file.
* `meanLoss_shard`, `dpMeanGrad_eq_globalBatchGrad_of_perExample`,
  `dpMeanGrad_eq_globalBatchGrad_contiguous` — the no-coupling half.
* `bnToyLoss` / `dpToyShard` / `dpToyBatch` / `dpToyShard_eq_batch` / `lossGrad_smul_coord` /
  `lossGrad_bnToyLoss` / `dpMeanGrad_ne_globalBatchGrad` — the coupled half.
* `dpStep` / `dpSingleStep` / `dpStep_const` / `dpIterate_lockstep` — the induction.
* `dpSingleStep_eq_meanLoss_step` / `dpIterate_eq_meanLossTrain` — the two composed: `n` steps of
  `R` replicas ARE `n` steps of ordinary single-device training on the mean loss.

⭐⭐ **The negative half is a THEOREM, which the scoping only asked for as prose.** The scoping said
that for a training-BN net the DP step is "a different function from the single-device batch-`R·N`
step". `dpMeanGrad_ne_globalBatchGrad` proves it: two replicas, one example each, slices `{0}` and
`{2}` whose union is the global batch `{0,2}`; the DP mean gradient is 2 and the global-batch
gradient is 1. So the batch-BN nets trained data-parallel provably did not descend the
batch-`R·N` loss, at any learning rate and however small the gradients.

⭐⭐ **And the witness needs no BatchNorm at all**, which is the finding worth carrying. `bnToyLoss`
is `(slice mean)² · θ₀` — linear in the parameter so the gradient is a constant, quadratic in the
batch statistic so the coupling bites. ANY nonlinear read of a per-slice statistic separates the
two functions. The split is structural, not a property of the normalisation's formula, and no
all-reduce repairs it because nothing all-reduces μ/var.

⭐ **The no-coupling half is cheaper than scoped, because it needs no derivatives.** `meanLoss_shard`
is an identity between FUNCTIONS — `(1/R) Σ_r (1/N) Σ_n ℓ_{r,n} = (1/(R·N)) Σ_k ℓ_k` — proved by
`Equiv.sum_comp` and `Fintype.sum_prod_type` with `div_mul_div_comm` for the constant. The gradient
statement is that identity under `congrArg`. The scoping put the whole package at "linearity of
`pdiv`"; only `lossGrad_meanLoss` actually is.

⭐ **The shard is a BINDER, and that is a result in itself.** `dpMeanGrad_eq_globalBatchGrad_of_perExample`
takes an arbitrary `e : Fin R × Fin N ≃ Fin (R·N)`, so WHICH examples land on which replica does
not enter — the contiguous cut the DP path makes (`elems / replicas`) and the interleave the
sharded producers make (`ds.shard`) give the same theorem. That answers, for the no-coupling nets,
half of what §4d piece 3 was reserving for the driver: the sharding policy is provably irrelevant,
and only "the union is the batch" matters. ⛔ It does NOT answer it for a batch-BN net, where the
partition changes the function.

⚠ **`pdiv_const_smul` belongs in `Tensor.lean` and is in this file instead.** `Tensor.lean` carries
`pdiv_add`, `pdiv_mul` and `pdiv_finset_sum` but not the scalar-multiple rule, because a constant
is one of `pdiv_mul`'s factors and nobody had needed the specialisation. It is stated in
`DataParallel.lean` with a note, because `Tensor.lean` is the root of the whole corpus and a
declaration added to it rebuilds all 3980 `Certs` jobs. ▶ Move it up the next time `Tensor.lean`
has to change for another reason.

⚠ **`dpIterate_eq_meanLossTrain` asks for differentiability at EVERY point**, not just at the
starting parameters, because the trajectory passes through states the statement cannot name. For a
relu net that is stronger than the truth; the honest weakening is differentiability along the
trajectory and it costs a mutual induction the payoff does not justify. Said in the docstring.

⚠ **What this does NOT do, and every tie still says so.** Nothing here is about the emitted
`all_reduce`. `den (allReduceMeanF R g) = (1/R) Σ_r den (g r)` is piece 2 — an `SHlo` constructor
with a `den`, a `pretty` and a parser case — and it is gated behind 4c like 4b's last two
capstones. Until it lands a tie composes with these lemmas only through the reader. The
`*-dp-check` gates remain the only evidence for piece 3.

### 4d.2 DONE 2026-09-07 — the collective as an AST node; the per-replica disclaimer is composable

`SHlo.allReduceMeanF R hR t ds g` in `Codegen/StableHLO.lean` (constructor, `den` arm,
`den_allReduceMeanF`, `skel` arm, `Raw`/`Tok` cases, `toToks`, `emitTok` via `allReduceMeanText`,
`prettyAllReduceMean`), one `parseStack` case + one induction case in `StableHLOParse.lean`, and
`Foundation/DataParallelNode.lean` (five declarations, ~2 s, 3-axiom clean). `Certs` 3996 →
**3997**.

⭐⭐ **The design, and why `R` graphs of one skeleton is the honest encoding.** Every `SHlo` node
carries its operands' VALUES (`.operand name v`, the saved activations in `convWeightGradB`'s
arguments, …), so "the same program on `R` replicas with each replica's own data" IS a family
`g : Fin R → SHlo n` whose members share a skeleton and differ in values. `den` sums that family
— `(1/R) Σ_r den (g r)`, piece 1's `dpMean` — while `skel` (and hence `pretty`) reads `g 0`,
which is exactly SPMD: `skel` erases values, so any member prints the same program
(`skel_allReduceMeanF_of_spmd`). In a render the family is `.operand grad` at every `r` (renders
are value-independent), so the hypothesis is free there and the sum is what the tie sees.

⭐ **Byte-identity, by carrying the names.** `emitGradAllReduce` named its lines `%arsum{t}` …
`%armean{t}` from the parameter's tag, not from `pretty`'s `%v{k}` counter. The node carries `t`
and its `emitTok` arm is the old body verbatim, so `prettyAllReduceMean` re-renders every committed
`*dp*` artifact byte-identically — MEASURED on all 68 of them before the banners moved. The one
deliberate movement is the banner comment each DP artifact opens with, which used to say the
collective was "a TRUSTED CARVE-OUT, emitted text outside the faithfulness theorems" and now says
it is `pretty(allReduceMeanF)`: comment lines only (measured: every changed line begins `//`), and
`check_fwd_prefix` / `check_adam_prefix` are unaffected.

⭐ **What it composes to.** `adamW_at_allReduceMeanF`: `den (adamW tail (allReduceMeanF R g))` is
`adamWStep` at `dpMean (fun r => den (g r))` — one `rw` of `adamW_triple_faithful`, and the same
line closes the heavy-ball, RMSProp and LAMB tails. `den_allReduceMeanF_convWeightGradB`: the
all-reduced node over `R` replicas' `convWeightGradB` is the replica mean of the certified `Σ_n`
gradients (every other `*GradB` composes by `Finset.sum_congr` and its own fold lemma).
`den_allReduceMeanF_eq_lossGrad_meanLoss`: piece 1 composed — if each replica's node denotes its
loss gradient, the node denotes the gradient of the MEAN loss, the function a DP run minimises.

⚠ **What changed in the trusted surface.** The collective's TEXT moved from a hand-written
function into `emitTok`, i.e. into the same audited per-op `Tok ↔ text` boundary every other op
sits behind; its STRUCTURE is now inside `roundtrip`. What is still trusted is the lowerer's
`all_reduce`, as every op's lowering is. ⚠ Piece 3 (one host batch sharded, same initial
parameters, `replica_groups` names all devices) is the driver's and the `*-dp-check` gates'.

⚠ **Cost.** `StableHLO.lean` is the root of the Codegen cone: the edit rebuilt it (~6 min alone)
and every module below it — the whole corpus, which is why this piece was sequenced after the
capstones rather than interleaved. ⚠ One binder-placement trap: a helper inserted between a
declaration's docstring and its `def` reads as two docstrings and the parser reports "expected
'lemma'" at the second `/--`, a message that names neither.

**Gates.** `lake build Certs` 3997; `lake env lean tests/AuditAxioms.lean` 3-axiom clean on all
seven new prints (`den_allReduceMeanF`, `roundtrip`, the five in `DataParallelNode`);
`lake exe docstring-checkrefs`; `check_audit_coverage.py`; `check_render_coverage.py`;
`regen_verified_mlir.sh check` (20 paired / 0 known-split / 0 unaccounted); `git diff
verified_mlir/` empty before the banner change and comment-only after.

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
* ⛔ **An audit's named gap can be the wrong one, and it happened TWICE.** §3.4 as scoped said
  ViT's block tie was at `heads = 1`; it is general in `h`, and says so in its own header — the real
  gap was the LayerNorm form, which the audit did not mention and which the same net had already
  been caught on twice (`planning/float_budget_numbers.md` §4, row 3). Then §3.5 and 4b both said
  "LAMB has no faithfulness theorem"; `lambDirF_faithful` and `lambScaleF_faithful` had existed
  since LAMB landed and only the `(θ', m', v')` assembly was absent, which made the package a
  fraction of its price. ▶ A row that says "X has no theorem" must NAME the theorem it looked for.
  Re-read the file before costing the package, and cost it against the SHIPPED spelling of every
  convention, not against the one the row names.
  ⛔⛔ **Third instance, 2026-09-07, and it extends the rule to BLOCKERS.** Four rows across two
  planning files said 4c legs 3–4 were blocked because "the licensing gate is IREE-linked and does
  not link on this box". `convnext-adam-tie` links in 2 s, `ireeLink` stopped existing on
  2026-08-25, and the XLA-side A/B those rows say to *build* had already been built AND run, with a
  negative control, and written up in `planning/xla_pjrt_handoff.md` §0.10 — whose own verdict is
  "so the blocker is gone". ▶ **Before costing a package on a blocker, grep the other planning docs
  for the blocker itself.** A stale blocker is worse than a stale gap: it stops work that is free.
  ⛔⛔ **Fourth instance, 2026-09-07, and it is the one no gate could catch.** This document priced
  4c leg 4 as *"a one-line renderer change with no artifact movement"*. The bytes were indeed
  identical — 19 of 19, measured — but the two traversals emit different CONSTRUCTORS, and every
  `den` lemma is about the AST rather than the bytes, so the swap alone would have orphaned 4b.3's
  ten ViT lemmas. ▶ **Byte-identity is not tier-identity: a leg whose artifacts do not move still
  moves the tier.** Before pricing a renderer swap, ask which constructors the new traversal emits,
  not only which bytes.
* ⛔ **A fold header's coverage claim must name the CONSTRUCTOR each variant emits.** Three fold
  headers say "the bf16 twins consume the same node"; the bf16 renders emit `*GradBBf16`
  constructors, whose `den` is `rnd ∘ Σ_n ∘ (VJP at rnd-ed operands)`. ConvNeXt's batched fold is
  the first to state them (§4c-quater, four lemmas); r34's, B0's and MobileNetV2's should gain the
  same. Found by listing what `ConvNeXtRenderB` emits rather than reading what the header claimed —
  the same move that found the ImageNet artifacts had no batched fold at all.
* ⛔⛔ **A KINKED net's whole-net tie stops at opaque blocks, and the reason is the kink.** B0's
  T6 goes one step further than MobileNetV2's — instantiate the generic tie at the concrete blocks,
  then `HasVJP.backward_unique` — and §3.3(c) read that as a lemma B0 had and the others lacked.
  It is not. B0's block witnesses are GLOBAL `HasVJP` and carry no point, so the instantiation is
  free; a `HasVJPAt` net's carry a saved activation, and the caller's spelling of it
  (`r34Pre{k-1} N w x`) is not the tie's (`r34OpaqueA{k-1} … x`), so instantiating is sixteen defeq
  checks between sixteen-deep nested applications and a **kernel** deterministic timeout (measured,
  4.2d). ▶ Budget a kinked net's T6 as "generic tie + `*_eq_slots` shape check" and do not price
  the extra step — and price the whole package at FOUR declarations if the net's stem and head are
  another net's, which §3.5d showed for ResNet-50. B0, ConvNeXt-T and ViT-Tiny are `HasVJP` nets and can take it; r34 and
  MobileNetV2 cannot. (B0's three-block representative stopped one rung short for an unrelated
  reason — a `▸`-transported witness — which §3.3(c) already records.)
* ⚠ **A backward whose saved activation is per example is `batchMapAux`, not `batchMap`.** A
  `batchMap` hands example 0's value to the whole batch — same type, different function, and the
  emitter side already had this trap recorded in `StableHLO.batchMapAux`'s header. The float lift
  is `FloatClose.batchMapAux` (4.2d), `FloatClose.batchMap`'s proof with the function allowed to
  depend on the row.
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
* ⛔ **A tie's row must name its artifact and its optimizer form.** Five T3 rows said "✓ N params"
  and every one was at the SGD-inline file while the book named the Adam one; it went unnoticed
  because the tier table had no artifact column. Section 2b is that column; keep it filled.
* **A file header can name an artifact that does not exist.** `MobileNetV2FaithfulPoCPaper.lean`
  says `mobilenetv2_paper_train_step.mlir`; the 17-block artifact is `mobilenetv2_train_step.mlir`
  (the 6-block one is `mobilenetv2_reduced_train_step.mlir`). Fix it with 4b.4.
* **Emitted text outside the AST is invisible to every `den` lemma.** The all-reduce is the one
  such carve-out in the ImageNet artifacts (4d); `psW`'s hand-written SGD wrap was another and 4b
  makes its un-fused form the norm. Grep a renderer for `s!"    %` literals that are not `pretty`
  before claiming an artifact is `pretty(provenGraph)` end to end.
* ⛔ **A negative witness must prove its two sides are about the same data.**
  `dpMeanGrad_ne_globalBatchGrad` compares the sharded gradient with the global-batch one; without
  `dpToyShard_eq_batch` (the two slices ARE the batch, under the split it claims) it would be a
  true theorem about two unrelated datasets and would establish nothing. Same shape as the
  ONNX-replica lesson: a comparison against a re-derivation tests the re-derivation.
* ⚠ **A lemma that belongs in a ROOT file costs the whole corpus.** `pdiv_const_smul` and
  `pdiv_coordFun` are `Tensor.lean` lemmas by content, `lambStep` and `lambScale_zero_weight` are
  `Lamb.lean`'s, and all four live in leaf modules by economics — `Tensor.lean` has 423 downstream
  modules, `Lamb.lean` and `StableHLO.lean` 315 apiece, and the leaves have none. State them in the
  leaf with a note saying where they belong, and move them up when that file has to change anyway.
  ▶ Measure before deciding: the reverse-dependency count is a ten-line script over the `import`
  graph. Three such lemmas accumulated in two sessions; they should move together.
* **Scripts under the pinned venv.** `convention_audit.py` imports the JAX reference; bare
  `python3` on this box is anaconda's and has no jaxlib.

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
and the most expensive of the three declined backwards); §4 (the batch-BN port) landed 2026-09-06: `Float/BnBatchFloatBridge.lean` (the batched
BatchNorm leaves, both directions, plus the three layout-free `Maps` cores),
`Foundation/BatchMapVJPAt.lean`, `Architectures/ResNet34FullB.lean`,
`Architectures/ResNet34FullBVJP.lean` and `Foundation/ResNet34FaithfulPoCB.lean`; still to
write there are r34's `ResNet34TiePoCB.lean` (4.2a) and MobileNetV2's five peers; 4b
`Architectures/EfficientNetFaithfulPoCG.lean`, `ConvNeXtFaithfulPoCG.lean`, `ViTFaithfulPoCG.lean`,
`MobileNetV2FaithfulPoCPaperG.lean` ALL LANDED 2026-09-06 (⛔ `Foundation/SmoothedLossCot.lean` is
NOT part of 4b after all — it is a prerequisite for re-pointing the capstones, not for the folds,
which are `∀ cot`; it moved to 4.2a and LANDED there, with
`Foundation/ResNet34TiePoCB.lean`); 4c: `Architectures/ViTFaithfulPoCGB.lean` LANDED 2026-09-07 for leg 4 (⛔ the module §4c
originally said the legs needed none — byte-identity is not tier-identity, §4c-ter) and
`Architectures/ConvNeXtFaithfulPoCGB.lean` LANDED 2026-09-07 for leg 3 (§4c-quater; eighteen
declarations, the four bf16 node lemmas the first anywhere), and otherwise
no new Lean module beyond the `.sgd` tail in `MobileNetV2RenderB.lean`; it RETIRED
`ResNet34Render.lean` and `MobileNetV2Render.lean`, moved ViT's nineteen writers onto
`ViTRenderB.lean` and ConvNeXt's seventeen onto `ConvNeXtRenderB.lean` (⛔ neither `ViTRender.lean`
nor `ConvNeXtRender.lean` is retired — each still writes its SGD-inline `*_train_step.mlir`), and
re-points every T2/T3 file at the batched constructors; §4.2d's T6 LANDED 2026-09-07 for both nets: `Float/Resnet34WholeBackFloatBridgeB.lean` + `Foundation/Resnet34BackCertifiedTieB.lean` and `Float/MobileNetV2WholeBackFloatBridgeB.lean` + `Foundation/MobileNetV2WholeBackCertifiedTieB.lean` (⛔ no `*BackFloatBudget` peer for either — T4/T5 are the vacuous half); 4b.6 `Architectures/ConvNeXtTiePoCGB.lean` LANDED 2026-09-07 with `smoothedLossCotGraphDiv` in
`Foundation/SmoothedLossCot.lean`, and 4b.7 `Architectures/ViTTiePoCGB.lean` LANDED the same day on
both shapes — the five capstones are all re-pointed; 4d `Foundation/DataParallel.lean` LANDED 2026-09-06 (piece 1) and the
`allReduceMeanF` constructor in `StableHLO.lean` with its `den`, `pretty` and parser cases plus
`Foundation/DataParallelNode.lean` LANDED 2026-09-07 (piece 2, §4d.2);
3.5a `Codegen/LambTriple.lean` and
`Foundation/BceLossCot.lean` both LANDED 2026-09-06, as did 3.5(a)+(b)'s
`Architectures/ResNet50FullB.lean` (forward AND graph) + `ResNet50FullBVJP.lean` (§3.5b); still to
write for 3.5 are `Resnet50FloatBudget.lean`,
`Resnet50BackFloatBudget.lean`, `Resnet50WholeBackCertifiedTie.lean`; 3.6 the same six for
MobileNetV4. Every new file: a `lakefile.lean` `Certs` root or an import of one, an
`AuditAxioms.lean` block, a `formalization.yaml` row with the convention comment, a
`planning/float_budget_numbers.md` section-1 row for each number.

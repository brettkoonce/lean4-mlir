# book_theorems_pass.md — the certificate theorems read like the maintainers' index

Started 2026-10-01 from the user's read of the current book: "the theorems and copy sort of
bleed together a little now / i'm fine with that pattern (Theorem 13 is an example) ... in other
spots it feels like a lab notebook / just hitting people with random technical facts (eg The
twins matter)". This is the audit of all 163 numbered blocks in `blueprint/src/content.tex`
(128 theorems, 35 definitions) and the plan for the pass.

## 0. Diagnosis

The book has two theorem registers and only one of them was designed.

**Register A — the calculus (about 70 blocks).** Chapter 1's `pdiv` rules and VJP records, the
dense/softmax Jacobians, Chapter 4's seven BatchNorm lemmas, Chapter 9's matrix-level machinery
and attention proofs. These are the Lamport design the Foundations paragraph announces: an
`\Assume{}` list with the Lean hypothesis name in brackets, one `\Prove{}` formula, a numbered
proof whose every step carries its own `\pf{}`, `\Qed{}` last. 55 of them have the numbered
proof. Theorem 13 (Identity VJP) is a one-liner whose statement carries its own reading ("the
identity passes the upstream gradient through unchanged") — the bleed the user is fine with.
Nothing here needs touching.

**Register B — the certificates (56 blocks, ~900 lines).** Every chapter's whole-net VJP, "the
rendered backward is the VJP", "emitted step is the certified step", "every gradient node is the
loss's derivative", the seals, the data-parallel ties, "SGD descends". These were never given a
form. They accreted: WP2 (loss gradients, 09-30) added a sentence per capstone, WP5 (capstone
reach) another, WP6 (honesty: "statements say what they prove") another, WP8/WP9 and the
only-mentions triage (10-01: "cite what is real") added the corollary names. Each pass was right
on its own terms and the sum is the lab notebook:

* hypotheses as run-on prose after `\Assume{}` (21 blocks are `\Assume{}` with no list; 23
  have the list);
* Lean identifiers in the statement body — 162 `\texttt{}` names across the 56 blocks, up to
  nine in one statement (`thm:efficientnetFullHasVJP`), where register A has zero outside the
  brackets;
* corollaries cited by name as sentences ("`r34_net_lossGrad_smoothedCE` discharges the loss
  hypothesis for …", "`mnv4_lossCot_is_smoothedCE_grad` instantiates it at …"): 14 of them,
  every one a row of the only-mentions triage that landed in the book because the book was the
  cheapest place to cite a declaration. The `\lean{}` header already lists those names and the
  web view links them; the PDF never printed the header, so the sentences were the only way the
  names reached the PDF — and the PDF reader is the one they help least;
* process facts in statements: "because instantiating the tie at the concrete blocks times out
  the kernel" (`thm:resnet50_whole_back`), "real batches meet this stem clause, the
  input-gradient one they do not" (`thm:resnet34_loss_grad`), "the loss of the 76.66% … run"
  (`thm:resnet50_step_tie`), "not necessarily the one the denotation names", "the committed
  steps are bf16" (`thm:mobilenetv4FullHasVJP`), "unlike the other nets' forward blocks it is
  not tied to this graph by a text check" (`thm:vitFwdGraphKMHV_faithful`);
* the same ladder re-explained in every chapter: "the chain cotangents it threads are the
  certified block backwards (…); the ε > 0 and smooth-point hypotheses enter there, not in the
  tie" appears in four step ties with four different name lists;
* proofs that are "See `\leandocref{…}`" (38 in the PDF) with no sentence saying what kind of
  argument the Lean is — next to register A's numbered proofs the contrast reads as "we did not
  write this one";
* commentary with no home. Chapters 2–9 put all teaching prose in "Run it first" and stack the
  theorems with no sentence between them (148 of 163 blocks start within two lines of the
  previous block). So when a hypothesis needed explaining, the explanation went into the
  statement, or — "The twins matter." — into a paragraph between the theorem and its proof. That
  paragraph is the Lean docstring of `cnn_conv2_sgd_descends` (`Training/SgdDescent/Cnn.lean`)
  copied into the book, including the probe script's finding and a concrete-witness name.

Register A is what the Lamport paper bought. Register B is what the gates bought. The pass is to
give register B the form register A has, and to give explanations a place to go.

## 1. The template for a certificate theorem

One form, applied to all 56. The user's Theorem-13 pattern (a gloss inside the statement) is
kept as rule 4.

1. **Title**: the claim in plain words, as now.
2. **`\Assume{}`** is an `enumerate`, one hypothesis per item, the Lean bundle or hypothesis
   name in `\hfill[\texttt{…}]` — register A's form. A hypothesis that needs a definition gets it
   in the item, in words, once ("up to twins: cells that read the same input patch, so are equal
   at every kernel"). Bundle names (`R34PosB`, `R34LossSmoothAtB`, `CnnLossSmoothAt`) live in the
   bracket, never in the sentence. No `\Assume{}` at all when there is nothing to assume.
3. **`\Prove{}`** is one sentence or one displayed formula. It names the forward function once
   (`\texttt{resnet34ForwardBFull}`) and, when the theorem is about an artifact, the artifact
   family once. Counts that are the statement (110 gradients, 146 slots, twenty-two updates)
   stay; counts that are a different theorem's go there.
4. **One gloss sentence**, at most, after `\Prove{}`: what the statement means or what it
   covers ("Instantiated at the two losses the artifacts train", "The same holds at the first
   convolution"). Corollaries are described, not named: the names are in `\lean{}`.
5. **Nothing else in the statement.** Process facts (kernel timeouts, what real batches do or do
   not meet, bf16 vs f32 of the committed steps, text-check coverage, run accuracies) go to the
   Lean docstring (most are already there) or to the section that reports the run. A caveat
   that changes what a theorem claims (the strict pool clause real data fails) is stated once,
   on the theorem it caveats, as one sentence — not on a neighbour.
6. **Proof**: register A's numbered proof where the argument has steps (keep all 55). For a
   kernel-checked tie or fold, one sentence naming the kind of argument, then the
   `\leandocref`: "A fold over the sixteen blocks, each step that block's VJP at its chain
   cotangent; see …". Never a bare "See …".
7. **No prose between a statement and its proof.** Commentary that explains a hypothesis goes
   into the hypothesis (rule 2). Commentary that introduces a group of theorems goes before the
   group as a lead-in, one short paragraph, in the chapter's "Run it first" voice.
8. **Witnesses are witnesses.** A concrete instance (`trained_cnn_conv2_sgd_descends_concrete`,
   the seals' `sealX`) is a gloss clause — "Witness: every hypothesis holds at a trained network
   and a real test image" — with the name in `\lean{}`, matching the front matter's
   contracts / witnesses / step-certificates taxonomy.

The only-mentions gate (`scripts/gates/audit_only_mentions.py`) reads whole-file text with `\_`
unescaped, so a name moved from a body sentence into `\lean{}` still counts as cited. Checked
2026-10-01; the report should not grow.

## 2. The ladder, said once

Chapters 5–9 prove the same rungs for each net: (i) the whole net at batch N has the VJP; (ii)
the hand-written backward chain is that VJP; (iii) the emitted step's gradient nodes are the
certified batch gradients at the chain cotangent; (iv) each node is the loss's derivative in its
parameter; (v) the backward is not the zero map (seal); (vi) the data-parallel render's mean
over replicas is one device's step on the global batch. The front matter already names (iii)–(vi)
as step certificates. Chapter 5 is where the full ladder first appears: its theorems section
opens with one paragraph naming the six rungs and the two facts every later chapter now repeats
per theorem — the hypotheses enter at the block backwards, not at the tie; the loss cotangent is
a binder, instantiated at the losses the artifacts train. Chapters 6–9 open "The same rungs for
MobileNetV2 and V4" and each statement is its Assume/Prove alone. Chapters 1–4 have the small-net
rungs (fold, loss gradient, descent) and get a two-sentence version in Chapter 1.

## 3. Two exemplars

### 3a. `thm:cnn_sgd_descends` and "The twins matter" (Chapter 3, lines 3428–3457)

Now: a run-on hypothesis sentence, a corollary named in the body, then a nine-line paragraph
between the statement and the proof that explains the twins clause, narrates why a stricter
margin was abandoned ("would never hold on the data"), and names the concrete witness.

Proposed (hypothesis names from `Training/SgdDescent/Cnn.lean`):

```latex
\begin{theorem}[SGD descends through the convolutions]
  \label{thm:cnn_sgd_descends}
  \lean{Proofs.cnn_conv2_sgd_descends, Proofs.cnn_conv1_sgd_descends,
        Proofs.cnn_conv2_float_sgd_descends,
        Proofs.trained_cnn_conv2_sgd_descends_concrete}
  \leanok
  \uses{...}
  \Assume{}
  \begin{enumerate}
    \item every pixel is bounded, \(|x_i| \le a\) \hfill[\texttt{hx}]
    \item a margin at every ReLU after \(\mathrm{conv}_2\) and in the dense head
          \hfill[\texttt{hm2}, \texttt{hm3}, \texttt{hm4}]
    \item in every live pooling window the maximum clears the other cells by a
          margin, up to twins: cells that read the same input patch, so are
          equal at every kernel --- a blank corner of an MNIST image is one
          \hfill[\texttt{hmq}, \texttt{hT}]
    \item an \(\eta\)-accurate gradient oracle, and the small-step and
          dominance conditions of Theorem~\ref{thm:linear_sgd_descends}
          \hfill[\texttt{hgh}, \texttt{hsmall}, \texttt{h1}, \texttt{h2}]
  \end{enumerate}
  \Prove{} one SGD step on the second convolution's kernel decreases the
  cross-entropy loss:
  \[
    L(W_2 - \alpha g) \le L(W_2) - \alpha\,\|\nabla L\|_2^2/2 .
  \]
  The same holds at the first convolution, and with the binary32 gradient in
  place of \(g\), its \(\eta\) the rounding budget of \S\ref{sec:precision}
  and its pool margin strict. Witness: every hypothesis holds at a trained
  network and a real test image.
\end{theorem}

\begin{proof}
\leanok
Within the step a tied window stays tied and a strict one stays strict, so
the pooled ReLU is a fixed gather and the descent lemma of
Theorem~\ref{thm:linear_sgd_descends} runs on the linear chain through it;
see \leandocref{Proofs.cnn\_conv2\_sgd\_descends}.
\end{proof}
```

The twins paragraph is gone: its one reader-facing fact (what a twin is and where one comes
from) is item 3; the rest is the docstring, where it already is. The same clause fixes
`thm:cnn_loss_grad`, `thm:cifar_loss_grad` and `thm:cifar8_loss_grad`, whose `\Assume{}` prose
spells the twins condition out in a different wording each time.

### 3b. `thm:resnet34_loss_grad` (Chapter 5, lines 5539–5575)

Now: 30 lines; six Lean names in the body; two corollaries cited as sentences; a probe finding
("real batches meet this stem clause, the input-gradient one they do not") that caveats a
different theorem; a `select_and_scatter` routing remark that is the docstring's.

Proposed:

```latex
\begin{theorem}[Every ResNet-34 gradient node is the loss's derivative]
  \label{thm:resnet34_loss_grad}
  \lean{... the same five names ...}
  \leanok
  \uses{...}
  \Assume{}
  \begin{enumerate}
    \item every BatchNorm's \(\varepsilon > 0\) \hfill[\texttt{R34PosB}]
    \item at the batch's activations, every ReLU is off its kink and every
          stem-pool window is entirely zero or has one maximal cell, up to
          twins \hfill[\texttt{R34LossSmoothAtB}]
    \item \(L\) is any loss of the logits, with gradient \(g\) there
          \hfill[\texttt{hL}]
  \end{enumerate}
  \Prove{} for each parameter slot of Theorem~\ref{thm:resnet34_step_tie},
  the gradient node at the chain cotangent from \(g\) is
  \(\partial L/\partial\theta\) of \texttt{resnet34ForwardBFull} with that one
  parameter varied, at any batch size and number of classes. Instantiated at
  the label-smoothed cross-entropy, and at the stem with each tied window's
  cotangent routed to any of its maximal cells, as the emitted
  \texttt{select\_and\_scatter} routes it.
\end{theorem}

\begin{proof}
\leanok
Each node is the step tie's node; its chain cotangent is \(g\) pulled back
through the certified block backwards, one application of
Theorem~\ref{thm:hasGradAt_comp} per stage; see
\leandocref{Proofs.ResNet34TieB.r34\_net\_lossGrad}.
\end{proof}
```

Decisions this one raises (user):

* "146 parameter slots (the 110 the artifacts emit among them)": the step tie says 110, the
  loss gradient 146. One number in the book; the 146-vs-110 accounting is the docstring's.
  Proposed: the step tie keeps 110 and the loss gradient says "each slot of Theorem X".
* "real batches meet this stem clause, the input-gradient one they do not": a caveat on
  `thm:resnet34FullHasVJP`, not on this theorem. Proposed: one sentence on
  `thm:resnet34FullHasVJP` ("a condition real batches do not meet; the parameter gradients
  below are stated under one they do"), and nothing here. Same for ResNet-50.

## 4. Worklist: the 56 register-B blocks

Columns: form = `\Assume{}` present / `enumerate` present; ids = Lean names in the statement
body outside brackets; cites = corollaries named as sentences; proc = process phrases. Every row
gets rules 1–8; the ids/cites/proc columns say how much leaves the statement.

| ch | label | line | lines | form | ids | cites | proc |
|---|---|---|---|---|---|---|---|
| 1 | `thm:linear_fold` | 1597 | 20 | -- | 3 | 0 | 0 |
| 1 | `thm:linear_loss_grad` | 1630 | 11 | A- | 2 | 1 | 0 |
| 1 | `thm:linear_sgd_descends` | 1649 | 24 | AE | 1 | 0 | 0 |
| 2 | `thm:mlpHasVJPAt` | 2428 | 22 | AE | 5 | 0 | 0 |
| 2 | `thm:mlp_fold` | 2459 | 11 | -- | 1 | 0 | 0 |
| 2 | `thm:mlp_loss_grad` | 2478 | 13 | A- | 2 | 1 | 0 |
| 2 | `thm:mlp_sgd_descends` | 2499 | 13 | -- | 2 | 1 | 0 |
| 3 | `thm:mnistCnnNoBnHasVJPAt` | 3342 | 21 | AE | 4 | 0 | 0 |
| 3 | `thm:cnn_fold` | 3371 | 13 | -- | 1 | 0 | 0 |
| 3 | `thm:cnn_loss_grad` | 3392 | 17 | A- | 3 | 1 | 0 |
| 3 | `thm:cnn_sgd_descends` | 3428 | 14 | -- | 1 | 1 | 0 |
| 4 | `thm:cifar_fold` | 4394 | 10 | -- | 2 | 0 | 0 |
| 4 | `thm:cifar_loss_grad` | 4412 | 16 | A- | 2 | 0 | 0 |
| 4 | `thm:cifar_bn_fold` | 4438 | 9 | -- | 0 | 0 | 0 |
| 4 | `thm:cifar8_step_tie` | 4455 | 11 | -- | 2 | 0 | 0 |
| 4 | `thm:cifar8bn_step_tie` | 4474 | 13 | -- | 2 | 0 | 0 |
| 4 | `thm:cifar8_step_tieG` | 4495 | 13 | -- | 2 | 0 | 0 |
| 4 | `thm:cifar8_loss_grad` | 4517 | 19 | A- | 1 | 0 | 0 |
| 4 | `thm:cifar8_sgd_descends` | 4545 | 9 | -- | 1 | 0 | 0 |
| 5 | `thm:resnet34FullHasVJP` | 5432 | 16 | A- | 2 | 0 | 0 |
| 5 | `thm:resnet50FullHasVJP` | 5456 | 13 | -- | 2 | 0 | 0 |
| 5 | `thm:resnet50_whole_back` | 5477 | 30 | AE | 8 | 0 | 1 |
| 5 | `thm:resnet34_step_tie` | 5515 | 16 | -- | 5 | 0 | 0 |
| 5 | `thm:resnet34_loss_grad` | 5539 | 30 | A- | 6 | 2 | 3 |
| 5 | `thm:resnet50_step_tie` | 5577 | 16 | -- | 5 | 0 | 1 |
| 5 | `thm:resnet50_loss_grad` | 5601 | 25 | A- | 6 | 1 | 0 |
| 5 | `thm:resnet34_seal` | 5634 | 11 | -- | 2 | 0 | 0 |
| 5 | `thm:resnet50_seal` | 5653 | 9 | -- | 0 | 0 | 0 |
| 5 | `thm:resnet34_sync_tie` | 5670 | 13 | -- | 1 | 0 | 0 |
| 5 | `thm:resnet50_sync_tie` | 5691 | 10 | -- | 0 | 0 | 0 |
| 6 | `thm:mobilenetv2FullHasVJP` | 7304 | 15 | A- | 2 | 0 | 0 |
| 6 | `thm:mobilenetv4FullHasVJP` | 7327 | 19 | -- | 4 | 0 | 1 |
| 6 | `thm:mobilenetv2_whole_back` | 7354 | 14 | -- | 2 | 0 | 0 |
| 6 | `thm:mobilenetv4_whole_back` | 7376 | 9 | -- | 1 | 0 | 0 |
| 6 | `thm:mobilenetv2_step_tie` | 7393 | 19 | -- | 6 | 1 | 0 |
| 6 | `thm:mobilenetv2_loss_grad` | 7420 | 22 | A- | 4 | 1 | 0 |
| 6 | `thm:mobilenetv4_step_tie` | 7450 | 19 | -- | 7 | 1 | 0 |
| 6 | `thm:mobilenetv4_loss_grad` | 7477 | 20 | A- | 3 | 1 | 0 |
| 6 | `thm:mobilenetv2_seal` | 7505 | 10 | -- | 0 | 0 | 0 |
| 6 | `thm:mobilenetv4_seal` | 7523 | 9 | -- | 0 | 0 | 0 |
| 6 | `thm:mobilenetv2_sync_tie` | 7540 | 13 | -- | 1 | 0 | 0 |
| 6 | `thm:mobilenetv4_sync_tie` | 7561 | 12 | -- | 0 | 0 | 0 |
| 7 | `thm:efficientnetFullHasVJP` | 8884 | 24 | A- | 7 | 0 | 0 |
| 7 | `thm:efficientnet_whole_back` | 8916 | 13 | -- | 2 | 0 | 0 |
| 7 | `thm:efficientnet_step_tie` | 8937 | 20 | -- | 4 | 0 | 0 |
| 7 | `thm:efficientnet_loss_grad` | 8965 | 21 | A- | 3 | 1 | 0 |
| 7 | `thm:efficientnet_sync_tie` | 8994 | 13 | -- | 0 | 0 | 0 |
| 8 | `thm:convnext_whole_back` | 10232 | 18 | A- | 3 | 0 | 0 |
| 8 | `thm:convnext_step_tie` | 10258 | 21 | -- | 7 | 0 | 0 |
| 8 | `thm:convnext_loss_grad` | 10287 | 22 | A- | 4 | 1 | 0 |
| 9 | `thm:vitForwardKVHasVJP` | 12285 | 18 | A- | 2 | 0 | 0 |
| 9 | `thm:vitFwdGraphKMHV_faithful` | 12311 | 14 | -- | 4 | 0 | 1 |
| 9 | `thm:vitTinyHasVJP_correct` | 12333 | 22 | AE | 3 | 0 | 0 |
| 9 | `thm:vit_whole_back` | 12385 | 16 | -- | 3 | 1 | 0 |
| 9 | `thm:vit_step_tie` | 12409 | 16 | -- | 5 | 0 | 0 |
| 9 | `thm:vit_loss_grad` | 12433 | 23 | A- | 4 | 1 | 0 |

Also in scope, smaller:

* The three `definition` blocks that are really implementation notes — `thm:reluHasVJP`,
  `ax:mlpHasVJP`, `ax:maxPool2HasVJP3` ("noncomputable def with the pdiv-derived backward;
  `HasVJP.correct` holds by `rfl`; codegen substitutes the standard argmax routing convention
  at tiebreaks"). A definition says what the object is; how it elaborates is the docstring's.
* `thm:efficientnetFullHasVJP` and `thm:mobilenetv4FullHasVJP` each bundle a VJP with two or
  three graph-faithfulness theorems in one block. Split: one VJP theorem, one "the graph denotes
  the forward (with and without drop masks)" theorem, as Chapter 9 already does
  (`thm:vitForwardKVHasVJP` / `thm:vitFwdGraphKMHV_faithful`).
* `thm:cifar8_step_tieG`'s opening sentence ("The runs this chapter reports train the packed
  steps …, not the fused SGD steps of the two theorems above") is the chapter's one
  reader-facing fact about which artifact trained; it becomes the lead-in sentence before the
  group, not the statement.

Not in scope: register A; the Bestiary's 22 `Layer` definitions; the front-matter budget table
(44 / 9 / 37 — recount after this pass, separately).

## 5. Order and review

One chapter per commit, reviewed on :8765 with `scripts/book/blueprint_preview.py` (current
vs proposed), the user's pattern. Order: Chapter 3 (the twins exemplar, 4 blocks) and Chapter 5
(the ladder preamble + 11 blocks) first, since they decide the template; then 1, 2, 4; then 6,
7, 8, 9, which inherit Chapter 5's preamble.

Gates per chapter: `lake build` is untouched (no Lean changes); `blueprint_uses.py --check`
(the `\uses` lists do not change); `scripts/gates/audit_only_mentions.py` (report must not
grow — every name that leaves a body goes into that block's `\lean{}`); `scripts/book/book_xrefs.py
--summary` (new `Theorem~\ref`s inside a chapter are pointers); the PDF build with no undefined
refs and a page count within one page of 219.

Expected size: the 56 blocks are 985 lines now; the template drops the name sentences and the
re-explained ladder and adds `enumerate` scaffolding; net somewhere around -150 lines, page
count flat.

## 6. Status

* 2026-10-01: Chapter 3 (`ax:maxPool2HasVJP3`, the four network blocks, a two-sentence lead-in,
  the twins paragraph folded into `thm:cnn_sgd_descends` item 3 and `thm:cnn_loss_grad` item 2,
  the "rendered backward makes the same choice" paragraph folded into `thm:cnn_loss_grad`'s gloss)
  and Chapter 5 (the ladder paragraph before `thm:resnet34FullHasVJP`, eleven blocks) rewritten
  in the working tree, uncommitted, previewed on :8765. Every name that left a body is in its
  block's `\lean{}`. Decisions taken as proposed in §3b: the loss gradients say "each node of the
  step tie" (110 stays, 146 goes); the strict-pool caveat sits on `thm:resnet34FullHasVJP`.
  Gates: `blueprint_uses.py --check` in sync (163 blocks, 1114 edges), only-mentions report
  unchanged (50), PDF 221 pages both sides, 0 LaTeX errors, no new undefined references.
  Net +218/−147 lines (the `enumerate` scaffolding costs more than the name sentences save).
* 2026-10-01 (later): Chapters 6–9 rewritten in the working tree, uncommitted (27 blocks: the
  twelve MobileNet rungs, the five EfficientNet rungs, `thm:chanLNTensor3HasVJP` + the three
  ConvNeXt rungs, `thm:layerNormVecHasVJP` + the six ViT rungs), each chapter opening its ladder
  with a one- or two-sentence lead-in that points at Chapter 5's. Honesty fix surfaced by the
  template: `thm:mobilenetv2_whole_back` and `thm:mobilenetv4_whole_back` are ResNet-50-shaped
  (opaque block witnesses `hb1…hb17` / `hfused, hb1…hb21`, stem and head conditions, a separate
  `*ForwardBFull_eq_slots` shape check) and the book had stated them with no hypotheses; they now
  carry R50's Assume list. `thm:efficientnet_whole_back` gains its `EpsPos` hypothesis the same
  way. Kept bundled (no new blocks, so the depgraph figures and the budget table are untouched):
  `thm:mobilenetv4FullHasVJP` (retitled "VJP and graph") and `thm:efficientnetFullHasVJP`; the
  split into a VJP block and a "graph denotes the forward" block waits for the budget recount.
  Scope sentence kept once per step tie: "at one replica, in f32, on the chain without stochastic
  depth; the drop and bf16 renders are outside it". Gone: the convBias-census conjunct count,
  "the AdamW, RMSProp, EMA tails consume that node", the `*GradBBf16` per-operator remark, every
  `X discharges …` sentence, the `*CotIn_eq_vjp` name lists (now in `\lean{}`). Gates:
  `blueprint_uses.py --check` in sync, checkdecls over 257 names, only-mentions 50.
* 2026-10-01 (end): Chapters 1, 2 and 4 rewritten (17 blocks: the three Chapter-1 certificates
  with a four-line "three certificates close every small-net chapter" lead-in, the two Chapter-2
  VJP definitions and four theorems, the eight Chapter-4 blocks with a lead-in that carries
  `thm:cifar8_step_tieG`'s "the runs train the packed steps" sentence). Every register-B block in
  the plan's §4 table is now on the template; `thm:hasGradAt_comp` was left as it was (it reads as
  a definition's gloss and its three `\lean{}` names are its three readings). Gates on the whole
  tree: `blueprint_uses.py --check` in sync (163 blocks, 1114 edges), checkdecls over 262 names,
  only-mentions 50, PDF builds (page count in §6's build line below). Whole pass: +745/−470 lines.

## 7. Proof-side follow-ups the pass surfaced (documented, not done)

The user's call (2026-10-01): document these; a separate doc for proof improvements if any is
taken up. None blocks the book pass.

1. **`thm:mobilenetv2_whole_back` / `thm:mobilenetv4_whole_back` are opaque-block ties.** Like
   ResNet-50's, they take block witnesses `hb1…hb17` (`hfused, hb1…hb21`) and a separate
   `*ForwardBFull_eq_slots` shape check; EfficientNet-B0's and ConvNeXt-T's chains are composed
   through to the concrete forward. R50's docstring says the composition times out the kernel; the
   MobileNet files should say whether the same holds or the composed form was never attempted. If
   it elaborates, the composed corollary makes the book statement one line shorter and stronger.
2. **The 146-vs-110 slot accounting (R34), 210-vs-158 (MNv2), 262-vs-213 (B0).** The loss-gradient
   theorems are stated over every parameter slot of the proof model, the step ties over the
   emitted nodes; the book now states the loss gradients over the step tie's nodes (the weaker,
   true reading). A corollary restricted to the emitted slots, or a docstring line per net saying
   which slots the renders fold away (conv biases), would let the book say one number with no gap.
3. **The strict stem-pool clause of `R34SmoothAtB` / `R50SmoothAtB`** (one maximal cell, no twins)
   is a condition real batches do not meet (`scripts/probes/stem_pool_smooth_probe.py`); the loss
   gradients already state the twin-tolerant clause. Re-stating the input-gradient VJP under the
   twin-tolerant bundle (as `r34_net_lossGrad` does via `IsMaxPool3s2SelectB`) would remove the
   caveat sentence from `thm:resnet34FullHasVJP`.
4. **The two VJP + graph bundles** (`thm:mobilenetv4FullHasVJP`, `thm:efficientnetFullHasVJP`)
   should be two blocks each, as Chapter 9's `thm:vitForwardKVHasVJP` / `thm:vitFwdGraphKMHV_faithful`
   are; this is a book change that moves the generated depgraph figures and the front-matter
   budget table, so it goes with the budget recount.
5. **`thm:hasGradAt_comp` reads `hf` as "f has a VJP at x"**; the Lean binder is
   `DifferentiableAt ℝ f x` with the VJP a separate argument. Minor; fix the wording when the
   block is next touched.
6. **The budget table (44 / 9 / 37)** was not recounted; the witnesses column should also count
   `trained_cnn_conv2_sgd_descends_concrete` and its conv-1 twin, which the pass now names as
   witnesses in Chapter 3.

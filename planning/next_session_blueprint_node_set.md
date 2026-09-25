# next_session_blueprint_node_set.md — the blueprint's node set, from the VJP spine to the whole ladder

**Where this stands.** As of `37104d98` (2026-09-22) the blueprint's dependency graph is honest and
self-maintaining: the `\uses` lines are generated from a Lean dependency walk and CI holds them to it
(`blueprint-checkdecls … blueprint/lean_deps` + `scripts/book/blueprint_uses.py --check`), the web graph
draws chapters in order with portal nodes, and every theorems section opens with its chapter's graph
as TikZ (`scripts/book/blueprint_depgraph_tikz.py` → `blueprint/src/figures/depgraph/`). What the graph
*covers* has not moved since the original suite: 87 proof nodes, all on the per-layer-VJP spine —
the calculus of chapter 1, one `*_has_vjp` per layer, and a net-level `*_has_vjp_at` (ViT:
`vitTiny_has_vjp_correct`) as each chapter's last node. Everything the suite grew since 2026-08 has
no node: the full-width batched nets, the folds, the step ties, the seals, the sync-BN twins, the
descent capstones. The book names them in `\texttt` prose only (§2 lists where). This session adds
them, chapter by chapter, so that each chapter's figure ends where the chapter's argument ends —
"the rendered graph is this math, and the step descends" — not at "the net has a VJP".

Nothing here needs a GPU.

**Status 2026-09-22, end of session:** the whole set — chapter 1's two nodes and chapters 2–9's 38, the
shared-box rule in the generator, the two section retitles, the legend clause — is applied to the tree,
verified (checkdecls 162 names, `--check` 149 blocks / 794 edges, PDF 0 errors, plastex web parse clean)
and staged as ONE commit at the user's call ("I can tell it's better than what we had"), overriding §0's
one-chapter-per-commit rule for this landing. The book's node count goes 87 → 127; the intro's census table gains a Steps column (45 contracts, 9 witnesses, 30 step certificates = 84) and Appendix C's sentence matches.

---

## 0. How this work gets reviewed — the rule

* **One chapter per commit.** Stage, then STOP; commit only on the word. Approval for one commit does
  not carry to the next. `git diff --cached` to show it.
* **Every new environment's statement is one or two sentences** in the voice of the chapter's existing
  ones; the book already narrates each of these theorems, so the statement is a restatement, not new
  prose. No per-figure text: the figure legend is said once, under chapter 1's figure, and the user
  has said the figures need nothing around them.
* Figures are top-down (sources upper left, capstone lower right) whenever that reads at ≥ 5.5 pt;
  the generator decides. If a chapter's new nodes push its figure below that, cut the section at a
  named node (`CUTS` in the generator) rather than shrinking — the attention section is the model.
* Rebuild the PDF under `timeout` and eyeball the changed pages before handing over
  (`leanblueprint pdf` runs xelatex interactively; a TeX error hangs it at a prompt).

---

## 1. State of the tree

| what | where | status |
|---|---|---|
| node set | `blueprint/src/content.tex` theorem/definition environments with `\lean{}` | 87 proof nodes + 22 Bestiary `Layer.*` (constructors; the walker skips them) |
| edges | `\uses{…}` lines | GENERATED — never hand-edit; `scripts/book/blueprint_uses.py --fix` |
| real edges | `blueprint/lean_deps` (gitignored) | written by `lake exe blueprint-checkdecls blueprint/lean_decls blueprint/lean_deps` |
| cited names | `blueprint/lean_decls` (gitignored) | written by `leanblueprint web` in CI; locally the imager fails, so write it yourself: the `\lean{}` names of content.tex, one per line |
| web graph | `blueprint/src/templates/dep_graph.html` | chapters in order, portals, `pdiv`/`hasvjp` drawn once; nothing to change for new nodes |
| print figures | `blueprint/src/figures/depgraph/*.tex`, committed | `python3 scripts/book/blueprint_depgraph_tikz.py` after any `\uses` change; CI has no graphviz |
| gates | blueprint.yml | checkdecls (names exist) → `--check` (edges match) → web build rasterizes the TikZ |

The walker (`tests/BlueprintCheckDecls.lean`, `writeDeps`) is only as good as the oleans it loads:
CI's fresh build is the authority. A locally REMOVED edge must be re-checked against CI before it is
believed (2026-09-21: a local run walked 255 edges, CI 274 — the 19 were the ViT finale's).

---

## 2. The ladder, per chapter — what to add

Each net's chapter should carry these rungs above its existing `*_has_vjp_at` node. Modules are
under `LeanMlir/Proofs/`; the capstone is the theorem the module's docstring names as such — the
"probable" names below came from the last theorem in each file and MUST be checked against the
module (several are `*_den` lemmas that sit after the capstone). `\lean{}` takes the fully qualified
name; `sealX_backward_nontrivial` and `headBGradB_den` repeat across modules, so the namespace is
part of the name.

| rung | what it says | module pattern | probable capstone |
|---|---|---|---|
| batched net | the full-width net at batch B has the VJP | `Nets/<f>/<Net>FullBVJP.lean` | `<net>ForwardB_full_has_vjp_at_correct` |
| fold | the emitted train step denotes the certified gradient step | `Nets/<f>/<Net>Fold*.lean` | Small: `mlp_train_step_tied_certified`, `cnn_conv_tied_certified`, `cifar_conv_tied_certified`, `poc_train_step_tail_certified` (Linear); others: check the module |
| step tie | the whole-net step ties to the batched spec at the loss | `Nets/<f>/<Net>StepTie*.lean` | `<net>_net_tied*` (`cnx_net_tiedGB`, `efficientnet_net_tiedG`, `vit_net_tiedGB`), ResNet/MobileNet: `*_lossCot_is_*_grad` |
| seal | non-degeneracy on the full-width net | `Nets/<f>/<Net>FullBSeal.lean` | `…sealX_backward_nontrivial` (namespaced) |
| sync-BN twin | the DP sync render's step equals one device's batch of R·N | `Nets/<f>/<Net>SyncStepTieB.lean` | `r34_net_syncTiedB`, `r50_net_syncTiedB_bce`, `mnv2_net_syncTiedB`, `mnv4_net_syncTiedB_smoothedCE`, `efficientnet_net_syncTiedG` |
| descent | the step decreases the loss | `Training/SgdDescent<Net>.lean` | `linear_float_sgd_descends`, `mlp_input_float_sgd_descends`, `cnn_conv1_bias_float_sgd_descends`, `cifar8_lastConv_sgd_descends` |

Per chapter — VERIFIED against the modules 2026-09-22 (three readers, then one walk: every name below
resolves; `lean_deps` for the whole set is 938 edges). The drafted environments are in
`planning/blueprint_ladder_environments.tex`, one block per chapter, ready to paste before the named anchor.

| ch | nodes | `\lean{}` names (label -> name) |
|---|---|---|
| 1 | 2 (landed) | `thm:linear_fold` -> `Proofs.LinPoC.poc_train_step_tail_certified`; `thm:linear_sgd_descends` -> `Proofs.linear_sgd_descends, Proofs.linear_float_sgd_descends` |
| 2 | 2 | `thm:mlp_fold` -> `Proofs.MlpPoC.mlp_train_step_tied_certified`; `thm:mlp_sgd_descends` -> `Proofs.mlp_input_sgd_descends, Proofs.mlp_input_float_sgd_descends` |
| 3 | 2 | `thm:cnn_fold` -> `Proofs.CnnPoC.cnn_conv_tied_certified`; `thm:cnn_sgd_descends` -> `Proofs.cnn_conv2_sgd_descends, Proofs.cnn_conv2_float_sgd_descends` (the header's capstone is conv2, not the conv1-bias rung) |
| 4 | 5 | `thm:cifar_fold` -> `Proofs.CifarPoC.cifar_conv_tied_certified`; `thm:cifar_bn_fold` -> `Proofs.CifarBnPoC.bnSgdPairTied_holds`; `thm:cifar8_step_tie` -> `Proofs.Cifar8PoC.cifar8_convs_tied_certified`; `thm:cifar8bn_step_tie` -> `Proofs.Cifar8BnPoC.cifar8Bn_convbn_tied_certified`; `thm:cifar8_sgd_descends` -> `Proofs.cifar8_lastConv_sgd_descends`. `Cifar8Fold.lean` has NO declarations |
| 5 | 9 | `resnet34/50ForwardB_full_has_vjp_at_correct`; `Proofs.r50InputGradB_eq_r34B_full_vjp, Proofs.r50InputGradB_correct`; `Proofs.ResNet34TieB.r34_net_tiedB`; `Proofs.ResNet50TieB.r50_net_tiedB, ...r50_lossCot_is_bce_grad`; `Proofs.R34FullBSeal.sealX_backward_nontrivial`; `Proofs.R50FullBSeal.sealX_backward_nontrivial`; `Proofs.ResNet34SyncTieB.r34_net_syncTiedB`; `Proofs.ResNet50SyncTieB.r50_net_syncTiedB, ...r50_net_syncTiedB_bce`. Retitle the section "The theorems" |
| 6 | 10 | `Proofs.mobilenetv2ForwardB_full_has_vjp_at_correct`; `Proofs.StableHLO.mobilenetv4ForwardB_full_has_vjp_at_correct`; `Proofs.mnv2InputGradB_eq_mobilenetv2B_full_vjp, Proofs.mnv2InputGradB_correct`; `Proofs.mnv4InputGradB_eq_mnv4B_full_vjp, Proofs.mnv4InputGradB_correct`; `Proofs.MobileNetV2TieB.mnv2_net_tiedB`; `Proofs.Mnv4TieB.mnv4_net_tiedB, ...mnv4_lossCot_is_smoothedCE_grad`; `Proofs.Mnv2FullBSeal.sealX_backward_nontrivial`; `Proofs.Mnv4FullBSeal.sealX_backward_nontrivial`; `Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB`; `Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB, ...mnv4_net_syncTiedB_smoothedCE`. MNv4 is the chapter's side quest; its nodes measured inside that section come out as a row of islands, so they stay in "The theorems" |
| 7 | 4 | `Proofs.efficientnetForwardB_full_has_vjp_correct, Proofs.StableHLO.efficientnetFwdGraphB_full_faithful`; `Proofs.efficientnetInputGradB_full_correct`; `Proofs.EnetTiePoCG.efficientnet_net_tiedG`; `Proofs.EnetSyncTieG.efficientnet_net_syncTiedG`. Retitle the section "The theorems" |
| 8 | 2 | `Proofs.convnextInputGradB_correct, Proofs.convnextImagenetInputGradB_eq_vjp`; `Proofs.CnxTiePoCGB.cnx_net_tiedGB` |
| 9 | 4 | before the finale: `Proofs.vitForwardKV_has_vjp_correct`; `Proofs.StableHLO.vitFwdGraphKMHV_faithful`; after it: `Proofs.vitTinyInputGradB_eq_vitTiny_vjp, Proofs.vitInputGradKB_correct`; `Proofs.ViTTiePoCGB.vit_net_tiedGB` |

Two rungs of the original table are gone and one was added. Gone: the "fold" rung for every ImageNet net,
because each `*Fold*`/`*FoldB`/`*FoldG(B)` module is an op table of per-op `*_den` lemmas with no capstone
(`ResNet34Fold` is retired outright) — the `*StepTie*` capstone IS the whole-net "emitted step = certified
step" theorem. Added: the whole-backward ties (`*InputGradB*_correct`, "the rendered backward is the VJP"),
which the doc had not listed and which are real whole-net theorems. Total 2 + 38 = 40 nodes.

**Finding (2026-09-22): the rungs are not stacked in Lean.** No ResNet certificate cites another: the sync
tie does not cite the step tie, the step tie does not cite the batched VJP, the seal cites neither. Each
is proved directly from the same 11–16 layer theorems of chapters 1, 3, 4 and 9 (`conv2d_has_vjp3`,
`bn_has_vjp`, `dense_has_vjp`, `rowwise_has_vjp_mat`, `vjp_comp`, …). The chapter's shape is a fan, and
the fan pushes 16 existing nodes over the generator's `AMBIENT = 4` threshold, which hid them as portals
everywhere — ch 5 rendered as nine islands with nothing listed. Fixed in `blueprint_depgraph_tikz.py`: a
node whose every source is hidden (ambient, listed or heavy) hangs from the figure's one shared box,
"N statements of ch 1, 3, 4, 9"; nodes with a drawn source are untouched, so the simplification the
ambient rule brings to the attention figures stays. Proposed sizes with everything in: ch 2 7.0, ch 3 5.6,
ch 4 7.0, ch 5 7.0 (170 × 251, a fan), ch 6 6.3, ch 7–9 7.0. The interactive graph keeps its own
`AMBIENT_CHAPTERS = 4` and names the hidden statements once under the graph; not changed.

Not in this pass: the comparator tier, the float/bf16/E4M3 folds, the Lipschitz certificates. They
are their own families and belong to appendix C's story, possibly as one appendix figure later.

---

## 3. Mechanics, in order, per chapter

1. Find the capstone: open the module, read its docstring, take the theorem it names. Get the fully
   qualified name (`#check` or the docs site). Two names on one node is fine: `\lean{a, b}`.
2. Add the environment where the chapter narrates that theorem, in the chapter's existing pattern:

   ```latex
   \begin{theorem}[Emitted step is the certified step]
     \label{thm:mlp_fold}
     \lean{Proofs.mlp_train_step_tied_certified}
     \leanok
     One or two sentences in the chapter's voice.
   \end{theorem}
   ```
   No `\uses` — it is generated. Placement decides the figures: a chapter whose nodes span more than
   one `\section` gets one figure per section, so putting the ladder nodes in the section that tells
   their story (e.g. "MLIR: Residual", "The verified trainer") gives that chapter a second figure at
   that section's head. Decide with the user before chapter 1: one figure per chapter (everything in
   "The theorems") or the ladder as its own figure. The attention section shows what a cut looks like.
3. Regenerate: write `blueprint/lean_decls` from content.tex, then
   `lake exe blueprint-checkdecls blueprint/lean_decls blueprint/lean_deps`,
   `python3 scripts/book/blueprint_uses.py --fix`, `python3 scripts/book/blueprint_depgraph_tikz.py`
   (its table prints each figure's font size; anything under 5.5 pt wants a `CUTS` entry).
4. `leanblueprint pdf` under `timeout 300`; look at the chapter's figure page. Stage. Stop.

---

## 4. Order

Chapter 1 first: two nodes (`LinearFold`, `SgdDescentLinear`), the pattern for everything after, and
the smallest possible diff to agree the placement rule on. Then 2–4 (fold / step tie / descent, the
Small tree), then 5–7 (the batched-net rungs, seals, sync twins), then 9, then 8 with the ConvNeXt
rewrite. Roughly 45 environments; a chapter is an hour including the read.

**Placement rule, agreed with chapter 1 (2026-09-22): the ladder nodes go in "The theorems", after
the chapter's last VJP node** — one figure per chapter, ending at the capstones, every edge drawn
(ch 1: 460 × 398 pt at 5.6 pt). The alternative, nodes in the section that tells their story
("MLIR: Training Step"), was measured on a scratch copy: its two-node figure sets at 7.0 pt but the
fold node has four imports, so the ≥ 4 rule lists them and draws it as an island. When a chapter's
figure falls under 5.5 pt, cut at the net's `*_has_vjp_at` node (`CUTS`) rather than move nodes.
The descent node carries both names, `\lean{Proofs.linear_sgd_descends,
Proofs.linear_float_sgd_descends}`; the walker gives a two-name block the union of their edges.

---

## 5. Traps already met

* `leanblueprint web` fails locally in the figure imager; use
  `plastex -c plastex.cfg --imager=none --vector-imager=none --dir=<scratch> web.tex` in
  `blueprint/src` to check the web parse, and write `lean_decls` by hand (§3.3).
* The walker must not enter `Proofs.StableHLO.den` / `denOp`: one recursion over every op, so
  the linear fold's edges arrived from conv2d, depthwise, GELU, maxPool and SE. `walkBoundary` in
  `tests/BlueprintCheckDecls.lean` stops there (prefix match, so the equation lemmas and `._f`
  stop too); a fold's real layer edges come through its proof, and the walk still finds them.
* Stale local oleans truncate the walk silently at the first missing constant; the walker now warns
  on stderr. `lake build Certs` is the honest refresh (bare `lake build` misses it) and is heavy —
  or trust CI's `--check` verdict and fix forward.
* A `\\` inside `\hyperref{}` in a TikZ node breaks `align`; the generator already emits one link per
  line. `\noindent\small` after a figure leaks into the chapter unless scoped.
* The web graph's `AMBIENT_CHAPTERS = 4` rule and the print figures' per-figure rule (a source
  feeding ≥ 4 nodes, or a node with ≥ 4 imports, is not drawn) will absorb the new hubs; check that
  a chapter's capstone does not vanish into the second rule (it has many imports by construction —
  it will keep its in-chapter edges and lose only the cross-chapter portals).

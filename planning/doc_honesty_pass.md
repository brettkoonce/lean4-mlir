# doc_honesty_pass.md — making the prose say what the theorems say

Started 2026-09-26 from a ten-way read of every docstring, module docstring and published document
against `doc_audit/rubric.md` (overclaim > stale > proof-narrative > missing > process-narrative >
copied), on origin/main `290187b1`. The ten reports (`doc_audit/find_*.md`) quote prose and
statement side by side at file:line and give replacement wording; 240 entries in all. They are
static reads — nothing was built — and the top findings of each slice were re-checked by hand.
Their "Fix:" lines are suggestions, not checked text, and their counts can be off (the reports say
AuditAxioms 1,611 and Heavy 76; the files have 1,610 and 62).

Most of the overclaims were already found by the 2026-09-24 correctness audit and never fixed in
the prose. This pass fixes them.

| Report | Slice | Entries |
|---|---|---|
| `find_A_published.md` | README, CHANGELOG, LeanMlir/README, NAMING, apps/README, content.tex 1–8750 | 20 |
| `find_B_blueprint2.md` | content.tex 8751–end, formalization.yaml | 18 |
| `find_J_toplevel.md` | LeanMlir.lean + LeanMlir/*.lean outside Proofs/ | 32 |
| `find_C_foundation_float.md` | Proofs/Foundation, Proofs/Float, SpecVJP | 37 |
| `find_D_arch_training.md` | Proofs/Architectures, Proofs/Training | 28 |
| `find_E_codegen.md` | Proofs/Codegen | 27 |
| `find_F_certificates.md` | Proofs/Certificates (+ scripts/certs generators) | 18 |
| `find_G_mobilenet.md` | Nets/MobileNet | 19 |
| `find_H_resnet_small.md` | Nets/ResNet, Nets/Small | 24 |
| `find_I_convnext_b0_vit.md` | Nets/ConvNeXt, Nets/EfficientNet, Nets/ViT | 17 |

## ▶ Start here (state at 2026-09-26, origin/main `290187b1`)

Nothing in this plan is landed. A first pass at §1 (54 hunks across the book, the API landing
page, README, yaml, CHANGELOG and three small READMEs) was drafted, previewed, and **backed out
whole** at the user's request: the review found a replacement sentence that was itself an
overclaim (see "What went wrong", below). The next session starts from a clean tree and works
§1 again under the protocol in §0, one theme at a time.

**What went wrong in the first attempt.** The draft rewrote the ChallengeArch sentence to say
"the whole-network rows are the small, two-block worked examples". Of the six whole-network rows
in `tests/comparator/ChallengeArch.lean`, two are the real chapter nets (`mnistLinearHasVJP`, the
MNIST `cnnHasVJPAt`), three are reduced-depth nets (`mobilenetv2HasVJPAt`, `efficientnetHasVJP`,
`convnextHasVJP`: stem, two blocks, head), and one is the weight-shared ViT body
(`vitFullHasVJP`). The draft had taken a report's "Fix:" line, coined a category label, and not
enumerated what the label covered. Three causes, each now a rule in §0: the reports' fixes were
treated as verified text; a new sentence introduced a summary term nobody checked; and 54 hunks
went to review at once.

## 0. Rules for this pass

**The protocol, per site.** For every sentence changed:

1. Read the prose in place, and read the Lean statement it describes: signature, every hypothesis,
   the conclusion. The report's quote is a pointer, not evidence.
2. Write the replacement so that every noun in it names something that exists: a declaration, a
   file, a hypothesis, an artifact. No new category words ("worked examples", "toy nets",
   "representative"). If the sentence summarises a set, enumerate the set once, check each member,
   and only then choose the summary word; if no one word fits every member, list them.
3. Every number in the replacement is re-counted from the source at write time, with the command
   recorded in the ledger (below).
4. Record the site in the ledger: file:line, old claim, the statement read, the new text, how
   each noun and number was checked.
5. Review on the preview server per theme (a handful of sites), not per phase. Nothing is staged
   until the user has read that theme.

**The ledger** is `doc_audit/ledger.md`, one row per changed site, written as the site is changed.
It is the review document: the user reads old claim / statement / new text side by side.

**Standing rules.**


* A doc fix states the theorem's actual scope; it does not apologise for it. Write "at one
  replica, f32, no drop-path" as the statement's scope, not as a caveat paragraph
  (right, then described, then coherent).
* Before rewording, re-read the statement — the reports are a day old and a static read. If the
  statement moved, the finding moves with it.
* A `\S\ref` names, never asserts (`scripts/book/book_xrefs.py`). A cross-reference to a
  declaration is a backticked name that `docstring-checkrefs` resolves.
* Generated files: fix the generator, regenerate, confirm byte-reproducible before and after.
* Docstring-only edits to a root file (`Foundation/Tensor.lean` 423 downstream,
  `Codegen/StableHLO.lean` 315) still rebuild everything below it: batch them, one landing each.
* No Lean statement changes in the prose phases (§1–§4). Statement changes are §5, decided per item.

Gates before each landing: `lake build`, `lake build Certs CertsHeavy Proofs Apps`,
`tests/AuditAxioms.lean`, `lake exe docstring-checkrefs`, `scripts/gates/name_lint.py`,
`scripts/book/blueprint_uses.py --check` (regenerate `lean_decls` first),
`scripts/gates/gen_comparator_tier.py --check`, and `git status verified_mlir` unchanged unless §4
says otherwise. Blueprint: `latexmk` print + web build, review on the preview server (:8765).

## 1. The published surface (first: this is what readers believe)

Files: `LeanMlir.lean` (the doc-gen4 home page), `README.md`, `blueprint/src/content.tex`,
`formalization.yaml`, `CHANGELOG.md`, `tests/comparator/README.md`.

(a) **Tie scope.** "every `*in_*` artifact", "at the ImageNet head its artifacts run", "each
    parameter-update node denotes the certified descent step" → the statements are one replica
    (sync twins at N := R·N), f32 (bf16 emits `*GradBBf16`), drop-free, at the gradient node; B0
    and ViT at 10 classes. Sites: `LeanMlir.lean:114`, content.tex 8384/8408/9619/9653,
    yaml:166. [A1, B2, J3]
(b) **"Two things are proved for every net … Descent"** (`LeanMlir.lean:80`) → descent is the
    linear/MLP/CNN rungs, one example, under small-step hypotheses. [J2]
(c) **"Whole-network VJPs"** list (`LeanMlir.lean:112`) names three reduced-depth nets
    (stem, two blocks, head) and the weight-shared `vitFullHasVJP`; point at the full-depth
    witnesses (table below). [J1]
(d) **Headline sentence** content.tex:16990 ("exact derivative … up to one printer, one lowerer,
    and floating point"): add the StableHLO-semantics trust item listed three lines above, the
    smooth-point condition, and the tie scope of (a). [B1]
(e) **Round trip.** content.tex:16982 "lexes and parses back" → the token skeleton round-trips;
    the token↔text map (`emitTok`) is trusted. [B1, E1]
(f) **MaxPool condition.** "argmax tie / measure-zero / unique argmax / off-the-kink" → "every
    pair of window entries distinct", which fails in any window with two dead ReLUs.
    content.tex:17030, 5212–5244; yaml:221–226. [A2, B3]
(g) **`HasVJP` framing.** ch 1 (675–683) and yaml:225 ("codegen-shaped witnesses") → ReLU/MLP/
    max-pool witnesses are `HasVJP.canonical`; the emitted backward is tied by separate pointwise
    bridges. ChallengeArch (content.tex:16522) has no ResNet-34 row; its whole-network rows are
    enumerated in "What went wrong" — the replacement must not summarise them with one word. [A5, B4]
(h) **Batch.** linear/MLP folds are one example; the α/128 render is not "line for line" a
    theorem. content.tex:1535, 2147–2150, 2389, 2518. [A3]
(i) **LayerNorm citations.** ConvNeXt → `chanLNTensor3HasVJP`, ViT → `layerNormVecHasVJP`, not
    `layerNormHasVJP`. content.tex:9362, 9819, 10488. [B]
(j) **Numbers.** comparator 73/21 → 87/35 (LeanMlir.lean, content.tex, comparator README);
    AuditAxioms 1,374 → about 1,600 (1,610 lines today); ViT theorem counts; README tier-3 R34 89.50 → 89.99 ± 0.32,
    tier-4 74.06 → 74.17; CHANGELOG v0.7.0 78.26% is the JAX reference's. Prefer generating
    counts over hand-typing them where a script already has the number.
(k) **Stale sections.** content.tex:5309 retired 76.66% run; "exactly one theorem" (ch 5);
    "stops at Imagenette" (6103); R50 section 6347–6409 vs 82445a97 (see Decisions, D3).
(l) **Wrong formula.** content.tex:10568 softmax Jacobian index. [B]

**Facts verified in the first attempt** (re-check before relying on them; each took a grep):

| Fact | Source |
|---|---|
| Full-depth whole-net witnesses: `resnet34ForwardBFullHasVJPAt`, `resnet50ForwardBFullHasVJPAt`, `mobilenetv2ForwardBFullHasVJPAt`, `Proofs.StableHLO.mobilenetv4ForwardBFullHasVJPAt` (note the namespace), `efficientnetForwardBFullHasVJP` (B0Weights, 10 classes), `convNextForwardTChHasVJP`, `vitForwardKVHasVJP` | `grep -rn "def <name>"` |
| Optimizer per-op ties: `Proofs.StableHLO.adamW_triple_faithful`, `Proofs.StableHLO.lamb_triple_faithful` | Codegen/StableHLO.lean:3438, Codegen/LambTriple.lean:72 |
| `cnx_net_tiedGB` and `vit_net_tiedGB` bind `nC`; both docstrings say one replica, drop-free chain. `vitTinyInputGradB_eq_vitTiny_vjp` and `efficientnet_net_tiedG` are pinned at 10 classes | the signatures |
| ConvNeXt-T has 23 channel-LNs (stem, 18 blocks, 3 downsamples, head) | `CnxTieWeights`, ConvNeXtStepTie.lean:475 |
| Comparator: 13 + 39 + 35 = 87 theorems; yaml `main_results` point 35 / 3 / 1 at tier / arch / challenge | `config*.json`; `grep -o comparator_config formalization.yaml` |
| `tests/AuditAxioms.lean` has 1,610 `#print axioms` lines, `AuditAxiomsHeavy.lean` 62 | `grep -c "^#print axioms"` |
| ViT chapter: Matrix-level machinery has 14 blocks (13 theorems + 1 definition), Attention proofs 25 | theorem environments between the section heads |
| ChallengeArch whole-network rows: see "What went wrong" | `grep -o "theorem chk_[A-Za-z0-9_]*" tests/comparator/ChallengeArch.lean` |
| Transformer block/tower/body witnesses take `hε : 0 < ε` and scalar `γ1 β1 : ℝ` | Attention.lean:1481, 1560, 1655 |
| No ResNet-34 per-example whole-net declaration exists (all `resnet34*` are batched) | `grep -rhoE "(def\|theorem) resnet34[A-Za-z0-9_]*"` |
| No MobileNetV4 `*CotIn_eq_vjp` exists | `grep -rhoE "theorem [A-Za-z0-9_]*CotIn_eq_vjp"` |

**Tooling that works locally.** `lake build LeanMlir` + `lake exe docstring-checkrefs` (the
landing page is a docstring; the gate caught a wrong namespace in the draft).
`scripts/gates/gen_comparator_tier.py --check` after any yaml edit. Book preview: copy
`blueprint/src` twice into the scratchpad (HEAD via `git archive HEAD blueprint/src`, and the
working tree), then in each `plastex -c plastex.cfg --imager=none --vector-imager=none --dir=<out>
web.tex` and `latexmk -xelatex -interaction=nonstopmode -output-directory=<out> print.tex`
(~25 s each, run all four in parallel); serve with `setsid nohup python3 -m http.server 8765 --bind
0.0.0.0 --directory <site>` and link http://100.76.1.97:8765/. A `difflib.HtmlDiff` page of every
changed file beside the builds made the review fast. `leanblueprint web` fails locally at HEAD
(plasTeX image step), so use the `--imager=none` form; CI builds the figures.

**Open from the first attempt.** The ch 5 row of the theorem-budget table (content.tex ~405) cites
"ResNet-34 per example", which does not exist; the row's contract count (10) and the 45 total
have to be re-derived by the table's own counting rule (content.tex ~380–392) before the phrase
is dropped.

## 2. Hand-written Lean docstrings — overclaims and wrong facts

Grouped by theme; each group is one commit. Report entries carry the replacement text.

(a) **Tie scope in the capstones** — mirror §1(a) in `cnx_net_tiedGB`, `vit_net_tiedGB`,
    `EfficientNetStepTieG`, `ConvNeXtWholeBackCertifiedTieB:519` (T only, not S/B),
    `ViTWholeBackCertifiedTieB:290` (512 is global), `ViTStepTie:14`. [I1–I4]
(b) **Cotangents called "the certified ∂L".** State what is proved: the op denotes θ − lr·(certified
    local Jacobian · the emitted chain cotangent). Name where a `*CotIn_eq_vjp` exists (R34, R50,
    MNv2, B0-sync) and where it does not (MNv4, ConvNeXt, ViT, small CNNs). `CnnFold`,
    `CifarFold`, `Cifar8StepTie`, `Cifar8BnStepTie`, `MlpFold`/`MlpCanonical` (only W₂),
    `MobileNetV4StepTieB`, R34/R50 "together" sentences. [H1, H4, H5, G1, I5]
(c) **"Every input".** `r34/r50/mnv2/mnv4InputGradB_correct`: list the stem/pool smoothness
    hypotheses and the block witnesses. [H3, G2]
(d) **Round trip.** `StableHLOParse` module (also dead `parse_skel`, `den (emit g) = fderiv`),
    `StableHLOPretty`, the seven `*_faithful` "roundtrip covers it structurally". [E1–E3]
(e) **"Every artifact is `pretty` of one term".** `StableHLO.lean` module, `MlpRender`,
    `CnnRender`, `ConvNeXtRender` ("ENTIRELY", 180 → 182): name the hand-written `%loss` block
    and ConvNeXt's stem/`%dy`/GAP text. [E3, E4]
(f) **Unverified paths labelled verified.** `VerifiedAttack` PGD docstrings, `VjpOracleNets`
    "the verified path", `ViTRender` "verified-faithful". [J4]
(g) **Float tier.** `FloatSubnormalBridge` ("closed", "stay normal"), `dot_close`/`sum_close`
    ("every association" → left fold), the whole-net-fold promises and dead `r34_float_close`,
    Muon "hardware computes the optimum" (the tuned quintic fails the hypothesis). [C1–C4]
(h) **Certificates.** `hp` justified by MC estimates; "NO smoothing-side hypotheses left";
    scorecards "prove 92/100" / "certifies 34/100" (8 per radius proved); "the certified network
    is the deployed one" (IBP conv); "strictly stronger" pixel-L2 (it is weaker); Schatten-4
    "any spread" → rank ≥ 2. Mostly in `scripts/certs/*.py`. [F1–F5]
(i) **Descent rungs.** `SgdDescentCnn/Cifar` "off-the-kink", "non-vacuous", "EVERY parameter";
    float rungs "one binary32 SGD step … exactly as the rendered trainer computes it". [D1, D2]
(j) **Wrong facts.** `BatchNorm.lean:75,106` (variance lemma "cannot" — it is at :145) and the
    sync-BN docstrings still describing an E[x²] exchange; `Residual.lean:54` ("never smaller" is
    false); `vitFull` "full ViT"; `MnistCNN` "trained weights, real test input" (a 6×6 toy);
    `transformerTowerHasVJPMat` instances; ConvNeXt MMR "only whole-net tie with no smoothness
    condition". [D3, D4, H2, I, B5]

## 3. Stale references, counts, and missing docstrings

Mechanical, low-risk; one commit per slice. Every `stale` entry in the ten reports, e.g.:
dead names (`sdpa_back_{Q,K,V}_correct` ×5 files, `resnet34TrainStepFaithfulV`,
`r34InputGrad_eq_resnet34_vjp`, `mnv2PaperInputGrad_eq_…`, `adamGraph`, `bnForward_lb`,
`conv_bias_grad`, `cnnVerifiedHasVJPAt`, `SHlo.maxPool3s2F`); counts (22 → 23 LN positivities ×4,
~60 → 38 relu clauses, twelve → ten ReLU kinks, "five" → seven stride-2 sites, eleven →
seventeen β = 0, five → nine mnv4in twins); "future work" that is proved further down the same
file (Attention, SgdDescentMlp, SE, MaxPool3s2, MuonGeometry, `mlp_w2_step_float_close`);
scalar-LN leftovers in ConvNeXt; per-example BN in ResNet34FullB; per-replica wording in the
MNv2/MNv4 single-device ties; R50 "SKELETON"; `initKind 0` He fan-in; the draft "… wait, … = 12"
at `EfficientNetRender.lean:600`.

Missing docstrings to write: `HasVJP` (the central contract), `dense`, `relu`, `relu6`, `softmax`,
`crossEntropy`, `mlpForward`, `Layer` (constructors are `--` comments doc-gen4 drops), `NetSpec`,
`TrainConfig`, `cifarCnnBn8*`, `cnnBackGraph_faithful`; `LeanMlir/README.md` module list
(`Pong`, `FloatFmt`).

## 4. Process narrative (scope set by D4)

136 Lean files carry ⭐/⛔/⚠ markers (2,440 lines); 70 carry dates. Module docstrings also cite
tiers/phases/§ numbers of planning docs, "4b left open", "after the audit". The rubric says
purpose, what it feeds, what a reader needs — history goes to commit messages.

## 5. Statement changes (decided per item, D1)

Each of these makes a §1/§2 overclaim true instead of rewording it:

| Gap | Lean change | Size |
|---|---|---|
| B0 tie at 10 classes | `nCls` binder through `B0Weights` / `efficientnet_net_tiedG` | small–medium |
| ViT tie at 10 classes | `nCls` binder on `vitTinyInputGradB_eq_vitTiny_vjp` + chain | small–medium |
| small-CNN capstones omit W₃/W₄ … | add the dense-head conjuncts | small |
| MaxPool all-pairs-distinct | live-cell predicate (distinct only where it matters) | medium |
| MNv4 / ConvNeXt / ViT no `*CotIn_eq_vjp` | prove them (MNv2/R34 templates) | large |
| `ViTRender.lean:579` hardcodes `ty [10]` | use `nClasses` (correctness lead) | small |

## Decisions (answered 2026-09-26)

D1. §5: **prove all five rows** (nCls for B0 + ViT, small-CNN dense heads, MaxPool live-cell
    predicate, `*CotIn_eq_vjp` for MNv4/ConvNeXt/ViT). Consequence for ordering: prose that one of
    these rows will make true is NOT reworded down in §1/§2 — it is left for the landing of that
    row, which rewrites it to the new statement. Everything else is reworded to the current
    statement as planned. The ViT `ty [10]` fix rides with the ViT nCls row.
D2. **Regenerate all 18** ConvNeXt artifacts after the generator fix; regenerate
    `verified_mlir/MANIFEST.md` (`gen_mlir_manifest --check`) in the same commit.
D3. content.tex R50 section 6347–6409: **out of scope** — a separate book pass. §1(k) keeps only
    the non-R50 stale items; content.tex:5309 (the retired 76.66% run) goes with the R50 pass.
D4. §4: **full sweep**, one mechanical commit after §1–§3, markers and dates in all 136/70 files.
D5. Landing (default, not asked): branch `doc-pass` off origin/main, commit per reviewed theme,
    fast-forward only; each commit and each push needs the user's word. (The first attempt's branch
    was deleted with the back-out.)

## Order

1. §1 minus the §5-dependent sites and minus R50, one theme per review: (j)+(l) numbers and the
   formula first (pure re-counts, the protocol's warm-up), then (a)+(b)+(c) the landing page,
   then (d)+(e)+(g) the verification appendix, then (h)+(i)+(k) the chapters.
2. §2 and §3, theme by theme (same exclusion).
3. §4 full sweep.
4. §5 rows, smallest first: small-CNN dense heads → nCls B0 → nCls ViT (+ `ty [10]`) → MaxPool
   live-cell → `*CotIn_eq_vjp` (MNv4, then ConvNeXt, then ViT). Each lands with its prose.
5. ConvNeXt generator fix + regenerate the 18 artifacts (D2) — may go with §2(e).

## Declined / deferred

* R50 book section and content.tex:5309 — separate R50 book pass (D3).

# book_xrefs.md — cross-references that assert instead of point

Started 2026-09-23, the day §5.7's verified ResNet-34 row became the sync-BN run. Two sentences
in other chapters described that section ("§6.5 is the pair that isolates the BatchNorm group",
"§5.7 carries the same fourfold difference") and both were false the moment the run landed.
LaTeX checks that a label exists; nothing checks that the sentence around `\S\ref` is still
true. The user's rule, same day: cross-chapter refs of that kind are "the code form of
instantly out of date" — a ref may NAME where something lives, never ASSERT what it says.

`scripts/book_xrefs.py` is the census: every `\S\ref` / `Chapter~\ref` / `Appendix` / `Figure`
ref and every cross-chapter `Theorem~\ref`, grouped by target, each with the sentence it sits
in, so the claim and the target can be read side by side. `--summary` gives one line per target.

## 0. Rules for this thread

* Fine: "the trainer of §X", "Chapter 4's BatchNorm", "see Appendix C", a proof step citing a
  theorem by its title or restating that theorem's own statement (labels do not drift).
* Not fine: a number ("the floor §5.7 set, ±0.39"), a result ("landed 0.10 apart"), a
  characterization ("§X is the pair that isolates it", "as §X built", "for the reason given
  there"), a promise ("Chapter 5 was going to assume"), or a shape claim ("closes with").
* A number is stated where it is measured. Elsewhere it is restated with its own source
  (a `runs/…/RESULTS.md` path, a log) or dropped — never carried through a ref.
* A retired run is not mentioned. No "an earlier run landed 74.06", no "ran both ways".
* A generated figure's caption carries generated numbers: `figures/depgraph/book.tex` now
  `\gdef`s `\depgraphCitations`, `\depgraphCitesTensor`, `\depgraphCitesCnn` and the Figure C.1
  caption uses them. Same pattern for any other count that a script already computes.
* Editing a results section: `grep -n 'ref{<its label>}'` and re-read every citing sentence,
  the same discipline `per_replica_identity_gates` applies to artifacts.

## 1. The audit (2026-09-23) and what it found

341 refs to 90 targets, read by six checkers (one per target range, each reading a target once
and judging every sentence that points at it). 100 were bare pointers (97 fine, 3 dangling);
241 asserted something about the target: 223 true that day, 12 stale, 6 doubtful.

All 21 fixed the same day, staged with the §5.7 update:

| where | was | now |
|---|---|---|
| front matter "Target: ViT", §9.1 lead-in | ViT "the first and only chapter" to use the matrix kit; "every earlier chapter routes around it" | the kit is proved in §9.1; the row-wise lift is in every batched certificate (`thm:*_sync_tie` `\uses thm:rowwise_has_vjp_mat`), which the generated Figure 1.1 / C.1 already showed |
| Figure C.1 caption | 308 / 622 hand-carried | generator macros (312 / 626 today) |
| §5.5 | "the `bare` arm of §5.6 at 83.20 ± 1.57" | no such arm; "what each one is worth is measured in §5.6" |
| §5.7 lead-in | "§5.9 is the table that says which is which" | the paper/polish split stated locally |
| §5.10 table | "run (§5.9)" | §5.8, where the run is |
| §6.5 | "the way Chapter 5 built ResNet-50's" | §5.7 built ResNet-34's (the only head swap shown) |
| §6.7 | `.fusedMbConvNB` (§6.7) | the ref moved to "MobileNetV4's stage 0" |
| Bestiary WRN | "same `.residualBlock` as Chapter 5" | "the residual block of Chapter 5" (ch 5 spells `.residualStage`) |
| §4, §7 ×2 | dense theorems credited to Chapter 2 | `Theorem~\ref{ax:pdiv_dense}` / `thm:dense_has_vjp` (ch 1) |
| §9.6 | "first network since Chapter 2's dense-only graphs to reproduce exactly" | ch 2 never claims it; the XLA-algorithm reason stands alone |
| §6.7, §7.7 | tables "closing" Chapters 7 / 8 | "the recipe tables of" / "recipe table is" |
| §4.6, §6.1, §1.x, Bestiary GW, §6 proof, §9.1 | the six doubtful (forward promise, unsourced 98.46 top-5, inference-statistics clause, 0.83M "as ch 4 spells it", "three steps" vs four, "both halves are the same fact") | claim dropped, pointer kept |

## 2. The pass: the 220 claim-bearing refs still standing, by kind

Counts are from the audit; rerun `scripts/book_xrefs.py` for today's list. Targets with the
most incoming refs: `chap:residual` 35, `chap:tensor` 32, `chap:mlp` 20, `chap:cnn` 17,
`chap:bn` 15, `chap:depthwise` 11, `sec:r34_pjrt` 9, `chap:se` 9, `sec:r34_ablation` 8,
`app:verification` 8.

(a) Proof steps that restate the cited theorem's statement or discharge its hypothesis
    (~60, nearly all against `ax:pdiv_*`, `thm:vjp_comp`, `thm:biPath_has_vjp`, the conv2d
    VJPs). Keep. A label is stable; the statement is what the proof needs.

(b) "Theorem T of Chapter N" — the chapter named as the theorem's address (~20, chapters 2–3,
    5–9 and the Bestiary, almost all at `chap:tensor`). Cite the label, drop "Chapter N's":
    `Theorem~\ref{thm:dense_has_vjp}` says where it lives.

(c) Numbers restated through a ref (~25). Chapter 1's 92.10 % / 224 ms / 604 ms compile cited
    from chapter 2; §5.4's 89.99 ± 0.32 cited from chapters 6, 7, 8, 9 and the Bestiary; §5.7's
    Wilson floor cited from §5.8, §6.5, §7.6, §8.6, §9.6 (each already recomputes its own ±;
    drop "the floor §5.7 set" and keep the local number); run-it numbers cited across
    chapters (§6.1 → §5.1, §7.6 → §7.1, §8.x → §8.1, §9.6 → §9.1). Rule: local number with
    its source, or the ref alone.

(d) Characterizations of another chapter's method or argument (~40): "the pattern every later
    chapter reuses", "as §X built MobileNetV2's", "for the reason given there", "§X is about why",
    "Chapter 6 gave the reason", "where this book first had to take spread seriously",
    "the same sub-proportional scaling §X sees". Drop the characterization, keep the pointer,
    or say the thing locally in one clause.

(e) Status restated in the Track-4 appendix table (`app:getting_started`, the job table):
    "rendered, not run", "untrained", A2/A1 "neither tier has been run yet". These go stale on
    the day each run lands. Either the table carries only target / recipe / job name and the
    chapter carries the status, or the status column is generated from `scripts/jobs/*.conf`
    plus each job's `.epoch` file.

(f) Front matter ("How this book is organized", the reading-order targets). It describes
    every chapter and cannot avoid claims about them. Keep, but it is re-read whenever a
    chapter's results section changes; the ViT paragraph was the stale one this time.

(g) Whole-book claims made in one chapter. §6.5's "every BatchNorm pair in this book but
    ResNet-34's was trained under that asymmetry" is true today and false the day the MobileNetV2
    sync-BN leg lands (`global_bn_verified.md` §3.5). Each such sentence is a debt the next run
    pays; prefer "this pair" statements.

## 3. Order of work

One target chapter per commit, reading its block of `scripts/book_xrefs.py` output: chapter 1
(32 refs, mostly (b)), chapter 5 (35, mostly (c)/(d)), then 2, 3, 4, 6–9, the appendices.
Rebuild the PDF after each (`cd blueprint/src && latexmk -xelatex -interaction=nonstopmode
-output-directory=../print print.tex`, ~1 min, 0 `^!` lines in `print.log`). The near-term
risk is new text: the MobileNetV2 / EfficientNet-B0 / ResNet-50 sync-BN legs each rewrite an
ImageNet section, and each of those sections is cited by the ones after it.

## 4. Not this doc

* The Wilson-floor paragraph's home. It is stated once in §5.7 and recomputed in each later
  chapter; whether the shared sentence moves to chapter 1 is a book-structure call.
* A lint. `book_xrefs.py` could flag sentences around a ref that contain a digit, a `%`, or
  "showed / measured / landed / carries / is the"; advisory at best, and not built.

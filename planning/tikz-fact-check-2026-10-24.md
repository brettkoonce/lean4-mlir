# TikZ fact-check — figures and the prose around them (2026-10-24, at 7ba70a26)

All 60 TikZ pictures were checked: the 40 inline in `blueprint/src/content.tex`, the 13 generated
dependency graphs in `blueprint/src/figures/depgraph/`, and the 7 standalone pgfplots files. The
checks covered:
- architecture diagrams against `NetsCore.lean` and the emitted MLIR;
- data plots against `runs/` and `jax/runs/` logs;
- dependency graphs against a fresh run of `scripts/book/blueprint_depgraph_tikz.py`.

**Every plotted data point that can be traced matches its log.** The problems are:
- wrong labels in the architecture diagrams;
- prose that overstates what a figure shows;
- dependency graphs that predate dc49fea3 and d6085da4.

Some items were already in `book-fact-check-2026-10-18.md` and are still open; they are marked
**(10-18)**.

Line numbers are at 7ba70a26. Fix bottom-up, or re-grep the quoted text. All book copy needs a chat
preview before it's committed.

---

## 1. Wrong

- [x] **ResNet-34 diagram, stem pool** (`:6554`).
  - The figure says `maxPool $2\times2$`.
  - The code is `.maxPool 3 2` (`NetsCore.lean:462`), and the MLIR is a 3×3 stride-2 window with
    pad 1 (`resnet34_fwd.mlir:49`).
  - The listing below the figure already says 3×3.
  - Fix: `maxPool $3\times3$/2`.
- [x] **ConvNeXt-T diagram, stem** (`:11911`).
  - The figure says `\textbf{ConvBN} patchify`.
  - The code is `.conv 3 96 4 4, .layerNorm 96` (`NetsCore.lean:1013`), and the prose right after
    says there is no BatchNorm.
  - Fix: `conv + LN patchify`, drawn with the `norm`/conv style, not `convbn`.
- [x] **ConvNeXt-T diagram, head.** The figure draws GAP → dense. The code is GAP → LN → dense
  (`NetsCore.lean:1023`). Add the LN box.
- [x] **Operation inventory** (`:15228`, `\opchiprowD{cTe}{\PA}…`).
  - `tokenEmbed/r` is listed as a Part 1 op, but no Part 1 net uses it; ViT uses `patchEmbed`.
  - It first appears in TinyGPT, in Part 2.
  - Fix: move it to the `\PB` row.
- [x] **ResNet-34 ablation prose** (`:7075`).
  - The text says "three times the next-largest knob".
  - Augmentation's −7.33 is about 1.5× cosine's −5.00. It is 3× only against AdamW (−2.27).
  - Fix: "half again the next-largest knob (cosine, −5.00) and three times AdamW's".
- [x] **Dependency graphs are stale** (`blueprint/src/figures/depgraph/`).
  - Five statements added in dc49fea3/d6085da4 are missing:
    - `optimizer_tails_faithful` (ch4);
    - `lamb_clip_faithful` and `accum_mean_step` (ch5);
    - `rmsprop_faithful` and `ema_faithful` (ch6).
  - The statement counts in `book.tex` are wrong: Ch 4 says 16 (should be 17), Ch 5 says 12 (14),
    and Ch 6 says 16 (18).
  - `spines.tex` differs in layout only; every other graph is identical.
  - The `\uses` edges are correct: CI (blueprint run 40042763746) passed both the declarations check
    and the `\uses` check.
  - Fix: rerun the generator (it needs pygraphviz) and commit the ch4, ch5, ch6, book and spines
    outputs.

## 2. Misleading

- [x] **R34 paths prose** (`:7351`).
  - The text says "The verified path leads by one to three points through the first third".
  - Per `runs/2026-10-20-r34-bot-{jax,verified}/full.log`, over epochs 1–30 the verified path
    trails at 11 epochs: e3 −4.38, e7 −1.20, e14 −1.62, e19 −1.01.
  - Its leads of +1.5 to +2.7 are scattered (e5, 9, 13, 15, 16). The mean lead is about +0.2.
  - Fix: "neither path leads for long through the first third; the gap swings up to four points
    either way and closes as the cosine anneals".
- [x] **R34 optimizer figure, right panel** (`:7045`–`7047`, `fig:r34_optimizer_matched`)
  **(10-18)**.
  - No per-epoch log exists for any of its three curves.
  - The prose at `:7055` says SGD "finishes slightly ahead", but the plotted SGD run ends at 87.82
    against momentum's 88.13. The +0.403 is the five-seed paired mean, not this run.
  - Fix: find or regenerate the logs and cite them, or drop the panel.
  - Also add "on the paired mean; the single runs plotted here finish in the other order".
- [x] **TinyGPT causal mask caption** (`:17366`, comment `:17329`) **(10-18)**.
  - "The sampler's own first eight characters" are `ROMEO:\nI`, but seven of the eight are the
    prompt, and the cited run's 8th character is `O`.
  - Fix: "the prompt and the first sampled character", with the real character.
- [x] **ResNet-34 inset.** It draws only an identity skip, but the first block of stages 2–4 uses a
  1×1 projection shortcut. Draw a dashed projection or say "identity shown".
- [x] **Operation inventory, chapter colours** (`:15224`–`15236`).
  - `bottleneck` is in the Part 2 row, but ResNet-50 trains and proves it in Ch 5. Move it to
    `\PA`, coloured green.
  - `$1{\times}1$ conv/y` should be `/g`, since it is first used in Ch 5.
- [x] **CNN head-width caption** (`:4123`) **(10-18)**.
  - The caption doesn't say the grid ran 10 epochs. That is why d=512 plots at 98.49, against the
    canonical 12-epoch run's 98.77.
  - Fix: add "10 epochs".
- [x] **MNIST epoch-1 range** (`:3268`) **(10-18)**. The text says "92.5 and 94.8%", but the logs
  show 91.98–96.42.

## 3. Nits

- [x] `:3040` "$755\times$" should be 756×.
- [x] The MLP width plot draws symmetric whiskers, but the caption calls them Wilson intervals,
  which are asymmetric. Use the real Wilson bounds (`error bars/explicit` with separate plus and
  minus).
- [x] `:4133` "four runs cluster and one falls away": two runs are low at d=512.
- [x] `:7350` "three at the worst on either path": the actual worst swings are 2.89 and 3.39.
- [x] `:9253`, `:9450` "every ten epochs": the curves also plot e1.
- [x] `:11124` EfficientNet "every ten epochs" / "per epoch": the data is e1–10, then every 5th.
- [x] `:14902` ViT "under two tenths by epoch 200": this only holds from about e230.
- [x] The median 98.785 is printed as 98.78. Decide on round-half-even or say 98.79.
- [x] The bf16 `_s1_resumed.log` files are not documented in `runs/RESULTS.md`.

## 4. Suggested additions (not errors)

- [x] `:3741` (the CNN step theorem): "for odd kernels" leaves the reader wondering about even
  kernels. Add a pointer to `:3500` (the `padOdd` paragraph) and to ConvNeXt's even-kernel tie
  (`:11785`).

## 5. Outside the book

- [x] Seven standalone pgfplots files are not `\input` anywhere:
  - `runs/smoothing_frontier.tex` and `runs/lipschitz_cnn.tex`, whose generators were deleted in
    e4216436;
  - five `jax/runs/*/*_val_curve.tex` files. Of those:
    - `jax/runs/enet_b0_imagenet_bf16_350ep/enet_val_curve.tex` has a header saying it is the Ch 7
      figure, but it plots an older run;
    - the 80- and 90-epoch files are superseded;
    - the ConvNeXt 300-epoch file duplicates the inline figure.
  - Fix: delete them, or move them to `archive/`.
- [x] The `mobilenetv2Verified` and `efficientnetVerified` blurbs (`NetsCore.lean:787`, `:885`)
  mention a head LayerNorm that neither net has.
- [x] CI: the blueprint workflow now installs graphviz. Add a step that regenerates the depgraphs and
  fails on a diff, so §1's staleness can't happen again.

## Checked and fine

- **Architecture diagrams:** MNIST linear, MLP, CNN, cifar8, MobileNetV2, EfficientNet-B0 and ViT.
- **ImageNet curves:** all 14 inline figures match their logs at every point.
- **MNIST and CIFAR plots:** every value that can be traced matches.
- **Depgraphs:** ch1–3, ch7, ch8 and all three ch9 graphs.

## Applied 2026-10-27 (branch `tikz-fixes`, from main at 34a71da8)

Every box above is ticked. Three were already closed on main before this pass: the ResNet-34 stem
pool (the figure reads $3\times3$, /2), the ablation prose ("half again the next-largest knob"),
and the dependency graphs (regenerated in 69675dca, with the five statements and the counts).
`tokenEmbed` was already in the Part 2 row and `bottleneck` is no longer in the inventory, so
those two moved nothing. Three were closed differently from the fix proposed:

- the optimizer figure's right panel stays (its source comment already says one run per
  optimizer, shape not level); the prose now says the +0.403 is the paired mean and that the
  plotted single runs finish in the other order, 87.82 against 88.13;
- the ResNet-34 inset keeps the identity skip and the caption says so, naming the $1\times1$
  projection that opens stages 2--4, rather than a dashed second arrow;
- the median is printed as 98.785 with the two middle runs named, rather than rounded.

Six of the seven orphaned pgfplots files are under `historical/pgfplots/` (the four `jax/runs`
RESULTS files point there); `runs/smoothing_frontier.tex` went with the ConvNeXt-T certificate in
f889452d, so it is deleted, not moved; the `*_s1_resumed.log` note is in the bf16 sweep's RESULTS.md; the blueprint
workflow regenerates the depgraphs and fails on drift. The R34 "first third" sentence reads from
`runs/2026-10-20-r34-bot-{jax,verified}/full.log`: ahead at 18 of 30 epochs, behind at 12, gap
$-4.37$ to $+2.66$, mean $+0.18$; the worst mid-schedule swings are 2.89 (JAX, e45) and 3.39
(verified, e15). `print.tex` compiles; the five book gates pass.

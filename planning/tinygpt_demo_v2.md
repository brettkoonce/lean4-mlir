# tinygpt_demo_v2.md — tinyGPT demo, second pass (Shakespeare polish + TinyStories)

Goal, in two halves. **Part I**: finish the char-level Shakespeare
demo as it stands — validation perplexity (the val split has never
been read), the LR schedule the config pretends to have, a modest
scale-up, and real sampling controls. All cheap; compute is minutes.
**Part II**: the TinyStories upgrade — small-vocab BPE, longer
context, ~10M params — which turns "recognizably Shakespearean
fragments" into "fluent little stories," the qualitative jump that
made the TinyStories paper famous (Eldan & Li 2023: even 1–10M-param
models tell coherent stories when the corpus is simple enough).
Part II includes the one real primitive decision: bridge via
in-graph one-hot, or land a true gather/scatter.

Prerequisite reading: `planning/tinygpt_demo.md` (v1 plan + results;
Phases 1–3 shipped in one session).

## Where v1 landed (recap, one paragraph)

A 212K-param char-level GPT (V=65, T=64, D=64, 2 heads, 4 blocks,
causal mask) trains on tinyshakespeare and samples text with real
character names and dialog structure. The load-bearing tricks: token
embedding is a **host-side one-hot** (`F32.tokenOneHot`, `[B, V·T]`
f32) into a learnable `[V, D]` matmul inside `tokenPositionEmbed`;
the `lmHead` emits `[B, V, T, 1]` so **per-token CE rides the
`useSeg` per-pixel-CE path unchanged**; `transformerEncoder` grew
`causalMask := true`. 10K steps / 11 min lands at 1.45 nats/char
(2.10 bits/char) vs the bigram baseline's 3.55 bits. Trainer and
sampler live in `MainTinyGptShakespeare.lean` (own loop, not
`trainGeneric`); bigram baseline validates the data path.

## What's open (v1's own list, audited 2026-07-03)

v1's "what's left" ranking, plus code-audit findings:

1. **`val.bin` (111K tokens) has never been read** — no validation
   perplexity, anywhere. Train loss is the only signal (the same
   gap every other v2 doc starts by closing).
2. **The LR schedule is decorative.** `trainConfig` sets
   `cosineDecay := false, warmupEpochs := 0` and the custom
   `runTinyGptTrain` loop passes a constant `lr` regardless — the
   fields exist but aren't threaded.
3. **Sampling is temperature-only** (plus greedy at `t ≤ 0`); v1
   planned top-k/top-p ("each a 1-line modification") — never
   shipped. The xorshift seed is derived from the step index, so
   repeated runs are identical with no way to vary them.
4. **The left-pad wart**: short prompts pad the context with token
   id 0 — which is a *real character* in the sorted 65-char vocab,
   not a pad symbol. The model was never trained on that prefix
   distribution. Mostly harmless at T=64; worth fixing before
   anything is measured against it.
5. Wider/deeper model, longer context — v1's #1 and #2 levers,
   untouched.
6. **No RESULTS.md row** — the bits/char table lives only inside
   the planning doc.

# Part I — improve the demo as-is (char-level, no new primitives)

**Status 2026-07-03: Phases 0–2 DONE** (one session, klawd/CUDA).
Val bits/char + cosine/warmup + pad fix + top-k/top-p/seed + `suite`
subcommand all shipped in `MainTinyGptShakespeare.lean`; results in
RESULTS.md §tinyshakespeare. Findings: (a) the cosine schedule alone
improved nano's train loss 1.45 → 1.386 nats at matched steps;
(b) nano val = **2.27 bits/char** — the demo's first held-out number;
(c) **Gate B failed informatively**: tiny (1.2M) overfits from
~step 3500–4500 (10K-step run: train 1.26 bits vs val 2.78) and even
annealed to its val minimum only matches nano (2.30 vs 2.27) while
producing visibly more fluent samples. At 1M chars the corpus, not
capacity, is binding — which is Part II's whole argument. Follow-up
if wanted before TinyStories: a wd sweep (1e-4 → 1e-2/5e-2) as the
cheap dropout substitute.

## Workstream A — metric + schedule (do first)

1. **Validation bits/char** (~half session): every N steps, run the
   eval forward over `val.bin` in fixed chunks, report mean
   nats→bits/char. The eval vmfb already compiles at batch 1 for
   sampling; compile at an eval batch instead and reuse. Gives the
   overfitting answer (212K params vs 1M tokens is probably fine;
   the Part-I scale-up vs 1M tokens is the real question — v1's
   `weightDecay := 1e-4` is currently an act of faith).
2. **Thread cosine + warmup through the custom loop** (~15 min):
   the loop already owns `lr` per step; compute the schedule
   host-side. (Or migrate the trainer onto `trainGeneric`'s seg
   path — bigger refactor, only worth it if the token modality
   grows more trainers; default: keep the custom loop.)
3. **Fix the pad wart** (~15 min): generate from the prompt length
   upward (feed only filled positions by masking loss-side, or
   simply prime generation by consuming the prompt
   autoregressively from position 0 instead of left-padding).

Gate A: a train-vs-val bits/char curve for the v1 checkpoint
config. Everything after gets measured against it.

## Workstream B — the scale-up run (v1's own #1/#2)

One spec bump, minutes of compute: **D=128, 4 heads, 6 blocks,
T=128, mlp 512** (~1.6M params). Expected per v1's estimate:
≈1.7–1.8 bits/char and visibly longer coherence (the ~50-char
cliff is partly the T=64 window). Keep the 212K nano config as the
second cell — the "more params → better text" pairing v1's cell
plan wanted. Report both in RESULTS.md with the bigram row.

Gate B: val bits/char improves and samples hold coherence past one
sentence. (If val diverges from train early, revisit weight decay
before blaming capacity — that's what Workstream A's curve is for.)

## Workstream C — sampling polish

Top-k and top-p (~1 hr, host-side in `sampleToken`), a real
`--seed` arg, and a small fixed **prompt suite** (4–6 prompts ×
fixed seeds) regenerated after every training change — the LM
version of the DDPM fixed-seed grid / CAM deletion curve: samples
you can diff across runs instead of vibes. Refresh
`blueprint/src/figures/tinygpt/tinygpt_sample.txt` from the suite.

# Part II — TinyStories

**Status 2026-07-03: Phases 3–5 DONE (Gate F met early).** Shipped:
`preprocess_tinystories.py` (4096-vocab byte-level BPE → same int32
stream the Shakespeare loader reads, 50.3M train / 4.86M val tokens,
`<|endoftext|>`=0); the **in-graph one-hot** primitive
(`tokenPositionEmbed idsInput`, Option 1) — validated byte-identical
to the host one-hot on nano (`nano-ids` reproduces nano's loss
sequence exactly), killing the ~134 MB/step upload at V=4096;
`MainTinyStories.lean` (8.49M params: V=4096, T=256, D=256, 8h, 8
blocks) + `scripts/tinystories_decode.py`. **Gate F result**: a
checkpoint at only step ~1000/12000 already produces a fully coherent,
grammatical children's story — named character held across the whole
piece, complete narrative arc — the paper's signature effect, inside
the verified pipeline (sample in `blueprint/src/figures/tinystories/`).
Val fell 8.4 → 3.06 bits/tok by step 3000; the run was stopped at
step ~3200/12000 (2026-07-03) — `_params.bin` is checkpointed every
500 steps, so resume by re-running `tinystories train 12000 …` for a
sharper final model (or just sample from the saved checkpoint).
Option 2 (true gather/scatter) stays deferred — Option 1 carried the
whole path with zero new VJP machinery, as predicted.

## Why TinyStories, in one paragraph

Char-level Shakespeare caps out at "recognizable fragments" because
Early-Modern-English drama is *hard* relative to model size.
TinyStories (≈2.1M GPT-generated children's stories, ~1–2 GB raw
text, few-hundred-million tokens) was built to be the opposite:
simple vocabulary, short sentences, complete narrative arcs — so
small models produce *fluent, coherent, grammatical* stories. It's
the ideal corpus for this framework's scale: the demo output jumps
a category ("the verified pipeline trains a model that tells
stories") without needing more than overnight compute on mars.

## Workstream D — tokenizer + data (Python-side, ~1 session)

1. **Small BPE, vocab ~4K**, trained on the corpus
   (`preprocess_tinystories.py`; byte-level BPE, minbpe-style —
   no HF runtime dependency, just merges). Deliberately small:
   keeps the embedding/head matmuls cheap (see E) and the
   tokenizer auditable. Emit the same
   `train.bin`/`val.bin`/`vocab` format the Shakespeare loader
   already reads (token stream as int32; loader is
   vocab-size-agnostic).
2. **Lean-side decode** is a table lookup (id → bytes), same as
   now. **Encode** (for prompts) is greedy merge application —
   ~100 lines of Lean, or accept Python-encoded prompt files for
   the first cut. Include an `<|endoftext|>` token between stories
   (the sampler can stop on it — v1 had no EOS concept).
3. Start with a **100–200M-token subset**; the full corpus is an
   ambition knob, not a requirement (the paper's small models are
   trained on far fewer tokens than an epoch of modern LLM
   budgets).

## Workstream E — the embedding primitive decision

The host-side one-hot dies here: at V=4096, T=256, batch 32 it's a
134 MB f32 upload *per step*. Two ways out:

- **Option 1 — in-graph one-hot (the bridge, ~1 session).** Change
  the model input to `[B, T]` int32 and build the one-hot *inside*
  the MLIR with the iota+broadcast+compare+select pattern the
  per-pixel-CE block already uses for labels — then the existing
  `[B,T,V]·[V,D]` matmul and its backward carry over untouched (the
  one-hot is a constant w.r.t. params; `dE = onehotᵀ·dOut` is the
  existing matmul VJP). Kills the host transfer entirely. The FLOP
  waste (V·D·T per sample) is ≈ the *same cost as the unavoidable
  lmHead matmul* (D·V·T), so at V≈4K this is merely 2× head-cost,
  not a bottleneck. No new VJP, no new proof surface.
- **Option 2 — true gather/scatter (the primitive, ~1–2
  sessions).** `stablehlo.gather` forward, `stablehlo.scatter`
  (add) backward, Layer kind + FD test + the usual pidx audit.
  The proper answer at V=16K+, and scatter-add is a gateway op
  (embedding bags, MoE routing, sparse updates). But it's new VJP
  machinery with a genuinely fiddly backward.

**Recommendation: Option 1 first.** It de-risks the entire
TinyStories path with zero new derivative machinery and keeps the
"everything reduces to already-proved VJPs" story intact; Option 2
becomes worthwhile the day a 16K+ vocab or MoE work shows up, and
can be validated *against* the running Option-1 model when it does.

## Workstream F — the TinyStories model + runs

- Spec: **V=4096, T=256, D=256, 8 heads, mlp 1024, 8 blocks** —
  ≈8–10M params. Same three-Layer NetSpec shape as v1
  (`tokenPositionEmbed` variant from E → `transformerEncoder` →
  `lmHead`); the `useSeg` loss ride is vocab-agnostic.
- Compute: batch 32 × T 256 = 8K tokens/step; a 100M-token pass ≈
  12K steps. At a guessed 100–300 ms/step on mars this is an
  evening, not a week; 2–3 passes overnight. (klawd works too with
  the usual duty-cycle caveat.)
- Eval: val bits/token + the fixed prompt suite ("Once upon a
  time, there was a", "The little dog saw a" …). Optionally the
  paper's qualitative axes (grammar / consistency / plot) judged
  by eye across checkpoints — no GPT-judge machinery, just the
  suite printed side by side.
- Cells: keep Shakespeare-nano + Shakespeare-tiny from Part I, add
  `tinystories-10m`. Three rungs: same architecture family, three
  data/scale regimes.

Gate F: samples are complete grammatical mini-stories with stable
characters over ≥3 sentences — the paper's signature result
reproduced inside the verified pipeline. Val bits/token reported
alongside (expect roughly ~1.5–2.5 bits/token at this scale;
the sample quality is the headline, the number keeps us honest).

## Running it (reproduce Part II end-to-end)

```bash
# 1. Data: download (~1.9GB) + train a 4096-vocab BPE, encode 200M
#    chars → data/tinystories/{train,val}.bin (~50M/4.9M tokens).
#    Needs Python `tokenizers` + `numpy` in .venv. One-time, ~10 min.
bash download_tinystories.sh
python3 preprocess_tinystories.py 4096 200000000

# 2. Build the exe.
lake build tinystories

# 3. Train. 8.49M params, T=256, V=4096. Checkpoints _params.bin every
#    500 steps (safe to interrupt/resume-by-rerun), logs val bits/token.
#    On klawd (RTX 4060 Ti, CUDA) ≈2.7 s/step host-bound → ~9h for the
#    full 12K; coherent stories already emerge by step ~1000 (~1h).
#    Pick a GPU with CUDA_VISIBLE_DEVICES; args: [steps] [batch] [lr_x1e4].
CUDA_VISIBLE_DEVICES=0 .lake/build/bin/tinystories train 12000 32 30

# 4. Sample. BPE encode a prompt, generate ids, BPE decode back to text.
python3 scripts/tinystories_decode.py encode "Once upon a time, there was a little"
CUDA_VISIBLE_DEVICES=0 .lake/build/bin/tinystories sample 200 80 40 95 1 > /tmp/gen.txt
python3 scripts/tinystories_decode.py decode "Once upon a time, there was a little" < /tmp/gen.txt
#   sample args: [n_toks] [temp_x100] [topk] [topp_x100] [seed]
```

Char-level Part I (Shakespeare) is self-contained and faster
(minutes): `bash download_shakespeare.sh && python3
preprocess_shakespeare.py`, then `lake build tinygpt-shakespeare` and
`… train {nano|tiny} 10000`, `… sample {nano|tiny} …`, `… suite …`.
The in-graph one-hot equivalence check is `… train nano-ids 200`
(losses match `… train nano 200` byte-for-byte).

## Sequencing

```
Phase 0 (½ session):            A  val bits/char + LR schedule + pad fix (Gate A)
Phase 1 (½ session + minutes):  B  1.6M-param scale-up, two-cell table (Gate B)
Phase 2 (½ session):            C  top-k/top-p + prompt suite + figure refresh
Phase 3 (1 session):            D  BPE + TinyStories preprocessing
Phase 4 (1 session):            E  in-graph one-hot embedding (Option 1)
Phase 5 (1 session + overnight): F  10M-param TinyStories runs (Gate F)
Phase 6 (later, on demand):     E' true gather/scatter primitive
```

Part I (Phases 0–2) is roughly two sessions with compute measured
in minutes — the cheapest committed core of any v2 doc. Part II is
~3 sessions + one overnight, and each phase is useful alone (the
BPE pipeline and in-graph one-hot are worth having even if the big
run waits).

## Deliverables

- Val bits/char in the trainer log + RESULTS.md table: bigram /
  nano / tiny / tinystories-10m
- Top-k/top-p + `--seed`, the fixed prompt suite, refreshed
  `tinygpt_sample.txt` (and a `tinystories_sample.txt` beside it)
- `preprocess_tinystories.py` + `download_tinystories.sh` (HF
  dataset mirror), vocab/merges artifacts
- The int32-input `tokenPositionEmbed` variant (in-graph one-hot)
  + FD test
- Bestiary/blueprint: GPT entry gains the three-rung scale story;
  v1 planning doc gets a status pointer here

## Out of scope (unchanged from v1, plus)

True gather/scatter until a >8K vocab or MoE demands it (tracked
as Phase 6, not dropped); KV-cache sampling (500-token generations
at T=256 are seconds; revisit only if generation becomes a
bottleneck); dropout in transformer blocks (the codegen pattern
was deliberately punted repo-wide; weight decay + data scale carry
regularization here); instruction tuning / chat formatting; RLHF /
DPO; `transformerDecoder` cross-attention and `mambaBlock`
promotion (both stay shape-only Bestiary entries); GPT-judge
automatic evals (the paper's method, but a distraction at demo
scale).

# next_session_convnext_verified_run.md — the shim fix first, then ConvNeXt's phase-4 half

**Opened 2026-09-16.** Successor to `next_session_enet_verified_run.md`, which closed the same day:
EfficientNet-B0 ran to **76.878 / 93.154 in 73 h 23 m**, and §7 got its phase-4 half. Two jobs here,
**in this order**:

1. **Make the shim loaders survive a long run** — staggered respawn, proven on the R34 bf16 job.
2. **Run `convnext-imagenet-verified` to 300 epochs** and write §8's phase-4 half.

⚠ **MobileNetV4 is deliberately NOT in this sequence** (Brett, 2026-09-16): its phase-4 peer needs a
phase-2 JAX run first. ConvNeXt is the last phase-4 gap in the primary sequence.

---

## 0. What the EfficientNet run established — carry these forward

* **Resume is lossless as of `367bb28b`.** The blob is `[θ|m|v|(G)|E]`, the BN running stats and
  their EMA go in a `<ckpt>.bn` companion, and companion/blob/marker are each written through a temp
  file and renamed. Exercised 8 times on that run with no accuracy step at any boundary.
  ⛔ **ConvNeXt is LayerNorm: it writes no companion, so its resume path is the one that has never
  been tested.** That is R2 in the test matrix below and it is a blocker, not a nicety — every
  planned restart and every thermal rest rides on it.
* **One tf.data loader degrades and the round-robin feed runs at its pace** (`planning/
  shim_loader_health_and_resume_tests.md` §1): 690 → 1,250 s/epoch on EfficientNet, cleared
  instantly by a restart. Mitigated there with `REST_EPOCHS` (six rests, ~10 min total).
* ⭐ **ConvNeXt's pair will be CLEANER than EfficientNet's or MobileNetV2's.** Those two carry the
  BatchNorm statistic-group confound — 64 per replica on the verified path against 256 global in
  JAX, now a book-level `[TODO: revisit BN sharding]` in §7. **ConvNeXt has no BatchNorm at all**, so
  that confound is absent and the comparison isolates the lowerer far better. ▶ Say so in §8; it is
  the strongest lowerer evidence the book will have.
  ⚠ Still not one-variable: drop-path and dropout masks are host-drawn here and JAX-PRNG there —
  a variance source between runs, not a bias.

## 1. ⛔ THE CONF SHIPS THE f32 ARM — the same trap MNv2 and B0 both had

`scripts/jobs/cnx-default-4gpu.conf` sets `LEAN_MLIR_VARIANT=adamdpwxclipdrop` (f32) with
`CKPT_EPOCH_FILE=".lake/build/convnextin_adamdpwxclipdrop_ckpt_xla.bin.epoch"`, while
`verified_mlir/convnextin_adamdpwxclipdropbf16_train_step.mlir` **is committed**. Fix both together
and rewrite the precheck on `13d9b460`'s model: assert the variant string, the bf16 render,
ckpt/variant agreement, shim freshness (`regen_jax_generated.sh box`), residency, the worker count,
stray checkpoints, GPU idleness — **and exe freshness**, which caught nothing here only because the
rebuild was done by hand.

⚠ **First confirm the phase-2 reference's precision.** §6 and §7 flipped to bf16 *because their
references were bf16*; pairing a bf16 verified run against an fp32 reference moves two axes. The
ConvNeXt-T reference is `81.53 / 95.50`, 300 epochs, 76.49 h on this box (`cnxInit`) — find its log
and read the precision off it, not off a recipe table.

## 2. Re-probe throughput on this box before quoting an ETA

The conf's `SHIM_WORKERS=4` and its `ETA=` come from a **2026-08-31** sweep, which predates both
feed fixes (`4a0a2781` determinism OFF, `13d90e68` mimalloc + stale runtime shims). EfficientNet's
re-probe moved its worker count *and* halved the ETA's error.

    for w in 2 4 6 8; do WORKERS=$w ARMS=fed PRECS=bf16 NETS=cnx WARM=200 STEPS=1000 \
      scripts/bf16_probe_3060.sh runs/2026-09-XX-cnx-sweep.tsv; done
    # then one synth arm for the compute floor

⚠ ConvNeXt runs **bs 32 per replica ⇒ 10,009 steps/epoch**, double the 64-batch nets: per-step
figures do not transfer, and neither do per-epoch ones. ⚠ Rank on `mean` and check `mean` vs `p90`,
but do not expect EfficientNet's answer — ConvNeXt's shim is a different augmentation load.

## 3. The shim work — DO IT FIRST

`planning/shim_loader_health_and_resume_tests.md` holds the design and the evidence. Order:
§3a Level 0 (consume whichever loader is ready, so one slow loader costs a quarter of its shortfall
rather than pacing all four) → §3e prerequisites (keep the child process so a loader can be killed
and reaped; seed the loader shuffle, which is hardcoded to 42 today) → §3b staggered respawn.

**Why first:** ConvNeXt is the longest run in the book (~107 h at the conf's current ETA). At the
degraded rate EfficientNet hit, that is days of loss, and a 300-epoch schedule gives the defect
five or six chances to appear.

▶ **Test vehicle: the R34 bf16 job** (Brett's call — it is wanted anyway). ⚠ There is no
`r34-2018-bf16` conf here or on `origin/main`; R34's `default` recipe already *is* the 2018 recipe
(momentum SGD, lr 0.1, batch 256, 90 epochs, flip-only shim), so this means `r34-default-4gpu.conf`
switched to `momdp64bf16` — which is committed and which that conf does not name. R34's shim is
flip-only, so it also answers whether the defect is augmentation-weight-dependent.

**Interim safety net** if the respawn is not ready when ConvNeXt launches: `REST_EPOCHS` every ~40
epochs with `REST_SECS=15`, exactly as EfficientNet ran it.

## 4. Resume tests before the ConvNeXt run

§4 of the shim doc: R1 bit-exact (BN+EMA, `efficientnet-verified-adam` / `emarms`), **R2 bit-exact
on ConvNeXt's own path** (`convnext-verified-adam` / `ema`, LayerNorm, no companion), R3 kill fuzz
across the checkpoint write, R4 the missing-companion fallback, R5 a wrong-size companion, R6 a
short supervisor job with a planned rest running to `COMPLETE`. Both Imagenette exes are stale
(July) — rebuild before trusting them.

## 5. §8's phase-4 half — the shape is now fixed

Copy §7, which copied §6: **side-by-side lowerer table with the differing rows above the rule**
(lowering, Python-in-step, ms/step, per epoch, total — and for ConvNeXt **no BN statistic-group
row**, which is the point), the convergence table, both curves on one plot, and an explicit
statement of what the pair does not isolate. Editorial rule, enforced on §6 and §7: **describe what
the code does, not the run's operational history** — restarts, thermal narration and host-memory
forensics belong in the run's `RESULTS.md`.

⛔ **`content.tex` anchors moved when §7 landed (+101 lines).** Re-derive every line number.
As of this writing: `What it has not done is run.` is **count 1** (ConvNeXt's, ~9325) and
`\subsection*{Phase 4: the verified trainer}` is count 2 (EfficientNet ~8175, ConvNeXt ~9269).
▶ The safe edit is the one used for §7: splice by line range with **asserted boundaries**, e.g.

    python3 - <<'PY'
    L=open(p).read().split('\n'); a,b = <first>, <last>
    assert L[a-1].startswith('\\medskip\\noindent\\textbf{What it has not done is run.}')
    assert 'convnext' in '\n'.join(L[a-1:b])      # prove it is the RIGHT chapter's block
    L[a-1:b] = new
    PY

and afterwards assert: 13 `\chapter{`, `center`/`tabular`/`tikzpicture` balanced, the chapter's TODO
gone, the other chapters' anchors still present. A global replace on one of these once ate 3,939
lines and three chapters.

## 6. Archive shape

`runs/<date>-cnx-verified-300ep/`: `RESULTS.md`, the curve CSV, `reference_curve.tsv`, `full.log`,
`attempt.log`, `master.log`, `epoch_clock.tsv`, `loader_rss.tsv`, and `summarize.sh` — copy
EfficientNet's, which rebuilds every number in its write-up from the logs and takes one argument
change (slug, epochs, reference log path). The per-epoch clock and the loader-memory logger are
systemd units in that run dir; start both at launch, since a plain background watcher gets killed
under memory pressure.

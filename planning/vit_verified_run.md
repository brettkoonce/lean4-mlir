# vit_verified_run.md — run `vit-imagenet-verified` and land it in §9.6

**Opened 2026-08-29.** Handoff for a fresh session. One job: run the **phase-4 ViT** — the verified
StableHLO peer of the phase-2 reference that landed 2026-08-29 at **72.31 / 91.12** — then write it
into the blueprint the way §9.6's phase-2 half was written this session.

▶ Everything the run needs is committed. Nothing here is blocked on a decision.

---

## 0. WHY NOW, AND WHAT THE PAIR IS FOR

§9.6's phase-4 subsection ends on a bare `[TODO: run vit-imagenet-verified.]` and says outright:
*"What it has not done is run. No phase-4 ImageNet result exists for this network, which is why
every accuracy above comes from phase 2."* This fills it, and gives ViT the {reference, verified}
pair that Chapter 5 already has for ResNet-50.

⭐ **The pair is one-variable as of `HEAD`, and it was not before.** Three axes differed and all
three were closed 2026-08-29 (see that commit):

| axis | was | now |
|---|---|---|
| weight init | Glorot Linears (3.6× wide at d=192), CLS 5× wide | `vitInit := true` — σ 0.02, patch conv on PyTorch `Conv2d` default |
| EMA | no artifact combined it with wx+clip+drop | `vitin_emadp128x4wxclipdropbf16` |
| precision | fp32 | bf16, matching the reference |

Everything else already matched and was checked individually: 128/replica × 4 = global 512 and
2,502 steps/epoch; 300 epochs, warmup 5, cosine; AdamW lr 5e-4, wd 0.05, `wdExclude`, clip 1.0;
label smoothing 0.1 (baked — `vitBackAllB … (some (alphaStr, negAlphaKStr, bStr))`); dropPath 0.1
with 24 independent masks; soft targets via `%onehot: tensor<128x1000xf32>`; and the data side is
the SAME shim the reference's recipe emits (RandAugment m9/mstd0.5/inc1, mixup+cutmix alternating,
random-erase 0.25, repeated-aug 3×, antialiased BICUBIC), eval over all 50,000.

---

## 1. ▶ THE LAUNCH

GPUs must be idle first — see §3's stale-process trap.

```bash
cd /home/skoonce/lean/proof_verify_demo/verify-v2
mkdir -p /tmp/supervise_vit-verified-4gpu
setsid nohup scripts/supervise.sh vit-verified-4gpu \
  > /tmp/supervise_vit-verified-4gpu.out 2>&1 &
```

⚠ **The job config does not exist yet — write it first.** Model it on
`scripts/jobs/r50-2018-bf16-4gpu.conf`, which is this box's working example, NOT on
`scripts/jobs/vit-imagenet-4gpu.conf`, which is ares' (its `DEVS="0,2,3,4"` excludes AER-dirty
cards this box does not have, and its `PJRT_PLUGIN` says cuda12). The settings:

```bash
DEVS="0,1,2,3"
EPOCHS=300
CMD=(.lake/build/bin/vit-imagenet-verified data)
ENV_EXTRA=(
  SHIM_PYTHON="/home/skoonce/.venv-cuda/bin/python3"
  PJRT_PLUGIN="/home/skoonce/.venv-cuda/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so"
  PJRT_REPLICAS=4
  LEAN_MLIR_REPLICAS=4
  LEAN_MLIR_VARIANT=emadp128x4wxclipdropbf16   # ⛔ LOAD BEARING — see below
  LEAN_MLIR_BATCH=128                          # PER REPLICA -> global 512
  LEAN_MLIR_EPOCHS=300                         # SETS the cosine, not just the loop bound
  PJRT_FFI_RESIDENT=1                          # opt-in; OFF silently halves throughput
  SHIM_WORKERS=8
)
TEMP_MAX=78
TEMP_RESUME=62
REST_EPOCHS=""                                  # temperature governs; this box needs no duty cycle
CKPT_EPOCH_FILE=".lake/build/vitin_emadp128x4wxclipdropbf16_ckpt_xla.bin.epoch"
```

⛔ **`LEAN_MLIR_VARIANT` is the whole run.** `adamdp128x4wxclipdropbf16` is a committed artifact one
character-class away and it has **no EMA** — it would train fine, report a number, and not be the
pair. The PRECHECK must assert the variant string AND that
`verified_mlir/vitin_emadp128x4wxclipdropbf16_train_step.mlir` exists.

⚠ `baseLR` defaults to 5e-4 in `MainViTImagenet.lean`, which is already the DeiT batch-512 rate, so
unlike R50-2018 there is **no** `LEAN_MLIR_BASE_LR_U` to set. Do not add one.

### Verify from the RUN'S OWN OUTPUT, never the conf

The driver announces every one of these. All five appeared in the 2026-08-29 smoke:

```
▸ INIT: timm/DeiT (σ=0.02 weights, patch-embed on PyTorch Conv2d default)
▸ EMA: region 3 of 4 [θ|m|v|ema], shadow starts AT the weights,
       decay min(0.999960, (1+t)/(10+t)) — EVAL AND CHECKPOINT SCORE THE SHADOW.
▸ STOCHASTIC DEPTH: 24 drop sites, keeps #[1.000000, …, 0.900000]
▸ MIXUP/CUTMIX: SHIM_MIX=both; ON (wire v2, soft float32 targets)
▸ val = all 50,000 (timm's denominator; drop_remainder=False on the val split)
[pjrt_ffi] compiled …vitin_emadp128x4wxclipdropbf16_train_step.mlir (829 outputs, 4 replicas)
```

⛔ **`decay min(0.999960, …)` is the line to check.** `trainAdamSched`'s `emaDecay` default is
**0.9999**; the reference's is 0.99996. `MainViTImagenet.lean` passes it by name, but a future edit
that reorders those positional args would silently revert it, and a shadow averaging 4× too fast
still trains and still reports.

---

## 2. ETA — MEASURE IT, do not quote this paragraph

Blueprint §9.6's phase-4 table carries **159 ms/step → 6.6 min/epoch → ~36.3 h** for this box, but
⚠ that row is the **non-EMA** render. EMA adds a fourth `nP`-wide region (200 params ≈ 23 MB) of
read+write per step. Expect ~33–37 h; **the first full epoch settles it, and that is what to quote.**

⚠⚠ The driver prints a **cumulative** average that includes compile. Difference it over steps
200→600, exactly as `scripts/bf16_probe_4gpu.sh` does. Reading the printed line is the mistake.

⭐ For calibration: the phase-2 reference ran **34.22 h**, 401–403 s/epoch, one attempt, zero
restarts, zero thermal rests, peak 55/62/61/66 °C. This box has now done two multi-day runs back to
back (that one plus R50-2018's 30.72 h). Cooldowns are OFF deliberately.

---

## 3. TRAPS — every one of these has bitten on this box

* ⚠⚠ **Stale GPU process fakes OOM and NCCL errors.** Before trusting ANY CUDA failure, run
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and confirm it is empty.
  `kill -9` on a wrapper leaves the Lean exe grandchild holding ~9 GB on all four cards.
* ⚠⚠ **`pkill -f NAME` SELF-MATCHES** when the same command line mentions NAME — it has eaten
  commands here three times. Use `pkill "vit-imagenet-ve"` (no `-f`, 15-char comm limit) or a PID
  captured beforehand.
* ⚠ **Kill the SUPERVISOR, not the trainer.** `scripts/supervise.sh` traps EXIT and takes its run
  down with it; the reverse is not true, and killing the trainer alone just gets it relaunched.
* ⚠ **mtime is not staleness.** `.lake/build/bin/vit-imagenet-verified` looked stale against its
  source this session and was not — `lake build` replayed from cache and left the binary
  byte-identical. **Ask the artifact what it is** (`--help`, or the announcements above), never its
  timestamp. R50-2018's conf PRECHECK false-positives for exactly this reason.
* ⚠ **`lake build <module>` does not re-run `#eval`s** when it replays from cache. If an artifact
  needs regenerating, `touch` the module first. `LeanMlir/Proofs/Codegen/ViTRenderB.lean` is the
  heavy one (~11 min / ~14 GB from cold, per [[vit-backb0-ci-cost]]).
* ⚠ **Checkpoints are per-VARIANT and outlive the artifact they trained on.** A `vitin_adamdp…`
  checkpoint must never be resumed into an `emadp…` run — the blob has a different region count.

---

## 4. ▶ WHEN IT LANDS: the LaTeX pass

Mirror what §9.6's phase-2 half got on 2026-08-29 (commit `93e8f413`). Brett's instructions that
session, which apply again:

* **Numbers and what differs, not framing.** Cut "the point of this section is", "the familiar
  shape", "where the recipe stops being optional". The phase-2 result went from three paragraphs to
  four lines and was better for it.
* **Delete superseded rows, do not annotate them.** The ROCm row and its 70.28/90.05 were removed
  outright rather than kept as history.
* **Re-measure rather than patch a stale figure.** The bf16 paragraph's 176-vs-260 ms was replaced
  by a measurement taken that day on the box that exists (`jax/scripts/jax_vit_bench.py tiny 128`).

### The specific edits

1. **§9.6's phase-4 table** (`Box | ms/step | min/epoch | 300 epochs | Val top-1`) — fill the `TBD`
   in the 4× 3060 row, and correct the ms/step to the EMA render's measured figure.
2. **Delete `\noindent\textbf{[TODO: run \texttt{vit-imagenet-verified}.]}`** — the whole point.
3. **Rewrite "What it has not done is run."** That paragraph exists to explain an absence.
4. ⚠ **§9.6 says "the one place in Part 1 where the verified row is the FASTER of the two, at 6.8
   minutes per epoch against phase 2's 7.6."** That compares 4060 Ti to 4060 Ti. The 3060 numbers
   are now 6.6 (verified, non-EMA) against a measured 6.84 (phase 2) — same direction, so the claim
   strengthens; **re-check it against the EMA render's real figure before restating it.**
5. **The regularizer-knob paragraph is now WRONG and must be rewritten.** It says *"EMA is a variant
   too (`vitin_emadp128x4`), but no committed artifact combines it with the other three."* As of
   `HEAD` one does.
6. **Add the init to the phase-4 listing**, the way `vitInit` was added to the phase-2 one — it is
   a `VerifiedConfig` field now and the run depends on it.

### What the comparison does and does not establish

⭐ Quote both numbers side by side; both are over 50,000 under the same protocol.
⛔ **Do not call it one-variable without saying which variable.** It is one-variable in *recipe* —
init, EMA, precision, schedule, optimizer, augmentation and eval all match. It is not one-variable
in *implementation*: that is the whole point, the lowerer differs.
⚠ **Carry the gate caveat.** `tests/TestViTDpCheck.lean` hands both replicas the SAME rows, so it is
structurally blind to a shard-offset bug, and the genuinely-sharded check covers Imagenette `vit`
but not `vitin`. §9.6 already says this chapter has "one and a half" gates where ConvNeXt has two —
that sentence stays true and matters more once a phase-4 number exists.
⚠ The remaining paper deviations in §9.6's "Where this still differs from the paper" are unchanged
by this run: batch 512 vs 1024, our gradient clipping (**the "DeiT default" claim is still
unverified against DeiT's `main.py`** — there is no timm on this box), from-scratch
RandomResizedCrop/Random-Erasing, bilinear geometric RandAugment.

### Build and check

```bash
cd blueprint/src && latexmk -pdf -interaction=nonstopmode print.tex
```
⚠⚠ **Do not edit `content.tex` with `str.index`-style Python.** Four chapters share the heading
`\subsection*{Phase 4: the verified trainer}`, and slicing between a section-local anchor and the
FIRST match of that string duplicated ~4,100 lines this session (two Vision Transformer chapters,
duplicate `\label{sec:vit_imagenet}`). Use anchored edits, and check
`grep -c "^\\\\chapter{Vision Transformer}"` is 1 and the diffstat is plausible before building.

---

## 5. AFTERWARDS

* Update [[vit-deitinit-300ep-inflight]] with the pair, and `planning/imagenet_rerun_sweep.md` §5
  item 5 — ViT is then fully closed and **ConvNeXt is the last C1-blocked net**, with its paper init
  already committed as `a8bda98` ([[convnext-cnxinit]]).
* ⚠ ConvNeXt-T at 300 ep costs **~74.9 h** on this box, more than twice ViT. The 80-epoch `default`
  tier is ~20 h. Decide which before booking it.
* ⭐ **ConvNeXt has the IDENTICAL EMA gap, and §8's own prose says so** — `content.tex:9134`:
  *"(`convnextin_emadp`), but no committed artifact combines it with the other three."* ViT's §9
  sentence even calls it out as shared (*"the same combination is missing here as there"*). So when
  ConvNeXt's phase-4 run comes up, the fix is the same one-line shape as this session's: re-emit
  through `cnxAdamTrainStepFaithful…` with `(ema := true)` alongside `wdExclude`/`clip`/`sd`, and
  extend `tests/TestVitEmaDropRender.lean`'s pattern to it. ⚠ ConvNeXt derives its entry name from
  the variant (`cnxAdamVariant`) where ViT takes `funcName` explicitly — `ViTRenderB.lean`'s own
  note flags that as a different failure route, so a flag that misses the naming call writes a
  MISMATCHED artifact rather than a wrong one.
* Open, and unrelated to this run: the classifier-head init gap (torchvision's `nn.Linear` default
  is `U(±1/√fan_in)`; the emitter gives R50's 2048→1000 head 0.0444 vs 0.0221, **2× wide**, on every
  ResNet/MobileNet/EfficientNet). Deliberately not fixed — it moves nets with landed numbers and
  wants its own A/B.

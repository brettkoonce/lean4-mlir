# 2026-08-27 — does the per-example drop-path fix move the A2/A1 step cost?

**The question.** `planning/verified_side_quest_counterparts.md` §6b left one thing open: the
book's phase-2 A2/A1 figures (**1,368 ms/step**, `runs/2026-08-27-jax-sb-tier-step-probe/`) were
measured against a reference whose stochastic depth drew a per-BLOCK scalar bernoulli. The fix
(`49ef99a`) makes it per-example — a `(512,1,1,1)` mask per block instead of a scalar. That was
recorded as *"unmeasured, not measured-and-negligible"*, which is the honest version and not an
answer.

**Method.** `runs/2026-08-27-jax-sb-tier-step-probe/probe_r50.sh`, unchanged and copied here — same
script, same `STEPS=10`, same 4× RTX 4060 Ti, same synthetic input, effective batch 2048 as
4×512 grad-accum.

⭐ **A3 IS THE CONTROL, AND IT IS A GOOD ONE.** `resnet50ImagenetConfigRSBFaithful` sets
`dropPath := 0.0`, so its generated trainer emits **no `dpkeys` at all** (checked: 0 occurrences).
Its drop path is not merely inactive, it is absent — so whatever A3 moves by is pure session
variance, and A2's move has to be read against it.

⚠ The probe passes a live key (`jax.random.fold_in(PRNGKey(123), step)`), so A2's
`_drop_branch` really fires at `keep_prob < 1.0`. A probe that passed `drop_key=None` would
early-return and compare nothing; that was checked before believing the numbers.

---

## Result

| tier | dropPath | ms/step before | ms/step after | Δ | peak before | peak after |
|---|---|---|---|---|---|---|
| **A3 (control)** | **0.0 — absent** | 715.3 | 711.5 | **−0.53 %** | 4.30 GiB | 4.30 GiB |
| **A2 / A1** | **0.05 — live** | 1368.3 | 1359.7 | **−0.63 %** | 7.61 GiB | 7.48 GiB |

⭐⭐ **THE CONTROL MOVED BY THE SAME AMOUNT AS THE TEST** — −0.53 % against −0.63 %. A net with no
stochastic depth at all got 0.5 % faster between sessions, so A2's 0.6 % is that same drift and not
the mask shape. **The per-example drop-path costs nothing measurable at this resolution.**

▶ Sixteen `(512,1,1,1)` bernoulli draws and broadcasts per step, against a 1.36-second step. The
arithmetic said this and the control is what makes it a measurement.

⚠ Peak memory moved 7.61 → 7.48 GiB on A2 while the control stayed at 4.30 exactly. A per-example
mask is *larger* than a scalar, so this is not the mask — it is XLA fusing the multiply differently
now that it is elementwise over the batch. Noted rather than explained; it is 1.7 % and in the
helpful direction.

## What this settles, and what it does NOT

✅ **The book's phase-2 table stands as committed.** 1,368 and 715 ms/step remain the quoted
figures, and §6b's caveat is closed.

⚠ **This session's numbers are NOT promoted into the book**, deliberately and by this repo's own
standing practice: a fresh session that reproduces a committed figure to within 1 % confirms it, it
does not replace it. `bf16_renderer.md` §8.6 kept its committed basis for exactly this reason when
ViT-Tiny read 2–3 % slow one session and ConvNeXt-T 6 % fast another.

⛔ **This is a JAX-side measurement and prices the tier, not the verified path.** No epoch of A2 or
A1 has run on either path.

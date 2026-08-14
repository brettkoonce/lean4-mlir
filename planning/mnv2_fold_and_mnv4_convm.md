# Two assembly jobs: MobileNetV2's full-depth fold, and MobileNetV4 Conv-S → Conv-M

**Who this is for.** An agent picking up either job after the ch1–ch9 blueprint pass
(`planning/chapter_makeover.md`). They are in one doc because they share a shape:
**neither needs new mathematics.** Both are connecting structures that already exist,
and in both the cost is concentrated in one step you can measure in an hour before
committing to the rest.

Job 1 is **DONE** (2026-08-14). Job 2 is open, and independent of it.

| job | what it closes | measured scope | risk sits in |
|---|---|---|---|
| **1. MNv2 fold** ✅ | the book's "representative scale" caveat, stated in ch6 §6.1 and contrasted in ch7 §7.1 | **611 lines**, landed | not where this doc said — see the post-mortem |
| **2. MNv4 Conv-M** | the verified net is Conv-**S** while ch6 §6.5 prints Conv-**M**'s 75.48%/92.37% | two hand-written lists + a head + 4 `#guard`s | the two-conv head |

---

## Job 1 — prove `mobilenetv2_full_has_vjp_at` at all seventeen blocks — ✅ DONE 2026-08-14

Landed as `LeanMlir/Proofs/Architectures/MobileNetV2FullVJP.lean` (611 lines, 0 sorries,
3-axiom-clean, builds in ~2.5 s). `mobilenetv2_full_has_vjp_at` folds stem + 17 bottlenecks +
head; `mobilenetv2_full_has_vjp_at_correct` ties the backward to the `pdiv`-contracted Jacobian
of `mobilenetv2ForwardPaper` itself through `mobilenetv2ForwardPaper_eq_chain`. ch6 §6.1 and
ch7 §7.1 rewritten, the SpecVJP baseline line deleted, `VerifiedNets.lean`'s caveat updated.

### ⚠⚠ POST-MORTEM — read this before trusting a scope estimate in this doc

**The two headline claims below about cost were both wrong, in opposite directions.** Recorded
because the same two mistakes are available on any other net in this repo.

1. **"MobileNetV2 has no `ivResidFwdB_has_vjp` to delegate to" is FALSE, and it was the whole
   estimate.** The grep that established it searched EfficientNet's naming convention
   (`iv[A-Za-z]*Fwd[A-Za-z]*_has_vjp`). MobileNetV2 does not put `Fwd` in the name. The
   delegation targets existed the whole time as `invresBodyPC_has_vjp_at` and
   `invresBodyStridedPC_has_vjp_at` (`MobileNetV2BackCertifiedTie.lean:125,250`), built there
   as the §B float-bridge tie targets. So the eight block lemmas ARE EfficientNet-style
   three-liners: the de-risk probe this doc asks for landed at **5 lines**, not 10 and not 60.
   ▶ **A negative grep is evidence about the pattern, not about the repo.** Search for the
   *concept* across naming conventions before pricing work off its absence.

2. **The real cost was somewhere this doc never mentions: stating the hypotheses.** Seventeen
   blocks carry 33 relu6 sites plus the stem's and the head's — **35 kink conditions, each of
   which must be stated AT its running activation**, and by block 5 the activation is an
   unreadable nested term. Claim ⭐1 below ("takes only BN-epsilon hypotheses and **no explicit
   smooth-point binders** — the `_at` form absorbs the kink") is simply not what
   `mobilenetv2_has_vjp_at` does: read `MobileNetV2.lean:502–534`, which spells out five
   `h_stem`/`h_b1e`/`h_b1d`/`h_b2e`/`h_b2d` binders inline. The `_at` form absorbs nothing; it
   relocates the kink condition into a binder. At 2 blocks that is invisible and at 17 it is
   the design problem. The fix that made it tractable: name the running activations
   (`mnv2StemW`, `mnv2Pre1 … mnv2Pre17`, each one `∘` deeper) and bundle the conditions per
   block (`IVSmoothAt` / `IVStridedSmoothAt` / `IVNoExpSmoothAt`, plus `IVPos`/`IVNoExpPos` for
   the epsilons), which turns 35 + 52 loose binders into 19 + 19 bundles.

3. **What actually had to be written from scratch** was small and not on this doc's list: the
   differentiability peers of the two body VJPs, and the per-channel STRIDED stem stage
   `convBnRelu6StridedPC_*`. `MobileNetV2.lean`'s `convBnRelu6Strided_*` is the *global*-
   `bnForward` twin, not the `bnPerChannelTensor3` one the paper-spec net renders.

### ⚠⚠ The one genuine trap, and this doc does not mention it either

**`rfl` is not uniformly safe in the `_eq_chain` bridge, and 16 of the 17 layers hide it.**
Three proof shapes were measured:

| shape | result |
|---|---|
| one-step `rfl` on the whole chain | elaborator recursion depth blown (fails in 2 s) |
| one-step `simp only [<19 defs>, Function.comp_apply]` | elaborates in ~140 s into a term the **kernel** rejects, deterministic timeout |
| per-layer peel by `rfl` | instant peeling block off block, **~80 s then kernel timeout** peeling block 1 off the STEM |
| per-layer peel by `rw [<def>, Function.comp_apply]` | ✅ whole file 2.5 s |

The stem is the one layer whose body carries a type ascription: `flatConvStride2 (h := 112)
(w := 112)` has natural domain `Vec (3 * (2 * 112) * (2 * 112))` while `mnv2StemW` declares
`Vec (3 * 224 * 224)`. At that seam the kernel stops matching heads and descends into the block
body. Going through `rw` never unfolds the inner layer at all, so it closes on syntactically
identical terms. ▶ **Any net whose stem is strided has this seam.** EfficientNet's
`eq_chain` docstring warns about `rfl`/`simp` generically; the specific reason is this.

### The original scoping, kept for the record

Everything from here to the Job 2 divider is the pre-flight estimate. Read it against the
post-mortem above; the two disagree, and the post-mortem is the measured one.

#### What it is

`mobilenetv2_has_vjp_at` (`Proofs/Architectures/MobileNetV2.lean:489`) folds **stem +
two inverted-residual blocks + head**. The network has seventeen. The book says so out
loud in two places, so closing this edits prose as well as proofs:

- ch6 §6.1 states the caveat (`content.tex` ~5618).
- ch7 §7.1 contrasts EfficientNet's full-depth fold *against* it.
- `content.tex:5630` carries the TODO, and `Proofs/Foundation/SpecVJP.lean` names the
  theorem it would produce — as of 2026-08-14 that docstring says honestly that it does
  not exist, and it is a PROPOSED entry in `scripts/docstring_ref_baseline.txt`.

### ⚠⚠ The trap, and it is the expensive one

**The obvious plan — "copy `EfficientNetFullB0.lean`" — does not transfer.**
`efficientnetForwardB_full_has_vjp` is a **global** `HasVJP` over all sixteen MBConv
blocks, and it can be global because EfficientNet's activation is **swish**, smooth
everywhere. MobileNetV2 is **relu6**. A global `HasVJP` through a kink is FALSE.

▶ **There are two axes and only one is movable.** Depth (2 → 17) is the gap. Pointwise
→ global is not, and the result must stay in the `_at` form. Anyone who conflates them
spends the day proving something untrue. `MobileNetV2FullPaper.lean`'s own header
already says this ("relu6 is kinked, so the whole-net input-VJP stays pointwise-only,
the repo standard for relu-family nets"), which is why that file delivers forward +
faithfulness for all 17 and stops.

### ⭐ Two things that make it cheaper than it sounds

1. **The relu6 problem is already solved.** The existing 2-block
   `mobilenetv2_has_vjp_at` takes **only BN-epsilon hypotheses** (`0 < εs`, `0 < e₁`,
   `0 < d₁`, `0 < p₁`, …) and **no explicit smooth-point binders** — the `_at` form
   absorbs the kink. So the part everyone fears has an established pattern and does not
   need inventing.
2. **The hypothesis count is not the blocker.** At 17 blocks you thread ~52 epsilon
   hypotheses, which sounds unmanageable until you notice EfficientNet threads ~48 and
   handles them by packaging weights in a structure (`w.b1.dε`). `IVW` already exists on
   the MobileNetV2 side.

### What exists

- `Proofs/Architectures/MobileNetV2FullPaper.lean` (323 lines): all four block shapes
  over **packaged** weights — `ivNoExpW` (:136), `ivExpOnlyW` (:143), `ivResidW` (:149),
  `ivStridedW` (:155) — plus the full 17-block chain `mobilenetv2ForwardPaper`, 0 sorries.
- `vjp_comp_at` (`Proofs/Foundation/Tensor.lean:342`).
- The per-operation VJPs to compose, all proved: `depthwiseFlat_has_vjp`,
  `relu6_has_vjp_at`, `residual_has_vjp`, `flatConv_has_vjp`.
- `Proofs/Architectures/EfficientNetFullB0.lean` (531 lines) — the destination, complete.

### What is missing, and the honest cost

The eight `iv*W_{differentiable, has_vjp_at}` lemmas.

⚠⚠ **These are NOT EfficientNet's three-liners, and that is the whole estimate.** Its
eight sit in 50 lines total (:207–256) because each is `unfold` + `exact
mbResidFwdB_has_vjp …` — pure delegation to an already-proved *per-block forward*
lemma. MobileNetV2 has **no `ivResidFwdB_has_vjp` to delegate to**:

```
grep -rE "(def|theorem) iv[A-Za-z]*Fwd[A-Za-z]*_(has_vjp|differentiable)" LeanMlir/
  → nothing
```

So each of the eight has to compose ~6 operations itself rather than forward to one.

| piece | EfficientNet's | estimate here |
|---|---|---|
| 8 block lemmas | 50 lines (delegation) | **250–400 lines (real composition)** |
| the N-block chain | 88 lines (:381–469) | ~90 |
| `_eq_chain` + `_correct` | ~60 lines (:469–531) | ~60 |

**~400–550 lines against a 531-line template.**

### ⚠ The structural obstacle nobody mentions

The current 2-block proof is **91 lines (:489–580) written fully expanded** — every
weight a separate binder, every stage `set` inline. It cannot reach 17 by copy-paste;
the binder list alone would be unreadable. So this is a **rewrite into the structured
style**, not an extension. That is what makes it assembly: the structured pieces exist
in `FullPaper`, they have simply never been connected to the VJP side.

### ▶▶ De-risk it in an hour before committing

**Write ONE of the eight — `ivResidW_has_vjp_at` — and see whether it lands at 10 lines
or 60.** That single number decides the whole job, because the other seven are the same
shape. 10 → this is an afternoon. 60 → two focused sessions, schedule it deliberately.

Do not start with the chain. The chain is the part that is known to work.

### Verification

1. `lake build Proofs` typechecks, 0 sorries in the touched files.
2. `lake exe docstring-checkrefs` — the SpecVJP citation flips from PROPOSED back to a
   real name, so **delete its baseline line**. That is the ratchet shrinking as designed.
3. Update the book in **both** places: ch6 §6.1's caveat and ch7 §7.1's contrast. ⚠ ch7's
   sentence exists specifically to contrast against ch6's gap; closing the gap without
   rewriting ch7 leaves it asserting a difference that no longer exists.
4. `scripts/verify_excerpt.py` + `measure_prose.py` over ch6 and ch7 (§5 discipline).

---

## Job 2 — convert the verified MobileNetV4 from Conv-S to Conv-M

### Why

`mobilenetv4Verified` is **Conv-S**; `mnv4ImagenetVerified` (slug `mnv4in`) is that same
Conv-S trunk with a 1000-class head. But ch6 §6.5 prints **Conv-M's** 75.48% / 92.37%,
and the 100-epoch JAX reference behind that number is Conv-M
(`/home/skoonce/mnv4_convm_100ep`, outside the repo, so repo searches miss it).

So today the verified side cannot target the number the chapter prints. §4a-quater says
this in bold and leaves the phase-4 Val top-1 `TBD`; a Conv-S run there would be a first
measurement, not a reproduction. **Converting to Conv-M is what makes the row
comparable.**

### ⭐⭐ The renderer is already generic, which is most of the job

`mnv4ShapeList` builds the signature as
`mnv4Blocks.flatMap (fun b => uibSig b.p b.ic b.oc b.expand b.preDWk b.postDWk)`
(`Proofs/Codegen/MobileNetV4RenderB.lean:139`). Depth is a fold over a list, not
hardcoded. **Adding blocks needs no renderer logic.**

And the block table is already written out, in the arg order the verified spec uses:
`jax/MainMobilenetV4Imagenet.lean:24–57`. ⭐ The `.uib` signature is **identical** on both
sides (`.uib ic oc expand stride preDWk postDWk`), so the Conv-M table transcribes
directly. ▶ **Read it from that file; do not retype it from the paper.**

### What actually changes

| | Conv-S (today) | Conv-M (target) |
|---|---|---|
| UIB blocks | **14** | **21** |
| head | one `.convBnNB 256 1280 1 1` | **two**: `256→960` then `960→1280` |
| params | 4,124,426 (10-class) / 5,392,616 (`mnv4in`) | ~9.7M |
| stem, fused stage | `.convBnNB 3 32 3 2`, `.fusedMbConvNB 32 48 4 3 2` | unchanged |

Four edits, in this order:

1. **`LeanMlir/VerifiedNets.lean`** — `mobilenetv4Verified.layers` (:1146 area) and
   `mnv4ImagenetVerified.layers` (:1223 area). Both, together.
2. **`Proofs/Codegen/MobileNetV4RenderB.lean`** — `mnv4Blocks` (the 14-entry hand list
   ending at :128) and the head entry in `mnv4ShapeList` (:142).
3. **The four `#guard`s** at `VerifiedNets.lean:1166–1177`: `toSpecs.size == 158`, the
   param-count fold, the `bnChannels` list, `bnChannels.size == 52`. All four move.
   ⭐ The `mnv4in` guards (:1232–1235) are *relative* to `mobilenetv4Verified`, so they
   need no edit — which is exactly what they were written for.
4. **Re-render** the six artifacts (`mnv4_{fwd,fwd_eval,adam_train_step}`,
   `mnv4in_{fwd,fwd_eval,adam64_train_step}`) and re-run the drift guard, which now
   diffs all six (`ci(drift guard)`, 2026-08-13).

### ⚠ The one real risk: the two-conv head

`mnv4ShapeList` hardcodes a single head conv:

```lean
[("%hW", [1280, 256, 1, 1]), ("%hg", [1280]), ("%hbt", [1280])]
```

Conv-M needs `256→960` then `960→1280`. That is a small edit, but it changes the
**signature order**, and `mnv4ShapeList` is the single source for both the function
signature and the return order. ▶ Do this edit first and render immediately — if
anything is going to fight, it is here, not in the 7 extra blocks.

### ⚠⚠ Two hand-written lists, and the gate that pins them

`mnv4Blocks` and `VLayer.toSpecs` are **two independent readings of the same layout**
(the renderer cannot import the spec without inverting the dependency). The renderer's
own docstring says so at :134. **`mnv4-fwd-smoke` (`lakefile.lean:1434`) is what pins
them**, and it is the gate that will catch a half-done conversion.

▶ Run `mnv4-fwd-smoke` after **every** one of the four edits above, not once at the end.
A mismatch between the two lists is far cheaper to locate one edit at a time.

### What this does NOT fix

§4a-quater's gap table still applies to Conv-M unchanged, and converting does not touch
any of it:

- **No data-parallel render.** `mnv4AdamVariant 64 2` names `adamdp64` and the renderer
  takes `replicas`, but there is no `shard-check` row and no `TestMnv4DpCheck.lean`. So
  it stays single-device, and the blueprint's 4× row stays `TBD`. ▶ **This is still the
  blocking item for a printable phase-4 row**, and it is independent of Conv-M.
- **Drop-path and classifier dropout are not wired into UIB** (they exist as shared
  render variants; `MobileNetV4RenderB.lean` has no marker for them, and
  `jax/MainMobilenetV4Imagenet.lean:114` says the reference side is not wired either).
- **The whole verified path is fp32.**

⚠ Conv-M being ~2.4× Conv-S's parameters makes the single-card wall clock worse, and the
single-card figure is deliberately not in the book (§4a-quater: every other ImageNet row
was measured at 4×, so a 1-GPU row invites a false architecture comparison).

### Verification

1. `lake build LeanMlir` + the four `#guard`s pass (they fail loudly if any count drifts).
2. `mnv4-fwd-smoke` green after each edit.
3. `scripts/check_render_coverage.py` — all six artifacts stay drift-guarded.
4. `scripts/regen_verified_mlir.sh check`.
5. `scripts/grad_tie.py --net mnv4` — the phase-2 gradient tie, which is the check that
   the converted net still agrees with its JAX peer.
6. ▶ **Then** the chapter: ch6 §6.5's Conv-S/Conv-M mismatch note can come out, because
   the mismatch is gone. That is the point of the job.

---

## Shared discipline

Both jobs are governed by `planning/chapter_makeover.md` §5, and two of its rules matter
here more than usual:

- **Run the binary the change claims to affect.** Every real defect in the ch4–ch6 passes
  was invisible to reading and to `lake build`, and two printed normal-looking output
  while doing nothing.
- **Never a number you did not measure.** Job 2 changes a parameter count that the book
  prints; re-derive it from the spec rather than from the paper.

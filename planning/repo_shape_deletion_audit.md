# Repo shape & deletion audit — what's load-bearing, what's accretion

*2026-07-06. Prompted by: "what would a repo reduced to `lake run
mnist/cifar/imagenette` look like?" → "keep the book and its demos" → "I fear
we are accreting a lot of complexity." This doc records the measured
dependency structure, and for each remaining piece, the honest case for and
against deleting it. Default outcome: **no split, a few hygiene moves** — but
the numbers are worth having on paper before the next year of accretion.*

---

## 0. The headline numbers (measured 2026-07-06)

| thing | size |
|---|---|
| working tree | 23 GB |
| …of which build artifacts (`.lake` 13G + `tests/comparator/.lake` 7.3G + `jax/.lake` 0.7G) | **21 GB** |
| …of which data | 2.3 GB |
| **git-tracked content** | **47 MB / 964 files** |
| Lean source | 392 files / 9.6 MB |
| import cone of `lake run mnist+cifar+imagenette` (14 exes) | 40 modules / 1.8 MB |
| import cone of *every* book-referenced demo (~50 exes) | 146 modules / 2.5 MB |
| `Proofs/` IR+render layer (in the demo cone) | 18 files / 1.45 MB / 24.6k lines |
| `Proofs/` math certificates (outside every demo cone) | **155 files / 5.6 MB / 86.8k lines** |

So: the *disk* problem doesn't exist (it's untracked build junk), and even the
*source* problem is concentrated — the certificate suite is 58% of all Lean
source and ~0% of what any demo runs. The complexity is real but it is
**deliberately shaped**: an engine everyone uses, and a proof corpus that only
CI reads.

Mathlib boundary: it enters through exactly the 18-module IR layer
(`StableHLO`/`IR`/`Tensor` + the per-net op/VJP definitions the renderers are
built on). The runtime layer (`VerifiedTrain`, `IreeRuntime`, `F32Array`,
`Types`, `E4M3Quant`, `MlirCodegen`) is Mathlib-free. Only ONE demo exe pulls
the proofs at build time: `MainViTVerifiedAdam` renders its MLIR in-process;
the other 13 verified trainers read committed `verified_mlir/*.mlir`.

## 1. The refactor shapes considered (and shelved)

* **Verified-only repo** (`lake run` 3 tiers): ~15 MB source. One-file change
  (ViT reads its committed render) severs Mathlib entirely → `.lake` drops
  from 8.5 GB to a few hundred MB, clean build in minutes. Trade: the demo
  repo becomes a consumer of artifacts proven elsewhere.
* **Book repo** (book + literally its demos): ~30 MB source, 146 modules,
  keeps Mathlib (the render layer is structural here), keeps `jax/` unchanged
  (it already `require`s the parent), rescues ~15 render-writer `tests/Test*`
  files into `tools/render/` so the drift guard survives. CI = build ~50 exes
  + render drift; the proof CI stays in the mother repo.
* **Decision: neither, for now.** The split is mechanical whenever wanted
  (cone lists archived from this analysis); the real question is below.

## 2. Piece-by-piece: delete or keep?

Format: **piece (size)** — case for deleting / case against / verdict.

### The engine (untouchable)

* **`LeanMlir/` non-Proofs (~1.6 MB, 128 modules)** — codegen, emitters,
  verified-train runtime, IREE FFI driver, data loaders, renderers. Everything
  imports it. **KEEP.**
* **`LeanMlir/Proofs/` IR layer (18 files, 1.45 MB)** — `StableHLO` AST +
  denotation (411 KB), `IR`, `Tensor`, plus the per-net op/VJP modules
  (`Attention`, `CNN`, `MobileNetV2`, …, 890 KB) the proven renderers
  reference. Deleting any of it breaks the "trains the proven graph" claim.
  It is also the entire Mathlib dependency. **KEEP.**
* **`ffi/` (1.5 MB), `verified_mlir/` needed renders, `run.sh`,
  `download_*.sh`** — the runnable path. **KEEP.**

### The certificate corpus (155 files, 5.6 MB, 86.8k lines)

This is where "are we accreting?" has a real answer: yes — and mostly on
purpose. It grows along two axes: **nets × certificate-kinds** (each new kind
of certificate sweeps the 5–12 nets) and **generated proof files** (step
functions: the two LipSDP scorecards added 1.03 MB in one day, instantly the
#2 and #3 largest proof files). Family by family:

* **Per-net VJP + tie certificates (TiePoC sweep, FaithfulPoC, render ties)**
  — *delete?* they're the biggest sweep and nobody runs them. *keep?* they ARE
  the thesis: without the tie, "verified trainer" means "trainer plus some
  unrelated theorems." **KEEP — this is the product, not accretion.**
* **FloatBridges matrix (whole-net fwd/bwd × 5 nets, MNIST chain, adjoint
  chain)** — *delete?* pure research output, no demo touches it, meaningful CI
  cost. *keep?* it's the float-soundness story the blueprint (Appendix C)
  rests on; deleting published results to save CI minutes is backwards.
  **KEEP; prime candidate for a `certs` lib built on scheduled CI rather than
  per-push (see §4).**
* **Robustness line (`LipschitzCert*`, smoothing, the new LipSDP pair certs)**
  — *delete?* the two generated LipSDP files are 1 MB of machine-written Lean;
  regenerable from `scripts/lipschitz_cert_pair_sdp.py` in ~5 min. *keep?*
  active thread, and "regenerable" assumes the generator + scipy environment
  stay healthy — committed proofs are the artifact of record; CI checks them,
  not the generator. **KEEP, but adopt the generated-proof policy (§3).**
* **Descent capstones (`SgdDescentCnn` 615 KB/10.2k lines — the single
  largest proof file; `SgdDescentMlp`, linear)** — *delete?* one theorem
  family, enormous files, endpoint reached (full-depth descent is an honest
  open-by-design stop). *keep?* it closes audit gap #2 (real descent at
  trained weights); the size is exact-rational witness data, not code anyone
  maintains. **KEEP — it's finished, it won't grow.**
* **Seals/witnesses (`TrainedCnnSeal` 287 KB of Jacobian tables, Mnv2/R34
  live seals)** — same shape as descent: big, done, inert. **KEEP.**
* **`MuonGeometry` (14 theorems, small)** — self-contained jewel, cheap.
  **KEEP.**
* **`Binary32Instance` / E4M3 / TrustedBridge pattern** — the axiom-hygiene
  spine; AuditAxioms leans on it. **KEEP.**
* **Lexer/SSA syntactic line (`IRPrint` 125 KB, `StableHLOParse`)** — *delete?*
  Gap-3 is the least-loved thread. *keep?* small, and it's the only purely
  syntactic faithfulness result. **KEEP (low priority to extend).**

**Net verdict on the corpus: nothing here is deletable without giving up a
result we currently claim.** The accretion risk isn't the existing files —
it's the *default* that every future certificate kind sweeps every net and
lands in the same per-push CI. That default is worth breaking (§4).

### Tests & CI

* **`tests/AuditAxioms.lean` (216 KB, 1,334 entries)** — the honesty
  mechanism; the repo's credibility is this file. **KEEP.**
* **`tests/Test*.lean` render writers** — the drift guard (committed `.mlir`
  == proven renderer). **KEEP** — and they're what a book-repo split must
  rescue.
* **`tests/comparator` (8 tracked files, 130 KB; 7.3 GB untracked `.lake`)** —
  the Diderot formal-verification model. Source: keep. The 7.3 GB: **local
  build junk — clean it and gitignore `tests/comparator/.lake`** (hygiene item
  H1).
* **`tests/vjp_oracle` (84 KB)** — small, useful. KEEP.

### Everything else

* **`blueprint/` (7.1 MB), `demos/` (5.4 MB — 5.2 is figures), `Bestiary/`
  (0.5 MB), `apps/` (0.3 MB), `jax/` (~1 MB source, own lake project)** — the
  book and its demos. **KEEP** (the point of the repo).
* **`verified_mlir/` (47 renders, 8.8 MB; 14 needed by the run tiers)** — the
  other 33 are older variants (SGD imagenette, momentum, fp8, …), each
  drift-guard-tied to a Test writer. *delete orphans?* any render no exe or
  book chapter references is pure weight (~4 MB). **AUDIT: prune renders with
  no referencing exe** (hygiene item H4).
* **`lakefile.lean` (161 exes)** — the quietest accretion point: every
  experiment added an exe, and lately root-level `run_*.sh` scripts too.
  *delete?* dead exes cost reader attention and CI surface. **AUDIT** (H4);
  move stray root `run_*.sh` into `scripts/` (H2).
* **`planning/` (83 docs, 1.5 MB)** — the lab notebook. *delete?* never —
  it's the only honest history of why things are shaped this way. *but*: ~half
  are handoffs for threads that are DONE. **Add `planning/archive/` and sweep
  completed-thread docs into it** (H3). Cheap, big signal-to-noise win.
* **`runs/` (3.5 MB, mostly untracked logs), `traces/` (15 MB), `logs/`
  (0.7 MB)** — experiment artifacts. Adopt a retention rule: logs cited by a
  chapter/results table get committed; everything else is gitignored and
  expendable. **PRUNE + policy.**
* **`historical/`, `mnist-lean4/`, `mlir_poc/` (~400 KB total)** —
  predecessors; git history preserves them anyway. *keep?* negligible cost,
  occasional archaeology value. **DELETE-SAFE; genuinely optional.**
* **`upstream-issues/`, `home_page/`, `figures/` (tiny)** — KEEP, noise-level.
* **`scripts/` (944 KB)** — mixed bag: proof generators (load-bearing —
  they're how generated certificates regenerate), GPU probes, one-off
  plotting. Worth a light dead-script pass during H4.

## 3. What actually accretes (patterns, not pieces)

1. **The certs matrix**: kinds × nets. Each new certificate kind costs
   5–12 files and a CI tail. The engine only grows with new *ops*. All real
   growth pressure is in the corpus — which is also the research output, so
   the answer is separation, not abstinence.
2. **Generated proofs are step functions.** LipSDP: +1.03 MB/day. Policy
   worth adopting now: every generated `.lean` must (a) name its generator in
   the module docstring, (b) be regenerable offline, (c) get a size note in
   the commit message. If a generated family crosses ~2 MB, it moves to its
   own lake lib.
3. **Every experiment sheds: an exe + a planning doc + logs + sometimes a
   root script.** The exe and doc stay forever by default. A "thread
   retirement" habit (archive the doc, delete the exe if the result is
   committed elsewhere) caps this.
4. **Untracked build junk masquerades as repo weight** (21 of 23 GB). Not a
   problem, but it distorts every "how big is this" intuition — hence this
   doc's tracked-size table.

## 4. The one structural change worth considering (short of a split)

Split the **lake libs** (not the repo): `engine` (codegen + runtime + IR
layer, ~3 MB, Mathlib via the 18 modules), `certs` (the 155-file corpus,
scheduled/nightly CI instead of per-push), `book` (apps/demos/Bestiary exes).
Nothing moves on disk; imports don't change; the demo cone becomes an
enforced boundary instead of a measured one, and per-push CI drops to
engine+book+AuditAxioms-spot-checks. This gets ~all of the split's benefit
with none of the two-repo drift risk.

**IMPLEMENTED 2026-07-06** (the certs half — Brett: "make it a submodule …
set up a new ci thing → dump it in there"). Not a git submodule: the
dependency is bidirectional (certs import the IR layer; `AuditAxioms` audits
both sides), so a separate repo would force restructuring the audit. Instead:

* `lean_lib «Proofs»` (default target) is now the **fast per-push slice**:
  the 11 demo-cone IR/op roots + the 8 renderers the drift guard
  re-elaborates. Builds in minutes.
* `lean_lib «Certs»` is the corpus (the old 157-root list, unchanged; its
  roots subsume the slice, so `lake build Certs` builds everything the audit
  needs).
* `.github/workflows/proofs.yml` (per-push): coverage check + build the
  slice + verified-render drift guard + zero-axiom-decl grep + stats. The
  three-axiom closure step moved out.
* `.github/workflows/certs.yml` (new): corpus build + three-axiom closure +
  faithful-scorecard summary + corpus sorry count. Triggers: pushes/PRs that
  touch `LeanMlir/Proofs/**` / `AuditAxioms` / lakefile / toolchain, nightly
  cron 08:17 UTC, manual dispatch. Shares proofs.yml's rolling `.lake` cache
  key family.
* `scripts/check_audit_coverage.py` parses the union of both libs' roots.

Net effect: proof-touching pushes still get the full audit immediately;
demo/book/engine/docs pushes stop paying the corpus tail; nightly catches
drift the path filter can't see (toolchain bumps, Mathlib cache rot).

## 5. Recommended now (even though the answer is "don't split")

* **H1**: `rm -rf tests/comparator/.lake` + gitignore it (frees 7.3 GB).
* **H2**: root `run_*.sh` strays → `scripts/`.
* **H3**: `planning/archive/` + sweep completed-thread handoffs.
* **H4**: dead-exe / orphan-render / dead-script audit (one sitting,
  lakefile + verified_mlir + scripts).
* Adopt the generated-proof policy (§3.2) before the next generated family.

Cone module lists from this analysis: session scratchpad (regenerable by the
import-cone script in ~2 s; roots = the lakefile exe roots minus `tests.*`).

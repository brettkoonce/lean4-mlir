# `Proofs/` directory refactor — subdivide the flat 199-file folder

*2026-07-21. Prompted by running [lean-code-reuse](https://github.com/Vilin97/lean-code-reuse)
(Vilin97) against this repo and comparing to its published 34-project cohort.
Two findings: our body-duplication number is entirely the generated
certificate corpus, and our cross-directory reuse is ~0 **because we only have
two directories**. This doc records the measurement, the proposed split, and
what it would cost. Status: **executed 2026-07-21** on branch
`proofs-refactor` — see §5 for the execution record and one honest surprise
about the metric.*

---

## 0. Why (the measurement that prompted it)

lean-code-reuse's thesis: raw reuse volume doesn't discriminate quality, but
*organization* (cross-directory reuse), *process discipline* (docstrings,
sorry hygiene) and *economy* do. Textual tier, `roots=["LeanMlir"]` (their
harness excludes `tests/`, `blueprint/`, `scripts/` by default), against their
34-project cohort:

| metric | ours | cohort median | percentile |
|---|---|---|---|
| LOC | 168,590 | 38,888 | 85% |
| declarations | 17,801 | 3,037 | 85% |
| never-reused decls | **5.7%** | 36.0% | 3% (2nd best) |
| cross-file edges | 58.8% | 49.4% | 76% |
| sorry rate | **0** | 0.0005 | tied best |
| axioms declared | **0** | 0 | tied best |
| trivial proofs | 1.4% | 5.1% | 18% |
| **body duplication** | **8.1%** | 1.2% | 88% *(bad end)* |
| **cross-directory edges** | **0.05%** | 13.2% | 18% *(bad end)* |
| doc coverage (public) | 34.8% | 41.5% | 44% |

### The duplication flag is the certificate corpus, not the proofs

Splitting the corpus by filename (`Lipschitz*`/`*Scorecard*`/`Smoothing*` /
`*Seal*`/`SgdDescent*` = 13,940 of 17,801 decls):

| metric | all | **certs excluded** | certs only |
|---|---|---|---|
| body duplication | 8.1% | **2.0%** | 9.9% |
| max identical-body cluster | 281 | **12** | 281 |
| doc coverage (public) | 34.8% | **83.2%** | 22.5% |
| cross-file edges | 57.6% | **65.4%** | 46.7% |
| never-reused | 5.7% | 23.2% | 0.9% |
| trivial proofs | 1.1% | 3.9% | 0.3% |
| median proof lines | 6 | 9 | 6 |

Top duplicate-body files are all scorecards (`SmoothingCPScorecard` 279,
`LipschitzCertScorecardIBP` 270, `LipschitzCertScorecardSDPFull` 182 …). So the
repo is really **two corpora**: a dense, 83%-documented, 2%-duplication proof
library, and a large machine-emitted certificate payload with the inverse
profile. Every metric where we score badly is the second one. That is a fine
thing to be — but the directory layout should *say so*, which is the other
half of this doc.

### Cross-directory is an artifact of the flat layout

`LeanMlir/` = 16 root files + `LeanMlir/Proofs/` = 199 files flat. Two
directories, so essentially no edge can cross one. Simulating the split below
against the real edge table: **cross-directory reuse 0.05% → 19.3%**, above
the cohort median (13.2%), from filing alone — no proof changes.

---

## 1. Proposed layout

Buckets follow seams the codebase already has (the lakefile's `Proofs`/`Certs`/
`CertsHeavy` target split is roughly this line already):

| directory | files | what lives there |
|---|---|---|
| `Proofs/Foundation/` | 33 | pdiv/HasVJP kit, `Tensor`, `Vec`/`Mat`, `MLP`, `IR`, canonical instantiations |
| `Proofs/Architectures/` | 56 | `Attention`, `CNN`, `BatchNorm`, `MobileNetV2`, `ConvNeXt`, `EfficientNet`, ViT, detection/segmentation |
| `Proofs/Float/` | 43 | `FloatBridge`, `Binary32Instance`, bf16/E4M3, per-net float bridges |
| `Proofs/Codegen/` | 22 | proof↔IR bridges, `*Render`, `IRPrint`, `StableHLO`, Adam step/render |
| `Proofs/Certificates/` | 30 | Lipschitz + smoothing scorecards (the generated payload) |
| `Proofs/Training/` | 15 | `SgdDescent*`, Jacobian seals, trained witnesses |
| `LeanMlir/` (engine) | 16 | codegen/spec/train — **unchanged** |

A `Proofs/Certificates/` directory also makes the two-corpora story legible to
a newcomer, and gives an obvious place to put a README explaining that its
contents are emitted, not hand-written (which is the honest answer to the
duplication metric).

---

## 2. Blast radius

Mechanical but wide — measured 2026-07-21:

| touch point | count | note |
|---|---|---|
| `import LeanMlir.Proofs.*` lines | 682 | across `LeanMlir/`, `demos/`, `jax/`, `tests/` |
| `lakefile.lean` module references | 184 | `Proofs`/`Certs`/`CertsHeavy`/`Codegen` root lists name modules explicitly |
| `tests/AuditAxioms.lean` decl references | 644 | `#print axioms Proofs.foo` — **namespaces don't change**, so these are safe |
| blueprint `\lean{}` refs | 107 | also namespace-based → safe; `blueprint-checkdecls` re-verifies |

Key mitigation: **the Lean namespace stays `Proofs.*`** regardless of
directory, so anything referring to *declarations* (audits, blueprint,
checkdecls) is unaffected. Only *module paths* (`import`s, lakefile roots) move.

## 3. Execution sketch

1. `git mv` in bucket order, one commit per bucket, so review is per-slice and
   a bad bucket can be reverted alone.
2. Rewrite imports: `sed -i 's|LeanMlir\.Proofs\.<M>|LeanMlir.Proofs.<Bucket>.<M>|g'`
   driven by a generated module→bucket map (avoid hand-editing 682 lines).
3. Update `lakefile.lean` root lists the same way.
4. Full rebuild: `lake build LeanMlir` + `Proofs` + `Certs` + `CertsHeavy`.
5. Re-run both axiom audits and `blueprint-checkdecls` (expect zero diffs —
   they are namespace-keyed).
6. Re-run lean-code-reuse to confirm cross-directory ≈ 19%.

**Cost:** a day, mostly rebuild wall-clock. **Risk:** low (mechanical, and the
build is the oracle) but it touches nearly every file, so it should land alone
on a quiet main, not interleaved with proof work.

## 4. Do it or not?

*For:* navigability (199 files in one folder is genuinely hard to browse),
per-directory build targets become natural, the two-corpora structure becomes
self-documenting, and the tool's headline metric goes from bottom-quintile to
above-median.

*Against:* it is pure churn against a working tree; it will invalidate every
open branch's imports (rebase pain); and the metric alone is not a reason —
lean-code-reuse is one person's tool, published 2026, not a standard.

**Recommendation:** do it for the navigability and the two-corpora legibility,
with the metric as a bonus — and do it *between* chapters of book work, when
no long-lived branch is outstanding. Not urgent.

---

## 5. Execution record (2026-07-21, branch `proofs-refactor`)

Done as sketched in §3: six `git mv` commits (one per bucket, map recovered
from the measurement session's `buckets.json`) + a fixups commit. Blast-radius
surfaces §2 missed, all mechanical: `apps/*/Main*Verified.lean` imports,
`formalization.yaml`, `tests/Audit{Bridge,Mutation,Probes,Sanity}.lean` and
`tests/comparator/*.lean`, the blueprint getting-started page, f-string
import/path templates in `smooth_dec_scorecard_gen.py` +
`lipschitz_cert_scorecard_full.py` (invisible to a literal rewrite — grep for
`Proofs.` inside f-strings when adding generators), flat `LeanMlir/Proofs/*.lean`
shell globs in both workflows, and the relative links in `Proofs/README.md`.
`planning/`, `runs/`, `CHANGELOG.md` and the dated audit reports were left
as historical records. `Proofs/Certificates/README.md` now carries the
emitted-not-hand-written note; `Proofs/README.md` got a layout table.

**The metric surprise:** re-running the harness on the new tree leaves
`pct_edges_cross_dir` at **0.05%, n_dirs=2 — unchanged**. lean-code-reuse's
`_dir_of` truncates every path to its first **two** components
(`cross_file.py`), so `LeanMlir/Proofs/<Bucket>/X.lean` still counts as
directory `LeanMlir/Proofs`: the tool assumes Mathlib-style `Pkg/Topic/…`
layout, and our buckets sit one level too deep for it. With `_dir_of`
patched to the full dirname, the new tree measures **19.28%** across 7
directories — exactly the §0 simulation — so the organization is real; the
tool just can't see it at this depth. §4's ordering of reasons (navigability
first, metric as bonus) turned out to be the right call. If the headline
number ever matters, the options are upstreaming a depth fix to
lean-code-reuse or flattening buckets to `LeanMlir/<Bucket>/` — neither
worth doing for its own sake.

## Appendix: reproducing the measurement

```bash
git clone https://github.com/Vilin97/lean-code-reuse
cd lean-code-reuse
# register this repo in lean_reuse/repos.py (roots=["LeanMlir"], deps=[mathlib4,
# batteries, aesop, core]) AND add its key to BUILD_ORDER — run_all silently
# emits an empty result if the key is missing from BUILD_ORDER.
mkdir checkouts && cd checkouts
ln -s <repo> lean4-mlir
ln -s <repo>/.lake/packages/mathlib mathlib4      # vendored deps work fine
ln -s <repo>/.lake/packages/batteries batteries
ln -s <repo>/.lake/packages/aesop aesop
mkdir lean4mlir-core && ln -s ~/.elan/toolchains/leanprover--lean4---v4.32.0/src/lean lean4/src
cd .. && python3 -m lean_reuse.run_all --repos-dir checkouts --cache-dir cache \
    --out results_ours.json --only core,batteries,aesop,mathlib4,lean4mlir
```

Their `results/results_textual.json` ships the 34-project cohort for
comparison. The per-bucket recomputations above come from
`cache/lean4mlir.pkl` (decl table has `file`/`body_hash`/`body_len`, edge
table has `src`/`dst`/`dst_repo`), so slicing by filename regex is a few lines
of numpy — no need to re-run the harness.

**Not yet done:** the *exact* tier (`extract_runner` against built `.olean`s),
which resolves dot-notation and instances — our textual run reports 74%
dotted coverage, so the exact numbers will shift somewhat.

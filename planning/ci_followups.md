# CI: what landed 2026-09-09 and what is still open

**Opened 2026-09-09.** The review and the fix commit (`370ea19a`, "ci: faster, honest builds")
happened the same day; this is the residue. Measured on the live run history, not the yml comments.

## §0 What landed

* **Blueprint cache split.** The old whole-`.lake` entry was keyed on manifest+toolchain only, hit
  exactly every run since 2026-08-28 and was therefore never re-saved — a frozen snapshot every
  run rebuilt from (~25 min "Build Lean library"). NOT eviction, as `cache-prune.yml` claimed: the
  entry was read daily. Now: stable `lake-v3-` (packages + doc-gen4 output) and rolling
  `lake-blueprint-` (`.lake/build` minus `doc`/`doc-data`) that restores certs.yml's `lake-certs-`.
* **Heavy cache renamed.** `certs-heavy.yml` saved into the `lake-certs-` prefix as
  `…-heavy-<id>`; finishing last, it was the newest match and what certs/comparator restored — a
  different module set, so the corpus rebuilt every run (288 modules, 0 replays, on an UNCHANGED
  commit at the 12:57 nightly). Proof that traces survive the round trip: proofs.yml on a
  docs-only push restored its own cache and rebuilt nothing (2 min). Heavy now saves
  `lake-heavy-` and only restores `lake-certs-`.
* **Push path filters** on comparator (had none on push) and proofs (had none at all); the
  docstring gate's source + ratchet + two published-content scripts on blueprint; gate scripts +
  `proofs.yml` + bestiary table on certs; toolchain/manifest on heavy; two tools on jax; jax on
  `main` only.
* **Concurrency** on all six Lean/JAX workflows; blueprint's group moved to the build job;
  `deploy_pages` has a never-cancel `pages` group.
* **Heavy loop** names the two Crown roots and a new step derives the root list from the lakefile
  and fails if the loop drifts (tested positive and negative).
* `timeout-minutes` on blueprint (150/15) and all four jax jobs; comparator's `cache get || true`;
  the homepage copy no longer swallows failure; cache-prune groups by namespace (hash AND run-id
  stripped).

First evidence it works: the 17:08 blueprint build was cancelled by the 17:33 push; the 17:46 push
(landing page) fired blueprint alone; the heavy run on the CI commit took 3 min (warm).

## §1 Open

1. ✅ **Diagnosed the same evening — the 35.2 min was a fully cold build (242 Built, 0 Replayed),
   and doc-gen4 took 43.5 min because it rebuilt Mathlib's pages too.** Both restores missed:
   `Cache not found for input keys: lake-v3-…, lake-v3-Linux-, lake-v2-Linux-` and the same for
   `lake-blueprint-…`/`lake-certs-…` — while the 3 GB `lake-v2-` entry and a 5-minute-old
   `lake-certs-` entry both existed. Cause: actions/cache hashes the `path` list into a cache
   VERSION and restore only matches entries of the same version; the new rolling step's path
   (`.lake/build` + two `!` exclusions) differs from the corpus workflows' bare `.lake/build`, and
   the new stable step's path differs from the old whole-`.lake`. Fix staged: the identical
   three-line path block in certs/comparator/heavy (the exclusions are no-ops there). One cold run
   each for those three (their old entries are now the wrong version), then blueprint restores
   `lake-certs-` for real. Also learned: `LeanMlir.Proofs.Codegen.StableHLO` alone elaborates in
   **1285 s** on the runner (next-slowest 481 s) — the first target for the compile-time session.
2. Verify the second blueprint run under the new layout restores `lake-blueprint-` and drops to
   doc-gen time (~30 min) plus minutes.
3. A shared `.github/actions/setup-elan` composite with the 3× retry that only blueprint.yml has
   (nine elan installs, one retry). Low risk, medium value.
4. Comparator has been green on every run for days; its header still says NON-REQUIRED. Make it
   a required check — main has **no branch protection at all** today.
5. The blueprint doc-gen4 comment says `lake build LeanMlir:docs` does not regenerate
   `declarations/declaration-data.bmp` (the `/find/` index) — if true, the search index is frozen
   at whatever full build last ran. Verify by searching for a 2026-09 declaration on the live site.
6. doc-gen4 at the pinned rev (`092d631`) has no hosted-Mathlib-docs option; the site ships a
   private Mathlib copy (~5,600 pages, the 36 MB index) that the sitemap hides. Re-check on the
   next `lake update`.
7. certs-heavy's `lakefile.lean` trigger fires the heavy corpus on ~25% of pushes; documented as
   deliberate in its header, and the warm path now makes it ~3 min, so the cost is gone unless a
   root file changes.
8. `LeanMlir/Train.lean` prints "Compiling vmfbs..." on the XLA path — a code string, cosmetic.

## §2 Rules

Every workflow watches its own yml, so a CI commit fires everything once — that is the full test.
Run `python3 -c "import yaml; yaml.safe_load(open(f))"` on each edited file before pushing;
`scripts/check_render_coverage.py` reads `proofs.yml` and must still pass after editing it.

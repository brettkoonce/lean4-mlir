# upgrade.md — how to cut a new release / bump the version

The project version lives in **four spots across three files**, plus the git
tag and (out-of-repo) the Zenodo deposit. Bump them together. Last done:
`0.5.7 → 0.6.0`.

## The version-string spots

1. **`lakefile.lean`** — the Lake package version (the canonical one):
   ```
   version := v!"0.6.0"
   ```

2. **`README.md`** — two occurrences:
   - **Headline** (top of the file): `**Current version: \`v0.6.0\`** — <blurb>`.
     Convention: write a fresh blurb for the new release, and *demote* the
     previous headline to a chained `The \`v0.5.7\` headline still holds: …`
     paragraph right below it (each release stacks on the prior — search for
     "headline still holds" to see the chain).
   - **Citation block** (the `@software{…}` BibTeX near the bottom):
     ```
     version = {0.6.0},
     ```

3. **`blueprint/src/print.tex`** — the book's title-page version:
   ```
   \date{Version 0.6.0}
   ```
   **Requires a PDF rebuild** afterward: `cd blueprint && leanblueprint pdf`
   (the print PDF is a build artifact, not committed, but rebuild so the
   title page is current before tagging).

## Quick check

```
grep -rnE "0\.5\.7" lakefile.lean README.md blueprint/src/print.tex   # old → should be empty after bump
grep -rnE "0\.6\.0" lakefile.lean README.md blueprint/src/print.tex   # new → should show 4 hits (lakefile, README ×2, print.tex)
```
(Ignore `README.md` line ~39's `v0.5.6` — that's narrative chaining to a past
milestone, not the current version; leave historical "headline still holds"
references alone.)

## Tag + release

How `v0.5.7` was cut (mirror it):
1. Commit the bumps: `git commit -m "bump to v0.6.0"` (touches lakefile.lean +
   README.md; print.tex can ride the same commit).
2. Tag: `git tag -a v0.6.0 -m "v0.6.0 — <one-line release note>"`.
3. Push commit + tag: `git push origin main --tags` (only on explicit go).

## Zenodo (out-of-repo, manual)

The README DOI badge + the citation's `doi = {10.5281/zenodo.20402133}` point
at the Zenodo deposit. Zenodo mints a **new version DOI per release**; after
the GitHub release/tag, update the deposit on Zenodo, then (if you want the
per-version DOI rather than the concept DOI) refresh the badge URL and the
`doi`/`version` in the README citation. The concept DOI stays stable across
versions, so leaving it is also fine.

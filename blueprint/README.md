# Blueprint

Interactive web visualization of the `LeanMlir/Proofs/` suite. Each node
in the dependency graph links back to its Lean declaration; click around
to navigate the proof compositionally from primitives to the ViT finale.

## CI build (automatic)

Pushing to `main` triggers `.github/workflows/blueprint.yml`:

1. Builds the Lean project.
2. Runs `blueprint-checkdecls` — verifies every `\lean{…}` reference
   points at a real Lean declaration, and writes the real dependency edges
   among them; `scripts/blueprint_uses.py --check` holds every `\uses{…}`
   line to those edges.
3. Runs `leanblueprint web` — compiles `src/content.tex` into HTML with
   an interactive dependency DAG.
4. Runs `doc-gen4` — generates Lean API docs.
5. Deploys everything to GitHub Pages: `home_page/` → `/`, blueprint → `/blueprint/`, docs → `/docs/`.

Enable GitHub Pages in the repo settings (Settings → Pages → Source: GitHub Actions) for the first deploy to work.

## Local build

Requires a full TeX Live installation plus graphviz.

```bash
sudo apt-get install texlive-full graphviz
pip install leanblueprint

leanblueprint checkdecls       # verify \lean{…} refs
leanblueprint pdf              # build PDF → blueprint/print/*.pdf
leanblueprint web              # build HTML → blueprint/web/index.html
leanblueprint serve            # local web server on :8000
```

## Structure

```
blueprint/
├── src/
│   ├── content.tex            ← every axiom/theorem, with \lean{…}+\uses{…}
│   ├── macros/common.tex      ← shared math notation
│   ├── web.tex / print.tex    ← entry points for HTML / PDF builds
│   └── plastex.cfg            ← plasTeX config
├── lean_decls                 ← auto-generated list of cited Lean decls
└── README.md                  ← this file
```

## The dependency graph, in print

`src/figures/depgraph/*.tex` are the per-chapter dependency graphs at the head of
every theorems section — TikZ, laid out by graphviz, every node a `\hyperref` to
its statement. They are generated from the `\uses` lines:

```bash
python3 scripts/blueprint_depgraph_tikz.py     # needs pygraphviz; writes src/figures/depgraph/
```

Re-run after `scripts/blueprint_uses.py --fix` changes an edge or a statement
moves between sections, and commit the result (CI has no graphviz for this step).
The web build draws the same graph interactively (`src/templates/dep_graph.html`).

## Adding a new theorem to the blueprint

```latex
\begin{theorem}[Short description]
  \label{thm:my_new_theorem}
  \lean{Proofs.my_new_theorem}
  \leanok
  \uses{thm:dependency_1, ax:primitive_axiom}
  Optional math content; the key fields are \lean, \leanok, and \uses.
\end{theorem}
```

- `\lean{…}` — the exact Lean declaration name (tied to the source).
- `\leanok` — mark as complete. Removed if the theorem is still in progress.
- `\uses{…}` — other labels this theorem depends on; drives the DAG edges.
  **Generated, not written:** after adding or re-proving a block, run
  `lake exe blueprint-checkdecls blueprint/lean_decls blueprint/lean_deps &&
  python3 scripts/blueprint_uses.py --fix`. CI fails on a `\uses` line that
  disagrees with the Lean proof.

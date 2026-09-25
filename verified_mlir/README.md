# `verified_mlir/` — the programs that train

Each file here is a complete train step or forward pass in StableHLO: forward, loss, backward
and optimizer update in one module. The verified trainers load these bytes and hand them to the
lowerer; nothing regenerates them at run time.

The committed file is tied to the proofs. The renders are written by proven renderers in
[`LeanMlir/Proofs/Codegen/`](../LeanMlir/Proofs/Codegen/), and each net's `*StepTie` /
`*Fold` theorems in [`LeanMlir/Proofs/Nets/`](../LeanMlir/Proofs/Nets/) state that the
emitted update denotes the certified descent step. CI re-renders every certified artifact and
fails if the bytes differ from what is committed.

[`MANIFEST.md`](MANIFEST.md) indexes all of them: which net, which variant, how the file name
encodes the recipe, which file writes it, and which runs used it. The one artifact whose
writer is a hand-typed emitter in `tests/` rather than a proven renderer shows it there.

```bash
scripts/regen_verified_mlir.sh          # regenerate; afterwards `git diff verified_mlir/` should be empty
scripts/regen_verified_mlir.sh check    # write nothing, audit writers and pairings
```

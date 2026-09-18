import LeanMlir.Proofs.Codegen.StableHLO

/-! # R4 — the syntactic half (4b): verified-lexer **numeric core**

`StableHLOParse.lean` closes the *structural* core of syntactic faithfulness:
`parse (toToks (skel a)) = some (skel a)` — the op-graph is recovered exactly
from its **token** serialization. The remaining trusted edge is the **lexical**
one: that the emitted `.mlir` **text** (the bytes `iree-compile` reads) is a
faithful, recoverable rendering of those tokens — `parse (lex (pretty g)) = some (skel g)`.

This file lands the **numeric keystone** of that lexer: the decimal
`Nat ⟷ String` round-trip. Every per-op recognizer must read shapes (`784`,
`10`, …) back out of `tensor<784x10xf32>` type annotations, and `emitTok`
renders every such shape with `toString` (Lean's decimal `Nat.repr`). So
`parseNat (toString n) = n` is the one lemma the *whole* lexer is built on.

## A `List Char` codec, and core already has it

A 2026-06-27 probe established that Lean-core string *parsing* primitives —
`String.toNat?`, `String.splitOn` — are **kernel-opaque**: they do not reduce
under `decide`/`rfl` (they fold over `String.Pos`/`Substring` iterators), so a
concrete `decide`-the-instance shortcut is impossible and a verified lexer has to
work at the `List Char` level with structurally-recursive functions. Core ships
exactly that level's decimal codec: `Nat.ofDigitChars` is a `List.foldl` Horner
step (so it reduces), and `Nat.ofDigitChars_ten_toDigits` is its round-trip
against `Nat.toDigits`, which is what `toString` renders. `parseNat` below is that
fold. (Rendering — `toString`, `ty`, `++` — reduces by `rfl`, so the emit side
needs nothing.)

## Honest scope of the remaining lexer (corrects the planning doc)

Three sub-problems remain above this keystone; the planning doc
(`tier23_float_and_syntactic_faithfulness.md` §B) under-modeled the first two:

1. **Per-op recognizers (volume).** ~90 `Tok` constructors, each emitting a
   *fixed-shape multi-line block* (0 lines for `operand`, up to ~20 for the BN-γ
   SGD op). Many blocks share a leading `stablehlo.constant dense<0.0>` line, so
   recognition needs block-delimiting + lookahead, not a 1-line-per-token map.
   Each needs a recognizer + an `emitTok_lexTok` inverse lemma. This is the bulk.
2. **Operand re-synthesis.** `emitTok (.operand nm _) = ("", nm :: st)` — operand
   tokens emit the **empty string**; the name appears only as a *reference* inside
   a later op's line. But `toToks (skel g)` *contains* operand tokens and `parse`
   *consumes* them. So the doc's step-2 target `lex (pretty g) = toToks (skel g)`
   is **false as written**: `lex` must *regenerate* operand tokens from operand
   references (distinguishing leaf names from fresh `%v{k}` results — `fresh` at
   `StableHLO.lean:2138`). The correct end target is the composite
   `parse (lex (pretty g)) = some (skel g)`.
3. **The `ty`-string parser.** `ty dims = "tensor<" ++ intercalate "x" (…) ++ ">"`
   inverts to a `List Char` splitter on `'x'` + `parseNat` per field — needs a
   `split (intercalate)` round-trip at the `List Char` level (core `splitOn` is
   kernel-opaque, see above). Reuses `parseNat`.

Effort for the full lexer is therefore **large / multi-session** (volume + the
two design subtleties), not the "medium finite case-split" the doc billed.

## Status: deliberate STOP, not work-in-progress (decided 2026-06-27)

The full lexer is **not being pursued** — low ROI: the CI drift guard (`proofs.yml`)
*already* byte-for-byte diffs every committed `verified_mlir/<net>_train_step.mlir`
against the renderer, so the practical risk is caught; and a finished lexer would
close only the lexical edge, leaving the spec/IREE/`float32≈ℝ` edges trusted
anyway. This file is kept as a small proven down-payment **and** as the record of
*why Part B is a poor target* (the three findings above), so the scoping is not
re-discovered from scratch. See `planning/archive/tier23_float_and_syntactic_faithfulness.md`
(Part B VERDICT block).

## Residue (unchanged, state wherever cited)

Even the full `parse (lex (pretty g)) = some (skel g)` closes only the **lexer**.
It does *not* close per-op StableHLO *spec* conformance, IREE lowering, or
`float32 ≈ ℝ` — those stay validated by `iree-compile` + the GPU runs.
-/

namespace Proofs
namespace StableHLO

/-- The lexer's numeric core: decode a string's chars as a big-endian decimal — core's
    `Nat.ofDigitChars`, a `List.foldl`, so it still reduces under `decide`.
    (Non-validating — it only ever sees `emit`'s output, which is always digits;
    rejecting non-digit input is a robustness property, not a faithfulness one.) -/
def parseNat (s : String) : Nat := Nat.ofDigitChars 10 s.toList 0

/-- **Decimal round-trip keystone.** Parsing the rendered decimal of any `n`
    recovers `n` — the one lemma the whole verified lexer rests on. `toString n` is
    `Nat.repr n`, whose chars are `Nat.toDigits 10 n`, and core's
    `Nat.ofDigitChars_ten_toDigits` inverts exactly that. -/
theorem parseNat_toString (n : Nat) : parseNat (toString n) = n := by
  simp [parseNat, Nat.ofDigitChars_ten_toDigits]

end StableHLO
end Proofs

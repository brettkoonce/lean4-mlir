import LeanMlir.Proofs.Codegen.StableHLO

/-! # StableHLOLex — the decimal `Nat ⟷ String` round trip

`parseNat_toString : parseNat (toString n) = n`, the numeric piece a text lexer for `pretty`'s
output would need: `emitTok` renders every shape with `toString`, and reading `tensor<784x10xf32>`
back means parsing those decimals. No lexer is built. The correspondence between `pretty`'s text
and the token stream `StableHLOParse.roundtrip` is about stays trusted. Committed artifacts are
byte-diffed against a fresh render by the CI drift guard (proofs.yml) and parsed as StableHLO by
scripts/gates/parse_verified_mlir.py.

`parseNat` works at the `List Char` level because Lean-core string parsing (`String.toNat?`,
`String.splitOn`) folds over `String.Pos`/`Substring` iterators and does not reduce under
`decide`/`rfl`. Core's `Nat.ofDigitChars` is a `List.foldl` Horner step (so it reduces), and
`Nat.ofDigitChars_ten_toDigits` is its round trip against `Nat.toDigits`, which is what `toString`
renders.

Note, for a lexer built on this: operand tokens emit the empty string
(`emitTok (.operand nm _)` pushes `nm` and prints nothing), so `lex (pretty g) = toToks (skel g)`
is false as written; a lexer has to regenerate operand tokens from operand references, and the
statement to aim at is `parse (lex (pretty g)) = some (skel g)`. Even that closes only the
lexical edge: per-op StableHLO semantics, the lowering and `float32 ≈ ℝ` stay trusted.
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

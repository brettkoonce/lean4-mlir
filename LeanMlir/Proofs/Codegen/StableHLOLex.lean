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
`parseNat (toString n) = n` is the one lemma the *whole* lexer is built on, and
it is the genuinely-hard, fully-reusable part (a fuel induction over
`Nat.toDigitsCore`, no Mathlib analysis — `[propext, Classical.choice, Quot.sound]`).

## Why a from-scratch `List Char` codec (not `String.toNat?`)

A 2026-06-27 probe established that Lean-core string *parsing* primitives —
`String.toNat?`, `String.splitOn` — are **kernel-opaque**: they do not reduce
under `decide`/`rfl` (they fold over `String.Pos`/`Substring` iterators). So
(a) there is no off-the-shelf `(toString n).toNat? = some n`, and (b) a concrete
`decide`-the-instance shortcut is impossible. A verified lexer must therefore be
built at the `List Char` level with *structurally-recursive* functions and proven
by induction — which is exactly what `parseNat`/`dstep` below are. (Rendering —
`toString`, `ty`, `++` — *does* reduce by `rfl`, so the emit side needs nothing.)

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
re-discovered from scratch. See `planning/tier23_float_and_syntactic_faithfulness.md`
(Part B VERDICT block).

## Residue (unchanged, state wherever cited)

Even the full `parse (lex (pretty g)) = some (skel g)` closes only the **lexer**.
It does *not* close per-op StableHLO *spec* conformance, IREE lowering, or
`float32 ≈ ℝ` — those stay validated by `iree-compile` + the GPU runs.
-/

namespace Proofs
namespace StableHLO

open Nat

-- ════════════════════════════════════════════════════════════════
-- § Decimal digit ⟷ Char
-- ════════════════════════════════════════════════════════════════

/-- Inverse of `Nat.digitChar` on `'0'..'9'` (the chars `toString` emits). -/
def digitVal (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then some (c.toNat - '0'.toNat) else none

/-- One big-endian Horner step: `acc * 10 + digit`. -/
def dstep (acc : Nat) (c : Char) : Nat := acc * 10 + (digitVal c).getD 0

/-- `digitVal` is the value inverse of `Nat.digitChar` on single decimal digits. -/
theorem digitVal_digitChar_all : ∀ d, d < 10 → (digitVal (Nat.digitChar d)).getD 0 = d := by
  decide

/-- A Horner step on a rendered digit is `acc * 10 + d`. -/
theorem dstep_digitChar (acc d : Nat) (h : d < 10) :
    dstep acc (Nat.digitChar d) = acc * 10 + d := by
  unfold dstep; rw [digitVal_digitChar_all d h]

-- ════════════════════════════════════════════════════════════════
-- § The decimal round-trip (fuel induction over `Nat.toDigitsCore`)
-- ════════════════════════════════════════════════════════════════

/-- The accumulator argument of `toDigitsCore` is always a pure **suffix** of
    its output: digits are prepended in front of `l`. -/
theorem toDigitsCore_suffix (f : Nat) :
    ∀ (n : Nat) (l : List Char),
      Nat.toDigitsCore 10 f n l = Nat.toDigitsCore 10 f n [] ++ l := by
  induction f with
  | zero => intro n l; simp [Nat.toDigitsCore]
  | succ f ih =>
    intro n l
    simp only [Nat.toDigitsCore]
    by_cases hx : n / 10 = 0
    · simp [hx]
    · simp only [hx, if_false]
      rw [ih (n/10) (Nat.digitChar (n % 10) :: l), ih (n/10) [Nat.digitChar (n % 10)]]
      simp

/-- **Horner value.** Folding `dstep` over the digit list of `n` (big-endian)
    from `acc` yields `acc * 10^(#digits) + n`. The inductive heart of the
    round-trip. -/
theorem foldl_dstep_toDigitsCore (f : Nat) :
    ∀ (n acc : Nat), n < 10 ^ f →
      List.foldl dstep acc (Nat.toDigitsCore 10 f n []) =
        acc * 10 ^ (Nat.toDigitsCore 10 f n []).length + n := by
  induction f with
  | zero => intro n acc h; simp at h; simp [Nat.toDigitsCore, h]
  | succ f ih =>
    intro n acc h
    simp only [Nat.toDigitsCore]
    by_cases hx : n / 10 = 0
    · have hn : n < 10 := by omega
      have hmod : n % 10 = n := Nat.mod_eq_of_lt hn
      simp only [hx, if_true]
      simp only [List.foldl, List.length]
      rw [hmod, dstep_digitChar acc n hn]
      ring
    · have hlt : n / 10 < 10 ^ f := by
        rw [pow_succ] at h; exact Nat.div_lt_of_lt_mul (by omega)
      have hmodlt : n % 10 < 10 := Nat.mod_lt n (by omega)
      simp only [hx, if_false]
      rw [toDigitsCore_suffix f (n/10) [Nat.digitChar (n % 10)]]
      rw [List.foldl_append, ih (n/10) acc hlt]
      simp only [List.foldl, List.length_append, List.length]
      rw [dstep_digitChar _ (n % 10) hmodlt]
      have : 10 * (n / 10) + n % 10 = n := by omega
      ring_nf
      omega

/-- `n < 10^(n+1)` — the fuel `Nat.toDigits` allocates (`n+1`) always suffices. -/
theorem lt_ten_pow_succ (n : Nat) : n < 10 ^ (n + 1) := by
  have h1 : n < 2 ^ n := Nat.lt_two_pow_self
  have h2 : (2:Nat) ^ n ≤ 10 ^ n := Nat.pow_le_pow_left (by omega) n
  have h3 : (10:Nat) ^ n ≤ 10 ^ (n+1) := Nat.pow_le_pow_right (by omega) (by omega)
  omega

/-- `toString n` (= `Nat.repr n`) is exactly `Nat.toDigitsCore 10 (n+1) n []`. -/
theorem toString_toList (n : Nat) :
    (toString n).toList = Nat.toDigitsCore 10 (n+1) n [] := by
  show (Nat.repr n).toList = _
  unfold Nat.repr Nat.toDigits
  rw [String.toList_ofList]

-- ════════════════════════════════════════════════════════════════
-- § Keystone
-- ════════════════════════════════════════════════════════════════

/-- The lexer's numeric core: decode a string's chars as a big-endian decimal.
    (Non-validating — it only ever sees `emit`'s output, which is always digits;
    rejecting non-digit input is a robustness property, not a faithfulness one.) -/
def parseNat (s : String) : Nat := List.foldl dstep 0 s.toList

/-- **Decimal round-trip keystone.** Parsing the rendered decimal of any `n`
    recovers `n` — the one lemma the whole verified lexer rests on. -/
theorem parseNat_toString (n : Nat) : parseNat (toString n) = n := by
  unfold parseNat
  rw [toString_toList n, foldl_dstep_toDigitsCore (n+1) n 0 (lt_ten_pow_succ n)]
  simp

end StableHLO
end Proofs

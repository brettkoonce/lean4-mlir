import LeanMlir.Proofs.StableHLO

/-! # R4 — the syntactic half (4a), structural core

`StableHLO.lean` closes the **semantic** half of R4: `den (emit g) = fderiv`.
The **syntactic** half asks that the emitted text be a *faithful, recoverable*
encoding of the graph — `parse (pretty a) = a` (the doc's "4a"). A verified
lexer/parser of the literal SSA StableHLO text (with multi-instruction op
expansion + name resolution) is a large separate build; this file lands its
**structural core**, which is the load-bearing part:

* `Raw` — the renderable **skeleton** of an `SHlo` graph (opcodes + shapes +
  leaf SSA names, with the `ℝ` operand values and the shape index erased: the
  text never carries the runtime values, only the op structure).
* `skel : SHlo n → Raw` — extract that skeleton.
* `toToks : Raw → List Tok` — a postorder token serialization (the order
  `pretty` already emits in: children before parent).
* `parse : List Tok → Option Raw` — a stack reconstructor.
* **`parse_skel` / `roundtrip`** — `parse (toToks (skel a)) = some (skel a)`:
  the op-graph is recovered exactly from its serialization. Proven by
  structural induction (no string reasoning, no SSA-freshness bookkeeping).

What this buys: the *structure* of the emitted graph (which op, which operands,
which shapes) is now **proven** recoverable — it leaves the trusted surface.
What remains audited (the thin lexical boundary): the per-op `Tok ↔ StableHLO
text` map — i.e. that `stablehlo.dot_general … contracting_dims = [1] x [0]`
*is* the string for a `dotIn` token. That, plus per-op spec conformance, IREE
lowering, and `float32 ≈ ℝ`, is the residue (validated by `iree-compile` + the
GPU runs). Closes under `[propext, Classical.choice, Quot.sound]`.
-/

namespace Proofs
namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Renderable skeleton + its postorder token serialization
-- ════════════════════════════════════════════════════════════════

/-- The renderable skeleton of an `SHlo` graph: op structure + shapes + leaf
    SSA names, with `ℝ` values and the shape index erased. This is exactly the
    information that reaches the emitted text. -/
inductive Raw where
  | operand    (name : String) (n : Nat)            : Raw
  | dotIn      (w : String) (m n : Nat)             : Raw → Raw
  | dotOut     (w : String) (m n : Nat)             : Raw → Raw
  | addBcast   (b : String) (n : Nat)               : Raw → Raw
  | expe       (n : Nat)                            : Raw → Raw
  | softmaxDiv (n : Nat)                            : Raw → Raw
  | sub        (n : Nat)                            : Raw → Raw → Raw
  | reluF      (n : Nat)                            : Raw → Raw
  | selectPos  (x : String) (n : Nat)               : Raw → Raw
deriving DecidableEq, Repr, Inhabited

/-- Erase an `SHlo` graph to its renderable skeleton. -/
def skel : {k : Nat} → SHlo k → Raw
  | k, .operand name _      => .operand name k
  | k, .dotIn (m := m) w _ e  => .dotIn w m k (skel e)
  | k, .dotOut (n := n) w _ e => .dotOut w k n (skel e)
  | k, .addBcast b _ e      => .addBcast b k (skel e)
  | k, .expe e              => .expe k (skel e)
  | k, .softmaxDiv e        => .softmaxDiv k (skel e)
  | k, .sub a b             => .sub k (skel a) (skel b)
  | k, .reluF e             => .reluF k (skel e)
  | k, .selectPos x _ e     => .selectPos x k (skel e)

/-- One serialized token: an opcode with its shapes/names but no operands
    (operands are recovered positionally from a postorder stream). -/
inductive Tok where
  | operand    (name : String) (n : Nat)  : Tok
  | dotIn      (w : String) (m n : Nat)    : Tok
  | dotOut     (w : String) (m n : Nat)    : Tok
  | addBcast   (b : String) (n : Nat)      : Tok
  | expe       (n : Nat)                   : Tok
  | softmaxDiv (n : Nat)                   : Tok
  | sub        (n : Nat)                   : Tok
  | reluF      (n : Nat)                   : Tok
  | selectPos  (x : String) (n : Nat)      : Tok
deriving DecidableEq, Repr

/-- Postorder serialization: children, then the node's opcode token. -/
def toToks : Raw → List Tok
  | .operand nm n    => [.operand nm n]
  | .dotIn w m n e   => toToks e ++ [.dotIn w m n]
  | .dotOut w m n e  => toToks e ++ [.dotOut w m n]
  | .addBcast b n e  => toToks e ++ [.addBcast b n]
  | .expe n e        => toToks e ++ [.expe n]
  | .softmaxDiv n e  => toToks e ++ [.softmaxDiv n]
  | .sub n a b       => toToks a ++ toToks b ++ [.sub n]
  | .reluF n e       => toToks e ++ [.reluF n]
  | .selectPos x n e => toToks e ++ [.selectPos x n]

/-- Stack reconstructor: fold the token stream, pushing operands and applying
    each opcode to the top of the stack (popping its arity). -/
def parseStack : List Tok → List Raw → Option (List Raw)
  | [], st                       => some st
  | .operand nm n :: ts, st      => parseStack ts (.operand nm n :: st)
  | .dotIn w m n :: ts, e :: st  => parseStack ts (.dotIn w m n e :: st)
  | .dotOut w m n :: ts, e :: st => parseStack ts (.dotOut w m n e :: st)
  | .addBcast b n :: ts, e :: st => parseStack ts (.addBcast b n e :: st)
  | .expe n :: ts, e :: st       => parseStack ts (.expe n e :: st)
  | .softmaxDiv n :: ts, e :: st => parseStack ts (.softmaxDiv n e :: st)
  | .sub n :: ts, b :: a :: st   => parseStack ts (.sub n a b :: st)
  | .reluF n :: ts, e :: st      => parseStack ts (.reluF n e :: st)
  | .selectPos x n :: ts, e :: st => parseStack ts (.selectPos x n e :: st)
  | _ :: _, _                    => none  -- stack underflow / malformed

/-- Parse a full token stream back to a single graph. -/
def parse (ts : List Tok) : Option Raw :=
  match parseStack ts [] with
  | some [r] => some r
  | _        => none

-- ════════════════════════════════════════════════════════════════
-- § Round-trip: the serialization recovers the op-graph exactly
-- ════════════════════════════════════════════════════════════════

/-- **Stack invariant.** Serializing `r` and folding it onto a stack `st`
    pushes exactly `r`. The generalized statement that drives the round-trip. -/
theorem parseStack_toToks (r : Raw) :
    ∀ (ts : List Tok) (st : List Raw),
      parseStack (toToks r ++ ts) st = parseStack ts (r :: st) := by
  induction r with
  | operand nm n => intro ts st; rfl
  | dotIn w m n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | dotOut w m n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | addBcast b n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | expe n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | softmaxDiv n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | sub n a b iha ihb => intro ts st; simp only [toToks, List.append_assoc, iha, ihb]; rfl
  | reluF n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | selectPos x n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl

/-- **Serialization round-trip.** `parse` recovers any skeleton from its
    postorder token stream. -/
theorem parse_toToks (r : Raw) : parse (toToks r) = some r := by
  unfold parse
  have h : parseStack (toToks r) [] = some [r] := by
    have := parseStack_toToks r [] []
    simpa using this
  rw [h]

/-- **R4 syntactic core.** The emitted op-graph (skeleton) of any `SHlo` is a
    faithful, recoverable serialization: `parse (toToks (skel a)) = some (skel a)`.
    The op structure / shapes / SSA names leave the trusted surface; only the
    per-op `Tok ↔ StableHLO-text` lexing stays audited. -/
theorem roundtrip {k : Nat} (a : SHlo k) : parse (toToks (skel a)) = some (skel a) :=
  parse_toToks (skel a)

end StableHLO
end Proofs

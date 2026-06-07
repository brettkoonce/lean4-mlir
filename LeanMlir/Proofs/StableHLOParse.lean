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
-- § Parser for the postorder token serialization (`Raw`/`Tok`/`toToks`/`skel`
--   live in StableHLO.lean — the emitter shares them).
-- ════════════════════════════════════════════════════════════════

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
  | .relu6F n :: ts, e :: st     => parseStack ts (.relu6F n e :: st)
  | .selectMid x n :: ts, e :: st => parseStack ts (.selectMid x n e :: st)
  | .flatConvF w b ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.flatConvF w b ic oc h w' kH kW e :: st)
  | .maxPoolF c h w :: ts, e :: st => parseStack ts (.maxPoolF c h w e :: st)
  | .convBack w ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convBack w ic oc h w' kH kW e :: st)
  | .maxPoolBack x c h w :: ts, e :: st => parseStack ts (.maxPoolBack x c h w e :: st)
  | .bnF g b eps n :: ts, e :: st => parseStack ts (.bnF g b eps n e :: st)
  | .bnBack g x eps n :: ts, e :: st => parseStack ts (.bnBack g x eps n e :: st)
  | .addV n :: ts, b :: a :: st  => parseStack ts (.addV n a b :: st)
  | .gapF c h w :: ts, e :: st   => parseStack ts (.gapF c h w e :: st)
  | .flatConvStridedF w b ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.flatConvStridedF w b ic oc h w' kH kW e :: st)
  | .convStridedBack w ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convStridedBack w ic oc h w' kH kW e :: st)
  | .bnPerChannelF g b eps oc h w :: ts, e :: st =>
      parseStack ts (.bnPerChannelF g b eps oc h w e :: st)
  | .bnPerChannelBack g x eps oc h w :: ts, e :: st =>
      parseStack ts (.bnPerChannelBack g x eps oc h w e :: st)
  | .depthwiseF w b c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseF w b c h w' kH kW e :: st)
  | .depthwiseBack w c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseBack w c h w' kH kW e :: st)
  | .depthwiseStridedF w b c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseStridedF w b c h w' kH kW e :: st)
  | .depthwiseStridedBack w c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseStridedBack w c h w' kH kW e :: st)
  | .swishF n :: ts, e :: st     => parseStack ts (.swishF n e :: st)
  | .swishBack x n :: ts, e :: st => parseStack ts (.swishBack x n e :: st)
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
  | relu6F n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | selectMid x n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | flatConvF w b ic oc h w' kH kW e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | maxPoolF c h w e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | convBack w ic oc h w' kH kW e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | maxPoolBack x c h w e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | bnF g b eps n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | bnBack g x eps n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | addV n a b iha ihb => intro ts st; simp only [toToks, List.append_assoc, iha, ihb]; rfl
  | gapF c h w e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | flatConvStridedF w b ic oc h w' kH kW e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | convStridedBack w ic oc h w' kH kW e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | bnPerChannelF g b eps oc h w e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | bnPerChannelBack g x eps oc h w e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | depthwiseF w b c h w' kH kW e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | depthwiseBack w c h w' kH kW e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | depthwiseStridedF w b c h w' kH kW e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | depthwiseStridedBack w c h w' kH kW e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | swishF n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl
  | swishBack x n e ih => intro ts st; simp only [toToks, List.append_assoc, ih]; rfl

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

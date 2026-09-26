import LeanMlir.Proofs.Codegen.StableHLO.Pretty

/-! # StableHLOParse — the token skeleton of an `SHlo` term round-trips

`StableHLO.Basic` states what each `SHlo` graph denotes: its `*_faithful` theorems equate `den` of a node or
graph with a named ℝ function (`fwdGraph_faithful : den (fwdGraph W b x) = mnistLinear W b x`, …).
This file is about the other side, the encoding the printer works from:

* `Raw` — the skeleton of an `SHlo` graph (opcodes, shapes, leaf SSA names; the `ℝ` operand
  values and the shape index erased).
* `skel : SHlo n → Raw` — extract that skeleton.
* `toToks : Raw → List Tok` — the postorder token serialization `pretty` prints from
  (children before parent).
* `parse : List Tok → Option Raw` — a stack reconstructor.
* `parse_toToks` / `roundtrip` — `parse (toToks (skel a)) = some (skel a)`, by structural
  induction.

Scope: the round trip is a statement about `toToks`, not about the text. Which operands each op's
emitted line reads, in what order, with what types, is decided by `emitTok` (in
StableHLO/Pretty.lean) and is trusted, together with the per-op lexical syntax, the per-op
StableHLO semantics, the lowering, and `float32 ≈ ℝ`. -/

namespace Proofs
namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Parser for the postorder token serialization (`Raw`/`Tok`/`toToks`/`skel`
--   live in StableHLO/Basic.lean — the emitter shares them).
-- ════════════════════════════════════════════════════════════════

/-- Stack reconstructor: fold the token stream, pushing operands and applying
    each opcode to the top of the stack (popping its arity). -/
def parseStack : List Tok → List Raw → Option (List Raw)
  | [], st                       => some st
  | .operand nm n :: ts, st      => parseStack ts (.operand nm n :: st)
  | .allReduceMean R t ds :: ts, e :: st => parseStack ts (.allReduceMean R t ds e :: st)
  | .dotIn w m n :: ts, e :: st  => parseStack ts (.dotIn w m n e :: st)
  | .dotInBf16 w m n :: ts, e :: st => parseStack ts (.dotInBf16 w m n e :: st)
  | .convertF n :: ts, e :: st   => parseStack ts (.convertF n e :: st)
  | .dotOut w m n :: ts, e :: st => parseStack ts (.dotOut w m n e :: st)
  | .addBcast b n :: ts, e :: st => parseStack ts (.addBcast b n e :: st)
  | .expe n :: ts, e :: st       => parseStack ts (.expe n e :: st)
  | .softmaxDiv n :: ts, e :: st => parseStack ts (.softmaxDiv n e :: st)
  | .sub n :: ts, b :: a :: st   => parseStack ts (.sub n a b :: st)
  | .weightSgd xN wN lrS m n :: ts, e :: st => parseStack ts (.weightSgd xN wN lrS m n e :: st)
  | .biasSgd bN lrS n :: ts, e :: st        => parseStack ts (.biasSgd bN lrS n e :: st)
  | .convWeightSgd xN wN lrS ic oc h w kH kW :: ts, e :: st =>
      parseStack ts (.convWeightSgd xN wN lrS ic oc h w kH kW e :: st)
  | .convBiasSgd bN lrS oc h w :: ts, e :: st =>
      parseStack ts (.convBiasSgd bN lrS oc h w e :: st)
  | .bnGammaSgd gN vN es lrS oc h w :: ts, e :: st =>
      parseStack ts (.bnGammaSgd gN vN es lrS oc h w e :: st)
  | .bnBetaSgd bN lrS oc h w :: ts, e :: st =>
      parseStack ts (.bnBetaSgd bN lrS oc h w e :: st)
  | .layerScaleChGammaSgd gN xN lrS c h w :: ts, e :: st =>
      parseStack ts (.layerScaleChGammaSgd gN xN lrS c h w e :: st)
  | .lnGammaSgd gN xN es lrS n :: ts, e :: st =>
      parseStack ts (.lnGammaSgd gN xN es lrS n e :: st)
  | .veclnGammaSgd gN xN es lrS N D :: ts, e :: st =>
      parseStack ts (.veclnGammaSgd gN xN es lrS N D e :: st)
  | .patchEmbedWeightSgd wN xN lrS ic H W P N D :: ts, e :: st =>
      parseStack ts (.patchEmbedWeightSgd wN xN lrS ic H W P N D e :: st)
  | .lnBetaSgd bN lrS n :: ts, e :: st =>
      parseStack ts (.lnBetaSgd bN lrS n e :: st)
  | .reluF n :: ts, e :: st      => parseStack ts (.reluF n e :: st)
  | .selectPos x n :: ts, e :: st => parseStack ts (.selectPos x n e :: st)
  | .relu6F n :: ts, e :: st     => parseStack ts (.relu6F n e :: st)
  | .selectMid x n :: ts, e :: st => parseStack ts (.selectMid x n e :: st)
  | .flatConvF w b ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.flatConvF w b ic oc h w' kH kW e :: st)
  | .flatConvFBf16 w b ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.flatConvFBf16 w b ic oc h w' kH kW e :: st)
  | .maxPoolF c h w :: ts, e :: st => parseStack ts (.maxPoolF c h w e :: st)
  | .convBack w ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convBack w ic oc h w' kH kW e :: st)
  | .maxPoolBack x c h w :: ts, e :: st => parseStack ts (.maxPoolBack x c h w e :: st)
  | .bnF g b eps n :: ts, e :: st => parseStack ts (.bnF g b eps n e :: st)
  | .bnBack g x eps n :: ts, e :: st => parseStack ts (.bnBack g x eps n e :: st)
  | .addV n :: ts, b :: a :: st  => parseStack ts (.addV n a b :: st)
  | .gapF c h w :: ts, e :: st   => parseStack ts (.gapF c h w e :: st)
  | .gapBack c h w :: ts, e :: st => parseStack ts (.gapBack c h w e :: st)
  | .broadcastBack c h w :: ts, e :: st => parseStack ts (.broadcastBack c h w e :: st)
  | .flatConvStridedF w b ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.flatConvStridedF w b ic oc h w' kH kW e :: st)
  | .flatConvStridedXlaF w b ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.flatConvStridedXlaF w b ic oc h w' kH kW e :: st)
  | .depthwiseStridedXlaF w b c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseStridedXlaF w b c h w' kH kW e :: st)
  | .convStridedBack w ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convStridedBack w ic oc h w' kH kW e :: st)
  | .convStridedWeightSgd xN wN lrS ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convStridedWeightSgd xN wN lrS ic oc h w' kH kW e :: st)
  | .depthwiseWeightSgd xN wN lrS c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseWeightSgd xN wN lrS c h w' kH kW e :: st)
  | .depthwiseStridedWeightSgd xN wN lrS c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseStridedWeightSgd xN wN lrS c h w' kH kW e :: st)
  | .convStridedXlaWeightSgd xN wN lrS ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convStridedXlaWeightSgd xN wN lrS ic oc h w' kH kW e :: st)
  | .depthwiseStridedXlaWeightSgd xN wN lrS c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseStridedXlaWeightSgd xN wN lrS c h w' kH kW e :: st)
  | .flatConvStride4F w b ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.flatConvStride4F w b ic oc h w' kH kW e :: st)
  | .bnPerChannelF g b eps oc h w :: ts, e :: st =>
      parseStack ts (.bnPerChannelF g b eps oc h w e :: st)
  | .bnPerChannelBack g x eps oc h w :: ts, e :: st =>
      parseStack ts (.bnPerChannelBack g x eps oc h w e :: st)
  | .bnPerChannelEvalF g b mu var eps oc h w :: ts, e :: st =>
      parseStack ts (.bnPerChannelEvalF g b mu var eps oc h w e :: st)
  | .weightGrad x m n :: ts, e :: st => parseStack ts (.weightGrad x m n e :: st)
  | .biasGrad n :: ts, e :: st => parseStack ts (.biasGrad n e :: st)
  | .convWeightGrad x ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convWeightGrad x ic oc h w' kH kW e :: st)
  | .convBiasGrad ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convBiasGrad ic oc h w' kH kW e :: st)
  | .convStridedWeightGrad x ic oc h w' kH kW :: ts, e :: st =>
      parseStack ts (.convStridedWeightGrad x ic oc h w' kH kW e :: st)
  | .bnGammaGrad v eps oc h w' :: ts, e :: st => parseStack ts (.bnGammaGrad v eps oc h w' e :: st)
  | .bnBetaGrad oc h w' :: ts, e :: st => parseStack ts (.bnBetaGrad oc h w' e :: st)
  | .adamMNextF m b1 ob1 ds :: ts, e :: st => parseStack ts (.adamMNextF m b1 ob1 ds e :: st)
  | .adamVNextF v b2 ob2 ds :: ts, e :: st => parseStack ts (.adamVNextF v b2 ob2 ds e :: st)
  | .adamWParamF θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd ds :: ts, e :: st =>
      parseStack ts (.adamWParamF θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd ds e :: st)
  | .sgdParamF θ lr ds :: ts, e :: st => parseStack ts (.sgdParamF θ lr ds e :: st)
  | .momVNextF v mu ds :: ts, e :: st => parseStack ts (.momVNextF v mu ds e :: st)
  | .momParamF θ v mu lr ds :: ts, e :: st => parseStack ts (.momParamF θ v mu lr ds e :: st)
  | .rmsBufNextF sq buf rho orho mu eps ds :: ts, e :: st =>
      parseStack ts (.rmsBufNextF sq buf rho orho mu eps ds e :: st)
  -- Global-norm grad clip. ⚠ The two BINARY ones pop right-then-left (`.addV`'s shape), and for
  -- `clipScaleF` the deeper element is the FACTOR — `toToks` emits factor-then-gradient.
  | .gradSumSqAccF ds :: ts, g :: acc :: st => parseStack ts (.gradSumSqAccF ds acc g :: st)
  | .clipScaleF cS eS ds :: ts, g :: sN :: st => parseStack ts (.clipScaleF cS eS ds sN g :: st)
  -- LAMB. ⚠ `lambDirF` is UNARY (θ/m/v ride as NAMES, like `adamWParamF`); `lambScaleF` is binary
  -- with the deeper element the ‖θ‖² SCALAR — `toToks` emits scalar-then-direction, the
  -- `clipScaleF` order exactly.
  | .lambDirF θ m v b1 ob1 b2 ob2 bc1 bc2 eps wd ds :: ts, e :: st =>
      parseStack ts (.lambDirF θ m v b1 ob1 b2 ob2 bc1 bc2 eps wd ds e :: st)
  | .lambScaleF ds :: ts, r :: wn2 :: st => parseStack ts (.lambScaleF ds wn2 r :: st)
  | .depthwiseF w b c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseF w b c h w' kH kW e :: st)
  | .depthwiseBack w c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseBack w c h w' kH kW e :: st)
  | .depthwiseStridedF w b c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseStridedF w b c h w' kH kW e :: st)
  | .depthwiseStridedBack w c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseStridedBack w c h w' kH kW e :: st)
  | .depthwiseStridedXlaBack w c h w' kH kW :: ts, e :: st =>
      parseStack ts (.depthwiseStridedXlaBack w c h w' kH kW e :: st)
  | .swishF n :: ts, e :: st     => parseStack ts (.swishF n e :: st)
  | .swishBack x n :: ts, e :: st => parseStack ts (.swishBack x n e :: st)
  | .sigmoidF n :: ts, e :: st   => parseStack ts (.sigmoidF n e :: st)
  | .sigmoidBack x n :: ts, e :: st => parseStack ts (.sigmoidBack x n e :: st)
  | .geluF n :: ts, e :: st      => parseStack ts (.geluF n e :: st)
  | .geluBack x n :: ts, e :: st => parseStack ts (.geluBack x n e :: st)
  | .layerScaleF γ n :: ts, e :: st => parseStack ts (.layerScaleF γ n e :: st)
  | .layerScaleChF γ c h w :: ts, e :: st => parseStack ts (.layerScaleChF γ c h w e :: st)
  | .softmaxRowF m n :: ts, e :: st => parseStack ts (.softmaxRowF m n e :: st)
  | .softmaxRowBack x m n :: ts, e :: st => parseStack ts (.softmaxRowBack x m n e :: st)
  | .matmulF m k n :: ts, b :: a :: st => parseStack ts (.matmulF m k n a b :: st)
  | .transposeF m n :: ts, e :: st => parseStack ts (.transposeF m n e :: st)
  | .scaleF s n :: ts, e :: st => parseStack ts (.scaleF s n e :: st)
  | .lnRowF g b eps m n :: ts, e :: st => parseStack ts (.lnRowF g b eps m n e :: st)
  | .lnRowBack g x eps m n :: ts, e :: st => parseStack ts (.lnRowBack g x eps m n e :: st)
  | .denseRowF w b N a c :: ts, e :: st => parseStack ts (.denseRowF w b N a c e :: st)
  | .denseRowBack w N a c :: ts, e :: st => parseStack ts (.denseRowBack w N a c e :: st)
  | .patchEmbedF w b cls pos ic H W P N D :: ts, e :: st =>
      parseStack ts (.patchEmbedF w b cls pos ic H W P N D e :: st)
  | .clsSliceF N D :: ts, e :: st => parseStack ts (.clsSliceF N D e :: st)
  | .clsPadF N D :: ts, e :: st => parseStack ts (.clsPadF N D e :: st)
  | .rowScaleF g m n :: ts, e :: st => parseStack ts (.rowScaleF g m n e :: st)
  | .rowBiasF b m n :: ts, e :: st => parseStack ts (.rowBiasF b m n e :: st)
  | .headSliceF N heads d hIdx :: ts, e :: st => parseStack ts (.headSliceF N heads d hIdx e :: st)
  | .headPadF N heads d hIdx :: ts, e :: st => parseStack ts (.headPadF N heads d hIdx e :: st)
  | .batched tag names info :: ts, e :: st => parseStack ts (.batched tag names info e :: st)
  | .batched2 tag names info :: ts, b :: a :: st =>
      parseStack ts (.batched2 tag names info a b :: st)
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
  -- every constructor pushes its children then itself: peel `toToks`, reassociate, apply the
  -- children's hypotheses, and `parseStack` reduces the pushed token by `rfl`.
  induction r <;> intro ts st <;> simp only [toToks, List.append_assoc, *] <;> rfl

/-- **Serialization round-trip.** `parse` recovers any skeleton from its
    postorder token stream. -/
theorem parse_toToks (r : Raw) : parse (toToks r) = some r := by
  unfold parse
  have h : parseStack (toToks r) [] = some [r] := by
    have := parseStack_toToks r [] []
    rw [List.append_nil] at this
    exact this
  rw [h]

/-- **Skeleton round trip.** The postorder token encoding of the skeleton of any `SHlo`
    term is invertible: `parse (toToks (skel a)) = some (skel a)`. It is a statement about
    `toToks`, not about the text: which operands each op's emitted line reads, in what order,
    and with what types is decided by `emitTok` and remains trusted along with the lexical
    syntax. -/
theorem roundtrip {k : Nat} (a : SHlo k) : parse (toToks (skel a)) = some (skel a) :=
  parse_toToks (skel a)

end StableHLO
end Proofs

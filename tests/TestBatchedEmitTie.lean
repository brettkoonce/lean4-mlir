import LeanMlir.Proofs.Codegen.StableHLO

/-! # The batched pointwise/row forms emit exactly what their per-example peers emit

`planning/xla_pjrt_handoff.md` §2b moved the batched renderers off the `N := 1` batch-unit
convention onto the honest batched index `N := B`. That required batched peers for every op whose
emitter reads its width off the SHlo index — the pointwise ops and the two row ops — because at
index `N·n` those emitted `tensor<B×(N·n)>`, a type that does not match their own operand.

Each batched form exists ONLY to move the batch out of the emit width. So the whole design claim is:
**the batched form renders byte-for-byte what the per-example form renders.** For EfficientNet that
was witnessed by `verified_mlir/efficientnet_train_step.mlir` coming back byte-identical, but that
is a whole-net check on one net that happens to use eight of the nine forms. This file pins each
form individually, including `relu`/`selectPos`, which ResNet-34 needs and EfficientNet never
exercises — so the tie is nailed down BEFORE the batched R34 render depends on it.

What this does NOT check is the `den` side: the render is value-independent (`skel` erases values),
so a form with the wrong denotation emits identical bytes. That half is
`den_batchOp_swish_eq_swishF` / `den_batchOp_relu_eq_reluF` / `selectPosB_faithful` and the `rfl`
faithfulness lemmas in `StableHLO.lean`. Both halves are needed; neither implies the other.

    lake env lean tests/TestBatchedEmitTie.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def n : Nat := 12
private def rows : Nat := 3
private def a : Nat := 5
private def c : Nat := 7
-- pool dims (input is 2h×2w) and conv dims for the bias-grad cases
private def pc : Nat := 2
private def ph : Nat := 3
private def ic : Nat := 2
private def oc : Nat := 3
private def ch : Nat := 4
private def kk : Nat := 3
-- ViT-shaped stand-ins for the transformer gradient cases: `tk` tokens (so the token axis is
-- `tk+1` with CLS), `dm` model dim, `pp × pp` patches over a `pi × pH × pW` image. `pH/pp = 2` and
-- `pW/pp = 2` gives tk = 4 patch tokens, which is what makes the patch-embed shapes consistent.
private def tk : Nat := 4
private def dm : Nat := 6
private def pp : Nat := 2
private def pi : Nat := 3
private def pH : Nat := 4
private def pW : Nat := 4

private def render (g : StateM Nat (String × String)) : String := (g.run' 0).1

/-- Per-example peer (SHlo index `n`) vs batched form (SHlo index `BS*n`), same `pretty BS`. -/
private def cases : List (String × String × String) :=
  let zv  : Vec n := fun _ => 0
  let zvb : Vec (BS*n) := fun _ => 0
  let zr  : Vec (rows*c) := fun _ => 0
  let zrb : Vec (BS*(rows*c)) := fun _ => 0
  let zW  : Mat a c := fun _ _ => 0
  let zc2  : Vec pc := fun _ => 0
  let zc   : Vec c := fun _ => 0
  let zq0  : Vec (pc*ph*ph) := fun _ => 0
  let zq0b : Vec (BS*(pc*ph*ph)) := fun _ => 0
  [ ("swish",
     render (pretty BS (.swishF (.operand "%x" zv))),
     render (pretty BS (.batchOp (N := BS) (.swish (n := n)) (.operand "%x" zvb))))
  , ("relu",
     render (pretty BS (.reluF (.operand "%x" zv))),
     render (pretty BS (.batchOp (N := BS) (.relu (n := n)) (.operand "%x" zvb))))
  , ("swishBack",
     render (pretty BS (.swishBack "%s" zv (.operand "%x" zv))),
     render (pretty BS (.swishBackB "%s" zvb (.operand "%x" zvb))))
  , ("sigmoidBack",
     render (pretty BS (.sigmoidBack "%s" zv (.operand "%x" zv))),
     render (pretty BS (.sigmoidBackB "%s" zvb (.operand "%x" zvb))))
  , ("selectPos",
     render (pretty BS (.selectPos "%s" zv (.operand "%x" zv))),
     render (pretty BS (.selectPosB "%s" zvb (.operand "%x" zvb))))
  -- ── MobileNetV2's activation (§2f). `relu6` is a descriptor (pointwise, carries nothing);
  --    `selectMid` is NOT (the two-sided mask reads the saved per-example pre-activation), so it
  --    gets its own `selectMidB` holding the whole-batch `x`. Same split as relu/selectPos. ──
  , ("relu6",
     render (pretty BS (.relu6F (.operand "%x" zv))),
     render (pretty BS (.batchOp (N := BS) (.relu6 (n := n)) (.operand "%x" zvb))))
  , ("selectMid",
     render (pretty BS (.selectMid "%s" zv (.operand "%x" zv))),
     render (pretty BS (.selectMidB "%s" zvb (.operand "%x" zvb))))
  -- ── the eval forwards (§2a, the `_fwd_eval` move). INFERENCE BN is a descriptor because its
  --    γ/β/μ/var are the driver's frozen running stats — graph inputs shared by the whole batch,
  --    i.e. batch-INVARIANT data, the one thing a descriptor may carry. Contrast `bnBatch`, which
  --    needs its own ctor precisely because it reduces ACROSS examples. ──
  , ("bnEval",
     render (pretty BS (.bnPerChannelEvalF (oc := pc) (h := ph) (w := ph)
                          "%g" "%bt" "%mu" "%var" "1.0e-5" 0 zc2 zc2 zc2 zc2
                          (.operand "%x" zq0))),
     render (pretty BS (.batchOp (N := BS) (.bnEval (oc := pc) (h := ph) (w := ph)
                          "%g" "%bt" "%mu" "%var" "1.0e-5" 0 zc2 zc2 zc2 zc2)
                          (.operand "%x" zq0b))))
  , ("addV",
     render (pretty BS (.addV (.operand "%x" zv) (.operand "%y" zv))),
     render (pretty BS (.addVB (.operand "%x" zvb) (.operand "%y" zvb))))
  , ("sub",
     render (pretty BS (.sub (.operand "%x" zv) (.operand "%y" zv))),
     render (pretty BS (.subB (.operand "%x" zvb) (.operand "%y" zvb))))
  , ("softmaxRow",
     render (pretty BS (.softmaxRowF (m := rows) (n := c) (.operand "%x" zr))),
     render (pretty BS (.batchOp (N := BS) (.softmaxRow (m := rows) (n := c))
                          (.operand "%x" zrb))))
  , ("denseRowBack",
     render (pretty BS (.denseRowBack (N := rows) (a := a) (c := c) "%W" zW (.operand "%x" zr))),
     render (pretty BS (.batchOp (N := BS)
                          (.denseRowBack (rows := rows) (a := a) (c := c) "%W" zW)
                          (.operand "%x" zrb))))
  -- ── ViT / ConvNeXt, the batched-index move (§0.2 ▶2). Five row/pointwise FORWARD forms, shared
  --    by both nets — which is why they go first: every one of them is on ConvNeXt's critical path
  --    AND on ViT's, so the cheaper net pays down roughly half of the dearer one.
  --
  --    ⚠ `rows` here is the token / spatial axis PER EXAMPLE, never the batch. A descriptor's
  --    whole job is to keep those two numbers apart — `N` for the denotation, `rows*c` for the
  --    emit — and a tie at `rows ≠ BS` is what makes a swapped pair a type error rather than a
  --    silent agreement. (`rows` is 3 against `BS` 32 here, so they cannot be confused.)
  , ("gelu",
     render (pretty BS (.geluF (.operand "%x" zv))),
     render (pretty BS (.batchOp (N := BS) (.gelu (n := n)) (.operand "%x" zvb))))
  , ("transpose",
     render (pretty BS (.transposeF (m := rows) (n := c) (.operand "%x" zr))),
     render (pretty BS (.batchOp (N := BS) (.transpose (m := rows) (n := c))
                          (.operand "%x" zrb))))
  , ("lnRow",
     render (pretty BS (.lnRowF "%g" "%bt" "1.0e-5" 0 1 0 (m := rows) (n := c)
                          (.operand "%x" zr))),
     render (pretty BS (.batchOp (N := BS)
                          (.lnRow (m := rows) (n := c) "%g" "%bt" "1.0e-5" 0 1 0)
                          (.operand "%x" zrb))))
  , ("rowScale",
     render (pretty BS (.rowScaleF (m := rows) (n := c) "%g" zc (.operand "%x" zr))),
     render (pretty BS (.batchOp (N := BS) (.rowScale (m := rows) (n := c) "%g" zc)
                          (.operand "%x" zrb))))
  , ("rowBias",
     render (pretty BS (.rowBiasF (m := rows) (n := c) "%bt" zc (.operand "%x" zr))),
     render (pretty BS (.batchOp (N := BS) (.rowBias (m := rows) (n := c) "%bt" zc)
                          (.operand "%x" zrb))))
  -- ── the two SAVED-ACTIVATION backwards (increment 2). Not descriptors, and the reason is the
  --    descriptor rule itself: `batchMap N (denOp op)` is ONE fixed function, so a descriptor
  --    would hand example 0's saved activation to all N. They take the whole-batch `x` instead —
  --    `geluBackB` pointwise (`swishBackB`'s exact shape), `lnRowBackB` via `batchMapAux`.
  --    `den_lnRowBackB_per_example` is the den-side statement of that; this is the emit side.
  , ("geluBack",
     render (pretty BS (.geluBack "%s" zv (.operand "%x" zv))),
     render (pretty BS (.geluBackB "%s" zvb (.operand "%x" zvb))))
  , ("lnRowBack",
     render (pretty BS (.lnRowBack "%g" "%s" "1.0e-5" 0 1 (m := rows) (n := c) zr
                          (.operand "%x" zr))),
     render (pretty BS (.lnRowBackB (N := BS) (m := rows) (n := c) "%g" "%s" "1.0e-5" 0 1 zrb
                          (.operand "%x" zrb))))
  -- ── increment 3: ConvNeXt's stem conv, LayerScale, and the loss-path softmax pair.
  --    ⚠ `expe` and `softmaxDiv` are here for OPPOSITE halves of the §2b defect: `expe`'s den was
  --    already honest at the batched index and only its EMIT read the width off the SHlo index;
  --    `softmaxDiv`'s emit already reduced per example while its DEN would have divided by the
  --    sum over the whole batch. This file can only see the first kind — the second is
  --    `den_batchOp_softmaxDiv`'s job — which is the standing argument for gating both halves.
  , ("expe",
     render (pretty BS (.expe (.operand "%x" zv))),
     render (pretty BS (.batchOp (N := BS) (.expe (n := n)) (.operand "%x" zvb))))
  , ("softmaxDiv",
     render (pretty BS (.softmaxDiv (.operand "%x" zv))),
     render (pretty BS (.batchOp (N := BS) (.softmaxDiv (n := n)) (.operand "%x" zvb))))
  , ("layerScaleCh",
     render (pretty BS (.layerScaleChF (c := pc) (h := ph) (w := ph) "%g" zc2
                          (.operand "%x" zq0))),
     render (pretty BS (.batchOp (N := BS) (.layerScaleCh (c := pc) (h := ph) (w := ph) "%g" zc2)
                          (.operand "%x" zq0b))))
  -- ── ViT increment 1 (handoff §0.2 ▶3): the six forms whose data is batch-INVARIANT.
  --
  --    ⚠⚠ EVERY ONE OF THESE IS DRIVEN AT `tk = 4` AGAINST `BS = 32`, AND THAT IS THE POINT.
  --    ViT's per-example renderer already calls its TOKEN axis `N`, so "the batch" and "the token
  --    count" are the same letter on the two sides of each pair. At `tk ≠ BS` a form that read one
  --    for the other emits a different shape and this file goes red; at `tk = BS` it would agree
  --    silently. That is the same argument the ConvNeXt five carry for `rows ≠ BS`, one net over
  --    and considerably sharper, because here the collision is in the SOURCE TEXT and not only in
  --    the semantics.
  , ("denseRow",
     render (pretty BS (.denseRowF (N := rows) (a := a) (c := c) "%W" "%b" zW zc
                          (.operand "%x" (fun _ => 0 : Vec (rows*a))))),
     render (pretty BS (.batchOp (N := BS) (.denseRow (N := rows) (a := a) (c := c) "%W" "%b" zW zc)
                          (.operand "%x" (fun _ => 0 : Vec (BS*(rows*a)))))))
  , ("clsSlice",
     render (pretty BS (.clsSliceF (N := tk) (D := dm)
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))),
     render (pretty BS (.batchOp (N := BS) (.clsSlice (N := tk) (D := dm))
                          (.operand "%x" (fun _ => 0 : Vec (BS*((tk+1)*dm)))))))
  , ("clsPad",
     render (pretty BS (.clsPadF (N := tk) (D := dm)
                          (.operand "%x" (fun _ => 0 : Vec dm)))),
     render (pretty BS (.batchOp (N := BS) (.clsPad (N := tk) (D := dm))
                          (.operand "%x" (fun _ => 0 : Vec (BS*dm))))))
  -- ⚠ head 1 of 4, NOT head 0, and NOT head 1 of 3: at `h = 0` the pad's `low` is 0 and the slice's offset is 0, so an
  -- emit that dropped the index entirely would still agree. The interior head is the only one that
  -- pins the slice offset. And 1-of-4 rather than 1-of-3 because at 1-of-3 the pad's
  -- `low` and `high` are both `d` — symmetric, so swapping them would be inert too.
  , ("headSlice",
     render (pretty BS (.headSliceF (N := rows) (heads := 4) (d := a) ⟨1, by decide⟩
                          (.operand "%x" (fun _ => 0 : Vec (rows*(4*a)))))),
     render (pretty BS (.batchOp (N := BS) (.headSlice (N := rows) (heads := 4) (d := a)
                          ⟨1, by decide⟩)
                          (.operand "%x" (fun _ => 0 : Vec (BS*(rows*(4*a))))))))
  , ("headPad",
     render (pretty BS (.headPadF (N := rows) (heads := 4) (d := a) ⟨1, by decide⟩
                          (.operand "%x" (fun _ => 0 : Vec (rows*a))))),
     render (pretty BS (.batchOp (N := BS) (.headPad (N := rows) (heads := 4) (d := a)
                          ⟨1, by decide⟩)
                          (.operand "%x" (fun _ => 0 : Vec (BS*(rows*a)))))))
  -- ── ViT increment 2: the six forms that CANNOT be descriptors. Every one of them ALIASES its
  --    per-example peer's `Raw`, so these cases check something slightly different from the rest of
  --    this file: not "the copied emit body still agrees" but "the alias really is wired to that
  --    tag". A mis-aliased form falls through to `// MALFORMED` or emits another op's text, and
  --    both show up here immediately.
  --
  --    ⚠ The four gradients are driven through their `*Sgd` peers in the un-fused block below as
  --    well; here they are pinned against the per-example GRADIENT, which is the comparison that
  --    says the batched ctor did not change the tag's dims.
  -- ⚠ `scale` was relied on by ConvNeXt's byte tie and by 36 ViT SDPA sites without ever being
  --    pinned individually. Added when ViT's forward started depending on it: a whole-net byte tie
  --    localises nothing, which is the standing argument for this file existing at all.
  , ("scale",
     render (pretty BS (.scaleF "0.125" 0 (.operand "%x" zv))),
     render (pretty BS (.scaleB (N := BS) (n := n) "0.125" 0 (.operand "%x" zvb))))
  , ("matmulF",
     render (pretty BS (.matmulF (m := rows) (k := a) (n := c)
                          (.operand "%q" (fun _ => 0 : Vec (rows*a)))
                          (.operand "%k" (fun _ => 0 : Vec (a*c))))),
     render (pretty BS (.matmulFB (N := BS) (m := rows) (k := a) (n := c)
                          (.operand "%q" (fun _ => 0 : Vec (BS*(rows*a))))
                          (.operand "%k" (fun _ => 0 : Vec (BS*(a*c)))))))
  , ("softmaxRowBack",
     render (pretty BS (.softmaxRowBack (m := rows) (n := c) "%s" zr (.operand "%x" zr))),
     render (pretty BS (.softmaxRowBackB (N := BS) (m := rows) (n := c) "%s" zrb
                          (.operand "%x" zrb))))
  , ("rowDenseWeightGradB",
     render (pretty BS (.rowDenseWeightGrad (N := rows) (a := a) (c := c) "%h"
                          (fun _ => 0 : Vec (rows*a)) (.operand "%x" zr))),
     render (pretty BS (.rowDenseWeightGradB (N := BS) (tk := rows) (a := a) (c := c) "%h"
                          (fun _ => 0 : Vec (BS*(rows*a))) (.operand "%x" zrb))))
  , ("posEmbedGradB",
     render (pretty BS (.posEmbedGrad (N := tk) (D := dm)
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))),
     render (pretty BS (.posEmbedGradB (N := BS) (tk := tk) (D := dm)
                          (.operand "%x" (fun _ => 0 : Vec (BS*((tk+1)*dm)))))))
  , ("patchEmbedBiasGradB",
     render (pretty BS (.patchEmbedBiasGrad (N := tk) (c := c)
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*c))))),
     render (pretty BS (.patchEmbedBiasGradB (N := BS) (tk := tk) (c := c)
                          (.operand "%x" (fun _ => 0 : Vec (BS*((tk+1)*c)))))))
  , ("patchEmbedWeightGradB",
     render (pretty BS (.patchEmbedWeightGrad (ic := pi) (H := pH) (W := pW) (P := pp)
                          (N := tk) (D := dm) "%img" (fun _ => 0 : Vec (pi*pH*pW))
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))),
     render (pretty BS (.patchEmbedWeightGradB (N := BS) (ic := pi) (H := pH) (W := pW) (P := pp)
                          (tk := tk) (D := dm) "%img" (fun _ => 0 : Vec (BS*(pi*pH*pW)))
                          (.operand "%x" (fun _ => 0 : Vec (BS*((tk+1)*dm)))))))
  , ("patchEmbed",
     render (pretty BS (.patchEmbedF (ic := pi) (H := pH) (W := pW) (P := pp) (N := tk) (D := dm)
                          "%W" "%b" "%cls" "%pos" (fun _ _ _ _ => 0 : Kernel4 dm pi pp pp)
                          (fun _ => 0 : Vec dm) (fun _ => 0 : Vec dm)
                          (fun _ _ => 0 : Mat (tk+1) dm)
                          (.operand "%x" (fun _ => 0 : Vec (pi*pH*pW))))),
     render (pretty BS (.batchOp (N := BS)
                          (.patchEmbed (ic := pi) (H := pH) (W := pW) (P := pp) (N := tk) (D := dm)
                            "%W" "%b" "%cls" "%pos" (fun _ _ _ _ => 0 : Kernel4 dm pi pp pp)
                            (fun _ => 0 : Vec dm) (fun _ => 0 : Vec dm)
                            (fun _ _ => 0 : Mat (tk+1) dm))
                          (.operand "%x" (fun _ => 0 : Vec (BS*(pi*pH*pW)))))))
  ] ++
  -- ── the stem conv and the four batch-contracting PARAMETER gradients ──────────────────────────
  --    ⚠⚠ The four gradients ALIAS their per-example peer's `Raw` (their emitted MLIR already
  --    contracts the batch axis — `layerScaleChGammaGrad` reduces `dimensions = [0, 2, 3]`), so
  --    these four rows are true BY CONSTRUCTION and cannot fail while that holds. They are here
  --    anyway, and the reason is worth stating: if anyone later gives one of them its own tag —
  --    the obvious "tidy-up" — the byte-identity stops being structural and starts being a claim,
  --    and this is the row that would then carry it. A gate that is currently trivial but becomes
  --    load-bearing on a foreseeable edit is worth its four lines.
  (let zs4  : Vec (ic*(2*(2*ch))*(2*(2*ch))) := fun _ => 0
   let zs4b : Vec (BS*(ic*(2*(2*ch))*(2*(2*ch)))) := fun _ => 0
   let zoc  : Vec oc := fun _ => 0
   let zk4  : Kernel4 oc ic kk kk := fun _ _ _ _ => 0
   let zov  : Vec (oc*ch*ch) := fun _ => 0
   let zovb : Vec (BS*(oc*ch*ch)) := fun _ => 0
   let zq0  : Vec (pc*ph*ph) := fun _ => 0
   let zq0b : Vec (BS*(pc*ph*ph)) := fun _ => 0
   let zr   : Vec (rows*c) := fun _ => 0
   let zrb  : Vec (BS*(rows*c)) := fun _ => 0
   let zvc  : Vec c := fun _ => 0
   let zvcb : Vec (BS*c) := fun _ => 0
   let zva  : Vec a := fun _ => 0
   let zvab : Vec (BS*a) := fun _ => 0
   let zWo  : Mat a c := fun _ _ => 0
   [ ("convStride4",
      render (pretty BS (.flatConvStride4F (ic := ic) (oc := oc) (h := ch) (w := ch)
                           (kH := kk) (kW := kk) "%W" "%b" zk4 zoc (.operand "%x" zs4))),
      render (pretty BS (.batchOp (N := BS)
                           (.convStride4 (ic := ic) (oc := oc) (h := ch) (w := ch)
                              (kH := kk) (kW := kk) "%W" "%b" zk4 zoc)
                           (.operand "%x" zs4b))))
   , ("convStride4WeightGrad",
      render (pretty BS (.convStride4WeightGrad (ic := ic) (oc := oc) (h := ch) (w := ch)
                           (kH := kk) (kW := kk) "%x" zoc zs4 zk4 (.operand "%dy" zov))),
      render (pretty BS (.convStride4WeightGradB (N := BS) (ic := ic) (oc := oc) (h := ch) (w := ch)
                           (kH := kk) (kW := kk) "%x" zoc zs4b zk4 (.operand "%dy" zovb))))
   , ("layerScaleChGammaGrad",
      render (pretty BS (.layerScaleChGammaGrad (c := pc) (h := ph) (w := ph) "%x" zq0
                           (.operand "%dy" zq0))),
      render (pretty BS (.layerScaleChGammaGradB (N := BS) (c := pc) (h := ph) (w := ph) "%x" zq0b
                           (.operand "%dy" zq0b))))
   , ("veclnGammaGrad",
      render (pretty BS (.veclnGammaGrad (N := rows) (D := c) "%x" "1.0e-5" 0 zr
                           (.operand "%dy" zr))),
      render (pretty BS (.veclnGammaGradB (N := BS) (R := rows) (D := c) "%x" "1.0e-5" 0 zrb
                           (.operand "%dy" zrb))))
   , ("rowDenseBiasGrad",
      render (pretty BS (.rowDenseBiasGrad (N := rows) (c := c) (.operand "%dy" zr))),
      render (pretty BS (.rowDenseBiasGradB (N := BS) (R := rows) (c := c)
                           (.operand "%dy" zrb))))
   -- ── the classifier head: `dotOut` (input-VJP) and the two dense param grads.
   --    ⚠ `dotOut` is `softmaxDiv`'s situation one op over — its `dot_general` was ALREADY
   --    per-example (contracting dim 1 of tensor<B,n>), so only the `den` was wrong at the batched
   --    index. ⚠ `weightGradB`/`biasGradB` exist even though `denseWeightGradB`/`denseBiasGradB`
   --    denote the same thing: those carry different Raw TAGS, so reusing them would change the
   --    emitted text. Two ops can denote one function and still be two renders.
   , ("dotOut",
      render (pretty BS (.dotOut (m := a) (n := c) "%W" (zWo : Mat a c) (.operand "%dy" zvc))),
      render (pretty BS (.batchOp (N := BS) (.dotOut (m := a) (n := c) "%W" (zWo : Mat a c))
                           (.operand "%dy" zvcb))))
   , ("weightGrad",
      render (pretty BS (.weightGrad (m := a) (n := c) "%x" (zva : Vec a) (.operand "%dy" zvc))),
      render (pretty BS (.weightGradB (N := BS) (m := a) (n := c) "%x" (zvab)
                           (.operand "%dy" zvcb))))
   , ("biasGrad",
      render (pretty BS (.biasGrad (n := c) (.operand "%dy" zvc))),
      render (pretty BS (.biasGradB (N := BS) (n := c) (.operand "%dy" zvcb))))
   ]) ++
  -- ── step 3: max-pool + the conv bias param grads ──
  (let zp   : Vec (pc*(2*ph)*(2*ph)) := fun _ => 0
   let zpb  : Vec (BS*(pc*(2*ph)*(2*ph))) := fun _ => 0
   let zq   : Vec (pc*ph*ph) := fun _ => 0
   let zqb  : Vec (BS*(pc*ph*ph)) := fun _ => 0
   let zK   : Kernel4 oc ic kk kk := fun _ _ _ _ => 0
   let zT   : Tensor3 ic ch ch := fun _ _ _ => 0
   let zXb  : Vec (BS*(ic*ch*ch)) := fun _ => 0
   let zS   : Vec (ic*(2*ch)*(2*ch)) := fun _ => 0
   let zSb  : Vec (BS*(ic*(2*ch)*(2*ch))) := fun _ => 0
   let zB   : Vec oc := fun _ => 0
   let zdy  : Vec (oc*ch*ch) := fun _ => 0
   let zdyb : Vec (BS*(oc*ch*ch)) := fun _ => 0
   [ ("maxPool",
      render (pretty BS (.maxPoolF (c := pc) (h := ph) (w := ph) (.operand "%x" zp))),
      render (pretty BS (.batchOp (N := BS) (.maxPool (c := pc) (h := ph) (w := ph))
                           (.operand "%x" zpb))))
   , ("maxPoolBack",
      render (pretty BS (.maxPoolBack (c := pc) (h := ph) (w := ph) "%s" zp (.operand "%x" zq))),
      render (pretty BS (.maxPoolBackB (c := pc) (h := ph) (w := ph) "%s" zpb
                           (.operand "%x" zqb))))
   , ("convBiasSgd",
      render (pretty BS (.convBiasSgd (h := ch) (w := ch) "%b" "0.05" zK zT zB 0
                           (.operand "%x" zdy))),
      render (pretty BS (.convBiasSgdB (h := ch) (w := ch) "%b" "0.05" zK zXb zB 0
                           (.operand "%x" zdyb))))
   , ("convStridedBiasSgd",
      render (pretty BS (.convStridedBiasSgd (h := ch) (w := ch) "%b" "0.05" zK zS zB 0
                           (.operand "%x" zdy))),
      render (pretty BS (.convStridedBiasSgdB (h := ch) (w := ch) "%b" "0.05" zK zSb zB 0
                           (.operand "%x" zdyb)))) ])

/-- **The un-fused batched gradients** (`*GradB`) are their `*SgdB` peers with the SGD tail
    (const lr / multiply / subtract) cut off, so each render must be a byte-PREFIX of the fused
    one. Prefix rather than equality is the honest statement: the shared fresh-name counter means
    the gradient's lines carry the *same* SSA names in the fused render, and the tail is pure
    suffix. Denotationally this is the `*SgdB_eq_grad` set (`den (xSgdB …) = θ − lr·den (xGradB …)`,
    all `rfl`); this is its emit-side twin. -/
private def gradPrefixCases : List (String × String × String) :=
  let zK   : Kernel4 oc ic kk kk := fun _ _ _ _ => 0
  let zXb  : Vec (BS*(ic*ch*ch)) := fun _ => 0
  let zSb  : Vec (BS*(ic*(2*ch)*(2*ch))) := fun _ => 0
  let zB   : Vec oc := fun _ => 0
  let zdyb : Vec (BS*(oc*ch*ch)) := fun _ => 0
  let zbn  : Vec (BS*(oc*(ch*ch))) := fun _ => 0
  let zG   : Vec oc := fun _ => 0
  let zda  : Vec (BS*a) := fun _ => 0
  let zdc  : Vec (BS*c) := fun _ => 0
  let zWd  : Mat a c := fun _ _ => 0
  [ ("convWeightGradB",
     render (pretty BS (.convWeightGradB "%a" zB zXb zK (.operand "%x" zdyb))),
     render (pretty BS (.convWeightSgdB "%a" "%W" "0.05" zB zXb zK 0 (.operand "%x" zdyb))))
  , ("convStridedWeightGradB",
     render (pretty BS (.convStridedWeightGradB "%a" zB zSb zK (.operand "%x" zdyb))),
     render (pretty BS (.convStridedWeightSgdB "%a" "%W" "0.05" zB zSb zK 0 (.operand "%x" zdyb))))
  , ("convBiasGradB",
     render (pretty BS (.convBiasGradB (h := ch) (w := ch) zK zXb zB (.operand "%x" zdyb))),
     render (pretty BS (.convBiasSgdB (h := ch) (w := ch) "%b" "0.05" zK zXb zB 0
                          (.operand "%x" zdyb))))
  , ("convStridedBiasGradB",
     render (pretty BS (.convStridedBiasGradB (h := ch) (w := ch) zK zSb zB (.operand "%x" zdyb))),
     render (pretty BS (.convStridedBiasSgdB (h := ch) (w := ch) "%b" "0.05" zK zSb zB 0
                          (.operand "%x" zdyb))))
  , ("bnGammaGradB",
     render (pretty BS (.bnGammaGradB (oc := oc) (h := ch) (w := ch) "%v" "1.0e-5" 0 zbn
                          (.operand "%x" zbn))),
     render (pretty BS (.bnGammaSgdB (oc := oc) (h := ch) (w := ch) "%g" "%v" "1.0e-5" "0.05" 0 zG
                          zbn 0 (.operand "%x" zbn))))
  , ("bnBetaGradB",
     render (pretty BS (.bnBetaGradB (N := BS) (oc := oc) (h := ch) (w := ch) (.operand "%x" zbn))),
     render (pretty BS (.bnBetaSgdB (oc := oc) (h := ch) (w := ch) "%b" "0.05" zG 0
                          (.operand "%x" zbn))))
  , ("denseWeightGradB",
     render (pretty BS (.denseWeightGradB (c := c) "%a" zda (.operand "%x" zdc))),
     render (pretty BS (.denseWeightSgdB "%a" "%W" "0.05" zda zWd 0 (.operand "%x" zdc))))
  , ("denseBiasGradB",
     render (pretty BS (.denseBiasGradB (N := BS) (.operand "%x" zdc))),
     render (pretty BS (.denseBiasSgdB "%b" "0.05" (fun _ => 0 : Vec c) 0 (.operand "%x" zdc))))
  -- ── the TRANSFORMER family (§2a-quinquies follow-on: the ViT AdamW render needs these) ──
  -- Small stand-in shapes: `tk` tokens, `dm` model dim, `pp` patch, on a `pi × pH × pW` image.
  -- The property under test is textual, so the numbers only have to be consistent.
  , ("rowDenseWeightGrad",
     render (pretty BS (.rowDenseWeightGrad (N := tk) (a := a) (c := c) "%h"
                          (fun _ => 0 : Vec (tk*a)) (.operand "%x" (fun _ => 0 : Vec (tk*c))))),
     render (pretty BS (.rowDenseWeightSgd (N := tk) (a := a) (c := c) "%h" "%W" "0.05"
                          (fun _ => 0 : Vec (tk*a)) (fun _ _ => 0 : Mat a c) 0
                          (.operand "%x" (fun _ => 0 : Vec (tk*c))))))
  , ("rowDenseBiasGrad",
     render (pretty BS (.rowDenseBiasGrad (N := tk) (c := c) (.operand "%x" (fun _ => 0 : Vec (tk*c))))),
     render (pretty BS (.rowDenseBiasSgd (N := tk) (c := c) "%b" "0.05" (fun _ => 0 : Vec c) 0
                          (.operand "%x" (fun _ => 0 : Vec (tk*c))))))
  , ("veclnGammaGrad",
     render (pretty BS (.veclnGammaGrad (N := tk) (D := dm) "%h" "1.0e-5" 0
                          (fun _ => 0 : Vec (tk*dm)) (.operand "%x" (fun _ => 0 : Vec (tk*dm))))),
     render (pretty BS (.veclnGammaSgd (N := tk) (D := dm) "%g" "%h" "1.0e-5" "0.05" 0
                          (fun _ => 0 : Vec (tk*dm)) (fun _ => 0 : Vec dm) 0
                          (.operand "%x" (fun _ => 0 : Vec (tk*dm))))))
  , ("patchEmbedBiasGrad",
     render (pretty BS (.patchEmbedBiasGrad (N := tk) (c := c)
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*c))))),
     render (pretty BS (.patchEmbedBiasSgd (N := tk) (c := c) "%b" "0.05" (fun _ => 0 : Vec c) 0
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*c))))))
  , ("posEmbedGrad",
     render (pretty BS (.posEmbedGrad (N := tk) (D := dm)
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))),
     render (pretty BS (.posEmbedSgd (N := tk) (D := dm) "%p" "0.05"
                          (fun _ _ => 0 : Mat (tk+1) dm) 0
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))))
  -- the depthwise weight grads (EfficientNet's last blockers; mnv2/convnext reuse the shape)
  , ("depthwiseWeightGradB",
     render (pretty BS (.depthwiseWeightGradB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk) "%a" zB
                          (fun _ => 0 : Vec (BS*(oc*ch*ch)))
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))),
     render (pretty BS (.depthwiseWeightSgdB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk) "%a" "%W" "0.05" zB
                          (fun _ => 0 : Vec (BS*(oc*ch*ch)))
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk) 0
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))))
  , ("depthwiseStridedWeightGradB",
     render (pretty BS (.depthwiseStridedWeightGradB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk) "%a" zB
                          (fun _ => 0 : Vec (BS*(oc*(2*ch)*(2*ch))))
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))),
     render (pretty BS (.depthwiseStridedWeightSgdB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk) "%a" "%W" "0.05" zB
                          (fun _ => 0 : Vec (BS*(oc*(2*ch)*(2*ch))))
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk) 0
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))))
  , ("patchEmbedWeightGrad",
     render (pretty BS (.patchEmbedWeightGrad (ic := pi) (H := pH) (W := pW) (P := pp)
                          (N := tk) (D := dm) "%img" (fun _ => 0 : Vec (pi*pH*pW))
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))),
     render (pretty BS (.patchEmbedWeightSgd (ic := pi) (H := pH) (W := pW) (P := pp)
                          (N := tk) (D := dm) "%W" "%img" "0.05" (fun _ => 0 : Vec (pi*pH*pW))
                          (fun _ _ _ _ => 0 : Kernel4 dm pi pp pp) 0
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))))
  -- ── the ConvNeXt five (§2f). These ride the generic `.batched` tag, where an unmatched name
  --    falls through to `// MALFORMED` SILENTLY — these five cases are what catches that. ──
  , ("depthwiseWeightGrad (per-example)",
     render (pretty BS (.depthwiseWeightGrad (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          "%a" zB (fun _ _ _ => 0 : Tensor3 oc ch ch)
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.depthwiseWeightSgd (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          "%a" "%W" "0.05" zB (fun _ _ _ => 0 : Tensor3 oc ch ch)
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk) 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("depthwiseBiasGrad",
     render (pretty BS (.depthwiseBiasGrad (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (fun _ _ _ => 0 : Tensor3 oc ch ch) zB
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.depthwiseBiasSgd (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          "%b" "0.05" (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (fun _ _ _ => 0 : Tensor3 oc ch ch) zB 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  -- ── the MobileNetV2 two (§2f): the BATCHED depthwise bias grads. There is no fused
  --    `depthwise{,Strided}BiasSgdB` peer to check against — `MobileNetV2RenderB` is AdamW-only,
  --    like `ResNet34RenderB` — so both are checked against the PER-EXAMPLE fused op, which is
  --    sound precisely because the bias grad's emit is batch-, stride- and kernel-independent
  --    (`Σ_{batch,spatial} dy`) and all three constructors alias ONE emitter. If that aliasing ever
  --    stops holding, these two cases go red. ──
  , ("depthwiseBiasGradB",
     render (pretty BS (.depthwiseBiasGradB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk)
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (fun _ => 0 : Vec (BS*(oc*ch*ch))) zB
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))),
     render (pretty BS (.depthwiseBiasSgd (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          "%b" "0.05" (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (fun _ _ _ => 0 : Tensor3 oc ch ch) zB 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("depthwiseStridedBiasGradB",
     render (pretty BS (.depthwiseStridedBiasGradB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk)
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (fun _ => 0 : Vec (BS*(oc*(2*ch)*(2*ch)))) zB
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))),
     render (pretty BS (.depthwiseStridedBiasSgd (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          "%b" "0.05" (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (fun _ => 0 : Vec (oc*(2*ch)*(2*ch))) zB 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("lnGammaGrad",
     render (pretty BS (.lnGammaGrad (n := oc*ch*ch) "%a" "1.0e-6" 0
                          (fun _ => 0 : Vec (oc*ch*ch))
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.lnGammaSgd (n := oc*ch*ch) "%g" "%a" "1.0e-6" "0.05" 0
                          (fun _ => 0 : Vec (oc*ch*ch)) (fun _ => 0 : Vec 1) 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("lnBetaGrad",
     render (pretty BS (.lnBetaGrad (n := oc*ch*ch)
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.lnBetaSgd (n := oc*ch*ch) "%bt" "0.05" (fun _ => 0 : Vec 1) 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("layerScaleChGammaGrad",
     render (pretty BS (.layerScaleChGammaGrad (c := oc) (h := ch) (w := ch) "%a"
                          (fun _ => 0 : Vec (oc*ch*ch))
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.layerScaleChGammaSgd (c := oc) (h := ch) (w := ch) "%g" "%a" "0.05"
                          (fun _ => 0 : Vec (oc*ch*ch)) (fun _ => 0 : Vec oc) 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch)))))) ]

/-- **Stochastic depth's emit guard** (`planning/stochastic_depth.md`). `dropPathB` has **no
    per-example peer** — the mask is per-example by construction, so there is nothing to tie it
    against in either section above. What it needs pinning instead is the one structural property
    that could plausibly be wrong and that no numeric gate can see:

    > **the mask is per-SAMPLE, not per-element.**

    A `tensor<B×n>` mask input in place of `tensor<B>` + `broadcast_in_dim dims = [0]` typechecks,
    compiles, runs, descends, and is **per-element dropout** — a different regulariser entirely. The
    reference's mask is `(B, 1, …, 1)` (`jax/Jax/Codegen.lean:1037`), and `dims = [0]` is the whole
    of that claim in the emitted text.

    Also checked: the `1/keep` literal is BAKED (so a wrong ramp is visible in the text rather than
    hidden in an input), and — the emit-side twin of `dropPathB_back_faithful` — the backward emits
    **byte-identically** to the forward at the same mask, which is what says there is genuinely one
    emitter here rather than two that must be kept in step. -/
private def dropPathEmit : String :=
  render (pretty BS (.dropPathB (N := BS) (n := n) "%dp0"
                       (fun _ => 0 : Vec BS) (.operand "%x" (fun _ => 0 : Vec (BS*n)))))

/-- The same op on a COTANGENT — same mask, same `invKeep`. `dropPath_vjp_is_self` says this IS the
    certified VJP; this says the two render the same bytes. -/
private def dropPathBackEmit : String :=
  render (pretty BS (.dropPathB (N := BS) (n := n) "%dp0"
                       (fun _ => 0 : Vec BS) (.operand "%x" (fun _ => 0 : Vec (BS*n)))))

/-- **Classifier dropout's emit guard** (`recipe_gaps.md` gap C) — and it is `dropPathB`'s read
    BACKWARDS, which is the point of putting the two side by side.

    `dropoutB` has no per-example peer either, and the one structural property that could plausibly
    be wrong is the mirror of stochastic depth's:

    > **the mask is per-ELEMENT, not per-sample.**

    The reference draws `bernoulli(key, keep, x.shape)` (`jax/Jax/Codegen.lean:1971`) — the full
    `(B, 1280)` shape, not `(B, 1, …, 1)`. So a `tensor<B>` input plus `broadcast_in_dim dims = [0]`
    here typechecks, compiles, runs, descends, and is **stochastic depth on the classifier** — a
    different regulariser, exactly as the reverse substitution is on a residual branch.

    ⚠ Until this op existed the hazard was one-directional and every comment in the kit wrote it
    that way. It is now symmetric, and these two blocks are the pair that says so: `dropPathB`'s
    asserts the broadcast is PRESENT, this one asserts it is ABSENT. Neither alone distinguishes
    the two ops. `Proofs.dropout_of_dropScale` and `Proofs.dropPath_scales_uniformly` are the
    denotation-side peers — this is a claim about bytes, those are claims about functions, and a
    render that swapped the ops would be wrong on both counts. -/
private def dropoutEmit : String :=
  render (pretty BS (.dropoutB (N := BS) (n := n) "%do"
                       (fun _ => 0 : Vec (BS*n)) (.operand "%x" (fun _ => 0 : Vec (BS*n)))))

/-- The same op on a COTANGENT. `dropout_vjp_is_self` says this IS the certified VJP; this says the
    two render the same bytes, i.e. that there is genuinely one emitter. -/
private def dropoutBackEmit : String :=
  render (pretty BS (.dropoutB (N := BS) (n := n) "%do"
                       (fun _ => 0 : Vec (BS*n)) (.operand "%x" (fun _ => 0 : Vec (BS*n)))))

/-- Fail loudly. NOT `IO.Process.exit 1`: under `#eval` the elaborator buffers the eval's output
    and prints it only after the eval returns, so `exit` kills the process with **every diagnostic
    discarded** — you get a bare non-zero status and no idea which form broke. (Verified against a
    deliberately broken `relu` emit case.) `throw` surfaces the message as an elaboration error and
    still makes `lake env lean` exit non-zero. Several older `tests/*.lean` use the `exit` form and
    have the same blind-failure problem. -/
private def die (msg : String) : IO α := throw (IO.userError msg)

def main : IO Unit := do
  let mut bad := 0
  for (name, perExample, batched) in cases do
    -- A form whose emit case is missing falls through to a COMMENT, not an error, so an empty
    -- or marker-bearing render must fail here rather than tie trivially against another one.
    if perExample.isEmpty || (perExample.splitOn "MALFORMED").length != 1 then
      die s!"DEGENERATE: {name} per-example render is empty or fell through"
    if perExample == batched then
      IO.println s!"  ✓ {name}"
    else
      bad := bad + 1
      IO.eprintln s!"  ✖ {name}\n    per-example:\n{perExample}    batched:\n{batched}"
  if bad != 0 then
    die s!"MISMATCH: {bad} batched form(s) do not emit their per-example peer's text"
  IO.println s!"✓ all {cases.length} batched forms emit their per-example peer's text byte-for-byte"
  IO.println "── un-fused gradients ⊂ their fused *SgdB peers ──"
  for (name, grad, sgd) in gradPrefixCases do
    if grad.isEmpty || grad.length >= sgd.length then
      die s!"DEGENERATE: {name} gradient render is empty or not shorter than the fused op"
    if sgd.startsWith grad then
      IO.println s!"  ✓ {name}  ({grad.length} of {sgd.length} bytes, tail = the SGD update)"
    else
      bad := bad + 1
      IO.eprintln s!"  ✖ {name} is NOT a prefix of its fused peer\n    grad:\n{grad}    sgd:\n{sgd}"
  if bad != 0 then
    die s!"MISMATCH: {bad} gradient render(s) are not a prefix of their fused peer"
  IO.println s!"✓ all {gradPrefixCases.length} un-fused gradients are byte-prefixes of their *SgdB peers"
  -- ── stochastic depth: the mask is per-SAMPLE (§ dropPathEmit above) ──
  IO.println "── stochastic depth: dropPathB ──"
  let e := dropPathEmit
  if e.isEmpty || (e.splitOn "MALFORMED").length != 1 then
    die "DEGENERATE: dropPathB render is empty or fell through to // MALFORMED"
  -- ⭐ the load-bearing one: a `tensor<32xf32>` mask broadcast over dim 0. Per-ELEMENT dropout
  --    would read `tensor<32x12xf32>` here and be a different regulariser that still trains.
  if (e.splitOn "stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x12xf32>").length != 2 then
    die s!"dropPathB's mask is NOT a per-SAMPLE tensor<32xf32> broadcast over dim 0:\n{e}"
  -- ⚠ There must be NO baked keep constant: the driver folds `1/keep_i` into the supplied scale,
  -- which is what makes the ones-scale forward the exact identity (`den_dropPathB_ones`) and lets
  -- the op be emitted in the forward at all. A baked constant here would silently rescale eval.
  if (e.splitOn "stablehlo.constant").length != 1 then
    die s!"dropPathB emits a baked constant — eval at a ones scale would not be the identity:\n{e}"
  if e != dropPathBackEmit then
    die s!"dropPathB's backward does not emit its forward's text — a diagonal map is its own \
transpose, so these must be one emitter:\n forward:\n{e} backward:\n{dropPathBackEmit}"
  IO.println "  ✓ dropPathB: per-SAMPLE scale (dims = [0] over tensor<32xf32>), no baked keep"
  IO.println "  ✓ dropPathB: backward emits the forward's text byte-for-byte (VJP is itself)"
  -- ── classifier dropout: the mask is per-ELEMENT (§ dropoutEmit above) ──
  IO.println "── classifier dropout: dropoutB ──"
  let d := dropoutEmit
  if d.isEmpty || (d.splitOn "MALFORMED").length != 1 then
    die "DEGENERATE: dropoutB render is empty or fell through to // MALFORMED"
  -- ⭐ the load-bearing one, and it is the EXACT MIRROR of dropPathB's above: the mask carries the
  --    value's own shape and is multiplied in directly. A `broadcast_in_dim` here would be
  --    stochastic depth on the classifier — it trains, and no numeric gate sees it.
  if (d.splitOn "stablehlo.multiply %do, %x : tensor<32x12xf32>").length != 2 then
    die s!"dropoutB does not multiply a per-ELEMENT tensor<32x12xf32> mask in directly:\n{d}"
  if (d.splitOn "broadcast_in_dim").length != 1 then
    die s!"dropoutB BROADCASTS its mask — that is stochastic depth, not dropout:\n{d}"
  -- ⚠ Same no-baked-constant requirement as dropPathB, for the same reason: the driver folds
  -- `1/keep` into the mask, which is what makes the ones-mask forward the exact identity.
  if (d.splitOn "stablehlo.constant").length != 1 then
    die s!"dropoutB emits a baked constant — eval at a ones mask would not be the identity:\n{d}"
  if d != dropoutBackEmit then
    die s!"dropoutB's backward does not emit its forward's text — a diagonal map is its own \
transpose, so these must be one emitter:\n forward:\n{d} backward:\n{dropoutBackEmit}"
  -- ⭐⭐ THE PAIR, stated as one check: the two regularisers must not render the same bytes. Each
  --    block above passes on its own op; only this says they are DISTINGUISHABLE, which is the
  --    property that makes either emit test worth running.
  if d == e then
    die s!"dropoutB and dropPathB emit IDENTICAL text — one of them is the wrong regulariser:\n{d}"
  IO.println "  ✓ dropoutB: per-ELEMENT mask (tensor<32x12xf32> multiplied in, NO broadcast)"
  IO.println "  ✓ dropoutB: backward emits the forward's text byte-for-byte (VJP is itself)"
  IO.println "  ✓ dropoutB ≠ dropPathB: the two regularisers render distinguishable text"

#eval main

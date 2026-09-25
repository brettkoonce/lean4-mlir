import LeanMlir.Proofs.Codegen.StableHLO

/-! # StableHLOPretty — the printer: `SHlo` terms → StableHLO text

The syntactic half of [`StableHLO`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Codegen/StableHLO.lean):
`pretty B g` renders a term of the AST whose denotation `den` the proofs are about. SSA names are
annotations `den` ignores, and `skel` erases the ℝ values, so the rendered text depends on the
graph's shape and names only.

| part | what |
|---|---|
| `ty` / `tyI1` / `tyBf16` / `tyF8`, `fresh`, the `ShapeTbl` helpers | type strings, SSA names, the name ↦ shape table |
| `Raw`, `skel` | the value-erased skeleton of an `SHlo` term |
| `Tok`, `toToks` | the post-order token stream |
| `emitContract`, `emitTok`, `serializeToks` | StableHLO text per token (f32 / bf16 / fp8) |
| `pretty`, `prettyAdamW`, `prettyAllReduceMean`, `renderModule` | whole graphs and modules |
| `fmt6` / `fmt12`, `OptKind`, `RmsHyper`, bias-slot helpers | the literal and constant blocks the renderers share |
| `*ModuleV`, `linTrainStepFaithfulV` | the chapter 1–3 renderers |

A module that states `den` facts imports `StableHLO` alone; one that renders text imports this.
**Trusted residue:** the text's lexical conformance to the StableHLO spec is checked by execution
(`iree-compile` / PJRT) and by the `StableHLOParse` round-trip, not proved. -/

open Finset BigOperators

namespace Proofs
namespace StableHLO

/-- Tensor-type string `tensor<d₀x…xf32>`. -/
def ty (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["f32"]) ++ ">"

/-- Boolean (i1) tensor-type string, for `compare`/`select` masks. -/
def tyI1 (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["i1"]) ++ ">"

/-- bf16 tensor-type string, for the `convertF` round node (planning/archive/bf16_renderer.md).
    Only the round trip uses it today; when a bf16-operand `dot_general` lands (rung 2+)
    this is the type its operands carry. -/
def tyBf16 (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["bf16"]) ++ ">"

/-- fp8 peer of `tyBf16`. **E4M3 only** — `planning/archive/cifar_lowprec_stability.md` §2.3 measured
    that `f8E5M2` compiles, lowers to a plain `__cublas$lt$matmul`, and leaves ZERO `f8e5m2`
    values in the optimized HLO: the type is silently widened away. Only E4M3 reaches the fp8
    units on sm_89, so there is deliberately no E5M2 spelling here. -/
def tyF8 (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["f8E4M3FN"]) ++ ">"

/-- SSA name ↦ the `[c,h,w]` the value bound to that name really carries. See `liftPointwise`.

    ⚠⚠ **Keyed by NAME, not by flat width — and that is not a refinement, it is the whole
    correctness of the table.** A width table collides whenever two layers have the same element
    count, and on the real nets they do: ConvNeXt-T's stage-2 MLP is `1536·14·14 = 301056` and its
    stage-0 block is `96·56·56 = 301056`; stage 3's `3072·7·7` equals stage 1's `192·28·28`. First
    writer won, so 24 of ConvNeXt's pointwise blocks unflattened to a shape with the right element
    count and the wrong layout — which is not a wrong program (the bracket is still an inverse
    reshape pair) but is exactly the relayout the bracket exists to remove. Measured: 2.434 GB of
    transposes and 84.45 ms/step keyed by width, **0.122 GB and 68.28 ms** keyed by name.

    Newest entry first, and no dedup: `fresh` never reuses a name, so a lookup for a value the
    previous token produced hits the head of the list.

    ⚠⚠ The `Bool` is the value's LAYOUT: `true` means this is the map's **row view** `[h·w, c]`
    rather than the map `[c, h, w]`. It is not bookkeeping — it is what makes ConvNeXt's channel-LN
    transparent. That chain is `transpose → lnRow → rowScale → rowBias → transpose`, a layout ROUND
    TRIP whose two ends are the same `[c,h,w]` map; without the flag the closing transpose's result
    has no entry, the drop-path multiply that consumes it falls back to flat, and every pointwise op
    after it on the residual chain goes with it — 0.223 GB of relayout against 0.122 (measured,
    ConvNeXt-T bf16). And `liftPointwise` must NOT fire on a row view: `[h·w, c]` reshaped to
    `[B,c,h,w]` is a DIFFERENT permutation, not an inverse pair, so that one would be a wrong
    program rather than a slow one. -/
abbrev ShapeTbl := List (String × Nat × Nat × Nat × Bool)

/-- Emitter state: the fresh-name counter, plus the name ↦ `[c,h,w]` table.

    ⚠ The table lives in the STATE rather than in a `pretty` argument because a net renderer
    calls `pretty` once per graph FRAGMENT — a conv and the activation that consumes it land in
    different calls — and only the state is threaded across them. -/
abbrev EmitS := Nat × ShapeTbl

/-- Fresh SSA name `%v{k}`. -/
def fresh : StateM EmitS String := do
  let (k, tbl) ← get; set (k + 1, tbl); pure s!"%v{k}"

/-- **The 3×3/s2 pool's emitted forward text**, given already-freshened names.

    ⚠⚠ It is a shared helper rather than two copies for the reason `sWGradGeom` is (§2f-bis): the
    per-example `.maxPool3s2F` and the batched `BatchableOp.maxPool3s2` are two `emitTok` arms
    emitting **one** program, and a window or padding that drifted between them would be a pair of
    renders that agree on every structural check and compute different functions — which is the
    exact failure this whole op exists to fix. With one writer they cannot drift, and
    `TestBatchedEmitTie` then measures rather than assumes it.

    `window_dimensions = 3, window_strides = 2, padding = [[1,1],[1,1]]` on the spatial axes: He
    et al./torchvision `MaxPool2d(3, stride=2, padding=1)`, window `i` = input `[2i−1, 2i+1]`.
    ⚠ NOT XLA `'SAME'`, which pads `(0,1)` and slides the grid one input position — the two are
    different functions everywhere. -/
def maxPool3s2FwdText (B c h w : Nat) (r xn ninf p o : String) : String :=
  s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
  s!"    {ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
  s!"    {p} = \"stablehlo.reduce_window\"({xn}, {ninf}) (" ++ "{\n" ++
  "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
  "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
  "        stablehlo.return %pm : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, " ++
  "padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>}" ++
  s!" : ({ty [B,c,2*h,2*w]}, tensor<f32>) -> {ty [B,c,h,w]}\n" ++
  s!"    {o} = stablehlo.reshape {p} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n"

/-- **The 3×3/s2 pool's emitted backward text**, given already-freshened names. Shared by the
    per-example and batched arms, for `maxPool3s2FwdText`'s reason.

    ⭐ Only the window attributes differ from `maxPoolBack`'s emit — **nothing else** — because
    `select_and_scatter`'s scatter region already reduces with `add`, which is exactly the
    accumulation overlapping windows need. The emitter was general enough before the op existed.

    ⚠ `%sa`/`%sb`/`%sc`/`%sd` are hardcoded region block arguments and are therefore RESERVED SSA
    names (§4): a top-level value of the same name is a redefinition error that surfaces only at
    XLA compile time. -/
def maxPool3s2BackText (B c h w : Nat) (xN r xr dr z scn o : String) : String :=
  s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
  s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
  s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    {scn} = \"stablehlo.select_and_scatter\"({xr}, {dr}, {z}) (" ++ "{\n" ++
  "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
  "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
  "        stablehlo.return %sge : tensor<i1>\n" ++
  "    }, " ++ "{\n" ++
  "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
  "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
  "        stablehlo.return %ss : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, " ++
  "padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>}" ++
  s!" : ({ty [B,c,2*h,2*w]}, {ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
  s!"    {o} = stablehlo.reshape {scn} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n"

/-- **The stochastic-depth mask input name for ramp index `i`** — the `mName` a `dropPathB` carries,
    and the `tensor<Bxf32>` the signature declares for it.

    ⚠ It lives HERE, beside the emitter, rather than in one net's renderer, because the spelling is
    load-bearing in three places that must agree and only one of them is Lean: `dropPathP`'s emit
    reads it as an operand, every SD render's signature declares it, and
    **`scripts/misplace_drop_sites.py` matches `%dp\d+` textually** to build the placement control.
    A second definition would be the double-writer disease with a committed shell script as the
    third writer. (It started in `EfficientNetRender.lean` and moved when ConvNeXt needed it too;
    both renderers are in this namespace, so no call site changed and no artifact byte moved.) -/
def dpName (i : Nat) : String := s!"%dp{i}"

/-- **The classifier-dropout mask input name** — the `mName` a `dropoutB` carries, and the
    `tensor<B×n×f32>` the signature declares for it.

    ⚠⚠ **IT IS DELIBERATELY NOT `%dp{i}`-SHAPED, and that is not cosmetic.**
    `scripts/misplace_drop_sites.py` builds the stochastic-depth placement control by matching
    `%dp\d+` textually; a dropout input spelled `%dp9` would be swept into that rewrite, silently
    changing a control's meaning on a render it was never written for. Handoff §0.11 records the
    other half of this hazard on ViT — a control that quietly does nothing reads exactly like a
    control that ran — and the cheap defence is a name the SD tooling cannot match.
    `grep -c '%do' verified_mlir/*.mlir` is 0 across every committed artifact. -/
def doName : String := "%do"

-- ── Renderable skeleton + postorder tokenization (one form, shared with the
--    parser in StableHLOParse.lean) ──

/-- The renderable skeleton of an `SHlo` graph: opcodes + shapes + leaf SSA
    names, with `ℝ` operand values and the shape index erased — exactly what
    reaches the emitted text. -/
inductive Raw where
  | operand    (name : String) (n : Nat)  : Raw
  | dotIn      (w : String) (m n : Nat)    : Raw → Raw
  | dotInBf16  (w : String) (m n : Nat)    : Raw → Raw
  | dotOut     (w : String) (m n : Nat)    : Raw → Raw
  | addBcast   (b : String) (n : Nat)      : Raw → Raw
  | expe       (n : Nat)                   : Raw → Raw
  | softmaxDiv (n : Nat)                   : Raw → Raw
  | sub        (n : Nat)                   : Raw → Raw → Raw
  | weightSgd  (xName wName lrStr : String) (m n : Nat) : Raw → Raw
  | biasSgd    (bName lrStr : String) (n : Nat)         : Raw → Raw
  | convWeightSgd (xName wName lrStr : String) (ic oc h w kH kW : Nat) : Raw → Raw
  | convBiasSgd   (bName lrStr : String) (oc h w : Nat)               : Raw → Raw
  | bnGammaSgd    (gName vName epsStr lrStr : String) (oc h w : Nat)  : Raw → Raw
  | bnBetaSgd     (bName lrStr : String) (oc h w : Nat)               : Raw → Raw
  | layerScaleChGammaSgd (gName xName lrStr : String) (c h w : Nat)   : Raw → Raw
  | lnGammaSgd    (gName xName epsStr lrStr : String) (n : Nat)       : Raw → Raw
  | lnBetaSgd     (bName lrStr : String) (n : Nat)                    : Raw → Raw
  | veclnGammaSgd (gName xName epsStr lrStr : String) (N D : Nat)     : Raw → Raw
  | patchEmbedWeightSgd (wName xName lrStr : String) (ic H W P N D : Nat) : Raw → Raw
  | reluF      (n : Nat)                   : Raw → Raw
  | selectPos  (x : String) (n : Nat)      : Raw → Raw
  | relu6F     (n : Nat)                   : Raw → Raw
  | selectMid  (x : String) (n : Nat)      : Raw → Raw
  | convertF   (n : Nat)                   : Raw → Raw
  | flatConvF  (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | flatConvFBf16 (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | maxPoolF   (c h w : Nat)               : Raw → Raw
  | convBack   (w : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | maxPoolBack (x : String) (c h w : Nat) : Raw → Raw
  | bnF        (g b eps : String) (n : Nat) : Raw → Raw
  | bnBack     (g x eps : String) (n : Nat) : Raw → Raw
  | addV       (n : Nat)                   : Raw → Raw → Raw
  | gapF       (c h w : Nat)               : Raw → Raw
  | gapBack    (c h w : Nat)               : Raw → Raw
  | broadcastBack (c h w : Nat)            : Raw → Raw
  | flatConvStridedF (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | flatConvStridedXlaF (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedBack  (w : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedWeightSgd (xName wName lrStr : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | depthwiseWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Raw → Raw
  | convStridedXlaWeightSgd (xName wName lrStr : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedXlaWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Raw → Raw
  | flatConvStride4F (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | bnPerChannelF    (g b eps : String) (oc h w : Nat) : Raw → Raw
  | bnPerChannelBack (g x eps : String) (oc h w : Nat) : Raw → Raw
  | bnPerChannelEvalF (g b mu var eps : String) (oc h w : Nat) : Raw → Raw
  | weightGrad (x : String) (m n : Nat) : Raw → Raw
  | biasGrad (n : Nat) : Raw → Raw
  | convWeightGrad (x : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convBiasGrad (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedWeightGrad (x : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | bnGammaGrad (v eps : String) (oc h w' : Nat) : Raw → Raw
  | bnBetaGrad (oc h w' : Nat) : Raw → Raw
  | adamMNextF (m b1 ob1 : String) (ds : List Nat) : Raw → Raw
  | adamVNextF (v b2 ob2 : String) (ds : List Nat) : Raw → Raw
  | adamWParamF (θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd : String) (ds : List Nat) : Raw → Raw
  | sgdParamF (θ lr : String) (ds : List Nat) : Raw → Raw
  | momVNextF (v mu : String) (ds : List Nat) : Raw → Raw
  | momParamF (θ v mu lr : String) (ds : List Nat) : Raw → Raw
  | rmsBufNextF (sq buf rho orho mu eps : String) (ds : List Nat) : Raw → Raw
  -- Global-norm grad clip. `gradClipFacF` keeps only its two literal strings (the ℝs are
  -- denotation-only, as everywhere in `Raw`); `clipScaleF`/`addScalarF` are BINARY.
  | gradSumSqAccF (ds : List Nat)                            : Raw → Raw → Raw
  | clipScaleF    (clipStr epsStr : String) (ds : List Nat)  : Raw → Raw → Raw
  | lambDirF (θ m v b1 ob1 b2 ob2 bc1 bc2 eps wd : String) (ds : List Nat) : Raw → Raw
  | lambScaleF    (ds : List Nat)                            : Raw → Raw → Raw
  | depthwiseF    (w b : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseBack (w : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedF    (w b : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedXlaF (w b : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedBack (w : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedXlaBack (w : String) (c h w' kH kW : Nat) : Raw → Raw
  | swishF     (n : Nat)                   : Raw → Raw
  | swishBack  (x : String) (n : Nat)      : Raw → Raw
  | sigmoidF   (n : Nat)                   : Raw → Raw
  | sigmoidBack (x : String) (n : Nat)     : Raw → Raw
  | geluF      (n : Nat)                   : Raw → Raw
  | geluBack   (x : String) (n : Nat)      : Raw → Raw
  | layerScaleF (γ : String) (n : Nat)     : Raw → Raw
  | layerScaleChF (γ : String) (c h w : Nat) : Raw → Raw
  | softmaxRowF    (m n : Nat)             : Raw → Raw
  | softmaxRowBack (x : String) (m n : Nat) : Raw → Raw
  | matmulF    (m k n : Nat)               : Raw → Raw → Raw
  | transposeF (m n : Nat)                 : Raw → Raw
  | scaleF     (s : String) (n : Nat)      : Raw → Raw
  | lnRowF     (g b eps : String) (m n : Nat) : Raw → Raw
  | lnRowBack  (g x eps : String) (m n : Nat) : Raw → Raw
  | denseRowF  (w b : String) (N a c : Nat) : Raw → Raw
  | denseRowBack (w : String) (N a c : Nat) : Raw → Raw
  | patchEmbedF (w b cls pos : String) (ic H W P N D : Nat) : Raw → Raw
  | clsSliceF  (N D : Nat)                 : Raw → Raw
  | clsPadF    (N D : Nat)                 : Raw → Raw
  | headSliceF (N heads d hIdx : Nat)      : Raw → Raw
  | headPadF   (N heads d hIdx : Nat)      : Raw → Raw
  | rowScaleF  (g : String) (m n : Nat)    : Raw → Raw
  | rowBiasF   (b : String) (m n : Nat)    : Raw → Raw
  -- EfficientNet batched ops (`batchOp`/`bnBatchF`/the batched backward ops): the
  -- renderable skeleton keeps a tag discriminating the op, the SSA names the emit
  -- references (weight/bias/BN-input/γ/ε/SE-input names), and shape info. The tag
  -- is the BatchableOp variant ("conv"/"depthwise"/"seBlock"/…) for forward ops or
  -- the backward op name; this is what lets `emitTok` reconstruct real StableHLO.
  | batched    (tag : String) (names : List String) (info : List Nat) : Raw → Raw
  -- The BINARY peer of `batched`, for the pointwise two-operand ops (`addV`/`sub`)
  -- at the batched index. Same reason as the unary descriptors: their
  -- descriptor-less tokens read the emit width off the SHlo index, so they cannot
  -- sit at `N·n`. `info` is `[N, n]`; the emit uses `n`, like every batched tag.
  | batched2   (tag : String) (names : List String) (info : List Nat) : Raw → Raw → Raw
  -- 4d piece 2: the cross-replica mean over `R` replicas, at the parameter shape `ds`, with the
  -- SSA tag `t` the emit names its lines by. Replica 0's skeleton is the operand.
  | allReduceMean (R : Nat) (t : String) (ds : List Nat) : Raw → Raw
deriving Repr, Inhabited

/-- The `(tag, names, info)` skeleton descriptor of a batched per-example op — the
    discriminator + the SSA names the emit references + the shape dims. Keeps the
    `batchOp` skel one line and isolates the 7-variant match into a pure function. -/
def batchOpDescr {a b : Nat} (N : Nat) : BatchableOp a b → (String × List String × List Nat)
  | .conv (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("conv", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .convStrided (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("convStrided", [wN, bN], [N, ic, oc, h, w, kH, kW])
  -- ⚠ DISTINCT tags, for the `convStridedXla` reason one case down and one more: the emitted
  -- TEXT differs (converts + a bf16 result type), so sharing a Raw with the f32 tag would make
  -- two different graphs indistinguishable after `skel`.
  | .convBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convBf16", [wN, bN], [N, ic, oc, h, w, kH, kW])
  -- DISTINCT tag for the same reason `convBf16` has one: the emitted TEXT differs (f8 converts
  -- and an f8 result type), so sharing a Raw would make two different graphs indistinguishable.
  | .convF8 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convF8", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .convStridedBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convStridedBf16", [wN, bN], [N, ic, oc, h, w, kH, kW])
  -- ⚠ A DISTINCT tag, deliberately. Aliasing this onto "convStrided" (the way the bias-grads
  -- legitimately alias, because their emitted text is stride-independent) would be wrong here:
  -- the emitted `pad` differs, so the two tags must not share a Raw.
  | .convStridedXla (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("convStridedXla", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .convStridedXlaBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convStridedXlaBf16", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .depthwise (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("depthwise", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseBf16 (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("depthwiseBf16", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseStrided (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("depthwiseStrided", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseStridedBf16 (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("depthwiseStridedBf16", [wN, bN], [N, c, h, w, kH, kW])
  -- Distinct tag, for the same reason `convStridedXla` is: the emitted `pad` differs.
  | .depthwiseStridedXla (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("depthwiseStridedXla", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseStridedXlaBf16 (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("depthwiseStridedXlaBf16", [wN, bN], [N, c, h, w, kH, kW])
  | .dense (c := c) wN bN _ _ => ("dense", [wN, bN], [N, a, c])
  | .gap (c := c) (h := h) (w := w) => ("gap", [], [N, c, h, w])
  | .seBlock (c := c) (h := h) (w := w) (r := r) w1 b1 w2 b2 _ _ _ _ =>
      ("seBlock", [w1, b1, w2, b2], [N, c, h, w, r])
  | .bnEval (oc := oc) (h := h) (w := w) gN bN muN varN es _ _ _ _ _ =>
      ("bnEval", [gN, bN, muN, varN, es], [N, oc, h, w])
  | .swish (n := n) => ("swish", [], [N, n])
  | .relu (n := n) => ("relu", [], [N, n])
  | .relu6 (n := n) => ("relu6", [], [N, n])
  | .maxPool (c := c) (h := h) (w := w) => ("maxPool", [], [N, c, h, w])
  -- ⚠ A DIFFERENT tag from `.maxPool`, deliberately. The two denote different functions at the
  -- same type, so sharing a tag would make the emitted text the only thing separating them — and
  -- the emitted text is what a reader checks last. `denOp`'s `.maxPool3s2` arm is the den-side
  -- half of the same pin.
  | .maxPool3s2 (c := c) (h := h) (w := w) => ("maxPool3s2", [], [N, c, h, w])
  | .softmaxRow (m := m) (n := n) => ("softmaxRow", [], [N, m, n])
  | .denseRowBack (rows := rows) (a := a) (c := c) wN _ => ("denseRowBackP", [wN], [N, rows, a, c])
  -- ⚠ A DISTINCT tag, for the reason every bf16 tag in this function is distinct: the emitted TEXT
  -- differs (two operand converts and bf16 operand types), so sharing a Raw with the f32 tag would
  -- make two different graphs indistinguishable after `skel`.
  | .denseRowBackBf16 (rows := rows) (a := a) (c := c) _ wN _ =>
      ("denseRowBackPBf16", [wN], [N, rows, a, c])
  -- ⚠ `epsStr` rides in `names` though it is a LITERAL, not an SSA name — `bnEval` set that
  -- precedent and `emitTok` splices both the same way. The alternative is a second string list.
  | .gelu (n := n) => ("gelu", [], [N, n])
  | .transpose (m := m) (n := n) => ("transposeP", [], [N, m, n])
  | .convStride4 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("convStride4P", [wN, bN], [N, ic, oc, h, w, kH, kW])
  -- ⚠ A DISTINCT tag, for the reason every other bf16 conv tag is distinct: the emitted TEXT
  -- differs (two operand converts, a bf16 result type, a convert back), so sharing a Raw with the
  -- f32 tag would make two different graphs indistinguishable after `skel`.
  | .convStride4Bf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convStride4PBf16", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .layerScaleCh (c := c) (h := h) (w := w) gN _ => ("layerScaleChP", [gN], [N, c, h, w])
  | .dotOut (m := m) (n := n) wN _ => ("dotOutP", [wN], [N, m, n])
  | .expe (n := n) => ("expeP", [], [N, n])
  | .softmaxDiv (n := n) => ("softmaxDivP", [], [N, n])
  | .lnRow (m := m) (n := n) gN bN es _ _ _ => ("lnRowP", [gN, bN, es], [N, m, n])
  | .rowScale (m := m) (n := n) gN _ => ("rowScaleP", [gN], [N, m, n])
  | .rowBias (m := m) (n := n) bN _ => ("rowBiasP", [bN], [N, m, n])
  -- ViT increment 1. ⚠ The FIRST entry of `info` is the batch and every following one is the tag's
  -- own dims — so `[N, tk, a, c]` reads "batch N, token count tk". The emitter below uses only the
  -- tail (it takes the batch from `pretty`'s `B`), which is why the batched text is its
  -- per-example peer's byte for byte; `tests/TestBatchedEmitTie.lean` is what pins that.
  | .denseRow (N := tk) (a := a) (c := c) wN bN _ _ => ("denseRowP", [wN, bN], [N, tk, a, c])
  | .denseRowBf16 (N := tk) (a := a) (c := c) _ wN bN _ _ =>
      ("denseRowPBf16", [wN, bN], [N, tk, a, c])
  | .patchEmbed (ic := ic) (H := H) (W := W) (P := P) (N := tk) (D := D) wN bN clsN posN _ _ _ _ =>
      ("patchEmbedP", [wN, bN, clsN, posN], [N, ic, H, W, P, tk, D])
  | .patchEmbedBf16 (ic := ic) (H := H) (W := W) (P := P) (N := tk) (D := D)
        _ wN bN clsN posN _ _ _ _ =>
      ("patchEmbedPBf16", [wN, bN, clsN, posN], [N, ic, H, W, P, tk, D])
  | .clsSlice (N := tk) (D := D) => ("clsSliceP", [], [N, tk, D])
  | .clsPad (N := tk) (D := D) => ("clsPadP", [], [N, tk, D])
  | .headSlice (N := tk) (heads := heads) (d := d) h => ("headSliceP", [], [N, tk, heads, d, h.val])
  | .headPad (N := tk) (heads := heads) (d := d) h => ("headPadP", [], [N, tk, heads, d, h.val])

/-- Erase an `SHlo` graph to its renderable skeleton (drops `ℝ` values + shape
    index; keeps op structure, shapes, leaf names). -/
def skel : {k : Nat} → SHlo k → Raw
  | k, .operand name _        => .operand name k
  | k, .dotIn (m := m) w _ e  => .dotIn w m k (skel e)
  | k, .dotInBf16 (m := m) _ w _ e => .dotInBf16 w m k (skel e)
  | k, .dotOut (n := n) w _ e => .dotOut w k n (skel e)
  | k, .addBcast b _ e        => .addBcast b k (skel e)
  | k, .expe e                => .expe k (skel e)
  | k, .softmaxDiv e          => .softmaxDiv k (skel e)
  | k, .sub a b               => .sub k (skel a) (skel b)
  | _, .weightSgd (m := m) (n := n) xN wN lrS _ _ _ e => .weightSgd xN wN lrS m n (skel e)
  | k, .biasSgd bN lrS _ _ e  => .biasSgd bN lrS k (skel e)
  | _, .convWeightSgd (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .convWeightSgd xN wN lrS ic oc h w kH kW (skel e)
  | _, .convBiasSgd (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS oc h w (skel e)
  | _, .bnGammaSgd (oc := oc) (h := h) (w := w) gN vN es lrS _ _ _ _ e =>
      .bnGammaSgd gN vN es lrS oc h w (skel e)
  | _, .bnBetaSgd (oc := oc) (h := h) (w := w) bN lrS _ _ e =>
      .bnBetaSgd bN lrS oc h w (skel e)
  | _, .layerScaleChGammaSgd (c := c) (h := h) (w := w) gN xN lrS _ _ _ e =>
      .layerScaleChGammaSgd gN xN lrS c h w (skel e)
  | _, .lnGammaSgd (n := n) gN xN es lrS _ _ _ _ e =>
      .lnGammaSgd gN xN es lrS n (skel e)
  | _, .lnBetaSgd (n := n) bN lrS _ _ e =>
      .lnBetaSgd bN lrS n (skel e)
  | _, .veclnGammaSgd (N := N) (D := D) gN xN es lrS _ _ _ _ e =>
      .veclnGammaSgd gN xN es lrS N D (skel e)
  | _, .patchEmbedWeightSgd (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) wN xN lrS _ _ _ e =>
      .patchEmbedWeightSgd wN xN lrS ic H W P N D (skel e)
  | k, .reluF e               => .reluF k (skel e)
  | k, .selectPos x _ e       => .selectPos x k (skel e)
  | k, .relu6F e              => .relu6F k (skel e)
  | k, .selectMid x _ e       => .selectMid x k (skel e)
  | k, .convertF _ e          => .convertF k (skel e)
  | _, .flatConvF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvF wN bN ic oc h w kH kW (skel e)
  | _, .flatConvFBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ e =>
      .flatConvFBf16 wN bN ic oc h w kH kW (skel e)
  | _, .maxPoolF (c := c) (h := h) (w := w) e => .maxPoolF c h w (skel e)
  | _, .convBack (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .convBack wN ic oc h w kH kW (skel e)
  | _, .maxPoolBack (c := c) (h := h) (w := w) xN _ e => .maxPoolBack xN c h w (skel e)
  | k, .bnF gN bN es _ _ _ e => .bnF gN bN es k (skel e)
  | k, .bnBack gN xN es _ _ _ e => .bnBack gN xN es k (skel e)
  | k, .addV a b              => .addV k (skel a) (skel b)
  | _, .maxPoolBackB (N := N) (c := c) (h := h) (w := w) xN _ e =>
      .batched "maxPoolBackP" [xN] [N, c, h, w] (skel e)
  -- ⭐ The three 3×3/s2 pool forms all ride the generic `.batched` tag, so they cost NO
  -- `Raw`/`Tok`/`toToks`/`parseStack`/`parse_toToks` work (§0.2 increment 2's five-site route).
  -- ⚠ The per-example and batched BACKWARDS share one tag and are distinguished by the nat list's
  -- ARITY (3 vs 4) — `depthwiseWeightGrad`'s convention, and legitimate here for its reason: the
  -- emitter ignores `N` (it reads the batch off `pretty`'s `B`), so the two emit identical text by
  -- construction and the arity carries only the `den`-side difference.
  | _, .maxPool3s2F (c := c) (h := h) (w := w) e =>
      .batched "maxPool3s2" [] [c, h, w] (skel e)
  | _, .maxPool3s2Back (c := c) (h := h) (w := w) xN _ e =>
      .batched "maxPool3s2BackP" [xN] [c, h, w] (skel e)
  | _, .maxPool3s2BackB (N := N) (c := c) (h := h) (w := w) xN _ e =>
      .batched "maxPool3s2BackP" [xN] [N, c, h, w] (skel e)
  | _, .convBiasSgdB (N := N) (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .batched "convBiasSgd" [bN, lrS] [N, oc, h, w] (skel e)
  | _, .convStridedBiasSgdB (N := N) (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .batched "convBiasSgd" [bN, lrS] [N, oc, h, w] (skel e)
  | _, .selectPosB (N := N) (n := n) xN _ e => .batched "selectPosP" [xN] [N, n] (skel e)
  | _, .selectMidB (N := N) (n := n) xN _ e => .batched "selectMidP" [xN] [N, n] (skel e)
  -- Two name slots: the mask INPUT and the baked `1/keep` literal. Same two-string shape
  -- `convStridedWeightSgd` uses for `xN`/`lrS`, so the generic `.batched` tag needs no widening.
  | _, .dropPathB (N := N) (n := n) mN _ e => .batched "dropPathP" [mN] [N, n] (skel e)
  -- ⚠ A DIFFERENT TAG, not a flag on `dropPathP`. The two emit different text and denote different
  -- functions, so they must be distinguishable in the skeleton — a shared tag would make the
  -- round-trip parser unable to tell a per-sample render from a per-element one.
  | _, .dropoutB (N := N) (n := n) mN _ e => .batched "dropoutP" [mN] [N, n] (skel e)
  | _, .swishBackB (N := N) (n := n) xN _ e => .batched "swishBackP" [xN] [N, n] (skel e)
  -- ⚠ Routed through the GENERIC `.batched` tag, like every batched op above — which is why these
  -- cost five sites (ctor, den, the `rfl` theorem, this line, `emitTok`) and not §4's ten: `Raw`,
  -- `Tok`, `toToks`, `parseStack` and the `parse_toToks` induction all already handle `.batched`.
  | _, .geluBackB (N := N) (n := n) xN _ e => .batched "geluBackP" [xN] [N, n] (skel e)
  -- ⚠⚠ THESE FOUR ALIAS THEIR PER-EXAMPLE PEER'S `Raw` — deliberately, and it is the pattern the
  -- depthwise bias grads already use. Their emitted MLIR ALREADY contracts the batch axis
  -- (`layerScaleChGammaGrad` reduces `dimensions = [0, 2, 3]`, `rowDenseBiasGrad` `[0, 1]`), because
  -- values flow as `tensor<B, …>` and `B` is `pretty`'s, never the SHlo index. So the batched form
  -- emits the same text BY CONSTRUCTION rather than by a copied body — no new `emitTok` case, and
  -- no way for the two to drift. It is the `den` that was per-example and is now honest.
  --
  -- ⚠ Note what is dropped: the batch `N` does NOT ride in `info`. The emitter never used it (it
  -- reads `B`), so passing it would add a number that means nothing at the emit and everything at
  -- the denotation — the exact conflation this whole thread exists to remove.
  | _, .convStride4WeightGradB (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convStride4WeightGrad" [xN] [ic, oc, h, w, kH, kW] (skel e)
  | _, .convStride4WeightGradBBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convStride4WeightGradBf16" [xN] [ic, oc, h, w, kH, kW] (skel e)
  | _, .layerScaleChGammaGradB (c := c) (h := h) (w := w) xN _ e =>
      .batched "layerScaleChGammaGrad" [xN] [c, h, w] (skel e)
  | _, .veclnGammaGradB (R := R) (D := D) xN es _ _ e =>
      .batched "veclnGammaGrad" [xN, es] [R, D] (skel e)
  | _, .rowDenseBiasGradB (R := R) (c := c) e =>
      .batched "rowDenseBiasGrad" [] [R, c] (skel e)
  -- ── ViT increment 2: SIX ALIASES AND NOT ONE NEW EMIT CASE. Every one of these emits its
  --    per-example peer's `Raw` verbatim — the batch never appears in the tag, because the emitter
  --    reads `B` from `pretty` and its dims from the tag, and (for the four gradients) already
  --    contracts the batch axis. So these forms cannot drift from their peers by construction
  --    rather than by a copied body kept honest by a test. Increment 3's finding, generalised.
  --    ⚠ `matmulFB` rides the BINARY `.matmulF` tag: `.batched2` exists for `addVB`/`subB`, but a
  --    direct alias is better still, because it shares the emit rather than restating it.
  | _, .matmulFB (m := m) (k := k) (n := n) a b => .matmulF m k n (skel a) (skel b)
  -- ⚠ Its bf16 peer can NOT alias `.matmulF` — the text differs — so it rides the generic BINARY
  -- skeleton `.batched2` that `addVB`/`subB` already use, and needs no new `Raw`/`Tok` of its own.
  | _, .matmulFBBf16 (m := m) (k := k) (n := n) _ a b =>
      .batched2 "matmulFBf16" [] [m, k, n] (skel a) (skel b)
  | _, .softmaxRowBackB (m := m) (n := n) x _ e => .softmaxRowBack x m n (skel e)
  | _, .rowDenseWeightGradB (tk := tk) (a := a) (c := c) xN _ e =>
      .batched "rowDenseWeightGrad" [xN] [tk, a, c] (skel e)
  | _, .rowDenseWeightGradBBf16 (tk := tk) (a := a) (c := c) _ xN _ e =>
      .batched "rowDenseWeightGradBf16" [xN] [tk, a, c] (skel e)
  | _, .posEmbedGradB (tk := tk) (D := D) e =>
      .batched "posEmbedGrad" [] [tk, D] (skel e)
  | _, .patchEmbedWeightGradB (ic := ic) (H := H) (W := W) (P := P) (tk := tk) (D := D) xN _ e =>
      .batched "patchEmbedWeightGrad" [xN] [ic, H, W, P, tk, D] (skel e)
  | _, .patchEmbedWeightGradBBf16 (ic := ic) (H := H) (W := W) (P := P) (tk := tk) (D := D)
        _ xN _ e =>
      .batched "patchEmbedWeightGradBf16" [xN] [ic, H, W, P, tk, D] (skel e)
  | _, .patchEmbedBiasGradB (tk := tk) (c := c) e =>
      .batched "patchEmbedBiasGrad" [] [tk, c] (skel e)
  | _, .weightGradB (m := m) (n := n) xN _ e => .weightGrad xN m n (skel e)
  | _, .biasGradB (n := n) e => .biasGrad n (skel e)
  | _, .lnRowBackB (N := N) (m := m) (n := n) gN xN es _ _ _ e =>
      .batched "lnRowBackP" [gN, xN, es] [N, m, n] (skel e)
  | _, .sigmoidB (N := N) (n := n) e => .batched "sigmoidP" [] [N, n] (skel e)
  | _, .sigmoidBackB (N := N) (n := n) xN _ e => .batched "sigmoidBackP" [xN] [N, n] (skel e)
  | _, .addVB (N := N) (n := n) a b => .batched2 "addV" [] [N, n] (skel a) (skel b)
  | _, .subB (N := N) (n := n) a b  => .batched2 "sub" [] [N, n] (skel a) (skel b)
  | _, .gapF (c := c) (h := h) (w := w) e => .gapF c h w (skel e)
  | _, .gapBack (c := c) (h := h) (w := w) e => .gapBack c h w (skel e)
  | _, .broadcastBack (c := c) (h := h) (w := w) e => .broadcastBack c h w (skel e)
  | _, .flatConvStridedF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStridedF wN bN ic oc h w kH kW (skel e)
  | _, .flatConvStridedXlaF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStridedXlaF wN bN ic oc h w kH kW (skel e)
  | _, .convStridedBack (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .convStridedBack wN ic oc h w kH kW (skel e)
  | _, .convStridedWeightSgd (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .convStridedWeightSgd xN wN lrS ic oc h w kH kW (skel e)
  | _, .convStridedBiasSgd (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS oc h w (skel e)
  | _, .depthwiseWeightSgd (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .depthwiseWeightSgd xN wN lrS c h w kH kW (skel e)
  | _, .depthwiseBiasSgd (c := c) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS c h w (skel e)
  | _, .depthwiseStridedWeightSgd (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .depthwiseStridedWeightSgd xN wN lrS c h w kH kW (skel e)
  | _, .depthwiseStridedBiasSgd (c := c) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS c h w (skel e)
  | _, .convStridedXlaWeightSgd (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .convStridedXlaWeightSgd xN wN lrS ic oc h w kH kW (skel e)
  | _, .convStridedXlaBiasSgd (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS oc h w (skel e)
  | _, .depthwiseStridedXlaWeightSgd (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .depthwiseStridedXlaWeightSgd xN wN lrS c h w kH kW (skel e)
  | _, .depthwiseStridedXlaBiasSgd (c := c) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS c h w (skel e)
  | _, .flatConvStride4F (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStride4F wN bN ic oc h w kH kW (skel e)
  | _, .bnPerChannelF (oc := oc) (h := h) (w := w) gN bN es _ _ _ e =>
      .bnPerChannelF gN bN es oc h w (skel e)
  | _, .bnPerChannelBack (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .bnPerChannelBack gN xN es oc h w (skel e)
  | _, .bnPerChannelEvalF (oc := oc) (h := h) (w := w) gN bN muN varN es _ _ _ _ _ e =>
      .bnPerChannelEvalF gN bN muN varN es oc h w (skel e)
  | _, .weightGrad (m := m) (n := n) xN _ e => .weightGrad xN m n (skel e)
  | _, .biasGrad (n := n) e => .biasGrad n (skel e)
  | _, .convWeightGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .convWeightGrad xN ic oc h w kH kW (skel e)
  | _, .convBiasGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .convBiasGrad ic oc h w kH kW (skel e)
  | _, .convStridedWeightGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .convStridedWeightGrad xN ic oc h w kH kW (skel e)
  -- rides the generic `.batched` tag (the four-site route, §4) — no new Raw/Tok/parse case.
  | _, .convStride4WeightGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convStride4WeightGrad" [xN] [ic, oc, h, w, kH, kW] (skel e)
  -- aliases convBiasGrad's Raw: the bias grad is stride-independent, so the text is identical
  | _, .convStridedBiasGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .convBiasGrad ic oc h w kH kW (skel e)
  | _, .bnGammaGrad (oc := oc) (h := h) (w := w) vN es _ _ e => .bnGammaGrad vN es oc h w (skel e)
  | _, .bnBetaGrad (oc := oc) (h := h) (w := w) e => .bnBetaGrad oc h w (skel e)
  | _, .adamMNextF mN b1N ob1N ds _ _ e => .adamMNextF mN b1N ob1N ds (skel e)
  | _, .adamVNextF vN b2N ob2N ds _ _ e => .adamVNextF vN b2N ob2N ds (skel e)
  | _, .adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds
        _ _ _ _ _ _ _ _ _ _ e =>
      .adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds (skel e)
  | _, .sgdParamF θN lrN ds _ _ e => .sgdParamF θN lrN ds (skel e)
  | _, .momVNextF vN muN ds _ _ e => .momVNextF vN muN ds (skel e)
  | _, .momParamF θN vN muN lrN ds _ _ _ _ e => .momParamF θN vN muN lrN ds (skel e)
  | _, .rmsBufNextF sqN bufN rhoN orhoN muN epsN ds _ _ _ _ _ e =>
      .rmsBufNextF sqN bufN rhoN orhoN muN epsN ds (skel e)
  | _, .gradSumSqAccF ds acc e         => .gradSumSqAccF ds (skel acc) (skel e)
  | _, .clipScaleF cS eS _ _ ds s e    => .clipScaleF cS eS ds (skel s) (skel e)
  | _, .lambDirF a b c d e' f g h i j k ds _ _ _ _ _ _ _ _ _ x =>
      .lambDirF a b c d e' f g h i j k ds (skel x)
  | _, .lambScaleF ds s e              => .lambScaleF ds (skel s) (skel e)
  | _, .depthwiseF (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .depthwiseF wN bN c h w kH kW (skel e)
  | _, .depthwiseBack (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .depthwiseBack wN c h w kH kW (skel e)
  | _, .depthwiseStridedF (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .depthwiseStridedF wN bN c h w kH kW (skel e)
  | _, .depthwiseStridedXlaF (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .depthwiseStridedXlaF wN bN c h w kH kW (skel e)
  | _, .depthwiseStridedBack (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .depthwiseStridedBack wN c h w kH kW (skel e)
  | _, .depthwiseStridedXlaBack (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .depthwiseStridedXlaBack wN c h w kH kW (skel e)
  | k, .swishF e             => .swishF k (skel e)
  | k, .swishBack x _ e      => .swishBack x k (skel e)
  | k, .sigmoidF e           => .sigmoidF k (skel e)
  | k, .sigmoidBack x _ e    => .sigmoidBack x k (skel e)
  | k, .geluF e              => .geluF k (skel e)
  | k, .geluBack x _ e       => .geluBack x k (skel e)
  | k, .layerScaleF γN _ e   => .layerScaleF γN k (skel e)
  | _, .layerScaleChF (c := c) (h := h) (w := w) γN _ e => .layerScaleChF γN c h w (skel e)
  | _, .softmaxRowF (m := m) (n := n) e => .softmaxRowF m n (skel e)
  | _, .softmaxRowBack (m := m) (n := n) x _ e => .softmaxRowBack x m n (skel e)
  | _, .matmulF (m := m) (k := k) (n := n) a b => .matmulF m k n (skel a) (skel b)
  | _, .transposeF (m := m) (n := n) e => .transposeF m n (skel e)
  | k, .scaleF sStr _ e => .scaleF sStr k (skel e)
  | _, .lnRowF (m := m) (n := n) gN bN es _ _ _ e => .lnRowF gN bN es m n (skel e)
  | _, .lnRowBack (m := m) (n := n) gN xN es _ _ _ e => .lnRowBack gN xN es m n (skel e)
  | _, .denseRowF (N := N) (a := a) (c := c) wN bN _ _ e => .denseRowF wN bN N a c (skel e)
  | _, .denseRowBack (N := N) (a := a) (c := c) wN _ e => .denseRowBack wN N a c (skel e)
  | _, .patchEmbedF (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) wN bN cN pN _ _ _ _ e =>
      .patchEmbedF wN bN cN pN ic H W P N D (skel e)
  | _, .patchEmbedBack (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ _ e =>
      .batched "patchEmbedBack" [] [ic, H, W, P, N, D] (skel e)
  | _, .clsSliceF (N := N) (D := D) e => .clsSliceF N D (skel e)
  | _, .clsPadF (N := N) (D := D) e => .clsPadF N D (skel e)
  | _, .headSliceF (N := N) (heads := heads) (d := d) h e => .headSliceF N heads d h.val (skel e)
  | _, .headPadF (N := N) (heads := heads) (d := d) h e => .headPadF N heads d h.val (skel e)
  | _, .rowScaleF (m := m) (n := n) gN _ e => .rowScaleF gN m n (skel e)
  | _, .rowBiasF (m := m) (n := n) bN _ e => .rowBiasF bN m n (skel e)
  | _, .batchOp (N := N) op e =>
      let (tag, nms, inf) := batchOpDescr N op; .batched tag nms inf (skel e)
  | _, .bnBatchF (N := N) (oc := oc) (h := h) (w := w) gN bN es _ _ _ e =>
      .batched "bnBatch" [gN, bN, es] [N, oc, h, w] (skel e)
  | _, .bnBatchBack (N := N) (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .batched "bnBatchBack" [gN, xN, es] [N, oc, h, w] (skel e)
  | _, .convBackBatched (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "convBackBatched" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedBackBatched (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "convStridedBackBatched" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convBackBatchedBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "convBackBatchedBf16" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convBackBatchedF8 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "convBackBatchedF8" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedBackBatchedBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "convStridedBackBatchedBf16" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .depthwiseBackBatched (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "depthwiseBackBatched" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "depthwiseBackBatchedBf16" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedBackBatched (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "depthwiseStridedBackBatched" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "depthwiseStridedBackBatchedBf16" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedXlaBackBatched (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "depthwiseStridedXlaBackBatched" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedXlaBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "depthwiseStridedXlaBackBatchedBf16" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .bnBatchLABack (N := N) (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .batched "bnBatchLABack" [gN, xN, es] [N, oc, h, w] (skel e)
  | _, .seBackBatched (N := N) (c := c) (h := h) (w := w) (r := r) w1 b1 w2 b2 vN _ _ _ _ _ e =>
      .batched "seBackBatched" [w1, b1, w2, b2, vN] [N, c, h, w, r] (skel e)
  | _, .seReduceB (N := N) (c := c) (h := h) (w := w) xN _ e =>
      .batched "seReduceB" [xN] [N, c, h, w] (skel e)
  | _, .gapBackBatched (N := N) (c := c) (h := h) (w := w) e =>
      .batched "gapBackBatched" [] [N, c, h, w] (skel e)
  | _, .bnGammaSgdB (N := N) (oc := oc) (h := h) (w := w) gN vN es lrS _ _ _ _ e =>
      .batched "bnGammaSgd" [gN, vN, es, lrS] [N, oc, h, w] (skel e)
  | _, .bnBetaSgdB (N := N) (oc := oc) (h := h) (w := w) bN lrS _ _ e =>
      .batched "bnBetaSgd" [bN, lrS] [N, oc, h, w] (skel e)
  | _, .denseWeightSgdB (N := N) (a := a) (c := c) xN wN lrS _ _ _ e =>
      .batched "denseWeightSgd" [xN, wN, lrS] [N, a, c] (skel e)
  | _, .denseBiasSgdB (N := N) (c := c) bN lrS _ _ e =>
      .batched "denseBiasSgd" [bN, lrS] [N, c] (skel e)
  | _, .convWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convWeightGrad" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convStridedWeightGrad" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convWeightGradBf16" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convWeightGradBF8 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convWeightGradF8" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convStridedWeightGradBf16" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedXlaWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convStridedXlaWeightGrad" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedXlaWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convStridedXlaWeightGradBf16" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convBiasGradB (N := N) (oc := oc) (h := h) (w := w) _ _ _ e =>
      .batched "convBiasGrad" [] [N, oc, h, w] (skel e)
  | _, .convStridedBiasGradB (N := N) (oc := oc) (h := h) (w := w) _ _ _ e =>
      .batched "convBiasGrad" [] [N, oc, h, w] (skel e)
  -- ⭐ Aliases "convBiasGrad" DELIBERATELY: `Σ_{batch,spatial} dy` is padding-independent, so the
  -- emitted text is character-identical and only `den` distinguishes them. Same aliasing
  -- `convStridedBiasGradB` already does for stride.
  | _, .convStridedXlaBiasGradB (N := N) (oc := oc) (h := h) (w := w) _ _ _ e =>
      .batched "convBiasGrad" [] [N, oc, h, w] (skel e)
  | _, .bnGammaGradB (N := N) (oc := oc) (h := h) (w := w) vN es _ _ e =>
      .batched "bnGammaGrad" [vN, es] [N, oc, h, w] (skel e)
  | _, .bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) e =>
      .batched "bnBetaGrad" [] [N, oc, h, w] (skel e)
  | _, .denseWeightGradB (N := N) (a := a) (c := c) xN _ e =>
      .batched "denseWeightGrad" [xN] [N, a, c] (skel e)
  | _, .denseBiasGradB (N := N) (c := c) e =>
      .batched "denseBiasGrad" [] [N, c] (skel e)
  | _, .bnBatchMeanB (N := N) (oc := oc) (h := h) (w := w) e =>
      .batched "bnBatchMean" [] [N, oc, h, w] (skel e)
  | _, .bnBatchVarB (N := N) (oc := oc) (h := h) (w := w) e =>
      .batched "bnBatchVar" [] [N, oc, h, w] (skel e)
  | _, .bnBatchVarAtB (N := N) (oc := oc) (h := h) (w := w) e mu =>
      .batched2 "bnBatchVarAt" [] [N, oc, h, w] (skel e) (skel mu)
  | _, .bnPackB (oc := oc) a b => .batched2 "bnPack" [] [oc] (skel a) (skel b)
  | _, .bnSyncF (N := N) (oc := oc) (h := h) (w := w) gN bN es _ _ _ x st =>
      .batched2 "bnSync" [gN, bN, es] [N, oc, h, w] (skel x) (skel st)
  | _, .bnSyncDyStatsB (N := N) (oc := oc) (h := h) (w := w) gN xN es _ _ _ dy st =>
      .batched2 "bnSyncDyStats" [gN, xN, es] [N, oc, h, w] (skel dy) (skel st)
  | _, .bnSyncBack (N := N) (oc := oc) (h := h) (w := w) gN xN es _ _ _ dy ds =>
      .batched2 "bnSyncBack" [gN, xN, es] [N, oc, h, w] (skel dy) (skel ds)
  | _, .bnSyncGammaGradB (N := N) (oc := oc) (h := h) (w := w) xN es _ _ dy st =>
      .batched2 "bnSyncGammaGrad" [xN, es] [N, oc, h, w] (skel dy) (skel st)
  | _, .bnStatsMeanB (oc := oc) e => .batched "bnStatsMean" [] [oc] (skel e)
  | _, .bnStatsVarB  (oc := oc) e => .batched "bnStatsVar"  [] [oc] (skel e)
  | _, .scaleB (N := N) (n := n) sS _ e    => .batched "scale" [sS] [N, n] (skel e)
  | _, .shiftB (N := N) (n := n) sS _ e    => .batched "shift" [sS] [N, n] (skel e)
  | _, .divConstB (N := N) (n := n) sS _ e => .batched "divConst" [sS] [N, n] (skel e)
  | _, .allReduceMeanF R hR t ds g => .allReduceMean R t ds (skel (g ⟨0, hR⟩))
  | _, .rowDenseWeightSgd (N := N) (a := a) (c := c) xN wN lrS _ _ _ e =>
      .batched "rowDenseWeightSgd" [xN, wN, lrS] [N, a, c] (skel e)
  | _, .rowDenseBiasSgd (N := N) (c := c) bN lrS _ _ e =>
      .batched "rowDenseBiasSgd" [bN, lrS] [N, c] (skel e)
  | _, .patchEmbedBiasSgd (N := N) (c := c) bN lrS _ _ e =>
      .batched "patchEmbedBiasSgd" [bN, lrS] [N, c] (skel e)
  | _, .posEmbedSgd (N := N) (D := D) pN lrS _ _ e =>
      .batched "posEmbedSgd" [pN, lrS] [N, D] (skel e)
  -- The un-fused transformer peers. They ride the same generic `.batched` tag (so `Raw`/`Tok`/
  -- `toToks`/`parseStack`/`parse_toToks` need no new cases), carrying only the operands the
  -- gradient actually reads — no param name, no lr.
  | _, .rowDenseWeightGrad (N := N) (a := a) (c := c) xN _ e =>
      .batched "rowDenseWeightGrad" [xN] [N, a, c] (skel e)
  | _, .rowDenseBiasGrad (N := N) (c := c) e =>
      .batched "rowDenseBiasGrad" [] [N, c] (skel e)
  | _, .patchEmbedBiasGrad (N := N) (c := c) e =>
      .batched "patchEmbedBiasGrad" [] [N, c] (skel e)
  | _, .posEmbedGrad (N := N) (D := D) e =>
      .batched "posEmbedGrad" [] [N, D] (skel e)
  | _, .veclnGammaGrad (N := N) (D := D) xN es _ _ e =>
      .batched "veclnGammaGrad" [xN, es] [N, D] (skel e)
  | _, .patchEmbedWeightGrad (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) xN _ e =>
      .batched "patchEmbedWeightGrad" [xN] [ic, H, W, P, N, D] (skel e)
  | _, .convWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "convWeightSgd" [xN, wN, lrS] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "convStridedWeightSgd" [xN, wN, lrS] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedXlaWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "convStridedXlaWeightSgd" [xN, wN, lrS] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .depthwiseWeightSgdB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "depthwiseWeightSgd" [xN, wN, lrS] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseWeightGradB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "depthwiseWeightGrad" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "depthwiseWeightGradBf16" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedXlaWeightGradB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "depthwiseStridedXlaWeightGrad" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedXlaWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "depthwiseStridedXlaWeightGradBf16" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedWeightGradB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "depthwiseStridedWeightGrad" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "depthwiseStridedWeightGradBf16" [xN] [N, c, h, w, kH, kW] (skel e)
  -- Both depthwise BIAS grads alias ConvNeXt's per-example `depthwiseBiasGrad` Raw — the bias
  -- gradient is `Σ_{batch,spatial} dy`, stride-independent AND kernel-independent, so one emitter
  -- serves all three. `N` is dropped for the same reason every batched tag drops it: the runtime
  -- batch is `B`. This is the aliasing route, so there is nothing to add in `emitTok`.
  | _, .depthwiseBiasGradB (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .batched "depthwiseBiasGrad" [] [c, h, w, kH, kW] (skel e)
  -- ⚠ info is `[c, h, w, kH, kW]` — NO leading `N`, matching the symmetric peer below rather than
  -- the `*WeightGrad` convention. It aliases that op's Raw, so the shape must agree exactly.
  | _, .depthwiseStridedXlaBiasGradB (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .batched "depthwiseBiasGrad" [] [c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedBiasGradB (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .batched "depthwiseBiasGrad" [] [c, h, w, kH, kW] (skel e)
  -- The ConvNeXt five ride the generic `.batched` Raw/Tok tag, so they need no new
  -- `Raw`/`Tok`/`toToks`/`parseStack`/`parse_toToks` cases — the four-site route (§4).
  --
  -- `depthwiseWeightGrad` deliberately **aliases the batched op's tag**: its emitted text is
  -- byte-identical (that emitter ignores its `N` and reads the width off the render batch `B`), so
  -- only `den` differs — per-example here, a sum over `Fin N` there. Exactly the aliasing
  -- `convStridedBiasGrad` already does against `convBiasGrad`. It is distinguished by ARITY: six
  -- nats for the batched form, five for this one, so the two `emitTok` cases cannot collide.
  | _, .depthwiseWeightGrad (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "depthwiseWeightGrad" [xN] [c, h, w, kH, kW] (skel e)
  | _, .depthwiseBiasGrad (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .batched "depthwiseBiasGrad" [] [c, h, w, kH, kW] (skel e)
  | _, .lnGammaGrad (n := n) xN es _ _ e => .batched "lnGammaGrad" [xN, es] [n] (skel e)
  | _, .lnBetaGrad (n := n) e => .batched "lnBetaGrad" [] [n] (skel e)
  | _, .layerScaleChGammaGrad (c := c) (h := h) (w := w) xN _ e =>
      .batched "layerScaleChGammaGrad" [xN] [c, h, w] (skel e)
  | _, .depthwiseStridedWeightSgdB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "depthwiseStridedWeightSgd" [xN, wN, lrS] [N, c, h, w, kH, kW] (skel e)

/-- One serialized token: an opcode with shapes/names; operands are positional. -/
inductive Tok where
  | operand    (name : String) (n : Nat)  : Tok
  | dotIn      (w : String) (m n : Nat)    : Tok
  | dotInBf16  (w : String) (m n : Nat)    : Tok
  | dotOut     (w : String) (m n : Nat)    : Tok
  | addBcast   (b : String) (n : Nat)      : Tok
  | expe       (n : Nat)                   : Tok
  | softmaxDiv (n : Nat)                   : Tok
  | sub        (n : Nat)                   : Tok
  | weightSgd  (xName wName lrStr : String) (m n : Nat) : Tok
  | biasSgd    (bName lrStr : String) (n : Nat)         : Tok
  | convWeightSgd (xName wName lrStr : String) (ic oc h w kH kW : Nat) : Tok
  | convBiasSgd   (bName lrStr : String) (oc h w : Nat)               : Tok
  | bnGammaSgd    (gName vName epsStr lrStr : String) (oc h w : Nat)  : Tok
  | bnBetaSgd     (bName lrStr : String) (oc h w : Nat)               : Tok
  | layerScaleChGammaSgd (gName xName lrStr : String) (c h w : Nat)   : Tok
  | lnGammaSgd    (gName xName epsStr lrStr : String) (n : Nat)       : Tok
  | lnBetaSgd     (bName lrStr : String) (n : Nat)                    : Tok
  | veclnGammaSgd (gName xName epsStr lrStr : String) (N D : Nat)     : Tok
  | patchEmbedWeightSgd (wName xName lrStr : String) (ic H W P N D : Nat) : Tok
  | reluF      (n : Nat)                   : Tok
  | selectPos  (x : String) (n : Nat)      : Tok
  | relu6F     (n : Nat)                   : Tok
  | selectMid  (x : String) (n : Nat)      : Tok
  | convertF   (n : Nat)                   : Tok
  | flatConvF  (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | flatConvFBf16 (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | maxPoolF   (c h w : Nat)               : Tok
  | convBack   (w : String) (ic oc h w' kH kW : Nat) : Tok
  | maxPoolBack (x : String) (c h w : Nat) : Tok
  | bnF        (g b eps : String) (n : Nat) : Tok
  | bnBack     (g x eps : String) (n : Nat) : Tok
  | addV       (n : Nat)                   : Tok
  | gapF       (c h w : Nat)               : Tok
  | gapBack    (c h w : Nat)               : Tok
  | broadcastBack (c h w : Nat)            : Tok
  | flatConvStridedF (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | flatConvStridedXlaF (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | convStridedBack  (w : String) (ic oc h w' kH kW : Nat) : Tok
  | convStridedWeightSgd (xName wName lrStr : String) (ic oc h w' kH kW : Nat) : Tok
  | depthwiseWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Tok
  | convStridedXlaWeightSgd (xName wName lrStr : String) (ic oc h w' kH kW : Nat) : Tok
  | depthwiseStridedXlaWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Tok
  | flatConvStride4F (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | bnPerChannelF    (g b eps : String) (oc h w : Nat) : Tok
  | bnPerChannelBack (g x eps : String) (oc h w : Nat) : Tok
  | bnPerChannelEvalF (g b mu var eps : String) (oc h w : Nat) : Tok
  | weightGrad (x : String) (m n : Nat) : Tok
  | biasGrad (n : Nat) : Tok
  | convWeightGrad (x : String) (ic oc h w' kH kW : Nat) : Tok
  | convBiasGrad (ic oc h w' kH kW : Nat) : Tok
  | convStridedWeightGrad (x : String) (ic oc h w' kH kW : Nat) : Tok
  | bnGammaGrad (v eps : String) (oc h w' : Nat) : Tok
  | bnBetaGrad (oc h w' : Nat) : Tok
  | adamMNextF (m b1 ob1 : String) (ds : List Nat) : Tok
  | adamVNextF (v b2 ob2 : String) (ds : List Nat) : Tok
  | adamWParamF (θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd : String) (ds : List Nat) : Tok
  | sgdParamF (θ lr : String) (ds : List Nat) : Tok
  | momVNextF (v mu : String) (ds : List Nat) : Tok
  | momParamF (θ v mu lr : String) (ds : List Nat) : Tok
  | rmsBufNextF (sq buf rho orho mu eps : String) (ds : List Nat) : Tok
  | gradSumSqAccF (ds : List Nat)                           : Tok
  | clipScaleF    (clipStr epsStr : String) (ds : List Nat) : Tok
  | lambDirF (θ m v b1 ob1 b2 ob2 bc1 bc2 eps wd : String) (ds : List Nat) : Tok
  | lambScaleF    (ds : List Nat)                           : Tok
  | depthwiseF    (w b : String) (c h w' kH kW : Nat) : Tok
  | depthwiseBack (w : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedF    (w b : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedXlaF (w b : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedBack (w : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedXlaBack (w : String) (c h w' kH kW : Nat) : Tok
  | swishF     (n : Nat)                   : Tok
  | swishBack  (x : String) (n : Nat)      : Tok
  | sigmoidF   (n : Nat)                   : Tok
  | sigmoidBack (x : String) (n : Nat)     : Tok
  | geluF      (n : Nat)                   : Tok
  | geluBack   (x : String) (n : Nat)      : Tok
  | layerScaleF (γ : String) (n : Nat)     : Tok
  | layerScaleChF (γ : String) (c h w : Nat) : Tok
  | softmaxRowF    (m n : Nat)             : Tok
  | softmaxRowBack (x : String) (m n : Nat) : Tok
  | matmulF    (m k n : Nat)               : Tok
  | transposeF (m n : Nat)                 : Tok
  | scaleF     (s : String) (n : Nat)      : Tok
  | lnRowF     (g b eps : String) (m n : Nat) : Tok
  | lnRowBack  (g x eps : String) (m n : Nat) : Tok
  | denseRowF  (w b : String) (N a c : Nat) : Tok
  | denseRowBack (w : String) (N a c : Nat) : Tok
  | patchEmbedF (w b cls pos : String) (ic H W P N D : Nat) : Tok
  | clsSliceF  (N D : Nat)                 : Tok
  | clsPadF    (N D : Nat)                 : Tok
  | headSliceF (N heads d hIdx : Nat)      : Tok
  | headPadF   (N heads d hIdx : Nat)      : Tok
  | rowScaleF  (g : String) (m n : Nat)    : Tok
  | rowBiasF   (b : String) (m n : Nat)    : Tok
  | batched    (tag : String) (names : List String) (info : List Nat) : Tok
  | batched2   (tag : String) (names : List String) (info : List Nat) : Tok
  | allReduceMean (R : Nat) (t : String) (ds : List Nat) : Tok
deriving Repr

/-- Postorder serialization: children, then the node's opcode token. -/
def toToks : Raw → List Tok
  | .operand nm n    => [.operand nm n]
  | .dotIn w m n e   => toToks e ++ [.dotIn w m n]
  | .dotInBf16 w m n e => toToks e ++ [.dotInBf16 w m n]
  | .dotOut w m n e  => toToks e ++ [.dotOut w m n]
  | .addBcast b n e  => toToks e ++ [.addBcast b n]
  | .expe n e        => toToks e ++ [.expe n]
  | .softmaxDiv n e  => toToks e ++ [.softmaxDiv n]
  | .sub n a b       => toToks a ++ toToks b ++ [.sub n]
  | .weightSgd xN wN lrS m n e => toToks e ++ [.weightSgd xN wN lrS m n]
  | .biasSgd bN lrS n e        => toToks e ++ [.biasSgd bN lrS n]
  | .convWeightSgd xN wN lrS ic oc h w kH kW e => toToks e ++ [.convWeightSgd xN wN lrS ic oc h w kH kW]
  | .convBiasSgd bN lrS oc h w e               => toToks e ++ [.convBiasSgd bN lrS oc h w]
  | .bnGammaSgd gN vN es lrS oc h w e          => toToks e ++ [.bnGammaSgd gN vN es lrS oc h w]
  | .bnBetaSgd bN lrS oc h w e                 => toToks e ++ [.bnBetaSgd bN lrS oc h w]
  | .layerScaleChGammaSgd gN xN lrS c h w e    => toToks e ++ [.layerScaleChGammaSgd gN xN lrS c h w]
  | .lnGammaSgd gN xN es lrS n e               => toToks e ++ [.lnGammaSgd gN xN es lrS n]
  | .lnBetaSgd bN lrS n e                      => toToks e ++ [.lnBetaSgd bN lrS n]
  | .veclnGammaSgd gN xN es lrS N D e          => toToks e ++ [.veclnGammaSgd gN xN es lrS N D]
  | .patchEmbedWeightSgd wN xN lrS ic H W P N D e => toToks e ++ [.patchEmbedWeightSgd wN xN lrS ic H W P N D]
  | .reluF n e       => toToks e ++ [.reluF n]
  | .selectPos x n e => toToks e ++ [.selectPos x n]
  | .relu6F n e      => toToks e ++ [.relu6F n]
  | .selectMid x n e => toToks e ++ [.selectMid x n]
  | .convertF n e    => toToks e ++ [.convertF n]
  | .flatConvF w b ic oc h w' kH kW e => toToks e ++ [.flatConvF w b ic oc h w' kH kW]
  | .flatConvFBf16 w b ic oc h w' kH kW e => toToks e ++ [.flatConvFBf16 w b ic oc h w' kH kW]
  | .maxPoolF c h w e => toToks e ++ [.maxPoolF c h w]
  | .convBack w ic oc h w' kH kW e => toToks e ++ [.convBack w ic oc h w' kH kW]
  | .maxPoolBack x c h w e => toToks e ++ [.maxPoolBack x c h w]
  | .bnF g b eps n e => toToks e ++ [.bnF g b eps n]
  | .bnBack g x eps n e => toToks e ++ [.bnBack g x eps n]
  | .addV n a b      => toToks a ++ toToks b ++ [.addV n]
  | .gapF c h w e    => toToks e ++ [.gapF c h w]
  | .gapBack c h w e => toToks e ++ [.gapBack c h w]
  | .broadcastBack c h w e => toToks e ++ [.broadcastBack c h w]
  | .flatConvStridedF w b ic oc h w' kH kW e => toToks e ++ [.flatConvStridedF w b ic oc h w' kH kW]
  | .flatConvStridedXlaF w b ic oc h w' kH kW e => toToks e ++ [.flatConvStridedXlaF w b ic oc h w' kH kW]
  | .convStridedBack w ic oc h w' kH kW e => toToks e ++ [.convStridedBack w ic oc h w' kH kW]
  | .convStridedWeightSgd xN wN lrS ic oc h w' kH kW e => toToks e ++ [.convStridedWeightSgd xN wN lrS ic oc h w' kH kW]
  | .depthwiseWeightSgd xN wN lrS c h w' kH kW e => toToks e ++ [.depthwiseWeightSgd xN wN lrS c h w' kH kW]
  | .depthwiseStridedWeightSgd xN wN lrS c h w' kH kW e => toToks e ++ [.depthwiseStridedWeightSgd xN wN lrS c h w' kH kW]
  | .convStridedXlaWeightSgd xN wN lrS ic oc h w' kH kW e => toToks e ++ [.convStridedXlaWeightSgd xN wN lrS ic oc h w' kH kW]
  | .depthwiseStridedXlaWeightSgd xN wN lrS c h w' kH kW e => toToks e ++ [.depthwiseStridedXlaWeightSgd xN wN lrS c h w' kH kW]
  | .flatConvStride4F w b ic oc h w' kH kW e => toToks e ++ [.flatConvStride4F w b ic oc h w' kH kW]
  | .bnPerChannelF g b eps oc h w e => toToks e ++ [.bnPerChannelF g b eps oc h w]
  | .bnPerChannelBack g x eps oc h w e => toToks e ++ [.bnPerChannelBack g x eps oc h w]
  | .bnPerChannelEvalF g b mu var eps oc h w e => toToks e ++ [.bnPerChannelEvalF g b mu var eps oc h w]
  | .weightGrad x m n e => toToks e ++ [.weightGrad x m n]
  | .biasGrad n e => toToks e ++ [.biasGrad n]
  | .convWeightGrad x ic oc h w' kH kW e => toToks e ++ [.convWeightGrad x ic oc h w' kH kW]
  | .convBiasGrad ic oc h w' kH kW e => toToks e ++ [.convBiasGrad ic oc h w' kH kW]
  | .convStridedWeightGrad x ic oc h w' kH kW e => toToks e ++ [.convStridedWeightGrad x ic oc h w' kH kW]
  | .bnGammaGrad v eps oc h w' e => toToks e ++ [.bnGammaGrad v eps oc h w']
  | .bnBetaGrad oc h w' e => toToks e ++ [.bnBetaGrad oc h w']
  | .adamMNextF m b1 ob1 ds e => toToks e ++ [.adamMNextF m b1 ob1 ds]
  | .adamVNextF v b2 ob2 ds e => toToks e ++ [.adamVNextF v b2 ob2 ds]
  | .adamWParamF θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd ds e =>
      toToks e ++ [.adamWParamF θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd ds]
  | .sgdParamF θ lr ds e => toToks e ++ [.sgdParamF θ lr ds]
  | .momVNextF v mu ds e => toToks e ++ [.momVNextF v mu ds]
  | .momParamF θ v mu lr ds e => toToks e ++ [.momParamF θ v mu lr ds]
  | .rmsBufNextF sq buf rho orho mu eps ds e =>
      toToks e ++ [.rmsBufNextF sq buf rho orho mu eps ds]
  -- ⚠ Both push LEFT then RIGHT, so `parseStack` pops right-then-left (`.addV`'s shape). The
  -- LEFT child is the scalar in both cases — the accumulator, and the summed global total.
  | .gradSumSqAccF ds acc e  => toToks acc ++ toToks e ++ [.gradSumSqAccF ds]
  | .clipScaleF cS eS ds s e => toToks s ++ toToks e ++ [.clipScaleF cS eS ds]
  | .lambDirF a b c d e' f g h i j k ds x =>
      toToks x ++ [.lambDirF a b c d e' f g h i j k ds]
  | .lambScaleF ds s e       => toToks s ++ toToks e ++ [.lambScaleF ds]
  | .depthwiseF w b c h w' kH kW e => toToks e ++ [.depthwiseF w b c h w' kH kW]
  | .depthwiseBack w c h w' kH kW e => toToks e ++ [.depthwiseBack w c h w' kH kW]
  | .depthwiseStridedF w b c h w' kH kW e => toToks e ++ [.depthwiseStridedF w b c h w' kH kW]
  | .depthwiseStridedXlaF w b c h w' kH kW e => toToks e ++ [.depthwiseStridedXlaF w b c h w' kH kW]
  | .depthwiseStridedBack w c h w' kH kW e => toToks e ++ [.depthwiseStridedBack w c h w' kH kW]
  | .depthwiseStridedXlaBack w c h w' kH kW e => toToks e ++ [.depthwiseStridedXlaBack w c h w' kH kW]
  | .swishF n e      => toToks e ++ [.swishF n]
  | .swishBack x n e => toToks e ++ [.swishBack x n]
  | .sigmoidF n e    => toToks e ++ [.sigmoidF n]
  | .sigmoidBack x n e => toToks e ++ [.sigmoidBack x n]
  | .geluF n e       => toToks e ++ [.geluF n]
  | .geluBack x n e  => toToks e ++ [.geluBack x n]
  | .layerScaleF γN n e => toToks e ++ [.layerScaleF γN n]
  | .layerScaleChF γN c h w e => toToks e ++ [.layerScaleChF γN c h w]
  | .softmaxRowF m n e    => toToks e ++ [.softmaxRowF m n]
  | .softmaxRowBack x m n e => toToks e ++ [.softmaxRowBack x m n]
  | .matmulF m k n a b    => toToks a ++ toToks b ++ [.matmulF m k n]
  | .transposeF m n e     => toToks e ++ [.transposeF m n]
  | .scaleF s n e         => toToks e ++ [.scaleF s n]
  | .lnRowF g b eps m n e => toToks e ++ [.lnRowF g b eps m n]
  | .lnRowBack g x eps m n e => toToks e ++ [.lnRowBack g x eps m n]
  | .denseRowF w b N a c e => toToks e ++ [.denseRowF w b N a c]
  | .denseRowBack w N a c e => toToks e ++ [.denseRowBack w N a c]
  | .patchEmbedF w b cls pos ic H W P N D e => toToks e ++ [.patchEmbedF w b cls pos ic H W P N D]
  | .clsSliceF N D e      => toToks e ++ [.clsSliceF N D]
  | .clsPadF N D e        => toToks e ++ [.clsPadF N D]
  | .headSliceF N heads d hIdx e => toToks e ++ [.headSliceF N heads d hIdx]
  | .headPadF N heads d hIdx e   => toToks e ++ [.headPadF N heads d hIdx]
  | .rowScaleF g m n e    => toToks e ++ [.rowScaleF g m n]
  | .rowBiasF b m n e     => toToks e ++ [.rowBiasF b m n]
  | .batched tag names info e   => toToks e ++ [.batched tag names info]
  | .batched2 tag names info a b => toToks a ++ toToks b ++ [.batched2 tag names info]
  | .allReduceMean R t ds e => toToks e ++ [.allReduceMean R t ds]

/-- **Stride-2 weight-gradient window geometry — odd AND even kernels.** Returns
    `(up, ext, lo, hi)` for one spatial axis: `up` is the trailing zero row of the
    decimate-backward upsample, `ext` the resulting cotangent extent, and `[lo, hi]` the
    correlation's padding.

    The strided weight grad zero-upsamples the cotangent onto the stride-1 grid (the `decimate`
    backward) and then correlates the saved input against it VALID-style, so the result is `kH×kW`.
    For an **odd** kernel the upsample carries a trailing zero row (`up = 1`, extent `2s`) and the
    correlation pads symmetrically by `p = (k−1)/2`. That is the committed spelling for the 1×1, 3×3
    and 7×7 kernels every other net uses, and it is reproduced here **byte-identically** — the odd
    branch is exactly the old inline formula.

    For an **even** kernel `p = (k−1)/2` floors to `k/2 − 1`, so a symmetric pad emits a result one
    short of `k`. Measured at `k = 2`, input 8×8 → output 4×4: it declared `2x3x2x2` against a
    convolution yielding `1x1` — **type-invalid MLIR**, which is why ConvNeXt's 2×2/s2 downsample
    weight grad was hand-written in `ConvNeXtRender.lean` rather than using this op. The fix is
    the same asymmetry `convStridedBack` already applies on the input-VJP: drop the trailing zero
    row (extent `2s−1`) and pad `[p, k−2−p]`.

    Output width is `ext_padded − ext + 1 = (2s + lo + hi) − ext + 1`:
    `k=1 → (1, 2s, 0, 0) ⇒ 1`; `k=3 → (1, 2s, 1, 1) ⇒ 3`; `k=7 → (1, 2s, 3, 3) ⇒ 7`;
    `k=2 → (0, 2s−1, 0, 0) ⇒ 2`; `k=4 → (0, 2s−1, 1, 1) ⇒ 4`. -/
private def sWGradGeom (k s : Nat) : Nat × Nat × Nat × Nat :=
  let p := (k - 1) / 2
  if k % 2 == 1 then (1, 2 * s, p, p) else (0, 2 * s - 1, p, k - 2 - p)

-- ════════════════════════════════════════════════════════════════
-- § Pointwise ops at their 4-D shape — the NHWC↔NCHW relayout fix
--
-- ⚠⚠ WHY THIS EXISTS, and why it is worth 2.3× on EfficientNet-B0.
--
-- The proof IR carries activations as flat `[B, c*h*w]` vectors, so every 4-D op brackets
-- itself with `reshape` glue and every pointwise op is emitted at the flat type. A flatten is
-- a free bitcast ONLY in NCHW layout — and XLA runs bf16 convolutions in **NHWC**, for the
-- tensor cores. So a pointwise op emitted flat pins its tensor to NCHW between two NHWC convs
-- and XLA materialises a physical relayout going in and coming out.
--
-- Measured 2026-08-29, one 4060 Ti, node-granularity nsys: B0 bf16 @64 spent **72.61 ms/step,
-- 54.6% of all GPU time**, in those relayouts (10.486 GB) against its JAX reference's 0.75 ms.
-- ConvNeXt-T 53.2%. MobileNetV2 and ViT, which end up with no relayouts, are FASTER than their
-- references. ⚠ f32 is immune: XLA keeps f32 convs in NCHW, so the effect cannot appear there
-- and an f32 A/B reads as a clean null — which is exactly how it stayed hidden.
--
-- The fix is to emit the pointwise ops at the 4-D shape. `liftPointwise` brackets a block with
-- an inverse reshape pair, so the value crossing the token boundary keeps its flat type and no
-- other `emitTok` arm changes; XLA then folds each pair against the neighbouring conv's own
-- reshape and the chain is 4-D end to end.
-- ════════════════════════════════════════════════════════════════

/-- The full entry — `[c,h,w]` plus the row-view flag — recorded for SSA name `nm`. -/
def lookupEntry (tbl : ShapeTbl) (nm : String) : Option (Nat × Nat × Nat × Bool) :=
  tbl.lookup nm

/-- The `[c,h,w]` `nm` carries **as a map**. A row view answers `none`: it holds the same elements
    in a different order, so unflattening it to `[B,c,h,w]` would not be an inverse pair. -/
def lookupShape (tbl : ShapeTbl) (nm : String) : Option (Nat × Nat × Nat) :=
  match lookupEntry tbl nm with
  | some (c, h, w, false) => some (c, h, w)
  | _                     => none

/-- The `[c,h,w]` an op's INPUT and OUTPUT activations carry, when it has them. Strided and
    pooled tags carry their **output** spatial dims, so the input side is `2h × 2w`.

    ⚠⚠ Each batched tag and its per-example peer MUST answer IDENTICALLY. The two renders are tied
    byte for byte ([`tests/TestBatchedEmitTie.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestBatchedEmitTie.lean), `convnext-fwd-b-tie`), so a shape one path
    knows and the other does not is a `liftPointwise` that fires on one side only — a tie failure
    with no wrong answer anywhere to point at. Add tags in pairs. -/
private def tokIO : Tok → Option (Nat × Nat × Nat) × Option (Nat × Nat × Nat)
  | .batched tag _ info =>
      match tag, info with
      | "conv",                   [_, ic, oc, h, w, _, _]
      | "convBf16",               [_, ic, oc, h, w, _, _]
      | "convF8",                 [_, ic, oc, h, w, _, _] => (some (ic, h, w), some (oc, h, w))
      | "convStrided",            [_, ic, oc, h, w, _, _]
      | "convStridedBf16",        [_, ic, oc, h, w, _, _]
      | "convStridedXla",         [_, ic, oc, h, w, _, _]
      | "convStridedXlaBf16",     [_, ic, oc, h, w, _, _] => (some (ic, 2*h, 2*w), some (oc, h, w))
      | "depthwise",              [_, c, h, w, _, _]
      | "depthwiseBf16",          [_, c, h, w, _, _]      => (some (c, h, w), some (c, h, w))
      | "depthwiseStrided",       [_, c, h, w, _, _]
      | "depthwiseStridedBf16",   [_, c, h, w, _, _]
      | "depthwiseStridedXla",    [_, c, h, w, _, _]
      | "depthwiseStridedXlaBf16",[_, c, h, w, _, _]      => (some (c, 2*h, 2*w), some (c, h, w))
      | "bnBatch",                [_, oc, h, w]
      | "bnEval",                 [_, oc, h, w]           => (some (oc, h, w), some (oc, h, w))
      | "layerScaleChP",          [_, c, h, w]            => (some (c, h, w), some (c, h, w))
      -- ⚠ `gap` CONTRACTS the spatial extent: its result is `[c]`, so it has an input shape and
      -- no output one. The width table conflated the two and could hand a `[c]` value a `[c,h,w]`.
      | "gap",                    [_, c, h, w]            => (some (c, h, w), none)
      | "maxPool",                [_, c, h, w]
      | "maxPool3s2",             [_, c, h, w]            => (some (c, 2*h, 2*w), some (c, h, w))
      | "seBlock",                [_, c, h, w, _]         => (some (c, h, w), some (c, h, w))
      | "convStride4P",           [_, ic, oc, h, w, _, _]
      | "convStride4PBf16",       [_, ic, oc, h, w, _, _] =>
          (some (ic, 2*(2*h), 2*(2*w)), some (oc, h, w))
      -- ── the BACKWARD half. ⚠⚠ Omitting it is not a smaller version of this table, it is a
      --    BROKEN one: a cotangent chain whose head has no shape drops back to flat at the first
      --    `addV`, and every pointwise op after it in that chain goes with it. The width table
      --    covered the backward by accident — one entry served the forward activation and the
      --    cotangent alike, because they have the same width — and losing that is what took the
      --    first name-keyed cut to 0.999 GB where the scratch pass reached 0.122.
      | "convBackBatched",        [_, ic, oc, h, w, _, _]
      | "convBackBatchedBf16",    [_, ic, oc, h, w, _, _]
      | "convBackBatchedF8",      [_, ic, oc, h, w, _, _] => (some (oc, h, w), some (ic, h, w))
      | "convStridedBackBatched", [_, ic, oc, h, w, _, _]
      | "convStridedBackBatchedBf16", [_, ic, oc, h, w, _, _] =>
          (some (oc, h, w), some (ic, 2*h, 2*w))
      | "depthwiseBackBatched",   [_, c, h, w, _, _]
      | "depthwiseBackBatchedBf16", [_, c, h, w, _, _]    => (some (c, h, w), some (c, h, w))
      | "depthwiseStridedBackBatched", [_, c, h, w, _, _]
      | "depthwiseStridedBackBatchedBf16", [_, c, h, w, _, _]
      | "depthwiseStridedXlaBackBatched", [_, c, h, w, _, _]
      | "depthwiseStridedXlaBackBatchedBf16", [_, c, h, w, _, _]
      | "maxPoolBackP",           [_, c, h, w]
      | "maxPool3s2BackP",        [_, c, h, w]            => (some (c, h, w), some (c, 2*h, 2*w))
      -- ⚠ the per-example `maxPool3s2BackP` Raw carries THREE nats, not four (`skel` aliases the
      -- batched tag but not its `N`), so the batched pattern above cannot match it.
      | "maxPool3s2BackP",        [c, h, w]               => (some (c, h, w), some (c, 2*h, 2*w))
      | "bnBatchBack",            [_, oc, h, w]
      | "bnBatchLABack",          [_, oc, h, w]           => (some (oc, h, w), some (oc, h, w))
      | "seBackBatched",          [_, c, h, w, _]         => (some (c, h, w), some (c, h, w))
      -- GAP contracts and its adjoint expands, so each has a shape on ONE side only.
      | "gapBackBatched",         [_, c, h, w]            => (none, some (c, h, w))
      | _, _                                              => (none, none)
  -- ── the per-example peers of exactly the tags above ──
  | .flatConvF _ _ ic oc h w _ _
  | .flatConvFBf16 _ _ ic oc h w _ _          => (some (ic, h, w), some (oc, h, w))
  | .flatConvStridedF _ _ ic oc h w _ _
  | .flatConvStridedXlaF _ _ ic oc h w _ _    => (some (ic, 2*h, 2*w), some (oc, h, w))
  | .depthwiseF _ _ c h w _ _                 => (some (c, h, w), some (c, h, w))
  | .depthwiseStridedF _ _ c h w _ _
  | .depthwiseStridedXlaF _ _ c h w _ _       => (some (c, 2*h, 2*w), some (c, h, w))
  | .bnPerChannelF _ _ _ oc h w
  | .bnPerChannelEvalF _ _ _ _ _ oc h w       => (some (oc, h, w), some (oc, h, w))
  | .layerScaleChF _ c h w                    => (some (c, h, w), some (c, h, w))
  | .gapF c h w                               => (some (c, h, w), none)
  | .maxPoolF c h w                           => (some (c, 2*h, 2*w), some (c, h, w))
  | .flatConvStride4F _ _ ic oc h w _ _       => (some (ic, 2*(2*h), 2*(2*w)), some (oc, h, w))
  -- ── their backward peers, pair for pair with the batched tags above ──
  | .convBack _ ic oc h w _ _                 => (some (oc, h, w), some (ic, h, w))
  | .convStridedBack _ ic oc h w _ _          => (some (oc, h, w), some (ic, 2*h, 2*w))
  | .depthwiseBack _ c h w _ _                => (some (c, h, w), some (c, h, w))
  | .depthwiseStridedBack _ c h w _ _
  | .depthwiseStridedXlaBack _ c h w _ _
  | .maxPoolBack _ c h w                      => (some (c, h, w), some (c, 2*h, 2*w))
  | .bnPerChannelBack _ _ _ oc h w            => (some (oc, h, w), some (oc, h, w))
  | .gapBack c h w                            => (none, some (c, h, w))
  | .broadcastBack c h w                      => (some (c, h, w), none)
  | _                                         => (none, none)

/-- Record `nm ↦ [c,h,w]` + layout, newest first. `fresh` never reuses a name, so an entry can
    never be contradicted by a later one; an `.operand` name re-pushed in a later fragment repeats. -/
def noteEntry (nm : String) : Option (Nat × Nat × Nat × Bool) → StateM EmitS Unit
  | none               => pure ()
  | some (c, h, w, rv) => modify fun (k, tbl) => (k, (nm, c, h, w, rv) :: tbl)

/-- Record `nm` as carrying the `[c,h,w]` MAP (not a row view). -/
def noteShapeOf (nm : String) : Option (Nat × Nat × Nat) → StateM EmitS Unit
  | none           => pure ()
  | some (c, h, w) => noteEntry nm (some (c, h, w, false))

/-- The `[c,h,w]` the running table has for the value bound to `nm`, as a map. -/
def lookupShapeM (nm : String) : StateM EmitS (Option (Nat × Nat × Nat)) := do
  let (_, tbl) ← get
  pure (lookupShape tbl nm)

/-- The full entry the running table has for `nm`. -/
def lookupEntryM (nm : String) : StateM EmitS (Option (Nat × Nat × Nat × Bool)) := do
  let (_, tbl) ← get
  pure (lookupEntry tbl nm)

/-- The ops that keep a value in its `[h·w, c]` ROW VIEW: ConvNeXt's channel-LN chain, which
    normalises over the transposed layout and hands the result back to the closing transpose. -/
private def rowViewPass (t : Tok) : Bool :=
  match t with
  | .batched tag _ _ =>
      tag == "lnRowP" || tag == "rowScaleP" || tag == "rowBiasP" || tag == "lnRowBackP"
  | .lnRowF _ _ _ _ _ | .lnRowBack _ _ _ _ _ | .rowScaleF _ _ _ | .rowBiasF _ _ _ => true
  | _ => false

/-- The `(m, n)` of a transpose token, batched or per-example. -/
private def transposeMN (t : Tok) : Option (Nat × Nat) :=
  match t with
  | .batched "transposeP" _ [_, m, n] => some (m, n)
  | .transposeF m n                   => some (m, n)
  | _                                 => none

/-- Record what one token's operand and result carry, given the operand-name stack before and
    after it was emitted. Called from `serializeToks`, so no `emitTok` arm has to know about the
    table — which is what keeps the 94 arms free of it.

    ⭐ Three cases, and the first two exist only for the channel-LN round trip: a transpose FLIPS
    the layout flag when its `(m,n)` match the operand's `[c,h,w]` (and records nothing when they
    do not, e.g. ViT's attention transposes, which are not maps at all), and the row ops carry it
    through unchanged. Everything else reads its shapes off the tag. -/
def noteTokShapes (t : Tok) (before after : List String) : StateM EmitS Unit := do
  if rowViewPass t then
    match before, after with
    | i :: _, o :: _ => noteEntry o (← lookupEntryM i)
    | _, _           => pure ()
  else match transposeMN t with
  | some (m, n) =>
      match before, after with
      | i :: _, o :: _ =>
          match ← lookupEntryM i with
          | some (c, h, w, false) =>
              if c == m && h * w == n then noteEntry o (some (c, h, w, true)) else pure ()
          | some (c, h, w, true)  =>
              if h * w == m && c == n then noteEntry o (some (c, h, w, false)) else pure ()
          | none                  => pure ()
      | _, _ => pure ()
  | none => do
      let (i, o) := tokIO t
      match before with
      | r :: _ => noteShapeOf r i
      | _      => pure ()
      match after with
      | r :: _ => noteShapeOf r o
      | _      => pure ()

/-- Render a pointwise block at its 4-D shape when the OPERAND's producer recorded one.
    `k` receives the (possibly unflattened) input name and the dims to type its ops with, and
    returns `(text, result name)`.

    ⚠ The `c*h*w == n` guard is what keeps a mismatched entry from emitting an ill-typed reshape
    rather than merely a suboptimal one. It cannot fire today — an entry is written by the token
    that produced the name — and it is the difference between a missed optimisation and a render
    that does not parse, so it stays.

    ⭐ The block's own RESULT is recorded too, which is what lets a pointwise CHAIN stay 4-D: the
    value crossing the token boundary keeps its flat type, so without this the second op in a
    swish→multiply→add chain would find nothing for its operand and drop back to flat. -/
def liftPointwise (B n : Nat) (r : String)
    (k : String → List Nat → StateM EmitS (String × String)) : StateM EmitS (String × String) := do
  match ← lookupShapeM r with
  | some (c, h, w) =>
      if c * h * w == n then do
        let xi ← fresh
        let (body, res) ← k xi [B, c, h, w]
        let o ← fresh
        noteShapeOf o (some (c, h, w))
        pure (s!"    {xi} = stablehlo.reshape {r} : ({ty [B,n]}) -> {ty [B,c,h,w]}\n" ++ body ++
              s!"    {o} = stablehlo.reshape {res} : ({ty [B,c,h,w]}) -> {ty [B,n]}\n", o)
      else k r [B, n]
  | none => k r [B, n]

/-- Two-tensor-operand peer of `liftPointwise`; both operands carry the same flat width.
    The shape comes from whichever operand has one — the cotangent first, since it is the stack
    operand and was produced nearby, then the saved activation. -/
def liftPointwise2 (B n : Nat) (r s : String)
    (k : String → String → List Nat → StateM EmitS (String × String)) :
    StateM EmitS (String × String) := do
  let sh ← match ← lookupShapeM r with
           | some p => pure (some p)
           | none   => lookupShapeM s
  match sh with
  | some (c, h, w) =>
      if c * h * w == n then do
        let xi ← fresh; let yi ← fresh
        let (body, res) ← k xi yi [B, c, h, w]
        let o ← fresh
        noteShapeOf o (some (c, h, w))
        pure (s!"    {xi} = stablehlo.reshape {r} : ({ty [B,n]}) -> {ty [B,c,h,w]}\n" ++
              s!"    {yi} = stablehlo.reshape {s} : ({ty [B,n]}) -> {ty [B,c,h,w]}\n" ++ body ++
              s!"    {o} = stablehlo.reshape {res} : ({ty [B,c,h,w]}) -> {ty [B,n]}\n", o)
      else k r s [B, n]
  | none => k r s [B, n]

/-- **The text of the cross-replica mean** — `ViTRender.emitGradAllReduce`'s body, verbatim, so
    that the `allReduceMean` token re-renders every committed `*dp*` artifact byte-identically.
    `all_reduce(add)` over `replica_groups = [[0..R-1]]`, then a divide by `R`; the names are
    `%arsum{t}` … `%armean{t}` from the tag rather than `fresh`. At `R ≤ 1` there is no text and
    the operand's name is the result, exactly as the text function did. -/
def allReduceMeanText (g : String) (ds : List Nat) (t : String) (R : Nat) : String × String :=
  if R ≤ 1 then ("", g) else
  let T := ty ds
  let grp := String.intercalate ", " ((List.range R).map toString)
  let lbrace := "{"
  let rbrace := "}"
  let s :=
    s!"    %arsum{t} = \"stablehlo.all_reduce\"({g}) ({lbrace}\n" ++
    s!"    ^bb0(%ara{t}: tensor<f32>, %arb{t}: tensor<f32>):\n" ++
    s!"      %aradd{t} = stablehlo.add %ara{t}, %arb{t} : tensor<f32>\n" ++
    s!"      stablehlo.return %aradd{t} : tensor<f32>\n" ++
    s!"    {rbrace}) {lbrace} replica_groups = dense<[[{grp}]]> : tensor<1x{R}xi64> {rbrace} : ({T}) -> {T}\n" ++
    s!"    %arn{t} = stablehlo.constant dense<{R}.0> : {T}\n" ++
    s!"    %armean{t} = stablehlo.divide %arsum{t}, %arn{t} : {T}\n"
  (s, s!"%armean{t}")

/-- The compute precision a contraction tag asks for: `…Bf16` bf16, `…F8` fp8 (E4M3), otherwise f32
    (`none`) — as the type printer of the low-precision operands. -/
def lowOf (tag : String) : Option (List Nat → String) :=
  if tag.endsWith "Bf16" then some tyBf16 else if tag.endsWith "F8" then some tyF8 else none

/-- **One contraction at a compute precision** — the `stablehlo.convolution` / `dot_general` line of
    an emit arm, `op lhs rhs` being its text between `=` and the type signature. `lp = none` is the
    f32 op. `lp = some t` converts both operands to `t` and types the op in `t`; with `lowResult`
    the result is `t` too and is converted back to f32 — the bf16 / fp8 shape (an f32-typed conv
    result compiles to pure f32, `flatConvFBf16`) — and without it the result stays f32, the dot
    shape whose f32 result IS the accumulator (`dotInBf16`). Names are drawn in that order (the two
    converts, the op, the convert back), so an arm draws the same `%v` numbers at every precision.
    Returns the text and the f32 result's name. -/
def emitContract (lp : Option (List Nat → String)) (x y : String) (xs ys rs : List Nat)
    (op : String → String → String) (lowResult : Bool := true) : StateM EmitS (String × String) := do
  match lp with
  | none =>
    let o ← fresh
    pure (s!"    {o} = {op x y} : ({ty xs}, {ty ys}) -> {ty rs}\n", o)
  | some t =>
    let xb ← fresh; let yb ← fresh; let o ← fresh
    let cvt := s!"    {xb} = stablehlo.convert {x} : ({ty xs}) -> {t xs}\n" ++
               s!"    {yb} = stablehlo.convert {y} : ({ty ys}) -> {t ys}\n"
    if lowResult then
      let of ← fresh
      pure (cvt ++ s!"    {o} = {op xb yb} : ({t xs}, {t ys}) -> {t rs}\n" ++
              s!"    {of} = stablehlo.convert {o} : ({t rs}) -> {ty rs}\n", of)
    else
      pure (cvt ++ s!"    {o} = {op xb yb} : ({t xs}, {t ys}) -> {ty rs}\n", o)

/-- The input-side dense contraction `x · W` (`dotIn` / `dotInBf16`). -/
def dotInOp (lhs rhs : String) : String :=
  s!"stablehlo.dot_general {lhs}, {rhs}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]"

/-- The flat-carrier SAME conv + bias (`flatConvF` / `flatConvFBf16`): reshape the `[B, ic·h·w]`
    carrier to NCHW, convolve at `lp`'s precision (`emitContract`), add the broadcast bias, flatten. -/
def emitFlatConv (B : Nat) (lp : Option (List Nat → String)) (w b : String)
    (ic oc h w' kH kW : Nat) (r : String) : StateM EmitS (String × String) := do
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let xn ← fresh
  let (cs, cv) ← emitContract lp xn w [B,ic,h,w'] [oc,ic,kH,kW] [B,oc,h,w'] fun lhs rhs =>
      s!"stablehlo.convolution({lhs}, {rhs})\n" ++
      "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
      s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
      "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let bb ← fresh; let ob ← fresh; let o ← fresh
  pure (
    s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*h*w']}) -> {ty [B,ic,h,w']}\n" ++ cs ++
    s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
    s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
    s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o)

/-- The flattened batched matrix multiply `C = A·B` (`matmulF` / `"matmulFBf16"`): reshape both
    operands to rank 3, `dot_general` with batching dim 0 (A's last axis against B's middle) at
    `lp`'s precision, reshape back to flat. -/
def emitMatmul (B : Nat) (lp : Option (List Nat → String)) (a b : String) (m k n : Nat) :
    StateM EmitS (String × String) := do
  let an ← fresh; let bn ← fresh
  let (cs, mm) ← emitContract lp an bn [B,m,k] [B,k,n] [B,m,n] fun lhs rhs =>
      s!"stablehlo.dot_general {lhs}, {rhs}, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT]"
  let o ← fresh
  pure (s!"    {an} = stablehlo.reshape {a} : ({ty [B, m*k]}) -> {ty [B,m,k]}\n" ++
    s!"    {bn} = stablehlo.reshape {b} : ({ty [B, k*n]}) -> {ty [B,k,n]}\n" ++ cs ++
    s!"    {o} = stablehlo.reshape {mm} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o)

/-- **The squeeze-excite activations inside one `seBlock` emission**, by name: the squeeze (GAP,
    `[B,c]`), the reduce dense + bias (`[B,r]`), its swish, and the excite dense + bias (`[B,c]`) —
    the four values the SE backward reads. `k0` is the fresh-name counter when the `seBlock` token
    is emitted. The offsets are the order the `"seBlock"` arm of `emitTok` below calls `fresh` in
    (`sq` 5th, `ex` 8th, `a1` 10th, `h2` 13th); the `#guard` after `emitTok` checks each name is
    defined by the op it claims. A renderer that saves these reuses the SE the forward already
    computes instead of emitting a second, un-fused copy. -/
def seBlockSavedNames (k0 : Nat) : String × String × String × String :=
  (s!"%v{k0 + 4}", s!"%v{k0 + 7}", s!"%v{k0 + 9}", s!"%v{k0 + 12}")

-- Compiling this one 99-arm def needs ~2× the default budget (more under `trace.profiler`, which
-- trips 400000); 5× leaves room for new arms. Nothing else in the file needs a bump.
set_option maxHeartbeats 1000000 in
/-- Render one token: pop its operands' result-names off the stack, emit its
    StableHLO line(s), push its fresh result name. The per-op StableHLO *syntax*
    here is the audited lexical boundary (validated by `iree-compile` + GPU run);
    the *structure* it consumes is the proven-faithful token stream. -/
def emitTok (B : Nat) : Tok → List String → StateM EmitS (String × List String)
  | .operand nm _, st => pure ("", nm :: st)
  | .dotIn w m n, r :: st => do
      let (s, o) ← emitContract none r w [B,m] [m,n] [B,n] dotInOp
      pure (s, o :: st)
  -- The ONLY emit shape that reaches tensor cores: both operands bf16, result f32.
  -- The f32 result type IS the "fp32 accumulate" — it is not a convert of a bf16 product.
  | .dotInBf16 w m n, r :: st => do
      let (s, o) ← emitContract (some tyBf16) r w [B,m] [m,n] [B,n] dotInOp (lowResult := false)
      pure (s, o :: st)
  | .dotOut w m n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.dot_general {r}, {w}, contracting_dims = [1] x [1], " ++
            s!"precision = [DEFAULT, DEFAULT] : ({ty [B,n]}, {ty [m,n]}) -> {ty [B,m]}\n", o :: st)
  | .addBcast b n, r :: st => do
      let bb ← fresh; let o ← fresh
      pure (s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [n]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.add {r}, {bb} : {ty [B,n]}\n", o :: st)
  | .expe n, r :: st => do
      let (txt4, res4) ← liftPointwise B n r fun r d => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.exponential {r} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .softmaxDiv n, r :: st => do
      let z ← fresh; let s ← fresh; let sb ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {s} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.divide {r}, {sb} : {ty [B,n]}\n", o :: st)
  | .sub n, b :: a :: st => do
      let (txt4, res4) ← liftPointwise2 B n a b fun a b d => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.subtract {a}, {b} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .weightSgd xN wN lrS m n, r :: st => do
      let dW ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (s!"    {dW} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [B,n]}) -> {ty [m,n]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [m,n]}\n" ++
            s!"    {sW} = stablehlo.multiply {dW}, {lW} : {ty [m,n]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [m,n]}\n", o :: st)
  | .biasSgd bN lrS n, r :: st => do
      let z ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dB} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,n]}, tensor<f32>) -> {ty [n]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [n]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [n]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [n]}\n", o :: st)
  | .convWeightSgd xN wN lrS ic oc h w kH kW, r :: st => do
      -- conv weight grad (transpose trick) then SGD: reshape flat acts/cotangent to
      -- 4-D, transpose batch↔feature, convolve (batch as contraction), transpose back
      -- to [oc,ic,kH,kW], then θ' = θ − lr·dW. Same op text as `CnnRender.convWGrad`+`sgd`.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
        s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
  | .convBiasSgd bN lrS oc h w, r :: st => do
      -- conv bias grad (reduce over batch+spatial [0,2,3]) then SGD. Same op text as
      -- `CnnRender.convBiasGrad`+`sgd`.
      let dr ← fresh; let z ← fresh; let g ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {g} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sB} = stablehlo.multiply {g}, {lB} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [oc]}\n", o :: st)
  | .bnGammaSgd gN vN epsStr lrS oc h w, r :: st => do
      -- BN per-channel γ grad: recompute x̂ from the saved conv output {vN} (reduce μ/var
      -- over spatial [2,3]), dγ_c = Σ_{b,h,w} dy·x̂, then SGD.
      let z ← fresh; let xr ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let dyr ← fresh; let p ← fresh; let dg ← fresh
      let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {p} = stablehlo.multiply {dyr}, {xhat} : {ty [B,oc,h,w]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : {ty [oc]}\n", o :: st)
  | .bnBetaSgd bN lrS oc h w, r :: st => do
      -- BN per-channel β grad: dβ_c = Σ_{b,h,w} dy, then SGD (β grad needs no x̂).
      let z ← fresh; let dyr ← fresh; let db ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {db} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sB} = stablehlo.multiply {db}, {lB} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [oc]}\n", o :: st)
  | .layerScaleChGammaSgd gN xN lrS c h w, r :: st => do
      -- per-channel layer-scale γ grad: dγ_c = reduce[0,2,3](x⊙dy), then SGD (`lsGradCh` + wrap).
      let z ← fresh; let xr ← fresh; let dr ← fresh; let p ← fresh
      let dg ← fresh; let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {p} = stablehlo.multiply {xr}, {dr} : {ty [B,c,h,w]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [c]}\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : {ty [c]}\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : {ty [c]}\n", o :: st)
  | .lnGammaSgd gN xN epsStr lrS n, r :: st => do
      -- scalar-LN γ grad: recompute x̂ from the saved LN input {xN} (μ/var over [1]),
      -- dγ = Σ_{b,k} dy·x̂ → tensor<f32>, reshape to the Vec-1 param, SGD (`lnParamGrad` dγ half + wrap).
      let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
      let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
      let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
      let dg ← fresh; let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xN} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xN}, {mu} : {ty [B,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
        s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {p} = stablehlo.multiply {r}, {xh} : {ty [B,n]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : tensor<f32>\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : tensor<f32>\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : tensor<f32>\n", o :: st)
  | .lnBetaSgd bN lrS n, r :: st => do
      -- scalar-LN β grad: dβ = Σ_{b,k} dy → tensor<f32> (rank-0, matches the scalar-LN bnF param), SGD.
      let z ← fresh; let db ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {db} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : tensor<f32>\n" ++
        s!"    {sB} = stablehlo.multiply {db}, {lB} : tensor<f32>\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : tensor<f32>\n", o :: st)
  | .veclnGammaSgd gN xN epsStr lrS N D, r :: st => do
      -- vector-[D] LN γ grad: recompute x̂ from the saved LN input {xN} (μ/var over [2], per token),
      -- dγ = Σ_{b,n} dy·x̂ → tensor<Dxf32> (reduce over [0,1], KEEP D), SGD (`lnParamGrad` dγ half + wrap).
      -- {xN}/{r} arrive flat [B,N*D] (the SHlo thread convention) → reshape both to [B,N,D] first.
      let x3 ← fresh; let d3 ← fresh
      let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
      let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
      let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
      let dg ← fresh; let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {x3} = stablehlo.reshape {xN} : ({ty [B, N*D]}) -> {ty [B,N,D]}\n" ++
        s!"    {d3} = stablehlo.reshape {r} : ({ty [B, N*D]}) -> {ty [B,N,D]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{D}.0> : {ty [B,N,D]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,N,D]}\n" ++
        s!"    {smr} = stablehlo.reduce({x3} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,N,D]}, tensor<f32>) -> {ty [B,N]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,N]}) -> {ty [B,N,D]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,N,D]}\n" ++
        s!"    {xc} = stablehlo.subtract {x3}, {mu} : {ty [B,N,D]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,N,D]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,N,D]}, tensor<f32>) -> {ty [B,N]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,N]}) -> {ty [B,N,D]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,N,D]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,N,D]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,N,D]}\n" ++
        s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,N,D]}\n" ++
        s!"    {p} = stablehlo.multiply {d3}, {xh} : {ty [B,N,D]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,D]}, tensor<f32>) -> {ty [D]}\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : {ty [D]}\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : {ty [D]}\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : {ty [D]}\n", o :: st)
  | .patchEmbedWeightSgd wN xN lrS ic H W P N D, r :: st => do
      -- patch-embed (16×16/s16) conv WEIGHT grad: slice patch tokens [1..N] from the embed cotangent
      -- {r} [B,(N+1)*D] (drop CLS row 0), reshape→[B,ph,pw,D]→transpose→[B,D,ph,pw], dilate interior P-1,
      -- valid conv with the saved image {xN} [B,ic,H,W] → dW [D,ic,P,P], SGD: W − lr·dW.
      let ph := H / P; let pw := W / P
      let dilH := H - (P - 1); let dilW := W - (P - 1)
      let zc ← fresh; let dtr ← fresh; let dsl ← fresh; let drs ← fresh; let dy3 ← fresh
      let u ← fresh; let xt ← fresh; let dt ← fresh; let raw ← fresh; let dw ← fresh
      let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {zc} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {dtr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
        s!"    {dsl} = stablehlo.slice {dtr} [0:{B}, 1:{N+1}, 0:{D}] : ({ty [B,N+1,D]}) -> {ty [B,N,D]}\n" ++
        s!"    {drs} = stablehlo.reshape {dsl} : ({ty [B,N,D]}) -> {ty [B,ph,pw,D]}\n" ++
        s!"    {dy3} = stablehlo.transpose {drs}, dims = [0, 3, 1, 2] : ({ty [B,ph,pw,D]}) -> {ty [B,D,ph,pw]}\n" ++
        s!"    {u} = stablehlo.pad {dy3}, {zc}, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, {P-1}, {P-1}] : ({ty [B,D,ph,pw]}, tensor<f32>) -> {ty [B,D,dilH,dilW]}\n" ++
        s!"    {xt} = stablehlo.transpose {xN}, dims = [1, 0, 2, 3] : ({ty [B,ic,H,W]}) -> {ty [ic,B,H,W]}\n" ++
        s!"    {dt} = stablehlo.transpose {u}, dims = [1, 0, 2, 3] : ({ty [B,D,dilH,dilW]}) -> {ty [D,B,dilH,dilW]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,H,W]}, {ty [D,B,dilH,dilW]}) -> {ty [ic,D,P,P]}\n" ++
        s!"    {dw} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,D,P,P]}) -> {ty [D,ic,P,P]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [D,ic,P,P]}\n" ++
        s!"    {sW} = stablehlo.multiply {dw}, {lW} : {ty [D,ic,P,P]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [D,ic,P,P]}\n", o :: st)
  | .reluF n, r :: st => do
      let (txt4, res4) ← liftPointwise B n r fun r d => do
          let z ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty d}\n" ++
                s!"    {o} = stablehlo.maximum {r}, {z} : {ty d}\n", o)
      -- The round node: down to bf16 and straight back. Two converts, not one, because
      -- `den` is `ℝ → ℝ` — the VALUE stays f32 and only its precision is degraded, which
      -- is what "round to bf16" means as a function on reals.
      --
      -- ⚠⚠ **MEASURED 2026-08-01 ON ares: XLA DELETES THIS PAIR, SO THIS EMIT IS A NO-OP
      -- ON HARDWARE.** jax 0.10.2 / CUDA 12.9, `.astype(bf16).astype(f32)` under `jit`:
      -- eager rounds 1.7640524 → 1.765625, but the jitted result is 1.7640524 unchanged and
      -- the optimized HLO contains no `convert` at all — the algebraic simplifier treats the
      -- round trip as removable. So a graph carrying this node computes in FULL f32: no
      -- speedup and, worse, not even the bf16 numerics.
      --
      -- The `den` equation is still correct and the ties built on it still hold — what is
      -- refuted is this EMIT STRATEGY, not the op. To make bf16 real the value has to stay
      -- bf16 ACROSS an operation, i.e. a `dot_general` whose operands are bf16-typed with
      -- `preferred_element_type = f32`. That changes the value's type and so cannot be a
      -- `SHlo n → SHlo n` node; it is the rung-2 emitter change in
      -- planning/archive/bf16_renderer.md. Keep this node — it is the proof-side round and the
      -- depth > 1 ingredient — but do NOT read a graph containing it as running bf16.
      pure (txt4, res4 :: st)
  | .convertF n, r :: st => do
      let b ← fresh; let o ← fresh
      pure (s!"    {b} = stablehlo.convert {r} : ({ty [B,n]}) -> {tyBf16 [B,n]}\n" ++
            s!"    {o} = stablehlo.convert {b} : ({tyBf16 [B,n]}) -> {ty [B,n]}\n", o :: st)
  | .selectPos x n, r :: st => do
      let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
          let z ← fresh; let msk ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty d}\n" ++
            s!"    {msk} = stablehlo.compare GT, {x}, {z} : ({ty d}, {ty d}) -> {tyI1 d}\n" ++
            s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 d}, {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .relu6F n, r :: st => do
      let (txt4, res4) ← liftPointwise B n r fun r d => do
          -- ReLU6 forward: clamp to [0,6] as `min(max(x,0),6)` (matches `relu6`'s def).
          let z ← fresh; let six ← fresh; let mx ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty d}\n" ++
                s!"    {six} = stablehlo.constant dense<6.0> : {ty d}\n" ++
                s!"    {mx} = stablehlo.maximum {r}, {z} : {ty d}\n" ++
                s!"    {o} = stablehlo.minimum {mx}, {six} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .selectMid x n, r :: st => do
      let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
          -- ReLU6 backward mask: route dy where `0 < x < 6`, else 0 (the two-sided kink).
          let z ← fresh; let six ← fresh; let g0 ← fresh; let l6 ← fresh; let msk ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty d}\n" ++
            s!"    {six} = stablehlo.constant dense<6.0> : {ty d}\n" ++
            s!"    {g0} = stablehlo.compare GT, {x}, {z} : ({ty d}, {ty d}) -> {tyI1 d}\n" ++
            s!"    {l6} = stablehlo.compare LT, {x}, {six} : ({ty d}, {ty d}) -> {tyI1 d}\n" ++
            s!"    {msk} = stablehlo.and {g0}, {l6} : {tyI1 d}\n" ++
            s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 d}, {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .flatConvF w b ic oc h w' kH kW, r :: st => do
      let (s, o) ← emitFlatConv B none w b ic oc h w' kH kW r
      pure (s, o :: st)
  -- ⚠ The convolution's RESULT is bf16-typed. An f32-typed result here reads as the same
  -- computation and compiles to pure f32 — see the constructor's note. Do not "simplify".
  | .flatConvFBf16 w b ic oc h w' kH kW, r :: st => do
      let (s, o) ← emitFlatConv B (some tyBf16) w b ic oc h w' kH kW r
      pure (s, o :: st)
  | .maxPoolF c h w, r :: st => do
      let xn ← fresh; let ninf ← fresh; let p ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
        s!"    {p} = \"stablehlo.reduce_window\"({xn}, {ninf}) (" ++ "{\n" ++
        "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
        "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
        "        stablehlo.return %pm : tensor<f32>\n" ++
        "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
        s!" : ({ty [B,c,2*h,2*w]}, tensor<f32>) -> {ty [B,c,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {p} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
  | .convBack w ic oc h w' kH kW, r :: st => do
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let wt ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w']}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {wt} = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {wr} = stablehlo.reverse {wt}, dims = [2, 3] : {ty [ic,oc,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({dn}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,oc,h,w']}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,h,w']}) -> {ty [B, ic*h*w']}\n", o :: st)
  | .maxPoolBack xN c h w, r :: st => do
      let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {scn} = \"stablehlo.select_and_scatter\"({xr}, {dr}, {z}) (" ++ "{\n" ++
        "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
        "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
        "        stablehlo.return %sge : tensor<i1>\n" ++
        "    }, " ++ "{\n" ++
        "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
        "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
        "        stablehlo.return %ss : tensor<f32>\n" ++
        "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
        s!" : ({ty [B,c,2*h,2*w]}, {ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {o} = stablehlo.reshape {scn} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
  | .bnF gN bN epsStr n, r :: st => do
      -- per-example BatchNorm forward `γ·(x−μ)·istd + β` (reduce μ/var over [1])
      let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {r}, {mu} : {ty [B,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.add {gx}, {bb} : {ty [B,n]}\n", o :: st)
  | .bnBack gN xN epsStr n, r :: st => do
      -- BN input-VJP: recompute x̂/istd from saved input {xN}, then the
      -- consolidated three-term `(istd/N)·(N·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))`, dx̂ = γ·dy.
      let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xN} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xN}, {mu} : {ty [B,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {r} : {ty [B,n]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,n]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,n]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,n]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,n]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,n]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.multiply {sN}, {i2} : {ty [B,n]}\n", o :: st)
  | .addV n, b :: a :: st => do
      let (txt4, res4) ← liftPointwise2 B n a b fun a b d => do
          -- residual fan-in: dy of the two operands summed (`F(x) + skip`)
          let o ← fresh
          pure (s!"    {o} = stablehlo.add {a}, {b} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .gapF c h w, r :: st => do
      -- global average pool: reshape to [B,c,h,w], reduce-add over the spatial
      -- axes [2,3], divide by h·w. Denotes `globalAvgPoolFlat` (mean over H×W).
      let xn ← fresh; let z ← fresh; let sm ← fresh; let nf ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {sm} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
        s!"    {o} = stablehlo.divide {sm}, {nf} : {ty [B,c]}\n", o :: st)
  | .gapBack c h w, r :: st => do
      -- GAP backward (VJP): divide the per-channel cotangent by h·w, broadcast
      -- it back over the H×W spatial grid, reshape to flat. Reverse of `.gapF`.
      -- Denotes `globalAvgPoolFlat`'s VJP backward `dy[chan idx] / (h·w)`.
      -- (Text emission best-effort/unverified-vs-IREE; the `den` is proven.)
      let nf ← fresh; let dv ← fresh; let bb ← fresh; let o ← fresh
      pure (
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
        s!"    {dv} = stablehlo.divide {r}, {nf} : {ty [B,c]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {dv}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {bb} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
  | .broadcastBack c h w, r :: st => do
      -- broadcast backward (VJP) = sum over H×W per channel (adjoint of broadcast):
      -- reshape to [B,c,h,w], reduce-add over spatial axes [2,3] → [B,c]. No divide.
      -- (Text emission best-effort/unverified-vs-IREE; the `den` is proven.)
      let xn ← fresh; let z ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {o} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n", o :: st)
  | .flatConvStride4F w b ic oc h w' kH kW, r :: st => do
      -- stride-4 patchify conv (the ConvNeXt 4×4/s4 stem): reshape, convolution
      -- with window_strides=[4,4], +bias. The denotation reads the SAME conv
      -- (pad (k-1)/2) at the offset-1 positions 4i+1 (decimate ∘ decimateOdd),
      -- so the emitted pad is one less: (k-1)/2 − 1 — for the 4×4 stem pad 0,
      -- the left-aligned window x[4i..4i+3] of the paper's pad-0 Conv2d(4, s=4).
      let pH := (kH - 1) / 2 - 1; let pW := (kW - 1) / 2 - 1
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*(2*h))*(2*(2*w'))]}) -> {ty [B,ic,2*(2*h),2*(2*w')]}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [4, 4], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,2*(2*h),2*(2*w')]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .flatConvStridedF w b ic oc h w' kH kW, r :: st => do
      -- stride-2 SAME conv: reshape, convolution with window_strides=[2,2], +bias
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w')]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,2*h,2*w']}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .flatConvStridedXlaF w b ic oc h w' kH kW, r :: st => do
      -- stride-2 XLA-`SAME` conv: identical to `flatConvStridedF` above but `pad = [p-1, p]`,
      -- the asymmetric split XLA uses at an even input (k=3 -> (0,1)).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w')]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH-1}, {pH}], [{pW-1}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,2*h,2*w']}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .convStridedBack w ic oc h w' kH kW, r :: st => do
      -- stride-2 conv input-VJP: zero-upsample dy (pad with interior=1, high=1) to
      -- the 2h×2w grid, then the reversed-kernel stride-1 conv (= decimate.back ▸ conv.back).
      -- Transpose-conv pad: low = k−1−p, high = p (p = the forward pad (k−1)/2) —
      -- symmetric (k−1)/2 for odd k (3×3 MNV2/r34, unchanged), [[1,0]] for the
      -- even 2×2 ConvNeXt downsample (the left-aligned forward window).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let z ← fresh; let up ← fresh; let wt ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w']}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {up} = stablehlo.pad {dn}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w']}, tensor<f32>) -> {ty [B,oc,2*h,2*w']}\n" ++
        s!"    {wt} = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {wr} = stablehlo.reverse {wt}, dims = [2, 3] : {ty [ic,oc,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({up}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{kH - 1 - pH}, {pH}], [{kW - 1 - pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,oc,2*h,2*w']}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,2*h,2*w']}) -> {ty [B, ic*(2*h)*(2*w')]}\n", o :: st)
  | .convStridedWeightSgd xN wN lrS ic oc h w kH kW, r :: st => do
      -- strided (stride-2) conv weight grad then SGD: reshape x to the 2h×2w grid and dy
      -- to h×w, zero-upsample dy (interior+high=1 → 2h×2w, the decimate-backward), then the
      -- SAME transpose-trick stride-1 weight-grad conv as `convWeightSgd` on the 2h×2w grid →
      -- [oc,ic,kH,kW], then θ' = θ − lr·dW. Same op text as `TestResnet34Train.convWGradStrided`.
      -- `sWGradGeom` is the odd/even split; odd reproduces the old inline formula byte-for-byte.
      let (upH, extH, loH, hiH) := sWGradGeom kH h
      let (upW, extW, loW, hiW) := sWGradGeom kW w
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
  | .convStridedXlaWeightSgd xN wN lrS ic oc h w kH kW, r :: st => do
      -- The XLA-`SAME` peer of `convStridedWeightSgd`: identical text, with the weight-grad
      -- correlation pad shifted one position (`loH-1`, `hiH+1`) so the saved input is read at
      -- `2*ho + 1 + kh - p` — the same shift as the batched `convStridedXlaWeightSgdB`.
      let (upH, extH, loH, hiH) := sWGradGeom kH h
      let (upW, extW, loW, hiW) := sWGradGeom kW w
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH-1}, {hiH+1}], [{loW-1}, {hiW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
  | .depthwiseWeightSgd xN wN lrS c h w kH kW, r :: st => do
      -- depthwise (grouped) weight grad: per-channel transpose-trick conv with
      -- `batch_group_count = c` (each output kernel reads only its own channel) → [1,c,kH,kW],
      -- reshape to the depthwise kernel layout [c,1,kH,kW], then θ' = θ − lr·dW. Same op text as
      -- `TestMobilenetV2Train.dwconvWGrad` + sgd.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
        s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
        s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
  | .depthwiseStridedWeightSgd xN wN lrS c h w kH kW, r :: st => do
      -- strided depthwise weight grad: reshape x to the 2h×2w grid and dy to h×w, zero-upsample dy
      -- (interior+high=1 → 2h×2w, the decimate-backward), then the SAME per-channel transpose-trick
      -- weight-grad conv (`batch_group_count = c`) on the 2h×2w grid → [1,c,kH,kW], reshape to the
      -- depthwise layout [c,1,kH,kW], then θ' = θ − lr·dW. Same op text as
      -- `TestMobilenetV2Train.dwconvWGradStrided` + sgd.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
        s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
  | .depthwiseStridedXlaWeightSgd xN wN lrS c h w kH kW, r :: st => do
      -- The XLA-`SAME` peer of `depthwiseStridedWeightSgd`: the per-channel correlation pad
      -- shifts to `[p-1, p+1]` (as the batched `depthwiseStridedXlaWeightGradB`).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH-1}, {pH+1}], [{pW-1}, {pW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
        s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
  | .bnPerChannelF gN bN epsStr oc h w, r :: st => do
      -- PER-CHANNEL BatchNorm forward: reshape to [B,oc,h,w], reduce μ/var over the
      -- spatial axes [2,3] (per channel), normalize, then γ·x̂+β with rank-1 γ/β
      -- (broadcast dims=[1]). Mirrors `bnF` but 4-D + per-channel.
      let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,oc,h,w]}\n" ++
        s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
  -- ══ PARAM GRADIENTS: the `*Sgd` emitters with the `constant lr / multiply / subtract`
  --    tail cut off. Same gradient text, so `θ − lr·grad` reproduces the SGD op exactly. ══
  | .weightGrad xN m n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [B,n]}) -> {ty [m,n]}\n", o :: st)
  | .biasGrad n, r :: st => do
      let z ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,n]}, tensor<f32>) -> {ty [n]}\n", o :: st)
  | .convWeightGrad xN ic oc h w kH kW, r :: st => do
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh; let raw ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
        s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n",
        o :: st)
  | .convBiasGrad _ic oc h w _kH _kW, r :: st => do
      let dr ← fresh; let z ← fresh; let o ← fresh
      pure (
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {o} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n",
        o :: st)
  | .convStridedWeightGrad xN ic oc h w kH kW, r :: st => do
      -- `sWGradGeom` is the odd/even split; at odd kernels it reproduces the old inline
      -- `pH = (kH-1)/2` formula byte-for-byte.
      let (upH, extH, loH, hiH) := sWGradGeom kH h
      let (upW, extW, loW, hiW) := sWGradGeom kW w
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n",
        o :: st)
  | .bnGammaGrad vN epsStr oc h w, r :: st => do
      -- dγ_c = Σ_{b,h,w} dy·x̂, with x̂ recomputed from the saved BN input {vN} (μ/var over
      -- the spatial axes [2,3] — per-channel, per-example, as `bnPerChannelF` normalises).
      let z ← fresh; let xr ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let dyr ← fresh; let p ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {p} = stablehlo.multiply {dyr}, {xhat} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n",
        o :: st)
  | .bnBetaGrad oc h w, r :: st => do
      -- dβ_c = Σ_{b,h,w} dy — needs no x̂.
      let z ← fresh; let dyr ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n",
        o :: st)
  -- ══ ADAMW: op-for-op `Proofs.adamMNext` / `adamVNext` / `adamWParam`, matching the
  --    hand-written `ViTRender.emitAdamV` block it replaces. Scalar hyperparameters are
  --    `tensor<f32>` function args, broadcast to the param shape `ds`. ══
  | .adamMNextF mN b1N ob1N ds, r :: st => do
      let T := ty ds
      let bb ← fresh; let ob ← fresh; let ms ← fresh; let mg ← fresh; let o ← fresh
      pure (
        s!"    {bb} = stablehlo.broadcast_in_dim {b1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob} = stablehlo.broadcast_in_dim {ob1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ms} = stablehlo.multiply {bb}, {mN} : {T}\n" ++
        s!"    {mg} = stablehlo.multiply {ob}, {r} : {T}\n" ++
        s!"    {o} = stablehlo.add {ms}, {mg} : {T}\n", o :: st)
  | .adamVNextF vN b2N ob2N ds, r :: st => do
      let T := ty ds
      let bb ← fresh; let ob ← fresh; let vs ← fresh; let g2 ← fresh; let vg ← fresh; let o ← fresh
      pure (
        s!"    {bb} = stablehlo.broadcast_in_dim {b2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob} = stablehlo.broadcast_in_dim {ob2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vs} = stablehlo.multiply {bb}, {vN} : {T}\n" ++
        s!"    {g2} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {vg} = stablehlo.multiply {ob}, {g2} : {T}\n" ++
        s!"    {o} = stablehlo.add {vs}, {vg} : {T}\n", o :: st)
  | .adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds, r :: st => do
      let T := ty ds
      -- m' and v' are recomputed here rather than shared with the two moment ops: SHlo is
      -- single-result, so each output is its own node. XLA's CSE folds the duplicates.
      let b1b ← fresh; let ob1b ← fresh; let ms ← fresh; let mg ← fresh; let mn ← fresh
      let b2b ← fresh; let ob2b ← fresh; let vs ← fresh; let g2 ← fresh; let vg ← fresh; let vn ← fresh
      let bc1b ← fresh; let bc2b ← fresh; let mh ← fresh; let vh ← fresh
      let lrb ← fresh; let epsb ← fresh; let sq ← fresh; let dn ← fresh; let rat ← fresh
      let stp ← fresh; let sub ← fresh; let wdb ← fresh; let wdlr ← fresh; let wdp ← fresh; let o ← fresh
      pure (
        s!"    {b1b} = stablehlo.broadcast_in_dim {b1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob1b} = stablehlo.broadcast_in_dim {ob1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ms} = stablehlo.multiply {b1b}, {mN} : {T}\n" ++
        s!"    {mg} = stablehlo.multiply {ob1b}, {r} : {T}\n" ++
        s!"    {mn} = stablehlo.add {ms}, {mg} : {T}\n" ++
        s!"    {b2b} = stablehlo.broadcast_in_dim {b2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob2b} = stablehlo.broadcast_in_dim {ob2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vs} = stablehlo.multiply {b2b}, {vN} : {T}\n" ++
        s!"    {g2} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {vg} = stablehlo.multiply {ob2b}, {g2} : {T}\n" ++
        s!"    {vn} = stablehlo.add {vs}, {vg} : {T}\n" ++
        s!"    {bc1b} = stablehlo.broadcast_in_dim {bc1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {bc2b} = stablehlo.broadcast_in_dim {bc2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {mh} = stablehlo.divide {mn}, {bc1b} : {T}\n" ++
        s!"    {vh} = stablehlo.divide {vn}, {bc2b} : {T}\n" ++
        s!"    {lrb} = stablehlo.broadcast_in_dim {lrN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {epsb} = stablehlo.broadcast_in_dim {epsN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {sq} = stablehlo.sqrt {vh} : {T}\n" ++
        s!"    {dn} = stablehlo.add {sq}, {epsb} : {T}\n" ++
        s!"    {rat} = stablehlo.divide {mh}, {dn} : {T}\n" ++
        s!"    {stp} = stablehlo.multiply {lrb}, {rat} : {T}\n" ++
        s!"    {sub} = stablehlo.subtract {θN}, {stp} : {T}\n" ++
        s!"    {wdb} = stablehlo.broadcast_in_dim {wdN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {wdlr} = stablehlo.multiply {wdb}, {lrb} : {T}\n" ++
        s!"    {wdp} = stablehlo.multiply {wdlr}, {θN} : {T}\n" ++
        s!"    {o} = stablehlo.subtract {sub}, {wdp} : {T}\n", o :: st)
  -- ══ SGD / NESTEROV (§2i): op-for-op `Proofs.sgdParam` / `momVNext` / `momParam`, matching the
  --    retired `tests/TestCifar8AdamTrain.emit{Sgd,Momentum}` blocks byte-for-byte modulo SSA
  --    freshness. `%lr` / `%mu` are runtime `tensor<f32>` args, broadcast to the param shape. ══
  | .sgdParamF θN lrN ds, r :: st => do
      let T := ty ds
      let lrb ← fresh; let stp ← fresh; let o ← fresh
      pure (
        s!"    {lrb} = stablehlo.broadcast_in_dim {lrN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {stp} = stablehlo.multiply {lrb}, {r} : {T}\n" ++
        s!"    {o} = stablehlo.subtract {θN}, {stp} : {T}\n", o :: st)
  | .momVNextF vN muN ds, r :: st => do
      let T := ty ds
      let mub ← fresh; let vg ← fresh; let o ← fresh
      pure (
        s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vg} = stablehlo.multiply {mub}, {vN} : {T}\n" ++
        s!"    {o} = stablehlo.add {vg}, {r} : {T}\n", o :: st)
  | .momParamF θN vN muN lrN ds, r :: st => do
      let T := ty ds
      -- v' is recomputed here rather than shared with `momVNextF`: SHlo is single-result, so each
      -- output is its own node. XLA's CSE folds the duplicate (§2b-bis measured that on R34's
      -- 108 → 36 rsqrt, at no run-time cost).
      let mub ← fresh; let vg ← fresh; let vel ← fresh
      let nv ← fresh; let lk ← fresh; let lrb ← fresh; let stp ← fresh; let o ← fresh
      pure (
        s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vg} = stablehlo.multiply {mub}, {vN} : {T}\n" ++
        s!"    {vel} = stablehlo.add {vg}, {r} : {T}\n" ++
        s!"    {nv} = stablehlo.multiply {mub}, {vel} : {T}\n" ++
        s!"    {lk} = stablehlo.add {nv}, {r} : {T}\n" ++
        s!"    {lrb} = stablehlo.broadcast_in_dim {lrN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {stp} = stablehlo.multiply {lrb}, {lk} : {T}\n" ++
        s!"    {o} = stablehlo.subtract {θN}, {stp} : {T}\n", o :: st)
  -- ══ RMSPROP (TensorFlow flavour): op-for-op `Proofs.rmsBufNext`, matching the JAX reference's
  --    `MOMENTUM * b + g / jnp.sqrt(s + EPS)` where `s = RHO*s + (1-RHO)*g*g`. `%rho`/`%orho`/
  --    `%mu`/`%eps` are runtime `tensor<f32>` args, broadcast to the param shape.
  --    ⚠ `{ep}` is added to the mean-square BEFORE the `sqrt`, not to the root after it. That one
  --    line is the entire difference from textbook RMSProp and it is a different optimizer —
  --    `Proofs.rmsBufNext_eps_placement_at_zero` states the gap (1/√ε against 1/ε).
  --    s' is recomputed here rather than shared with `adamVNextF`: SHlo is single-result, so each
  --    output is its own node. XLA's CSE folds the duplicate (§2b-bis measured that on R34's
  --    108 → 36 rsqrt, at no run-time cost). ══
  | .rmsBufNextF sqN bufN rhoN orhoN muN epsN ds, r :: st => do
      let T := ty ds
      let rhob ← fresh; let orhob ← fresh; let ss ← fresh; let g2 ← fresh; let sg ← fresh
      let sn ← fresh; let ep ← fresh; let da ← fresh; let dn ← fresh; let nrm ← fresh
      let mub ← fresh; let bs ← fresh; let o ← fresh
      pure (
        s!"    {rhob} = stablehlo.broadcast_in_dim {rhoN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {orhob} = stablehlo.broadcast_in_dim {orhoN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ss} = stablehlo.multiply {rhob}, {sqN} : {T}\n" ++
        s!"    {g2} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {sg} = stablehlo.multiply {orhob}, {g2} : {T}\n" ++
        s!"    {sn} = stablehlo.add {ss}, {sg} : {T}\n" ++
        s!"    {ep} = stablehlo.broadcast_in_dim {epsN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {da} = stablehlo.add {sn}, {ep} : {T}\n" ++
        s!"    {dn} = stablehlo.sqrt {da} : {T}\n" ++
        s!"    {nrm} = stablehlo.divide {r}, {dn} : {T}\n" ++
        s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {bs} = stablehlo.multiply {mub}, {bufN} : {T}\n" ++
        s!"    {o} = stablehlo.add {bs}, {nrm} : {T}\n", o :: st)
  -- ══ GLOBAL-NORM GRADIENT CLIPPING: op-for-op the reference's two lines
  --      gn    = sqrt(sum(jnp.sum(g*g) for g in tree.leaves(grads)))
  --      grads = tree.map(lambda g: g * minimum(1.0, CLIP/(gn + 1e-6)), grads)
  --    `emitLossAndTraining` in `jax/Jax/Codegen.lean`. All four emit at the PARAMETER shape `ty ds`, not at
  --    `ty [B,n]` — the clip runs on parameter gradients, after the batch has been contracted. ══
  | .gradSumSqAccF ds, g :: acc :: st => do
      -- One leaf's `jnp.sum(g*g)`, reduced over EVERY axis to rank 0, added to the running total.
      -- The dims list is `List.range ds.length` rather than a literal, so it is right at rank 1 and
      -- rank 4 alike; ViT and ConvNeXt have no rank-0 parameters, so it is never empty
      -- (`vitParamSig`, ConvNeXt's `allParams` — both checked, minimum rank 1).
      -- ⚠ Emits at the PARAMETER shape `ty ds`, not `ty [B,n]`: the clip runs on parameter
      -- gradients, after the batch has been contracted.
      let T := ty ds
      let dims := String.intercalate ", " ((List.range ds.length).map toString)
      let z ← fresh; let sq ← fresh; let red ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {sq} = stablehlo.multiply {g}, {g} : {T}\n" ++
        s!"    {red} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [{dims}] : ({T}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {o} = stablehlo.add {acc}, {red} : tensor<f32>\n", o :: st)
  | .clipScaleF clipS epsS ds, g :: sN :: st => do
      -- `g * min(1, CLIP/(sqrt(total) + 1e-6))` — the reference's second line.
      -- ⚠ ε is added to the ROOT, not under it: the opposite of `rmsBufNextF`'s TF placement, and
      -- this one follows the reference literally (`CLIP / (gn + 1e-6)`).
      -- ⚠ The `minimum` against 1.0 is not decoration — without it a SMALL gradient is AMPLIFIED
      -- by `c/gn`, which compiles, trains and descends (`Proofs.clipFactor_le_one`).
      -- The broadcast-then-multiply is `adamWParamF`'s `%lr` shape verbatim; it is the
      -- scale-by-a-RUNTIME-scalar the kit lacked (`scaleF` bakes a `constant dense<…>` instead).
      let T := ty ds
      let gn ← fresh; let ep ← fresh; let dn ← fresh; let cc ← fresh
      let rat ← fresh; let one ← fresh; let fac ← fresh; let fb ← fresh; let o ← fresh
      pure (
        s!"    {gn} = stablehlo.sqrt {sN} : tensor<f32>\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsS}> : tensor<f32>\n" ++
        s!"    {dn} = stablehlo.add {gn}, {ep} : tensor<f32>\n" ++
        s!"    {cc} = stablehlo.constant dense<{clipS}> : tensor<f32>\n" ++
        s!"    {rat} = stablehlo.divide {cc}, {dn} : tensor<f32>\n" ++
        s!"    {one} = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
        s!"    {fac} = stablehlo.minimum {one}, {rat} : tensor<f32>\n" ++
        s!"    {fb} = stablehlo.broadcast_in_dim {fac}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {o} = stablehlo.multiply {fb}, {g} : {T}\n", o :: st)
  -- ══ LAMB (`Proofs.Lamb`), RSB-A3's optimizer. `lambDirF` is `adamWParamF`'s block truncated at
  --    the ratio with `wd·θ` ADDED rather than subtracted at the end — the decay is decoupled and
  --    lands INSIDE the direction, hence inside the norm the trust ratio takes. ══
  | .lambDirF θN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN ds, r :: st => do
      let T := ty ds
      let b1b ← fresh; let ob1b ← fresh; let ms ← fresh; let mg ← fresh; let mn ← fresh
      let b2b ← fresh; let ob2b ← fresh; let vs ← fresh; let g2 ← fresh; let vg ← fresh; let vn ← fresh
      let bc1b ← fresh; let bc2b ← fresh; let mh ← fresh; let vh ← fresh
      let epsb ← fresh; let sq ← fresh; let dn ← fresh; let rat ← fresh
      let wdb ← fresh; let wdp ← fresh; let o ← fresh
      pure (
        s!"    {b1b} = stablehlo.broadcast_in_dim {b1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob1b} = stablehlo.broadcast_in_dim {ob1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ms} = stablehlo.multiply {b1b}, {mN} : {T}\n" ++
        s!"    {mg} = stablehlo.multiply {ob1b}, {r} : {T}\n" ++
        s!"    {mn} = stablehlo.add {ms}, {mg} : {T}\n" ++
        s!"    {b2b} = stablehlo.broadcast_in_dim {b2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob2b} = stablehlo.broadcast_in_dim {ob2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vs} = stablehlo.multiply {b2b}, {vN} : {T}\n" ++
        s!"    {g2} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {vg} = stablehlo.multiply {ob2b}, {g2} : {T}\n" ++
        s!"    {vn} = stablehlo.add {vs}, {vg} : {T}\n" ++
        s!"    {bc1b} = stablehlo.broadcast_in_dim {bc1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {bc2b} = stablehlo.broadcast_in_dim {bc2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {mh} = stablehlo.divide {mn}, {bc1b} : {T}\n" ++
        s!"    {vh} = stablehlo.divide {vn}, {bc2b} : {T}\n" ++
        s!"    {epsb} = stablehlo.broadcast_in_dim {epsN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        -- ⚠ ε OUTSIDE the root. `sqrt(vh + eps)` is RMSProp-TF's placement and a different
        -- optimizer; the reference is literal — `mc / (jnp.sqrt(vc) + EPS)`.
        s!"    {sq} = stablehlo.sqrt {vh} : {T}\n" ++
        s!"    {dn} = stablehlo.add {sq}, {epsb} : {T}\n" ++
        s!"    {rat} = stablehlo.divide {mh}, {dn} : {T}\n" ++
        s!"    {wdb} = stablehlo.broadcast_in_dim {wdN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {wdp} = stablehlo.multiply {wdb}, {θN} : {T}\n" ++
        s!"    {o} = stablehlo.add {rat}, {wdp} : {T}\n", o :: st)
  | .lambScaleF ds, r :: wn2 :: st => do
      -- `trust · r` with `trust = ‖θ‖/‖r‖`, guarded to 1 when either norm vanishes.
      -- ⚠⚠ `‖r‖²` is reduced HERE, from this op's own tensor child, so the trust ratio is
      -- PER PARAMETER TENSOR. That is the whole difference from `clipScaleF` above, whose factor
      -- is one scalar shared across every parameter — the two blocks look alike and differ in the
      -- quantifier (`Proofs.lambScale_not_shared` states it).
      -- ⚠ The guard is not a corner case: the driver inits every BN β and dense bias to 0, so
      -- `wn2 = 0` on those tensors at step 1 and `select` takes the `1.0` branch on a real run.
      let T := ty ds
      let dims := String.intercalate ", " ((List.range ds.length).map toString)
      let z ← fresh; let rsq ← fresh; let rn2 ← fresh
      let wn ← fresh; let rn ← fresh; let rat ← fresh; let one ← fresh
      let okW ← fresh; let okR ← fresh; let ok ← fresh; let tr ← fresh; let tb ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {rsq} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {rn2} = stablehlo.reduce({rsq} init: {z}) applies stablehlo.add across dimensions = [{dims}] : ({T}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {wn} = stablehlo.sqrt {wn2} : tensor<f32>\n" ++
        s!"    {rn} = stablehlo.sqrt {rn2} : tensor<f32>\n" ++
        s!"    {rat} = stablehlo.divide {wn}, {rn} : tensor<f32>\n" ++
        s!"    {one} = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
        s!"    {okW} = stablehlo.compare GT, {wn2}, {z} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
        s!"    {okR} = stablehlo.compare GT, {rn2}, {z} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
        s!"    {ok} = stablehlo.and {okW}, {okR} : tensor<i1>\n" ++
        s!"    {tr} = stablehlo.select {ok}, {rat}, {one} : tensor<i1>, tensor<f32>\n" ++
        s!"    {tb} = stablehlo.broadcast_in_dim {tr}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {o} = stablehlo.multiply {tb}, {r} : {T}\n", o :: st)
  | .bnPerChannelEvalF gN bN muN varN epsStr oc h w, r :: st => do
      -- INFERENCE per-channel BatchNorm: reshape to [B,oc,h,w], then the affine map
      -- γ·(x − μ)·rsqrt(var + ε) + β with μ/var/γ/β all rank-1 `[oc]` graph inputs
      -- (broadcast dims=[1]). No reduce and no normalizer constant — that is the whole
      -- difference from `bnPerChannelF`, and why eval is class-batch-independent.
      let xn ← fresh; let mub ← fresh; let xc ← fresh; let vb ← fresh; let ep ← fresh
      let ve ← fresh; let istd ← fresh; let xhat ← fresh; let gb ← fresh; let bb ← fresh
      let gx ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mub} : {ty [B,oc,h,w]}\n" ++
        s!"    {vb} = stablehlo.broadcast_in_dim {varN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vb}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,oc,h,w]}\n" ++
        s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
  | .bnPerChannelBack gN xN epsStr oc h w, r :: st => do
      -- PER-CHANNEL BN input-VJP: recompute x̂/istd per channel from saved input {xN},
      -- then the block-diagonal three-term `(istd/m)·(m·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))`,
      -- dx̂ = γ·dy, with all Σ reductions over the spatial axes [2,3] (m = h·w).
      let dn ← fresh; let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o0 ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {dn} : {ty [B,oc,h,w]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,oc,h,w]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,oc,h,w]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,oc,h,w]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,oc,h,w]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {o0} = stablehlo.multiply {sN}, {i2} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {o0} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
  | .depthwiseF w b c h w' kH kW, r :: st => do
      -- depthwise conv forward: reshape to [B,c,h,w'], grouped `stablehlo.convolution`
      -- (feature_group_count = c, [c,1,kH,kW] kernel — one filter per channel, no
      -- cross-channel mixing), SAME pad, + per-channel bias, reshape back.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,h,w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseBack w c h w' kH kW, r :: st => do
      -- depthwise conv input-VJP: reshape dy, reverse the per-channel filters over the
      -- spatial axes [2,3] (the channel groups are 1×1, so no o↔i transpose), then the
      -- reversed-kernel SAME-pad depthwise conv (feature_group_count = c).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {wr} = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({dn}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,h,w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseStridedF w b c h w' kH kW, r :: st => do
      -- stride-2 depthwise conv: reshape, grouped convolution with window_strides=[2,2]
      -- (feature_group_count = c, [c,1,kH,kW] kernel), SAME pad, + bias. Halves spatial.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w')]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseStridedXlaF w b c h w' kH kW, r :: st => do
      -- stride-2 XLA-`SAME` depthwise: as above with `pad = [p-1, p]`.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w')]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH-1}, {pH}], [{pW-1}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseStridedBack w c h w' kH kW, r :: st => do
      -- stride-2 depthwise input-VJP: zero-upsample dy (pad interior/high=1) back to
      -- 2h×2w', reverse the per-channel filters over [2,3] (no transpose, 1×1 groups),
      -- then the reversed-kernel stride-1 depthwise conv (feature_group_count = c).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let z ← fresh; let up ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {up} = stablehlo.pad {dn}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w']}, tensor<f32>) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {wr} = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({up}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w']}) -> {ty [B, c*(2*h)*(2*w')]}\n", o :: st)
  | .depthwiseStridedXlaBack w c h w' kH kW, r :: st => do
      -- The XLA-`SAME` peer of `depthwiseStridedBack`: the transposed-conv pad shifts to
      -- `[p+1, p-1]`. ⚠⚠ The OPPOSITE direction from the two weight grads (`[p-1, p+1]`),
      -- because the kernel is reversed here — see `depthwiseStridedXlaBackBatched`'s note; a
      -- version derived "by symmetry" with its siblings type-checks, descends, and is wrong.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let z ← fresh; let up ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {up} = stablehlo.pad {dn}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w']}, tensor<f32>) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {wr} = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({up}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH+1}, {pH-1}], [{pW+1}, {pW-1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w']}) -> {ty [B, c*(2*h)*(2*w')]}\n", o :: st)
  | .swishF n, r :: st => do
      let (txt4, res4) ← liftPointwise B n r fun r d => do
          -- swish forward: y = x · σ(x), σ = logistic (smooth everywhere, no kink/mask).
          let s ← fresh; let o ← fresh
          pure (s!"    {s} = stablehlo.logistic {r} : {ty d}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {s} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .swishBack x n, r :: st => do
      let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
          -- swish input-VJP: dy ⊙ σ(x)·(1 + x·(1−σ(x))), recomputing σ from the saved
          -- pre-activation {x} (`swishScalarDeriv_eq`, IRPrint `swishB`).
          let s ← fresh; let one ← fresh; let om ← fresh; let xom ← fresh
          let inr ← fresh; let sp ← fresh; let o ← fresh
          pure (s!"    {s} = stablehlo.logistic {x} : {ty d}\n" ++
                s!"    {one} = stablehlo.constant dense<1.0> : {ty d}\n" ++
                s!"    {om} = stablehlo.subtract {one}, {s} : {ty d}\n" ++
                s!"    {xom} = stablehlo.multiply {x}, {om} : {ty d}\n" ++
                s!"    {inr} = stablehlo.add {one}, {xom} : {ty d}\n" ++
                s!"    {sp} = stablehlo.multiply {s}, {inr} : {ty d}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {sp} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .sigmoidF n, r :: st => do
      let (txt4, res4) ← liftPointwise B n r fun r d => do
          -- sigmoid forward: σ(x) = logistic(x) (smooth, the SE gate's output nonlinearity).
          let o ← fresh
          pure (s!"    {o} = stablehlo.logistic {r} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .sigmoidBack x n, r :: st => do
      let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
          -- sigmoid input-VJP: dy ⊙ σ(x)·(1−σ(x)), recomputing σ from the saved
          -- pre-activation {x} (`sigmoidScalarDeriv`, IRPrint `sigmoidBackM`).
          let s ← fresh; let one ← fresh; let om ← fresh; let sp ← fresh; let o ← fresh
          pure (s!"    {s} = stablehlo.logistic {x} : {ty d}\n" ++
                s!"    {one} = stablehlo.constant dense<1.0> : {ty d}\n" ++
                s!"    {om} = stablehlo.subtract {one}, {s} : {ty d}\n" ++
                s!"    {sp} = stablehlo.multiply {s}, {om} : {ty d}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {sp} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .layerScaleF gN n, r :: st => do
      -- per-element layer-scale `γ ⊙ x`: broadcast γ:[n] over the batch, then multiply.
      let gb ← fresh; let o ← fresh
      pure (s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [n]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {gb} : {ty [B,n]}\n", o :: st)
  | .layerScaleChF gN c h w', r :: st => do
      -- per-channel layer-scale: reshape flat→NCHW, broadcast γ:[c] over
      -- batch+spatial (dims=[1]), multiply, reshape back.
      let xn ← fresh; let gb ← fresh; let m ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
            s!"    {m} = stablehlo.multiply {xn}, {gb} : {ty [B,c,h,w']}\n" ++
            s!"    {o} = stablehlo.reshape {m} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .geluF n, r :: st => do
      let (txt4, res4) ← liftPointwise B n r fun r d => do
          -- gelu forward (tanh approximation): y = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
          -- Smooth everywhere (no kink/mask); `stablehlo.tanh` is the only non-arith op.
          let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
          let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
          let chalf ← fresh; let hx ← fresh; let o ← fresh
          pure (s!"    {x2} = stablehlo.multiply {r}, {r} : {ty d}\n" ++
                s!"    {x3} = stablehlo.multiply {x2}, {r} : {ty d}\n" ++
                s!"    {ck} = stablehlo.constant dense<0.044715> : {ty d}\n" ++
                s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty d}\n" ++
                s!"    {inn} = stablehlo.add {r}, {kx3} : {ty d}\n" ++
                s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty d}\n" ++
                s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty d}\n" ++
                s!"    {t} = stablehlo.tanh {u} : {ty d}\n" ++
                s!"    {one} = stablehlo.constant dense<1.0> : {ty d}\n" ++
                s!"    {opt} = stablehlo.add {one}, {t} : {ty d}\n" ++
                s!"    {chalf} = stablehlo.constant dense<0.5> : {ty d}\n" ++
                s!"    {hx} = stablehlo.multiply {chalf}, {r} : {ty d}\n" ++
                s!"    {o} = stablehlo.multiply {hx}, {opt} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .geluBack x n, r :: st => do
      let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
          -- gelu input-VJP: dy ⊙ gelu'(x), recomputing tanh(u(x)) from the saved
          -- pre-activation {x}. gelu'(x) = 0.5·(1+t) + 0.5·x·(1−t²)·√(2/π)·(1+3·0.044715·x²),
          -- t = tanh(√(2/π)·(x+0.044715·x³)). (Matches IRPrint `renderGeluB`.)
          let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
          let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
          let chalf ← fresh; let term1 ← fresh; let t2 ← fresh; let omt2 ← fresh
          let hx ← fresh; let hxo ← fresh; let c3b ← fresh; let a3x2 ← fresh
          let in2 ← fresh; let up ← fresh; let term2 ← fresh; let gp ← fresh; let o ← fresh
          pure (s!"    {x2} = stablehlo.multiply {x}, {x} : {ty d}\n" ++
                s!"    {x3} = stablehlo.multiply {x2}, {x} : {ty d}\n" ++
                s!"    {ck} = stablehlo.constant dense<0.044715> : {ty d}\n" ++
                s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty d}\n" ++
                s!"    {inn} = stablehlo.add {x}, {kx3} : {ty d}\n" ++
                s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty d}\n" ++
                s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty d}\n" ++
                s!"    {t} = stablehlo.tanh {u} : {ty d}\n" ++
                s!"    {one} = stablehlo.constant dense<1.0> : {ty d}\n" ++
                s!"    {opt} = stablehlo.add {one}, {t} : {ty d}\n" ++
                s!"    {chalf} = stablehlo.constant dense<0.5> : {ty d}\n" ++
                s!"    {term1} = stablehlo.multiply {chalf}, {opt} : {ty d}\n" ++
                s!"    {t2} = stablehlo.multiply {t}, {t} : {ty d}\n" ++
                s!"    {omt2} = stablehlo.subtract {one}, {t2} : {ty d}\n" ++
                s!"    {hx} = stablehlo.multiply {chalf}, {x} : {ty d}\n" ++
                s!"    {hxo} = stablehlo.multiply {hx}, {omt2} : {ty d}\n" ++
                s!"    {c3b} = stablehlo.constant dense<0.134145> : {ty d}\n" ++
                s!"    {a3x2} = stablehlo.multiply {c3b}, {x2} : {ty d}\n" ++
                s!"    {in2} = stablehlo.add {one}, {a3x2} : {ty d}\n" ++
                s!"    {up} = stablehlo.multiply {csqrt}, {in2} : {ty d}\n" ++
                s!"    {term2} = stablehlo.multiply {hxo}, {up} : {ty d}\n" ++
                s!"    {gp} = stablehlo.add {term1}, {term2} : {ty d}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {gp} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .softmaxRowF m n, r :: st => do
      -- ROW-softmax: reshape flat `[B,m*n]` → `[B,m,n]`, exp, reduce add over the
      -- LAST axis [2] (per row), broadcast back over dims [0,1], divide, reshape to
      -- flat. Plain exp/sum (no max-shift), matching the proven `softmax` (3-D
      -- analogue of `.softmaxDiv`).
      let xn ← fresh; let z ← fresh; let e ← fresh; let s ← fresh; let sb ← fresh
      let dv ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {e} = stablehlo.exponential {xn} : {ty [B,m,n]}\n" ++
        s!"    {s} = stablehlo.reduce({e} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {dv} = stablehlo.divide {e}, {sb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {dv} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .softmaxRowBack x m n, r :: st => do
      -- ROW-softmax input-VJP `p ⊙ (dy − ⟨p,dy⟩)` per row: reshape flat→`[B,m,n]`,
      -- recompute `p` from the saved pre-softmax scores {x} (exp/reduce[2]/broadcast/
      -- divide), then the rank-1 correction (`pdy`, reduce[2], subtract, multiply),
      -- reshape to flat. {r} is dy.
      let xn ← fresh; let dn ← fresh; let z ← fresh; let e ← fresh; let s ← fresh
      let sb ← fresh; let p ← fresh; let pdy ← fresh; let sr ← fresh; let srb ← fresh
      let d ← fresh; let dz ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {x} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {e} = stablehlo.exponential {xn} : {ty [B,m,n]}\n" ++
        s!"    {s} = stablehlo.reduce({e} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {p} = stablehlo.divide {e}, {sb} : {ty [B,m,n]}\n" ++
        s!"    {pdy} = stablehlo.multiply {p}, {dn} : {ty [B,m,n]}\n" ++
        s!"    {sr} = stablehlo.reduce({pdy} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {srb} = stablehlo.broadcast_in_dim {sr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {d} = stablehlo.subtract {dn}, {srb} : {ty [B,m,n]}\n" ++
        s!"    {dz} = stablehlo.multiply {p}, {d} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {dz} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .matmulF m k n, b :: a :: st => do
      -- flattened matrix multiply C = A·B (`emitMatmul`). (Postorder pushes a then b, so b is on top.)
      let (s, o) ← emitMatmul B none a b m k n
      pure (s, o :: st)
  | .transposeF m n, r :: st => do
      -- flattened matrix transpose: reshape to rank 3, swap the matrix axes
      -- (dims = [0, 2, 1], batch axis fixed), reshape back.
      let xn ← fresh; let t ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {t} = stablehlo.transpose {xn}, dims = [0, 2, 1] : ({ty [B,m,n]}) -> {ty [B,n,m]}\n" ++
        s!"    {o} = stablehlo.reshape {t} : ({ty [B,n,m]}) -> {ty [B, n*m]}\n", o :: st)
  | .scaleF sStr n, r :: st => do
      let (txt4, res4) ← liftPointwise B n r fun r d => do
          -- scalar multiply s·x against a splat constant (SDPA's 1/√d).
          let c ← fresh; let o ← fresh
          pure (s!"    {c} = stablehlo.constant dense<{sStr}> : {ty d}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {c} : {ty d}\n", o)
      pure (txt4, res4 :: st)
  | .lnRowF gN bN epsStr m n, r :: st => do
      -- ROW-wise LayerNorm forward: reshape flat [B,m*n] → [B,m,n], then `bnF`'s
      -- normalize/affine graph at rank 3 — μ/var reduced over the LAST axis [2]
      -- (per token row), broadcast back over dims [0,1], scalar γ/β (dims = []),
      -- reshape to flat. LayerNorm IS per-example BN per row.
      let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,m,n]}\n" ++
        s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .lnRowBack gN xN epsStr m n, r :: st => do
      -- ROW-wise LN input-VJP: recompute x̂/istd per row from the saved flat
      -- pre-LN input {xN}, then `bnBack`'s consolidated three-term
      -- `(istd/n)·(n·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))` (dx̂ = γ·dy) at rank 3, all Σ
      -- reductions over the row axis [2], reshape to flat. {r} is dy.
      let dn ← fresh; let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o0 ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {dn} : {ty [B,m,n]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,m,n]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,m,n]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,m,n]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,m,n]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {o0} = stablehlo.multiply {sN}, {i2} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {o0} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .denseRowF wN bN N a c, r :: st => do
      -- per-token dense: reshape [B,N*a] → [B,N,a], dot_general contracting the
      -- feature axis with W:[a,c] ([2] x [0] — every token row through the same W),
      -- bias broadcast dims = [2], reshape back. (ViTRender `mlpRowFwd` form.)
      let xn ← fresh; let dg ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, N*a]}) -> {ty [B,N,a]}\n" ++
        s!"    {dg} = stablehlo.dot_general {xn}, {wN}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,N,a]}, {ty [a,c]}) -> {ty [B,N,c]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [c]}) -> {ty [B,N,c]}\n" ++
        s!"    {ob} = stablehlo.add {dg}, {bb} : {ty [B,N,c]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,N,c]}) -> {ty [B, N*c]}\n", o :: st)
  | .denseRowBack wN N a c, r :: st => do
      -- per-token dense input-VJP dX = dY·Wᵀ: contract dy's feature axis with W's
      -- OUTPUT axis ([2] x [1] — the GPU-validated ViTRender backward form).
      let dn ← fresh; let dg ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
        s!"    {dg} = stablehlo.dot_general {dn}, {wN}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,N,c]}, {ty [a,c]}) -> {ty [B,N,a]}\n" ++
        s!"    {o} = stablehlo.reshape {dg} : ({ty [B,N,a]}) -> {ty [B, N*a]}\n", o :: st)
  | .patchEmbedF wN bN clsN posN ic H W P N D, r :: st => do
      -- ViT patch embedding: reshape image to [B,ic,H,W], stride-P VALID conv
      -- (kernel [D,ic,P,P] — the non-overlapping patch projection) + bias, move
      -- channels last (transpose [0,2,3,1]) and flatten the patch grid to [B,N,D],
      -- prepend the broadcast CLS token (concatenate at dim 1), add the position
      -- embedding (broadcast dims = [1,2]), reshape to flat [B,(N+1)*D].
      let hp := H / P; let wp := W / P
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let cb ← fresh
      let tr ← fresh; let tk ← fresh; let clsb ← fresh; let cat ← fresh
      let pb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {wN})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [{P}, {P}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,H,W]}, {ty [D,ic,P,P]}) -> {ty [B,D,hp,wp]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [D]}) -> {ty [B,D,hp,wp]}\n" ++
        s!"    {cb} = stablehlo.add {cv}, {bb} : {ty [B,D,hp,wp]}\n" ++
        s!"    {tr} = stablehlo.transpose {cb}, dims = [0, 2, 3, 1] : ({ty [B,D,hp,wp]}) -> {ty [B,hp,wp,D]}\n" ++
        s!"    {tk} = stablehlo.reshape {tr} : ({ty [B,hp,wp,D]}) -> {ty [B,N,D]}\n" ++
        s!"    {clsb} = stablehlo.broadcast_in_dim {clsN}, dims = [2] : ({ty [D]}) -> {ty [B,1,D]}\n" ++
        s!"    {cat} = stablehlo.concatenate {clsb}, {tk}, dim = 1 : ({ty [B,1,D]}, {ty [B,N,D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {pb} = stablehlo.broadcast_in_dim {posN}, dims = [1, 2] : ({ty [N+1,D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {ob} = stablehlo.add {cat}, {pb} : {ty [B,N+1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,N+1,D]}) -> {ty [B, (N+1)*D]}\n", o :: st)
  | .clsSliceF N D, r :: st => do
      -- CLS-token gather (row 0): reshape [B,(N+1)*D] → [B,N+1,D], slice the
      -- first token row, reshape to [B,D]. (ViTRender `headFwd` slice form.)
      let xn ← fresh; let sl ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:1, 0:{D}] : ({ty [B,N+1,D]}) -> {ty [B,1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {sl} : ({ty [B,1,D]}) -> {ty [B,D]}\n", o :: st)
  | .clsPadF N D, r :: st => do
      -- CLS-slice VJP (scatter dy to row 0): reshape [B,D] → [B,1,D], zero-pad
      -- N token rows below (high = [0, N, 0]), reshape to flat [B,(N+1)*D].
      -- (ViTRender `headBack` pad form.)
      let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B,D]}) -> {ty [B,1,D]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, 0], high = [0, {N}, 0], interior = [0, 0, 0] : ({ty [B,1,D]}, tensor<f32>) -> {ty [B,N+1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {pd} : ({ty [B,N+1,D]}) -> {ty [B, (N+1)*D]}\n", o :: st)
  | .headSliceF N heads d hIdx, r :: st => do
      -- per-head column slice: reshape [B,N*(H*d)] → [B,N,H*d], slice head h's
      -- contiguous feature block [h*d:(h+1)*d] (row-major layout), reshape to flat.
      let xn ← fresh; let sl ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, N*(heads*d)]}) -> {ty [B,N,heads*d]}\n" ++
        s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:{N}, {hIdx*d}:{(hIdx+1)*d}] : ({ty [B,N,heads*d]}) -> {ty [B,N,d]}\n" ++
        s!"    {o} = stablehlo.reshape {sl} : ({ty [B,N,d]}) -> {ty [B, N*d]}\n", o :: st)
  | .headPadF N heads d hIdx, r :: st => do
      -- per-head column scatter: reshape [B,N*d] → [B,N,d], zero-pad the feature
      -- axis into head h's block (low = h*d, high = (heads-1-h)*d), reshape to flat.
      let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*d]}) -> {ty [B,N,d]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, {hIdx*d}], high = [0, 0, {(heads-1-hIdx)*d}], interior = [0, 0, 0] : ({ty [B,N,d]}, tensor<f32>) -> {ty [B,N,heads*d]}\n" ++
        s!"    {o} = stablehlo.reshape {pd} : ({ty [B,N,heads*d]}) -> {ty [B, N*(heads*d)]}\n", o :: st)
  | .rowScaleF gN m n, r :: st => do
      -- per-token broadcast scale: reshape [B,m*n] -> [B,m,n], broadcast the shared
      -- gamma:[n] over batch+rows (dims = [2]), multiply, reshape back.
      let xn <- fresh; let gb <- fresh; let mu <- fresh; let o <- fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.multiply {xn}, {gb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {mu} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .rowBiasF bN m n, r :: st => do
      -- per-token broadcast bias: same bracket, broadcast beta:[n] dims = [2], add.
      let xn <- fresh; let bb <- fresh; let ad <- fresh; let o <- fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
        s!"    {ad} = stablehlo.add {xn}, {bb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {ad} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .batched tag names info, r :: st =>
      -- EfficientNet batched op: emit the concrete `[N,C,H,W]` StableHLO from the
      -- tag (which op) + names (weight/bias/BN-input/SE-input/γ/ε SSA names) + info
      -- (shape dims). Batched values flow as 2-D `[B, c·h·w]` (B = batch); each op
      -- reshapes its operand to 4-D, computes, reshapes back — uniform with the
      -- per-example ops (`convWeightSgd`/`denseRowF` do the same). Backward ops are
      -- self-contained: they recompute forward intermediates from the carried
      -- input/weight names (the mnv2 pattern). `den` never calls `emit`; this text
      -- is iree-validated, not theorem-tied (the per-op lexing trust the whole
      -- suite carries). Backward tags are filled in the next pass.
      match tag, names, info with
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      -- fp8 (E4M3) emit: byte-for-byte `convBf16`'s shape with `tyF8` in place of
      -- `tyBf16`. f8 operands, f8-TYPED conv result, convert back, bias added in f32.
      | "conv", [wN, bN], [_N, ic, oc, h, w, kH, kW] | "convBf16", [wN, bN], [_N, ic, oc, h, w, kH, kW] | "convF8", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh
          let (cs, cc) ← emitContract (lowOf tag) xr wN [B,ic,h,w] [oc,ic,kH,kW] [B,oc,h,w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convStrided", [wN, bN], [_N, ic, oc, h, w, kH, kW] | "convStridedBf16", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh
          let (cs, cc) ← emitContract (lowOf tag) xr wN [B,ic,2*h,2*w] [oc,ic,kH,kW] [B,oc,h,w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      -- ⭐ The asymmetric-pad peer. `pad_low = (k-2)/2 = p-1`, `pad_high = k/2 = p` for odd `k`
      -- (k=3 → [[0,1]], k=5 → [[1,2]], k=7 → [[2,3]]) — exactly what XLA computes for `'SAME'` at
      -- an even input, which is the only input shape this token's type admits (`2*h`, `2*w`).
      -- Everything else is byte-identical to "convStrided", which is the point: the ONLY
      -- difference between the two nets is these four numbers.
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "convStridedXla", [wN, bN], [_N, ic, oc, h, w, kH, kW] | "convStridedXlaBf16", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let lo := p - 1
          let xr ← fresh
          let (cs, cc) ← emitContract (lowOf tag) xr wN [B,ic,2*h,2*w] [oc,ic,kH,kW] [B,oc,h,w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{lo}, {p}], [{lo}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      -- ⭐ The asymmetric-pad depthwise. `pad_low = p-1`, `pad_high = p` (k=3 → [[0,1]], k=5 →
      -- [[1,2]]) — XLA `'SAME'` at an even input, which is the only shape this token's type admits.
      -- Byte-identical to "depthwiseStrided" apart from those four numbers.
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "depthwise", [wN, bN], [_N, c, h, w, kH, kW] | "depthwiseBf16", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh
          let (cs, cc) ← emitContract (lowOf tag) xr wN [B,c,h,w] [c,1,kH,kW] [B,c,h,w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}"
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "depthwiseStridedXla", [wN, bN], [_N, c, h, w, kH, kW] | "depthwiseStridedXlaBf16", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let lo := p - 1
          let xr ← fresh
          let (cs, cc) ← emitContract (lowOf tag) xr wN [B,c,2*h,2*w] [c,1,kH,kW] [B,c,h,w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{lo}, {p}], [{lo}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}"
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. The f32-result
      -- shape folds to pure f32, for grouped convolutions exactly as for ordinary ones
      -- (measured). ⚠ SYMMETRIC pad — this is the torchvision-origin variant, NOT `Xla`.
      | "depthwiseStrided", [wN, bN], [_N, c, h, w, kH, kW] | "depthwiseStridedBf16", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh
          let (cs, cc) ← emitContract (lowOf tag) xr wN [B,c,2*h,2*w] [c,1,kH,kW] [B,c,h,w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}"
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "dense", [wN, bN], [_N, a, c] => do
          let dg ← fresh; let bb ← fresh; let o ← fresh
          pure (
            s!"    {dg} = stablehlo.dot_general {r}, {wN}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,a]}, {ty [a,c]}) -> {ty [B,c]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {o} = stablehlo.add {dg}, {bb} : {ty [B,c]}\n", o :: st)
      | "gap", [], [_N, c, h, w] => do
          let xr ← fresh; let z ← fresh; let sr ← fresh; let nf ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {o} = stablehlo.divide {sr}, {nf} : {ty [B,c]}\n", o :: st)
      | "seBlock", [w1, b1, w2, b2], [_N, c, h, w, rr] => do
          let xr ← fresh; let z ← fresh; let sqs ← fresh; let sqnf ← fresh; let sq ← fresh
          let exd ← fresh; let exbb ← fresh; let ex ← fresh; let a1s ← fresh; let a1 ← fresh
          let h2d ← fresh; let h2bb ← fresh; let h2 ← fresh; let gate ← fresh; let gb ← fresh
          let se ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sqs} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {sqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {sq} = stablehlo.divide {sqs}, {sqnf} : {ty [B,c]}\n" ++
            s!"    {exd} = stablehlo.dot_general {sq}, {w1}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [c,rr]}) -> {ty [B,rr]}\n" ++
            s!"    {exbb} = stablehlo.broadcast_in_dim {b1}, dims = [1] : ({ty [rr]}) -> {ty [B,rr]}\n" ++
            s!"    {ex} = stablehlo.add {exd}, {exbb} : {ty [B,rr]}\n" ++
            s!"    {a1s} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {a1} = stablehlo.multiply {ex}, {a1s} : {ty [B,rr]}\n" ++
            s!"    {h2d} = stablehlo.dot_general {a1}, {w2}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [rr,c]}) -> {ty [B,c]}\n" ++
            s!"    {h2bb} = stablehlo.broadcast_in_dim {b2}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {h2} = stablehlo.add {h2d}, {h2bb} : {ty [B,c]}\n" ++
            s!"    {gate} = stablehlo.logistic {h2} : {ty [B,c]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gate}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {se} = stablehlo.multiply {xr}, {gb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {se} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "swish", [], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            -- Pointwise swish at the BATCHED index: byte-for-byte the `.swishF` emit,
            -- except the width comes from the descriptor's per-example `n` rather than
            -- from the SHlo index (which here is `N·n`). `_N` is discarded for the same
            -- reason every batched tag discards it — the runtime batch is `B`.
            let s ← fresh; let o ← fresh
            pure (s!"    {s} = stablehlo.logistic {r} : {ty d}\n" ++
                  s!"    {o} = stablehlo.multiply {r}, {s} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "relu", [], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            -- byte-for-byte `.reluF`'s emit, width from the descriptor's `n`.
            let z ← fresh; let o ← fresh
            pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty d}\n" ++
                  s!"    {o} = stablehlo.maximum {r}, {z} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "bnEval", [gN, bN, muN, varN, es], [_N, oc, h, w] => do
          -- byte-for-byte `.bnPerChannelEvalF`'s emit, dims from the descriptor rather than off
          -- the SHlo index (§2b). INFERENCE BN: reshape to [B,oc,h,w], then the affine map
          -- γ·(x − μ)·rsqrt(var + ε) + β with μ/var/γ/β all rank-1 `[oc]` graph inputs. No reduce
          -- and no normalizer constant — that is the whole difference from `bnBatch`, and why the
          -- descriptor form is denotationally honest at any `N`.
          let xn ← fresh; let mub ← fresh; let xc ← fresh; let vb ← fresh; let ep ← fresh
          let ve ← fresh; let istd ← fresh; let xhat ← fresh; let gb ← fresh; let bb ← fresh
          let gx ← fresh; let ob ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xn}, {mub} : {ty [B,oc,h,w]}\n" ++
            s!"    {vb} = stablehlo.broadcast_in_dim {varN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vb}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,oc,h,w]}\n" ++
            s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "relu6", [], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            -- byte-for-byte `.relu6F`'s emit, width from the descriptor's `n`.
            let z ← fresh; let six ← fresh; let mx ← fresh; let o ← fresh
            pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty d}\n" ++
                  s!"    {six} = stablehlo.constant dense<6.0> : {ty d}\n" ++
                  s!"    {mx} = stablehlo.maximum {r}, {z} : {ty d}\n" ++
                  s!"    {o} = stablehlo.minimum {mx}, {six} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "maxPool", [], [_N, c, h, w] => do
          -- byte-for-byte `.maxPoolF`'s emit; dims from the descriptor, batch from `B`.
          let xn ← fresh; let ninf ← fresh; let pp ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
            s!"    {pp} = \"stablehlo.reduce_window\"({xn}, {ninf}) (" ++ "{\n" ++
            "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
            "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
            "        stablehlo.return %pm : tensor<f32>\n" ++
            "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
            s!" : ({ty [B,c,2*h,2*w]}, tensor<f32>) -> {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {pp} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⭐ The 3×3/s2 stem pool, forward — both index conventions, ONE text writer
      -- (`maxPool3s2FwdText`), so the per-example and batched forms cannot drift.
      | "maxPool3s2", [], [c, h, w] => do
          let xn ← fresh; let ninf ← fresh; let pp ← fresh; let o ← fresh
          pure (maxPool3s2FwdText B c h w r xn ninf pp o, o :: st)
      | "maxPool3s2", [], [_N, c, h, w] => do
          let xn ← fresh; let ninf ← fresh; let pp ← fresh; let o ← fresh
          pure (maxPool3s2FwdText B c h w r xn ninf pp o, o :: st)
      | "maxPool3s2BackP", [xN], [c, h, w] => do
          let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
          pure (maxPool3s2BackText B c h w xN r xr dr z scn o, o :: st)
      | "maxPool3s2BackP", [xN], [_N, c, h, w] => do
          let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
          pure (maxPool3s2BackText B c h w xN r xr dr z scn o, o :: st)
      | "maxPoolBackP", [xN], [_N, c, h, w] => do
          -- byte-for-byte `.maxPoolBack`'s emit. NOTE the region block arguments %sa/%sb/%sc/%sd
          -- are HARDCODED here, so they are reserved SSA names: a top-level value of the same
          -- name is a redefinition error, and it only surfaces at XLA compile time.
          let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {scn} = \"stablehlo.select_and_scatter\"({xr}, {dr}, {z}) (" ++ "{\n" ++
            "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
            "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
            "        stablehlo.return %sge : tensor<i1>\n" ++
            "    }, " ++ "{\n" ++
            "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
            "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
            "        stablehlo.return %ss : tensor<f32>\n" ++
            "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
            s!" : ({ty [B,c,2*h,2*w]}, {ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {scn} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      | "convBiasSgd", [bN, lrS], [_N, oc, h, w] => do
          -- byte-for-byte `.convBiasSgd`'s emit. Stride-independent, so the strided peer
          -- shares this case (both `skel` to the same Raw).
          let dr ← fresh; let z ← fresh; let g ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {g} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
            s!"    {sB} = stablehlo.multiply {g}, {lB} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [oc]}\n", o :: st)
      | "selectPosP", [x], [_N, n] => do
          let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
            -- byte-for-byte `.selectPos`'s emit, width from the descriptor's `n`.
            let z ← fresh; let msk ← fresh; let o ← fresh
            pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty d}\n" ++
              s!"    {msk} = stablehlo.compare GT, {x}, {z} : ({ty d}, {ty d}) -> {tyI1 d}\n" ++
              s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 d}, {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "selectMidP", [x], [_N, n] => do
          let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
            -- byte-for-byte `.selectMid`'s emit, width from the ctor's `n`. Two-sided kink, so
            -- two compares AND-ed — unlike `selectPosP`'s single GT.
            let z ← fresh; let six ← fresh; let g0 ← fresh; let l6 ← fresh
            let msk ← fresh; let o ← fresh
            pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty d}\n" ++
              s!"    {six} = stablehlo.constant dense<6.0> : {ty d}\n" ++
              s!"    {g0} = stablehlo.compare GT, {x}, {z} : ({ty d}, {ty d}) -> {tyI1 d}\n" ++
              s!"    {l6} = stablehlo.compare LT, {x}, {six} : ({ty d}, {ty d}) -> {tyI1 d}\n" ++
              s!"    {msk} = stablehlo.and {g0}, {l6} : {tyI1 d}\n" ++
              s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 d}, {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "dropPathP", [mN], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            -- ▶ STOCHASTIC DEPTH: the per-SAMPLE residual-branch scale
            -- (`planning/archive/stochastic_depth.md`). `mN` is a graph INPUT of type `tensor<Bxf32>` — one
            -- value per EXAMPLE, computed on the host — and `dims = [0]` is what makes it the
            -- reference's `(B, 1, …, 1)` mask: every position within an example is scaled
            -- identically, every example independently. Emitting a `tensor<B×n>` scale instead
            -- typechecks, compiles and trains, and is per-ELEMENT dropout — a different regulariser.
            -- ⚠ NO BAKED `1/keep`. The driver folds the inversion into the supplied value
            -- (`bernoulli(keep_i)/keep_i` at train, `1.0` at eval), which is what makes the ones-scale
            -- forward the EXACT identity and lets this op be emitted in the forward too — keeping the
            -- `forward ⊂ train-step` prefix audit alive. See `Proofs.dropPath`'s note on why a baked
            -- constant and that audit cannot both hold.
            let mb ← fresh; let o ← fresh
            pure (s!"    {mb} = stablehlo.broadcast_in_dim {mN}, dims = [0] : ({ty [B]}) -> {ty d}\n" ++
              s!"    {o} = stablehlo.multiply {mb}, {r} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "sigmoidP", [], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            -- σ(z) at the batched shape, for BCE-with-logits' cotangent `(σ(z) − t)/(B·K)`.
            -- ⚠ ONE op, and `stablehlo.logistic` is the same primitive `sigmoidF` emits — the
            -- difference is only the shape it is emitted at.
            let o ← fresh
            pure (s!"    {o} = stablehlo.logistic {r} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "dropoutP", [mN], [_N, n] => do
          -- ⚠⚠ NEVER LIFTED to the operand's recorded 4-D shape (`liftPointwise`), unlike the other
          -- pointwise ops: the mask is a graph INPUT whose type is fixed at `tensor<B×n>`, so a 4-D
          -- emission types `mN` two ways and the artifact does not parse. MobileNetV4's head relu
          -- carries `[1280,1,1]` from its 1×1 BN and hit exactly that (2026-09-25); EfficientNet's
          -- GAP output has no recorded shape, so its renders are unchanged.
          let (txt4, res4) ← (fun r d => do
            -- ▶ CLASSIFIER DROPOUT (`recipe_gaps.md` gap C): the per-ELEMENT inverted mask, applied
            -- immediately before the classifier dense. `mN` is a graph INPUT of type
            -- `tensor<B×n×f32>` — one value per (example, feature), computed on the host.
            --
            -- ⚠⚠ **NO `broadcast_in_dim`, AND THAT ABSENCE IS THE WHOLE CLAIM.** The mask already
            -- has the value's shape, because the reference draws `bernoulli(key, keep, x.shape)`
            -- (`emitForward`'s classifier dropout in `jax/Jax/Codegen.lean`) rather than the `(B, 1, …, 1)` shape stochastic depth
            -- uses. A `dims = [0]` broadcast off a `tensor<B>` input here typechecks, compiles, runs,
            -- descends — and is stochastic depth on the classifier, a different regulariser. That is
            -- `dropPathP`'s warning read backwards, and `tests/TestBatchedEmitTie.lean` pins both
            -- directions: that one asserts the broadcast is PRESENT, this one that it is ABSENT.
            -- ⚠ NO BAKED `1/keep`, for `dropPathP`'s reason exactly: the driver folds the inversion
            -- into the supplied mask, so the ones-mask forward is the exact identity and this op can
            -- be emitted in the forward artifact without rescaling eval.
            let o ← fresh
            pure (s!"    {o} = stablehlo.multiply {mN}, {r} : {ty d}\n", o)) r [B, n]
          pure (txt4, res4 :: st)
      | "swishBackP", [x], [_N, n] => do
          let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
            -- byte-for-byte `.swishBack`'s emit, width from the descriptor's `n`.
            let s ← fresh; let one ← fresh; let om ← fresh; let xom ← fresh
            let inr ← fresh; let sp ← fresh; let o ← fresh
            pure (s!"    {s} = stablehlo.logistic {x} : {ty d}\n" ++
                  s!"    {one} = stablehlo.constant dense<1.0> : {ty d}\n" ++
                  s!"    {om} = stablehlo.subtract {one}, {s} : {ty d}\n" ++
                  s!"    {xom} = stablehlo.multiply {x}, {om} : {ty d}\n" ++
                  s!"    {inr} = stablehlo.add {one}, {xom} : {ty d}\n" ++
                  s!"    {sp} = stablehlo.multiply {s}, {inr} : {ty d}\n" ++
                  s!"    {o} = stablehlo.multiply {r}, {sp} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "sigmoidBackP", [x], [_N, n] => do
          let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
            -- byte-for-byte `.sigmoidBack`'s emit, width from the descriptor's `n`.
            let s ← fresh; let one ← fresh; let om ← fresh; let sp ← fresh; let o ← fresh
            pure (s!"    {s} = stablehlo.logistic {x} : {ty d}\n" ++
                  s!"    {one} = stablehlo.constant dense<1.0> : {ty d}\n" ++
                  s!"    {om} = stablehlo.subtract {one}, {s} : {ty d}\n" ++
                  s!"    {sp} = stablehlo.multiply {s}, {om} : {ty d}\n" ++
                  s!"    {o} = stablehlo.multiply {r}, {sp} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "softmaxRow", [], [_N, m, n] => do
          -- byte-for-byte `.softmaxRowF`'s emit. `m` is rows PER EXAMPLE (it always
          -- was); the batch is `_N` on the proof side and `B` in the emit.
          let xn ← fresh; let z ← fresh; let e ← fresh; let s ← fresh; let sb ← fresh
          let dv ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {e} = stablehlo.exponential {xn} : {ty [B,m,n]}\n" ++
            s!"    {s} = stablehlo.reduce({e} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {dv} = stablehlo.divide {e}, {sb} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {dv} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      -- ── the five ViT/ConvNeXt row/pointwise descriptors (§0.2 ▶2). Each body is a BYTE-FOR-BYTE
      --    copy of its descriptor-less peer's, with the width read from the descriptor's `m`/`n`
      --    instead of the SHlo index — which is the entire content of the batched-index move on
      --    the emit side. `tests/TestBatchedEmitTie.lean` ties each pair, so "byte-for-byte" is
      --    checked rather than intended.
      | "geluBackP", [x], [_N, n] => do
          let (txt4, res4) ← liftPointwise2 B n r x fun r x d => do
            -- byte-for-byte `.geluBack`'s emit, width from the descriptor's `n`.
            let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
            let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
            let chalf ← fresh; let term1 ← fresh; let t2 ← fresh; let omt2 ← fresh
            let hx ← fresh; let hxo ← fresh; let c3b ← fresh; let a3x2 ← fresh
            let in2 ← fresh; let up ← fresh; let term2 ← fresh; let gp ← fresh; let o ← fresh
            pure (s!"    {x2} = stablehlo.multiply {x}, {x} : {ty d}\n" ++
                  s!"    {x3} = stablehlo.multiply {x2}, {x} : {ty d}\n" ++
                  s!"    {ck} = stablehlo.constant dense<0.044715> : {ty d}\n" ++
                  s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty d}\n" ++
                  s!"    {inn} = stablehlo.add {x}, {kx3} : {ty d}\n" ++
                  s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty d}\n" ++
                  s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty d}\n" ++
                  s!"    {t} = stablehlo.tanh {u} : {ty d}\n" ++
                  s!"    {one} = stablehlo.constant dense<1.0> : {ty d}\n" ++
                  s!"    {opt} = stablehlo.add {one}, {t} : {ty d}\n" ++
                  s!"    {chalf} = stablehlo.constant dense<0.5> : {ty d}\n" ++
                  s!"    {term1} = stablehlo.multiply {chalf}, {opt} : {ty d}\n" ++
                  s!"    {t2} = stablehlo.multiply {t}, {t} : {ty d}\n" ++
                  s!"    {omt2} = stablehlo.subtract {one}, {t2} : {ty d}\n" ++
                  s!"    {hx} = stablehlo.multiply {chalf}, {x} : {ty d}\n" ++
                  s!"    {hxo} = stablehlo.multiply {hx}, {omt2} : {ty d}\n" ++
                  s!"    {c3b} = stablehlo.constant dense<0.134145> : {ty d}\n" ++
                  s!"    {a3x2} = stablehlo.multiply {c3b}, {x2} : {ty d}\n" ++
                  s!"    {in2} = stablehlo.add {one}, {a3x2} : {ty d}\n" ++
                  s!"    {up} = stablehlo.multiply {csqrt}, {in2} : {ty d}\n" ++
                  s!"    {term2} = stablehlo.multiply {hxo}, {up} : {ty d}\n" ++
                  s!"    {gp} = stablehlo.add {term1}, {term2} : {ty d}\n" ++
                  s!"    {o} = stablehlo.multiply {r}, {gp} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "lnRowBackP", [gN, xN, epsStr], [_N, m, n] => do
          -- byte-for-byte `.lnRowBack`'s emit; `m` is rows PER EXAMPLE.
          let dn ← fresh; let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
          let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
          let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
          let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
          let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
          let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o0 ← fresh; let o ← fresh
          pure (
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
            s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
            s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
            s!"    {dxh} = stablehlo.multiply {gb}, {dn} : {ty [B,m,n]}\n" ++
            s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,m,n]}\n" ++
            s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,m,n]}\n" ++
            s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,m,n]}\n" ++
            s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,m,n]}\n" ++
            s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {o0} = stablehlo.multiply {sN}, {i2} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {o0} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      | "dotOutP", [w], [_N, m, n] => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.dot_general {r}, {w}, contracting_dims = [1] x [1], " ++
                s!"precision = [DEFAULT, DEFAULT] : ({ty [B,n]}, {ty [m,n]}) -> {ty [B,m]}\n", o :: st)
      | "expeP", [], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            let o ← fresh
            pure (s!"    {o} = stablehlo.exponential {r} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "softmaxDivP", [], [_N, n] => do
          -- byte-for-byte `.softmaxDiv`'s emit — which already reduced over `dimensions = [1]`,
          -- i.e. per example. It is the DEN that this descriptor fixes.
          let z ← fresh; let s ← fresh; let sb ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {s} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
            s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.divide {r}, {sb} : {ty [B,n]}\n", o :: st)
      | "layerScaleChP", [gN], [_N, c, h, w'] => do
          let xn ← fresh; let gb ← fresh; let m ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
                s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
                s!"    {m} = stablehlo.multiply {xn}, {gb} : {ty [B,c,h,w']}\n" ++
                s!"    {o} = stablehlo.reshape {m} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed result reads
      -- identically and compiles to pure f32 — measured on THIS shape (4×4/s4) before the op was
      -- written, so stride 4 buys no exemption from §9.2 any more than grouping did.
      -- ⚠⚠ The pad is `convStride4P`'s `(k-1)/2 − 1`, NOT `convBf16`'s `(k-1)/2`. At the 4×4 stem
      -- that is `[[0,0]]`. The two spellings produce the same output SIZE, so nothing structural
      -- separates them — do not "tidy" this to match the other bf16 convs.
      | "convStride4P", [w, b], [_N, ic, oc, h, w', kH, kW] | "convStride4PBf16", [w, b], [_N, ic, oc, h, w', kH, kW] => do
          -- byte-for-byte `.flatConvStride4F`'s emit, including its ⚠ pad-one-less rule: the
          -- denotation reads the SAME conv at the offset-1 positions 4i+1, so the emitted pad is
          -- (k-1)/2 − 1 — for the 4×4 stem that is 0, the paper's left-aligned window.
          let pH := (kH - 1) / 2 - 1; let pW := (kW - 1) / 2 - 1
          let xn ← fresh
          let (cs, cv) ← emitContract (lowOf tag) xn w [B,ic,2*(2*h),2*(2*w')] [oc,ic,kH,kW] [B,oc,h,w'] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [4, 4], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let bb ← fresh; let ob ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*(2*h))*(2*(2*w'))]}) -> {ty [B,ic,2*(2*h),2*(2*w')]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
            s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
      | "gelu", [], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
            let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
            let chalf ← fresh; let hx ← fresh; let o ← fresh
            pure (s!"    {x2} = stablehlo.multiply {r}, {r} : {ty d}\n" ++
                  s!"    {x3} = stablehlo.multiply {x2}, {r} : {ty d}\n" ++
                  s!"    {ck} = stablehlo.constant dense<0.044715> : {ty d}\n" ++
                  s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty d}\n" ++
                  s!"    {inn} = stablehlo.add {r}, {kx3} : {ty d}\n" ++
                  s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty d}\n" ++
                  s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty d}\n" ++
                  s!"    {t} = stablehlo.tanh {u} : {ty d}\n" ++
                  s!"    {one} = stablehlo.constant dense<1.0> : {ty d}\n" ++
                  s!"    {opt} = stablehlo.add {one}, {t} : {ty d}\n" ++
                  s!"    {chalf} = stablehlo.constant dense<0.5> : {ty d}\n" ++
                  s!"    {hx} = stablehlo.multiply {chalf}, {r} : {ty d}\n" ++
                  s!"    {o} = stablehlo.multiply {hx}, {opt} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "transposeP", [], [_N, m, n] => do
          let xn ← fresh; let t ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {t} = stablehlo.transpose {xn}, dims = [0, 2, 1] : ({ty [B,m,n]}) -> {ty [B,n,m]}\n" ++
            s!"    {o} = stablehlo.reshape {t} : ({ty [B,n,m]}) -> {ty [B, n*m]}\n", o :: st)
      | "lnRowP", [gN, bN, epsStr], [_N, m, n] => do
          let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
          let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
          let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
          let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let ob ← fresh
          let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
            s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
            s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
            s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,m,n]}\n" ++
            s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      | "rowScaleP", [gN], [_N, m, n] => do
          let xn ← fresh; let gb ← fresh; let mu ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
            s!"    {mu} = stablehlo.multiply {xn}, {gb} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {mu} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      | "rowBiasP", [bN], [_N, m, n] => do
          let xn ← fresh; let bb ← fresh; let ad ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
            s!"    {ad} = stablehlo.add {xn}, {bb} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {ad} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      -- ⚠⚠ bf16 operands, **bf16-TYPED result**, convert back — the CONV shape, applied to a dot.
      -- §9.2 established that `dot_general` reaches the tensor cores with EITHER result type and
      -- read that as "the result type is inert for dot". It is inert for CORRECTNESS and it is not
      -- inert for SPEED: an f32 result makes the gemm write twice the bytes and take a worse
      -- epilogue. Measured on ViT's own MLP chain (§20.1): f32-result 1.18×, bf16-result **1.60×**.
      -- ▶ So the convert-back is not "a node that buys nothing" — it is most of the win.
      | "denseRowBackP", [wN], [_N, rows, a, c] | "denseRowBackPBf16", [wN], [_N, rows, a, c] => do
          -- byte-for-byte `.denseRowBack`'s emit; `rows` is per-example rows.
          let dn ← fresh
          let (cs, dg) ← emitContract (lowOf tag) dn wN [B,rows,c] [a,c] [B,rows,a] fun lhs rhs =>
              s!"stablehlo.dot_general {lhs}, {rhs}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT]"
          let o ← fresh
          pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, rows*c]}) -> {ty [B,rows,c]}\n" ++
            cs ++
            s!"    {o} = stablehlo.reshape {dg} : ({ty [B,rows,a]}) -> {ty [B, rows*a]}\n", o :: st)
      -- ══ ViT increment 1: the six batch-invariant forms. Every one is byte-for-byte its
      --    per-example peer's emit with the TOKEN count read off the tag (`tk`) instead of off the
      --    SHlo index — which is the whole content of the move, since `B` was always `pretty`'s.
      --    ⚠ `_N` (the batch) is deliberately unused in all six: an emit that read it would be
      --    reintroducing the conflation. `tests/TestBatchedEmitTie.lean` pins each against its peer.
      -- ⚠⚠ bf16-TYPED result then convert back, per `denseRowBackPBf16`'s note — and this is the op
      -- that carries ViT, six sites per block × 12 blocks. The BIAS is added after the convert, in
      -- f32, which is what `den`'s outer `rnd` sits inside of.
      | "denseRowP", [wN, bN], [_N, tk, a, c] | "denseRowPBf16", [wN, bN], [_N, tk, a, c] => do
          let xn ← fresh
          let (cs, dg) ← emitContract (lowOf tag) xn wN [B,tk,a] [a,c] [B,tk,c] fun lhs rhs =>
              s!"stablehlo.dot_general {lhs}, {rhs}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT]"
          let bb ← fresh; let ob ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, tk*a]}) -> {ty [B,tk,a]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [c]}) -> {ty [B,tk,c]}\n" ++
            s!"    {ob} = stablehlo.add {dg}, {bb} : {ty [B,tk,c]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,tk,c]}) -> {ty [B, tk*c]}\n", o :: st)
      -- ⚠⚠ **THE CONV SHAPE, NOT THE DOT SHAPE** — bf16 operands, **bf16-TYPED convolution
      -- result**, convert back. ViT's patchify stem is the one op in this net that is a
      -- `convolution`, and giving it an f32-typed result folds the whole thing to pure f32
      -- (measured standalone at this exact shape before the constructor existed, §17.2). Stride 16
      -- buys no exemption from §9.2 any more than grouping (§12.2) or stride 4 (§16.1) did.
      -- ▶ Everything after the convert-back — bias, transpose, CLS concat, position add — is
      -- byte-for-byte "patchEmbedP" and stays f32, which is what `patchEmbedFlatBf16`'s `den` says.
      | "patchEmbedP", [wN, bN, clsN, posN], [_N, ic, H, W, P, tk, D] | "patchEmbedPBf16", [wN, bN, clsN, posN], [_N, ic, H, W, P, tk, D] => do
          let hp := H / P; let wp := W / P
          let xn ← fresh
          let (cs, cv) ← emitContract (lowOf tag) xn wN [B,ic,H,W] [D,ic,P,P] [B,D,hp,wp] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [{P}, {P}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let bb ← fresh; let cb ← fresh
          let tr ← fresh; let tkn ← fresh; let clsb ← fresh; let cat ← fresh
          let pb ← fresh; let ob ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
            cs ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [D]}) -> {ty [B,D,hp,wp]}\n" ++
            s!"    {cb} = stablehlo.add {cv}, {bb} : {ty [B,D,hp,wp]}\n" ++
            s!"    {tr} = stablehlo.transpose {cb}, dims = [0, 2, 3, 1] : ({ty [B,D,hp,wp]}) -> {ty [B,hp,wp,D]}\n" ++
            s!"    {tkn} = stablehlo.reshape {tr} : ({ty [B,hp,wp,D]}) -> {ty [B,tk,D]}\n" ++
            s!"    {clsb} = stablehlo.broadcast_in_dim {clsN}, dims = [2] : ({ty [D]}) -> {ty [B,1,D]}\n" ++
            s!"    {cat} = stablehlo.concatenate {clsb}, {tkn}, dim = 1 : ({ty [B,1,D]}, {ty [B,tk,D]}) -> {ty [B,tk+1,D]}\n" ++
            s!"    {pb} = stablehlo.broadcast_in_dim {posN}, dims = [1, 2] : ({ty [tk+1,D]}) -> {ty [B,tk+1,D]}\n" ++
            s!"    {ob} = stablehlo.add {cat}, {pb} : {ty [B,tk+1,D]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,tk+1,D]}) -> {ty [B, (tk+1)*D]}\n", o :: st)
      | "clsSliceP", [], [_N, tk, D] => do
          let xn ← fresh; let sl ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, (tk+1)*D]}) -> {ty [B,tk+1,D]}\n" ++
            s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:1, 0:{D}] : ({ty [B,tk+1,D]}) -> {ty [B,1,D]}\n" ++
            s!"    {o} = stablehlo.reshape {sl} : ({ty [B,1,D]}) -> {ty [B,D]}\n", o :: st)
      | "clsPadP", [], [_N, tk, D] => do
          let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
          pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B,D]}) -> {ty [B,1,D]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, 0], high = [0, {tk}, 0], interior = [0, 0, 0] : ({ty [B,1,D]}, tensor<f32>) -> {ty [B,tk+1,D]}\n" ++
            s!"    {o} = stablehlo.reshape {pd} : ({ty [B,tk+1,D]}) -> {ty [B, (tk+1)*D]}\n", o :: st)
      | "headSliceP", [], [_N, tk, heads, d, hIdx] => do
          let xn ← fresh; let sl ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, tk*(heads*d)]}) -> {ty [B,tk,heads*d]}\n" ++
            s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:{tk}, {hIdx*d}:{(hIdx+1)*d}] : ({ty [B,tk,heads*d]}) -> {ty [B,tk,d]}\n" ++
            s!"    {o} = stablehlo.reshape {sl} : ({ty [B,tk,d]}) -> {ty [B, tk*d]}\n", o :: st)
      | "headPadP", [], [_N, tk, heads, d, hIdx] => do
          let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
          pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, tk*d]}) -> {ty [B,tk,d]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, {hIdx*d}], high = [0, 0, {(heads-1-hIdx)*d}], interior = [0, 0, 0] : ({ty [B,tk,d]}, tensor<f32>) -> {ty [B,tk,heads*d]}\n" ++
            s!"    {o} = stablehlo.reshape {pd} : ({ty [B,tk,heads*d]}) -> {ty [B, tk*(heads*d)]}\n", o :: st)
      -- ══ The un-fused BATCHED gradients: each is its `*SgdB` peer's emit with the SGD tail
      --    (const lr / multiply / subtract) removed, so the text is a byte-PREFIX of it. ══
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convWeightGrad", [xN], [_N, ic, oc, h, w, kH, kW] | "convWeightGradBf16", [xN], [_N, ic, oc, h, w, kH, kW] | "convWeightGradF8", [xN], [_N, ic, oc, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let (cs, raw) ← emitContract (lowOf tag) xt dt [ic,B,h,w] [oc,B,h,w] [ic,oc,kH,kW] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
            cs ++
            s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convStridedWeightGrad", [xN], [_N, ic, oc, h, w, kH, kW] | "convStridedWeightGradBf16", [xN], [_N, ic, oc, h, w, kH, kW] => do
          -- odd/even split via `sWGradGeom`; odd is byte-for-byte the old inline formula.
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let (cs, raw) ← emitContract (lowOf tag) xt dt [ic,B,2*h,2*w] [oc,B,extH,extW] [ic,oc,kH,kW] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            cs ++
            s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      -- ⭐ The XLA-`SAME` conv weight grad. Same `sWGradGeom` extents; only the correlation pad
      -- shifts by one (`loH-1`, `hiH+1`), so the saved input is read at `2·ho + 1 + kh - p`.
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "convStridedXlaWeightGrad", [xN], [_N, ic, oc, h, w, kH, kW] | "convStridedXlaWeightGradBf16", [xN], [_N, ic, oc, h, w, kH, kW] => do
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let (cs, raw) ← emitContract (lowOf tag) xt dt [ic,B,2*h,2*w] [oc,B,extH,extW] [ic,oc,kH,kW] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH-1}, {hiH+1}], [{loW-1}, {hiW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            cs ++
            s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back — measured on this exact
      -- shape (`[3,B,224,224]` × `[96,B,221,221]` → `[3,96,4,4]`) before the op was written.
      -- ⚠ Every geometry number is `convStride4WeightGrad`'s verbatim: the `interior = 3`
      -- upsample, the `4h−3` extent with NO trailing row, and the `lo = p−1` / `hi = kH−3−p`
      -- window. Only the four dtypes and the two converts move. ⚠ The `stablehlo.pad`'s zero stays
      -- f32 — it pads the cotangent BEFORE the cast, so it is an f32 tensor at that point.
      | "convStride4WeightGrad", [xN], [ic, oc, h, w, kH, kW] | "convStride4WeightGradBf16", [xN], [ic, oc, h, w, kH, kW] => do
          -- ConvNeXt's 4×4/s4 patchify weight grad. `flatConvStride4` decimates TWICE, so the
          -- cotangent is zero-upsampled with `interior = 3` (extent `4h−3`, no trailing row) and
          -- correlated VALID-style against the saved input at `4h`, giving `kH×kW`.
          -- The window offset: the stride-1 SAME conv is read at position `4i+1`, so a tap `kh`
          -- lands at `x[4i + 1 + kh − p]` with `p = (kH−1)/2` — hence `lo = p − 1`, and
          -- `hi = kH − 3 − p` makes the result exactly `kH` wide. At the 4×4 stem `p = 1`, so
          -- `[[0,0]]` and extent `4h−3` — byte-for-byte `ConvNeXtRender.patchWGrad`'s geometry.
          -- Nothing else in the kit is stride-4; this is exercised only at 4×4.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let extH := 4 * h - 3; let extW := 4 * w - 3
          let loH := pH - 1; let hiH := kH - 3 - pH
          let loW := pW - 1; let hiW := kW - 3 - pW
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh
          let dt ← fresh
          let (cs, raw) ← emitContract (lowOf tag) xt dt [ic,B,4*h,4*w] [oc,B,extH,extW] [ic,oc,kH,kW] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(4*h)*(4*w)]}) -> {ty [B,ic,4*h,4*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,4*h,4*w]}) -> {ty [ic,B,4*h,4*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            cs ++
            s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      | "convBiasGrad", [], [_N, oc, h, w] => do
          let dr ← fresh; let z ← fresh; let o ← fresh
          pure (
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n", o :: st)
      | "bnGammaGrad", [vN, es], [_N, oc, h, w] => do
          -- x̂ recomputed with μ/var over [0,2,3] — BATCH BN, not `bnGammaGrad`'s per-example [2,3].
          let xr ← fresh; let z ← fresh; let nf ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ep ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
          let dyr ← fresh; let dgp ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dgp} = stablehlo.multiply {dyr}, {xh} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reduce({dgp} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n", o :: st)
      | "bnBetaGrad", [], [_N, oc, h, w] => do
          let dyr ← fresh; let z ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n", o :: st)
      | "denseWeightGrad", [xN], [_N, a, c] => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,a]}, {ty [B,c]}) -> {ty [a,c]}\n", o :: st)
      | "denseBiasGrad", [], [_N, c] => do
          let z ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,c]}, tensor<f32>) -> {ty [c]}\n", o :: st)
      | "scale", [sS], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            let c ← fresh; let o ← fresh
            pure (s!"    {c} = stablehlo.constant dense<{sS}> : {ty d}\n" ++
                  s!"    {o} = stablehlo.multiply {r}, {c} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "shift", [sS], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            let c ← fresh; let o ← fresh
            pure (s!"    {c} = stablehlo.constant dense<{sS}> : {ty d}\n" ++
                  s!"    {o} = stablehlo.add {r}, {c} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "divConst", [sS], [_N, n] => do
          let (txt4, res4) ← liftPointwise B n r fun r d => do
            let c ← fresh; let o ← fresh
            pure (s!"    {c} = stablehlo.constant dense<{sS}> : {ty d}\n" ++
                  s!"    {o} = stablehlo.divide {r}, {c} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "bnStatsMean", [], [oc] => do
          -- μ, sliced off the packed `[μ ‖ σ²]` — under DP the all-reduced statistics.
          let o ← fresh
          pure (s!"    {o} = stablehlo.slice {r} [0:{oc}] : ({ty [oc+oc]}) -> {ty [oc]}\n", o :: st)
      | "bnStatsVar", [], [oc] => do
          -- σ², sliced off the packed `[μ ‖ σ²]`.
          let o ← fresh
          pure (s!"    {o} = stablehlo.slice {r} [{oc}:{oc+oc}] : ({ty [oc+oc]}) -> {ty [oc]}\n", o :: st)
      | "bnBatchMean", [], [_N, oc, h, w] => do
          -- μ_c = reduce[0,2,3](x) / (B·h·w). Numerically the `%{p}bnmu` the hand-written
          -- emitter divides out of its own BN fragment's `smr`.
          let xr ← fresh; let z ← fresh; let nf ← fresh; let smr ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [oc]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {o} = stablehlo.divide {smr}, {nf} : {ty [oc]}\n", o :: st)
      | "bnBatchVar", [], [_N, oc, h, w] => do
          -- var_c = reduce[0,2,3]((x−μ)²) / (B·h·w), μ recomputed inline — the biased (÷n)
          -- variance `bnVar` uses, matching `%{p}bnvar`.
          let xr ← fresh; let z ← fresh; let nfb ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let nf ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nfb} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nfb} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.divide {vsr}, {nf} : {ty [oc]}\n", o :: st)
      | "bnBatch", [gN, bN, es], [_N, oc, h, w] => do
          let xr ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh
          let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh
          let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
          let gb ← fresh; let btb ← fresh; let gx ← fresh; let o4 ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {btb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {gx} = stablehlo.multiply {xh}, {gb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o4} = stablehlo.add {gx}, {btb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {o4} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | t, [gN, xN, es], [_N, oc, h, w] =>
          -- bnBatchBack / bnBatchLABack: the 3-term true-BN input-VJP. Self-contained
          -- recompute of x̂/istd from the saved BN input `xN` + γ `gN` + ε `es`
          -- (mnv2 pattern), then dx = (istd/nf)·(nf·(γ⊙dy) − Σ(γ⊙dy) − x̂·Σ(x̂·γ⊙dy)).
          -- `r` is the upstream cotangent dy. (dγ/dβ are param grads, not here.)
          if t == "bnBatchBack" || t == "bnBatchLABack" then do
            let xr ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh
            let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh
            let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
            let gb ← fresh; let dyr ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
            let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
            let xs ← fresh; let i2 ← fresh; let sN ← fresh; let dx4 ← fresh; let o ← fresh
            pure (
              s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
              s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
              s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
              s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
              s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
              s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
              s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
              s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
              s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {dxh} = stablehlo.multiply {gb}, {dyr} : {ty [B,oc,h,w]}\n" ++
              s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {xd} = stablehlo.multiply {xh}, {dxh} : {ty [B,oc,h,w]}\n" ++
              s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,oc,h,w]}\n" ++
              s!"    {xs} = stablehlo.multiply {xh}, {sxd} : {ty [B,oc,h,w]}\n" ++
              s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,oc,h,w]}\n" ++
              s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {dx4} = stablehlo.multiply {sN}, {i2} : {ty [B,oc,h,w]}\n" ++
              s!"    {o} = stablehlo.reshape {dx4} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
          else
            pure (s!"    // [EfficientNet Item B] batched {tag} {names} {info} — render TODO\n", r :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convBackBatched", [wN], [_N, ic, oc, h, w, kH, kW] | "convBackBatchedBf16", [wN], [_N, ic, oc, h, w, kH, kW] | "convBackBatchedF8", [wN], [_N, ic, oc, h, w, kH, kW] => do
          -- conv input-VJP: dx = conv(dy, reverse(W,[2,3])ᵀ), reversed+transposed
          -- kernel, stride 1, same-pad p. (1×1 in enet ⇒ p=0, reverse a no-op.)
          let p := (kH - 1) / 2
          let dyr ← fresh; let rev ← fresh; let wt ← fresh
          let (cs, dx) ← emitContract (lowOf tag) dyr wt [B,oc,h,w] [ic,oc,kH,kW] [B,ic,h,w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {wt} = stablehlo.transpose {rev}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            cs ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,h,w]}) -> {ty [B, ic*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      -- ⚠ ASYMMETRIC pad, exactly as the f32 peer above — the bf16 twin must not "tidy" it.
      | "convStridedBackBatched", [wN], [_N, ic, oc, h, w, kH, kW] | "convStridedBackBatchedBf16", [wN], [_N, ic, oc, h, w, kH, kW] => do
          -- stride-2 conv input-VJP: upsample dy (zero-interleave to 2h×2w) then the
          -- stride-1 conv input-VJP. Produces dx at the 2h×2w input resolution.
          -- ⚠⚠ ASYMMETRIC pad, matching `.convStridedBack`. The symmetric `[[p,p],[p,p]]` this
          -- emitted AGREES at every odd kernel (kH=3 ⇒ pH=1 ⇒ kH−1−pH=1) and is WRONG at even
          -- ones (kH=2 ⇒ [[0,0]] where the VJP needs [[1,0]]). §2f-bis fixed exactly this in the
          -- per-example emitter and it was never carried here, because no batched net had an
          -- even strided kernel until ConvNeXt's 2×2/s2 downsample. Found by the whole-net
          -- backward tie — 3 lines, at the 3 downsamples. Inert on every committed batched
          -- artifact, all of which are odd (R34 3×3, mnv2/enet 3×3 and 5×5).
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh; let wt ← fresh
          let (cs, dx) ← emitContract (lowOf tag) up wt [B,oc,2*h,2*w] [ic,oc,kH,kW] [B,ic,2*h,2*w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{kH - 1 - pH}, {pH}], [{kW - 1 - pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {wt} = stablehlo.transpose {rev}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            cs ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,2*h,2*w]}) -> {ty [B, ic*(2*h)*(2*w)]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "depthwiseBackBatched", [wN], [_N, c, h, w, kH, kW] | "depthwiseBackBatchedBf16", [wN], [_N, c, h, w, kH, kW] => do
          -- depthwise input-VJP: dx = depthwise_conv(dy, reverse(W,[2,3])), fgc=c,
          -- same-pad p (no transpose — one input channel per group).
          let p := (kH - 1) / 2
          let dyr ← fresh; let rev ← fresh
          let (cs, dx) ← emitContract (lowOf tag) dyr rev [B,c,h,w] [c,1,kH,kW] [B,c,h,w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}"
          let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            cs ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. The f32-result
      -- shape folds to pure f32, for grouped convolutions exactly as for ordinary ones
      -- (measured). ⚠ SYMMETRIC pad — this is the torchvision-origin variant, NOT `Xla`.
      | "depthwiseStridedBackBatched", [wN], [_N, c, h, w, kH, kW] | "depthwiseStridedBackBatchedBf16", [wN], [_N, c, h, w, kH, kW] => do
          -- stride-2 depthwise input-VJP: upsample dy then the stride-1 depthwise
          -- input-VJP. dx at the 2h×2w input resolution.
          let p := (kH - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh
          let (cs, dx) ← emitContract (lowOf tag) up rev [B,c,2*h,2*w] [c,1,kH,kW] [B,c,2*h,2*w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}"
          let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            cs ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      -- ⭐ The XLA-`SAME` depthwise input-VJP: conv pad shifts to `[p+1, p-1]`.
      -- ⚠⚠ **NOTE THE DIRECTION — it is the OPPOSITE of the two weight grads**, which shift to
      -- `[p-1, p+1]`. The kernel is REVERSED here (`stablehlo.reverse`, dims [2,3]), and that
      -- reversal flips the sign of the index shift. Deriving it "by symmetry" with the weight
      -- grads gives `[p-1, p+1]`, which type-checks, has the right shape, descends, and is WRONG
      -- — `scripts/xla_pad_op_check.py` caught exactly that (2.6e0 against both references) and
      -- a numeric sweep over (upsample phase, pad_low) pinned the true answer at both k=3 and
      -- k=5. Do not "fix" this to match its siblings. Total pad is `2p`, so the extent stays `2h`.
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      -- ⚠⚠ Keeps the `[p+1, p-1]` pad of its f32 peer — the OPPOSITE shift from the weight
      -- grads, because the kernel is reversed. Do not "fix" it to match its siblings.
      | "depthwiseStridedXlaBackBatched", [wN], [_N, c, h, w, kH, kW] | "depthwiseStridedXlaBackBatchedBf16", [wN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh
          let (cs, dx) ← emitContract (lowOf tag) up rev [B,c,2*h,2*w] [c,1,kH,kW] [B,c,2*h,2*w] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p+1}, {p-1}], [{p+1}, {p-1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}"
          let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            cs ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      | "seBackBatched", [w1, b1, w2, b2, vN], [_N, c, h, w, rr] => do
          -- SE backward: recompute the SE forward (GAP → dense W₁ b₁ → swish → dense
          -- W₂ b₂ → sigmoid gate) from the SE input `vN`, then the SE-input cotangent
          -- dx = gate⊙dse + GAP-adjoint(W₁ᵀ·swish'·W₂ᵀ·(gate·(1−gate))·Σ(x⊙dse)).
          -- `r` is the SE-output cotangent dse.
          let xr ← fresh; let z ← fresh; let sqs ← fresh; let sqnf ← fresh; let sq ← fresh
          let exd ← fresh; let exbb ← fresh; let ex ← fresh; let a1s ← fresh; let a1 ← fresh
          let h2d ← fresh; let h2bb ← fresh; let h2 ← fresh; let gate ← fresh
          let dser ← fresh; let gb2 ← fresh; let dleft ← fresh; let xdse ← fresh; let dgate ← fresh
          let one ← fresh; let omg ← fresh; let sg ← fresh; let dh2 ← fresh; let da1 ← fresh
          let dexs ← fresh; let dexone ← fresh; let dexom ← fresh; let dexxom ← fresh; let dexin ← fresh
          let dexsp ← fresh; let dex ← fresh; let dsq ← fresh; let dsqnf ← fresh; let dsqd ← fresh
          let dgsp ← fresh; let dds ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sqs} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {sqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {sq} = stablehlo.divide {sqs}, {sqnf} : {ty [B,c]}\n" ++
            s!"    {exd} = stablehlo.dot_general {sq}, {w1}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [c,rr]}) -> {ty [B,rr]}\n" ++
            s!"    {exbb} = stablehlo.broadcast_in_dim {b1}, dims = [1] : ({ty [rr]}) -> {ty [B,rr]}\n" ++
            s!"    {ex} = stablehlo.add {exd}, {exbb} : {ty [B,rr]}\n" ++
            s!"    {a1s} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {a1} = stablehlo.multiply {ex}, {a1s} : {ty [B,rr]}\n" ++
            s!"    {h2d} = stablehlo.dot_general {a1}, {w2}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [rr,c]}) -> {ty [B,c]}\n" ++
            s!"    {h2bb} = stablehlo.broadcast_in_dim {b2}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {h2} = stablehlo.add {h2d}, {h2bb} : {ty [B,c]}\n" ++
            s!"    {gate} = stablehlo.logistic {h2} : {ty [B,c]}\n" ++
            s!"    {dser} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {gb2} = stablehlo.broadcast_in_dim {gate}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dleft} = stablehlo.multiply {gb2}, {dser} : {ty [B,c,h,w]}\n" ++
            s!"    {xdse} = stablehlo.multiply {xr}, {dser} : {ty [B,c,h,w]}\n" ++
            s!"    {dgate} = stablehlo.reduce({xdse} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,c]}\n" ++
            s!"    {omg} = stablehlo.subtract {one}, {gate} : {ty [B,c]}\n" ++
            s!"    {sg} = stablehlo.multiply {gate}, {omg} : {ty [B,c]}\n" ++
            s!"    {dh2} = stablehlo.multiply {dgate}, {sg} : {ty [B,c]}\n" ++
            s!"    {da1} = stablehlo.dot_general {dh2}, {w2}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [rr,c]}) -> {ty [B,rr]}\n" ++
            s!"    {dexs} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {dexone} = stablehlo.constant dense<1.0> : {ty [B,rr]}\n" ++
            s!"    {dexom} = stablehlo.subtract {dexone}, {dexs} : {ty [B,rr]}\n" ++
            s!"    {dexxom} = stablehlo.multiply {ex}, {dexom} : {ty [B,rr]}\n" ++
            s!"    {dexin} = stablehlo.add {dexone}, {dexxom} : {ty [B,rr]}\n" ++
            s!"    {dexsp} = stablehlo.multiply {dexs}, {dexin} : {ty [B,rr]}\n" ++
            s!"    {dex} = stablehlo.multiply {da1}, {dexsp} : {ty [B,rr]}\n" ++
            s!"    {dsq} = stablehlo.dot_general {dex}, {w1}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [c,rr]}) -> {ty [B,c]}\n" ++
            s!"    {dsqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {dsqd} = stablehlo.divide {dsq}, {dsqnf} : {ty [B,c]}\n" ++
            s!"    {dgsp} = stablehlo.broadcast_in_dim {dsqd}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dds} = stablehlo.add {dleft}, {dgsp} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dds} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "bnGammaSgd", [gN, vN, es, lrS], [_N, oc, h, w] => do
          -- BN γ update: recompute x̂ from the BN input `vN`, dγ = reduce[0,2,3](dy⊙x̂),
          -- γ' = γ − lr·dγ. Output is the channel-shaped updated γ.
          let xr ← fresh; let z ← fresh; let nf ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ep ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
          let dyr ← fresh; let dgp ← fresh; let dg ← fresh; let lc ← fresh; let sc ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dgp} = stablehlo.multiply {dyr}, {xh} : {ty [B,oc,h,w]}\n" ++
            s!"    {dg} = stablehlo.reduce({dgp} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {lc} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
            s!"    {sc} = stablehlo.multiply {dg}, {lc} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.subtract {gN}, {sc} : {ty [oc]}\n", o :: st)
      | "bnBetaSgd", [bN, lrS], [_N, oc, h, w] => do
          -- BN β update: dβ = reduce[0,2,3](dy), β' = β − lr·dβ.
          let dyr ← fresh; let z ← fresh; let db ← fresh; let lc ← fresh; let sc ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {db} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {lc} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
            s!"    {sc} = stablehlo.multiply {db}, {lc} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sc} : {ty [oc]}\n", o :: st)
      | "denseWeightSgd", [xN, wN, lrS], [_N, a, c] => do
          -- dense weight update: dW = aᵀ·dy (dot_general contracts the batch), W' = W − lr·dW.
          let dW ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {dW} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,a]}, {ty [B,c]}) -> {ty [a,c]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [a,c]}\n" ++
            s!"    {sW} = stablehlo.multiply {dW}, {lW} : {ty [a,c]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [a,c]}\n", o :: st)
      | "denseBiasSgd", [bN, lrS], [_N, c] => do
          -- dense bias update: dβ = reduce[0](dy) (sum over batch), β' = β − lr·dβ.
          let z ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dB} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,c]}, tensor<f32>) -> {ty [c]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [c]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [c]}\n", o :: st)
      | "rowDenseWeightSgd", [xN, wN, lrS], [N, a, c] => do
          -- per-token (rowwise) dense weight grad: reshape activation/cotangent to the [B,N,·] token
          -- matrix, dW = Σ_{B,N} xᵀ·dy (contract batch×tokens [0,1]x[0,1]), W' = W − lr·dW.
          let xn ← fresh; let dn ← fresh; let dW ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, N*a]}) -> {ty [B,N,a]}\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dW} = stablehlo.dot_general {xn}, {dn}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [B,N,a]}, {ty [B,N,c]}) -> {ty [a,c]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [a,c]}\n" ++
            s!"    {sW} = stablehlo.multiply {dW}, {lW} : {ty [a,c]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [a,c]}\n", o :: st)
      | "rowDenseBiasSgd", [bN, lrS], [N, c] => do
          -- per-token dense bias grad: db = reduce[0,1](dy) over batch×tokens ([B,N,c] → [c]),
          -- b' = b − lr·db. (Also the vector-LN β reduce.)
          let z ← fresh; let dn ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dB} = stablehlo.reduce({dn} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,c]}, tensor<f32>) -> {ty [c]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [c]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [c]}\n", o :: st)
      | "patchEmbedBiasSgd", [bN, lrS], [N, c] => do
          -- patch-embed bias grad: slice the N patch tokens [1..N] from the embed cotangent (drop the
          -- CLS row 0), reduce[0,1] over batch×patches → [c], b' = b − lr·db.
          let z ← fresh; let dr ← fresh; let dsl ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, (N+1)*c]}) -> {ty [B, N+1, c]}\n" ++
            s!"    {dsl} = stablehlo.slice {dr} [0:{B}, 1:{N+1}, 0:{c}] : ({ty [B,N+1,c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dB} = stablehlo.reduce({dsl} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,c]}, tensor<f32>) -> {ty [c]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [c]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [c]}\n", o :: st)
      | "posEmbedSgd", [pN, lrS], [N, D] => do
          -- pos-embed grad: reshape the embed cotangent [B,(N+1)*D] → [B,N+1,D], reduce ONLY the batch
          -- axis [0] (KEEP all N+1 tokens) → [N+1,D], pos' = pos − lr·dpos. (identity pos-Jacobian.)
          let z ← fresh; let dr ← fresh; let dP ← fresh; let lP ← fresh; let sP ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
            s!"    {dP} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,N+1,D]}, tensor<f32>) -> {ty [N+1,D]}\n" ++
            s!"    {lP} = stablehlo.constant dense<{lrS}> : {ty [N+1,D]}\n" ++
            s!"    {sP} = stablehlo.multiply {dP}, {lP} : {ty [N+1,D]}\n" ++
            s!"    {o} = stablehlo.subtract {pN}, {sP} : {ty [N+1,D]}\n", o :: st)
      -- ── the un-fused transformer gradients ─────────────────────────────────────────────────
      -- Each is its `*Sgd` peer above with the trailing `constant lr / multiply / subtract` cut
      -- off, so the emitted text is a byte PREFIX of the fused one. `tests/TestBatchedEmitTie.lean`
      -- checks exactly that, which is the emit-side twin of the `*Sgd_eq_grad` theorems.
      -- ⚠ Dot shape. The contraction is over BOTH the batch and the token axis in one op, so the
      -- f32-typed result IS the accumulator for the whole reduction — which is what keeps this
      -- gradient out of §9.3's vacuity argument (a bf16 accumulate at this fan-in would be).
      | "rowDenseWeightGrad", [xN], [N, a, c] | "rowDenseWeightGradBf16", [xN], [N, a, c] => do
          let xn ← fresh; let dn ← fresh
          let (cs, dW) ← emitContract (lowOf tag) xn dn [B,N,a] [B,N,c] [a,c] (lowResult := false) fun lhs rhs =>
              s!"stablehlo.dot_general {lhs}, {rhs}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT]"
          pure (
            s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, N*a]}) -> {ty [B,N,a]}\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            cs, dW :: st)
      | "rowDenseBiasGrad", [], [N, c] => do
          let z ← fresh; let dn ← fresh; let dB ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dB} = stablehlo.reduce({dn} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,c]}, tensor<f32>) -> {ty [c]}\n", dB :: st)
      | "patchEmbedBiasGrad", [], [N, c] => do
          let z ← fresh; let dr ← fresh; let dsl ← fresh; let dB ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, (N+1)*c]}) -> {ty [B, N+1, c]}\n" ++
            s!"    {dsl} = stablehlo.slice {dr} [0:{B}, 1:{N+1}, 0:{c}] : ({ty [B,N+1,c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dB} = stablehlo.reduce({dsl} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,c]}, tensor<f32>) -> {ty [c]}\n", dB :: st)
      | "posEmbedGrad", [], [N, D] => do
          let z ← fresh; let dr ← fresh; let dP ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
            s!"    {dP} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,N+1,D]}, tensor<f32>) -> {ty [N+1,D]}\n", dP :: st)
      | "veclnGammaGrad", [xN, epsStr], [N, D] => do
          let x3 ← fresh; let d3 ← fresh
          let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
          let dg ← fresh
          pure (
            s!"    {x3} = stablehlo.reshape {xN} : ({ty [B, N*D]}) -> {ty [B,N,D]}\n" ++
            s!"    {d3} = stablehlo.reshape {r} : ({ty [B, N*D]}) -> {ty [B,N,D]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{D}.0> : {ty [B,N,D]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,N,D]}\n" ++
            s!"    {smr} = stablehlo.reduce({x3} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,N,D]}, tensor<f32>) -> {ty [B,N]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,N]}) -> {ty [B,N,D]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,N,D]}\n" ++
            s!"    {xc} = stablehlo.subtract {x3}, {mu} : {ty [B,N,D]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,N,D]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,N,D]}, tensor<f32>) -> {ty [B,N]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,N]}) -> {ty [B,N,D]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,N,D]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,N,D]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,N,D]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,N,D]}\n" ++
            s!"    {p} = stablehlo.multiply {d3}, {xh} : {ty [B,N,D]}\n" ++
            s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,D]}, tensor<f32>) -> {ty [D]}\n", dg :: st)
      -- ⚠⚠ **CONV shape** — the second of ViT's two convolutions, and the second place the result
      -- type is load-bearing. The pad/transpose preamble is byte-for-byte "patchEmbedWeightGrad";
      -- the two converts go on the CONVOLUTION's operands only, after the dilating pad, because
      -- that pad is exact data movement and casting before it would round the same values twice.
      -- ⚠ The convolution contracts the BATCH axis (`[ic,B,H,W] × [D,B,dilH,dilW]`), so the single
      -- bf16 store lands on the already-summed gradient — which is exactly where
      -- `patchEmbedWeightGradBBf16`'s `den` puts its outer `rnd`.
      | "patchEmbedWeightGrad", [xN], [ic, H, W, P, N, D] | "patchEmbedWeightGradBf16", [xN], [ic, H, W, P, N, D] => do
          let ph := H / P; let pw := W / P
          let dilH := H - (P - 1); let dilW := W - (P - 1)
          let zc ← fresh; let dtr ← fresh; let dsl ← fresh; let drs ← fresh; let dy3 ← fresh
          let u ← fresh; let xt ← fresh; let dt ← fresh
          let (cs, raw) ← emitContract (lowOf tag) xt dt [ic,B,H,W] [D,B,dilH,dilW] [ic,D,P,P] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
              "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
          let dw ← fresh
          pure (
            s!"    {zc} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dtr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
            s!"    {dsl} = stablehlo.slice {dtr} [0:{B}, 1:{N+1}, 0:{D}] : ({ty [B,N+1,D]}) -> {ty [B,N,D]}\n" ++
            s!"    {drs} = stablehlo.reshape {dsl} : ({ty [B,N,D]}) -> {ty [B,ph,pw,D]}\n" ++
            s!"    {dy3} = stablehlo.transpose {drs}, dims = [0, 3, 1, 2] : ({ty [B,ph,pw,D]}) -> {ty [B,D,ph,pw]}\n" ++
            s!"    {u} = stablehlo.pad {dy3}, {zc}, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, {P-1}, {P-1}] : ({ty [B,D,ph,pw]}, tensor<f32>) -> {ty [B,D,dilH,dilW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xN}, dims = [1, 0, 2, 3] : ({ty [B,ic,H,W]}) -> {ty [ic,B,H,W]}\n" ++
            s!"    {dt} = stablehlo.transpose {u}, dims = [1, 0, 2, 3] : ({ty [B,D,dilH,dilW]}) -> {ty [D,B,dilH,dilW]}\n" ++
            cs ++
            s!"    {dw} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,D,P,P]}) -> {ty [D,ic,P,P]}\n", dw :: st)
      | "convWeightSgd", [xN, wN, lrS], [_N, ic, oc, h, w, kH, kW] => do
          -- conv weight update via the transpose-trick wgrad (batch as the conv
          -- contraction), then W' = W − lr·dW. Same text as the per-example convWeightSgd.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
      | "convStridedWeightSgd", [xN, wN, lrS], [_N, ic, oc, h, w, kH, kW] => do
          -- stem 3×3 s2 weight: zero-upsample dy to 2h×2w then the transpose-trick wgrad.
          -- odd/even split via `sWGradGeom`; odd is byte-for-byte the old inline formula.
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
      -- The XLA-`SAME` peer: identical, with the weight-grad correlation pad shifted one
      -- (`loH-1`, `hiH+1`) so the saved input is read at `2*ho + 1 + kh - p`.
      | "convStridedXlaWeightSgd", [xN, wN, lrS], [_N, ic, oc, h, w, kH, kW] => do
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH-1}, {hiH+1}], [{loW-1}, {hiW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
      | "depthwiseWeightSgd", [xN, wN, lrS], [_N, c, h, w, kH, kW] => do
          -- depthwise weight: per-channel transpose-trick wgrad (batch_group_count=c).
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
      | "depthwiseStridedWeightSgd", [xN, wN, lrS], [_N, c, h, w, kH, kW] => do
          -- strided depthwise weight: upsample dy to 2h×2w then the per-channel wgrad.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
      -- ── the ConvNeXt five. Each is its `*Sgd` peer's emit with the const-lr / multiply /
      --    subtract tail cut off, so each render is a byte-PREFIX of the fused one. ──
      | "depthwiseBiasGrad", [], [c, h, w, _kH, _kW] => do
          -- depthwise bias grad: Σ_{batch,spatial} dy, per channel. Note the RESHAPE precedes the
          -- zero constant — that is `depthwiseBiasSgd`'s order, and the emit-prefix test in
          -- `tests/TestBatchedEmitTie.lean` fails if the two are emitted the other way round even
          -- though the MLIR would be equivalent. (It caught exactly that here.)
          let dr ← fresh; let z ← fresh; let db ← fresh
          pure (
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {db} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [c]}\n", db :: st)
      | "lnGammaGrad", [xN, epsStr], [n] => do
          -- scalar-LN γ grad: recompute x̂ from the saved LN input, dγ = Σ_{b,k} dy·x̂ → tensor<f32>.
          let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
          let dg ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
            s!"    {smr} = stablehlo.reduce({xN} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
            s!"    {xc} = stablehlo.subtract {xN}, {mu} : {ty [B,n]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
            s!"    {p} = stablehlo.multiply {r}, {xh} : {ty [B,n]}\n" ++
            s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n", dg :: st)
      | "lnBetaGrad", [], [n] => do
          -- scalar-LN β grad: dβ = Σ_{b,k} dy → tensor<f32> (rank-0, the scalar-LN param shape).
          let z ← fresh; let db ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {db} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n", db :: st)
      | "layerScaleChGammaGrad", [xN], [c, h, w] => do
          -- per-channel layer-scale γ grad: dγ_c = reduce[0,2,3](x ⊙ dy).
          let z ← fresh; let xr ← fresh; let dr ← fresh; let p ← fresh; let dg ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {p} = stablehlo.multiply {xr}, {dr} : {ty [B,c,h,w]}\n" ++
            s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [c]}\n", dg :: st)
      -- The PER-EXAMPLE depthwise weight grad — five nats, where the batched form below has six.
      -- Same emitted text (that emitter ignores its `N`), so this shares the body by construction.
      | "depthwiseWeightGrad", [xN], [c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "depthwiseWeightGrad", [xN], [_N, c, h, w, kH, kW] | "depthwiseWeightGradBf16", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let (cs, raw) ← emitContract (lowOf tag) xt dt [c,B,h,w] [c,B,h,w] [1,c,kH,kW] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}"
          let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            cs ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. The f32-result
      -- shape folds to pure f32, for grouped convolutions exactly as for ordinary ones
      -- (measured). ⚠ SYMMETRIC pad — this is the torchvision-origin variant, NOT `Xla`.
      | "depthwiseStridedWeightGrad", [xN], [_N, c, h, w, kH, kW] | "depthwiseStridedWeightGradBf16", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let (cs, raw) ← emitContract (lowOf tag) xt dt [c,B,2*h,2*w] [c,B,2*h,2*w] [1,c,kH,kW] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}"
          let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            cs ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      -- ⭐ The XLA-`SAME` depthwise weight grad — the same one-position shift.
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      -- ⚠ Keeps the `[p-1, p+1]` weight-grad pad — the opposite direction from the dgrad.
      | "depthwiseStridedXlaWeightGrad", [xN], [_N, c, h, w, kH, kW] | "depthwiseStridedXlaWeightGradBf16", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let (cs, raw) ← emitContract (lowOf tag) xt dt [c,B,2*h,2*w] [c,B,2*h,2*w] [1,c,kH,kW] fun lhs rhs =>
              s!"stablehlo.convolution({lhs}, {rhs})\n" ++
              "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
              s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH-1}, {pH+1}], [{pW-1}, {pW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
              "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}"
          let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            cs ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      | "seReduceB", [xN], [_N, c, h, w] => do
          -- SE gate cotangent: dgate = reduce[2,3](x ⊙ dy). `xN` = SE input, `r` = the
          -- SE-output cotangent dy. Output is the per-example per-channel gate cotangent
          -- [B,c] (= the broadcast-adjoint of the Hadamard x⊙dy). Feeds the SE param grads.
          let xr ← fresh; let dyr ← fresh; let z ← fresh; let xd ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {xd} = stablehlo.multiply {xr}, {dyr} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n", o :: st)
      | "gapBackBatched", [], [_N, c, h, w] => do
          -- GAP backward: broadcast the per-channel cotangent `r` ([B,c]) over the h×w
          -- grid and scale by 1/(h·w) — the `globalAvgPoolFlat` adjoint, batched.
          let bb ← fresh; let nf ← fresh; let dv ← fresh; let o ← fresh
          pure (
            s!"    {bb} = stablehlo.broadcast_in_dim {r}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c,h,w]}\n" ++
            s!"    {dv} = stablehlo.divide {bb}, {nf} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dv} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | _, _, _ =>
          pure (s!"    // [EfficientNet Item B] batched {tag} {names} {info} — backward render TODO\n", r :: st)
  | .batched2 tag names info, b :: a :: st =>
      -- Pointwise binary ops at the batched index. Byte-for-byte the `.addV`/`.sub`
      -- emits; the width is `info`'s per-example `n`, not the SHlo index `N·n`.
      -- ⚠ `names` used to be discarded here — every `batched2` was nameless. `bnSync` is the
      -- first that carries any (γ, β, ε), so the match took a third component (2026-09-20).
      match tag, names, info with
      | "addV", [], [_N, n] => do
          let (txt4, res4) ← liftPointwise2 B n a b fun a b d => do
            let o ← fresh
            pure (s!"    {o} = stablehlo.add {a}, {b} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      | "sub", [], [_N, n] => do
          let (txt4, res4) ← liftPointwise2 B n a b fun a b d => do
            let o ← fresh
            pure (s!"    {o} = stablehlo.subtract {a}, {b} : {ty d}\n", o)
          pure (txt4, res4 :: st)
      -- ⭐⭐ **The only NON-pointwise `batched2`, and the only ACTIVATION × ACTIVATION bf16 op in
      -- the kit** — SDPA's `QKᵀ` and `P·V`, plus the four backward matmuls. It rides this binary
      -- skeleton rather than aliasing `.matmulF` because its text differs; `info` is `[m, k, n]`
      -- (no batch), since `B` is `pretty`'s exactly as it is for every other case here.
      -- ⚠ Both operand converts are on VALUES, not on a weight — nothing in the emit cares, and
      -- neither does `dot_close_mixed`, which rounds both sides.
      | "bnBatchVarAt", [], [_N, oc, h, w] => do
          -- σ²_r + (μ_r − μ)²: the replica's TWO-PASS variance about its own mean, plus its
          -- mean's squared offset from the all-reduced global mean `b`. The replica mean of these
          -- is the global σ² exactly (`bnVar_shard_chan`) — no `E[x²] − μ²` anywhere.
          let xr ← fresh; let z ← fresh; let nf ← fresh; let smr ← fresh; let mu ← fresh
          let mub ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vr ← fresh
          let d ← fresh; let dd ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {a} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [oc]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {mu} = stablehlo.divide {smr}, {nf} : {ty [oc]}\n" ++
            s!"    {mub} = stablehlo.broadcast_in_dim {mu}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mub} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vr} = stablehlo.divide {vsr}, {nf} : {ty [oc]}\n" ++
            s!"    {d} = stablehlo.subtract {mu}, {b} : {ty [oc]}\n" ++
            s!"    {dd} = stablehlo.multiply {d}, {d} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.add {vr}, {dd} : {ty [oc]}\n", o :: st)
      | "bnPack", [], [oc] => do
          -- `[μ ‖ σ²]`: the two all-reduced `[oc]` statistics as the one operand the sync ops read.
          let o ← fresh
          pure (s!"    {o} = stablehlo.concatenate {a}, {b}, dim = 0 : ({ty [oc]}, {ty [oc]}) -> {ty [oc+oc]}\n", o :: st)
      | "bnSync", [gN, bN, es], [_N, oc, h, w] => do
          -- ⭐ Sync-BN forward: μ and σ² SLICED out of the packed `[oc+oc]` operand `b` (the
          -- two all-reduced statistics under DP), σ² used as it arrives. The
          -- tail from `istd` on is `bnBatch`'s text verbatim — only where the statistics come
          -- from differs, which is exactly the claim `bnSyncTensor4_at_own_stats` makes.
          let xr ← fresh; let mus ← fresh; let vs ← fresh; let mub ← fresh; let vb ← fresh
          let ep ← fresh; let ve ← fresh; let istd ← fresh
          let xc ← fresh; let xh ← fresh; let gb ← fresh; let btb ← fresh
          let gx ← fresh; let o4 ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {a} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mus} = stablehlo.slice {b} [0:{oc}] : ({ty [oc+oc]}) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.slice {b} [{oc}:{oc+oc}] : ({ty [oc+oc]}) -> {ty [oc]}\n" ++
            s!"    {mub} = stablehlo.broadcast_in_dim {mus}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vb} = stablehlo.broadcast_in_dim {vs}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vb}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mub} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {btb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {gx} = stablehlo.multiply {xh}, {gb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o4} = stablehlo.add {gx}, {btb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {o4} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "bnSyncDyStats", [gN, xN, es], [_N, oc, h, w] => do
          -- [μ ‖ σ² ‖ mdy ‖ mdyx]. The operand `b` IS `[μ ‖ σ²]`, so the pass-through is a
          -- concatenate of `b` itself — no re-slice. Both new entries are MEANS (÷ B·h·w),
          -- which is what lets one mean-collective carry them.
          let xr ← fresh; let mus ← fresh; let vs ← fresh; let mub ← fresh; let vb ← fresh
          let ep ← fresh; let ve ← fresh; let istd ← fresh
          let xc ← fresh; let xh ← fresh; let gb ← fresh; let dyr ← fresh; let dxh ← fresh
          let z ← fresh; let nf ← fresh; let sdxr ← fresh; let mdy ← fresh
          let xd ← fresh; let sxdr ← fresh; let mdyx ← fresh; let c2 ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mus} = stablehlo.slice {b} [0:{oc}] : ({ty [oc+oc]}) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.slice {b} [{oc}:{oc+oc}] : ({ty [oc+oc]}) -> {ty [oc]}\n" ++
            s!"    {mub} = stablehlo.broadcast_in_dim {mus}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vb} = stablehlo.broadcast_in_dim {vs}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vb}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mub} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {a} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dxh} = stablehlo.multiply {gb}, {dyr} : {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [oc]}\n" ++
            s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {mdy} = stablehlo.divide {sdxr}, {nf} : {ty [oc]}\n" ++
            s!"    {xd} = stablehlo.multiply {xh}, {dxh} : {ty [B,oc,h,w]}\n" ++
            s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {mdyx} = stablehlo.divide {sxdr}, {nf} : {ty [oc]}\n" ++
            s!"    {c2} = stablehlo.concatenate {mdy}, {mdyx}, dim = 0 : ({ty [oc]}, {ty [oc]}) -> {ty [oc+oc]}\n" ++
            s!"    {o} = stablehlo.concatenate {b}, {c2}, dim = 0 : ({ty [oc+oc]}, {ty [oc+oc]}) -> {ty [oc+oc+(oc+oc)]}\n", o :: st)
      | "bnSyncBack", [gN, xN, es], [_N, oc, h, w] => do
          -- dx = istd·(dx̂ − mdy − x̂·mdyx), all four statistics sliced out of the all-reduced
          -- `[4·oc]` operand. ⚠ The MEAN form: no `·B·h·w` then `÷B·h·w` round trip, because
          -- the reductions arrive already divided — see `bnSync_grad_input`.
          let xr ← fresh; let mus ← fresh; let vs ← fresh; let mdys ← fresh; let mdyxs ← fresh
          let mub ← fresh; let vb ← fresh; let mdyb ← fresh; let mdyxb ← fresh
          let ep ← fresh; let ve ← fresh; let istd ← fresh
          let xc ← fresh; let xh ← fresh; let gb ← fresh; let dyr ← fresh; let dxh ← fresh
          let i1 ← fresh; let xs ← fresh; let i2 ← fresh; let dx4 ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mus} = stablehlo.slice {b} [0:{oc}] : ({ty [oc+oc+(oc+oc)]}) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.slice {b} [{oc}:{oc+oc}] : ({ty [oc+oc+(oc+oc)]}) -> {ty [oc]}\n" ++
            s!"    {mdys} = stablehlo.slice {b} [{oc+oc}:{oc+oc+oc}] : ({ty [oc+oc+(oc+oc)]}) -> {ty [oc]}\n" ++
            s!"    {mdyxs} = stablehlo.slice {b} [{oc+oc+oc}:{oc+oc+oc+oc}] : ({ty [oc+oc+(oc+oc)]}) -> {ty [oc]}\n" ++
            s!"    {mub} = stablehlo.broadcast_in_dim {mus}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vb} = stablehlo.broadcast_in_dim {vs}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mdyb} = stablehlo.broadcast_in_dim {mdys}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mdyxb} = stablehlo.broadcast_in_dim {mdyxs}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vb}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mub} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {a} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dxh} = stablehlo.multiply {gb}, {dyr} : {ty [B,oc,h,w]}\n" ++
            s!"    {i1} = stablehlo.subtract {dxh}, {mdyb} : {ty [B,oc,h,w]}\n" ++
            s!"    {xs} = stablehlo.multiply {xh}, {mdyxb} : {ty [B,oc,h,w]}\n" ++
            s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,oc,h,w]}\n" ++
            s!"    {dx4} = stablehlo.multiply {istd}, {i2} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx4} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "matmulFBf16", [], [m, k, n] => do
          let (s, o) ← emitMatmul B (some tyBf16) a b m k n
          pure (s, o :: st)
      | "bnSyncGammaGrad", [xN, es], [_N, oc, h, w] => do
          -- dγ_c = Σ_{[0,2,3]} dy·x̂ with x̂ at the statistics sliced out of the packed operand
          -- `b` — `bnSync`'s prologue verbatim, then `bnGammaGrad`'s tail. A SUM, not a mean:
          -- this is a parameter gradient, and the parameter collective takes the mean over
          -- replicas of exactly these shard sums.
          let xr ← fresh; let mus ← fresh; let vs ← fresh; let mub ← fresh; let vb ← fresh
          let ep ← fresh; let ve ← fresh; let istd ← fresh
          let xc ← fresh; let xh ← fresh; let dyr ← fresh; let dgp ← fresh; let z ← fresh
          let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mus} = stablehlo.slice {b} [0:{oc}] : ({ty [oc+oc]}) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.slice {b} [{oc}:{oc+oc}] : ({ty [oc+oc]}) -> {ty [oc]}\n" ++
            s!"    {mub} = stablehlo.broadcast_in_dim {mus}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vb} = stablehlo.broadcast_in_dim {vs}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vb}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mub} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {a} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dgp} = stablehlo.multiply {dyr}, {xh} : {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({dgp} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n", o :: st)
      | _, _, _ => pure (s!"    // MALFORMED batched2 {tag} {info}\n", a :: st)
  | .allReduceMean R t ds, r :: st =>
      let (txt, o) := allReduceMeanText r ds t R
      pure (txt, o :: st)
  | _, st => pure ("    // MALFORMED token stream\n", st)

/-- Fold a token stream to accumulated `(code, result-name-stack)`. -/
def serializeToks (B : Nat) : List Tok → (String × List String) → StateM EmitS (String × List String)
  | [], acc           => pure acc
  | t :: ts, (code, st) => do
      let (c, st') ← emitTok B t st
      -- ⭐ The ONE place the shape table is written. Doing it here rather than in the 94 `emitTok`
      -- arms is what keeps them free of it — and it is why the table can be keyed by NAME at all:
      -- only here are the operand stack before and after the token both in hand.
      noteTokShapes t st st'
      serializeToks B ts (code ++ c, st')

/-- The bias's slot in a **return-name list**, gated the way `biasName` gates the operand: with
    `convBias := false` no bias SGD op is emitted, so the slot must LEAVE the list rather than carry
    the empty string the `if convBias then … else pure ("", "")` idiom hands back.

    ⚠ This exists because leaving it in is **silent twice over**. An empty name renders
    `return %a, , %b` — malformed text, but only the lowerer ever sees it; and the name list keeps
    its FULL length, so an arity `#guard` on the signature still passes. Measured on the first swap
    attempt: `mobilenetv2_train_step` at `convBias := false` returned 210 names (52 of them empty)
    against 160 types. Use this at every site where a `names := [...]` list is built from gated ops. -/
def biasSlot (convBias : Bool) (nm : String) : List String :=
  if convBias then [nm] else []

/-- The zero-bias constants the `convBias := false` render consumes, one per channel width used as
    a conv bias. Emitted once at the top of the body; XLA folds the resulting `add`. -/
def zeroBiasPrelude (convBias : Bool) (widths : List Nat) : String :=
  if convBias then "" else
    "    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s\n" ++
    "    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a\n" ++
    "    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.\n" ++
    String.join (widths.map (fun c =>
      s!"    %zb{c} = stablehlo.constant dense<0.0> : {ty [c]}\n"))

/-- Fixed-6-decimal float literal, so a computed smoothing constant emits in the SAME textual form
    the hand-written literals used and `nClasses = 10` re-renders byte-identical. -/
def fmt6 (x : Float) : String :=
  let neg := x < 0.0
  let n := ((if neg then -x else x) * 1000000.0 + 0.5).toUInt64.toNat
  let ip := n / 1000000
  let fp := n % 1000000
  let fs := (toString fp).leftpad 6 '0'
  (if neg then "-" else "") ++ toString ip ++ "." ++ fs

/-- Fixed-12-decimal float literal, for constants `fmt6` would destroy.

    ⚠ It exists because `fmt6` is not a formatting preference, it is a PRECISION CEILING, and small
    derived constants fall straight through it. Gradient accumulation's second-moment coefficient is
    `(1−β₂)/K²`; at K = 4 that is `6.25e-5`, which `fmt6` emits as `0.000063` — **0.8% wrong**, in a
    baked literal, in the optimizer, where nothing downstream would question it. Same class as §2k's
    hardcoded `0.010000` label-smoothing mass. `fmt6` stays the default so every committed artifact
    re-renders byte-identically; this is for constants that need the room. -/
def fmt12 (x : Float) : String :=
  let neg := x < 0.0
  let n := ((if neg then -x else x) * 1000000000000.0 + 0.5).toUInt64.toNat
  let ip := n / 1000000000000
  let fp := n % 1000000000000
  let fs := (toString fp).leftpad 12 '0'
  (if neg then "-" else "") ++ toString ip ++ "." ++ fs

/-- **The label-smoothing mass per class, α/K.** α = 0.1 throughout; K is `nClasses`.

    ⚠ **This was hardcoded `0.010000` — correct at K = 10 and WRONG at every other K**, and it sat
    in the COTANGENT, not just in the report-only `%loss`. At `nClasses = 1000` it made the smoothing
    term 100× too large: it removes 10.0 of probability mass instead of 0.1, i.e. a different
    objective, silently. Caught 2026-07-30 by the first ImageNet smoke run reporting loss ≈ 87 where
    1000-class CE at init must be ≈ ln(1000) = 6.9 — the number was implausible, and that is the only
    reason it surfaced. Nothing in the repo's proofs covers it: `α` is a *literal in emitted text*,
    which is exactly the carve-out class §5 says needs its own numeric check, and §2b's `%loss` bug
    is the standing precedent for it going wrong unnoticed. -/
def alphaOverK (nClasses : Nat) (alpha : Float := 0.1) : String :=
  fmt6 (alpha / nClasses.toFloat)

/-- `1 − α`, the ON-class weight of label-smoothed CE. Emitted beside `alphaOverK`, because the two
    always move together and splitting them is how one of them gets updated alone. -/
def oneMinusAlpha (alpha : Float := 0.1) : String := fmt6 (1.0 - alpha)

/-- **`1 − ρ`, the RMSProp mean-square mixing weight.** Derived from ρ, never written as a second
    literal beside it — the `oneMinusAlpha` precedent, and the K-constant lesson (§2k): *any
    emitted constant that depends on a hyperparameter must be DERIVED*, because the copy is what
    gets left behind when the original moves. Five copies of one label-smoothing constant were
    found across four nets in a single session for exactly this reason. -/
def oneMinusRho (rho : Float) : String := fmt6 (1.0 - rho)

/-- Which optimizer tail a whole-net render emits. `.adamw` is every net's committed default and
    reproduces the existing artifacts byte-identically; `.rmsprop` is what the MobileNetV2 and
    EfficientNet ImageNet references actually use (`planning/archive/recipe_gaps.md` v1.2).

    Lives here rather than in either renderer because **both** need it: a per-net copy of a
    two-constructor choice is the double-writer disease one level down, in code — the same argument
    `vitBackAll`/`enetBackAll` exist for (§2a-quater). Each renderer threads it through ONE
    traversal, so gate 1 applies for free: at `.adamw` every committed artifact must re-render
    byte-identical. -/
inductive OptKind where
  | adamw
  | rmsprop
deriving DecidableEq, Repr

/-- The RMSProp hyperparameters, as the JAX reference configs state them. `ρ`/`μ` are 0.9 on both
    nets that use this optimizer; **ε and wd are what differ**, and ε differs in the way that
    matters most (see `Proofs.rmsBufNext_eps_placement_at_zero`). -/
structure RmsHyper where
  /-- `rmspropDecay` — the running mean-square decay. -/
  rho : Float := 0.9
  /-- `momentum` — μ for the buffer on the normalised gradient. -/
  mu  : Float := 0.9
  /-- `rmspropEps` — ⚠ emitted INSIDE the square root (TensorFlow), not added to the root. -/
  eps : Float
  /-- COUPLED L2 (folded into the gradient), not AdamW's decoupled decay. -/
  wd  : Float

/-- ρ / (1−ρ) / μ / ε / wd as graph constants — the RMSProp peer of each renderer's `adamConsts`
    block. `%lr` stays a runtime `tensor<f32>` arg so one graph serves a whole LR schedule. -/
def rmsConstsBlock (h : RmsHyper) : String :=
  s!"    %rho = stablehlo.constant dense<{fmt6 h.rho}> : tensor<f32>\n" ++
  s!"    %orho = stablehlo.constant dense<{oneMinusRho h.rho}> : tensor<f32>\n" ++
  s!"    %mu = stablehlo.constant dense<{fmt6 h.mu}> : tensor<f32>\n" ++
  s!"    %eps = stablehlo.constant dense<{fmt6 h.eps}> : tensor<f32>\n" ++
  s!"    %wd = stablehlo.constant dense<{fmt6 h.wd}> : tensor<f32>\n"

/-- **MobileNetV2's RMSProp knobs** ([`jax/MainMobilenetV2Imagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainMobilenetV2Imagenet.lean)): ε = **1.0**. -/
def mnv2RmsHyper : RmsHyper := { eps := 1.0, wd := 4.0e-5 }

/-- **EfficientNet-B0's RMSProp knobs** ([`jax/MainEfficientNetImagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainEfficientNetImagenet.lean)): ε = **1e-3**. -/
def enetRmsHyper : RmsHyper := { eps := 1.0e-3, wd := 1.0e-5 }

-- ▶ The DRIVER-side half of the same two recipes — peak LR, exponential decay, warmup — is
-- `RmsSchedule` in `LeanMlir/VerifiedNetsCore.lean`, deliberately NOT here. Two reasons, and the second
-- is the load-bearing one: the four trainer entry points that read it would otherwise have to
-- import this whole proof module, and nothing that lives in this file can reach `rmsConstsBlock`
-- by accident. `%lr` is a runtime `tensor<f32>` argument precisely so one graph serves a whole
-- schedule; a learning rate must never become a graph constant.

-- ⚠ `fmt6` is a SIX-DECIMAL fixed-point formatter, so any hyperparameter below 5e-7 silently
-- renders as `0.000000` — a graph constant of zero, which for `wd` is "no weight decay" and for
-- `eps` is a divide-by-zero at a dead coordinate. Neither is a compile error and neither is
-- visible in a green build. These pin every constant these two nets actually emit; add a line
-- here before adding a third net's knobs.
#guard fmt6 mnv2RmsHyper.eps == "1.000000"
#guard fmt6 mnv2RmsHyper.wd  == "0.000040"
#guard fmt6 enetRmsHyper.eps == "0.001000"
#guard fmt6 enetRmsHyper.wd  == "0.000010"
#guard oneMinusRho 0.9 == "0.100000"
#guard fmt6 (0.9 : Float) == "0.900000"

/-- **`pretty`** — render an `SHlo` graph to StableHLO, now defined as
    `serialize ∘ toToks ∘ skel`: tokenize the graph (postorder), then print the
    tokens. The emitter shares ONE structured form with the parser, so the
    round-trip `parse (toToks (skel a)) = skel a` (StableHLOParse.lean) is about
    the very tokens this prints — the printer can't structurally drift. -/
def pretty (B : Nat) {k : Nat} (g : SHlo k) : StateM EmitS (String × String) := do
  let toks := toToks (skel g)
  let (code, st) ← serializeToks B toks ("", [])
  match st with
  | [r] => pure (code, r)
  | _   => pure (code, "%MALFORMED")

-- `seBlockSavedNames` names the four SE activations of the `seBlock` it is handed the counter of:
-- the squeeze is the `divide`, the reduce and excite pre-activations the bias `add`s, the swish
-- the `multiply` of the reduce pre-activation.
#guard
  let k0 := 7
  let (sq, ex, a1, h2) := seBlockSavedNames k0
  let txt := (Id.run ((pretty 2 (.batchOp (N := 2) (.seBlock (c := 8) (h := 3) (w := 3) (r := 2)
    "%W1" "%b1" "%W2" "%b2" (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0))
    (.operand "%x" (fun _ => 0)))).run' (k0, []))).1
  txt.contains s!"{sq} = stablehlo.divide " ∧ txt.contains s!"{ex} = stablehlo.add " ∧
    txt.contains s!"{a1} = stablehlo.multiply {ex}, " ∧ txt.contains s!"{h2} = stablehlo.add " ∧
    txt.contains s!"stablehlo.dot_general {sq}, %W1" ∧ txt.contains s!"stablehlo.dot_general {a1}, %W2" ∧
    txt.contains s!"stablehlo.logistic {h2} "

/-- **The rounding a render hands the bf16/fp8 ops: the identity.** A placeholder, exactly as a
    renderer's zero kernels are: a render produces TEXT, `skel` erases every ℝ payload before a
    token is emitted, and the emitted text is decided by the tag. The rounding-bearing `den` lives
    in the tie theorems, where `rnd` is arbitrary and the accuracy statement
    ([`Proofs/Float/*MixedFloatBridge.lean`](https://github.com/brettkoonce/lean4-mlir/tree/main/LeanMlir/Proofs/Float))
    instantiates it at bf16 round-to-nearest with `|rnd x − x| ≤ 2⁻⁸|x|`. A render that baked a
    concrete rounding here would be claiming the emitter knows about it, which it does not. -/
abbrev zrnd : ℝ → ℝ := fun r => r

/-- **The cross-replica gradient mean as `pretty` of the `allReduceMeanF` node** — the drop-in
    for `ViTRender.emitGradAllReduce` in every batched render (4d piece 2, 2026-09-07), measured
    byte-identical on every committed `*dp*` artifact. At `replicas ≤ 1` it emits nothing and
    threads the gradient's name, exactly as the text function did. The `R` operand graphs are
    all `.operand grad` at a zero placeholder, because a render is value-independent — `skel`
    erases values — while the family is what `den` sums over in the tie. -/
def prettyAllReduceMean (grad : String) (ds : List Nat) (t : String) (replicas : Nat) :
    StateM EmitS (String × String) :=
  if h : replicas ≤ 1 then pure ("", grad)
  else pretty 1 (.allReduceMeanF (n := ds.foldl (· * ·) 1) replicas (by omega) t ds
        (fun _ => .operand grad (fun _ => 0)))

/-- **One parameter's AdamW update, as `pretty` of the proven triple** — `adamMNextF`, `adamVNextF`
    and the decoupled-decay `adamWParamF` on the gradient `grad`, reading the graph constants
    `adamWConsts` binds, the step's `%bc1`/`%bc2`/`%lr`, and the decay operand `wdName` (`"%wdz"` for
    a timm no-decay parameter). Returns `(code, θ', m', v')`; every batched renderer's AdamW tail is
    this. ⚠ `wdName` and `den`'s `wd` must move together: both are 0 here only because every ℝ slot
    is a placeholder the emit ignores; if that changes, a no-decay site must pass `wd := 0` as well
    as `%wdz`, or the artifact and the denotation describe different optimizers. -/
def prettyAdamW (B : Nat) (nm : String) (ds : List Nat) (grad : String) (wdName : String := "%wd") :
    StateM EmitS (String × String × String × String) := do
  let z : Vec (ds.foldl (· * ·) 1) := fun _ => 0
  let gr : SHlo (ds.foldl (· * ·) 1) := .operand grad z
  let (cM, nM) ← pretty B (.adamMNextF s!"%{nm}m" "%b1" "%ob1" ds 0 z gr)
  let (cV, nV) ← pretty B (.adamVNextF s!"%{nm}v" "%b2" "%ob2" ds 0 z gr)
  let (cT, nT) ← pretty B (.adamWParamF s!"%{nm}" s!"%{nm}m" s!"%{nm}v" "%b1" "%ob1"
                    "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" wdName ds 0 0 0 0 0 0 0 z z z gr)
  pure (cM ++ cV ++ cT, nT, nM, nV)

/-- **The AdamW graph constants** — β₁ = 0.9, β₂ = 0.999, ε = 1e-8 and the baked decay `wdStr`
    (1e-4 is the Imagenette recipe every batched net shares; the ImageNet configs pass their own). -/
def adamWConsts (wdStr : String := "0.0001") : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  s!"    %wd = stablehlo.constant dense<{wdStr}> : tensor<f32>\n"

/-- Wrap a rendered single-result graph as a `func.func` module. -/
def renderModule (name argSig : String) (B retLen : Nat) (g : SHlo retLen) : String :=
  let (body, res) := (pretty B g).run' (0, [])
  "module @m {\n" ++ s!"  func.func @{name}({argSig}) -> {ty [B, retLen]} " ++ "{\n" ++
  body ++ s!"    return {res} : {ty [B, retLen]}\n" ++ "  }\n}\n"

/-- `@linear_fwd` rendered **from the verified AST**. -/
def linearFwdModuleV (B d₀ d₁ : Nat) (W : Mat d₀ d₁) (b : Vec d₁) (x : Vec d₀) : String :=
  renderModule "linear_fwd" s!"%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}" B d₁ (fwdGraph W b x)

/-- `@linear_back` rendered **from the verified AST**. -/
def linearBackModuleV (B d₀ d₁ : Nat) (W : Mat d₀ d₁) (dy : Vec d₁) : String :=
  renderModule "linear_back" s!"%dy: {ty [B,d₁]}, %W0: {ty [d₀,d₁]}" B d₀ (backGraph W dy)

/-- The full **`@linear_train_step`** rendered from the verified AST: forward +
    softmax-CE cotangent come from `pretty (lossCotGraph …)` (the `%onehot`
    operand value is `pretty`-irrelevant, so any placeholder renders the same
    text — at runtime `%onehot` is a graph input); the weight grad
    (`dot_general` over the batch axis), bias grad (`reduce`), and the SGD
    `multiply`/`subtract` updates are appended. Returns the two updated params.
    The verified-AST peer of `IRPrint.linearTrainStepModule`. -/
def linearTrainStepModuleV (B d₀ d₁ : Nat) (lr : String)
    (W : Mat d₀ d₁) (b : Vec d₁) (x : Vec d₀) : String :=
  let (body, dy) := (pretty B (lossCotGraph W b x (fun _ => 0))).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @linear_train_step(%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, " ++
  s!"%onehot: {ty [B,d₁]}) -> ({ty [d₀,d₁]}, {ty [d₁]}) " ++ "{\n" ++
  "    // ── forward + softmax-CE cotangent — rendered from the verified AST (lossCotGraph) ──\n" ++
  body ++
  s!"    // dy = {dy} = ⟦lossCotGraph⟧ = ∂CE/∂logits (lossCotGraph_isCEgrad)\n" ++
  "    // ── param grads: dW0 = x⊗dy, db0 = Σ_batch dy (wGrad/bGrad_is*Jacobian) ──\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %dW0 = stablehlo.dot_general %x, {dy}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,d₀]}, {ty [B,d₁]}) -> {ty [d₀,d₁]}\n" ++
  s!"    %db0 = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,d₁]}, tensor<f32>) -> {ty [d₁]}\n" ++
  "    // ── SGD update θ' = θ − lr·∇ (sgdW/sgdB_isCertifiedGradStep) ──\n" ++
  s!"    %lW0 = stablehlo.constant dense<{lr}> : {ty [d₀,d₁]}\n" ++
  s!"    %sW0 = stablehlo.multiply %dW0, %lW0 : {ty [d₀,d₁]}\n" ++
  s!"    %W0n = stablehlo.subtract %W0, %sW0 : {ty [d₀,d₁]}\n" ++
  s!"    %lb0 = stablehlo.constant dense<{lr}> : {ty [d₁]}\n" ++
  s!"    %sb0 = stablehlo.multiply %db0, %lb0 : {ty [d₁]}\n" ++
  s!"    %b0n = stablehlo.subtract %b0, %sb0 : {ty [d₁]}\n" ++
  s!"    return %W0n, %b0n : {ty [d₀,d₁]}, {ty [d₁]}\n" ++
  "  }\n}\n"

/-- **The linear train step rendered ENTIRELY from the verified AST.** Unlike
    `linearTrainStepModuleV` (forward via `pretty`, tail hand-written), here the
    *whole* module is `pretty` of denoted nodes: the cotangent (`lossCotGraph`,
    rendered once → shared `%dy`), then the two fused SGD ops `weightSgd`/`biasSgd`
    that consume `%dy`. So every emitted line is `pretty(provenNode)` and
    `LinearFold` proves the two outputs' `den` = the certified loss-descent
    SGD step. The `lr` ℝ / operand values are `skel`-erased (render is
    value-independent), so placeholders here render identically to the live graph
    the `den` theorems use. -/
def linTrainStepFaithfulV (B m n : Nat) (lrStr : String)
    (W : Mat m n) (b : Vec n) (x : Vec m) : String :=
  -- FULLY TIED: each SGD op consumes the proven `lossCotGraph` node DIRECTLY (not a
  -- name-pinned `.operand %dy <placeholder>`), so `den(output) = certified` is one composed
  -- theorem with the forward = the proven `fwdGraph` (nested inside `lossCotGraph`) — no
  -- SSA-name pin. The shared cotangent is rendered once per output (2× here); iree CSEs it.
  let act : StateM EmitS (String × String × String) := do
    let (wBody, wRes) ← pretty B (SHlo.weightSgd "%x" "%W0" lrStr x W 0 (lossCotGraph W b x (fun _ => 0)))
    let (bBody, bRes) ← pretty B (SHlo.biasSgd "%b0" lrStr b 0 (lossCotGraph W b x (fun _ => 0)))
    pure (wBody ++ bBody, wRes, bRes)
  let (body, wRes, bRes) := act.run' (0, [])
  "module @m {\n" ++
  s!"  func.func @linear_train_step(%x: {ty [B,m]}, %W0: {ty [m,n]}, %b0: {ty [n]}, " ++
  s!"%onehot: {ty [B,n]}) -> ({ty [m,n]}, {ty [n]}) " ++ "{\n" ++
  "    // ── linear train step: every line is pretty(verified AST node) ──\n" ++
  body ++
  s!"    return {wRes}, {bRes} : {ty [m,n]}, {ty [n]}\n" ++
  "  }\n}\n"

/-- `@mlp_fwd` rendered from the verified forward AST `mlpFwdGraph`. -/
def mlpFwdModuleV (B d₀ d₁ d₂ d₃ : Nat)
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) : String :=
  renderModule "mlp_fwd"
    s!"%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, %W1: {ty [d₁,d₂]}, %b1: {ty [d₂]}, %W2: {ty [d₂,d₃]}, %b2: {ty [d₃]}"
    B d₃ (mlpFwdGraph W₀ b₀ W₁ b₁ W₂ b₂ x)

/-- `@cnn_fwd` rendered from the verified CNN forward AST `cnnFwdGraph`. -/
def cnnFwdModuleV (B ic c h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) : String :=
  renderModule "cnn_fwd"
    s!"%x: {ty [B,ic*(2*h)*(2*w)]}, %W1: {ty [c,ic,kH,kW]}, %b1: {ty [c]}, %W2: {ty [c,c,kH,kW]}, %b2: {ty [c]}, %W3: {ty [c*h*w,d1]}, %b3: {ty [d1]}, %W4: {ty [d1,d1]}, %b4: {ty [d1]}, %W5: {ty [d1,nClasses]}, %b5: {ty [nClasses]}"
    B nClasses (cnnFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x)

/-- `@cifar_fwd` rendered from the verified CIFAR forward AST `cifarFwdGraph`. -/
def cifarFwdModuleV (B ic c1 c2 h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  renderModule "cifar_fwd"
    s!"%x: {ty [B,ic*(2*(2*h))*(2*(2*w))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c2*h*w,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}"
    B nClasses (cifarFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)

/-- `@cifar8_fwd` rendered from the verified 8-conv CIFAR forward AST `cifar8FwdGraph`
    (`cifar8FwdGraph_faithful` proves it denotes `cifarCnn8Forward`). The 4-stage peer of
    `cifarFwdModuleV` — the committed `verified_mlir/cifar8_fwd.mlir` is
    `renderModule(provenGraph)`. -/
def cifar8FwdModuleV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : String :=
  renderModule "cifar8_fwd"
    s!"%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [c4*h*w,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}"
    B nClasses (cifar8FwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x)

/-- `@cifar8_bn_fwd` rendered from the verified 8-conv per-channel-BN CIFAR forward AST
    `cifar8BnFwdGraph` (`cifar8BnFwdGraph_faithful` proves it denotes `cifarCnnBn8Forward`).
    The BN peer of `cifar8FwdModuleV`. -/
def cifar8BnFwdModuleV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : String :=
  renderModule "cifar8_bn_fwd"
    s!"%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %g5: {ty [c3]}, %bt5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %g6: {ty [c3]}, %bt6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %g7: {ty [c4]}, %bt7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %g8: {ty [c4]}, %bt8: {ty [c4]}, %W9: {ty [c4*h*w,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}"
    B nClasses (cifar8BnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
      W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x)

end StableHLO
end Proofs

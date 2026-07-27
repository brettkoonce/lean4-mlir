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

private def render (g : StateM Nat (String × String)) : String := (g.run' 0).1

/-- Per-example peer (SHlo index `n`) vs batched form (SHlo index `BS*n`), same `pretty BS`. -/
private def cases : List (String × String × String) :=
  let zv  : Vec n := fun _ => 0
  let zvb : Vec (BS*n) := fun _ => 0
  let zr  : Vec (rows*c) := fun _ => 0
  let zrb : Vec (BS*(rows*c)) := fun _ => 0
  let zW  : Mat a c := fun _ _ => 0
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
  ] ++
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

#eval main

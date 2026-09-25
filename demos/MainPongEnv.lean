import LeanMlir.Pong

/-! Pong in Lean — Phase 0 of `planning/pong_dqn_demo.md`: the sanity baselines
    (random, tracker, self-play) and a rendered frame strip over the game in
    `LeanMlir/Pong.lean`. No stack, no GPU: `lake exe pong-env [games=100]`. -/

open PongEnv

def main (args : List String) : IO Unit := do
  let n := (args.head? >>= String.toNat?).getD 100
  let o : Opp := {}
  IO.println s!"Pong: {n} games per arm, opponent speed {fmt o.speed 2} px/frame, delay {o.delay} frames"
  let t0 ← IO.monoMsNow
  let (m1, s1, f1) ← runGames n 1000 o randomPol
  let (m2, s2, f2) ← runGames n 2000 o trackerPol
  let (m3, s3, f3) ← runGames n 3000 { speed := 2.0, delay := 1 } trackerPol
  let (m4, s4, f4) ← runGames n 4000 { speed := 4.0, delay := 1 } trackerPol
  let t1 ← IO.monoMsNow
  let total := f1 + f2 + f3 + f4
  IO.println "arm                                        mean points/game   s.e.    frames/game"
  IO.println s!"random                                     {fmt m1 2}            {fmt s1 2}    {f1 / n}"
  IO.println s!"tracker vs default opponent                {fmt m2 2}            {fmt s2 2}    {f2 / n}"
  IO.println s!"tracker vs opponent (2.0, 1) ~ symmetric    {fmt m3 2}            {fmt s3 2}    {f3 / n}"
  IO.println s!"tracker vs opponent (4.0, 1) faster         {fmt m4 2}            {fmt s4 2}    {f4 / n}"
  IO.println s!"{total} raw frames in {t1 - t0} ms = {total * 1000 / (max 1 (t1 - t0))} frames/s (single thread, incl. RNG)"
  -- determinism
  let (d1, _) := playGame 7 o trackerPol
  let (d2, _) := playGame 7 o trackerPol
  IO.println s!"same seed twice: {d1} {d2} {if d1 == d2 then "(deterministic)" else "(NOT deterministic)"}"
  -- a four-frame stack from a rally, as one strip
  let mut gm := Game.reset 11
  let mut strip := ByteArray.mk (Array.replicate (84 * 84 * 4) 0)
  for k in [0:60] do
    let (a, g') := trackerPol gm.p gm.g
    gm := { gm with g := g' }
    let (gm', _, _) := gm.step o a
    gm := gm'
    if k >= 56 then
      let fr := Pong.render gm.p
      let col := k - 56
      for y in [0:84] do
        for x in [0:84] do
          strip := strip.set! (y * 336 + col * 84 + x) (fr.get! (y * 84 + x))
  IO.FS.createDirAll ".lake/build"
  writePgm ".lake/build/pong_stack.pgm" 336 84 strip
  IO.println "wrote .lake/build/pong_stack.pgm (four consecutive observations, 336x84)"

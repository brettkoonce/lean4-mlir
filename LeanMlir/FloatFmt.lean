/-! Fixed-point decimal for the Chapter 10 demos' tables (`Float.toString` prints 17 digits).
    Import-free, so the pure-Lean game modules `Blackjack`, `Pong` and `TicTacToe` share it with
    the demos that drive them and with `MainNqsIsing`. -/

namespace FloatFmt

/-- `x` rounded to `d` decimals, zero-padded: `fmt 0.5 3 = "0.500"`. -/
def fmt (x : Float) (d : Nat) : String :=
  let m := Float.pow 10.0 d.toFloat
  let y := Float.round (x * m)
  let neg := y < 0.0
  let yi := (Float.abs y).toUInt64.toNat
  let ip := yi / (10 ^ d)
  let fp := yi % (10 ^ d)
  let fs := toString fp
  let fs := String.ofList (List.replicate (d - fs.length) '0') ++ fs
  (if neg then "-" else "") ++ toString ip ++ "." ++ fs

#guard fmt 0.5 3 == "0.500"

end FloatFmt

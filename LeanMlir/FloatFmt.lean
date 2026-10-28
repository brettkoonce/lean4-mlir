module

@[expose] public section

/-! Fixed-point decimal for the Chapter 10 demos' tables (`Float.toString` prints 17 digits).
    Import-free, so the pure-Lean game modules `Pong` and `TicTacToe` share it with
    the demos that drive them and with `MainNqsIsing`. -/

namespace FloatFmt

/-- `x` rounded to `d ≥ 1` decimals, zero-padded: `fmt 0.5 3 = "0.500"` (at `d = 0` it still
    prints a trailing `.0`). A NaN prints `nan` and an
    infinity `inf` / `-inf`, so a diverged value cannot pass for a number in a table; a value too
    large for the fixed-point path (`|x|·10^d ≥ 2^63`) prints as `Float.toString` does. -/
def fmt (x : Float) (d : Nat) : String :=
  let m := Float.pow 10.0 d.toFloat
  if x.isNaN then "nan"
  else if x.isInf then (if x > 0.0 then "inf" else "-inf")
  else if (x * m).abs ≥ 9.223372036854775807e18 then toString x
  else
  let y := Float.round (x * m)
  let neg := y < 0.0
  let yi := (Float.abs y).toUInt64.toNat
  let ip := yi / (10 ^ d)
  let fp := yi % (10 ^ d)
  let fs := toString fp
  let fs := String.ofList (List.replicate (d - fs.length) '0') ++ fs
  (if neg then "-" else "") ++ toString ip ++ "." ++ fs

#guard fmt 0.5 3 == "0.500"
#guard fmt (0.0 / 0.0) 4 == "nan"
#guard fmt (-1.0 / 0.0) 2 == "-inf"

end FloatFmt

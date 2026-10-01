/-! # `key=value` command-line arguments

The demos take their knobs as `key=value` words (`lr=0.001 epochs=30 tag=x`). This is the one
reader for them, and the one float parser: `parseFloat?` rounds a token exactly as the same
literal written in Lean source would. Import-free. -/

namespace CliArgs

/-- Parse one float token (`-0.00623606`, `1.3e-05`, `42`, `nan`, `-inf`); `none` on anything
    else. Decimal and scientific forms go through the elaborator's own literal decoder
    (`Lean.Syntax.decodeScientificLitVal?` + `Float.ofScientific`), so a token rounds exactly as the
    same literal written in Lean source would; a bare integer, which that decoder rejects, falls
    back to `String.toNat?`. `nan`/`inf` are kept as NaN/∞ so a non-finite value fails a check
    rather than reading as a number. -/
def parseFloat? (tok : String) : Option Float :=
  let (neg, body) :=
    if tok.startsWith "-" then (true, (tok.drop 1).toString)
    else if tok.startsWith "+" then (false, (tok.drop 1).toString)
    else (false, tok)
  let v? : Option Float :=
    if body == "nan" then some (0.0 / 0.0)
    else if body == "inf" then some (1.0 / 0.0)
    else match Lean.Syntax.decodeScientificLitVal? body with
      | some (m, s, e) => some (Float.ofScientific m s e)
      | none => body.toNat?.map Nat.toFloat
  v?.map fun v => if neg then -v else v

/-- `parseFloat?` with `0.0` for a token it cannot read. -/
def parseFloat (tok : String) : Float := (parseFloat? tok).getD 0.0

/-- The value of the first `key=…` argument, if there is one. -/
def kv (args : List String) (key : String) : Option String :=
  (args.find? (·.startsWith (key ++ "="))).map (·.drop (key.length + 1) |>.toString)

/-- The value of `key=…`, or `dflt`. -/
def parseArg (args : List String) (key : String) (dflt : String) : String :=
  (kv args key).getD dflt

/-- `key=…` as a natural number, or `d` when absent or unreadable. -/
def natArg (args : List String) (key : String) (d : Nat) : Nat :=
  ((kv args key) >>= String.toNat?).getD d

/-- `key=…` as a float (`parseFloat?`), or `d` when absent or unreadable. -/
def floatArg (args : List String) (key : String) (d : Float) : Float :=
  ((kv args key) >>= parseFloat?).getD d

#guard parseFloat? "1e-4" == some 1e-4
#guard parseFloat? "-2.5" == some (-2.5)
#guard parseFloat? "42" == some 42.0
#guard parseFloat? "abc" == none
#guard parseArg ["lr=0.01", "tag=x"] "tag" "" == "x"
#guard natArg ["epochs=7"] "epochs" 30 == 7
#guard floatArg ["lr=x"] "lr" 0.5 == 0.5

end CliArgs

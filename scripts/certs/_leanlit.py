"""Lean literal printers for the certificate generators. Their output is committed Lean, so a
change here is a change to every generated file: regenerate them all and diff."""
from fractions import Fraction


def zlit(x):
    """An integer as a `ℤ` literal; negatives as `Int.negSucc n`, which `decide` reduces fastest."""
    x = int(x)
    return str(x) if x >= 0 else f"Int.negSucc {-x - 1}"


def zlist(vals):
    return "[" + ", ".join(zlit(v) for v in vals) + "]"


def frac(q):
    """A rational as an `ℝ` literal: `((n : ℝ)/d)`, or `(n : ℝ)` when integral."""
    q = Fraction(q)
    return f"(({q.numerator} : ℝ)/{q.denominator})" if q.denominator != 1 else f"({q.numerator} : ℝ)"


def rrow(vals, den):
    """Integer numerators over one denominator as an `ℝ` vector literal `![...]`."""
    return "![" + ", ".join(f"(({int(v)} : ℝ)/{den})" for v in vals) + "]"


def rmat(M, den):
    return "![" + ",\n    ".join(rrow(r, den) for r in M) + "]"


def qrow(vals, den):
    """As `rrow`, over `ℚ`."""
    return "![" + ", ".join(f"(({int(v)} : ℚ)/{den})" for v in vals) + "]"


def qmat(M, den):
    return "![" + ",\n    ".join(qrow(r, den) for r in M) + "]"

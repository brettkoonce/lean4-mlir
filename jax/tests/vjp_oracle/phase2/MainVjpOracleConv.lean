import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.convOnly` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.convOnly.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.convOnly VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_conv.py"

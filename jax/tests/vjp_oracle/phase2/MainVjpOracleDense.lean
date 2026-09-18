import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.denseOnly` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.denseOnly.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.denseOnly VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_dense.py"

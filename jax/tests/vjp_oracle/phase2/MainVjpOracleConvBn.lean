import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.convBnOnly` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.convBnOnly.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.convBnOnly VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_convbn.py"

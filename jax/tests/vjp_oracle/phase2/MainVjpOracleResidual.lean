import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.residualNet` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.residualNet.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.residualNet VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_residual.py"

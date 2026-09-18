import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.denseRelu` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.denseRelu.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.denseRelu VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_dense_relu.py"

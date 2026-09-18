import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.depthwiseNet` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.depthwiseNet.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.depthwiseNet VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_depthwise.py"

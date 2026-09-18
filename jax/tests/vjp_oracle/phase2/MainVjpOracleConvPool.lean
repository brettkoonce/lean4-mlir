import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.convPool` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.convPool.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.convPool VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_conv_pool.py"

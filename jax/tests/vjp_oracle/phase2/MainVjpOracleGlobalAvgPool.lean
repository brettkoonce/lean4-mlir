import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.gapNet` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.gapNet.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.gapNet VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_global_avg_pool.py"

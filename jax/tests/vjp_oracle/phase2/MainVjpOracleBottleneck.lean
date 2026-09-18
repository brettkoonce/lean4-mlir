import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.bneckNet` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.bneckNet.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.bneckNet VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_bottleneck.py"

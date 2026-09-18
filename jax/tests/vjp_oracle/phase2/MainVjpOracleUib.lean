import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.uibNet` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.uibNet.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.uibNet VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_uib.py"

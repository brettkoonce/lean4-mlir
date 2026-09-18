import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.mbConvNet` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.mbConvNet.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.mbConvNet VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_mbconv.py"

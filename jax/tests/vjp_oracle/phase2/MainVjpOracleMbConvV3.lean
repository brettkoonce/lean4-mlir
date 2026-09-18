import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.mbConvV3Net` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.mbConvV3Net.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.mbConvV3Net VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_mbconv_v3.py"

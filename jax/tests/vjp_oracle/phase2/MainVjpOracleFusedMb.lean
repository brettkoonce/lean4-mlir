import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.fusedMbNet` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.fusedMbNet.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.fusedMbNet VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_fused_mbconv.py"

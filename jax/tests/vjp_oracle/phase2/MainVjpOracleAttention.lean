import Jax
import LeanMlir.VjpOracleNets

-- VJP oracle, JAX side: `VjpOracle.attentionNet` (spec shared with tests/vjp_oracle/phase3/).

#eval VjpOracle.attentionNet.validate!

def main (args : List String) : IO Unit :=
  runJax VjpOracle.attentionNet VjpOracle.cfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_attention.py"

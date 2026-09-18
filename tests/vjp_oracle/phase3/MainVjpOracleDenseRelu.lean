import LeanMlir
import LeanMlir.VjpOracleNets

-- VJP oracle, verified side: `VjpOracle.denseRelu` (spec shared with jax/tests/vjp_oracle/phase2/).

def main (args : List String) : IO Unit :=
  VjpOracle.denseRelu.train VjpOracle.cfg (args.head?.getD "data") .mnist

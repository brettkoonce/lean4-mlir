import LeanMlir
import LeanMlir.VjpOracleNets

-- VJP oracle, verified side: `VjpOracle.bneckNet` (spec shared with jax/tests/vjp_oracle/phase2/).

def main (args : List String) : IO Unit :=
  VjpOracle.bneckNet.train VjpOracle.cfg (args.head?.getD "data") .mnist

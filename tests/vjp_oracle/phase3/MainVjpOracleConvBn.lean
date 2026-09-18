import LeanMlir
import LeanMlir.VjpOracleNets

-- VJP oracle, verified side: `VjpOracle.convBnOnly` (spec shared with jax/tests/vjp_oracle/phase2/).

def main (args : List String) : IO Unit :=
  VjpOracle.convBnOnly.train VjpOracle.cfg (args.head?.getD "data") .mnist

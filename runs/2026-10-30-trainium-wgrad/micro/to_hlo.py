"""to_hlo.py IN.mlir OUT.hlo — StableHLO text (func @main) → serialized HloModuleProto for neuronx-cc."""
import sys
from jax._src.lib import xla_client as xc
txt = open(sys.argv[1]).read()
c = xc._xla.mlir.mlir_module_to_xla_computation(txt, use_tuple_args=False, return_tuple=False)
open(sys.argv[2], 'wb').write(c.as_serialized_hlo_module_proto())
print(c.as_hlo_text()[:2000])

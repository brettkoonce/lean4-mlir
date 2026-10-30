# Source on an x86 box with no Trainium: Neuron's fake device lets jax-neuronx compile (it fails only at load).
# Used to check compiler-side repros (03, 04) without an instance. Not needed on a trn1.
R=/home/skoonce/lean/lean4-mlir
. $R/.venv-neuron/bin/activate
export LD_LIBRARY_PATH=$HOME/.cache/neuron-warm/2.35${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} NEURON_RT_FAKE_INSTANCE_TYPE=trn1.32xlarge \
  NEURON_RT_VISIBLE_CORES=0 NEURON_FORCE_PJRT_PLUGIN_REGISTRATION=1 JAX_PLATFORMS=neuron,cpu \
  NEURON_COMPILE_CACHE_URL=$(mktemp -d /var/tmp/neuron-repro.XXXX)

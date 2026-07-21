# Certificates/ — machine-emitted, not hand-written

Every `.lean` file in this directory is **generated** by a script in
`scripts/` (`lipschitz_cert_*.py`, `smooth_*scorecard*_gen.py`,
`smoothing_net_witness_gen.py`) and then checked by Lean like any other
proof: same kernel, zero `sorry`s, three-axiom audit.

That provenance explains their shape: thousands of short, structurally
identical theorem statements (one per network / image / radius row of a
scorecard), high body duplication, thin docstrings. Judged as *prose* they
look nothing like the hand-written library in the sibling directories —
and that's expected; they are certified *data*, not exposition. If a
code-quality metric flags this directory, the honest answer is "yes, it's
an emitted payload, and here is the emitter."

To regenerate a scorecard, run its generator script from the repo root and
rebuild `lake build CertsHeavy` (the heavy scorecards) or `Certs` (the
rest). Don't hand-edit files here — edits will be clobbered by the next
generator run.

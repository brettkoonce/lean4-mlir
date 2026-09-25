# `tests/` — the gates behind the verified renders

| what | files |
|---|---|
| independent kernel re-check | [`comparator/`](comparator/): the headline theorems re-checked by Lean's kernel outside this project's build; `Challenge.lean` imports only Mathlib |
| differential oracle | [`vjp_oracle/`](vjp_oracle/): each backward checked against JAX autodiff, with the Lean side lowered through IREE |
| axiom audit | `AuditAxioms.lean`, `AuditAxiomsHeavy.lean`: `#print axioms` over every pinned theorem |
| citation gates | `BlueprintCheckDecls.lean`, `DocstringCheckRefs.lean`: every name the book or a docstring cites resolves |
| ties and checks | `Test*.lean`, one `lake exe` each: a committed render against the proven math (`*Tie`), data-parallel and sync-BN equivalence (`*DpCheck`, `*SyncBnCheck`, `*ShardCheck`), and regression guards for specific bugs |
| Bestiary | `bestiary_params.yml`, `verify_bestiary_timm.py`: the catalogue's parameter counts against timm |

The exe names are in [`lakefile.lean`](../lakefile.lean) under `tests/`. The proofs themselves
are in [`LeanMlir/Proofs/`](../LeanMlir/Proofs/); `lake build Certs` checks them.

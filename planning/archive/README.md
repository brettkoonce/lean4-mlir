# planning/archive — closed threads

Planning docs whose work has landed. Nothing here is deleted or wrong;
it is the lab notebook for finished threads, moved out of `planning/`
so what remains there is what is still open.

Hygiene item H3 of `planning/repo_shape_deletion_audit.md`.

## What's in here

| doc | thread, and where the work lives now |
|---|---|
| `verified_mnv2.md`, `verified_enet.md`, `verified_convnext.md`, `verified_vit.md` | Ch 7–10 verified-net handoffs — all four nets ship |
| `resnet34_close.md`, `efficientnet_close.md` | "close it both ways" plans for ch 6 and ch 8 |
| `cifar_handoff.md`, `validation_sweep.md` | Ch 5 verified codegen + the sweep that put it in the book |
| `backward_graph_faithfulness_convnext_vit.md` | backward-graph faithfulness for ConvNeXt + ViT |
| `conv2d.md`, `mat_matrix_phase2.md` | input-VJP elimination; `Proofs.Mat` → Mathlib `Matrix` |
| `floatbridge_descent_cnn.md`, `floatbridge_descent_pass.md`, `floatbridge_honesty_pass.md`, `forward_wholenet_handoff.md`, `backward_certified_tie_and_vit_fwd_tie.md` | the FloatBridge §3 descent rungs and the whole-net forward/backward ties |

## What deliberately stayed in planning/

Two kinds of doc did **not** move:

**Standing references, not threads** — `upgrade.md` (release procedure),
`SIDE_QUESTS.md` (backlog), `codegen_scope.md` (what earns a real emit),
`math_threads.md`, `paper_faithfulness.md` (the fidelity ledger),
`post_audit_roadmap.md`, `repo_shape_deletion_audit.md`. These are old but
live; age is not the criterion, openness is.

**Docs cited from code** — 25 otherwise-closed threads are referenced by
name from Lean proof docstrings, the three CI workflows, `lakefile.lean`
and `AuditAxioms.lean` (`audit.md`, `VJP.md`, `pdiv.md`, `yolo_final.md`,
`whole_network_backward.md`, `a3_backward_deepnet_assembly.md`, the
`*_close.md` set, …). Those citations are not uniform — some carry the
`planning/` prefix, some are bare filenames mid-sentence — so moving them
would leave half-rewritten pointers inside the proof suite. They are
archivable, but only as a deliberate pass that rewrites the citations
too, with a proof rebuild behind it.

## Adding to this directory

Move a doc here when its thread is closed and nothing outside `planning/`
cites it. If something does cite it, repoint the citation in the same
commit or leave the doc where it is. Cross-references from docs that stay
behind get an `archive/` prefix.

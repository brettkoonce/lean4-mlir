/-
lean-atlas metadata annotations (https://github.com/NyxFoundation/lean-atlas).

Applied post-hoc via `attribute [...]` so the proof modules themselves never
import the visualization tooling — deleting this file (and its import in
`LeanMlir.lean`) removes the lean-atlas coupling entirely.

`lake exe atlas` uses the `mainTheorem` flags below as the roots of its
"review cone" (Lean Compass); the interactive graph serves at localhost:5326.
-/
import LeanAtlas.Metadata.Attribute.Meta
import LeanMlir.Proofs.Attention

-- The ViT capstone: the blueprint's Theorem `thm:vit_body_has_vjp_mat` —
-- a machine-checked backward for the full k-block ViT body (pre-LN MHSA +
-- MLP), composed from the ~30 attention-chapter VJP lemmas.
attribute [formalMeta
  "ViT body VJP (capstone)"
  "Machine-checked backward for the k-block ViT body: pre-LN multi-head self-attention + MLP, composed blockwise"
  "Blueprint Ch 9, thm:vit_body_has_vjp_mat"
  mainTheorem] Proofs.vit_body_has_vjp_mat

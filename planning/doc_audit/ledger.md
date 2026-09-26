# ledger.md — doc_honesty_pass §1, one row per changed site

Branch `doc-pass` off origin/main `c5c1aeb6`. Recounts, run 2026-09-26:

* comparator: `python3 -c "import json;print(len(json.load(open('tests/comparator/<cfg>.json'))['theorem_names']))"`
  → config 13, config-arch 39, config-tier 35, total 87. Same at tag v0.7.1 (`git show v0.7.1:…`);
  52 at v0.7.0 (no config-tier).
* AuditAxioms: `grep -c '^#print axioms' tests/AuditAxioms.lean` → 1,596 (Heavy 62). The followups'
  1,610 predates `c5c1aeb6` (§9 private helpers).

## §1(j) numbers

| Site | Old | New | Checked by |
|---|---|---|---|
| tests/comparator/README.md:3, 63, 208, 212 | 73 | 87 | comparator recount |
| tests/comparator/README.md:39 | remaining 21 | remaining 35 | config-tier count |
| tests/comparator/README.md:208 | all 1,374 | all 1,596 | AuditAxioms recount |
| LeanMlir/Proofs/README.md:417 | "73 theorems … the five whole-network VJPs (ViT, ResNet, MobileNetV2, ConvNeXt, EfficientNet)" | "87 theorems; its README lists them by configuration" | the list named nine nets across two configs and mixed reduced-depth with full-depth rows (see doc_honesty_pass "What went wrong"); pointer instead of a summary |
| README.md:53 tier 3 | R34 89.50 | R34 89.99 (mean of five seeds) | runs/2026-09-12-r34-ablation-fp32-seeds/RESULTS.md, `full` row |
| CHANGELOG.md:46 (v0.7.1) | re-checks 73 (was 52) | 87 (was 52) | count at tag v0.7.1 |
| CHANGELOG.md:88–90 (v0.7.0) | "verified path to 78.26%, ahead of its JAX reference" | "verified path to 77.91%, beside its JAX reference's 78.26" | 78.26 is the JAX reference (v0.7.1 entry; v0.7.0 README tier-4 row quotes references); 77.91 fp32 verified, runs/2026-08-27-r50-a2-a1-verified-eta/README.md:66 |
| content.tex:16525, 16919, 16936, 16939 | 73 | 87 | comparator recount — reviewed 2026-09-26 |
| content.tex:16550, 16926 | 21 | 35 | config-tier count — reviewed 2026-09-26 |
| content.tex:16937 | 1{,}374 | 1{,}596 | AuditAxioms recount — reviewed 2026-09-26 |

Left for later: CHANGELOG v0.7.0 also says ResNet-34, MobileNetV2 and ViT-Tiny "land at 74.16,
71.90 and 72.31" and ConvNeXt-T "at 81.53%"; those read as the JAX references' numbers too. Not
re-derived here.

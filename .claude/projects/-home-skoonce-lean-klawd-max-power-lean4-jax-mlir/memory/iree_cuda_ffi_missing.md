---
name: IREE FFI triple return bug
description: C FFI for IO (A × B × C) must use nested Prod pairs, not 3-field ctor
type: feedback
---

Lean's `A × B × C` is `Prod A (Prod B C)` — a nested pair, NOT a 3-element tuple. In C FFI, return it as:
```c
lean_object* inner = lean_alloc_ctor(0, 2, 0);  // (B, C)
lean_object* outer = lean_alloc_ctor(0, 2, 0);  // (A, inner)
```
NOT `lean_alloc_ctor(0, 3, 0)` which causes silent memory corruption/segfault.

**Why:** Used wrong ctor for loadImagenette return type, caused segfault that looked like an IREE/CUDA issue.

**How to apply:** Always check Lean's ABI for tuple types in FFI code. `A × B` = `Prod A B` = ctor(0, 2, 0). `A × B × C` = `Prod A (Prod B C)` = nested two ctors.

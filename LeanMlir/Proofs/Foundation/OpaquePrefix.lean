import LeanMlir.Proofs.Foundation.Tensor

/-! # The opaque running activations of a layered net — one `def` per slot

`opaqueA k stem b1 … bk x` is the activation after the `k`-th stage of a chain whose stages are
all still VARIABLES. Every whole-net certified backward tie states its apex over these: the tie
keeps its blocks opaque, and a `*_eq_slots` shape check says the concrete stages ARE the committed
forward. They are plain `def`s so the closing `rfl` of a tie can unfold them.

⭐ Net-agnostic and generic in every dimension. Until 2026-09-08 ResNet-34, EfficientNet-B0,
MobileNetV2 and MobileNetV4 each carried a private copy of this construction — seventeen, seventeen,
eighteen and twenty-five slots, under four names — and ResNet-50 reused ResNet-34's. One copy, to
the deepest ladder in the suite. -/

namespace Proofs

/-- The first stage's output. -/
noncomputable def opaqueA0 {s0 s1 : Nat} (stem : Vec s0 → Vec s1) (x : Vec s0) : Vec s1 :=
  stem x

/-- The activation after stage 1. -/
noncomputable def opaqueA1 {s0 s1 s2 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2)
    (x : Vec s0) : Vec s2 := b1 (opaqueA0 stem x)

/-- The activation after stage 2. -/
noncomputable def opaqueA2 {s0 s1 s2 s3 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3)
    (x : Vec s0) : Vec s3 := b2 (opaqueA1 stem b1 x)

/-- The activation after stage 3. -/
noncomputable def opaqueA3 {s0 s1 s2 s3 s4 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4)
    (x : Vec s0) : Vec s4 := b3 (opaqueA2 stem b1 b2 x)

/-- The activation after stage 4. -/
noncomputable def opaqueA4 {s0 s1 s2 s3 s4 s5 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5)
    (x : Vec s0) : Vec s5 := b4 (opaqueA3 stem b1 b2 b3 x)

/-- The activation after stage 5. -/
noncomputable def opaqueA5 {s0 s1 s2 s3 s4 s5 s6 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6)
    (x : Vec s0) : Vec s6 := b5 (opaqueA4 stem b1 b2 b3 b4 x)

/-- The activation after stage 6. -/
noncomputable def opaqueA6 {s0 s1 s2 s3 s4 s5 s6 s7 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7)
    (x : Vec s0) : Vec s7 := b6 (opaqueA5 stem b1 b2 b3 b4 b5 x)

/-- The activation after stage 7. -/
noncomputable def opaqueA7 {s0 s1 s2 s3 s4 s5 s6 s7 s8 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8)
    (x : Vec s0) : Vec s8 := b7 (opaqueA6 stem b1 b2 b3 b4 b5 b6 x)

/-- The activation after stage 8. -/
noncomputable def opaqueA8 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9)
    (x : Vec s0) : Vec s9 := b8 (opaqueA7 stem b1 b2 b3 b4 b5 b6 b7 x)

/-- The activation after stage 9. -/
noncomputable def opaqueA9 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10)
    (x : Vec s0) : Vec s10 := b9 (opaqueA8 stem b1 b2 b3 b4 b5 b6 b7 b8 x)

/-- The activation after stage 10. -/
noncomputable def opaqueA10 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11)
    (x : Vec s0) : Vec s11 := b10 (opaqueA9 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 x)

/-- The activation after stage 11. -/
noncomputable def opaqueA11 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12)
    (x : Vec s0) : Vec s12 := b11 (opaqueA10 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)

/-- The activation after stage 12. -/
noncomputable def opaqueA12 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13)
    (x : Vec s0) : Vec s13 := b12 (opaqueA11 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)

/-- The activation after stage 13. -/
noncomputable def opaqueA13 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14)
    (x : Vec s0) : Vec s14 := b13 (opaqueA12 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)

/-- The activation after stage 14. -/
noncomputable def opaqueA14 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15)
    (x : Vec s0) : Vec s15 := b14 (opaqueA13 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)

/-- The activation after stage 15. -/
noncomputable def opaqueA15 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16)
    (x : Vec s0) : Vec s16 := b15 (opaqueA14 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)

/-- The activation after stage 16. -/
noncomputable def opaqueA16 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17)
    (x : Vec s0) : Vec s17 := b16 (opaqueA15 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)

/-- The activation after stage 17. -/
noncomputable def opaqueA17 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17) (b17 : Vec s17 → Vec s18)
    (x : Vec s0) : Vec s18 := b17 (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)

/-- The activation after stage 18. -/
noncomputable def opaqueA18 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17) (b17 : Vec s17 → Vec s18) (b18 : Vec s18 → Vec s19)
    (x : Vec s0) : Vec s19 := b18 (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)

/-- The activation after stage 19. -/
noncomputable def opaqueA19 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17) (b17 : Vec s17 → Vec s18) (b18 : Vec s18 → Vec s19) (b19 : Vec s19 → Vec s20)
    (x : Vec s0) : Vec s20 := b19 (opaqueA18 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x)

/-- The activation after stage 20. -/
noncomputable def opaqueA20 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17) (b17 : Vec s17 → Vec s18) (b18 : Vec s18 → Vec s19) (b19 : Vec s19 → Vec s20) (b20 : Vec s20 → Vec s21)
    (x : Vec s0) : Vec s21 := b20 (opaqueA19 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x)

/-- The activation after stage 21. -/
noncomputable def opaqueA21 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17) (b17 : Vec s17 → Vec s18) (b18 : Vec s18 → Vec s19) (b19 : Vec s19 → Vec s20) (b20 : Vec s20 → Vec s21) (b21 : Vec s21 → Vec s22)
    (x : Vec s0) : Vec s22 := b21 (opaqueA20 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x)

/-- The activation after stage 22. -/
noncomputable def opaqueA22 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17) (b17 : Vec s17 → Vec s18) (b18 : Vec s18 → Vec s19) (b19 : Vec s19 → Vec s20) (b20 : Vec s20 → Vec s21) (b21 : Vec s21 → Vec s22) (b22 : Vec s22 → Vec s23)
    (x : Vec s0) : Vec s23 := b22 (opaqueA21 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)

/-- The activation after stage 23. -/
noncomputable def opaqueA23 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17) (b17 : Vec s17 → Vec s18) (b18 : Vec s18 → Vec s19) (b19 : Vec s19 → Vec s20) (b20 : Vec s20 → Vec s21) (b21 : Vec s21 → Vec s22) (b22 : Vec s22 → Vec s23) (b23 : Vec s23 → Vec s24)
    (x : Vec s0) : Vec s24 := b23 (opaqueA22 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 x)

/-- The activation after stage 24. -/
noncomputable def opaqueA24 {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17) (b17 : Vec s17 → Vec s18) (b18 : Vec s18 → Vec s19) (b19 : Vec s19 → Vec s20) (b20 : Vec s20 → Vec s21) (b21 : Vec s21 → Vec s22) (b22 : Vec s22 → Vec s23) (b23 : Vec s23 → Vec s24) (b24 : Vec s24 → Vec s25)
    (x : Vec s0) : Vec s25 := b24 (opaqueA23 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 x)

end Proofs

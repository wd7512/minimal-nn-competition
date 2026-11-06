import Proofs.Architecture

/-!
The CartPole architectural lower bound.

This file proves that every accepted CartPole MLP architecture has at least
nine parameters, and that a nine-parameter architecture exists.  It does not
claim that any particular model passes the benchmark; the passing witness is
evaluated by the competition's Python code and recorded in `proofs/bounds.json`.
-/

namespace MinimalNN

theorem cartPole_architecturalLowerBound
    (hidden : List Nat) (hpos : forall h, h ∈ hidden -> 1 <= h) :
    9 <= parameterCount 4 2 hidden := by
  have h := parameterCount_ge_floor 4 2 (by decide) (by decide) hidden hpos
  have hfloor : architecturalFloor 4 2 = 9 := by decide
  simpa [hfloor] using h

theorem cartPole_attained : parameterCount 4 2 [1] = 9 :=
  Claim.check_sound (Claim.mk 4 2 [1] 9) (by decide)

theorem cartPole_architecturalMinimum : IsArchitecturalMinimum 4 2 9 := by
  constructor
  · intro hidden hpos
    exact cartPole_architecturalLowerBound hidden hpos
  · refine ⟨[1], ?_, cartPole_attained⟩
    intro h hh
    simp at hh
    omega

end MinimalNN

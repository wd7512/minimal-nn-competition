/-!
Parameter accounting for the accepted feed-forward MLP family.

The competition builds `nn.Sequential` stacks of `Linear` layers, so every
layer contributes `(previous width + 1) * width` scalars.  The formal counter
below is intentionally independent of the benchmark code; `Proofs/Generated.lean`
checks it against a pinned table derived from the evaluator.
-/

namespace MinimalNN

/-- Number of scalar parameters in a fully connected MLP with the given input,
output and hidden widths.  The final hidden width is the input width of the
output layer. -/
def parameterCount (input output : Nat) : List Nat -> Nat
  | [] => (input + 1) * output
  | hidden :: rest => (input + 1) * hidden + parameterCount hidden output rest

/-- The architectural floor: the smaller of the linear model and one hidden
unit.  More layers or wider hidden layers only add parameters. -/
def architecturalFloor (input output : Nat) : Nat :=
  min ((input + 1) * output) (input + 1 + 2 * output)

/-- A number is an architectural minimum when every positive-width hidden list
costs at least that much and some positive-width hidden list costs exactly that
much.  This is the architecture-only statement; it says nothing about whether a
model passes the benchmark. -/
def IsArchitecturalMinimum (input output n : Nat) : Prop :=
  (forall hidden : List Nat, (forall h, h ∈ hidden -> 1 <= h) ->
      n <= parameterCount input output hidden) ∧
  ∃ hidden : List Nat, (forall h, h ∈ hidden -> 1 <= h) ∧
      parameterCount input output hidden = n

/-- With a positive starting width and positive output width, any list of
positive hidden widths costs at least `2 * output` after the first layer. -/
theorem two_mul_output_le_parameterCount
    (input output : Nat) (_hinput : 1 <= input) (_houtput : 1 <= output)
    (hidden : List Nat) (hpos : forall h, h ∈ hidden -> 1 <= h) :
    2 * output <= parameterCount input output hidden := by
  induction hidden generalizing input with
  | nil =>
      simp [parameterCount]
      exact Nat.mul_le_mul_right output (by omega)
  | cons head tail ih =>
      have hhead : 1 <= head := hpos head (by simp)
      have htail : forall h, h ∈ tail -> 1 <= h := by
        intro h hh
        exact hpos h (List.mem_cons_of_mem head hh)
      have ih' := ih head hhead htail
      simp [parameterCount]
      omega

/-- The architectural floor is a lower bound for every architecture with
positive input, output and hidden widths. -/
theorem parameterCount_ge_floor
    (input output : Nat) (_hinput : 1 <= input) (_houtput : 1 <= output)
    (hidden : List Nat) (hpos : forall h, h ∈ hidden -> 1 <= h) :
    architecturalFloor input output <= parameterCount input output hidden := by
  cases hidden with
  | nil =>
      simp [architecturalFloor, parameterCount]
      exact Nat.min_le_left _ _
  | cons head tail =>
      have hhead : 1 <= head := hpos head (by simp)
      have htail : forall h, h ∈ tail -> 1 <= h := by
        intro h hh
        exact hpos h (List.mem_cons_of_mem head hh)
      have hrest :=
        two_mul_output_le_parameterCount head output hhead _houtput tail htail
      have hfirst : input + 1 <= (input + 1) * head := by
        simpa using Nat.mul_le_mul_left (input + 1) hhead
      have hle :
          input + 1 + 2 * output <= parameterCount input output (head :: tail) := by
        simp [parameterCount]
        omega
      exact Nat.le_trans (Nat.min_le_right _ _) hle

/-- An externally generated architecture claim: a dimensions list and the
parameter count the generator computed for it.  `check_sound` is the bridge that
lets `decide` turn a generator claim into a theorem. -/
structure Claim where
  input : Nat
  output : Nat
  hidden : List Nat
  parameters : Nat
  deriving Repr, DecidableEq

def Claim.check (claim : Claim) : Bool :=
  decide (parameterCount claim.input claim.output claim.hidden = claim.parameters)

theorem Claim.check_sound (claim : Claim) (h : claim.check = true) :
    parameterCount claim.input claim.output claim.hidden = claim.parameters := by
  have h' :
      decide (parameterCount claim.input claim.output claim.hidden = claim.parameters) = true := by
    simpa [Claim.check] using h
  exact of_decide_eq_true h'

end MinimalNN

# Asymptotic lower bounds for the leaderboard problems

An analysis of how small each benchmark's optimal MLP could be, and how far the
current leaderboard is from the bounds that can actually be proved. The numbers
are taken from `docs/leaderboard.json` (82 entries, best passing submission per
benchmark; the generated `docs/index.html` can lag the JSON) and from the
benchmark implementations in `src/minimal_nn_competition/benchmarks/`.

The short version: one problem (CartPole) is solved exactly. The control
problems are close to their architectural floors. The shuffled-label and chess
problems are in a memorization regime where the only strong bound is an
information bound, and the leaderboard is 15–26× above it. The clean image
problems have no useful task-specific lower bound: the best passing models are
only 1.04× (MNIST) and 2.58× (Fashion-MNIST) the size of a linear model, even
though the linear model fails. The obstacle-Snake problem may be impossible for
the observation encoding, not merely too small; a concrete observation collision
is shown below.

## 1. What counts as a lower bound

Let the network have input width $d$, output width $k$, hidden widths
$h_1,\dots,h_L$, and

$$
P = (d+1)h_1 + \sum_{i=2}^{L}(h_{i-1}+1)h_i + (h_L+1)k .
$$

No hidden layer means $P=(d+1)k$; a single hidden unit means $P=d+1+2k$. These
are the two smallest possible architectures, so the **architectural floor** is

$$
P_{\text{arch}} = \min\{(d+1)k,\; d+1+2k\}.
$$

This is rigorous but ignores the task. Other bounds use the task:

1. **Information / description.** The checkpoint is float32, so a model with
   $P$ parameters has at most $2^{32P}$ distinct functions. If the pass
   criterion requires one of $M$ equally likely output vectors on $N$ fixed
   examples, then $P \ge N\log_2 M / 32$ with high probability. This is the
   strongest simple bound for random labels and random targets.
2. **Capacity / VC.** A ReLU network with $P$ parameters and $L$ layers has
   $\mathrm{VCdim} = O(PL\log P)$ (Bartlett et al., 2019). Fitting arbitrary
   labels on $N$ points needs $PL\log P = \Omega(N)$, i.e.
   $P=\Omega(N/(L\log N))$ for fixed $L$, and $P=\Omega(\sqrt{N/\log N})$ when
   depth is unrestricted (Yun, Sra and Jadbabaie, 2019). The derivation is a
   counting argument: a class with VC dimension $V$ realizes at most
   $\sum_{i\le V}\binom{N}{i}\le(eN/V)^V$ labelings, so a uniformly random
   labeling is realized with probability at most
   $2^{-N}(eN/V)^V$; unless $V=\Omega(N)$ this is exponentially small. The
   universal bound follows because $L\le P$.
3. **Approximation.** For a Hölder-$\alpha$ target on an $r$-dimensional
   manifold, any fixed-depth ReLU network with error $\varepsilon$ needs
   $P=\Omega(\varepsilon^{-r/(2\alpha)})$ (up to logarithmic factors; the exact
   exponent depends on depth). With $N$ samples and
   $\varepsilon\sim N^{-1/r}$, this is $P=\Omega(N^{1/(2\alpha)})$. This is a
   worst-case bound over the Hölder class, not a bound for the specific
   benchmark function.
4. **Structural / observability.** Control-theoretic rank arguments,
   Hamiltonian-cycle/region arguments, and collisions in the observation map.
   These are the only bounds that can be tight for structured tasks.

The important consequence is that for structured data, the information and VC
bounds are usually *not* lower bounds for the benchmark at all. They describe a
worst-case class that contains the benchmark. They are still useful as a scale:
they tell us which regime a task is in.

## 2. Summary table

`arch` is the rigorous architectural floor. `info` is the float32 information
floor, which is a high-probability bound for random labels/targets and only a
reference for structured ones (marked \*).

| Benchmark | Best $P$ | arch | info | $P/\text{arch}$ | $P/\text{info}$ | Regime |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CartPole | 9 | 9 | — | 1.00 | — | solved exactly |
| Pendulum | 11 | 4 | — | 2.75 | — | likely near-optimal |
| LunarLander | 69 | 17 | — | 4.06 | — | optimization-limited |
| Snake 8×8 | 37 | 15 | 4\* | 2.47 | 9.3\* | near lookup floor |
| Snake 16×16 | 203 | 15 | 16\* | 13.5 | 12.7\* | scaling open |
| Snake obstacles 8×8 | none (22%) | 15 | ∞? | — | — | observation-blocked |
| MNIST | 8,190 | 805 | 7,267\* | 10.2 | 1.13\* | empirical frontier |
| Fashion-MNIST | 20,241 | 805 | 7,267\* | 25.1 | 2.79\* | empirical frontier |
| MNIST shuffled | 115,932 | 805 | 7,267 | 144 | 16.0 | $\Theta(N)$ |
| Fashion shuffled | 115,904 | 805 | 7,267 | 144 | 15.95 | $\Theta(N)$ |
| Chess 10k | 31,001 | 774 | 2,077\* | 40.1 | 14.9\* | memorization-like |
| Chess 100k | 541,093 | 774 | 20,763\* | 699 | 26.1\* | memorization-like |

The `info` column for the image tasks uses $N=70{,}000$, $k=10$
($70{,}000\log_2 10/32=7{,}267$). For chess it uses the ±0.1-pawn tolerance:
100 acceptance bins over a 20-pawn range, so $10{,}000\log_2 100/32=2{,}077$ and
$100{,}000\log_2 100/32=20{,}763$. For snake it uses 2 bits per cell for an
arbitrary action table, $n^2/16$ (4 and 16 parameters).

### 2.1 Scaling-law summary

| Family | Variable | Best lower bound (order) | Matching construction | Observed exponent | Current $P$ |
| --- | --- | --- | --- | ---: | --- |
| Clean image | $N$ | $\Omega(\sqrt N)$ for Lipschitz boundaries; $O(N)$ worst case | $O(N)$ memorization | one point | 8,190 / 20,241 |
| Shuffled image | $N$ | $\Theta(N)$, constant $0.104N$ | $O(N)$ table | 1 (by definition) | 115,932 |
| Chess | $N$ | $\Theta(N)$ worst case, constant $0.207N$ | $O(N)$ memorization | 1.24 (10k→100k) | 31,001 / 541,093 |
| Snake | $n$ | $\Omega(n^{1/L})$; $\Omega(n)$ for $L=1$ | $O(n)$ phase, $O(n^2)$ lookup | 2.46 | 37 / 203 |
| Control | $d,k$ | $\Theta(dk)$ for linear policies; exact 9 for CartPole | $O(dk)$ linear | n/a | 9 / 11 / 69 |

The exponents in the last two rows are empirical fits to two leaderboard points,
not asymptotic theorems. The clean-image lower bound is a worst-case Hölder
statement; the actual exponent for MNIST is unknown.

## 3. Image classification: MNIST and Fashion-MNIST

Both benchmarks score exact accuracy over all 70,000 images and rank passing
entries by parameter count. The current best models are
`mnist/mlp-8-100-exact` at 8,190 parameters ($784\to8\to100\to10$) and
`fashion-mnist/mlp-23-64` at 20,241 ($784\to23\to64\to10$).

### 3.1 The linear baseline fails

A linear softmax classifier (7,850 parameters) reaches 91.1% on MNIST and 84.0%
on Fashion-MNIST with a short training run; least-squares one-hot
regression reaches 85.8% and 82.9%. So $L=0$ is not enough. This is the only
task-specific fact that is easy to prove computationally, and it immediately
rules out the 7,850-parameter linear model.

### 3.2 How close is the leaderboard?

The architectural floor is 805 (one hidden unit). The best MNIST model is 10.2×
that, but it is only **1.04×** the size of the linear model: 8,190 versus 7,850.
The nonlinearity buys the last 9% of accuracy for 340 extra parameters. This is
the most striking “closeness” result in the competition. Whatever the true
minimum is, it is bracketed by 805 and 8,190, and the practical frontier is very
near the linear count.

Fashion-MNIST is harder: 20,241 parameters is 2.58× the linear count and 25.1×
the architectural floor. The extra capacity is used for a second hidden layer
and wider features, not for a larger input layer.

### 3.3 The bounds that do not apply

- The float32 information floor of 7,267 is *not* a lower bound here. The
  labels are a deterministic function of the images, so a model does not have
  to store 70,000 labels; it has to compute the function. The coincidence that
  MNIST's 8,190 parameters is 1.13× the 7,267-parameter information floor is
  interesting but not a proof.
- The VC bound for fixed depth $L=2$ is
  $P=\Omega(N/(L\log N))\approx2{,}174$, and the universal bound is
  $\sqrt{N/\log N}\approx66$ (with base-2 logs; the constant depends on the
  log base). The best MNIST model is 3.8× the former; the bound is far too weak
  to be informative.
- The smooth-approximation bound is $P=\Omega(N^{1/(2\alpha)})$. For a Lipschitz
  boundary ($\alpha=1$) this is $\sqrt{N}=265$; for $\alpha=2$ it is
  $N^{1/4}=16.3$. The best models are 31× and 76× the $\alpha=1$ value. The
  bound is a worst-case statement over Hölder functions, not a lower bound for
  MNIST.

### 3.4 Asymptotic interpretation

If the decision boundary is smooth and the task is scaled to $N$ examples with
the same distribution, the required parameter count should grow *sublinearly*
— at least $\Omega(\sqrt N)$ under the Lipschitz assumption, and at most $O(N)$
by memorization. The clean-image benchmark is therefore not in the same
asymptotic regime as the shuffled-label benchmark, which is $\Theta(N)$. The
current MNIST model's $P/N=0.117$ is close to the arbitrary-label floor
($0.104N$), but that is a coincidence of this finite sample, not evidence that
clean MNIST requires linear capacity.

The open problem is a lower bound that uses the image distribution. Region
counting is not enough in 784 dimensions: a two-layer network can realize a
huge number of linear regions even with a narrow first layer, and the data lie
on a low-dimensional manifold (PCA needs 87 dimensions for 90% variance, 153 for
95%, 329 for 99%). A proof would need a data-dependent boundary-complexity
argument, for example a lower bound on the number of affine pieces needed to
separate the nearest opposite-label pairs. No such bound is known for this
benchmark.

## 4. Shuffled labels: the clean asymptotic case

`mnist-shuffled` and `fashion-mnist-shuffled` use a frozen random permutation of
the 70,000 labels. The labels are independent of the images, so the model must
encode the image-to-label mapping. The entropy of the label vector is
approximately

$$
N\log_2 10 = 70{,}000 \times 3.3219 = 232{,}535\ \text{bits}.
$$

A float32 checkpoint with $P$ parameters has $32P$ bits, so

$$
P \ge \frac{232{,}535}{32} = 7{,}267.
$$

This is a high-probability bound: there are at most $2^{32P}$ models, and a
uniform random labeling is realized by a given model with probability
$10^{-N}$; the union bound gives $2^{32P-N\log_2 10}$, which is exponentially
small below the floor. The frozen labels are a random draw, so the bound
applies up to a lower-order $O(\log N)$ term from the fixed class counts.

The current best is 115,932 parameters (MNIST) and 115,904 (Fashion), i.e.
**16× the floor**, or $P/N=1.66$ versus the floor's $0.104$. Both are
$\Theta(N)$, so the leaderboard is asymptotically optimal in order but a
constant factor 16 above the information limit. A naive four-bit-per-label table
would need $70{,}000\times4/32=8{,}750$ parameters (1.2× the floor); the current
architecture is 13.2× that table. There is substantial room for a bit-efficient
construction.

The VC bound is much weaker here: with depth $L=18$, $N/(L\log N)\approx242$;
the universal bound is $\sqrt{N/\log N}\approx66$ with base-2 logs. The float32
information floor dominates because the weights are finite precision. This is a good
illustration of why the precision model matters: in the real-valued model, deep
narrow networks can memorize $N$ points with $\Theta(\sqrt N)$ parameters, but
those parameters would carry far less than the 232,535 bits the labels require.

## 5. Chess evaluation

The two tiers score the fraction of positions whose predicted score is within
0.1 pawns of Stockfish; a pass requires all positions. The best models are
31,001 parameters ($773\to40\to1$) for 10k and 541,093
($773\to530\to246\to1$) for 100k.

### 5.1 The linear baseline is far away

A linear model with 774 parameters has mean absolute error 1.37 pawns (10k) and
1.46 pawns (100k), and only 7.4% of positions are within 0.1. The target is
strongly nonlinear. The targets have standard deviation 2.6 pawns, range ±10,
1,485 distinct centipawn values in the 10k tier and 1,974 in the 100k tier
(empirical entropy 9.15 and 9.30 bits per sample).

### 5.2 Bounds

- **Architectural:** 774 (the linear model is actually one parameter smaller
  than the one-hidden-unit model, 776). The best 10k model is 40× this; the 100k
  model is 699×. This floor is rigorous and almost useless.
- **Random-target information floor:** 2,077 (10k) and 20,763 (100k). The current
  models are 14.9× and 26.1× above. This is *not* a lower bound for the
  Stockfish targets, which are structured; it is the floor if the target vector
  were incompressible at 0.1-pawn resolution. Because there is no held-out split
  and the task is exact recall, the benchmark is closer to the memorization
  regime than a typical regression task, so the reference is informative.
- **VC:** for depth 2, $P=\Omega(N/(2\log N))\approx376$ (10k) and 4,348 (100k);
  the current models are 82× and 124× above. This is weak and not
  task-specific.

### 5.3 Scaling

The ratio $P(100k)/P(10k)=17.45$ for 10× more data gives an empirical exponent
of 1.24. If the target function were fixed and smooth, adding positions should
have diminishing returns and $P$ should grow sublinearly (or saturate). The
superlinear growth indicates that the larger set covers more of the complexity
of the evaluation function, or that the model is in a memorization regime. The
10k model uses 40 hidden units in one layer; the 100k model needs 530+246 units
in two layers. That is a depth-width response to the larger set.

The true asymptotic exponent is open. The worst-case bound is $\Theta(N)$; the
observed exponent is above 1, consistent with a growing effective complexity.
If the Stockfish evaluation is effectively generic at 0.1-pawn resolution, the
current models are within a factor 15–26 of the right order. If it is
structured enough that a much smaller model could fit, the current models are
far from optimal. There is no known lower bound that distinguishes these cases.

## 6. Snake

The Snake benchmarks fill an $n\times n$ board for 100 seeds; a pass requires
all 100 seeds. The observation has 24 features; the winning models select six
(head row, head column, distance to the bottom/right edges, and two corner
averages). With those six features the head cell is identifiable.

### 6.1 Simple 8×8 and 16×16

The current bests are 37 parameters ($6\to3\to4$) for 8×8 and 203 parameters
($6\to13\to6\to4$) for 16×16. The winning construction is a fixed Hamiltonian
cycle: in these submissions the action is a function of the head cell only, so
a compressed lookup table can implement it. The parameter density is
0.58 parameters per cell at 8×8 and 0.79 at 16×16; scaling from the two points
gives $P\propto n^{2.46}$, or $P\propto(\text{cells})^{1.23}$.

- **Architectural floor:** 15 (one hidden unit). The 8×8 model is 2.47× this,
  the 16×16 model 13.5×. The 8×8 model is close; the 16×16 model is not.
- **Arbitrary-action-table floor:** 2 bits per cell, $n^2$ cells, so
  $P\ge n^2/16$: 4 parameters at 8×8 and 16 at 16×16. The current models are
  9.3× and 12.7× above. The cycle is structured, so the true floor is lower;
  this is a reference, not a lower bound for the cycle policy.
- **Region-count bound:** a depth-$L$ network with $P$ parameters has at most
  $O(P^L)$ linear pieces along a one-dimensional slice. A Hamiltonian cycle has
  $\Omega(n)$ direction changes, so $P=\Omega(n^{1/L})$: $\Omega(n)$ for
  $L=1$, $\Omega(\sqrt n)$ for $L=2$. The current 37 (8×8) and 203 (16×16) are
  far above the $L=1$ values (8 and 16) and even further above the $L=2$ values
  (about 2.8 and 4), but the constants in the region argument are loose.

The asymptotic picture is a depth-width tradeoff. A phase-based construction
should be constructible with $O(\log n)$ parameters and $O(\log n)$ depth
(computing the row/column phase by binary search); a shallow lookup needs
$O(n^2)$; the current models are between, with an empirical exponent around 2.5
in $n$. The true minimum is open, but the scaling suggests the 16×16 model is
not asymptotically optimal.

### 6.2 Obstacles 8×8: an observation obstruction

No submission passes `snake-obstacles-8x8`; the best score is 22% of seeds. The
layouts are generated with a verified closed Hamiltonian tour, so a tour-following
policy exists in principle. The problem is that the policy sees only 24 local
features, not the global obstacle layout.

To test this, the search followed the canonical closed tour for all 100 seeds and
recorded the 24-feature observation and the tour's next action at every state. There are
4,493 unique observations and **329 collisions** where the same observation
requires different next actions across the 100 tours. In at least one collision
the *safe*-action sets are disjoint:

- observation: `(0.833, 0.0, 0.333, 0.0, 0.5, 0.25, 0.833, 1.083, 0, …, 0)`
- seed 6, step 9: the only action that does not immediately die is `{3}`
- seed 48, step 13: the only safe action is `{0}`

A deterministic memoryless policy must output the same action for both states.
Whichever it chooses, one state's only safe action is missed, so the snake dies.
If both states are unavoidable for any winning play, **no MLP can pass all 100
seeds, regardless of parameter count**. The states are reachable by following a
valid Hamiltonian tour; a different winning path might avoid one of them, but
the density of collisions (7.3% of observations) makes that unlikely. The
current best of 22% is consistent with this obstruction.

This is the only benchmark where the binding constraint may be observability
rather than parameter count. Proving unavoidability, or finding a winning policy
that avoids the collisions, is the key open problem.

## 7. Control

The three control tasks are small-parameter problems; the architecture floors are
correspondingly small.

### 7.1 CartPole: solved exactly

For $d=4$, $k=2$:

- no hidden layer: $P=(4+1)2=10$;
- one hidden layer of width $h\ge1$:
  $P=(4+1)h+(h+1)2=7h+2\ge9$;
- more layers add positive parameters.

So $P\ge9$. The submission `relu-threshold-9` has $[4,1,2]$, $P=4+1+2+2=9$, and
passes all 100 seeds. Therefore **$P_{\min}=9$ exactly**. This is the only
benchmark whose lower bound is proved and attained.

The architectural lower bound is machine-checked in Lean 4:
`MinimalNN.cartPole_architecturalMinimum` in
[`proofs/lean/Proofs/CartPole.lean`](../proofs/lean/Proofs/CartPole.lean)
proves that every accepted CartPole MLP architecture has at least nine
parameters and that nine is attained. The 100-seed pass is an empirical upper
bound recorded in [`proofs/bounds.json`](../proofs/bounds.json); together they
give the exact passing minimum. The Lean theorem does not formalise Gymnasium or
Box2D.

### 7.2 Pendulum

For $d=3$, $k=1$: linear $P=4$; one hidden unit $P=6$; the current best
`es-2` has $[3,2,1]$, $P=11$.

A random search over 3,000 linear policies and 3,000 one-ReLU policies (20
training seeds, then 100-seed evaluation of the top candidates) found best pass
fractions of 15/100 and 16/100 respectively, with mean returns below −1100, far
below the −500 threshold. This is evidence, not proof, that 4 and 6 parameters are
insufficient. The bracket is therefore $[6,11]$ (or $[11,11]$ if no 6-parameter
policy exists); the current best is 1.83× the one-hidden floor (11/6) and 2.75×
the linear floor (11/4). The threshold is lenient, but the random-search
evidence suggests a linear policy is insufficient for swing-up from random
initial angles.

### 7.3 LunarLander

For $d=8$, $k=4$: linear $P=36$; one hidden unit $P=17$; two hidden units
$P=30$; three $P=43$; four $P=56$; five $P=69$. The current best `distill-5` is
$[8,5,4]$, $P=69$, distilled from a 51×51 teacher.

A linear controller is unlikely to land safely on all 100 seeds with discrete
thrusters and the wide-start reset force, but this is not proved. The
architectural bracket is $[17,69]$; if linear policies are excluded, $[36,69]$.
The current best is 1.9× the linear floor and 4.1× the architectural floor. The
gap is optimization, not expressiveness: a 69-parameter policy has only five
hidden units, so its action map is piecewise affine with at most $2^5=32$ linear
cells in the general-position bound — already a very small controller.

### 7.4 Asymptotic in state dimension

For a linear state-feedback policy, a $d$-state, $k$-action controller needs a
$d\times k$ gain plus $k$ biases, $P=dk+k=\Theta(dk)$. A ReLU network can beat
this by a constant factor by factorizing the gain through a narrow hidden layer:
CartPole's 9-parameter policy is one parameter smaller than the 10-parameter
linear policy. Heuristically, a piecewise-linear policy with $R$ action regions
needs $P=\Omega(dk+R)$; for small $d$ the architectural floor dominates. This group is
the closest to solved: CartPole exact, Pendulum and LunarLander within small
constants of the floor.

## 8. What the gaps mean

- **CartPole:** done. The lower bound is exact and attained.
- **Pendulum, LunarLander, Snake 8×8:** within 2–4× of a rigorous architectural
  floor; the remaining gap is likely optimization/search, not a missing
  theoretical ingredient.
- **Clean MNIST:** within 1.04× of the linear model's parameter count, but 10×
  the architectural floor. No task-specific bound exists; the empirical
  frontier is the only guide.
- **Clean Fashion-MNIST:** 2.58× the linear count, 25× the architectural floor;
  the task needs more nonlinear capacity, and the gap is larger.
- **Shuffled labels:** 16× the float32 information floor, and $\Theta(N)$ like
  the floor. This is the cleanest asymptotic gap; a bit-efficient construction
  is the obvious challenge.
- **Chess:** 15–26× the random-target information floor and $\Theta(N)$ in
  order, but the floor is not a true bound for the structured Stockfish
  targets. The observed exponent 1.24 suggests a memorization-like regime.
- **Snake 16×16:** 13.5× the architectural floor and $P\propto n^{2.46}$;
  asymptotically the most clearly suboptimal of the small tasks.
- **Snake obstacles:** no passing entry; the collision evidence suggests an
  observability obstruction. This is a different kind of lower bound: not a
  parameter count, but, if the colliding states are unavoidable, a statement
  that no memoryless policy with this observation can be correct on all seeds.

## 9. Caveats

1. The information floors are high-probability statements for random
   labels/targets. The frozen shuffled labels are a random draw, so the bound
   applies; the clean image labels and the Stockfish targets are structured, so
   their floors are references, not lower bounds.
2. The VC and Hölder bounds are worst-case over function classes. They do not
   prove that the specific benchmark requires that many parameters.
3. The architectural floors are exact for the MLP family but usually far below
   the truth. They are useful for detecting when a task is essentially solved
   (CartPole) and for bracketing the small control tasks.
4. The float32 assumption is essential for the information floor. With
   real-valued weights, deep narrow networks can memorize $N$ points with
   $\Theta(\sqrt N)$ parameters; with float32, the labels' entropy sets a
   stronger linear floor for large $N$.
5. The obstacle collision is a proof that the canonical tour is not
   observation-realizable; it is not a proof that no winning policy exists. A
   policy that chooses a different Hamiltonian path could, in principle, avoid
   the colliding states.
6. The Pendulum search is a random search, not an exhaustive proof; a targeted
   optimization might find a 4- or 6-parameter policy.

## 10. Reproduction notes

The parameter counts follow the formula in Section 1 and match the evaluator.
The linear baselines were trained with the repository's `torch`/`numpy`
environment: linear softmax 91.1% (MNIST) and 84.0% (Fashion); least-squares
one-hot 85.8% and 82.9%. PCA on a 20,000-image sample needs 87/153/329
dimensions for 90/95/99% variance. The chess linear fit used the 773-feature
encoder and achieved MAE 1.37/1.46 pawns and 7.4% within 0.1. The obstacle
collision search followed `find_closed_tour` for all 100 seeds, recorded
`observe(engine)` at each step, and compared the tour's next action across
states with identical observations; safe-action sets were computed by cloning
the engine and stepping each of the four actions.

The CartPole architectural theorem is checked with
`cd proofs/lean && lake build`; the certificate is checked with
`uv run python -m proofs check`. See [`proofs/README.md`](../proofs/README.md)
for the trust boundary and evidence classes.

The main targets for future work are: a two-layer lower bound for MNIST, a
bit-efficient construction for the shuffled labels, a structure-aware lower
bound for chess, and a proof or refutation of the obstacle observability
obstruction.

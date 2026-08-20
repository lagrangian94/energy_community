# Applying the Rawlsian Maximin Criterion over the Optimal Dual Face of $(\mathrm{DWR}_N)$

Scope: this note concerns **Sec. `sec:lr` only** (the Lagrangian-relaxation route that we
actually compute). It does **not** touch Sec. `sec:cpp`.

---

## 1. Where the freedom is

The dual of the Dantzig--Wolfe master `eq:dwr` is

$$
(\mathrm{DWR}_N^{\,*}):\quad
\min_{\theta,\sigma}\ \ \theta^{\top}\!\!\sum_{j\in N}\! b_j+\sum_{j\in N}\sigma_j
$$
$$
\text{s.t.}\quad
\theta^{\top}a_j^{q}+\sigma_j\ \ge\ w_j^{q}\quad \forall j\in N,\ q\in Q_j,
\qquad
A_0^{\top}\theta\ \ge\ c_0,\qquad \theta\ge 0 ,
$$

with $\theta$ dual to the linking rows `eq:dwr_link` and $\sigma_j$ dual to the convexity
row `eq:dwr_conv` (an equality, hence $\sigma_j$ free). By Lemma `lem:lpg`(b) the Owen
solution is

$$
\chi_j(\theta,\sigma)\ =\ \theta^{\top}b_j+\sigma_j ,
$$

an **affine** function of the dual variables. Whenever $(\mathrm{DWR}_N)$ is primal
degenerate, its optimal dual face is not a singleton and a **set** of Owen solutions
exists, all of them equally stable and equally efficient. Classical Owen picks one of
them arbitrarily — whichever vertex the LP solver happens to return.

**Where the freedom is *not*.** In the dual objective $\sigma_j$ carries coefficient
$+1$ under a minimisation and is bounded only from below, so at any dual optimum

$$
\sigma_j=\max_{q\in Q_j}\{w_j^{q}-\theta^{\top}a_j^{q}\}
=\max_{(x_j,y_j)\in\mathcal{X}_j^{\mathrm{MIP}}}\{f_j(x_j,y_j)-\theta^{\top}A_jx_j\},
$$

i.e. $\theta$ pins $\sigma$. All non-uniqueness lives in

$$
\Theta^{*}:=\operatorname*{arg\,min}_{\theta\in\Lambda} v^{\mathrm{LR}}(N;\theta)
=\Bigl\{\theta\in\Lambda:\ \textstyle\sum_{j\in N}\sigma_j(\theta)=z^{*}\Bigr\},
\qquad z^{*}=\mathrm{val}(\mathrm{DWR}_N).
$$

Economically, $\Theta^{*}$ is the set of community price vectors (balance / reserve /
peak) supporting the same optimal welfare $z^{*}$ — the familiar multiplicity of optimal
prices in electricity markets.

---

## 2. The selection rule

Apply the Rawlsian maximin criterion: among Owen allocations, which are otherwise
equivalent in stability and efficiency, select the one maximising the payoff of the
worst-off member.

$$
(\mathrm{MM}):\quad
\max_{\theta,\sigma,t}\ t
$$
$$
\text{s.t.}\quad
\theta^{\top}b_j+\sigma_j\ \ge\ t \quad\forall j\in P
\tag{R1}
$$
$$
\theta^{\top}\!\!\sum_{j\in N}\!b_j+\sum_{j\in N}\sigma_j\ \le\ z^{*}
\tag{R2}
$$
$$
\theta^{\top}a_j^{q}+\sigma_j\ \ge\ w_j^{q}\ \ \forall j\in N,\ q\in Q_j,
\qquad A_0^{\top}\theta\ge c_0,\ \theta\ge 0 .
\tag{R3}
$$

- $P\subseteq N$ is the **protected set**. $P=N$ is the plain Rawlsian rule; a strict
  subset protects only those members. Nothing else in the formulation changes — the
  index set of (R1) is the only knob. (R2) and (R3) always range over all of $N$.
- (R2) **must be imposed as an equality.** This note originally wrote it as $\le z^{*}$
  "for numerical safety", relying on dual feasibility to supply $\ge z^{*}$. In a
  *restricted* (MM) that reasoning fails: (R3) holds only over the pool, so nothing forces
  $\ge z^{*}$, and any slack lets the LP lower the worst-off share by **under-collecting**
  — buying it with subsidy instead of redistributing along $\Theta^{*}$. Implemented with
  a $10^{-7}|z^{*}|$ tolerance this produced a complete and entirely spurious set of
  results; see §6.1.

**What $(\mathrm{MM})$ determines, and what it does not.** The optimal value $t^{*}$ is
unique; it is the object the Rawlsian criterion defines. The optimal *argument* $\chi$
need not be: coordinates outside $\arg\min_{j\in P}$ can carry slack that the objective
does not see. This is a property of the criterion, not a defect of the formulation, and is
addressed in §6.2 rather than by adding machinery here.

---

## 3. Properties

**(P1) Exactness of the LP form.** $(\mathrm{MM})$ optimises over the *exact* set of Owen
solutions, not a relaxation. Reason: (R2)+(R3) force $\theta\in\Theta^{*}$ and
$\sigma_j=\sigma_j(\theta)-\theta^{\top}b_j$ simultaneously, since
$\sum_j\sigma_j\ge\sum_j\sigma_j^{\min}(\theta)$ and
$\sum_j\sigma_j^{\min}(\theta)+\theta^{\top}b(N)=v^{\mathrm{LR}}(N;\theta)\ge z^{*}$
leave no slack.

**(P2) $\chi$ is affine on $\Theta^{*}$.** Each $\sigma_j(\cdot)$ is convex piecewise
linear on $\Lambda$; for $\theta_1,\theta_2\in\Theta^{*}$ and
$\theta_\lambda=\lambda\theta_1+(1-\lambda)\theta_2$, convexity gives
$\sigma_j(\theta_\lambda)\le\lambda\sigma_j(\theta_1)+(1-\lambda)\sigma_j(\theta_2)$ for
every $j$, while summing over $j$ gives $z^{*}$ on both sides — so every inequality is
tight. Hence the Owen set $\{\chi(\theta):\theta\in\Theta^{*}\}$ is a polytope (the affine
image of $\Theta^{*}$) and the max-min over it is genuinely an LP. Individually the
$\sigma_j$ are kinked on $\Lambda$; they flatten only on the optimal face.

**(P3) Stability is free.** Weak duality `eq:weakdual` holds at *every* $\theta\in\Lambda$,
so $\sum_{j\in S}\chi_j\ge v^{\mathrm{MIP}}(S)$ for all $S$ regardless of which optimum is
selected. In particular $S=\{j\}$ gives $\chi_j\ge v^{\mathrm{MIP}}(\{j\})$: **protecting a
subset $P$ can never push $N\setminus P$ below stand-alone.** This is what licenses
restricting (R1) to $P\subsetneq N$. Caveat: weak duality applies to the *exact*
$\sigma_j(\theta)$, which is why the reported allocation must be the re-evaluated one —
see Step (2d) in §4.

**(P4) The budget is free.** $\theta\in\Theta^{*}$, so the residual subsidy is unchanged:
$\omega^{\mathrm{LR}}=\sum_j\chi_j-v^{\mathrm{MIP}}(N)$ exactly as in
Prop. `prop:opap`. Redistribution costs nothing.

**(P5) No homogeneity needed.** (P1)--(P4) use only Assumption `as:pool` (additivity of the
linking rows). $b_j=0$, required by Thm. `thm:core`, is *not* used; $\theta^{\top}b_j$
stays linear in $\theta$. So the selection sits inside Sec. `sec:lr` without importing the
assumptions of Sec. `sec:cpp`. With $b_j\neq0$ the reading is in fact sharper: $\chi_j$
splits into an endowment value $\theta^{\top}b_j$ and a pattern price $\sigma_j$, and
moving along $\Theta^{*}$ trades one against the other — choosing $\theta$ *is* choosing
whose endowment is priced highly.

**(P6) No coalition oracle.** $(\mathrm{MM})$ never evaluates $v^{\mathrm{MIP}}(S)$ for
$|S|>1$ and never solves the separation problem `eq:rg_sep`.

---

## 4. The algorithm

**Input:** $\{\mathcal{X}_j^{\mathrm{MIP}},f_j,A_j,b_j\}_{j\in N}$, $A_0,c_0$, protected set
$P\subseteq N$, tolerance $\epsilon\ge0$.

### Phase 0 — preparation
- Solve the $n$ single-prosumer MIPs $v^{\mathrm{MIP}}(\{j\})$, needed for the individual
  rationality check.
- Solve the grand-coalition MIP $v^{\mathrm{MIP}}(N)$ → benchmark and subsidy base.

### Phase 1 — column generation (unchanged)
- Solve the restricted master `eq:dwr` over the current pool $\hat Q_j$ → $(\theta,\sigma)$.
- Solve the $n$ independent pricing MIPs `eq:pricing` in parallel at $\theta$.
- Add violated columns and repeat; terminate when none prices out.
- Output: $z^{*}$, $\theta^{*}$, pool $\hat Q_j$, and the classical Owen allocation
  $\chi_j^{\mathrm{LR}}=\theta^{*\top}b_j+\sigma_j^{*}$ (with the correction of §7), with
  residual subsidy $\omega^{\mathrm{LR}}=\sum_j\chi_j^{\mathrm{LR}}-v^{\mathrm{MIP}}(N)$.

### Phase 1.5 — singleton screen
- For each $j$, maximise and minimise $\chi_j$ subject to (R2)+(R3)$|_{\hat Q}$: $2n$ LPs.
- Since $\hat\Theta^{*}\supseteq\Theta^{*}$, each width is an **upper bound** on the true
  width. **Width $=0$ therefore certifies $\Theta^{*}=\{\theta^{*}\}$** — skip Phase 2 and
  report $\chi^{\mathrm{LR}}$.
- A positive width certifies nothing (it may be a truncation artefact), so it is a
  screen for skipping work, not a reportable measurement of the true spread.

### Phase 2 — maximin selection (second cutting-plane loop)
- **(2a)** Solve $(\mathrm{MM})$ with (R3) restricted to $q\in\hat Q_j$ →
  $(\hat\theta,\hat\sigma,\hat t)$.
- **(2b)** Re-price: solve the same $n$ MIPs `eq:pricing` at $\hat\theta$ → exact
  $\sigma_j(\hat\theta)$ and maximising pattern $q_j^{\mathrm{new}}$.
- **(2c)** Let $\delta_j:=\sigma_j(\hat\theta)-\hat\sigma_j\ (\ge0)$. If
  $\max_j\delta_j\le\epsilon$, stop. Otherwise append $q_j^{\mathrm{new}}$ for every $j$
  with $\delta_j>\epsilon$ and return to (2a). Note the re-pricing runs over **all**
  $j\in N$, not only $j\in P$: (R3) is a global constraint and a stale column anywhere
  loosens the face everywhere.
- **(2d) Output.** Report the *re-evaluated* allocation, not the LP's $\hat\sigma$:
  $$\chi_j=\hat\theta^{\top}b_j+\sigma_j(\hat\theta),\qquad
    \omega=\textstyle\sum_j\chi_j-v^{\mathrm{MIP}}(N).$$
  This is what makes (P3) applicable at any stopping point.

### Phase 3 — reporting
Iteration count of Phase 2, total pricing-MIP calls, $\omega$ against
$\omega^{\mathrm{LR}}$ (any discrepancy *is* the truncation cost), $t^{*}$ against the
worst-off position under classical Owen, and — as verification only — a blocking-coalition
check via `eq:rg_sep`.

### Why the converged pool is not already enough

The natural objection is that Phase 1 ran to convergence, so $\hat Q_j$ ought to be
sufficient. It is sufficient for what column generation set out to prove, and that is
strictly less than what $(\mathrm{MM})$ needs.

$\sigma_j(\theta)=\max_{q\in Q_j}\{w_j^{q}-\theta^{\top}a_j^{q}\}$ is a max of
exponentially many affine functions, hence convex piecewise linear in $\theta$. The pool
gives

$$
\hat\sigma_j(\theta):=\max_{q\in\hat Q_j}\{w_j^{q}-\theta^{\top}a_j^{q}\}\ \le\ \sigma_j(\theta)
\qquad\text{for every }\theta,
$$

a **lower** approximation, since the max runs over fewer terms. Convergence of column
generation says exactly that no column prices out at $\theta^{*}$, i.e.
$\hat\sigma_j(\theta^{*})=\sigma_j(\theta^{*})$ for all $j$. The two agree **at that one
point**. Away from it $\hat\sigma_j$ still underestimates.

Ordinary column generation never needs more. It checks dual feasibility at a single
incumbent point; once no column is violated there, that point is dual feasible for the
full problem and optimality follows. Read as a cutting-plane method on the dual, each
column is a cut, and the algorithm only ever accumulates the cuts required to remove the
infeasible points met on the way to $\theta^{*}$. It stops the moment the current point is
feasible, so the outer approximation of $\Lambda$ is tight at $\theta^{*}$ and loose
everywhere else.

$(\mathrm{MM})$ instead searches the whole optimal face. Dropping (R3) for
$q\notin\hat Q_j$ removes constraints, so $\hat\Theta^{*}\supseteq\Theta^{*}$, and the only
way $(\mathrm{MM})$ can redistribute is by moving $\theta$ off $\theta^{*}$ — **precisely
the direction in which the pool is loose.** The restricted program exploits the gap
$\sigma_j-\hat\sigma_j$: it believes a smaller $\sigma_j$ is attainable and returns an
allocation more equal than anything actually feasible. Re-pricing exposes this — some $j$
has $\sigma_j(\hat\theta)>\hat\sigma_j$, whence $\sum_j\sigma_j(\hat\theta)>z^{*}$ and
$\hat\theta\notin\Theta^{*}$.

**The accounting is exact: convergence of Phase 1 buys $z^{*}$, and nothing else.** That
matters because it is what keeps (R2) identical between the restricted and full programs,
which is in turn what makes $\hat\Theta^{*}\supseteq\Theta^{*}$ true. Had Phase 1 been
stopped early we would have $\hat z<z^{*}$, (R2) would read $\le\hat z$ — tighter, not
looser — and the inclusion would fail in the wrong direction, possibly to the point of
$\hat F\cap F=\emptyset$. Dual feasibility at the single point $\theta^{*}$ comes along for
free but expires the moment $\theta$ moves.

**Truncation is safe but not free.** Stopping at any $\hat\theta$ with
$A_0^{\top}\hat\theta\ge c_0$ and reporting per (2d) still yields a stable allocation (P3),
with excess

$$
\bigl[v^{\mathrm{LR}}(N;\hat\theta)-v^{\mathrm{MIP}}(N)\bigr]-\omega^{\mathrm{LR}}
=\sum_j\sigma_j(\hat\theta)-z^{*}\ \le\ n\epsilon .
$$

Fairness is then bought with subsidy, and the amount is measurable — which makes the
fairness/subsidy trade-off plottable as a function of Phase-2 iteration count.

---

## 5. Convergence

**Assumption.** Each $\mathcal X_j^{\mathrm{MIP}}$ is bounded, so
$\mathrm{conv}(\mathcal X_j^{\mathrm{MIP}})$ has finitely many extreme points and
$|Q_j|<\infty$. (Capacity limits give this, but it must be stated.)

**Feasibility and boundedness.** $(\theta^{*},\sigma^{*})$ from Phase 1 is feasible for
every restricted $(\mathrm{MM})$, so no iterate is infeasible. Summing (R1) over $j\in P$,
using (R2), and bounding the unprotected members below by (P3),

$$
t\ \le\ \frac{1}{|P|}\Bigl(z^{*}-\sum_{j\in N\setminus P}v^{\mathrm{MIP}}(\{j\})\Bigr),
$$

which for $P=N$ reduces to $t\le z^{*}/n$. So $\hat t$ is bounded however truncated the
pool, and each (2a) is a well-posed LP.

**Finite termination.** At any non-terminating iteration there is a $j$ with

$$
w_j^{q_j^{\mathrm{new}}}-\hat\theta^{\top}a_j^{q_j^{\mathrm{new}}}
=\sigma_j(\hat\theta)>\hat\sigma_j
=\max_{q\in\hat Q_j}\{w_j^{q}-\hat\theta^{\top}a_j^{q}\},
$$

hence $q_j^{\mathrm{new}}\notin\hat Q_j$: **every iteration adds at least one genuinely new
column.** The pool is monotone increasing inside the finite set $\prod_j Q_j$, so the loop
terminates after finitely many iterations. Cycling is impossible for the same reason — if
$\hat\theta$ recurred, its maximising patterns would already be pooled and $\delta_j=0$.

**Exactness at termination.** With $\epsilon=0$, termination gives
$\hat\sigma_j=\sigma_j(\hat\theta)$ for all $j$, so (R3) holds not only on $\hat Q_j$ but on
all of $Q_j$; hence $(\hat\theta,\hat\sigma)$ is feasible for the full $(\mathrm{MM})$ and
$\hat t\le t^{*}$. The restricted program is a relaxation, so $\hat t\ge t^{*}$. Therefore
$\hat t=t^{*}$ and $\hat\theta\in\Theta^{*}$: termination is itself the optimality
certificate.

**What this does not give.**
- *Finite is not practical.* No polynomial bound on the iteration count; worst case scales
  with $|Q_j|$. Warm-starting from the Phase-1 pool should help, but that is an empirical
  claim and the count must be **measured and reported, not asserted**.
- *Anytime safety.* Stopping early is legitimate under (2d); the price is the subsidy
  excess bounded by $n\epsilon$ above.
- *Pricing MIP gaps.* If `eq:pricing` is stopped with a gap, use the incumbent (lower
  bound) to generate cuts but the MIP **dual bound (upper bound)** as $\sigma_j$ in any
  stability claim, so weak duality stays conservative. Exact termination then degrades to
  $\epsilon$-termination and must be stated as such.
- *Scope.* Under a non-additive coupling such as the shared export cap `eq:expcap`,
  Lemma `lem:lpg` fails, the decomposition of $\sigma_j$ is void, and the entire argument
  collapses. Assumption `as:pool` must appear in the hypothesis of any convergence
  statement.

---

## 6. Open issues to settle before writing this up

1. **Is $\Theta^{*}$ a singleton?** If so the whole construction is vacuous. Phase 1.5
   answers this in the safe direction at cost $2n$ LPs; report the widths against
   $\omega^{\mathrm{LR}}$. If they collapse, drop the section.

   *Measured* (`maximin.py`; $|T|=24$, reserve 56 + peak 150, 4-hour symmetric, one day,
   both run to convergence):

   | | 6p | 15p |
   |---|---|---|
   | screen width | $0.396$ | $0.279$ |
   | worst-off position actually moved | $0.0026$ | $0.0048$ |
   | as a share of $\omega^{\mathrm{LR}}$ | $0.05\%$ | $0.06\%$ |

   **The screen overstated the freedom by 150x and 58x respectively**, since
   $\hat\Theta^{*}\supseteq\Theta^{*}$ makes its widths an upper bound and almost all of
   both was pool truncation. The rigorous cheap statement is instead the bracket the
   Phase-2 loop provides for free: $\hat t$ lower bounds $t^{*}$ at every iteration and
   classical Owen's worst-off value upper bounds it (since $\theta^{*}\in\Theta^{*}$), so
   their difference caps the remaining gain whether or not the loop has converged.

   **MEASURED, and the answer is that $\Theta^{*}$ is a point.** With (R2) imposed as a
   hard equality $\sum_j\sigma_j=z^{*}$, the maximin improvement is numerically zero at
   $n=6$ on every day tested (five spike days, all terminating in **one** iteration — there
   is no cutting-plane work to do), and $10^{-6}$–$10^{-5}$ at $n=15,30$, which on payoffs
   of order 40 EUR is $10^{-7}$ relative and inside the LP's own accuracy. **The section is
   vacuous on these instances and should be cut.**

   *A full day of results said otherwise and every one of them was an artefact*, recorded
   here because the failure mode is easy to repeat. (R2) had been implemented as
   $\sum_j\sigma_j\ge z^{*}-\texttt{r2\_slack}$ with $\texttt{r2\_slack}=10^{-7}|z^{*}|$
   "for numerical safety" — as this note itself suggests in §2. That inequality lets the LP
   lower the worst-off share by **under-collecting**, i.e. by buying it with subsidy rather
   than redistributing along $\Theta^{*}$ — exactly the trade §4 calls *"fairness is then
   bought with subsidy"*. It produced a rich and entirely spurious phenomenology: a
   68.7%-of-$\epsilon^{\mathrm{LR}}$ spike on one day, a plausible dependence on the peak
   channel, a size trend. Halving the slack halved every one of them; a $100\times$
   tightening scaled all of them by exactly $1/100$, leaving
   improvement$/\texttt{r2\_slack}$ fixed to four significant figures. **A quantity
   proportional to a tolerance is that tolerance.**

   Two lessons for anything built on this LP. First, `truncation_cost` equalling
   `r2_slack` in every single run was the tell, printed all along and read as noise;
   (P4) says the residual subsidy is *unchanged*, so any run where it moves by the
   tolerance is not on $\Theta^{*}$. Second, "the effect is $1649\times$ the tolerance so
   it cannot be the tolerance" is **not** a valid safeguard — day 4 was exactly that and
   was entirely artefact. Vary the tolerance and check the answer does not move with it.

3. **Raw payoff, not surplus.** With $\underline v_j$ removed, (R1) equalises the level of
   $\chi_j$ rather than the gain $\chi_j-v^{\mathrm{MIP}}(\{j\})$. A reviewer will ask why
   a member with a strong stand-alone position and a pure consumer should be measured on
   the same absolute scale. The formulation is unchanged either way — $\underline v_j$
   would enter (R1) as a constant shift — so this is a presentation decision to make once
   the numbers are in, not a modelling commitment now.

   *The numbers are in, and it is causal rather than cosmetic — it is why $t^{*}$ does
   not move.* On the level scale a pure consumer with no assets sits at the $\arg\min$
   essentially always (it only ever pays), and its $\chi_j$ is capped from above by the
   value of its own pattern set, which no choice of $\theta$ can lift. Measured: the
   worst-off member is an **asset-less consumer at all three sizes** (u4, u4, u25), and
   members with almost no patterns (9–10 columns) shift **exactly zero**, while asset
   owners with large pattern sets move most (u18 at $n=30$ shifts $0.0655$, **16x** the
   worst-off member's entire gain). So $\Theta^{*}$ is not a point — it has width, in
   coordinates (R1) never looks at. The $n=15$, two-electrolyzer instance is the limiting
   case: the worst-off member is idle, pinned at $\chi_j=0.000000$, improvement exactly
   $0$, with $\omega^{\mathrm{LR}}=8.279$ sitting there.

   Recomputing the $\arg\min$ on $v^{\mathrm{MIP}}(\{j\})-\chi_j$ moves it off the
   asset-less consumer at 15p (u4 $\to$ u12) and 30p (u25 $\to$ u21, an electrolyzer and
   solar owner with 217 columns) — into the region where the face demonstrably has width.
   Whether $t^{*}$ then improves is **unmeasured**: those shifts came from the level-scale
   objective and say nothing about how far u21 moves when it *is* the objective. (R1) has
   to be rewritten as $\sigma_j-v^{\mathrm{MIP}}(\{j\})\le t$ — a constant per-member
   shift — and the loop re-run before §5 is judged.
4. **Novelty positioning.** Fairness selection over stable sets is **not** new: Valencia
   Zuluaga & Oren optimise a preference $f$ over the uniform-price core (which they
   identify with Owen imputations), and Fioriti et al. (*Fair Least Core*, IEEE TEMPR 2025)
   maximise a strictly concave $f$ over the core / least core. The claim must be
   computational, not conceptual: selection over a **mixed-integer** community game's
   optimal-multiplier face, at zero budget cost, using only pricing MIPs and no coalition
   separation. Note that switching to a strictly concave objective to force uniqueness
   would move us onto exactly their ground; this is a further reason to keep the plain
   maximin LP.
5. **Knock-on effect on prices.** $\theta$ also generates the CHP settlement
   $\pi^{\mathrm{chp}}\!\cdot x^{\mathrm{MIP}}$. The same selection therefore perturbs the
   CHP allocation; whether it also compresses the uneven deductions $\Delta_j$ borne by
   nonconvex prosumers is testable in the same run at no extra cost.
6. **Scope limit.** As in §5: this is an additivity issue, not a homogeneity issue.

---

## 7. Notation fix required in the draft

`eq:sigma` defines $\sigma_j(\theta):=\theta^{\top}b_j+\max_q\{\cdot\}$, whereas the dual
variable $\sigma_j^{*}$ of `eq:dwr_conv` equals $\max_q\{\cdot\}$ alone. Hence

$$
\sigma_j(\theta^{*})=\theta^{*\top}b_j+\sigma_j^{*}\ \neq\ \sigma_j^{*}
\quad\text{unless } b_j=0,
$$

so `eq:chi_lr`, which writes $\chi^{\mathrm{LR}}:=\sigma(\theta^{*})=\sigma^{*}$, is
consistent only in the homogeneous case introduced later in Sec. `sec:cpp`. Since (R1)
must be written with the correct object, resolve this first — either rename the dual
variable or restate `eq:chi_lr` as $\chi_j^{\mathrm{LR}}=\theta^{*\top}b_j+\sigma_j^{*}$.

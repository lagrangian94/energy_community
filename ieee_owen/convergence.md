# Why the two loops fail to converge, and what was done about each

Written 2026-08-08, while implementing `maximin criterion.md` sec.4.

There are **two** cutting-plane loops in this pipeline and they fail for unrelated
reasons. Keeping them apart matters, because one is a defect in an algorithm
specification that this work introduced, and the other is a pre-existing numerical
tolerance problem in the shared layer that has been silently costing 1800 seconds a run.

| | loop | status |
|---|---|---|
| §1 | **Phase 2** — maximin selection over `Theta*` (`maximin.py`, new) | diagnosed and **fixed** |
| §2 | **Phase 1** — Dantzig-Wolfe column generation (`pricer.py`, shared) | diagnosed and **fixed** |

Neither is a failure of the underlying theory. In both cases the mathematics is right
and the *procedure* is underspecified.

---

## 1. Phase 2 — the maximin loop

### 1.1 What (MM) is supposed to do

Solve the restricted (MM) over the current column pool, re-price at the resulting
`theta_hat` with `n` independent pricing MIPs, append every column that prices out, and
repeat. `maximin criterion.md` sec.5 proves this terminates: at any non-terminating
iteration some `q_j^new` is not in `Q_hat_j`, so **every iteration adds at least one
genuinely new column**, and the pool grows monotonically inside the finite set
`prod_j Q_j`.

That proof is correct and nothing below contradicts it. But it is a statement about the
sequence of *pools*, and it quietly assumes each iteration hands back a well-defined
`theta_hat`. It does not.

### 1.2 Failure A — the restricted face has no vertex

A coupling row that **every pooled column has coefficient zero in** does not appear in
any (R3) constraint. In the 6-prosumer instance this is the heat balance at `t = 6`: the
heat pump is off in every pattern column generation has generated so far, so no column
touches that row. If the row also carries no shared-variable constraint (R4), its dual
is completely free — both directions are directions of recession.

The feasible set therefore **contains a line**, and a polyhedron containing a line has no
vertex at all. Simplex has nowhere to land and returns the variable at its artificial
infinity bound (SCIP: `1e20`). The first run died there, not in the LP but one step
later: `Error setting objective` from the pricing subproblem, because a dual of `1e20`
makes the re-pricing objective unrepresentable.

Note what is and is not unbounded. The *objective* `t` is bounded — sec.5's
`t <= z*/n` argument is untouched. It is the *argmin* that is unbounded. So this is not
a wrong theorem; it is a missing rule. Sec.6.2 of the note already flags that (MM) pins
`t*` but not the allocation vector, and treats it as a reporting nuisance. In `theta`
space the same freedom is not a nuisance — it stops the loop from running.

### 1.3 Failure B — an unstabilised cutting-plane method makes no progress

Give `theta` a finite box and the crash goes away, but the loop still does not work.

Only `max_j sigma_j` enters the objective, so every iterate is free to pick **any** point
of the restricted face achieving it, and a simplex solver picks an extreme one. Extreme
is precisely where the pool is loosest — the pool was built to certify optimality at
`theta*` and is tight nowhere else — so the re-priced violation is large, the cut is
generated far from anything that matters, and the region around the true optimum is
never tightened.

Measured, 6p, `|T| = 24`, reserve 56 + peak 150, 4-hour symmetric, one day:

```
box only, no tie-break, 50 iterations:
  t_hat      = 29.991633   for all 50 iterations, to the last digit
  max_j d_j  oscillates between 3.9e-1 and 1.3e+1, no trend
  box        1 PRICED row pinned at the bound, every iteration
```

Zero progress. Not slow — zero. The note does say `Finite is not practical` and that the
iteration count must be measured rather than asserted; this is the extreme value of that
warning.

### 1.4 The remedies

**(a) A finite box on `theta`.** `default_pi_bound()` scales it off the model's own price
data — two orders of magnitude above the largest tariff, floor `1e4`. The number of rows
sitting on the box is counted every iteration, split into rows some column touches and
rows none does. Only the first kind could distort the answer; at 6p convergence there
were **zero** of them.

The box does not paper over anything. A `theta` at infinity on an untouched row genuinely
IS dual infeasible for the full problem, and exposing that is exactly Phase 2's job. The
box only keeps the iterate representable long enough for the loop to do it.

**(b) A lexicographic second stage.** Solve (MM) for `t*`, then **fix `t` at `t*`** and
minimise `||theta - theta*||_1` over the remaining face. Two LPs per iteration instead of
one.

Lexicographic, not a weighted sum — the fairness objective is never traded against the
proximity term, so `t*` is exactly what it was. What changes is only the choice *among*
optima, and it now has a statable rule behind it: **stay as close as possible to the price
vector column generation actually produced.**

This is the same idea Phase 1 already applies via Wentges smoothing, transplanted to the
second loop. It also disposes of sec.6.2: `theta_hat` is reproducible instead of
solver-dependent, and since (2d) reads `chi` off the pricing oracle at `theta_hat`, so is
the reported allocation.

### 1.5 What the fix does not change

- **`t*` is unchanged.** Stage B runs at fixed `t`.
- **The certificate is unchanged.** Termination is still `max_j delta_j <= eps`, which
  still gives `sigma_hat_j = sigma_j(theta_hat)` for all `j`, hence (R3) on all of `Q_j`,
  hence `theta_hat in Theta*` and `t_hat = t*`. How the iterates got there is irrelevant
  to whether the endpoint is optimal.
- **Stability is unchanged.** (P3) holds at every dual-feasible point regardless.

### 1.6 Evidence

| configuration | outcome |
|---|---|
| no box, no tie-break | crash — dual at `1e20`, pricing objective unrepresentable |
| box only | 50 iterations, `t_hat` frozen, `max_j delta_j` oscillating 0.4 ↔ 13 |
| box + `l1` proximity | `t_hat` monotone up, `max_j delta_j` monotone down, **converged in 143 iterations / 858 pricing MIPs / 61 s** |

At convergence: `t*` in `[30.063114, 30.065750]`, truncation cost `3.0e-4`, and the
selected point passes the separation stability check (`excess <= 0` for `chi`,
`<= eps^LR` for the gap-corrected point).

### 1.7 Residual: the truncation floor is the (R2) tolerance, not the pricing tolerance

(R2) is written `sum_j sigma_j >= z* - r2_slack` with `r2_slack = 1e-7 |z*|`, for the
numerical-safety reason the note gives. That slack widens the face by the same amount, so
the honest bound on the truncation cost is `n * eps + r2_slack`, and at 6p the second term
dominates: measured `3.04e-4`, against `n * eps = 6e-6` and `r2_slack = 3.04e-4`. The
residual subsidy at termination is LP tolerance, not early stopping. Reported as
`truncation_bound` and `r2_slack` rather than left implicit.

---

## 2. Phase 1 — column generation, at 15 prosumers

### 2.1 Symptom

`run_maximin.py --sizes 15` never reached Phase 2. Column generation ran **6474 pricing
rounds**, added roughly 78 000 columns, hit the master's hard-coded 1800 s limit, and
returned `timelimit`. From about iteration 500 onward the log showed `LP Obj` and `L_bar`
equal to displayed precision while 12–13 columns were added every single round.

### 2.2 Measurement

The existing log line could not distinguish "still working" from "stalled", so two fields
were added to it (logging only, no numerical effect): the worst reduced cost of the round
and the bound gap. Re-run:

```
Iter    1 | LP Obj: -6490.37 | L_bar: -17460.61 | Cols: 13 | minRC: -3.63e+03 | LB-Z: -1.10e+04
Iter  600 | LP Obj: -6499.01 | L_bar:  -6499.01 | Cols: 12 | minRC: -9.72e-08 | LB-Z: -6.58e-07
Iter 1200 | LP Obj: -6499.01 | L_bar:  -6499.01 | Cols: 12 | minRC: -9.72e-08 | LB-Z: -6.58e-07
Iter 1500 | LP Obj: -6499.01 | L_bar:  -6499.01 | Cols: 12 | minRC: -9.72e-08 | LB-Z: -6.58e-07
   ... unchanged to the last digit through iteration 6474, then the time limit
```

**`minRC` and `LB-Z` are frozen — bit-identical for six thousand iterations.** The loop is
not converging slowly. It is not converging at all, and it is not learning anything.
`z*` was pinned by iteration ~500: the Lagrangian lower bound met the master's upper
bound to `6.6e-7` on a value of `6499`, i.e. **`1e-10` relative.** By any relative
criterion the algorithm had already finished. Everything after that was waste.

Hashing the pool afterwards (round to `1e-9`, one bucket per distinct extreme point)
says how much waste:

| instance | columns | distinct | duplicates | worst single column |
|---|---|---|---|---|
| 6p, 4-hour | 781 | 781 | **0** | 1 copy |
| 15p, 24-hour | 1527 | 1526 | 1 | 2 copies |
| 15p, 4-hour | **71 494** | 3 563 | **67 931 (95.0%)** | **6 148 copies** |

It was re-adding one column six thousand times. A column already in the pool has
reduced cost `>= 0` as the master's LP sees it — otherwise the LP was not optimal — so
generating one at all is proof that the threshold is below the noise floor.

### 2.3 Cause — three thresholds that do not agree

`pricer.py::_price_smoothed` has three absolute tolerances:

| line | test | meaning | at the measured values |
|---|---|---|---|
| early termination | `lb > Z_RM - 1e-7` | stop, bound has met the LP | `LB-Z = -6.58e-07` → **fails**, by 6.6x |
| main branch | add column if `rc_rm < -1e-8` | | `-9.72e-08` → **adds 12 columns** |
| misprice fallback | add column if `rc_rm < -1e-7` | | `-9.72e-08` → would add **nothing** |

The fallback's threshold is the one that would declare convergence — at `rc = -9.72e-08`
it adds nothing, `columns_added` stays 0, and the loop stops. But **the fallback only runs
`if columns_added == 0` after the main branch**, and the main branch, using a threshold
ten times tighter, always adds. So a reduced cost anywhere in the band

```
    -1e-7  <  rc  <  -1e-8
```

is a trap: the branch that adds columns is permanently active and the branch that can stop
is permanently unreachable. `-9.72e-08` sits almost exactly in the middle of it.

### 2.4 What triggers it — measured, not guessed

Same instance, same code, one parameter changed:

| `reserve_block_hours` | status | iterations | time | final `minRC` |
|---|---|---|---|---|
| 24 | `optimal` | 142 | 58 s | `-4.55e-13` |
| 4 | `timelimit` | 6474 | 1800 s | `-9.72e-08` (frozen) |

So it is the **4-hour reserve product**, not the community size. 15p on the 24-hour
product converges in under a minute and lands at machine precision.

The block change does **not** add linking rows — the count is `2|T|` either way, and the
master has 159 rows in both cases. What it changes is the number of shared variables,
`dim(x_0)` from 1 to 6, and through that the *density of the dual solution*: under one
24-hour product a single equation `sum_{t} (pi_up + pi_dn) = -1344` lets almost every
hourly dual sit at zero, whereas six blocks impose `sum_{t in blk} = -224` **per block**
and force a nonzero dual in each. The reduced cost is then a difference of many more
terms of size `1e+2`, and its noise floor rises accordingly — at 6p it is `1e-13`
(`sum |pi a| ~ 2e+3`, i.e. `1e-16` relative, plain double precision), at 15p on 4-hour
blocks it is `1e-7`.

**The model itself is not at fault, and this was checked rather than assumed.** At 6p on
the 4-hour product every block's shared-variable dual feasibility holds with equality to
`1e-14`:

```
block 0 hours 0..3   sum(pi_up+pi_dn) = -224.00000000 <= -224.0   slack 2.8e-14   r_sym = 0.748
block 1 hours 4..7   ...                                          slack 0.0e+00   r_sym = 0.143
... all six blocks, and peak: sum(pi_peak) = -150.00000000 >= -150.0
```

and the pricer's reduced-cost recomputation agrees with SCIP's own to `1e-13`. Block
assignment, dual extraction and the reduced-cost formula are all correct. The 4-hour
product is the trigger; the absolute thresholds are the defect.

**This is not the same phenomenon as `validation_plan.md`'s "row generation fails to
converge at n = 30".** That is a different algorithm and a genuine scaling statement about
coalition separation. This one is arithmetic.

### 2.5 The fix — one relative tolerance

`LEMPricer._pricing_tol(scale) = 1e-9 * (1 + |scale|)`, used by **every** add rule and by
the early-termination test:

- `price()` — `reduced_cost < -tol` (Farkas pricing keeps the absolute threshold; it
  measures infeasibility, for which the LP objective is not a scale)
- `_price_smoothed()` — `lb > Z_RM - tol` for early termination, and `rc_rm < -tol` in
  **both** the main branch and the misprice fallback

That the two add rules are now the same number matters more than what the number is.
Convergence is declared when nothing is added, so the add rule *is* the stop rule; two
different add rules made stopping unreachable.

Accuracy given up: termination allows `|rc| <= tol` per prosumer, so the bound is within
`n * tol` of `z*` — about `1e-4` EUR at 15p, `1e-8` relative. Below every reported figure.

**A duplicate-column guard was considered and rejected.** With the pricing step correct a
duplicate cannot arise: a pooled column has reduced cost `>= 0` by LP optimality, so it
is never a strict improver. Guarding would treat the symptom and, worse, would silently
swallow the evidence if the duals were ever genuinely wrong. The measured duplicate rate
is used as a *test* of the fix, not as a mechanism in it.

### 2.6 Measured after the fix

| instance | before | after |
|---|---|---|
| 15p, 4-hour | `timelimit`, 6474 iters, 1800 s, 71 494 cols, 95.0% dup | **`optimal`, 394 iters, 98 s, 2807 cols, 0.0% dup** |
| 6p, 4-hour (regression) | `optimal`, 286 iters, 781 cols | **identical** — `v_chp = -3039.9442970417203` to the last digit |

`v_chp = -6499.008394` at 15p, the value the stalled run had already been sitting on. The
answer was right the whole time; the loop could not stop.

Not touched, deliberately: `_compute_alpha`'s `gap < 1e-2`, which is also absolute and is
what switched smoothing off at 15p. It is a heuristic knob rather than a correctness
threshold, and one change at a time is worth more than two.

### 2.7 Also added — logging

- `pricer.py::_price_smoothed` — `minRC` and `LB-Z` on the iteration line. Without them a
  stalled run is indistinguishable from a working one, which is why 1800-second no-ops
  went unnoticed.
- `chp.py` — the pricer is kept on the solver (`self.pricer`), so a caller can reach the
  Lagrangian bound and the iteration count. Table II of the validation plan wants exactly
  these as measured oracle counts.

### 2.8 Consequence for sec.5 of the validation plan

`run_maximin.py` refuses to run (MM) when Phase 1 did not converge, and records the fact
instead of crashing. The refusal is not caution, it is required: (R2) is written against
`z*`, and a truncated Phase 1 gives `z_hat >= z*` in the cost convention, making (R2)
**tighter** rather than looser. The inclusion `hat Theta* superset Theta*` — the one thing
that makes the whole restricted-face argument valid — then fails in the wrong direction,
possibly to the point of an empty intersection.

With §2.5 applied, 15p now converges and the gate can be run there. That is the next
measurement, not a blocker.

---

## 3. Separately fixed along the way

**The runners could not start at all with reserve enabled.** `reserve_block_hours` turned
`r_sym` from a bare variable into a dict keyed by block index. The `isinstance(v, dict)`
filter that guarded the column-generation warm start was written to drop the shared
community variables, and it stopped catching them the moment they became dicts — so a
block-indexed community variable was handed to `_add_initial_columns` as if it were a
private `(u,t)` dispatch: `TypeError: 'int' object is not subscriptable`, in
`run_experiment.py` and `run_multiday.py` alike.

The key-shape test is what the contract always needed. It now lives in
`solver.private_dispatch()` and is used by `_add_initial_columns` and by the smoothing
incumbent in `chp.py`. Nothing had been run since the block change, which is why a total
blocker surfaced only now.

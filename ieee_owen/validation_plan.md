# What is still missing from the numerical study

Everything the experiments can already support has been written into `ieee_draft.txt`
Sec. `sec:num` and removed from here. This file is only the remainder: what is not yet
measured, what is blocked, and what was tried and abandoned.

Baseline for everything below: `|T| = 24`, reserve 56 EUR/MW.h on **4-hour symmetric**
blocks, peak 150 EUR/MW, `m = 144`. Entry points are
`weak_eps_experiment/run_experiment.py` (one day per size) and `run_multiday.py` (day
sweeps, two phases). Anything whose **time** is reported must be run one solve at a time —
three concurrent jobs inflated the grand-coalition MILP by 1.8x and the maximin loop by
1.6x on this machine (24 cores, 7 GB), and Sec. `sec:complexity` rests on those timings.

---

## 0. Row generation cycled on an absolute tolerance -- FIXED, sweeps redone

Every instance in the sweeps below that failed to certify failed the same way, and it was
not coalition separation being hard. It was re-adding a coalition already in the master,
forever.

Counted off the run logs (`Adding violated coalition` lines, total vs distinct), the split
is clean rather than graded:

| | days | duplicate cut rate |
|---|---|---|
| certified | 17/31 at 15p, 30/31 at 6p | **0.0%**, every one |
| cut off at the budget | 14/31 at 15p, 1/31 at 6p | **39-82%**, every one |

15p day 28: 2197 cuts added, 436 distinct, one coalition added 1735 times. 6p day 24:
14 467 iterations, 17 distinct coalitions, one coalition added 14 458 times in 3600 s.

**Mechanism.** Printed at full precision, 6p day 24 converges at iteration 11 and then
freezes: the violation is `5.958251982463025e-06`, **bit-identical for every one of the
next 200+ iterations**. That is `6.8e-9` relative to `|c(S)| = 870` -- LP noise. But
`compute_core` tested `violation <= tolerance` with `tolerance = 1e-6` **absolute**, so the
noise read as a live violation; the coalition was re-added although already present, the LP
did not move, and the same coalition came back. The add rule is the negation of the stop
rule, so an add that changes nothing makes stopping unreachable. Same shape as
`convergence.md` sec.2, where the pricer's absolute thresholds froze `minRC` for six
thousand iterations; row generation was never audited the same way because the 24-hour
product happened not to reach the floor.

**The answer never depended on the tolerance.** Swept over five decades, at 6p (day 24, the
cycling one, and day 1, a clean one) and at 15p (day 8, 82.5% duplicates):

| | 6p day 24 | 15p day 8 |
|---|---|---|
| certifies from | `1e-5` up | `1e-3` up |
| cuts, every tolerance | 17 | 137 |
| `v*`, every tolerance | `-1.634e-13` | `0.0` |
| allocation | bit-identical across all six | bit-identical across all six |
| time, once it can stop | 3600 s -> **13.4 s** | 3600 s -> **259 s** |

The runs that failed to certify were holding the same answer as the ones that did. The
tolerance decides only whether the loop can stop.

**Fix.** `tolerance` is now relative: `tol_eff = tolerance * (1 + |c(N)|)`, one number used
by the only test that consumes it, printed at the top of every run. At `tolerance = 1e-6`
that is `2.2e-3` at 6p and `6.3e-3` at 15p. The measured room to choose in -- above the
residual that must be ignored, below the smallest genuine violation that must not be -- is
`[1e-5, 1.7]` at 6p and `[1e-3, 0.022]` at 15p, and the effective threshold sits inside
both. No duplicate-cut guard: `convergence.md` sec.2.5 rejected the duplicate-column
equivalent for the reason that applies here too, that it hides the evidence.

Regression, 6p: day 1 reproduces `v* = -4.973799150320701e-14` and its 14 cuts exactly;
day 24 goes from 3600 s uncertified to **14.1 s certified**, 17 cuts, at the `v*` the sweep
predicted.

**The residual does not keep climbing with `n`.** It jumps a hundredfold from 6p to 15p
while `|c(N)|` only doubles, which was reason to doubt `|c(N)|` as the scale; the 30p probe
settles it. At 30p the loop behaves identically at `1e-3`, `1e-2` and `1e-1` -- same 135
cuts, same allocation -- and differs only at `1e-4`, so the floor is in `(1e-4, 1e-3)`,
level with 15p rather than another decade up. `1e-6 * (1 + 8658) = 8.7e-3` clears it ninefold
(15p sixfold, 6p five-hundredfold). The rule stands at all three sizes.

**Which days were re-run, and why not all of them.** Only the days that cycled: 14 at 15p,
1 at 6p. The certified days were left alone because the smallest violation any of them
carried into a cut is `0.022` at 15p and `1.7` at 6p, both **above** the new threshold
(`6.3e-3`, `2.2e-3`) -- so no cut they added is dropped and no trajectory changes. The
margin at 15p is a factor of `3.5`, not the "three decades" an earlier draft of this note
claimed; the conclusion is exact rather than comfortable, since a cut is added iff its
violation exceeds the threshold and every one of theirs does.

**Result of the redo.**

| | before | after |
|---|---|---|
| 6p | 30/31 certified, gmean 6.3 s | **31/31**, gmean 6.4 s, 15 cuts |
| 15p | 17/31 certified, gmean 1042 s | **31/31**, gmean 982 s, 234 cuts |
| 30p | 0/1, 3955 s, 160 cuts (old code) | **0/1**, 3604 s, 146 cuts |

Every day that had been cut off certified, none took more than 1735 s, and 15p day 28 went
from 3600 s uncertified to 312 s certified. `sec:num` and `tab:runtime` are rewritten
against these: row generation now certifies everything at 6p and 15p, costs 150x more at 15p
than at 6p against column generation's 2.7x, and fails only at 30p -- where the tolerance
probe shows the failure is the method, not the stopping test.

Left as a known seam: the 17 days at 15p that certified first time were measured under the
pre-fix code. Their numbers cannot differ, by the argument above, but they are a different
vintage; re-running them for a single-vintage dataset is about 5 h and has not been done.

## 1. The multi-day spread — DONE for the core results

`sec:num` now reports geometric means over **30 daily instances at every size**, and every
figure in its two tables is regenerated from disk by `paper_tables.py`. Sources:

| file | produced by | contents |
|---|---|---|
| `eps_30day.json` | extracted from `maximin_slack1e-7/` | `v^MIP`, `v^CHP`, `omega`, `eps^LR`, MIP/CG times, 30 days x 3 sizes |
| `excess_{6,15,30}p.json` | `measure_excess.py` | per-capita excess of the executed allocation, per day |
| `cg_*.json`, `rowgen_*.json` | `run_experiment.py` | the single-instance row-generation column |

What this does **not** cover, and what `run_multiday.py` is still for: the parameter
variations (`param6p`) and the reserve/peak metrics. The `--groups core,param6p` sweep
(248 rows) has not been run on the 4-hour product.

**Row generation: the `n = 15` sweep is DONE, `n = 6` and `n = 30` are still one instance.**
31 days at `n = 15` under a 3600 s per-day budget (19.2 h of compute), in
`baseline_15p/rowgen.csv`:

| | certified | time (gmean, certified only) | cuts (gmean) |
|---|---|---|---|
| certified | 17/31 | 1042 s (410 / 1061 / 1735) | 242 |
| stopped at the budget | 14/31 | --- | 227 |

The expectation going in was that most days would not converge. **They do — 55% of them.**
The claim the draft can carry is therefore not "row generation does not terminate at 15
prosumers" but the weaker and more interesting one now in `sec:num`: it terminates on
barely half the instances, takes 1042 s when it does, and gives no way to tell in advance
which half. Cut counts are nearly identical on both sides (242 vs 227), so it is not that
the stopped instances were doing more work.

On the 17 certified days `v* = 0` to `1.5e-12`, so the core of `Gamma^MIP` is nonempty at
`n = 15` as well — a separate fact from `tab:results`' "3/31 days in the exact core", which
is about where the *proposed allocation* lands, not about whether the core is inhabited.

Averaging censored runs in with the rest would report the budget rather than the algorithm;
`paper_tables.py` therefore takes the row-generation geometric means over certified
instances only and prints the certified count next to them.

`n = 6` (about 10 minutes for 31 days) and `n = 30` (31 h, and every day expected to be
censored) have not been swept. The 6p one is cheap enough to be worth it for symmetry.

**Stale row-generation CSVs are parked, not deleted.** `baseline_{6,15,30}p/rowgen.csv` were
all pre-4-hour-product. They now sit next to their folders as
`rowgen_stale_24h_product.csv`, and `paper_tables.load_rowgen` treats the presence of a
plain `rowgen.csv` as meaning "current sweep". The five `*_6p` parameter-variation folders
still hold July `rowgen.csv` files; nothing reads them yet, but the same rule will pick them
up when something does.

**Stale data on disk.** Every per-run folder under `weak_eps_experiment/` predates the
4-hour product and must not be mixed with current results; the seven `scenario` folders in
particular will not be refreshed by the default sweep. Delete them or re-run with
`--force`, and do not let `summarize.py` build a table across both vintages.

## 2. Channel decomposition and incidence

`reserve.txt` Sec. 3.1/3.3 wants `v(N)` decomposed over the four channel corners (balance
only, `+reserve`, `+peak`, both) and the incidence question — who captures the reserve
value, who bears the peak charge — as a counterfactual `Delta chi_j` with the channel on
and off. None of it exists on the 4-hour product. The runs are defined and switched off:

```
run_multiday.py --phase owen --groups all             # adds 7 scenario runs, 217 rows
```

Also unmeasured on the 4-hour product: the reserve pooling gain and its time/direction
split. The §5b claim that the 24-hour product suppressed pooling entirely (gain exactly
zero on all 372 days measured) is the reason the product was changed, and it has not been
re-measured under the change. The four-way product table in that section is still one day,
one 6-prosumer instance.

## 3. Admissibility boundary (`as:pool`)

Reserve and peak are homogeneous linking rows and satisfy `as:pool`; a **shared** export
cap does not (`rmk:nonadditive`). The intended demonstration is that `v*` and `eps^LR` are
undisturbed as the reserve and peak channels are switched on, and that both break once a
shared cap is imposed. The channel half is a `--groups all` sweep; **the shared-cap
counterexample does not exist and has to be built** — note that `export_cap_020_6p` is a
per-player cap, which is additive and therefore not the counterexample.

## 4. Non-convex fraction at fixed `n`

The configurations dilute the integer players as `n` grows (1 of 6, 3 of 15, 6 of 30), so
a decay measured along `n` cannot be separated from consumer dilution — the first question
a referee will ask. The sweep exists (`run_maximin.py --electrolyzers 2,3,5,7`, which
rebuilds the instance with a given number of electrolyzer owners) but has only been run
inside the retracted maximin work; it needs to be run for `eps^LR`. Nothing blocks it.

## 5. The excess measurement clamps at zero — do not reuse the harness value

`stability_check.check_allocations` reports `excess = 0, worst S = (none)` whenever an
allocation is in the core, and that zero is a **floor, not a margin**:
`_measure_weak_eps_separation` short-circuits when the raw excess is `<= 1e-6` or the
returned coalition is empty, handing back the raw value with the empty set attaining it.
It also switches units — the short-circuit returns the **raw** excess, the Dinkelbach loop
below it returns the **per-capita** one — so the two cases are not comparable.

The draft therefore uses enumeration at `n = 6` (all `2^n - 1 = 63` coalitions, about a
minute) and the identity that adding a constant `c` to every member raises the per-capita
excess by exactly `c`, which recovers the margin of `chi^LR` from the corrected point's
measurement at 15p and 30p. That gave the margins `-1.414 / -0.546 / -0.121` now in
Table `tab:alloc`, and the finding that the **corrected** allocation is still in the exact
core at `n = 6`.

Outstanding: nothing for the draft, but the harness should be fixed so this does not bite
again — report `-inf`/`n.a.` rather than `0` when no coalition attains, and return the same
unit on both paths. `_measure_violation_brute_force` has the same clamp (it solves
`min v s.t. ..., v >= 0`).

---

## Closed, with the reasons

**Fairness selection over `Theta*` (the maximin criterion).** Built
(`maximin.py`, `run_maximin.py`), run to convergence at all three sizes, over 30 days each,
across a 2-to-7 sweep of the non-convex share, with `P` restricted to asset owners, and
with the peak channel on and off. **Result: `Theta*` is a point in the coordinate the
criterion optimises** — with (R2) imposed as a hard equality the improvement is numerically
zero at 6p (one iteration; no cutting-plane work exists) and `1e-6`–`1e-5` at 15p/30p,
which on payoffs of order 40 EUR is inside the LP's own accuracy. Sec. 5 of
`maximin criterion.md` should be cut.

The cautionary part is worth keeping. For most of a day this measured a rich phenomenology
— a spike worth 69% of `eps^LR` on one day, an apparent dependence on the peak channel, a
size trend — all of which was the `1e-7 |z*|` tolerance on (R2). That inequality lets the
LP lower the worst-off share by **under-collecting**, i.e. by buying it with subsidy rather
than redistributing along `Theta*`. Two lessons:

- `truncation_cost` equalled `r2_slack` in every run and was printed throughout. (P4) says
  the residual subsidy is *unchanged* on `Theta*`, so a run where it moves by the tolerance
  is not on `Theta*`. The diagnostic was there and was read as noise.
- **"the effect is 1649x the tolerance, so it is not the tolerance" is not a valid check.**
  Day 4 was exactly that and was entirely artefact. Vary the tolerance and confirm the
  answer does not move with it.

`maximin.py` itself is sound — pool extraction agrees with the RMP duals to `1e-12` — and
is reusable for any other selection rule over `Theta*`. The 90 day-JSONs produced under the
loose tolerance are kept at `weak_eps_experiment/maximin_slack1e-7/` as documentation, not
as results.

**The `2n`-LP singleton screen.** Removed from the default path (`--screen` keeps it as a
diagnostic). Its widths are computed over `hat Theta* ⊇ Theta*`, so only a width of exactly
zero proves anything, and it never returns zero because it looks in precisely the direction
the column pool was never built to constrain — it overstated the freedom 150x at 6p and 58x
at 15p and skipped no work at either size. The loop's own bracket is rigorous in the
direction that matters and is available from iteration one.

**Measuring `rho_bar`, the numerator of `prop:eps`.** Dropped, and `rho_bound.py` with its
`rho_bar.json` / `rho_gap.json` outputs deleted. Two estimators were built and they did not
agree: the pricing-LP gap `sigma_j^LP(theta*) - sigma_j^MIP(theta*)` gave `1.42 / 0 / 0` at
`n = 6/15/30` — under `(m+1) rho_bar / n` that is a bound *below* the measured `eps^LR` at
the two larger sizes, so it is not a bound at all — while the master-side fractional-column
gap gave `57 / 183 / 526`, growing with `n` and so contradicting the flatness the scaling
argument rests on. At least one of the two is measuring the wrong object, and resolving
which is a modelling question, not a run.

Nothing in the draft depends on it. `sec:num` uses `prop:eps` for its *form* only — neither
`m = 144` nor `rho_bar` involves `n` — and reports `omega^LR` itself, measured as the
integrality gap of a single grand-coalition solve, which is flat across the fivefold
increase in `n` without any appeal to `rho_bar`. There is no Figure 1 and no measured
numerator anywhere in the manuscript. Do not rebuild the module without first settling
which gap `rho_j` actually is.

**Shrinking `|T|` to enter the `1/n` decay regime.** Attempted and abandoned. Block-averaged
profiles at `|T|` in {24, 8, 4, 2} (machinery in `horizon.py`; `|T| = 24` verified
bit-identical) dissolve the commitment the experiment was meant to measure:
`min_down_time_G = 2` is two *hours*, so below `|T| = 8` it stops binding, and the
non-convexity disappears along with `m`. A redesign at `|T| = 6` with reserve and peak off
(`m = 18`, boundary at `n > 19`) and recalibrated commitment timing would be needed. Low
priority — `sec:num` reports the decay of `eps^LR` directly and does not need the regime.

---

## Fixed along the way (both were blocking)

**The runners could not start with reserve enabled.** `reserve_block_hours` turned `r_sym`
from a bare variable into a dict keyed by block index, which the `isinstance(v, dict)`
filter guarding the CG warm start no longer caught, so a shared community variable reached
`_add_initial_columns` as if it were a private `(u,t)` dispatch. The key-shape test now
lives in `solver.private_dispatch`.

**Column generation did not converge at 15p on the 4-hour product**, burning the full
1800~s master limit. Every threshold in `pricer.py` was absolute — `-1e-8` is `1e-12`
relative on a master worth 6499 — and the two column-adding branches used *different* ones
(`-1e-8` and `-1e-7`), so a reduced cost landing between them kept the adding branch active
and the stopping branch (convergence is declared when nothing is added) permanently
unreachable: 6474 rounds, 71 494 columns of which **95% duplicates**, one column re-added
6148 times, with the Lagrangian bound already equal to the LP objective to `1e-10` relative
since round 500. Now one relative tolerance, `1e-9 (1 + |Z_RM|)`, shared by every add rule
and the stop rule. 15p/4-hour: `timelimit`/6474 iters/1800~s → **`optimal`/394 iters/98~s,
0% duplicates**; 6p reproduces its old `v_chp` to the last digit. Full account in
`convergence.md`.

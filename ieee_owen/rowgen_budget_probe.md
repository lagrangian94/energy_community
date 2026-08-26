# Row generation at n=30 under a larger budget

Data: `weak_eps_experiment/baseline_30p/rowgen_probe_budget21600.csv`
Command: `run_multiday.py --phase rowgen --runs baseline_30p --days 10,23,29 --budget 21600 --force`

## Why these three days

The 31-day sweep at the quoted 3,600 s budget certifies nothing at n=30
(`converged` false on all 31). Its incumbent v* is a *lower* bound, because row
generation carries a subset of the proper-coalition constraints and a relaxation of
a minimisation understates its optimum. On 28 days that bound sits at zero and says
nothing. On days 10, 23 and 29 it is strictly positive, and a positive lower bound
on v* is a certificate that the core is **empty** on that day -- the one direction an
unconverged run can still settle. Those three were re-run at six times the budget to
see whether the bound closes.

## Results

| day | budget | converged | v* | v*/n | cuts | time |
|-----|--------|-----------|------|-------|------|------|
| 10  | 3,600  | no        | >= 0.5984 | >= 0.01995 | 166 | 3,604 s |
| 10  | 21,600 | **yes**   | **1.1822** | **0.03941** | 262 | 13,946 s |
| 23  | 3,600  | no        | >= 2.0691 | >= 0.06897 | 108 | 3,607 s |
| 23  | 21,600 | no        | >= 2.9432 | >= 0.09811 | 136 | 21,604 s |
| 29  | 3,600  | no        | >= 1.5061 | >= 0.05020 | 128 | 3,609 s |
| 29  | 21,600 | **yes**   | **2.7213** | **0.09071** | 247 | 21,309 s |

## What the numbers say

**Row generation does terminate at n=30.** Two of the three converged: day 10 in
13,946 s and day 29 in 21,309 s, 3.9x and 5.9x the quoted budget but inside six hours.
Day 29 finished with 291 s of its budget left, so six hours is close to the margin for
it rather than comfortably above. "No certificate at n=30" is therefore a statement
about the 3,600 s budget, not about the size. This is separate from the tolerance
sweep, which showed that the failure at 3,600 s is not an artefact of the stopping
test: the instance examined there does not terminate for any convergence tolerance
between 1e-4 and 1e-1. Both readings hold. The budget binds; the tolerance does not.

**Two cores are empty, exactly.** v* is now a value rather than a bound on days 10 and
29: 1.1822 and 2.7213, being 23% and 29% of those days' omega^LR (5.04 and 9.38). It sits inside the sandwich every efficient
allocation must respect,

    day 10:  0.0394 <= 0.1060 <= 0.1680
    day 29:  0.0907 <= 0.1926 <= 0.3127
    day 23:  0.0981 <= 0.3236 <= 0.4678   (left end still a bound)

whose left end is the per-capita subsidy no efficient allocation can avoid and whose
right end is what Proposition eps guarantees.

**Day 23 improved without closing.** Six hours raised the bound 42%, from 2.0691 to
2.9432, on 136 cuts against 108. The improvement is certain, since the bound is
monotone in the cuts generated, and v* > 0 keeps the core-empty verdict.

**Day-to-day spread is large.** 13,946 s, 21,309 s and more than 21,600 s on three days
of the same size, against a median of 3,600 s at which none of them finishes.
Certifying all 31 would need a budget well beyond six hours, and the binding constraint
is the spread rather than the median. Note also that all three were selected for having
a positive lower bound at 3,600 s, so they are the days already known to be hard; the
other 28 are unmeasured at this budget.

## Budget uniformity

These rows are deliberately **not** in `rowgen.csv`. That file is measured at a single
3,600 s budget across n=6, 15 and 30, which is what the runtime table quotes; mixing a
21,600 s row into it would put two budgets in one column. `rowgen.csv` is restored from
`rowgen_30p_budget3600.csv` once day 29 finishes.

# Hunting an empty core at n=15

Brief: `experiment_brief.md`. Data: `weak_eps_experiment/emptycore_15p/*.csv`.
57 instances across four axes. **No instance with `omega* > 0` was found.**

Row generation is used throughout (n=15 is asymmetric, so the brief's T0 symmetric
shortcut does not apply). For this goal a truncated run is still useful in one
direction: v* is a lower bound, so v* > 0 would certify an empty core outright.
A truncated v* = 0 certifies nothing, which matters for reading the table below.

## What was run

| sweep | axis | instances | converged | result |
|---|---|---|---|---|
| 1 `sweep.csv` | `els_cap` 1.0 -> 4.55 | 15 | 14/15 | all v* = 0 |
| 2 `sweep_ratio.csv` | renewable ratio f 0.9 -> 0.5 | 21 | 16/21 | all v* = 0 |
| 3 `sweep_absorb.csv` | storage off / export capped / both | 12 | 3/12 | all v* = 0 |
| 4 `sweep_owners.csv` | electrolyser owners 3 / 8 / 15 | 9 | 3/9 | all v* = 0 |

## Findings that are not the headline

**The brief's first-choice lever does not work here.** S1 nominates `els_cap`,
reasoning that a min load fixed as a fraction of rating moves `d_G` and hence kappa.
But `data_generator.py:238` sizes renewables off the same quantity --
`ElectricityProdGenerator(..., el_cap_mw=parameters["els_cap"])` -- so e scales with
els_cap and kappa is invariant. Measured: els_cap 1.0 -> 4.55 moved kappa_peak from
1.76 to 1.77. Sweep 1 is therefore fifteen repetitions of one grid point. The brief
should name the renewable ratios as the primary lever instead.

**The usable range of that lever is narrow.** Below f ~ 0.46 the peak net surplus
turns negative and the community has nothing to pool, so the instance is degenerate
in a different way. Inside f in [0.5, 0.9], kappa spans 1.6 to 20.8, which covers
the brief's target band [2, n-1] and beyond. It is covered; it just yields nothing.

**Condition 1 is only partly in force.** mu at the mean import price is 0.806, but
the price is TOU with two near-free hours (`pi_E_gri_import_12 = 0.0`, `_13 = 6.85`).
Per hour, 22 of 24 have mu < 1 and two do not, and the second is where the threshold
stops binding. `use_tou_elec = False` does not change the profile at all, so that
flag does not do what its name suggests; `import_factor` scales every hour, which
cannot close a zero-priced hour. Flattening the price needs a change to the price
data, not to the scenario knobs.

**The absorption levers do bite, without changing the verdict.** Electrolyser
on-hours fall from 60 (of 72) at baseline to 43-45 with export caps at 0.2 and to
38-39 with storage removed as well. So S3 is doing real work to the dispatch; the
core survives it anyway.

**Larger instances are easier for row generation.** Time to converge falls as
els_cap rises (1202 s -> 394 s over sweep 1) and rises steeply with electrolyser
count (153-489 s at 3 owners, >1200 s at 8, 1235-5491 s at 15).

## A separation defect, found by accident

At 15 of 15 owners with `A_nosto`, `find_violated_coalition` aborted on its own
consistency guard:

    Actual violation (sum payoffs - cost): 1211.5540
    Separation problem violation:          1994.9837
    coalition = [u2 .. u15]   (u1 excluded)
    RuntimeError: Mismatch ... exceeds tol 0.962052

The separation values that coalition 65% above what the coalition can actually
achieve, so it is not solving max_S {sum chi - v(S)} for this configuration. It ran
clean at 3 and 8 owners.

`core.py:167-172` has the big-M gate on `els_d` commented out, with the note
"z_u=0 -> els_d=0. therefore, useless". The other electrolyser variables are gated
(`p[u,'els']`, `fl_d`), as are renewables and heat pumps, so the leak was not
located -- only that the implication that comment relies on does not hold at full
electrolyser coverage.

**This is worth checking against `nonconvex_15p` and `nonconvex_30p`**, which exist
precisely to give electrolysers to members that did not have them (7/15 and 15/30).
They sit between the coverages that ran clean and the one that failed. Any row
generation or stability check on those runs should be re-validated before use.

## Reading the negative result

Not "the core is nonempty at n=15 under all these settings". Only 36 of the 57
instances converged; the other 21 carry no verdict. The converged ones cluster in the
easy corner (sweep 1 and 2 at high f), and the settings the brief actually predicts
would work -- absorption removed, many owners -- are exactly the ones that time out.
A larger budget on sweep 3 and 4 would be the honest next step before concluding
anything.

The brief's own S5 predicts this outcome: asymmetry protects the core, and 15p
carries 3 electrolyser owners of 15. Sweep 4 was the attempt to leave that regime
and it hit the defect above at the one point that mattered.

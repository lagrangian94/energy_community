# What the duality allocation gives up in fairness

Status: **6 and 15 prosumers, 31 daily instances each, complete.** 30 prosumers is out of
reach, since row generation returns no certificate there at any tolerance and every number
below needs one.

**The headline is the size effect.** At 6 prosumers the duality allocation is measurably
less even than the fairest stable one; by 15 it is essentially the same point. Whatever
fairness it gives up is a small-community phenomenon.

The Owen point `chi^LR` is read off a dual. Nothing in its construction asks for it to be
even-handed, only stable. This measures what that costs, against the core element a
fairness criterion would pick instead.

Everything is in the repo's **cost convention**: a share is what the member pays, so
negative means the member profits. Two invariances make the comparison safe -- dispersion
is unchanged by the flip to profits (deviations flip with the mean), and unchanged by
adding a constant to everyone, so `chi^LR` and the executed `chi^LR - eps^LR*1` have the
same spread and the choice between them does not arise.

Produced by `weak_eps_experiment/run_fairness.py`; per-day files are
`baseline_6p/fair_{mode}_day{d}.json`, tables `fairness_{mode}.csv`.

---

## 1. The benchmark, and why it is the Variance Core

Two ways to ask for "least spread over the core":

| | criterion | problem | minimiser |
|---|---|---|---|
| `range` | `min (max_i p_i - min_i p_i)` | LP | Drechsel & Kimms (2010) MP_I |
| `variance` | `min sum_i (p_i - c(N)/n)^2` | QP | Fioriti et al. (2025) eq.(27), Variance Core |

Efficiency pins the mean at `c(N)/n`, so the second objective **is** `n` times the variance
of the shares. The first is an LP and fixes the objective value but not the argument; the
second is strictly convex and fixes both. That difference is not academic here:

**The MP_I optimum is a face, measured exactly.** At 6 prosumers all `2^6 - 1` coalitions
can be enumerated, so the core is known exactly rather than through generated rows. Solving
MP_I twice -- once over the 14 rows row generation produced, once over all 62 proper
coalitions -- gives

```
objective        1436.0916   both, to the last digit
allocation       u2: -495.72 vs -388.06,  u5: -47.08 vs -154.74
max difference   107.66  (7.258% of the largest share)
face width       219.6   (hold the range at r*, swing each share: exact, not a bound)
```

Both points are in the exact core; neither is wrong. The criterion simply does not name one.
Fioriti et al. report the same thing across procedures and make it their reason for
requiring strict concavity -- *"different procedures point to allocations that are far from
each other, although having comparable surplus"*. Repeating the test on the Variance Core:

| | max difference between the two procedures |
|---|---|
| MP_I (`range`) | **107.66**, i.e. 7.258% |
| Variance Core (`variance`) | **0.0000**, i.e. 0.000% |

**Adopting the L2 measure costs nothing.** Over 31 days the Variance Core attains the
minimum range as well -- `range(VC) - r*` never exceeds `4.5e-13` -- while cutting the
variance by a median 13.2% and up to 41.5% relative to MP_I. On 2 of 31 days MP_I was
already a point and the two agree. So the L2 criterion is not a trade against the L∞ one; it
selects within its optimal face. Everything below therefore uses the Variance Core.

---

## 2. Stability is not the question here

Whether `chi^LR - eps^LR*1` lies in the exact core is settled in `sec:num` and measured by
`measure_excess.py`: **31/31 days at 6p, 3/31 at 15p, 1/31 at 30p**. Leaving the core by at
most `eps^LR` is what the construction trades for budget balance, so the counts below 6p are
the design working, not a defect.

`run_fairness.py` also reports an `owen_in_core` flag, but it tests only the coalitions that
run generated -- a subset, so it is optimistic and disagrees with the counts above at 15p.
Use `excess_{n}p.json`, not the flag.

## 3. What it gives up, and how that changes with `n`

`chi^LR` against the Variance Core, geometric means over 31 daily instances:

| | 6 prosumers | 15 prosumers |
|---|---|---|
| spread `max - min` | **1.24x**  (1.14 – 1.43) | **1.02x**  (1.00 – 1.04) |
| variance `sum (p_i - mean)^2` | **1.65x**  (1.40 – 2.04) | **1.03x**  (1.00 – 1.07) |
| largest per-member gap | **258 EUR**  (156 – 480) | **21.8 EUR**  (down 12x) |
| row generation, per day | 7 s | 281 s |

At 6 prosumers the gap is real: a quarter more spread, two thirds more variance, and single
members up to 480 EUR from where a fairness criterion would put them. At 15 it is gone --
`chi^LR` is within 2-3% of the least-dispersed stable allocation on every day, and the worst
any member is displaced is 22 EUR on shares of order a thousand.

So the duality allocation needs no fairness correction at the size the paper argues for. That
is an observation over two sizes, not an explanation; what makes the core narrow enough to
squeeze every stable allocation together has not been measured here.

At 6 prosumers the variance ratio exceeds the spread ratio on every day, so the excess
dispersion there is not two outliers at the ends -- the members in between are unevenly
treated too, which a report built on the range alone would have missed.

Against MP_I instead of the Variance Core the variance ratio falls to 1.43 at 6p, and on day
28 it is **0.95** -- the MP_I point is *less even than `chi^LR` itself* by the L2 measure
while being optimal by the L∞ one. One more reason not to report MP_I's allocation.

## 4. Who pays -- a 6-prosumer phenomenon only

| days on which some member's share is positive (pays in) | 6p | 15p |
|---|---|---|
| `chi^LR` | **31 / 31**, always `u4` | 0 / 31 |
| Variance Core | 5 / 31 | 0 / 31 |

At 6 prosumers `u4` is the consumer-only member: no wind, no electrolyser, no storage, and a
stand-alone cost of `+67.85`. Day 1:

| player | `chi^LR` | Variance Core | difference |
|---|---|---|---|
| u1 | -1785.05 | -1483.17 | -301.87 |
| u2 | -229.41 | -388.06 | +158.65 |
| u3 | -505.28 | -377.26 | -128.02 |
| **u4** | **+39.14** | **-47.08** | **+86.22** |
| u5 | -103.27 | -154.74 | +51.46 |
| u6 | -133.13 | -266.69 | +133.56 |

Under `chi^LR` `u4` still pays 39.14 to belong. It is not irrational to join -- 39.14 is
below the 67.85 it would pay alone, and that inequality is exactly the individual-rationality
constraint the core enforces -- but it is the only member that does not turn a profit, and
that holds on every one of the 31 days. The Variance Core hands `u4` a profit instead, taking
it from `u1`, the member with the wind and the storage.

Under `chi^LR` `u4` pays on every one of the 31 days; the Variance Core hands it a profit
instead, taken from `u1`, the member with the wind and the storage. So at this size the
duality allocation concentrates the surplus on the asset owners systematically rather than on
unlucky days.

**It does not carry to 15 prosumers**, where no member pays under either allocation. The
effect belongs to the 6-player configuration, in which one of six members owns nothing, and
should not be reported as a property of the method.

---

## 5. What this does not claim

Strict concavity buys **uniqueness, not fairness**. The Variance Core is "the least dispersed
core element, uniquely defined", not fair in the axiomatic sense the Shapley value is. That
distinction matters in this community in particular, where stand-alone costs run from `-950`
to `0`, so "everyone's share should be equal" is a premise rather than a conclusion.

What is defensible without that premise: `chi^LR` costs one grand-coalition MILP and one
column generation, and at 15 prosumers it is within 2-3% of the most even stable allocation
that exists -- which costs 281 s of row generation per day against 79 s of column generation,
and does not exist at all at 30. A community that wants the even one can compute it by the
same row generation with a different master objective, while it still terminates.

## 6. Open

- 30 prosumers. Row generation returns no certificate there for any tolerance from `1e-4` to
  `1e-1`, and every quantity here is defined against a certified core element. The size trend
  in §3 therefore rests on two points.
- The narrowing itself is unexplained. §3 reports it; nothing here measures the core's width
  at either size.
- Variant II of Drechsel & Kimms (percentage cost savings) is not computable in this
  community: `c({u5}) = c({u6}) = 0`, and the formulation divides by it.

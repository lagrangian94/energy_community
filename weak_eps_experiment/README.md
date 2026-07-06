# Weak-ε-core experiment — Owen vs. Row generation (cost-of-stability)

Compares two core-stability allocations for the energy community across n ∈ {6, 15, 30}
players, **reserve + peak coupling ON**, single representative day, T = 24.
Deliverable: **computation time + ε** per method.

## The two methods

| method | ε reported | cost | scales? |
|--------|-----------|------|---------|
| **Owen** (1 column-generation solve) | `eps_bound = \|gap\|/N` (Shapley–Folkman upper bound) | seconds–~1 min | ✅ |
| **Row generation (CoS)** | exact minimal weak-ε `= v*/n` (v* = cost of stability) | separation MIP per iter | ❌ beyond ~small n |

Bracket: `v*/n ≤ eps_min ≤ eps_bound = |gap|/N`. When row-gen does not converge, its
`v*` is a **lower bound** (master slack is monotone ↑), so the run still brackets from below.

## How to run

```bash
source ../../.venv/bin/activate
python run_experiment.py --sizes 6                 # default
python run_experiment.py --sizes 6,15,30 --rowgen-time-limit 5400
python run_experiment.py --sizes 30 --skip-rowgen  # Owen only
python run_experiment.py --sizes 15 --force        # recompute
```

- **JSON files are the source of truth; `results.csv` is regenerated from them every run.**
  A size is skipped if its JSON exists (`--force` to recompute). Each stage is saved immediately.

## Outputs

- `results.csv` — summary: n_players, method, time_s, eps, eps_bound, v_mip, v_chp, gap, converged, n_coalitions
- `cg_<n>p.json` — Owen side. Besides eps/time, stores the **offline CHP-settlement toolkit**
  (reserve/peak CHP allocation is unresolved — this is why): Owen duals `owen_sigma_cost`,
  gap-corrected `owen_alloc_cost`, `master_coupling_duals` (convex-hull prices: balance +
  reserve_up/dn + peak), and `mip_quantities` (per-(u,t) i_E_gri, e_E_gri, r_plus, r_minus).
  **price × quantity ⇒ prototype any CHP settlement rule offline, no CG re-run.** Columns not stored.
- `rowgen_<n>p.json` — v*, weak_eps (exact or lower bound), converged, core allocation,
  and `coalition_costs` (warm-start / reconstruct partial bound).

## Findings (2026-07-06)

Separation solver matters enormously — use `--sep-solver gurobi` (default).

- **6p:** both converge. Core non-empty (row-gen v*≈0, ~2 s). Owen bound 0.0955 (loose). Owen 25 s.
- **15p:** Owen 47 s, bound 0.0308. Row-gen with **Gurobi separation CONVERGED in 359 s (6 min),
  v*≈0 → core non-empty, eps_min = 0** (130 coalitions). With SCIP it did NOT converge in 1.6 h —
  a solver-speed artifact (~40 s/iter SCIP vs ~3 s/iter Gurobi, ≈13×), not a fundamental limit.
- **30p:** Owen ~100 s, bound 0.0197. Row-gen (Gurobi) — see `results.csv` / `rowgen_30p.json`.

Take-away: with Gurobi, exact CoS row generation is tractable well beyond toy sizes, and for these
instances the core is non-empty (eps_min = 0), so Owen's `|gap|/N` is a loose upper bound.

## Solver note

The **separation MIP** uses Gurobi (`--sep-solver gurobi`, via `CoreComputation(mipsolver='gurobi')`,
implemented as `SeparationProblem._solve_with_gurobi`). Coalition-cost solves still use SCIP.
Gurobi keeps its default `MIPGap=1e-4`; `find_violated_coalition`'s violation verification uses a
matching **relative** tolerance so the 1e-4 gap doesn't trip a false mismatch.

## Open lever

- **Single-day vs 30-day.** This harness runs one representative day; `results_15p/`, `results_30p/`
  run 30 days each. Owen multi-day is cheap; row-gen multi-day is now feasible too (Gurobi) but 30×.

## Scope note

CHP allocation is **deliberately excluded** — with reserve/peak coupling, the CHP price-based
settlement rule is unresolved. The ingredients (prices + quantities) are saved so it can be
prototyped offline later.

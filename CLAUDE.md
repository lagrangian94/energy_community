# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The repo holds **two papers** that share one dispatch model. Keep them separate;
the shared layer at the root is the only thing both depend on.

```
.                        shared model layer (both papers import from here)
├── compact_utility.py     LocalEnergyMarket: the community dispatch MILP
├── data_generator.py      setup_lem_parameters(): all parameters, all scenarios
├── components.py
├── core.py                CoreComputation (row generation) + SeparationProblem
├── chp.py / solver.py / pricer.py    Dantzig-Wolfe column generation
├── ElecGen / HeatGen / HydroGen / ElsGen.py   load, price, device generators
├── data/                  raw inputs (Jeju KR, DK2) -- read as ./data/*, so CWD
│                          must be the repo root
├── ieee_owen/           IEEE submission (current work)
└── applied_energy/      earlier Applied Energy submission (archive)
```

**CWD must be the repo root** for anything that touches `data/`. Scripts in the
two paper folders bootstrap this themselves (`sys.path` + `os.chdir`); do not
remove those headers.

## ieee_owen/ — current paper

"Profit Allocation in Energy Community Games under Non-convex Operations:
A Duality-based Perspective" (Kim, Moon, Hesamzadeh, Kronqvist, Choi).

Stable profit allocation when the community dispatch is a MILP. Partially
dualizing the linking constraints leaves a Lagrangian sub-game that is a linear
production game, whose Owen solution is read off the Dantzig-Wolfe dual of a
single grand-coalition solve. It is stable against every coalition and gives up
efficiency by exactly the integrality gap, bounded as `O(|T|/n)`.

```
ieee_owen/
├── ieee_draft.txt          manuscript (LaTeX, IEEEtran)
├── reserve.txt             implementation spec for the reserve/peak block
├── large_community.py      15-, 30- and 60-prosumer configurations
├── reserve_metrics.py      reserve/peak output schema + derived metrics
├── config_tables.py/.tex   parameter + member-configuration tables (GENERATED)
├── weak_eps_experiment/    the experiment harness (entry points below)
└── copositive/             exact core existence via Burer lifting (paper Sec. 4.3)
```

Entry points, all run from anywhere:

```bash
python ieee_owen/weak_eps_experiment/run_experiment.py --sizes 6,15,30
python ieee_owen/weak_eps_experiment/run_multiday.py            # 31 days x scenarios
python ieee_owen/weak_eps_experiment/summarize.py               # regenerate all tables
python ieee_owen/weak_eps_experiment/paper_tables.py            # tab:results, tab:runtime
python ieee_owen/config_tables.py                               # tab:params, tab:config*
```

**Paper tables are generated, not typed.** Both table scripts read the CSVs and the
live `CONFIGURATION_*` objects. This is not a stylistic preference: the hand-typed
`n=30` column of `tab:results`/`tab:runtime` drifted off its data, carrying `v^MIP`,
`omega^LR` and both timings from a single instance while the rest of the column came
from the 31-day sweep. Re-run the scripts rather than editing the numbers.

The separation is the expensive half of the Owen phase (~85% of it at 60 prosumers).
`--no-stab` defers it and `--phase stab` fills the columns in afterwards, reading the
allocations back from `cg_day<D>.json` so the MILP and column generation are not
re-solved. A day whose separation hit its budget carries `stab_certified=False`, and
its `holds` verdict is then only meaningful when it says VIOLATED.

`run_multiday.py` is the main harness. JSON files are the durable source of
truth; CSVs are regenerated from them. A run is skipped if its rows already
exist -- use `--force` to recompute. `--runs`/`--days` filter.

### Model type matters

- `model_type='mip'` — the MILP dispatch. **This is what the paper uses**, everywhere.
- `model_type='mip_fix_binaries'` — same, with commitment fixed from a prior solve.
- `model_type='lp'` — the *linear production game* of Definition `def:lpg`, a convex
  contrast case, **not** a relaxation of the MILP. It creates no commitment
  variables and requires `eff_type=2`. Reserve is refused here (guarded), because
  electrolyzer/heat-pump headroom cannot be expressed without commitment binaries.

Note `v^LR` in the manuscript is the **Lagrangian** relaxation (computed from MIP
subproblems via column generation), *not* `model_type='lp'`. Do not conflate them.

### Reserve / peak coupling

Two community-level channels beyond the carrier balances (`reserve.txt`):

- **Reserve**: one symmetric FCR-N-type capacity product `r_sym` — a single scalar,
  no time index — that must be deliverable in *both* directions in *every* hour.
  The same variable enters both coupling row families, so at the optimum
  `r_sym* = min_t min(sum_j r+, sum_j r-)` without that min ever being written down.
  Held for the whole horizon, so the payment is `|T| * pi_res * r_sym`
  (baseline `24 * 56 = 1344` EUR/MW/day) — the `|T|` factor is easy to drop by mistake.
- **Peak**: `sum_j (i_E_gri - e_E_gri) <= p`, cost `delta_peak * p`.

Prices are **absolute**, not fractions of the import price:
`reserve_price` [EUR/MW.h] ∈ {0, 11, 56}, `peak_penalty` [EUR/MW] ∈ {0, 150, 200}.
Both default to 0, which leaves the model byte-identical to the pre-reserve version.

Per-asset headroom is private (it enters every subproblem); the coupling rows and
the shared variables `r_sym`, `p` belong to the master only. Heat-pump reserve is
in **electric** MW while its window is on **heat** output, so `nu_cop` converts
between them.

### reserve_metrics.py

`r_plus`/`r_minus` carry no objective cost, so the solver reports only as much as
`r_sym` needs — reading capability off them measures degeneracy, not capability.
Use `max_headroom_at_dispatch()` (closed form, no extra solve) for anything about
capability, and `solve_standalone_r_sym()` for the no-pooling baseline.

The pooling gain and the time/direction mechanism split are **separate objects
with different baselines**; do not chain them.

## applied_energy/ — earlier submission (archive)

Core-selecting mechanism design for local energy markets; compares three pricing
methods (IP / convex hull / LP relaxation) and produces the price figures. The
IEEE paper explicitly has no price figures, so nothing here feeds it.

```
applied_energy/
├── scripts/   analysis_{mip,lp,pwl_plot}.py, visualize.py, sensitivity_analysis*.py, ...
├── docs/      analysis_pwl_pricing.md, cg_smoothing.md, 15player_config.md, idea.txt
├── figures/   price/storage PNGs, LaTeX tables
└── results/   results_*, working_*, compare/, with_ess/, chp_ip_lp_*  (~580 files)
```

These scripts still run (`python applied_energy/scripts/analysis_mip.py`) — each
has a path bootstrap that puts the repo root on `sys.path` and resolves result
folders under `applied_energy/results/`. Treat them as archive: prefer not to
change the shared layer in ways that break them, but the IEEE paper is what is
maintained.

## Environment

```bash
source ../.venv/bin/activate     # venv lives in the parent directory
pip install -r requirements.txt
```

Key dependencies: `pyscipopt` (SCIP), `gurobipy` (separation, optional), `highspy`
(fast pricing, optional), `numpy`, `pandas`, `matplotlib`, `scipy`.

## Known wart

`compact_utility.py` writes plots straight into the CWD (`plt.savefig('price.png')`
and similar, several places), so the repo root re-clutters with PNGs after any
solve. Left as is deliberately; do not "fix" it without asking.

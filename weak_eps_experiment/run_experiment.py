"""
Weak-ε-core experiment: Owen allocation vs. Row generation (cost-of-stability).

For each community size n ∈ {6, 15, 30}, reserve+peak coupling ON, T=24:
  METHOD 1  Owen allocation  (one column-generation solve)
      - eps_bound = |gap|/N            (Shapley-Folkman guarantee, free)
      - eps_meas  = weak-ε of the gap-corrected Owen point (measured by separation)
  METHOD 2  Row generation (CoS)  (exact minimal weak-ε)
      - eps = v*/n                     (v* = cost of stability)
  Relationship:  v*/n ≤ eps_meas ≤ eps_bound = |gap|/N.

DESIGN (to avoid expensive re-runs)
  - JSON files are the durable source of truth; results.csv is REGENERATED from
    them every run. Deleting a JSON re-runs only that piece.
  - A size is skipped if its JSON already exists (use --force to recompute).
  - Each stage is written to disk immediately after it finishes.
  - cg_<n>p.json stores, besides eps/time, the ingredients needed to prototype a
    CHP settlement OFFLINE later (the reserve/peak case we haven't settled):
       * Owen duals  sigma_u  (convexity duals)  and gap-corrected owen_u
       * master LP coupling duals (convex_hull_prices: balance/reserve/peak)
       * per-player MIP dispatch quantities on the coupling rows
         (i_E_gri, e_E_gri, r_plus, r_minus)  ⇒  price × quantity settlement.
    Columns/extreme points are intentionally NOT stored (compact).

Usage:
  python weak_eps_experiment/run_experiment.py                 # sizes 6 (default)
  python weak_eps_experiment/run_experiment.py --sizes 6,15,30 # all
  python weak_eps_experiment/run_experiment.py --sizes 30 --rowgen-time-limit 7200
  python weak_eps_experiment/run_experiment.py --sizes 15 --force
"""
import os, sys, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data_generator import setup_lem_parameters
from compact_utility import LocalEnergyMarket
from chp import ColumnGenerationSolver
from core import CoreComputation
import large_community as LC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Reserve/peak scenario (fractions of mean import price) — same as the validated
# smoke tests so results are comparable across sizes.
RESERVE_UP_RATIO = 0.10
RESERVE_DN_RATIO = 0.05
PEAK_PENALTY_RATIO = 0.05

# ----------------------------------------------------------------------------- helpers
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x

def _qty_by_player(results, key, players, T):
    """Per-player MIP dispatch quantity {u: {t: val}} for coupling-row settlement."""
    d = results.get(key, {}) if isinstance(results, dict) else {}
    out = {}
    for u in players:
        out[u] = {int(t): float(d.get((u, t), 0.0)) for t in T}
    return out

# ----------------------------------------------------------------------------- instances
def build_instance(n):
    """Return (players, configuration, time_periods, parameters, scenario_name)."""
    T = list(range(24))
    if n == 6:
        players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
        config = {
            "players_with_renewables": ['u1'], "players_with_solar": [], "players_with_wind": ['u1'],
            "players_with_electrolyzers": ['u2'], "players_with_heatpumps": ['u3'],
            "players_with_elec_storage": ['u1'], "players_with_hydro_storage": ['u2'],
            "players_with_heat_storage": ['u3'],
            "players_with_nfl_elec_demand": ['u4'], "players_with_nfl_hydro_demand": ['u5'],
            "players_with_nfl_heat_demand": ['u6'],
            "players_with_fl_elec_demand": ['u2', 'u3'],
            "players_with_fl_hydro_demand": [], "players_with_fl_heat_demand": [],
        }
        # base defaults, then switch reserve/peak on at the smoke prices
        params = setup_lem_parameters(players, config, T)
        mean_imp = float(np.mean(params["pi_E_gri"]["import"]))
        params['enable_reserve'] = True
        params['enable_peak'] = True
        params['pi_up'] = RESERVE_UP_RATIO * mean_imp
        params['pi_dn'] = RESERVE_DN_RATIO * mean_imp
        params['pi_E_peak'] = PEAK_PENALTY_RATIO * mean_imp
        return players, config, T, params, "6p_reserve_peak"

    if n == 15:
        players, config = LC.PLAYERS_15, LC.CONFIGURATION_15
        sens = {k: v[0] for k, v in LC.BASELINE_CANDIDATES_15.items()}
        override_fn = LC.apply_15player_overrides
        scenario = "15p_reserve_peak"
    elif n == 30:
        players, config = LC.PLAYERS_30, LC.CONFIGURATION_30
        sens = {k: v[0] for k, v in LC.BASELINE_CANDIDATES_30.items()}
        override_fn = LC.apply_30player_overrides
        scenario = "30p_reserve_peak"
    else:
        raise ValueError(f"unsupported size {n}")

    sens['reserve_up_ratio'] = RESERVE_UP_RATIO
    sens['reserve_dn_ratio'] = RESERVE_DN_RATIO
    sens['peak_penalty_ratio'] = PEAK_PENALTY_RATIO
    params = setup_lem_parameters(players, config, T, sens)
    params = override_fn(params, T)
    return players, config, T, params, scenario

# ----------------------------------------------------------------------------- Owen stage
def run_owen(n, players, T, params, scenario):
    print(f"\n{'='*72}\n[Owen] n={n}\n{'='*72}")
    # (1) grand-coalition MIP: v_mip + per-player dispatch quantities
    t0 = time.time()
    lem = LocalEnergyMarket(players, T, params, model_type='mip')
    lem.model.hideOutput()
    ret = lem.solve_complete_model(analyze_revenue=False)
    status_mip, results_ip = ret[0], ret[1]
    v_mip = float(lem.model.getObjVal())
    t_mip = time.time() - t0
    print(f"  MIP status={status_mip}  v_mip={v_mip:.4f}  ({t_mip:.1f}s)")

    # (2) column generation (convex-hull pricing, smoothed)
    # Warm-start only from per-(u,t) private dispatch dicts; drop the community
    # shared scalars (r_up/r_dn/chi_peak_E) that reserve/peak adds to results_ip —
    # they are master RMP variables, not subproblem columns, and _add_initial_columns
    # assumes every init_sol value is a (u,t)-indexed dict.
    init_priv = {k: v for k, v in results_ip.items() if isinstance(v, dict)}
    t0 = time.time()
    cg = ColumnGenerationSolver(players, T, params, model_type='mip',
                                init_sol=init_priv, smoothing=True)
    status_cg, solution, v_chp, _ = cg.solve()
    t_cg = time.time() - t0
    print(f"  CG status={status_cg}  v_chp={v_chp:.4f}  ({t_cg:.1f}s)")

    # (3) Owen allocation + duals
    owen_res = cg.compute_owen_allocation(v_mip)
    sigma = owen_res['sigma']              # convexity duals (raw Owen, cost conv.)
    owen = owen_res['owen']                # gap-corrected, budget-balanced (Σ=v_mip)
    gap = owen_res['gap']
    eps_bound = owen_res['eps']            # |gap|/N
    prices = solution.get('convex_hull_prices', {})   # master LP coupling duals

    # (Owen's own core-violation re-check via separation is intentionally SKIPPED —
    #  we report only the free |gap|/N bound; row generation provides the exact/
    #  lower-bound eps for comparison.)
    print(f"  Owen eps_bound=|gap|/N={eps_bound:.6f}")

    doc = {
        'scenario': scenario, 'n_players': n, 'players': players, 'T': len(T),
        'reserve_peak': {'pi_up': params.get('pi_up'), 'pi_dn': params.get('pi_dn'),
                         'pi_E_peak': params.get('pi_E_peak'),
                         'enable_reserve': params.get('enable_reserve'),
                         'enable_peak': params.get('enable_peak')},
        'v_mip': v_mip, 'v_chp': v_chp, 'gap': gap,
        'eps_bound_gap_over_N': eps_bound,
        'time_mip_s': t_mip, 'time_cg_s': t_cg,
        'owen_sigma_cost': sigma,       # raw Owen point (Σ = v_chp)
        'owen_alloc_cost': owen,        # gap-corrected weak-eps-core point (Σ = v_mip)
        'master_coupling_duals': prices,   # convex_hull_prices: balance/reserve/peak
        # per-player MIP dispatch quantities on the coupling rows (for offline CHP settlement)
        'mip_quantities': {
            'i_E_gri': _qty_by_player(results_ip, 'i_E_gri', players, T),
            'e_E_gri': _qty_by_player(results_ip, 'e_E_gri', players, T),
            'r_plus':  _qty_by_player(results_ip, 'r_plus', players, T),
            'r_minus': _qty_by_player(results_ip, 'r_minus', players, T),
        },
    }
    path = os.path.join(OUT, f"cg_{n}p.json")
    with open(path, 'w') as f:
        json.dump(_jsonable(doc), f, indent=2)
    print(f"  -> saved {path}")
    return doc

# ----------------------------------------------------------------------------- RowGen stage
def run_rowgen(n, players, T, params, scenario, time_limit, sep_solver=None):
    print(f"\n{'='*72}\n[RowGen / CoS] n={n}  (time_limit={time_limit}s, sep_solver={sep_solver or 'scip'})\n{'='*72}")
    core_comp = CoreComputation(players, 'mip', T, params, mipsolver=sep_solver)
    t0 = time.time()
    alloc, success = core_comp.compute_core(max_iterations=int(1e8),
                                            tolerance=1e-6, time_limit=time_limit)
    t_rg = time.time() - t0
    v_star = getattr(core_comp, 'cost_of_stability_value', None)
    weak_eps = getattr(core_comp, 'weak_eps', None)
    converged = bool(getattr(core_comp, 'cos_converged', False))
    # When NOT converged, v*/weak_eps are LOWER BOUNDS (bracket eps_min from below);
    # when converged they are exact.
    print(f"  success={success} converged={converged}  v*={v_star}  weak_eps={weak_eps}  ({t_rg:.1f}s)")

    doc = {
        'scenario': scenario, 'n_players': n, 'T': len(T),
        'converged': converged,
        'core_nonempty': bool(success),
        'vstar_is_lower_bound': (not converged),
        'cost_of_stability_vstar': (float(v_star) if v_star is not None else None),
        'weak_eps_vstar_over_n': (float(weak_eps) if weak_eps is not None else None),
        'time_rowgen_s': t_rg,
        'time_limit_s': time_limit,
        'n_coalitions_generated': len(core_comp.coalition_costs),
        'core_allocation_cost': (alloc if isinstance(alloc, dict) else None),
        'cos_raw_payoffs': getattr(core_comp, 'cos_raw_payoffs', None),
        'coalition_costs': {"_".join(k): v for k, v in core_comp.coalition_costs.items()},
    }
    path = os.path.join(OUT, f"rowgen_{n}p.json")
    with open(path, 'w') as f:
        json.dump(_jsonable(doc), f, indent=2)
    print(f"  -> saved {path}")
    return doc

# ----------------------------------------------------------------------------- CSV rebuild
def rebuild_csv():
    import csv
    rows = []
    for cgf in sorted(glob.glob(os.path.join(OUT, "cg_*p.json"))):
        d = json.load(open(cgf))
        rows.append({'n_players': d['n_players'], 'method': 'owen',
                     'time_s': round(d['time_mip_s'] + d['time_cg_s'], 2),
                     'time_measure_s': '',
                     'eps': '', 'eps_bound': d['eps_bound_gap_over_N'],
                     'v_mip': d['v_mip'], 'v_chp': d['v_chp'], 'gap': d['gap'],
                     'converged': True, 'n_coalitions': ''})
    for rgf in sorted(glob.glob(os.path.join(OUT, "rowgen_*p.json"))):
        d = json.load(open(rgf))
        rows.append({'n_players': d['n_players'], 'method': 'rowgen',
                     'time_s': round(d['time_rowgen_s'], 2), 'time_measure_s': '',
                     'eps': d.get('weak_eps_vstar_over_n'), 'eps_bound': '',
                     'v_mip': '', 'v_chp': '', 'gap': '',
                     'converged': d['converged'], 'n_coalitions': d['n_coalitions_generated']})
    rows.sort(key=lambda r: (r['n_players'], r['method']))
    cols = ['n_players', 'method', 'time_s', 'time_measure_s', 'eps', 'eps_bound',
            'v_mip', 'v_chp', 'gap', 'converged', 'n_coalitions']
    path = os.path.join(OUT, "results.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> results.csv rebuilt ({len(rows)} rows)")
    return path

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='6', help='comma list, e.g. 6,15,30')
    ap.add_argument('--rowgen-time-limit', type=float, default=3600)
    ap.add_argument('--sep-solver', default='gurobi',
                    help="separation MIP solver for row-gen: gurobi (default) | highs | scip")
    ap.add_argument('--force', action='store_true', help='recompute even if JSON exists')
    ap.add_argument('--skip-rowgen', action='store_true')
    ap.add_argument('--skip-owen', action='store_true')
    args = ap.parse_args()
    sep_solver = None if args.sep_solver.lower() == 'scip' else args.sep_solver.lower()
    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]

    for n in sizes:
        players, config, T, params, scenario = build_instance(n)
        cg_path = os.path.join(OUT, f"cg_{n}p.json")
        rg_path = os.path.join(OUT, f"rowgen_{n}p.json")
        if not args.skip_owen and (args.force or not os.path.exists(cg_path)):
            run_owen(n, players, T, params, scenario)
        else:
            print(f"[Owen]   n={n}: skip (exists)" if os.path.exists(cg_path) else f"[Owen]   n={n}: skipped")
        if not args.skip_rowgen and (args.force or not os.path.exists(rg_path)):
            run_rowgen(n, players, T, params, scenario, args.rowgen_time_limit, sep_solver=sep_solver)
        else:
            print(f"[RowGen] n={n}: skip (exists)" if os.path.exists(rg_path) else f"[RowGen] n={n}: skipped")

    rebuild_csv()

if __name__ == "__main__":
    main()

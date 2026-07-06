"""
Multi-day weak-ε-core experiment: Owen vs. Row generation (cost-of-stability),
structured like results_15p/ results_30p/ results_53/ (one row per day).

Runs, IN ORDER (sensitivity 6p scenarios → 15p → 30p), each over 31 days, T=24,
reserve+peak coupling ON (0.10 / 0.05 / 0.05 × mean import):

  6p sensitivity scenarios (players u1..u6, from results_53 / sensitivity_analysis_claude):
    baseline, low_h2_margin, full_storage, community_size_350, community_size_1000,
    export_cap_020  (per-player electricity export cap via variable ub)
  15p (large_community CONFIGURATION_15), 30p (CONFIGURATION_30)  — baseline only.

TWO-PHASE per run (so the slow/possibly-non-converging row-gen never blocks Owen):
  Phase 1  Owen for ALL 31 days   → owen.csv + cg_day<D>.json (offline-settlement ingredients)
  Phase 2  RowGen for ALL 31 days → rowgen.csv   (Gurobi separation; per-day time budget)

RESUMABLE: a (run, phase, day) already present in its CSV is skipped. Safe to kill and
relaunch; only missing pieces recompute. Row-gen for 30p is expected to hit the per-day
budget without converging — the partial CoS lower bound is stored (v* ≥ ...).

Usage:
  python weak_eps_experiment/run_multiday.py                       # everything, both phases
  python weak_eps_experiment/run_multiday.py --phase owen          # Owen only (fast, all runs)
  python weak_eps_experiment/run_multiday.py --runs baseline_6p,baseline_15p
  python weak_eps_experiment/run_multiday.py --days 1,2,3 --force
"""
import os, sys, json, time, csv, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data_generator import setup_lem_parameters
from compact_utility import LocalEnergyMarket
from chp import ColumnGenerationSolver
from core import CoreComputation
import large_community as LC

OUT = os.path.dirname(os.path.abspath(__file__))
T = list(range(24))
DAYS = list(range(1, 32))                      # 31 days, matching results_53
RESERVE_UP, RESERVE_DN, PEAK = 0.10, 0.05, 0.05
SEP_SOLVER = 'gurobi'

# 6-player config (u1..u6) — same as sensitivity_analysis_claude / analysis_mip
P6 = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
C6 = {
    "players_with_renewables": ['u1'], "players_with_solar": [], "players_with_wind": ['u1'],
    "players_with_electrolyzers": ['u2'], "players_with_heatpumps": ['u3'],
    "players_with_elec_storage": ['u1'], "players_with_hydro_storage": ['u2'],
    "players_with_heat_storage": ['u3'],
    "players_with_nfl_elec_demand": ['u4'], "players_with_nfl_hydro_demand": ['u5'],
    "players_with_nfl_heat_demand": ['u6'],
    "players_with_fl_elec_demand": ['u2', 'u3'],
    "players_with_fl_hydro_demand": [], "players_with_fl_heat_demand": [],
}
# 6p baseline candidate set (scalars) — from sensitivity_analysis_claude.BASELINE_CANDIDATES
BASE6 = {
    'use_korean_price': True, 'use_tou_elec': False, 'import_factor': 1.5, 'month': 1,
    'hp_cap': 0.8, 'els_cap': 1, 'num_households': 700, 'nu_cop': 3.28,
    'c_su_G': 50.0, 'c_su_H': 10.0, 'base_h2_price_eur': 5000 / 1500,
    'e_E_cap_ratio': 1.0, 'e_H_cap_ratio': 1.0, 'e_G_cap_ratio': 1.0,
    'eff_type': 1, 'segments': 6, 'peak_penalty_ratio': 0.0,
    'wind_el_ratio': 1.0, 'solar_el_ratio': 1.0,
    'storage_power_ratio_E': 0.25, 'storage_power_ratio_G': 0.25, 'storage_power_ratio_H': 0.25,
    'storage_capacity_ratio_E': 3.0, 'storage_capacity_ratio_G': 0.0, 'storage_capacity_ratio_H': 0.0,
    'initial_soc_ratio_E': 0.2, 'initial_soc_ratio_G': 0.2, 'initial_soc_ratio_H': 0.2,
}

def scalarize(cand):
    return {k: (v[0] if isinstance(v, list) else v) for k, v in cand.items()}

# run registry (executed in this order)
RUNS = [
    dict(name='baseline_6p',           players=P6, config=C6, base=BASE6, ov={}, ovfn=None, budget=300),
    dict(name='low_h2_margin_6p',      players=P6, config=C6, base=BASE6,
         ov={'base_h2_price_eur': 2.0, 'import_factor': 3.0}, ovfn=None, budget=300),
    dict(name='full_storage_6p',       players=P6, config=C6, base=BASE6,
         ov={'storage_capacity_ratio_G': 3.0, 'storage_capacity_ratio_H': 3.0}, ovfn=None, budget=300),
    dict(name='community_size_350_6p', players=P6, config=C6, base=BASE6,
         ov={'num_households': 350}, ovfn=None, budget=300),
    dict(name='community_size_1000_6p',players=P6, config=C6, base=BASE6,
         ov={'num_households': 1000}, ovfn=None, budget=300),
    dict(name='export_cap_020_6p',     players=P6, config=C6, base=BASE6,
         ov={'e_E_cap_ratio': 0.2, 'e_G_cap_ratio': 0.2, 'e_H_cap_ratio': 0.2}, ovfn=None, budget=300),
    dict(name='baseline_15p', players=LC.PLAYERS_15, config=LC.CONFIGURATION_15,
         base=scalarize(LC.BASELINE_CANDIDATES_15), ov={}, ovfn=LC.apply_15player_overrides, budget=900),
    dict(name='baseline_30p', players=LC.PLAYERS_30, config=LC.CONFIGURATION_30,
         base=scalarize(LC.BASELINE_CANDIDATES_30), ov={}, ovfn=LC.apply_30player_overrides, budget=3600),
]

# --------------------------------------------------------------------------- helpers
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x

def _qty(results, key, players):
    d = results.get(key, {}) if isinstance(results, dict) else {}
    return {u: {int(t): float(d.get((u, t), 0.0)) for t in T} for u in players}

def build_params(run, day):
    sens = dict(run['base']); sens.update(run['ov'])
    sens['day'] = day
    sens['reserve_up_ratio'] = RESERVE_UP
    sens['reserve_dn_ratio'] = RESERVE_DN
    sens['peak_penalty_ratio'] = PEAK
    params = setup_lem_parameters(run['players'], run['config'], T, sens)
    if run['ovfn'] is not None:
        params = run['ovfn'](params, T)
    return params

def run_dir(run):
    d = os.path.join(OUT, run['name'])
    os.makedirs(d, exist_ok=True)
    return d

def existing_days(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        with open(csv_path) as f:
            return {int(r['day']) for r in csv.DictReader(f) if r.get('day')}
    except Exception:
        return set()

def append_row(csv_path, row, fieldnames):
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)

OWEN_COLS = ['run', 'day', 'n_players', 'num_households', 'base_h2_price_eur', 'import_factor',
             'v_mip', 'v_chp', 'gap', 'eps_bound', 'time_mip_s', 'time_cg_s']
ROWGEN_COLS = ['run', 'day', 'n_players', 'converged', 'vstar_is_lower_bound',
               'cost_of_stability_vstar', 'weak_eps', 'time_rowgen_s', 'n_coalitions']

# --------------------------------------------------------------------------- Owen phase
def owen_day(run, day):
    players = run['players']
    params = build_params(run, day)
    t0 = time.time()
    lem = LocalEnergyMarket(players, T, params, model_type='mip')
    lem.model.hideOutput()
    ret = lem.solve_complete_model(analyze_revenue=False)
    results_ip = ret[1]
    v_mip = float(lem.model.getObjVal())
    t_mip = time.time() - t0

    init_priv = {k: v for k, v in results_ip.items() if isinstance(v, dict)}
    t0 = time.time()
    cg = ColumnGenerationSolver(players, T, params, model_type='mip',
                                init_sol=init_priv, smoothing=True)
    _, solution, v_chp, _ = cg.solve()
    t_cg = time.time() - t0

    owen_res = cg.compute_owen_allocation(v_mip)
    gap, eps_bound = owen_res['gap'], owen_res['eps']

    # offline CHP-settlement ingredients per (run, day)
    doc = {
        'run': run['name'], 'day': day, 'n_players': len(players), 'T': len(T),
        'reserve_peak': {'pi_up': params.get('pi_up'), 'pi_dn': params.get('pi_dn'),
                         'pi_E_peak': params.get('pi_E_peak')},
        'v_mip': v_mip, 'v_chp': v_chp, 'gap': gap, 'eps_bound_gap_over_N': eps_bound,
        'time_mip_s': t_mip, 'time_cg_s': t_cg,
        'owen_sigma_cost': owen_res['sigma'], 'owen_alloc_cost': owen_res['owen'],
        'master_coupling_duals': solution.get('convex_hull_prices', {}),
        'mip_quantities': {k: _qty(results_ip, k, players)
                           for k in ('i_E_gri', 'e_E_gri', 'r_plus', 'r_minus')},
    }
    with open(os.path.join(run_dir(run), f"cg_day{day}.json"), 'w') as f:
        json.dump(_jsonable(doc), f, indent=2)

    row = {'run': run['name'], 'day': day, 'n_players': len(players),
           'num_households': dict(run['base'], **run['ov']).get('num_households', ''),
           'base_h2_price_eur': (dict(run['base'], **run['ov']).get('base_h2_price_eur', '')),
           'import_factor': (dict(run['base'], **run['ov']).get('import_factor', '')),
           'v_mip': v_mip, 'v_chp': v_chp, 'gap': gap, 'eps_bound': eps_bound,
           'time_mip_s': round(t_mip, 2), 'time_cg_s': round(t_cg, 2)}
    print(f"  [Owen] {run['name']} day {day}: v_mip={v_mip:.3f} eps_bound={eps_bound:.5f} "
          f"({t_mip+t_cg:.0f}s)")
    return row

# --------------------------------------------------------------------------- RowGen phase
def rowgen_day(run, day, budget):
    players = run['players']
    params = build_params(run, day)
    cc = CoreComputation(players, 'mip', T, params, mipsolver=SEP_SOLVER)
    t0 = time.time()
    _, success = cc.compute_core(max_iterations=int(1e8), tolerance=1e-6, time_limit=budget)
    t_rg = time.time() - t0
    v_star = getattr(cc, 'cost_of_stability_value', None)
    weak_eps = getattr(cc, 'weak_eps', None)
    converged = bool(getattr(cc, 'cos_converged', False))
    row = {'run': run['name'], 'day': day, 'n_players': len(players),
           'converged': converged, 'vstar_is_lower_bound': (not converged),
           'cost_of_stability_vstar': (float(v_star) if v_star is not None else ''),
           'weak_eps': (float(weak_eps) if weak_eps is not None else ''),
           'time_rowgen_s': round(t_rg, 2),
           'n_coalitions': len(cc.coalition_costs)}
    print(f"  [RowGen] {run['name']} day {day}: converged={converged} "
          f"weak_eps={weak_eps} ({t_rg:.0f}s, {len(cc.coalition_costs)} coalitions)")
    return row

# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['owen', 'rowgen', 'both'], default='both')
    ap.add_argument('--runs', default='', help='comma list of run names (default: all)')
    ap.add_argument('--days', default='', help='comma list of days (default: 1..31)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    run_filter = set(args.runs.split(',')) if args.runs else None
    days = [int(d) for d in args.days.split(',')] if args.days else DAYS

    for run in RUNS:
        if run_filter and run['name'] not in run_filter:
            continue
        rd = run_dir(run)
        owen_csv = os.path.join(rd, 'owen.csv')
        rowgen_csv = os.path.join(rd, 'rowgen.csv')

        # Phase 1: Owen for all days
        if args.phase in ('owen', 'both'):
            done = set() if args.force else existing_days(owen_csv)
            todo = [d for d in days if d not in done]
            print(f"\n=== [Owen phase] {run['name']}: {len(todo)} days ===")
            for d in todo:
                try:
                    append_row(owen_csv, owen_day(run, d), OWEN_COLS)
                except Exception as e:
                    print(f"  !! Owen {run['name']} day {d} FAILED: {e}")

        # Phase 2: RowGen for all days
        if args.phase in ('rowgen', 'both'):
            done = set() if args.force else existing_days(rowgen_csv)
            todo = [d for d in days if d not in done]
            print(f"\n=== [RowGen phase] {run['name']}: {len(todo)} days (budget {run['budget']}s) ===")
            for d in todo:
                try:
                    append_row(rowgen_csv, rowgen_day(run, d, run['budget']), ROWGEN_COLS)
                except Exception as e:
                    print(f"  !! RowGen {run['name']} day {d} FAILED: {e}")

    print("\nDONE.")

if __name__ == "__main__":
    main()

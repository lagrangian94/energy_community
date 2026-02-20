"""
large_community.py — 15-Player Energy Community Experiment

Reuses run_experiment() pattern from sensitivity_analysis_claude.py with
15-player configuration and per-player parameter overrides.

Player roles:
  u1:  Wind + ESS              u7:  Solar + ESS
  u2:  Electrolyzer + H2 Sto   u8:  Electrolyzer (no storage)
  u3:  Heat Pump + Heat Sto    u9:  Heat Pump (no storage)
  u4:  Elec Consumer           u10: Wind + Electrolyzer + ESS + H2 Sto (vertically integrated)
  u5:  H2 Consumer             u11: Pure ESS (workaround: solar cap=0, demand=0)
  u6:  Heat Consumer           u12: Elec Consumer
                                u13: H2 Consumer
                                u14: Heat Consumer
                                u15: Multi-sector Consumer (Elec + H2)
"""

import sys
import os
import gc
import time
import argparse
import itertools

import numpy as np
import pandas as pd

from data_generator import setup_lem_parameters
from compact_utility import LocalEnergyMarket
from chp import ColumnGenerationSolver
from core import CoreComputation


# ============================================================
# 15-Player Configuration
# ============================================================

PLAYERS_15 = [
    'u1', 'u2', 'u3', 'u4', 'u5', 'u6',
    'u7', 'u8', 'u9', 'u10', 'u11', 'u12',
    'u13', 'u14', 'u15',
]

CONFIGURATION_15 = {
    # u11 in solar with cap=0 (workaround for Pure ESS to get export vars)
    # 만약 다른 pure hydro/heat storage player가 들어온다면 얘네도 players_with_electrolyzers, players_with_heatpumps에 추가해야 함; cap=0으로 두고.
    "players_with_renewables": ['u1', 'u7', 'u10', 'u11'],
    "players_with_solar": ['u7', 'u11'],          # u11: cap=0
    "players_with_wind": ['u1', 'u10'],
    "players_with_electrolyzers": ['u2', 'u8', 'u10'],
    "players_with_heatpumps": ['u3', 'u9'],
    "players_with_elec_storage": ['u1', 'u7', 'u10', 'u11'],
    "players_with_hydro_storage": ['u8', 'u10'],
    "players_with_heat_storage": ['u9'],
    # u11 in nfl_elec_demand with demand=0 (workaround for import vars)
    "players_with_nfl_elec_demand": ['u4', 'u11', 'u12', 'u15'],
    "players_with_nfl_hydro_demand": ['u5', 'u13', 'u15'],
    "players_with_nfl_heat_demand": ['u6', 'u14'],
    # auto-set by setup_lem_parameters (electrolyzers + heatpumps)
    "players_with_fl_elec_demand": [],
    "players_with_fl_hydro_demand": [],
    "players_with_fl_heat_demand": [],
}

BASELINE_CANDIDATES_15 = {
    'use_korean_price': [True],
    'use_tou_elec': [False],
    'import_factor': [1.5],
    'month': [1],
    'hp_cap': [0.8],
    'els_cap': [1],
    'num_households': [700],
    'nu_cop': [3.28],
    'c_su_G': [50],
    'c_su_H': [10],
    'base_h2_price_eur': [5000 / 1500],
    'e_E_cap_ratio': [1.0],
    'e_H_cap_ratio': [1.0],
    'e_G_cap_ratio': [1.0],
    'eff_type': [1],
    'segments': [6],
    'peak_penalty_ratio': [0.0],
    'wind_el_ratio': [1.0],
    'solar_el_ratio': [1.0],
    'storage_power_ratio_E': [0.25],
    'storage_power_ratio_G': [0.25],
    'storage_power_ratio_H': [0.25],
    'storage_capacity_ratio_E': [3.0],
    'storage_capacity_ratio_G': [3.0],
    'storage_capacity_ratio_H': [3.0],
    'initial_soc_ratio_E': [0.2],
    'initial_soc_ratio_G': [0.2],
    'initial_soc_ratio_H': [0.2],
    'day': [1],
}


# ============================================================
# Per-player parameter overrides
# ============================================================

def apply_15player_overrides(parameters, time_periods):
    """
    Apply per-player capacity differentiation after setup_lem_parameters().

    Key overrides:
      u10: small wind (0.5x), small ESS (0.5x)
      u11: Pure ESS workaround — solar cap=0, demand=0, export cap = storage power
    """
    # --- u10: vertically integrated, smaller wind + ESS ---
    for t in time_periods:
        key = f'renewable_cap_u10_{t}'
        if key in parameters:
            parameters[key] *= 0.5

    for suffix in ['storage_power_E_u10', 'storage_capacity_E_u10']:
        if suffix in parameters:
            parameters[suffix] *= 0.5
    # Recompute initial SOC after capacity change
    if 'storage_capacity_E_u10' in parameters:
        parameters['initial_soc_E_u10'] = (
            parameters.get('initial_soc_ratio_E', 0.2) * parameters['storage_capacity_E_u10']
        )

    # --- u11: Pure ESS workaround ---
    # Zero out renewable production (solar cap=0)
    for t in time_periods:
        parameters[f'renewable_cap_u11_{t}'] = 0.0

    # Export cap = storage power (ESS can only export what it discharges)
    storage_power_u11 = parameters.get('storage_power_E_u11', 0)
    parameters[f'e_E_cap_u11'] = storage_power_u11

    # Zero out demand and utility (u11 has no real demand)
    for t in time_periods:
        parameters[f'd_E_nfl_u11_{t}'] = 0.0
        parameters[f'u_E_u11_{t}'] = 0.0

    return parameters


# ============================================================
# Main experiment runner
# ============================================================

def run_single_day_pricing(sensitivity_analysis, players, configuration,
                           time_periods, scenario_name, run_id, total_runs):
    """
    Run IP → CHP → CHP Smoothing + core violation check for a single parameter combo.
    (Pricing only — no row generation.)
    Returns a dict with all pricing results for this run.
    """
    print(f"\n--- [Pricing] Run {run_id + 1}/{total_runs} | Day {sensitivity_analysis.get('day', '?')} ---")

    parameters = setup_lem_parameters(
        players, configuration, time_periods, sensitivity_analysis
    )
    parameters = apply_15player_overrides(parameters, time_periods)

    row = {
        'scenario': scenario_name,
        'run_id': run_id,
        **{k: v for k, v in sensitivity_analysis.items()},
    }

    # --- IP ---
    lem = LocalEnergyMarket(players, time_periods, parameters, model_type='mip')
    t0 = time.time()
    status, results_ip, _, community_prices, _ = lem.solve_complete_model(
        analyze_revenue=False
    )
    row['solve_time_ip'] = time.time() - t0

    player_profits, prices = lem.calculate_player_profits_with_community_prices(
        results_ip, community_prices
    )
    comparison = lem.compare_individual_vs_community_profits(
        results_ip, players, lem.model_type, time_periods,
        parameters, player_profits, community_prices
    )
    profit_ip = {u: player_profits[u]["net_profit"] for u in players}
    for u in players:
        row[f'profit_ip_{u}'] = profit_ip[u]

    # IP core violation (separation only — brute force infeasible for 15 players)
    core_comp = CoreComputation(players, 'mip', time_periods, parameters)
    cost_ip = {u: -1 * profit_ip[u] for u in players}
    coalition_ip, violation_ip, isimp_ip = core_comp.measure_stability_violation(cost_ip)
    row['violation_ip'] = violation_ip
    row['blocking_coalition_ip'] = str(coalition_ip) if violation_ip > 1e-6 else ''
    row['isimp_ip'] = isimp_ip
    del lem
    gc.collect()

    # # --- CHP ---
    # cg = ColumnGenerationSolver(
    #     players, time_periods, parameters,
    #     model_type='mip', init_sol=results_ip
    # )
    # t0 = time.time()
    # status_chp, results_chp, obj_val, solution_by_player = cg.solve()
    # row['solve_time_chp'] = time.time() - t0
    # print(f"CHP time: {row['solve_time_chp']:.2f}s")

    # if status_chp == "optimal":
    #     print(f"CHP optimal objective: {obj_val:.2f} EUR")
    #     community_prices_chp = results_chp.get('convex_hull_prices', {})
    #     if "capacity_prices" in results_chp:
    #         community_prices_chp = community_prices_chp | results_chp['capacity_prices']
    #     synergy_results = cg.analyze_synergy_with_convex_hull_prices(
    #         results_ip, obj_val, community_prices_chp
    #     )
    #     profit_chp = {u: synergy_results['community_profits'][u]['net_profit'] for u in players}
    # else:
    #     print(f"CHP failed: {status_chp}")
    #     profit_chp = {u: np.nan for u in players}

    # for u in players:
    #     row[f'profit_chp_{u}'] = profit_chp[u]

    # if status_chp == "optimal":
    #     cost_chp = {u: -1 * profit_chp[u] for u in players}
    #     coalition_chp, violation_chp, isimp_chp = core_comp.measure_stability_violation(cost_chp)
    #     row['violation_chp'] = violation_chp
    #     row['blocking_coalition_chp'] = str(coalition_chp) if violation_chp > 1e-6 else ''
    #     row['isimp_chp'] = isimp_chp
    #     for carrier in ['electricity', 'heat', 'hydrogen']:
    #         if carrier in community_prices_chp:
    #             for t in time_periods:
    #                 row[f'chp_price_{carrier}_{t}'] = community_prices_chp[carrier][t]
    # else:
    #     row['violation_chp'] = np.nan
    #     row['blocking_coalition_chp'] = np.nan
    #     row['isimp_chp'] = np.nan
    # del cg
    # gc.collect()

    # --- CHP Smoothing ---
    cg_smooth = ColumnGenerationSolver(
        players, time_periods, parameters,
        model_type='mip', init_sol=results_ip, smoothing=True
    )
    t0 = time.time()
    status_smooth, results_chp_smooth, obj_val_smooth, _ = cg_smooth.solve()
    row['solve_time_chp_smooth'] = time.time() - t0

    if status_smooth == "optimal":
        community_prices_chp_smooth = results_chp_smooth.get('convex_hull_prices', {})
        if "capacity_prices" in results_chp_smooth:
            community_prices_chp_smooth = community_prices_chp_smooth | results_chp_smooth['capacity_prices']
        synergy_smooth = cg_smooth.analyze_synergy_with_convex_hull_prices(
            results_ip, obj_val_smooth, community_prices_chp_smooth
        )
        profit_chp_smooth = {u: synergy_smooth['community_profits'][u]['net_profit'] for u in players}
    else:
        profit_chp_smooth = {u: np.nan for u in players}

    for u in players:
        row[f'profit_chp_smooth_{u}'] = profit_chp_smooth[u]

    if status_smooth == "optimal":
        cost_chp_smooth = {u: -1 * profit_chp_smooth[u] for u in players}
        coalition_smooth, violation_smooth, isimp_smooth = core_comp.measure_stability_violation(
            cost_chp_smooth
        )
        row['violation_chp_smooth'] = violation_smooth
        row['blocking_coalition_chp_smooth'] = str(coalition_smooth) if violation_smooth > 1e-6 else ''
        row['isimp_chp_smooth'] = isimp_smooth
    else:
        row['violation_chp_smooth'] = np.nan
        row['blocking_coalition_chp_smooth'] = np.nan
        row['isimp_chp_smooth'] = np.nan

    # # Performance comparison
    # if row.get('solve_time_chp_smooth', 0) > 0:
    #     row['speedup_chp_smooth'] = row.get('solve_time_chp', 0) / row['solve_time_chp_smooth']
    # else:
    #     row['speedup_chp_smooth'] = np.nan
    # if status_chp == "optimal" and status_smooth == "optimal":
    #     row['obj_diff_chp_smooth'] = abs(obj_val - obj_val_smooth)
    # else:
    #     row['obj_diff_chp_smooth'] = np.nan
    del cg_smooth
    gc.collect()

    # Log
    flag = " *** IP VIOLATED ***" if row.get('violation_ip', 0) > 1e-6 else ""
    print(f"  IP: {row.get('violation_ip', 0):.4f}{flag} | "
          f"CHP: {row.get('violation_chp', 0):.4f} | "
          f"CHP_S: {row.get('violation_chp_smooth', 0):.4f} | "
          f"t_IP={row.get('solve_time_ip', 0):.1f}s "
          f"t_CHP={row.get('solve_time_chp', 0):.1f}s "
          f"t_CHP_S={row.get('solve_time_chp_smooth', 0):.1f}s "
          f"speedup={row.get('speedup_chp_smooth', 0):.2f}x")

    return row


def run_single_day_rowgen(sensitivity_analysis, players, configuration,
                          time_periods, scenario_name, run_id, total_runs,
                          time_limit=3600):
    """
    Run Row Generation (Core computation) for a single parameter combo.
    Returns a dict with rowgen results for this run.
    """
    print(f"\n--- [RowGen] Run {run_id + 1}/{total_runs} | Day {sensitivity_analysis.get('day', '?')} ---")

    parameters = setup_lem_parameters(
        players, configuration, time_periods, sensitivity_analysis
    )
    parameters = apply_15player_overrides(parameters, time_periods)

    row = {
        'scenario': scenario_name,
        'run_id': run_id,
        **{k: v for k, v in sensitivity_analysis.items()},
    }

    core_comp = CoreComputation(players, 'mip', time_periods, parameters)

    t0 = time.time()
    core_rowgen, success_rowgen = core_comp.compute_core(
        max_iterations=int(1e8), tolerance=1e-6, time_limit=time_limit
    )
    row['solve_time_rowgen'] = time.time() - t0
    print(f"Row generation time: {row['solve_time_rowgen']:.2f}s")

    if success_rowgen:
        # Verify stability
        coalition_rg, violation_rg, isimp_rg = core_comp.measure_stability_violation(core_rowgen)
        row['violation_rowgen'] = violation_rg
        row['blocking_coalition_rowgen'] = str(coalition_rg) if violation_rg > 1e-6 else ''
        row['isimp_rowgen'] = isimp_rg
        for u in players:
            row[f'profit_rowgen_{u}'] = -1 * core_rowgen[u]
    else:
        print("Row generation: core not found (empty or time limit)")
        row['violation_rowgen'] = np.nan
        row['blocking_coalition_rowgen'] = ''
        row['isimp_rowgen'] = np.nan
        for u in players:
            row[f'profit_rowgen_{u}'] = np.nan
    gc.collect()

    print(f"  RG: {row.get('violation_rowgen', 'N/A')} | "
          f"t_RG={row.get('solve_time_rowgen', 0):.1f}s")

    return row


def _save_with_day_update(outpath, new_df, day_values):
    """
    If outpath exists, replace rows matching day_values and keep the rest.
    If a day doesn't exist yet, the new rows are inserted.
    Result is sorted by day.
    """
    if os.path.exists(outpath):
        existing_df = pd.read_csv(outpath)
        keep = existing_df[~existing_df['day'].isin(day_values)]
        merged = pd.concat([keep, new_df], ignore_index=True)
        merged.sort_values('day', inplace=True)
        merged.to_csv(outpath, index=False)
    else:
        new_df.to_csv(outpath, index=False)


def run_experiment_pricing(sensitivity_analysis_candidates, players, configuration,
                           time_periods, scenario_name, output_dir='results_15p'):
    """
    Phase 1: Run all pricing (IP → CHP → CHP Smoothing) for all parameter combinations.
    """
    os.makedirs(output_dir, exist_ok=True)

    param_names = list(sensitivity_analysis_candidates.keys())
    param_values = [sensitivity_analysis_candidates[name] for name in param_names]

    total_runs = 1
    for v in param_values:
        total_runs *= len(v)
    print(f"\n[Phase 1: Pricing] Scenario: {scenario_name}")
    print(f"Total runs: {total_runs}")
    print(f"Players: {len(players)}")

    outpath = os.path.join(output_dir, f'{scenario_name}_pricing.csv')
    day_values = sensitivity_analysis_candidates.get('day', [])
    results_summary = []

    for i, values in enumerate(itertools.product(*param_values)):
        sensitivity_analysis = dict(zip(param_names, values))
        row = run_single_day_pricing(
            sensitivity_analysis, players, configuration,
            time_periods, scenario_name, i, total_runs
        )
        results_summary.append(row)

        # Incremental save — overwrite current days, keep others
        df_new = pd.DataFrame(results_summary)
        _save_with_day_update(outpath, df_new, day_values)

    df = pd.DataFrame(results_summary)
    print(f"\nPricing results saved to {outpath}")
    return df


def run_experiment_rowgen(sensitivity_analysis_candidates, players, configuration,
                          time_periods, scenario_name, output_dir='results_15p',
                          time_limit=3600):
    """
    Phase 2: Run row generation (core computation) for all parameter combinations.
    """
    os.makedirs(output_dir, exist_ok=True)

    param_names = list(sensitivity_analysis_candidates.keys())
    param_values = [sensitivity_analysis_candidates[name] for name in param_names]

    total_runs = 1
    for v in param_values:
        total_runs *= len(v)
    print(f"\n[Phase 2: RowGen] Scenario: {scenario_name}")
    print(f"Total runs: {total_runs}")
    print(f"Players: {len(players)}")
    print(f"Time limit per run: {time_limit}s")

    outpath = os.path.join(output_dir, f'{scenario_name}_rowgen.csv')
    day_values = sensitivity_analysis_candidates.get('day', [])
    results_summary = []

    for i, values in enumerate(itertools.product(*param_values)):
        sensitivity_analysis = dict(zip(param_names, values))
        row = run_single_day_rowgen(
            sensitivity_analysis, players, configuration,
            time_periods, scenario_name, i, total_runs,
            time_limit=time_limit
        )
        results_summary.append(row)

        # Incremental save — overwrite current days, keep others
        df_new = pd.DataFrame(results_summary)
        _save_with_day_update(outpath, df_new, day_values)

    df = pd.DataFrame(results_summary)
    print(f"\nRowGen results saved to {outpath}")
    return df


def run_experiment(sensitivity_analysis_candidates, players, configuration,
                   time_periods, scenario_name, output_dir='results_15p',
                   time_limit=3600):
    """
    Run both phases sequentially: pricing first, then row generation.
    Results are saved separately and merged at the end.
    """
    # Phase 1: Pricing
    df_pricing = run_experiment_pricing(
        sensitivity_analysis_candidates, players, configuration,
        time_periods, scenario_name, output_dir
    )

    # Phase 2: Row Generation
    df_rowgen = run_experiment_rowgen(
        sensitivity_analysis_candidates, players, configuration,
        time_periods, scenario_name, output_dir, time_limit=time_limit
    )

    # Merge on shared keys
    merge_keys = ['scenario', 'run_id'] + list(sensitivity_analysis_candidates.keys())
    df = df_pricing.merge(df_rowgen, on=merge_keys, how='left')

    outpath = os.path.join(output_dir, f'{scenario_name}.csv')
    df.to_csv(outpath, index=False)
    print(f"\nMerged results saved to {outpath}")

    print_scenario_summary(df, scenario_name)
    return df


def print_scenario_summary(df, scenario_name):
    """Print summary statistics for a scenario."""
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {scenario_name}")
    print(f"{'=' * 60}")
    total = len(df)
    n_ip = (df['violation_ip'] > 1e-6).sum()
    n_chp = (df['violation_chp'] > 1e-6).sum()
    print(f"  Total runs: {total}")
    print(f"  IP violations:  {n_ip}/{total} ({100 * n_ip / total:.0f}%)")
    print(f"  CHP violations: {n_chp}/{total} ({100 * n_chp / total:.0f}%)")
    if 'violation_chp_smooth' in df.columns:
        n_chp_s = (df['violation_chp_smooth'] > 1e-6).sum()
        print(f"  CHP Smooth violations: {n_chp_s}/{total} ({100 * n_chp_s / total:.0f}%)")
    if n_ip > 0:
        avg_mag = df.loc[df['violation_ip'] > 1e-6, 'violation_ip'].mean()
        max_mag = df.loc[df['violation_ip'] > 1e-6, 'violation_ip'].max()
        print(f"  Avg IP violation magnitude: {avg_mag:.4f}")
        print(f"  Max IP violation magnitude: {max_mag:.4f}")
    if 'violation_rowgen' in df.columns:
        n_rg_success = df['violation_rowgen'].notna().sum()
        n_rg_stable = (df['violation_rowgen'].dropna() <= 1e-6).sum()
        print(f"  Row gen success: {n_rg_success}/{total}")
        print(f"  Row gen stable:  {n_rg_stable}/{n_rg_success}")
    if 'speedup_chp_smooth' in df.columns:
        avg_speedup = df['speedup_chp_smooth'].mean()
        med_speedup = df['speedup_chp_smooth'].median()
        print(f"  Avg smoothing speedup: {avg_speedup:.2f}x")
        print(f"  Median smoothing speedup: {med_speedup:.2f}x")
    if 'obj_diff_chp_smooth' in df.columns:
        max_diff = df['obj_diff_chp_smooth'].max()
        print(f"  Max obj diff (CHP vs CHP_smooth): {max_diff:.6f}")
    # Timing comparison
    time_cols = ['solve_time_ip', 'solve_time_chp', 'solve_time_chp_smooth', 'solve_time_rowgen']
    available = [c for c in time_cols if c in df.columns]
    if available:
        print(f"\n  --- Timing (avg) ---")
        for c in available:
            label = c.replace('solve_time_', '').upper()
            print(f"  {label:>12s}: {df[c].mean():.1f}s")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='15-Player Energy Community Experiment')
    parser.add_argument('--day', type=int, nargs='+', default=[31],
                        help='Day(s) to run (e.g. --day 1 or --day 1 2 3)')
    parser.add_argument('--output', type=str, default='results_15p',
                        help='Output directory')
    parser.add_argument('--phase', type=str, default='pricing',
                        choices=['all', 'pricing', 'rowgen'],
                        help='Which phase to run: pricing only, rowgen only, or all (default: all)')
    parser.add_argument('--time-limit', type=float, default=3600,
                        help='Row generation time limit in seconds (default: 3600)')
    args = parser.parse_args()

    time_periods = list(range(24))

    candidates = dict(BASELINE_CANDIDATES_15)
    candidates['day'] = args.day

    if args.phase == 'pricing':
        run_experiment_pricing(
            candidates, PLAYERS_15, CONFIGURATION_15,
            time_periods, scenario_name='15player_baseline',
            output_dir=args.output,
        )
    elif args.phase == 'rowgen':
        run_experiment_rowgen(
            candidates, PLAYERS_15, CONFIGURATION_15,
            time_periods, scenario_name='15player_baseline',
            output_dir=args.output, time_limit=args.time_limit,
        )
    else:
        run_experiment(
            candidates, PLAYERS_15, CONFIGURATION_15,
            time_periods, scenario_name='15player_baseline',
            output_dir=args.output, time_limit=args.time_limit,
        )

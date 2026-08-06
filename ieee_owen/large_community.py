"""
large_community.py — 15/30-Player Energy Community Experiment

Reuses run_experiment() pattern from sensitivity_analysis_claude.py with
15- or 30-player configuration and per-player parameter overrides.

15-Player roles:
  u1:  Wind + ESS              u7:  Solar + ESS
  u2:  Electrolyzer            u8:  Electrolyzer + H2 Sto
  u3:  Heat Pump               u9:  Heat Pump + Heat Sto
  u4:  Elec Consumer           u10: Wind + Electrolyzer + ESS + H2 Sto (vertically integrated)
  u5:  H2 Consumer             u11: Pure ESS (workaround: solar cap=0, demand=0)
  u6:  Heat Consumer           u12: Solar + Elec Consumer (prosumer)
                                u13: H2 Consumer
                                u14: Heat Pump + Heat Consumer (prosumer)
                                u15: Elec + Heat Consumer

30-Player additions (u16–u30):
  Supply side:
    u16: Wind (second wind producer)
    u17: Solar (second solar generator)
    u18: Electrolyzer + H2 Storage
    u19: Electrolyzer (no storage)
    u20: Heat Pump + Heat Storage
    u21: Solar + Electrolyzer (solar-to-hydrogen pathway)
    u22: Pure ESS (standalone battery, like u11)
  Demand side (residential=elec+heat, refueling=H2, industrial=elec+H2):
    u23: Residential prosumer with rooftop solar (elec 0.5x + heat 0.4x demand + solar)
    u24: Residential consumer with ESS (elec 0.8x + heat 0.3x demand + ESS)
    u25: Residential pure consumer (elec 0.6x + heat 0.3x demand)
    u26: Refueling station with solar canopy (H2 0.6x demand + solar)
    u27: Refueling pure consumer (H2 0.5x demand)
    u28: Industrial pure consumer (elec 0.9x + H2 0.5x demand)
    u29: Industrial consumer with ESS (elec 0.7x + H2 0.4x demand + ESS)
    u30: Heat pure consumer (heat 0.3x demand)
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
    "players_with_renewables": ['u1', 'u7', 'u10', 'u11', 'u12'],
    "players_with_solar": ['u7', 'u11', 'u12'],   # u11: cap=0, u12: rooftop solar
    "players_with_wind": ['u1', 'u10'],
    "players_with_electrolyzers": ['u2', 'u8', 'u10'],
    "players_with_heatpumps": ['u3', 'u9', 'u14'],
    "players_with_elec_storage": ['u1', 'u7', 'u10', 'u11'],
    "players_with_hydro_storage": ['u8', 'u10'],
    "players_with_heat_storage": ['u9'],
    # u11 in nfl_elec_demand with demand=0 (workaround for import vars)
    "players_with_nfl_elec_demand": ['u4', 'u11', 'u12', 'u15'],
    "players_with_nfl_hydro_demand": ['u5', 'u13'],
    "players_with_nfl_heat_demand": ['u6', 'u14', 'u15'],
    # auto-set by setup_lem_parameters (electrolyzers + heatpumps)
    "players_with_fl_elec_demand": [],
    "players_with_fl_hydro_demand": [],
    "players_with_fl_heat_demand": [],
}

# ============================================================
# 30-Player Configuration
# ============================================================

PLAYERS_30 = [f'u{i}' for i in range(1, 31)]

CONFIGURATION_30 = {
    # All 15-player renewables + u16(wind), u17(solar), u21(solar+els), u22(pure ESS→solar cap=0), u23(solar), u26(solar)
    "players_with_renewables": [
        'u1', 'u7', 'u10', 'u11', 'u12',
        'u16', 'u17', 'u21', 'u22', 'u23', 'u26',
    ],
    "players_with_solar": [
        'u7', 'u11', 'u12',
        'u17', 'u21', 'u22', 'u23', 'u26',
    ],
    "players_with_wind": ['u1', 'u10', 'u16'],
    "players_with_electrolyzers": ['u2', 'u8', 'u10', 'u18', 'u19', 'u21'],
    "players_with_heatpumps": ['u3', 'u9', 'u14', 'u20'],
    "players_with_elec_storage": ['u1', 'u7', 'u10', 'u11', 'u22', 'u24', 'u29'],
    "players_with_hydro_storage": ['u8', 'u10', 'u18'],
    "players_with_heat_storage": ['u9', 'u20'],
    # Demand: u22 workaround (demand=0), residential(elec+heat), industrial(elec)
    "players_with_nfl_elec_demand": [
        'u4', 'u11', 'u12', 'u15',
        'u22', 'u23', 'u24', 'u25', 'u28', 'u29',
    ],
    "players_with_nfl_hydro_demand": [
        'u5', 'u13',
        'u26', 'u27', 'u28', 'u29',
    ],
    "players_with_nfl_heat_demand": [
        'u6', 'u14', 'u15',
        'u23', 'u24', 'u25', 'u30',
    ],
    "players_with_fl_elec_demand": [],
    "players_with_fl_hydro_demand": [],
    "players_with_fl_heat_demand": [],
}

BASELINE_CANDIDATES_30 = {
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
      u12: small rooftop solar (0.3x)
      u14: small heat pump (0.3x)
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

    # --- u12: residential prosumer, small rooftop solar (0.3x) ---
    for t in time_periods:
        key = f'renewable_cap_u12_{t}'
        if key in parameters:
            parameters[key] *= 0.3

    # --- u14: heat prosumer, small heat pump (0.3x) ---
    if 'hp_cap_u14' in parameters:
        parameters['hp_cap_u14'] *= 0.3

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


def apply_30player_overrides(parameters, time_periods):
    """
    Apply per-player capacity differentiation for 30-player community.
    First applies the 15-player overrides, then adds u16–u30 overrides.

    Capacity multipliers (0.5–1.5x baseline) to introduce heterogeneity:
      u16: wind 0.8x
      u17: solar 1.3x
      u18: electrolyzer 1.2x, H2 storage 1.2x
      u19: electrolyzer 0.6x
      u20: heat pump 1.0x, heat storage 1.0x
      u21: solar 0.5x, electrolyzer 0.8x
      u22: Pure ESS workaround (solar cap=0, demand=0), ESS 1.5x
      u23: rooftop solar 0.3x (residential prosumer)
      u26: solar canopy 0.4x (refueling station)

    Demand multipliers (match supply growth ~2.18x/1.87x/1.43x for E/G/H):
      Elec: u23(0.5), u24(0.8), u25(0.6), u28(0.9), u29(0.7) → sum 3.5D
      H2:   u26(0.6), u27(0.5), u28(0.5), u29(0.4) → sum 2.0D
      Heat: u23(0.4), u24(0.3), u25(0.3), u30(0.3) → sum 1.3D
    """
    # Apply base 15-player overrides first
    parameters = apply_15player_overrides(parameters, time_periods)

    # --- u16: second wind producer, 0.8x ---
    for t in time_periods:
        key = f'renewable_cap_u16_{t}'
        if key in parameters:
            parameters[key] *= 0.8

    # --- u17: second solar generator, 1.3x ---
    for t in time_periods:
        key = f'renewable_cap_u17_{t}'
        if key in parameters:
            parameters[key] *= 1.3

    # --- u18: electrolyzer + H2 storage, 1.2x ---
    if 'els_cap_u18' in parameters:
        parameters['els_cap_u18'] *= 1.2
    for suffix in ['storage_power_G_u18', 'storage_capacity_G_u18']:
        if suffix in parameters:
            parameters[suffix] *= 1.2
    if 'storage_capacity_G_u18' in parameters:
        parameters['initial_soc_G_u18'] = (
            parameters.get('initial_soc_ratio_G', 0.2) * parameters['storage_capacity_G_u18']
        )

    # --- u19: electrolyzer without storage, 0.6x ---
    if 'els_cap_u19' in parameters:
        parameters['els_cap_u19'] *= 0.6

    # --- u20: heat pump + heat storage, 1.0x (baseline) ---
    # No override needed

    # --- u21: solar-integrated electrolyzer ---
    # Solar 0.5x
    for t in time_periods:
        key = f'renewable_cap_u21_{t}'
        if key in parameters:
            parameters[key] *= 0.5
    # Electrolyzer 0.8x
    if 'els_cap_u21' in parameters:
        parameters['els_cap_u21'] *= 0.8

    # --- u22: Pure ESS workaround (same pattern as u11) ---
    # Zero out renewable production (solar cap=0)
    for t in time_periods:
        parameters[f'renewable_cap_u22_{t}'] = 0.0
    # Export cap = storage power
    storage_power_u22 = parameters.get('storage_power_E_u22', 0)
    parameters['e_E_cap_u22'] = storage_power_u22
    # ESS 1.5x
    for suffix in ['storage_power_E_u22', 'storage_capacity_E_u22']:
        if suffix in parameters:
            parameters[suffix] *= 1.5
    if 'storage_capacity_E_u22' in parameters:
        parameters['initial_soc_E_u22'] = (
            parameters.get('initial_soc_ratio_E', 0.2) * parameters['storage_capacity_E_u22']
        )
    # Update export cap after scaling
    parameters['e_E_cap_u22'] = parameters.get('storage_power_E_u22', 0)
    # Zero out demand and utility
    for t in time_periods:
        parameters[f'd_E_nfl_u22_{t}'] = 0.0
        parameters[f'u_E_u22_{t}'] = 0.0

    # --- u23: residential prosumer, small rooftop solar 0.3x ---
    for t in time_periods:
        key = f'renewable_cap_u23_{t}'
        if key in parameters:
            parameters[key] *= 0.3

    # --- u26: refueling station with solar canopy, solar 0.4x ---
    for t in time_periods:
        key = f'renewable_cap_u26_{t}'
        if key in parameters:
            parameters[key] *= 0.4

    # --- Demand differentiation for u16–u30 consumers ---
    # Match demand growth to supply growth (15p→30p):
    #   Elec supply grows ~2.18x, H2 supply ~1.87x, Heat supply ~1.43x
    #   Without scaling, demand grows 2.67x / 3.0x / 2.33x respectively
    #   Scale new consumers' demand to balance supply-demand ratio.
    #
    # Electricity demand multipliers (new consumers only):
    #   15p effective: u4(1.0), u12(1.0), u15(1.0) = 3.0D
    #   Target 30p total: ~6.5D (2.18x) → new players sum ≈ 3.5D
    elec_demand_scale = {
        'u23': 0.5,   # residential prosumer (small)
        'u24': 0.8,   # residential
        'u25': 0.6,   # residential (smaller)
        'u28': 0.9,   # industrial (larger)
        'u29': 0.7,   # industrial
    }
    # Hydrogen demand multipliers (new consumers only):
    #   15p effective: u5(1.0), u13(1.0) = 2.0D
    #   Target 30p total: ~4.0D (2.0x) → new players sum ≈ 2.0D
    hydro_demand_scale = {
        'u26': 0.6,   # refueling station with solar
        'u27': 0.5,   # small refueling station
        'u28': 0.5,   # industrial
        'u29': 0.4,   # industrial (smaller H2 need)
    }
    # Heat demand multipliers (new consumers only):
    #   15p effective: u6(1.0), u14(1.0), u15(1.0) = 3.0D
    #   Target 30p total: ~4.3D (1.43x) → new players sum ≈ 1.3D
    heat_demand_scale = {
        'u23': 0.4,   # residential
        'u24': 0.3,   # residential (smaller)
        'u25': 0.3,   # residential (smaller)
        'u30': 0.3,   # heat consumer (smaller)
    }

    for t in time_periods:
        for u, scale in elec_demand_scale.items():
            key = f'd_E_nfl_{u}_{t}'
            if key in parameters:
                parameters[key] *= scale
            # Scale utility proportionally to demand
            u_key = f'u_E_{u}_{t}'
            if u_key in parameters:
                parameters[u_key] *= scale
        for u, scale in hydro_demand_scale.items():
            key = f'd_G_nfl_{u}_{t}'
            if key in parameters:
                parameters[key] *= scale
            u_key = f'u_G_{u}_{t}'
            if u_key in parameters:
                parameters[u_key] *= scale
        for u, scale in heat_demand_scale.items():
            key = f'd_H_nfl_{u}_{t}'
            if key in parameters:
                parameters[key] *= scale
            u_key = f'u_H_{u}_{t}'
            if u_key in parameters:
                parameters[u_key] *= scale

    return parameters


# ============================================================
# Main experiment runner
# ============================================================

def run_single_day_pricing(sensitivity_analysis, players, configuration,
                           time_periods, scenario_name, run_id, total_runs,
                           override_fn=None, mipsolver=None):
    """
    Run IP → CHP → CHP Smoothing + core violation check for a single parameter combo.
    (Pricing only — no row generation.)
    Returns a dict with all pricing results for this run.
    """
    print(f"\n--- [Pricing] Run {run_id + 1}/{total_runs} | Day {sensitivity_analysis.get('day', '?')} ---")

    parameters = setup_lem_parameters(
        players, configuration, time_periods, sensitivity_analysis
    )
    if override_fn is not None:
        parameters = override_fn(parameters, time_periods)

    row = {
        'scenario': scenario_name,
        'run_id': run_id,
        **{k: v for k, v in sensitivity_analysis.items()},
    }

    # --- IP ---
    lem = LocalEnergyMarket(players, time_periods, parameters, model_type='mip', mipsolver=mipsolver)
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
    core_comp = CoreComputation(players, 'mip', time_periods, parameters, mipsolver=mipsolver)
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
        model_type='mip', init_sol=results_ip, smoothing=True, mipsolver=mipsolver
    )
    t0 = time.time()
    status_smooth, results_chp_smooth, obj_val_smooth, _ = cg_smooth.solve()
    row['solve_time_chp_smooth'] = time.time() - t0
    row['solve_time_chp_smooth_total'] = row['solve_time_ip'] + row['solve_time_chp_smooth']
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
                          time_limit=3600, override_fn=None, mipsolver=None):
    """
    Run Row Generation (Core computation) for a single parameter combo.
    Returns a dict with rowgen results for this run.
    """
    print(f"\n--- [RowGen] Run {run_id + 1}/{total_runs} | Day {sensitivity_analysis.get('day', '?')} ---")

    parameters = setup_lem_parameters(
        players, configuration, time_periods, sensitivity_analysis
    )
    if override_fn is not None:
        parameters = override_fn(parameters, time_periods)

    row = {
        'scenario': scenario_name,
        'run_id': run_id,
        **{k: v for k, v in sensitivity_analysis.items()},
    }

    core_comp = CoreComputation(players, 'mip', time_periods, parameters, mipsolver=mipsolver)

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
                           time_periods, scenario_name, output_dir='results_15p',
                           override_fn=None, mipsolver=None):
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
            time_periods, scenario_name, i, total_runs,
            override_fn=override_fn, mipsolver=mipsolver,
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
                          time_limit=3600, override_fn=None, mipsolver=None):
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
            time_limit=time_limit, override_fn=override_fn, mipsolver=mipsolver,
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
                   time_limit=3600, override_fn=None, mipsolver=None):
    """
    Run both phases sequentially: pricing first, then row generation.
    Results are saved separately and merged at the end.
    """
    # Phase 1: Pricing
    df_pricing = run_experiment_pricing(
        sensitivity_analysis_candidates, players, configuration,
        time_periods, scenario_name, output_dir, override_fn=override_fn,
        mipsolver=mipsolver,
    )

    # Phase 2: Row Generation
    df_rowgen = run_experiment_rowgen(
        sensitivity_analysis_candidates, players, configuration,
        time_periods, scenario_name, output_dir, time_limit=time_limit,
        override_fn=override_fn, mipsolver=mipsolver,
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
    parser = argparse.ArgumentParser(description='15/30-Player Energy Community Experiment')
    parser.add_argument('--day', type=int, nargs='+', default=list(range(1,2)),
                        help='Day(s) to run (e.g. --day 1 or --day 1 2 3)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results_15p or results_30p)')
    parser.add_argument('--phase', type=str, default='rowgen',
                        choices=['all', 'pricing', 'rowgen'],
                        help='Which phase to run: pricing only, rowgen only, or all (default: pricing)')
    parser.add_argument('--time-limit', type=float, default=36000,
                        help='Row generation time limit in seconds (default: 3600)')
    parser.add_argument('--players', type=int, default=30, choices=[15, 30],
                        help='Number of players: 15 or 30 (default: 15)')
    parser.add_argument('--mipsolver', type=str, default='highs', choices=[None, 'highs'],
                        help='MIP solver for separation problem: None (SCIP, default) or highs (HiGHS)')
    args = parser.parse_args()

    time_periods = list(range(24))

    if args.players == 30:
        players = PLAYERS_30
        configuration = CONFIGURATION_30
        candidates = dict(BASELINE_CANDIDATES_30)
        override_fn = apply_30player_overrides
        scenario_name = '30player_baseline'
        default_output = 'results_30p'
    else:
        players = PLAYERS_15
        configuration = CONFIGURATION_15
        candidates = dict(BASELINE_CANDIDATES_15)
        override_fn = apply_15player_overrides
        scenario_name = '15player_baseline'
        default_output = 'results_15p'

    output_dir = args.output or default_output
    candidates['day'] = args.day

    if args.phase == 'pricing':
        run_experiment_pricing(
            candidates, players, configuration,
            time_periods, scenario_name=scenario_name,
            output_dir=output_dir, override_fn=override_fn,
            mipsolver=args.mipsolver,
        )
    elif args.phase == 'rowgen':
        run_experiment_rowgen(
            candidates, players, configuration,
            time_periods, scenario_name=scenario_name,
            output_dir=output_dir, time_limit=args.time_limit,
            override_fn=override_fn, mipsolver=args.mipsolver,
        )
    else:
        run_experiment(
            candidates, players, configuration,
            time_periods, scenario_name=scenario_name,
            output_dir=output_dir, time_limit=args.time_limit,
            override_fn=override_fn, mipsolver=args.mipsolver,
        )

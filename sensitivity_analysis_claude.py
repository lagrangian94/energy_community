"""
sensitivity_analysis.py 수정 가이드
====================================

현재 코드 구조를 최대한 유지하면서, 5.3 시나리오를 추가하는 방법.

핵심 변경점:
1. sensitivity_analysis_candidates를 시나리오별로 분리
2. configuration도 시나리오에 따라 바꿀 수 있게 함수화
3. 결과 저장 경로를 시나리오별로 구분
"""

# =============================================================
# [변경 1] 기존 단일 candidates dict → 시나리오별 dict
# =============================================================

# 기존 코드:
#   sensitivity_analysis_candidates = { ... 하나의 dict ... }
#
# 변경: BASELINE을 기본으로 하고, 시나리오별 override만 정의

import sys
import os

# --- Baseline (5.2에서 쓰던 설정 그대로) ---
BASELINE_CANDIDATES = {
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
    'base_h2_price_eur': [5000/1500],
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
    'storage_capacity_ratio_G': [0.0],
    'storage_capacity_ratio_H': [0.0],
    'initial_soc_ratio_E': [0.2],
    'initial_soc_ratio_G': [0.2],
    'initial_soc_ratio_H': [0.2],
    'day': list(range(1, 32)),  # 31일 전체
}


def make_scenario(overrides: dict) -> dict:
    """baseline을 복사하고 특정 파라미터만 override"""
    scenario = {k: list(v) for k, v in BASELINE_CANDIDATES.items()}
    for k, v in overrides.items():
        scenario[k] = v if isinstance(v, list) else [v]
    return scenario


# =============================================================
# [변경 2] 시나리오 정의
# =============================================================

SCENARIOS = {
    # --- 5.2 baseline (비교 기준) ---
    'baseline': make_scenario({}),

    # --- 5.3.1 Low Hydrogen Margin ---
    # (a) 수소 싸짐 + 전기 비쌈 (import-export spread 큼)
    'low_h2_margin': make_scenario({
        'base_h2_price_eur': [2.0],       # 미래 그린수소
        'import_factor': [3.0],            # 높은 spread
    }),
    # 참고: baseline과 비교하려면 baseline도 같이 돌려야 함
    # 아니면 baseline 결과를 재사용

    # --- 5.3.2 Storage Availability ---
    # (b) hydro + heat storage ON
    'full_storage': make_scenario({
        'storage_capacity_ratio_G': [3.0],  # hydro storage ON
        'storage_capacity_ratio_H': [3.0],  # heat storage ON
    }),

    # --- 5.3.3 Community Size ---
    # (c) 500, 700(baseline), 1000 가구
    'community_size': make_scenario({
        'num_households': [350, 1000],
    }),

    # # --- 5.3.4 Export Capacity Constraint ---
    # # (d) export cap을 줄임
    # '5.3.4_export_cap': make_scenario({
    #     'e_E_cap_ratio': [0.2, 1.0],
    #     'e_G_cap_ratio': [0.2, 1.0],
    #     'e_H_cap_ratio': [0.2, 1.0],
    # }),
}

# !! 5.3.4 주의 !!
# 위처럼 하면 e_E, e_G, e_H가 독립적으로 조합됨 (3^3 = 27 조합)
# 실제로는 세 개를 동시에 같은 비율로 바꾸고 싶을 것.
# 그러면 아래처럼 수동으로:

# SCENARIOS['export_cap'] = []  # 리스트로 바꿔서 수동 지정
# for ratio in [0.2]:
#     SCENARIOS['export_cap'].append(
#         make_scenario({
#             'e_E_cap_ratio': [ratio],
#             'e_G_cap_ratio': [ratio],
#             'e_H_cap_ratio': [ratio],
#         })
#     )
# → 이 경우 아래 runner 코드에서 리스트 처리 필요 (아래 참조)


# =============================================================
# [변경 3] 메인 실행부 수정
# =============================================================

"""
기존 코드의 for loop를 함수로 감싸고, 시나리오 선택 인자를 추가.

기존:
    for i, values in enumerate(itertools.product(*param_values)):
        sensitivity_analysis = dict(zip(param_names, values))
        parameters = setup_lem_parameters(...)
        ...

변경:
"""

from data_generator import setup_lem_parameters
from core import CoreComputation
from compact_utility import LocalEnergyMarket
from chp import ColumnGenerationSolver
import numpy as np
import pandas as pd
import time
import itertools
import argparse


def run_experiment(sensitivity_analysis_candidates, players, configuration,
                   time_periods, scenario_name, output_dir='results'):
    """
    하나의 시나리오(candidates dict)에 대해 전 조합 실행.
    기존 for loop와 동일하되, 결과를 scenario별로 저장.
    """
    os.makedirs(output_dir, exist_ok=True)

    param_names = list(sensitivity_analysis_candidates.keys())
    param_values = [sensitivity_analysis_candidates[name] for name in param_names]

    results_summary = []
    ip, chp = True, True
    brute_force = False  # 5.3에서는 separation problem으로 충분

    total_runs = 1
    for v in param_values:
        total_runs *= len(v)
    print(f"\nScenario: {scenario_name}")
    print(f"Total runs: {total_runs}")

    outpath = os.path.join(output_dir, f'{scenario_name}_fixed.csv')  # ← 여기
    
    for i, values in enumerate(itertools.product(*param_values)):
        sensitivity_analysis = dict(zip(param_names, values))
        print(f"\n--- Run {i+1}/{total_runs} | Day {sensitivity_analysis.get('day', '?')} ---")

        parameters = setup_lem_parameters(
            players, configuration, time_periods, sensitivity_analysis
        )

        row = {
            'scenario': scenario_name,
            'run_id': i,
            **{k: v for k, v in sensitivity_analysis.items()},  # 모든 파라미터 기록
        }

        # --- IP ---
        if ip:
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

            # IP core violation
            core_comp = CoreComputation(players, 'mip', time_periods, parameters)
            cost_ip = {u: -1 * profit_ip[u] for u in players}
            coalition_ip, violation_ip, isimp_ip = core_comp.measure_stability_violation(cost_ip)
            row['violation_ip'] = violation_ip
            row['blocking_coalition_ip'] = str(coalition_ip) if violation_ip > 1e-6 else ''
            row['isimp_ip'] = isimp_ip

        # --- CHP ---
        if chp:
            cg = ColumnGenerationSolver(players, time_periods, parameters, model_type='mip', init_sol=results_ip)
            t0 = time.time()
            status, results_chp, obj_val, solution_by_player = cg.solve()
            row['solve_time_chp'] = time.time() - t0
            welfare_chp = cg.master.model.getObjVal()
            print(f"Time taken: {row['solve_time_chp']:.2f} seconds")
            if status == "optimal":
                print("\n" + "="*80)
                print("COLUMN GENERATION - SUCCESSFUL")
                print("="*80)
                print(f"Optimal objective: {obj_val:.2f} EUR")
                
                # Print convex hull prices
                if 'convex_hull_prices' in results_chp:
                    print("\n" + "="*80)
                    print("CONVEX HULL PRICES (Community Balance Shadow Prices)")
                    print("="*80)
                    community_prices_chp = results_chp['convex_hull_prices']
                    if "capacity_prices" in results_chp:
                        community_prices_chp = community_prices_chp | results_chp['capacity_prices']
                    # Perform synergy analysis
                    print("\n" + "="*80)
                    print("PERFORMING SYNERGY ANALYSIS")
                    print("="*80)
                    synergy_results = cg.analyze_synergy_with_convex_hull_prices(results_ip, obj_val, community_prices_chp)
                    profit_chp = {u: synergy_results['community_profits'][u]['net_profit'] for u in players}
            else:
                print(f"Column generation failed with status: {status}")
                profit_chp = {u: np.nan for u in players}
            for u in players:
                row[f'profit_chp_{u}'] = profit_chp[u]

            if status == "optimal":
                cost_chp = {u: -1 * profit_chp[u] for u in players}
                coalition_chp, violation_chp, isimp_chp = core_comp.measure_stability_violation(cost_chp)
                row['violation_chp'] = violation_chp
                row['blocking_coalition_chp'] = str(coalition_chp) if violation_chp > 1e-6 else ''
                row['isimp_chp'] = isimp_chp
                # CHP community prices 저장 (분석용)
                for carrier in ['electricity', 'heat', 'hydrogen']:
                    if carrier in community_prices_chp:
                        for t in time_periods:
                            row[f'chp_price_{carrier}_{t}'] = community_prices_chp[carrier][t]
            else:
                print(f"Column generation failed with status: {status}")
                row['violation_chp'] = np.nan
                row['blocking_coalition_chp'] = np.nan
                row['isimp_chp'] = np.nan

            if ip and chp:
                if (violation_ip > 1e-6) and (violation_chp > 1e-6):
                    core_comp = CoreComputation(players, 'mip', time_periods, parameters)
                    core_bf, success = core_comp.compute_core_brute_force()
                    if success:
                        row['violation_bf'] = 0.0
                        row['blocking_coalition_bf'] = []
                        row['isimp_bf'] = True
                    else:
                        row['violation_bf'] = core_bf
                        row['blocking_coalition_bf'] = np.nan
        # 로그
        flag = " *** IP VIOLATED ***" if row.get('violation_ip', 0) > 1e-6 else ""
        print(f"  IP: {row.get('violation_ip', 0):.4f}{flag} | "
              f"CHP: {row.get('violation_chp', 0):.4f} | "
              f"t_IP={row.get('solve_time_ip', 0):.1f}s t_CHP={row.get('solve_time_chp', 0):.1f}s")

        results_summary.append(row)

        # 매 run마다 중간 저장
        df = pd.DataFrame(results_summary)
        df.to_csv(outpath, index=False)
    # # 저장
    # df = pd.DataFrame(results_summary)
    # outpath = os.path.join(output_dir, f'{scenario_name}.csv')
    # df.to_csv(outpath, index=False)
    # print(f"\nSaved: {outpath}")

    # Summary 출력
    print_scenario_summary(df, scenario_name)
    return df


def print_scenario_summary(df, scenario_name):
    """시나리오 결과 요약"""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {scenario_name}")
    print(f"{'='*60}")
    total = len(df)
    n_ip = (df['violation_ip'] > 1e-6).sum()
    n_chp = (df['violation_chp'] > 1e-6).sum()
    print(f"  Total runs: {total}")
    print(f"  IP violations:  {n_ip}/{total} ({100*n_ip/total:.0f}%)")
    print(f"  CHP violations: {n_chp}/{total} ({100*n_chp/total:.0f}%)")
    if n_ip > 0:
        avg_mag = df.loc[df['violation_ip'] > 1e-6, 'violation_ip'].mean()
        max_mag = df.loc[df['violation_ip'] > 1e-6, 'violation_ip'].max()
        print(f"  Avg IP violation magnitude: {avg_mag:.4f}")
        print(f"  Max IP violation magnitude: {max_mag:.4f}")

    # 서브그룹별 (5.3.3은 num_households별, 5.3.4는 cap_ratio별)
    for col in ['num_households', 'e_E_cap_ratio', 'base_h2_price_eur', 'import_factor']:
        if col in df.columns and df[col].nunique() > 1:
            print(f"\n  By {col}:")
            for val, grp in df.groupby(col):
                n = len(grp)
                n_v = (grp['violation_ip'] > 1e-6).sum()
                print(f"    {col}={val}: IP {n_v}/{n}, CHP {(grp['violation_chp']>1e-6).sum()}/{n}")


# =============================================================
# [변경 4] MAIN - argparse로 시나리오 선택
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Section 5.3 Scenario Experiments')
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['baseline', 'low_h2_margin',
                                 'full_storage', 'community_size',
                                 'export_cap', 'all'],
                        help='시나리오 선택')
    parser.add_argument('--output', type=str, default='results_53',
                        help='결과 저장 디렉토리')
    args = parser.parse_args()

    # --- Player/Configuration은 기존 그대로 ---
    model_type = 'mip'
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))
    configuration = {
        "players_with_renewables": ['u1'],
        "players_with_solar": [],
        "players_with_wind": ['u1'],
        "players_with_electrolyzers": ['u2'],
        "players_with_heatpumps": ['u3'],
        "players_with_elec_storage": ['u1'],
        "players_with_hydro_storage": ['u2'],
        "players_with_heat_storage": ['u3'],
        "players_with_nfl_elec_demand": ['u4'],
        "players_with_nfl_hydro_demand": ['u5'],
        "players_with_nfl_heat_demand": ['u6'],
        "players_with_fl_elec_demand": ['u2', 'u3'],
        "players_with_fl_hydro_demand": [],
        "players_with_fl_heat_demand": [],
    }

    # --- 5.3.4 특수 처리: cap ratio를 동시에 맞추기 ---
    if args.scenario == 'export_cap':
        for ratio in [0.2]:
            sub_name = f'export_cap_{int(ratio*100):03d}'
            candidates = make_scenario({
                'e_E_cap_ratio': [ratio],
                'e_G_cap_ratio': [ratio],
                'e_H_cap_ratio': [ratio],
            })
            run_experiment(candidates, players, configuration,
                           time_periods, sub_name, args.output)

    elif args.scenario == 'all':
        for name in ['low_h2_margin',
                      'full_storage', 'community_size']:
            if name == 'low_h2_margin':
                continue
            candidates = SCENARIOS[name]
            run_experiment(candidates, players, configuration,
                           time_periods, name, args.output)
        # 5.3.4는 별도 처리
        for ratio in [0.2]:
            sub_name = f'export_cap_{int(ratio*100):03d}'
            candidates = make_scenario({
                'e_E_cap_ratio': [ratio],
                'e_G_cap_ratio': [ratio],
                'e_H_cap_ratio': [ratio],
            })
            run_experiment(candidates, players, configuration,
                           time_periods, sub_name, args.output)

    else:
        candidates = SCENARIOS[args.scenario]
        run_experiment(candidates, players, configuration,
                       time_periods, args.scenario, args.output)


# =============================================================
# 실행 예시:
#   python sensitivity_analysis.py --scenario baseline --output results_53
#   python sensitivity_analysis.py --scenario 5.3.1_low_h2_margin
#   python sensitivity_analysis.py --scenario all
#
# 총 예상 runs:
#   baseline:            31 days × 1 config = 31
#   5.3.1_low_h2_margin: 31 days × 1 config = 31
#   5.3.2_full_storage:  31 days × 1 config = 31
#   5.3.3_community_size:31 days × 3 configs = 93
#   5.3.4_export_cap:    31 days × 3 configs = 93
#   ────────────────────────────────────────────
#   Total:                                    279 runs
# =============================================================
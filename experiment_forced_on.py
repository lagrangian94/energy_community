"""
experiment_forced_on.py
========================
수전해 PWL 비선형성(non-convexity)만의 효과를 분리하기 위한 실험.

설정:
  - baseline 시나리오, c_su_G = 0.0 (startup cost 제거)
  - z_off_G >= 1.0 initial state 제약 삭제
  - z_on_G >= 1.0 모든 period에 강제 (항상 ON)
  - startup/shutdown/minimum_down_time 등 commitment 관련 제약 전부 삭제

이렇게 하면 commitment(ON/OFF) decision이 없는 상태에서
수전해 PWL efficiency approximation의 non-convexity가
IP pricing의 core stability에 미치는 영향만 따로 볼 수 있음.

결과 (2026-02-28):
  26/31일에서 violation_ip > 0 발생.
  commitment 유연성이 오히려 non-convexity를 완화하고 있었고,
  수전해가 항상 켜져 있으면 PWL의 non-convexity가 크게 드러남.


 결과가 나왔습니다.                                     
                                                                                                                                                                       
  결과: z_on_G=1 (항상 ON), c_su_G=0 → 수전해 비선형성만의 효과                                                                                                        
  
  26/31일에서 violation_ip > 0 발생.                                                                                                                                   
                                                                                                                                                                     
  ┌─────────────────────────┬───────────────────────────┐
  │    Violation 없는 날    │ Day 6, 8, 9, 15, 30 (5일) │
  ├─────────────────────────┼───────────────────────────┤
  │ Violation 있는 날       │ 26일 (84%)                │
  ├─────────────────────────┼───────────────────────────┤
  │ 최대 violation          │ Day 28: 28.09             │
  ├─────────────────────────┼───────────────────────────┤
  │ 평균 violation (양수만) │ ~14.5                     │
  └─────────────────────────┴───────────────────────────┘

  Commitment(ON/OFF)를 완전히 제거하고 수전해가 항상 ON인 상태에서도, PWL efficiency 비선형성만으로 31일 중 26일에서 IP pricing이 core를 벗어납니다. 이전 c_su_G=0 +
  자유 commitment 실험에서는 1/31일만 violated였던 것과 비교하면, commitment 유연성이 오히려 non-convexity를 완화하고 있었고, 수전해가 항상 켜져 있으면 PWL
  approximation의 non-convexity가 훨씬 크게 드러난다는 걸 보여줍니다
"""

import sensitivity_analysis_claude as sa
from data_generator import setup_lem_parameters
from compact_utility import LocalEnergyMarket, solve_and_extract_results_highs
from core import CoreComputation
import numpy as np
import pandas as pd
import tempfile
import os
import gc
import argparse


def modify_els_constraints(lem):
    """수전해 commitment 제약 삭제 + z_on_G >= 1 강제"""
    u = 'u2'
    keys_to_del = [k for k in lem.electrolyzer_cons.keys()
                   if isinstance(k, str) and any(w in k for w in
                   ['startup', 'shut_down', 'initial', 'minimum_down', 'forbid'])]
    for k in keys_to_del:
        lem.model.delCons(lem.electrolyzer_cons[k])
    for t in lem.time_periods:
        lem.model.addCons(lem.z_on_G[u, t] >= 1.0, name=f'force_on_{u}_{t}')


def run_single_day(day, players, time_periods, configuration, baseline):
    """단일 day 실험 실행. (violation_ip, row_dict) 반환."""
    import highspy

    sa_params = dict(baseline)
    sa_params['c_su_G'] = 0.0
    sa_params['day'] = day
    params = setup_lem_parameters(players, configuration, time_periods,
                                  sensitivity_analysis=sa_params)

    # --- 1) IP solve ---
    lem = LocalEnergyMarket(players, time_periods, params,
                            model_type='mip', mipsolver='highs')
    modify_els_constraints(lem)

    mps_path = tempfile.mktemp(suffix='.mps')
    lem.model.writeProblem(mps_path)
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.readModel(mps_path)
    h.run()
    os.remove(mps_path)
    status_ip, results_ip = solve_and_extract_results_highs(lem.model, h)
    del h

    if status_ip != 'optimal':
        del lem
        gc.collect()
        return None

    # --- 2) LP with fixed binaries (same constraint modifications) ---
    binary_values = {}
    for u in ['u2']:
        for t in time_periods:
            for vn in ['z_su_G', 'z_on_G', 'z_off_G', 'z_sb_G', 'z_sd_G']:
                if (u, t) in results_ip.get(vn, {}):
                    binary_values[(vn, u, t)] = results_ip[vn][(u, t)]
    for u in ['u3']:
        for t in time_periods:
            for vn in ['z_su_H', 'z_on_H', 'z_sd_H']:
                if (u, t) in results_ip.get(vn, {}):
                    binary_values[(vn, u, t)] = results_ip[vn][(u, t)]

    from pyscipopt import SCIP_PARAMSETTING
    lp_m = LocalEnergyMarket(players, time_periods, params,
                             model_type='mip_fix_binaries',
                             binary_values=binary_values)
    modify_els_constraints(lp_m)
    lp_m.model.setPresolve(SCIP_PARAMSETTING.OFF)
    lp_m.model.setHeuristics(SCIP_PARAMSETTING.OFF)
    lp_m.model.disablePropagation()
    lp_m.model.setSeparating(SCIP_PARAMSETTING.OFF)
    lp_m.model.optimize()

    if lp_m.model.getStatus() != 'optimal':
        del lem, lp_m
        gc.collect()
        return None

    # --- 3) Extract prices ---
    prices = {'electricity': {}, 'heat': {}, 'hydrogen': {}}
    for t in time_periods:
        for carrier, cd, prefix in [
            ('electricity', lp_m.community_elec_balance_cons, 'community_elec_balance_'),
            ('heat', lp_m.community_heat_balance_cons, 'community_heat_balance_'),
            ('hydrogen', lp_m.community_hydro_balance_cons, 'community_hydro_balance_'),
        ]:
            c = cd[f'{prefix}{t}']
            prices[carrier][t] = np.abs(
                lp_m.model.getDualsolLinear(lp_m.model.getTransformedCons(c))
            )

    # --- 4) Profit & violation ---
    player_profits, _ = lem.calculate_player_profits_with_community_prices(
        results_ip, prices
    )
    profit_ip = {u: player_profits[u]['net_profit'] for u in players}

    core_comp = CoreComputation(players, 'mip', time_periods, params,
                                mipsolver='highs')
    cost_ip = {u: -1 * profit_ip[u] for u in players}
    coalition, violation_ip, isimp = core_comp.measure_stability_violation(cost_ip)

    row = {
        'day': day,
        'violation_ip': violation_ip,
        'blocking_coalition': str(coalition) if violation_ip > 1e-4 else '',
    }
    for u in players:
        row[f'profit_ip_{u}'] = profit_ip[u]

    del lem, lp_m, core_comp
    gc.collect()
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Forced-ON electrolyzer experiment (PWL non-convexity only)')
    parser.add_argument('--day', type=int, nargs='+', default=list(range(1, 32)),
                        help='Day(s) to run (default: 1-31)')
    parser.add_argument('--output', type=str, default='results_forced_on.csv',
                        help='Output CSV path')
    args = parser.parse_args()

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
    baseline = {k: v[0] for k, v in sa.BASELINE_CANDIDATES.items()
                if k not in ['c_su_G', 'day']}

    results_all = []
    for day in args.day:
        row = run_single_day(day, players, time_periods, configuration, baseline)
        if row is None:
            print(f'Day {day:2d}: solve failed')
            results_all.append({'day': day, 'violation_ip': float('nan')})
        else:
            flag = ' ***' if row['violation_ip'] > 1e-4 else ''
            print(f'Day {day:2d}: violation_ip = {row["violation_ip"]:>10.6f}{flag}')
            results_all.append(row)

    df = pd.DataFrame(results_all)
    df.to_csv(args.output, index=False)
    print(f'\nSaved: {args.output}')

    violated = df[df['violation_ip'] > 1e-4]
    print(f'Total violated days: {len(violated)}/{len(df)}')
    for _, r in violated.iterrows():
        print(f'  Day {int(r["day"])}: {r["violation_ip"]:.6f}')

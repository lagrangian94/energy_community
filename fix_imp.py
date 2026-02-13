"""
Re-measure violation for non-imputation rows.
violation=inf인 row에 대해 실제 violation magnitude를 측정하여 덮어씀.

Usage:
    python fix_inf_violations.py --input results_53/low_h2_margin.csv
"""

import pandas as pd
import numpy as np
import argparse
import time
from data_generator import setup_lem_parameters
from core import CoreComputation

# Player/configuration (sensitivity_analysis.py와 동일)
PLAYERS = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
TIME_PERIODS = list(range(24))
CONFIGURATION = {
    "players_with_renewables": ['u1'],
    "players_with_solar": [],
    "players_with_wind": ['u1'],
    "players_with_electrolyzers": ['u2'],
    "players_with_heatpumps": ['u3'],
    "players_with_elec_storage": ['u1'],
    "players_with_hydro_storage": ['u2'],
    "players_with_heat_storage": ['u3'],
    "players_with_nfl_elec_demand": ['u1'],
    "players_with_nfl_hydro_demand": ['u2'],
    "players_with_nfl_heat_demand": ['u3'],
    "players_with_fl_elec_demand": ['u2', 'u3'],
    "players_with_fl_hydro_demand": [],
    "players_with_fl_heat_demand": [],
}

# sensitivity_analysis dict를 row에서 복원하기 위한 칼럼 목록
SA_COLUMNS = [
    'use_korean_price', 'use_tou_elec', 'import_factor', 'month',
    'hp_cap', 'els_cap', 'num_households', 'nu_cop',
    'c_su_G', 'c_su_H', 'base_h2_price_eur',
    'e_E_cap_ratio', 'e_H_cap_ratio', 'e_G_cap_ratio',
    'eff_type', 'segments', 'peak_penalty_ratio',
    'wind_el_ratio', 'solar_el_ratio',
    'storage_power_ratio_E', 'storage_power_ratio_G', 'storage_power_ratio_H',
    'storage_capacity_ratio_E', 'storage_capacity_ratio_G', 'storage_capacity_ratio_H',
    'initial_soc_ratio_E', 'initial_soc_ratio_G', 'initial_soc_ratio_H',
    'day',
]


def row_to_sa(row):
    """CSV row → sensitivity_analysis dict 복원"""
    sa = {}
    for col in SA_COLUMNS:
        val = row[col]
        # bool 처리 (csv에서 문자열로 읽힐 수 있음)
        if col in ('use_korean_price', 'use_tou_elec'):
            if isinstance(val, str):
                val = val.strip().lower() == 'true'
        # int 처리
        elif col in ('month', 'num_households', 'eff_type', 'segments', 'day'):
            val = int(val)
        else:
            val = float(val)
        sa[col] = val
    return sa


def fix_violations(df, pricing='ip'):
    """
    isimp_{pricing}=False인 row에 대해 violation 재측정

    Args:
        df: DataFrame
        pricing: 'ip' or 'chp'
    Returns:
        수정된 row 수
    """
    isimp_col = f'isimp_{pricing}'
    violation_col = f'violation_{pricing}'
    coalition_col = f'blocking_coalition_{pricing}'
    profit_prefix = f'profit_{pricing}_'

    # imputation 아닌 row 찾기
    mask = df[isimp_col] == False
    targets = df[mask].index.tolist()

    if len(targets) == 0:
        print(f"  No non-imputation rows for {pricing}. Skipping.")
        return 0

    print(f"  Found {len(targets)} non-imputation rows for {pricing}: "
          f"days {df.loc[targets, 'day'].tolist()}")

    fixed = 0
    for idx in targets:
        row = df.loc[idx]
        day = int(row['day'])
        print(f"\n  --- Re-measuring day {day} ({pricing}) ---")

        # 1. sensitivity_analysis dict 복원
        sa = row_to_sa(row)

        # 2. parameters 생성
        parameters = setup_lem_parameters(
            PLAYERS, CONFIGURATION, TIME_PERIODS, sa
        )

        # 3. profit → cost
        cost = {}
        for u in PLAYERS:
            profit_val = row[f'{profit_prefix}{u}']
            cost[u] = -1 * profit_val

        # 4. CoreComputation + measure_stability_violation
        core_comp = CoreComputation(PLAYERS, 'mip', TIME_PERIODS, parameters)
        t0 = time.time()
        coalition, violation, is_imp = core_comp.measure_stability_violation(cost)
        elapsed = time.time() - t0

        print(f"    violation: {violation:.6f}")
        print(f"    coalition: {coalition}")
        print(f"    is_imputation: {is_imp}")
        print(f"    time: {elapsed:.1f}s")

        # 5. 덮어쓰기
        df.at[idx, violation_col] = violation
        df.at[idx, coalition_col] = str(coalition) if coalition else ''
        fixed += 1

    return fixed

def compute_core_for_both_failed(df):
    """
    isimp_ip=False AND isimp_chp=False인 row에 대해
    row generation으로 core allocation을 직접 계산.

    새 칼럼 추가:
        - profit_core_u1 ~ u6
        - violation_core
        - blocking_coalition_core
        - isimp_core
        - solve_time_core
    """
    mask = (df['isimp_ip'] == False) & (df['isimp_chp'] == False)
    targets = df[mask].index.tolist()

    if len(targets) == 0:
        print("  No rows where both IP and CHP fail imputation. Skipping.")
        return 0

    print(f"  Found {len(targets)} rows where both fail: "
          f"days {df.loc[targets, 'day'].tolist()}")

    # 새 칼럼 초기화
    for u in PLAYERS:
        if f'profit_rowgen_{u}' not in df.columns:
            df[f'profit_rowgen_{u}'] = np.nan
    for col in ['violation_rowgen', 'blocking_coalition_rowgen', 'isimp_rowgen', 'solve_time_rowgen']:
        if col not in df.columns:
            df[col] = np.nan

    computed = 0
    for idx in targets:
        row = df.loc[idx]
        day = int(row['day'])
        print(f"\n  --- Computing core via row generation: day {day} ---")

        # 1. parameters 복원
        sa = row_to_sa(row)
        parameters = setup_lem_parameters(
            PLAYERS, CONFIGURATION, TIME_PERIODS, sa
        )

        # 2. Row generation으로 core 계산
        core_comp = CoreComputation(PLAYERS, 'mip', TIME_PERIODS, parameters)
        t0 = time.time()

        core_rowgen, success = core_comp.compute_core_brute_force()
        solve_time = time.time() - t0
        # core_rowgen, success = core_comp.compute_core(
        #     max_iterations=int(1e+8),
        #     tolerance=1e-6
        # )
        # solve_time = time.time() - t0

        # 3. Core allocation의 violation 측정
        if success:
            violation = 0.0
            coalition = []
            is_imp = True
            for u in PLAYERS:
                df.at[idx, f'profit_rowgen_{u}'] = -1 * core_rowgen[u]  # cost → profit
                df.at[idx, 'violation_rowgen'] = violation
                df.at[idx, 'blocking_coalition_rowgen'] = str(coalition) if coalition else ''
                df.at[idx, 'isimp_rowgen'] = is_imp
                df.at[idx, 'solve_time_rowgen'] = solve_time
        else:
            violation = core_rowgen
            coalition = np.nan
            is_imp = np.nan
            for u in PLAYERS:
                df.at[idx, f'profit_rowgen_{u}'] = np.nan
                df.at[idx, 'violation_rowgen'] = violation
                df.at[idx, 'blocking_coalition_rowgen'] = np.nan
                df.at[idx, 'isimp_rowgen'] = np.nan
                df.at[idx, 'solve_time_rowgen'] = solve_time
                

        print(f"    core allocation: {core_rowgen}")
        print(f"    violation: {violation:.6f}")
        print(f"    is_imputation: {is_imp}")
        print(f"    time: {solve_time:.1f}s")

        computed += 1


    return computed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results_53/low_h2_margin.csv',
                        help='입력 CSV 경로')
    parser.add_argument('--output', type=str, default='results_53/low_h2_margin.csv',
                        help='출력 CSV 경로 (기본: 입력 파일 덮어쓰기)')
    args = parser.parse_args()

    outpath = args.output or args.input

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Total rows: {len(df)}")

    # IP
    print(f"\n{'='*50}")
    print("Fixing IP violations...")
    print(f"{'='*50}")
    n_ip = fix_violations(df, 'ip')

    # CHP
    print(f"\n{'='*50}")
    print("Fixing CHP violations...")
    print(f"{'='*50}")
    n_chp = fix_violations(df, 'chp')
    print(f"Fixed: {n_ip} IP rows, {n_chp} CHP rows")

    # # Row generation
    # print(f"\n{'='*50}")
    # print("Computing core via row generation...")
    # print(f"{'='*50}")
    # n_rowgen = compute_core_for_both_failed(df)
    # print(f"Fixed: {n_rowgen} rows")

    # 저장
    df.to_csv(outpath, index=False)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
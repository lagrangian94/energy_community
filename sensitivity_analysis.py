from data_generator import setup_lem_parameters
from core import CoreComputation
from compact import LocalEnergyMarket
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from chp import ColumnGenerationSolver
import pandas as pd
import time
import sys
sys.path.append('/mnt/project')


if __name__ == "__main__":
    model_type = 'lp'
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6','u7']
    # players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']
    time_periods = list(range(24))  # 24 hours
    configuration = {}
    configuration["players_with_renewables"] = ['u1', 'u7']
    configuration["players_with_solar"] = ['u7']
    configuration["players_with_wind"] = ['u1']
    configuration["players_with_electrolyzers"] = ['u2'] #+ ['u7']
    configuration["players_with_heatpumps"] = ['u3']
    configuration["players_with_elec_storage"] = ['u1','u7']
    configuration["players_with_hydro_storage"] = ['u2'] #+ ['u7']
    configuration["players_with_heat_storage"] = ['u3']
    configuration["players_with_nfl_elec_demand"] = ['u4']
    configuration["players_with_nfl_hydro_demand"] = ['u5'] #+ ['u8']
    configuration["players_with_nfl_heat_demand"] = ['u6']
    configuration["players_with_fl_elec_demand"] = ['u2','u3']# + ['u7']
    configuration["players_with_fl_hydro_demand"] = []
    configuration["players_with_fl_heat_demand"] = []


    sensitivity_analysis_candidates = {
        'use_korean_price': [True],#,False],
        'use_tou': [True],#,, False],
        'month': [1],
        'storage_capacity_E': [0.5, 1.0, 1.5],
        'storage_capacity_G': [50, 100, 150],
        'storage_capacity_heat': [0.40, 0.60, 0.80],
        'hp_cap': [0.6, 0.8, 1.0],
        'els_cap': [0.5, 1.0, 1.5],
        'res_cap': [1,2],
        'num_households': [50, 75, 100],
        'nu_cop': [3.0, 3.28, 3.5],
        'c_su_G': [50, 75, 100],
        'c_su_H': [40, 50, 60],
        'base_h2_price_eur': [2.1*0.75, 2.1, 2.1*2],
        'e_E_cap': [0.5, 1.0, 1.5, 2.0],
        'e_H_cap': [0.5, 1.0, 1.5, 2.0]
    }


    # INSERT_YOUR_CODE
    import itertools

    # Prepare all param names and value list for combinations
    param_names = list(sensitivity_analysis_candidates.keys())
    param_values = [sensitivity_analysis_candidates[name] for name in param_names]
    results_summary = []
    violation_lp = []

    for i, values in enumerate(itertools.product(*param_values)):
        # Enum a single config
        sensitivity_analysis = dict(zip(param_names, values))
        # Generate parameters under this config
        parameters = setup_lem_parameters(players, configuration, time_periods, sensitivity_analysis)
        # Build and solve LP
        lem_lp = LocalEnergyMarket(players, time_periods, parameters, model_type=model_type)
        time_start = time.time()
        status_lp, results_complete_lp, revenue_analysis_lp, community_prices_lp = lem_lp.solve_complete_model(analyze_revenue=False)
        time_end = time.time()
        time_lp = time_end - time_start
        if status_lp != "optimal":
            result_row = {
            'sensitivity_analysis': sensitivity_analysis,
            'violation_lp': np.inf,
            'solve_time': time_lp,
            'profit_lp': np.inf
            }
            results_summary.append(result_row)
            continue
        
        # Compute profit and violation
        player_profits_lp, prices_lp = lem_lp.calculate_player_profits_with_community_prices(results_complete_lp, community_prices_lp)
        comparison_results_lp = lem_lp.compare_individual_vs_community_profits(
        players, 
        lem_lp.model_type,
        time_periods, 
        parameters,
        player_profits_lp
        )
        profit_lp = {u: player_profits_lp[u]["net_profit"] for u in players}
        cost_lp = {u: -1*price for (u,price) in profit_lp.items()}
        core_comp = CoreComputation(
            players=players,
            time_periods=time_periods,
            model_type=model_type,
            parameters=parameters
        )
        violation_lp = core_comp.measure_stability_violation(cost_lp)
        result_row = {
            'sensitivity_analysis': sensitivity_analysis,
            'violation_lp': violation_lp,
            'solve_time': time_lp,
            'profit_lp': profit_lp
        }
        if violation_lp > 1:
            print('debug')
        results_summary.append(result_row)
        # print('='*60)
        # print('Current Sensitivity Case:', sensitivity_analysis)
        # print(f'Core Violation: {violation_lp:.6f}, LP Solve Time: {time_lp:.2f}')
        # print('='*60)

        # Optionally: Save as DataFrame
        df = pd.DataFrame(results_summary)
        df.to_csv('sensitivity_analysis_results.csv', index=False)

        if i > 100:
            break

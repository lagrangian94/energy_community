from data_generator import setup_lem_parameters
from core import CoreComputation
from compact_utility import LocalEnergyMarket
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from chp import ColumnGenerationSolver
import pandas as pd
import time
import sys
sys.path.append('/mnt/project')


if __name__ == "__main__":
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']#,'u7']
    # players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']
    time_periods = list(range(24))  # 24 hours
    configuration = {}
    configuration["players_with_renewables"] = ['u1']
    configuration["players_with_solar"] = []
    configuration["players_with_wind"] = ['u1']
    configuration["players_with_electrolyzers"] = ['u2'] #+ ['u7']
    configuration["players_with_heatpumps"] = ['u3']
    configuration["players_with_elec_storage"] = ['u1'] #,'u7']
    configuration["players_with_hydro_storage"] = ['u2'] #+ ['u7']
    configuration["players_with_heat_storage"] = ['u3']
    configuration["players_with_nfl_elec_demand"] = ['u4']
    configuration["players_with_nfl_hydro_demand"] = ['u5'] #+ ['u8']
    configuration["players_with_nfl_heat_demand"] = ['u6']
    configuration["players_with_fl_elec_demand"] = ['u2','u3']# + ['u7']
    configuration["players_with_fl_hydro_demand"] = []
    configuration["players_with_fl_heat_demand"] = []
    parameters = setup_lem_parameters(players, configuration, time_periods)
    parameters['eff_type'], parameters['El']['eff_type'] = 2, 2
    # parameters["c_els_u7"] = parameters["c_els"]*2
    # parameters["c_su_u7"] = parameters["c_su_G"]*2
    # Create and solve model with Restricted Pricing    
    with open('working_chp_wins_all_2_effect_of_zsb/parameters.json', 'r') as f:
        parameters = json.load(f)
        parameters["El"]["eff_type"] = 2

    ## ========================================
    ## Marginal Pricing (Solve LP Relaxation)
    ## ========================================
    lem_lp = LocalEnergyMarket(players, time_periods, parameters, model_type='lp')
    time_start = time.time()
    status_lp, results_complete_lp, revenue_analysis_lp, community_prices_lp, market_prices = lem_lp.solve_complete_model(analyze_revenue=False)
    time_end = time.time()
    time_lp = time_end - time_start
    welfare_lp = lem_lp.model.getObjVal()
    print(f"Time taken: {time_lp:.2f} seconds")
    ## 실제 실현된 수익은 original "results_complete"를 사용하여 계산
    results_to_be_compared = results_complete_lp
    player_profits_lp, prices_lp = lem_lp.calculate_player_profits_with_community_prices(results_to_be_compared, community_prices_lp)

    profit_lp = {u: player_profits_lp[u]["net_profit"] for u in players} # marginal pricing의 수익
    ## 커뮤니티 수입을 계산할때도 original "lem" instance를 사용하여 계산 (lp는 relaxation이니까 부정확)
    comparison_results_lp = lem_lp.compare_individual_vs_community_profits(
        results_to_be_compared,
        players, 
        lem_lp.model_type,
        time_periods, 
        parameters,
        player_profits_lp,
        community_prices_lp 
    )
    lem_lp.generate_beamer_synergy_table(comparison_results_lp, players, filename='synergy_analysis_linear_games.tex')

    ## ========================================
    ## Computational Stability Analysis
    ## ========================================
    print("Create Core Computation Instance")
    print("="*80)
    # Create core computation instance
    core_comp = CoreComputation(
        players=players,
        time_periods=time_periods,
        model_type='lp',
        parameters=parameters
    )

    # Compute core allocation
    compute_core = False
    if compute_core:
        time_start = time.time()
        core_allocation = core_comp.compute_core(
            max_iterations=500,
            tolerance=1e-6
        )
        time_end = time.time()
        time_core = time_end - time_start
        print(f"Time taken: {time_core:.2f} seconds")
        if core_allocation:
            print("double check the stability of the found core allocation")
            coalition_found_core, violation_found_core, isimp_found_core = core_comp.measure_stability_violation(core_allocation)
            if violation_found_core <= 1e-6:    
                print("\n" + "="*70)
                print("SUCCESS: Core allocation found")
                print("="*70)

                comparison_results_core = comparison_results_lp
                for u in players:
                    ### !!!net profit key만 업데이트함에 유의!!!
                    comparison_results_core["community"]["player_profits"][u]["net_profit"] = -1*core_allocation[u]
                lem_lp.generate_beamer_synergy_table(comparison_results_core, players, filename='synergy_analysis_core_linear_games.tex')
            else:
                raise RuntimeError("Row generation algorithm has bug")
        else:
            print("\n" + "="*70)
            print("RESULT: Core is empty for this game instance")
            print("="*70)
    
    cost_lp = {u: -1*price for (u,price) in profit_lp.items()}
    coalition_lp, violation_lp, isimp_lp = core_comp.measure_stability_violation(cost_lp)
    print(f"Violation LP: {violation_lp:.4f}")
    # if compute_core:
    #     ## find all core
    #     time_start = time.time()
    #     violation_core = core_comp.measure_stability_violation(core_allocation, brute_force=True)
    #     time_end = time.time()
    #     time_core_bf = time_end - time_start
    #     print(f"Time taken: {time_core_bf:.2f} seconds")
    #     # print(f"Violation Core: {violation_core:.4f}")
    # print('='*70)
    
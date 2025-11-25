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
    # players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10']
    time_periods = list(range(24))  # 24 hours
    configuration = {}
    configuration["players_with_renewables"] = ['u1']
    configuration["players_with_electrolyzers"] = ['u2'] + ['u7', 'u8', 'u10']
    configuration["players_with_heatpumps"] = ['u3']
    configuration["players_with_elec_storage"] = ['u1']
    configuration["players_with_hydro_storage"] = ['u2'] + ['u7', 'u10']
    configuration["players_with_heat_storage"] = ['u3']
    configuration["players_with_nfl_elec_demand"] = ['u4']
    configuration["players_with_nfl_hydro_demand"] = ['u5'] + ['u9']
    configuration["players_with_nfl_heat_demand"] = ['u6']
    configuration["players_with_fl_elec_demand"] = ['u2','u3'] + ['u7', 'u8']
    configuration["players_with_fl_hydro_demand"] = []
    configuration["players_with_fl_heat_demand"] = []
    parameters = setup_lem_parameters(players, configuration, time_periods)

    # Create and solve model with Restricted Pricing
    lem = LocalEnergyMarket(players, time_periods, parameters, isLP=False)
    ip, lp, chp = True, True, False #True, True, False
    ## ========================================
    ## Restricted Pricing
    ## ========================================
    if ip:
        time_start = time.time()
        status_complete, results_complete, revenue_analysis, community_prices = lem.solve_complete_model(analyze_revenue=False)
        time_end = time.time()
        time_ip = time_end - time_start
        print(f"Time taken: {time_ip:.2f} seconds")
        player_profits, prices = lem.calculate_player_profits_with_community_prices(results_complete, community_prices)
        comparison_results = lem.compare_individual_vs_community_profits(
            players, 
            time_periods, 
            parameters,
            player_profits
        )
        profit_ip = {u: player_profits[u]["net_profit"] for u in players} # restricted pricing의 또 다른 이름인 integer programming pricing의 수익
        # 음의 시너지 분석
        if comparison_results['synergy']['absolute_gain'] < 0:
            lem.analyze_negative_synergy(comparison_results, players)
        lem.generate_beamer_synergy_table(comparison_results, players, filename='synergy_analysis_ip.tex')

        # 추가 분석: 어떤 플레이어가 가장 큰 이익을 보는지
        print("\n" + "="*80)
        print("PLAYER BENEFIT ANALYSIS")
        print("="*80)
    
    ## ========================================
    ## Marginal Pricing (Solve LP Relaxation)
    ## ========================================
    if lp:
        lem_lp = LocalEnergyMarket(players, time_periods, parameters, isLP=True)
        time_start = time.time()
        status_lp, _, revenue_analysis_lp, community_prices_lp = lem_lp.solve_complete_model(analyze_revenue=False)
        time_end = time.time()
        time_lp = time_end - time_start
        print(f"Time taken: {time_lp:.2f} seconds")
        ## 실제 실현된 수익은 original "results_complete"를 사용하여 계산
        results_to_be_compared = results_complete
        player_profits_lp, prices_lp = lem_lp.calculate_player_profits_with_community_prices(results_to_be_compared, community_prices_lp)
        profit_lp = {u: player_profits_lp[u]["net_profit"] for u in players} # marginal pricing의 수익
        ## 커뮤니티 수입을 계산할때도 original "lem" instance를 사용하여 계산 (lp는 relaxation이니까 부정확)
        comparison_results_lp = lem.compare_individual_vs_community_profits(
            players, 
            time_periods, 
            parameters,
            player_profits_lp
        )
        lem_lp.generate_beamer_synergy_table(comparison_results_lp, players, filename='synergy_analysis_lp.tex')
    ## ========================================
    ## Convex Hull Pricing
    ## ========================================
    if chp:
        print("COLUMN GENERATION FOR LOCAL ENERGY MARKET")
        print("Dantzig-Wolfe Decomposition Implementation")
        print("="*80)
        cg_solver = ColumnGenerationSolver(players, time_periods, parameters)
        time_start = time.time()
        status, solution, obj_val = cg_solver.solve()
        time_end = time.time()
        time_chp = time_end - time_start
        print(f"Time taken: {time_chp:.2f} seconds")
        if status == "optimal":
            print("\n" + "="*80)
            print("COLUMN GENERATION - SUCCESSFUL")
            print("="*80)
            print(f"Optimal objective: {obj_val:.2f} EUR")
            
            # Print convex hull prices
            if 'convex_hull_prices' in solution:
                print("\n" + "="*80)
                print("CONVEX HULL PRICES (Community Balance Shadow Prices)")
                print("="*80)
                chp = solution['convex_hull_prices']
                print("\nElectricity Prices (EUR/MWh):")
                for t in time_periods:
                    print(f"  t={t:2d}: {chp['electricity'][t]:8.4f}")
                print("\nHeat Prices (EUR/MWh):")
                for t in time_periods:
                    print(f"  t={t:2d}: {chp['heat'][t]:8.4f}")
                print("\nHydrogen Prices (EUR/kg):")
                for t in time_periods:
                    print(f"  t={t:2d}: {chp['hydrogen'][t]:8.4f}")

                # Perform synergy analysis
                print("\n" + "="*80)
                print("PERFORMING SYNERGY ANALYSIS")
                print("="*80)
                synergy_results = cg_solver.analyze_synergy_with_convex_hull_prices(solution, obj_val, chp)
                profit_chp = {u: synergy_results['community_profits'][u]['net_profit'] for u in players}
                comparison_results_chp = comparison_results_lp
                for u in players:
                    ### !!!net profit key만 업데이트함에 유의!!!
                    comparison_results_chp["community"]["player_profits"][u]["net_profit"] = player_profits_chp[u]
                lem.generate_beamer_synergy_table(comparison_results_chp, players, filename='synergy_analysis_chp.tex')
            print("\n" + "="*80)
            print("COMPLETED SUCCESSFULLY")
            print("="*80)

    ## ========================================
    ## Computational Stability Analysis
    ## ========================================
    print("Create Core Computation Instance")
    print("="*80)
    # Create core computation instance
    core_comp = CoreComputation(
        players=players,
        time_periods=time_periods,
        parameters=parameters
    )

    # Compute core allocation
    compute_core = True
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
            violation_found_core = core_comp.measure_stability_violation(core_allocation)
            if violation_found_core <= 1e-6:    
                print("\n" + "="*70)
                print("SUCCESS: Core allocation found")
                print("="*70)

                comparison_results_core = comparison_results_lp
                for u in players:
                    ### !!!net profit key만 업데이트함에 유의!!!
                    comparison_results_core["community"]["player_profits"][u]["net_profit"] = -1*core_allocation[u]
                lem.generate_beamer_synergy_table(comparison_results_core, players, filename='synergy_analysis_core.tex')
            else:
                raise RuntimeError("Row generation algorithm has bug")
        else:
            print("\n" + "="*70)
            print("RESULT: Core is empty for this game instance")
            print("="*70)
    if ip:
        cost_ip = {u: -1*price for (u,price) in profit_ip.items()}
        violation_ip = core_comp.measure_stability_violation(cost_ip)
        print(f"Violation IP: {violation_ip:.4f}")
    if lp:
        cost_lp = {u: -1*price for (u,price) in profit_lp.items()}
        violation_lp = core_comp.measure_stability_violation(cost_lp)
        print(f"Violation LP: {violation_lp:.4f}")
    if chp:
        cost_chp = {u: -1*price for (u,price) in profit_chp.items()}
        violation_chp = core_comp.measure_stability_violation(cost_chp)
        print(f"Violation CHP: {violation_chp:.4f}")
    print('='*70)
    
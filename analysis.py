from data_generator import setup_lem_parameters
from core import CoreComputation
from compact import LocalEnergyMarket
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from chp import ColumnGenerationSolver
import pandas as pd

import sys
sys.path.append('/mnt/project')


if __name__ == "__main__":
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))  # 24 hours
    parameters = setup_lem_parameters(players, time_periods)
    # Create and solve model with Restricted Pricing
    lem = LocalEnergyMarket(players, time_periods, parameters, isLP=False)
    ip, lp, chp = False, False, False #True, True, True
    ## ========================================
    ## Restricted Pricing
    ## ========================================
    if ip:
        status_complete, results_complete, revenue_analysis, community_prices = lem.solve_complete_model(analyze_revenue=False)
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
        lem.generate_beamer_synergy_table(comparison_results, players)

        # 추가 분석: 어떤 플레이어가 가장 큰 이익을 보는지
        print("\n" + "="*80)
        print("PLAYER BENEFIT ANALYSIS")
        print("="*80)
    
    ## ========================================
    ## Marginal Pricing (Solve LP Relaxation)
    ## ========================================
    if lp:
        lem_lp = LocalEnergyMarket(players, time_periods, parameters, isLP=True)
        status_lp, _, revenue_analysis_lp, community_prices_lp = lem_lp.solve_complete_model(analyze_revenue=False)
        ## 실제 실현된 수익은 original "results_complete"를 사용하여 계산
        results_to_be_compared = results_complete
        player_profits_lp, prices_lp = lem_lp.calculate_player_profits_with_community_prices(results_to_be_compared, community_prices_lp)
        profit_lp = {u: player_profits_lp[u]["net_profit"] for u in players} # marginal pricing의 수익

    ## ========================================
    ## Convex Hull Pricing
    ## ========================================
    if chp:
        print("COLUMN GENERATION FOR LOCAL ENERGY MARKET")
        print("Dantzig-Wolfe Decomposition Implementation")
        print("="*80)
        cg_solver = ColumnGenerationSolver(players, time_periods, parameters)
        status, solution, obj_val = cg_solver.solve()

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
        core_allocation = core_comp.compute_core(
            max_iterations=500,
            tolerance=1e-6
        )

        if core_allocation:
            print("double check the stability of the found core allocation")
            violation_found_core = core_comp.measure_stability_violation(core_allocation)
            if violation_found_core <= 1e-6:    
                print("\n" + "="*70)
                print("SUCCESS: Core allocation found")
                print("="*70)
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
    
from data_generator import setup_lem_parameters
from core import CoreComputation
from compact_utility import LocalEnergyMarket
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from chp import ColumnGenerationSolver
import pandas as pd
import time
import sys
sys.path.append('/mnt/project')


if __name__ == "__main__":
    model_type = 'mip'
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    # players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']
    time_periods = list(range(24))  # 24 hours
    configuration = {}
    configuration["players_with_renewables"] = ['u1']
    configuration["players_with_solar"] = []
    configuration["players_with_wind"] = ['u1']
    configuration["players_with_electrolyzers"] = ['u2'] #+ ['u7']
    configuration["players_with_heatpumps"] = ['u3']
    configuration["players_with_elec_storage"] = ['u1']
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
        'use_tou_elec': [True],#,, False],
        'import_factor': [1.5],
        'month': [1],
        'hp_cap': [0.8],
        'els_cap': [1],
        'num_households': [700], #적정, oversupply, undersupply
        'nu_cop': [3.28],
        'c_su_G': [50],
        'c_su_H': [10],
        'base_h2_price_eur': [5000/1500], #AS-IS, TO-BE
        'e_E_cap_ratio': [1.0],
        'e_H_cap_ratio': [1.0],
        'e_G_cap_ratio': [1.0],
        'eff_type': [1], # 1: HYP-MIL, 2: HYP-L,
        'segments': [6], # 정교한 근사, 간략한 근사,
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
        'day': [5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    }


    # INSERT_YOUR_CODE
    import itertools

    # Prepare all param names and value list for combinations
    param_names = list(sensitivity_analysis_candidates.keys())
    param_values = [sensitivity_analysis_candidates[name] for name in param_names]
    results_summary = []
    violation_lp = []
    ip, lp_relax, chp = True, True, True
    for i, values in enumerate(itertools.product(*param_values)):
        # Enum a single config
        sensitivity_analysis = dict(zip(param_names, values))
        # Generate parameters under this config
        parameters = setup_lem_parameters(players, configuration, time_periods, sensitivity_analysis)
        # Build and solve IP
        if ip:
            lem = LocalEnergyMarket(players, time_periods, parameters, model_type='mip')
            time_start = time.time()
            status_complete, results_complete_ip, revenue_analysis, community_prices, market_prices = lem.solve_complete_model(analyze_revenue=False)
            welfare = lem.model.getObjVal()
            time_end = time.time()
            time_ip = time_end - time_start
            print(f"Time taken: {time_ip:.2f} seconds")
            player_profits, prices = lem.calculate_player_profits_with_community_prices(results_complete_ip, community_prices)
            comparison_results = lem.compare_individual_vs_community_profits(
                results_complete_ip,
                players, 
                lem.model_type,
                time_periods, 
                parameters,
                player_profits,
                community_prices
            )
            profit_ip = {u: player_profits[u]["net_profit"] for u in players} # restricted pricing의 또 다른 이름인 integer programming pricing의 수익
            profit_pca = lem.proportional_cost_allocation(comparison_results["individual"], comparison_results["community"]["total_profit"])
        if lp_relax:
            lem.model.freeTransform()
            lem.model.relax()
            time_start = time.time()
            status_complete_lp, results_complete_lp, revenue_analysis_lp, community_prices_lp, _ = lem.solve_complete_model(analyze_revenue=False)
            time_end = time.time()
            time_lp = time_end - time_start
            print(f"Time taken: {time_lp:.2f} seconds")
            welfare_lp = lem.model.getObjVal()
            player_profits_lp, prices_lp = lem.calculate_player_profits_with_community_prices(results_complete_ip, community_prices_lp)
            comparison_results_lp = lem.compare_individual_vs_community_profits(
                results_complete_lp,
                players, 
                lem.model_type,
                time_periods, 
                parameters,
                player_profits_lp,
                community_prices_lp
            )
            profit_lp = {u: player_profits_lp[u]["net_profit"] for u in players}
        
        if chp:
            print("COLUMN GENERATION FOR LOCAL ENERGY MARKET")
            print("Dantzig-Wolfe Decomposition Implementation")
            print("="*80)
            init_sol = results_complete_ip
            # init_sol = None
            cg_solver = ColumnGenerationSolver(players, time_periods, parameters, model_type='mip', init_sol=init_sol)
            time_start = time.time()
            status, results_chp, obj_val, solution_by_player = cg_solver.solve()
            time_end = time.time()
            time_chp = time_end - time_start
            welfare_chp = cg_solver.master.model.getObjVal()
            print(f"Time taken: {time_chp:.2f} seconds")
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
                    synergy_results = cg_solver.analyze_synergy_with_convex_hull_prices(results_complete_ip, obj_val, community_prices_chp)
                    profit_chp = {u: synergy_results['community_profits'][u]['net_profit'] for u in players}
        
        core_comp = CoreComputation(
        players=players,
        time_periods=time_periods,
        model_type='mip',
        parameters=parameters
        )
        if ip:
            cost_ip = {u: -1*price for (u,price) in profit_ip.items()}
            violation_ip = core_comp.measure_stability_violation(cost_ip)
            print(f"Violation IP: {violation_ip:.4f}")

            cost_pca = {u: -1*price for (u,price) in profit_pca.items()}
            violation_pca = core_comp.measure_stability_violation(cost_pca)
            print(f"Violation PCA: {violation_pca:.4f}")
        if lp_relax:
            cost_lp = {u: -1*price for (u,price) in profit_lp.items()}
            violation_lp = core_comp.measure_stability_violation(cost_lp)
            print(f"Violation LP: {violation_lp:.4f}")
        if chp:
            cost_chp = {u: -1*price for (u,price) in profit_chp.items()}
            violation_chp = core_comp.measure_stability_violation(cost_chp)
            print(f"Violation CHP: {violation_chp:.4f}")
        result_row = {
            'sensitivity_analysis': sensitivity_analysis,
            'violation_ip': violation_ip,
            'solve_time': time_ip,
            'profit_ip': profit_ip
        }
        if lp_relax:
            result_row = result_row | {
                'violation_lp': violation_lp,
                'solve_time_lp': time_lp,
                'profit_lp': profit_lp
            }
        if chp:
            result_row = result_row | {
                'violation_chp': violation_chp,
                'solve_time_chp': time_chp,
                'profit_chp': profit_chp
            }
        results_summary.append(result_row)
        # print('='*60)
        # print('Current Sensitivity Case:', sensitivity_analysis)
        # print(f'Core Violation: {violation_lp:.6f}, LP Solve Time: {time_lp:.2f}')
        # print('='*60)

        # Optionally: Save as DataFrame
        df = pd.DataFrame(results_summary)
        df.to_csv('sensitivity_analysis_results_all.csv', index=False)

        if i > 100:
            break

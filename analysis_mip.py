from data_generator import setup_lem_parameters
from core import CoreComputation
# from compact import LocalEnergyMarket
from compact_utility import LocalEnergyMarket
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from chp import ColumnGenerationSolver
import pandas as pd
import time
import sys
import copy
from visualize import plot_community_prices_comparison
# sys.path.append('/mnt/project')


if __name__ == "__main__":
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))  # 24 hours
    configuration = {}
    configuration["players_with_renewables"] = ['u1']
    configuration["players_with_solar"] = []
    configuration["players_with_wind"] = ['u1']
    configuration["players_with_electrolyzers"] = ['u2']# + ['u7']
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

    # players = ['u2', 'u5']
    # time_periods = list(range(24))  # 24 hours
    # configuration = {}
    # configuration["players_with_renewables"] = []
    # configuration["players_with_solar"] = []
    # configuration["players_with_wind"] = []
    # configuration["players_with_electrolyzers"] = ['u2']# + ['u7']
    # configuration["players_with_heatpumps"] = []
    # configuration["players_with_elec_storage"] = [] #,'u7']
    # configuration["players_with_hydro_storage"] = ['u2'] #+ ['u7']
    # configuration["players_with_heat_storage"] = []
    # configuration["players_with_nfl_elec_demand"] = []
    # configuration["players_with_nfl_hydro_demand"] = ['u5'] #+ ['u8']
    # configuration["players_with_nfl_heat_demand"] = []
    # configuration["players_with_fl_elec_demand"] = []# + ['u7']
    # configuration["players_with_fl_hydro_demand"] = []
    # configuration["players_with_fl_heat_demand"] = []

    parameters = setup_lem_parameters(players, configuration, time_periods)
    # parameters["c_els_u7"] = parameters["c_els"]*2
    # parameters["c_su_u7"] = parameters["c_su_G"]*2
    # Create and solve model with Restricted Pricing
    ip, chp = True, True
    lp_relax = False
    compute_core = True
    brute_force = True
    analyze_revenue = False
    base_path = '.'
    # INSERT_YOUR_CODE
    import os, json
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/parameters.json", "w") as f:
        json.dump(parameters, f, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x), indent=2)

    """
    참고용: 아래 hydrogen efficient, inefficient 다시 계산해볼것.
    """
    hydro_efficient = parameters["El"]["a"][0]*(parameters["El"]["p_val"][1])+parameters["El"]["b"][0]
    hydro_inefficient = parameters["El"]["a"][-1]*(parameters["El"]["p_val"][-1]-parameters["El"]["p_val"][-2])/(parameters["El"]["p_val"][-1]-parameters["El"]["p_val"][-2])
    ## 수소 1kg 위해 구매해야 하는 전기비용
    hydro_cost_per_kg_efficient = 1/hydro_efficient*parameters["pi_E_gri"]["import"].mean()
    hydro_cost_per_kg_inefficient = 1/hydro_inefficient*parameters["pi_E_gri"]["import"].mean()
    print(f"수소 1kg 위해 구매해야 하는 전기비용 (efficient): {hydro_cost_per_kg_efficient} EUR")
    print(f"수소 1kg 위해 구매해야 하는 전기비용 (inefficient): {hydro_cost_per_kg_inefficient} EUR")
    hydro_efficiency = np.array([parameters['El']['a'][jj] + parameters['El']['b'][jj]/parameters['El']['p_val'][jj] for jj in range(len(parameters['El']['a']))])
    print(f"수소 Efficiency (kg/MWh): {hydro_efficiency} ")
    print(f"단위 투입 전력 당 수소의 외부 판매 수익: {parameters["pi_G_gri"]["export"].mean() * hydro_efficiency}")
    heat_cost_per_mwh = (1/parameters["nu_cop"])*parameters["pi_E_gri"]["import"].mean()
    print(f"heat 1MWh 생산단가: {heat_cost_per_mwh} EUR")
    # with open('working_chp_wins_all_2/parameters.json', 'r') as f:
    #     parameters = json.load(f)
    #     print(1)
    ## ========================================
    ## Restricted Pricing
    ## ========================================
    if ip:
        lem = LocalEnergyMarket(players, time_periods, parameters, model_type='mip')
        time_start = time.time()
        status_complete, results_complete_ip, revenue_analysis, community_prices, market_prices = lem.solve_complete_model(analyze_revenue=analyze_revenue)
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
        # 음의 시너지 분석
        if comparison_results['synergy']['absolute_gain'] < 0:
            lem.analyze_negative_synergy(comparison_results, players)
        lem.generate_beamer_synergy_table(comparison_results, players, filename=f'{base_path}/synergy_analysis_ip.tex')

        # 추가 분석: 어떤 플레이어가 가장 큰 이익을 보는지
        print("\n" + "="*80)
        print("PLAYER BENEFIT ANALYSIS")
        print("="*80)
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
        # 음의 시너지 분석
        if comparison_results_lp['synergy']['absolute_gain'] < 0:
            lem.analyze_negative_synergy(comparison_results_lp, players)
        lem.generate_beamer_synergy_table(comparison_results_lp, players, filename=f'{base_path}/synergy_analysis_lp_relax.tex')

        # 추가 분석: 어떤 플레이어가 가장 큰 이익을 보는지
        print("\n" + "="*80)
        print("PLAYER BENEFIT ANALYSIS")
        print("="*80)
    else:
        community_prices_lp = None
    ## ========================================
    ## Convex Hull Pricing
    ## ========================================
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
                comparison_results_chp = copy.deepcopy(comparison_results)
                for u in players:
                    ### !!!net profit key만 업데이트함에 유의!!!
                    comparison_results_chp["community"]["player_profits"][u]["net_profit"] = profit_chp[u]
                comparison_results_chp["community"]["prices"] = community_prices_chp
                lem.generate_beamer_synergy_table(comparison_results_chp, players, filename=f'{base_path}/synergy_analysis_chp.tex')
            print("\n" + "="*80)
            print("COMPLETED SUCCESSFULLY")
            print("="*80)
    else:
        community_prices_chp = None
    ## ========================================
    ## Visualize Community Prices Comparison
    ## ========================================
    plot_community_prices_comparison(community_prices, community_prices_lp, community_prices_chp, market_prices, save_path=f'{base_path}/community_prices_comparison.png')
    with open(f'{base_path}/comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    if lp_relax:
        with open(f'{base_path}/comparison_results_lp.json', 'w') as f:
            json.dump(comparison_results_lp, f, indent=2)
    if chp:
        with open(f'{base_path}/comparison_results_chp.json', 'w') as f:
            json.dump(comparison_results_chp, f, indent=2)

    import pickle
    # Save with pickle instead of json because results may contain tuples.
    with open(f'{base_path}/results_ip.pkl', 'wb') as f:
        pickle.dump(results_complete_ip, f)
    if lp_relax:
        with open(f'{base_path}/results_lp.pkl', 'wb') as f:
            pickle.dump(results_complete_lp, f)
    if chp:
        with open(f'{base_path}/results_chp.pkl', 'wb') as f:
            pickle.dump(results_chp, f)
    
    # # results_ip.pkl 읽는 코드
    # import pickle

    # with open(f'{base_path}/results_ip.pkl', 'rb') as f:
    #     results_ip_loaded = pickle.load(f)
    ## ========================================
    ## Computational Stability Analysis
    ## ========================================
    print("Create Core Computation Instance")
    print("="*80)
    # Create core computation instance
    core_comp = CoreComputation(
        players=players,
        time_periods=time_periods,
        model_type='mip',
        parameters=parameters
    )

    # Compute core allocation
    if compute_core:
        time_start = time.time()
        core_rowgen = core_comp.compute_core(
            max_iterations=1e+8,
            tolerance=1e-6
        )
        time_end = time.time()
        time_rowgen = time_end - time_start
        print(f"Time taken: {time_rowgen:.2f} seconds")
        if core_rowgen:
            print("double check the stability of the found core allocation")
            coalition_rowgen, violation_rowgen, isimp_rowgen = core_comp.measure_stability_violation(core_rowgen)
            if violation_rowgen <= 1e-6:    
                print("\n" + "="*70)
                print("SUCCESS: Core allocation found")
                print("="*70)

                comparison_results_core = copy.deepcopy(comparison_results)
                for u in players:
                    ### !!!net profit key만 업데이트함에 유의!!!
                    comparison_results_core["community"]["player_profits"][u]["net_profit"] = -1*core_rowgen[u]
                lem.generate_beamer_synergy_table(comparison_results_core, players, filename=f'{base_path}/synergy_analysis_core.tex')
            else:
                raise RuntimeError("Row generation algorithm has bug")
        else:
            print("\n" + "="*70)
            print("RESULT: Core is empty for this game instance")
            print("="*70)
    if ip:
        cost_ip = {u: -1*price for (u,price) in profit_ip.items()}
        coalition_ip, violation_ip, isimp_ip = core_comp.measure_stability_violation(cost_ip)
        print(f"Violation IP: {violation_ip:.4f}")

        cost_pca = {u: -1*price for (u,price) in profit_pca.items()}
        coalition_pca, violation_pca, isimp_pca = core_comp.measure_stability_violation(cost_pca)
        print(f"Violation PCA: {violation_pca:.4f}")
    if lp_relax:
        cost_lp = {u: -1*price for (u,price) in profit_lp.items()}
        coalition_lp, violation_lp, isimp_lp = core_comp.measure_stability_violation(cost_lp)
        print(f"Violation LP: {violation_lp:.4f}")
    if chp:
        cost_chp = {u: -1*price for (u,price) in profit_chp.items()}
        coalition_chp, violation_chp, isimp_chp = core_comp.measure_stability_violation(cost_chp)
        print(f"Violation CHP: {violation_chp:.4f}")
    if brute_force:
        ## find all core
        time_start = time.time()
        coalition_core, violation_core, isimp_core = core_comp.measure_stability_violation(core_rowgen, brute_force=True)
        time_end = time.time()
        time_core_bf = time_end - time_start
        print(f"Time taken: {time_core_bf:.2f} seconds")

        ## IP, LP, CHP도 다시 검증
        temp = 0
        if ip:
            coalition_ip_bf, violation_ip_bf, isimp_ip_bf = core_comp.measure_stability_violation(cost_ip, brute_force=True)
            temp += np.abs(violation_ip - violation_ip_bf)
        if lp_relax:
            coalition_lp_bf, violation_lp_bf, isimp_lp_bf = core_comp.measure_stability_violation(cost_lp, brute_force=True)
            temp += np.abs(violation_lp - violation_lp_bf)
        if chp:
            coalition_chp_bf, violation_chp_bf, isimp_chp_bf = core_comp.measure_stability_violation(cost_chp, brute_force=True)
            temp += np.abs(violation_chp - violation_chp_bf)

        if temp > 1e-6:
            raise RuntimeError("Mismatch between actual and computed violation!")
        # print(f"Violation Core: {violation_core:.4f}")
    # INSERT_YOUR_CODE
    import pandas as pd
    import os

    # INSERT_YOUR_CODE

    # Assemble result DataFrame comparing the methods
    results = []

    # First, collect all required allocations per method for each player
    players_sorted = sorted(players)  # ensure order

    # Helper to safely get allocation/cost dictionaries or fill nan
    def get_cost_dict(cost_dict):
        if cost_dict is not None:
            return {u: cost_dict.get(u, float('nan')) for u in players_sorted}
        else:
            return {u: float('nan') for u in players_sorted}

    # Prepare all allocations/costs for each method
    cost_ip_dict = get_cost_dict(locals().get('cost_ip', None))
    cost_lp_dict = get_cost_dict(locals().get('cost_lp', None))
    cost_chp_dict = get_cost_dict(locals().get('cost_chp', None))
    cost_pca_dict = get_cost_dict(locals().get('cost_pca', None))
    rowgen_dict = get_cost_dict(locals().get('core_rowgen', None))

    for method, violation, t, cost_dict in [
        ('ip', violation_ip if 'violation_ip' in locals() else float('nan'), time_ip if 'time_ip' in locals() else float('nan'), cost_ip_dict),
        ('lp', violation_lp if 'violation_lp' in locals() else float('nan'), time_lp if 'time_lp' in locals() else float('nan'), cost_lp_dict),
        ('pca', violation_pca if 'violation_pca' in locals() else float('nan'), time_ip if 'time_ip' in locals() else float('nan'), cost_pca_dict),
        ('chp', violation_chp if 'violation_chp' in locals() else float('nan'), time_chp if 'time_chp' in locals() else float('nan'), cost_chp_dict),
        ('rowgen', violation_rowgen if 'violation_rowgen' in locals() else float('nan'), time_rowgen if 'time_rowgen' in locals() else float('nan'), rowgen_dict),
    ]:
        row = {
            'method': method,
            'violation': violation,
            'time': t
        }
        total_cost = 0
        # Add allocation for each player
        for u in players_sorted:
            row[u] = cost_dict[u]
            # sum for total cost
            if not (cost_dict[u] is None or (isinstance(cost_dict[u], float) and pd.isna(cost_dict[u]))):
                total_cost += cost_dict[u]
        row['total_cost'] = total_cost
        row['total_profit'] = -1 * total_cost
        results.append(row)

    # If brute_force True, add its row
    if brute_force:
        results.append({
            'method': 'brute_force',
            'violation': float('nan'),
            'time': time_core_bf if 'time_core_bf' in locals() else float('nan')
        })

    df_results = pd.DataFrame(results, columns=['method', 'violation', 'time'] + players_sorted + ['total_cost', 'total_profit'])

    # Save to CSV in base_path
    csv_path = os.path.join(base_path, "core_stability_comparison.csv")
    df_results.to_csv(csv_path, index=False)


    print(f"Result saved to {csv_path}")
    print('='*70)

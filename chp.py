"""
Column Generation for Local Energy Market with Convex Hull Pricing
Main solver implementation using Dantzig-Wolfe Decomposition
"""
import sys
sys.path.append('/mnt/project')
# sys.path.append('/home/claude')

from pyscipopt import SCIP_PARAMSETTING
from typing import Dict, List, Tuple

from data_generator import setup_lem_parameters
from solver import PlayerSubproblem, MasterProblem
from pricer import LEMPricer
import numpy as np


class ColumnGenerationSolver:
    """
    Main column generation solver for Local Energy Market
    Implements Dantzig-Wolfe decomposition to compute convex hull prices
    """
    def __init__(self, players: List[str], time_periods: List[int], parameters: Dict, model_type: str, init_sol: Dict = None):
        """
        Initialize column generation solver
        
        Args:
            players: List of player IDs
            time_periods: List of time periods
            parameters: Model parameters
        """
        self.players = players
        self.time_periods = time_periods
        self.parameters = parameters
        self.init_sol = init_sol
        # Create subproblems for each player
        print("=== Creating Subproblems ===")
        self.subproblems = {}
        for player in players:
            print(f"Creating subproblem for player {player}...")
            self.subproblems[player] = PlayerSubproblem(
                player=player,
                time_periods=time_periods,
                parameters=parameters,
                model_type=model_type
            )
        
        # Create master problem
        print("\n=== Creating Master Problem ===")
        self.master = MasterProblem(players, time_periods)
        
        print("Note: Initial columns will be generated via initial solve")
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[str, Dict, float]:
        """
        Solve using column generation to obtain convex hull prices
        
        Args:
            max_iterations: Maximum number of CG iterations (not used with SCIP pricer)
            tolerance: Convergence tolerance for reduced cost (not used with SCIP pricer)
            
        Returns:
            tuple: (status, solution, objective_value)
        """
        print("\n" + "="*80)
        print("COLUMN GENERATION - STARTING ITERATIONS")
        print("="*80)
                
        # Generate initial columns for each player
        print("\n=== Generating Initial Columns ===")
        self.master._add_initial_columns(self.subproblems, self.init_sol)
        # Create master constraints
        self.master._create_master_constraints()
        # Disable presolving to allow pricer to work properly
        self.master.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.master.model.setHeuristics(SCIP_PARAMSETTING.OFF)
        self.master.model.setSeparating(SCIP_PARAMSETTING.OFF)
        self.master.model.disablePropagation()
        # Create pricer
        pricer = LEMPricer(
            subproblems=self.subproblems,
            time_periods=self.time_periods,
            players=self.players
        )
        
        # Include pricer in master problem
        self.master.model.includePricer(
            pricer,
            "LEMPricer",
            "Pricer for Local Energy Market column generation"
        )

        # Solve with column generation
        print("\nSolving master problem with pricing...")
        self.master.model.optimize()
        status = self.master.model.getStatus()
        
        if status == "optimal":
            obj_val = self.master.get_objective_value()
            solution, solution_by_player = self.master.get_solution()
            
            print("\n" + "="*80)
            print("COLUMN GENERATION - COMPLETED")
            print("="*80)
            print(f"Optimal objective value: {obj_val:.2f} EUR")
            
            # Count columns per player
            print("\nColumns per player:")
            for player in self.players:
                player_cols = [k for k in self.master.model.data['vars'].keys() if k[0] == player]
                print(f"  {player}: {len(player_cols)} columns")
            
            # Extract and print convex hull prices (dual prices of community balance constraints)
            print("\n" + "="*80)
            print("CONVEX HULL PRICES (Shadow Prices of Community Balance)")
            print("="*80)
            
            chp_elec = {}
            chp_heat = {}
            chp_hydro = {}
            
            for t in self.time_periods:
                elec_cons = self.master.model.data['cons']['community_elec_balance'][t]
                heat_cons = self.master.model.data['cons']['community_heat_balance'][t]
                hydro_cons = self.master.model.data['cons']['community_hydro_balance'][t]
                
                try:
                    t_elec_cons = self.master.model.getTransformedCons(elec_cons)
                    t_heat_cons = self.master.model.getTransformedCons(heat_cons)
                    t_hydro_cons = self.master.model.getTransformedCons(hydro_cons)
                    chp_elec[t] = np.abs(self.master.model.getDualsolLinear(t_elec_cons))
                    chp_heat[t] = np.abs(self.master.model.getDualsolLinear(t_heat_cons))
                    chp_hydro[t] = np.abs(self.master.model.getDualsolLinear(t_hydro_cons))
                except:
                    raise Exception("Error getting dual multipliers")
            
            # Print convex hull prices
            print("\nConvex Hull Prices (EUR/MWh or EUR/kg):")
            sample_times = self.time_periods
            print(f"{'Time':>6} {'Electricity':>12} {'Heat':>12} {'Hydrogen':>12}")
            print("-" * 48)
            for t in sample_times:
                print(f"{t:6d} {chp_elec[t]:12.4f} {chp_heat[t]:12.4f} {chp_hydro[t]:12.4f}")
            
            # Store convex hull prices in solution
            solution['convex_hull_prices'] = {
                'electricity': chp_elec,
                'heat': chp_heat,
                'hydrogen': chp_hydro
            }
            
            return status, solution, obj_val
        else:
            print(f"\nColumn generation failed with status: {status}")
            return status, None, None
    def analyze_synergy_with_convex_hull_prices(self, solution: Dict, obj_val: float, community_prices: Dict):
        """
        Analyze individual vs community profits using convex hull prices
        
        Args:
            solution: Solution from column generation including convex hull prices
            obj_val: Objective value from column generation
        """
        from compact import LocalEnergyMarket, solve_and_extract_results
        
        print("\n" + "="*80)
        print("SYNERGY ANALYSIS WITH CONVEX HULL PRICING")
        print("="*80)
        
        # Extract convex hull prices
        chp = community_prices
        
        # Step 1: Calculate community player profits using convex hull prices
        print("\nSTEP 1: Computing player profits in community (using convex hull prices)")
        print("-"*80)
        
        # For this, we need to solve the community problem again to get detailed results
        lem_community = LocalEnergyMarket(
            self.players, 
            self.time_periods, 
            self.parameters, 
            model_type = 'mip',
            dwr=False
        )
        
        # Solve with SCIP settings for consistency
        from pyscipopt import SCIP_PARAMSETTING
        lem_community.model.setPresolve(SCIP_PARAMSETTING.OFF)
        lem_community.model.setHeuristics(SCIP_PARAMSETTING.OFF)
        lem_community.model.disablePropagation()
        lem_community.model.setSeparating(SCIP_PARAMSETTING.OFF)
        # lem_community.model.hideOutput()
        lem_community.model.optimize()
        
        status_comm = lem_community.model.getStatus()
        if status_comm != "optimal":
            print(f"âš ï¸ Community optimization failed: {status_comm}")
            return
        
        _, results_comm = solve_and_extract_results(lem_community.model)
        
        # Calculate player profits with convex hull prices
        player_profits_chp = self._calculate_player_profits_with_chp(
            results_comm, chp, lem_community
        )
        
        # Step 2: Optimize each player individually
        print("\nSTEP 2: Computing individual player profits")
        print("-"*80)
        
        individual_profits = {}
        for player in self.players:
            print(f"Optimizing {player} individually...", end=" ")
            
            # Create individual parameters
            individual_params = self.parameters.copy()
            for key in individual_params.keys():
                if key.startswith('players_with_'):
                    if player in individual_params[key]:
                        individual_params[key] = [player]
                    else:
                        individual_params[key] = []
            individual_params['dwr'] = False
            
            # Solve individual problem
            lem_individual = LocalEnergyMarket(
                [player],
                self.time_periods,
                individual_params,
                model_type = 'mip',
                dwr=False
            )
            lem_individual.model.hideOutput()
            status_ind = lem_individual.solve()
            
            if status_ind == "optimal":
                _, results_ind = solve_and_extract_results(lem_individual.model)
                revenue_ind = lem_individual._analyze_revenue_by_resource(results_ind)
                individual_profits[player] = revenue_ind['net_profit']
                print(f"Profit: {revenue_ind['net_profit']:.2f} EUR")
            else:
                individual_profits[player] = 0
                print(f"Failed ({status_ind})")
        
        # Step 3: Synergy analysis
        print("\n" + "="*80)
        print("SYNERGY ANALYSIS: INDIVIDUAL VS COMMUNITY")
        print("="*80)
        
        print(f"\n{'Player':^10} | {'Individual':^15} | {'Community':^15} | {'Gain':^15} | {'Gain %':^12}")
        print("-"*80)
        
        total_individual = 0
        total_community = 0
        
        for player in self.players:
            ind_profit = individual_profits.get(player, 0)
            comm_profit = player_profits_chp[player]['net_profit']
            gain = comm_profit - ind_profit
            
            # Calculate gain percentage
            if ind_profit == 0:
                if gain > 0:
                    gain_pct_str = "N/A (+)"
                elif gain < 0:
                    gain_pct_str = "N/A (-)"
                else:
                    gain_pct_str = "0.0"
            else:
                gain_pct = (gain / abs(ind_profit)) * 100
                gain_pct_str = f"{gain_pct:.1f}"
            
            total_individual += ind_profit
            total_community += comm_profit
            
            # Gain marker
            if gain > 0:
                gain_marker = "UP"
            elif gain < 0:
                gain_marker = "DN"
            else:
                gain_marker = ""
            
            print(f"{player:^10} | {ind_profit:^15.2f} | {comm_profit:^15.2f} | "
                  f"{gain:^15.2f} {gain_marker} | {gain_pct_str:^12}")
        
        print("-"*80)
        
        # Total synergy
        total_gain = total_community - total_individual
        if total_individual == 0:
            total_gain_pct_str = "N/A"
        else:
            total_gain_pct = (total_gain / abs(total_individual)) * 100
            total_gain_pct_str = f"{total_gain_pct:.1f}%"
        
        print(f"{'Total':^10} | {total_individual:^15.2f} | {total_community:^15.2f} | "
              f"{total_gain:^15.2f} | {total_gain_pct_str:^12}")
        
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("Individual: Player's profit when operating alone")
        print("Community: Player's profit in community (using convex hull prices)")
        print("Gain: Additional profit from community participation")
        print("\nNote: Convex hull prices ensure efficient market clearing and")
        print("      reflect true opportunity costs in the community market.")
        
        # results_comparison = {"individual":{u: }}
        return {
            'individual_profits': individual_profits,
            'community_profits': player_profits_chp,
            'total_gain': total_gain,
            'convex_hull_prices': chp
        }
    def _calculate_player_profits_with_chp(self, results: Dict, chp: Dict, lem: 'LocalEnergyMarket') -> Dict:
        """
        Calculate player profits using convex hull prices
        
        Args:
            results: Optimization results from community model
            chp: Convex hull prices (electricity, heat, hydrogen)
            lem: LocalEnergyMarket instance for parameters
            
        Returns:
            dict: Player profits with detailed breakdown
        """
        player_profits = {}
        
        for u in self.players:
            profit = {
                'grid_revenue': 0.0,
                'grid_cost': 0.0,
                'community_revenue': 0.0,
                'community_cost': 0.0,
                'production_cost': 0.0,
                'storage_cost': 0.0,
                'startup_cost': 0.0,
                'net_profit': 0.0
            }
            
            for t in self.time_periods:
                # 1. Grid trading
                if 'e_E_gri' in results and (u,t) in results['e_E_gri']:
                    export = results['e_E_gri'][u,t]
                    if export > 0:
                        grid_price = self.parameters.get(f'pi_E_gri_export_{t}', 0)
                        profit['grid_revenue'] += export * grid_price
                
                if 'i_E_gri' in results and (u,t) in results['i_E_gri']:
                    import_val = results['i_E_gri'][u,t]
                    if import_val > 0:
                        grid_price = self.parameters.get(f'pi_E_gri_import_{t}', 0)
                        profit['grid_cost'] += import_val * grid_price
                # Hydrogen (새로 추가!)
                if 'e_G_gri' in results and (u,t) in results['e_G_gri']:
                    export = results['e_G_gri'][u,t]
                    if export > 0:
                        grid_price = self.parameters.get(f'pi_G_gri_export_{t}', 0)
                        profit['grid_revenue'] += export * grid_price

                if 'i_G_gri' in results and (u,t) in results['i_G_gri']:
                    import_val = results['i_G_gri'][u,t]
                    if import_val > 0:
                        grid_price = self.parameters.get(f'pi_G_gri_import_{t}', 0)
                        profit['grid_cost'] += import_val * grid_price

                # Heat (새로 추가!)
                if 'e_H_gri' in results and (u,t) in results['e_H_gri']:
                    export = results['e_H_gri'][u,t]
                    if export > 0:
                        grid_price = self.parameters.get(f'pi_H_gri_export_{t}', 0)
                        profit['grid_revenue'] += export * grid_price

                if 'i_H_gri' in results and (u,t) in results['i_H_gri']:
                    import_val = results['i_H_gri'][u,t]
                    if import_val > 0:
                        grid_price = self.parameters.get(f'pi_H_gri_import_{t}', 0)
                        profit['grid_cost'] += import_val * grid_price

                # 2. Community trading (using convex hull prices)
                # Electricity
                if 'e_E_com' in results and (u,t) in results['e_E_com']:
                    export = results['e_E_com'][u,t]
                    if export > 0:
                        profit['community_revenue'] += export * chp['electricity'][t]
                
                if 'i_E_com' in results and (u,t) in results['i_E_com']:
                    import_val = results['i_E_com'][u,t]
                    if import_val > 0:
                        profit['community_cost'] += import_val * chp['electricity'][t]
                
                # Heat
                if 'e_H_com' in results and (u,t) in results['e_H_com']:
                    export = results['e_H_com'][u,t]
                    if export > 0:
                        profit['community_revenue'] += export * chp['heat'][t]
                
                if 'i_H_com' in results and (u,t) in results['i_H_com']:
                    import_val = results['i_H_com'][u,t]
                    if import_val > 0:
                        profit['community_cost'] += import_val * chp['heat'][t]
                
                # Hydrogen
                if 'e_G_com' in results and (u,t) in results['e_G_com']:
                    export = results['e_G_com'][u,t]
                    if export > 0:
                        profit['community_revenue'] += export * chp['hydrogen'][t]
                
                if 'i_G_com' in results and (u,t) in results['i_G_com']:
                    import_val = results['i_G_com'][u,t]
                    if import_val > 0:
                        profit['community_cost'] += import_val * chp['hydrogen'][t]
                
                # 3. Production costs
                if 'p' in results:
                    if (u,'res',t) in results['p']:
                        profit['production_cost'] += results['p'][u,'res',t] * self.parameters.get(f'c_res_{u}', 0)
                    if (u,'els',t) in results['p']:
                        profit['production_cost'] += results['p'][u,'els',t] * self.parameters.get(f'c_els_{u}', 0)
                    if (u,'hp',t) in results['p']:
                        profit['production_cost'] += results['p'][u,'hp',t] * self.parameters.get(f'c_hp_{u}', 0)
                
                # 4. Storage costs
                c_E_sto = self.parameters.get('c_E_sto', 0.01)
                c_G_sto = self.parameters.get('c_G_sto', 0.01)
                c_H_sto = self.parameters.get('c_H_sto', 0.01)
                nu_ch = self.parameters.get('nu_ch', 0.9)
                nu_dis = self.parameters.get('nu_dis', 0.9)
                
                if 'b_ch_E' in results and (u,t) in results['b_ch_E']:
                    profit['storage_cost'] += results['b_ch_E'][u,t] * c_E_sto * nu_ch
                if 'b_dis_E' in results and (u,t) in results['b_dis_E']:
                    profit['storage_cost'] += results['b_dis_E'][u,t] * c_E_sto * (1/nu_dis)
                
                if 'b_ch_G' in results and (u,t) in results['b_ch_G']:
                    profit['storage_cost'] += results['b_ch_G'][u,t] * c_G_sto * nu_ch
                if 'b_dis_G' in results and (u,t) in results['b_dis_G']:
                    profit['storage_cost'] += results['b_dis_G'][u,t] * c_G_sto * (1/nu_dis)
                
                if 'b_ch_H' in results and (u,t) in results['b_ch_H']:
                    profit['storage_cost'] += results['b_ch_H'][u,t] * c_H_sto * nu_ch
                if 'b_dis_H' in results and (u,t) in results['b_dis_H']:
                    profit['storage_cost'] += results['b_dis_H'][u,t] * c_H_sto * (1/nu_dis)
                
                # 5. Startup costs
                if 'z_su' in results and (u,t) in results['z_su']:
                    profit['startup_cost'] += results['z_su'][u,t] * self.parameters.get(f'c_su_{u}', 50)
            
            # Calculate net profit
            profit['net_profit'] = (
                profit['grid_revenue'] + 
                profit['community_revenue'] - 
                profit['grid_cost'] - 
                profit['community_cost'] - 
                profit['production_cost'] - 
                profit['storage_cost'] - 
                profit['startup_cost']
            )
            
            player_profits[u] = profit
        
        return player_profits
def main():
    """
    Main test function for column generation
    """
    print("\n" + "="*80)
    print("COLUMN GENERATION FOR LOCAL ENERGY MARKET")
    print("Dantzig-Wolfe Decomposition Implementation")
    print("="*80)
    
    # Setup problem
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))
    
    print("\nSetting up parameters...")
    parameters = setup_lem_parameters(players, time_periods)
    
    print(f"\n✓ Parameters configured")
    print(f"  Players: {len(players)}")
    print(f"  Time periods: {len(time_periods)}")
    print(f"  Parameter entries: {len(parameters)}")
    
    # Solve with column generation
    print("\n" + "="*80)
    print("SOLVING WITH COLUMN GENERATION")
    print("="*80)
    
    try:
        cg_solver = ColumnGenerationSolver(players, time_periods, parameters, model_type='mip')
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
                # Perform synergy analysis
                print("\n" + "="*80)
                print("PERFORMING SYNERGY ANALYSIS")
                print("="*80)
                synergy_results = cg_solver.analyze_synergy_with_convex_hull_prices(solution, obj_val, chp)
            print("\n" + "="*80)
            print("COMPLETED SUCCESSFULLY")
            print("="*80)
        
        else:
            print(f"\nColumn generation failed with status: {status}")
    
    except Exception as e:
        print(f"\n✗ Error during column generation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    print(1)
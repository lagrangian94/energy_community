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
# from compact import LocalEnergyMarket, solve_and_extract_results
from compact_utility import LocalEnergyMarket, solve_and_extract_results
from solver import PlayerSubproblem, MasterProblem
from pricer import LEMPricer
import numpy as np


class ColumnGenerationSolver:
    """
    Main column generation solver for Local Energy Market
    Implements Dantzig-Wolfe decomposition to compute convex hull prices
    """
    def __init__(self, players: List[str], time_periods: List[int], parameters: Dict, model_type: str, init_sol: Dict = None, smoothing: bool = False, mipsolver: str = 'highs'):
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
        self.smoothing = smoothing
        self.mipsolver = mipsolver
        # Create subproblems for each player
        print("=== Creating Subproblems ===")
        self.subproblems = {}
        for player in players:
            print(f"Creating subproblem for player {player}...")
            self.subproblems[player] = PlayerSubproblem(
                player=player,
                time_periods=time_periods,
                parameters=parameters,
                model_type=model_type,
                mipsolver=mipsolver
            )
        
        # Create master problem
        print("\n=== Creating Master Problem ===")
        self.master = MasterProblem(players, time_periods, parameters)
        
        print("Note: Initial columns will be generated via initial solve")
    
    def _shared_block_cost(self, init_sol: Dict) -> float:
        """Objective contribution of the master's shared block (r_sym, p) at a solution.

        The incumbent passed to the pricer is assembled from per-player column costs,
        which by construction exclude the community variables. Given the members'
        private quantities, the master would set

            r_sym = min_t min(sum_j r_plus[j,t], sum_j r_minus[j,t])
            p     = max_t sum_j (i_E_gri[j,t] - e_E_gri[j,t])

        so their objective contribution is recoverable here without needing the
        scalars themselves (init_sol carries only (u,t)-indexed dicts).
        Returns 0.0 when both channels are off.
        """
        params = self.parameters
        T = self.time_periods
        cost = 0.0

        if params.get('enable_reserve'):
            rp = init_sol.get('r_plus') or {}
            rm = init_sol.get('r_minus') or {}
            if rp and rm:
                from compact_utility import reserve_blocks as _rblocks
                sym = params.get('reserve_product', 'symmetric') == 'symmetric'
                pi = params.get('pi_res', 0.0)
                pu, pd = params.get('pi_up', pi), params.get('pi_dn', pi)
                for blk in _rblocks(T, params.get('reserve_block_hours', 24)):
                    up = [sum(rp.get((u, t), 0.0) for u in self.players) for t in blk]
                    dn = [sum(rm.get((u, t), 0.0) for u in self.players) for t in blk]
                    if sym:
                        cost -= len(blk) * pi * max(0.0, min(min(up), min(dn)))
                    else:
                        cost -= len(blk) * (pu * max(0.0, min(up))
                                            + pd * max(0.0, min(dn)))

        if params.get('enable_peak'):
            imp = init_sol.get('i_E_gri') or {}
            exp = init_sol.get('e_E_gri') or {}
            if imp or exp:
                net = [sum(imp.get((u, t), 0.0) - exp.get((u, t), 0.0) for u in self.players)
                       for t in T]
                cost += params.get('pi_E_peak', 0.0) * max(0.0, max(net))

        return cost

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
        self.master.model.setParam("limits/time", 1800)
        # Create pricer
        pricer = LEMPricer(
            subproblems=self.subproblems,
            time_periods=self.time_periods,
            players=self.players,
            smoothing=self.smoothing
        )

        # Set incumbent value for smoothing
        if self.smoothing and self.init_sol is not None:
            from solver import calculate_column_cost, private_dispatch
            Z_INC = sum(
                calculate_column_cost(
                    p,
                    private_dispatch(self.init_sol, p),
                    self.subproblems[p].parameters,
                    self.time_periods
                )
                for p in self.players
            )
            # calculate_column_cost covers the priced columns only, and rightly so:
            # r_plus/r_minus are private vars with zero objective cost. The reserve
            # revenue and the peak penalty live on the MASTER's shared block
            # (r_sym, p), so without this term the incumbent is off by exactly that
            # amount and the smoothing schedule is driven by a wrong Z_INC.
            Z_INC += self._shared_block_cost(self.init_sol)
            pricer.Z_INC = Z_INC

        # Kept for reporting: the pricer carries the Lagrangian bound and the iteration
        # count, which Table II of the validation plan wants as measured oracle counts
        # and which a caller otherwise has no way to reach.
        self.pricer = pricer

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
                player_cols = len(self.master.model.data['vars'][player].keys())
                print(f"  {player}: {player_cols} columns")
            
            # Extract and print convex hull prices (dual prices of community balance constraints)
            print("\n" + "="*80)
            print("CONVEX HULL PRICES (Shadow Prices of Community Balance)")
            print("="*80)
            
            chp_elec = {}
            chp_heat = {}
            chp_hydro = {}
            # RAW duals, kept alongside the absolute ones below. `convex_hull_prices`
            # stores |dual| because a settlement wants a positive price, but the pricing
            # subproblem applies RC = c - pi*a and therefore needs the sign SCIP reports
            # (negative on these rows). Any offline re-solve of eq:pricing must read THIS
            # dict, not that one -- feeding it the absolute values negates the price
            # vector silently and the pricing problem still solves.
            raw_duals = {'electricity': {}, 'heat': {}, 'hydrogen': {}}

            for t in self.time_periods:
                elec_cons = self.master.model.data['cons']['community_elec_balance'][t]
                heat_cons = self.master.model.data['cons']['community_heat_balance'][t]
                hydro_cons = self.master.model.data['cons']['community_hydro_balance'][t]
                try:
                    t_elec_cons = self.master.model.getTransformedCons(elec_cons)
                    t_heat_cons = self.master.model.getTransformedCons(heat_cons)
                    t_hydro_cons = self.master.model.getTransformedCons(hydro_cons)
                    raw_duals['electricity'][t] = float(self.master.model.getDualsolLinear(t_elec_cons))
                    raw_duals['heat'][t] = float(self.master.model.getDualsolLinear(t_heat_cons))
                    raw_duals['hydrogen'][t] = float(self.master.model.getDualsolLinear(t_hydro_cons))
                    chp_elec[t] = np.abs(raw_duals['electricity'][t])
                    chp_heat[t] = np.abs(raw_duals['heat'][t])
                    chp_hydro[t] = np.abs(raw_duals['hydrogen'][t])

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

            # ---- Reserve / peak coupling prices (adding_cons.txt) ----
            # Same dual-price settlement as the energy carriers: the shadow price of
            # each homogeneous coupling row is the per-t capacity/coincidence price.
            # abs() as for the balance rows (<=0 dual on a <= row -> positive price).
            # Budget-balanced at the LP optimum. r_sym is the single symmetric
            # product shared by both row families, so its zero reduced cost gives
            #   sum_t (mu_plus[t] + mu_minus[t]) = |T| * pi_res,
            # and mu is nonzero only on rows binding at r_sym. Hence
            #   sum_u sum_t (mu_plus[t]*r_plus[u,t] + mu_minus[t]*r_minus[u,t])
            #     = r_sym * |T| * pi_res = the community reserve revenue.
            # Likewise sum_t p_peak[t] = delta_peak for the peak penalty.
            # Only present when the rows exist (reserve/peak enabled); plotting ignores them.
            cons = self.master.model.data['cons']
            if cons.get('reserve_up'):
                chp_resup, chp_resdn = {}, {}
                raw_duals['reserve_up'], raw_duals['reserve_dn'] = {}, {}
                for t in self.time_periods:
                    up_c = self.master.model.getTransformedCons(cons['reserve_up'][t])
                    dn_c = self.master.model.getTransformedCons(cons['reserve_dn'][t])
                    raw_duals['reserve_up'][t] = float(self.master.model.getDualsolLinear(up_c))
                    raw_duals['reserve_dn'][t] = float(self.master.model.getDualsolLinear(dn_c))
                    chp_resup[t] = np.abs(raw_duals['reserve_up'][t])
                    chp_resdn[t] = np.abs(raw_duals['reserve_dn'][t])
                solution['convex_hull_prices']['reserve_up'] = chp_resup
                solution['convex_hull_prices']['reserve_dn'] = chp_resdn
            if cons.get('peak'):
                chp_peak = {}
                raw_duals['peak'] = {}
                for t in self.time_periods:
                    pk_c = self.master.model.getTransformedCons(cons['peak'][t])
                    raw_duals['peak'][t] = float(self.master.model.getDualsolLinear(pk_c))
                    chp_peak[t] = np.abs(raw_duals['peak'][t])
                solution['convex_hull_prices']['peak'] = chp_peak
            solution['coupling_duals_raw'] = raw_duals

            return status, solution, obj_val, solution_by_player
        else:
            print(f"\nColumn generation failed with status: {status}")
            return status, None, None, None
    def analyze_synergy_with_convex_hull_prices(self, results: Dict, obj_val: float, community_prices: Dict):
        """
        Analyze individual vs community profits using convex hull prices
        
        Args:
            solution: Solution from column generation including convex hull prices
            obj_val: Objective value from column generation
        """
        
        print("\n" + "="*80)
        print("SYNERGY ANALYSIS WITH CONVEX HULL PRICING")
        print("="*80)
        
        # Extract convex hull prices
        chp = community_prices
        
        # # Step 1: Calculate community player profits using convex hull prices
        # print("\nSTEP 1: Computing player profits in community (using convex hull prices)")
        # print("-"*80)
        
        # # For this, we need to solve the community problem again to get detailed results
        # lem_community = LocalEnergyMarket(
        #     self.players, 
        #     self.time_periods, 
        #     self.parameters, 
        #     model_type = 'mip',
        #     dwr=False
        # )
        
        # # Solve with SCIP settings for consistency
        # from pyscipopt import SCIP_PARAMSETTING
        # # lem_community.model.hideOutput()
        # lem_community.model.optimize()
        
        # status_comm = lem_community.model.getStatus()
        # if status_comm != "optimal":
        #     print(f"âš ï¸ Community optimization failed: {status_comm}")
        #     return
        
        # _, results_comm = solve_and_extract_results(lem_community.model)
        
        # Calculate player profits with convex hull prices
        player_profits_chp = self._calculate_player_profits_with_chp(
            results, chp
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
                dwr=False,
                mipsolver=self.mipsolver
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
    def _calculate_player_profits_with_chp(self, results: Dict, chp: Dict) -> Dict:
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
                'utility': 0.0,
                'reserve_revenue': 0.0,   # reserve capacity payment (adding_cons.txt)
                'peak_cost': 0.0,         # peak coincidence charge
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
                c_sto_E = self.parameters.get('c_sto_E', np.inf)
                c_sto_G = self.parameters.get('c_sto_G', np.inf)
                c_sto_H = self.parameters.get('c_sto_H', np.inf)
                nu_ch = self.parameters.get('nu_ch', 0.9)
                nu_dis = self.parameters.get('nu_dis', 0.9)
                
                if 'b_ch_E' in results and (u,t) in results['b_ch_E']:
                    profit['storage_cost'] += results['b_ch_E'][u,t] * c_sto_E * nu_ch
                if 'b_dis_E' in results and (u,t) in results['b_dis_E']:
                    profit['storage_cost'] += results['b_dis_E'][u,t] * c_sto_E * (1/nu_dis)
                
                if 'b_ch_G' in results and (u,t) in results['b_ch_G']:
                    profit['storage_cost'] += results['b_ch_G'][u,t] * c_sto_G * nu_ch
                if 'b_dis_G' in results and (u,t) in results['b_dis_G']:
                    profit['storage_cost'] += results['b_dis_G'][u,t] * c_sto_G * (1/nu_dis)
                
                if 'b_ch_H' in results and (u,t) in results['b_ch_H']:
                    profit['storage_cost'] += results['b_ch_H'][u,t] * c_sto_H * nu_ch
                if 'b_dis_H' in results and (u,t) in results['b_dis_H']:
                    profit['storage_cost'] += results['b_dis_H'][u,t] * c_sto_H * (1/nu_dis)
                
                # 5. Startup costs
                if 'z_su_G' in results and (u,t) in results['z_su_G']:
                    profit['startup_cost'] += results['z_su_G'][u,t] * self.parameters.get(f'c_su_G_{u}', np.inf)
                if 'z_su_H' in results and (u,t) in results['z_su_H']:
                    profit['startup_cost'] += results['z_su_H'][u,t] * self.parameters.get(f'c_su_H_{u}', np.inf)
                # 6. Utility
                if 'nfl_d' in results and (u, 'elec', t) in results['nfl_d']:
                    demand = results['nfl_d'][u, 'elec', t]
                    if demand > 0:
                        profit['utility'] += demand * self.parameters.get(f'u_E_{u}_{t}', 0)
                if 'nfl_d' in results and (u, 'hydro', t) in results['nfl_d']:
                    demand = results['nfl_d'][u, 'hydro', t]
                    if demand > 0:
                        profit['utility'] += demand * self.parameters.get(f'u_G_{u}_{t}', 0)
                if 'nfl_d' in results and (u, 'heat', t) in results['nfl_d']:
                    demand = results['nfl_d'][u, 'heat', t]
                    if demand > 0:
                        profit['utility'] += demand * self.parameters.get(f'u_H_{u}_{t}', 0)

                # 7. Reserve capacity payment (adding_cons.txt): each provider is
                #    paid the reserve price x its offered up/down headroom (MIP qty).
                if 'reserve_up' in chp:
                    r_plus = results.get('r_plus', {}).get((u, t), 0.0)
                    r_minus = results.get('r_minus', {}).get((u, t), 0.0)
                    profit['reserve_revenue'] += r_plus * chp['reserve_up'][t]
                    profit['reserve_revenue'] += r_minus * chp['reserve_dn'][t]

                # 8. Peak coincidence charge: net grid import at t x peak price.
                #    Net importers at the binding hour pay; net exporters are credited.
                if 'peak' in chp:
                    i_gri = results.get('i_E_gri', {}).get((u, t), 0.0)
                    e_gri = results.get('e_E_gri', {}).get((u, t), 0.0)
                    profit['peak_cost'] += (i_gri - e_gri) * chp['peak'][t]
            # Calculate net profit
            profit['net_profit'] = (
                profit['grid_revenue'] +
                profit['community_revenue'] -
                profit['grid_cost'] -
                profit['community_cost'] -
                profit['production_cost'] -
                profit['storage_cost'] -
                profit['startup_cost'] +
                profit['utility'] +
                profit['reserve_revenue'] -
                profit['peak_cost']
            )

            player_profits[u] = profit

        return player_profits

    def compute_owen_allocation(self, v_mip: float) -> Dict:
        """
        Owen (1975) linear-production-game allocation from the Dantzig-Wolfe master.

        The shadow price sigma_u of player u's convexity constraint (sum_k lambda_{u,k}=1)
        is u's Owen value. Because EVERY coupling row (balance + reserve + peak) is
        homogeneous (RHS 0), the LP dual objective reduces to sum_u 1*sigma_u, so by strong
        duality  sum_u sigma_u = v^CHP  (the master objective) exactly — reserve/peak are
        already baked into sigma_u, no separate term is needed.

        Since v^CHP generally differs from v^MIP, the raw Owen point is NOT efficient for
        the integer game (budget balance is broken by the duality gap). Spreading the gap
        equally over the N players restores efficiency and yields a weak eps-core cost
        allocation (Liu-Qi-Xu 2016 style):

            gap    = v^CHP - v^MIP
            owen_u = sigma_u - gap / N            =>   sum_u owen_u = v^MIP,   eps = |gap|/N

        Args:
            v_mip: optimal objective of the grand-coalition MIP (same cost-min sign
                   convention as the CHP master objective).

        Returns dict with sigma (raw Owen point), owen (gap-corrected), and diagnostics.
        Must be called after solve() (master duals must be available).
        """
        m = self.master.model
        sigma = {}
        for u in self.players:
            conv_cons = m.getTransformedCons(m.data['cons']['convexity'][u])
            sigma[u] = m.getDualsolLinear(conv_cons)

        v_chp = m.getObjVal()
        N = len(self.players)
        gap = v_chp - v_mip
        owen = {u: sigma[u] - gap / N for u in self.players}

        sum_sigma = sum(sigma.values())
        result = {
            'sigma': sigma,               # raw Owen point (sums to v^CHP)
            'owen': owen,                 # gap-corrected (sums to v^MIP)
            'v_chp': v_chp,
            'v_mip': v_mip,
            'gap': gap,
            'eps': abs(gap) / N,
            'sum_sigma': sum_sigma,       # ~= v^CHP  (LP strong-duality check)
            'sum_owen': sum(owen.values())  # ~= v^MIP (efficiency check)
        }

        print("\n" + "="*80)
        print("OWEN ALLOCATION (LP production-game value + gap correction)")
        print("="*80)
        print(f"  v^CHP = {v_chp:.6f}   sum_u sigma_u = {sum_sigma:.6f}   (diff = {abs(v_chp-sum_sigma):.3e})")
        print(f"  v^MIP = {v_mip:.6f}   duality gap = {gap:.6f}   eps = |gap|/N = {result['eps']:.6f}")
        print(f"  sum_u owen_u = {result['sum_owen']:.6f}   (diff from v^MIP = {abs(result['sum_owen']-v_mip):.3e})")
        print(f"  {'Player':>8} {'sigma_u (Owen)':>16} {'owen_u (eps-core)':>18}")
        for u in self.players:
            print(f"  {u:>8} {sigma[u]:>16.4f} {owen[u]:>18.4f}")
        return result

    def compare_owen_vs_chp(self, owen_result: Dict, chp_profits: Dict) -> Dict:
        """
        Owen vs. price-based (CHP) allocation comparison (owen_chp.txt).

        Both are corrections of the raw Owen point sigma_j (which over-distributes,
        summing to v^CHP):
          - Owen eps-core:  sigma_j - eps           (UNIFORM gap split, eps=|gap|/N)
          - CHP:            sigma_j - Delta_j        (NON-UNIFORM: Delta_j=sigma_j-chi_j^CHP)
        Reported in PROFIT (payoff) convention. sum_j Delta_j = gap = eps*N. A player
        with Delta_j > eps is charged MORE than the fair uniform share by the CHP rule
        (the "nonconvexity loser", typically the electrolyzer); Delta_j < eps means less.

        Args:
            owen_result: dict from compute_owen_allocation().
            chp_profits: dict from _calculate_player_profits_with_chp() (per-player
                         'net_profit' at CHP prices on the MIP dispatch).
        """
        sigma_cost = owen_result['sigma']
        eps = owen_result['eps']
        gap_value = -owen_result['gap']   # value-convention duality gap (>=0), = sum_j Delta_j

        rows, sum_owen_p, sum_chp_p, sum_delta = {}, 0.0, 0.0, 0.0
        for u in self.players:
            owen_sigma_p = -sigma_cost[u]                 # Owen payoff (raw)
            owen_eps_p = owen_sigma_p - eps               # uniform-corrected (weak eps-core)
            chp_p = chp_profits[u]['net_profit']          # CHP payoff (budget-balanced)
            delta = owen_sigma_p - chp_p                  # nonconvexity loss (doc Delta_j)
            rows[u] = {'owen_sigma': owen_sigma_p, 'owen_eps_core': owen_eps_p,
                       'chp': chp_p, 'delta': delta, 'delta_minus_eps': delta - eps}
            sum_owen_p += owen_sigma_p; sum_chp_p += chp_p; sum_delta += delta

        print("\n" + "="*80)
        print("OWEN vs CHP ALLOCATION  (profit convention; Delta_j = sigma_j - chi_j^CHP)")
        print("="*80)
        print(f"  eps = |gap|/N = {eps:.4f}   sum_j Delta_j = {sum_delta:.4f}   (= gap = {gap_value:.4f})")
        print(f"  sum Owen(sigma) = {sum_owen_p:.4f} (=v^CHP)   sum CHP = {sum_chp_p:.4f} (=v^MIP up to reserve/peak residual)")
        print(f"  {'Player':>7} {'Owen sigma':>12} {'Owen-eps':>12} {'CHP':>12} {'Delta_j':>10} {'Delta-eps':>10}")
        for u in self.players:
            r = rows[u]
            flag = "  <- loses more" if r['delta_minus_eps'] > 1e-6 else ""
            print(f"  {u:>7} {r['owen_sigma']:>12.4f} {r['owen_eps_core']:>12.4f} "
                  f"{r['chp']:>12.4f} {r['delta']:>10.4f} {r['delta_minus_eps']:>10.4f}{flag}")
        return {'eps': eps, 'gap': gap_value, 'sum_delta': sum_delta,
                'sum_owen': sum_owen_p, 'sum_chp': sum_chp_p, 'rows': rows}

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
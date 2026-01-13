"""
Solver components for column generation: Subproblem and Master Problem
"""
from pyscipopt import Model, quicksum
import sys
sys.path.append('/mnt/project')
import numpy as np
from compact import LocalEnergyMarket, solve_and_extract_results
# from compact_debug import LocalEnergyMarket, solve_and_extract_results
from typing import Dict, List, Tuple


class PlayerSubproblem:
    """
    Individual player's subproblem for column generation
    Reuses LocalEnergyMarket class with single player
    """
    def __init__(self, player: str, time_periods: List[int], parameters: Dict, model_type: str):
        """
        Create subproblem for a single player
        
        Args:
            player: Single player ID
            time_periods: List of time periods
            parameters: Model parameters
            isLP: Whether to use LP relaxation (True for column generation)
        """
        self.player = player
        self.time_periods = time_periods
        self.parameters = parameters
        if model_type not in ('mip', 'lp'):
            raise ValueError("model_type must be either 'mip' or 'lp', got: {}".format(model_type))
        self.model_type = model_type
        # Create LocalEnergyMarket with single player and dwr=True
        # dwr=True removes community balance constraints
        self.lem = LocalEnergyMarket(
            players=[player],
            time_periods=time_periods,
            parameters=parameters,
            model_type=model_type,
            dwr=True  # This removes community balance constraints
        )
        
        self.model = self.lem.model
        # Disable output for subproblems
        self.model.hideOutput()
        
    def solve_pricing(self, dual_elec: Dict[int, float], 
                      dual_heat: Dict[int, float], 
                      dual_hydro: Dict[int, float],
                      dual_convexity: float,
                      farkas: bool=False) -> Tuple[float, Dict]:
        """
        Solve pricing problem with modified objective based on dual prices
        
        Args:
            dual_elec: Dual prices for electricity community balance constraints
            dual_heat: Dual prices for heat community balance constraints
            dual_hydro: Dual prices for hydrogen community balance constraints
            dual_convexity: Dual price for convexity constraint (sum lambda = 1)

        Returns:
            tuple: (reduced_cost, solution_dict)
        """
        # Free transform to allow objective modification
        self.model.freeTransform()
        
        # Build modified objective with dual prices
        new_obj = 0
        u = self.player

        # Original grid costs
        for t in self.time_periods:
            if not farkas:
                # Electricity grid costs
                if (u, t) in self.lem.i_E_gri:
                    pi_import = self.parameters.get(f'pi_E_gri_import_{t}', np.inf)
                    new_obj += pi_import * self.lem.i_E_gri[u, t]
                if (u, t) in self.lem.e_E_gri:
                    pi_export = self.parameters.get(f'pi_E_gri_export_{t}', np.inf)
                    new_obj -= pi_export * self.lem.e_E_gri[u, t]
                
                # Heat grid costs
                if (u, t) in self.lem.i_H_gri:
                    pi_import = self.parameters.get(f'pi_H_gri_import_{t}', np.inf)
                    new_obj += pi_import * self.lem.i_H_gri[u, t]
                if (u, t) in self.lem.e_H_gri:
                    pi_export = self.parameters.get(f'pi_H_gri_export_{t}', np.inf)
                    new_obj -= pi_export * self.lem.e_H_gri[u, t]
                
                # Hydrogen grid costs
                if (u, t) in self.lem.i_G_gri:
                    pi_import = self.parameters.get(f'pi_G_gri_import_{t}', np.inf)
                    new_obj += pi_import * self.lem.i_G_gri[u, t]
                if (u, t) in self.lem.e_G_gri:
                    pi_export = self.parameters.get(f'pi_G_gri_export_{t}', np.inf)
                    new_obj -= pi_export * self.lem.e_G_gri[u, t]
                
                # Production costs
                if (u, 'res', t) in self.lem.p:
                    c_res = self.parameters.get(f'c_res_{u}', np.inf)
                    new_obj += c_res * self.lem.p[u, 'res', t]
                if (u, 'hp', t) in self.lem.p:
                    c_hp = self.parameters.get(f'c_hp_{u}', np.inf)
                    new_obj += c_hp * self.lem.p[u, 'hp', t]
                if (u, 'els', t) in self.lem.p:
                    c_els = self.parameters.get(f'c_els_{u}', np.inf)
                    new_obj += c_els * self.lem.p[u, 'els', t]
                
                # Startup costs
                if (u, t) in self.lem.z_su_G:
                    c_su = self.parameters.get(f'c_su_G_{u}', np.inf)
                    new_obj += c_su * self.lem.z_su_G[u, t]
                # Storage costs
                if u in self.lem.params["players_with_elec_storage"]:
                    c_E_sto = self.parameters.get(f'c_sto_E_{u}', np.inf)
                    new_obj += c_E_sto * self.lem.b_ch_E[u, t]
                    new_obj += c_E_sto * self.lem.b_dis_E[u, t]
                if u in self.lem.params["players_with_hydro_storage"]:
                    c_G_sto = self.parameters.get(f'c_sto_G_{u}', np.inf)
                    new_obj += c_G_sto * self.lem.b_ch_G[u, t]
                    new_obj += c_G_sto * self.lem.b_dis_G[u, t]
                if u in self.lem.params["players_with_heat_storage"]:
                    c_H_sto = self.parameters.get(f'c_sto_H_{u}', np.inf)
                    new_obj += c_H_sto * self.lem.b_ch_H[u, t]
                    new_obj += c_H_sto * self.lem.b_dis_H[u, t]
            # Community trading with dual prices
            # For reduced cost: original_cost - dual_price * coefficient
            # Electricity
            if (u, t) in self.lem.i_E_com:
                # Import from community: coefficient is +1 in balance
                new_obj -= dual_elec[t] * self.lem.i_E_com[u, t]
            if (u, t) in self.lem.e_E_com:
                # Export to community: coefficient is -1 in balance
                new_obj += dual_elec[t] * self.lem.e_E_com[u, t]
            
            # Heat
            if (u, t) in self.lem.i_H_com:
                new_obj -= dual_heat[t] * self.lem.i_H_com[u, t]
            if (u, t) in self.lem.e_H_com:
                new_obj += dual_heat[t] * self.lem.e_H_com[u, t]
            
            # Hydrogen
            if (u, t) in self.lem.i_G_com:
                new_obj -= dual_hydro[t] * self.lem.i_G_com[u, t]
            if (u, t) in self.lem.e_G_com:
                new_obj += dual_hydro[t] * self.lem.e_G_com[u, t]
        
        # Set modified objective
        try:
            self.model.setObjective(new_obj, "minimize")
        except:
            raise Exception(f"Error setting objective: {new_obj}")
        # Solve
        self.model.optimize()
        status = self.model.getStatus()

        if status == "optimal":
            reduced_cost = self.model.getObjVal() - dual_convexity
            # Extract solution
            _, results = solve_and_extract_results(self.model)
            
            return reduced_cost, results, self.model.getObjVal()
        else:
            return float('inf'), None, None


class MasterProblem:
    """
    Restricted Master Problem for column generation
    Contains only community balance constraints
    """
    def __init__(self, players: List[str], time_periods: List[int]):
        """
        Initialize master problem
        
        Args:
            players: List of player IDs
            time_periods: List of time periods
        """
        self.players = players
        self.time_periods = time_periods
        self.model = Model("RMP_LocalEnergyMarket")
        self.model.data = {}
        # Storage for variables and constraints
        self.model.data['vars'] = {
            player:{} for player in players
        }  # {(player, col_idx): {'var': var, 'solution': dict}}
        
        # Data dictionary for storing constraints (similar to LocalEnergyMarket)
        self.model.data['cons'] = {
            'community_elec_balance': {},
            'community_heat_balance': {},
            'community_hydro_balance': {},
            'convexity': {}
        }
    
    def _create_master_constraints(self):
        """
        Create master problem constraints:
        1. Community balance constraints (linking constraints)
        2. Convexity constraints (one per player)
        """
        # Convexity constraints: sum of lambdas = 1 for each player
        for u in self.players:
            initial_var = self.model.data["vars"][u][0]["var"]
            cons = self.model.addCons(
                quicksum([initial_var]) - 1 == 0, 
                name=f"convexity_{u}", modifiable=True
            )
            self.model.data['cons']['convexity'][u] = cons
        
        # Community balance constraints (no slack variables)
        # Farkas pricing will ensure feasibility
        for t in self.time_periods:
            elec_expr, heat_expr, hydro_expr = 0, 0, 0
            for u in self.players:
                var = self.model.data["vars"][u][0]["var"]
                solution = self.model.data["vars"][u][0]["solution"]
                """
                여기선 player u와 t가 존재하지 않을 수도 있고, 그것만 확인하는거니 0.0
                """
                e_E_com_val = solution.get('e_E_com', {}).get((u, t), 0.0)
                i_E_com_val = solution.get('i_E_com', {}).get((u, t), 0.0)
                elec_expr += var * (i_E_com_val - e_E_com_val)
                e_H_com_val = solution.get('e_H_com', {}).get((u, t), 0.0)
                i_H_com_val = solution.get('i_H_com', {}).get((u, t), 0.0)
                heat_expr += var * (i_H_com_val - e_H_com_val)
                e_G_com_val = solution.get('e_G_com', {}).get((u, t), 0.0)
                i_G_com_val = solution.get('i_G_com', {}).get((u, t), 0.0)
                hydro_expr += var * (i_G_com_val - e_G_com_val)
            # # Add artificial variables (positive and negative slack with big-M penalty)
            # BIG_M = 1e6  # Large penalty cost
            # art_elec_pos = self.model.addVar(f"art_elec_pos_{t}", vtype="C", lb=0, obj=BIG_M)
            # art_elec_neg = self.model.addVar(f"art_elec_neg_{t}", vtype="C", lb=0, obj=BIG_M)
            # art_heat_pos = self.model.addVar(f"art_heat_pos_{t}", vtype="C", lb=0, obj=BIG_M)
            # art_heat_neg = self.model.addVar(f"art_heat_neg_{t}", vtype="C", lb=0, obj=BIG_M)
            # art_hydro_pos = self.model.addVar(f"art_hydro_pos_{t}", vtype="C", lb=0, obj=BIG_M)
            # art_hydro_neg = self.model.addVar(f"art_hydro_neg_{t}", vtype="C", lb=0, obj=BIG_M)
            art_elec_pos, art_elec_neg, art_heat_pos, art_heat_neg, art_hydro_pos, art_hydro_neg = 0, 0, 0, 0, 0, 0
            if t == 0:
                print(f"\n  Time t=0 balance check:")
                print(f"    Elec imbalance: {elec_expr}")
                print(f"    Heat imbalance: {heat_expr}") 
                print(f"    Hydro imbalance: {hydro_expr}")
                # print(f"    Adding artificial variables with penalty {BIG_M}")
            # Electricity balance: sum(i_E_com - e_E_com) = 0
            cons = self.model.addCons(
                elec_expr + art_elec_pos - art_elec_neg == 0,
                name=f"community_elec_balance_{t}", modifiable=True
            )
            self.model.data['cons']['community_elec_balance'][t] = cons
            
            # Heat balance: sum(i_H_com - e_H_com) = 0
            cons = self.model.addCons(
                heat_expr + art_heat_pos - art_heat_neg == 0,
                name=f"community_heat_balance_{t}", modifiable=True
            )
            self.model.data['cons']['community_heat_balance'][t] = cons
            
            # Hydrogen balance: sum(i_G_com - e_G_com) = 0
            cons = self.model.addCons(
                hydro_expr + art_hydro_pos - art_hydro_neg == 0,
                name=f"community_hydro_balance_{t}", modifiable=True
            )
            self.model.data['cons']['community_hydro_balance'][t] = cons
        print(f"  Added {len(self.time_periods)} community balance constraints (with artificial vars)")
        # print(f"  Artificial variable penalty: {BIG_M}")
        print("=== Master Constraints Created ===\n")
    def _add_initial_columns(self, subproblems: Dict[str, 'PlayerSubproblem'], init_sol: Dict = None):
        """
        Generate initial columns by solving each subproblem independently
        
        Args:
            subproblems: Dictionary of player subproblems
        """
        print("\n=== Generating Initial Columns ===")
        
        # Zero dual prices for initial solve
        zero_duals_elec = {t: 0.0 for t in self.time_periods}
        zero_duals_heat = {t: 0.0 for t in self.time_periods}
        zero_duals_hydro = {t: 0.0 for t in self.time_periods}
        zero_duals_convexity = {player: 0.0 for player in self.players}
        
        for player in self.players:
            print(f"Generating initial column for player {player}...")
            
            if not init_sol:
                # Solve subproblem with zero dual prices
                reduced_cost, solution, obj_val = subproblems[player].solve_pricing(
                    zero_duals_elec, zero_duals_heat, zero_duals_hydro, zero_duals_convexity[player]
                )
            else:
                # Solve subproblem with initial solution
                solution = {k:{key:value for key, value in init_sol[k].items() if key[0] == player } for k,v in init_sol.items()}
            
            if solution is None:
                raise Exception(f"  WARNING: Could not generate initial column for {player}")
            
            col_idx = 0
            var_name = f"lambda_{player}_{col_idx}"
        
            # Calculate column cost (original objective value)
            cost = calculate_column_cost(player, solution, subproblems[player].parameters, self.time_periods)
            
            # Create variable
            new_var = self.model.addVar(
                name=var_name,
                vtype="C",
                lb=0.0,
                obj=cost
            )
        
            # Store variable and solution
            self.model.data['vars'][player][col_idx] = {
                'var': new_var,
                'solution': solution
            }
            print(f"  {player}: Initial column added (cost={cost:.4f})")
    
    def solve(self):
        """Solve the restricted master problem"""
        self.model.optimize()
        return self.model.getStatus()
    
    def get_objective_value(self):
        """Get current objective value"""
        return self.model.getObjVal()
    
    def get_solution(self) -> Dict:
        """
        Extract solution from master problem
        Combines columns based on lambda values
        """
        solution = {}
        solution_by_player = {player:{} for player in self.players}
        
        for player in self.players:
            for idx, col_data in self.model.data['vars'][player].items():
                lambda_val = self.model.getVal(col_data['var'])
                
                if lambda_val > 1e-6:  # Only consider active columns
                    col_solution = col_data['solution']
                    
                    # Add weighted contribution of this column to solution
                    for var_name, var_dict in col_solution.items():
                        if var_name not in solution:
                            solution[var_name] = {}
                        
                        for key, value in var_dict.items():
                            if key not in solution[var_name]:
                                solution[var_name][key] = 0.0
                            solution[var_name][key] += lambda_val * value
                    solution_by_player[player][idx] = (lambda_val, col_solution)
        
        return solution, solution_by_player

def calculate_column_cost(player: str, solution: Dict, params: Dict, time_periods: List[int]) -> float:
    """Calculate cost for a column"""
    cost = 0.0

    for t in time_periods:
        # Grid costs
        """
        solution에서 get하는건 없을 경우 0.0 (player type별로 없을수도 있으니)
        """
        e_E_gri = solution.get('e_E_gri', {}).get((player, t), 0.0)
        i_E_gri = solution.get('i_E_gri', {}).get((player, t), 0.0)
        cost += i_E_gri * params.get(f'pi_E_gri_import_{t}', np.inf)
        cost -= e_E_gri * params.get(f'pi_E_gri_export_{t}', np.inf)
        
        e_H_gri = solution.get('e_H_gri', {}).get((player, t), 0.0)
        i_H_gri = solution.get('i_H_gri', {}).get((player, t), 0.0)
        cost += i_H_gri * params.get(f'pi_H_gri_import_{t}', np.inf)
        cost -= e_H_gri * params.get(f'pi_H_gri_export_{t}', np.inf)
        
        e_G_gri = solution.get('e_G_gri', {}).get((player, t), 0.0)
        i_G_gri = solution.get('i_G_gri', {}).get((player, t), 0.0)
        cost += i_G_gri * params.get(f'pi_G_gri_import_{t}', np.inf)
        cost -= e_G_gri * params.get(f'pi_G_gri_export_{t}', np.inf)
        
        # Production and startup costs
        p_res = solution.get('p', {}).get((player, 'res', t), 0.0)
        p_hp = solution.get('p', {}).get((player, 'hp', t), 0.0)
        p_els = solution.get('p', {}).get((player, 'els', t), 0.0)
        z_su_G = solution.get('z_su_G', {}).get((player, t), 0.0)
        z_su_H = solution.get('z_su_H', {}).get((player, t), 0.0)

        cost += p_res * params.get(f'c_res_{player}', 0.0)
        cost += p_hp * params.get(f'c_hp_{player}', 0.0)
        cost += p_els * params.get(f'c_els_{player}', 0.0)
        cost += z_su_G * params.get(f'c_su_G_{player}', 0.0)
        cost += z_su_H * params.get(f'c_su_H_{player}', 0.0)
        # Storage costs
        b_dis_E, b_ch_E = solution.get('b_dis_E', {}).get((player, t), 0.0), solution.get('b_ch_E', {}).get((player, t), 0.0)
        b_dis_G, b_ch_G = solution.get('b_dis_G', {}).get((player, t), 0.0), solution.get('b_ch_G', {}).get((player, t), 0.0)
        b_dis_H, b_ch_H = solution.get('b_dis_H', {}).get((player, t), 0.0), solution.get('b_ch_H', {}).get((player, t), 0.0)
        nu_ch_E, nu_dis_E = params.get(f'nu_ch_E', np.inf), params.get(f'nu_dis_E', np.inf)
        nu_ch_G, nu_dis_G = params.get(f'nu_ch_G', np.inf), params.get(f'nu_dis_G', np.inf)
        nu_ch_H, nu_dis_H = params.get(f'nu_ch_H', np.inf), params.get(f'nu_dis_H', np.inf)

        cost += b_dis_E * params.get(f'c_sto_E_{player}', 0.0) * (1/nu_dis_E)
        cost += b_ch_E * params.get(f'c_sto_E_{player}', 0.0) * nu_ch_E
        cost += b_dis_G * params.get(f'c_sto_G_{player}', 0.0) * (1/nu_dis_G)
        cost += b_ch_G * params.get(f'c_sto_G_{player}', 0.0) * nu_ch_G
        cost += b_dis_H * params.get(f'c_sto_H_{player}', 0.0) * (1/nu_dis_H)
        cost += b_ch_H * params.get(f'c_sto_H_{player}', 0.0) * nu_ch_H

    if (cost == np.inf) or (cost == -np.inf) or (np.isnan(cost)):
        raise Exception(f"Cost is infinite for player {player} at time {t}")
    return cost
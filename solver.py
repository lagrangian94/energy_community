"""
Solver components for column generation: Subproblem and Master Problem

_add_initial_columns, solve_pricing, calculate_column_cost 세 개 다 업데이트하는거 까먹지 말 것.
"""
from pyscipopt import Model, quicksum
import sys
sys.path.append('/mnt/project')
import numpy as np
import tempfile
import os
# from compact import LocalEnergyMarket, solve_and_extract_results
from compact_utility import LocalEnergyMarket, solve_and_extract_results, solve_and_extract_results_highs
from typing import Dict, List, Tuple


class PlayerSubproblem:
    """
    Individual player's subproblem for column generation
    Reuses LocalEnergyMarket class with single player
    """
    def __init__(self, player: str, time_periods: List[int], parameters: Dict, model_type: str, mipsolver: str = None):
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

        self.mipsolver = mipsolver
        if mipsolver == 'highs':
            self._init_highs_model()

    def _init_highs_model(self):
        """
        Export SCIP model to MPS once, load into HiGHS, and build column maps
        for efficient objective updates in subsequent pricing iterations.
        """
        import highspy

        # Export SCIP model to MPS (need to optimize once first to have a valid problem)
        mps_path = tempfile.mktemp(suffix='.mps')
        self.model.writeProblem(mps_path)

        # Create HiGHS instance and load MPS
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(mps_path)

        # Clean up temp file
        os.remove(mps_path)

        # Build SCIP variable name → HiGHS column index map
        num_cols = h.getNumCol()
        self._highs_col_map = {}
        for j in range(num_cols):
            name = h.getColName(j)
            if isinstance(name, tuple):
                name = name[1]
            self._highs_col_map[name] = j

        self._highs_num_cols = num_cols

        # Build community variable column indices for fast access
        u = self.player
        self._com_col_indices = {}
        com_var_types = ['e_E_com', 'i_E_com', 'e_H_com', 'i_H_com', 'e_G_com', 'i_G_com']
        for var_type in com_var_types:
            lem_dict = getattr(self.lem, var_type)
            for t in self.time_periods:
                if (u, t) in lem_dict:
                    scip_var = lem_dict[u, t]
                    scip_name = scip_var.name
                    if scip_name in self._highs_col_map:
                        self._com_col_indices[(var_type, t)] = self._highs_col_map[scip_name]

        # Build base cost vector from SCIP model's objective coefficients
        # We need to solve the pricing once with zero duals using SCIP to get the objective set
        # Instead, compute base costs directly from parameters (matching solve_pricing logic)
        self._compute_highs_base_costs(h)

        self._highs = h
        self._last_was_farkas = False

    def _compute_highs_base_costs(self, h):
        """
        Compute base objective costs for all variables from parameters,
        matching the non-farkas, non-dual portion of solve_pricing.
        """
        u = self.player
        costs = np.zeros(self._highs_num_cols)

        def _set_cost(var_dict, key, cost_val):
            if key in var_dict:
                scip_name = var_dict[key].name
                if scip_name in self._highs_col_map:
                    idx = self._highs_col_map[scip_name]
                    costs[idx] += cost_val

        for t in self.time_periods:
            # Grid costs
            if (u, t) in self.lem.i_E_gri:
                _set_cost(self.lem.i_E_gri, (u, t), self.parameters.get(f'pi_E_gri_import_{t}', np.inf))
            if (u, t) in self.lem.e_E_gri:
                _set_cost(self.lem.e_E_gri, (u, t), -self.parameters.get(f'pi_E_gri_export_{t}', np.inf))
            if (u, 'elec', t) in self.lem.nfl_d:
                _set_cost(self.lem.nfl_d, (u, 'elec', t), -self.parameters.get(f'u_E_{u}_{t}', 0.0))

            if (u, t) in self.lem.i_H_gri:
                _set_cost(self.lem.i_H_gri, (u, t), self.parameters.get(f'pi_H_gri_import_{t}', np.inf))
            if (u, t) in self.lem.e_H_gri:
                _set_cost(self.lem.e_H_gri, (u, t), -self.parameters.get(f'pi_H_gri_export_{t}', np.inf))
            if (u, 'heat', t) in self.lem.nfl_d:
                _set_cost(self.lem.nfl_d, (u, 'heat', t), -self.parameters.get(f'u_H_{u}_{t}', 0.0))

            if (u, t) in self.lem.i_G_gri:
                _set_cost(self.lem.i_G_gri, (u, t), self.parameters.get(f'pi_G_gri_import_{t}', np.inf))
            if (u, t) in self.lem.e_G_gri:
                _set_cost(self.lem.e_G_gri, (u, t), -self.parameters.get(f'pi_G_gri_export_{t}', np.inf))
            if (u, 'hydro', t) in self.lem.nfl_d:
                _set_cost(self.lem.nfl_d, (u, 'hydro', t), -self.parameters.get(f'u_G_{u}_{t}', 0.0))

            # Production costs
            if (u, 'res', t) in self.lem.p:
                _set_cost(self.lem.p, (u, 'res', t), self.parameters.get(f'c_res_{u}', np.inf))
            if (u, 'hp', t) in self.lem.p:
                _set_cost(self.lem.p, (u, 'hp', t), self.parameters.get(f'c_hp_{u}', np.inf))
            if (u, 'els', t) in self.lem.p:
                _set_cost(self.lem.p, (u, 'els', t), self.parameters.get(f'c_els_{u}', np.inf))

            # Startup costs
            if (u, t) in self.lem.z_su_G:
                _set_cost(self.lem.z_su_G, (u, t), self.parameters.get(f'c_su_G_{u}', np.inf))
            if (u, t) in self.lem.z_su_H:
                _set_cost(self.lem.z_su_H, (u, t), self.parameters.get(f'c_su_H_{u}', np.inf))

            # Storage costs
            if u in self.lem.params["players_with_elec_storage"]:
                c_sto_E = self.parameters.get(f'c_sto_E_{u}', np.inf)
                nu_ch = self.parameters.get(f'nu_ch_E', np.inf)
                nu_dis = self.parameters.get(f'nu_dis_E', np.inf)
                if (u, t) in self.lem.b_ch_E:
                    _set_cost(self.lem.b_ch_E, (u, t), c_sto_E * nu_ch)
                if (u, t) in self.lem.b_dis_E:
                    _set_cost(self.lem.b_dis_E, (u, t), c_sto_E * (1/nu_dis))

            if u in self.lem.params["players_with_hydro_storage"]:
                c_sto_G = self.parameters.get(f'c_sto_G_{u}', np.inf)
                nu_ch = self.parameters.get(f'nu_ch_G', np.inf)
                nu_dis = self.parameters.get(f'nu_dis_G', np.inf)
                if (u, t) in self.lem.b_ch_G:
                    _set_cost(self.lem.b_ch_G, (u, t), c_sto_G * nu_ch)
                if (u, t) in self.lem.b_dis_G:
                    _set_cost(self.lem.b_dis_G, (u, t), c_sto_G * (1/nu_dis))

            if u in self.lem.params["players_with_heat_storage"]:
                c_sto_H = self.parameters.get(f'c_sto_H_{u}', np.inf)
                nu_ch = self.parameters.get(f'nu_ch_H', np.inf)
                nu_dis = self.parameters.get(f'nu_dis_H', np.inf)
                if (u, t) in self.lem.b_ch_H:
                    _set_cost(self.lem.b_ch_H, (u, t), c_sto_H * nu_ch)
                if (u, t) in self.lem.b_dis_H:
                    _set_cost(self.lem.b_dis_H, (u, t), c_sto_H * (1/nu_dis))

        self._highs_base_costs = costs

    def _solve_pricing_highs(self, dual_elec, dual_heat, dual_hydro, dual_convexity, farkas=False):
        """
        Solve pricing subproblem using HiGHS with efficient objective updates.
        Only community variable costs change per iteration; base costs are precomputed.
        """
        from highspy import HighsModelStatus

        h = self._highs
        u = self.player

        if farkas:
            # Farkas mode: set all costs to 0, then only apply dual terms on community vars
            if not self._last_was_farkas:
                # Reset all columns to zero cost
                zero_costs = np.zeros(self._highs_num_cols, dtype=np.float64)
                indices = np.arange(self._highs_num_cols, dtype=np.int32)
                h.changeColsCost(self._highs_num_cols, indices, zero_costs)
            self._last_was_farkas = True

            # Apply dual terms on community variables only
            for t in self.time_periods:
                # i_E_com: coefficient is -dual (import from community)
                key = ('i_E_com', t)
                if key in self._com_col_indices:
                    h.changeColCost(self._com_col_indices[key], -dual_elec[t])
                # e_E_com: coefficient is +dual (export to community)
                key = ('e_E_com', t)
                if key in self._com_col_indices:
                    h.changeColCost(self._com_col_indices[key], dual_elec[t])

                key = ('i_H_com', t)
                if key in self._com_col_indices:
                    h.changeColCost(self._com_col_indices[key], -dual_heat[t])
                key = ('e_H_com', t)
                if key in self._com_col_indices:
                    h.changeColCost(self._com_col_indices[key], dual_heat[t])

                key = ('i_G_com', t)
                if key in self._com_col_indices:
                    h.changeColCost(self._com_col_indices[key], -dual_hydro[t])
                key = ('e_G_com', t)
                if key in self._com_col_indices:
                    h.changeColCost(self._com_col_indices[key], dual_hydro[t])
        else:
            if self._last_was_farkas:
                # Restore all base costs after farkas mode
                indices = np.arange(self._highs_num_cols, dtype=np.int32)
                h.changeColsCost(self._highs_num_cols, indices, self._highs_base_costs.copy())
            self._last_was_farkas = False

            # Update community variable costs: base_cost + dual_term
            for t in self.time_periods:
                key = ('i_E_com', t)
                if key in self._com_col_indices:
                    idx = self._com_col_indices[key]
                    h.changeColCost(idx, self._highs_base_costs[idx] - dual_elec[t])
                key = ('e_E_com', t)
                if key in self._com_col_indices:
                    idx = self._com_col_indices[key]
                    h.changeColCost(idx, self._highs_base_costs[idx] + dual_elec[t])

                key = ('i_H_com', t)
                if key in self._com_col_indices:
                    idx = self._com_col_indices[key]
                    h.changeColCost(idx, self._highs_base_costs[idx] - dual_heat[t])
                key = ('e_H_com', t)
                if key in self._com_col_indices:
                    idx = self._com_col_indices[key]
                    h.changeColCost(idx, self._highs_base_costs[idx] + dual_heat[t])

                key = ('i_G_com', t)
                if key in self._com_col_indices:
                    idx = self._com_col_indices[key]
                    h.changeColCost(idx, self._highs_base_costs[idx] - dual_hydro[t])
                key = ('e_G_com', t)
                if key in self._com_col_indices:
                    idx = self._com_col_indices[key]
                    h.changeColCost(idx, self._highs_base_costs[idx] + dual_hydro[t])

        # Solve
        h.run()
        model_status = h.getModelStatus()

        if model_status == HighsModelStatus.kOptimal:
            obj_val = h.getInfoValue("objective_function_value")[1]
            reduced_cost = obj_val - dual_convexity

            # Extract results using existing helper
            _, results = solve_and_extract_results_highs(self.model, h)

            return reduced_cost, results, obj_val
        else:
            return float('inf'), None, None

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
        if self.mipsolver == 'highs':
            return self._solve_pricing_highs(dual_elec, dual_heat, dual_hydro, dual_convexity, farkas)

        # Free transform to allow objective modification
        self.model.freeTransform()
        
        # Build modified objective with dual prices
        new_obj = 0
        u = self.player

        # Original grid costs
        if not farkas:
            # Electricity grid costs
            if (u, 0) in self.lem.i_E_gri:
                new_obj += quicksum(self.parameters.get(f'pi_E_gri_import_{t}', np.inf) * self.lem.i_E_gri[u, t] for t in self.time_periods)
            if (u, 0) in self.lem.e_E_gri:
                new_obj -= quicksum(self.parameters.get(f'pi_E_gri_export_{t}', np.inf) * self.lem.e_E_gri[u, t] for t in self.time_periods)
            if (u, 'elec', 0) in self.lem.nfl_d:
                # u_E = self.parameters.get(f'u_E_{u}_{t}', 0.0)
                new_obj -= quicksum(self.parameters.get(f'u_E_{u}_{t}', 0.0) * self.lem.nfl_d[u, 'elec', t] for t in self.time_periods)
            # Heat grid costs
            if (u, 0) in self.lem.i_H_gri:
                new_obj += quicksum(self.parameters.get(f'pi_H_gri_import_{t}', np.inf) * self.lem.i_H_gri[u, t] for t in self.time_periods)
            if (u, 0) in self.lem.e_H_gri:
                new_obj -= quicksum(self.parameters.get(f'pi_H_gri_export_{t}', np.inf) * self.lem.e_H_gri[u, t] for t in self.time_periods)
            if (u, 'heat', 0) in self.lem.nfl_d:
                new_obj -= quicksum(self.parameters.get(f'u_H_{u}_{t}', 0.0) * self.lem.nfl_d[u, 'heat', t] for t in self.time_periods)
            # Hydrogen grid costs
            if (u, 0) in self.lem.i_G_gri:
                new_obj += quicksum(self.parameters.get(f'pi_G_gri_import_{t}', np.inf) * self.lem.i_G_gri[u, t] for t in self.time_periods)
            if (u, 0) in self.lem.e_G_gri:
                new_obj -= quicksum(self.parameters.get(f'pi_G_gri_export_{t}', np.inf) * self.lem.e_G_gri[u, t] for t in self.time_periods)
            if (u, 'hydro', 0) in self.lem.nfl_d:
                new_obj -= quicksum(self.parameters.get(f'u_G_{u}_{t}', 0.0) * self.lem.nfl_d[u, 'hydro', t] for t in self.time_periods)
            # Production costs
            if (u, 'res', 0) in self.lem.p:
                new_obj += quicksum(self.parameters.get(f'c_res_{u}', np.inf) * self.lem.p[u, 'res', t] for t in self.time_periods)
            if (u, 'hp', 0) in self.lem.p:
                new_obj += quicksum(self.parameters.get(f'c_hp_{u}', np.inf) * self.lem.p[u, 'hp', t] for t in self.time_periods)
            if (u, 'els', 0) in self.lem.p:
                new_obj += quicksum(self.parameters.get(f'c_els_{u}', np.inf) * self.lem.p[u, 'els', t] for t in self.time_periods)
            
            # Startup costs
            if (u, 0) in self.lem.z_su_G:
                new_obj += quicksum(self.parameters.get(f'c_su_G_{u}', np.inf) * self.lem.z_su_G[u, t] for t in self.time_periods)
            if (u, 0) in self.lem.z_su_H:
                new_obj += quicksum(self.parameters.get(f'c_su_H_{u}', np.inf) * self.lem.z_su_H[u, t] for t in self.time_periods)
            # Storage costs
            if u in self.lem.params["players_with_elec_storage"]:
                c_sto_E = self.parameters.get(f'c_sto_E_{u}', np.inf)
                nu_ch, nu_dis = self.parameters.get(f'nu_ch_E', np.inf), self.parameters.get(f'nu_dis_E', np.inf)
                new_obj += quicksum(c_sto_E * self.lem.b_ch_E[u, t] * nu_ch for t in self.time_periods)
                new_obj += quicksum(c_sto_E * self.lem.b_dis_E[u, t] * (1/nu_dis) for t in self.time_periods)
            if u in self.lem.params["players_with_hydro_storage"]:
                c_sto_G = self.parameters.get(f'c_sto_G_{u}', np.inf)
                nu_ch, nu_dis = self.parameters.get(f'nu_ch_G', np.inf), self.parameters.get(f'nu_dis_G', np.inf)
                new_obj += quicksum(c_sto_G * self.lem.b_ch_G[u, t] * nu_ch for t in self.time_periods)
                new_obj += quicksum(c_sto_G * self.lem.b_dis_G[u, t] * (1/nu_dis) for t in self.time_periods)
            if u in self.lem.params["players_with_heat_storage"]:
                c_sto_H = self.parameters.get(f'c_sto_H_{u}', np.inf)
                nu_ch, nu_dis = self.parameters.get(f'nu_ch_H', np.inf), self.parameters.get(f'nu_dis_H', np.inf)
                new_obj += quicksum(c_sto_H * self.lem.b_ch_H[u, t] * nu_ch for t in self.time_periods)
                new_obj += quicksum(c_sto_H * self.lem.b_dis_H[u, t] * (1/nu_dis) for t in self.time_periods)
        # Community trading with dual prices
        # For reduced cost: original_cost - dual_price * coefficient
        # Electricity
        if (u, 0) in self.lem.i_E_com:
            # Import from community: coefficient is +1 in balance
            new_obj -= quicksum(dual_elec[t] * self.lem.i_E_com[u, t] for t in self.time_periods)
        if (u, 0) in self.lem.e_E_com:
            # Export to community: coefficient is -1 in balance
            new_obj += quicksum(dual_elec[t] * self.lem.e_E_com[u, t] for t in self.time_periods)
        
        # Heat
        if (u, 0) in self.lem.i_H_com:
            new_obj -= quicksum(dual_heat[t] * self.lem.i_H_com[u, t] for t in self.time_periods)
        if (u, 0) in self.lem.e_H_com:
            new_obj += quicksum(dual_heat[t] * self.lem.e_H_com[u, t] for t in self.time_periods)
        # Hydrogen
        if (u, 0) in self.lem.i_G_com:
            new_obj -= quicksum(dual_hydro[t] * self.lem.i_G_com[u, t] for t in self.time_periods)
        if (u, 0) in self.lem.e_G_com:
            new_obj += quicksum(dual_hydro[t] * self.lem.e_G_com[u, t] for t in self.time_periods)
        
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
    def __init__(self, players: List[str], time_periods: List[int], params: Dict):
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
        self.params = params
        # Storage for variables and constraints
        self.model.data['vars'] = {
            player:{} for player in players
        }  # {(player, col_idx): {'var': var, 'solution': dict}}
        
        # Data dictionary for storing constraints (similar to LocalEnergyMarket)
        self.model.data['cons'] = {
            'community_elec_balance': {},
            'community_heat_balance': {},
            'community_hydro_balance': {},
            'convexity': {},
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
                                if not isinstance(value, list):
                                    solution[var_name][key] = 0.0
                                else:
                                    solution[var_name][key] = np.zeros(len(value))
                            if not isinstance(value, list):
                                solution[var_name][key] += lambda_val * value
                            else:
                                solution[var_name][key] += lambda_val * np.asarray(value)
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

        # Utility (Minimization - 음수)
        nfl_d_elec = solution.get('nfl_d', {}).get((player, 'elec', t), 0.0)
        nfl_d_hydro = solution.get('nfl_d', {}).get((player, 'hydro', t), 0.0)
        nfl_d_heat = solution.get('nfl_d', {}).get((player, 'heat', t), 0.0)
        cost -= nfl_d_elec * params.get(f'u_E_{player}_{t}', 0.0)
        cost -= nfl_d_hydro * params.get(f'u_G_{player}_{t}', 0.0)
        cost -= nfl_d_heat * params.get(f'u_H_{player}_{t}', 0.0)

    if (cost == np.inf) or (cost == -np.inf) or (np.isnan(cost)):
        raise Exception(f"Cost is infinite for player {player} at time {t}")
    return cost
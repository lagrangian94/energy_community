from pyscipopt import Model, quicksum, Pricer, SCIP_RESULT, SCIP_PARAMSETTING
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

class EnergyMarketPricer(Pricer):
    def __init__(self, players, time_periods, parameters, constraints, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.players = players
        self.time_periods = time_periods
        self.params = parameters
        self.constraints = constraints
        self.iteration = 0
        
        # Store player sets
        self.players_with_renewables = self.params.get('players_with_renewables', [])
        self.players_with_electrolyzers = self.params.get('players_with_electrolyzers', [])
        self.players_with_heatpumps = self.params.get('players_with_heatpumps', [])
        self.players_with_elec_storage = self.params.get('players_with_elec_storage', [])
        self.players_with_hydro_storage = self.params.get('players_with_hydro_storage', [])
        self.players_with_heat_storage = self.params.get('players_with_heat_storage', [])
        
    def price(self, farkas):
        """Solve pricing subproblems for all players"""
        
        # Get dual multipliers from master problem
        dual_sol = self._get_dual_multipliers(farkas)
        
        # Solve subproblem for each player
        min_reduced_cost = float('inf')
        best_player = None
        best_pattern = None
        
        for u in self.players:
            reduced_cost, pattern = self._solve_player_subproblem(u, dual_sol, farkas)
            
            if reduced_cost < min_reduced_cost:
                min_reduced_cost = reduced_cost
                best_player = u
                best_pattern = pattern
        
        print(f"Iteration {self.iteration}: Best reduced cost = {min_reduced_cost:.6f}")
        
        # Add new column if reduced cost is negative
        if min_reduced_cost < -1e-6:  # numerical tolerance
            self._add_new_column(best_player, best_pattern)
            print(f"Added new column for player {best_player}")
            
        return {'result': SCIP_RESULT.SUCCESS}
    
    def _get_dual_multipliers(self, farkas):
        """Extract dual multipliers from master problem constraints"""
        dual_sol = {
            'electricity': {},
            'heat': {},
            'hydrogen': {},
            'peak': {}
        }
        
        try:
            for t in self.time_periods:
                # Community balance constraints - initialize to 0 if not available
                if f'elec_balance_{t}' in self.constraints:
                    cons = self.model.getTransformedCons(self.constraints[f'elec_balance_{t}'])
                    try:
                        if farkas:
                            dual_sol['electricity'][t] = self.model.getDualfarkasLinear(cons)
                        else:
                            dual_sol['electricity'][t] = self.model.getDualsolLinear(cons)
                    except:
                        # dual_sol['electricity'][t] = 0.0
                        raise Exception("Error getting dual multipliers")
                else:
                    dual_sol['electricity'][t] = 0.0
                
                if f'heat_balance_{t}' in self.constraints:
                    cons = self.model.getTransformedCons(self.constraints[f'heat_balance_{t}'])
                    try:
                        if farkas:
                            dual_sol['heat'][t] = self.model.getDualfarkasLinear(cons)
                        else:
                            dual_sol['heat'][t] = self.model.getDualsolLinear(cons)
                    except:
                        # dual_sol['heat'][t] = 0.0
                        raise Exception("Error getting dual multipliers")
                else:
                    dual_sol['heat'][t] = 0.0
                
                if f'hydrogen_balance_{t}' in self.constraints:
                    cons = self.model.getTransformedCons(self.constraints[f'hydrogen_balance_{t}'])
                    try:
                        if farkas:
                            dual_sol['hydrogen'][t] = self.model.getDualfarkasLinear(cons)
                        else:
                            dual_sol['hydrogen'][t] = self.model.getDualsolLinear(cons)
                    except:
                        # dual_sol['hydrogen'][t] = 0.0
                        raise Exception("Error getting dual multipliers")
                else:
                    dual_sol['hydrogen'][t] = 0.0
                

            
            # Debug output for dual values
            if not farkas and self.iteration <= 3:  # Only show first few iterations
                print(f"Dual values - Electricity: {dual_sol['electricity']}")
                print(f"Dual values - Heat: {dual_sol['heat']}")
                print(f"Dual values - Hydrogen: {dual_sol['hydrogen']}")
                print(f"Dual values - Peak: {dual_sol['peak']}")
                
        except Exception as e:
            print(f"Error getting dual multipliers: {e}")
            # Return zero dual values as fallback
            for t in self.time_periods:
                dual_sol['electricity'][t] = 0.0
                dual_sol['heat'][t] = 0.0
                dual_sol['hydrogen'][t] = 0.0
                dual_sol['peak'][t] = 0.0
        
        return dual_sol
    
    def _solve_player_subproblem(self, player, dual_sol, farkas):
        """Solve pricing subproblem for a specific player"""
        
        # Create subproblem model for this player
        sub_model = Model(f"Subproblem_Player_{player}")
        sub_model.setParam("display/verblevel", 0)  # Suppress output
        
        # Set solver parameters for better stability
        sub_model.setParam("numerics/feastol", 1e-6)
        sub_model.setParam("numerics/epsilon", 1e-9)
        sub_model.setParam("limits/time", 300)  # 5 minute time limit
        
        
        # Create variables for this player
        variables = self._create_player_variables(sub_model, player)
        
        # Create player-specific objective (original cost - dual contribution)
        obj_terms = self._create_player_objective(player, variables, dual_sol, farkas)
        
        # Check if objective has terms
        if not obj_terms:
            print(f"Warning: No objective terms for player {player}")
            # Add small penalty to avoid unbounded problem
            obj_terms = [1e-6 * var for var in variables.values() if hasattr(var, 'name')]
        
        sub_model.setObjective(quicksum(obj_terms), "minimize")
        
        # Add player-specific constraints
        self._add_player_constraints(sub_model, player, variables)
        
        print(f"Solving subproblem for player {player}...")
        
        # Solve subproblem
        sub_model.optimize()
        
        status = sub_model.getStatus()
        print(f"Player {player} subproblem status: {status}")
        
        if status == "optimal":
            try:
                # Extract solution pattern
                pattern = self._extract_pattern(sub_model, player, variables)
                
                # Reduced cost calculation for energy market (minimize problem)
                # Unlike bin packing (maximize), here reduced cost = objective value directly
                reduced_cost = sub_model.getObjVal()
                
                if farkas:
                    reduced_cost -= 1  # Farkas pricing adjustment
                
                print(f"Player {player} reduced cost: {reduced_cost:.6f}")
                
                # Additional debug info for first few iterations
                if self.iteration <= 3:
                    print(f"  Pattern cost breakdown:")
                    print(f"  - Objective value: {sub_model.getObjVal():.6f}")
                    if 'cost' in pattern:
                        print(f"  - Pattern cost: {pattern['cost']:.6f}")
                
                return reduced_cost, pattern
                
            except Exception as e:
                print(f"Error extracting pattern for player {player}: {e}")
                return float('inf'), None
                
        elif status in ["infeasible", "unbounded", "inforunbd"]:
            print(f"Player {player} subproblem {status}")
            if farkas:
                # In Farkas pricing, infeasible subproblem might still be useful
                return float('inf'), None
            else:
                return float('inf'), None
                
        else:
            print(f"Player {player} subproblem not solved to optimality: {status}")
            # Try to extract partial solution if available
            try:
                if sub_model.getNSols() > 0:  # Check if any solution is available
                    pattern = self._extract_pattern(sub_model, player, variables)
                    reduced_cost = sub_model.getObjVal()
                    print(f"Using non-optimal solution with cost: {reduced_cost:.6f}")
                    return reduced_cost, pattern
                else:
                    return float('inf'), None
            except:
                return float('inf'), None
    
    def _create_player_variables(self, model, player):
        """Create all variables for a specific player"""
        variables = {}
        
        # Grid interaction variables
        for t in self.time_periods:
            variables[f'e_E_gri_{t}'] = model.addVar(vtype="C", name=f"e_E_gri_{t}", lb=0, 
                                                   ub=self.params.get(f'e_E_cap_{player}_{t}', 1000))
            variables[f'i_E_gri_{t}'] = model.addVar(vtype="C", name=f"i_E_gri_{t}", lb=0,
                                                   ub=self.params.get(f'i_E_cap_{player}_{t}', 1000))
            variables[f'e_E_com_{t}'] = model.addVar(vtype="C", name=f"e_E_com_{t}", lb=0, ub=1000)
            variables[f'i_E_com_{t}'] = model.addVar(vtype="C", name=f"i_E_com_{t}", lb=0, ub=1000)
            
            variables[f'e_H_gri_{t}'] = model.addVar(vtype="C", name=f"e_H_gri_{t}", lb=0,
                                                   ub=self.params.get(f'e_H_cap_{player}_{t}', 500))
            variables[f'i_H_gri_{t}'] = model.addVar(vtype="C", name=f"i_H_gri_{t}", lb=0,
                                                   ub=self.params.get(f'i_H_cap_{player}_{t}', 500))
            variables[f'e_H_com_{t}'] = model.addVar(vtype="C", name=f"e_H_com_{t}", lb=0, ub=500)
            variables[f'i_H_com_{t}'] = model.addVar(vtype="C", name=f"i_H_com_{t}", lb=0, ub=500)
            
            variables[f'e_G_gri_{t}'] = model.addVar(vtype="C", name=f"e_G_gri_{t}", lb=0,
                                                   ub=self.params.get(f'e_G_cap_{player}_{t}', 100))
            variables[f'i_G_gri_{t}'] = model.addVar(vtype="C", name=f"i_G_gri_{t}", lb=0,
                                                   ub=self.params.get(f'i_G_cap_{player}_{t}', 100))
            variables[f'e_G_com_{t}'] = model.addVar(vtype="C", name=f"e_G_com_{t}", lb=0, ub=100)
            variables[f'i_G_com_{t}'] = model.addVar(vtype="C", name=f"i_G_com_{t}", lb=0, ub=100)
        
        # Production and consumption variables
        for t in self.time_periods:
            if player in self.players_with_renewables:
                renewable_cap = self.params.get(f'renewable_cap_{player}', 200)
                variables[f'p_res_{t}'] = model.addVar(vtype="C", name=f"p_res_{t}", 
                                                     lb=0, ub=renewable_cap)
            
            if player in self.players_with_heatpumps:
                hp_cap = self.params.get(f'hp_cap_{player}', 100)
                variables[f'p_hp_{t}'] = model.addVar(vtype="C", name=f"p_hp_{t}", 
                                                    lb=0, ub=hp_cap)
                variables[f'd_hp_{t}'] = model.addVar(vtype="C", name=f"d_hp_{t}", 
                                                    lb=0, ub=hp_cap/3)
            
            if player in self.players_with_electrolyzers:
                els_cap = self.params.get(f'els_cap_{player}', 150)
                variables[f'p_els_{t}'] = model.addVar(vtype="C", name=f"p_els_{t}", 
                                                     lb=0, ub=els_cap)
                variables[f'd_els_{t}'] = model.addVar(vtype="C", name=f"d_els_{t}", 
                                                     lb=0, ub=200)
                
                # Binary variables for electrolyzer commitment
                variables[f'z_su_{t}'] = model.addVar(vtype="B", name=f"z_su_{t}")
                variables[f'z_on_{t}'] = model.addVar(vtype="B", name=f"z_on_{t}")
                variables[f'z_off_{t}'] = model.addVar(vtype="B", name=f"z_off_{t}")
                variables[f'z_sb_{t}'] = model.addVar(vtype="B", name=f"z_sb_{t}")
        
        # Storage variables
        storage_power = self.params.get('storage_power', 50)
        storage_capacity = self.params.get('storage_capacity', 200)
        
        for t in self.time_periods:
            if player in self.players_with_elec_storage:
                variables[f'b_dis_E_{t}'] = model.addVar(vtype="C", name=f"b_dis_E_{t}", 
                                                       lb=0, ub=storage_power)
                variables[f'b_ch_E_{t}'] = model.addVar(vtype="C", name=f"b_ch_E_{t}", 
                                                      lb=0, ub=storage_power)
                variables[f's_E_{t}'] = model.addVar(vtype="C", name=f"s_E_{t}", 
                                                   lb=0, ub=storage_capacity)
            
            if player in self.players_with_hydro_storage:
                variables[f'b_dis_G_{t}'] = model.addVar(vtype="C", name=f"b_dis_G_{t}", 
                                                       lb=0, ub=storage_power)
                variables[f'b_ch_G_{t}'] = model.addVar(vtype="C", name=f"b_ch_G_{t}", 
                                                      lb=0, ub=storage_power)
                variables[f's_G_{t}'] = model.addVar(vtype="C", name=f"s_G_{t}", 
                                                   lb=0, ub=storage_capacity)
            
            if player in self.players_with_heat_storage:
                variables[f'b_dis_H_{t}'] = model.addVar(vtype="C", name=f"b_dis_H_{t}", 
                                                       lb=0, ub=storage_power)
                variables[f'b_ch_H_{t}'] = model.addVar(vtype="C", name=f"b_ch_H_{t}", 
                                                      lb=0, ub=storage_power)
                variables[f's_H_{t}'] = model.addVar(vtype="C", name=f"s_H_{t}", 
                                                   lb=0, ub=storage_capacity)
        
        return variables
    
    def _create_player_objective(self, player, variables, dual_sol, farkas):
        """Create objective function for player subproblem (original cost - dual contribution)"""
        obj_terms = []
        
        # Original cost terms
        for t in self.time_periods:
            # Production costs
            if player in self.players_with_renewables and f'p_res_{t}' in variables:
                c_res = self.params.get(f'c_res_{player}', 0)
                obj_terms.append(c_res * variables[f'p_res_{t}'])
            
            if player in self.players_with_heatpumps and f'p_hp_{t}' in variables:
                c_hp = self.params.get(f'c_hp_{player}', 0)
                obj_terms.append(c_hp * variables[f'p_hp_{t}'])
            
            if player in self.players_with_electrolyzers:
                if f'p_els_{t}' in variables:
                    c_els = self.params.get(f'c_els_{player}', 0)
                    obj_terms.append(c_els * variables[f'p_els_{t}'])
                if f'z_su_{t}' in variables:
                    c_su = self.params.get(f'c_su_{player}', 0)
                    obj_terms.append(c_su * variables[f'z_su_{t}'])
            
            # Grid interaction costs
            pi_E_gri = self.params.get(f'pi_E_gri_{t}', 0)
            pi_H_gri = self.params.get(f'pi_H_gri_{t}', 0)
            pi_G_gri = self.params.get(f'pi_G_gri_{t}', 0)
            
            obj_terms.append(pi_E_gri * (variables[f'i_E_gri_{t}'] - variables[f'e_E_gri_{t}']))
            obj_terms.append(pi_H_gri * (variables[f'i_H_gri_{t}'] - variables[f'e_H_gri_{t}']))
            obj_terms.append(pi_G_gri * (variables[f'i_G_gri_{t}'] - variables[f'e_G_gri_{t}']))
            
            # Storage costs
            c_sto = self.params.get('c_sto', 0.01)
            nu_ch = self.params.get('nu_ch', 0.9)
            nu_dis = self.params.get('nu_dis', 0.9)
            
            if player in self.players_with_elec_storage:
                if f'b_ch_E_{t}' in variables and f'b_dis_E_{t}' in variables:
                    obj_terms.append(c_sto * (nu_ch * variables[f'b_ch_E_{t}'] + (1/nu_dis) * variables[f'b_dis_E_{t}']))
            
            if player in self.players_with_hydro_storage:
                if f'b_ch_G_{t}' in variables and f'b_dis_G_{t}' in variables:
                    obj_terms.append(c_sto * (nu_ch * variables[f'b_ch_G_{t}'] + (1/nu_dis) * variables[f'b_dis_G_{t}']))
            
            if player in self.players_with_heat_storage:
                if f'b_ch_H_{t}' in variables and f'b_dis_H_{t}' in variables:
                    obj_terms.append(c_sto * (nu_ch * variables[f'b_ch_H_{t}'] + (1/nu_dis) * variables[f'b_dis_H_{t}']))
        
        # Subtract dual contribution (pricing mechanism)
        # This is the key part: we subtract the dual values times the contribution coefficients
        if not farkas:  # Only for regular pricing, not Farkas
            for t in self.time_periods:
                # Electricity community contribution: coefficient is (i_E_com - e_E_com)
                if 'electricity' in dual_sol and t in dual_sol['electricity']:
                    pi_E = dual_sol['electricity'][t]
                    community_elec_contrib = variables[f'i_E_com_{t}'] - variables[f'e_E_com_{t}']
                    obj_terms.append(-pi_E * community_elec_contrib)  # Subtract dual * coefficient
                
                # Heat community contribution: coefficient is (i_H_com - e_H_com)
                if 'heat' in dual_sol and t in dual_sol['heat']:
                    pi_H = dual_sol['heat'][t]
                    community_heat_contrib = variables[f'i_H_com_{t}'] - variables[f'e_H_com_{t}']
                    obj_terms.append(-pi_H * community_heat_contrib)
                
                # Hydrogen community contribution: coefficient is (e_G_com - i_G_com) [note the sign]
                if 'hydrogen' in dual_sol and t in dual_sol['hydrogen']:
                    pi_G = dual_sol['hydrogen'][t]
                    community_hydrogen_contrib = variables[f'e_G_com_{t}'] - variables[f'i_G_com_{t}']
                    obj_terms.append(-pi_G * community_hydrogen_contrib)
                
                # Peak power contribution: coefficient is (i_E_gri - e_E_gri)
                if 'peak' in dual_sol and t in dual_sol['peak']:
                    pi_peak = dual_sol['peak'][t]
                    peak_contrib = variables[f'i_E_gri_{t}'] - variables[f'e_E_gri_{t}']
                    obj_terms.append(-pi_peak * peak_contrib)
        
        return obj_terms
    
    def _add_player_constraints(self, model, player, variables):
        """Add all player-specific constraints to subproblem"""
        
        # Electricity balance constraints
        for t in self.time_periods:
            lhs = (variables[f'i_E_gri_{t}'] - variables[f'e_E_gri_{t}'] + 
                   variables[f'i_E_com_{t}'] - variables[f'e_E_com_{t}'])
            
            # Add renewable generation
            if player in self.players_with_renewables and f'p_res_{t}' in variables:
                lhs += variables[f'p_res_{t}']
            
            # Add electricity storage
            if player in self.players_with_elec_storage:
                if f'b_dis_E_{t}' in variables and f'b_ch_E_{t}' in variables:
                    lhs += variables[f'b_dis_E_{t}'] - variables[f'b_ch_E_{t}']
            
            # RHS: demand
            rhs = self.params.get(f'd_E_nfl_{player}_{t}', 0)
            
            if player in self.players_with_heatpumps and f'd_hp_{t}' in variables:
                rhs += variables[f'd_hp_{t}']
            if player in self.players_with_electrolyzers and f'd_els_{t}' in variables:
                rhs += variables[f'd_els_{t}']
            
            model.addCons(lhs == rhs, name=f"elec_balance_{t}")
        
        # Heat balance constraints
        for t in self.time_periods:
            lhs = (variables[f'i_H_gri_{t}'] - variables[f'e_H_gri_{t}'] + 
                   variables[f'i_H_com_{t}'] - variables[f'e_H_com_{t}'])
            
            if player in self.players_with_heatpumps and f'p_hp_{t}' in variables:
                lhs += variables[f'p_hp_{t}']
            
            if player in self.players_with_heat_storage:
                if f'b_dis_H_{t}' in variables and f'b_ch_H_{t}' in variables:
                    lhs += variables[f'b_dis_H_{t}'] - variables[f'b_ch_H_{t}']
            
            rhs = self.params.get(f'd_H_nfl_{player}_{t}', 0)
            model.addCons(lhs == rhs, name=f"heat_balance_{t}")
        
        # Hydrogen balance constraints
        for t in self.time_periods:
            lhs = (variables[f'i_G_gri_{t}'] - variables[f'e_G_gri_{t}'] + 
                   variables[f'i_G_com_{t}'] - variables[f'e_G_com_{t}'])
            
            if player in self.players_with_electrolyzers and f'p_els_{t}' in variables:
                lhs += variables[f'p_els_{t}']
            
            if player in self.players_with_hydro_storage:
                if f'b_dis_G_{t}' in variables and f'b_ch_G_{t}' in variables:
                    lhs += variables[f'b_dis_G_{t}'] - variables[f'b_ch_G_{t}']
            
            rhs = self.params.get(f'd_G_nfl_{player}_{t}', 0)
            model.addCons(lhs == rhs, name=f"hydrogen_balance_{t}")
        
        # Heat pump coupling constraints
        if player in self.players_with_heatpumps:
            for t in self.time_periods:
                if f'd_hp_{t}' in variables and f'p_hp_{t}' in variables:
                    nu_COP = self.params.get(f'nu_COP_{player}', 3.0)
                    model.addCons(nu_COP * variables[f'd_hp_{t}'] == variables[f'p_hp_{t}'], 
                                name=f"heatpump_coupling_{t}")
        
        # Electrolyzer constraints
        if player in self.players_with_electrolyzers:
            for t in self.time_periods:
                # Coupling constraint
                if f'p_els_{t}' in variables and f'd_els_{t}' in variables:
                    phi1 = self.params.get(f'phi1_{player}', 0.7)
                    phi0 = self.params.get(f'phi0_{player}', 0.0)
                    model.addCons(variables[f'p_els_{t}'] <= phi1 * variables[f'd_els_{t}'] + phi0,
                                name=f"electrolyzer_coupling_{t}")
                
                # Commitment constraints
                if all(f'z_{state}_{t}' in variables for state in ['on', 'off', 'sb']):
                    model.addCons(variables[f'z_on_{t}'] + variables[f'z_off_{t}'] + variables[f'z_sb_{t}'] == 1,
                                name=f"electrolyzer_state_{t}")
                    
                    if f'd_els_{t}' in variables:
                        C_max = self.params.get(f'C_max_{player}', 100)
                        C_sb = self.params.get(f'C_sb_{player}', 10)
                        C_min = self.params.get(f'C_min_{player}', 20)
                        
                        model.addCons(variables[f'd_els_{t}'] <= C_max * variables[f'z_on_{t}'] + C_sb * variables[f'z_sb_{t}'],
                                    name=f"electrolyzer_max_{t}")
                        model.addCons(variables[f'd_els_{t}'] >= C_min * variables[f'z_on_{t}'] + C_sb * variables[f'z_sb_{t}'],
                                    name=f"electrolyzer_min_{t}")
                    
                    # Startup logic
                    if t > 0 and f'z_su_{t}' in variables:
                        model.addCons(variables[f'z_su_{t}'] >= variables[f'z_off_{t-1}'] + variables[f'z_on_{t}'] + variables[f'z_sb_{t}'] - 1,
                                    name=f"electrolyzer_startup_{t}")
        
        # Storage SOC transition constraints
        for storage_type, player_set in [('E', self.players_with_elec_storage), 
                                       ('G', self.players_with_hydro_storage), 
                                       ('H', self.players_with_heat_storage)]:
            if player in player_set:
                # Initial SOC
                if f's_{storage_type}_0' in variables:
                    initial_soc = self.params.get('initial_soc', 50)
                    model.addCons(variables[f's_{storage_type}_0'] == initial_soc, 
                                name=f"initial_soc_{storage_type}")
                
                # SOC transitions
                for t in self.time_periods:
                    if (t > 0 and f's_{storage_type}_{t}' in variables and 
                        f's_{storage_type}_{t-1}' in variables):
                        nu_ch = self.params.get('nu_ch', 0.9)
                        nu_dis = self.params.get('nu_dis', 0.9)
                        
                        model.addCons(
                            variables[f's_{storage_type}_{t}'] == 
                            variables[f's_{storage_type}_{t-1}'] + 
                            nu_ch * variables[f'b_ch_{storage_type}_{t}'] - 
                            (1/nu_dis) * variables[f'b_dis_{storage_type}_{t}'],
                            name=f"soc_transition_{storage_type}_{t}"
                        )
        
        # Renewable availability constraints
        if player in self.players_with_renewables:
            for t in self.time_periods:
                if f'p_res_{t}' in variables:
                    availability = self.params.get(f'renewable_availability_{player}_{t}', 1.0)
                    renewable_cap = self.params.get(f'renewable_cap_{player}', 200)
                    model.addCons(variables[f'p_res_{t}'] <= availability * renewable_cap,
                                name=f"renewable_availability_{t}")
    
    def _extract_pattern(self, model, player, variables):
        """Extract solution pattern from solved subproblem"""
        
        def safe_get_val(var):
            """Safely get variable value with error handling"""
            try:
                if var is None:
                    return 0.0
                return model.getVal(var)
            except Exception as e:
                print(f"Warning: Could not get value for variable {var.name if var else 'None'}: {e}")
                return 0.0
        
        pattern = {
            'player': player,
            'electricity': {},
            'heat': {},
            'hydrogen': {},
            'production': {},
            'storage': {},
            'cost': 0.0
        }
        
        # Safely get objective value
        try:
            pattern['cost'] = model.getObjVal()
        except:
            pattern['cost'] = 0.0
            print(f"Warning: Could not get objective value for player {player}")
        
        for t in self.time_periods:
            # Community contributions (key for master problem coefficients)
            e_E_com = safe_get_val(variables.get(f'e_E_com_{t}'))
            i_E_com = safe_get_val(variables.get(f'i_E_com_{t}'))
            e_E_gri = safe_get_val(variables.get(f'e_E_gri_{t}'))
            i_E_gri = safe_get_val(variables.get(f'i_E_gri_{t}'))
            
            pattern['electricity'][t] = {
                'e_com': e_E_com,
                'i_com': i_E_com,
                'e_gri': e_E_gri,
                'i_gri': i_E_gri,
                'net_com': i_E_com - e_E_com,
                'net_gri': i_E_gri - e_E_gri
            }
            
            e_H_com = safe_get_val(variables.get(f'e_H_com_{t}'))
            i_H_com = safe_get_val(variables.get(f'i_H_com_{t}'))
            e_H_gri = safe_get_val(variables.get(f'e_H_gri_{t}'))
            i_H_gri = safe_get_val(variables.get(f'i_H_gri_{t}'))
            
            pattern['heat'][t] = {
                'e_com': e_H_com,
                'i_com': i_H_com,
                'e_gri': e_H_gri,
                'i_gri': i_H_gri,
                'net_com': i_H_com - e_H_com
            }
            
            e_G_com = safe_get_val(variables.get(f'e_G_com_{t}'))
            i_G_com = safe_get_val(variables.get(f'i_G_com_{t}'))
            e_G_gri = safe_get_val(variables.get(f'e_G_gri_{t}'))
            i_G_gri = safe_get_val(variables.get(f'i_G_gri_{t}'))
            
            pattern['hydrogen'][t] = {
                'e_com': e_G_com,
                'i_com': i_G_com,
                'e_gri': e_G_gri,
                'i_gri': i_G_gri,
                'net_com': e_G_com - i_G_com  # Note: sign difference for hydrogen
            }
            
            # Production and storage details (for reporting)
            if f'p_res_{t}' in variables:
                pattern['production'][f'res_{t}'] = safe_get_val(variables[f'p_res_{t}'])
            if f'p_hp_{t}' in variables:
                pattern['production'][f'hp_{t}'] = safe_get_val(variables[f'p_hp_{t}'])
            if f'p_els_{t}' in variables:
                pattern['production'][f'els_{t}'] = safe_get_val(variables[f'p_els_{t}'])
            
            # Storage SOC values
            if f's_E_{t}' in variables:
                pattern['storage'][f'elec_{t}'] = safe_get_val(variables[f's_E_{t}'])
            if f's_G_{t}' in variables:
                pattern['storage'][f'hydro_{t}'] = safe_get_val(variables[f's_G_{t}'])
            if f's_H_{t}' in variables:
                pattern['storage'][f'heat_{t}'] = safe_get_val(variables[f's_H_{t}'])
        
        return pattern
    
    def _add_new_column(self, player, pattern):
        """Add new column (pattern) to master problem"""
        
        # Create new variable for this pattern
        new_var = self.model.addVar(vtype="C", name=f"lambda_{player}_{self.iteration}", 
                                  obj=pattern['cost'], pricedVar=True, lb=0, ub=1)
        
        # Add coefficients to master problem constraints
        for t in self.time_periods:
            # Community electricity balance: Σ_p (net_electricity_contribution_p * λ_p) = 0
            if f'elec_balance_{t}' in self.constraints:
                cons = self.model.getTransformedCons(self.constraints[f'elec_balance_{t}'])
                coeff = pattern['electricity'][t]['net_com']
                if abs(coeff) > 1e-9:  # Only add non-zero coefficients
                    self.model.addConsCoeff(cons, new_var, coeff)
            
            # Community heat balance
            if f'heat_balance_{t}' in self.constraints:
                cons = self.model.getTransformedCons(self.constraints[f'heat_balance_{t}'])
                coeff = pattern['heat'][t]['net_com']
                if abs(coeff) > 1e-9:
                    self.model.addConsCoeff(cons, new_var, coeff)
            
            # Community hydrogen balance
            if f'hydrogen_balance_{t}' in self.constraints:
                cons = self.model.getTransformedCons(self.constraints[f'hydrogen_balance_{t}'])
                coeff = pattern['hydrogen'][t]['net_com']
                if abs(coeff) > 1e-9:
                    self.model.addConsCoeff(cons, new_var, coeff)
            
        
        # Convexity constraint: each player can only use one pattern (if enforcing convexity)
        if hasattr(self, 'convexity_constraints') and player in self.convexity_constraints:
            cons = self.model.getTransformedCons(self.convexity_constraints[player])
            self.model.addConsCoeff(cons, new_var, 1.0)  # Each pattern contributes 1 to its player's convexity
        
    def pricerredcost(self):
        """Regular pricing"""
        self.iteration += 1
        return self.price(farkas=False)
    
    def pricerfarkas(self):
        """Farkas pricing"""
        self.iteration += 1
        return self.price(farkas=True)


class EnergyMarketColumnGeneration:
    def __init__(self, players, time_periods, parameters):
        """Initialize Column Generation for Energy Market"""
        self.players = players
        self.time_periods = time_periods
        self.params = parameters
        self.model = Model("EnergyMarket_Master")
        
        # Store constraint references for pricer
        self.constraints = {}
        
        # Initialize master problem
        self._create_master_problem()
        self._setup_pricer()
    
    def _create_master_problem(self):
        """Create master problem with only community coupling constraints"""
        
        # Master problem only has:
        # 1. Community balance constraints
        # 2. Convexity constraints (optional)
        # Community balance constraints
        for t in self.time_periods:
            # Electricity balance: Σ_p (net_elec_contribution_p * λ_p) = 0
            cons = self.model.addCons(quicksum([]) == 0.0, modifiable=True, name=f"elec_balance_{t}")
            self.constraints[f'elec_balance_{t}'] = cons
            
            # Heat balance: Σ_p (net_heat_contribution_p * λ_p) = 0  
            cons = self.model.addCons(quicksum([]) == 0.0, modifiable=True, name=f"heat_balance_{t}")
            self.constraints[f'heat_balance_{t}'] = cons
            
            # Hydrogen balance: Σ_p (net_hydrogen_contribution_p * λ_p) = 0
            cons = self.model.addCons(quicksum([]) == 0.0, modifiable=True, name=f"hydrogen_balance_{t}")
            self.constraints[f'hydrogen_balance_{t}'] = cons
            
        
        # Optional: Add convexity constraints for each player
        # Σ_p∈P_u λ_p = 1 for each player u (if enforcing convex combinations)
        self.convexity_constraints = {}
        for u in self.players:
            cons = self.model.addCons(quicksum([]) == 1.0, modifiable=True, name=f"convexity_{u}")
            self.convexity_constraints[u] = cons
    
    def _setup_pricer(self):
        """Setup pricer for column generation"""
        self.pricer = EnergyMarketPricer(
            self.players, self.time_periods, self.params, self.constraints
        )
        self.model.includePricer(self.pricer, "EnergyMarketPricer", 
                               "Pricer for Energy Market Problem")
    
    def solve(self):
        """Solve using column generation"""
        
        # Configure SCIP settings for column generation
        self.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.model.setSeparating(SCIP_PARAMSETTING.OFF)
        self.model.disablePropagation()
        self.model.setSeparating(SCIP_PARAMSETTING.OFF)
        # self.model.setParam("display/freq", 1)
        
        print("Starting Column Generation for Energy Market...")
        self.model.optimize()
        
        return self.model.getStatus()
    
    def get_results(self):
        """Extract results from column generation"""
        if self.model.getStatus() != "optimal":
            return None
        
        results = {
            'objective_value': self.model.getObjVal(),
            'selected_patterns': {},
            'community_prices': {}
        }
        
        # Get selected patterns (non-zero lambda variables)
        for var in self.model.getVars(transformed=True):
            if var.name.startswith("lambda_") and self.model.getVal(var) > 1e-6:
                results['selected_patterns'][var.name] = self.model.getVal(var)
        
        # Get dual prices (shadow prices) from community constraints
        for t in self.time_periods:
            results['community_prices'][f'electricity_{t}'] = self.model.getDualsolLinear(
                self.model.getTransformedCons(self.constraints[f'elec_balance_{t}']))
            results['community_prices'][f'heat_{t}'] = self.model.getDualsolLinear(
                self.model.getTransformedCons(self.constraints[f'heat_balance_{t}']))
            results['community_prices'][f'hydrogen_{t}'] = self.model.getDualsolLinear(
                self.model.getTransformedCons(self.constraints[f'hydrogen_balance_{t}']))
        
        return results


# Example usage
if __name__ == "__main__":
    # Define example data (same as before)
    players = ['u1', 'u2', 'u3']
    time_periods = list(range(6))  # 6 hours for faster testing
    
    parameters = {
        'players_with_renewables': ['u1'],
        'players_with_electrolyzers': ['u2'],
        'players_with_heatpumps': ['u3'],
        'players_with_elec_storage': ['u1'],
        'players_with_hydro_storage': ['u2'],
        'players_with_heat_storage': ['u3'],
        'nu_ch': 0.9,
        'nu_dis': 0.9,
        'pi_peak': 100,
        'storage_power': 50,
        'storage_capacity': 200,
        'initial_soc': 100,
        'renewable_cap_u1': 150,
        'hp_cap_u3': 80,
        'els_cap_u2': 100,
        'C_max_u2': 100,
        'C_min_u2': 20,
        'C_sb_u2': 10,
        'phi1_u2': 0.7,
        'phi0_u2': 0.0,
        'c_sto': 0.01,
    }
    
    # Add demand and cost data
    for u in players:
        for t in time_periods:
            parameters[f'd_E_nfl_{u}_{t}'] = 10 + 5 * np.sin(2 * np.pi * t / 24)
            parameters[f'd_H_nfl_{u}_{t}'] = 5 + 2 * np.sin(2 * np.pi * t / 24)
            parameters[f'd_G_nfl_{u}_{t}'] = 2
            parameters[f'c_res_{u}'] = 0.05
            parameters[f'c_hp_{u}'] = 0.1
            parameters[f'c_els_{u}'] = 0.08
            parameters[f'c_su_{u}'] = 50
            parameters[f'pi_E_gri_{t}'] = 0.2 + 0.1 * np.sin(2 * np.pi * t / 24)
            parameters[f'pi_H_gri_{t}'] = 0.15
            parameters[f'pi_G_gri_{t}'] = 0.3
            parameters[f'renewable_availability_{u}_{t}'] = 0.8
    
    # Create and solve with column generation
    cg_model = EnergyMarketColumnGeneration(players, time_periods, parameters)
    status = cg_model.solve()
    
    if status == "optimal":
        results = cg_model.get_results()
        print(f"\n=== COLUMN GENERATION RESULTS ===")
        print(f"Optimal objective value: {results['objective_value']:.2f}")
        print(f"Number of selected patterns: {len(results['selected_patterns'])}")
        print(f"Community prices: {results['community_prices']}")
    else:
        print(f"Column generation failed with status: {status}")
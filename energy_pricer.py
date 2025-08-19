from pyscipopt import Pricer, SCIP_RESULT, quicksum
import numpy as np
from energy_pricing_network import solve_energy_pricing_problem

class EnergyPricer(Pricer):
    """
    Pricer for Energy Community Column Generation
    """
    def __init__(self, players, time_periods, parameters, cons_flatten, cons_coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.players = players
        self.time_periods = time_periods
        self.params = parameters
        self.cons_flatten = cons_flatten
        self.cons_coeff = {k: cons_coeff[k] for k in cons_coeff if k in self.cons_flatten.keys()}
        
        # Player sets
        self.players_with_renewables = self.params.get('players_with_renewables', [])
        self.players_with_electrolyzers = self.params.get('players_with_electrolyzers', [])  
        self.players_with_heatpumps = self.params.get('players_with_heatpumps', [])
        self.players_with_elec_storage = self.params.get('players_with_elec_storage', [])
        self.players_with_hydro_storage = self.params.get('players_with_hydro_storage', [])
        self.players_with_heat_storage = self.params.get('players_with_heat_storage', [])
        
        # Combined sets
        self.U_E = list(set(self.players_with_renewables + self.players_with_elec_storage))
        self.U_G = list(set(self.players_with_electrolyzers + self.players_with_hydro_storage))
        self.U_H = list(set(self.players_with_heatpumps + self.players_with_heat_storage))
        
        # Iteration counter
        self.iter = 0
        self.global_iter = 0
        self.has_new_patterns = False
        self.patterns = {}
        
        # subproblem
        self.subproblems = {}
    def create_subproblems(self):
        self.subproblems = self.create_individual_player_models(self.players, self.time_periods, self.params)
        return self
    def pricerredcost(self):
        """Called by SCIP to perform pricing"""
        self.iter += 1
        self.global_iter += 1
        self.farkas = False
        return self.price(farkas=False)
    
    def price(self, farkas):
        print(f"-------------iteration {self.iter}-------------")
        
        # Get dual values
        if farkas:
            func = self.model.getDualfarkasLinear
        else:
            func = self.model.getDualsolLinear
        
        dual_values = self.get_dual_values(self.cons_coeff, self.cons_flatten, func)
        
        # Track if any negative reduced cost patterns found
        found_improving_pattern = False
        
        # Solve pricing problem for each player
        for u in self.players:
            # Prepare subproblem objective coefficients
            subprob_obj = self.prepare_subproblem_objective(u, dual_values, farkas)
            
            # Solve pricing problem
            min_redcost, pattern, objval = self.solve_subproblem(
                player_instance=self.subproblems[u],
                subprob_obj=subprob_obj
            )
            
            print(f"Player {u}: min_redcost = {min_redcost:.4f}, objval = {objval:.4f}")
            
            # If negative reduced cost (or positive for farkas), add pattern
            if farkas:
                if min_redcost > 1e-6:  # For farkas, we want positive reduced cost
                    found_improving_pattern = True
                    self.add_pattern_to_rmp(u, pattern)
            else:
                if min_redcost < -1e-6:  # For regular pricing, we want negative reduced cost
                    found_improving_pattern = True
                    self.add_pattern_to_rmp(u, pattern)

        # Return result to SCIP
        if found_improving_pattern:
            return {'result': SCIP_RESULT.SUCCESS}
        else:
            return {'result': SCIP_RESULT.SUCCESS}  # No more improving patterns
    def solve_subproblem(self, player_instance, subprob_obj):
        model = player_instance.model
        subprob = player_instance
        player = player_instance.players[0]
        if len(subprob.players) > 1:
            raise ValueError("Player instance should have only one player")
        dual_obj_terms = []
        for t in subprob.time_periods:
            # Electricity community balance dual
            if player in subprob.players_with_renewables or player in subprob.players_with_elec_storage or player in subprob.players_with_nfl_elec_demand or player in subprob.players_with_fl_elec_demand:
                pi_com_elec_export = subprob_obj.get(f"e_E_com_{player}_{t}",0)
                pi_com_elec_import = subprob_obj.get(f"i_E_com_{player}_{t}",0)
                dual_obj_terms.append(pi_com_elec_import * subprob.i_E_com.get((player,t),0))
                dual_obj_terms.append(pi_com_elec_export * subprob.e_E_com.get((player,t),0))
        
            # Heat community balance dual
            if player in subprob.players_with_heatpumps or player in subprob.players_with_heat_storage or player in subprob.players_with_nfl_heat_demand:
                pi_com_heat_export = subprob_obj.get(f"e_H_com_{player}_{t}",0)
                pi_com_heat_import = subprob_obj.get(f"i_H_com_{player}_{t}",0)
                dual_obj_terms.append(pi_com_heat_import * subprob.i_H_com.get((player,t),0))
                dual_obj_terms.append(pi_com_heat_export * subprob.e_H_com.get((player,t),0))
            
            # Hydrogen community balance dual (note: opposite sign)
            if player in subprob.players_with_electrolyzers or player in subprob.players_with_hydro_storage or player in subprob.players_with_nfl_hydro_demand:
                pi_com_hydro_export = subprob_obj.get(f"e_G_com_{player}_{t}",0)
                pi_com_hydro_import = subprob_obj.get(f"i_G_com_{player}_{t}",0)
                dual_obj_terms.append(pi_com_hydro_import * subprob.i_G_com.get((player,t),0))
                dual_obj_terms.append(pi_com_hydro_export * subprob.e_G_com.get((player,t),0))
        cons_convexity = subprob_obj[f"cons_convexity_{player}"]
        primal_obj = subprob.primal_obj
        dual_obj = quicksum(dual_obj_terms)
        model.setObjective(primal_obj + dual_obj, "minimize")
        model.optimize()
        objval = model.getObjVal()
        min_redcost = objval + cons_convexity
        pattern = self.extract_pattern_from_subproblem(subprob)
        model.freeTransform()
        return min_redcost, pattern, objval
    def extract_pattern_from_subproblem(self, player_subprob):
        """
        Extract pattern from solved subproblem
        
        Args:
            player_instance: LocalEnergyMarket instance after solving
        
        Returns:
            dict: Pattern dictionary with variable values
        """
        pattern = {}
        model = player_subprob.model
        player = player_subprob.players[0]
        
        for t in player_subprob.time_periods:
            # Extract electricity trading variables
            if (player, t) in player_subprob.e_E_gri:
                if 'e_E_gri' not in pattern:
                    pattern['e_E_gri'] = {}
                pattern['e_E_gri'][t] = model.getVal(player_subprob.e_E_gri[player, t])
            
            if (player, t) in player_subprob.i_E_gri:
                if 'i_E_gri' not in pattern:
                    pattern['i_E_gri'] = {}
                pattern['i_E_gri'][t] = model.getVal(player_subprob.i_E_gri[player, t])
            
            if (player, t) in player_subprob.e_E_com:
                if 'e_E_com' not in pattern:
                    pattern['e_E_com'] = {}
                pattern['e_E_com'][t] = model.getVal(player_subprob.e_E_com[player, t])
            
            if (player, t) in player_subprob.i_E_com:
                if 'i_E_com' not in pattern:
                    pattern['i_E_com'] = {}
                pattern['i_E_com'][t] = model.getVal(player_subprob.i_E_com[player, t])
            
            # Extract heat trading variables
            if (player, t) in player_subprob.e_H_gri:
                if 'e_H_gri' not in pattern:
                    pattern['e_H_gri'] = {}
                pattern['e_H_gri'][t] = model.getVal(player_subprob.e_H_gri[player, t])
            
            if (player, t) in player_subprob.i_H_gri:
                if 'i_H_gri' not in pattern:
                    pattern['i_H_gri'] = {}
                pattern['i_H_gri'][t] = model.getVal(player_subprob.i_H_gri[player, t])
            
            if (player, t) in player_subprob.e_H_com:
                if 'e_H_com' not in pattern:
                    pattern['e_H_com'] = {}
                pattern['e_H_com'][t] = model.getVal(player_subprob.e_H_com[player, t])
            
            if (player, t) in player_subprob.i_H_com:
                if 'i_H_com' not in pattern:
                    pattern['i_H_com'] = {}
                pattern['i_H_com'][t] = model.getVal(player_subprob.i_H_com[player, t])
            
            # Extract hydrogen trading variables
            if (player, t) in player_subprob.e_G_gri:
                if 'e_G_gri' not in pattern:
                    pattern['e_G_gri'] = {}
                pattern['e_G_gri'][t] = model.getVal(player_subprob.e_G_gri[player, t])
            
            if (player, t) in player_subprob.i_G_gri:
                if 'i_G_gri' not in pattern:
                    pattern['i_G_gri'] = {}
                pattern['i_G_gri'][t] = model.getVal(player_subprob.i_G_gri[player, t])
            
            if (player, t) in player_subprob.e_G_com:
                if 'e_G_com' not in pattern:
                    pattern['e_G_com'] = {}
                pattern['e_G_com'][t] = model.getVal(player_subprob.e_G_com[player, t])
            
            if (player, t) in player_subprob.i_G_com:
                if 'i_G_com' not in pattern:
                    pattern['i_G_com'] = {}
                pattern['i_G_com'][t] = model.getVal(player_subprob.i_G_com[player, t])
            
            # Extract production variables
            if (player, 'res', t) in player_subprob.p:
                if 'p_res' not in pattern:
                    pattern['p_res'] = {}
                pattern['p_res'][t] = model.getVal(player_subprob.p[player, 'res', t])
            
            if (player, 'hp', t) in player_subprob.p:
                if 'p_hp' not in pattern:
                    pattern['p_hp'] = {}
                pattern['p_hp'][t] = model.getVal(player_subprob.p[player, 'hp', t])
            
            if (player, 'els', t) in player_subprob.p:
                if 'p_els' not in pattern:
                    pattern['p_els'] = {}
                pattern['p_els'][t] = model.getVal(player_subprob.p[player, 'els', t])
            
            # Extract electrolyzer binary variables
            if (player, t) in player_subprob.z_su:
                if 'z_su' not in pattern:
                    pattern['z_su'] = {}
                pattern['z_su'][t] = model.getVal(player_subprob.z_su[player, t])
            
            if (player, t) in player_subprob.z_on:
                if 'z_on' not in pattern:
                    pattern['z_on'] = {}
                pattern['z_on'][t] = model.getVal(player_subprob.z_on[player, t])
            
            if (player, t) in player_subprob.z_off:
                if 'z_off' not in pattern:
                    pattern['z_off'] = {}
                pattern['z_off'][t] = model.getVal(player_subprob.z_off[player, t])
            
            if (player, t) in player_subprob.z_sb:
                if 'z_sb' not in pattern:
                    pattern['z_sb'] = {}
                pattern['z_sb'][t] = model.getVal(player_subprob.z_sb[player, t])
            
            # Extract electricity storage variables
            if (player, t) in player_subprob.b_dis_E:
                if 'b_dis_E' not in pattern:
                    pattern['b_dis_E'] = {}
                pattern['b_dis_E'][t] = model.getVal(player_subprob.b_dis_E[player, t])
            
            if (player, t) in player_subprob.b_ch_E:
                if 'b_ch_E' not in pattern:
                    pattern['b_ch_E'] = {}
                pattern['b_ch_E'][t] = model.getVal(player_subprob.b_ch_E[player, t])
            
            if (player, t) in player_subprob.s_E:
                if 's_E' not in pattern:
                    pattern['s_E'] = {}
                pattern['s_E'][t] = model.getVal(player_subprob.s_E[player, t])
            
            # Extract hydrogen storage variables
            if (player, t) in player_subprob.b_dis_G:
                if 'b_dis_G' not in pattern:
                    pattern['b_dis_G'] = {}
                pattern['b_dis_G'][t] = model.getVal(player_subprob.b_dis_G[player, t])
            
            if (player, t) in player_subprob.b_ch_G:
                if 'b_ch_G' not in pattern:
                    pattern['b_ch_G'] = {}
                pattern['b_ch_G'][t] = model.getVal(player_subprob.b_ch_G[player, t])
            
            if (player, t) in player_subprob.s_G:
                if 's_G' not in pattern:
                    pattern['s_G'] = {}
                pattern['s_G'][t] = model.getVal(player_subprob.s_G[player, t])
            
            # Extract heat storage variables
            if (player, t) in player_subprob.b_dis_H:
                if 'b_dis_H' not in pattern:
                    pattern['b_dis_H'] = {}
                pattern['b_dis_H'][t] = model.getVal(player_subprob.b_dis_H[player, t])
            
            if (player, t) in player_subprob.b_ch_H:
                if 'b_ch_H' not in pattern:
                    pattern['b_ch_H'] = {}
                pattern['b_ch_H'][t] = model.getVal(player_subprob.b_ch_H[player, t])
            
            if (player, t) in player_subprob.s_H:
                if 's_H' not in pattern:
                    pattern['s_H'] = {}
                pattern['s_H'][t] = model.getVal(player_subprob.s_H[player, t])
        
        return pattern
    def create_individual_player_models(self, players, time_periods, parameters):
        import copy
        from compact import LocalEnergyMarket
        individual_problems = {}
        for player in players:
            print(f"Creating individual problem for player {player}...")
            
            # 해당 플레이어만 포함하는 파라미터 복사
            player_params = copy.deepcopy(parameters)
            
            # 플레이어별 세트 재정의 - 해당 플레이어만 포함
            if player in parameters.get('players_with_renewables', []):
                player_params['players_with_renewables'] = [player]
            else:
                player_params['players_with_renewables'] = []
                
            if player in parameters.get('players_with_electrolyzers', []):
                player_params['players_with_electrolyzers'] = [player]
            else:
                player_params['players_with_electrolyzers'] = []
                
            if player in parameters.get('players_with_heatpumps', []):
                player_params['players_with_heatpumps'] = [player]
            else:
                player_params['players_with_heatpumps'] = []
                
            if player in parameters.get('players_with_elec_storage', []):
                player_params['players_with_elec_storage'] = [player]
            else:
                player_params['players_with_elec_storage'] = []
                
            if player in parameters.get('players_with_hydro_storage', []):
                player_params['players_with_hydro_storage'] = [player]
            else:
                player_params['players_with_hydro_storage'] = []
                
            if player in parameters.get('players_with_heat_storage', []):
                player_params['players_with_heat_storage'] = [player]
            else:
                player_params['players_with_heat_storage'] = []
                
            if player in parameters.get('players_with_nfl_elec_demand', []):
                player_params['players_with_nfl_elec_demand'] = [player]
            else:
                player_params['players_with_nfl_elec_demand'] = []
                
            if player in parameters.get('players_with_nfl_hydro_demand', []):
                player_params['players_with_nfl_hydro_demand'] = [player]
            else:
                player_params['players_with_nfl_hydro_demand'] = []
                
            if player in parameters.get('players_with_nfl_heat_demand', []):
                player_params['players_with_nfl_heat_demand'] = [player]
            else:
                player_params['players_with_nfl_heat_demand'] = []
                
            if player in parameters.get('players_with_fl_elec_demand', []):
                player_params['players_with_fl_elec_demand'] = [player]
            else:
                player_params['players_with_fl_elec_demand'] = []
                
            if player in parameters.get('players_with_fl_hydro_demand', []):
                player_params['players_with_fl_hydro_demand'] = [player]
            else:
                player_params['players_with_fl_hydro_demand'] = []
                
            if player in parameters.get('players_with_fl_heat_demand', []):
                player_params['players_with_fl_heat_demand'] = [player]
            else:
                player_params['players_with_fl_heat_demand'] = []
            
            # dwr=True로 설정하여 community balance 제약 비활성화
            # (개별 플레이어 문제이므로 community trading 제약 불필요)
            individual_problem = LocalEnergyMarket(
                players=[player],  # 단일 플레이어만
                time_periods=time_periods,
                parameters=player_params,
                isLP=False,  # Binary variables 포함
                dwr=True  # Disable community balance constraints
            )
            individual_problem.primal_obj = individual_problem.model.getObjective()
            individual_problem.model.hideOutput()
            individual_problems[player] = individual_problem
            
        return individual_problems
    def pricerfarkas(self):
        self.iter += 1
        self.global_iter += 1
        self.farkas = True
        print(f"-------------farkas pricing started-------------")
        return self.price(farkas=True)
    def get_dual_values(self, coeff_dict, cons_dict, func):
        """Get dual values from current LP solution"""
        
        dual_values = {}
        
        # Get all constraint duals
        for outer_key, value in coeff_dict.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    try:
                        cons = cons_dict[outer_key][inner_key]
                    except:
                        print(f"Error: {outer_key} {inner_key} not found")
                    t_cons = self.model.getTransformedCons(cons)
                    pi = func(t_cons)
                    for varname, coeff in inner_value.items():
                        dual_values[varname] = pi * coeff + dual_values.get(varname, 0)
            else:
                raise("Error: constraint type is not a dictionary")
        for key in cons_dict["cons_convexity"]:
            cons = cons_dict["cons_convexity"][key]
            t_cons = self.model.getTransformedCons(cons)
            pi = func(t_cons)
            dual_values[key] = pi
        
        return dual_values
    # def get_dual_values(self, coeff_dict, cons_dict, func):
    #     """Get dual values from current LP solution"""
        
    #     dual_values = {}
        
    #     # Get constraint duals for community balance and peak power constraints
    #     for constraint_type in ['community_elec_balance', 'community_heat_balance', 
    #                            'community_hydrogen_balance']:
    #                            # 'peak_power']:
    #         if constraint_type in cons_dict:
    #             for key, cons in cons_dict[constraint_type].items():
    #                 t_cons = self.model.getTransformedCons(cons)
    #                 pi = func(t_cons)
    #                 dual_values[key] = pi
        
    #     # Get convexity constraint duals
    #     if 'cons_convexity' in cons_dict:
    #         for key, cons in cons_dict['cons_convexity'].items():
    #             t_cons = self.model.getTransformedCons(cons)
    #             pi = func(t_cons)
    #             dual_values[key] = pi
        
    #     # # Get chi_peak dual (reduced cost)
    #     # if 'chi_peak' in self.model.data['vars']:
    #     #     chi_peak_var = self.model.data['vars']['chi_peak']
    #     #     # For chi_peak, we need its reduced cost
    #     #     dual_values['chi_peak'] = self.model.getVarRedcost(chi_peak_var)
        
    #     return dual_values
    
    def prepare_subproblem_objective(self, player, dual_values, farkas):
        """Prepare objective coefficients for pricing subproblem"""
        
        # Pass dual values directly to the pricing problem
        # The pricing problem will handle the signs correctly
        subprob_obj = {k:-1*v for k,v in dual_values.items()}
        
        return subprob_obj
    
    def add_pattern_to_rmp(self, player, pattern):
        """Add a new pattern to the RMP"""
        
        pattern_id = f"pattern_{player}_{self.iter}_{self.global_iter}"
        
        # Calculate pattern cost
        pattern_cost = self.calculate_pattern_cost(player, pattern)
        
        # Create pattern variable with objective coefficient
        var = self.model.addVar(name=pattern_id, obj=pattern_cost, lb=0.0, pricedVar=True)
        
        # Store pattern
        self.patterns[pattern_id] = pattern
        
        # Add to convexity constraint
        cons_conv = self.cons_flatten["cons_convexity"][f"cons_convexity_{player}"]
        t_cons_conv = self.model.getTransformedCons(cons_conv)
        self.model.addConsCoeff(t_cons_conv, var, 1.0)
        
        # Calculate pattern coefficients for community constraints
        for t in self.time_periods:
            # Electricity balance coefficient
            e_coeff = pattern.get('i_E_com', {}).get(t, 0) - pattern.get('e_E_com', {}).get(t, 0)
            cons_e = self.cons_flatten["community_elec_balance"][f"community_elec_balance_{t}"]
            t_cons_e = self.model.getTransformedCons(cons_e)
            self.model.addConsCoeff(t_cons_e, var, e_coeff)
            
            # Heat balance coefficient
            h_coeff = pattern.get('i_H_com', {}).get(t, 0) - pattern.get('e_H_com', {}).get(t, 0)
            cons_h = self.cons_flatten["community_heat_balance"][f"community_heat_balance_{t}"]
            t_cons_h = self.model.getTransformedCons(cons_h)
            self.model.addConsCoeff(t_cons_h, var, h_coeff)
        
            # Hydrogen balance coefficient (note: opposite sign)
            g_coeff = pattern.get('i_G_com', {}).get(t, 0) - pattern.get('e_G_com', {}).get(t, 0)
            cons_g = self.cons_flatten["community_hydro_balance"][f"community_hydro_balance_{t}"]
            t_cons_g = self.model.getTransformedCons(cons_g)
            self.model.addConsCoeff(t_cons_g, var, g_coeff)
            
            # # Peak power coefficient
            # p_coeff = pattern.get('i_E_gri', {}).get(t, 0) - pattern.get('e_E_gri', {}).get(t, 0)
            # if abs(p_coeff) > 1e-10:
            #     cons_p = self.cons_flatten["peak_power"][f"peak_power_{t}"]
            #     t_cons_p = self.model.getTransformedCons(cons_p)
            #     self.model.addConsCoeff(t_cons_p, var, p_coeff)
        
        print(f"Added pattern {pattern_id} for player {player} with cost {pattern_cost:.4f}")
        if self.iter > 20:
            print(1)
    def calculate_pattern_cost(self, player, pattern):
        """Calculate the cost of a pattern"""
        
        cost = 0
        
        # Production costs
        if 'p_res' in pattern:
            c_res = self.params.get(f'c_res_{player}', 0)
            for t, val in pattern['p_res'].items():
                cost += c_res * val
        
        if 'p_hp' in pattern:
            c_hp = self.params.get(f'c_hp_{player}', 0)
            for t, val in pattern['p_hp'].items():
                cost += c_hp * val
        
        if 'p_els' in pattern:
            c_els = self.params.get(f'c_els_{player}', 0)
            for t, val in pattern['p_els'].items():
                cost += c_els * val
        
        # Startup costs for electrolyzers
        if 'z_su' in pattern:
            c_su = self.params.get(f'c_su_{player}', 0)
            for t, val in pattern['z_su'].items():
                cost += c_su * val
        
        # Grid interaction costs (use separate import/export prices)
        for t in self.time_periods:
            # Electricity
            pi_E_gri_import = self.params.get(f'pi_E_gri_import_{t}', 0)
            pi_E_gri_export = self.params.get(f'pi_E_gri_export_{t}', 0)
            
            if 'i_E_gri' in pattern and t in pattern['i_E_gri']:
                cost += pi_E_gri_import * pattern['i_E_gri'][t]
            if 'e_E_gri' in pattern and t in pattern['e_E_gri']:
                cost -= pi_E_gri_export * pattern['e_E_gri'][t]  # Export generates revenue
            
            # Heat
            pi_H_gri_import = self.params.get(f'pi_H_gri_import_{t}', 0)
            pi_H_gri_export = self.params.get(f'pi_H_gri_export_{t}', 0)
            
            if 'i_H_gri' in pattern and t in pattern['i_H_gri']:
                cost += pi_H_gri_import * pattern['i_H_gri'][t]
            if 'e_H_gri' in pattern and t in pattern['e_H_gri']:
                cost -= pi_H_gri_export * pattern['e_H_gri'][t]
            
            # Hydrogen
            pi_G_gri_import = self.params.get(f'pi_G_gri_import_{t}', 0)
            pi_G_gri_export = self.params.get(f'pi_G_gri_export_{t}', 0)
            
            if 'i_G_gri' in pattern and t in pattern['i_G_gri']:
                cost += pi_G_gri_import * pattern['i_G_gri'][t]
            if 'e_G_gri' in pattern and t in pattern['e_G_gri']:
                cost -= pi_G_gri_export * pattern['e_G_gri'][t]
        
        # Storage costs
        c_sto = self.params.get('c_sto', 0.01)
        nu_ch = self.params.get('nu_ch', 0.9)
        nu_dis = self.params.get('nu_dis', 0.9)
        
        # Electricity storage
        if 'b_ch_E' in pattern:
            for t, val in pattern['b_ch_E'].items():
                cost += c_sto * nu_ch * val
        if 'b_dis_E' in pattern:
            for t, val in pattern['b_dis_E'].items():
                cost += c_sto * (1/nu_dis) * val
        
        # Hydrogen storage
        if 'b_ch_G' in pattern:
            for t, val in pattern['b_ch_G'].items():
                cost += c_sto * nu_ch * val
        if 'b_dis_G' in pattern:
            for t, val in pattern['b_dis_G'].items():
                cost += c_sto * (1/nu_dis) * val
        
        # Heat storage
        if 'b_ch_H' in pattern:
            for t, val in pattern['b_ch_H'].items():
                cost += c_sto * nu_ch * val
        if 'b_dis_H' in pattern:
            for t, val in pattern['b_dis_H'].items():
                cost += c_sto * (1/nu_dis) * val
        return cost
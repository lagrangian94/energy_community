from pyscipopt import Pricer, SCIP_RESULT
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
            min_redcost, pattern, objval = solve_energy_pricing_problem(
                player=u,
                time_periods=self.time_periods,
                params=self.params,
                subprob_obj=subprob_obj,
                iter=self.iter,
                current_node=self.model.getCurrentNode()
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
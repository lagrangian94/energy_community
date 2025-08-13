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
        
        # Pricing subproblems (one per player)
        self.pricing_models = {}
        
    def pricerredcost(self):
        """Called by SCIP to perform pricing"""
        self.iter += 1
        self.global_iter +=1
        self.farkas = False
        return self.price(farkas=False)
    def price(self, farkas):
        print(f"-------------iteration {self.iter}-------------")
        # Get dual values
        if farkas:
            func = self.model.getDualfarkasLinear
        else:
            func = self.model.getDualsolLinear
        coeff_dict = self.cons_coeff
        cons_dict = self.cons_flatten
        dual_values = self.get_dual_values(coeff_dict, cons_dict,func)
        
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
                current_node=self.model.getCurrentNode()
            )
            
            print(f"Player {u}: min_redcost = {min_redcost}, objval = {objval}")
            
            # If negative reduced cost, add pattern
            if farkas:
                min_redcost -= 1
            if min_redcost < -1e-6:
                found_improving_pattern = True
                self.add_pattern_to_rmp(u, pattern)
        
        if self.iter > 20:
            print(1)
        
        # Return result to SCIP
        if found_improving_pattern:
            return {'result': SCIP_RESULT.SUCCESS}
        else:
            return {'result': SCIP_RESULT.SUCCESS}  # No more improving patterns
    
    def pricerfarkas(self):
        self.iter += 1
        self.global_iter +=1
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
                    cons = cons_dict[outer_key][inner_key]
                    t_cons = self.model.getTransformedCons(cons)
                    pi = func(t_cons)
                    for varname, coeff in inner_value.items():
                        dual_values[varname] = pi * coeff + dual_values.get(varname, 0)
            else:
                raise("이상한게 들어있음")
        for key in cons_dict["cons_convexity"]:
            cons = cons_dict["cons_convexity"][key]
            t_cons = self.model.getTransformedCons(cons)
            pi = func(t_cons)
            dual_values[key] = pi
        
        return dual_values
    
    def prepare_subproblem_objective(self, player, dual_values, farkas):
        """Prepare objective coefficients for pricing subproblem"""
        
        subprob_obj = {key:-1*val for key, val in dual_values.items()}
        if not farkas:
            subprob_obj["chi_peak"] = subprob_obj.get("chi_peak",0) +self.model.data["vars"]["chi_peak"].getObj()
        return subprob_obj
    
    def add_pattern_to_rmp(self, player, pattern):
        """Add a new pattern to the RMP"""
        
        pattern_id = f"pattern_{player}_{self.iter}_{self.global_iter}"
        
        # Calculate pattern cost first
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
            if e_coeff != 0:
                cons_e = self.cons_flatten["community_elec_balance"][f"community_elec_balance_{t}"]
                t_cons_e = self.model.getTransformedCons(cons_e)
                self.model.addConsCoeff(t_cons_e, var, e_coeff)
            
            # Heat balance coefficient
            h_coeff = pattern.get('i_H_com', {}).get(t, 0) - pattern.get('e_H_com', {}).get(t, 0)
            if h_coeff != 0:
                cons_h = self.cons_flatten["community_heat_balance"][f"community_heat_balance_{t}"]
                t_cons_h = self.model.getTransformedCons(cons_h)
                self.model.addConsCoeff(t_cons_h, var, h_coeff)
            
            # Hydrogen balance coefficient (note: opposite sign)
            g_coeff = pattern.get('e_G_com', {}).get(t, 0) - pattern.get('i_G_com', {}).get(t, 0)
            if g_coeff != 0:
                cons_g = self.cons_flatten["community_hydrogen_balance"][f"community_hydrogen_balance_{t}"]
                t_cons_g = self.model.getTransformedCons(cons_g)
                self.model.addConsCoeff(t_cons_g, var, g_coeff)
            
            # Peak power coefficient
            p_coeff = pattern.get('i_E_gri', {}).get(t, 0) - pattern.get('e_E_gri', {}).get(t, 0)
            if p_coeff != 0:
                cons_p = self.cons_flatten["peak_power"][f"peak_power_{t}"]
                t_cons_p = self.model.getTransformedCons(cons_p)
                self.model.addConsCoeff(t_cons_p, var, p_coeff)
        
        print(f"Added pattern {pattern_id} with cost {pattern_cost}")
        
    def calculate_pattern_cost(self, player, pattern):
        """Calculate the cost of a pattern"""
        
        cost = 0
        
        # Production costs
        if 'p_res' in pattern:
            c_res = self.params.get(f'c_res_{player}', 0)
            for t in self.time_periods:
                cost += c_res * pattern['p_res'].get(t, 0)
        
        if 'p_hp' in pattern:
            c_hp = self.params.get(f'c_hp_{player}', 0)
            for t in self.time_periods:
                cost += c_hp * pattern['p_hp'].get(t, 0)
        
        if 'p_els' in pattern:
            c_els = self.params.get(f'c_els_{player}', 0)
            for t in self.time_periods:
                cost += c_els * pattern['p_els'].get(t, 0)
        
        # Startup costs for electrolyzers
        if 'z_su' in pattern:
            c_su = self.params.get(f'c_su_{player}', 0)
            for t in self.time_periods:
                cost += c_su * pattern['z_su'].get(t, 0)
        
        # Grid interaction costs
        for t in self.time_periods:
            pi_E_gri = self.params.get(f'pi_E_gri_{t}', 0)
            pi_H_gri = self.params.get(f'pi_H_gri_{t}', 0)
            pi_G_gri = self.params.get(f'pi_G_gri_{t}', 0)
            
            cost += pi_E_gri * (pattern.get('i_E_gri', {}).get(t, 0) - pattern.get('e_E_gri', {}).get(t, 0))
            cost += pi_H_gri * (pattern.get('i_H_gri', {}).get(t, 0) - pattern.get('e_H_gri', {}).get(t, 0))
            cost += pi_G_gri * (pattern.get('i_G_gri', {}).get(t, 0) - pattern.get('e_G_gri', {}).get(t, 0))
        
        # Storage costs
        c_sto = self.params.get(f'c_sto', 0.01)
        nu_ch = self.params.get('nu_ch', 0.9)
        nu_dis = self.params.get('nu_dis', 0.9)
        
        for t in self.time_periods:
            # Electricity storage
            if 'b_ch_E' in pattern and 'b_dis_E' in pattern:
                cost += c_sto * (nu_ch * pattern['b_ch_E'].get(t, 0) + 
                               (1/nu_dis) * pattern['b_dis_E'].get(t, 0))
            
            # Hydrogen storage
            if 'b_ch_G' in pattern and 'b_dis_G' in pattern:
                cost += c_sto * (nu_ch * pattern['b_ch_G'].get(t, 0) + 
                               (1/nu_dis) * pattern['b_dis_G'].get(t, 0))
            
            # Heat storage
            if 'b_ch_H' in pattern and 'b_dis_H' in pattern:
                cost += c_sto * (nu_ch * pattern['b_ch_H'].get(t, 0) + 
                               (1/nu_dis) * pattern['b_dis_H'].get(t, 0))
        
        return cost
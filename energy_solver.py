from pyscipopt import Model, quicksum
import numpy as np
from compact import process_cons_arr, solve_and_extract_results
from create_initial_patterns import create_initial_patterns_for_players
class EnergyCommunitySolver:
    """
    Energy Community Solver using Convex Hull Pricing
    Manages the Restricted Master Problem (RMP)
    """
    def __init__(self, players, time_periods, parameters):
        self.players = players
        self.time_periods = time_periods
        self.params = parameters
        self.model = Model("EnergyCommunitySolver")
        
        # Player sets
        self.players_with_renewables = self.params.get('players_with_renewables', [])
        self.players_with_electrolyzers = self.params.get('players_with_electrolyzers', [])  
        self.players_with_heatpumps = self.params.get('players_with_heatpumps', [])
        self.players_with_elec_storage = self.params.get('players_with_elec_storage', [])
        self.players_with_hydro_storage = self.params.get('players_with_hydro_storage', [])
        self.players_with_heat_storage = self.params.get('players_with_heat_storage', [])
        self.players_with_nfl_elec_demand = self.params.get('players_with_nfl_elec_demand', [])
        self.players_with_nfl_hydro_demand = self.params.get('players_with_nfl_hydro_demand', [])
        self.players_with_nfl_heat_demand = self.params.get('players_with_nfl_heat_demand', [])
        self.players_with_fl_elec_demand = self.params.get('players_with_fl_elec_demand', [])
        self.players_with_fl_hydro_demand = self.params.get('players_with_fl_hydro_demand', [])
        self.players_with_fl_heat_demand = self.params.get('players_with_fl_heat_demand', [])
        # Combined sets
        self.U_E = list(set(self.players_with_renewables + self.players_with_elec_storage))
        self.U_G = list(set(self.players_with_electrolyzers + self.players_with_hydro_storage))
        self.U_H = list(set(self.players_with_heatpumps + self.players_with_heat_storage))
        ## 추후에 non-flexible demand를 가진 player들이 storage를 가지고 있다고 고려할 수도 있음.
        self.U_E_nfl = list(set(self.players_with_nfl_elec_demand))
        self.U_G_nfl = list(set(self.players_with_nfl_hydro_demand))
        self.U_H_nfl = list(set(self.players_with_nfl_heat_demand))
        self.U_E_fl = list(set(self.players_with_fl_elec_demand))
        self.U_G_fl = list(set(self.players_with_fl_hydro_demand))
        self.U_H_fl = list(set(self.players_with_fl_heat_demand))
        # Model data storage
        self.model.data = {}
        self.model.data["vars_pattern"] = {}
        self.model.data["cons"] = {}
        self.model.data["vars"] = {}
        
        # Constraint storage for dual access
        self.cons_flatten = {}
        self.cons_coeff = {}
        self.cons_cpt = {}
        # Pattern storage
        self.patterns = {}
        self.pattern_counter = 0
        
    def init_rmp(self, init_patterns):
        """Initialize the Restricted Master Problem with initial patterns"""
        
        # Create convexity constraints for each player
        self.model.data["cons"]["cons_convexity"] = {}
        for u in self.players:
            cons = self.model.addCons(quicksum([]) == 1, name=f"cons_convexity_{u}", modifiable=True)
            self.model.data["cons"]["cons_convexity"][f"cons_convexity_{u}"] = cons
        
        # Create community balance constraints (these will have patterns added)
        self.model.data["cons"]["community_elec_balance"] = {}
        self.model.data["cons"]["community_heat_balance"] = {}
        self.model.data["cons"]["community_hydro_balance"] = {}
        # self.model.data["cons"]["peak_power"] = {}
        
        # Create chi_peak variable first
        # pi_peak = self.params.get('pi_peak', 0)
        # self.chi_peak = self.model.addVar(vtype="C", name="chi_peak", lb=0, obj=pi_peak)
        # self.model.data["vars"]["chi_peak"] = self.chi_peak
        for t in self.time_periods:
            # Community electricity balance: sum(i_E_com - e_E_com) == 0
            cons_e = self.model.addCons(quicksum([]) == 0, name=f"community_elec_balance_{t}", modifiable=True)
            self.model.data["cons"]["community_elec_balance"][f"community_elec_balance_{t}"] = cons_e
            
            # Community heat balance: sum(i_H_com - e_H_com) == 0
            cons_h = self.model.addCons(quicksum([]) == 0, name=f"community_heat_balance_{t}", modifiable=True)
            self.model.data["cons"]["community_heat_balance"][f"community_heat_balance_{t}"] = cons_h
            
            # Community hydrogen balance: sum(e_G_com - i_G_com) == 0 (note: opposite sign)
            cons_g = self.model.addCons(quicksum([]) == 0, name=f"community_hydro_balance_{t}", modifiable=True)
            self.model.data["cons"]["community_hydro_balance"][f"community_hydro_balance_{t}"] = cons_g
            
            # # Peak power constraint: sum(i_E_gri - e_E_gri) <= chi_peak
            # cons_p = self.model.addCons(quicksum([]) <= self.chi_peak, name=f"peak_power_{t}", modifiable=True)
            # self.model.data["cons"]["peak_power"][f"peak_power_{t}"] = cons_p
        
        # Add initial patterns
        for u in self.players:
            if u in init_patterns:
                self.add_pattern(u, init_patterns[u])
        
        # Set objective direction
        self.model.setMinimize()
        self.reshape_and_create_cons_coeff()
        
    def add_pattern(self, player, pattern):
        """Add a new pattern (column) to the RMP"""
        
        pattern_id = f"pattern_{player}_{self.pattern_counter}"
        self.pattern_counter += 1
        
        # Calculate pattern cost first
        pattern_cost = self.calculate_pattern_cost(player, pattern)
        
        # Create pattern variable with objective coefficient
        var = self.model.addVar(vtype="C", lb=0, name=pattern_id, obj=pattern_cost)
        
        # Store pattern
        self.patterns[pattern_id] = pattern
        if player not in self.model.data["vars_pattern"]:
            self.model.data["vars_pattern"][player] = []
        self.model.data["vars_pattern"][player].append(var)
        
        # Add to convexity constraint
        cons_conv = self.model.data["cons"]["cons_convexity"][f"cons_convexity_{player}"]
        self.model.addConsCoeff(cons_conv, var, 1.0)
        
        # Calculate pattern coefficients for community constraints
        for t in self.time_periods:
            # Electricity balance coefficient
            e_coeff = pattern.get('i_E_com', {}).get(t, 0) - pattern.get('e_E_com', {}).get(t, 0)
            if e_coeff != 0:
                cons_e = self.model.data["cons"]["community_elec_balance"][f"community_elec_balance_{t}"]
                self.model.addConsCoeff(cons_e, var, e_coeff)
            
            # Heat balance coefficient
            h_coeff = pattern.get('i_H_com', {}).get(t, 0) - pattern.get('e_H_com', {}).get(t, 0)
            if h_coeff != 0:
                cons_h = self.model.data["cons"]["community_heat_balance"][f"community_heat_balance_{t}"]
                self.model.addConsCoeff(cons_h, var, h_coeff)
            
            # Hydrogen balance coefficient (note: opposite sign)
            g_coeff = pattern.get('i_G_com', {}).get(t, 0) - pattern.get('e_G_com', {}).get(t, 0)
            if g_coeff != 0:
                cons_g = self.model.data["cons"]["community_hydro_balance"][f"community_hydro_balance_{t}"]
                self.model.addConsCoeff(cons_g, var, g_coeff)
            
            # # Peak power coefficient
            # p_coeff = pattern.get('i_E_gri', {}).get(t, 0) - pattern.get('e_E_gri', {}).get(t, 0)
            # if p_coeff != 0:
            #     cons_p = self.model.data["cons"]["peak_power"][f"peak_power_{t}"]
            #     self.model.addConsCoeff(cons_p, var, p_coeff)
        
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
    
    def get_dual_values(self):
        """Extract dual values for pricing problem"""
        
        dual_values = {}
        
        # Get convexity constraint duals
        for u in self.players:
            cons = self.model.data["cons"]["cons_convexity"][u]
            trans_cons = self.model.getTransformedCons(cons)
            dual_values[f"cons_convexity_{u}"] = self.model.getDualsolLinear(trans_cons)
        
        # Get community balance duals
        for t in self.time_periods:
            # Electricity
            cons_e = self.model.data["cons"]["community_elec_balance"][f"community_elec_balance_{t}"]
            trans_cons_e = self.model.getTransformedCons(cons_e)
            dual_values[f"community_elec_balance_{t}"] = self.model.getDualsolLinear(trans_cons_e)
            
            # Heat
            cons_h = self.model.data["cons"]["community_heat_balance"][f"community_heat_balance_{t}"]
            trans_cons_h = self.model.getTransformedCons(cons_h)
            dual_values[f"community_heat_balance_{t}"] = self.model.getDualsolLinear(trans_cons_h)
            
            # Hydrogen
            cons_g = self.model.data["cons"]["community_hydro_balance"][f"community_hydro_balance_{t}"]
            trans_cons_g = self.model.getTransformedCons(cons_g)
            dual_values[f"community_hydro_balance_{t}"] = self.model.getDualsolLinear(trans_cons_g)
            

        
        return dual_values
    
    def solve(self):
        """Solve the RMP"""
        self.model.optimize()
        return self.model.getStatus()
    
    def get_objective_value(self):
        """Get the objective value"""
        return self.model.getObjVal()
    
    def get_solution(self):
        """Get the solution values"""
        solution = {}
        for player in self.players:
            if player in self.model.data["vars_pattern"]:
                solution[player] = {}
                for var in self.model.data["vars_pattern"][player]:
                    solution[player][var.name] = self.model.getVal(var)
        # solution['chi_peak'] = self.model.getVal(self.chi_peak)
        return solution
    def gen_initial_cols(self, folder_path=None):
        import os
        from compact import LocalEnergyMarket
        if not folder_path:
            folder_path = "."
        init_col_file = os.path.join(folder_path, f"init_col.npy")
        cons_coeff_file = os.path.join(folder_path, f"cons_coeff.npy")
        if os.path.exists(init_col_file) and os.path.exists(cons_coeff_file):
            pattern = np.load(init_col_file, allow_pickle=True).item()
            self.cons_cpt = np.load(cons_coeff_file, allow_pickle=True).item()
        else:
            cpt = LocalEnergyMarket(self.players, self.time_periods, self.params)
            status = cpt.solve()
            status, pattern = solve_and_extract_results(cpt.model)
            if status not in ["optimal", "gaplimit","interrupted", "time_limit"]:
                raise("initial column cannot be generated bc/ compact model infeasible")
            for key in cpt.model.data["cons"]:
                cons_arr = cpt.model.data["cons"][key]
                val = process_cons_arr(cons_arr, cpt.model.getValsLinear)
                if type(val) in [dict, np.ndarray]:
                    self.cons_cpt[key] = val
                else:
                    raise("이상한게 들어있음")
            cpt = None #Garbage collection
            np.save(init_col_file, pattern)
            np.save(cons_coeff_file, self.cons_cpt)

        return pattern
    def reshape_and_create_cons_coeff(self):
        for (key, val) in self.model.data["cons"].items():
            if key=="cons_convexity":
                self.cons_flatten[key] = val
            else:
                coeff_original = self.cons_cpt[key]
                self.cons_flatten[key] = val
                self.cons_coeff[key] = coeff_original
        return self
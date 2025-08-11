from pyscipopt import Model, quicksum
import numpy as np
from compact import LocalEnergyMarket, solve_and_extract_results

class PlayerPricingProblem(LocalEnergyMarket):
    """
    Individual player pricing problem for column generation
    Inherits from LocalEnergyMarket but modifies for single player
    """
    
    def __init__(self, player, time_periods, parameters, subprob_obj):
        """
        Initialize pricing problem for a single player
        
        Args:
            player: Single player ID
            time_periods: List of time period indices
            parameters: Dictionary containing all model parameters
            subprob_obj: Dictionary of dual values from RMP
        """
        # Store single player and subproblem objective
        self.single_player = player
        self.subprob_obj = subprob_obj
        
        # Call parent constructor with single player list
        super().__init__([player], time_periods, parameters)
        
        # Modify objective function to include dual values
        self._adjust_objective_with_duals()
    
    def _adjust_objective_with_duals(self):
        """Adjust all variable objective coefficients to include dual values"""
        
        # Get all variables from the model
        all_vars = self.model.getVars()
        
        # Create new objective by adding dual values to original coefficients
        obj_expr = quicksum(
            (var.getObj() + self.subprob_obj.get(var.name, 0)) * var 
            for var in all_vars
        )
        
        # Add convexity constraint dual as a constant term
        convexity_dual = self.subprob_obj.get(f'cons_convexity_{self.single_player}', 0)
        obj_expr += convexity_dual
        
        # Set the new objective
        self.model.setObjective(obj_expr, "minimize")
        
    def _create_objective(self):
        """Override objective function to include dual values"""
        
        obj_terms = []
        u = self.single_player  # Use single player directly
        
        # (1) Production costs (generator variable costs + startup costs)
        for t in self.time_periods:
            # Renewable generation costs
            if u in self.U_E and (u,'res',t) in self.p:
                c_res = self.params.get(f'c_res_{u}', 0)
                # Add dual value contribution
                dual_coeff = self.subprob_obj.get(f'p_res_{u}_{t}', 0)
                obj_terms.append((c_res + dual_coeff) * self.p[u,'res',t])
            
            # Heat pump costs
            if u in self.U_H and (u,'hp',t) in self.p:
                c_hp = self.params.get(f'c_hp_{u}', 0)
                dual_coeff = self.subprob_obj.get(f'p_hp_{u}_{t}', 0)
                obj_terms.append((c_hp + dual_coeff) * self.p[u,'hp',t])
            
            # Electrolyzer costs
            if u in self.U_G:
                if (u,'els',t) in self.p:
                    c_els = self.params.get(f'c_els_{u}', 0)
                    dual_coeff = self.subprob_obj.get(f'p_els_{u}_{t}', 0)
                    obj_terms.append((c_els + dual_coeff) * self.p[u,'els',t])
                if (u,t) in self.z_su:
                    c_su = self.params.get(f'c_su_{u}', 0)
                    dual_coeff = self.subprob_obj.get(f'z_su_{u}_{t}', 0)
                    obj_terms.append((c_su + dual_coeff) * self.z_su[u,t])
        
        # (2) Grid interaction costs with dual values
        for t in self.time_periods:
            pi_E_gri = self.params.get(f'pi_E_gri_{t}', 0)
            pi_H_gri = self.params.get(f'pi_H_gri_{t}', 0)
            pi_G_gri = self.params.get(f'pi_G_gri_{t}', 0)
            
            # Add dual values for each variable
            dual_i_E_gri = self.subprob_obj.get(f'i_E_gri_{u}_{t}', 0)
            dual_e_E_gri = self.subprob_obj.get(f'e_E_gri_{u}_{t}', 0)
            dual_i_H_gri = self.subprob_obj.get(f'i_H_gri_{u}_{t}', 0)
            dual_e_H_gri = self.subprob_obj.get(f'e_H_gri_{u}_{t}', 0)
            dual_i_G_gri = self.subprob_obj.get(f'i_G_gri_{u}_{t}', 0)
            dual_e_G_gri = self.subprob_obj.get(f'e_G_gri_{u}_{t}', 0)
            
            obj_terms.append((pi_E_gri + dual_i_E_gri) * self.i_E_gri[u,t])
            obj_terms.append((-pi_E_gri + dual_e_E_gri) * self.e_E_gri[u,t])
            obj_terms.append((pi_H_gri + dual_i_H_gri) * self.i_H_gri[u,t])
            obj_terms.append((-pi_H_gri + dual_e_H_gri) * self.e_H_gri[u,t])
            obj_terms.append((pi_G_gri + dual_i_G_gri) * self.i_G_gri[u,t])
            obj_terms.append((-pi_G_gri + dual_e_G_gri) * self.e_G_gri[u,t])
            
            # Add dual values for community trading variables
            dual_i_E_com = self.subprob_obj.get(f'i_E_com_{u}_{t}', 0)
            dual_e_E_com = self.subprob_obj.get(f'e_E_com_{u}_{t}', 0)
            dual_i_H_com = self.subprob_obj.get(f'i_H_com_{u}_{t}', 0)
            dual_e_H_com = self.subprob_obj.get(f'e_H_com_{u}_{t}', 0)
            dual_i_G_com = self.subprob_obj.get(f'i_G_com_{u}_{t}', 0)
            dual_e_G_com = self.subprob_obj.get(f'e_G_com_{u}_{t}', 0)
            
            obj_terms.append(dual_i_E_com * self.i_E_com[u,t])
            obj_terms.append(dual_e_E_com * self.e_E_com[u,t])
            obj_terms.append(dual_i_H_com * self.i_H_com[u,t])
            obj_terms.append(dual_e_H_com * self.e_H_com[u,t])
            obj_terms.append(dual_i_G_com * self.i_G_com[u,t])
            obj_terms.append(dual_e_G_com * self.e_G_com[u,t])
        
        # (3) Storage usage costs by type with dual values
        c_sto = self.params.get(f'c_sto', 0.01)
        nu_ch = self.params.get('nu_ch', 0.9)
        nu_dis = self.params.get('nu_dis', 0.9)
        
        for t in self.time_periods:
            # Electricity storage costs
            if u in self.players_with_elec_storage and (u,t) in self.b_ch_E and (u,t) in self.b_dis_E:
                dual_b_ch_E = self.subprob_obj.get(f'b_ch_E_{u}_{t}', 0)
                dual_b_dis_E = self.subprob_obj.get(f'b_dis_E_{u}_{t}', 0)
                obj_terms.append((c_sto * nu_ch + dual_b_ch_E) * self.b_ch_E[u,t])
                obj_terms.append((c_sto * (1/nu_dis) + dual_b_dis_E) * self.b_dis_E[u,t])
            
            # Hydrogen storage costs
            if u in self.players_with_hydro_storage and (u,t) in self.b_ch_G and (u,t) in self.b_dis_G:
                dual_b_ch_G = self.subprob_obj.get(f'b_ch_G_{u}_{t}', 0)
                dual_b_dis_G = self.subprob_obj.get(f'b_dis_G_{u}_{t}', 0)
                obj_terms.append((c_sto * nu_ch + dual_b_ch_G) * self.b_ch_G[u,t])
                obj_terms.append((c_sto * (1/nu_dis) + dual_b_dis_G) * self.b_dis_G[u,t])
            
            # Heat storage costs
            if u in self.players_with_heat_storage and (u,t) in self.b_ch_H and (u,t) in self.b_dis_H:
                dual_b_ch_H = self.subprob_obj.get(f'b_ch_H_{u}_{t}', 0)
                dual_b_dis_H = self.subprob_obj.get(f'b_dis_H_{u}_{t}', 0)
                obj_terms.append((c_sto * nu_ch + dual_b_ch_H) * self.b_ch_H[u,t])
                obj_terms.append((c_sto * (1/nu_dis) + dual_b_dis_H) * self.b_dis_H[u,t])
            
            # Add dual values for storage SOC variables
            if u in self.players_with_elec_storage and (u,t) in self.s_E:
                dual_s_E = self.subprob_obj.get(f's_E_{u}_{t}', 0)
                obj_terms.append(dual_s_E * self.s_E[u,t])
            
            if u in self.players_with_hydro_storage and (u,t) in self.s_G:
                dual_s_G = self.subprob_obj.get(f's_G_{u}_{t}', 0)
                obj_terms.append(dual_s_G * self.s_G[u,t])
            
            if u in self.players_with_heat_storage and (u,t) in self.s_H:
                dual_s_H = self.subprob_obj.get(f's_H_{u}_{t}', 0)
                obj_terms.append(dual_s_H * self.s_H[u,t])
        
        # (4) Peak power penalty - chi_peak contribution
        # Note: chi_peak is handled at RMP level, but we need to add the dual contribution
        pi_peak = self.params.get('pi_peak', 0)
        dual_chi_peak = self.subprob_obj.get('chi_peak', 0)
        obj_terms.append((pi_peak + dual_chi_peak) * self.chi_peak)
        
        # Add dual values for other variables (electrolyzer binary variables)
        if u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if (u,t) in self.z_on:
                    dual_z_on = self.subprob_obj.get(f'z_on_{u}_{t}', 0)
                    obj_terms.append(dual_z_on * self.z_on[u,t])
                if (u,t) in self.z_off:
                    dual_z_off = self.subprob_obj.get(f'z_off_{u}_{t}', 0)
                    obj_terms.append(dual_z_off * self.z_off[u,t])
                if (u,t) in self.z_sb:
                    dual_z_sb = self.subprob_obj.get(f'z_sb_{u}_{t}', 0)
                    obj_terms.append(dual_z_sb * self.z_sb[u,t])
                
                # Add dual values for flexible demand
                if (u,'hp',t) in self.d:
                    dual_d_hp = self.subprob_obj.get(f'd_hp_{u}_{t}', 0)
                    obj_terms.append(dual_d_hp * self.d[u,'hp',t])
                if (u,'els',t) in self.d:
                    dual_d_els = self.subprob_obj.get(f'd_els_{u}_{t}', 0)
                    obj_terms.append(dual_d_els * self.d[u,'els',t])
        
        # Add convexity constraint dual (from RMP)
        convexity_dual = self.subprob_obj.get(f'cons_convexity_{u}', 0)
        obj_terms.append(convexity_dual)  # This is a constant term
        
        # Set objective
        self.model.setObjective(quicksum(obj_terms), "minimize")
    
    
    def _create_constraints(self):
        """Override to exclude community constraints"""
        # Individual player constraints only
        self._add_electricity_constraints()
        self._add_heat_constraints()
        self._add_hydrogen_constraints()
    
    def _add_heat_constraints(self):
        """Override heat constraints for single player"""
        u = self.single_player  # Use single player directly
        
        # Heat flow balance for single player
        for t in self.time_periods:
            lhs = (self.i_H_gri[u,t] - self.e_H_gri[u,t] + 
                   self.i_H_com[u,t] - self.e_H_com[u,t])
            
            # Add heat pump production
            if u in self.players_with_heatpumps and (u,'hp',t) in self.p:
                lhs += self.p[u,'hp',t]
            
            # Add heat storage discharge/charge
            if u in self.players_with_heat_storage:
                if (u,t) in self.b_dis_H and (u,t) in self.b_ch_H:
                    lhs += self.b_dis_H[u,t] - self.b_ch_H[u,t]
            
            # RHS: heat demand
            rhs = self.params.get(f'd_H_nfl_{u}_{t}', 0)
            
            cons = self.model.addCons(lhs == rhs, name=f"heat_balance_{u}_{t}")
            self.heat_balance_cons[f"heat_balance_{u}_{t}"] = cons
        
        # Heat pump coupling constraint
        if u in self.players_with_heatpumps:
            for t in self.time_periods:
                if (u,'hp',t) in self.d and (u,'hp',t) in self.p:
                    nu_COP = self.params.get(f'nu_COP_{u}', 3.0)
                    cons = self.model.addCons(
                        nu_COP * self.d[u,'hp',t] == self.p[u,'hp',t],
                        name=f"heatpump_coupling_{u}_{t}"
                    )
                    self.heatpump_cons[f"heatpump_coupling_{u}_{t}"] = cons
        
        # Heat storage SOC transition
        if u in self.players_with_heat_storage:
            # Set initial SOC
            if (u,0) in self.s_H:
                initial_soc = self.params.get(f'initial_soc', 50)
                cons = self.model.addCons(self.s_H[u,0] == initial_soc, name=f"initial_soc_H_{u}")
                self.storage_cons[f"initial_soc_H_{u}"] = cons
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_H and (u,t-1) in self.s_H:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    cons = self.model.addCons(
                        self.s_H[u,t] == self.s_H[u,t-1] + nu_ch * self.b_ch_H[u,t] - (1/nu_dis) * self.b_dis_H[u,t],
                        name=f"soc_transition_H_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_H_{u}_{t}"] = cons
    
    def _add_hydrogen_constraints(self):
        """Override hydrogen constraints for single player"""
        u = self.single_player  # Use single player directly
        
        # Hydrogen flow balance for single player
        for t in self.time_periods:
            lhs = (self.i_G_gri[u,t] - self.e_G_gri[u,t] + 
                   self.i_G_com[u,t] - self.e_G_com[u,t])
            
            # Add electrolyzer production
            if u in self.players_with_electrolyzers and (u,'els',t) in self.p:
                lhs += self.p[u,'els',t]
            
            # Add hydrogen storage discharge/charge
            if u in self.players_with_hydro_storage:
                if (u,t) in self.b_dis_G and (u,t) in self.b_ch_G:
                    lhs += self.b_dis_G[u,t] - self.b_ch_G[u,t]
            
            # RHS: hydrogen demand
            rhs = self.params.get(f'd_G_nfl_{u}_{t}', 0)
            
            cons = self.model.addCons(lhs == rhs, name=f"hydrogen_balance_{u}_{t}")
            self.hydrogen_balance_cons[f"hydrogen_balance_{u}_{t}"] = cons
        
        # Electrolyzer coupling constraint
        if u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if (u,'els',t) in self.p and (u,'els',t) in self.d:
                    phi1 = self.params.get(f'phi1_{u}', 0.7)
                    phi0 = self.params.get(f'phi0_{u}', 0.0)
                    
                    cons = self.model.addCons(
                        self.p[u,'els',t] <= phi1 * self.d[u,'els',t] + phi0,
                        name=f"electrolyzer_coupling_{u}_{t}"
                    )
                    self.electrolyzer_cons['coupling', u, t] = cons
        
        # Electrolyzer commitment constraints
        if u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if all(var in self.z_on for var in [(u,t), (u,t), (u,t), (u,t)]):
                    # Constraint 17: exactly one state
                    cons = self.model.addCons(
                        self.z_on[u,t] + self.z_off[u,t] + self.z_sb[u,t] == 1,
                        name=f"electrolyzer_state_{u}_{t}"
                    )
                    self.electrolyzer_cons['state', u, t] = cons
                    
                    # Constraints 18-19: production bounds
                    if (u,'els',t) in self.d:
                        C_max = self.params.get(f'C_max_{u}', 100)
                        C_sb = self.params.get(f'C_sb_{u}', 10)
                        C_min = self.params.get(f'C_min_{u}', 20)
                        
                        cons = self.model.addCons(
                            self.d[u,'els',t] <= C_max * self.z_on[u,t] + C_sb * self.z_sb[u,t],
                            name=f"electrolyzer_max_{u}_{t}"
                        )
                        self.electrolyzer_cons[f"electrolyzer_max_{u}_{t}"] = cons
                        
                        cons = self.model.addCons(
                            self.d[u,'els',t] >= C_min * self.z_on[u,t] + C_sb * self.z_sb[u,t],
                            name=f"electrolyzer_min_{u}_{t}"
                        )
                        self.electrolyzer_cons[f"electrolyzer_min_{u}_{t}"] = cons
                    
                    # Constraint 20: startup logic
                    if t > 0:
                        cons = self.model.addCons(
                            self.z_su[u,t] >= self.z_off[u,t-1] + self.z_on[u,t] + self.z_sb[u,t] - 1,
                            name=f"electrolyzer_startup_{u}_{t}"
                        )
                        self.electrolyzer_cons[f"electrolyzer_startup_{u}_{t}"] = cons
        
        # Hydrogen storage SOC transition
        if u in self.players_with_hydro_storage:
            # Set initial SOC
            if (u,0) in self.s_G:
                initial_soc = self.params.get(f'initial_soc', 50)
                cons = self.model.addCons(self.s_G[u,0] == initial_soc, name=f"initial_soc_G_{u}")
                self.storage_cons[f"initial_soc_G_{u}"] = cons
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_G and (u,t-1) in self.s_G:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    cons = self.model.addCons(
                        self.s_G[u,t] == self.s_G[u,t-1] + nu_ch * self.b_ch_G[u,t] - (1/nu_dis) * self.b_dis_G[u,t],
                        name=f"soc_transition_G_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_G_{u}_{t}"] = cons
        """Modified electricity constraints without community balance"""
        # Only player-specific constraints
        u = self.single_player  # Use single player directly
        
        for t in self.time_periods:
            lhs = (self.i_E_gri[u,t] - self.e_E_gri[u,t] + 
                   self.i_E_com[u,t] - self.e_E_com[u,t])
            
            if u in self.players_with_renewables and (u,'res',t) in self.p:
                lhs += self.p[u,'res',t]
            
            if u in self.players_with_elec_storage:
                if (u,t) in self.b_dis_E and (u,t) in self.b_ch_E:
                    lhs += self.b_dis_E[u,t] - self.b_ch_E[u,t]
            
            rhs = self.params.get(f'd_E_nfl_{u}_{t}', 0)
            
            if u in self.players_with_heatpumps and (u,'hp',t) in self.d:
                rhs += self.d[u,'hp',t]
            if u in self.players_with_electrolyzers and (u,'els',t) in self.d:
                rhs += self.d[u,'els',t]
            
            cons = self.model.addCons(lhs == rhs, name=f"elec_balance_{u}_{t}")
            self.elec_balance_cons[f"elec_balance_{u}_{t}"] = cons
        
        # Storage SOC constraints
        if u in self.players_with_elec_storage:
            if (u,0) in self.s_E:
                initial_soc = self.params.get(f'initial_soc', 50)
                cons = self.model.addCons(self.s_E[u,0] == initial_soc, name=f"initial_soc_E_{u}")
                self.storage_cons[f"initial_soc_E_{u}"] = cons
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_E and (u,t-1) in self.s_E:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    cons = self.model.addCons(
                        self.s_E[u,t] == self.s_E[u,t-1] + nu_ch * self.b_ch_E[u,t] - (1/nu_dis) * self.b_dis_E[u,t],
                        name=f"soc_transition_E_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_E_{u}_{t}"] = cons
        
        # Skip community electricity balance and peak power constraints
    
    def _create_constraints(self):
        """Override to exclude community-level constraints"""
        # Only create individual player constraints
        self._add_electricity_constraints()
        self._add_heat_constraints()
        self._add_hydrogen_constraints()
        self._add_community_constraints()  # This only adds renewable availability
    
    def _add_heat_constraints(self):
        """Override heat constraints for single player"""
        u = self.single_player  # Use single player directly
        
        # Heat flow balance for single player
        for t in self.time_periods:
            lhs = (self.i_H_gri[u,t] - self.e_H_gri[u,t] + 
                   self.i_H_com[u,t] - self.e_H_com[u,t])
            
            # Add heat pump production
            if u in self.players_with_heatpumps and (u,'hp',t) in self.p:
                lhs += self.p[u,'hp',t]
            
            # Add heat storage discharge/charge
            if u in self.players_with_heat_storage:
                if (u,t) in self.b_dis_H and (u,t) in self.b_ch_H:
                    lhs += self.b_dis_H[u,t] - self.b_ch_H[u,t]
            
            # RHS: heat demand
            rhs = self.params.get(f'd_H_nfl_{u}_{t}', 0)
            
            cons = self.model.addCons(lhs == rhs, name=f"heat_balance_{u}_{t}")
            self.heat_balance_cons[f"heat_balance_{u}_{t}"] = cons
        
        # Heat pump coupling constraint
        if u in self.players_with_heatpumps:
            for t in self.time_periods:
                if (u,'hp',t) in self.d and (u,'hp',t) in self.p:
                    nu_COP = self.params.get(f'nu_COP_{u}', 3.0)
                    cons = self.model.addCons(
                        nu_COP * self.d[u,'hp',t] == self.p[u,'hp',t],
                        name=f"heatpump_coupling_{u}_{t}"
                    )
                    self.heatpump_cons[f"heatpump_coupling_{u}_{t}"] = cons
        
        # Heat storage SOC transition
        if u in self.players_with_heat_storage:
            # Set initial SOC
            if (u,0) in self.s_H:
                initial_soc = self.params.get(f'initial_soc', 50)
                cons = self.model.addCons(self.s_H[u,0] == initial_soc, name=f"initial_soc_H_{u}")
                self.storage_cons[f"initial_soc_H_{u}"] = cons
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_H and (u,t-1) in self.s_H:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    cons = self.model.addCons(
                        self.s_H[u,t] == self.s_H[u,t-1] + nu_ch * self.b_ch_H[u,t] - (1/nu_dis) * self.b_dis_H[u,t],
                        name=f"soc_transition_H_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_H_{u}_{t}"] = cons
    
    def _add_hydrogen_constraints(self):
        """Override hydrogen constraints for single player"""
        u = self.single_player  # Use single player directly
        
        # Hydrogen flow balance for single player
        for t in self.time_periods:
            lhs = (self.i_G_gri[u,t] - self.e_G_gri[u,t] + 
                   self.i_G_com[u,t] - self.e_G_com[u,t])
            
            # Add electrolyzer production
            if u in self.players_with_electrolyzers and (u,'els',t) in self.p:
                lhs += self.p[u,'els',t]
            
            # Add hydrogen storage discharge/charge
            if u in self.players_with_hydro_storage:
                if (u,t) in self.b_dis_G and (u,t) in self.b_ch_G:
                    lhs += self.b_dis_G[u,t] - self.b_ch_G[u,t]
            
            # RHS: hydrogen demand
            rhs = self.params.get(f'd_G_nfl_{u}_{t}', 0)
            
            cons = self.model.addCons(lhs == rhs, name=f"hydrogen_balance_{u}_{t}")
            self.hydrogen_balance_cons[f"hydrogen_balance_{u}_{t}"] = cons
        
        # Electrolyzer coupling constraint
        if u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if (u,'els',t) in self.p and (u,'els',t) in self.d:
                    phi1 = self.params.get(f'phi1_{u}', 0.7)
                    phi0 = self.params.get(f'phi0_{u}', 0.0)
                    
                    cons = self.model.addCons(
                        self.p[u,'els',t] <= phi1 * self.d[u,'els',t] + phi0,
                        name=f"electrolyzer_coupling_{u}_{t}"
                    )
                    self.electrolyzer_cons['coupling', u, t] = cons
        
        # Electrolyzer commitment constraints
        if u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if all(var in self.z_on for var in [(u,t), (u,t), (u,t), (u,t)]):
                    # Constraint 17: exactly one state
                    cons = self.model.addCons(
                        self.z_on[u,t] + self.z_off[u,t] + self.z_sb[u,t] == 1,
                        name=f"electrolyzer_state_{u}_{t}"
                    )
                    self.electrolyzer_cons['state', u, t] = cons
                    
                    # Constraints 18-19: production bounds
                    if (u,'els',t) in self.d:
                        C_max = self.params.get(f'C_max_{u}', 100)
                        C_sb = self.params.get(f'C_sb_{u}', 10)
                        C_min = self.params.get(f'C_min_{u}', 20)
                        
                        cons = self.model.addCons(
                            self.d[u,'els',t] <= C_max * self.z_on[u,t] + C_sb * self.z_sb[u,t],
                            name=f"electrolyzer_max_{u}_{t}"
                        )
                        self.electrolyzer_cons[f"electrolyzer_max_{u}_{t}"] = cons
                        
                        cons = self.model.addCons(
                            self.d[u,'els',t] >= C_min * self.z_on[u,t] + C_sb * self.z_sb[u,t],
                            name=f"electrolyzer_min_{u}_{t}"
                        )
                        self.electrolyzer_cons[f"electrolyzer_min_{u}_{t}"] = cons
                    
                    # Constraint 20: startup logic
                    if t > 0:
                        cons = self.model.addCons(
                            self.z_su[u,t] >= self.z_off[u,t-1] + self.z_on[u,t] + self.z_sb[u,t] - 1,
                            name=f"electrolyzer_startup_{u}_{t}"
                        )
                        self.electrolyzer_cons[f"electrolyzer_startup_{u}_{t}"] = cons
        
        # Hydrogen storage SOC transition
        if u in self.players_with_hydro_storage:
            # Set initial SOC
            if (u,0) in self.s_G:
                initial_soc = self.params.get(f'initial_soc', 50)
                cons = self.model.addCons(self.s_G[u,0] == initial_soc, name=f"initial_soc_G_{u}")
                self.storage_cons[f"initial_soc_G_{u}"] = cons
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_G and (u,t-1) in self.s_G:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    cons = self.model.addCons(
                        self.s_G[u,t] == self.s_G[u,t-1] + nu_ch * self.b_ch_G[u,t] - (1/nu_dis) * self.b_dis_G[u,t],
                        name=f"soc_transition_G_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_G_{u}_{t}"] = cons


def solve_energy_pricing_problem(player, time_periods, params, subprob_obj, current_node=None):
    """
    Solve the pricing problem for a single player
    
    Args:
        player: Player ID
        time_periods: List of time periods
        params: Parameters dictionary
        subprob_obj: Dictionary of dual values from RMP
        current_node: Current branch-and-bound node (for warm start if needed)
    
    Returns:
        tuple: (min_redcost, pattern, objval)
            - min_redcost: Reduced cost (objective value with duals)
            - pattern: Solution pattern (variable values)
            - objval: Original objective value (without duals)
    """
    
    # Create pricing problem for single player
    pricing_prob = PlayerPricingProblem(player, time_periods, params, subprob_obj)
    
    # Solve the problem
    pricing_prob.model.hideOutput()
    status = pricing_prob.solve()
    
    if status not in ["optimal", "gaplimit"]:
        print(f"Warning: Pricing problem for player {player} not optimal. Status: {status}")
        return float('inf'), {}, float('inf')
    
    # Get solution
    status, pattern = solve_and_extract_results(pricing_prob.model)
    
    if pattern is None:
        print(f"Warning: Could not extract pattern for player {player}")
        return float('inf'), {}, float('inf')
    
    # Calculate reduced cost (objective with duals)
    min_redcost = pricing_prob.model.getObjVal()
    
    # Calculate original objective value (without duals)
    objval = calculate_original_objective(player, pattern, params)
    
    # Clean up pattern to only include relevant variables
    cleaned_pattern = {}
    for var_name, var_dict in pattern.items():
        if isinstance(var_dict, dict):
            # Filter to only include this player's variables
            filtered_dict = {}
            for key, value in var_dict.items():
                # Check if key contains player ID
                if isinstance(key, tuple) and player in key:
                    # Extract time index
                    if len(key) == 2:  # (player, time)
                        t = key[1]
                        filtered_dict[t] = value
                    elif len(key) == 3:  # (player, type, time)
                        t = key[2]
                        filtered_dict[t] = value
                elif isinstance(key, str) and player in key:
                    filtered_dict[key] = value
            if filtered_dict:
                cleaned_pattern[var_name] = filtered_dict
        else:
            # Single variables (like chi_peak) - skip as it's RMP level
            pass
    
    # Restructure pattern to match expected format
    final_pattern = {}
    for var_name in ['e_E_gri', 'i_E_gri', 'e_E_com', 'i_E_com',
                     'e_H_gri', 'i_H_gri', 'e_H_com', 'i_H_com',
                     'e_G_gri', 'i_G_gri', 'e_G_com', 'i_G_com',
                     'b_dis_E', 'b_ch_E', 's_E',
                     'b_dis_G', 'b_ch_G', 's_G',
                     'b_dis_H', 'b_ch_H', 's_H',
                     'z_su', 'z_on', 'z_off', 'z_sb']:
        if var_name in cleaned_pattern:
            final_pattern[var_name] = cleaned_pattern[var_name]
    
    # Add production and demand variables
    if 'p' in pattern:
        for key, value in pattern['p'].items():
            if isinstance(key, tuple) and player in key:
                if key[1] == 'res':
                    if 'p_res' not in final_pattern:
                        final_pattern['p_res'] = {}
                    final_pattern['p_res'][key[2]] = value
                elif key[1] == 'hp':
                    if 'p_hp' not in final_pattern:
                        final_pattern['p_hp'] = {}
                    final_pattern['p_hp'][key[2]] = value
                elif key[1] == 'els':
                    if 'p_els' not in final_pattern:
                        final_pattern['p_els'] = {}
                    final_pattern['p_els'][key[2]] = value
    
    if 'd' in pattern:
        for key, value in pattern['d'].items():
            if isinstance(key, tuple) and player in key:
                if key[1] == 'hp':
                    if 'd_hp' not in final_pattern:
                        final_pattern['d_hp'] = {}
                    final_pattern['d_hp'][key[2]] = value
                elif key[1] == 'els':
                    if 'd_els' not in final_pattern:
                        final_pattern['d_els'] = {}
                    final_pattern['d_els'][key[2]] = value
    
    return min_redcost, final_pattern, objval


def calculate_original_objective(player, pattern, params):
    """
    Calculate the original objective value without dual contributions
    
    Args:
        player: Player ID
        pattern: Solution pattern
        params: Parameters dictionary
    
    Returns:
        float: Original objective value
    """
    
    cost = 0
    time_periods = list(range(len(pattern.get('e_E_gri', {}))))
    
    # Production costs
    if 'p_res' in pattern:
        c_res = params.get(f'c_res_{player}', 0)
        for t, value in pattern['p_res'].items():
            cost += c_res * value
    
    if 'p_hp' in pattern:
        c_hp = params.get(f'c_hp_{player}', 0)
        for t, value in pattern['p_hp'].items():
            cost += c_hp * value
    
    if 'p_els' in pattern:
        c_els = params.get(f'c_els_{player}', 0)
        for t, value in pattern['p_els'].items():
            cost += c_els * value
    
    # Startup costs
    if 'z_su' in pattern:
        c_su = params.get(f'c_su_{player}', 0)
        for t, value in pattern['z_su'].items():
            cost += c_su * value
    
    # Grid interaction costs
    for t in time_periods:
        pi_E_gri = params.get(f'pi_E_gri_{t}', 0)
        pi_H_gri = params.get(f'pi_H_gri_{t}', 0)
        pi_G_gri = params.get(f'pi_G_gri_{t}', 0)
        
        if 'i_E_gri' in pattern and t in pattern['i_E_gri']:
            cost += pi_E_gri * pattern['i_E_gri'][t]
        if 'e_E_gri' in pattern and t in pattern['e_E_gri']:
            cost -= pi_E_gri * pattern['e_E_gri'][t]
        
        if 'i_H_gri' in pattern and t in pattern['i_H_gri']:
            cost += pi_H_gri * pattern['i_H_gri'][t]
        if 'e_H_gri' in pattern and t in pattern['e_H_gri']:
            cost -= pi_H_gri * pattern['e_H_gri'][t]
        
        if 'i_G_gri' in pattern and t in pattern['i_G_gri']:
            cost += pi_G_gri * pattern['i_G_gri'][t]
        if 'e_G_gri' in pattern and t in pattern['e_G_gri']:
            cost -= pi_G_gri * pattern['e_G_gri'][t]
    
    # Storage costs
    c_sto = params.get(f'c_sto', 0.01)
    nu_ch = params.get('nu_ch', 0.9)
    nu_dis = params.get('nu_dis', 0.9)
    
    for t in time_periods:
        # Electricity storage
        if 'b_ch_E' in pattern and t in pattern['b_ch_E']:
            cost += c_sto * nu_ch * pattern['b_ch_E'][t]
        if 'b_dis_E' in pattern and t in pattern['b_dis_E']:
            cost += c_sto * (1/nu_dis) * pattern['b_dis_E'][t]
        
        # Hydrogen storage
        if 'b_ch_G' in pattern and t in pattern['b_ch_G']:
            cost += c_sto * nu_ch * pattern['b_ch_G'][t]
        if 'b_dis_G' in pattern and t in pattern['b_dis_G']:
            cost += c_sto * (1/nu_dis) * pattern['b_dis_G'][t]
        
        # Heat storage
        if 'b_ch_H' in pattern and t in pattern['b_ch_H']:
            cost += c_sto * nu_ch * pattern['b_ch_H'][t]
        if 'b_dis_H' in pattern and t in pattern['b_dis_H']:
            cost += c_sto * (1/nu_dis) * pattern['b_dis_H'][t]
    
    return cost
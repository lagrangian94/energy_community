from pyscipopt import Model, quicksum
import numpy as np
from typing import Dict, List, Tuple, Optional

class LocalEnergyMarket:
    def __init__(self, 
                 players: List[str],
                 time_periods: List[int],
                 parameters: Dict):
        """
        Initialize the Local Energy Market optimization model
        
        Args:
            players: List of player IDs
            time_periods: List of time period indices
            parameters: Dictionary containing all model parameters
        """
        self.players = players
        self.time_periods = time_periods
        self.params = parameters
        self.model = Model("LocalEnergyMarket")
        
        # Sets definition based on slides
        self.players_with_renewables = self.params.get('players_with_renewables', [])
        self.players_with_electrolyzers = self.params.get('players_with_electrolyzers', [])  
        self.players_with_heatpumps = self.params.get('players_with_heatpumps', [])
        self.players_with_elec_storage = self.params.get('players_with_elec_storage', [])
        self.players_with_hydro_storage = self.params.get('players_with_hydro_storage', [])
        self.players_with_heat_storage = self.params.get('players_with_heat_storage', [])
        
        # Combined sets for energy types
        self.U_E = list(set(self.players_with_renewables + self.players_with_elec_storage))  # Players with electricity assets
        self.U_G = list(set(self.players_with_electrolyzers + self.players_with_hydro_storage))  # Players with hydrogen assets
        self.U_H = list(set(self.players_with_heatpumps + self.players_with_heat_storage))  # Players with heat assets
        
        # Store community balance constraints for dual access
        self.community_elec_balance_cons = {}
        self.community_heat_balance_cons = {}
        self.community_hydrogen_balance_cons = {}
        
        # Initialize variables
        self._create_variables()
        self._create_objective()
        self._create_constraints()
    
    def _create_variables(self):
        """Create decision variables based on slides 6"""
        
        # Electricity variables
        self.e_E_gri = {}  # Electricity exported to grid
        self.i_E_gri = {}  # Electricity imported from grid
        self.e_E_com = {}  # Electricity exported to community
        self.i_E_com = {}  # Electricity imported from community
        
        # Heat variables
        self.e_H_gri = {}  # Heat exported to grid
        self.i_H_gri = {}  # Heat imported from grid
        self.e_H_com = {}  # Heat exported to community
        self.i_H_com = {}  # Heat imported from community
        
        # Hydrogen variables  
        self.e_G_gri = {}  # Hydrogen exported to grid
        self.i_G_gri = {}  # Hydrogen imported from grid
        self.e_G_com = {}  # Hydrogen exported to community
        self.i_G_com = {}  # Hydrogen imported from community
        
        # Production variables
        self.p = {}  # Energy production by generators
        self.d = {}  # Flexible energy consumption
        
        # Storage variables by type
        self.b_dis_E = {}  # Electricity discharged from storage
        self.b_ch_E = {}   # Electricity charged to storage
        self.s_E = {}      # Electricity storage SOC level
        
        self.b_dis_G = {}  # Hydrogen discharged from storage
        self.b_ch_G = {}   # Hydrogen charged to storage
        self.s_G = {}      # Hydrogen storage SOC level
        
        self.b_dis_H = {}  # Heat discharged from storage
        self.b_ch_H = {}   # Heat charged to storage
        self.s_H = {}      # Heat storage SOC level
        
        # Electrolyzer commitment variables
        self.z_su = {}   # Start-up decision
        self.z_on = {}   # Turn on decision
        self.z_off = {}  # Turn off decision
        self.z_sb = {}   # Stand-by decision
        
        # Peak power variable
        self.chi_peak = self.model.addVar(vtype="C", name="chi_peak", lb=0)
        
        # Create variables for each player and time period
        for u in self.players:
            for t in self.time_periods:
                # Electricity variables with reasonable upper bounds
                self.e_E_gri[u,t] = self.model.addVar(vtype="C", name=f"e_E_gri_{u}_{t}", lb=0, 
                                                     ub=self.params.get(f'e_E_cap_{u}_{t}', 1000))
                self.i_E_gri[u,t] = self.model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_E_cap_{u}_{t}', 1000))
                self.e_E_com[u,t] = self.model.addVar(vtype="C", name=f"e_E_com_{u}_{t}", lb=0, ub=1000)
                self.i_E_com[u,t] = self.model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0, ub=1000)
                
                # Heat variables with upper bounds
                self.e_H_gri[u,t] = self.model.addVar(vtype="C", name=f"e_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'e_H_cap_{u}_{t}', 500))
                self.i_H_gri[u,t] = self.model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_H_cap_{u}_{t}', 500))
                self.e_H_com[u,t] = self.model.addVar(vtype="C", name=f"e_H_com_{u}_{t}", lb=0, ub=500)
                self.i_H_com[u,t] = self.model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0, ub=500)
                
                # Hydrogen variables with upper bounds
                self.e_G_gri[u,t] = self.model.addVar(vtype="C", name=f"e_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'e_G_cap_{u}_{t}', 100))
                self.i_G_gri[u,t] = self.model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_G_cap_{u}_{t}', 100))
                self.e_G_com[u,t] = self.model.addVar(vtype="C", name=f"e_G_com_{u}_{t}", lb=0, ub=100)
                self.i_G_com[u,t] = self.model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0, ub=100)
                
                # Production variables (for renewables, heat pumps, electrolyzers) with capacity limits
                if u in self.players_with_renewables:  # Renewable generators
                    renewable_cap = self.params.get(f'renewable_cap_{u}', 200)  # Default 200 kW
                    self.p[u,'res',t] = self.model.addVar(vtype="C", name=f"p_res_{u}_{t}", 
                                                        lb=0, ub=renewable_cap)
                if u in self.players_with_heatpumps:  # Heat pumps
                    hp_cap = self.params.get(f'hp_cap_{u}', 100)  # Default 100 kW thermal
                    self.p[u,'hp',t] = self.model.addVar(vtype="C", name=f"p_hp_{u}_{t}", 
                                                       lb=0, ub=hp_cap)
                    self.d[u,'hp',t] = self.model.addVar(vtype="C", name=f"d_hp_{u}_{t}", 
                                                       lb=0, ub=hp_cap/3)  # Assuming COP=3
                if u in self.players_with_electrolyzers:  # Electrolyzers
                    els_cap = self.params.get(f'els_cap_{u}', 150)  # Default 150 kg/day
                    self.p[u,'els',t] = self.model.addVar(vtype="C", name=f"p_els_{u}_{t}", 
                                                        lb=0, ub=els_cap)
                    self.d[u,'els',t] = self.model.addVar(vtype="C", name=f"d_els_{u}_{t}", 
                                                        lb=0, ub=200)  # 200 kW electrical
                    
                    # Electrolyzer commitment variables
                    self.z_su[u,t] = self.model.addVar(vtype="B", name=f"z_su_{u}_{t}")
                    self.z_on[u,t] = self.model.addVar(vtype="B", name=f"z_on_{u}_{t}")
                    self.z_off[u,t] = self.model.addVar(vtype="B", name=f"z_off_{u}_{t}")
                    self.z_sb[u,t] = self.model.addVar(vtype="B", name=f"z_sb_{u}_{t}")
                
                # Storage variables by type with capacity constraints
                storage_power = self.params.get(f'storage_power', 50)  # kW power rating
                storage_capacity = self.params.get(f'storage_capacity', 200)  # kWh capacity
                
                # Electricity storage
                if u in self.players_with_elec_storage:
                    self.b_dis_E[u,t] = self.model.addVar(vtype="C", name=f"b_dis_E_{u}_{t}", 
                                                        lb=0, ub=storage_power)
                    self.b_ch_E[u,t] = self.model.addVar(vtype="C", name=f"b_ch_E_{u}_{t}", 
                                                       lb=0, ub=storage_power)
                    self.s_E[u,t] = self.model.addVar(vtype="C", name=f"s_E_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
                
                # Hydrogen storage
                if u in self.players_with_hydro_storage:
                    self.b_dis_G[u,t] = self.model.addVar(vtype="C", name=f"b_dis_G_{u}_{t}", 
                                                        lb=0, ub=storage_power)
                    self.b_ch_G[u,t] = self.model.addVar(vtype="C", name=f"b_ch_G_{u}_{t}", 
                                                       lb=0, ub=storage_power)
                    self.s_G[u,t] = self.model.addVar(vtype="C", name=f"s_G_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
                
                # Heat storage
                if u in self.players_with_heat_storage:
                    self.b_dis_H[u,t] = self.model.addVar(vtype="C", name=f"b_dis_H_{u}_{t}", 
                                                        lb=0, ub=storage_power)
                    self.b_ch_H[u,t] = self.model.addVar(vtype="C", name=f"b_ch_H_{u}_{t}", 
                                                       lb=0, ub=storage_power)
                    self.s_H[u,t] = self.model.addVar(vtype="C", name=f"s_H_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
    
    def _create_objective(self):
        """Create objective function based on slide 8"""
        
        obj_terms = []
        
        # (1) Production costs (generator variable costs + startup costs)
        for u in self.players:
            for t in self.time_periods:
                # Renewable generation costs
                if u in self.U_E and (u,'res',t) in self.p:
                    c_res = self.params.get(f'c_res_{u}', 0)
                    obj_terms.append(c_res * self.p[u,'res',t])
                
                # Heat pump costs
                if u in self.U_H and (u,'hp',t) in self.p:
                    c_hp = self.params.get(f'c_hp_{u}', 0)
                    obj_terms.append(c_hp * self.p[u,'hp',t])
                
                # Electrolyzer costs
                if u in self.U_G:
                    if (u,'els',t) in self.p:
                        c_els = self.params.get(f'c_els_{u}', 0)
                        obj_terms.append(c_els * self.p[u,'els',t])
                    if (u,t) in self.z_su:
                        c_su = self.params.get(f'c_su_{u}', 0)
                        obj_terms.append(c_su * self.z_su[u,t])
        
        # (2) Grid interaction costs
        for u in self.players:
            for t in self.time_periods:
                pi_E_gri = self.params.get(f'pi_E_gri_{t}', 0)
                pi_H_gri = self.params.get(f'pi_H_gri_{t}', 0)
                pi_G_gri = self.params.get(f'pi_G_gri_{t}', 0)
                
                obj_terms.append(pi_E_gri * (self.i_E_gri[u,t] - self.e_E_gri[u,t]))
                obj_terms.append(pi_H_gri * (self.i_H_gri[u,t] - self.e_H_gri[u,t]))
                obj_terms.append(pi_G_gri * (self.i_G_gri[u,t] - self.e_G_gri[u,t]))
        
        # (3) Storage usage costs by type
        for u in self.players:
            c_sto = self.params.get(f'c_sto', 0.01)  # Common storage cost
            nu_ch = self.params.get('nu_ch', 0.9)
            nu_dis = self.params.get('nu_dis', 0.9)
            
            for t in self.time_periods:
                # Electricity storage costs
                if u in self.players_with_elec_storage and (u,t) in self.b_ch_E and (u,t) in self.b_dis_E:
                    obj_terms.append(c_sto * (nu_ch * self.b_ch_E[u,t] + (1/nu_dis) * self.b_dis_E[u,t]))
                
                # Hydrogen storage costs
                if u in self.players_with_hydro_storage and (u,t) in self.b_ch_G and (u,t) in self.b_dis_G:
                    obj_terms.append(c_sto * (nu_ch * self.b_ch_G[u,t] + (1/nu_dis) * self.b_dis_G[u,t]))
                
                # Heat storage costs
                if u in self.players_with_heat_storage and (u,t) in self.b_ch_H and (u,t) in self.b_dis_H:
                    obj_terms.append(c_sto * (nu_ch * self.b_ch_H[u,t] + (1/nu_dis) * self.b_dis_H[u,t]))
        
        # (4) Peak power penalty
        pi_peak = self.params.get('pi_peak', 0)
        obj_terms.append(pi_peak * self.chi_peak)
        
        # Set objective
        self.model.setObjective(quicksum(obj_terms), "minimize")
    
    def _create_constraints(self):
        """Create constraints based on slides 9-15"""
        
        # Electricity flow balance constraints (slide 9)
        self._add_electricity_constraints()
        
        # Heat flow balance constraints (slide 12)
        self._add_heat_constraints()
        
        # Hydrogen flow balance constraints (slide 13-14)
        self._add_hydrogen_constraints()
        
        # Community-level constraints
        self._add_community_constraints()
    
    def _add_electricity_constraints(self):
        """Add electricity-related constraints from slides 9-10"""
        
        # Constraint (5): Electricity flow balance for each player
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.i_E_gri[u,t] - self.e_E_gri[u,t] + 
                       self.i_E_com[u,t] - self.e_E_com[u,t])
                
                # Add renewable generation
                if u in self.players_with_renewables and (u,'res',t) in self.p:
                    lhs += self.p[u,'res',t]
                
                # Add electricity storage discharge/charge
                if u in self.players_with_elec_storage:
                    if (u,t) in self.b_dis_E and (u,t) in self.b_ch_E:
                        lhs += self.b_dis_E[u,t] - self.b_ch_E[u,t]
                
                # RHS: demand
                rhs = self.params.get(f'd_E_nfl_{u}_{t}', 0)  # Non-flexible demand
                
                # Add flexible demand (heat pump, electrolyzer)
                if u in self.players_with_heatpumps and (u,'hp',t) in self.d:
                    rhs += self.d[u,'hp',t]
                if u in self.players_with_electrolyzers and (u,'els',t) in self.d:
                    rhs += self.d[u,'els',t]
                
                self.model.addCons(lhs == rhs, name=f"elec_balance_{u}_{t}")
        
        # Constraint (6): Electricity storage SOC transition with initial condition
        for u in self.players_with_elec_storage:
            # Set initial SOC
            if (u,0) in self.s_E:
                initial_soc = self.params.get(f'initial_soc', 50)  # Default 50% SOC
                self.model.addCons(self.s_E[u,0] == initial_soc, name=f"initial_soc_E_{u}")
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_E and (u,t-1) in self.s_E:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    self.model.addCons(
                        self.s_E[u,t] == self.s_E[u,t-1] + nu_ch * self.b_ch_E[u,t] - (1/nu_dis) * self.b_dis_E[u,t],
                        name=f"soc_transition_E_{u}_{t}"
                    )
        
        # Constraint (9): Community electricity balance
        for t in self.time_periods:
            community_balance = quicksum(self.i_E_com[u,t] - self.e_E_com[u,t] for u in self.players)
            cons = self.model.addCons(community_balance == 0, name=f"community_elec_balance_{t}")
            self.community_elec_balance_cons[t] = cons
        
        # Constraint (10): Peak power constraint
        for t in self.time_periods:
            grid_import = quicksum(self.i_E_gri[u,t] - self.e_E_gri[u,t] for u in self.players)
            self.model.addCons(grid_import <= self.chi_peak, name=f"peak_power_{t}")
    
    def _add_heat_constraints(self):
        """Add heat-related constraints from slide 12"""
        
        # Heat flow balance for each player
        for u in self.players:
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
                rhs = self.params.get(f'd_H_nfl_{u}_{t}', 0)  # Non-flexible heat demand
                
                self.model.addCons(lhs == rhs, name=f"heat_balance_{u}_{t}")
        
        # Heat pump coupling constraint (constraint 12)
        for u in self.players_with_heatpumps:
            for t in self.time_periods:
                if (u,'hp',t) in self.d and (u,'hp',t) in self.p:
                    nu_COP = self.params.get(f'nu_COP_{u}', 3.0)
                    self.model.addCons(
                        nu_COP * self.d[u,'hp',t] == self.p[u,'hp',t],
                        name=f"heatpump_coupling_{u}_{t}"
                    )
        
        # Heat storage SOC transition
        for u in self.players_with_heat_storage:
            # Set initial SOC
            if (u,0) in self.s_H:
                initial_soc = self.params.get(f'initial_soc', 50)
                self.model.addCons(self.s_H[u,0] == initial_soc, name=f"initial_soc_H_{u}")
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_H and (u,t-1) in self.s_H:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    self.model.addCons(
                        self.s_H[u,t] == self.s_H[u,t-1] + nu_ch * self.b_ch_H[u,t] - (1/nu_dis) * self.b_dis_H[u,t],
                        name=f"soc_transition_H_{u}_{t}"
                    )
        
        # Community heat balance
        for t in self.time_periods:
            community_heat_balance = quicksum(self.i_H_com[u,t] - self.e_H_com[u,t] for u in self.players)
            cons = self.model.addCons(community_heat_balance == 0, name=f"community_heat_balance_{t}")
            self.community_heat_balance_cons[t] = cons
    
    def _add_hydrogen_constraints(self):
        """Add hydrogen-related constraints from slides 13-15"""
        
        # Hydrogen flow balance for each player
        for u in self.players:
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
                rhs = self.params.get(f'd_G_nfl_{u}_{t}', 0)  # Non-flexible hydrogen demand
                
                self.model.addCons(lhs == rhs, name=f"hydrogen_balance_{u}_{t}")
        
        # Electrolyzer coupling constraint (constraint 15)
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if (u,'els',t) in self.p and (u,'els',t) in self.d:
                    phi1 = self.params.get(f'phi1_{u}', 0.7)
                    phi0 = self.params.get(f'phi0_{u}', 0.0)
                    
                    self.model.addCons(
                        self.p[u,'els',t] <= phi1 * self.d[u,'els',t] + phi0,
                        name=f"electrolyzer_coupling_{u}_{t}"
                    )
        
        # Electrolyzer commitment constraints (constraints 17-21)
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if all(var in self.z_on for var in [(u,t), (u,t), (u,t), (u,t)]):
                    # Constraint 17: exactly one state
                    self.model.addCons(
                        self.z_on[u,t] + self.z_off[u,t] + self.z_sb[u,t] == 1,
                        name=f"electrolyzer_state_{u}_{t}"
                    )
                    
                    # Constraints 18-19: production bounds
                    if (u,'els',t) in self.d:
                        C_max = self.params.get(f'C_max_{u}', 100)
                        C_sb = self.params.get(f'C_sb_{u}', 10)
                        C_min = self.params.get(f'C_min_{u}', 20)
                        
                        self.model.addCons(
                            self.d[u,'els',t] <= C_max * self.z_on[u,t] + C_sb * self.z_sb[u,t],
                            name=f"electrolyzer_max_{u}_{t}"
                        )
                        self.model.addCons(
                            self.d[u,'els',t] >= C_min * self.z_on[u,t] + C_sb * self.z_sb[u,t],
                            name=f"electrolyzer_min_{u}_{t}"
                        )
                    
                    # Constraint 20: startup logic
                    if t > 0:
                        self.model.addCons(
                            self.z_su[u,t] >= self.z_off[u,t-1] + self.z_on[u,t] + self.z_sb[u,t] - 1,
                            name=f"electrolyzer_startup_{u}_{t}"
                        )
        
        # Hydrogen storage SOC transition
        for u in self.players_with_hydro_storage:
            # Set initial SOC
            if (u,0) in self.s_G:
                initial_soc = self.params.get(f'initial_soc', 50)
                self.model.addCons(self.s_G[u,0] == initial_soc, name=f"initial_soc_G_{u}")
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_G and (u,t-1) in self.s_G:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    self.model.addCons(
                        self.s_G[u,t] == self.s_G[u,t-1] + nu_ch * self.b_ch_G[u,t] - (1/nu_dis) * self.b_dis_G[u,t],
                        name=f"soc_transition_G_{u}_{t}"
                    )
        
        # Community hydrogen balance
        for t in self.time_periods:
            community_hydrogen_balance = quicksum(self.e_G_com[u,t] - self.i_G_com[u,t] for u in self.players)
            cons = self.model.addCons(community_hydrogen_balance == 0, name=f"community_hydrogen_balance_{t}")
            self.community_hydrogen_balance_cons[t] = cons
    
    def _add_community_constraints(self):
        """Add additional community-level constraints"""
        
        # Grid capacity constraints - using parameter values instead of infinity
        for u in self.players:
            for t in self.time_periods:
                # These constraints are now handled in variable bounds, but kept for explicit clarity
                pass
        
        # Add renewable generation constraints based on weather/availability
        for u in self.players_with_renewables:
            for t in self.time_periods:
                if (u,'res',t) in self.p:
                    # Renewable availability factor (0-1, e.g., solar irradiance, wind speed)
                    availability = self.params.get(f'renewable_availability_{u}_{t}', 1.0)
                    renewable_cap = self.params.get(f'renewable_cap_{u}', 200)
                    
                    self.model.addCons(
                        self.p[u,'res',t] <= availability * renewable_cap,
                        name=f"renewable_availability_{u}_{t}"
                    )
        
        # Add non-negativity and simultaneity constraints for all storage types
        for u in self.players:
            for t in self.time_periods:
                # Electricity storage constraints
                if u in self.players_with_elec_storage:
                    if (u,t) in self.b_ch_E and (u,t) in self.b_dis_E:
                        # Storage cannot charge and discharge simultaneously (optional constraint)
                        # This can be implemented with binary variables if needed
                        pass
                
                # Hydrogen storage constraints  
                if u in self.players_with_hydro_storage:
                    if (u,t) in self.b_ch_G and (u,t) in self.b_dis_G:
                        pass
                
                # Heat storage constraints
                if u in self.players_with_heat_storage:
                    if (u,t) in self.b_ch_H and (u,t) in self.b_dis_H:
                        pass
    
    def solve(self):
        """Solve the optimization model"""
        self.model.optimize()
        return self.model.getStatus()
    
    def solve_with_restricted_pricing(self):
        """
        Solve with Restricted Pricing mechanism:
        1. First solve MILP to get optimal binary variables
        2. Fix binary variables and solve LP to get shadow prices
        
        Returns:
            tuple: (status, results, prices)
                - status: optimization status
                - results: optimization results dictionary
                - prices: dictionary with electricity, heat, hydrogen prices per time period
        """
        
        print("Step 1: Solving MILP to get optimal commitment decisions...")
        
        # Step 1: Solve original MILP
        status = self.solve()
        
        if status != "optimal":
            print(f"MILP optimization failed with status: {status}")
            return status, None, None
        
        print("MILP solved successfully. Extracting binary variable values...")
        
        # Extract optimal binary variable values
        binary_values = {}
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if (u,t) in self.z_su:
                    binary_values['z_su', u, t] = self.model.getVal(self.z_su[u,t])
                if (u,t) in self.z_on:
                    binary_values['z_on', u, t] = self.model.getVal(self.z_on[u,t])
                if (u,t) in self.z_off:
                    binary_values['z_off', u, t] = self.model.getVal(self.z_off[u,t])
                if (u,t) in self.z_sb:
                    binary_values['z_sb', u, t] = self.model.getVal(self.z_sb[u,t])
        
        print("Step 2: Creating LP relaxation with fixed binary variables...")
        
        # Step 2: Create new LP model with fixed binary variables
        lp_model = Model("LocalEnergyMarket_LP")
        
        # Recreate all continuous variables
        self._recreate_continuous_variables_for_lp(lp_model)
        
        # Recreate objective function
        self._recreate_objective_for_lp(lp_model)
        
        # Recreate all constraints with fixed binary variables
        lp_community_elec_cons, lp_community_heat_cons, lp_community_hydrogen_cons = \
            self._recreate_constraints_for_lp(lp_model, binary_values)
        
        print("Step 3: Solving LP relaxation...")
        
        # Solve LP
        from pyscipopt import SCIP_PARAMSETTING
        lp_model.setPresolve(SCIP_PARAMSETTING.OFF)
        lp_model.setHeuristics(SCIP_PARAMSETTING.OFF)
        lp_model.disablePropagation()
        lp_model.optimize()
        
        if lp_model.getStatus() != "optimal":
            print(f"LP optimization failed with status: {lp_model.getStatus()}")
            return lp_model.getStatus(), None, None
        
        print("LP solved successfully. Extracting shadow prices...")
        
        # Step 3: Extract shadow prices (dual multipliers) from community balance constraints
        prices = {
            'electricity': {},
            'heat': {},
            'hydrogen': {}
        }
        
        for t in self.time_periods:
            # Get dual multipliers for community balance constraints
            # Note: Must use getTransformedCons() to get transformed constraints for dual solution
            if t in lp_community_elec_cons:
                t_cons = lp_model.getTransformedCons(lp_community_elec_cons[t])
                prices['electricity'][t] = lp_model.getDualsolLinear(t_cons)
            if t in lp_community_heat_cons:
                t_cons = lp_model.getTransformedCons(lp_community_heat_cons[t])
                prices['heat'][t] = lp_model.getDualsolLinear(t_cons)
            if t in lp_community_hydrogen_cons:
                t_cons = lp_model.getTransformedCons(lp_community_hydrogen_cons[t])
                prices['hydrogen'][t] = lp_model.getDualsolLinear(t_cons)
        
        # Get LP results
        lp_results = self._extract_lp_results(lp_model)
        
        print("Restricted Pricing completed successfully!")
        print(f"Electricity prices: {prices['electricity']}")
        print(f"Heat prices: {prices['heat']}")
        print(f"Hydrogen prices: {prices['hydrogen']}")
        
        return "optimal", lp_results, prices
    
    def _recreate_continuous_variables_for_lp(self, lp_model):
        """Recreate all continuous variables for LP relaxation"""
        
        # Store variable references for LP model
        self.lp_e_E_gri = {}
        self.lp_i_E_gri = {}
        self.lp_e_E_com = {}
        self.lp_i_E_com = {}
        
        self.lp_e_H_gri = {}
        self.lp_i_H_gri = {}
        self.lp_e_H_com = {}
        self.lp_i_H_com = {}
        
        self.lp_e_G_gri = {}
        self.lp_i_G_gri = {}
        self.lp_e_G_com = {}
        self.lp_i_G_com = {}
        
        self.lp_p = {}
        self.lp_d = {}
        
        self.lp_b_dis_E = {}
        self.lp_b_ch_E = {}
        self.lp_s_E = {}
        
        self.lp_b_dis_G = {}
        self.lp_b_ch_G = {}
        self.lp_s_G = {}
        
        self.lp_b_dis_H = {}
        self.lp_b_ch_H = {}
        self.lp_s_H = {}
        
        self.lp_chi_peak = lp_model.addVar(vtype="C", name="chi_peak", lb=0)
        
        # Recreate all continuous variables with same bounds
        for u in self.players:
            for t in self.time_periods:
                # Electricity variables
                self.lp_e_E_gri[u,t] = lp_model.addVar(vtype="C", name=f"e_E_gri_{u}_{t}", lb=0, 
                                                     ub=self.params.get(f'e_E_cap_{u}_{t}', 1000))
                self.lp_i_E_gri[u,t] = lp_model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_E_cap_{u}_{t}', 1000))
                self.lp_e_E_com[u,t] = lp_model.addVar(vtype="C", name=f"e_E_com_{u}_{t}", lb=0, ub=1000)
                self.lp_i_E_com[u,t] = lp_model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0, ub=1000)
                
                # Heat variables
                self.lp_e_H_gri[u,t] = lp_model.addVar(vtype="C", name=f"e_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'e_H_cap_{u}_{t}', 500))
                self.lp_i_H_gri[u,t] = lp_model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_H_cap_{u}_{t}', 500))
                self.lp_e_H_com[u,t] = lp_model.addVar(vtype="C", name=f"e_H_com_{u}_{t}", lb=0, ub=500)
                self.lp_i_H_com[u,t] = lp_model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0, ub=500)
                
                # Hydrogen variables
                self.lp_e_G_gri[u,t] = lp_model.addVar(vtype="C", name=f"e_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'e_G_cap_{u}_{t}', 100))
                self.lp_i_G_gri[u,t] = lp_model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_G_cap_{u}_{t}', 100))
                self.lp_e_G_com[u,t] = lp_model.addVar(vtype="C", name=f"e_G_com_{u}_{t}", lb=0, ub=100)
                self.lp_i_G_com[u,t] = lp_model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0, ub=100)
                
                # Production variables
                if u in self.players_with_renewables:
                    renewable_cap = self.params.get(f'renewable_cap_{u}', 200)
                    self.lp_p[u,'res',t] = lp_model.addVar(vtype="C", name=f"p_res_{u}_{t}", 
                                                        lb=0, ub=renewable_cap)
                if u in self.players_with_heatpumps:
                    hp_cap = self.params.get(f'hp_cap_{u}', 100)
                    self.lp_p[u,'hp',t] = lp_model.addVar(vtype="C", name=f"p_hp_{u}_{t}", 
                                                       lb=0, ub=hp_cap)
                    self.lp_d[u,'hp',t] = lp_model.addVar(vtype="C", name=f"d_hp_{u}_{t}", 
                                                       lb=0, ub=hp_cap/3)
                if u in self.players_with_electrolyzers:
                    els_cap = self.params.get(f'els_cap_{u}', 150)
                    self.lp_p[u,'els',t] = lp_model.addVar(vtype="C", name=f"p_els_{u}_{t}", 
                                                        lb=0, ub=els_cap)
                    self.lp_d[u,'els',t] = lp_model.addVar(vtype="C", name=f"d_els_{u}_{t}", 
                                                        lb=0, ub=200)
                
                # Storage variables
                storage_power = self.params.get(f'storage_power', 50)
                storage_capacity = self.params.get(f'storage_capacity', 200)
                
                if u in self.players_with_elec_storage:
                    self.lp_b_dis_E[u,t] = lp_model.addVar(vtype="C", name=f"b_dis_E_{u}_{t}", 
                                                        lb=0, ub=storage_power)
                    self.lp_b_ch_E[u,t] = lp_model.addVar(vtype="C", name=f"b_ch_E_{u}_{t}", 
                                                       lb=0, ub=storage_power)
                    self.lp_s_E[u,t] = lp_model.addVar(vtype="C", name=f"s_E_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
                
                if u in self.players_with_hydro_storage:
                    self.lp_b_dis_G[u,t] = lp_model.addVar(vtype="C", name=f"b_dis_G_{u}_{t}", 
                                                        lb=0, ub=storage_power)
                    self.lp_b_ch_G[u,t] = lp_model.addVar(vtype="C", name=f"b_ch_G_{u}_{t}", 
                                                       lb=0, ub=storage_power)
                    self.lp_s_G[u,t] = lp_model.addVar(vtype="C", name=f"s_G_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
                
                if u in self.players_with_heat_storage:
                    self.lp_b_dis_H[u,t] = lp_model.addVar(vtype="C", name=f"b_dis_H_{u}_{t}", 
                                                        lb=0, ub=storage_power)
                    self.lp_b_ch_H[u,t] = lp_model.addVar(vtype="C", name=f"b_ch_H_{u}_{t}", 
                                                       lb=0, ub=storage_power)
                    self.lp_s_H[u,t] = lp_model.addVar(vtype="C", name=f"s_H_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
    
    def _recreate_objective_for_lp(self, lp_model):
        """Recreate objective function for LP model (without binary startup costs)"""
        
        obj_terms = []
        
        # Production costs (excluding startup costs since binaries are fixed)
        for u in self.players:
            for t in self.time_periods:
                if u in self.U_E and (u,'res',t) in self.lp_p:
                    c_res = self.params.get(f'c_res_{u}', 0)
                    obj_terms.append(c_res * self.lp_p[u,'res',t])
                
                if u in self.U_H and (u,'hp',t) in self.lp_p:
                    c_hp = self.params.get(f'c_hp_{u}', 0)
                    obj_terms.append(c_hp * self.lp_p[u,'hp',t])
                
                if u in self.U_G and (u,'els',t) in self.lp_p:
                    c_els = self.params.get(f'c_els_{u}', 0)
                    obj_terms.append(c_els * self.lp_p[u,'els',t])
        
        # Grid interaction costs
        for u in self.players:
            for t in self.time_periods:
                pi_E_gri = self.params.get(f'pi_E_gri_{t}', 0)
                pi_H_gri = self.params.get(f'pi_H_gri_{t}', 0)
                pi_G_gri = self.params.get(f'pi_G_gri_{t}', 0)
                
                obj_terms.append(pi_E_gri * (self.lp_i_E_gri[u,t] - self.lp_e_E_gri[u,t]))
                obj_terms.append(pi_H_gri * (self.lp_i_H_gri[u,t] - self.lp_e_H_gri[u,t]))
                obj_terms.append(pi_G_gri * (self.lp_i_G_gri[u,t] - self.lp_e_G_gri[u,t]))
        
        # Storage usage costs
        for u in self.players:
            c_sto = self.params.get(f'c_sto', 0.01)
            nu_ch = self.params.get('nu_ch', 0.9)
            nu_dis = self.params.get('nu_dis', 0.9)
            
            for t in self.time_periods:
                if u in self.players_with_elec_storage and (u,t) in self.lp_b_ch_E and (u,t) in self.lp_b_dis_E:
                    obj_terms.append(c_sto * (nu_ch * self.lp_b_ch_E[u,t] + (1/nu_dis) * self.lp_b_dis_E[u,t]))
                
                if u in self.players_with_hydro_storage and (u,t) in self.lp_b_ch_G and (u,t) in self.lp_b_dis_G:
                    obj_terms.append(c_sto * (nu_ch * self.lp_b_ch_G[u,t] + (1/nu_dis) * self.lp_b_dis_G[u,t]))
                
                if u in self.players_with_heat_storage and (u,t) in self.lp_b_ch_H and (u,t) in self.lp_b_dis_H:
                    obj_terms.append(c_sto * (nu_ch * self.lp_b_ch_H[u,t] + (1/nu_dis) * self.lp_b_dis_H[u,t]))
        
        # Peak power penalty
        pi_peak = self.params.get('pi_peak', 0)
        obj_terms.append(pi_peak * self.lp_chi_peak)
        
        lp_model.setObjective(quicksum(obj_terms), "minimize")
    
    def _recreate_constraints_for_lp(self, lp_model, binary_values):
        """Recreate all constraints for LP model with fixed binary variables"""
        
        lp_community_elec_cons = {}
        lp_community_heat_cons = {}
        lp_community_hydrogen_cons = {}
        
        # Electricity constraints
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.lp_i_E_gri[u,t] - self.lp_e_E_gri[u,t] + 
                       self.lp_i_E_com[u,t] - self.lp_e_E_com[u,t])
                
                if u in self.players_with_renewables and (u,'res',t) in self.lp_p:
                    lhs += self.lp_p[u,'res',t]
                
                if u in self.players_with_elec_storage:
                    if (u,t) in self.lp_b_dis_E and (u,t) in self.lp_b_ch_E:
                        lhs += self.lp_b_dis_E[u,t] - self.lp_b_ch_E[u,t]
                
                rhs = self.params.get(f'd_E_nfl_{u}_{t}', 0)
                
                if u in self.players_with_heatpumps and (u,'hp',t) in self.lp_d:
                    rhs += self.lp_d[u,'hp',t]
                if u in self.players_with_electrolyzers and (u,'els',t) in self.lp_d:
                    rhs += self.lp_d[u,'els',t]
                
                lp_model.addCons(lhs == rhs, name=f"elec_balance_{u}_{t}")
        
        # Electricity storage SOC constraints
        for u in self.players_with_elec_storage:
            if (u,0) in self.lp_s_E:
                initial_soc = self.params.get(f'initial_soc', 50)
                lp_model.addCons(self.lp_s_E[u,0] == initial_soc, name=f"initial_soc_E_{u}")
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.lp_s_E and (u,t-1) in self.lp_s_E:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    lp_model.addCons(
                        self.lp_s_E[u,t] == self.lp_s_E[u,t-1] + nu_ch * self.lp_b_ch_E[u,t] - (1/nu_dis) * self.lp_b_dis_E[u,t],
                        name=f"soc_transition_E_{u}_{t}"
                    )
        
        # Community electricity balance (IMPORTANT: store constraint reference for dual)
        for t in self.time_periods:
            community_balance = quicksum(self.lp_i_E_com[u,t] - self.lp_e_E_com[u,t] for u in self.players)
            cons = lp_model.addCons(community_balance == 0, name=f"community_elec_balance_{t}")
            lp_community_elec_cons[t] = cons
        
        # Peak power constraint
        for t in self.time_periods:
            grid_import = quicksum(self.lp_i_E_gri[u,t] - self.lp_e_E_gri[u,t] for u in self.players)
            lp_model.addCons(grid_import <= self.lp_chi_peak, name=f"peak_power_{t}")
        
        # Heat constraints
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.lp_i_H_gri[u,t] - self.lp_e_H_gri[u,t] + 
                       self.lp_i_H_com[u,t] - self.lp_e_H_com[u,t])
                
                if u in self.players_with_heatpumps and (u,'hp',t) in self.lp_p:
                    lhs += self.lp_p[u,'hp',t]
                
                if u in self.players_with_heat_storage:
                    if (u,t) in self.lp_b_dis_H and (u,t) in self.lp_b_ch_H:
                        lhs += self.lp_b_dis_H[u,t] - self.lp_b_ch_H[u,t]
                
                rhs = self.params.get(f'd_H_nfl_{u}_{t}', 0)
                
                lp_model.addCons(lhs == rhs, name=f"heat_balance_{u}_{t}")
        
        # Heat pump coupling
        for u in self.players_with_heatpumps:
            for t in self.time_periods:
                if (u,'hp',t) in self.lp_d and (u,'hp',t) in self.lp_p:
                    nu_COP = self.params.get(f'nu_COP_{u}', 3.0)
                    lp_model.addCons(
                        nu_COP * self.lp_d[u,'hp',t] == self.lp_p[u,'hp',t],
                        name=f"heatpump_coupling_{u}_{t}"
                    )
        
        # Heat storage SOC constraints
        for u in self.players_with_heat_storage:
            if (u,0) in self.lp_s_H:
                initial_soc = self.params.get(f'initial_soc', 50)
                lp_model.addCons(self.lp_s_H[u,0] == initial_soc, name=f"initial_soc_H_{u}")
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.lp_s_H and (u,t-1) in self.lp_s_H:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    lp_model.addCons(
                        self.lp_s_H[u,t] == self.lp_s_H[u,t-1] + nu_ch * self.lp_b_ch_H[u,t] - (1/nu_dis) * self.lp_b_dis_H[u,t],
                        name=f"soc_transition_H_{u}_{t}"
                    )
        
        # Community heat balance (IMPORTANT: store constraint reference for dual)
        for t in self.time_periods:
            community_heat_balance = quicksum(self.lp_i_H_com[u,t] - self.lp_e_H_com[u,t] for u in self.players)
            cons = lp_model.addCons(community_heat_balance == 0, name=f"community_heat_balance_{t}")
            lp_community_heat_cons[t] = cons
        
        # Hydrogen constraints
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.lp_i_G_gri[u,t] - self.lp_e_G_gri[u,t] + 
                       self.lp_i_G_com[u,t] - self.lp_e_G_com[u,t])
                
                if u in self.players_with_electrolyzers and (u,'els',t) in self.lp_p:
                    lhs += self.lp_p[u,'els',t]
                
                if u in self.players_with_hydro_storage:
                    if (u,t) in self.lp_b_dis_G and (u,t) in self.lp_b_ch_G:
                        lhs += self.lp_b_dis_G[u,t] - self.lp_b_ch_G[u,t]
                
                rhs = self.params.get(f'd_G_nfl_{u}_{t}', 0)
                
                lp_model.addCons(lhs == rhs, name=f"hydrogen_balance_{u}_{t}")
        
        # Electrolyzer coupling (with fixed binary variables)
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if (u,'els',t) in self.lp_p and (u,'els',t) in self.lp_d:
                    phi1 = self.params.get(f'phi1_{u}', 0.7)
                    phi0 = self.params.get(f'phi0_{u}', 0.0)
                    
                    lp_model.addCons(
                        self.lp_p[u,'els',t] <= phi1 * self.lp_d[u,'els',t] + phi0,
                        name=f"electrolyzer_coupling_{u}_{t}"
                    )
        
        # Electrolyzer commitment constraints with FIXED binary values
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if (u,'els',t) in self.lp_d:
                    C_max = self.params.get(f'C_max_{u}', 100)
                    C_sb = self.params.get(f'C_sb_{u}', 10)
                    C_min = self.params.get(f'C_min_{u}', 20)
                    
                    # Use FIXED binary values instead of binary variables
                    z_on_val = binary_values.get(('z_on', u, t), 0)
                    z_sb_val = binary_values.get(('z_sb', u, t), 0)
                    
                    lp_model.addCons(
                        self.lp_d[u,'els',t] <= C_max * z_on_val + C_sb * z_sb_val,
                        name=f"electrolyzer_max_{u}_{t}"
                    )
                    lp_model.addCons(
                        self.lp_d[u,'els',t] >= C_min * z_on_val + C_sb * z_sb_val,
                        name=f"electrolyzer_min_{u}_{t}"
                    )
        
        # Hydrogen storage SOC constraints
        for u in self.players_with_hydro_storage:
            if (u,0) in self.lp_s_G:
                initial_soc = self.params.get(f'initial_soc', 50)
                lp_model.addCons(self.lp_s_G[u,0] == initial_soc, name=f"initial_soc_G_{u}")
            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.lp_s_G and (u,t-1) in self.lp_s_G:
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    lp_model.addCons(
                        self.lp_s_G[u,t] == self.lp_s_G[u,t-1] + nu_ch * self.lp_b_ch_G[u,t] - (1/nu_dis) * self.lp_b_dis_G[u,t],
                        name=f"soc_transition_G_{u}_{t}"
                    )
        
        # Community hydrogen balance (IMPORTANT: store constraint reference for dual)
        for t in self.time_periods:
            community_hydrogen_balance = quicksum(self.lp_e_G_com[u,t] - self.lp_i_G_com[u,t] for u in self.players)
            cons = lp_model.addCons(community_hydrogen_balance == 0, name=f"community_hydrogen_balance_{t}")
            lp_community_hydrogen_cons[t] = cons
        
        # Renewable availability constraints
        for u in self.players_with_renewables:
            for t in self.time_periods:
                if (u,'res',t) in self.lp_p:
                    availability = self.params.get(f'renewable_availability_{u}_{t}', 1.0)
                    renewable_cap = self.params.get(f'renewable_cap_{u}', 200)
                    
                    lp_model.addCons(
                        self.lp_p[u,'res',t] <= availability * renewable_cap,
                        name=f"renewable_availability_{u}_{t}"
                    )
        
        return lp_community_elec_cons, lp_community_heat_cons, lp_community_hydrogen_cons
    
    def _extract_lp_results(self, lp_model):
        """Extract results from LP model"""
        
        results = {
            'objective_value': lp_model.getObjVal(),
            'electricity': {},
            'heat': {},
            'hydrogen': {},
            'storage': {},
            'production': {},
            'peak_power': lp_model.getVal(self.lp_chi_peak)
        }
        
        for u in self.players:
            for t in self.time_periods:
                # Electricity results
                results['electricity'][u,t] = {
                    'e_gri': lp_model.getVal(self.lp_e_E_gri[u,t]),
                    'i_gri': lp_model.getVal(self.lp_i_E_gri[u,t]),
                    'e_com': lp_model.getVal(self.lp_e_E_com[u,t]),
                    'i_com': lp_model.getVal(self.lp_i_E_com[u,t])
                }
                
                # Heat results
                results['heat'][u,t] = {
                    'e_gri': lp_model.getVal(self.lp_e_H_gri[u,t]),
                    'i_gri': lp_model.getVal(self.lp_i_H_gri[u,t]),
                    'e_com': lp_model.getVal(self.lp_e_H_com[u,t]),
                    'i_com': lp_model.getVal(self.lp_i_H_com[u,t])
                }
                
                # Hydrogen results
                results['hydrogen'][u,t] = {
                    'e_gri': lp_model.getVal(self.lp_e_G_gri[u,t]),
                    'i_gri': lp_model.getVal(self.lp_i_G_gri[u,t]),
                    'e_com': lp_model.getVal(self.lp_e_G_com[u,t]),
                    'i_com': lp_model.getVal(self.lp_i_G_com[u,t])
                }
                
                # Production results
                if (u,'res',t) in self.lp_p:
                    results['production'][u,'res',t] = lp_model.getVal(self.lp_p[u,'res',t])
                if (u,'hp',t) in self.lp_p:
                    results['production'][u,'hp',t] = lp_model.getVal(self.lp_p[u,'hp',t])
                if (u,'els',t) in self.lp_p:
                    results['production'][u,'els',t] = lp_model.getVal(self.lp_p[u,'els',t])
                
                # Storage results
                if (u,t) in self.lp_s_E:
                    results['storage']['elec',u,t] = {
                        'soc': lp_model.getVal(self.lp_s_E[u,t]),
                        'charge': lp_model.getVal(self.lp_b_ch_E[u,t]),
                        'discharge': lp_model.getVal(self.lp_b_dis_E[u,t])
                    }
                
                if (u,t) in self.lp_s_G:
                    results['storage']['hydro',u,t] = {
                        'soc': lp_model.getVal(self.lp_s_G[u,t]),
                        'charge': lp_model.getVal(self.lp_b_ch_G[u,t]),
                        'discharge': lp_model.getVal(self.lp_b_dis_G[u,t])
                    }
                
                if (u,t) in self.lp_s_H:
                    results['storage']['heat',u,t] = {
                        'soc': lp_model.getVal(self.lp_s_H[u,t]),
                        'charge': lp_model.getVal(self.lp_b_ch_H[u,t]),
                        'discharge': lp_model.getVal(self.lp_b_dis_H[u,t])
                    }
        
        return results
    
    def get_results(self):
        """Get optimization results from original MILP"""
        if self.model.getStatus() != "optimal":
            return None
        
        results = {
            'objective_value': self.model.getObjVal(),
            'electricity': {},
            'heat': {},
            'hydrogen': {},
            'storage': {},
            'production': {},
            'peak_power': self.model.getVal(self.chi_peak)
        }
        
        # Extract variable values
        for u in self.players:
            for t in self.time_periods:
                # Electricity results
                results['electricity'][u,t] = {
                    'e_gri': self.model.getVal(self.e_E_gri[u,t]),
                    'i_gri': self.model.getVal(self.i_E_gri[u,t]),
                    'e_com': self.model.getVal(self.e_E_com[u,t]),
                    'i_com': self.model.getVal(self.i_E_com[u,t])
                }
                
                # Heat results
                results['heat'][u,t] = {
                    'e_gri': self.model.getVal(self.e_H_gri[u,t]),
                    'i_gri': self.model.getVal(self.i_H_gri[u,t]),
                    'e_com': self.model.getVal(self.e_H_com[u,t]),
                    'i_com': self.model.getVal(self.i_H_com[u,t])
                }
                
                # Hydrogen results
                results['hydrogen'][u,t] = {
                    'e_gri': self.model.getVal(self.e_G_gri[u,t]),
                    'i_gri': self.model.getVal(self.i_G_gri[u,t]),
                    'e_com': self.model.getVal(self.e_G_com[u,t]),
                    'i_com': self.model.getVal(self.i_G_com[u,t])
                }
                
                # Production results
                if (u,'res',t) in self.p:
                    results['production'][u,'res',t] = self.model.getVal(self.p[u,'res',t])
                if (u,'hp',t) in self.p:
                    results['production'][u,'hp',t] = self.model.getVal(self.p[u,'hp',t])
                if (u,'els',t) in self.p:
                    results['production'][u,'els',t] = self.model.getVal(self.p[u,'els',t])
                
                # Storage results by type
                if (u,t) in self.s_E:
                    results['storage']['elec',u,t] = {
                        'soc': self.model.getVal(self.s_E[u,t]),
                        'charge': self.model.getVal(self.b_ch_E[u,t]),
                        'discharge': self.model.getVal(self.b_dis_E[u,t])
                    }
                
                if (u,t) in self.s_G:
                    results['storage']['hydro',u,t] = {
                        'soc': self.model.getVal(self.s_G[u,t]),
                        'charge': self.model.getVal(self.b_ch_G[u,t]),
                        'discharge': self.model.getVal(self.b_dis_G[u,t])
                    }
                
                if (u,t) in self.s_H:
                    results['storage']['heat',u,t] = {
                        'soc': self.model.getVal(self.s_H[u,t]),
                        'charge': self.model.getVal(self.b_ch_H[u,t]),
                        'discharge': self.model.getVal(self.b_dis_H[u,t])
                    }
        
        return results


# Example usage with Restricted Pricing
if __name__ == "__main__":
    # Define example data
    players = ['u1', 'u2', 'u3']
    time_periods = list(range(24))  # 24 hours
    
    # Example parameters with proper bounds and storage types
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
        
        # Storage parameters (common for all types)
        'storage_power': 50,        # 50 kW power rating
        'storage_capacity': 200,    # 200 kWh capacity
        'initial_soc': 100,         # Initial 100 kWh
        
        # Equipment capacities
        'renewable_cap_u1': 150,    # 150 kW solar
        'hp_cap_u3': 80,           # 80 kW thermal heat pump
        'els_cap_u2': 100,         # 100 kg/day electrolyzer
        
        # Grid connection limits
        'e_E_cap_u1_t': 100,       # 100 kW export limit
        'i_E_cap_u1_t': 150,       # 150 kW import limit
        'e_H_cap_u3_t': 60,        # 60 kW heat export
        'i_H_cap_u3_t': 80,        # 80 kW heat import
        'e_G_cap_u2_t': 50,        # 50 kg/day hydrogen export
        'i_G_cap_u2_t': 30,        # 30 kg/day hydrogen import
        
        # Cost parameters
        'c_sto': 0.01,             # Common storage cost
        
        # Electrolyzer parameters
        'C_max_u2': 100,           # Maximum capacity
        'C_min_u2': 20,            # Minimum capacity
        'C_sb_u2': 10,             # Standby capacity
        'phi1_u2': 0.7,            # Electrolyzer efficiency parameter
        'phi0_u2': 0.0,            # Electrolyzer efficiency parameter
    }
    
    # Add demand data
    for u in players:
        for t in time_periods:
            parameters[f'd_E_nfl_{u}_{t}'] = 10 + 5 * np.sin(2 * np.pi * t / 24)  # Example demand pattern
            parameters[f'd_H_nfl_{u}_{t}'] = 5 + 2 * np.sin(2 * np.pi * t / 24)
            parameters[f'd_G_nfl_{u}_{t}'] = 2
    
    # Add cost parameters
    for u in players:
        parameters[f'c_res_{u}'] = 0.05
        parameters[f'c_hp_{u}'] = 0.1
        parameters[f'c_els_{u}'] = 0.08
        parameters[f'c_su_{u}'] = 50
        parameters[f'c_sto_{u}'] = 0.01
    
    # Add grid prices
    for t in time_periods:
        parameters[f'pi_E_gri_{t}'] = 0.2 + 0.1 * np.sin(2 * np.pi * t / 24)
        parameters[f'pi_H_gri_{t}'] = 0.15
        parameters[f'pi_G_gri_{t}'] = 0.3
    
    # Create and solve model with Restricted Pricing
    lem = LocalEnergyMarket(players, time_periods, parameters)
    
    # Solve using Restricted Pricing
    status, results, prices = lem.solve_with_restricted_pricing()
    
    if status == "optimal":
        print(f"Optimal objective value: {results['objective_value']:.2f}")
        print(f"Peak power: {results['peak_power']:.2f}")
        print("\n=== RESTRICTED PRICING RESULTS ===")
        print(f"Electricity prices: {prices['electricity']}")
        print(f"Heat prices: {prices['heat']}")
        print(f"Hydrogen prices: {prices['hydrogen']}")
    else:
        print(f"Optimization failed with status: {status}")
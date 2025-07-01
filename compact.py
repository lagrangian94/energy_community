"""
Local Energy Market Optimization Model
Based on the presentation slides (pages 4-15)
Using PySCIPOpt for optimization
"""

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
        self.U_E = self.params.get('players_with_renewables', [])  # Players with renewable generators
        self.U_G = self.params.get('players_with_electrolyzers', [])  # Players with electrolyzers  
        self.U_H = self.params.get('players_with_heatpumps', [])  # Players with heat pumps
        
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
        
        # Storage variables
        self.b_dis = {}  # Energy discharged from storage
        self.b_ch = {}   # Energy charged to storage
        self.s = {}      # State of charge level
        
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
                # Electricity variables
                self.e_E_gri[u,t] = self.model.addVar(vtype="C", name=f"e_E_gri_{u}_{t}", lb=0)
                self.i_E_gri[u,t] = self.model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", lb=0)
                self.e_E_com[u,t] = self.model.addVar(vtype="C", name=f"e_E_com_{u}_{t}", lb=0)
                self.i_E_com[u,t] = self.model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0)
                
                # Heat variables
                self.e_H_gri[u,t] = self.model.addVar(vtype="C", name=f"e_H_gri_{u}_{t}", lb=0)
                self.i_H_gri[u,t] = self.model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", lb=0)
                self.e_H_com[u,t] = self.model.addVar(vtype="C", name=f"e_H_com_{u}_{t}", lb=0)
                self.i_H_com[u,t] = self.model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0)
                
                # Hydrogen variables
                self.e_G_gri[u,t] = self.model.addVar(vtype="C", name=f"e_G_gri_{u}_{t}", lb=0)
                self.i_G_gri[u,t] = self.model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", lb=0)
                self.e_G_com[u,t] = self.model.addVar(vtype="C", name=f"e_G_com_{u}_{t}", lb=0)
                self.i_G_com[u,t] = self.model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0)
                
                # Production variables (for renewables, heat pumps, electrolyzers)
                if u in self.U_E:  # Renewable generators
                    self.p[u,'res',t] = self.model.addVar(vtype="C", name=f"p_res_{u}_{t}", lb=0)
                if u in self.U_H:  # Heat pumps
                    self.p[u,'hp',t] = self.model.addVar(vtype="C", name=f"p_hp_{u}_{t}", lb=0)
                    self.d[u,'hp',t] = self.model.addVar(vtype="C", name=f"d_hp_{u}_{t}", lb=0)
                if u in self.U_G:  # Electrolyzers
                    self.p[u,'els',t] = self.model.addVar(vtype="C", name=f"p_els_{u}_{t}", lb=0)
                    self.d[u,'els',t] = self.model.addVar(vtype="C", name=f"d_els_{u}_{t}", lb=0)
                    
                    # Electrolyzer commitment variables
                    self.z_su[u,t] = self.model.addVar(vtype="B", name=f"z_su_{u}_{t}")
                    self.z_on[u,t] = self.model.addVar(vtype="B", name=f"z_on_{u}_{t}")
                    self.z_off[u,t] = self.model.addVar(vtype="B", name=f"z_off_{u}_{t}")
                    self.z_sb[u,t] = self.model.addVar(vtype="B", name=f"z_sb_{u}_{t}")
                
                # Storage variables (if player has storage)
                if self.params.get(f'has_storage_{u}', False):
                    self.b_dis[u,t] = self.model.addVar(vtype="C", name=f"b_dis_{u}_{t}", lb=0)
                    self.b_ch[u,t] = self.model.addVar(vtype="C", name=f"b_ch_{u}_{t}", lb=0)
                    self.s[u,t] = self.model.addVar(vtype="C", name=f"s_{u}_{t}", lb=0)
    
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
        
        # (3) Storage usage costs
        for u in self.players:
            if self.params.get(f'has_storage_{u}', False):
                for t in self.time_periods:
                    c_sto = self.params.get(f'c_sto_{u}', 0)
                    nu_ch = self.params.get('nu_ch', 0.9)
                    nu_dis = self.params.get('nu_dis', 0.9)
                    
                    if (u,t) in self.b_ch and (u,t) in self.b_dis:
                        obj_terms.append(c_sto * (nu_ch * self.b_ch[u,t] + (1/nu_dis) * self.b_dis[u,t]))
        
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
                if u in self.U_E and (u,'res',t) in self.p:
                    lhs += self.p[u,'res',t]
                
                # Add storage discharge/charge
                if self.params.get(f'has_storage_{u}', False):
                    if (u,t) in self.b_dis and (u,t) in self.b_ch:
                        lhs += self.b_dis[u,t] - self.b_ch[u,t]
                
                # RHS: demand
                rhs = self.params.get(f'd_E_nfl_{u}_{t}', 0)  # Non-flexible demand
                
                # Add flexible demand (heat pump, electrolyzer)
                if u in self.U_H and (u,'hp',t) in self.d:
                    rhs += self.d[u,'hp',t]
                if u in self.U_G and (u,'els',t) in self.d:
                    rhs += self.d[u,'els',t]
                
                self.model.addCons(lhs == rhs, name=f"elec_balance_{u}_{t}")
        
        # Constraint (6): Storage SOC transition
        for u in self.players:
            if self.params.get(f'has_storage_{u}', False):
                for t in self.time_periods:
                    if t > 0 and (u,t) in self.s and (u,t-1) in self.s:
                        nu_ch = self.params.get('nu_ch', 0.9)
                        nu_dis = self.params.get('nu_dis', 0.9)
                        
                        self.model.addCons(
                            self.s[u,t] == self.s[u,t-1] + nu_ch * self.b_ch[u,t] - (1/nu_dis) * self.b_dis[u,t],
                            name=f"soc_transition_{u}_{t}"
                        )
        
        # Constraint (9): Community electricity balance
        for t in self.time_periods:
            community_balance = quicksum(self.i_E_com[u,t] - self.e_E_com[u,t] for u in self.players)
            self.model.addCons(community_balance == 0, name=f"community_elec_balance_{t}")
        
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
                if u in self.U_H and (u,'hp',t) in self.p:
                    lhs += self.p[u,'hp',t]
                
                # Add heat storage discharge/charge
                if self.params.get(f'has_heat_storage_{u}', False):
                    if (u,t) in self.b_dis and (u,t) in self.b_ch:
                        lhs += self.b_dis[u,t] - self.b_ch[u,t]
                
                # RHS: heat demand
                rhs = self.params.get(f'd_H_nfl_{u}_{t}', 0)  # Non-flexible heat demand
                if (u,'hp',t) in self.d:
                    rhs += self.d[u,'hp',t]
                
                self.model.addCons(lhs == rhs, name=f"heat_balance_{u}_{t}")
        
        # Heat pump coupling constraint (constraint 12)
        for u in self.U_H:
            for t in self.time_periods:
                if (u,'hp',t) in self.d and (u,'hp',t) in self.p:
                    nu_COP = self.params.get(f'nu_COP_{u}', 3.0)
                    self.model.addCons(
                        nu_COP * self.d[u,'hp',t] == self.p[u,'hp',t],
                        name=f"heatpump_coupling_{u}_{t}"
                    )
        
        # Community heat balance
        for t in self.time_periods:
            community_heat_balance = quicksum(self.i_H_com[u,t] - self.e_H_com[u,t] for u in self.players)
            self.model.addCons(community_heat_balance == 0, name=f"community_heat_balance_{t}")
    
    def _add_hydrogen_constraints(self):
        """Add hydrogen-related constraints from slides 13-15"""
        
        # Hydrogen flow balance for each player
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.i_G_gri[u,t] - self.e_G_gri[u,t] + 
                       self.i_G_com[u,t] - self.e_G_com[u,t])
                
                # Add electrolyzer production
                if u in self.U_G and (u,'els',t) in self.p:
                    lhs += self.p[u,'els',t]
                
                # Add hydrogen storage discharge/charge
                if self.params.get(f'has_hydrogen_storage_{u}', False):
                    if (u,t) in self.b_dis and (u,t) in self.b_ch:
                        lhs += self.b_dis[u,t] - self.b_ch[u,t]
                
                # RHS: hydrogen demand
                rhs = self.params.get(f'd_G_nfl_{u}_{t}', 0)  # Non-flexible hydrogen demand
                
                self.model.addCons(lhs == rhs, name=f"hydrogen_balance_{u}_{t}")
        
        # Electrolyzer coupling constraint (constraint 15)
        for u in self.U_G:
            for t in self.time_periods:
                if (u,'els',t) in self.p and (u,'els',t) in self.d:
                    phi1 = self.params.get(f'phi1_{u}', 0.7)
                    phi0 = self.params.get(f'phi0_{u}', 0.0)
                    
                    self.model.addCons(
                        self.p[u,'els',t] <= phi1 * self.d[u,'els',t] + phi0,
                        name=f"electrolyzer_coupling_{u}_{t}"
                    )
        
        # Electrolyzer commitment constraints (constraints 17-21)
        for u in self.U_G:
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
        
        # Community hydrogen balance
        for t in self.time_periods:
            community_hydrogen_balance = quicksum(self.e_G_com[u,t] - self.i_G_com[u,t] for u in self.players)
            self.model.addCons(community_hydrogen_balance == 0, name=f"community_hydrogen_balance_{t}")
    
    def _add_community_constraints(self):
        """Add additional community-level constraints"""
        
        # Grid capacity constraints
        for u in self.players:
            for t in self.time_periods:
                # Electricity export/import capacity
                e_E_cap = self.params.get(f'e_E_cap_{u}_{t}', float('inf'))
                i_E_cap = self.params.get(f'i_E_cap_{u}_{t}', float('inf'))
                
                self.model.addCons(self.e_E_gri[u,t] <= e_E_cap, name=f"elec_export_cap_{u}_{t}")
                self.model.addCons(self.i_E_gri[u,t] <= i_E_cap, name=f"elec_import_cap_{u}_{t}")
                
                # Heat export/import capacity
                e_H_cap = self.params.get(f'e_H_cap_{u}_{t}', float('inf'))
                i_H_cap = self.params.get(f'i_H_cap_{u}_{t}', float('inf'))
                
                self.model.addCons(self.e_H_gri[u,t] <= e_H_cap, name=f"heat_export_cap_{u}_{t}")
                self.model.addCons(self.i_H_gri[u,t] <= i_H_cap, name=f"heat_import_cap_{u}_{t}")
                
                # Hydrogen export/import capacity
                e_G_cap = self.params.get(f'e_G_cap_{u}_{t}', float('inf'))
                i_G_cap = self.params.get(f'i_G_cap_{u}_{t}', float('inf'))
                
                self.model.addCons(self.e_G_gri[u,t] <= e_G_cap, name=f"hydrogen_export_cap_{u}_{t}")
                self.model.addCons(self.i_G_gri[u,t] <= i_G_cap, name=f"hydrogen_import_cap_{u}_{t}")
    
    def solve(self):
        """Solve the optimization model"""
        self.model.optimize()
        return self.model.getStatus()
    
    def get_results(self):
        """Get optimization results"""
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
                
                # Storage results
                if (u,t) in self.s:
                    results['storage'][u,t] = {
                        'soc': self.model.getVal(self.s[u,t]),
                        'charge': self.model.getVal(self.b_ch[u,t]),
                        'discharge': self.model.getVal(self.b_dis[u,t])
                    }
        
        return results

# Example usage
if __name__ == "__main__":
    # Define example data
    players = ['u1', 'u2', 'u3']
    time_periods = list(range(24))  # 24 hours
    
    # Example parameters
    parameters = {
        'players_with_renewables': ['u1'],
        'players_with_electrolyzers': ['u2'],
        'players_with_heatpumps': ['u3'],
        'has_storage_u1': True,
        'nu_ch': 0.9,
        'nu_dis': 0.9,
        'pi_peak': 100,
        # Add more parameters as needed
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
    
    # Create and solve model
    lem = LocalEnergyMarket(players, time_periods, parameters)
    status = lem.solve()
    
    if status == "optimal":
        results = lem.get_results()
        print(f"Optimal objective value: {results['objective_value']:.2f}")
        print(f"Peak power: {results['peak_power']:.2f}")
    else:
        print(f"Optimization failed with status: {status}")
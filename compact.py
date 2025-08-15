from pyscipopt import Model, quicksum
import numpy as np
from typing import Dict, List, Tuple, Optional

def solve_and_extract_results(model):
    """
    모델을 풀고 결과를 반환합니다.
    
    Returns:
    --------
    results : dict
        test.py의 generate_initial_patterns와 동일한 구조의 딕셔너리
    """
   
    # 최적화 상태 확인
    status = model.getStatus()
    time = model.getSolvingTime()
    print(f"MIP model status: {status}, time: {time}")
    if status in ["optimal", "gaplimit"]:            
        # 결과 저장할 딕셔너리 초기화
        results = {}
        if model.data != None:
            # 저장된 모든 변수에 대해 최적해 값 추출
            for var_name, var_dict in model.data["vars"].items():
                if isinstance(var_dict, dict):
                    # 딕셔너리 형태의 변수들 처리 (예: e_E_gri, i_E_gri 등)
                    result_dict = {}
                    for key, var in var_dict.items():
                        try:
                            result_dict[key] = model.getVal(var)
                        except:
                            print(f"Could not get value for {var_name}_{key}")
                    results[var_name] = result_dict
                else:
                    # 단일 변수 처리 (예: chi_peak)
                    try:
                        results[var_name] = model.getVal(var_dict)
                    except:
                        print(f"Could not get value for {var_name}")
        else:
            vars_name = ["e_E_gri", "i_E_gri", "e_E_com", "i_E_com", "e_H_gri", "i_H_gri", 
                        "e_H_com", "i_H_com", "e_G_gri", "i_G_gri", "e_G_com", "i_G_com",
                        "p", "d", "b_dis_E", "b_ch_E", "s_E", "b_dis_G", "b_ch_G", "s_G",
                        "b_dis_H", "b_ch_H", "s_H", "z_su", "z_on", "z_off", "z_sb", "chi_peak"]
            raise Exception(f"Model data is None. Cannot extract results for {vars_name}")
            
        return status, results
    else:
        print(f"Model not solved to optimality. Status: {status}")
        return status, None

def process_cons_arr(arr, process_func):
    """
    Recursively process arrays of constraint objects
    
    Args:
        arr: A numpy array (potentially nested) of constraint objects
        process_func: Function to apply to each constraint object
        
    Returns:
        Processed array with the same structure
    """
    # # Base case: if it's not an array or not of object type, just return it
    # if not isinstance(arr, np.ndarray) or arr.dtype != object:
    #     return arr
    
    # Create a new array with the same shape
    new_arr = {}
    
    # Process each element
    for key, val in arr.items():
        new_arr[key] = process_func(val)
    
    return new_arr

class LocalEnergyMarket:
    def __init__(self, 
                 players: List[str],
                 time_periods: List[int],
                 parameters: Dict,
                 isLP: bool = False):
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
        self.isLP = isLP
        # Initialize model.data dictionary to store variables and constraints
        self.model.data = {"vars": {}, "cons": {}}
        
        # Sets definition based on slides
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
        # Combined sets for energy types
        self.U_E = list(set(self.players_with_renewables + self.players_with_elec_storage))  # Players with electricity assets
        self.U_G = list(set(self.players_with_electrolyzers + self.players_with_hydro_storage))  # Players with hydro assets
        self.U_H = list(set(self.players_with_heatpumps + self.players_with_heat_storage))  # Players with heat assets
        ## 추후에 non-flexible demand를 가진 player들이 storage를 가지고 있다고 고려할 수도 있음.
        self.U_E_nfl = list(set(self.players_with_nfl_elec_demand))
        self.U_G_nfl = list(set(self.players_with_nfl_hydro_demand))
        self.U_H_nfl = list(set(self.players_with_nfl_heat_demand))
        self.U_E_fl = list(set(self.players_with_fl_elec_demand))
        self.U_G_fl = list(set(self.players_with_fl_hydro_demand))
        self.U_H_fl = list(set(self.players_with_fl_heat_demand))
        # Store community balance constraints for dual access
        self.community_elec_balance_cons = {}
        self.community_heat_balance_cons = {}
        self.community_hydro_balance_cons = {}
        
        # Initialize constraint storage dictionaries
        self.elec_balance_cons = {}
        self.heat_balance_cons = {}
        self.hydro_balance_cons = {}
        self.storage_cons = {}
        self.production_cons = {}
        self.electrolyzer_cons = {}
        self.heatpump_cons = {}
        self.renewable_cons = {}
        self.peak_power_cons = {}
        
        # Initialize variables
        self._create_variables(isLP=self.isLP)
        self._create_constraints()
        
        # Store all variables and constraints in model.data
        self._store_model_data()
    
    def _create_variables(self, isLP=False):
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
        
        # hydro variables  
        self.e_G_gri = {}  # hydro exported to grid
        self.i_G_gri = {}  # hydro imported from grid
        self.e_G_com = {}  # hydro exported to community
        self.i_G_com = {}  # hydro imported from community
        
        # Production variables
        self.p = {}  # Energy production by generators
        self.fl_d = {}  # Flexible energy consumption
        self.nfl_d = {}  # Non-flexible energy consumption
        
        # Storage variables by type
        self.b_dis_E = {}  # Electricity discharged from storage
        self.b_ch_E = {}   # Electricity charged to storage
        self.s_E = {}      # Electricity storage SOC level
        
        self.b_dis_G = {}  # hydro discharged from storage
        self.b_ch_G = {}   # hydro charged to storage
        self.s_G = {}      # hydro storage SOC level
        
        self.b_dis_H = {}  # Heat discharged from storage
        self.b_ch_H = {}   # Heat charged to storage
        self.s_H = {}      # Heat storage SOC level
        
        # Electrolyzer commitment variables
        self.z_su = {}   # Start-up decision
        self.z_on = {}   # Turn on decision
        self.z_off = {}  # Turn off decision
        self.z_sb = {}   # Stand-by decision
        
        # Peak power variable
        self.chi_peak = self.model.addVar(vtype="C", name="chi_peak", lb=0, obj=self.params.get('pi_peak', 0))
        
        # Create variables for each player and time period
        for u in self.players:
            for t in self.time_periods:
                if u in self.U_E:
                    self.e_E_gri[u,t] = self.model.addVar(vtype="C", name=f"e_E_gri_{u}_{t}", lb=0, 
                                                        ub=self.params.get(f'e_E_cap_{u}_{t}', 1000), obj=-1*self.params.get(f'pi_E_gri_export_{t}', 0))
                    self.e_E_com[u,t] = self.model.addVar(vtype="C", name=f"e_E_com_{u}_{t}", lb=0, ub=1000)
                if u in self.U_H:
                    self.e_H_gri[u,t] = self.model.addVar(vtype="C", name=f"e_H_gri_{u}_{t}", lb=0,
                                                        ub=self.params.get(f'e_H_cap_{u}_{t}', 500), obj=-1*self.params.get(f'pi_H_gri_export_{t}', 0))
                    self.e_H_com[u,t] = self.model.addVar(vtype="C", name=f"e_H_com_{u}_{t}", lb=0, ub=500)
                if u in self.U_G:
                    self.e_G_gri[u,t] = self.model.addVar(vtype="C", name=f"e_G_gri_{u}_{t}", lb=0,
                                                        ub=self.params.get(f'e_G_cap_{u}_{t}', 100), obj=-1*self.params.get(f'pi_G_gri_export_{t}', 0))
                    self.e_G_com[u,t] = self.model.addVar(vtype="C", name=f"e_G_com_{u}_{t}", lb=0, ub=100)
                # Production variables (for renewables, heat pumps, electrolyzers) with capacity limits
                if u in self.players_with_renewables:  # Renewable generators
                    renewable_cap = self.params.get(f'renewable_cap_{u}', 200)  # Default 200 kW
                    c_res = self.params.get(f'c_res_{u}', 0)
                    self.p[u,'res',t] = self.model.addVar(vtype="C", name=f"p_res_{u}_{t}", 
                                                        lb=0, ub=renewable_cap, obj=c_res)
                if u in self.players_with_heatpumps:  # Heat pumps
                    hp_cap = self.params.get(f'hp_cap_{u}', 100)  # Default 100 kW thermal
                    c_hp = self.params.get(f'c_hp_{u}', 0)
                    self.p[u,'hp',t] = self.model.addVar(vtype="C", name=f"p_hp_{u}_{t}", 
                                                       lb=0, ub=hp_cap, obj=c_hp)

                if u in self.players_with_electrolyzers:  # Electrolyzers
                    els_cap = self.params.get(f'els_cap_{u}', 150)  # Default 150 kg/day
                    c_els = self.params.get(f'c_els_{u}', 0)
                    self.p[u,'els',t] = self.model.addVar(vtype="C", name=f"p_els_{u}_{t}", 
                                                        lb=0, ub=els_cap, obj=c_els)
                    
                    # Electrolyzer commitment variables
                    c_su = self.params.get(f'c_su_{u}', 0)
                    vartype = "C" if isLP else "B"
                    self.z_su[u,t] = self.model.addVar(vtype=vartype, name=f"z_su_{u}_{t}", obj=c_su)
                    self.z_on[u,t] = self.model.addVar(vtype=vartype, name=f"z_on_{u}_{t}")
                    self.z_off[u,t] = self.model.addVar(vtype=vartype, name=f"z_off_{u}_{t}")
                    self.z_sb[u,t] = self.model.addVar(vtype=vartype, name=f"z_sb_{u}_{t}")
            
                    
                # Non-flexible demand variables
                if u in self.players_with_nfl_elec_demand:
                    nfl_elec_demand_t = self.params.get(f'd_E_nfl_{u}_{t}', 0)
                    self.nfl_d[u,'elec',t] = self.model.addVar(vtype="C", name=f"d_elec_{u}_{t}", 
                                                       lb=nfl_elec_demand_t, ub=nfl_elec_demand_t)
                    self.i_E_gri[u,t] = self.model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_E_cap_{u}_{t}', 1000), obj=self.params.get(f'pi_E_gri_import_{t}', 0))
                    self.i_E_com[u,t] = self.model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0, ub=1000)
                if u in self.players_with_nfl_hydro_demand:
                    nfl_hydro_demand_t = self.params.get(f'd_G_nfl_{u}_{t}', 0)
                    self.nfl_d[u,'hydro',t] = self.model.addVar(vtype="C", name=f"d_hydro_{u}_{t}", 
                                                       lb=nfl_hydro_demand_t, ub=nfl_hydro_demand_t)
                    self.i_G_gri[u,t] = self.model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_G_cap_{u}_{t}', 100), obj=self.params.get(f'pi_G_gri_import_{t}', 0))
                    self.i_G_com[u,t] = self.model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0, ub=100)
                if u in self.players_with_nfl_heat_demand:
                    nfl_heat_demand_t = self.params.get(f'd_H_nfl_{u}_{t}', 0)
                    self.nfl_d[u,'heat',t] = self.model.addVar(vtype="C", name=f"d_heat_{u}_{t}", 
                                                       lb=nfl_heat_demand_t, ub=nfl_heat_demand_t)
                    self.i_H_gri[u,t] = self.model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_H_cap_{u}_{t}', 500), obj=self.params.get(f'pi_H_gri_import_{t}', 0))
                    self.i_H_com[u,t] = self.model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0, ub=500)
                
                # Flexible demand variables
                if u in self.players_with_fl_elec_demand:
                    fl_elec_demand_cap = 10**6 ## 일단 assuming large numbers just to bound the subproblem
                    self.fl_d[u,'elec',t] = self.model.addVar(vtype="C", name=f"d_elec_{u}_{t}", 
                                                       lb=0.0, ub=fl_elec_demand_cap)
                    self.i_E_gri[u,t] = self.model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_E_cap_{u}_{t}', 1000), obj=self.params.get(f'pi_E_gri_import_{t}', 0))
                    self.i_E_com[u,t] = self.model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0, ub=1000)
                if u in self.players_with_fl_hydro_demand:
                    fl_hydro_demand_cap = 10**6
                    self.fl_d[u,'hydro',t] = self.model.addVar(vtype="C", name=f"d_hydro_{u}_{t}", 
                                                       lb=0.0, ub=fl_hydro_demand_cap)
                    self.i_G_gri[u,t] = self.model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_G_cap_{u}_{t}', 100), obj=self.params.get(f'pi_G_gri_import_{t}', 0))
                    self.i_G_com[u,t] = self.model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0, ub=100)
                if u in self.players_with_fl_heat_demand:
                    fl_heat_demand_cap = 10**6
                    self.fl_d[u,'heat',t] = self.model.addVar(vtype="C", name=f"d_heat_{u}_{t}", 
                                                       lb=0.0, ub=fl_heat_demand_cap)
                    self.i_H_gri[u,t] = self.model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_H_cap_{u}_{t}', 500), obj=self.params.get(f'pi_H_gri_import_{t}', 0))
                    self.i_H_com[u,t] = self.model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0, ub=500)
                # Storage variables by type with capacity constraints
                storage_power = self.params.get(f'storage_power', 50)  # kW power rating
                storage_capacity = self.params.get(f'storage_capacity', 200)  # kWh capacity
                c_sto = self.params.get(f'c_sto', 0.01)  # Common storage cost
                nu_ch = self.params.get('nu_ch', 0.9)
                nu_dis = self.params.get('nu_dis', 0.9)
                # Electricity storage
                if u in self.players_with_elec_storage:
                    self.b_dis_E[u,t] = self.model.addVar(vtype="C", name=f"b_dis_E_{u}_{t}", 
                                                        lb=0, ub=storage_power, obj=c_sto*(1/nu_dis))
                    self.b_ch_E[u,t] = self.model.addVar(vtype="C", name=f"b_ch_E_{u}_{t}", 
                                                       lb=0, ub=storage_power, obj=c_sto*nu_ch)
                    self.s_E[u,t] = self.model.addVar(vtype="C", name=f"s_E_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
                
                # hydro storage
                if u in self.players_with_hydro_storage:
                    self.b_dis_G[u,t] = self.model.addVar(vtype="C", name=f"b_dis_G_{u}_{t}", 
                                                        lb=0, ub=storage_power, obj=c_sto*(1/nu_dis))
                    self.b_ch_G[u,t] = self.model.addVar(vtype="C", name=f"b_ch_G_{u}_{t}", 
                                                       lb=0, ub=storage_power, obj=c_sto*nu_ch)
                    self.s_G[u,t] = self.model.addVar(vtype="C", name=f"s_G_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
                
                # Heat storage
                if u in self.players_with_heat_storage:
                    self.b_dis_H[u,t] = self.model.addVar(vtype="C", name=f"b_dis_H_{u}_{t}", 
                                                        lb=0, ub=storage_power, obj=c_sto*(1/nu_dis))
                    self.b_ch_H[u,t] = self.model.addVar(vtype="C", name=f"b_ch_H_{u}_{t}", 
                                                       lb=0, ub=storage_power, obj=c_sto*nu_ch)
                    self.s_H[u,t] = self.model.addVar(vtype="C", name=f"s_H_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
    
    def _create_constraints(self):
        """Create constraints based on slides 9-15"""
        
        # Electricity flow balance constraints (slide 9)
        self._add_electricity_constraints()
        
        # Heat flow balance constraints (slide 12)
        self._add_heat_constraints()
        
        # hydro flow balance constraints (slide 13-14)
        self._add_hydro_constraints()
        
    
    def _store_model_data(self):
        """Store all variables and constraints in model.data dictionary"""
        
        # Store all variables
        self.model.data["vars"]["e_E_gri"] = self.e_E_gri
        self.model.data["vars"]["i_E_gri"] = self.i_E_gri
        self.model.data["vars"]["e_E_com"] = self.e_E_com
        self.model.data["vars"]["i_E_com"] = self.i_E_com
        
        self.model.data["vars"]["e_H_gri"] = self.e_H_gri
        self.model.data["vars"]["i_H_gri"] = self.i_H_gri
        self.model.data["vars"]["e_H_com"] = self.e_H_com
        self.model.data["vars"]["i_H_com"] = self.i_H_com
        
        self.model.data["vars"]["e_G_gri"] = self.e_G_gri
        self.model.data["vars"]["i_G_gri"] = self.i_G_gri
        self.model.data["vars"]["e_G_com"] = self.e_G_com
        self.model.data["vars"]["i_G_com"] = self.i_G_com
        
        self.model.data["vars"]["p"] = self.p
        self.model.data["vars"]["fl_d"] = self.fl_d
        self.model.data["vars"]["nfl_d"] = self.nfl_d
        
        self.model.data["vars"]["b_dis_E"] = self.b_dis_E
        self.model.data["vars"]["b_ch_E"] = self.b_ch_E
        self.model.data["vars"]["s_E"] = self.s_E
        
        self.model.data["vars"]["b_dis_G"] = self.b_dis_G
        self.model.data["vars"]["b_ch_G"] = self.b_ch_G
        self.model.data["vars"]["s_G"] = self.s_G
        
        self.model.data["vars"]["b_dis_H"] = self.b_dis_H
        self.model.data["vars"]["b_ch_H"] = self.b_ch_H
        self.model.data["vars"]["s_H"] = self.s_H
        
        self.model.data["vars"]["z_su"] = self.z_su
        self.model.data["vars"]["z_on"] = self.z_on
        self.model.data["vars"]["z_off"] = self.z_off
        self.model.data["vars"]["z_sb"] = self.z_sb
        
        self.model.data["vars"]["chi_peak"] = self.chi_peak
        
        # Store all constraints (we need to collect them during creation)
        # For now, we'll store the constraint dictionaries that were created
        self.model.data["cons"]["community_elec_balance"] = self.community_elec_balance_cons
        self.model.data["cons"]["community_heat_balance"] = self.community_heat_balance_cons
        self.model.data["cons"]["community_hydro_balance"] = self.community_hydro_balance_cons
        
        # Store all other constraint types
        self.model.data["cons"]["elec_balance"] = self.elec_balance_cons
        self.model.data["cons"]["heat_balance"] = self.heat_balance_cons
        self.model.data["cons"]["hydro_balance"] = self.hydro_balance_cons
        self.model.data["cons"]["storage"] = self.storage_cons
        self.model.data["cons"]["production"] = self.production_cons
        self.model.data["cons"]["electrolyzer"] = self.electrolyzer_cons
        self.model.data["cons"]["heatpump"] = self.heatpump_cons
        self.model.data["cons"]["renewable"] = self.renewable_cons
        self.model.data["cons"]["peak_power"] = self.peak_power_cons
    
    def _add_electricity_constraints(self):
        """Add electricity-related constraints from slides 9-10"""
        
        # Constraint (5): Electricity flow balance for each player
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.i_E_gri.get((u,t),0) - self.e_E_gri.get((u,t),0) + 
                       self.i_E_com.get((u,t),0) - self.e_E_com.get((u,t),0))
                
                # Add renewable generation
                lhs += self.p.get((u,'res',t),0)
                
                # Add electricity storage discharge/charge
                lhs += self.b_dis_E.get((u,t),0) - self.b_ch_E.get((u,t),0)
                
                # RHS: demand
                rhs = self.nfl_d.get((u,'elec',t),0) + self.fl_d.get((u,'elec',t),0)  # Non-flexible + flexible demand
                
                if not (type(lhs) == int):
                    cons = self.model.addCons(lhs == rhs, name=f"elec_balance_{u}_{t}")
                    self.elec_balance_cons[f"elec_balance_{u}_{t}"] = cons
        
        # Constraint (6): Electricity storage SOC transition with initial condition
        for u in self.players_with_elec_storage:
            # Set initial SOC
            if (u,0) in self.s_E:
                initial_soc = self.params.get(f'initial_soc', 50)  # Default 50% SOC
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
        
        # Constraint (9): Community electricity balance
        for t in self.time_periods:
            community_balance = quicksum(self.i_E_com.get((u,t),0) - self.e_E_com.get((u,t),0) for u in self.players)
            cons = self.model.addCons(community_balance == 0, name=f"community_elec_balance_{t}")
            self.community_elec_balance_cons[f"community_elec_balance_{t}"] = cons
        
        # Constraint (10): Peak power constraint
        for t in self.time_periods:
            grid_import = quicksum(self.i_E_gri.get((u,t),0) - self.e_E_gri.get((u,t),0) for u in self.players)
            cons = self.model.addCons(grid_import <= self.chi_peak, name=f"peak_power_{t}")
            self.peak_power_cons[f"peak_power_{t}"] = cons
    
    def _add_heat_constraints(self):
        """Add heat-related constraints from slide 12"""
        
        # Heat flow balance for each player
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.i_H_gri.get((u,t),0) - self.e_H_gri.get((u,t),0) + 
                       self.i_H_com.get((u,t),0) - self.e_H_com.get((u,t),0))
                
                # Add heat pump production
                lhs += self.p.get((u,'hp',t),0)
                
                # Add heat storage discharge/charge
                lhs += self.b_dis_H.get((u,t),0) - self.b_ch_H.get((u,t),0)
                
                # RHS: heat demand
                rhs = self.nfl_d.get((u,'heat',t),0) + self.fl_d.get((u,'heat',t),0)  # Non-flexible + flexible demand
                
                if not (type(lhs) == int):
                    cons = self.model.addCons(lhs == rhs, name=f"heat_balance_{u}_{t}")
                    self.heat_balance_cons[f"heat_balance_{u}_{t}"] = cons
        
        # Heat pump coupling constraint (constraint 12)
        for u in self.players_with_heatpumps:
            for t in self.time_periods:
                nu_COP = self.params.get(f'nu_COP_{u}', 3.0)
                cons = self.model.addCons(
                    nu_COP * self.fl_d.get((u,'elec',t),0) == self.p.get((u,'hp',t),0),
                    name=f"heatpump_coupling_{u}_{t}"
                )
                self.heatpump_cons[f"heatpump_coupling_{u}_{t}"] = cons
        
        # Heat storage SOC transition
        for u in self.players_with_heat_storage:
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
        
        # Community heat balance
        for t in self.time_periods:
            community_heat_balance = quicksum(self.i_H_com.get((u,t),0) - self.e_H_com.get((u,t),0) for u in self.players)
            cons = self.model.addCons(community_heat_balance == 0, name=f"community_elec_balance_{t}")
            self.community_heat_balance_cons[f"community_heat_balance_{t}"] = cons
    
    def _add_hydro_constraints(self):
        """Add hydro-related constraints from slides 13-15"""
        
        # hydro flow balance for each player
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.i_G_gri.get((u,t),0) - self.e_G_gri.get((u,t),0) + 
                       self.i_G_com.get((u,t),0) - self.e_G_com.get((u,t),0))
                
                # Add electrolyzer production
                lhs += self.p.get((u,'els',t),0)
                
                # Add hydro storage discharge/charge
                lhs += self.b_dis_G.get((u,t),0) - self.b_ch_G.get((u,t),0)
                
                # RHS: hydro demand
                rhs = self.nfl_d.get((u,'hydro',t),0) + self.fl_d.get((u,'hydro',t),0)  # Non-flexible + flexible demand
                
                if not (type(lhs) == int):
                    cons = self.model.addCons(lhs == rhs, name=f"hydro_balance_{u}_{t}")
                    self.hydro_balance_cons[f"hydro_balance_{u}_{t}"] = cons

        # Electrolyzer coupling constraint (constraint 15)
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                phi1 = self.params.get(f'phi1_{u}', 0.7)
                phi0 = self.params.get(f'phi0_{u}', 0.0)
                
                cons = self.model.addCons(
                    self.p.get((u,'els',t),0) <= phi1 * self.fl_d.get((u,'elec',t),0) + phi0,
                    name=f"electrolyzer_coupling_{u}_{t}"
                )
                self.electrolyzer_cons['coupling', u, t] = cons
        
        # Electrolyzer commitment constraints (constraints 17-21)
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                # Constraint 17: exactly one state
                cons = self.model.addCons(
                    self.z_on[u,t] + self.z_off[u,t] + self.z_sb[u,t] == 1,
                    name=f"electrolyzer_state_{u}_{t}"
                )
                self.electrolyzer_cons['state', u, t] = cons
                
                # Constraints 18-19: production bounds
                C_max = self.params.get(f'C_max_{u}', 100)
                C_sb = self.params.get(f'C_sb_{u}', 10)
                C_min = self.params.get(f'C_min_{u}', 20)
                
                cons = self.model.addCons(
                    self.fl_d[u,'elec',t] <= C_max * self.z_on[u,t] + C_sb * self.z_sb[u,t],
                    name=f"electrolyzer_max_{u}_{t}"
                )
                self.electrolyzer_cons[f"electrolyzer_max_{u}_{t}"] = cons
                
                cons = self.model.addCons(
                    self.fl_d[u,'elec',t] >= C_min * self.z_on[u,t] + C_sb * self.z_sb[u,t],
                    name=f"electrolyzer_min_{u}_{t}"
                )
                self.electrolyzer_cons[f"electrolyzer_min_{u}_{t}"] = cons
                
                # Constraint 20: startup logic
                if t > 0:
                    cons = self.model.addCons(
                        self.z_su[u,t] >= self.z_on[u,t] - self.z_on[u,t-1] - self.z_sb[u,t],
                        name=f"electrolyzer_startup_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"electrolyzer_startup_{u}_{t}"] = cons
                    
                    # off to standby is not allowed
                    cons = self.model.addCons(
                        self.z_off[u,t-1] + self.z_sb[u,t] <= 1.0,
                        name=f"electrolyzer_forbid_off_to_sb_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"electrolyzer_forbid_off_to_sb_{u}_{t}"] = cons
                else:
                    cons = self.model.addCons(
                        self.z_off[u,t] >= 1.0,
                        name=f"electrolyzer_initial_state_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"electrolyzer_initial_state_{u}_{t}"] = cons

        
        # hydro storage SOC transition
        for u in self.players_with_hydro_storage:
            # Set initial SOC
            if (u,0) in self.s_G:
                initial_soc = self.params.get(f'initial_soc', 0)
                initial_soc = 0.0
                # cons = self.model.addCons(self.s_G[u,0] == initial_soc, name=f"initial_soc_G_{u}")
                # self.storage_cons[f"initial_soc_G_{u}"] = cons
            nu_ch = self.params.get('nu_ch', 0.9)
            nu_dis = self.params.get('nu_dis', 0.9)            
            for t in self.time_periods:
                if t > 0 and (u,t) in self.s_G and (u,t-1) in self.s_G:
                    cons = self.model.addCons(
                        self.s_G[u,t] == self.s_G[u,t-1] + nu_ch * self.b_ch_G[u,t] - (1/nu_dis) * self.b_dis_G[u,t],
                        name=f"soc_transition_G_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_G_{u}_{t}"] = cons
                elif t==0:
                    cons = self.model.addCons(
                        self.s_G[u,t] == initial_soc + nu_ch * self.b_ch_G[u,t] - (1/nu_dis) * self.b_dis_G[u,t],
                        name=f"soc_transition_G_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_G_{u}_{t}"] = cons
        
        # Community hydro balance
        for t in self.time_periods:
            community_hydro_balance = quicksum(self.e_G_com.get((u,t),0) - self.i_G_com.get((u,t),0) for u in self.players)
            cons = self.model.addCons(community_hydro_balance == 0, name=f"community_hydro_balance_{t}")
            self.community_hydro_balance_cons[f"community_hydro_balance_{t}"] = cons
        
        # Force u5 to only import hydrogen from community (not from grid)
        # for t in self.time_periods:
        #     if t==10:
        #         cons = self.model.addCons(
        #             self.i_G_gri.get(('u5',t),0) == 0, 
        #             name=f"force_u5_community_only_{t}"
        #         )
        #         self.storage_cons[f"force_u5_community_only_{t}"] = cons
    
    
    def solve(self):
        """Solve the optimization model"""
        self.model.optimize()
        return self.model.getStatus()
    
    def solve_complete_model(self):
        """
        Solve the complete optimization model and analyze revenue by resource type
        
        Returns:
            tuple: (status, results, revenue_analysis)
                - status: optimization status
                - results: optimization results dictionary
                - revenue_analysis: dictionary with revenue breakdown by resource type
        """
        print("Solving complete optimization model...")
        
        # Solve the model
        status = self.solve()
        
        if status != "optimal":
            print(f"Optimization failed with status: {status}")
            return status, None, None
        
        print("Model solved successfully. Extracting results and analyzing revenue...")
        
        # Extract results using existing function
        status, results = solve_and_extract_results(self.model)
        
        if status != "optimal":
            print(f"Failed to extract results. Status: {status}")
            return status, None, None
        
        # Analyze revenue by resource type
        revenue_analysis = self._analyze_revenue_by_resource(results)
        # Analyze energy flows
        flow_analysis = self.analyze_energy_flows(results)
        return status, results, revenue_analysis
    
    def _analyze_revenue_by_resource(self, results):
        """
        Analyze revenue contribution by resource type from objective function
        Based ONLY on addVar(obj=...) coefficients defined in the model
        
        Args:
            results: Dictionary containing optimization results
            
        Returns:
            Dictionary with revenue breakdown by resource type
        """
        revenue_analysis = {
            'electricity': {
                'grid_export_revenue': 0.0,
                'grid_import_cost': 0.0,
                'production_cost': 0.0,
                'storage_cost': 0.0,
                'net': 0.0
            },
            'hydrogen': {
                'grid_export_revenue': 0.0,
                'grid_import_cost': 0.0,
                'production_cost': 0.0,
                'storage_cost': 0.0,
                'startup_cost': 0.0,
                'net': 0.0
            },
            'heat': {
                'grid_export_revenue': 0.0,
                'grid_import_cost': 0.0,
                'production_cost': 0.0,
                'storage_cost': 0.0,
                'net': 0.0
            },
            'common': {
                'peak_power_cost': 0.0
            },
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'net_profit': 0.0
        }
        
        c_sto = self.params.get('c_sto', 0.01)
        nu_ch = self.params.get('nu_ch', 0.9)
        nu_dis = self.params.get('nu_dis', 0.9)
        
        # ========== ELECTRICITY ==========
        # Export revenue
        if 'e_E_gri' in results:
            for (u, t), val in results['e_E_gri'].items():
                if val > 0:
                    price = self.params.get(f'pi_E_gri_export_{t}', 0)
                    revenue_analysis['electricity']['grid_export_revenue'] += val * price
        
        # Import cost
        if 'i_E_gri' in results:
            for (u, t), val in results['i_E_gri'].items():
                if val > 0:
                    price = self.params.get(f'pi_E_gri_import_{t}', 0)
                    revenue_analysis['electricity']['grid_import_cost'] += val * price
        
        # Renewable production cost
        if 'p' in results:
            for (u, resource_type, t), val in results['p'].items():
                if val > 0 and resource_type == 'res':
                    cost = self.params.get(f'c_res_{u}', 0)
                    revenue_analysis['electricity']['production_cost'] += val * cost
        
        # Electricity storage cost
        if 'b_ch_E' in results:
            for (u, t), val in results['b_ch_E'].items():
                if val > 0:
                    revenue_analysis['electricity']['storage_cost'] += val * c_sto * nu_ch
        
        if 'b_dis_E' in results:
            for (u, t), val in results['b_dis_E'].items():
                if val > 0:
                    revenue_analysis['electricity']['storage_cost'] += val * c_sto * (1/nu_dis)
        
        # ========== HYDROGEN ==========
        # Export revenue
        if 'e_G_gri' in results:
            for (u, t), val in results['e_G_gri'].items():
                if val > 0:
                    price = self.params.get(f'pi_G_gri_export_{t}', 0)
                    revenue_analysis['hydrogen']['grid_export_revenue'] += val * price
        
        # Import cost
        if 'i_G_gri' in results:
            for (u, t), val in results['i_G_gri'].items():
                if val > 0:
                    price = self.params.get(f'pi_G_gri_import_{t}', 0)
                    revenue_analysis['hydrogen']['grid_import_cost'] += val * price
        
        # Electrolyzer production cost
        if 'p' in results:
            for (u, resource_type, t), val in results['p'].items():
                if val > 0 and resource_type == 'els':
                    cost = self.params.get(f'c_els_{u}', 0)
                    revenue_analysis['hydrogen']['production_cost'] += val * cost
        
        # Hydrogen storage cost
        if 'b_ch_G' in results:
            for (u, t), val in results['b_ch_G'].items():
                if val > 0:
                    revenue_analysis['hydrogen']['storage_cost'] += val * c_sto * nu_ch
        
        if 'b_dis_G' in results:
            for (u, t), val in results['b_dis_G'].items():
                if val > 0:
                    revenue_analysis['hydrogen']['storage_cost'] += val * c_sto * (1/nu_dis)
        
        # Electrolyzer startup cost
        if 'z_su' in results:
            for (u, t), val in results['z_su'].items():
                if val > 0:
                    startup_cost = self.params.get(f'c_su_{u}', 50)
                    revenue_analysis['hydrogen']['startup_cost'] += val * startup_cost
        
        # ========== HEAT ==========
        # Export revenue
        if 'e_H_gri' in results:
            for (u, t), val in results['e_H_gri'].items():
                if val > 0:
                    price = self.params.get(f'pi_H_gri_export_{t}', 0)
                    revenue_analysis['heat']['grid_export_revenue'] += val * price
        
        # Import cost
        if 'i_H_gri' in results:
            for (u, t), val in results['i_H_gri'].items():
                if val > 0:
                    price = self.params.get(f'pi_H_gri_import_{t}', 0)
                    revenue_analysis['heat']['grid_import_cost'] += val * price
        
        # Heat pump production cost
        if 'p' in results:
            for (u, resource_type, t), val in results['p'].items():
                if val > 0 and resource_type == 'hp':
                    cost = self.params.get(f'c_hp_{u}', 0)
                    revenue_analysis['heat']['production_cost'] += val * cost
        
        # Heat storage cost
        if 'b_ch_H' in results:
            for (u, t), val in results['b_ch_H'].items():
                if val > 0:
                    revenue_analysis['heat']['storage_cost'] += val * c_sto * nu_ch
        
        if 'b_dis_H' in results:
            for (u, t), val in results['b_dis_H'].items():
                if val > 0:
                    revenue_analysis['heat']['storage_cost'] += val * c_sto * (1/nu_dis)
        
        # ========== COMMON COSTS ==========
        if 'chi_peak' in results:
            peak_penalty = self.params.get('pi_peak', 0)
            revenue_analysis['common']['peak_power_cost'] = results['chi_peak'] * peak_penalty
        
        # ========== CALCULATE NET VALUES ==========
        # Electricity net
        revenue_analysis['electricity']['net'] = (
            revenue_analysis['electricity']['grid_export_revenue'] -
            revenue_analysis['electricity']['grid_import_cost'] -
            revenue_analysis['electricity']['production_cost'] -
            revenue_analysis['electricity']['storage_cost']
        )
        
        # Hydrogen net
        revenue_analysis['hydrogen']['net'] = (
            revenue_analysis['hydrogen']['grid_export_revenue'] -
            revenue_analysis['hydrogen']['grid_import_cost'] -
            revenue_analysis['hydrogen']['production_cost'] -
            revenue_analysis['hydrogen']['storage_cost'] -
            revenue_analysis['hydrogen']['startup_cost']
        )
        
        # Heat net
        revenue_analysis['heat']['net'] = (
            revenue_analysis['heat']['grid_export_revenue'] -
            revenue_analysis['heat']['grid_import_cost'] -
            revenue_analysis['heat']['production_cost'] -
            revenue_analysis['heat']['storage_cost']
        )
        
        # Total calculations
        revenue_analysis['total_revenue'] = (
            revenue_analysis['electricity']['grid_export_revenue'] +
            revenue_analysis['hydrogen']['grid_export_revenue'] +
            revenue_analysis['heat']['grid_export_revenue']
        )
        
        revenue_analysis['total_cost'] = (
            revenue_analysis['electricity']['grid_import_cost'] +
            revenue_analysis['electricity']['production_cost'] +
            revenue_analysis['electricity']['storage_cost'] +
            revenue_analysis['hydrogen']['grid_import_cost'] +
            revenue_analysis['hydrogen']['production_cost'] +
            revenue_analysis['hydrogen']['storage_cost'] +
            revenue_analysis['hydrogen']['startup_cost'] +
            revenue_analysis['heat']['grid_import_cost'] +
            revenue_analysis['heat']['production_cost'] +
            revenue_analysis['heat']['storage_cost'] +
            revenue_analysis['common']['peak_power_cost']
        )
        
        revenue_analysis['net_profit'] = revenue_analysis['total_revenue'] - revenue_analysis['total_cost']
        
        # ========== PRINT SUMMARY ==========
        print("\n=== OBJECTIVE FUNCTION ANALYSIS BY ENERGY TYPE ===")
        
        print(f"\n[ELECTRICITY]")
        print(f"  Grid export revenue:  {revenue_analysis['electricity']['grid_export_revenue']:10.4f}")
        print(f"  Grid import cost:    -{revenue_analysis['electricity']['grid_import_cost']:10.4f}")
        print(f"  Production cost:     -{revenue_analysis['electricity']['production_cost']:10.4f}")
        print(f"  Storage cost:        -{revenue_analysis['electricity']['storage_cost']:10.4f}")
        print(f"  Net:                  {revenue_analysis['electricity']['net']:10.4f}")
        
        print(f"\n[HYDROGEN]")
        print(f"  Grid export revenue:  {revenue_analysis['hydrogen']['grid_export_revenue']:10.4f}")
        print(f"  Grid import cost:    -{revenue_analysis['hydrogen']['grid_import_cost']:10.4f}")
        print(f"  Production cost:     -{revenue_analysis['hydrogen']['production_cost']:10.4f}")
        print(f"  Storage cost:        -{revenue_analysis['hydrogen']['storage_cost']:10.4f}")
        print(f"  Startup cost:        -{revenue_analysis['hydrogen']['startup_cost']:10.4f}")
        print(f"  Net:                  {revenue_analysis['hydrogen']['net']:10.4f}")
        
        print(f"\n[HEAT]")
        print(f"  Grid export revenue:  {revenue_analysis['heat']['grid_export_revenue']:10.4f}")
        print(f"  Grid import cost:    -{revenue_analysis['heat']['grid_import_cost']:10.4f}")
        print(f"  Production cost:     -{revenue_analysis['heat']['production_cost']:10.4f}")
        print(f"  Storage cost:        -{revenue_analysis['heat']['storage_cost']:10.4f}")
        print(f"  Net:                  {revenue_analysis['heat']['net']:10.4f}")
        
        print(f"\n[COMMON]")
        print(f"  Peak power penalty:  -{revenue_analysis['common']['peak_power_cost']:10.4f}")
        
        print(f"\n[TOTAL]")
        print(f"  Total revenue:        {revenue_analysis['total_revenue']:10.4f}")
        print(f"  Total cost:          -{revenue_analysis['total_cost']:10.4f}")
        print(f"  Net profit:           {revenue_analysis['net_profit']:10.4f}")
        
        print(f"\n[VERIFICATION]")
        print(f"  Calculated profit:    {revenue_analysis['net_profit']:10.6f}")
        print(f"  Solver objective:     {-1*self.model.getObjVal():10.6f}")
        print(f"  Difference:           {abs(revenue_analysis['net_profit'] - (-1*self.model.getObjVal())):10.10f}")
        
        if abs(revenue_analysis['net_profit'] - (-1*self.model.getObjVal())) > 1e-6:
            print("  ⚠️  WARNING: Calculated profit doesn't match solver objective value!")
        else:
            print("  ✓  Verification passed!")
        
        return revenue_analysis
    def analyze_energy_flows(self, results):
        """
        Analyze energy flows including community internal conversions
        
        Args:
            results: Dictionary containing optimization results
            
        Returns:
            Dictionary with energy flow analysis
        """
        flow_analysis = {}
        
        for t in self.time_periods:
            flow_analysis[t] = {
                'elec_to_hydro': 0.0,
                'hydro_produced': 0.0,
                'elec_to_heat': 0.0,
                'heat_produced': 0.0,
                'elec_to_grid': 0.0,
                'elec_from_grid': 0.0,
                'elec_renewable': 0.0,
                'elec_storage_charge': 0.0,    # 추가: 전기 저장소 충전
                'elec_storage_discharge': 0.0,  # 추가: 전기 저장소 방전
                'elec_nfl_demand': 0.0,         # 추가: 비유연 전기 수요
                'elec_net_comm': 0.0,
                'hydro_to_grid': 0.0,
                'hydro_from_grid': 0.0,
                'heat_to_grid': 0.0,
                'heat_from_grid': 0.0,
            }
            
            # Electricity flows
            if 'e_E_gri' in results:
                for u in self.players:
                    if (u, t) in results['e_E_gri']:
                        flow_analysis[t]['elec_to_grid'] += results['e_E_gri'][u, t]
            
            if 'i_E_gri' in results:
                for u in self.players:
                    if (u, t) in results['i_E_gri']:
                        flow_analysis[t]['elec_from_grid'] += results['i_E_gri'][u, t]
            
            # Renewable generation
            if 'p' in results:
                for u in self.players_with_renewables:
                    if (u, 'res', t) in results['p']:
                        flow_analysis[t]['elec_renewable'] += results['p'][u, 'res', t]
            
            # Electrolyzer (Electricity → Hydrogen)
            if 'fl_d' in results:
                for u in self.players_with_electrolyzers:
                    if (u, 'elec', t) in results['fl_d']:
                        flow_analysis[t]['elec_to_hydro'] += results['fl_d'][u, 'elec', t]
            
            if 'p' in results:
                for u in self.players_with_electrolyzers:
                    if (u, 'els', t) in results['p']:
                        flow_analysis[t]['hydro_produced'] += results['p'][u, 'els', t]
            
            # Heat pump (Electricity → Heat)
            if 'fl_d' in results:
                for u in self.players_with_heatpumps:
                    if (u, 'elec', t) in results['fl_d']:
                        flow_analysis[t]['elec_to_heat'] += results['fl_d'][u, 'elec', t]
            
            if 'p' in results:
                for u in self.players_with_heatpumps:
                    if (u, 'hp', t) in results['p']:
                        flow_analysis[t]['heat_produced'] += results['p'][u, 'hp', t]
            
            # Hydrogen flows
            if 'e_G_gri' in results:
                for u in self.players:
                    if (u, t) in results['e_G_gri']:
                        flow_analysis[t]['hydro_to_grid'] += results['e_G_gri'][u, t]
            
            if 'i_G_gri' in results:
                for u in self.players:
                    if (u, t) in results['i_G_gri']:
                        flow_analysis[t]['hydro_from_grid'] += results['i_G_gri'][u, t]
            
            # Heat flows
            if 'e_H_gri' in results:
                for u in self.players:
                    if (u, t) in results['e_H_gri']:
                        flow_analysis[t]['heat_to_grid'] += results['e_H_gri'][u, t]
            
            if 'i_H_gri' in results:
                for u in self.players:
                    if (u, t) in results['i_H_gri']:
                        flow_analysis[t]['heat_from_grid'] += results['i_H_gri'][u, t]
            
# Electricity storage
        if 'b_ch_E' in results:
            for u in self.players:
                if (u, t) in results['b_ch_E']:
                    flow_analysis[t]['elec_storage_charge'] += results['b_ch_E'][u, t]
        
        if 'b_dis_E' in results:
            for u in self.players:
                if (u, t) in results['b_dis_E']:
                    flow_analysis[t]['elec_storage_discharge'] += results['b_dis_E'][u, t]
        
        # Non-flexible electricity demand
        if 'nfl_d' in results:
            for u in self.players:
                if (u, 'elec', t) in results['nfl_d']:
                    flow_analysis[t]['elec_nfl_demand'] += results['nfl_d'][u, 'elec', t]
        
        # Recalculate Net Community with complete balance
        flow_analysis[t]['elec_net_comm'] = (
            flow_analysis[t]['elec_renewable'] +
            flow_analysis[t]['elec_from_grid'] +
            flow_analysis[t]['elec_storage_discharge'] -
            flow_analysis[t]['elec_to_grid'] -
            flow_analysis[t]['elec_to_hydro'] -
            flow_analysis[t]['elec_to_heat'] -
            flow_analysis[t]['elec_storage_charge'] -
            flow_analysis[t]['elec_nfl_demand']
        )
        
        # Print analysis
        self._print_energy_flow_analysis(flow_analysis)
        
        return flow_analysis

    def _print_energy_flow_analysis(self, flow_analysis):
        """
        Print energy flow analysis in a formatted table
        """
        print("\n" + "="*130)
        print("ENERGY FLOW ANALYSIS BY TIME PERIOD")
        print("="*130)
        
        # Header - 포맷 수정
        print(f"{'Time':^4} | {'='*27} ELECTRICITY {'='*27} | {'='*5} HYDROGEN {'='*5} | {'='*7} HEAT {'='*7}")
        print(f"{'':^4} | {'Renewable':^10} {'From Grid':^10} {'To Grid':^10} {'→Hydro':^10} {'→Heat':^10} {'Net Comm':^10} | {'Produced':^10} {'To/From':^10} | {'Produced':^10} {'To/From':^10}")
        print("-"*130)
        
        # Data rows - 포맷 조정
        for t in self.time_periods:
            flow = flow_analysis[t]
            
            # Calculate net flows for display
            hydro_net = flow['hydro_to_grid'] - flow['hydro_from_grid']
            hydro_net_str = f"{hydro_net:+10.2f}" if hydro_net != 0 else "      0.00"
            
            heat_net = flow['heat_to_grid'] - flow['heat_from_grid']
            heat_net_str = f"{heat_net:+10.2f}" if heat_net != 0 else "      0.00"
            
            print(f"{t:^4} | "
                f"{flow['elec_renewable']:^10.2f} "
                f"{flow['elec_from_grid']:^10.2f} "
                f"{flow['elec_to_grid']:^10.2f} "
                f"{flow['elec_to_hydro']:^10.2f} "
                f"{flow['elec_to_heat']:^10.2f} "
                f"{flow['elec_net_comm']:^10.2f} | "
                f"{flow['hydro_produced']:^10.2f} "
                f"{hydro_net_str} | "
                f"{flow['heat_produced']:^10.2f} "
                f"{heat_net_str}")
        
        # Summary statistics
        print("-"*120)
        print("\nSUMMARY STATISTICS:")
        
        total_renewable = sum(f['elec_renewable'] for f in flow_analysis.values())
        total_elec_import = sum(f['elec_from_grid'] for f in flow_analysis.values())
        total_elec_export = sum(f['elec_to_grid'] for f in flow_analysis.values())
        total_elec_to_hydro = sum(f['elec_to_hydro'] for f in flow_analysis.values())
        total_elec_to_heat = sum(f['elec_to_heat'] for f in flow_analysis.values())
        total_hydro_produced = sum(f['hydro_produced'] for f in flow_analysis.values())
        total_heat_produced = sum(f['heat_produced'] for f in flow_analysis.values())
        
        print(f"\nElectricity:")
        print(f"  Total renewable generated:     {total_renewable:10.2f} kWh")
        print(f"  Total imported from grid:      {total_elec_import:10.2f} kWh")
        print(f"  Total exported to grid:        {total_elec_export:10.2f} kWh")
        print(f"  Total used for hydrogen:       {total_elec_to_hydro:10.2f} kWh")
        print(f"  Total used for heat:           {total_elec_to_heat:10.2f} kWh")
        print(f"  Net grid position:             {total_elec_import - total_elec_export:+10.2f} kWh (+ = import, - = export)")
        
        print(f"\nHydrogen:")
        print(f"  Total produced:                {total_hydro_produced:10.2f} kg")
        print(f"  Conversion efficiency:         {total_hydro_produced/total_elec_to_hydro*100 if total_elec_to_hydro > 0 else 0:10.2f} %")
        
        print(f"\nHeat:")
        print(f"  Total produced:                {total_heat_produced:10.2f} kWh")
        print(f"  COP (efficiency):              {total_heat_produced/total_elec_to_heat if total_elec_to_heat > 0 else 0:10.2f}")
        
        # Check for community internal hydrogen trading
        hydro_community_trade = self._check_community_hydrogen_trade(flow_analysis)
        if hydro_community_trade > 0:
            print(f"\n⚠️  Community hydrogen trade detected: {hydro_community_trade:.2f} kg")
        else:
            print(f"\n❌ No community hydrogen trading detected (all hydrogen goes through grid)")

    def _check_community_hydrogen_trade(self, results):
        """
        Check if there's any hydrogen trading within the community
        """
        total_community_hydro = 0.0
        
        if 'e_G_com' in results:
            for key, val in results['e_G_com'].items():
                total_community_hydro += val
        
        if 'i_G_com' in results:
            for key, val in results['i_G_com'].items():
                total_community_hydro += val
        
        return total_community_hydro
    def solve_with_restricted_pricing(self):
        """
        Solve with Restricted Pricing mechanism:
        1. First solve MILP to get optimal binary variables
        2. Fix binary variables and solve LP to get shadow prices
        
        Returns:
            tuple: (status, results, prices)
                - status: optimization status
                - results: optimization results dictionary
                - prices: dictionary with electricity, heat, hydro prices per time period
        """
        
        print("Step 1: Solving MILP to get optimal commitment decisions...")
        
        # Step 1: Solve original MILP
        # status = self.solve()
        
        # if status != "optimal":
        #     print(f"MILP optimization failed with status: {status}")
        #     return status, None, None
        
        # print("MILP solved successfully. Extracting binary variable values...")
        # Extract optimal binary variable values
        from pyscipopt import SCIP_PARAMSETTING
        self.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.model.setHeuristics(SCIP_PARAMSETTING.OFF)
        self.model.disablePropagation()
        self.model.optimize()
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
        # Step 2: Create new LP model with fixed binary variables
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                var = self.model.data["vars"]["z_su"][(u,t)]
                # self.model.fixVar(var, binary_values["z_su", u, t])
                var = self.model.data["vars"]["z_on"][(u,t)]
                # self.model.fixVar(var, binary_values["z_on", u, t])
                var = self.model.data["vars"]["z_off"][(u,t)]
                # self.model.fixVar(var, binary_values["z_off", u, t])
                # self.model.chgVarUb(var, binary_values["z_off", u, t])
                var = self.model.data["vars"]["z_sb"][(u,t)]
                # self.model.fixVar(var, binary_values["z_sb", u, t])
                # self.model.chgVarUb(var, binary_values["z_sb", u, t])
        
        self.model.optimize()
        # # if status != "optimal":
        #     print(f"LP optimization failed with status: {status}")
        #     return status, None, None
        
        print("LP solved successfully. Extracting shadow prices...")
        
        # Step 3: Extract shadow prices (dual multipliers) from community balance constraints
        prices = {
            'electricity': {},
            'heat': {},
            'hydro': {}
        }
        
        for t in self.time_periods:
            # Get dual multipliers for community balance constraints
            # Note: Must use getTransformedCons() to get transformed constraints for dual solution
            t_cons = self.model.getTransformedCons(self.community_elec_balance_cons[f"community_elec_balance_{t}"])
            prices['electricity'][t] = self.model.getDualsolLinear(t_cons)
            t_cons = self.model.getTransformedCons(self.community_heat_balance_cons[f"community_heat_balance_{t}"])
            prices['heat'][t] = self.model.getDualsolLinear(t_cons)
            t_cons = self.model.getTransformedCons(self.community_hydro_balance_cons[f"community_hydro_balance_{t}"])
            prices['hydro'][t] = self.model.getDualsolLinear(t_cons)
            



        
        
        print("Step 2: Creating LP relaxation with fixed binary variables...")
        self.model.freeTransform()
        self.model.relax()


        print("Step 3: Solving LP relaxation...")
        
        # Solve LP
        from pyscipopt import SCIP_PARAMSETTING
        self.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.model.setHeuristics(SCIP_PARAMSETTING.OFF)
        self.model.disablePropagation()
        self.model.optimize()
        # if status != "optimal":
        #     print(f"LP optimization failed with status: {status}")
        #     return status, None, None
        
        print("LP solved successfully. Extracting shadow prices...")
        
        # Step 3: Extract shadow prices (dual multipliers) from community balance constraints
        prices = {
            'electricity': {},
            'heat': {},
            'hydro': {}
        }
        
        for t in self.time_periods:
            # Get dual multipliers for community balance constraints
            # Note: Must use getTransformedCons() to get transformed constraints for dual solution
            t_cons = self.model.getTransformedCons(self.community_elec_balance_cons[f"community_elec_balance_{t}"])
            prices['electricity'][t] = self.model.getDualsolLinear(t_cons)
            t_cons = self.model.getTransformedCons(self.community_heat_balance_cons[f"community_heat_balance_{t}"])
            prices['heat'][t] = self.model.getDualsolLinear(t_cons)
            t_cons = self.model.getTransformedCons(self.community_hydro_balance_cons[f"community_hydro_balance_{t}"])
            prices['hydro'][t] = self.model.getDualsolLinear(t_cons)
    
        # Get LP results
        lp_results = self._extract_lp_results(self.model)
        
        print("Restricted Pricing completed successfully!")
        print(f"Electricity prices: {prices['electricity']}")
        print(f"Heat prices: {prices['heat']}")
        print(f"hydro prices: {prices['hydro']}")
        
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
                
                # hydro variables
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
        lp_community_hydro_cons = {}
        
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
            lp_community_elec_cons[f"community_elec_balance_{t}"] = cons
        
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
            lp_community_heat_cons[f"community_heat_balance_{t}"] = cons
        
        # hydro constraints
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
                
                lp_model.addCons(lhs == rhs, name=f"hydro_balance_{u}_{t}")
        
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
        
        # hydro storage SOC constraints
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
        
        # Community hydro balance (IMPORTANT: store constraint reference for dual)
        for t in self.time_periods:
            community_hydro_balance = quicksum(self.lp_e_G_com[u,t] - self.lp_i_G_com[u,t] for u in self.players)
            cons = lp_model.addCons(community_hydro_balance == 0, name=f"community_hydro_balance_{t}")
            lp_community_hydro_cons[f"community_hydro_balance_{t}"] = cons
        
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
        
        return lp_community_elec_cons, lp_community_heat_cons, lp_community_hydro_cons
    
    def _extract_lp_results(self, lp_model):
        """Extract results from LP model"""
        
        results = {
            'objective_value': lp_model.getObjVal(),
            'electricity': {},
            'heat': {},
            'hydro': {},
            'storage': {},
            'production': {},
            'peak_power': lp_model.getVal(self.chi_peak)
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
                
                # hydro results
                results['hydro'][u,t] = {
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
            'hydro': {},
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
                
                # hydro results
                results['hydro'][u,t] = {
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
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))  # 24 hours
    
    # Example parameters with proper bounds and storage types
    parameters = {
        'players_with_renewables': ['u1'],
        'players_with_electrolyzers': ['u2'],
        'players_with_heatpumps': ['u3'],
        'players_with_elec_storage': ['u1'],
        'players_with_hydro_storage': ['u2'],
        'players_with_heat_storage': ['u3'],
        'players_with_nfl_elec_demand': ['u4'],
        'players_with_nfl_hydro_demand': ['u5'],
        'players_with_nfl_heat_demand': ['u6'],
        'players_with_fl_elec_demand': ['u2'],  # u2 needs electricity to run electrolyzer
        'players_with_fl_hydro_demand': [],
        'players_with_fl_heat_demand': [],
        'nu_ch': 0.9,
        'nu_dis': 0.9,
        'pi_peak': 100,
        
        # Storage parameters (common for all types)
        'storage_power': 50,        # 50 kW power rating
        'storage_capacity': 200,    # 200 kWh capacity
        'initial_soc': 0.0,         # Initial 100 kWh
        
        # Equipment capacities
        'renewable_cap_u1': 150,    # 150 kW solar
        'hp_cap_u3': 80,           # 80 kW thermal heat pump
        'els_cap_u2': 100,         # 100 kg/day electrolyzer
        
        # Grid connection limits
        'e_E_cap_u1_t': 100,       # 100 kW export limit
        'i_E_cap_u1_t': 150,       # 150 kW import limit
        'e_H_cap_u3_t': 60,        # 60 kW heat export
        'i_H_cap_u3_t': 80,        # 80 kW heat import
        'e_G_cap_u2_t': 50,        # 50 kg/day hydro export
        'i_G_cap_u2_t': 30,        # 30 kg/day hydro import
        
        # Cost parameters
        'c_sto': 0.01,             # Common storage cost
        
        # Electrolyzer parameters
        'C_max_u2': 100,           # Maximum capacity
        'C_min_u2': 20,            # Minimum capacity
        'C_sb_u2': 10,             # Standby capacity
        'phi1_u2': 0.7,            # Electrolyzer efficiency parameter
        'phi0_u2': 0.0,            # Electrolyzer efficiency parameter
    }
    parameters['players_with_fl_elec_demand'] = list(set(parameters['players_with_electrolyzers'] + parameters['players_with_heatpumps']))
    # Add demand data - Increased demand to encourage community trading
    for u in players:
        for t in time_periods:
            parameters[f'd_E_nfl_{u}_{t}'] = 30 + 15 * np.sin(2 * np.pi * t / 24)  # Increased: 15~45 kW
            parameters[f'd_H_nfl_{u}_{t}'] = 20 + 10 * np.sin(2 * np.pi * t / 24)  # Increased: 10~30 kW
            parameters[f'd_G_nfl_{u}_{t}'] = 8 + 4 * np.sin(2 * np.pi * t / 24)    # Increased: 4~12 kg/day
    
    # Add cost parameters
    for u in players:
        parameters[f'c_res_{u}'] = 0.05
        parameters[f'c_hp_{u}'] = 0.1
        parameters[f'c_els_{u}'] = 0.08
        parameters[f'c_su_{u}'] = 50
        parameters[f'c_sto_{u}'] = 0.01
    
    # Add grid prices - Grid import is 0.1% more expensive than export to encourage community trading
    for t in time_periods:
        base_price_E = 0.5 + 0.2 * np.sin(2 * np.pi * t / 24)  # Base electricity price: 0.3~0.7
        base_price_H = 0.4  # Base heat price
        base_price_G = 1.0  # Base hydrogen price
        
        # Export prices (negative in objective function)
        parameters[f'pi_E_gri_export_{t}'] = base_price_E
        parameters[f'pi_H_gri_export_{t}'] = base_price_H
        parameters[f'pi_G_gri_export_{t}'] = base_price_G
        
        # Import prices (positive in objective function) - 0.1% more expensive
        parameters[f'pi_E_gri_import_{t}'] = base_price_E * 1.001
        parameters[f'pi_H_gri_import_{t}'] = base_price_H * 1.001
        parameters[f'pi_G_gri_import_{t}'] = base_price_G * 1.001
    
    # Create and solve model with Restricted Pricing
    lem = LocalEnergyMarket(players, time_periods, parameters, isLP=False)
    
    # First solve complete model and analyze revenue
    print("\n" + "="*60)
    print("SOLVING COMPLETE MODEL AND ANALYZING REVENUE")
    print("="*60)
    status_complete, results_complete, revenue_analysis = lem.solve_complete_model()
    
    if status_complete == "optimal":
        print(f"\nComplete model solved successfully!")
        print(f"Objective value: {results_complete.get('objective_value', 'N/A')}")
    else:
        print(f"\nComplete model failed with status: {status_complete}")
    
    print("\n" + "="*60)
    print("SOLVING WITH RESTRICTED PRICING")
    print("="*60)
    
    # Solve using Restricted Pricing
    status, results, prices = lem.solve_with_restricted_pricing()
    
    if status == "optimal":
        print(f"Optimal objective value: {results['objective_value']:.2f}")
        print(f"Peak power: {results['peak_power']:.2f}")
        print("\n=== RESTRICTED PRICING RESULTS ===")
        print(f"Electricity prices: {prices['electricity']}")
        print(f"Heat prices: {prices['heat']}")
        print(f"hydro prices: {prices['hydro']}")
    else:
        print(f"Optimization failed with status: {status}")

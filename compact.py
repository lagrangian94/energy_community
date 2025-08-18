from pyscipopt import Model, quicksum
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd

# CSV 파일 읽기 (인코딩 처리)
def load_korean_electricity_prices():
    """한국 전력가격 데이터 로드 및 처리"""
    
    # CSV 읽기 (cp949 또는 euc-kr 인코딩 사용)
    df = pd.read_csv('./data/kor_elec_grid_price.csv', encoding='cp949')
    
    # 컬럼명 정리 (한글 깨짐 수정)
    column_mapping = {
        df.columns[0]: 'period',  # 기간
        **{df.columns[i]: f'{i:02d}시' for i in range(1, 25)},  # 01시 ~ 24시
        df.columns[25]: 'max',     # 최대
        df.columns[26]: 'min',     # 최소
        df.columns[27]: 'weighted_avg'  # 가중평균
    }
    df = df.rename(columns=column_mapping)
    
    # 시간대별 평균 가격 계산 (전체 기간)
    hourly_avg_prices = []
    for hour in range(1, 25):
        col_name = f'{hour:02d}시'
        # 음수인 값은 0으로 대체 후 평균 계산
        # avg_price = df[col_name].mask(df[col_name] < 0, 0).mean()
        avg_price = df[col_name].mean()
        hourly_avg_prices.append(avg_price)
    
    # 특정 날짜 선택 (예: 중간값에 가까운 날)
    median_idx = len(df) // 2
    selected_day_prices = []
    for hour in range(1, 25):
        col_name = f'{hour:02d}시'
        price = df.iloc[median_idx][col_name]
        if price < 0:
            price = 0
        price = df.iloc[median_idx][col_name]
        selected_day_prices.append(price)
    
    return hourly_avg_prices, selected_day_prices
# 수소 가격 설정 - 논문 참고 (동적 가격)
def calculate_hydrogen_prices(elec_prices_eur):
    """
    논문 기반 수소 가격 계산
    - 논문: 고정 €2.1/kg
    - 여기서는 전력 가격에 반비례하도록 동적 설정
    """
    h2_prices = []
    
    # 전력 가격 평균 및 범위 계산
    avg_elec = np.mean(elec_prices_eur)
    min_elec = min(elec_prices_eur)
    max_elec = max(elec_prices_eur)
    
    for elec_price in elec_prices_eur:
        # 기본 수소 가격: €2.1/kg
        """
        https://doi.org/10.1016/j.compchemeng.2023.108450
        근데, 논문에선 elec price range가 25-60 정도임. 우리 smp는 euro/MWh로 계산시 60-90 사이. 그래서 수소가격도 2배 해줌 일단.
        """
        base_h2_price = 2.1 *2
        
        # 전력 가격에 반비례하는 조정 계수
        # 전력이 싸면 수소 가격 낮춤 (수소 생산 유도)
        # 전력이 비싸면 수소 가격 높임 (전력 판매 유도)
        if avg_elec > 0:
            adjustment = 1.0 - 0.3 * (elec_price - avg_elec) / avg_elec
        else:
            adjustment = 1.0
        
        # 최종 수소 가격 (€1.5 ~ €3.5/kg 범위)
        h2_price = base_h2_price * adjustment
        h2_price = np.clip(h2_price, 1.5, 3.5)
        
        h2_prices.append(h2_price)
    
    return h2_prices
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
                        "b_dis_H", "b_ch_H", "s_H", "z_su", "z_on", "z_off", "z_sb"]
                        # "chi_peak"]
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
                 isLP: bool = False,
                 dwr: bool = False):
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
        self.dwr = dwr
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
        self.els_d = {}  # Electrolyzer power consumption at 'ON' state
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
        # self.chi_peak = self.model.addVar(vtype="C", name="chi_peak", lb=0, obj=self.params.get('pi_peak', 0))
        
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
                    renewable_cap = self.params.get(f'renewable_cap_{u}_{t}', 200)  # Default 200 kW, now time-dependent
                    c_res = self.params.get(f'c_res_{u}', 0)
                    self.p[u,'res',t] = self.model.addVar(vtype="C", name=f"p_res_{u}_{t}", 
                                                        lb=0, ub=renewable_cap, obj=c_res)
                if u in self.players_with_heatpumps:  # Heat pumps
                    hp_cap = self.params.get(f'hp_cap_{u}', 100)  # Default 100 kW thermal
                    c_hp = self.params.get(f'c_hp_{u}', 0)
                    self.p[u,'hp',t] = self.model.addVar(vtype="C", name=f"p_hp_{u}_{t}", 
                                                       lb=0, ub=hp_cap, obj=c_hp)

                if u in self.players_with_electrolyzers:  # Electrolyzers
                    els_cap = self.params.get(f'els_cap_{u}', 1)  # Default 1 MW
                    c_els = self.params.get(f'c_els_{u}', 0)
                    self.p[u,'els',t] = self.model.addVar(vtype="C", name=f"p_els_{u}_{t}", 
                                                        lb=0, obj=c_els)
                    
                    # Electrolyzer commitment variables
                    c_su = self.params.get(f'c_su_{u}', 0)
                    vartype = "C" if isLP else "B"
                    self.z_su[u,t] = self.model.addVar(vtype=vartype, name=f"z_su_{u}_{t}", obj=c_su)
                    self.z_on[u,t] = self.model.addVar(vtype=vartype, name=f"z_on_{u}_{t}")
                    self.z_off[u,t] = self.model.addVar(vtype=vartype, name=f"z_off_{u}_{t}")
                    self.z_sb[u,t] = self.model.addVar(vtype=vartype, name=f"z_sb_{u}_{t}")
            
                    self.els_d[u,t] = self.model.addVar(vtype="C", name=f"els_d_{u}_{t}", 
                                                        lb=0, ub=els_cap)
                # Non-flexible demand variables
                if u in self.players_with_nfl_elec_demand:
                    nfl_elec_demand_t = self.params.get(f'd_E_nfl_{u}_{t}', 0)
                    res_capacity = 2
                    self.nfl_d[u,'elec',t] = self.model.addVar(vtype="C", name=f"d_elec_{u}_{t}", 
                                                       lb=nfl_elec_demand_t, ub=nfl_elec_demand_t)
                    self.i_E_gri[u,t] = self.model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_E_cap', 0.5) * res_capacity, obj=self.params.get(f'pi_E_gri_import_{t}', 0))
                    self.i_E_com[u,t] = self.model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0, ub=self.params.get(f'i_E_cap', 0.5) * res_capacity)
                if u in self.players_with_nfl_hydro_demand:
                    nfl_hydro_demand_t = self.params.get(f'd_G_nfl_{u}_{t}', 0)
                    self.nfl_d[u,'hydro',t] = self.model.addVar(vtype="C", name=f"d_hydro_{u}_{t}", 
                                                       lb=nfl_hydro_demand_t, ub=nfl_hydro_demand_t)
                    self.i_G_gri[u,t] = self.model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_G_cap', 100), obj=self.params.get(f'pi_G_gri_import_{t}', 0))
                    self.i_G_com[u,t] = self.model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0, ub=100)
                if u in self.players_with_nfl_heat_demand:
                    nfl_heat_demand_t = self.params.get(f'd_H_nfl_{u}_{t}', 0)
                    self.nfl_d[u,'heat',t] = self.model.addVar(vtype="C", name=f"d_heat_{u}_{t}", 
                                                       lb=nfl_heat_demand_t, ub=nfl_heat_demand_t)
                    self.i_H_gri[u,t] = self.model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_H_cap', 500), obj=self.params.get(f'pi_H_gri_import_{t}', 0))
                    self.i_H_com[u,t] = self.model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0, ub=500)
                
                # Flexible demand variables
                if u in self.players_with_fl_elec_demand:
                    fl_elec_demand_cap = 1 ## Total electrolyzer power consumption capacity [MW]
                    self.fl_d[u,'elec',t] = self.model.addVar(vtype="C", name=f"d_elec_{u}_{t}", 
                                                       lb=0.0, ub=fl_elec_demand_cap)
                    self.i_E_gri[u,t] = self.model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_E_cap', 1000), obj=self.params.get(f'pi_E_gri_import_{t}', 0))
                    self.i_E_com[u,t] = self.model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0, ub=1000)
                if u in self.players_with_fl_hydro_demand:
                    fl_hydro_demand_cap = 10**6
                    self.fl_d[u,'hydro',t] = self.model.addVar(vtype="C", name=f"d_hydro_{u}_{t}", 
                                                       lb=0.0, ub=fl_hydro_demand_cap)
                    self.i_G_gri[u,t] = self.model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_G_cap', 100), obj=self.params.get(f'pi_G_gri_import_{t}', 0))
                    self.i_G_com[u,t] = self.model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0, ub=100)
                if u in self.players_with_fl_heat_demand:
                    fl_heat_demand_cap = 10**6
                    self.fl_d[u,'heat',t] = self.model.addVar(vtype="C", name=f"d_heat_{u}_{t}", 
                                                       lb=0.0, ub=fl_heat_demand_cap)
                    self.i_H_gri[u,t] = self.model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_H_cap', 500), obj=self.params.get(f'pi_H_gri_import_{t}', 0))
                    self.i_H_com[u,t] = self.model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0, ub=500)
                # Storage variables by type with capacity constraints
                storage_power = self.params.get(f'storage_power', 50)  # kW power rating
                storage_capacity = self.params.get(f'storage_capacity', 100)  # kWh capacity
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
        
        # self.model.data["vars"]["chi_peak"] = self.chi_peak
        
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
        # self.model.data["cons"]["peak_power"] = self.peak_power_cons
    
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
        
        # Constraint (6): Electricity storage SOC transition with special 23→0 transition
        for u in self.players_with_elec_storage:
            nu_ch = self.params.get('nu_ch', 0.9)
            nu_dis = self.params.get('nu_dis', 0.9)
                    
            # Set initial SOC at 6시 (논리적 시작점)
            if (u,6) in self.s_E:
                initial_soc = self.params.get(f'initial_soc', 1)  # Default 50% SOC
                cons = self.model.addCons(self.s_E[u,6] == initial_soc, name=f"initial_soc_E_{u}")
                self.storage_cons[f"initial_soc_E_{u}"] = cons
            
            # 일반적인 SOC transition (1시~23시)
            for t in range(1, 24):
                if (u,t) in self.s_E and (u,t-1) in self.s_E:
                    cons = self.model.addCons(
                        self.s_E[u,t] == self.s_E[u,t-1] + nu_ch * self.b_ch_E[u,t] - (1/nu_dis) * self.b_dis_E[u,t],
                        name=f"soc_transition_E_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_E_{u}_{t}"] = cons
        
            # 특별한 23→0시 transition (심야 충전 → 아침 방전)
            if (u,23) in self.s_E and (u,0) in self.s_E:
                cons = self.model.addCons(
                    self.s_E[u,0] == self.s_E[u,23] + nu_ch * self.b_ch_E[u,0] - (1/nu_dis) * self.b_dis_E[u,0],
                    name=f"soc_transition_E_{u}_23_to_0"
                )
                self.storage_cons[f"soc_transition_E_{u}_23_to_0"] = cons
        
        if not self.dwr:
            # Constraint (9): Community electricity balance
            for t in self.time_periods:
                community_balance = quicksum(self.i_E_com.get((u,t),0) - self.e_E_com.get((u,t),0) for u in self.players)
                cons = self.model.addCons(community_balance == 0, name=f"community_elec_balance_{t}")
                self.community_elec_balance_cons[f"community_elec_balance_{t}"] = cons
            
                # # Constraint (10): Peak power constraint
                # for t in self.time_periods:
                #     grid_import = quicksum(self.i_E_gri.get((u,t),0) - self.e_E_gri.get((u,t),0) for u in self.players)
                #     cons = self.model.addCons(grid_import <= self.chi_peak, name=f"peak_power_{t}")
                #     self.peak_power_cons[f"peak_power_{t}"] = cons
    
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
        for u in self.players:
            if u in self.players_with_heatpumps:
                for t in self.time_periods:
                    nu_COP = self.params.get(f'nu_COP_{u}', 3.0)
                    cons = self.model.addCons(
                        nu_COP * self.fl_d.get((u,'elec',t),0) == self.p.get((u,'hp',t),0),
                        name=f"heatpump_coupling_{u}_{t}"
                    )
                    self.heatpump_cons[f"heatpump_coupling_{u}_{t}"] = cons
        
        # Heat storage SOC transition with special 23→0 transition
        for u in self.players:
            if u in self.players_with_heat_storage:
                nu_ch = self.params.get('nu_ch', 0.9)
                nu_dis = self.params.get('nu_dis', 0.9)
                    
                # Set initial SOC at 6시 (논리적 시작점)
                if (u,6) in self.s_H:
                    initial_soc = self.params.get(f'initial_soc', 50)
                    cons = self.model.addCons(self.s_H[u,6] == initial_soc, name=f"initial_soc_H_{u}")
                    self.storage_cons[f"initial_soc_H_{u}"] = cons
                
                # 일반적인 SOC transition (1시~23시)
                for t in range(1, 24):
                    if (u,t) in self.s_H and (u,t-1) in self.s_H:
                        cons = self.model.addCons(
                            self.s_H[u,t] == self.s_H[u,t-1] + nu_ch * self.b_ch_H[u,t] - (1/nu_dis) * self.b_dis_H[u,t],
                            name=f"soc_transition_H_{u}_{t}"
                        )
                    self.storage_cons[f"soc_transition_H_{u}_{t}"] = cons
        
                # 특별한 23→0시 transition (심야 충전 → 아침 방전)
                if (u,23) in self.s_H and (u,0) in self.s_H:
                    cons = self.model.addCons(
                        self.s_H[u,0] == self.s_H[u,23] + nu_ch * self.b_ch_H[u,0] - (1/nu_dis) * self.b_dis_H[u,0],
                        name=f"soc_transition_H_{u}_23_to_0"
                    )
                    self.storage_cons[f"soc_transition_H_{u}_23_to_0"] = cons
        if not self.dwr:
            # Community heat balance
            for t in self.time_periods:
                community_heat_balance = quicksum(self.i_H_com.get((u,t),0) - self.e_H_com.get((u,t),0) for u in self.players)
                cons = self.model.addCons(community_heat_balance == 0, name=f"community_heat_balance_{t}")
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
        for u in self.players:
            if u in self.players_with_electrolyzers:
                els_cap = self.params.get(f'els_cap_{u}', 1)
                C_sb = self.params.get(f'C_sb_{u}', 0.01)
                C_min = self.params.get(f'C_min_{u}', 0.15)
                for t in self.time_periods:
                    phi1_1 = self.params.get(f'phi1_1_{u}', 21.12266316)
                    phi0_1 = self.params.get(f'phi0_1_{u}', -0.37924094)
                    phi1_2 = self.params.get(f'phi1_2_{u}', 16.66883134)
                    phi0_2 = self.params.get(f'phi0_2_{u}', 0.87814262)
                
                    cons = self.model.addCons(
                        self.p.get((u,'els',t),0) <= phi1_1 * self.els_d.get((u,t),0) + phi0_1 * self.z_on[u,t],
                        name=f"electrolyzer_production_curve_1_{u}_{t}"
                    )
                    self.electrolyzer_cons['production_curve_1', u, t] = cons
                    cons = self.model.addCons(
                        self.p.get((u,'els',t),0) <= phi1_2 * self.els_d.get((u,t),0) + phi0_2 * self.z_on[u,t],
                        name=f"electrolyzer_production_curve_2_{u}_{t}"
                    )
                    self.electrolyzer_cons['production_curve_2', u, t] = cons

                    cons = self.model.addCons(
                        self.els_d.get((u,t),0) - C_min * els_cap * self.z_on[u,t] >= 0.0,
                        name=f"electrolyzer_min_power_consumption_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"electrolyzer_min_power_consumption_{u}_{t}"] = cons
                    cons = self.model.addCons(
                        self.els_d.get((u,t),0) - els_cap * self.z_on[u,t] <= 0.0,
                        name=f"electrolyzer_max_power_consumption_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"electrolyzer_max_power_consumption_{u}_{t}"] = cons
                    
                    # Constraint   : Power consumption coupling
                    cons = self.model.addCons(
                        self.fl_d[u,'elec',t] == self.els_d[u,t] + C_sb * els_cap * self.z_sb[u,t],
                        name=f"electrolyzer_power_consumption_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"electrolyzer_power_consumption_{u}_{t}"] = cons
        # Electrolyzer commitment constraints (constraints 17-21)
            if u in self.players_with_electrolyzers:
                for t in self.time_periods:
                    # Constraint 17: exactly one state
                    cons = self.model.addCons(
                        self.z_on[u,t] + self.z_off[u,t] + self.z_sb[u,t] == 1,
                        name=f"electrolyzer_state_{u}_{t}"
                    )
                    self.electrolyzer_cons['state', u, t] = cons
                    
                    # Constraints 18-19: production bounds
                    els_cap = self.params.get(f'els_cap_u2', 1)
                    C_sb = self.params.get(f'C_sb_{u}', 0.01)
                    C_min = self.params.get(f'C_min_{u}', 0.15)
                    
                    cons = self.model.addCons(
                            self.fl_d[u,'elec',t] <= els_cap * self.z_on[u,t] + C_sb * els_cap * self.z_sb[u,t],
                        name=f"electrolyzer_max_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"electrolyzer_max_{u}_{t}"] = cons
                    
                    cons = self.model.addCons(
                            self.fl_d[u,'elec',t] >= C_min * els_cap * self.z_on[u,t] + C_sb * els_cap * self.z_sb[u,t],
                        name=f"electrolyzer_min_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"electrolyzer_min_{u}_{t}"] = cons
                    
                    # Constraint 20: startup logic
                    if t != 6:
                        if t != 0:
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
                                    self.z_su[u,t] >= self.z_on[u,t] - self.z_on[u,23] - self.z_sb[u,t],
                                    name=f"electrolyzer_startup_23_to_0_{u}_{t}"
                                )
                            self.electrolyzer_cons[f"electrolyzer_startup_23_to_0_{u}_{t}"] = cons
                            cons = self.model.addCons(
                                    self.z_off[u,23] + self.z_sb[u,t] <= 1.0,
                                    name=f"electrolyzer_forbid_off_to_sb_23_to_0_{u}_{t}"
                                )
                            self.electrolyzer_cons[f"electrolyzer_forbid_off_to_sb_23_to_0_{u}_{t}"] = cons
                    else:
                        cons = self.model.addCons(
                            self.z_off[u,t] >= 1.0,
                            name=f"electrolyzer_initial_state_{u}_{t}"
                        )
                        self.electrolyzer_cons[f"electrolyzer_initial_state_{u}_{t}"] = cons
                        cons = self.model.addCons(
                            self.z_su[u,t] <= 0.0,
                            name=f"electrolyzer_initial_su_{u}_{t}"
                        )
                        self.electrolyzer_cons[f"electrolyzer_initial_su_{u}_{t}"] = cons
        # # Maximum up time constraints (최대 연속 운전 시간)
        # for u in self.players_with_electrolyzers:
        #     max_up = self.params.get('max_up_time', 4)
            
        #     for t in range(max_up, len(self.time_periods)):
        #         # Cannot be ON for more than max_up consecutive periods
        #         cons = self.model.addCons(
        #             quicksum(self.z_on[u, tau] for tau in range(t - max_up + 1, t + 1)) <= max_up,
        #             name=f"max_up_time_{u}_{t}"
        #         )
        #         self.electrolyzer_cons[f"max_up_time_{u}_{t}"] = cons
            
        # Minimum down time constraints (최소 정지 시간)
            if u in self.players_with_electrolyzers:
                min_down = self.params.get('min_down_time', 1)
                # for t in range(1, len(self.time_periods)):  # t ∈ T \ {1}
                for t in [tau for tau in self.time_periods if tau != 6]:
                # t시점에 off로 전환되었는지 확인 (z_off_t - z_off_{t-1})
                # 만약 전환되었다면, 다음 min_down 기간 동안 off 유지
                    down_time_idx = [tau for tau in range(t, t + min_down)]
                    down_time_idx = [tau if tau < 24 else tau - 24 for tau in down_time_idx]
                    # for n in range(t, min(t + min_down, len(self.time_periods))):
                    if t != 0:
                        for n in down_time_idx:
                            cons = self.model.addCons(
                            self.z_off[u,t] - self.z_off[u,t-1] <= self.z_off[u,n],
                            name=f"min_downtime_{u}_{t}_{n}"
                        )
                        self.electrolyzer_cons[f"min_downtime_{u}_{t}_{n}"] = cons
                    else:
                        for n in down_time_idx:
                            cons = self.model.addCons(
                                    self.z_off[u,t] - self.z_off[u,23] <= self.z_off[u,n],
                                    name=f"min_downtime_{u}_{t}_{n}"
                                )
                        self.electrolyzer_cons[f"min_downtime_{u}_{t}_{n}"] = cons
        # hydro storage SOC transition with special 23→0 transition
            if u in self.players_with_hydro_storage:
                nu_ch = self.params.get('nu_ch', 0.9)
                nu_dis = self.params.get('nu_dis', 0.9)            
                initial_soc = 0.0  # 수소 저장소 초기값은 0
                
                # Set initial SOC at 6시 (논리적 시작점)
                if (u,6) in self.s_G:
                    cons = self.model.addCons(self.s_G[u,6] == initial_soc, name=f"initial_soc_G_{u}")
                    self.storage_cons[f"initial_soc_G_{u}"] = cons
                
                # 일반적인 SOC transition (1시~23시)
                for t in range(1, 24):
                    if (u,t) in self.s_G and (u,t-1) in self.s_G:
                        cons = self.model.addCons(
                        self.s_G[u,t] == self.s_G[u,t-1] + nu_ch * self.b_ch_G[u,t] - (1/nu_dis) * self.b_dis_G[u,t],
                        name=f"soc_transition_G_{u}_{t}"
                    )
                    self.storage_cons[f"soc_transition_G_{u}_{t}"] = cons
                
                # 특별한 53→6시 transition (심야 충전 → 아침 방전)
                if (u,6) in self.s_G and (u,53) in self.s_G:
                    cons = self.model.addCons(
                        self.s_G[u,6] == self.s_G[u,53] + nu_ch * self.b_ch_G[u,6] - (1/nu_dis) * self.b_dis_G[u,6],
                        name=f"soc_transition_G_{u}_53_to_6"
                    )
                    self.storage_cons[f"soc_transition_G_{u}_53_to_6"] = cons
        

        if not self.dwr:
        # Community hydro balance
            for t in self.time_periods:
                community_hydro_balance = quicksum(self.i_G_com.get((u,t),0) - self.e_G_com.get((u,t),0) for u in self.players)
                cons = self.model.addCons(community_hydro_balance == 0, name=f"community_hydro_balance_{t}")
                self.community_hydro_balance_cons[f"community_hydro_balance_{t}"] = cons
    
    
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
        flow_analysis = self._analyze_energy_flows(results)
        # Analyze electrolyzer operation
        electrolyzer_operation = self._analyze_electrolyzer_operations(results)
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
            revenue_analysis['heat']['storage_cost'] 
            # revenue_analysis['common']['peak_power_cost']
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
        # print(f"  Peak power penalty:  -{revenue_analysis['common']['peak_power_cost']:10.4f}")
        
        print(f"\n[TOTAL]")
        print(f"  Total revenue:        {revenue_analysis['total_revenue']:10.4f}")
        print(f"  Total cost:          -{revenue_analysis['total_cost']:10.4f}")
        print(f"  Net profit:           {revenue_analysis['net_profit']:10.4f}")
        
        print(f"\n[VERIFICATION]")
        print(f"  Calculated cost:    {revenue_analysis['net_profit']:10.6f}")
        print(f"  Solver objective:     {self.model.getObjVal():10.6f}")
        print(f"  Difference:           {abs(revenue_analysis['net_profit'] - (-1*self.model.getObjVal())):10.10f}")
        
        if abs(revenue_analysis['net_profit'] - (-1*self.model.getObjVal())) > 1e-6:
            print("  ⚠️  WARNING: Calculated profit doesn't match solver objective value!")
        else:
            print("  ✓  Verification passed!")
        
        return revenue_analysis
    def _analyze_energy_flows(self, results):
        """
        Analyze energy flows including community internal conversions and demand sources
        
        Args:
            results: Dictionary containing optimization results
            
        Returns:
            Dictionary with energy flow analysis
        """
        flow_analysis = {}
        
        for t in self.time_periods:
            flow_analysis[t] = {
                # Electricity flows
                'elec_renewable': 0.0,
                'elec_to_grid': 0.0,
                'elec_from_grid': 0.0,
                'elec_to_hydro': 0.0,
                'elec_to_heat': 0.0,
                'elec_storage_charge': 0.0,
                'elec_storage_discharge': 0.0,
                'elec_nfl_demand': 0.0,
                'elec_nfl_from_grid': 0.0,      # NEW: Grid portion of nfl demand
                'elec_nfl_from_community': 0.0,  # NEW: Community portion of nfl demand
                'elec_net_comm': 0.0,
                
                # Hydrogen flows
                'hydro_produced': 0.0,
                'hydro_from_grid': 0.0,
                'hydro_to_grid': 0.0,
                'hydro_nfl_demand': 0.0,
                'hydro_nfl_from_grid': 0.0,      # NEW
                'hydro_nfl_from_community': 0.0,  # NEW
                'hydro_storage_charge': 0.0,
                'hydro_storage_discharge': 0.0,
                
                # Heat flows
                'heat_produced': 0.0,
                'heat_from_grid': 0.0,
                'heat_to_grid': 0.0,
                'heat_nfl_demand': 0.0,
                'heat_nfl_from_grid': 0.0,       # NEW
                'heat_nfl_from_community': 0.0,   # NEW
                'heat_storage_charge': 0.0,
                'heat_storage_discharge': 0.0,
                
                # Storage SOC values
                'elec_soc': 0.0,
                'hydro_soc': 0.0,
                'heat_soc': 0.0,
            }
            
            # === ELECTRICITY FLOWS ===
            # Renewable generation
            if 'p' in results:
                for u in self.players_with_renewables:
                    if (u, 'res', t) in results['p']:
                        flow_analysis[t]['elec_renewable'] += results['p'][u, 'res', t]
            
            # Grid interactions (total)
            if 'e_E_gri' in results:
                for u in self.players:
                    if (u, t) in results['e_E_gri']:
                        flow_analysis[t]['elec_to_grid'] += results['e_E_gri'][u, t]
            
            if 'i_E_gri' in results:
                for u in self.players:
                    if (u, t) in results['i_E_gri']:
                        flow_analysis[t]['elec_from_grid'] += results['i_E_gri'][u, t]
            
            # Storage
            if 'b_ch_E' in results:
                for u in self.players:
                    if (u, t) in results['b_ch_E']:
                        flow_analysis[t]['elec_storage_charge'] += results['b_ch_E'][u, t]
            
            if 'b_dis_E' in results:
                for u in self.players:
                    if (u, t) in results['b_dis_E']:
                        flow_analysis[t]['elec_storage_discharge'] += results['b_dis_E'][u, t]
            
            # Electricity demand and sources
            if 'nfl_d' in results:
                for u in self.players_with_nfl_elec_demand:
                    if (u, 'elec', t) in results['nfl_d']:
                        demand = results['nfl_d'][u, 'elec', t]
                        flow_analysis[t]['elec_nfl_demand'] += demand
                        
                        # Check how this demand is met
                        grid_import = results.get('i_E_gri', {}).get((u, t), 0)
                        comm_import = results.get('i_E_com', {}).get((u, t), 0)
                        
                        flow_analysis[t]['elec_nfl_from_grid'] += grid_import
                        flow_analysis[t]['elec_nfl_from_community'] += comm_import
            
            # Flexible demand (for conversion)
            if 'fl_d' in results:
                for u in self.players_with_electrolyzers:
                    if (u, 'elec', t) in results['fl_d']:
                        flow_analysis[t]['elec_to_hydro'] += results['fl_d'][u, 'elec', t]
            
                for u in self.players_with_heatpumps:
                    if (u, 'elec', t) in results['fl_d']:
                        flow_analysis[t]['elec_to_heat'] += results['fl_d'][u, 'elec', t]
            
            # === HYDROGEN FLOWS ===
            # Production
            if 'p' in results:
                for u in self.players_with_electrolyzers:
                    if (u, 'els', t) in results['p']:
                        flow_analysis[t]['hydro_produced'] += results['p'][u, 'els', t]
            
            # Grid interactions (total)
            if 'e_G_gri' in results:
                for u in self.players:
                    if (u, t) in results['e_G_gri']:
                        flow_analysis[t]['hydro_to_grid'] += results['e_G_gri'][u, t]
            
            if 'i_G_gri' in results:
                for u in self.players:
                    if (u, t) in results['i_G_gri']:
                        flow_analysis[t]['hydro_from_grid'] += results['i_G_gri'][u, t]
            
            # Storage
            if 'b_ch_G' in results:
                for u in self.players:
                    if (u, t) in results['b_ch_G']:
                        flow_analysis[t]['hydro_storage_charge'] += results['b_ch_G'][u, t]
            
            if 'b_dis_G' in results:
                for u in self.players:
                    if (u, t) in results['b_dis_G']:
                        flow_analysis[t]['hydro_storage_discharge'] += results['b_dis_G'][u, t]
            
            # Hydrogen demand and sources
            if 'nfl_d' in results:
                for u in self.players_with_nfl_hydro_demand:
                    if (u, 'hydro', t) in results['nfl_d']:
                        demand = results['nfl_d'][u, 'hydro', t]
                        flow_analysis[t]['hydro_nfl_demand'] += demand
                        
                        # Check how this demand is met
                        grid_import = results.get('i_G_gri', {}).get((u, t), 0)
                        comm_import = results.get('i_G_com', {}).get((u, t), 0)
                        
                        flow_analysis[t]['hydro_nfl_from_grid'] += grid_import
                        flow_analysis[t]['hydro_nfl_from_community'] += comm_import
            
            # === HEAT FLOWS ===
            # Production
            if 'p' in results:
                for u in self.players_with_heatpumps:
                    if (u, 'hp', t) in results['p']:
                        flow_analysis[t]['heat_produced'] += results['p'][u, 'hp', t]
            
            # Grid interactions (total)
            if 'e_H_gri' in results:
                for u in self.players:
                    if (u, t) in results['e_H_gri']:
                        flow_analysis[t]['heat_to_grid'] += results['e_H_gri'][u, t]
            
            if 'i_H_gri' in results:
                for u in self.players:
                    if (u, t) in results['i_H_gri']:
                        flow_analysis[t]['heat_from_grid'] += results['i_H_gri'][u, t]
            
            # Storage
            if 'b_ch_H' in results:
                for u in self.players:
                    if (u, t) in results['b_ch_H']:
                        flow_analysis[t]['heat_storage_charge'] += results['b_ch_H'][u, t]
        
            if 'b_dis_H' in results:
                for u in self.players:
                    if (u, t) in results['b_dis_H']:
                        flow_analysis[t]['heat_storage_discharge'] += results['b_dis_H'][u, t]
        
            # Heat demand and sources
            if 'nfl_d' in results:
                for u in self.players_with_nfl_heat_demand:
                    if (u, 'heat', t) in results['nfl_d']:
                        demand = results['nfl_d'][u, 'heat', t]
                        flow_analysis[t]['heat_nfl_demand'] += demand
                        
                        # Check how this demand is met
                        grid_import = results.get('i_H_gri', {}).get((u, t), 0)
                        comm_import = results.get('i_H_com', {}).get((u, t), 0)
                        
                        flow_analysis[t]['heat_nfl_from_grid'] += grid_import
                        flow_analysis[t]['heat_nfl_from_community'] += comm_import
            
            # Net Community calculation for electricity
            flow_analysis[t]['elec_net_comm'] = (
            flow_analysis[t]['elec_renewable'] +
            flow_analysis[t]['elec_from_grid'] +
            flow_analysis[t]['elec_storage_discharge'] -
            flow_analysis[t]['elec_to_grid'] -
            flow_analysis[t]['elec_storage_charge'] -
                flow_analysis[t]['elec_nfl_demand'] -
                flow_analysis[t]['elec_to_hydro'] -
                flow_analysis[t]['elec_to_heat']
            )
            
            # Calculate Storage SOC values
            # Electricity storage SOC - 해당 시간대의 SOC 레벨
            if 's_E' in results:
                # 모든 전력 저장소 플레이어의 SOC 레벨 합계
                flow_analysis[t]['elec_soc'] = sum(
                    results['s_E'][u, t] 
                    for u in self.players_with_elec_storage 
                    if (u, t) in results['s_E']
                )
            
            # Hydrogen storage SOC - 해당 시간대의 SOC 레벨
            if 's_G' in results:
                # 모든 수소 저장소 플레이어의 SOC 레벨 합계
                flow_analysis[t]['hydro_soc'] = sum(
                    results['s_G'][u, t] 
                    for u in self.players_with_hydro_storage 
                    if (u, t) in results['s_G']
                )
            
            # Heat storage SOC - 해당 시간대의 SOC 레벨
            if 's_H' in results:
                # 모든 열 저장소 플레이어의 SOC 레벨 합계
                flow_analysis[t]['heat_soc'] = sum(
                    results['s_H'][u, t] 
                    for u in self.players_with_heat_storage 
                    if (u, t) in results['s_H']
        )
        
        # Print analysis
        self._print_energy_flow_analysis(flow_analysis)
        
        return flow_analysis

    def _print_energy_flow_analysis(self, flow_analysis):
        """
        Print energy flow analysis in a formatted table
        """
        # ========== ORIGINAL TABLE (유지) ==========
        print("\n" + "="*130)
        print("ENERGY FLOW ANALYSIS BY TIME PERIOD")
        print("="*130)
        
        # Header - 기존 포맷 유지
        print(f"{'Time':^4} | {'='*27} ELECTRICITY {'='*27} | {'='*5} HYDROGEN {'='*5} | {'='*7} HEAT {'='*7}")
        print(f"{'':^4} | {'Renewable':^10} {'From Grid':^10} {'To Grid':^10} {'→Hydro':^10} {'→Heat':^10} {'Net Comm':^10} | {'Produced':^10} {'To/From':^10} | {'Produced':^10} {'To/From':^10}")
        print("-"*130)
        
        # Data rows - 실제 시간으로 표시 (6시~5시)
        for t in list(range(6, 24)) + list(range(0, 6)):
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
        
        print("-"*130)
        
        # ========== NEW TABLE 1: DEMAND SOURCE BREAKDOWN ==========
        print("\n" + "="*130)
        print("DEMAND SOURCE BREAKDOWN BY TIME PERIOD")
        print("="*130)
        
        print(f"{'Time':^4} | {'='*27} ELECTRICITY {'='*27} | {'='*15} HYDROGEN {'='*15} | {'='*15} HEAT {'='*15}")
        print(f"{'':^4} | {'NFL Demand':^10} {'From Grid':^10} {'From Comm':^10} {'Comm %':^10} | {'NFL Demand':^10} {'From Grid':^10} {'From Comm':^10} {'Comm %':^10} | {'NFL Demand':^10} {'From Grid':^10} {'From Comm':^10} {'Comm %':^10}")
        print("-"*130)
        
        for t in list(range(6, 24)) + list(range(0, 6)):
            flow = flow_analysis[t]
            
            # Electricity demand source
            elec_comm_pct = (flow['elec_nfl_from_community'] / flow['elec_nfl_demand'] * 100) if flow['elec_nfl_demand'] > 0 else 0
            
            # Hydrogen demand source
            hydro_comm_pct = (flow['hydro_nfl_from_community'] / flow['hydro_nfl_demand'] * 100) if flow['hydro_nfl_demand'] > 0 else 0
            
            # Heat demand source
            heat_comm_pct = (flow['heat_nfl_from_community'] / flow['heat_nfl_demand'] * 100) if flow['heat_nfl_demand'] > 0 else 0
            
            print(f"{t:^4} | "
                f"{flow['elec_nfl_demand']:^10.2f} "
                f"{flow['elec_nfl_from_grid']:^10.2f} "
                f"{flow['elec_nfl_from_community']:^10.2f} "
                f"{elec_comm_pct:^10.1f} | "
                f"{flow['hydro_nfl_demand']:^10.2f} "
                f"{flow['hydro_nfl_from_grid']:^10.2f} "
                f"{flow['hydro_nfl_from_community']:^10.2f} "
                f"{hydro_comm_pct:^10.1f} | "
                f"{flow['heat_nfl_demand']:^10.2f} "
                f"{flow['heat_nfl_from_grid']:^10.2f} "
                f"{flow['heat_nfl_from_community']:^10.2f} "
                f"{heat_comm_pct:^10.1f}")
        
        print("-"*130)
        
        # ========== NEW TABLE 2: STORAGE OPERATION ==========
        print("\n" + "="*130)
        print("STORAGE OPERATION BY TIME PERIOD")
        print("="*130)
        
        print(f"{'Time':^4} | {'='*22} ELECTRICITY STORAGE {'='*22} | {'='*12} HYDROGEN STORAGE {'='*12} | {'='*14} HEAT STORAGE {'='*14}")
        print(f"{'':^4} | {'Charge':^10} {'Discharge':^10} {'Net':^10} {'SOC':^10} | {'Charge':^10} {'Discharge':^10} {'Net':^10} | {'Charge':^10} {'Discharge':^10} {'Net':^10}")
        print("-"*130)
        
        for t in list(range(6, 24)) + list(range(0, 6)):
            flow = flow_analysis[t]
            
            # Calculate net flows
            elec_net = flow['elec_storage_discharge'] - flow['elec_storage_charge']
            hydro_net = flow['hydro_storage_discharge'] - flow['hydro_storage_charge']
            heat_net = flow['heat_storage_discharge'] - flow['heat_storage_charge']
            
            # SOC values (need to be added to flow_analysis if available)
            elec_soc = flow.get('elec_soc', 0.0)
            
            print(f"{t:^4} | "
                f"{flow['elec_storage_charge']:^10.2f} "
                f"{flow['elec_storage_discharge']:^10.2f} "
                f"{elec_net:^10.2f} "
                f"{elec_soc:^10.2f} | "
                f"{flow['hydro_storage_charge']:^10.2f} "
                f"{flow['hydro_storage_discharge']:^10.2f} "
                f"{hydro_net:^10.2f} | "
                f"{flow['heat_storage_charge']:^10.2f} "
                f"{flow['heat_storage_discharge']:^10.2f} "
                f"{heat_net:^10.2f}")
        
        print("-"*130)
        
        # ========== NEW TABLE 3: STORAGE STATUS ==========
        print("\n" + "="*130)
        print("STORAGE STATUS BY TIME PERIOD")
        print("="*130)
        
        print(f"{'Time':^4} | {'='*15} ELECTRICITY {'='*15} | {'='*15} HYDROGEN {'='*15} | {'='*15} HEAT {'='*15}")
        print(f"{'':^4} | {'Charge':^10} {'Discharge':^10} {'Net':^10} {'SOC':^10} | {'Charge':^10} {'Discharge':^10} {'Net':^10} {'SOC':^10} | {'Charge':^10} {'Discharge':^10} {'Net':^10} {'SOC':^10}")
        print("-"*130)
        
        for t in list(range(6, 24)) + list(range(0, 6)):
            flow = flow_analysis[t]
            
            # Calculate net flows for storage
            elec_net = flow['elec_storage_discharge'] - flow['elec_storage_charge']
            hydro_net = flow['hydro_storage_discharge'] - flow['hydro_storage_charge']
            heat_net = flow['heat_storage_discharge'] - flow['heat_storage_charge']
            
            print(f"{t:^4} | "
                f"{flow['elec_storage_charge']:^10.2f} "
                f"{flow['elec_storage_discharge']:^10.2f} "
                f"{elec_net:^10.2f} "
                f"{flow['elec_soc']:^10.2f} | "
                f"{flow['hydro_storage_charge']:^10.2f} "
                f"{flow['hydro_storage_discharge']:^10.2f} "
                f"{hydro_net:^10.2f} "
                f"{flow['hydro_soc']:^10.2f} | "
                f"{flow['heat_storage_charge']:^10.2f} "
                f"{flow['heat_storage_discharge']:^10.2f} "
                f"{heat_net:^10.2f} "
                f"{flow['heat_soc']:^10.2f}")
        
        print("-"*130)
        
        # ========== SUMMARY STATISTICS (기존 유지) ==========
        print("\nSUMMARY STATISTICS:")
        
        total_renewable = sum(f['elec_renewable'] for f in flow_analysis.values())
        total_elec_import = sum(f['elec_from_grid'] for f in flow_analysis.values())
        total_elec_export = sum(f['elec_to_grid'] for f in flow_analysis.values())
        total_elec_to_hydro = sum(f['elec_to_hydro'] for f in flow_analysis.values())
        total_elec_to_heat = sum(f['elec_to_heat'] for f in flow_analysis.values())
        total_hydro_produced = sum(f['hydro_produced'] for f in flow_analysis.values())
        total_heat_produced = sum(f['heat_produced'] for f in flow_analysis.values())
        
        # Community self-sufficiency 계산
        total_elec_nfl_demand = sum(f['elec_nfl_demand'] for f in flow_analysis.values())
        total_elec_from_comm = sum(f['elec_nfl_from_community'] for f in flow_analysis.values())
        total_hydro_nfl_demand = sum(f['hydro_nfl_demand'] for f in flow_analysis.values())
        total_hydro_from_comm = sum(f['hydro_nfl_from_community'] for f in flow_analysis.values())
        total_heat_nfl_demand = sum(f['heat_nfl_demand'] for f in flow_analysis.values())
        total_heat_from_comm = sum(f['heat_nfl_from_community'] for f in flow_analysis.values())
        
        print(f"\nElectricity:")
        print(f"  Total renewable generated:     {total_renewable:10.2f} kWh")
        print(f"  Total imported from grid:      {total_elec_import:10.2f} kWh")
        print(f"  Total exported to grid:        {total_elec_export:10.2f} kWh")
        print(f"  Total used for hydrogen:       {total_elec_to_hydro:10.2f} kWh")
        print(f"  Total used for heat:           {total_elec_to_heat:10.2f} kWh")
        print(f"  Net grid position:             {total_elec_import - total_elec_export:+10.2f} kWh (+ = import, - = export)")
        if total_elec_nfl_demand > 0:
            print(f"  Community self-sufficiency:    {total_elec_from_comm/total_elec_nfl_demand*100:10.1f}%")
        
        print(f"\nHydrogen:")
        print(f"  Total produced:                {total_hydro_produced:10.2f} kg")
        if total_hydro_nfl_demand > 0:
            print(f"  Community self-sufficiency:    {total_hydro_from_comm/total_hydro_nfl_demand*100:10.1f}%")
        
        print(f"\nHeat:")
        print(f"  Total produced:                {total_heat_produced:10.2f} kWh")
        print(f"  COP (efficiency):              {total_heat_produced/total_elec_to_heat if total_elec_to_heat > 0 else 0:10.2f}")
        if total_heat_nfl_demand > 0:
            print(f"  Community self-sufficiency:    {total_heat_from_comm/total_heat_nfl_demand*100:10.1f}%")
        
        # Check for community internal hydrogen trading
        hydro_community_trade = self._check_community_hydrogen_trade(flow_analysis)
        if hydro_community_trade > 0:
            print(f"\n⚠️  Community hydrogen trade detected: {hydro_community_trade:.2f} kg")
        else:
            print(f"\n❌ No community hydrogen trading detected (all hydrogen goes through grid)")
    def _analyze_electrolyzer_operations(self, results):
        """
        Analyze electrolyzer operations including commitment states and production
        
        Args:
            results: Dictionary containing optimization results
            
        Returns:
            Dictionary with electrolyzer operation analysis
        """
        electrolyzer_analysis = {}
        
        for u in self.players_with_electrolyzers:
            electrolyzer_analysis[u] = {
                'summary': {
                    'total_hydrogen_produced': 0.0,
                    'total_electricity_consumed': 0.0,
                    'total_startup_cost': 0.0,
                    'conversion_efficiency': 0.0,
                    'hours_online': 0,
                    'hours_standby': 0,
                    'hours_offline': 0,
                    'number_of_startups': 0,
                    'capacity_utilization': 0.0,
                    'average_production_when_on': 0.0
                },
                'hourly_operations': {}
            }
            
            # Collect hourly data
            total_hydrogen = 0.0
            total_electricity = 0.0
            total_startups = 0.0
            hours_on = 0
            hours_sb = 0
            hours_off = 0
            total_production_when_on = 0.0
            hours_actually_producing = 0
            
            for t in self.time_periods:
                hour_data = {
                    'state': 'unknown',
                    'startup': 0.0,
                    'hydrogen_production': 0.0,
                    'electricity_consumption': 0.0,
                    'capacity_utilization': 0.0
                }
                
                # Get state information
                if 'z_on' in results and (u, t) in results['z_on']:
                    z_on = results['z_on'][u, t]
                    z_off = results['z_off'].get((u, t), 0)
                    z_sb = results['z_sb'].get((u, t), 0)
                    z_su = results['z_su'].get((u, t), 0)
                    
                    # Determine state
                    if z_on > 0.5:
                        hour_data['state'] = 'ON'
                        hours_on += 1
                    elif z_sb > 0.5:
                        hour_data['state'] = 'STANDBY'
                        hours_sb += 1
                    elif z_off > 0.5:
                        hour_data['state'] = 'OFF'
                        hours_off += 1
                    
                    hour_data['startup'] = z_su
                    total_startups += z_su
                
                # Get production data
                if 'p' in results and (u, 'els', t) in results['p']:
                    hour_data['hydrogen_production'] = results['p'][u, 'els', t]
                    total_hydrogen += hour_data['hydrogen_production']
                    
                    if hour_data['hydrogen_production'] > 0.01:  # Consider non-zero production
                        total_production_when_on += hour_data['hydrogen_production']
                        hours_actually_producing += 1
                
                # Get electricity consumption
                if 'fl_d' in results and (u, 'elec', t) in results['fl_d']:
                    hour_data['electricity_consumption'] = results['fl_d'][u, 'elec', t]
                    total_electricity += hour_data['electricity_consumption']
                
                # Calculate capacity utilization
                max_capacity = self.params.get(f'C_max_{u}', 100)
                if max_capacity > 0:
                    hour_data['capacity_utilization'] = (hour_data['electricity_consumption'] / max_capacity) * 100
                
                electrolyzer_analysis[u]['hourly_operations'][t] = hour_data
            
            # Calculate summary statistics
            summary = electrolyzer_analysis[u]['summary']
            summary['total_hydrogen_produced'] = total_hydrogen
            summary['total_electricity_consumed'] = total_electricity
            summary['total_startup_cost'] = total_startups * self.params.get(f'c_su_{u}', 50)
            summary['hours_online'] = hours_on
            summary['hours_standby'] = hours_sb
            summary['hours_offline'] = hours_off
            summary['number_of_startups'] = int(round(total_startups))
            
            # Calculate efficiency and utilization
            if total_electricity > 0:
                summary['conversion_efficiency'] = total_hydrogen / total_electricity
            
            max_capacity = self.params.get(f'C_max_{u}', 100)
            if max_capacity > 0 and len(self.time_periods) > 0:
                summary['capacity_utilization'] = (total_electricity / (max_capacity * len(self.time_periods))) * 100
            
            if hours_actually_producing > 0:
                summary['average_production_when_on'] = total_production_when_on / hours_actually_producing
        
        # Print analysis
        self._print_electrolyzer_analysis(electrolyzer_analysis)
        
        return electrolyzer_analysis
    def _print_electrolyzer_analysis(self, electrolyzer_analysis):
        """
        Print electrolyzer operation analysis in a formatted table
        """
        print("\n" + "="*120)
        print("ELECTROLYZER OPERATIONS ANALYSIS")
        print("="*120)
        
        for u in self.players_with_electrolyzers:
            analysis = electrolyzer_analysis[u]
            summary = analysis['summary']
            
            print(f"\n[ELECTROLYZER {u.upper()}] - OPERATIONAL SUMMARY")
            print("-" * 60)
            print(f"  Total hydrogen produced:       {summary['total_hydrogen_produced']:10.2f} kg")
            print(f"  Total electricity consumed:    {summary['total_electricity_consumed']:10.2f} kWh")
            print(f"  Conversion efficiency:         {summary['conversion_efficiency']:10.4f} kg/kWh")
            print(f"  Average production when ON:    {summary['average_production_when_on']:10.2f} kg/h")
            print(f"  Capacity utilization:          {summary['capacity_utilization']:10.2f} %")
            print(f"  Number of startups:            {summary['number_of_startups']:10d}")
            print(f"  Total startup cost:            {summary['total_startup_cost']:10.2f}")
            print(f"  Hours online:                  {summary['hours_online']:10d}")
            print(f"  Hours standby:                 {summary['hours_standby']:10d}")
            print(f"  Hours offline:                 {summary['hours_offline']:10d}")
            
            # Hourly operations table - 간단한 시간대별 표시
            print(f"\n[ELECTROLYZER {u.upper()}] - 시간대별 운영 현황")
            print("-" * 80)
            print(f"{'시간':^4} | {'상태':^8} | {'시작':^6} | {'수소생산':^8} | {'전력소비':^10} | {'가동률':^8}")
            print(f"{'':^4} | {'':^8} | {'':^6} | {'(kg/h)':^8} | {'(MWh)':^10} | {'(%)':^8}")
            print("-" * 80)
            
            # 시간대별로 한 줄씩 깔끔하게 표시
            for t in list(range(6, 24)) + list(range(0, 6)):
                data = analysis['hourly_operations'][t]
                startup_mark = "★" if data['startup'] > 0.5 else " "
                
                print(f"{t:^4} | {data['state']:^8} | {startup_mark:^6} | {data['hydrogen_production']:^8.2f} | {data['electricity_consumption']:^10.2f} | {data['capacity_utilization']:^8.1f}")
            
            # 상태 전환 분석 - 간단하게
            print(f"\n[ELECTROLYZER {u.upper()}] - 상태 변화")
            print("-" * 50)
            prev_state = None
            
            for t in sorted(analysis['hourly_operations'].keys()):
                current_state = analysis['hourly_operations'][t]['state']
                startup = analysis['hourly_operations'][t]['startup']
                
                if prev_state is not None and prev_state != current_state:
                    startup_text = " (시작)" if startup > 0.5 else ""
                    print(f"  {t:2d}시: {prev_state} → {current_state}{startup_text}")
                elif startup > 0.5:
                    print(f"  {t:2d}시: {current_state} 시작")
                
                prev_state = current_state
            
            # 연속 운영 구간 분석
            print(f"\n[ELECTROLYZER {u.upper()}] - 연속 운영 구간")
            print("-" * 50)
            
            # 연속 운영 구간 찾기
            on_periods = []
            current_on_start = None
            
            for t in sorted(analysis['hourly_operations'].keys()):
                state = analysis['hourly_operations'][t]['state']
                
                if state == 'ON':
                    if current_on_start is None:
                        current_on_start = t
                else:
                    if current_on_start is not None:
                        on_periods.append((current_on_start, t - 1))
                        current_on_start = None
            
            # 마지막까지 켜져있는 경우 처리
            if current_on_start is not None:
                on_periods.append((current_on_start, max(analysis['hourly_operations'].keys())))
            
            if on_periods:
                for start, end in on_periods:
                    duration = end - start + 1
                    avg_production = sum(analysis['hourly_operations'][t]['hydrogen_production'] 
                                    for t in range(start, end + 1)) / duration
                    print(f"  {start:2d}시-{end:2d}시 ({duration:2d}시간): 평균 생산량 {avg_production:.2f} kg/h")
            else:
                print("  연속 운영 구간 없음")
        
        print("\n" + "="*120)
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
                    renewable_cap = self.params.get(f'renewable_cap_{u}', 2)
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
                    C_min = self.params.get(f'C_min_{u}', 80)
                    
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
                    renewable_cap = self.params.get(f'renewable_cap_{u}', 2)
                    
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
            # 'peak_power': self.model.getVal(self.chi_peak)
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
    try:
        avg_prices, daily_prices = load_korean_electricity_prices()
        
        # 가격을 EUR/kWh로 변환 (KRW/kWh → EUR/kWh)
        # 가정: 1 EUR = 1400 KRW
        exchange_rate = 1400
        unit_adjustment = 1000 ## smp 데이터는 원/kWh, 그리고 수소 가격 등 논문에선 €/MWh 단위로 주어짐. 즉, 1000을 곱해줘야 함
        # 전기가격 일일 패턴 사용 (더 변동성 있는 패턴)
        korean_prices_eur = [price / exchange_rate * unit_adjustment for price in avg_prices]
        # 수소가격 계산
        h2_prices_eur = calculate_hydrogen_prices(korean_prices_eur)
        print("\n" + "="*80)
        print("KOREAN ELECTRICITY PRICE DATA LOADED")
        print("="*80)
        print(f"Price range: {min(korean_prices_eur):.4f} - {max(korean_prices_eur):.4f} EUR/kWh")
        print(f"Average: {np.mean(korean_prices_eur):.4f} EUR/kWh")
    except Exception as e:
        print(f"Error loading electricity prices: {e}")
        exit(1)
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
        'storage_power': 0.5,        # 30 kW power rating (0.03 MW)
        'storage_capacity': 2.0,    # 2000 kWh capacity (2 MWh)
        'initial_soc': 0.5*1,         # 50% of 1 MWh
        'nu_ch': 0.95,
        'nu_dis': 0.95,
        'pi_peak': 50,  # Peak penalty reduced
        # Equipment capacities
    # 'renewable_cap_u1' is now replaced by time-dependent 'renewable_cap_u1_t' below
        # Heat pump parameters
        'hp_cap_u3': 0.08,           # 80 kW thermal heat pump (0.08 MW)

        # Electrolyzer parameters
        'els_cap_u2': 1,          # Total electrolyzer capacity  [MW]
        'C_min_u2': 0.15,            # % minimum load
        'C_sb_u2': 0.01,              # % Power consumption in stand-by state
        'phi1_1_u2': 21.12266316,            # kg H2/kWh efficiency
        'phi0_1_u2': -0.37924094,
        'phi1_2_u2': 16.66883134,            # kg H2/kWh efficiency
        'phi0_2_u2': 0.87814262,
        'c_els_u2': 0.05,          # Small production cost
        'c_su_u2': 50,             # Startup cost reduced
        'max_up_time': 6,
        'min_down_time': 2,
        
        # Grid connection limits
        'e_E_cap': 0.1,       # 100 kW export limit (0.1 MW)
        'i_E_cap': 0.5,       # 50%
        'i_E_cap': 0.5,       # 50%
        'i_E_cap': 0.5,       # 50%
        'e_H_cap': 0.06,        # 60 kW heat export (0.06 MW)
        'i_H_cap': 0.08,        # 80 kW heat import (0.08 MW)
        'e_G_cap': 50,        # 50 kg/day hydro export
        'i_G_cap': 30,        # 30 kg/day hydro import
        
        # Cost parameters
        'c_sto': 0.01,             # Common storage cost
        
    }
    parameters['players_with_fl_elec_demand'] = list(set(parameters['players_with_electrolyzers'] + parameters['players_with_heatpumps']))
    # # Add demand data - Increased demand to encourage community trading
    # for u in players:
    #     for t in time_periods:
    #         parameters[f'd_E_nfl_{u}_{t}'] = 30 + 15 * np.sin(2 * np.pi * t / 24)  # Increased: 15~45 kW
    #         parameters[f'd_H_nfl_{u}_{t}'] = 20 + 10 * np.sin(2 * np.pi * t / 24)  # Increased: 10~30 kW
    #         parameters[f'd_G_nfl_{u}_{t}'] = 20 + 4 * np.sin(2 * np.pi * t / 24)    # Increased: 4~12 kg/day

    # RENEWABLE AVAILABILITY - Natural solar curve
    for t in time_periods:
        # Solar PV generation curve (bell-shaped, peaks at noon)
        if 6 <= t <= 18:
            # Bell curve centered at noon
            solar_factor = np.exp(-((t - 12) / 3.5)**2)
            parameters[f'renewable_cap_u1_{t}'] = 2 * solar_factor  # Unit: MW
        else:
            parameters[f'renewable_cap_u1_{t}'] = 0  # No solar at night
    
    # Add cost parameters
    for u in players:
        parameters[f'c_res_{u}'] = 0.05
        parameters[f'c_hp_{u}'] = 0.1
        parameters[f'c_els_{u}'] = 0.08
        parameters[f'c_su_{u}'] = 50
        parameters[f'c_sto_{u}'] = 0.01
    
    # Add grid prices - Grid import is 0.1% more expensive than export to encourage community trading
    # 전력 가격 설정
    if korean_prices_eur:
        # 실제 한국 데이터 사용
        for t in time_periods:
            # 시간 인덱스 조정 (CSV는 1시부터, 코드는 0시부터)
            csv_hour = t + 1 if t < 23 else 0
            base_price = korean_prices_eur[csv_hour]
            # 시간대별 가격 조정 계수 (변동성 증가)
            if 0 <= t <= 5:  # 심야: 더 저렴하게
                price_multiplier = 0.7
            elif 10 <= t <= 15:  # 태양광 시간: 매우 저렴
                price_multiplier = 0.5
            elif 17 <= t <= 20:  # 저녁 피크: 더 비싸게
                price_multiplier = 1.5
            else:
                price_multiplier = 1.0
            
            adjusted_price = base_price * price_multiplier
            
            # 수출 가격에 인센티브 제공
            parameters[f'pi_E_gri_export_{t}'] = adjusted_price * 1.05  # 수출 프리미엄
            parameters[f'pi_E_gri_import_{t}'] = adjusted_price * 1.10  # 수입은 더 비싸게
            
            # HYDROGEN PRICE
            h2_price = h2_prices_eur[t]
            parameters[f'pi_G_gri_export_{t}'] = h2_price
            parameters[f'pi_G_gri_import_{t}'] = h2_price * 1.2
            
        print("\n[HOURLY ELECTRICITY PRICES FROM KOREAN DATA]")
        print("-"*60)
        print(f"{'Hour':^6} | {'Import Price':^15} | {'Export Price':^15}")
        print(f"{'':^6} | {'(EUR/kWh)':^15} | {'(EUR/kWh)':^15}")
        print("-"*60)
        
        for t in time_periods:
            import_price = parameters[f'pi_E_gri_import_{t}']
            export_price = parameters[f'pi_E_gri_export_{t}']
            
            # 가격 수준 표시
            if import_price < 0.08:
                level = "LOW"
            elif import_price < 0.12:
                level = "MED"
            else:
                level = "HIGH"
            
            print(f"{t:^6} | {import_price:^15.4f} | {export_price:^15.4f} | {level}")
    else:
        for t in time_periods:
            # Base price follows typical duck curve
            # High in morning (6-9), low at midday (10-15), very high in evening (17-20)
            
            # Morning ramp
            if 6 <= t <= 9:
                base_price = 0.6 + 0.2 * np.sin((t-6) * np.pi / 6)  # 0.6 to 0.8
            # Midday valley (solar peak)
            elif 10 <= t <= 15:
                base_price = 0.2 + 0.1 * np.cos((t-12.5) * np.pi / 3)  # 0.1 to 0.3
            # Evening peak
            elif 17 <= t <= 20:
                base_price = 0.9 + 0.3 * np.sin((t-17) * np.pi / 6)  # 0.9 to 1.2
            # Night hours
            elif 21 <= t <= 23 or 0 <= t <= 5:
                base_price = 0.3 + 0.1 * np.sin(t * np.pi / 12)  # 0.2 to 0.4
            # Transition hours
            else:
                base_price = 0.5
            
            # Add small random variations
            variation = 0.02 * np.sin(2 * np.pi * t / 4)
            
            # Set grid prices with spread
            parameters[f'pi_E_gri_export_{t}'] = base_price + variation
            parameters[f'pi_E_gri_import_{t}'] = (base_price + variation) * 1.15  # 15% markup
    
    # HEAT PRICES
    for t in time_periods:        
        # Heat prices - higher in morning and evening
        heat_demand_factor = 1.0 + 0.3 * np.cos(2 * np.pi * (t - 7) / 24)
        parameters[f'pi_H_gri_export_{t}'] = 0.25 * heat_demand_factor
        parameters[f'pi_H_gri_import_{t}'] = 0.30 * heat_demand_factor

    # ELEC/HYDRO/HEAT NON FLEXIBLE DEMAND PATTERN
    for u in players:
        for t in time_periods:
            # HYDROGEN DEMAND - 논문 기반 현실적 패턴 
            """ https://doi.org/10.7316/JHNE.2023.34.3.246 """
            if u == 'u5':
                # 오전 버스 수요 (6-11시)
                if 6 <= t <= 11:
                    morning_demand = 6 + 4 * np.exp(-((t-9)/2)**2)  # 6-10 kg/h
                # 오후 승용차 수요 (14-20시)
                elif 14 <= t <= 20:
                    afternoon_demand = 4 + 3 * np.exp(-((t-17)/2)**2)  # 4-7 kg/h
                # 기타 시간
                elif 12 <= t <= 13:
                    h2_demand = 3.0  # 점심시간 최소
                elif 21 <= t <= 23:
                    h2_demand = 2.0  # 야간 최소
                else:  # 0-5시
                    h2_demand = 1.0  # 새벽 최소
                
                # 시간대별 수요 설정
                if 6 <= t <= 11:
                    h2_demand = morning_demand
                elif 14 <= t <= 20:
                    h2_demand = afternoon_demand
                    
                parameters[f'd_G_nfl_{u}_{t}'] = h2_demand
            else:
                parameters[f'd_G_nfl_{u}_{t}'] = 0
            
            # ELEC DEMAND
            if u == 'u4':
                # 시간대별 전형적인 가정용 수요 패턴
                # 아침 피크(7-9시) + 저녁 피크(18-21시) + 낮 시간 최소
                
                # 아침 피크: 7-9시 중심
                morning_peak = 20 * np.exp(-((t - 8) / 2)**2)
                
                # 저녁 피크: 18-21시 중심 (더 높은 피크)
                evening_peak = 40 * np.exp(-((t - 19.5) / 2)**2)
                
                # 기본 부하 + 피크 수요
                base_demand = 60  # 기본 60kW
                elec_demand = base_demand + morning_peak + evening_peak
                
                # MW 단위로 변환
                elec_demand = elec_demand * 0.001  # Unit: MWh
                parameters[f'd_E_nfl_{u}_{t}'] = elec_demand
            # HEAT DEMAND
            if u == 'u6':
                heat_demand = 6 + 3 * np.cos(2 * np.pi * (t - 3) / 24)
                parameters[f'd_H_nfl_{u}_{t}'] = heat_demand
            else:
                parameters[f'd_H_nfl_{u}_{t}'] = 0

    # Create visualization of input curves
    print("\n" + "="*60)
    print("ELECTROLYZER STRATEGIC OPERATION SIMULATION")
    print("="*60)
    
    # Print price and renewable curves
    # Print expected operation patterns
    print("\n" + "="*80)
    print("EXPECTED ELECTROLYZER OPERATION PATTERNS")
    print("="*80)
    print("\n[PRICE ZONES AND EXPECTED BEHAVIOR]")
    print("-"*80)
    print("Time Period    | Elec Price | H2 Price | H2 Demand | Expected Operation")
    print("-"*80)
    
    for desc, hours in [
        ("Night Valley", "00-05"),
        ("Morning Peak", "06-09"),
        ("Solar Hours", "10-15"),
        ("Evening Peak", "16-20"),
        ("Evening", "21-23")
    ]:
        # Get representative hour
        if hours == "00-05":
            t_rep = 3
            expected = "MAX (80-100 kW)"
        elif hours == "06-09":
            t_rep = 7
            expected = "MIN (10-20 kW)"
        elif hours == "10-15":
            t_rep = 12
            expected = "HIGH (60-80 kW)"
        elif hours == "16-20":
            t_rep = 18
            expected = "OFF or MIN"
        else:
            t_rep = 22
            expected = "MEDIUM (30-50 kW)"
        
        elec_p = parameters[f'pi_E_gri_import_{t_rep}']
        h2_p = parameters[f'pi_G_gri_export_{t_rep}']
        h2_d = parameters[f'd_G_nfl_u5_{t_rep}']
        
        print(f"{desc:14} | €{elec_p:5.3f}/kWh | €{h2_p:4.2f}/kg | {h2_d:5.1f} kg/h | {expected}")
    
    
    print("\n" + "="*60)
    # Create and solve model with Restricted Pricing
    lem = LocalEnergyMarket(players, time_periods, parameters, isLP=False)
    # lem.model.addCons(lem.z_on[("u2", 0)] == 1)
    from pyscipopt import SCIP_PARAMSETTING
    lem.model.setPresolve(SCIP_PARAMSETTING.OFF)
    lem.model.setHeuristics(SCIP_PARAMSETTING.OFF)
    lem.model.disablePropagation()
    # First solve complete model and analyze revenue
    print("\n" + "="*60)
    print("SOLVING COMPLETE MODEL AND ANALYZING REVENUE")
    print("="*60)
    status_complete, results_complete, revenue_analysis = lem.solve_complete_model()
    
    if status_complete == "optimal":
        # Analyze electrolyzer specific operation
        print("\n" + "="*80)
        print("ELECTROLYZER (u2) DETAILED OPERATION")
        print("="*80)
        print(f"{'Hour':^4} | {'Elec Price':^10} | {'State':^8} | {'Power':^8} | {'H2 Prod':^8} | {'Load %':^8} | {'Visual':^10}")
        print(f"{'':^4} | {'(€/kWh)':^10} | {'':^8} | {'(kW)':^8} | {'(kg/h)':^8} | {'':^8} | {'':^10}")
        print("-"*80)
        
        total_h2_produced = 0
        total_elec_consumed = 0
        hours_on = 0
        hours_off = 0
        hours_standby = 0
        
        for t in list(range(6, 24)) + list(range(0, 6)):
            price = parameters[f'pi_E_gri_import_{t}']
            
            # Get states
            z_on = results_complete.get('z_on', {}).get(('u2', t), 0)
            z_sb = results_complete.get('z_sb', {}).get(('u2', t), 0)
            z_off = results_complete.get('z_off', {}).get(('u2', t), 0)
            
            # Get operation values
            elec_use = results_complete.get('fl_d', {}).get(('u2', 'elec', t), 0)
            h2_prod = results_complete.get('p', {}).get(('u2', 'els', t), 0)
            
            # Determine state
            if z_on > 0.5:
                state = "ON"
                hours_on += 1
            elif z_sb > 0.5:
                state = "STANDBY"
                hours_standby += 1
            else:
                state = "OFF"
                hours_off += 1
            
            # Calculate load percentage
            load_pct = (elec_use / parameters['els_cap_u2'] * 100) if parameters['els_cap_u2'] > 0 else 0
            
            # Visual indicator
            if load_pct >= 80:
                visual = "████████"
            elif load_pct >= 60:
                visual = "██████░░"
            elif load_pct >= 40:
                visual = "████░░░░"
            elif load_pct >= 20:
                visual = "██░░░░░░"
            elif load_pct > 1e-7:
                visual = "█░░░░░░░"
            else:
                visual = "░░░░░░░░"
            
            total_h2_produced += h2_prod
            total_elec_consumed += elec_use
            
            print(f"{t:^4} | {price:^10.3f} | {state:^8} | {elec_use:^8.1f} | {h2_prod:^8.2f} | {load_pct:^8.1f} | {visual:^10}")
        
        print("-"*80)
        print(f"SUMMARY: H2 Total: {total_h2_produced:.1f} kg | Elec Total: {total_elec_consumed:.1f} kWh")
        print(f"Hours: ON={hours_on}, STANDBY={hours_standby}, OFF={hours_off}")
        print(f"Average H2 efficiency: {total_h2_produced/total_elec_consumed if total_elec_consumed > 0 else 0:.3f} kg/kWh")
    
    else:
        print(f"Optimization failed: {status_complete}")
    
    print("\n" + "="*60)
    print("SOLVING WITH RESTRICTED PRICING")
    print("="*60)
    
    # Solve using Restricted Pricing
    # status, results, prices = lem.solve_with_restricted_pricing()
    
    # if status == "optimal":
    #     print(f"Optimal objective value: {results['objective_value']:.2f}")
    #     print(f"Peak power: {results['peak_power']:.2f}")
    #     print("\n=== RESTRICTED PRICING RESULTS ===")
    #     print(f"Electricity prices: {prices['electricity']}")
    #     print(f"Heat prices: {prices['heat']}")
    #     print(f"hydro prices: {prices['hydro']}")
    # else:
    #     print(f"Optimization failed with status: {status}")

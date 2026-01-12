from pyscipopt import Model, quicksum
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from data_generator import setup_lem_parameters
from pyscipopt import SCIP_PARAMSETTING

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
def create_tou_import_prices(smp_prices_eur, time_periods):
    """
    한국의 TOU 요금제를 기반으로 import 가격 생성
    
    한국전력 산업용(을) 고압A 선택II 요금제 기준:
    - 경부하: 23:00~09:00 (기본요금 대비 약 60-70%)
    - 중간부하: 09:00~10:00, 12:00~13:00, 17:00~23:00 (기본요금)
    - 최대부하: 10:00~12:00, 13:00~17:00 (기본요금 대비 약 140-180%)
    
    SMP는 실시간 변동가격이지만, TOU는 고정된 시간대별 요금
    일반적으로 TOU는 SMP 평균가격에 안정성 프리미엄(10-20%)을 더한 수준
    """
    
    # SMP 평균가격 계산
    avg_smp = np.mean(smp_prices_eur)
    
    # TOU 기본요금 = SMP 평균 + 15% 안정성 프리미엄
    tou_base = avg_smp * 1.15
    
    tou_import_prices = []
    
    for t in time_periods:
        # 한국 TOU 시간대 구분 (0-23시 기준)
        if t >= 23 or t < 9:  # 경부하 시간대 (23:00~09:00)
            tou_multiplier = 0.65  # 기본요금의 65%
            
        elif (10 <= t < 12) or (13 <= t < 17):  # 최대부하 시간대
            # 여름철(7-8월) 기준으로 더 높은 요금 적용
            tou_multiplier = 1.60  # 기본요금의 160%
            
        else:  # 중간부하 시간대 (9-10, 12-13, 17-23)
            tou_multiplier = 1.00  # 기본요금
        
        # TOU 가격 계산
        tou_price = tou_base * tou_multiplier
        
        # 최소/최대 가격 제한 (SMP의 50%~200% 범위)
        min_price = min(smp_prices_eur) * 0.5
        max_price = max(smp_prices_eur) * 2.0
        tou_price = np.clip(tou_price, min_price, max_price)
        
        tou_import_prices.append(tou_price)
    
    return tou_import_prices
def generate_market_price(parameters, time_periods, korean_prices_eur, h2_prices_eur):
    """
    한국 전력시장 구조 반영:
    - Export: SMP (계통한계가격) - 발전사업자가 받는 가격
    - Import: TOU 요금제 - 수요자가 지불하는 가격
    """
    
    # 1. Export price = SMP (이미 있는 데이터 사용)
    for t in time_periods:
        # CSV에서 시간 인덱스 조정
        csv_hour = t + 1 if t < 23 else 0
        base_price = korean_prices_eur[csv_hour]
        if 0<=t<=5: # 심야: 더 저렴하게
            price_multiplier = 0.7
        elif 10<=t<=15: # 태양광 시간: 매우 저렴
            price_multiplier = 0.5
        elif 17<=t<=20: # 저녁 피크: 더 비싸게
            price_multiplier = 1.5
        else:
            price_multiplier = 1.0
        adjusted_price = base_price * price_multiplier
        # Export는 SMP 그대로 사용 (발전사업자 정산가격)
        parameters[f'pi_E_gri_export_{t}'] = adjusted_price
    
    # 2. Import price = TOU 요금제 생성
    tou_import_prices = create_tou_import_prices(korean_prices_eur, time_periods)
    
    for t in time_periods:
        parameters[f'pi_E_gri_import_{t}'] = tou_import_prices[t]
    
    # 3. 가격 비교 출력
    print("\n" + "="*80)
    print("한국 전력시장 가격 체계 (SMP vs TOU)")
    print("="*80)
    print(f"{'시간':^6} | {'SMP (Export)':^15} | {'TOU (Import)':^15} | {'차이':^10} | {'TOU 구간':^15}")
    print(f"{'':^6} | {'(EUR/MWh)':^15} | {'(EUR/MWh)':^15} | {'(%)':^10} | {'':^15}")
    print("-"*80)
    
    for t in time_periods:
        export_price = parameters[f'pi_E_gri_export_{t}']
        import_price = parameters[f'pi_E_gri_import_{t}']
        diff_pct = ((import_price - export_price) / export_price * 100) if export_price > 0 else 0
        
        # TOU 시간대 구분
        if t >= 23 or t < 9:
            tou_period = "경부하(Off-peak)"
        elif (10 <= t < 12) or (13 <= t < 17):
            tou_period = "최대부하(Peak)"
        else:
            tou_period = "중간부하(Mid)"
        
        print(f"{t:^6} | {export_price:^15.2f} | {import_price:^15.2f} | {diff_pct:^10.1f} | {tou_period:^15}")
    
    # 통계 요약
    print("-"*80)
    avg_smp = np.mean([parameters[f'pi_E_gri_export_{t}'] for t in time_periods])
    avg_tou = np.mean([parameters[f'pi_E_gri_import_{t}'] for t in time_periods])
    
    print(f"\n[요약 통계]")
    print(f"SMP (Export) 평균: {avg_smp:.2f} EUR/MWh")
    print(f"TOU (Import) 평균: {avg_tou:.2f} EUR/MWh")
    print(f"평균 스프레드: {avg_tou - avg_smp:.2f} EUR/MWh ({(avg_tou/avg_smp - 1)*100:.1f}%)")
    
    # 시간대별 평균
    off_peak_hours = list(range(23, 24)) + list(range(0, 9))
    peak_hours = list(range(10, 12)) + list(range(13, 17))
    mid_hours = [h for h in time_periods if h not in off_peak_hours and h not in peak_hours]
    
    off_peak_avg = np.mean([parameters[f'pi_E_gri_import_{t}'] for t in off_peak_hours])
    peak_avg = np.mean([parameters[f'pi_E_gri_import_{t}'] for t in peak_hours])
    mid_avg = np.mean([parameters[f'pi_E_gri_import_{t}'] for t in mid_hours])
    
    print(f"\n[TOU 시간대별 평균]")
    print(f"경부하 (23-09시): {off_peak_avg:.2f} EUR/MWh")
    print(f"중간부하: {mid_avg:.2f} EUR/MWh")
    print(f"최대부하 (10-12, 13-17시): {peak_avg:.2f} EUR/MWh")
    print(f"Peak/Off-peak 비율: {peak_avg/off_peak_avg:.2f}x")

    # 2. HEAT - 기존 코사인 패턴 유지
    for t in time_periods:
        # 기존 방식 그대로: 아침/저녁 높은 수요
        heat_demand_factor = 1.0 + 0.3 * np.cos(2 * np.pi * (t - 7) / 24)
        parameters[f'pi_H_gri_export_{t}'] = 0.25 * heat_demand_factor
        
        # Import는 Export에 TOU 승수 적용
        # 열 TOU: 피크(6-9, 17-23시) 1.2x, 심야(23-6시) 0.8x, 주간 1.0x
        if 6 <= t < 9 or 17 <= t < 23:
            tou_multiplier = 1.2
        elif 23 <= t or t < 6:
            tou_multiplier = 0.8
        else:
            tou_multiplier = 1.0
        
        # 기본 20% 마진 + TOU 조정
        parameters[f'pi_H_gri_import_{t}'] = parameters[f'pi_H_gri_export_{t}'] * 1.2 * tou_multiplier
    
    # 3. HYDROGEN - 기존 유지
    for t in time_periods:
        h2_price = h2_prices_eur[t]
        parameters[f'pi_G_gri_export_{t}'] = h2_price
        parameters[f'pi_G_gri_import_{t}'] = h2_price * 1.2
    return parameters
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
        base_h2_price = 2.1 *1.5
        
        # 전력 가격에 반비례하는 조정 계수
        # 전력이 싸면 수소 가격 낮춤 (수소 생산 유도)
        # 전력이 비싸면 수소 가격 높임 (전력 판매 유도)
        if avg_elec > 0:
            adjustment = 1.0 - 0.3 * (elec_price - avg_elec) / avg_elec
        else:
            adjustment = 1.0
        adjustment = 1.0
        # 최종 수소 가격 (€1.5 ~ €4.5/kg 범위)
        h2_price = base_h2_price * adjustment
        h2_price = np.clip(h2_price, 1.5, 5.0)
        
        h2_prices.append(h2_price)
    
    return h2_prices
def create_heat_tou_import_prices(base_heat_prices, time_periods):
    """
    한국 지역난방 TOU 요금제 반영
    
    지역난방 시간대별 요금 특징:
    - 난방 수요는 주로 아침/저녁에 집중
    - 전기와 달리 열저장 특성으로 인해 변동폭이 작음
    - 계절요금제와 시간대별 요금제 병행
    
    시간대 구분:
    - 심야: 23:00~06:00 (기본요금의 70%)
    - 주간: 09:00~17:00 (기본요금의 90%)  
    - 피크: 06:00~09:00, 17:00~23:00 (기본요금의 120%)
    """
    
    # 기본 열요금 = 평균가격 + 10% 안정성 프리미엄
    avg_heat_price = np.mean(base_heat_prices)
    heat_tou_base = avg_heat_price * 1.10
    
    heat_tou_prices = []
    
    for t in time_periods:
        # 열수요 시간대별 구분
        if 23 <= t or t < 6:  # 심야 시간대
            tou_multiplier = 0.70  # 난방수요 낮음
            
        elif 6 <= t < 9 or 17 <= t < 23:  # 피크 시간대 (아침/저녁)
            tou_multiplier = 1.20  # 난방수요 최대
            
        else:  # 주간 시간대 (9-17시)
            tou_multiplier = 0.90  # 난방수요 중간
        
        heat_tou_price = heat_tou_base * tou_multiplier
        
        # 가격 범위 제한 (기본가격의 60%~150%)
        min_price = min(base_heat_prices) * 0.6
        max_price = max(base_heat_prices) * 1.5
        heat_tou_price = np.clip(heat_tou_price, min_price, max_price)
        
        heat_tou_prices.append(heat_tou_price)
    
    return heat_tou_prices
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
    # print(f"model status: {status}, time: {time}")
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
                 model_type: str = 'mip',
                 dwr: bool = False,
                 binary_values: Optional[Dict] = None):
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
        if model_type not in ('mip', 'mip_fix_binaries','lp'):
            raise ValueError("model_type must be either 'mip' or 'mip_fix_binaries' or 'lp', got: {}".format(model_type))
        self.model_type = model_type
        self.dwr = dwr
        self.binary_values = binary_values
        # Initialize model.data dictionary to store variables and constraints
        self.model.data = {"vars": {}, "cons": {}}
        
        # Sets definition based on slides
        # Use intersection with self.players to ensure these sets contain only relevant players, preserve list type
        self.players_with_renewables = [u for u in self.params.get('players_with_renewables', []) if u in self.players]
        self.players_with_electrolyzers = [u for u in self.params.get('players_with_electrolyzers', []) if u in self.players]
        self.players_with_heatpumps = [u for u in self.params.get('players_with_heatpumps', []) if u in self.players]
        self.players_with_elec_storage = [u for u in self.params.get('players_with_elec_storage', []) if u in self.players]
        self.players_with_hydro_storage = [u for u in self.params.get('players_with_hydro_storage', []) if u in self.players]
        self.players_with_heat_storage = [u for u in self.params.get('players_with_heat_storage', []) if u in self.players]
        self.players_with_nfl_elec_demand = [u for u in self.params.get('players_with_nfl_elec_demand', []) if u in self.players]
        self.players_with_nfl_hydro_demand = [u for u in self.params.get('players_with_nfl_hydro_demand', []) if u in self.players]
        self.players_with_nfl_heat_demand = [u for u in self.params.get('players_with_nfl_heat_demand', []) if u in self.players]
        self.players_with_fl_elec_demand = [u for u in self.params.get('players_with_fl_elec_demand', []) if u in self.players]
        self.players_with_fl_hydro_demand = [u for u in self.params.get('players_with_fl_hydro_demand', []) if u in self.players]
        self.players_with_fl_heat_demand = [u for u in self.params.get('players_with_fl_heat_demand', []) if u in self.players]
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
        
        # Initialize non-flexible demand fixing constraints
        self.elec_nfl_demand_cons = {}
        self.hydro_nfl_demand_cons = {}
        self.heat_nfl_demand_cons = {}

        # Initialize variables
        self._create_variables()
        self._create_constraints()
        
        # Store all variables and constraints in model.data
        self._store_model_data()
    
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
                                                        ub=self.params.get(f'e_E_cap', -np.inf), obj=-1*self.params.get(f'pi_E_gri_export_{t}', 0))
                    self.e_E_com[u,t] = self.model.addVar(vtype="C", name=f"e_E_com_{u}_{t}", lb=0, ub=1000)
                if u in self.U_H:
                    self.e_H_gri[u,t] = self.model.addVar(vtype="C", name=f"e_H_gri_{u}_{t}", lb=0,
                                                        ub=self.params.get(f'e_H_cap', -np.inf), obj=-1*self.params.get(f'pi_H_gri_export_{t}', 0))
                    self.e_H_com[u,t] = self.model.addVar(vtype="C", name=f"e_H_com_{u}_{t}", lb=0, ub=500)
                if u in self.U_G:
                    self.e_G_gri[u,t] = self.model.addVar(vtype="C", name=f"e_G_gri_{u}_{t}", lb=0,
                                                        ub=self.params.get(f'e_G_cap', -np.inf), obj=-1*self.params.get(f'pi_G_gri_export_{t}', 0))
                    self.e_G_com[u,t] = self.model.addVar(vtype="C", name=f"e_G_com_{u}_{t}", lb=0, ub=100)
                
                # Production variables (for renewables, heat pumps, electrolyzers) with capacity limits
                if u in self.players_with_renewables:  # Renewable generators
                    renewable_cap = self.params.get(f'renewable_cap_{u}_{t}', -np.inf)  # Default 200 kW, now time-dependent
                    c_res = self.params.get(f'c_res_{u}', 0)
                    self.p[u,'res',t] = self.model.addVar(vtype="C", name=f"p_res_{u}_{t}", 
                                                        lb=0, ub=renewable_cap, obj=c_res)
                if u in self.players_with_heatpumps:  # Heat pumps
                    hp_cap = self.params.get(f'hp_cap', -np.inf)  # Default 100 kW thermal
                    c_hp = self.params.get(f'c_hp_{u}', 0)
                    self.p[u,'hp',t] = self.model.addVar(vtype="C", name=f"p_hp_{u}_{t}", 
                                                       lb=0, ub=hp_cap, obj=c_hp)

                if u in self.players_with_electrolyzers:  # Electrolyzers
                    els_cap = self.params.get(f'els_cap', -np.inf)  # Default 1 MW
                    c_els = self.params.get(f'c_els_{u}', 0)
                    self.p[u,'els',t] = self.model.addVar(vtype="C", name=f"p_els_{u}_{t}", 
                                                        lb=0, obj=c_els)
                    self.els_d[u,t] = self.model.addVar(vtype="C", name=f"els_d_{u}_{t}", 
                                                        lb=0, ub=els_cap)                    
                    # Electrolyzer commitment variables
                    c_su_G = self.params.get(f'c_su_G_{u}', 0)
                    if self.model_type in ['mip', 'mip_fix_binaries']:
                        vartype = "C" if (self.model_type == 'mip_fix_binaries') else "B"
                        self.z_su[u,t] = self.model.addVar(vtype=vartype, name=f"z_su_{u}_{t}", obj=c_su_G)
                        self.z_on[u,t] = self.model.addVar(vtype=vartype, name=f"z_on_{u}_{t}")
                        self.z_off[u,t] = self.model.addVar(vtype=vartype, name=f"z_off_{u}_{t}")
                        self.z_sb[u,t] = self.model.addVar(vtype=vartype, name=f"z_sb_{u}_{t}")
            

                # Non-flexible demand variables
                if u in self.players_with_nfl_elec_demand:
                    nfl_elec_demand_t = self.params.get(f'd_E_nfl_{u}_{t}', 0)
                    self.nfl_d[u,'elec',t] = self.model.addVar(vtype="C", name=f"d_elec_nfl_{u}_{t}")
                    cons = self.model.addCons(self.nfl_d[u,'elec',t] == nfl_elec_demand_t, name=f"fix_nfl_d_elec_{u}_{t}")
                    self.elec_nfl_demand_cons[f"elec_nfl_demand_cons_{u}_{t}"] = cons
                    self.i_E_gri[u,t] = self.model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_E_cap', -np.inf), obj=self.params.get(f'pi_E_gri_import_{t}', 0))
                    self.i_E_com[u,t] = self.model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0, ub=self.params.get(f'i_E_cap', -np.inf))
                if u in self.players_with_nfl_hydro_demand:
                    nfl_hydro_demand_t = self.params.get(f'd_G_nfl_{u}_{t}', 0)
                    self.nfl_d[u,'hydro',t] = self.model.addVar(vtype="C", name=f"d_hydro_nfl_{u}_{t}")
                    cons = self.model.addCons(self.nfl_d[u,'hydro',t] == nfl_hydro_demand_t, name=f"fix_nfl_d_hydro_{u}_{t}")
                    self.hydro_nfl_demand_cons[f"hydro_nfl_demand_cons_{u}_{t}"] = cons
                    self.i_G_gri[u,t] = self.model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_G_cap', 100), obj=self.params.get(f'pi_G_gri_import_{t}', 0))
                    self.i_G_com[u,t] = self.model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0, ub=100)
                if u in self.players_with_nfl_heat_demand:
                    nfl_heat_demand_t = self.params.get(f'd_H_nfl_{u}_{t}', 0)
                    self.nfl_d[u,'heat',t] = self.model.addVar(vtype="C", name=f"d_heat_nfl_{u}_{t}")
                    cons = self.model.addCons(self.nfl_d[u,'heat',t] == nfl_heat_demand_t, name=f"fix_nfl_d_heat_{u}_{t}")
                    self.heat_nfl_demand_cons[f"heat_nfl_demand_cons_{u}_{t}"] = cons
                    self.i_H_gri[u,t] = self.model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", lb=0,
                                                     ub=self.params.get(f'i_H_cap', 500), obj=self.params.get(f'pi_H_gri_import_{t}', 0))
                    self.i_H_com[u,t] = self.model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0, ub=500)
                
                # Flexible demand variables
                if u in self.players_with_fl_elec_demand:
                    fl_elec_demand_cap = 1 ## Total electrolyzer power consumption capacity [MW]
                    self.fl_d[u,'elec',t] = self.model.addVar(vtype="C", name=f"d_elec_{u}_{t}", 
                                                       lb=0.0, ub=fl_elec_demand_cap)
                    self.i_E_gri[u,t] = self.model.addVar(vtype="C", name=f"i_E_gri_{u}_{t}", obj=self.params.get(f'pi_E_gri_import_{t}', 0))
                    self.i_E_com[u,t] = self.model.addVar(vtype="C", name=f"i_E_com_{u}_{t}", lb=0, ub=1000)
                if u in self.players_with_fl_hydro_demand:
                    fl_hydro_demand_cap = 10**6
                    self.fl_d[u,'hydro',t] = self.model.addVar(vtype="C", name=f"d_hydro_{u}_{t}", 
                                                       lb=0.0, ub=fl_hydro_demand_cap)
                    self.i_G_gri[u,t] = self.model.addVar(vtype="C", name=f"i_G_gri_{u}_{t}", obj=self.params.get(f'pi_G_gri_import_{t}', 0))
                    self.i_G_com[u,t] = self.model.addVar(vtype="C", name=f"i_G_com_{u}_{t}", lb=0, ub=100)
                if u in self.players_with_fl_heat_demand:
                    fl_heat_demand_cap = 10**6
                    self.fl_d[u,'heat',t] = self.model.addVar(vtype="C", name=f"d_heat_{u}_{t}", 
                                                       lb=0.0, ub=fl_heat_demand_cap)
                    self.i_H_gri[u,t] = self.model.addVar(vtype="C", name=f"i_H_gri_{u}_{t}", obj=self.params.get(f'pi_H_gri_import_{t}', 0))
                    self.i_H_com[u,t] = self.model.addVar(vtype="C", name=f"i_H_com_{u}_{t}", lb=0, ub=500)
                # Storage variables by type with capacity constraints
                # Electricity storage
                if u in self.players_with_elec_storage:
                    storage_capacity = self.params.get(f'storage_capacity_E', -np.inf)  # kWh capacity
                    storage_power = storage_capacity * self.params.get(f'storage_power_E', -np.inf)
                    nu_ch = self.params.get('nu_ch_E', np.inf)
                    nu_dis = self.params.get('nu_dis_E', np.inf)
                    c_sto_E = self.params.get("c_sto_E", 0.0)
                    self.b_dis_E[u,t] = self.model.addVar(vtype="C", name=f"b_dis_E_{u}_{t}", 
                                                        lb=0, ub=storage_power, obj=c_sto_E*(1/nu_dis))
                    self.b_ch_E[u,t] = self.model.addVar(vtype="C", name=f"b_ch_E_{u}_{t}", 
                                                       lb=0, ub=storage_power, obj=c_sto_E*nu_ch)
                    self.s_E[u,t] = self.model.addVar(vtype="C", name=f"s_E_{u}_{t}", 
                                                    lb=0, ub=storage_capacity)
                
                # hydro storage
                if u in self.players_with_hydro_storage:
                    # 수소는 kg 단위
                    storage_capacity_G = self.params.get(f'storage_capacity_G', -np.inf)
                    storage_power_G = storage_capacity_G * self.params.get(f'storage_power_G', -np.inf)
                    nu_ch_G = self.params.get('nu_ch_G', np.inf)
                    nu_dis_G = self.params.get('nu_dis_G', np.inf)
                    c_sto_G = self.params.get("c_sto_G", np.inf)

                    self.b_dis_G[u,t] = self.model.addVar(vtype="C", name=f"b_dis_G_{u}_{t}", 
                                                        lb=0, ub=storage_power_G, obj=c_sto_G*(1/nu_dis_G))
                    self.b_ch_G[u,t] = self.model.addVar(vtype="C", name=f"b_ch_G_{u}_{t}", 
                                                       lb=0, ub=storage_power_G, obj=c_sto_G*nu_ch_G)
                    self.s_G[u,t] = self.model.addVar(vtype="C", name=f"s_G_{u}_{t}", 
                                                    lb=0, ub=storage_capacity_G)
                
                # Heat storage
                if u in self.players_with_heat_storage:
                    c_sto_H = self.params.get("c_sto_H", np.inf)
                    nu_ch_H = self.params.get('nu_ch_H', np.inf)
                    nu_dis_H = self.params.get('nu_dis_H', np.inf)
                    storage_capacity_heat = self.params.get('storage_capacity_heat', -np.inf)
                    storage_power_heat = storage_capacity_heat * self.params.get("storage_power_heat", -np.inf)
                    self.b_dis_H[u,t] = self.model.addVar(vtype="C", name=f"b_dis_H_{u}_{t}", 
                                                        lb=0, ub=storage_power_heat, obj=c_sto_H*(1/nu_dis_H))
                    self.b_ch_H[u,t] = self.model.addVar(vtype="C", name=f"b_ch_H_{u}_{t}", 
                                                       lb=0, ub=storage_power_heat, obj=c_sto_H*nu_ch_H)
                    self.s_H[u,t] = self.model.addVar(vtype="C", name=f"s_H_{u}_{t}", 
                                                    lb=0, ub=storage_capacity_heat)
    
    def _create_constraints(self):
        """Create constraints based on slides 9-15"""
        
        # Electricity flow balance constraints (slide 9)
        self._add_electricity_constraints()
        
        # Heat flow balance constraints (slide 12)
        self._add_heat_constraints()
        
        # hydro flow balance constraints (slide 13-14)
        self._add_hydro_constraints()

        # Non-convex Operation Constraints
        if self.model_type == 'mip':
            self._add_hydro_nonconvex_cons_mip()
        elif self.model_type == 'lp':
            self._add_hydro_nonconvex_cons_lp_relax()
        elif self.model_type == 'mip_fix_binaries':
            self._add_hydro_nonconvex_cons_mip()
            self._fix_binaries()
    def _fix_binaries(self):
        # If binary_values are provided, add constraints to fix binary variables
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if ('z_su', u, t) in self.binary_values:
                    cons = self.model.addCons(
                        self.z_su[u,t] == self.binary_values[('z_su', u, t)],
                        name=f"fix_z_su_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"fix_z_su_{u}_{t}"] = cons
                
                if ('z_on', u, t) in self.binary_values:
                    cons = self.model.addCons(
                        self.z_on[u,t] == self.binary_values[('z_on', u, t)],
                        name=f"fix_z_on_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"fix_z_on_{u}_{t}"] = cons
                
                if ('z_off', u, t) in self.binary_values:
                    cons = self.model.addCons(
                        self.z_off[u,t] == self.binary_values[('z_off', u, t)],
                        name=f"fix_z_off_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"fix_z_off_{u}_{t}"] = cons
                
                if ('z_sb', u, t) in self.binary_values:
                    cons = self.model.addCons(
                        self.z_sb[u,t] == self.binary_values[('z_sb', u, t)],
                        name=f"fix_z_sb_{u}_{t}"
                    )
                    self.electrolyzer_cons[f"fix_z_sb_{u}_{t}"] = cons
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
        self.model.data["cons"]["elec_nfl_demand"] = self.elec_nfl_demand_cons
        self.model.data["cons"]["hydro_nfl_demand"] = self.hydro_nfl_demand_cons
        self.model.data["cons"]["heat_nfl_demand"] = self.heat_nfl_demand_cons
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
                initial_soc = self.params.get(f'initial_soc_E', np.inf)  # Default 50% SOC
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
                    nu_cop = self.params.get(f'nu_cop_{u}', np.inf)
                    cons = self.model.addCons(
                        nu_cop * self.fl_d.get((u,'elec',t),0) == self.p.get((u,'hp',t),0),
                        name=f"heatpump_coupling_{u}_{t}"
                    )
                    self.heatpump_cons[f"heatpump_coupling_{u}_{t}"] = cons
        
        # Heat storage SOC transition with special 23→0 transition
        for u in self.players:
            if u in self.players_with_heat_storage:
                nu_ch = self.params.get('nu_ch_H', np.inf)
                nu_dis = self.params.get('nu_dis_H', np.inf)
                nu_loss = self.params.get('nu_loss_H', np.inf)
                # Set initial SOC at 6시 (논리적 시작점)
                if (u,6) in self.s_H:
                    initial_soc = self.params.get(f'initial_soc_H', np.inf)
                    cons = self.model.addCons(self.s_H[u,6] == initial_soc, name=f"initial_soc_H_{u}")
                    self.storage_cons[f"initial_soc_H_{u}"] = cons
                
                # 일반적인 SOC transition (1시~23시)
                for t in range(1, 24):
                    if (u,t) in self.s_H and (u,t-1) in self.s_H:
                        cons = self.model.addCons(
                            self.s_H[u,t] == (1-nu_loss)*self.s_H[u,t-1] + nu_ch * self.b_ch_H[u,t] - (1/nu_dis) * self.b_dis_H[u,t],
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
                community_balance = quicksum(self.i_H_com.get((u,t),0) - self.e_H_com.get((u,t),0) for u in self.players)
                cons = self.model.addCons(community_balance == 0, name=f"community_heat_balance_{t}")
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
        # hydro storage SOC transition with special 23→0 transition
            if u in self.players_with_hydro_storage:
                nu_ch = self.params.get('nu_ch', 0.9)
                nu_dis = self.params.get('nu_dis', 0.9)            
                initial_soc = self.params.get(f'initial_soc_G', np.inf)
                
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
                
                # 특별한 23→0시 transition (심야 충전 → 아침 방전)
                if (u,0) in self.s_G and (u,23) in self.s_G:
                    cons = self.model.addCons(
                        self.s_G[u,0] == self.s_G[u,23] + nu_ch * self.b_ch_G[u,0] - (1/nu_dis) * self.b_dis_G[u,0],
                        name=f"soc_transition_G_{u}_23_to_0"
                    )
                    self.storage_cons[f"soc_transition_G_{u}_23_to_0"] = cons
        

        if not self.dwr:
        # Community hydro balance
            for t in self.time_periods:
                community_balance = quicksum(self.e_G_com.get((u,t),0) - self.i_G_com.get((u,t),0) for u in self.players)
                cons = self.model.addCons(community_balance == 0, name=f"community_hydro_balance_{t}")
                self.community_hydro_balance_cons[f"community_hydro_balance_{t}"] = cons
    
    def _add_hydro_nonconvex_cons_mip(self):
        # Electrolyzer coupling constraint (constraint 15)
        for u in self.players_with_electrolyzers:
            els_cap = self.params.get(f'els_cap', -np.inf)
            C_sb = self.params.get(f'C_sb', -np.inf)
            C_min = self.params.get(f'C_min', -np.inf)
            for t in self.time_periods:
                phi1_1 = self.params.get(f'phi1_1', -np.inf)
                phi0_1 = self.params.get(f'phi0_1', -np.inf)
                phi1_2 = self.params.get(f'phi1_2', -np.inf)
                phi0_2 = self.params.get(f'phi0_2', -np.inf)
            
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
            for t in self.time_periods:
                # Constraint 17: exactly one state
                cons = self.model.addCons(
                    self.z_on[u,t] + self.z_off[u,t] + self.z_sb[u,t] == 1,
                    name=f"electrolyzer_state_{u}_{t}"
                )
                self.electrolyzer_cons['state', u, t] = cons
                
                # Constraints 18-19: production bounds
                els_cap = self.params.get(f'els_cap', -np.inf)
                C_sb = self.params.get(f'C_sb', -np.inf)
                C_min = self.params.get(f'C_min', -np.inf)
                
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
                    ## 아래는 왜 필요? 어차피 z_off + z_on + z_sb = 1 인데
                    # cons = self.model.addCons(
                    #     self.z_su[u,t] <= 0.0,
                    #     name=f"electrolyzer_initial_su_{u}_{t}"
                    # )
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
            min_down = self.params.get('min_down_time', -1)
            # for t in range(1, len(self.time_periods)):  # t ∈ T \ {1}
            for t in [tau for tau in self.time_periods if tau != 6]:
            # t시점에 off로 전환되었는지 확인 (z_off_t - z_off_{t-1})
            # 만약 전환되었다면, 다음 min_down 기간 동안 off 유지
                down_time_idx = [tau for tau in range(t+1, t + min_down+1)]
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
    def _add_hydro_nonconvex_cons_lp_relax(self):
        # Electrolyzer coupling constraint (constraint 15)
        for u in self.players_with_electrolyzers:
            els_cap = self.params.get(f'els_cap', -np.inf)
            for t in self.time_periods:
                phi1_1 = self.params.get(f'phi1_1', -np.inf)
                phi0_1 = self.params.get(f'phi0_1', -np.inf)
                phi1_2 = self.params.get(f'phi1_2', -np.inf)
                phi0_2 = self.params.get(f'phi0_2', -np.inf)
                try:
                    cons = self.model.addCons(
                        self.p.get((u,'els',t),0) <= phi1_1 * self.els_d.get((u,t),0) + phi0_1,
                        name=f"electrolyzer_production_curve_1_{u}_{t}"
                    )
                except:
                    print(f"Error adding constraint: electrolyzer_production_curve_1_{u}_{t}")
                    continue
                self.electrolyzer_cons['production_curve_1', u, t] = cons
                cons = self.model.addCons(
                    self.p.get((u,'els',t),0) <= phi1_2 * self.els_d.get((u,t),0) + phi0_2,
                    name=f"electrolyzer_production_curve_2_{u}_{t}"
                )
                self.electrolyzer_cons['production_curve_2', u, t] = cons
                cons = self.model.addCons(
                    self.els_d.get((u,t),0) - els_cap  <= 0.0,
                    name=f"electrolyzer_max_power_consumption_{u}_{t}"
                )
                self.electrolyzer_cons[f"electrolyzer_max_power_consumption_{u}_{t}"] = cons
                
                # Constraint   : Power consumption coupling
                cons = self.model.addCons(
                    self.fl_d[u,'elec',t] == self.els_d[u,t],
                    name=f"electrolyzer_power_consumption_{u}_{t}"
                )
                self.electrolyzer_cons[f"electrolyzer_power_consumption_{u}_{t}"] = cons
    def solve(self):
        """Solve the optimization model"""
        # self.model.setParam('lp/iterlim', 100)
        self.model.optimize()
        return self.model.getStatus()
    
    def solve_complete_model(self, analyze_revenue=True):
        """
        Solve the complete optimization model and analyze revenue by resource type
        
        Returns:
            tuple: (status, results, revenue_analysis)
                - status: optimization status
                - results: optimization results dictionary
                - revenue_analysis: dictionary with revenue breakdown by resource type
        """
        print("Solving complete optimization model...")
        if self.model_type in ['lp', 'mip_fix_binaries']:
            print('Relaxed LP model for Restricted Pricing')
            self.model.relax()
            self.model.setPresolve(SCIP_PARAMSETTING.OFF)
            self.model.setHeuristics(SCIP_PARAMSETTING.OFF)
            self.model.disablePropagation()
            self.model.setSeparating(SCIP_PARAMSETTING.OFF)
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
        if analyze_revenue:
            # Analyze electrolyzer operation
            electrolyzer_operation = self._analyze_electrolyzer_operations(results)        
            # Analyze revenue by resource type
            revenue_analysis = self._analyze_revenue_by_resource(results)
            # Analyze energy flows
            flow_analysis = self._analyze_energy_flows(results)

        if self.model_type == 'mip':
            ip_status, ip_results, prices = self.solve_with_restricted_pricing()
            if ip_status != "optimal":
                print(f"IP optimization failed with status: {ip_status}")
                return ip_status, None, None, None
        elif self.model_type == 'lp':
            ## solved LP model의 community balance constraints들을 순회하며 shadow price를 추출하여 prices로 저장
            prices = {
            'electricity': {},
            'heat': {},
            'hydro': {}
            }
            for t in self.time_periods:
                # Get dual multipliers for community balance constraints
                elec_cons = self.community_elec_balance_cons[f"community_elec_balance_{t}"]
                heat_cons = self.community_heat_balance_cons[f"community_heat_balance_{t}"]
                hydro_cons = self.community_hydro_balance_cons[f"community_hydro_balance_{t}"]
                
                pi = self.model.getDualsolLinear(self.model.getTransformedCons(elec_cons))
                prices['electricity'][t] = np.abs(pi)#np.round(np.abs(pi), 2)
                pi = self.model.getDualsolLinear(self.model.getTransformedCons(heat_cons))
                prices['heat'][t] = np.abs(pi)#np.round(np.abs(pi), 2)
                pi = self.model.getDualsolLinear(self.model.getTransformedCons(hydro_cons))
                prices['hydro'][t] = np.abs(pi)#np.round(np.abs(pi), 2)        
        else:
            raise ValueError("a function <solve_complete_model> has a model_type must be either 'mip' or 'lp', got: {}".format(self.model_type))
        
        import pandas as pd
        # 시간별 가격들을 리스트로 만들면서 동시에 print
        price_records = []
        for t in self.time_periods:
            row = {
                "time": t,
                "electricity": prices["electricity"][t],
                "heat": prices["heat"][t],
                "hydro": prices["hydro"][t]
            }
            price_records.append(row)
            print(f"{t:6d} {row['electricity']:12.4f} {row['heat']:12.4f} {row['hydro']:12.4f}")
        price_df = pd.DataFrame(price_records)
        price_df.to_csv(f"community_prices_{self.model_type}.csv", index=False)
        print(f"✓ Community prices by time exported to 'community_prices_{self.model_type}.csv'")
        e_import_prices = [self.params[f'pi_E_gri_import_{t}'] for t in self.time_periods]
        e_export_prices = [self.params[f'pi_E_gri_export_{t}'] for t in self.time_periods]
        g_import_prices = [self.params[f'pi_G_gri_import_{t}'] for t in self.time_periods]
        g_export_prices = [self.params[f'pi_G_gri_export_{t}'] for t in self.time_periods]
        h_import_prices = [self.params[f'pi_H_gri_import_{t}'] for t in self.time_periods]
        h_export_prices = [self.params[f'pi_H_gri_export_{t}'] for t in self.time_periods]
        import matplotlib.pyplot as plt
        
        if analyze_revenue:
            # matplotlib 백엔드를 Agg로 설정 (GUI 없이 파일 저장용)
            plt.switch_backend('Agg')
            
            print("Creating and saving individual price plots...")
            
            # 1. Electricity Price Plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_periods, [prices['electricity'][t] for t in self.time_periods], 
                    label='Community Elec Price', marker='o', linewidth=2, markersize=6)
            plt.plot(self.time_periods, e_import_prices, 
                    label='Grid Import Price', linestyle='--', marker='x', linewidth=2, markersize=6)
            plt.plot(self.time_periods, e_export_prices, 
                    label='Grid Export Price', linestyle=':', marker='s', linewidth=2, markersize=6)
            plt.ylabel('Electricity Price (€/kWh)', fontsize=12)
            plt.title('Electricity Prices by Time Period', fontsize=14, fontweight='bold')
            plt.xlabel('Time Period (Hour)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(self.time_periods)
            plt.tight_layout()
            plt.savefig('electricity_prices.png', dpi=300, bbox_inches='tight')
            print("✓ Electricity price plot saved as 'electricity_prices.png'")
            plt.close()
            
            # 2. Heat Price Plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_periods, [prices['heat'][t] for t in self.time_periods], 
                    label='Community Heat Price', marker='o', linewidth=2, markersize=6)
            plt.plot(self.time_periods, h_import_prices, 
                    label='Grid Import Price', linestyle='--', marker='x', linewidth=2, markersize=6)
            plt.plot(self.time_periods, h_export_prices, 
                    label='Grid Export Price', linestyle=':', marker='s', linewidth=2, markersize=6)
            plt.ylabel('Heat Price (€/kWh)', fontsize=12)
            plt.title('Heat Prices by Time Period', fontsize=14, fontweight='bold')
            plt.xlabel('Time Period (Hour)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(self.time_periods)
            plt.tight_layout()
            plt.savefig('heat_prices.png', dpi=300, bbox_inches='tight')
            print("✓ Heat price plot saved as 'heat_prices.png'")
            plt.close()
            
            # 3. Hydrogen Price Plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_periods, [prices['hydro'][t] for t in self.time_periods], 
                    label='Community Hydro Price', marker='o', linewidth=2, markersize=6)
            plt.plot(self.time_periods, g_import_prices, 
                    label='Grid Import Price', linestyle='--', marker='x', linewidth=2, markersize=6)
            plt.plot(self.time_periods, g_export_prices, 
                    label='Grid Export Price', linestyle=':', marker='s', linewidth=2, markersize=6)
            plt.ylabel('Hydrogen Price (€/kg)', fontsize=12)
            plt.title('Hydrogen Prices by Time Period', fontsize=14, fontweight='bold')
            plt.xlabel('Time Period (Hour)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(self.time_periods)
            plt.tight_layout()
            plt.savefig('hydrogen_prices.png', dpi=300, bbox_inches='tight')
            print("✓ Hydrogen price plot saved as 'hydrogen_prices.png'")
            plt.close()
            
            # 4. Combined Price Plot (all three energy types)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # Electricity
            ax1.plot(self.time_periods, [prices['electricity'][t] for t in self.time_periods], 
                    label='Community Elec Price', marker='o', linewidth=2, markersize=6)
            ax1.plot(self.time_periods, e_import_prices, 
                    label='Grid Import Price', linestyle='--', marker='x', linewidth=2, markersize=6)
            ax1.plot(self.time_periods, e_export_prices, 
                    label='Grid Export Price', linestyle=':', marker='s', linewidth=2, markersize=6)
            ax1.set_ylabel('Electricity Price (€/kWh)', fontsize=12)
            ax1.set_title('Electricity Prices', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Heat
            ax2.plot(self.time_periods, [prices['heat'][t] for t in self.time_periods], 
                    label='Community Heat Price', marker='o', linewidth=2, markersize=6)
            ax2.plot(self.time_periods, h_import_prices, 
                    label='Grid Import Price', linestyle='--', marker='x', linewidth=2, markersize=6)
            ax2.plot(self.time_periods, h_export_prices, 
                    label='Grid Export Price', linestyle=':', marker='s', linewidth=2, markersize=6)
            ax2.set_ylabel('Heat Price (€/kWh)', fontsize=12)
            ax2.set_title('Heat Prices', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Hydrogen
            ax3.plot(self.time_periods, [prices['hydro'][t] for t in self.time_periods], 
                    label='Community Hydro Price', marker='o', linewidth=2, markersize=6)
            ax3.plot(self.time_periods, g_import_prices, 
                    label='Grid Import Price', linestyle='--', marker='x', linewidth=2, markersize=6)
            ax3.plot(self.time_periods, g_export_prices, 
                    label='Grid Export Price', linestyle=':', marker='s', linewidth=2, markersize=6)
            ax3.set_ylabel('Hydrogen Price (€/kg)', fontsize=12)
            ax3.set_title('Hydrogen Prices', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time Period (Hour)', fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(self.time_periods)
            
            plt.tight_layout()
            plt.savefig('combined_energy_prices.png', dpi=300, bbox_inches='tight')
            print("✓ Combined energy prices plot saved as 'combined_energy_prices.png'")
            plt.close()
            
            print("\nAll price plots have been saved successfully!")
            print("Files created:")
            print("  - electricity_prices.png")
            print("  - heat_prices.png") 
            print("  - hydrogen_prices.png")
            print("  - combined_energy_prices.png")

            # 5. Plot storage operation
            self.plot_storage_operation(results)
        return status, results, revenue_analysis if analyze_revenue else None, prices
    
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
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'net_profit': 0.0
        }
        
        c_E_sto = self.params.get('c_E_sto', 0.01)
        c_G_sto = self.params.get('c_G_sto', 0.01)
        c_H_sto = self.params.get('c_H_sto', 0.01)
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
                    revenue_analysis['electricity']['storage_cost'] += val * c_E_sto * nu_ch
        
        if 'b_dis_E' in results:
            for (u, t), val in results['b_dis_E'].items():
                if val > 0:
                    revenue_analysis['electricity']['storage_cost'] += val * c_E_sto * (1/nu_dis)
        
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
                    revenue_analysis['hydrogen']['storage_cost'] += val * c_G_sto * nu_ch
        
        if 'b_dis_G' in results:
            for (u, t), val in results['b_dis_G'].items():
                if val > 0:
                    revenue_analysis['hydrogen']['storage_cost'] += val * c_G_sto * (1/nu_dis)
        
        # Electrolyzer startup cost
        if 'z_su' in results:
            for (u, t), val in results['z_su'].items():
                if val > 0:
                    startup_cost = self.params.get(f'c_su_G_{u}', np.inf)
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
                    revenue_analysis['heat']['storage_cost'] += val * c_H_sto * nu_ch
        
        if 'b_dis_H' in results:
            for (u, t), val in results['b_dis_H'].items():
                if val > 0:
                    revenue_analysis['heat']['storage_cost'] += val * c_H_sto * (1/nu_dis)
        
        
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
        
        print(f"{'Time':^4} | {'='*15} ELECTRICITY {'='*15} | {'='*15} HYDROGEN {'='*15} | {'='*15} HEAT {'='*15}")
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
        
        print(f"{'Time':^4} | {'='*12} ELECTRICITY STORAGE {'='*12} | {'='*12} HYDROGEN STORAGE {'='*12} | {'='*14} HEAT STORAGE {'='*14}")
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
        print(f"  Total renewable generated:     {total_renewable:10.2f} MWh")
        print(f"  Total imported from grid:      {total_elec_import:10.2f} MWh")
        print(f"  Total exported to grid:        {total_elec_export:10.2f} MWh")
        print(f"  Total used for hydrogen:       {total_elec_to_hydro:10.2f} MWh")
        print(f"  Total used for heat:           {total_elec_to_heat:10.2f} MWh")
        print(f"  Net grid position:             {total_elec_import - total_elec_export:+10.2f} MWh (+ = import, - = export)")
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
        

    def plot_storage_operation(self, results, save_plots=True):
        """
        저장소 SOC 및 충방전 패턴 시각화
        
        Args:
            results: 최적화 결과 딕셔너리
            save_plots: True면 파일로 저장
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # matplotlib 백엔드 설정
        plt.switch_backend('Agg')
        
        # 시간대별 데이터 수집
        time_hours = list(range(6, 24)) + list(range(0, 6))  # 6시부터 시작
        
        # 전기 저장소 데이터
        elec_soc = []
        elec_charge = []
        elec_discharge = []
        
        # 수소 저장소 데이터
        hydro_soc = []
        hydro_charge = []
        hydro_discharge = []
        
        # 열 저장소 데이터
        heat_soc = []
        heat_charge = []
        heat_discharge = []
        
        # 데이터 추출
        for t in time_hours:
            # 전기
            e_soc = sum(results.get('s_E', {}).get((u, t), 0) 
                    for u in self.players_with_elec_storage)
            e_ch = sum(results.get('b_ch_E', {}).get((u, t), 0) 
                    for u in self.players_with_elec_storage)
            e_dis = sum(results.get('b_dis_E', {}).get((u, t), 0) 
                    for u in self.players_with_elec_storage)
            
            elec_soc.append(e_soc)
            elec_charge.append(e_ch)
            elec_discharge.append(e_dis)
            
            # 수소
            h_soc = sum(results.get('s_G', {}).get((u, t), 0) 
                    for u in self.players_with_hydro_storage)
            h_ch = sum(results.get('b_ch_G', {}).get((u, t), 0) 
                    for u in self.players_with_hydro_storage)
            h_dis = sum(results.get('b_dis_G', {}).get((u, t), 0) 
                    for u in self.players_with_hydro_storage)
            
            hydro_soc.append(h_soc)
            hydro_charge.append(h_ch)
            hydro_discharge.append(h_dis)
            
            # 열
            heat_soc_val = sum(results.get('s_H', {}).get((u, t), 0) 
                            for u in self.players_with_heat_storage)
            heat_ch = sum(results.get('b_ch_H', {}).get((u, t), 0) 
                        for u in self.players_with_heat_storage)
            heat_dis = sum(results.get('b_dis_H', {}).get((u, t), 0) 
                        for u in self.players_with_heat_storage)
            
            heat_soc.append(heat_soc_val)
            heat_charge.append(heat_ch)
            heat_discharge.append(heat_dis)
        
        # X축 라벨 (시간)
        x_labels = [f"{h:02d}" for h in time_hours]
        x_pos = np.arange(len(time_hours))
        
        # 1. 전기 저장소 플롯
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # SOC 플롯
        ax1.plot(x_pos, elec_soc, 'b-', linewidth=2, marker='o', markersize=4, label='SOC')
        ax1.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Max Capacity (2.0 MWh)')
        ax1.fill_between(x_pos, 0, elec_soc, alpha=0.3, color='blue')
        ax1.set_ylabel('SOC (MWh)', fontsize=12)
        ax1.set_title('Electricity Storage Operation', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_ylim([0, 2.2])
        # 여기 추가!
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels)
        # 충방전 플롯
        width = 0.35
        ax2.bar(x_pos - width/2, elec_charge, width, label='Charge', color='green', alpha=0.7)
        ax2.bar(x_pos + width/2, [-d for d in elec_discharge], width, label='Discharge', color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel('Power (MW)', fontsize=12)
        ax2.set_xlabel('Hour', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('electricity_storage_operation.png', dpi=300, bbox_inches='tight')
            print("✓ Electricity storage operation plot saved as 'electricity_storage_operation.png'")
        plt.close()
        
        # 2. 수소 저장소 플롯
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # SOC 플롯
        ax1.plot(x_pos, hydro_soc, 'g-', linewidth=2, marker='o', markersize=4, label='SOC')
        ax1.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Max Capacity (100 kg)')
        ax1.fill_between(x_pos, 0, hydro_soc, alpha=0.3, color='green')
        ax1.set_ylabel('SOC (kg)', fontsize=12)
        ax1.set_title('Hydrogen Storage Operation', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_ylim([0, 110])
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels)
        
        # 충방전 플롯
        width = 0.35
        ax2.bar(x_pos - width/2, hydro_charge, width, label='Charge', color='green', alpha=0.7)
        ax2.bar(x_pos + width/2, [-d for d in hydro_discharge], width, label='Discharge', color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel('Flow Rate (kg/h)', fontsize=12)
        ax2.set_xlabel('Hour', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('hydrogen_storage_operation.png', dpi=300, bbox_inches='tight')
            print("✓ Hydrogen storage operation plot saved as 'hydrogen_storage_operation.png'")
        plt.close()
        
        # 3. 열 저장소 플롯
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # SOC 플롯
        ax1.plot(x_pos, heat_soc, 'r-', linewidth=2, marker='o', markersize=4, label='SOC')
        ax1.axhline(y=2.0, color='darkred', linestyle='--', alpha=0.5, label='Max Capacity (2.0 MWh)')
        ax1.fill_between(x_pos, 0, heat_soc, alpha=0.3, color='red')
        ax1.set_ylabel('SOC (MWh)', fontsize=12)
        ax1.set_title('Heat Storage Operation', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_ylim([0, 2.2])
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels)
        # 충방전 플롯
        width = 0.35
        ax2.bar(x_pos - width/2, heat_charge, width, label='Charge', color='green', alpha=0.7)
        ax2.bar(x_pos + width/2, [-d for d in heat_discharge], width, label='Discharge', color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel('Power (MW)', fontsize=12)
        ax2.set_xlabel('Hour', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('heat_storage_operation.png', dpi=300, bbox_inches='tight')
            print("✓ Heat storage operation plot saved as 'heat_storage_operation.png'")
        plt.close()
        
        # 4. Combined Storage Operation (모든 저장소를 한 번에)
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 전기 저장소
        axes[0, 0].plot(x_pos, elec_soc, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].axhline(y=2.0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].fill_between(x_pos, 0, elec_soc, alpha=0.3, color='blue')
        axes[0, 0].set_ylabel('SOC (MWh)', fontsize=10)
        axes[0, 0].set_title('Electricity Storage SOC', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 2.2])
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(x_labels)
        width = 0.35
        axes[0, 1].bar(x_pos - width/2, elec_charge, width, label='Charge', color='green', alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, [-d for d in elec_discharge], width, label='Discharge', color='red', alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[0, 1].set_ylabel('Power (MW)', fontsize=10)
        axes[0, 1].set_title('Electricity Charge/Discharge', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(x_labels)
        # 수소 저장소
        axes[1, 0].plot(x_pos, hydro_soc, 'g-', linewidth=2, marker='o', markersize=4)
        axes[1, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].fill_between(x_pos, 0, hydro_soc, alpha=0.3, color='green')
        axes[1, 0].set_ylabel('SOC (kg)', fontsize=10)
        axes[1, 0].set_title('Hydrogen Storage SOC', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 110])
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(x_labels)
        axes[1, 1].bar(x_pos - width/2, hydro_charge, width, label='Charge', color='green', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, [-d for d in hydro_discharge], width, label='Discharge', color='red', alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[1, 1].set_ylabel('Flow Rate (kg/h)', fontsize=10)
        axes[1, 1].set_title('Hydrogen Charge/Discharge', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(x_labels)
        # 열 저장소
        axes[2, 0].plot(x_pos, heat_soc, 'r-', linewidth=2, marker='o', markersize=4)
        axes[2, 0].axhline(y=2.0, color='darkred', linestyle='--', alpha=0.5)
        axes[2, 0].fill_between(x_pos, 0, heat_soc, alpha=0.3, color='red')
        axes[2, 0].set_ylabel('SOC (MWh)', fontsize=10)
        axes[2, 0].set_title('Heat Storage SOC', fontsize=12, fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylim([0, 2.2])
        axes[2, 0].set_xlabel('Hour', fontsize=10)
        axes[2, 0].set_xticks(x_pos)
        axes[2, 0].set_xticklabels(x_labels)
        axes[2, 0].set_xticks(x_pos[::2])  # 2시간 간격
        axes[2, 0].set_xticklabels(x_labels[::2])
        
        axes[2, 1].bar(x_pos - width/2, heat_charge, width, label='Charge', color='green', alpha=0.7)
        axes[2, 1].bar(x_pos + width/2, [-d for d in heat_discharge], width, label='Discharge', color='red', alpha=0.7)
        axes[2, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[2, 1].set_ylabel('Power (MW)', fontsize=10)
        axes[2, 1].set_title('Heat Charge/Discharge', fontsize=12, fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend(loc='upper right', fontsize=8)
        axes[2, 1].set_xlabel('Hour', fontsize=10)
        axes[2, 1].set_xticks(x_pos[::2])
        axes[2, 1].set_xticklabels(x_labels[::2])
        
        plt.suptitle('Combined Storage Operation Overview', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('combined_storage_operation.png', dpi=300, bbox_inches='tight')
            print("✓ Combined storage operation plot saved as 'combined_storage_operation.png'")
        plt.close()
        
        print("\nAll storage operation plots have been saved successfully!")
        print("Files created:")
        print("  - electricity_storage_operation.png")
        print("  - hydrogen_storage_operation.png")
        print("  - heat_storage_operation.png")
        print("  - combined_storage_operation.png")
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
            summary['total_startup_cost'] = total_startups * self.params.get(f'c_su_G_{u}', np.inf)
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
        1. First get optimal binary variables from solved MIP model
        2. Create new LP model with fixed binary variables to get shadow prices
        
        Returns:
            tuple: (status, results, prices)
                - status: optimization status
                - results: optimization results dictionary
                - prices: dictionary with electricity, heat, hydro prices per time period
        """

        
        # Step 2: Extract optimal binary variable values
        print("\n" + "="*80)
        print("STEP 2: Extracting binary variable values...")
        print("="*80)
        
        binary_values = {}
        for u in self.players_with_electrolyzers:
            for t in self.time_periods:
                if (u,t) in self.z_su:
                    binary_values[('z_su', u, t)] = self.model.getVal(self.z_su[u,t])
                if (u,t) in self.z_on:
                    binary_values[('z_on', u, t)] = self.model.getVal(self.z_on[u,t])
                if (u,t) in self.z_off:
                    binary_values[('z_off', u, t)] = self.model.getVal(self.z_off[u,t])
                if (u,t) in self.z_sb:
                    binary_values[('z_sb', u, t)] = self.model.getVal(self.z_sb[u,t])
        
        print(f"✓ Extracted {len(binary_values)} binary variable values")
        
        # Step 3: Create new LP model with fixed binary variables
        print("\n" + "="*80)
        print("STEP 3: Creating LP model with fixed binary variables...")
        print("="*80)
        
        lp_model = LocalEnergyMarket(
            players=self.players,
            time_periods=self.time_periods,
            parameters=self.params,
            model_type='mip_fix_binaries',  # Don't use LP relaxation - we want continuous z_ variables but with fixed values
            dwr=self.dwr,
            binary_values=binary_values
        )
        from pyscipopt import SCIP_PARAMSETTING
        lp_model.model.setPresolve(SCIP_PARAMSETTING.OFF)
        lp_model.model.setHeuristics(SCIP_PARAMSETTING.OFF)
        lp_model.model.disablePropagation()
        lp_model.model.setSeparating(SCIP_PARAMSETTING.OFF)
        print("✓ LP model created with fixed binary variables")
        
        # Step 4: Solve LP model
        print("\n" + "="*80)
        print("STEP 4: Solving LP model to get shadow prices...")
        print("="*80)
        
        lp_model.model.optimize()
        lp_status, lp_results = solve_and_extract_results(lp_model.model)
        
        if lp_status != "optimal":
            print(f"LP optimization failed with status: {lp_status}")
            return lp_status, None, None
        print("✓ LP solved successfully")
        # Step 5: Extract shadow prices (dual multipliers) from community balance constraints
        print("\n" + "="*80)
        print("STEP 5: Extracting shadow prices from community balance constraints...")
        print("="*80)
        
        prices = {
            'electricity': {},
            'heat': {},
            'hydro': {}
        }
        
        for t in self.time_periods:
            # Get dual multipliers for community balance constraints
            elec_cons = lp_model.community_elec_balance_cons[f"community_elec_balance_{t}"]
            heat_cons = lp_model.community_heat_balance_cons[f"community_heat_balance_{t}"]
            hydro_cons = lp_model.community_hydro_balance_cons[f"community_hydro_balance_{t}"]
            
            pi = lp_model.model.getDualsolLinear(lp_model.model.getTransformedCons(elec_cons))
            prices['electricity'][t] = np.abs(pi) #np.round(np.abs(pi), 2)
            pi = lp_model.model.getDualsolLinear(lp_model.model.getTransformedCons(heat_cons))
            prices['heat'][t] = np.abs(pi) #np.round(np.abs(pi), 2)
            pi = lp_model.model.getDualsolLinear(lp_model.model.getTransformedCons(hydro_cons))
            prices['hydro'][t] = np.abs(pi) #np.round(np.abs(pi), 2)
        
        print("✓ Shadow prices extracted")
        
        
        print("\n" + "="*80)
        print("RESTRICTED PRICING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n" + "="*80)
        print("RESTRICTED PRICING COMPLETED SUCCESSFULLY!")

        
        return lp_status, lp_results, prices

    def generate_beamer_economic_table(self, revenue_analysis, player_profits, community_total, filename='energy_community_results.tex'):
        """
        전체 결과를 하나의 Beamer 프레젠테이션으로 생성
        """
        latex_code = r"""\documentclass{beamer}
            \usepackage{kotex}
            \usepackage{booktabs}
            \usepackage{xcolor}
            \usepackage{multirow}

            \usetheme{Madrid}
            \usecolortheme{whale}

            \title{Local Energy Community Optimization Results}
            \date{\today}

            \begin{document}

            \begin{frame}
            \titlepage
            \end{frame}

            """
        
        # 1. 목적함수 분석 슬라이드
        latex_code += r"""
            \begin{frame}{목적함수 분석}
            \begin{table}[h]
            \centering
            \begin{tabular}{lrr}
            \toprule
            항목 & 수익(EUR) & 비용(EUR) \\
            \midrule
            \textbf{전력} & & \\
            \quad 그리드 수출 & """ + f"{revenue_analysis['electricity']['grid_export_revenue']:.2f}" + r""" & - \\
            \quad 그리드 수입 & - & """ + f"{revenue_analysis['electricity']['grid_import_cost']:.2f}" + r""" \\
            \quad 생산 비용 & - & """ + f"{revenue_analysis['electricity']['production_cost']:.2f}" + r""" \\
            \quad 저장 비용 & - & """ + f"{revenue_analysis['electricity']['storage_cost']:.2f}" + r""" \\
            \quad \textbf{소계} & \multicolumn{2}{r}{""" + ('+' if revenue_analysis['electricity']['net'] >= 0 else '') + f"{revenue_analysis['electricity']['net']:.2f}" + r"""} \\
            \midrule
            \textbf{수소} & & \\
            \quad 그리드 수출 & """ + f"{revenue_analysis['hydrogen']['grid_export_revenue']:.2f}" + r""" & - \\
            \quad 그리드 수입 & - & """ + f"{revenue_analysis['hydrogen']['grid_import_cost']:.2f}" + r""" \\
            \quad 생산/시작 비용 & - & """ + f"{revenue_analysis['hydrogen']['production_cost'] + revenue_analysis['hydrogen']['startup_cost']:.2f}" + r""" \\
            \quad \textbf{소계} & \multicolumn{2}{r}{""" + ('-' if revenue_analysis['hydrogen']['net'] < 0 else '+') + f"{abs(revenue_analysis['hydrogen']['net']):.2f}" + r"""} \\
            \midrule
            \textbf{열} & & \\
            \quad 그리드 수입 & - & """ + f"{revenue_analysis['heat']['grid_import_cost']:.2f}" + r""" \\
            \quad \textbf{소계} & \multicolumn{2}{r}{""" + ('-' if revenue_analysis['heat']['net'] < 0 else '+') + f"{abs(revenue_analysis['heat']['net']):.2f}" + r"""} \\
            \bottomrule
            \multicolumn{3}{r}{\textbf{총 순이익: """ + ('+' if revenue_analysis['net_profit'] >= 0 else '') + f"{revenue_analysis['net_profit']:.2f}" + r""" EUR/일}} \\
            \end{tabular}
            \end{table}
            \end{frame}

            """
        
        # 2. 플레이어별 수익 슬라이드
        latex_code += r"""
            \begin{frame}{플레이어별 수익 분석}
            \scriptsize
            \begin{table}[h]
            \centering
            \begin{tabular}{l|c|rr|rr|r|r}
            \toprule
            Player & Role & \multicolumn{2}{c|}{Grid} & \multicolumn{2}{c|}{Community} & Prod & Net \\
            & & Rev & Cost & Rev & Cost & Cost & Profit \\
            \midrule
            """
        
        player_roles = {
            'u1': 'Wind,Sto',
            'u2': 'Elz,Sto',
            'u3': 'HP,Sto',
            'u4': 'Elec load',
            'u5': 'H2 load',
            'u6': 'Heat load'
        }
        
        for u in sorted(player_profits.keys()):
            p = player_profits[u]
            role = player_roles.get(u, 'Unknown')
            
            latex_code += f"{u} & {role} & "
            latex_code += f"{p['grid_revenue']:.2f} & {p['grid_cost']:.2f} & "
            latex_code += f"{p['community_revenue']:.2f} & {p['community_cost']:.2f} & "
            latex_code += f"{p['production_cost']:.2f} & "
            
            if p['net_profit'] > 0:
                latex_code += r"\textcolor{blue}{+" + f"{p['net_profit']:.2f}" + r"}"
            else:
                latex_code += r"\textcolor{red}{" + f"{p['net_profit']:.2f}" + r"}"
            latex_code += r" \\" + "\n"
        
        total_net = sum(p['net_profit'] for p in player_profits.values())
        
        latex_code += r"""\midrule
            \textbf{Total} & & & & & & & \textbf{""" + f"{total_net:.2f}" + r"""} \\
            \bottomrule
            \end{tabular}
            \end{table}
            \end{frame}

            \end{document}
            """
        
        # 파일로 저장
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        print(f"\n✓ Complete Beamer presentation saved as '{filename}'")
        
        return latex_code
    def generate_beamer_synergy_table(self, results_comparison, players, filename=None):
        """
        시너지 분석 결과를 Beamer 테이블 LaTeX 코드로 생성
        
        Args:
            results_comparison: compare_individual_vs_community_profits의 결과
            players: 플레이어 리스트
            filename: 저장할 파일명
        """
        if filename is None:
            if self.model_type == 'lp':
                filename = 'synergy_analysis_lp.tex'
            elif self.model_type == 'mip':
                filename = 'synergy_analysis_ip.tex'
            else:
                raise ValueError("a function <generate_beamer_synergy_table> has a model_type must be either 'mip' or 'lp', got: {}".format(self.model_type))
        # 플레이어 역할 정의
        player_roles = {
            'u1': 'Wind + Sto',
            'u2': 'Elz + Sto',
            'u3': 'HP + Sto',
            'u4': 'Elec load',
            'u5': 'H2 load',
            'u6': 'Heat load'
        }
        
        latex_code = r"""\begin{frame}
    \begin{table}[h]
    \centering
    \small
    \begin{tabular}{l|c|r|r|r|r}
    \toprule
    Player & Role & Individual & In Community & Gain & Gain \% \\
        &      & (EUR/day) & (EUR/day) & (EUR/day) & \\
    \midrule
    """
        
        total_individual = 0
        total_community = 0
        
        # 각 플레이어 데이터
        for player in players:
            role = player_roles.get(player, 'Unknown')[:15]
            ind_profit = results_comparison['individual'].get(player, {}).get('profit', 0)
            comm_profit = results_comparison['community']['player_profits'][player]['net_profit']
            gain = comm_profit - ind_profit
            
            total_individual += ind_profit
            total_community += comm_profit
            
            # Gain percentage 계산
            if ind_profit == 0:
                if gain > 0:
                    gain_pct_str = r"\textcolor{green}{N/A (+)}"
                elif gain < 0:
                    gain_pct_str = r"\textcolor{red}{N/A (-)}"
                else:
                    gain_pct_str = "0.0"
            else:
                gain_pct = (gain / abs(ind_profit)) * 100
                if gain_pct > 0:
                    gain_pct_str = r"\textcolor{green}{+" + f"{gain_pct:.1f}" + r"\%}"
                elif gain_pct < 0:
                    gain_pct_str = r"\textcolor{red}{" + f"{gain_pct:.1f}" + r"\%}"
                else:
                    gain_pct_str = "0.0"
            
            # 수익 색상 처리
            if ind_profit > 0:
                ind_str = f"{ind_profit:.2f}"
            elif ind_profit < 0:
                ind_str = r"\textcolor{red}{" + f"{ind_profit:.2f}" + "}"
            else:
                ind_str = "0.00"
                
            if comm_profit > 0:
                comm_str = f"{comm_profit:.2f}"
            elif comm_profit < 0:
                comm_str = r"\textcolor{red}{" + f"{comm_profit:.2f}" + "}"
            else:
                comm_str = "0.00"
                
            if gain > 0:
                gain_str = r"\textcolor{green}{+" + f"{gain:.2f}" + "}"
            elif gain < 0:
                gain_str = r"\textcolor{red}{" + f"{gain:.2f}" + "}"
            else:
                gain_str = "0.00"
            
            latex_code += f"{player} & {role} & {ind_str} & {comm_str} & {gain_str} & {gain_pct_str} \\\\\n"
        
        # 합계
        total_gain = total_community - total_individual
        if total_individual != 0:
            total_gain_pct = (total_gain / abs(total_individual)) * 100
            if total_gain_pct > 0:
                total_gain_pct_str = r"\textcolor{green}{+" + f"{total_gain_pct:.1f}" + r"\%}"
            else:
                total_gain_pct_str = r"\textcolor{red}{" + f"{total_gain_pct:.1f}" + r"\%}"
        else:
            total_gain_pct_str = "N/A"
        
        if total_gain > 0:
            total_gain_str = r"\textcolor{green}{+" + f"{total_gain:.2f}" + "}"
        else:
            total_gain_str = r"\textcolor{red}{" + f"{total_gain:.2f}" + "}"
        
        latex_code += r"""\midrule
    \textbf{Total} & & \textbf{""" + f"{total_individual:.2f}" + r"""} & \textbf{""" + f"{total_community:.2f}" + r"""} & """ 
        latex_code += total_gain_str + " & " + total_gain_pct_str
        
        latex_code += r""" \\
    \bottomrule
    \end{tabular}
    \end{table}

    \vspace{0.5em}
    \begin{itemize}
    \item Community synergy effect: """ + f"{total_gain:.2f}" + r""" EUR/day (""" + f"{(total_gain/abs(total_individual)*100):.1f}" + r"""\% improvement)
    \item Producers (u1-u3): """ + f"{sum(results_comparison['community']['player_profits'][p]['net_profit'] for p in ['u1','u2','u3']):.2f}" + r""" EUR/day
    \item Consumers (u4-u6): """ + f"{sum(results_comparison['community']['player_profits'][p]['net_profit'] for p in ['u4','u5','u6']):.2f}" + r""" EUR/day
    \end{itemize}
    \end{frame}"""
        
        # 파일로 저장
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        print(f"\n✓ Beamer LaTeX synergy analysis table saved as '{filename}'")
        
        # 콘솔 출력용 간단 버전
        print("\n" + "="*80)
        print("BEAMER SYNERGY TABLE GENERATED")
        print("="*80)
        print(f"Total Individual Profit: {total_individual:.2f} EUR/day")
        print(f"Total Community Profit: {total_community:.2f} EUR/day")
        print(f"Synergy Gain: {total_gain:.2f} EUR/day ({(total_gain/abs(total_individual)*100):.1f}%)")
        
        return latex_code
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
            'production': {}        }
        
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
    def plot_synergy_comparison(self, results_comparison, players):
        """시너지 효과 시각화 (음수 처리 포함)"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.switch_backend('Agg')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 개별 vs 커뮤니티 수익 비교 막대그래프
        individual_profits = [results_comparison['individual'].get(p, {}).get('profit', 0) for p in players]
        community_profit = results_comparison['community']['total_profit']
        community_shares = [community_profit / len(players)] * len(players)
        
        x = np.arange(len(players))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, individual_profits, width, label='Individual', color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, community_shares, width, label='Community Share', color='green', alpha=0.7)
        
        ax1.set_xlabel('Players')
        ax1.set_ylabel('Profit (EUR/day)')
        ax1.set_title('Individual vs Community Operation Profit')
        ax1.set_xticks(x)
        ax1.set_xticklabels(players)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # 2. 총 수익 비교 (막대 차트로 변경 - 음수 처리)
        total_individual = results_comparison['synergy']['total_individual_profit']
        community_total = results_comparison['synergy']['community_profit']
        synergy_gain = results_comparison['synergy']['absolute_gain']
        
        categories = ['Individual\nSum', 'Community\nTotal', 'Synergy\nEffect']
        values = [total_individual, community_total, synergy_gain]
        colors_bar = ['lightcoral', 'lightgreen', 'gold' if synergy_gain >= 0 else 'lightgray']
        
        bars = ax2.bar(categories, values, color=colors_bar, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Profit (EUR/day)')
        ax2.set_title('Community Synergy Effect')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 막대 위에 값과 퍼센트 표시
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if i == 2:  # Synergy Effect
                pct = results_comparison['synergy']['relative_gain']
                label = f'{val:.0f}\n({pct:+.1f}%)'
            else:
                label = f'{val:.0f}'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('synergy_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Synergy comparison plot saved as 'synergy_comparison.png'")
        plt.close()

    def analyze_negative_synergy(self, results_comparison, players):
        """음의 시너지 원인 분석"""
        print("\n" + "="*80)
        print("NEGATIVE SYNERGY ANALYSIS")
        print("="*80)
        
        if results_comparison['synergy']['absolute_gain'] < 0:
            print("\n⚠️  WARNING: Community operation shows NEGATIVE synergy!")
            print("Possible reasons:")
            print("1. Community trading constraints reduce individual optimization flexibility")
            print("2. Forced internal trading at unfavorable prices")
            print("3. Peak power constraints affecting all players")
            print("4. Misaligned supply-demand timing within community")
            
            # 각 플레이어별 손실 분석
            print(f"\n{'Player':^10} | {'Individual':^15} | {'Community/6':^15} | {'Loss':^15}")
            print("-"*60)
            
            total_loss = 0
            for player in players:
                ind_profit = results_comparison['individual'].get(player, {}).get('profit', 0)
                comm_share = results_comparison['community']['profit'] / len(players)
                loss = comm_share - ind_profit
                total_loss += loss
                
                if loss < 0:
                    print(f"{player:^10} | {ind_profit:^15.2f} | {comm_share:^15.2f} | {loss:^15.2f} ⬇️")
                else:
                    print(f"{player:^10} | {ind_profit:^15.2f} | {comm_share:^15.2f} | {loss:^15.2f} ⬆️")
            
            print("\nRecommendations:")
            print("• Review community trading price mechanisms")
            print("• Consider flexible cooperation (partial community)")
            print("• Implement fair profit-sharing mechanisms")
            print("• Optimize storage and production coordination")
    def compare_individual_vs_community_profits(self, players, model_type,time_periods, base_parameters, player_profits):
        """
        각 플레이어의 개별 운영 수익과 커뮤니티 운영 수익을 비교
        
        Returns:
            dict: 비교 결과 및 시너지 효과 분석
        """
        results_comparison = {
        'community': {
            'total_profit': 0, 
            'player_profits': {}, 
            'prices': {},
            'status': None
        },
        'individual': {},
        'synergy': {}
        }
        
        # 1. 커뮤니티 전체 최적화 (기존 코드)
        print("\n" + "="*80)
        print("STEP 1: COMMUNITY OPTIMIZATION (All Players)")
        print("="*80)
        
        lem_community = LocalEnergyMarket(players, time_periods, base_parameters, model_type=model_type, dwr=False)
        lem_community.model.optimize()
        status = lem_community.model.getStatus()
        if status == "optimal":
            status, results = solve_and_extract_results(lem_community.model)
            revenue = lem_community._analyze_revenue_by_resource(results)
            results_comparison['community']['total_profit'] = revenue['net_profit']
            results_comparison['community']['player_profits'] = player_profits
            results_comparison['community']['status'] = status
            print(f"Community total profit: {revenue['net_profit']:.2f} EUR/day")
        
        # 2. 각 플레이어 개별 최적화
        print("\n" + "="*80)
        print("STEP 2: INDIVIDUAL PLAYER OPTIMIZATION")
        print("="*80)
        
        for player in players:
            print(f"\nOptimizing for player {player} alone...")
            # 파라미터 복사 및 수정
            individual_params = base_parameters.copy()
            
            # 현재 플레이어만 활성화하는 트릭
            # 각 카테고리에서 현재 플레이어만 남기기
            for key in individual_params.keys():
                if key.startswith('players_with_'):
                    if player in individual_params[key]:
                        individual_params[key] = [player]
                    else:
                        individual_params[key] = []
            
            individual_params['dwr'] = False 
            
            # 개별 최적화 실행
            lem_individual = LocalEnergyMarket(
                [player],  # 플레이어 리스트를 현재 플레이어만
                time_periods, 
                individual_params, 
                model_type=model_type,
                dwr=individual_params['dwr']
            )
            # 조용히 최적화 (출력 최소화)
            if player not in ['u4', 'u5', 'u6']:
                lem_individual.model.hideOutput()

            status_ind = lem_individual.solve()
            if status_ind == "optimal":
                # 결과 추출
                status_ind, results_ind = solve_and_extract_results(lem_individual.model)
                revenue_ind = lem_individual._analyze_revenue_by_resource(results_ind)
                results_comparison['individual'][player] = {
                    'profit': revenue_ind['net_profit'],
                    'status': status_ind,
                    'breakdown': {
                        'electricity': revenue_ind['electricity']['net'],
                        'hydrogen': revenue_ind['hydrogen']['net'],
                        'heat': revenue_ind['heat']['net']
                    }
                }
                
                print(f"  Player {player} individual profit: {revenue_ind['net_profit']:.2f} EUR/day")
            else:
                results_comparison['individual'][player] = {
                    'profit': 0,
                    'status': status_ind,
                    'breakdown': None
                }
                print(f"  Player {player} optimization failed: {status_ind}")
                    
        
        # 3. 시너지 효과 분석
        print("\n" + "="*80)
        print("SYNERGY ANALYSIS")
        print("="*80)
        
        print(f"\n{'Player':^10} | {'Individual':^15} | {'In Community':^15} | {'Gain':^15} | {'Gain %':^12}")
        print("-"*80)
        
        total_individual = 0
        total_community_player = 0
        
        for player in players:
            ind_profit = results_comparison['individual'].get(player, {}).get('profit', 0)
            comm_profit = results_comparison['community']['player_profits'][player]['net_profit']
            gain = comm_profit - ind_profit
            
            # gain_pct 계산 수정
            if ind_profit == 0:
                if gain > 0:
                    gain_pct_str = "N/A (+)"  # 개별 0에서 양수로
                elif gain < 0:
                    gain_pct_str = "N/A (-)"  # 개별 0에서 음수로
                else:
                    gain_pct_str = "0.0"
            else:
                gain_pct = (gain / abs(ind_profit)) * 100
                gain_pct_str = f"{gain_pct:.1f}"
            
            total_individual += ind_profit
            total_community_player += comm_profit
            
            # 색상 표시
            if gain > 0:
                gain_marker = "↑"
            elif gain < 0:
                gain_marker = "↓"
            else:
                gain_marker = ""
            
            print(f"{player:^10} | {ind_profit:^15.2f} | {comm_profit:^15.2f} | "
                f"{gain:^15.2f} {gain_marker} | {gain_pct_str:^12}")
        
        print("-"*80)
        
        # 전체 시너지 계산
        total_gain = total_community_player - total_individual
        if total_individual == 0:
            total_gain_pct_str = "N/A"
        else:
            total_gain_pct = (total_gain / abs(total_individual)) * 100
            total_gain_pct_str = f"{total_gain_pct:.1f}%"
        
        print(f"{'Total':^10} | {total_individual:^15.2f} | {total_community_player:^15.2f} | "
            f"{total_gain:^15.2f} | {total_gain_pct_str:^12}")
        
        # 추가 분석: Consumer의 특별한 상황
        print("\n" + "="*80)
        print("CONSUMER ANALYSIS (u4, u5, u6)")
        print("="*80)
        
        consumers = ['u4', 'u5', 'u6']
        for player in consumers:
            ind_profit = results_comparison['individual'].get(player, {}).get('profit', 0)
            comm_profit = results_comparison['community']['player_profits'][player]['net_profit']
            
            if ind_profit == 0 and comm_profit < 0:
                print(f"{player}: Individual operation = 0 (no assets to operate)")
                print(f"      Community participation = {comm_profit:.2f} (pays for energy)")
                print(f"      → This is expected: consumers pay for energy in community")
            elif ind_profit == 0 and comm_profit == 0:
                print(f"{player}: No change (0 → 0)")
        
        print("\nNote: Consumers (u4-u6) have no generation/storage assets,")
        print("      so their individual profit is 0 (cannot operate alone).")
        print("      In community, they pay for energy, resulting in negative profit.")
        results_comparison['synergy']['absolute_gain'] = total_community_player - total_individual
        return results_comparison
    def calculate_player_profits_with_community_prices(self, results, prices):
        """
        커뮤니티 가격(shadow price)으로 각 플레이어의 수익 계산
        player profits: MIP을 풀어 구한 central dispatch에 community price를 적용한 플레이어의 수익
        Returns:
            dict: 각 플레이어의 상세 수익 내역
        """        
        # 각 플레이어별 수익 계산
        player_profits = {}
        
        for u in self.players:
            profit_breakdown = {
                'grid_revenue': 0.0,
                'grid_cost': 0.0,
                'community_revenue': 0.0,
                'community_cost': 0.0,
                'production_cost': 0.0,
                'storage_cost': 0.0,
                'startup_cost': 0.0,
                'net_profit': 0.0
            }
            
            for t in self.time_periods:
                # 1. 그리드 거래 수익/비용
                # 전기
                if 'e_E_gri' in results and (u,t) in results['e_E_gri']:
                    export = results['e_E_gri'][u,t]
                    if export > 0:
                        profit_breakdown['grid_revenue'] += export * self.params.get(f'pi_E_gri_export_{t}', 0)
                
                if 'i_E_gri' in results and (u,t) in results['i_E_gri']:
                    import_val = results['i_E_gri'][u,t]
                    if import_val > 0:
                        profit_breakdown['grid_cost'] += import_val * self.params.get(f'pi_E_gri_import_{t}', 0)
                
                # 수소
                if 'e_G_gri' in results and (u,t) in results['e_G_gri']:
                    export = results['e_G_gri'][u,t]
                    if export > 0:
                        profit_breakdown['grid_revenue'] += export * self.params.get(f'pi_G_gri_export_{t}', 0)
                
                if 'i_G_gri' in results and (u,t) in results['i_G_gri']:
                    import_val = results['i_G_gri'][u,t]
                    if import_val > 0:
                        profit_breakdown['grid_cost'] += import_val * self.params.get(f'pi_G_gri_import_{t}', 0)
                
                # 열
                if 'e_H_gri' in results and (u,t) in results['e_H_gri']:
                    export = results['e_H_gri'][u,t]
                    if export > 0:
                        profit_breakdown['grid_revenue'] += export * self.params.get(f'pi_H_gri_export_{t}', 0)
                
                if 'i_H_gri' in results and (u,t) in results['i_H_gri']:
                    import_val = results['i_H_gri'][u,t]
                    if import_val > 0:
                        profit_breakdown['grid_cost'] += import_val * self.params.get(f'pi_H_gri_import_{t}', 0)
                
                # 2. 커뮤니티 내부 거래 (shadow price 기반)
                # 전기
                if 'e_E_com' in results and (u,t) in results['e_E_com']:
                    export = results['e_E_com'][u,t]
                    if export > 0:
                        profit_breakdown['community_revenue'] += export * prices['electricity'].get(t, 0)
                
                if 'i_E_com' in results and (u,t) in results['i_E_com']:
                    import_val = results['i_E_com'][u,t]
                    if import_val > 0:
                        profit_breakdown['community_cost'] += import_val * prices['electricity'].get(t, 0)
                
                # 수소
                if 'e_G_com' in results and (u,t) in results['e_G_com']:
                    export = results['e_G_com'][u,t]
                    if export > 0:
                        profit_breakdown['community_revenue'] += export * prices['hydro'].get(t, 0)
                
                if 'i_G_com' in results and (u,t) in results['i_G_com']:
                    import_val = results['i_G_com'][u,t]
                    if import_val > 0:
                        profit_breakdown['community_cost'] += import_val * prices['hydro'].get(t, 0)
                
                # 열
                if 'e_H_com' in results and (u,t) in results['e_H_com']:
                    export = results['e_H_com'][u,t]
                    if export > 0:
                        profit_breakdown['community_revenue'] += export * prices['heat'].get(t, 0)
                
                if 'i_H_com' in results and (u,t) in results['i_H_com']:
                    import_val = results['i_H_com'][u,t]
                    if import_val > 0:
                        profit_breakdown['community_cost'] += import_val * prices['heat'].get(t, 0)
                
                # 3. 생산 비용
                if 'p' in results:
                    if (u,'res',t) in results['p']:
                        profit_breakdown['production_cost'] += results['p'][u,'res',t] * self.params.get(f'c_res_{u}', 0)
                    if (u,'els',t) in results['p']:
                        profit_breakdown['production_cost'] += results['p'][u,'els',t] * self.params.get(f'c_els_{u}', 0)
                    if (u,'hp',t) in results['p']:
                        profit_breakdown['production_cost'] += results['p'][u,'hp',t] * self.params.get(f'c_hp_{u}', 0)
                
                # 4. 저장 비용
                c_E_sto = self.params.get('c_E_sto', 0.01)
                c_G_sto = self.params.get('c_G_sto', 0.01)
                c_H_sto = self.params.get('c_H_sto', 0.01)
                nu_ch = self.params.get('nu_ch', 0.9)
                nu_dis = self.params.get('nu_dis', 0.9)
                
                if 'b_ch_E' in results and (u,t) in results['b_ch_E']:
                    profit_breakdown['storage_cost'] += results['b_ch_E'][u,t] * c_E_sto * nu_ch
                if 'b_dis_E' in results and (u,t) in results['b_dis_E']:
                    profit_breakdown['storage_cost'] += results['b_dis_E'][u,t] * c_E_sto * (1/nu_dis)
                # 수소 저장 비용 추가
                if 'b_ch_G' in results and (u,t) in results['b_ch_G']:
                    profit_breakdown['storage_cost'] += results['b_ch_G'][u,t] * c_G_sto * nu_ch
                if 'b_dis_G' in results and (u,t) in results['b_dis_G']:
                    profit_breakdown['storage_cost'] += results['b_dis_G'][u,t] * c_G_sto * (1/nu_dis)
                
                # 열 저장 비용 추가
                if 'b_ch_H' in results and (u,t) in results['b_ch_H']:
                    profit_breakdown['storage_cost'] += results['b_ch_H'][u,t] * c_H_sto * nu_ch
                if 'b_dis_H' in results and (u,t) in results['b_dis_H']:
                    profit_breakdown['storage_cost'] += results['b_dis_H'][u,t] * c_H_sto * (1/nu_dis)
                # 5. 시작 비용
                if 'z_su' in results and (u,t) in results['z_su']:
                    profit_breakdown['startup_cost'] += results['z_su'][u,t] * self.params.get(f'c_su_G_{u}', 50)
            
            # 순이익 계산
            profit_breakdown['net_profit'] = (
                profit_breakdown['grid_revenue'] + 
                profit_breakdown['community_revenue'] - 
                profit_breakdown['grid_cost'] - 
                profit_breakdown['community_cost'] - 
                profit_breakdown['production_cost'] - 
                profit_breakdown['storage_cost'] - 
                profit_breakdown['startup_cost']
            )
            
            player_profits[u] = profit_breakdown
        
        # 결과 출력
        print("\n" + "="*80)
        print("PLAYER PROFITS WITH COMMUNITY PRICING")
        print("="*80)
        
        print(f"{'Player':^8} | {'Grid Rev':^10} | {'Grid Cost':^10} | {'Comm Rev':^10} | {'Comm Cost':^10} | {'Prod Cost':^10} | {'Net Profit':^12}")
        print("-"*80)
        
        for u in self.players:
            p = player_profits[u]
            print(f"{u:^8} | {p['grid_revenue']:^10.2f} | {p['grid_cost']:^10.2f} | "
                f"{p['community_revenue']:^10.2f} | {p['community_cost']:^10.2f} | "
                f"{p['production_cost']:^10.2f} | {p['net_profit']:^12.2f}")
        
        print("-"*80)
        total_profit = sum(p['net_profit'] for p in player_profits.values())
        print(f"{'Total':^8} | {'':^10} | {'':^10} | {'':^10} | {'':^10} | {'':^10} | {total_profit:^12.2f}")
        # 검증: 커뮤니티 내부 거래 균형
        total_comm_revenue = sum(p['community_revenue'] for p in player_profits.values())
        total_comm_cost = sum(p['community_cost'] for p in player_profits.values())
        
        if abs(total_comm_revenue - total_comm_cost) > 1e-6:
            print(f"⚠️ Community trade imbalance: {total_comm_revenue - total_comm_cost:.4f}")
        return player_profits, prices

# Example usage with Restricted Pricing
if __name__ == "__main__":
    # Define example data
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']
    time_periods = list(range(24))  # 24 hours
    configuration = {}
    configuration["players_with_renewables"] = ['u1']
    configuration["players_with_electrolyzers"] = ['u2'] + ['u7']
    configuration["players_with_heatpumps"] = ['u3']
    configuration["players_with_elec_storage"] = ['u1']
    configuration["players_with_hydro_storage"] = ['u2'] + ['u7']
    configuration["players_with_heat_storage"] = ['u3']
    configuration["players_with_nfl_elec_demand"] = ['u4']
    configuration["players_with_nfl_hydro_demand"] = ['u5'] + ['u8']
    configuration["players_with_nfl_heat_demand"] = ['u6']
    configuration["players_with_fl_elec_demand"] = ['u2','u3'] + ['u7']
    configuration["players_with_fl_hydro_demand"] = []
    configuration["players_with_fl_heat_demand"] = []
    parameters = setup_lem_parameters(players, configuration, time_periods)
    # Create and solve model with Restricted Pricing
    lem = LocalEnergyMarket(players, time_periods, parameters, model_type='mip')
    # lem.model.addCons(lem.z_on[("u2", 0)] == 1)
    # First solve complete model and analyze revenue
    print("\n" + "="*60)
    print("SOLVING COMPLETE MODEL AND ANALYZING REVENUE")
    print("="*60)
    status_complete, results_complete, revenue_analysis, community_prices = lem.solve_complete_model(analyze_revenue=True)
    
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
            load_pct = (elec_use / parameters['els_cap'] * 100) if parameters['els_cap'] > 0 else 0
            
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
        lem.plot_storage_operation(results_complete)
        player_profits, prices = lem.calculate_player_profits_with_community_prices(results_complete, community_prices)
        # LaTeX 파일로 저장
        latex_table = lem.generate_beamer_economic_table(revenue_analysis, player_profits, revenue_analysis['net_profit'])
    else:
        print(f"Optimization failed: {status_complete}")
    
    print("\n" + "="*60)
    print("SOLVING WITH RESTRICTED PRICING")
    print("="*60)
    # 개별 vs 커뮤니티 비교 분석
    comparison_results = lem.compare_individual_vs_community_profits(
        players, 
        lem.model_type,
        time_periods, 
        parameters,
        player_profits
    )
    # 음의 시너지 분석
    if comparison_results['synergy']['absolute_gain'] < 0:
        lem.analyze_negative_synergy(comparison_results, players)
    lem.generate_beamer_synergy_table(comparison_results, players)

    # 추가 분석: 어떤 플레이어가 가장 큰 이익을 보는지
    print("\n" + "="*80)
    print("PLAYER BENEFIT ANALYSIS")
    print("="*80)
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
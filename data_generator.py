"""
LEM Parameters Setup
Shared parameter configuration for both compact and column generation formulations
"""

import numpy as np
import pandas as pd


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
        avg_price = df[col_name].mean()
        hourly_avg_prices.append(avg_price)
    
    # 특정 날짜 선택 (예: 중간값에 가까운 날)
    median_idx = len(df) // 2
    selected_day_prices = []
    for hour in range(1, 25):
        col_name = f'{hour:02d}시'
        price = df.iloc[median_idx][col_name]
        selected_day_prices.append(price)
    
    return hourly_avg_prices, selected_day_prices


def calculate_hydrogen_prices(elec_prices_eur):
    """
    논문 기반 수소 가격 계산
    - 논문: 고정 €2.1/kg
    - 여기서는 전력 가격에 반비례하도록 동적 설정
    """
    h2_prices = []
    
    # 전력 가격 평균 및 범위 계산
    avg_elec = np.mean(elec_prices_eur)
    
    for elec_price in elec_prices_eur:
        # 기본 수소 가격: €2.1/kg
        base_h2_price = 2.1 * 1.5
        
        # 전력 가격에 반비례하는 조정 계수
        adjustment = 1.0
        
        # 최종 수소 가격 (€1.5 ~ €5.0/kg 범위)
        h2_price = base_h2_price * adjustment
        h2_price = np.clip(h2_price, 1.5, 5.0)
        
        h2_prices.append(h2_price)
    
    return h2_prices


def create_tou_import_prices(smp_prices_eur, time_periods):
    """
    한국의 TOU 요금제를 기반으로 import 가격 생성
    
    한국 전력 산업용(을) 고압A 선택II 요금제 기준:
    - 경부하: 23:00~09:00 (기본요금 대비 약 60-70%)
    - 중간부하: 09:00~10:00, 12:00~13:00, 17:00~23:00 (기본요금)
    - 최대부하: 10:00~12:00, 13:00~17:00 (기본요금 대비 약 140-180%)
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
            # 여름철 (7-8월) 기준으로 더 높은 요금 적용
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


def setup_lem_parameters(players, configuration, time_periods):
    """
    Setup parameters for Local Energy Market problem
    
    Args:
        players: List of player IDs
        time_periods: List of time period indices
        
    Returns:
        dict: Complete parameter dictionary
    """
    try:
        avg_prices, daily_prices = load_korean_electricity_prices()
        
        # 가격을 EUR/kWh로 변환 (KRW/kWh → EUR/kWh)
        exchange_rate = 1400
        unit_adjustment = 1000
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
        print("Using default synthetic prices...")
        korean_prices_eur = None
        h2_prices_eur = None
    
    # Example parameters with proper bounds and storage types
    parameters = {
        'players_with_renewables': configuration['players_with_renewables'],
        'players_with_electrolyzers': configuration['players_with_electrolyzers'],
        'players_with_heatpumps': configuration['players_with_heatpumps'],
        'players_with_elec_storage': configuration['players_with_elec_storage'],
        'players_with_hydro_storage': configuration['players_with_hydro_storage'],
        'players_with_heat_storage': configuration['players_with_heat_storage'],
        'players_with_nfl_elec_demand': configuration['players_with_nfl_elec_demand'],
        'players_with_nfl_hydro_demand': configuration['players_with_nfl_hydro_demand'],
        'players_with_nfl_heat_demand': configuration['players_with_nfl_heat_demand'],
        'players_with_fl_elec_demand': configuration['players_with_fl_elec_demand'],
        'players_with_fl_hydro_demand': configuration['players_with_fl_hydro_demand'],
        'players_with_fl_heat_demand': configuration['players_with_fl_heat_demand'],
        'nu_ch': 0.9,
        'nu_dis': 0.9,
        'pi_peak': 100,
        
        # Storage parameters
        'storage_power': 0.5,
        'storage_capacity': 2.0,
        'initial_soc_E': 0.5*1,
        'initial_soc_G': 25,
        'initial_soc_H': 0.2,
        'storage_power_heat': 0.10,
        'storage_capacity_heat': 0.40,
        'nu_ch': 0.95,
        'nu_dis': 0.95,        
        # Equipment capacities
        'hp_cap': 0.08,
        'els_cap': 1,
        'C_min': 0.15,
        'C_sb': 0.01,
        'phi1_1': 21.12266316,
        'phi0_1': -0.37924094,
        'phi1_2': 16.66883134,
        'phi0_2': 0.87814262,
        'c_res': 0.05,
        'c_hp': 0.1,
        'c_els': 0.05,
        'c_su': 50,
        'max_up_time': 3,
        'min_down_time': 2,
        
        # Grid connection limits
        'res_capacity': 2,
        'e_E_cap': 0.5*2,
        'i_E_cap': 0.5*2,
        'e_H_cap': 500, #0.06,
        'i_H_cap': 500, #0.08,
        'e_G_cap': 50,
        'i_G_cap': 100 , #30,
        
        # Cost parameters
        'c_E_sto': 0.01,
        'c_G_sto': 0.01,
        'c_H_sto': 0.01,
    }
    
    parameters['players_with_fl_elec_demand'] = list(set(
        parameters['players_with_electrolyzers'] + parameters['players_with_heatpumps']
    ))
    

    
    # Add cost parameters
    for u in parameters['players_with_renewables']:
        parameters[f'c_res_{u}'] = parameters['c_res']
        # RENEWABLE AVAILABILITY - Natural solar curve
        for t in time_periods:
            if 6 <= t <= 18:
                solar_factor = np.exp(-((t - 12) / 3.5)**2)
                parameters[f'renewable_cap_{u}_{t}'] = 2 * solar_factor  # MW
            else:
                parameters[f'renewable_cap_{u}_{t}'] = 0
    for u in parameters['players_with_heatpumps']:
        parameters[f'c_hp_{u}'] = parameters['c_hp']
    for u in parameters['players_with_electrolyzers']:
        parameters[f'c_els_{u}'] = parameters['c_els']
        parameters[f'c_su_{u}'] = parameters['c_su']
    for u in parameters['players_with_elec_storage']:
        parameters[f'c_E_sto_{u}'] = parameters['c_E_sto']
    for u in parameters['players_with_hydro_storage']:
        parameters[f'c_G_sto_{u}'] = parameters['c_G_sto']
    for u in parameters['players_with_heat_storage']:
        parameters[f'c_H_sto_{u}'] = parameters['c_H_sto']
    
    # Add grid prices
    if korean_prices_eur and h2_prices_eur:
        parameters = generate_market_price(parameters, time_periods, korean_prices_eur, h2_prices_eur)
    else:
        # Fallback to synthetic prices
        for t in time_periods:
            if 6 <= t <= 9:
                base_price = 0.6 + 0.2 * np.sin((t-6) * np.pi / 6)
            elif 10 <= t <= 15:
                base_price = 0.2 + 0.1 * np.cos((t-12.5) * np.pi / 3)
            elif 17 <= t <= 20:
                base_price = 0.9 + 0.3 * np.sin((t-17) * np.pi / 6)
            elif 21 <= t <= 23 or 0 <= t <= 5:
                base_price = 0.3 + 0.1 * np.sin(t * np.pi / 12)
            else:
                base_price = 0.5
            
            variation = 0.02 * np.sin(2 * np.pi * t / 4)
            parameters[f'pi_E_gri_export_{t}'] = base_price + variation
            parameters[f'pi_E_gri_import_{t}'] = (base_price + variation) * 1.15
    
    # HEAT PRICES
    for t in time_periods:
        heat_demand_factor = 1.0 + 0.3 * np.cos(2 * np.pi * (t - 7) / 24)
        parameters[f'pi_H_gri_export_{t}'] = (0.4 * parameters[f'pi_E_gri_import_{t}']) * heat_demand_factor
        
        if 6 <= t < 9 or 17 <= t < 23:
            tou_multiplier = 1.30
        elif 23 <= t or t < 6:
            tou_multiplier = 1.15
        else:
            tou_multiplier = 1.20
        
        parameters[f'pi_H_gri_import_{t}'] = parameters[f'pi_H_gri_export_{t}'] * tou_multiplier
    
    # DEMANDS
    for u in players:
        for t in time_periods:
            # HYDROGEN DEMAND
            if u in parameters['players_with_nfl_hydro_demand']:
                if 6 <= t <= 11:
                    h2_demand = 6 + 4 * np.exp(-((t-9)/2)**2)
                elif 14 <= t <= 20:
                    h2_demand = 4 + 3 * np.exp(-((t-17)/2)**2)
                elif 12 <= t <= 13:
                    h2_demand = 3.0
                elif 21 <= t <= 23:
                    h2_demand = 2.0
                else:
                    h2_demand = 1.0
                parameters[f'd_G_nfl_{u}_{t}'] = h2_demand
            else:
                parameters[f'd_G_nfl_{u}_{t}'] = 0
            
            # ELEC DEMAND
            if u in parameters['players_with_nfl_elec_demand']:
                morning_peak = 20 * np.exp(-((t - 8) / 2)**2)
                evening_peak = 40 * np.exp(-((t - 19.5) / 2)**2)
                base_demand = 60
                elec_demand = (base_demand + morning_peak + evening_peak) * 0.001
                parameters[f'd_E_nfl_{u}_{t}'] = elec_demand
            
            # HEAT DEMAND
            if u in parameters['players_with_nfl_heat_demand']:
                heat_demand_kw = 60 + 30 * np.cos(2 * np.pi * (t - 3) / 24)
                heat_demand = heat_demand_kw * 0.001
                parameters[f'd_H_nfl_{u}_{t}'] = heat_demand
            else:
                parameters[f'd_H_nfl_{u}_{t}'] = 0
    
    return parameters
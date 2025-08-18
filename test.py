import numpy as np
from pyscipopt import SCIP_PARAMSETTING
from energy_solver import EnergyCommunitySolver
from energy_pricer import EnergyPricer
from energy_pricing_network import solve_energy_pricing_problem
from compact import load_korean_electricity_prices, calculate_hydrogen_prices
def create_initial_patterns_for_players(players, time_periods, parameters):
    """
    Create simple initial patterns for each player
    Each player satisfies their demand using only grid imports/exports
    No community trading in initial patterns
    """
    
    initial_patterns = {}
    
    for player in players:
        pattern = {}
        
        # Player u1: Renewable generator with storage
        if player == 'u1':
            pattern['i_E_gri'] = {}
            pattern['e_E_gri'] = {}
            pattern['i_E_com'] = {}
            pattern['e_E_com'] = {}
            pattern['p_res'] = {}
            pattern['b_ch_E'] = {}
            pattern['b_dis_E'] = {}
            pattern['s_E'] = {}
            
            for t in time_periods:
                # Generate renewable when available
                renewable_cap = parameters.get(f'renewable_cap_{player}_{t}', 0)
                pattern['p_res'][t] = renewable_cap * 0.8  # Use 80% of capacity
                
                # Export excess to grid (not community initially)
                pattern['e_E_gri'][t] = 0.0
                pattern['i_E_gri'][t] = parameters.get(f'i_E_cap', 0.5)
                
                # No community trading initially
                pattern['i_E_com'][t] = 0.0
                pattern['e_E_com'][t] = 0.0
                
                # Simple storage operation
                pattern['b_ch_E'][t] = 0.0
                pattern['b_dis_E'][t] = 0.0
                pattern['s_E'][t] = parameters.get('initial_soc', 0.5)
        
        # Player u2: Electrolyzer with hydrogen storage
        elif player == 'u2':
            pattern['i_E_gri'] = {}
            pattern['e_E_gri'] = {}
            pattern['i_E_com'] = {}
            pattern['e_E_com'] = {}
            pattern['i_G_gri'] = {}
            pattern['e_G_gri'] = {}
            pattern['i_G_com'] = {}
            pattern['e_G_com'] = {}
            pattern['p_els'] = {}
            pattern['z_su'] = {}
            pattern['z_on'] = {}
            pattern['z_off'] = {}
            pattern['z_sb'] = {}
            pattern['b_ch_G'] = {}
            pattern['b_dis_G'] = {}
            pattern['s_G'] = {}
            
            for t in time_periods:
                # Simple operation: off most of the time, on during cheap hours
                if t in [0, 1, 2, 3, 4, 5]:  # Night hours - operate
                    pattern['z_on'][t] = 1.0
                    pattern['z_off'][t] = 0.0
                    pattern['z_sb'][t] = 0.0
                    pattern['p_els'][t] = 10.0  # Produce 10 kg/h hydrogen
                    pattern['i_E_gri'][t] = 0.5  # Import electricity for electrolyzer
                else:
                    pattern['z_on'][t] = 0.0
                    pattern['z_off'][t] = 1.0
                    pattern['z_sb'][t] = 0.0
                    pattern['p_els'][t] = 0.0
                    pattern['i_E_gri'][t] = 0.0
                
                # Startup detection
                if t == 0:
                    pattern['z_su'][t] = 1.0 if pattern['z_on'][t] > 0 else 0.0
                else:
                    pattern['z_su'][t] = 1.0 if (pattern['z_on'][t] > 0 and pattern['z_on'][t-1] == 0) else 0.0
                
                # Export hydrogen to grid
                pattern['e_G_gri'][t] = 0.0
                pattern['i_G_gri'][t] = parameters.get(f'i_G_cap', 30)
                
                # No electricity export
                pattern['e_E_gri'][t] = 0.0
                
                # No community trading
                pattern['i_E_com'][t] = 0.0
                pattern['e_E_com'][t] = 0.0
                pattern['i_G_com'][t] = 0.0
                pattern['e_G_com'][t] = 0.0
                
                # No storage use initially
                pattern['b_ch_G'][t] = 0.0
                pattern['b_dis_G'][t] = 0.0
                pattern['s_G'][t] = 0.0
        
        # Player u3: Heat pump with heat storage
        elif player == 'u3':
            pattern['i_E_gri'] = {}
            pattern['e_E_gri'] = {}
            pattern['i_E_com'] = {}
            pattern['e_E_com'] = {}
            pattern['i_H_gri'] = {}
            pattern['e_H_gri'] = {}
            pattern['i_H_com'] = {}
            pattern['e_H_com'] = {}
            pattern['p_hp'] = {}
            pattern['b_ch_H'] = {}
            pattern['b_dis_H'] = {}
            pattern['s_H'] = {}
            
            for t in time_periods:
                # Simple heat pump operation
                pattern['p_hp'][t] = 0.0  # No heat production initially
                pattern['i_E_gri'][t] = parameters.get(f'i_E_cap', 0.5)  # No electricity import for heat pump
                pattern['e_E_gri'][t] = 0.0
                
                # No heat trade
                pattern['i_H_gri'][t] = 0.0
                pattern['e_H_gri'][t] = 0.0
                
                # No community trading
                pattern['i_E_com'][t] = 0.0
                pattern['e_E_com'][t] = 0.0
                pattern['i_H_com'][t] = 0.0
                pattern['e_H_com'][t] = 0.0
                
                # No storage use
                pattern['b_ch_H'][t] = 0.0
                pattern['b_dis_H'][t] = 0.0
                pattern['s_H'][t] = parameters.get('initial_soc', 50)
        
        # Player u4: Electricity consumer
        elif player == 'u4':
            pattern['i_E_gri'] = {}
            pattern['e_E_gri'] = {}
            pattern['i_E_com'] = {}
            pattern['e_E_com'] = {}
            
            for t in time_periods:
                # Import all demand from grid
                demand = parameters.get(f'd_E_nfl_{player}_{t}', 0)
                pattern['i_E_gri'][t] = min(demand, parameters.get(f'i_E_cap', 0.5))
                pattern['e_E_gri'][t] = 0.0
                
                # No community trading
                pattern['i_E_com'][t] = 0.0
                pattern['e_E_com'][t] = 0.0
        
        # Player u5: Hydrogen consumer
        elif player == 'u5':
            pattern['i_G_gri'] = {}
            pattern['e_G_gri'] = {}
            pattern['i_G_com'] = {}
            pattern['e_G_com'] = {}
            
            for t in time_periods:
                # Import all demand from grid
                demand = parameters.get(f'd_G_nfl_{player}_{t}', 0)
                pattern['i_G_gri'][t] = min(demand, parameters.get(f'i_G_cap', 30))
                pattern['e_G_gri'][t] = 0.0
                
                # No community trading
                pattern['i_G_com'][t] = 0.0
                pattern['e_G_com'][t] = 0.0
        
        # Player u6: Heat consumer
        elif player == 'u6':
            pattern['i_H_gri'] = {}
            pattern['e_H_gri'] = {}
            pattern['i_H_com'] = {}
            pattern['e_H_com'] = {}
            
            for t in time_periods:
                # Import all demand from grid
                demand = parameters.get(f'd_H_nfl_{player}_{t}', 0)
                pattern['i_H_gri'][t] = min(demand, parameters.get(f'i_H_cap', 0.08))
                pattern['e_H_gri'][t] = 0.0
                
                # No community trading
                pattern['i_H_com'][t] = 0.0
                pattern['e_H_com'][t] = 0.0
        
        initial_patterns[player] = pattern
    
    return initial_patterns


def solve_convex_hull_pricing(players, time_periods, parameters):
    """
    Main function to solve Energy Community problem with Convex Hull Pricing
    """
    
    print("=== Energy Community Convex Hull Pricing ===\n")
    
    # Step 1: Generate initial patterns
    print("Step 1: Generating initial patterns...")
    # Step 2: Initialize RMP solver
    print("Step 2: Initializing RMP solver...")
    solver = EnergyCommunitySolver(players, time_periods, parameters)
    init_patterns = solver.gen_initial_cols(folder_path=None)
    init_patterns = create_initial_patterns_for_players(players, time_periods, parameters)

    solver.init_rmp(init_patterns)
    
    # Configure SCIP
    solver.model.setPresolve(SCIP_PARAMSETTING.OFF)
    solver.model.setHeuristics(SCIP_PARAMSETTING.OFF)
    solver.model.disablePropagation()
    solver.model.setSeparating(SCIP_PARAMSETTING.OFF)
    
    # Step 3: Create pricer and include in model
    print("Step 3: Setting up column generation pricer...")
    pricer = EnergyPricer(
        players=players,
        time_periods=time_periods,
        parameters=parameters,
        cons_flatten=solver.cons_flatten,
        cons_coeff=solver.cons_coeff
    )
    
    solver.model.includePricer(
        pricer, 
        "EnergyPricer",
        "Pricer for Energy Community Column Generation"
    )
    
    # Step 4: Solve with column generation
    print("Step 4: Solving with column generation...\n")
    solver.model.optimize()
    
    # Step 5: Extract solution
    print("\n=== SOLUTION ===")
    status = solver.model.getStatus()
    print(f"Optimization status: {status}")
    
    if status == "optimal":
        print(f"Optimal objective value: {solver.model.getObjVal():.2f}")
        
        # Get solution
        solution = solver.get_solution()
        
        # Print pattern usage
        print("\n=== Pattern Usage ===")
        for player in players:
            print(f"\nPlayer {player}:")
            if player in solver.model.data["vars_pattern"]:
                for var in solver.model.data["vars_pattern"][player]:
                    val = solver.model.getVal(var)
                    if val > 1e-6:
                        print(f"  {var.name}: {val:.4f}")
        
        print(f"\nPeak power (chi_peak): {solution['chi_peak']:.2f}")
        
        # Step 6: Get convex hull prices
        print("\n=== Convex Hull Prices ===")
        dual_values = solver.get_dual_values()
        
        print("\nElectricity prices by time period:")
        for t in time_periods[:5]:  # Show first 5 periods
            price = dual_values.get(f"community_elec_balance_{t}", 0)
            print(f"  t={t}: {price:.4f}")
        
        print("\nHeat prices by time period:")
        for t in time_periods[:5]:
            price = dual_values.get(f"community_heat_balance_{t}", 0)
            print(f"  t={t}: {price:.4f}")
        
        print("\nHydrogen prices by time period:")
        for t in time_periods[:5]:
            price = dual_values.get(f"community_hydrogen_balance_{t}", 0)
            print(f"  t={t}: {price:.4f}")
        
        print("\nPeak power shadow prices by time period:")
        for t in time_periods[:5]:
            price = dual_values.get(f"peak_power_{t}", 0)
            print(f"  t={t}: {price:.4f}")
        
        return solver, pricer, dual_values
    
    else:
        print(f"Optimization failed with status: {status}")
        return None, None, None


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
    
    # Solve with convex hull pricing
    solver, pricer, prices = solve_convex_hull_pricing(players, time_periods, parameters)
    
    if solver is not None:
        print("\n=== Convex Hull Pricing completed successfully ===")
    else:
        print("\n=== Convex Hull Pricing failed ===")
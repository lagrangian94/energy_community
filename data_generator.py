"""
LEM Parameters Setup
Shared parameter configuration for both compact and column generation formulations
"""

import numpy as np
import pandas as pd
from HeatGen import HeatLoadGenerator, HeatPriceGenerator
from HydroGen import HydrogenLoadGenerator, generate_hydrogen_price
from ElecGen import ElectricityLoadGenerator, ElectricityProdGenerator, ElectricityPriceGenerator
from ElsGen import generate_electrolyzer


def epsilon_log(n_consumers, scale=0.1):
    """
    Logarithmic growth: ε(N) = scale * log(N)
    
    Slower growth, more conservative
    """
    if n_consumers == 1:
        return 0.0
    return scale * np.log(n_consumers)


def update_market_price(parameters, time_periods, elec_prices, h2_prices, heat_prices):
    parameters['pi_E_gri'] = {}
    parameters['pi_G_gri'] = {}
    parameters['pi_H_gri'] = {}
    parameters['pi_E_gri']['import'] = elec_prices["import"]
    parameters['pi_E_gri']['export'] = elec_prices["export"]
    parameters['pi_G_gri']['export'] = h2_prices["export"]
    parameters['pi_G_gri']['import'] = h2_prices["import"]
    parameters['pi_H_gri']['import'] = heat_prices["import"]
    parameters['pi_H_gri']['export'] = heat_prices["export"]
    for t in time_periods:
        parameters[f'pi_E_gri_import_{t}'] = elec_prices["import"][t]
        parameters[f'pi_E_gri_export_{t}'] = elec_prices["export"][t]    

        parameters[f'pi_G_gri_export_{t}'] = h2_prices["export"][t]
        parameters[f'pi_G_gri_import_{t}'] = h2_prices["import"][t]

        parameters[f'pi_H_gri_import_{t}'] = heat_prices["import"][t]
        parameters[f'pi_H_gri_export_{t}'] = heat_prices["export"][t]
    return parameters


def setup_lem_parameters(players, configuration, time_periods, sensitivity_analysis = None):
    """
    Setup parameters for Local Energy Market problem
    
    Args:
        players: List of player IDs
        time_periods: List of time period indices
        
    Returns:
        dict: Complete parameter dictionary
    """
    if sensitivity_analysis:
        use_korean_price = sensitivity_analysis['use_korean_price']
        use_tou_elec = sensitivity_analysis['use_tou_elec']
        import_factor = sensitivity_analysis['import_factor'] # market import price가 export 대비 몇배 더 큰지
        month = sensitivity_analysis['month']
        hp_cap = sensitivity_analysis['hp_cap']
        els_cap = sensitivity_analysis['els_cap']
        num_households = sensitivity_analysis['num_households']
        nu_cop = sensitivity_analysis['nu_cop']
        c_su_G = sensitivity_analysis['c_su_G']
        c_su_H = sensitivity_analysis['c_su_H']
        base_h2_price_eur = sensitivity_analysis['base_h2_price_eur']
        e_E_cap_ratio = sensitivity_analysis['e_E_cap_ratio']
        e_H_cap_ratio = sensitivity_analysis['e_H_cap_ratio']
        e_G_cap_ratio = sensitivity_analysis['e_G_cap_ratio']
        eff_type = sensitivity_analysis['eff_type']
        segments = sensitivity_analysis['segments']
        peak_penalty_ratio = sensitivity_analysis['peak_penalty_ratio']
        wind_el_ratio = sensitivity_analysis['wind_el_ratio']
        solar_el_ratio = sensitivity_analysis['solar_el_ratio']
        storage_power_ratio_E = sensitivity_analysis['storage_power_ratio_E']
        storage_power_ratio_G = sensitivity_analysis['storage_power_ratio_G']
        storage_power_ratio_H = sensitivity_analysis['storage_power_ratio_H']
        storage_capacity_ratio_E = sensitivity_analysis['storage_capacity_ratio_E']
        storage_capacity_ratio_G = sensitivity_analysis['storage_capacity_ratio_G']
        storage_capacity_ratio_H = sensitivity_analysis['storage_capacity_ratio_H']
        initial_soc_ratio_E = sensitivity_analysis['initial_soc_ratio_E']
        initial_soc_ratio_G = sensitivity_analysis['initial_soc_ratio_G']
        initial_soc_ratio_H = sensitivity_analysis['initial_soc_ratio_H']
        day = sensitivity_analysis['day']
    else:
        use_korean_price = True
        use_tou_elec = False
        import_factor = 1.5 # market import price가 export 대비 몇배 더 큰지
        month = 1
        hp_cap = 0.8 #MW [0.6, 0.8, 1.0]
        els_cap = 1 #MW
        num_households = 700
        nu_cop = 3.28
        c_su_G = 50
        c_su_H = 10 
        base_h2_price_eur = 5000/1500 #2.1*1.5 # [2.1*0.75, 2.1, 2.1*2]
        e_E_cap_ratio = 1.0 # [0.2, 0.7, 1.0]
        e_H_cap_ratio = 1.0 # [0.2, 0.7, 1.0]
        e_G_cap_ratio = 1.0 # [0.2, 0.7, 1.0]
        eff_type = 1
        segments = 6
        peak_penalty_ratio = 0.0
        wind_el_ratio = 1.0# [1.0, 2.0]
        solar_el_ratio = 1.0
        storage_power_ratio_E = 0.25
        storage_power_ratio_G = 0.25
        storage_power_ratio_H = 0.25
        storage_capacity_ratio_E = 0.0 #3.0 -> storage_power의 3배가 총 storage 용량
        storage_capacity_ratio_G = 0.0 
        storage_capacity_ratio_H = 0.0 
        initial_soc_ratio_E = 0.2
        initial_soc_ratio_G = 0.2
        initial_soc_ratio_H = 0.2
        day = 28
    # Example parameters with proper bounds and storage types
    parameters = {
        'players_with_renewables': configuration['players_with_renewables'],
        'players_with_solar': configuration['players_with_solar'],
        'players_with_wind': configuration['players_with_wind'],
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
        
        # Storage parameters
         #capacity_E=2이고, power_E_가0.25일땐 u2가 이득, 근데 power_E_가 0.5일땐 손해. 왜 그럴까??
        'storage_power_ratio_E': storage_power_ratio_E,
        'storage_power_ratio_G': storage_power_ratio_G,
        'storage_power_ratio_H': storage_power_ratio_H,
        'storage_capacity_ratio_E': storage_capacity_ratio_E, 
        'storage_capacity_ratio_G': storage_capacity_ratio_G, 
        'storage_capacity_ratio_H': storage_capacity_ratio_H, 
        'initial_soc_ratio_E': initial_soc_ratio_E,
        'initial_soc_ratio_G': initial_soc_ratio_G,
        'initial_soc_ratio_H': initial_soc_ratio_H,
        'nu_ch_E': 0.95,
        'nu_dis_E': 0.95,        
        'nu_ch_G': 0.95,
        'nu_dis_G': 0.95,
        'nu_ch_H': 0.9,
        'nu_dis_H': 0.9,
        # 'nu_loss_H':0.002,
        # Equipment capacities (res_cap은 아래에서 els_cap에 비례하게 설정됨.)
        'hp_cap': hp_cap,
        'els_cap': els_cap,
        'e_E_cap_ratio': e_E_cap_ratio,
        'e_H_cap_ratio': e_H_cap_ratio,
        'e_G_cap_ratio': e_G_cap_ratio,
        'c_res': 0.05,
        'c_hp': 2.69,
        'c_els': 0.05,
        'nu_cop': nu_cop,
        # Cost parameters
        'c_sto_E': 0.0,
        'c_sto_G': 0.0,
        'c_sto_H': 0.0,

        # Unit commitment parameters
        'min_down_time_G': 2,
        'c_RU_H': 0.9,
        'c_RD_H': 0.9,
        'c_RSU_H': 0.9,
        'c_RSD_H': 0.9,
        'c_su_G': c_su_G,
        'c_su_H': c_su_H,
        'c_min_G': 0.15, #0.15MW
        'c_sb_G': 0.01, #0.01MW
        'c_max_G': 1.0, #1.0MW,
        'c_min_H': 0.2,
        'c_max_H': 0.8,

        'use_tou_elec': use_tou_elec,
    }
    parameters['players_with_fl_elec_demand'] = list(set(
        parameters['players_with_electrolyzers'] + parameters['players_with_heatpumps']
    ))
    

    electricity_prod_generator = ElectricityProdGenerator(num_units=1, wind_el_ratio=wind_el_ratio, solar_el_ratio=solar_el_ratio, el_cap_mw=parameters["els_cap"])
    wind_production = electricity_prod_generator.generate_wind_production(day=day)
    if len(wind_production)>1:
        wind_production = wind_production[0]
    pv_production = electricity_prod_generator.generate_solar_production()
    El = generate_electrolyzer(eff_type=eff_type, els_cap=parameters["els_cap"], wind_el_ratio=2.0, c_min=parameters["c_min_G"], c_sb=parameters["c_sb_G"], c_su_G=parameters["c_su_G"],
     segments=segments)
    parameters['El'] = El
    # Add cost parameters
    for u in parameters['players_with_solar']:
        parameters[f'c_res_{u}'] = parameters['c_res']
        for t in time_periods:
            parameters[f'renewable_cap_{u}_{t}'] = pv_production[t]  # MW
    for u in parameters['players_with_wind']:
        parameters[f'c_res_{u}'] = parameters['c_res']
        for t in time_periods:
            parameters[f'renewable_cap_{u}_{t}'] = wind_production[t]  # MW

    for u in parameters['players_with_heatpumps']:
        parameters[f'c_hp_{u}'] = parameters['c_hp']
        parameters[f'nu_cop_{u}'] = parameters['nu_cop']
        parameters[f'c_su_H_{u}'] = parameters['c_su_H']
    for i,u in enumerate(parameters['players_with_electrolyzers']):
        parameters[f'c_els_{u}'] = parameters['c_els']
        parameters[f'c_su_G_{u}'] = parameters['c_su_G']
    for u in parameters['players_with_heatpumps']:
        parameters[f'c_su_H_{u}'] = parameters['c_su_H']
    for u in parameters['players_with_elec_storage']:
        parameters[f'c_sto_E_{u}'] = parameters['c_sto_E']
    for u in parameters['players_with_hydro_storage']:
        parameters[f'c_sto_G_{u}'] = parameters['c_sto_G']
    for u in parameters['players_with_heat_storage']:
        parameters[f'c_sto_H_{u}'] = parameters['c_sto_H']
    
    # Add grid prices
    elec_prices = ElectricityPriceGenerator(use_korean_price=use_korean_price, tou=use_tou_elec).generate_price(import_factor=import_factor, month=month, time_horizon=24)
    """
    hydrogen, heat price의 tou는 차후 구현
    """
    h2_prices = generate_hydrogen_price(base_price_eur=base_h2_price_eur, import_factor=import_factor, time_horizon=24)
    heat_prices = HeatPriceGenerator().get_profiles(month=month, import_factor=import_factor, customer_type='residential', use_seasonal=False)
    parameters = update_market_price(parameters, time_periods, elec_prices, h2_prices, heat_prices)

    parameters[f"pi_E_peak"] = np.mean(elec_prices["import"])*peak_penalty_ratio
    
    
    # DEMANDS
    elec_generator = ElectricityLoadGenerator(num_households=num_households)
    heat_generator = HeatLoadGenerator(num_households=num_households)
    hydro_generator = HydrogenLoadGenerator()
    # hydro_generator.generate_profiles()
    elec_demand_mwh = elec_generator.generate_community_load(monthly_base_load_mwh_per_household=0.363, season='winter', num_days=10, variability='normal', method='empirical')
    hydro_demand_kg = hydro_generator.get_profiles()
    heat_demand_mwh = heat_generator.get_profiles(month=1)
    for u in players:
        for t in time_periods:
            # HYDROGEN DEMAND
            if u in parameters['players_with_nfl_hydro_demand']:
                parameters[f'd_G_nfl_{u}_{t}'] = hydro_demand_kg[t]           
            # ELEC DEMAND
            if u in parameters['players_with_nfl_elec_demand']:
                parameters[f'd_E_nfl_{u}_{t}'] = elec_demand_mwh[t]
                        
            # HEAT DEMAND - NEW: Using Busan data
            if u in configuration['players_with_nfl_heat_demand']:
                # Use Korean building data instead of synthetic profile
                parameters[f'd_H_nfl_{u}_{t}'] = heat_demand_mwh[t]
            else:
                parameters[f'd_H_nfl_{u}_{t}'] = 0
    parameters["i_E_cap"] = np.max(elec_demand_mwh)
    parameters["i_H_cap"] = np.max(heat_demand_mwh)
    parameters["i_G_cap"] = np.max(hydro_demand_kg)


    for u in parameters['players_with_renewables']:
        if u in parameters['players_with_solar']:
            total_solar_cap = np.max(pv_production)
            parameters[f"e_E_cap_{u}"] = parameters["e_E_cap_ratio"] * total_solar_cap
            if u in parameters['players_with_elec_storage']:
                parameters[f"storage_power_E_{u}"] = parameters["storage_power_ratio_E"] * total_solar_cap
                parameters[f"storage_capacity_E_{u}"] = parameters["storage_capacity_ratio_E"] * parameters[f"storage_power_E_{u}"]
                parameters[f"initial_soc_E_{u}"] = parameters["initial_soc_ratio_E"] * parameters[f"storage_capacity_E_{u}"]
        elif u in parameters['players_with_wind']:
            total_wind_cap = np.max(wind_production)
            parameters[f"e_E_cap_{u}"] = parameters["e_E_cap_ratio"] * total_wind_cap
            if u in parameters['players_with_elec_storage']:
                parameters[f"storage_power_E_{u}"] = parameters["storage_power_ratio_E"] * total_wind_cap
                parameters[f"storage_capacity_E_{u}"] = parameters["storage_capacity_ratio_E"] * parameters[f"storage_power_E_{u}"]
                parameters[f"initial_soc_E_{u}"] = parameters["initial_soc_ratio_E"] * parameters[f"storage_capacity_E_{u}"]
        else:
            raise ValueError(f"Player {u} is not in any of the renewable or demand players")
    for u in parameters['players_with_nfl_elec_demand']:
        if u not in parameters['players_with_renewables']:
            if u in parameters['players_with_elec_storage']:
                total_elec_cap = np.max(elec_demand_mwh)
                parameters[f"e_E_cap_{u}"] = parameters["e_E_cap_ratio"] * total_elec_cap
                parameters[f"storage_power_E_{u}"] = parameters["storage_power_ratio_E"] * total_elec_cap
                parameters[f"storage_capacity_E_{u}"] = parameters["storage_capacity_ratio_E"] * parameters[f"storage_power_E_{u}"]
                parameters[f"initial_soc_E_{u}"] = parameters["initial_soc_ratio_E"] * parameters[f"storage_capacity_E_{u}"]
    if set(parameters['players_with_solar']) & set(parameters['players_with_wind']):
        raise ValueError("Solar and wind players cannot be the same")
    # total_elec_cap = np.max(wind_production)
    total_els_cap = El['p_els_cap']# * len(parameters['players_with_electrolyzers'])
    total_heat_cap = parameters['hp_cap']# * len(parameters['players_with_heatpumps'])
    # parameters["e_E_cap"] = parameters["e_E_cap_ratio"] * total_elec_cap
    parameters["e_H_cap"] = parameters["e_H_cap_ratio"] * total_heat_cap
    parameters["e_G_cap"] = parameters["e_G_cap_ratio"] * total_els_cap
    # parameters["storage_power_E"] = parameters["storage_power_ratio_E"] * total_elec_cap
    parameters["storage_power_G"] = parameters["storage_power_ratio_G"] * total_els_cap
    parameters["storage_power_H"] = parameters["storage_power_ratio_H"] * total_heat_cap
    # parameters["storage_capacity_E"] = parameters["storage_capacity_ratio_E"] * parameters["storage_power_E"]
    parameters["storage_capacity_G"] = parameters["storage_capacity_ratio_G"] * parameters["storage_power_G"]
    parameters["storage_capacity_H"] = parameters["storage_capacity_ratio_H"] * parameters["storage_power_H"]
    # parameters["initial_soc_E"] = parameters["initial_soc_ratio_E"] * parameters["storage_capacity_E"]
    parameters["initial_soc_G"] = parameters["initial_soc_ratio_G"] * parameters["storage_capacity_G"]
    parameters["initial_soc_H"] = parameters["initial_soc_ratio_H"] * parameters["storage_capacity_H"]
    # Pure Consumer Utility Function
    N_E = len(parameters['players_with_nfl_elec_demand'])
    if N_E > 0:
        factor_E = np.random.uniform(low=1.0, high=1.0+epsilon_log(N_E), size=N_E)
    N_H = len(parameters['players_with_nfl_heat_demand'])
    if N_H > 0:
        factor_H = np.random.uniform(low=1.0, high=1.0+epsilon_log(N_H), size=N_H)
    N_G = len(parameters['players_with_nfl_hydro_demand'])
    if N_G > 0:
        factor_G = np.random.uniform(low=1.0, high=1.0+epsilon_log(N_G), size=N_G)
    for t in time_periods:
        for i,u in enumerate(parameters['players_with_nfl_elec_demand']):
            u_E = parameters[f"pi_E_gri_import_{t}"] * factor_E[i]
            parameters[f"u_E_{u}_{t}"] = u_E
        for i,u in enumerate(parameters['players_with_nfl_heat_demand']):
            u_H = parameters[f"pi_H_gri_import_{t}"] * factor_H[i]
            parameters[f"u_H_{u}_{t}"] = u_H
        for i,u in enumerate(parameters['players_with_nfl_hydro_demand']):
            u_G = parameters[f"pi_G_gri_import_{t}"] * factor_G[i]
            parameters[f"u_G_{u}_{t}"] = u_G
    return parameters
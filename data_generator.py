"""
LEM Parameters Setup
Shared parameter configuration for both compact and column generation formulations
"""

import numpy as np
import pandas as pd
from HeatGen import HeatLoadGenerator, HeatPriceGenerator
from HydroGen import HydrogenLoadGenerator, generate_hydrogen_price
from ElecGen import ElectricityLoadGenerator, ElectricityProdGenerator, ElectricityPriceGenerator






def update_market_price(parameters, time_periods, elec_prices, h2_prices, heat_prices):
    
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
        use_tou = sensitivity_analysis['use_tou']
        month = sensitivity_analysis['month']
        storage_capacity_E = sensitivity_analysis['storage_capacity_E']
        storage_capacity_G = sensitivity_analysis['storage_capacity_G']
        storage_capacity_heat = sensitivity_analysis['storage_capacity_heat']
        hp_cap = sensitivity_analysis['hp_cap']
        els_cap = sensitivity_analysis['els_cap']
        res_cap = sensitivity_analysis['res_cap']
        num_households = sensitivity_analysis['num_households']
        nu_cop = sensitivity_analysis['nu_cop']
        c_su_G = sensitivity_analysis['c_su_G']
        c_su_H = sensitivity_analysis['c_su_H']
        base_h2_price_eur = sensitivity_analysis['base_h2_price_eur']
        e_E_cap = res_cap * sensitivity_analysis['e_E_cap']
        e_H_cap = hp_cap * sensitivity_analysis['e_H_cap']
    else:
        use_korean_price = True
        use_tou = True
        month = 1
        storage_capacity_E = 1.0 # [0.5, 1.0, 1.5]
        storage_capacity_G = 50
        storage_capacity_heat = 0.40
        hp_cap = 0.8 # [0.6, 0.8, 1.0]
        els_cap = 1 # [0.5, 1.0, 1.5]
        res_cap = 2
        num_households = 52 # [50, 75, 100]
        nu_cop = 3.28 # [3.0, 3.28, 3.5]
        c_su_G = 50 # [50, 75, 100]
        c_su_H = 10 # [5, 10, 15]
        base_h2_price_eur = 2.1*1.5 # [2.1*0.75, 2.1, 2.1*2]
        e_E_cap = res_cap * 2 # [0.5, 1.0, 1.5, 2.0]
        e_H_cap = hp_cap * 2 # [0.5, 1.0, 1.5, 2.0]
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
        'storage_capacity_E': storage_capacity_E,
        'storage_capacity_G': storage_capacity_G, #kg
        'storage_capacity_heat': storage_capacity_heat,
        'storage_power_E': 0.5,
        'storage_power_G': 0.25,
        'storage_power_heat': 0.5,
        'initial_soc_E': 0.5*1,
        'initial_soc_G': 25,
        'initial_soc_H': 0.2,
        'nu_ch_E': 0.95,
        'nu_dis_E': 0.95,        
        'nu_ch_G': 0.95,
        'nu_dis_G': 0.95,
        'nu_ch_H': 0.9,
        'nu_dis_H': 0.9,
        'nu_loss_H':0.002,
        # Equipment capacities
        'res_cap': res_cap,
        'hp_cap': hp_cap,
        'els_cap': els_cap,
        'phi1_1': 21.12266316,
        'phi0_1': -0.37924094,
        'phi1_2': 16.66883134,
        'phi0_2': 0.87814262,
        'c_res': 0.05,
        'c_hp': 2.69,
        'c_els': 0.05,
        'nu_cop': nu_cop,
        # Grid connection limits
        'res_capacity': 2,
        'e_E_cap': e_E_cap,
        'i_E_cap': 500,
        'e_H_cap': e_H_cap, #0.06,
        'i_H_cap': 500, #0.08,
        'e_G_cap': 50,
        'i_G_cap': 100 , #30,
        
        # Cost parameters
        'c_sto_E': 0.01,
        'c_sto_G': 0.01,
        'c_sto_H': 0.01,

        # Unit commitment parameters
        'min_down_time_G': 2,
        'c_RU_H': 0.9,
        'c_RD_H': 0.9,
        'c_RSU_H': 0.9,
        'c_RSD_H': 0.9,
        'c_su_G': 50,
        'c_su_H': 10,
        'c_min_G': 0.15, #0.15MW
        'c_sb_G': 0.01, #0.01MW
        'c_max_G': 1.0, #1.0MW,
        'c_min_H': 0.2,
        'c_max_H': 0.8
    }
    parameters['players_with_fl_elec_demand'] = list(set(
        parameters['players_with_electrolyzers'] + parameters['players_with_heatpumps']
    ))
    

    electricity_prod_generator = ElectricityProdGenerator(num_units=1, wind_el_ratio=2.0, solar_el_ratio=1.0, el_cap_mw=parameters["els_cap"])
    wind_production = electricity_prod_generator.generate_wind_production()
    pv_production = electricity_prod_generator.generate_solar_production()
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
    elec_prices = ElectricityPriceGenerator(use_korean_price=use_korean_price, tou=use_tou).generate_price(month=month, time_horizon=24)
    """
    hydrogen, heat price의 tou는 차후 구현
    """
    h2_prices = generate_hydrogen_price(base_price_eur=base_h2_price_eur, time_horizon=24)
    heat_prices = HeatPriceGenerator().get_profiles(month=month, customer_type='residential', use_seasonal=False)
    parameters = update_market_price(parameters, time_periods, elec_prices, h2_prices, heat_prices)
    
    
    # DEMANDS
    num_households = 52
    elec_generator = ElectricityLoadGenerator(num_households=num_households)
    heat_generator = HeatLoadGenerator(num_households=num_households)
    hydro_generator = HydrogenLoadGenerator()
    # hydro_generator.generate_profiles()
    elec_demand_mwh = elec_generator.generate_community_load(monthly_base_load_mwh_per_household=0.363, season='winter', num_days=10, variability='normal', method='empirical')
    hydro_demand_kg = hydro_generator.get_profiles()*0.5
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
    
    return parameters
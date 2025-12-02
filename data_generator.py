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


def setup_lem_parameters(players, configuration, time_periods):
    """
    Setup parameters for Local Energy Market problem
    
    Args:
        players: List of player IDs
        time_periods: List of time period indices
        
    Returns:
        dict: Complete parameter dictionary
    """
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
        'pi_peak': 100,
        
        # Storage parameters
        'storage_power_E': 0.25, #0.25일땐 u2가 이득, 근데 0.5일땐 손해. 왜 그럴까??
        'storage_capacity_E': 2.0,
        'storage_power_G': 10, #kg/h
        'storage_capacity_G': 50, #kg
        'initial_soc_E': 0.5*1,
        'initial_soc_G': 25,
        'initial_soc_H': 0.2,
        'storage_power_heat': 0.5, #ratio
        'storage_capacity_heat': 0.40,
        'nu_ch_E': 0.95,
        'nu_dis_E': 0.95,        
        'nu_ch_G': 0.95,
        'nu_dis_G': 0.95,
        'nu_ch_H': 0.9,
        'nu_dis_H': 0.9,
        'nu_loss_H':0.002,
        # Equipment capacities
        'hp_cap': 0.8,
        'els_cap': 1,
        'C_min': 0.15,
        'C_sb': 0.01,
        'phi1_1': 21.12266316,
        'phi0_1': -0.37924094,
        'phi1_2': 16.66883134,
        'phi0_2': 0.87814262,
        'c_res': 0.05,
        'c_hp': 2.69,
        'c_els': 0.05,
        'c_su_G': 50,
        'c_su_H': 10,
        'max_up_time': 3,
        'min_down_time': 2,
        'nu_COP': 3.28,
        # Grid connection limits
        'res_capacity': 2,
        'e_E_cap': 0.5*2,
        'i_E_cap': 0.5*2,
        'e_H_cap': 500, #0.06,
        'i_H_cap': 500, #0.08,
        'e_G_cap': 50,
        'i_G_cap': 100 , #30,
        
        # Cost parameters
        'c_sto_E': 0.01,
        'c_sto_G': 0.01,
        'c_sto_H': 0.01,
    }
    
    parameters['players_with_fl_elec_demand'] = list(set(
        parameters['players_with_electrolyzers'] + parameters['players_with_heatpumps']
    ))
    

    electricity_prod_generator = ElectricityProdGenerator(num_units=1, wind_el_ratio=2.0, solar_el_ratio=1.0, el_cap_mw=parameters["els_cap"])
    wind_production = electricity_prod_generator.generate_wind_production()
    # Add cost parameters
    for u in parameters['players_with_renewables']:
        parameters[f'c_res_{u}'] = parameters['c_res']
        # RENEWABLE AVAILABILITY - Natural solar curve
        for t in time_periods:
            parameters[f'renewable_cap_{u}_{t}'] = wind_production[t]  # MW
    for u in parameters['players_with_heatpumps']:
        parameters[f'c_hp_{u}'] = parameters['c_hp']
        parameters[f'nu_COP_{u}'] = parameters['nu_COP']
        parameters[f'c_su_H_{u}'] = parameters['c_su_H']
    for i,u in enumerate(parameters['players_with_electrolyzers']):
        parameters[f'c_els_{u}'] = parameters['c_els']
        parameters[f'c_su_G_{u}'] = parameters['c_su_G']
    for u in parameters['players_with_elec_storage']:
        parameters[f'c_sto_E_{u}'] = parameters['c_sto_E']
    for u in parameters['players_with_hydro_storage']:
        parameters[f'c_sto_G_{u}'] = parameters['c_sto_G']
    for u in parameters['players_with_heat_storage']:
        parameters[f'c_sto_H_{u}'] = parameters['c_sto_H']
    
    # Add grid prices
    elec_prices = ElectricityPriceGenerator(use_korean_price=True, tou=True).generate_price(month=1, time_horizon=24)
    h2_prices = generate_hydrogen_price(base_price_eur=4.1, tou=False, time_horizon=24)
    heat_prices = HeatPriceGenerator().get_profiles(1, 'residential', False)
    parameters = update_market_price(parameters, time_periods, elec_prices, h2_prices, heat_prices)
    
    
    # DEMANDS
    num_households = 100
    elec_generator = ElectricityLoadGenerator(num_households=num_households)
    heat_generator = HeatLoadGenerator(num_households=num_households)
    hydro_generator = HydrogenLoadGenerator()
    # hydro_generator.generate_profiles()
    elec_demand_mwh = elec_generator.generate_community_load(monthly_base_load_mwh_per_household=0.036*3, season='summer', num_days=1, variability='normal', method='empirical')
    hydro_demand_kg = hydro_generator.get_profiles()
    heat_demand_mwh = heat_generator.get_profiles(month=11)
    for u in players:
        for t in time_periods:
            # HYDROGEN DEMAND
            if u in parameters['players_with_nfl_hydro_demand']:
                parameters[f'd_G_nfl_{u}_{t}'] = hydro_demand_kg[t]
            #     if 6 <= t <= 11:
            #         h2_demand = 6 + 4 * np.exp(-((t-9)/2)**2)
            #     elif 14 <= t <= 20:
            #         h2_demand = 4 + 3 * np.exp(-((t-17)/2)**2)
            #     elif 12 <= t <= 13:
            #         h2_demand = 3.0
            #     elif 21 <= t <= 23:
            #         h2_demand = 2.0
            #     else:
            #         h2_demand = 1.0
            #     parameters[f'd_G_nfl_{u}_{t}'] = h2_demand
            # else:
            #     parameters[f'd_G_nfl_{u}_{t}'] = 0
            
            # ELEC DEMAND
            if u in parameters['players_with_nfl_elec_demand']:
                # morning_peak = 20 * np.exp(-((t - 8) / 2)**2)
                # evening_peak = 40 * np.exp(-((t - 19.5) / 2)**2)
                # base_demand = 60
                # elec_demand = (base_demand + morning_peak + evening_peak) * 0.001
                parameters[f'd_E_nfl_{u}_{t}'] = elec_demand_mwh[t]
                        
            # HEAT DEMAND - NEW: Using Busan data
            if u in configuration['players_with_nfl_heat_demand']:
                # Use Korean building data instead of synthetic profile
                parameters[f'd_H_nfl_{u}_{t}'] = heat_demand_mwh[t]
            else:
                parameters[f'd_H_nfl_{u}_{t}'] = 0
            # if u in parameters['players_with_nfl_heat_demand']:
            #     heat_demand_kw = 60 + 30 * np.cos(2 * np.pi * (t - 3) / 24)
            #     heat_demand = heat_demand_kw * 0.001
            #     parameters[f'd_H_nfl_{u}_{t}'] = heat_demand
            # else:
            #     parameters[f'd_H_nfl_{u}_{t}'] = 0
    
    return parameters
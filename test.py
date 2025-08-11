import numpy as np
from pyscipopt import SCIP_PARAMSETTING
from energy_solver import EnergyCommunitySolver
from energy_pricer import EnergyPricer
from energy_pricing_network import solve_energy_pricing_problem

def generate_initial_patterns(players, time_periods, params):
    """Generate initial feasible patterns for each player"""
    
    init_patterns = {}
    
    for u in players:
        pattern = {}
        
        # Simple initial pattern: only use grid import/export
        pattern['i_E_gri'] = {}
        pattern['e_E_gri'] = {}
        pattern['i_E_com'] = {}
        pattern['e_E_com'] = {}
        pattern['i_H_gri'] = {}
        pattern['e_H_gri'] = {}
        pattern['i_H_com'] = {}
        pattern['e_H_com'] = {}
        pattern['i_G_gri'] = {}
        pattern['e_G_gri'] = {}
        pattern['i_G_com'] = {}
        pattern['e_G_com'] = {}
        
        for t in time_periods:
            # Meet electricity demand from grid
            d_E = params.get(f'd_E_nfl_{u}_{t}', 0)
            pattern['i_E_gri'][t] = d_E
            pattern['e_E_gri'][t] = 0
            pattern['i_E_com'][t] = 0
            pattern['e_E_com'][t] = 0
            
            # Meet heat demand from grid
            d_H = params.get(f'd_H_nfl_{u}_{t}', 0)
            pattern['i_H_gri'][t] = d_H
            pattern['e_H_gri'][t] = 0
            pattern['i_H_com'][t] = 0
            pattern['e_H_com'][t] = 0
            
            # Meet hydrogen demand from grid
            d_G = params.get(f'd_G_nfl_{u}_{t}', 0)
            pattern['i_G_gri'][t] = d_G
            pattern['e_G_gri'][t] = 0
            pattern['i_G_com'][t] = 0
            pattern['e_G_com'][t] = 0
        
        init_patterns[u] = pattern
    
    return init_patterns


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
    # init_patterns = generate_initial_patterns(players, time_periods, parameters)
    init_patterns = solver.gen_initial_cols(folder_path=None)
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
    players = ['u1', 'u2', 'u3']
    time_periods = list(range(24))  # 24 hours
    
    # Example parameters
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
        
        # Storage parameters
        'storage_power': 50,
        'storage_capacity': 200,
        'initial_soc': 100,
        
        # Equipment capacities
        'renewable_cap_u1': 150,
        'hp_cap_u3': 80,
        'els_cap_u2': 100,
        
        # Grid connection limits
        'e_E_cap_u1_t': 100,
        'i_E_cap_u1_t': 150,
        'e_H_cap_u3_t': 60,
        'i_H_cap_u3_t': 80,
        'e_G_cap_u2_t': 50,
        'i_G_cap_u2_t': 30,
        
        # Cost parameters
        'c_sto': 0.01,
        
        # Electrolyzer parameters
        'C_max_u2': 100,
        'C_min_u2': 20,
        'C_sb_u2': 10,
        'phi1_u2': 0.7,
        'phi0_u2': 0.0,
    }
    
    # Add demand data
    for u in players:
        for t in time_periods:
            parameters[f'd_E_nfl_{u}_{t}'] = 10 + 5 * np.sin(2 * np.pi * t / 24)
            parameters[f'd_H_nfl_{u}_{t}'] = 5 + 2 * np.sin(2 * np.pi * t / 24)
            parameters[f'd_G_nfl_{u}_{t}'] = 2
    
    # Add cost parameters
    for u in players:
        parameters[f'c_res_{u}'] = 0.05
        parameters[f'c_hp_{u}'] = 0.1
        parameters[f'c_els_{u}'] = 0.08
        parameters[f'c_su_{u}'] = 50
    
    # Add grid prices
    for t in time_periods:
        parameters[f'pi_E_gri_{t}'] = 0.2 + 0.1 * np.sin(2 * np.pi * t / 24)
        parameters[f'pi_H_gri_{t}'] = 0.15
        parameters[f'pi_G_gri_{t}'] = 0.3
    
    # Add renewable availability
    for t in time_periods:
        # Solar availability pattern (higher during day)
        if 6 <= t <= 18:
            parameters[f'renewable_availability_u1_{t}'] = 0.8 + 0.2 * np.sin((t - 6) * np.pi / 12)
        else:
            parameters[f'renewable_availability_u1_{t}'] = 0.0
    
    # Add COP for heat pump
    parameters['nu_COP_u3'] = 3.0
    
    # Solve with convex hull pricing
    solver, pricer, prices = solve_convex_hull_pricing(players, time_periods, parameters)
    
    if solver is not None:
        print("\n=== Convex Hull Pricing completed successfully ===")
    else:
        print("\n=== Convex Hull Pricing failed ===")
import numpy as np
from energy_solver import EnergyCommunitySolver
from energy_pricer import EnergyPricer
from create_initial_patterns import create_initial_patterns_for_players
from pyscipopt import SCIP_PARAMSETTING

# Define example data
players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
time_periods = list(range(24))  # 24 hours

# Create minimal parameters for testing
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
    'players_with_fl_elec_demand': ['u2', 'u3'],
    
    # Basic parameters
    'nu_ch': 0.95,
    'nu_dis': 0.95,
    'pi_peak': 50,
    'c_sto': 0.01,
    'storage_power': 0.5,
    'storage_capacity': 2.0,
    'initial_soc': 0.5,
    
    # Equipment capacities
    'hp_cap_u3': 0.08,
    'els_cap_u2': 1,
    'C_min_u2': 0.15,
    'C_sb_u2': 0.01,
    'phi1_1_u2': 21.12266316,
    'phi0_1_u2': -0.37924094,
    'phi1_2_u2': 16.66883134,
    'phi0_2_u2': 0.87814262,
    'c_els_u2': 0.05,
    'c_su_u2': 50,
    'nu_COP_u3': 3.0,
    'min_down_time': 1,
    
    # Costs
    'c_res_u1': 0.05,
    'c_hp_u3': 0.1,
}

# Add time-dependent data
for t in time_periods:
    # Renewable capacity (solar curve)
    if 6 <= t <= 18:
        solar_factor = np.exp(-((t - 12) / 3.5)**2)
        parameters[f'renewable_cap_u1_{t}'] = 2 * solar_factor
    else:
        parameters[f'renewable_cap_u1_{t}'] = 0
    
    # Grid prices
    base_price = 0.1 + 0.05 * np.sin(2 * np.pi * t / 24)
    parameters[f'pi_E_gri_import_{t}'] = base_price * 1.1
    parameters[f'pi_E_gri_export_{t}'] = base_price
    parameters[f'pi_H_gri_import_{t}'] = 0.3
    parameters[f'pi_H_gri_export_{t}'] = 0.25
    parameters[f'pi_G_gri_import_{t}'] = 2.5
    parameters[f'pi_G_gri_export_{t}'] = 2.1
    
    # Demands
    parameters[f'd_E_nfl_u4_{t}'] = 0.06 + 0.02 * np.sin(2 * np.pi * (t - 8) / 24)
    parameters[f'd_G_nfl_u5_{t}'] = 3 + 1 * np.sin(2 * np.pi * (t - 12) / 24)
    parameters[f'd_H_nfl_u6_{t}'] = 6 + 2 * np.cos(2 * np.pi * (t - 3) / 24)
    
    # Grid connection limits
    parameters[f'e_E_cap_u1_{t}'] = 0.1
    parameters[f'i_E_cap_u1_{t}'] = 0.5
    parameters[f'i_E_cap_u2_{t}'] = 0.5
    parameters[f'i_E_cap_u3_{t}'] = 0.5
    parameters[f'i_E_cap_u4_{t}'] = 0.5
    parameters[f'e_H_cap_u3_{t}'] = 0.06
    parameters[f'i_H_cap_u3_{t}'] = 0.08
    parameters[f'i_H_cap_u6_{t}'] = 0.08
    parameters[f'e_G_cap_u2_{t}'] = 50
    parameters[f'i_G_cap_u2_{t}'] = 30
    parameters[f'i_G_cap_u5_{t}'] = 30

print("="*80)
print("COLUMN GENERATION WITH SIMPLE INITIAL PATTERNS")
print("="*80)

# Step 1: Create initial patterns
print("\nStep 1: Creating initial patterns...")
init_patterns = create_initial_patterns_for_players(players, time_periods, parameters)

# Verify patterns
print("\nInitial patterns created:")
for player in players:
    pattern = init_patterns[player]
    print(f"  Player {player}: {len(pattern)} variable types")

# Step 2: Create solver and initialize RMP
print("\nStep 2: Initializing RMP...")
solver = EnergyCommunitySolver(players, time_periods, parameters)

# Initialize RMP with the simple patterns
solver.init_rmp(init_patterns)
print(f"  RMP initialized with {solver.pattern_counter} patterns")

# Check constraint structure
print("\n  Constraint counts:")
for cons_type in solver.cons_flatten:
    if isinstance(solver.cons_flatten[cons_type], dict):
        print(f"    {cons_type}: {len(solver.cons_flatten[cons_type])} constraints")

# Step 3: Create pricer
print("\nStep 3: Creating pricer...")
pricer = EnergyPricer(
    players=players,
    time_periods=time_periods,
    parameters=parameters,
    cons_flatten=solver.cons_flatten,
    cons_coeff=solver.cons_coeff
)

# Include pricer in the model
solver.model.includePricer(pricer, "EnergyPricer", "Energy Community Pricer")

# Step 4: Set SCIP parameters
print("\nStep 4: Setting SCIP parameters...")
solver.model.setPresolve(SCIP_PARAMSETTING.OFF)
solver.model.setHeuristics(SCIP_PARAMSETTING.OFF)
solver.model.disablePropagation()
solver.model.setSeparating(SCIP_PARAMSETTING.OFF)

# Add some debug output
solver.model.setIntParam('display/verblevel', 4)  # More verbose output

# Step 5: Solve with column generation
print("\nStep 5: Solving with column generation...")
print("-"*80)

try:
    solver.model.optimize()
    
    # Check results
    status = solver.model.getStatus()
    print("-"*80)
    print(f"\nOptimization status: {status}")
    
    if status == "optimal":
        obj_val = solver.model.getObjVal()
        print(f"Objective value: {obj_val:.4f}")
        
        # Check pattern usage
        print("\nPattern usage:")
        pattern_vars = [v for v in solver.model.getVars() if v.name.startswith("pattern_")]
        active_patterns = 0
        for var in pattern_vars:
            val = solver.model.getVal(var)
            if val > 1e-6:
                print(f"  {var.name}: {val:.4f}")
                active_patterns += 1
        
        if active_patterns == 0:
            print("  No patterns are active (all have value 0)")
        
        # Check chi_peak value
        chi_peak_val = solver.model.getVal(solver.chi_peak)
        print(f"\nPeak power (chi_peak): {chi_peak_val:.4f} MW")
        
    else:
        print(f"\nOptimization failed with status: {status}")
        
        # Try to understand why
        if status == "infeasible":
            print("\nThe problem is infeasible. Possible reasons:")
            print("  1. Initial patterns don't satisfy all constraints")
            print("  2. Community balance constraints cannot be satisfied")
            print("  3. Peak power constraint is too restrictive")
            
            # Check if we can write the LP to debug
            print("\nWriting LP file for debugging...")
            solver.model.writeProblem("debug_rmp.lp")
            print("  LP file written to debug_rmp.lp")
            
except Exception as e:
    print(f"\nException occurred: {e}")
    print("\nTrying to write the problem for debugging...")
    try:
        solver.model.writeProblem("debug_error.lp")
        print("  Problem written to debug_error.lp")
    except:
        print("  Could not write problem file")
    
    # Additional debug info
    print("\nDebug information:")
    print(f"  Number of variables: {solver.model.getNVars()}")
    print(f"  Number of constraints: {solver.model.getNConss()}")
    
    # Check if patterns were added correctly
    print("\nPattern variables:")
    for player in players:
        if player in solver.model.data.get("vars_pattern", {}):
            vars_list = solver.model.data["vars_pattern"][player]
            print(f"  Player {player}: {len(vars_list)} pattern variables")
        else:
            print(f"  Player {player}: No pattern variables")

print("\n" + "="*80)
print("DONE")
print("="*80)
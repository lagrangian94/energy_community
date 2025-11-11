"""
Column Generation Test Script for Local Energy Market
Uses shared parameters from lem_parameters.py
"""

import sys
sys.path.append('/mnt/project')
sys.path.append('/home/claude')
sys.path.append('/mnt/user-data/outputs')

from data_generator import setup_lem_parameters
from colgen import ColumnGenerationSolver


def main():
    """
    Main test function - uses compact.py parameters
    """
    print("\n" + "="*80)
    print("COLUMN GENERATION FOR LOCAL ENERGY MARKET")
    print("Dantzig-Wolfe Decomposition Implementation")
    print("="*80)
    
    # Setup problem using compact.py's parameter function
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))
    
    print("\nSetting up parameters (using compact.py configuration)...")
    parameters = setup_lem_parameters(players, time_periods)
    
    print(f"\nâœ“ Parameters configured")
    print(f"  Players: {len(players)}")
    print(f"  Time periods: {len(time_periods)}")
    print(f"  Parameter entries: {len(parameters)}")
    
    # Solve with column generation
    print("\n" + "="*80)
    print("SOLVING WITH COLUMN GENERATION")
    print("="*80)
    
    try:
        cg_solver = ColumnGenerationSolver(players, time_periods, parameters)
        status, solution, obj_val = cg_solver.solve()
        
        if status == "optimal":
            print("\n" + "="*80)
            print("COLUMN GENERATION - SUCCESSFUL")
            print("="*80)
            print(f"Optimal objective: {obj_val:.2f} EUR")
            
            # Print convex hull prices
            if 'convex_hull_prices' in solution:
                print("\n" + "="*80)
                print("CONVEX HULL PRICES (Community Balance Shadow Prices)")
                print("="*80)
                chp = solution['convex_hull_prices']
                
                print("\nElectricity Prices (EUR/MWh):")
                for t in [0, 6, 12, 18, 23]:
                    if t in chp['electricity']:
                        print(f"  t={t:2d}: {chp['electricity'][t]:8.4f}")
                
                print("\nHeat Prices (EUR/MWh):")
                for t in [0, 6, 12, 18, 23]:
                    if t in chp['heat']:
                        print(f"  t={t:2d}: {chp['heat'][t]:8.4f}")
                
                print("\nHydrogen Prices (EUR/kg):")
                for t in [0, 6, 12, 18, 23]:
                    if t in chp['hydrogen']:
                        print(f"  t={t:2d}: {chp['hydrogen'][t]:8.4f}")
            
            print("\n" + "="*80)
            print("COMPLETED SUCCESSFULLY")
            print("="*80)
        
        else:
            print(f"\nColumn generation failed with status: {status}")
    
    except Exception as e:
        print(f"\nâœ— Error during column generation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    print(1)
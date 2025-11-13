"""
Column Generation for Local Energy Market with Convex Hull Pricing
Main solver implementation using Dantzig-Wolfe Decomposition
"""
import sys
sys.path.append('/mnt/project')
sys.path.append('/home/claude')

from pyscipopt import SCIP_PARAMSETTING
from typing import Dict, List, Tuple

from data_generator import setup_lem_parameters
from solver import PlayerSubproblem, MasterProblem
from pricer import LEMPricer


class ColumnGenerationSolver:
    """
    Main column generation solver for Local Energy Market
    Implements Dantzig-Wolfe decomposition to compute convex hull prices
    """
    def __init__(self, players: List[str], time_periods: List[int], parameters: Dict):
        """
        Initialize column generation solver
        
        Args:
            players: List of player IDs
            time_periods: List of time periods
            parameters: Model parameters
        """
        self.players = players
        self.time_periods = time_periods
        self.parameters = parameters
        
        # Create subproblems for each player
        print("=== Creating Subproblems ===")
        self.subproblems = {}
        for player in players:
            print(f"Creating subproblem for player {player}...")
            self.subproblems[player] = PlayerSubproblem(
                player=player,
                time_periods=time_periods,
                parameters=parameters,
                isLP=True
            )
        
        # Create master problem
        print("\n=== Creating Master Problem ===")
        self.master = MasterProblem(players, time_periods)
        
        print("Note: Initial columns will be generated via initial solve")
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[str, Dict, float]:
        """
        Solve using column generation to obtain convex hull prices
        
        Args:
            max_iterations: Maximum number of CG iterations (not used with SCIP pricer)
            tolerance: Convergence tolerance for reduced cost (not used with SCIP pricer)
            
        Returns:
            tuple: (status, solution, objective_value)
        """
        print("\n" + "="*80)
        print("COLUMN GENERATION - STARTING ITERATIONS")
        print("="*80)
                
        # Generate initial columns for each player
        print("\n=== Generating Initial Columns ===")
        self.master._add_initial_columns(self.subproblems)
        # Create master constraints
        self.master._create_master_constraints()
        # Disable presolving to allow pricer to work properly
        self.master.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.master.model.setHeuristics(SCIP_PARAMSETTING.OFF)
        self.master.model.setSeparating(SCIP_PARAMSETTING.OFF)
        self.master.model.disablePropagation()
        # Create pricer
        pricer = LEMPricer(
            master_model=self.master.model,
            subproblems=self.subproblems,
            time_periods=self.time_periods,
            players=self.players
        )
        
        # Include pricer in master problem
        self.master.model.includePricer(
            pricer,
            "LEMPricer",
            "Pricer for Local Energy Market column generation"
        )

        # Solve with column generation
        print("\nSolving master problem with pricing...")
        self.master.model.optimize()
        status = self.master.model.getStatus()
        
        if status == "optimal":
            obj_val = self.master.get_objective_value()
            solution = self.master.get_solution()
            
            print("\n" + "="*80)
            print("COLUMN GENERATION - COMPLETED")
            print("="*80)
            print(f"Optimal objective value: {obj_val:.2f} EUR")
            
            # Count columns per player
            print("\nColumns per player:")
            for player in self.players:
                player_cols = [k for k in self.master.model.data['vars'].keys() if k[0] == player]
                print(f"  {player}: {len(player_cols)} columns")
            
            # Extract and print convex hull prices (dual prices of community balance constraints)
            print("\n" + "="*80)
            print("CONVEX HULL PRICES (Shadow Prices of Community Balance)")
            print("="*80)
            
            chp_elec = {}
            chp_heat = {}
            chp_hydro = {}
            
            for t in self.time_periods:
                elec_cons = self.master.model.data['cons']['community_elec_balance'][t]
                heat_cons = self.master.model.data['cons']['community_heat_balance'][t]
                hydro_cons = self.master.model.data['cons']['community_hydro_balance'][t]
                
                try:
                    t_elec_cons = self.master.model.getTransformedCons(elec_cons)
                    t_heat_cons = self.master.model.getTransformedCons(heat_cons)
                    t_hydro_cons = self.master.model.getTransformedCons(hydro_cons)
                    
                    chp_elec[t] = self.master.model.getDualsolLinear(t_elec_cons) if t_elec_cons is not None else 0.0
                    chp_heat[t] = self.master.model.getDualsolLinear(t_heat_cons) if t_heat_cons is not None else 0.0
                    chp_hydro[t] = self.master.model.getDualsolLinear(t_hydro_cons) if t_hydro_cons is not None else 0.0
                except:
                    chp_elec[t] = 0.0
                    chp_heat[t] = 0.0
                    chp_hydro[t] = 0.0
            
            # Print sample of convex hull prices
            print("\nSample Convex Hull Prices (EUR/MWh or EUR/kg):")
            sample_times = [0, 6, 12, 18, 23] if len(self.time_periods) >= 24 else self.time_periods[:5]
            print(f"{'Time':>6} {'Electricity':>12} {'Heat':>12} {'Hydrogen':>12}")
            print("-" * 48)
            for t in sample_times:
                print(f"{t:6d} {chp_elec[t]:12.4f} {chp_heat[t]:12.4f} {chp_hydro[t]:12.4f}")
            
            # Store convex hull prices in solution
            solution['convex_hull_prices'] = {
                'electricity': chp_elec,
                'heat': chp_heat,
                'hydrogen': chp_hydro
            }
            
            return status, solution, obj_val
        else:
            print(f"\nColumn generation failed with status: {status}")
            return status, None, None


def main():
    """
    Main test function for column generation
    """
    print("\n" + "="*80)
    print("COLUMN GENERATION FOR LOCAL ENERGY MARKET")
    print("Dantzig-Wolfe Decomposition Implementation")
    print("="*80)
    
    # Setup problem
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))
    
    print("\nSetting up parameters...")
    parameters = setup_lem_parameters(players, time_periods)
    
    print(f"\n✓ Parameters configured")
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
        print(f"\n✗ Error during column generation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    print(1)
"""
Core Computation using Row Generation Algorithm
Based on: Drechsel & Kimms (2010) "Computing core allocations in cooperative games 
with an application to cooperative procurement"

Implements:
1. SeparationProblem: Finds most violated coalition given current payoffs
2. CoreComputation: Main row generation algorithm to find core allocation
"""

from pyscipopt import Model, quicksum
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import sys
sys.path.append('/mnt/project')
from compact import LocalEnergyMarket


class SeparationProblem(LocalEnergyMarket):
    """
    Separation problem that extends LocalEnergyMarket with binary selection variables.
    
    Solves: max Σ_i payoffs[i] * z[i] - cost(selected coalition)
    where z[i] ∈ {0,1} indicates if player i is in the selected coalition
    """
    
    def __init__(self, 
                 players: List[str],
                 time_periods: List[int],
                 parameters: Dict,
                 current_payoffs: Dict[str, float]):
        """
        Initialize separation problem
        
        Args:
            players: List of all player IDs
            time_periods: List of time period indices
            parameters: Dictionary containing all model parameters
            current_payoffs: Current payoff allocation {player_id: payoff}
        """
        self.current_payoffs = current_payoffs
        
        # Initialize parent class
        super().__init__(players, time_periods, parameters, isLP=False, dwr=False)
        
        # Add binary selection variables and constraints
        self._add_selection_variables()
        self._add_bigm_constraints()
        self._modify_balance_constraints()
        
    def _add_selection_variables(self):
        """Add binary selection variables z[i] for each player"""
        self.z = {}
        
        for i in self.players:
            # Objective coefficient is NEGATIVE of current payoff
            # We want to maximize Σ payoff[i]*z[i] - cost
            # which is equivalent to minimize -Σ payoff[i]*z[i] + cost
            # Since original model minimizes cost, we just set z coefficient to -payoff
            payoff = self.current_payoffs.get(i, 0.0)
            self.z[i] = self.model.addVar(
                vtype="B", 
                name=f"z_{i}",
                obj=-payoff  # NEGATIVE payoff for minimization
            )
            
        print(f"Added {len(self.z)} binary selection variables")
    
    def _add_bigm_constraints(self):
        """Add Big-M constraints to force variables to 0 when player is not selected"""
        
        M_default = 10000  # Default big-M value
        
        for u in self.players:
            z_u = self.z[u]
            
            for t in self.time_periods:
                # Grid trading - Electricity
                if (u, t) in self.e_E_gri:
                    M = self.params.get(f'e_E_cap_{u}_{t}', M_default)
                    self.model.addCons(self.e_E_gri[u,t] <= M * z_u, 
                                      name=f"bigm_e_E_gri_{u}_{t}")
                
                if (u, t) in self.i_E_gri:
                    res_capacity = 2
                    M = self.params.get(f'i_E_cap_{u}_{t}', 0.5) * res_capacity
                    if M < 1:  # For non-demand players
                        M = M_default
                    self.model.addCons(self.i_E_gri[u,t] <= M * z_u,
                                      name=f"bigm_i_E_gri_{u}_{t}")
                
                # Community trading - Electricity
                if (u, t) in self.e_E_com:
                    M = 1000
                    self.model.addCons(self.e_E_com[u,t] <= M * z_u,
                                      name=f"bigm_e_E_com_{u}_{t}")
                
                if (u, t) in self.i_E_com:
                    M = self.params.get(f'i_E_cap_{u}_{t}', 0.5) * 2
                    if M < 1:
                        M = 1000
                    self.model.addCons(self.i_E_com[u,t] <= M * z_u,
                                      name=f"bigm_i_E_com_{u}_{t}")
                
                # Grid trading - Heat
                if (u, t) in self.e_H_gri:
                    M = self.params.get(f'e_H_cap_{u}_{t}', 500)
                    self.model.addCons(self.e_H_gri[u,t] <= M * z_u,
                                      name=f"bigm_e_H_gri_{u}_{t}")
                
                if (u, t) in self.i_H_gri:
                    M = self.params.get(f'i_H_cap_{u}_{t}', 500)
                    self.model.addCons(self.i_H_gri[u,t] <= M * z_u,
                                      name=f"bigm_i_H_gri_{u}_{t}")
                
                # Community trading - Heat
                if (u, t) in self.e_H_com:
                    M = 500
                    self.model.addCons(self.e_H_com[u,t] <= M * z_u,
                                      name=f"bigm_e_H_com_{u}_{t}")
                
                if (u, t) in self.i_H_com:
                    M = 500
                    self.model.addCons(self.i_H_com[u,t] <= M * z_u,
                                      name=f"bigm_i_H_com_{u}_{t}")
                
                # Grid trading - Hydrogen
                if (u, t) in self.e_G_gri:
                    M = self.params.get(f'e_G_cap_{u}_{t}', 100)
                    self.model.addCons(self.e_G_gri[u,t] <= M * z_u,
                                      name=f"bigm_e_G_gri_{u}_{t}")
                
                if (u, t) in self.i_G_gri:
                    M = self.params.get(f'i_G_cap_{u}_{t}', 100)
                    self.model.addCons(self.i_G_gri[u,t] <= M * z_u,
                                      name=f"bigm_i_G_gri_{u}_{t}")
                
                # Community trading - Hydrogen
                if (u, t) in self.e_G_com:
                    M = 100
                    self.model.addCons(self.e_G_com[u,t] <= M * z_u,
                                      name=f"bigm_e_G_com_{u}_{t}")
                
                if (u, t) in self.i_G_com:
                    M = 100
                    self.model.addCons(self.i_G_com[u,t] <= M * z_u,
                                      name=f"bigm_i_G_com_{u}_{t}")
                
                # Production variables
                if (u, 'res', t) in self.p:
                    M = self.params.get(f'renewable_cap_{u}_{t}', 200)
                    self.model.addCons(self.p[u,'res',t] <= M * z_u,
                                      name=f"bigm_p_res_{u}_{t}")
                
                if (u, 'hp', t) in self.p:
                    M = self.params.get(f'hp_cap_{u}', 100)
                    self.model.addCons(self.p[u,'hp',t] <= M * z_u,
                                      name=f"bigm_p_hp_{u}_{t}")
                
                if (u, 'els', t) in self.p:
                    M = self.params.get(f'els_cap_{u}', 1) * 25  # Upper bound for hydrogen production
                    self.model.addCons(self.p[u,'els',t] <= M * z_u,
                                      name=f"bigm_p_els_{u}_{t}")
                
                # Electrolyzer demand
                if (u, t) in self.els_d:
                    M = self.params.get(f'els_cap_{u}', 1)
                    self.model.addCons(self.els_d[u,t] <= M * z_u,
                                      name=f"bigm_els_d_{u}_{t}")
                
                # Flexible demand
                if (u, 'elec', t) in self.fl_d:
                    M = 1  # From compact.py line 533
                    self.model.addCons(self.fl_d[u,'elec',t] <= M * z_u,
                                      name=f"bigm_fl_d_elec_{u}_{t}")
                
                if (u, 'hydro', t) in self.fl_d:
                    M = 10**6
                    self.model.addCons(self.fl_d[u,'hydro',t] <= M * z_u,
                                      name=f"bigm_fl_d_hydro_{u}_{t}")
                
                if (u, 'heat', t) in self.fl_d:
                    M = 10**6
                    self.model.addCons(self.fl_d[u,'heat',t] <= M * z_u,
                                      name=f"bigm_fl_d_heat_{u}_{t}")
                
                # Storage variables - Electricity
                if (u, t) in self.s_E:
                    M = self.params.get('storage_capacity', 100)
                    self.model.addCons(self.s_E[u,t] <= M * z_u,
                                      name=f"bigm_s_E_{u}_{t}")
                    
                if (u, t) in self.b_ch_E:
                    M = self.params.get('storage_power', 50)
                    self.model.addCons(self.b_ch_E[u,t] <= M * z_u,
                                      name=f"bigm_b_ch_E_{u}_{t}")
                    
                if (u, t) in self.b_dis_E:
                    M = self.params.get('storage_power', 50)
                    self.model.addCons(self.b_dis_E[u,t] <= M * z_u,
                                      name=f"bigm_b_dis_E_{u}_{t}")
                
                # Storage variables - Hydrogen
                if (u, t) in self.s_G:
                    M = 50  # From compact.py line 572
                    self.model.addCons(self.s_G[u,t] <= M * z_u,
                                      name=f"bigm_s_G_{u}_{t}")
                
                if (u, t) in self.b_ch_G:
                    M = 10  # From compact.py line 571
                    self.model.addCons(self.b_ch_G[u,t] <= M * z_u,
                                      name=f"bigm_b_ch_G_{u}_{t}")
                
                if (u, t) in self.b_dis_G:
                    M = 10
                    self.model.addCons(self.b_dis_G[u,t] <= M * z_u,
                                      name=f"bigm_b_dis_G_{u}_{t}")
                
                # Storage variables - Heat
                if (u, t) in self.s_H:
                    M = self.params.get('storage_capacity_heat', 400)
                    if M < 0:  # Check if parameter was set
                        M = 400
                    self.model.addCons(self.s_H[u,t] <= M * z_u,
                                      name=f"bigm_s_H_{u}_{t}")
                
                if (u, t) in self.b_ch_H:
                    M = self.params.get('storage_power_heat', 100)
                    if M < 0:
                        M = 100
                    self.model.addCons(self.b_ch_H[u,t] <= M * z_u,
                                      name=f"bigm_b_ch_H_{u}_{t}")
                
                if (u, t) in self.b_dis_H:
                    M = self.params.get('storage_power_heat', 100)
                    if M < 0:
                        M = 100
                    self.model.addCons(self.b_dis_H[u,t] <= M * z_u,
                                      name=f"bigm_b_dis_H_{u}_{t}")
                
                # Electrolyzer binary commitment variables
                if (u, t) in self.z_su:
                    self.model.addCons(self.z_su[u,t] <= z_u,
                                      name=f"bigm_z_su_{u}_{t}")
                
                if (u, t) in self.z_on:
                    self.model.addCons(self.z_on[u,t] <= z_u,
                                      name=f"bigm_z_on_{u}_{t}")
                
                if (u, t) in self.z_off:
                    self.model.addCons(self.z_off[u,t] <= z_u,
                                      name=f"bigm_z_off_{u}_{t}")
                
                if (u, t) in self.z_sb:
                    self.model.addCons(self.z_sb[u,t] <= z_u,
                                      name=f"bigm_z_sb_{u}_{t}")
        
        print(f"Added Big-M constraints for all variables")
    
    def _modify_balance_constraints(self):
        """
        Modify individual balance constraints to incorporate z[i]
        
        Original: LHS == nfl_d + fl_d
        Modified: LHS - fl_d == nfl_d_param * z[i]
        
        This ensures that when z[i] = 0:
        - RHS = 0
        - All variables are forced to 0 by Big-M constraints
        - Balance is automatically satisfied
        """
        
        # Remove old balance constraints
        for cons_name in list(self.elec_balance_cons.keys()):
            cons = self.elec_balance_cons[cons_name]
            self.model.delCons(cons)
        self.elec_balance_cons = {}
        
        for cons_name in list(self.heat_balance_cons.keys()):
            cons = self.heat_balance_cons[cons_name]
            self.model.delCons(cons)
        self.heat_balance_cons = {}
        
        for cons_name in list(self.hydro_balance_cons.keys()):
            cons = self.hydro_balance_cons[cons_name]
            self.model.delCons(cons)
        self.hydro_balance_cons = {}
        
        # Add modified electricity balance constraints
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.i_E_gri.get((u,t),0) - self.e_E_gri.get((u,t),0) + 
                       self.i_E_com.get((u,t),0) - self.e_E_com.get((u,t),0))
                
                # Add renewable generation
                lhs += self.p.get((u,'res',t),0)
                
                # Add electricity storage discharge/charge
                lhs += self.b_dis_E.get((u,t),0) - self.b_ch_E.get((u,t),0)
                
                # Subtract flexible demand from LHS
                lhs -= self.fl_d.get((u,'elec',t),0)
                
                # RHS: non-flexible demand * z[u]
                nfl_demand_param = self.params.get(f'd_E_nfl_{u}_{t}', 0)
                rhs = nfl_demand_param * self.z[u]
                
                if not (type(lhs) == int):
                    cons = self.model.addCons(lhs == rhs, name=f"elec_balance_modified_{u}_{t}")
                    self.elec_balance_cons[f"elec_balance_modified_{u}_{t}"] = cons
        
        # Add modified heat balance constraints
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.i_H_gri.get((u,t),0) - self.e_H_gri.get((u,t),0) + 
                       self.i_H_com.get((u,t),0) - self.e_H_com.get((u,t),0))
                
                # Add heat pump production
                lhs += self.p.get((u,'hp',t),0)
                
                # Add heat storage discharge/charge
                lhs += self.b_dis_H.get((u,t),0) - self.b_ch_H.get((u,t),0)
                
                # Subtract flexible demand from LHS
                lhs -= self.fl_d.get((u,'heat',t),0)
                
                # RHS: non-flexible demand * z[u]
                nfl_demand_param = self.params.get(f'd_H_nfl_{u}_{t}', 0)
                rhs = nfl_demand_param * self.z[u]
                
                if not (type(lhs) == int):
                    cons = self.model.addCons(lhs == rhs, name=f"heat_balance_modified_{u}_{t}")
                    self.heat_balance_cons[f"heat_balance_modified_{u}_{t}"] = cons
        
        # Add modified hydrogen balance constraints
        for u in self.players:
            for t in self.time_periods:
                lhs = (self.i_G_gri.get((u,t),0) - self.e_G_gri.get((u,t),0) + 
                       self.i_G_com.get((u,t),0) - self.e_G_com.get((u,t),0))
                
                # Add electrolyzer production
                lhs += self.p.get((u,'els',t),0)
                
                # Add hydrogen storage discharge/charge
                lhs += self.b_dis_G.get((u,t),0) - self.b_ch_G.get((u,t),0)
                
                # Subtract flexible demand from LHS
                lhs -= self.fl_d.get((u,'hydro',t),0)
                
                # RHS: non-flexible demand * z[u]
                nfl_demand_param = self.params.get(f'd_G_nfl_{u}_{t}', 0)
                rhs = nfl_demand_param * self.z[u]
                
                if not (type(lhs) == int):
                    cons = self.model.addCons(lhs == rhs, name=f"hydro_balance_modified_{u}_{t}")
                    self.hydro_balance_cons[f"hydro_balance_modified_{u}_{t}"] = cons
        
        print(f"Modified balance constraints to incorporate z variables")
    
    def solve_separation(self):
        """
        Solve the separation problem
        
        Returns:
            tuple: (selected_coalition, violation)
                - selected_coalition: list of selected player IDs
                - violation: objective value (positive if constraint is violated)
        """
        print("\n" + "="*60)
        print("Solving separation problem...")
        
        # Model minimizes: -Σ payoffs[i]*z[i] + cost
        # This is equivalent to maximizing: Σ payoffs[i]*z[i] - cost
        # So we keep the default minimize objective
        
        # Solve
        status = self.solve()
        
        if status != "optimal":
            print(f"Separation problem failed with status: {status}")
            return [], 0.0
        
        obj_val = self.model.getObjVal()
        
        # Extract selected coalition
        selected_coalition = []
        for i in self.players:
            z_val = self.model.getVal(self.z[i])
            if z_val > 0.5:  # Binary variable threshold
                selected_coalition.append(i)
        
        # Compute violation: Σ payoffs[i] - cost(S)
        # The objective value is: -Σ payoffs[i] + cost(S)
        # So violation = -obj_val
        violation = -obj_val
        
        print(f"Selected coalition: {selected_coalition}")
        print(f"Violation (Σ payoffs - cost): {violation:.4f}")
        print("="*60)
        
        return selected_coalition, violation


class CoreComputation:
    """
    Main class for computing core allocations using row generation algorithm
    """
    
    def __init__(self, 
                 players: List[str],
                 time_periods: List[int],
                 parameters: Dict):
        """
        Initialize core computation
        
        Args:
            players: List of all player IDs
            time_periods: List of time period indices
            parameters: Dictionary containing all model parameters
        """
        self.players = players
        self.time_periods = time_periods
        self.params = parameters
        
        # Cache for coalition costs
        self.coalition_costs = {}
        
        # Master problem model
        self.master_model = None
        self.payoff_vars = {}
        self.slack_var = None
        
        print(f"\n{'='*70}")
        print(f"Core Computation Initialized")
        print(f"Players: {players}")
        print(f"Number of players: {len(players)}")
        print(f"Time periods: {len(time_periods)}")
        print(f"{'='*70}\n")
    
    def compute_coalition_cost(self, coalition: List[str]) -> float:
        """
        Compute the cost c(S) for a given coalition S
        
        Args:
            coalition: List of player IDs in the coalition
            
        Returns:
            float: Optimal cost for the coalition (negative = profit)
        """
        # Use tuple as cache key (lists are not hashable)
        coalition_tuple = tuple(sorted(coalition))
        
        if coalition_tuple in self.coalition_costs:
            print(f"  Using cached cost for {coalition}: {self.coalition_costs[coalition_tuple]:.4f}")
            return self.coalition_costs[coalition_tuple]
        
        print(f"  Computing cost for coalition {coalition}...")
        
        # Create and solve LocalEnergyMarket for this coalition
        lem = LocalEnergyMarket(
            players=list(coalition),
            time_periods=self.time_periods,
            parameters=self.params,
            isLP=False,
            dwr=False
        )
        
        status = lem.solve()
        
        if status != "optimal":
            print(f"  WARNING: Coalition {coalition} optimization failed with status {status}")
            # Return a very high cost (bad for the coalition)
            return float('inf')
        
        cost = lem.model.getObjVal()
        self.coalition_costs[coalition_tuple] = cost
        
        print(f"  Coalition {coalition} cost: {cost:.4f}")
        
        return cost
    
    def initialize_master_problem(self, initial_coalitions: List[List[str]]) -> None:
        """
        Initialize the master problem with initial set of coalitions
        
        Args:
            initial_coalitions: Initial set of coalitions (typically singletons)
        """
        print("\n" + "="*70)
        print("Initializing Master Problem")
        print("="*70)
        
        self.master_model = Model("MasterProblem")
        
        # Create payoff variables p[i] for each player
        for i in self.players:
            self.payoff_vars[i] = self.master_model.addVar(
                vtype="C",
                name=f"p_{i}",
                lb=-float('inf')  # Payoffs can be negative
            )
        
        # Create slack variable v >= 0
        self.slack_var = self.master_model.addVar(
            vtype="C",
            name="v",
            lb=0,
            obj=1.0  # Minimize v
        )
        
        # Constraint: Efficiency (sum of payoffs = grand coalition cost)
        grand_coalition_cost = self.compute_coalition_cost(self.players)
        print(f"\nGrand coalition cost c(N): {grand_coalition_cost:.4f}")
        
        efficiency_cons = self.master_model.addCons(
            quicksum(self.payoff_vars[i] for i in self.players) == grand_coalition_cost,
            name="efficiency"
        )
        
        # Add initial coalition constraints
        print(f"\nAdding {len(initial_coalitions)} initial coalition constraints:")
        for coalition in initial_coalitions:
            self._add_coalition_constraint(coalition)
        
        print("="*70 + "\n")
    
    def _add_coalition_constraint(self, coalition: List[str]) -> None:
        """
        Add a coalition stability constraint to the master problem
        
        Constraint: Σ_{i∈S} p[i] <= c(S) + v
        
        Args:
            coalition: List of player IDs in the coalition
        """
        coalition_cost = self.compute_coalition_cost(coalition)
        
        coalition_str = "_".join(sorted(coalition))
        
        # Need to free transformed problem before adding constraints
        self.master_model.freeTransform()
        
        cons = self.master_model.addCons(
            quicksum(self.payoff_vars[i] for i in coalition) <= coalition_cost + self.slack_var,
            name=f"stability_{coalition_str}"
        )
        
        print(f"  Added constraint for {coalition}: Σp[i] <= {coalition_cost:.4f} + v")
    
    def solve_master_problem(self) -> Tuple[Dict[str, float], float]:
        """
        Solve the master problem
        
        Returns:
            tuple: (payoffs, slack)
                - payoffs: Dictionary {player_id: payoff}
                - slack: Value of slack variable v
        """
        print("\n" + "="*60)
        print("Solving Master Problem...")
        
        self.master_model.optimize()
        
        status = self.master_model.getStatus()
        if status != "optimal":
            print(f"Master problem failed with status: {status}")
            return {}, float('inf')
        
        # Extract solution
        payoffs = {}
        for i in self.players:
            payoffs[i] = self.master_model.getVal(self.payoff_vars[i])
        
        slack = self.master_model.getVal(self.slack_var)
        
        print(f"\nMaster problem solved:")
        print(f"  Slack v = {slack:.6f}")
        print(f"  Payoffs:")
        for i in self.players:
            print(f"    {i}: {payoffs[i]:.4f}")
        print("="*60)
        
        return payoffs, slack
    
    def find_violated_coalition(self, payoffs: Dict[str, float]) -> Tuple[List[str], float]:
        """
        Solve separation problem to find most violated coalition
        
        Args:
            payoffs: Current payoff allocation
            
        Returns:
            tuple: (coalition, violation)
                - coalition: Most violated coalition (empty if none found)
                - violation: Amount of violation (positive if violated)
        """
        print("\n" + "="*60)
        print("Finding violated coalition via Separation Problem")
        print(f"Current payoffs: {payoffs}")
        
        # Create and solve separation problem
        sep_problem = SeparationProblem(
            players=self.players,
            time_periods=self.time_periods,
            parameters=self.params,
            current_payoffs=payoffs
        )
        
        coalition, violation = sep_problem.solve_separation()
        
        # Compute actual violation for verification
        if len(coalition) > 0:
            coalition_cost = self.compute_coalition_cost(coalition)
            payoff_sum = sum(payoffs[i] for i in coalition)
            actual_violation = payoff_sum - coalition_cost
            
            print(f"\nVerification:")
            print(f"  Coalition payoff sum: {payoff_sum:.4f}")
            print(f"  Coalition cost c(S): {coalition_cost:.4f}")
            print(f"  Actual violation (Σ payoffs - cost): {actual_violation:.4f}")
            print(f"  Separation problem violation: {violation:.4f}")
            
            if abs(actual_violation - violation) > 1e-4:
                raise RuntimeError("Mismatch between actual and computed violation!")
            
        print("="*60)
        
        return coalition, violation
    
    def compute_core(self, 
                     max_iterations: int = 100,
                     tolerance: float = 1e-6) -> Optional[Dict[str, float]]:
        """
        Main row generation algorithm to compute core allocation
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dict[str, float]: Core allocation if exists, None otherwise
        """
        print("\n" + "="*70)
        print("STARTING ROW GENERATION ALGORITHM")
        print("="*70)
        
        # Step 1: Initialize with singleton coalitions
        initial_coalitions = [[player] for player in self.players]
        self.initialize_master_problem(initial_coalitions)
        
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}")
            print(f"{'='*70}")
            
            # Step 2: Solve master problem
            payoffs, slack = self.solve_master_problem()
            
            # Step 3: Check if core is empty
            if slack > tolerance:
                print(f"\n{'='*70}")
                print(f"CORE IS EMPTY")
                print(f"Slack variable v = {slack:.6f} > {tolerance}")
                print(f"{'='*70}\n")
                return None
            
            # Step 4: Find violated coalition
            coalition, violation = self.find_violated_coalition(payoffs)
            
            # Step 5: Check convergence
            if len(coalition) == 0 or violation <= tolerance:
                print(f"\n{'='*70}")
                print(f"CORE ALLOCATION FOUND!")
                print(f"Converged after {iteration} iterations")
                print(f"Maximum violation: {violation:.6f}")
                print(f"\nCore Allocation:")
                total_payoff = 0
                for i in self.players:
                    print(f"  Player {i}: {payoffs[i]:.4f}")
                    total_payoff += payoffs[i]
                print(f"  Total: {total_payoff:.4f}")
                print(f"{'='*70}\n")
                return payoffs
            
            # Step 6: Add violated coalition constraint
            print(f"\nAdding violated coalition {coalition} to master problem")
            self._add_coalition_constraint(coalition)
        
        print(f"\n{'='*70}")
        print(f"WARNING: Maximum iterations ({max_iterations}) reached")
        print(f"{'='*70}\n")
        return payoffs
    def measure_stability_violation(self, payoffs: Dict[str, float]) -> float:
            """
            Measure the maximum stability violation for a given payoff allocation
            
            This function checks if the given payoff vector satisfies core stability
            by finding the coalition with the maximum violation.
            
            Args:
                payoffs: Payoff allocation dictionary {player_id: payoff}
                        Example: {'u1': -5.0, 'u2': 0.0, 'u3': -0.2}
                
            Returns:
                float: Maximum violation amount
                    - violation > 0: Payoff allocation is NOT in the core
                                    (some coalition can improve by deviation)
                    - violation ≤ 0: Payoff allocation IS in the core
                                    (no coalition has incentive to deviate)
            
            Example:
                >>> core_comp = CoreComputation(players, time_periods, parameters)
                >>> payoffs = {'u1': -5.15, 'u2': 0.0, 'u3': -0.13}
                >>> violation = core_comp.measure_stability_violation(payoffs)
                >>> if violation <= 0:
                ...     print("Payoff is in the core!")
                ... else:
                ...     print(f"Payoff violates core by {violation:.4f}")
            """
            coalition, violation = self.find_violated_coalition(payoffs)
            return violation
    def find_all_coalitions(self, verbose: bool = True) -> Dict:
            """
            Compute costs for all non-trivial sub-coalitions
            
            For N players, this computes 2^N - 2 coalitions (excluding empty set and grand coalition).
            Results are stored in self.coalition_costs and can be accessed by player combinations.
            
            Args:
                verbose: If True, print progress during computation
                
            Returns:
                Dict: Dictionary mapping coalition (as frozenset) to cost
                    Example: {frozenset({'u1', 'u2'}): -5.234, ...}
            
            Usage:
                >>> core_comp = CoreComputation(players, time_periods, parameters)
                >>> all_costs = core_comp.find_all_coalitions()
                >>> 
                >>> # Access specific coalition
                >>> cost_u1_u2 = all_costs[frozenset(['u1', 'u2'])]
                >>> 
                >>> # Or use the helper method
                >>> cost_u1_u2 = core_comp.get_coalition_cost(['u1', 'u2'])
            """
            from itertools import combinations
            
            print("\n" + "="*70)
            print("COMPUTING ALL COALITION COSTS")
            print("="*70)
            print(f"Players: {self.players}")
            print(f"Total players: {len(self.players)}")
            
            # Calculate total number of coalitions (excluding empty and grand)
            n_players = len(self.players)
            total_coalitions = 2**n_players - 2  # Exclude ∅ and N
            
            print(f"Total sub-coalitions to compute: {total_coalitions}")
            print("="*70 + "\n")
            
            # Generate all non-trivial coalitions
            all_coalitions = []
            for size in range(1, n_players + 1):
                for coalition_tuple in combinations(self.players, size):
                    coalition = list(coalition_tuple)
                    # Skip grand coalition (will be computed separately)
                    if len(coalition) < n_players:
                        all_coalitions.append(coalition)
            
            # Add grand coalition at the end
            all_coalitions.append(self.players)
            
            # Compute cost for each coalition
            computed = 0
            for coalition in all_coalitions:
                computed += 1
                
                if verbose:
                    coalition_str = "{" + ", ".join(sorted(coalition)) + "}"
                    print(f"[{computed}/{total_coalitions+1}] Computing c({coalition_str})...", end=" ")
                
                cost = self.compute_coalition_cost(coalition)
                
                if verbose:
                    print(f"= {cost:.4f}")
            
            print("\n" + "="*70)
            print(f"✓ All {total_coalitions + 1} coalitions computed")
            print(f"✓ Results cached in self.coalition_costs")
            print("="*70 + "\n")
            
            return self.coalition_costs.copy()
if __name__ == "__main__":
    # Example usage
    from data_generator import setup_lem_parameters
    
    print("="*70)
    print("CORE COMPUTATION TEST")
    print("="*70)
    
    # Setup parameters
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))
    
    print("\nGenerating parameters...")
    parameters = setup_lem_parameters(players, time_periods)
    
    # Create core computation instance
    core_comp = CoreComputation(
        players=players,
        time_periods=time_periods,
        parameters=parameters
    )
    
    # Compute core allocation
    core_allocation = core_comp.compute_core(
        max_iterations=50,
        tolerance=1e-6
    )
    
    if core_allocation:
        print("\n" + "="*70)
        print("SUCCESS: Core allocation found")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("RESULT: Core is empty for this game instance")
        print("="*70)

        # Compute stability violation of certain cost allocations
    # 줄리아(Julia)에서 하려면:
    # cost_allocation = -1.0 .* [822.61, 27.05, 44.58, -115.3, -367.99, -54.31]

    # 파이썬에서는:
    cost_allocation = {'u1': -822.61, 'u2': -27.05, 'u3': -44.58, 'u4': 115.3, 'u5': 367.99, 'u6': 54.31} # max violation:0.4025
    cost_allocation = {'u1': -829.24, 'u2': -26.75, 'u3': -44.41, 'u4': 115.71, 'u5': 373.74, 'u6': 54.31} # max violation: -0.0075
    violation = core_comp.measure_stability_violation(cost_allocation)

    coalition_values = core_comp.find_all_coalitions()
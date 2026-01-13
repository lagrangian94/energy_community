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
                 model_type: str,
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
        super().__init__(players, time_periods, parameters, model_type=model_type, dwr=False)
        
        # Add binary selection variables and constraints
        self._add_selection_variables()
        self._add_bigm_constraints()
        self._modify_constraints()
        
    def _add_selection_variables(self):
        """Add binary selection variables z[i] for each player"""
        self.z = {}
        
        for i in self.players:
            # Objective coefficient is NEGATIVE of current payoff
            # We want to maximize Σ payoff[i]*z[i] - cost
            # which is equivalent to minimize -Σ payoff[i]*z[i] + cost
            # Since original model minimizes cost, we just set z coefficient to -payoff
            payoff = self.current_payoffs.get(i, np.inf)
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
                    M = self.params.get(f'i_E_cap_{u}_{t}', M_default)
                    self.model.addCons(self.i_E_gri[u,t] <= M * z_u,
                                      name=f"bigm_i_E_gri_{u}_{t}")
                
                # Community trading - Electricity
                if (u, t) in self.e_E_com:
                    M = M_default
                    self.model.addCons(self.e_E_com[u,t] <= M * z_u,
                                      name=f"bigm_e_E_com_{u}_{t}")
                
                if (u, t) in self.i_E_com:
                    M = self.params.get(f'i_E_cap_{u}_{t}', M_default)
                    self.model.addCons(self.i_E_com[u,t] <= M * z_u,
                                      name=f"bigm_i_E_com_{u}_{t}")
                
                # Grid trading - Heat
                if (u, t) in self.e_H_gri:
                    M = self.params.get(f'e_H_cap_{u}_{t}', M_default)
                    self.model.addCons(self.e_H_gri[u,t] <= M * z_u,
                                      name=f"bigm_e_H_gri_{u}_{t}")
                
                if (u, t) in self.i_H_gri:
                    M = self.params.get(f'i_H_cap_{u}_{t}', M_default)
                    self.model.addCons(self.i_H_gri[u,t] <= M * z_u,
                                      name=f"bigm_i_H_gri_{u}_{t}")
                
                # Community trading - Heat
                if (u, t) in self.e_H_com:
                    M = M_default
                    self.model.addCons(self.e_H_com[u,t] <= M * z_u,
                                      name=f"bigm_e_H_com_{u}_{t}")
                
                if (u, t) in self.i_H_com:
                    M = M_default
                    self.model.addCons(self.i_H_com[u,t] <= M * z_u,
                                      name=f"bigm_i_H_com_{u}_{t}")
                
                # Grid trading - Hydrogen
                if (u, t) in self.e_G_gri:
                    M = self.params.get(f'e_G_cap_{u}_{t}', M_default)
                    self.model.addCons(self.e_G_gri[u,t] <= M * z_u,
                                      name=f"bigm_e_G_gri_{u}_{t}")
                
                if (u, t) in self.i_G_gri:
                    M = self.params.get(f'i_G_cap_{u}_{t}', M_default)
                    self.model.addCons(self.i_G_gri[u,t] <= M * z_u,
                                      name=f"bigm_i_G_gri_{u}_{t}")
                
                # Community trading - Hydrogen
                if (u, t) in self.e_G_com:
                    M = M_default
                    self.model.addCons(self.e_G_com[u,t] <= M * z_u,
                                      name=f"bigm_e_G_com_{u}_{t}")
                
                if (u, t) in self.i_G_com:
                    M = M_default
                    self.model.addCons(self.i_G_com[u,t] <= M * z_u,
                                      name=f"bigm_i_G_com_{u}_{t}")
                
                # Production variables
                if (u, 'res', t) in self.p:
                    M = self.params.get(f'renewable_cap_{u}_{t}', M_default)
                    self.model.addCons(self.p[u,'res',t] <= M * z_u,
                                      name=f"bigm_p_res_{u}_{t}")
                
                if (u, 'hp', t) in self.p:
                    M = self.params.get(f'hp_cap_{u}', M_default)
                    self.model.addCons(self.p[u,'hp',t] <= M * z_u,
                                      name=f"bigm_p_hp_{u}_{t}")
                
                if (u, 'els', t) in self.p:
                    M = self.params.get(f'els_cap_{u}', M_default) * 25  # Upper bound for hydrogen production
                    self.model.addCons(self.p[u,'els',t] <= M * z_u,
                                      name=f"bigm_p_els_{u}_{t}")
                
                # Electrolyzer demand
                if (u, t) in self.els_d:
                    M = self.params.get(f'els_cap_{u}', M_default)
                    self.model.addCons(self.els_d[u,t] <= M * z_u,
                                      name=f"bigm_els_d_{u}_{t}")
                
                # Flexible demand
                if (u, 'elec', t) in self.fl_d:
                    M = M_default  # From compact.py line 533
                    self.model.addCons(self.fl_d[u,'elec',t] <= M * z_u,
                                      name=f"bigm_fl_d_elec_{u}_{t}")
                
                if (u, 'hydro', t) in self.fl_d:
                    M = M_default
                    self.model.addCons(self.fl_d[u,'hydro',t] <= M * z_u,
                                      name=f"bigm_fl_d_hydro_{u}_{t}")
                
                if (u, 'heat', t) in self.fl_d:
                    M = M_default
                    self.model.addCons(self.fl_d[u,'heat',t] <= M * z_u,
                                      name=f"bigm_fl_d_heat_{u}_{t}")
                
                # Storage variables - Electricity
                if (u, t) in self.s_E:
                    M = self.params.get('storage_capacity', M_default)
                    self.model.addCons(self.s_E[u,t] <= M * z_u,
                                      name=f"bigm_s_E_{u}_{t}")
                    
                if (u, t) in self.b_ch_E:
                    M = self.params.get('storage_power', M_default)
                    self.model.addCons(self.b_ch_E[u,t] <= M * z_u,
                                      name=f"bigm_b_ch_E_{u}_{t}")
                    
                if (u, t) in self.b_dis_E:
                    M = self.params.get('storage_power', M_default)
                    self.model.addCons(self.b_dis_E[u,t] <= M * z_u,
                                      name=f"bigm_b_dis_E_{u}_{t}")
                
                # Storage variables - Hydrogen
                if (u, t) in self.s_G:
                    M = M_default  # From compact.py line 572
                    self.model.addCons(self.s_G[u,t] <= M * z_u,
                                      name=f"bigm_s_G_{u}_{t}")
                
                if (u, t) in self.b_ch_G:
                    M = M_default  # From compact.py line 571
                    self.model.addCons(self.b_ch_G[u,t] <= M * z_u,
                                      name=f"bigm_b_ch_G_{u}_{t}")
                
                if (u, t) in self.b_dis_G:
                    M = M_default
                    self.model.addCons(self.b_dis_G[u,t] <= M * z_u,
                                      name=f"bigm_b_dis_G_{u}_{t}")
                
                # Storage variables - Heat
                if (u, t) in self.s_H: #TODO
                    M = self.params.get('storage_capacity_heat', M_default)
                    self.model.addCons(self.s_H[u,t] <= M * z_u,
                                      name=f"bigm_s_H_{u}_{t}")
                
                if (u, t) in self.b_ch_H:
                    M = self.params.get('storage_power_heat', M_default)
                    self.model.addCons(self.b_ch_H[u,t] <= M * z_u,
                                      name=f"bigm_b_ch_H_{u}_{t}")
                
                if (u, t) in self.b_dis_H:
                    M = self.params.get('storage_power_heat', M_default)
                    self.model.addCons(self.b_dis_H[u,t] <= M * z_u,
                                      name=f"bigm_b_dis_H_{u}_{t}")
                
                # Electrolyzer binary commitment variables
                if (u, t) in self.z_su_G:
                    self.model.addCons(self.z_su_G[u,t] <= z_u,
                                      name=f"bigm_z_su_G_{u}_{t}")
                if (u, t) in self.z_on_G:
                    self.model.addCons(self.z_on_G[u,t] <= z_u,
                                      name=f"bigm_z_on_G_{u}_{t}")
                """
                z_on + z_off + z_sb == 1 이기 때문에 z_off는 bigm으로 가두면 안됨.
                """
                if (u, t) in self.z_sb_G:
                    self.model.addCons(self.z_sb_G[u,t] <= z_u,
                                      name=f"bigm_z_sb_G_{u}_{t}")

                # Heat pump binary commitment variables
                if (u, t) in self.z_su_H:
                    self.model.addCons(self.z_su_H[u,t] <= z_u,
                                      name=f"bigm_z_su_H_{u}_{t}")
                if (u, t) in self.z_on_H:
                    self.model.addCons(self.z_on_H[u,t] <= z_u,
                                      name=f"bigm_z_on_H_{u}_{t}")
                if (u, t) in self.z_ru_H:
                    self.model.addCons(self.z_ru_H[u,t] <= z_u,
                                      name=f"bigm_z_ru_H_{u}_{t}")
        print(f"Added Big-M constraints for all variables")
    
    def _modify_constraints(self):
        """
        Modify individual balance constraints to incorporate z[i]
        
        --- for Non-flexible demand ---
        Original: LHS == nfl_d + fl_d, nfl_d == nfl_d_param
        Modified: LHS == nfl_d + fl_d, nfl_d <= nfl_d_param * z[i], nfl_d >= nfl_d_param * z[i]
        
        --- for Storage ---
        Original: s_E, s_G, s_H at time 6 == initial SOC of E, G, H
        Modified: s_E, s_G, s_H at time 6 : s == initial SOC * z[i]
        """
        # Add modified electricity balance constraints
        for u in self.players:
            for t in self.time_periods:
                if (u,'elec',t) in self.nfl_d:
                    nfl_demand_param = self.params.get(f'd_E_nfl_{u}_{t}', np.inf)
                    cons = self.elec_nfl_demand_cons.get(f"elec_nfl_demand_cons_{u}_{t}", None)
                    self.model.addConsCoeff(cons, self.z[u], -1*nfl_demand_param)
                    self.model.chgRhs(cons, 0.0)
                    self.model.chgLhs(cons, 0.0)
        
                if (u,'hydro',t) in self.nfl_d:
                    nfl_demand_param = self.params.get(f'd_G_nfl_{u}_{t}', np.inf)
                    cons = self.hydro_nfl_demand_cons.get(f"hydro_nfl_demand_cons_{u}_{t}", None)
                    self.model.addConsCoeff(cons, self.z[u], -1*nfl_demand_param)
                    self.model.chgRhs(cons, 0.0)
                    self.model.chgLhs(cons, 0.0)
                if (u,'heat',t) in self.nfl_d:
                    nfl_demand_param = self.params.get(f'd_H_nfl_{u}_{t}', np.inf)
                    cons = self.heat_nfl_demand_cons.get(f"heat_nfl_demand_cons_{u}_{t}", None)
                    self.model.addConsCoeff(cons, self.z[u], -1*nfl_demand_param)
                    self.model.chgRhs(cons, 0.0)
                    self.model.chgLhs(cons, 0.0)

            if u in self.players_with_elec_storage:
                initial_soc = self.params.get(f'initial_soc_E', np.inf)
                cons_fix_s_E = self.storage_cons[f"initial_soc_E_{u}"]
                self.model.addConsCoeff(cons_fix_s_E, self.z[u], -initial_soc)
                self.model.chgRhs(cons_fix_s_E, 0.0)
                self.model.chgLhs(cons_fix_s_E, 0.0) ## equality constraint니까 LHS도 바꿔줘야 함.
            if u in self.players_with_hydro_storage:
                initial_soc = self.params.get(f'initial_soc_G', np.inf)
                cons_fix_s_G = self.storage_cons[f"initial_soc_G_{u}"]
                self.model.addConsCoeff(cons_fix_s_G, self.z[u], -initial_soc)
                self.model.chgRhs(cons_fix_s_G, 0.0)
                self.model.chgLhs(cons_fix_s_G, 0.0)
            if u in self.players_with_heat_storage:
                initial_soc = self.params.get(f'initial_soc_H', np.inf)
                cons_fix_s_H = self.storage_cons[f"initial_soc_H_{u}"]
                self.model.addConsCoeff(cons_fix_s_H, self.z[u], -initial_soc)
                self.model.chgRhs(cons_fix_s_H, 0.0)
                self.model.chgLhs(cons_fix_s_H, 0.0)
        
        
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
                 model_type: str,
                 time_periods: List[int],
                 parameters: Dict):
        """
        Initialize core computation
        
        Args:
            players: List of all player IDs
            model_type: Type of model to use ('mip' or 'lp')
            time_periods: List of time period indices
            parameters: Dictionary containing all model parameters
        """
        self.players = players
        if model_type not in ('mip', 'lp'):
            raise ValueError("model_type must be either 'mip' or 'lp', got: {}".format(model_type))
        self.model_type = model_type
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

        # Calculate individual costs (coalition costs에 저장하면 됨. 왜냐하면 플레이어 개개인도 각각이 sub-coalition이라서)
        for player in self.players:
            self.coalition_costs[tuple([player])] = self.compute_coalition_cost([player])
        
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
            model_type=self.model_type,
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
        
        try:
            cons = self.master_model.addCons(
                quicksum(self.payoff_vars[i] for i in coalition) <= coalition_cost + self.slack_var,
                name=f"stability_{coalition_str}"
            )
        except Exception as e:
            print(f"Error adding coalition constraint: {e}")
            print(f"Coalition: {coalition}")
            print(f"Coalition cost: {coalition_cost:.4f}")
            print(f"Slack variable: {self.slack_var:.4f}")
            print(f"Payoff variables: {self.payoff_vars}")
            raise e
        
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
            model_type=self.model_type,
            parameters=self.params,
            current_payoffs=payoffs
        )
        model = sep_problem.model
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
            print(f"  Found coalition: {coalition}")
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
    def measure_stability_violation(self, payoffs: Dict[str, float], brute_force: bool = False) -> float:
            """
            Measure the maximum stability violation for a given payoff allocation
            
            This function checks if the given payoff vector satisfies core stability
            by finding the coalition with the maximum violation.
            
            Args:
                payoffs: Payoff allocation dictionary {player_id: payoff}
                        Example: {'u1': -5.0, 'u2': 0.0, 'u3': -0.2}
                brute_force: If True, compute all coalition costs and solve LP with all constraints.
                        If False, use separation problem (faster for large N).
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
            ## First, check whether the cost allocation is the imputation (at least no worse than the individually played cost)
            is_imputation = self.check_imputation(payoffs)
            if not is_imputation:
                violation = np.inf
                print("Cost allocation is not an imputation")
                return violation
            if not brute_force:
                coalition, violation = self.find_violated_coalition(payoffs)
            else:
                violation = self._measure_violation_brute_force(payoffs)
            return violation
    def check_imputation(self, payoffs: Dict[str, float]) -> bool:
        """
        Check whether the cost allocation is the imputation (at least no worse than the individually played cost)
        """
        for player in self.players:
            if payoffs[player] - self.coalition_costs[tuple([player])] >= 1e-6:
                return False
        return True
    def _measure_violation_brute_force(self, payoffs: Dict[str, float]) -> float:
        """
        Measure violation by solving LP with all coalition constraints
        
        This method:
        1. Computes all coalition costs (if not already computed)
        2. Solves: min v
                  s.t. Σ_{i∈S} payoffs[i] ≤ c(S) + v  for all S
                       v ≥ 0
        3. Returns optimal v (maximum violation)
        
        Args:
            payoffs: Payoff allocation dictionary
            
        Returns:
            float: Maximum violation (optimal slack variable v)
        """
        print("\n" + "="*70)
        print("BRUTE FORCE VIOLATION MEASUREMENT")
        print("="*70)
        print(f"Payoffs: {payoffs}")
        
        # Ensure all coalition costs are computed
        if len(self.coalition_costs) < (2**len(self.players) - 1):
            print("\nComputing all coalition costs...")
            self.find_all_coalitions(verbose=False)
        else:
            print(f"\n✓ Using cached coalition costs ({len(self.coalition_costs)} coalitions)")
        
        # Create LP model
        print("\nCreating LP model with all stability constraints...")
        lp_model = Model("ViolationLP")
        
        # Create slack variable v
        v = lp_model.addVar(vtype="C", name="v", lb=0, obj=1.0)
        
        # Add constraint for each coalition
        num_constraints = 0
        for coalition_tuple, cost in self.coalition_costs.items():
            coalition = list(coalition_tuple)
                        
            # Constraint: Σ_{i∈S} payoffs[i] ≤ c(S) + v
            lhs = sum(payoffs[i] for i in coalition)
            lp_model.addCons(lhs <= cost + v, 
                           name=f"stability_{'_'.join(sorted(coalition))}")
            num_constraints += 1
        
        print(f"✓ Added {num_constraints} stability constraints")
        
        # Solve LP
        print("\nSolving LP...")
        lp_model.optimize()
        
        status = lp_model.getStatus()
        if status != "optimal":
            print(f"WARNING: LP failed with status {status}")
            return float('inf')
        
        violation = lp_model.getVal(v)
        
        print(f"\n✓ Optimal solution found")
        print(f"  Maximum violation (slack v): {violation:.6f}")
        
        if violation <= 1e-6:
            print(f"  → Payoff IS in the core (stable)")
        else:
            print(f"  → Payoff is NOT in the core (violation = {violation:.6f})")
            binding_cons = [cons for cons in lp_model.getConss(False) if lp_model.getSlack(cons) <= 1e-6]
            if len(binding_cons) >1:
                raise RuntimeError("Multiple binding constraints found!")
            coalition_str = binding_cons[0].name.replace("stability_", "")
            coalition = tuple(sorted(coalition_str.split("_")))
            coalition_cost = self.coalition_costs.get(coalition, None)
            payoff_sum = sum(payoffs[i] for i in coalition)

            print(f"  Binding coalition: {coalition}")
        print("="*70 + "\n")
        
        return coalition,violation
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
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
import os
import tempfile
sys.path.append('/mnt/project')
# from compact import LocalEnergyMarket
from compact_utility import LocalEnergyMarket


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
                 current_payoffs: Dict[str, float],
                 mipsolver: Optional[str] = None):
        """
        Initialize separation problem

        Args:
            players: List of all player IDs
            time_periods: List of time period indices
            parameters: Dictionary containing all model parameters
            current_payoffs: Current payoff allocation {player_id: payoff}
            mipsolver: MIP solver to use. None for SCIP (default), 'highs' for HiGHS
        """
        self.current_payoffs = current_payoffs
        self.mipsolver = mipsolver

        # Initialize parent class
        super().__init__(players, time_periods, parameters, model_type=model_type, dwr=False, mipsolver=mipsolver)
        
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
                
                # # Electrolyzer demand
                ## z_u=0 -> els_d=0. therefore, useless
                # if (u, t) in self.els_d:
                #     M = self.params.get(f'els_cap_{u}', M_default)
                #     self.model.addCons(self.els_d[u,t] <= M * z_u,
                #                       name=f"bigm_els_d_{u}_{t}")
                
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
                # # Non-flexible dmeand: 아래 modify에서 처리함.
                if (u, 'elec', t) in self.nfl_d:
                    M = M_default
                    self.model.addCons(self.nfl_d[u,'elec',t] <= M * z_u,
                                      name=f"bigm_nfl_d_elec_{u}_{t}")
                if (u, 'hydro', t) in self.nfl_d:
                    M = M_default
                    self.model.addCons(self.nfl_d[u,'hydro',t] <= M * z_u,
                                      name=f"bigm_nfl_d_hydro_{u}_{t}")
                if (u, 'heat', t) in self.nfl_d:
                    M = M_default
                    self.model.addCons(self.nfl_d[u,'heat',t] <= M * z_u,
                                      name=f"bigm_nfl_d_heat_{u}_{t}")
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

                # Reserve headroom (private): an unselected player must offer zero
                # reserve so it cannot contribute to the community coupling rows
                # (C-Res up/down). Gating the aggregates r_plus/r_minus suffices —
                # each equals the sum of its non-negative per-asset splits, so
                # r_plus[u,t] <= M*z_u forces every split to 0 when z_u=0. The
                # shared community singletons (r_sym, p) need no gating.
                if self.enable_reserve:
                    if (u, t) in self.r_plus:
                        self.model.addCons(self.r_plus[u,t] <= M_default * z_u,
                                          name=f"bigm_r_plus_{u}_{t}")
                    if (u, t) in self.r_minus:
                        self.model.addCons(self.r_minus[u,t] <= M_default * z_u,
                                          name=f"bigm_r_minus_{u}_{t}")

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

        나머지 제약식은 위의 bigm_constraints에서 변수가 0이 되므로 modify할 필요 없을 것으로 판단됨.
        Heap Pump commitment도 필요 없음.
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
                initial_soc = self.params.get(f'initial_soc_E_{u}', np.inf)
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

        # Electrolyzer state: z_on_G + z_off_G + z_sb_G == 1 -> == z[u] so when player not selected all are 0
        ##는 넣으면안됨! 왜냐면 특정 t에 대해서 z_off_G >=1 이어야 하기때문에..

        print(f"Modified balance constraints to incorporate z variables")
    
    def solve_separation(self, time_limit: Optional[float] = None):
        """
        Solve the separation problem

        Args:
            time_limit: wall-clock cap in seconds for this single separation solve.
                Honoured on the Gurobi and SCIP paths (HiGHS still ignores it). On expiry the
                incumbent is used instead of the optimum and `self.truncated` is set:
                an incumbent coalition still yields a VALID cut -- it is violated, just
                not necessarily the most violated one -- but it is NOT a certificate,
                so a caller must never read "no coalition found" off a truncated solve
                as convergence.

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

        self.truncated = False
        if self.mipsolver and self.mipsolver.lower() == 'highs':
            obj_val, selected_coalition = self._solve_with_highs()
        elif self.mipsolver and self.mipsolver.lower() == 'gurobi':
            obj_val, selected_coalition = self._solve_with_gurobi(time_limit)
        else:
            # Solve with SCIP (default). The budget has to be honoured here as well, not
            # only on the Gurobi path: this branch is what `check_allocations` reached by
            # default, so a time limit that existed only for Gurobi was silently no
            # limit at all -- one 60-prosumer separation ran six hours at a 118% gap
            # inside a measurement nominally capped at 3600 s.
            if time_limit is not None:
                self.model.setRealParam('limits/time', max(1.0, float(time_limit)))
            status = self.solve()

            if status == "optimal":
                pass
            elif self.model.getNSols() > 0:
                # Cut off with an incumbent: a violated coalition, just not provably the
                # most violated one. Same contract as the Gurobi path.
                self.truncated = True
                print(f"  SCIP separation cut off at the time limit (status {status}, "
                      f"{self.model.getNSols()} solution(s)); using the incumbent")
            else:
                self.truncated = True
                print(f"  SCIP separation cut off with no solution (status {status})")
                if status in ('timelimit', 'userinterrupt'):
                    return 0.0, []
                raise RuntimeError(f"Separation problem failed with status: {status}")

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

    def _solve_with_highs(self):
        """
        Export SCIP model to .mps, then solve with HiGHS via highspy.

        Returns:
            tuple: (obj_val, selected_coalition)
        """
        import highspy

        # Export SCIP model to temporary .mps file
        mps_path = os.path.join(tempfile.gettempdir(), "separation_problem.mps")
        self.model.writeProblem(mps_path)
        print(f"  Exported SCIP model to {mps_path}")

        # Solve with HiGHS
        h = highspy.Highs()
        h.setOptionValue("output_flag", True)
        h.readModel(mps_path)
        h.run()

        status = h.getInfoValue("primal_solution_status")[1]
        # primal_solution_status: 2 = feasible
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(
                f"HiGHS separation problem failed with status: {h.getModelStatus()}"
            )

        obj_val = h.getInfoValue("objective_function_value")[1]

        # Map HiGHS column indices back to z variable names to extract coalition
        # Get column names from the model
        num_cols = h.getNumCol()
        sol = h.getSolution()
        col_values = sol.col_value

        selected_coalition = []
        for j in range(num_cols):
            col_name = h.getColName(j)[1]
            if col_name.startswith("z_"):
                player_id = col_name[2:]  # strip "z_" prefix
                if player_id in [p for p in self.players]:
                    if col_values[j] > 0.5:
                        selected_coalition.append(player_id)

        print(f"  HiGHS objective: {obj_val:.4f}")

        # Clean up temp file
        try:
            os.remove(mps_path)
        except OSError:
            pass

        return obj_val, selected_coalition

    def _solve_with_gurobi(self, time_limit: Optional[float] = None):
        """
        Export SCIP model to .mps, then solve with Gurobi via gurobipy.

        The separation MIP is the row-generation bottleneck; Gurobi is typically
        much faster than SCIP/HiGHS on it. Only the selected coalition (z_ vars = 1)
        and the objective are needed, so we read the same MPS SCIP writes.

        Returns:
            tuple: (obj_val, selected_coalition)
        """
        import gurobipy as gp

        mps_path = os.path.join(tempfile.gettempdir(), "separation_problem.mps")
        self.model.writeProblem(mps_path)

        gm = gp.read(mps_path)
        gm.setParam("OutputFlag", 0)
        gm.setParam("MIPGap", 1e-4)   # default relative gap; the caller's
                                      # violation verification uses a matching
                                      # relative tolerance (see find_violated_coalition)
        if time_limit is not None:
            # Without this a single separation solve is unbounded and the row-generation
            # budget is only enforced between iterations. One second is the floor: asking
            # Gurobi for less returns no incumbent at all, which is strictly worse than a
            # cut of unknown quality.
            gm.setParam("TimeLimit", max(1.0, float(time_limit)))
        gm.optimize()

        if gm.Status == gp.GRB.OPTIMAL:
            pass
        elif gm.SolCount > 0:
            # Cut off with an incumbent. The incumbent selects a coalition whose
            # violation is real (the objective is a lower bound on it), so the cut is
            # valid and the master stays a relaxation. What is lost is the guarantee
            # that this is the MOST violated coalition, hence the certificate: see
            # `truncated` in solve_separation.
            self.truncated = True
            print(f"  Gurobi separation cut off at the time limit "
                  f"(status {gm.Status}, {gm.SolCount} incumbent(s), "
                  f"gap {gm.MIPGap:.2%}); using the incumbent coalition")
        else:
            # Out of time with nothing to show. Report no coalition and stay truncated;
            # the caller sees "nothing found" flagged as uncertified and stops rather
            # than mistaking it for convergence.
            self.truncated = True
            print(f"  Gurobi separation cut off with no incumbent (status {gm.Status})")
            if gm.Status in (gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED):
                return 0.0, []
            raise RuntimeError(
                f"Gurobi separation problem failed with status: {gm.Status}"
            )

        obj_val = gm.ObjVal
        players_set = set(self.players)
        selected_coalition = []
        for var in gm.getVars():
            if var.VarName.startswith("z_"):
                player_id = var.VarName[2:]  # strip "z_" prefix
                if player_id in players_set and var.X > 0.5:
                    selected_coalition.append(player_id)

        print(f"  Gurobi objective: {obj_val:.4f}")

        try:
            os.remove(mps_path)
        except OSError:
            pass

        return obj_val, selected_coalition


def fairness_mode(egalitarian) -> Optional[str]:
    """Normalise the `egalitarian` argument to a fairness mode, or None.

    'range'    Kimms MP_I: min (max_i p_i - min_i p_i). An LP, so the objective value
               is determined but the ARGMIN generally is not -- see egalitarian_width.
    'variance' Fioriti et al. (2025) Fair Core with f = -variance, i.e. the Variance
               Core of eq.(27): min sum_i (y_i - v(N)/n)^2 over the SURPLUS shares
               y_i = p_i - kappa_i of the zero-normalized game (Definition `def:game`).
               Efficiency pins their mean at v(N)/n, so this objective IS n times the
               variance of the shares. Strictly
               convex, hence a unique minimiser -- which is the whole point: the same
               allocation comes back whatever coalitions row generation happened to
               generate, and it is what makes the per-player numbers reportable.

    True is accepted as an alias for 'range'.
    """
    if not egalitarian:
        return None
    if egalitarian is True:
        return 'range'
    if egalitarian in ('range', 'variance'):
        return egalitarian
    raise ValueError(f"egalitarian must be False, 'range' or 'variance', got {egalitarian!r}")


class CoreComputation:
    """
    Main class for computing core allocations using row generation algorithm
    """
    
    def __init__(self,
                 players: List[str],
                 model_type: str,
                 time_periods: List[int],
                 parameters: Dict,
                 mipsolver: Optional[str] = None):
        """
        Initialize core computation

        Args:
            players: List of all player IDs
            model_type: Type of model to use ('mip' or 'lp')
            time_periods: List of time period indices
            parameters: Dictionary containing all model parameters
            mipsolver: MIP solver to use for separation problem.
                       None for SCIP (default), 'highs' for HiGHS
        """
        self.players = players
        if model_type not in ('mip', 'lp'):
            raise ValueError("model_type must be either 'mip' or 'lp', got: {}".format(model_type))
        self.model_type = model_type
        self.time_periods = time_periods
        self.params = parameters
        self.mipsolver = mipsolver
        
        # Cache for coalition costs
        self.coalition_costs = {}
        
        # Master problem model
        self.master_model = None
        self.payoff_vars = {}
        self.slack_var = None
        # Set by find_violated_coalition: True when the last separation solve was cut
        # off at its time limit, i.e. "no violated coalition" is not a certificate.
        self.last_separation_truncated = False
        
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
        lem.model.hideOutput()
        status = lem.solve()
        
        if status != "optimal":
            print(f"  WARNING: Coalition {coalition} optimization failed with status {status}")
            # Return a very high cost (bad for the coalition)
            return float('inf')
        
        cost = lem.model.getObjVal()
        self.coalition_costs[coalition_tuple] = cost
        
        print(f"  Coalition {coalition} cost: {cost:.4f}")
        
        return cost
    
    def initialize_master_problem(self, initial_coalitions: List[List[str]],
                                  cost_of_stability: bool = False,
                                  egalitarian: bool = False) -> None:
        """
        Initialize the master problem with initial set of coalitions

        Args:
            initial_coalitions: Initial set of coalitions (typically singletons)
            cost_of_stability: if True, the epsilon slack v is placed on the GRAND
                coalition only (external subsidy, efficiency row Σp = c(N) - v) and
                every proper-coalition constraint is left strict. Then v* = cost of
                stability. Default False = uniform epsilon on every coalition
                (strong ε-core / least-core, original behaviour).
            egalitarian: if True, build MP_I of Drechsel & Kimms (2010) sec.2.3 instead:
                the core is assumed nonempty and, among its elements, the one whose cost
                shares are least spread is selected,

                    min  P_hi - P_lo   s.t.  Σ_N p = c(N),  Σ_S p ≤ c(S) ∀S∈𝒮,
                                             P_hi ≥ y_i,  P_lo ≤ y_i  ∀i,

                over the surplus shares y_i = p_i - kappa_i (see the note at the
                objective). Minimisation drives P_hi down onto max_i y_i and P_lo up onto
                min_i y_i, so the objective is the range whatever the sign of the shares
                -- ours are costs, negative for a member that profits. There is no epsilon anywhere:
                the efficiency row is the exact c(N) and every coalition row is strict, so
                an infeasible master is not a failure but a proof that the core is empty
                (the rows present are a subset of the core's, hence a relaxation of it).
                Overrides `cost_of_stability`, whose subsidy has no meaning here.
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
        
        # Create slack variable v >= 0. In egalitarian mode it is pinned to zero and
        # carries no objective: the variable stays only so that solve_master_problem and
        # every caller that reads `slack` keep working unchanged.
        self.slack_var = self.master_model.addVar(
            vtype="C",
            name="v",
            lb=0,
            ub=0.0 if egalitarian else None,
            obj=0.0 if egalitarian else 1.0  # Minimize v
        )

        # Egalitarian objective: the two range variables. Free, because the shares they
        # bracket are costs and may be negative.
        self.range_vars = None
        mode = fairness_mode(egalitarian)
        if mode == 'range':
            p_hi = self.master_model.addVar(vtype="C", name="P_hi", lb=-float('inf'), obj=1.0)
            p_lo = self.master_model.addVar(vtype="C", name="P_lo", lb=-float('inf'), obj=-1.0)
            self.range_vars = (p_hi, p_lo)
        
        # Constraint: Efficiency (sum of payoffs = grand coalition cost)
        grand_coalition_cost = self.compute_coalition_cost(self.players)
        print(f"\nGrand coalition cost c(N): {grand_coalition_cost:.4f}")
        
        if mode:
            efficiency_cons = self.master_model.addCons(
                quicksum(self.payoff_vars[i] for i in self.players) == grand_coalition_cost,
                name="efficiency"
            )
        # BOTH fairness modes measure the dispersion of SURPLUS shares y_i = p_i -
        # kappa_i, not of the raw bills p_i. Definition `def:game` zero-normalizes the
        # game -- v(S) = c(S) - sum_{j in S} kappa_j in this cost convention -- so the
        # egalitarian reference is an equal split of v(N) on top of each member's
        # stand-alone value. On the raw shares it would instead call two members equally
        # treated when one brought a wind farm and the other a heat pump. Only the
        # OBJECTIVE moves: the feasible set is the same polytope in p, re-coordinated.
        kappa = {i: self.coalition_costs[(i,)] for i in self.players}
        y = {i: self.payoff_vars[i] - kappa[i] for i in self.players}
        if mode == 'range':
            p_hi, p_lo = self.range_vars
            for i in self.players:
                self.master_model.addCons(p_hi >= y[i], name=f"hi_{i}")
                self.master_model.addCons(p_lo <= y[i], name=f"lo_{i}")
        elif mode == 'variance':
            # SCIP takes no quadratic objective directly, so epigraph it: minimise q
            # subject to sum_i (y_i - a)^2 <= q. The constraint is convex, so the
            # relaxation is tight at the optimum and q equals the true objective.
            a = (grand_coalition_cost - sum(kappa.values())) / len(self.players)
            q = self.master_model.addVar(vtype="C", name="q", lb=0.0, obj=1.0)
            self.master_model.addCons(
                quicksum((y[i] - a) * (y[i] - a) for i in self.players) <= q,
                name="variance_epigraph"
            )
            print(f"Variance Core: minimising sum_i (p_i - kappa_i - {a:.4f})^2 "
                  f"over the core   [v(N)/n = {a:.4f}]")
        elif cost_of_stability:
            # Cost-of-stability: the epsilon slack v is an external subsidy sitting
            # on the grand coalition ONLY. In cost form the subsidy lowers the total
            # cost that must be allocated to members:  Σ_i p_i == c(N) - v.
            efficiency_cons = self.master_model.addCons(
                quicksum(self.payoff_vars[i] for i in self.players) == grand_coalition_cost - self.slack_var,
                name="efficiency"
            )
        else:
            efficiency_cons = self.master_model.addCons(
                quicksum(self.payoff_vars[i] for i in self.players) == grand_coalition_cost,
                name="efficiency"
            )

        # Add initial coalition constraints
        print(f"\nAdding {len(initial_coalitions)} initial coalition constraints:")
        for coalition in initial_coalitions:
            self._add_coalition_constraint(
                coalition, cost_of_stability=(cost_of_stability or egalitarian))
        
        print("="*70 + "\n")
    
    def _add_coalition_constraint(self, coalition: List[str],
                                  cost_of_stability: bool = False) -> None:
        """
        Add a coalition stability constraint to the master problem

        Constraint: Σ_{i∈S} p[i] <= c(S) + v   (uniform-epsilon mode)
                    Σ_{i∈S} p[i] <= c(S)        (cost-of-stability mode, strict)

        Args:
            coalition: List of player IDs in the coalition
            cost_of_stability: if True, no epsilon on the coalition (the subsidy
                lives only on the grand-coalition efficiency row).
        """
        coalition_cost = self.compute_coalition_cost(coalition)

        coalition_str = "_".join(sorted(coalition))

        # Need to free transformed problem before adding constraints
        self.master_model.freeTransform()

        try:
            if cost_of_stability:
                cons = self.master_model.addCons(
                    quicksum(self.payoff_vars[i] for i in coalition) <= coalition_cost,
                    name=f"stability_{coalition_str}"
                )
            else:
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
        self.master_model.hideOutput()
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
    
    def find_violated_coalition(self, payoffs: Dict[str, float],
                                time_limit: Optional[float] = None) -> Tuple[List[str], float]:
        """
        Solve separation problem to find most violated coalition

        Args:
            payoffs: Current payoff allocation
            time_limit: cap on this single separation solve (Gurobi path only). If it
                expires, `self.last_separation_truncated` is set and the returned
                coalition, if any, is violated but not necessarily the most violated.

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
            current_payoffs=payoffs,
            mipsolver=self.mipsolver
        )
        model = sep_problem.model
        ## debug: z_3=z_6=1, 나머지 0으로 고정
        # model.chgVarLb(sep_problem.z['u3'], 1.0)
        # model.chgVarLb(sep_problem.z['u6'], 1.0)
        # model.chgVarUb(sep_problem.z['u4'], 0.0)
        # model.chgVarUb(sep_problem.z['u5'], 0.0)

        # model.chgVarUb(sep_problem.z['u1'], 0.0)
        # model.chgVarUb(sep_problem.z['u2'], 0.0)
        ## 이렇게했더니 infeasible 뜸
        # model.hideOutput()
        coalition, violation = sep_problem.solve_separation(time_limit=time_limit)
        self.last_separation_truncated = getattr(sep_problem, 'truncated', False)
        # Compute actual violation for verification.
        #
        # `coalition` must be non-empty to be worth verifying: the violation of the empty
        # set is identically zero (no payoffs summed, c(empty) = 0), so an empty selection
        # IS the statement that nothing is violated. Its reported objective is pure solver
        # noise -- 9.3e-4 on 15p day 27 -- and comparing that against a freshly computed
        # exact 0 tripped the mismatch guard and killed the run one iteration before it
        # would have declared convergence.
        if coalition and violation > 1e-7:
            coalition_cost = self.compute_coalition_cost(coalition)
            payoff_sum = sum(payoffs[i] for i in coalition)
            actual_violation = payoff_sum - coalition_cost
            
            print(f"\nVerification:")
            print(f"  Coalition payoff sum: {payoff_sum:.4f}")
            print(f"  Coalition cost c(S): {coalition_cost:.4f}")
            print(f"  Actual violation (Σ payoffs - cost): {actual_violation:.4f}")
            print(f"  Separation problem violation: {violation:.4f}")
            print(f"  Found coalition: {coalition}")
            # The separation MIP is solved to a RELATIVE gap (MIPGap=1e-4, Gurobi or
            # SCIP), so the separation-vs-fresh-c(S) mismatch scales with the
            # objective magnitude. Use a relative tolerance (≈ MIPGap·|c(S)|) with an
            # absolute floor, rather than a fixed 1e-4 that only held for near-exact
            # solves and spuriously tripped with the Gurobi separation path.
            verify_tol = max(1e-4, 2e-4 * abs(coalition_cost))
            if self.last_separation_truncated:
                # A cut-off incumbent may carry a suboptimal dispatch for the coalition
                # it selects, so its objective only bounds the true violation from
                # below; equality is the wrong test. `compute_coalition_cost` re-solves
                # c(S) exactly, so the recomputed value is the one to carry forward.
                if actual_violation < violation - verify_tol:
                    raise RuntimeError(
                        f"Truncated separation reports violation {violation:.6f} above "
                        f"the exact {actual_violation:.6f} for the same coalition!")
                violation = actual_violation
            elif abs(actual_violation - violation) > verify_tol:
                raise RuntimeError(
                    f"Mismatch between actual ({actual_violation:.6f}) and separation "
                    f"({violation:.6f}) violation exceeds tol {verify_tol:.6f}!")
        else:
            coalition = []
        print("="*60)
        
        return coalition, violation
    
    def compute_core(self,
                     max_iterations: int = 100,
                     tolerance: float = 1e-6,
                     time_limit: float = 36000,
                     cost_of_stability: bool = True,
                     egalitarian: bool = False) -> Optional[Dict[str, float]]:
        """
        Main row generation algorithm to compute core allocation

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            time_limit: Time limit in seconds (default: 3600 = 1 hour)
            cost_of_stability: DEFAULT True. Runs the cost-of-stability formulation
                (epsilon subsidy on the grand coalition only; proper coalitions
                strict). On convergence v* = cost of stability is stored in
                self.cost_of_stability_value and self.weak_eps = v*/n, and the
                method RETURNS the budget-balanced weak-(v*/n)-core allocation
                q = p + v*/n (raw LP point p kept on self.cos_raw_payoffs); the
                returned success flag means "core is non-empty" (v* ≈ 0). When
                v*≈0 this q is exactly the core point, so it is a drop-in for the
                previous behaviour. Set False for the old uniform-epsilon
                strong-ε-core / least-core (early-exits on empty core).
            egalitarian: DEFAULT False. 'range' (or True) selects MP_I of Drechsel &
                Kimms (2010) sec.2.3, 'variance' the Variance Core of Fioriti et al.
                (2025) eq.(27). Both pick, among the core elements, the one whose cost
                shares are least spread; they differ in how spread is measured, and that
                difference decides whether the answer is a point. See `fairness_mode`.
                `cost_of_stability` is ignored. Row generation is
                otherwise unchanged: separation reads only the allocation, never the
                master's objective, so the same loop, the same convergence test and the
                same budget apply. On convergence the range is left in
                self.egalitarian_range and self.egalitarian_converged is True; the
                returned allocation is a core element by construction, so the returned
                flag means the same thing it does in the other modes.

                'range' is NOT a canonical point. min(max - min) fixes the objective
                value, not the argument: if the optimal face is more than a point, which
                element of it comes back is the solver's choice. Measured here at 6
                prosumers over the full 62-coalition core, that face is 220 EUR wide on
                shares of order 500, and two honest solves of the same MP_I -- one over
                the 14 rows row generation produced, one over all 62 -- return the same
                objective 1436.0916 and allocations 108 EUR apart. `egalitarian_width()`
                measures it; it is a diagnostic, not a tie-break.

                'variance' is a canonical point, by strict convexity. That is what makes
                per-player numbers reportable, and it is why Fioriti et al. propose it.

        Returns:
            (Dict[str, float], bool): allocation and success/core-nonempty flag.
        """
        import time

        print("\n" + "="*70)
        print("STARTING ROW GENERATION ALGORITHM")
        print(f"Time limit: {time_limit:.0f} seconds")
        print("="*70)

        start_time = time.time()

        # Step 1: Initialize with singleton coalitions
        initial_coalitions = [[player] for player in self.players]
        self.initialize_master_problem(initial_coalitions,
                                       cost_of_stability=cost_of_stability,
                                       egalitarian=egalitarian)
        mode = fairness_mode(egalitarian)
        strict = bool(cost_of_stability or mode)
        self.fairness_mode = mode
        self.egalitarian_converged = False

        # `tolerance` is RELATIVE to the value of the game. Read absolutely it was the
        # cause of a non-termination that looked like combinatorial hardness: a separation
        # residual of 5.96e-06 at 6 prosumers and roughly 5e-04 at 15 cleared an absolute
        # 1e-6, so the loop kept re-adding a coalition already in the master, the LP never
        # moved, and the identical coalition came back -- 14 458 times in one hour at 6
        # prosumers, on a day that had in fact converged at iteration 11. Because the add
        # rule is the negation of the stop rule, an add that changes nothing makes stopping
        # unreachable; this is `convergence.md` sec.2 in the other loop.
        #
        # The answer does not depend on the constant. Swept over five decades at 6p and 15p,
        # every tolerance that terminates at all returns the same cut count, the same v* and
        # a bit-identical allocation; what the tolerance decides is only whether the loop can
        # stop. The measured room to choose in, between the residual that must be ignored and
        # the smallest genuine violation that must not be:
        #
        #     6p   [1e-5,  1.7  ]      15p  [1e-3,  0.022]
        #
        # so `tolerance = 1e-6` puts the effective threshold at 3.0e-3 and 6.3e-3, inside
        # both. The residual grows much faster with n than |c(N)| does (100x against 2x from
        # 6p to 15p), so |c(N)| is a scale, not a predictor: the band has to be re-measured
        # before trusting this at a size where it has not been.
        _grand = tuple(sorted(self.players))
        tol_eff = tolerance * (1.0 + abs(self.coalition_costs.get(_grand, 0.0)))
        print(f"Convergence tolerance: {tolerance:.1e} relative -> {tol_eff:.3e} "
              f"(|c(N)| = {abs(self.coalition_costs.get(_grand, 0.0)):.1f})")

        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Time limit, checked EVERY iteration. It used to be checked every tenth,
            # which let a run overshoot its budget by up to nine iterations -- the
            # 30-prosumer instance reported 3955 s against a 3600 s budget for exactly
            # that reason. At the sizes where row generation does not converge a single
            # iteration is minutes, so the overshoot is not a rounding error. The
            # printout stays on the tenth iteration to keep the log readable.
            elapsed = time.time() - start_time
            if iteration % 10 == 0:
                print(f"  [Time check at iteration {iteration}] Elapsed: {elapsed:.1f}s / {time_limit:.0f}s")
            if elapsed > time_limit:
                print(f"\n{'='*70}")
                print(f"TIME LIMIT EXCEEDED ({elapsed:.1f}s > {time_limit:.0f}s)")
                print(f"Stopped after {iteration} iterations")
                print(f"{'='*70}\n")
                if cost_of_stability and 'slack' in dir():
                    # Not converged: the master slack over the coalitions
                    # generated so far is a LOWER BOUND on v* (slack is
                    # monotone non-decreasing as rows are added). Expose it so
                    # the partial run still brackets eps_min from below.
                    self.cost_of_stability_value = slack
                    self.weak_eps = slack / len(self.players)
                    self.cos_converged = False
                    self.cos_raw_payoffs = dict(payoffs)
                    print(f"Partial cost-of-stability LOWER BOUND v* >= {slack:.6f} "
                          f"(weak-eps >= {self.weak_eps:.6f})")
                    return self.weak_eps_core_allocation(payoffs, slack), False
                return payoffs if 'payoffs' in dir() else None, False

            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}")
            print(f"{'='*70}")
            
            # Step 2: Solve master problem
            payoffs, slack = self.solve_master_problem()
            if not payoffs:
                # No solution. In egalitarian mode the master carries no slack, so this
                # is infeasibility and infeasibility is a RESULT: the rows generated so
                # far are a subset of the core's, hence a relaxation, so an empty
                # relaxation proves the core itself is empty. In the other modes the
                # slack keeps the master feasible, so it is a solver failure instead.
                print(f"\n{'='*70}")
                print("CORE IS EMPTY (master infeasible on a relaxation)" if egalitarian
                      else "MASTER PROBLEM FAILED")
                print(f"Stopped after {iteration} iterations")
                print(f"{'='*70}\n")
                return None, False
            
            # Step 3: Check if core is empty
            # In cost-of-stability mode a positive slack is exactly the subsidy we
            # are trying to measure (not an emptiness certificate), so we must NOT
            # early-exit here — keep generating rows until separation is clean.
            # NOTE: `slack` is the subsidy v*, not a violation, and is still compared
            # absolutely here and in `core_nonempty` below. Measured |v*| has never exceeded
            # 1.5e-12, so the same noise-floor problem has not bitten; one change at a time.
            if not cost_of_stability and slack > tolerance:
                print(f"\n{'='*70}")
                print(f"CORE IS EMPTY")
                print(f"Slack variable v = {slack:.6f} > {tolerance}")
                print(f"{'='*70}\n")
                return slack, False
            
            # Step 4: Find violated coalition, within what is left of the budget.
            # Giving separation the remainder is what makes `time_limit` a bound on the
            # whole algorithm: the per-iteration check above cannot fire while a single
            # unbounded separation MIP is running.
            remaining = time_limit - (time.time() - start_time)
            coalition, violation = self.find_violated_coalition(payoffs, time_limit=remaining)

            # Step 5: Check convergence. A truncated separation proves nothing: it may
            # have missed a violated coalition it had not reached yet, so an empty
            # result is "out of time", not "converged".
            if (len(coalition) == 0 or violation <= tol_eff) and self.last_separation_truncated:
                print(f"\n{'='*70}")
                print(f"SEPARATION CUT OFF AT ITS TIME LIMIT -- no certificate")
                print(f"Stopped after {iteration} iterations")
                print(f"{'='*70}\n")
                if cost_of_stability:
                    self.cost_of_stability_value = slack
                    self.weak_eps = slack / len(self.players)
                    self.cos_converged = False
                    self.cos_raw_payoffs = dict(payoffs)
                    print(f"Partial cost-of-stability LOWER BOUND v* >= {slack:.6f} "
                          f"(weak-eps >= {self.weak_eps:.6f})")
                    return self.weak_eps_core_allocation(payoffs, slack), False
                return payoffs, False

            if len(coalition) == 0 or violation <= tol_eff:
                if mode:
                    # Both dispersion measures are read off the allocation rather than
                    # the solver's variables, so the two modes report the same pair of
                    # numbers and are directly comparable.
                    # Surplus shares, the coordinates both objectives are written in.
                    y = {i: payoffs[i] - self.coalition_costs[(i,)] for i in self.players}
                    a = sum(y.values()) / len(self.players)   # = v(N)/n by efficiency
                    self.egalitarian_range = max(y.values()) - min(y.values())
                    self.egalitarian_variance = sum((y[i] - a) ** 2 for i in self.players)
                    self.egalitarian_converged = True
                    print(f"\n{'='*70}")
                    print("EGALITARIAN CORE ELEMENT FOUND "
                          + ("(MP_I, min range)" if mode == 'range'
                             else "(Variance Core, unique)"))
                    print(f"Converged after {iteration} iterations")
                    print(f"Range max-min       = {self.egalitarian_range:.6f}")
                    print(f"Sum (p_i - mean)^2  = {self.egalitarian_variance:.6f}")
                    print(f"Equal split v(N)/n = {a:.4f}")
                    total = 0.0
                    for i in self.players:
                        print(f"  Player {i}: {payoffs[i]:.4f}")
                        total += payoffs[i]
                    print(f"  Total: {total:.4f}  (c(N) = {self.coalition_costs[_grand]:.4f})")
                    print(f"{'='*70}\n")
                    return payoffs, True
                if cost_of_stability:
                    # v* = cost of stability (external subsidy needed to make the
                    # core non-empty). Core is non-empty iff v* ≈ 0.
                    self.cost_of_stability_value = slack
                    self.weak_eps = slack / len(self.players)
                    self.cos_converged = True
                    core_nonempty = slack <= tolerance
                    # Raw LP point p (Σp = c(N) - v*) is NOT budget-balanced when the
                    # core is empty; redistribute the subsidy equally to return a
                    # budget-balanced weak-(v*/n)-core allocation q = p + v*/n. When
                    # v*≈0 this equals the true core point (drop-in for the old
                    # behaviour). Raw p kept on self.cos_raw_payoffs.
                    self.cos_raw_payoffs = dict(payoffs)
                    q = self.weak_eps_core_allocation(payoffs, slack)
                    print(f"\n{'='*70}")
                    print(f"COST-OF-STABILITY ROW GENERATION CONVERGED")
                    print(f"Converged after {iteration} iterations")
                    print(f"Cost of stability   v* = {slack:.6f}")
                    print(f"Weak-eps-core eps = v*/n = {self.weak_eps:.6f}")
                    print(f"Core non-empty (v*≈0): {core_nonempty}")
                    print(f"\nBudget-balanced weak-eps-core allocation (Σ q_i = c(N)):")
                    total_payoff = 0
                    for i in self.players:
                        print(f"  Player {i}: {q[i]:.4f}")
                        total_payoff += q[i]
                    print(f"  Total: {total_payoff:.4f}")
                    print(f"{'='*70}\n")
                    return q, core_nonempty
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
                return payoffs, True
            
            # Step 6: Add violated coalition constraint
            print(f"\nAdding violated coalition {coalition} to master problem")
            self._add_coalition_constraint(coalition, cost_of_stability=strict)
        
        print(f"\n{'='*70}")
        print(f"WARNING: Maximum iterations ({max_iterations}) reached")
        print(f"{'='*70}\n")
        if cost_of_stability and 'slack' in dir():
            # Same partial lower-bound capture as the time-limit exit above.
            self.cost_of_stability_value = slack
            self.weak_eps = slack / len(self.players)
            self.cos_converged = False
            self.cos_raw_payoffs = dict(payoffs)
            print(f"Partial cost-of-stability LOWER BOUND v* >= {slack:.6f} "
                  f"(weak-eps >= {self.weak_eps:.6f})")
            return self.weak_eps_core_allocation(payoffs, slack), False
        return payoffs, False

    def egalitarian_width(self, rel_tol: float = 1e-6) -> Dict[str, float]:
        """How much of the MP_I optimum is pinned down, and how much the solver chose.

        `min (P_hi - P_lo)` fixes the objective, not the argument. If the optimal face is
        more than a point, the per-player numbers that come back are whichever vertex the
        simplex landed on, and reporting them as "the egalitarian allocation" would be
        reporting a solver artefact -- the failure that retracted the maximin selection
        (`convergence.md` sec.1.2). This measures it instead of assuming either way: hold
        the range at its optimum and swing each share as far as it will go.

        Returns {player: width}. A width of zero means that share is determined.

        ONE-SIDED, and the direction matters. The rows in the master are the coalitions
        row generation happened to generate, a SUBSET of the core's, so the face measured
        here CONTAINS the true one. Zero width therefore proves the share is pinned;
        nonzero width does not prove it is free. This is the same asymmetry that made the
        2n-LP singleton screen useless as a skip test, and it is why this is reported
        rather than acted on.
        """
        if getattr(self, 'fairness_mode', None) == 'variance':
            raise RuntimeError("the Variance Core minimiser is unique by strict convexity; "
                               "there is no face to measure. Test it by re-solving with a "
                               "different coalition set and comparing allocations instead.")
        if self.range_vars is None or not getattr(self, 'egalitarian_converged', False):
            raise RuntimeError("egalitarian_width() needs a converged "
                               "compute_core(egalitarian='range') run")
        p_hi, p_lo = self.range_vars
        m = self.master_model
        r = self.egalitarian_range
        slack = max(rel_tol, abs(r) * rel_tol)

        m.freeTransform()
        m.addCons(p_hi - p_lo <= r + slack, name="fix_range")

        widths = {}
        for i in self.players:
            bounds = []
            for sense in ("minimize", "maximize"):
                m.freeTransform()
                m.setObjective(self.payoff_vars[i], sense)
                m.hideOutput()
                m.optimize()
                if m.getStatus() != "optimal":
                    bounds = None
                    break
                bounds.append(m.getVal(self.payoff_vars[i]))
            widths[i] = float('nan') if bounds is None else bounds[1] - bounds[0]

        worst = max((w for w in widths.values() if w == w), default=float('nan'))
        print(f"\nMP_I optimal face, width per share (0 = determined):")
        for i in self.players:
            print(f"  {i}: {widths[i]:.3e}")
        print(f"  worst: {worst:.3e}   [upper bound: measured over a superset of the core]")
        return widths

    def weak_eps_core_allocation(self, cos_payoffs: Dict[str, float],
                                 epsilon: Optional[float] = None) -> Dict[str, float]:
        """
        Turn a cost-of-stability allocation into a budget-balanced weak-ε-core point.

        A CoS solution satisfies  Σ_{i∈S} p_i ≤ c(S) for every proper S  and
        Σ_i p_i = c(N) - v*  (the members are collectively subsidised by v*).
        Redistributing the subsidy equally, q_i = p_i + v*/n, restores budget
        balance (Σ_i q_i = c(N)) while giving, for every coalition S,
            Σ_{i∈S} q_i = Σ_{i∈S} p_i + |S|·v*/n ≤ c(S) + |S|·(v*/n),
        i.e. q lies in the weak-(v*/n)-core with ε = v*/n.

        Args:
            cos_payoffs: allocation returned by compute_core(cost_of_stability=True)
                         (or compute_core_brute_force(cost_of_stability=True)).
            epsilon: the cost of stability v*. Defaults to self.cost_of_stability_value.

        Returns:
            Dict[str, float]: budget-balanced weak-(v*/n)-core allocation.
        """
        if epsilon is None:
            epsilon = getattr(self, 'cost_of_stability_value', 0.0)
        shift = epsilon / len(self.players)
        return {i: cos_payoffs[i] + shift for i in self.players}

    def measure_stability_violation(self, payoffs: Dict[str, float], brute_force: bool = False,
                                        time_limit: Optional[float] = None):
            """
            Measure the WEAK-ε-core violation of a given (fixed) payoff allocation.

            Consistent with the cost-of-stability / weak-ε-core framework used across
            the codebase: the reported violation is the PER-CAPITA excess

                ε(x) = max_{∅≠S⊊N}  ( Σ_{i∈S} x_i − c(S) ) / |S|

            (cost form; x_i are costs, negative = profit). ε(x) > 0 ⇒ NOT in the
            core; ε(x) ≤ 0 ⇒ in the core. The unit matches the game's v*/n
            (v* = cost of stability from compute_core), so IP/LP/CHP/PCA allocations
            are directly comparable to the best achievable weak-ε = v*/n.

            Args:
                payoffs: Payoff allocation dictionary {player_id: payoff}.
                brute_force: If True, solve the exact |S|-weighted LP over all 2^n
                        coalitions (small n only). If False (default), find ε(x) by
                        Dinkelbach iteration on the raw-excess separation problem
                        (a handful of separation solves; scalable to large N).
                time_limit: wall-clock budget in seconds for the whole Dinkelbach
                        sequence. Without one a single separation MIP can run
                        indefinitely -- at 15 prosumers with 7 electrolysers one sat at
                        a 100% gap for a quarter of an hour -- and the caller has no way
                        to bound the measurement. On expiry the search stops and
                        `self.last_stability_truncated` is set.
            Returns:
                tuple (coalition, weak_eps_violation, is_imputation)

            Example:
                >>> coalition, eps, isimp = core_comp.measure_stability_violation(payoffs)
                >>> print("in core" if eps <= 1e-6 else f"weak-eps violation {eps:.4f}")
            """
            ## First, check whether the cost allocation is the imputation (at least no worse than the individually played cost)
            is_imputation = self.check_imputation(payoffs)
            if not is_imputation:
                # violation = np.inf
                print("Cost allocation is not an imputation")
                # return [], violation
            if not brute_force:
                coalition, violation = self._measure_weak_eps_separation(
                    payoffs, time_limit=time_limit)
            else:
                self.last_stability_truncated = False
                coalition, violation = self._measure_violation_brute_force(payoffs)
            return coalition, violation, is_imputation

    def _measure_weak_eps_separation(self, payoffs: Dict[str, float],
                                     tolerance: float = 1e-6, max_iter: int = 50,
                                     time_limit: Optional[float] = None):
        """
        Weak-ε-core violation ε(x)=max_S (Σx_S−c(S))/|S| via Dinkelbach iteration.

        |S| in the denominator makes the objective fractional; Dinkelbach solves a
        sequence of parametric problems max_S (excess_S − λ·|S|), each of which is
        EXACTLY the existing raw-excess separation with every payoff shifted down by
        λ:  find_violated_coalition({i: x_i − λ}). λ increases monotonically to ε(x).

        Short-circuit: if the raw max excess ≤ tol the allocation is already in the
        exact core, hence ε(x) ≤ 0 too — report it without iterating (weak-ε only
        differs from strong-ε when there is a genuine positive violation).

        `time_limit` bounds the whole sequence. The asymmetry on expiry matters and is
        why the flag exists: whatever coalition has been found is genuinely violated, so
        a POSITIVE result stands, but the search may not have reached the worst one, so
        the reported value is only a lower bound and a NON-positive result proves
        nothing. Never read "in the core" off a truncated measurement.
        """
        import time as _time
        _t0 = _time.time()
        _left = (lambda: None if time_limit is None
                 else max(1.0, time_limit - (_time.time() - _t0)))
        self.last_stability_truncated = False

        # λ = 0: raw (strong) max excess
        coalition, raw_excess = self.find_violated_coalition(payoffs, time_limit=_left())
        self.last_stability_truncated = bool(self.last_separation_truncated)
        if raw_excess <= tolerance or len(coalition) == 0:
            return coalition, raw_excess          # in core: weak-ε ≤ 0 as well
        lam = raw_excess / len(coalition)
        best_coalition = coalition
        for _ in range(max_iter):
            if time_limit is not None and _time.time() - _t0 > time_limit:
                self.last_stability_truncated = True
                print(f"  Weak-ε measurement stopped at its {time_limit:.0f}s budget; "
                      f"reported violation is a LOWER bound")
                break
            shifted = {i: payoffs[i] - lam for i in self.players}
            S, _f_lam = self.find_violated_coalition(shifted, time_limit=_left())
            if self.last_separation_truncated:
                self.last_stability_truncated = True
            if len(S) == 0:                       # F(λ) ≤ 0 ⇒ λ is the max ratio
                break
            ratio = (sum(payoffs[i] for i in S) - self.compute_coalition_cost(S)) / len(S)
            best_coalition = S
            if ratio <= lam + tolerance:
                lam = ratio
                break
            lam = ratio
        print(f"  Weak-ε (per-capita) violation = {lam:.6f} on coalition {best_coalition}")
        return best_coalition, lam
    def check_imputation(self, payoffs: Dict[str, float]) -> bool:
        """
        Check whether the cost allocation is the imputation (at least no worse than the individually played cost)
        """
        for player in self.players:
            if payoffs[player] - self.coalition_costs[tuple([player])] >= 1e-6:
                return False
        return True
    def compute_core_brute_force(self, cost_of_stability: bool = False) -> Tuple[Dict[str, float], bool]:
        """
        Compute core allocation by solving LP with all coalition constraints.

        cost_of_stability=True switches to the CoS formulation (subsidy v on the
        grand coalition only; every proper coalition strict). Then v* is stored in
        self.cost_of_stability_value / self.weak_eps and the returned flag means
        "core non-empty" (v* ≈ 0). This is the exact (all-coalition) reference for
        the row-generation compute_core(cost_of_stability=True).
        """
        print("\n" + "="*70)
        print("COMPUTING CORE ALLOCATION BY BRUTE FORCE")
        print("="*70)
        lp_model = Model("ViolationLP")
        if len(self.coalition_costs) < (2**len(self.players) - 1):
            print("\nComputing all coalition costs...")
            self.find_all_coalitions(verbose=False)
        else:
            print(f"\n✓ Using cached coalition costs ({len(self.coalition_costs)} coalitions)")
        # Create slack variable v
        v = lp_model.addVar(vtype="C", name="v", lb=0, obj=1.0)
        payoff_dict = {player: None for player in self.players}
        for player in self.players:
            payoff_dict[player] = lp_model.addVar(vtype="C", name=f"chi_{player}", lb=-float('inf'))
        
        # Add constraint for each coalition
        num_constraints = 0
        for coalition_tuple, cost in self.coalition_costs.items():
            coalition = list(coalition_tuple)
                        
            # Constraint: Σ_{i∈S} payoffs[i] <= c(S) (+ v depending on mode)
            lhs = sum(payoff_dict[i] for i in coalition)
            if len(coalition) != len(self.players):
                if cost_of_stability:
                    # strict stability; subsidy sits only on the grand coalition
                    lp_model.addCons(lhs <= cost,
                                    name=f"stability_{'_'.join(sorted(coalition))}")
                else:
                    lp_model.addCons(lhs <= cost + v,
                                    name=f"stability_{'_'.join(sorted(coalition))}")
            else:
                if cost_of_stability:
                    lp_model.addCons(lhs == cost - v,
                                    name=f"stability_{'_'.join(sorted(coalition))}")
                else:
                    lp_model.addCons(lhs == cost,
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
        for player in self.players:
            payoff_dict[player] = lp_model.getVal(payoff_dict[player])
        
        print(f"\n✓ Optimal solution found")
        if cost_of_stability:
            self.cost_of_stability_value = violation
            self.weak_eps = violation / len(self.players)
            core_nonempty = violation <= 1e-6
            # return the budget-balanced weak-(v*/n)-core point (raw p on self)
            self.cos_raw_payoffs = dict(payoff_dict)
            q = self.weak_eps_core_allocation(payoff_dict, violation)
            print(f"  Cost of stability  v* = {violation:.6f}")
            print(f"  Weak-eps-core eps = v*/n = {self.weak_eps:.6f}")
            print(f"  → Core {'NON-EMPTY' if core_nonempty else 'EMPTY'} (v*≈0: {core_nonempty})")
            return q, core_nonempty
        print(f"  Maximum violation (slack v): {violation:.6f}")

        if violation <= 1e-6:
            print(f"  → Payoff IS in the core (stable)")
            return payoff_dict, True
        else:
            print(f"  → Payoff is NOT in the core (violation = {violation:.6f})")
            return payoff_dict, False
    def _measure_violation_brute_force(self, payoffs: Dict[str, float]) -> float:
        """
        Exact WEAK-ε-core violation by LP over all coalitions.

        1. Computes all coalition costs (if not already computed)
        2. Solves: min v
                  s.t. Σ_{i∈S} payoffs[i] ≤ c(S) + |S|·v   for all S
                       v ≥ 0
           whose optimum is  v = max(0, max_S (Σx_S − c(S))/|S|) = weak-ε(x).
           (The |S| weight on the slack is what turns the strong-ε excess into the
           per-capita weak-ε, matching self.weak_eps = v*/n from compute_core.)
        3. Returns (binding coalition, weak-ε).

        Args:
            payoffs: Payoff allocation dictionary

        Returns:
            float: weak-ε violation (optimal slack variable v)
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
                        
            # Constraint: Σ_{i∈S} payoffs[i] ≤ c(S) + |S|·v   (per-capita / weak-ε)
            lhs = sum(payoffs[i] for i in coalition)
            lp_model.addCons(lhs <= cost + len(coalition) * v,
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
        print(f"  Weak-ε (per-capita) violation: {violation:.6f}")

        if violation <= 1e-6:
            print(f"  → Payoff IS in the core (stable)")
        else:
            print(f"  → Payoff is NOT in the core (violation = {violation:.6f})")
            binding_cons = [cons for cons in lp_model.getConss(False) if lp_model.getSlack(cons) <= 1e-6]
            if len(binding_cons) >1:
                print("Multiple binding constraints found!")
            coalition_str = [cons.name.replace("stability_", "") for cons in binding_cons]
            coalition = [tuple(sorted(coalition.split("_"))) for coalition in coalition_str]
            coalition_cost = [self.coalition_costs.get(coalition_tuple, None) for coalition_tuple in coalition]
            payoff_sum = [sum(payoffs[i] for i in coalition_tuple) for coalition_tuple in coalition]

            print(f"  Binding coalition: {coalition}")
            print(f"  Binding coalition cost: {coalition_cost}")
            print(f"  Binding payoff sum: {payoff_sum}")
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
            # INSERT_YOUR_CODE
            import json
            import os

            # Prepare directory and file path for output
            output_dir = getattr(self, "output_dir", ".")
            os.makedirs(output_dir, exist_ok=True)
            coalition_json_path = os.path.join(output_dir, "all_coalition_costs.json")

            # Serialize coalition keys as stringified sorted list for JSON compatibility
            serializable_coalition_costs = {
                json.dumps(sorted(list(coalition))): cost
                for coalition, cost in self.coalition_costs.items()
            }
            with open(coalition_json_path, "w", encoding="utf-8") as f:
                json.dump(serializable_coalition_costs, f, indent=2, ensure_ascii=False)

            print(f"All coalition costs saved to {coalition_json_path}")
            return self.coalition_costs.copy()
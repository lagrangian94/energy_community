"""
Pricer for Local Energy Market column generation
"""
from pyscipopt import Pricer, SCIP_RESULT, quicksum
from typing import Dict, List
import numpy as np
from solver import calculate_column_cost
class LEMPricer(Pricer):
    """
    Pricer for Local Energy Market column generation
    """
    def __init__(self, subproblems: Dict[str, 'PlayerSubproblem'],
                 time_periods: List[int], players: List[str], smoothing: bool = False, *args, **kwargs):
        """
        Initialize pricer

        Args:
            subproblems: Dictionary of player subproblems
            time_periods: List of time periods
            players: List of player IDs
            smoothing: If True, use Wentges (1997) dual variable smoothing
        """
        super().__init__(*args, **kwargs)
        self.subproblems = subproblems
        self.time_periods = time_periods
        self.players = players
        self.iteration = 0
        self.farkas_iteration = 0
        self.lb= -np.inf

        # === Smoothing 관련 ===
        self.smoothing = smoothing
        if self.smoothing:
            # Stability center π̄ (전체 dual vector)
            self.pi_bar_elec = {t: 0.0 for t in time_periods}
            self.pi_bar_heat = {t: 0.0 for t in time_periods}
            self.pi_bar_hydro = {t: 0.0 for t in time_periods}
            self.pi_bar_conv = {player: 0.0 for player in players}
            # Best Lagrangean bound found so far
            self.L_bar = -np.inf
            # Incumbent (upper bound) — will be set from outside or from init_sol
            self.Z_INC = np.inf
    def price(self, farkas=False):
        """
        Common pricing logic for both regular and Farkas pricing
        
        Args:
            farkas: If True, use Farkas multipliers; if False, use regular duals
        """
        # Get current LP objective (only for regular pricing)
        if not farkas:
            lp_obj = self.model.getLPObjVal()

                
        # Get dual values or Farkas multipliers from master problem
        dual_elec = {}
        dual_heat = {}
        dual_hydro = {}
        dual_convexity = {}
        # Get community balance constraint duals/Farkas multipliers
        for t in self.time_periods:
            elec_cons = self.model.data['cons']['community_elec_balance'][t]
            heat_cons = self.model.data['cons']['community_heat_balance'][t]
            hydro_cons = self.model.data['cons']['community_hydro_balance'][t]
            # Get transformed constraints
            t_elec_cons = self.model.getTransformedCons(elec_cons)
            t_heat_cons = self.model.getTransformedCons(heat_cons)
            t_hydro_cons = self.model.getTransformedCons(hydro_cons)
            if farkas:
                # Get Farkas multipliers for infeasible problem
                dual_elec[t] = self.model.getDualfarkasLinear(t_elec_cons)
                dual_heat[t] = self.model.getDualfarkasLinear(t_heat_cons)
                dual_hydro[t] = self.model.getDualfarkasLinear(t_hydro_cons)
                """
                여기선 reduced cost test 어떻게 하는거더라?
                """
            else:
                # Get regular dual multipliers
                dual_elec[t] = self.model.getDualsolLinear(t_elec_cons)
                dual_heat[t] = self.model.getDualsolLinear(t_heat_cons)
                dual_hydro[t] = self.model.getDualsolLinear(t_hydro_cons)
        # Get convexity constraint duals/Farkas multipliers for each player
        for player in self.players:
            conv_cons = self.model.data['cons']['convexity'][player]
            t_conv_cons = self.model.getTransformedCons(conv_cons)
            if farkas:
                dual_convexity[player] = self.model.getDualfarkasLinear(t_conv_cons)
            else:
                dual_convexity[player] = self.model.getDualsolLinear(t_conv_cons)
                
        # DEBUG: Print dual prices for first few iterations
        # if not farkas and self.iteration <= 3:
        #     print(f"\n  [Iter {self.iteration}] Dual Prices Sample:")
        #     sample_times = [0, 6, 12, 18] if len(self.time_periods) >= 24 else self.time_periods[:4]
        #     print(f"    Time  Elec      Heat      Hydro")
        #     for t in sample_times:
        #         print(f"    {t:4d}  {dual_elec[t]:8.4f}  {dual_heat[t]:8.4f}  {dual_hydro[t]:8.4f}")
        #     print(f"    Convexity duals: {dual_convexity}")
                    
        # === Smoothing branch (non-farkas only) ===
        if not farkas and self.smoothing:
            return self._price_smoothed(dual_elec, dual_heat, dual_hydro, dual_convexity, lp_obj)

        # Solve pricing problems for each player
        columns_added = 0
        min_reduced_cost = float('inf')
        debug_sol = {}
        obj_val_list = []
        for player in self.players:
            reduced_cost, solution, obj_val = self.subproblems[player].solve_pricing(
                dual_elec, dual_heat, dual_hydro, dual_convexity[player],
                farkas=farkas)
            debug_sol[player] = solution  
            obj_val_list.append(obj_val)
            # Add column if reduced cost is negative
            if reduced_cost < -1e-8:
                columns_added += 1
                self._add_column(player, solution)
                if farkas:
                    print(f"  {player}: Farkas column added (RC={reduced_cost:.4f})")
                min_reduced_cost = min(min_reduced_cost, reduced_cost)
                # break ## column은 한 player만 넣어도 수렴에 충분.
        if len(obj_val_list) == len(self.players):
            self._update_lagrangian_bound(obj_val_list, farkas=farkas)
            lagrangian_gap = (self.model.getLPObjVal() - self.lb)/np.abs(self.lb) if self.lb != -1*np.inf else np.inf
            ## 사실 이걸 먼저 체크하고 termination 조건 체크한 뒤, 그 다음에 column을 넣어줘야 pricing이 제대로 끝날것임.
        if columns_added == 0:
            print(f"Reduced cost: {reduced_cost:.4f}")
            min_reduced_cost = 0.0
        # Print iteration summary
        if not farkas:
            print(f"Iter {self.iteration:3d} | LP Obj: {lp_obj:12.2f} | LB: {self.lb:12.2f} | Min RC: {min_reduced_cost:10.4f} | Columns added: {columns_added}")

        # else:
        #     print(f"  Total Farkas columns added: {columns_added}")
        
        # Check convergence
        if columns_added == 0:
            if farkas:
                print("WARNING: No Farkas columns found - problem may be infeasible")
                return {"result": SCIP_RESULT.DIDNOTRUN}
            else:
                print("\n>>> Column generation converged: No negative reduced cost found <<<\n")
                return {"result": SCIP_RESULT.SUCCESS}
        
        return {"result": SCIP_RESULT.SUCCESS}
    def pricerredcost(self):
        """
        Regular pricing callback - generate columns with negative reduced cost
        """
        self.iteration += 1
        return self.price(farkas=False)
    
    def pricerfarkas(self):
        """
        Farkas pricing callback - restore feasibility when master is infeasible
        """
        self.farkas_iteration += 1
        print(f"\n=== Farkas Pricing Iteration {self.farkas_iteration} ===")
        return self.price(farkas=True)
    def _add_column(self, player: str, solution: Dict):
        """
        Add a new column (extreme point) to the master problem
        
        Args:
            player: Player ID
            solution: Solution dictionary from subproblem
        """
        # Create new variable in master problem
        col_idx = len(self.model.data['vars'][player])
        var_name = f"lambda_{player}_{col_idx}"
        
        # Variable is continuous in [0, 1] (for RMP)
        new_var = self.model.addVar(
            name=var_name,
            vtype="C",
            lb=0.0,
            obj=calculate_column_cost(player, solution, self.subproblems[player].parameters, self.time_periods),
            pricedVar=True
        )
        
        # Store variable and solution
        self.model.data['vars'][player][col_idx] = {
            'var': new_var,
            'solution': solution
        }

        # Add variable to convexity constraint
        self.model.addConsCoeff(
            self.model.getTransformedCons(self.model.data['cons']['convexity'][player]),
            new_var,
            1.0
        )
        
        # Add variable to community balance constraints with appropriate coefficients
        for t in self.time_periods:
            # Electricity
            e_E_com_val = solution.get('e_E_com', {}).get((player, t), 0)
            i_E_com_val = solution.get('i_E_com', {}).get((player, t), 0)
            coeff_elec = i_E_com_val - e_E_com_val
            
            self.model.addConsCoeff(
                self.model.getTransformedCons(self.model.data['cons']['community_elec_balance'][t]),
                new_var,
                coeff_elec
            )
            
            # Heat
            e_H_com_val = solution.get('e_H_com', {}).get((player, t), 0)
            i_H_com_val = solution.get('i_H_com', {}).get((player, t), 0)
            coeff_heat = i_H_com_val - e_H_com_val
            
            self.model.addConsCoeff(
                self.model.getTransformedCons(self.model.data['cons']['community_heat_balance'][t]),
                new_var,
                coeff_heat
            )
            
            # Hydrogen
            e_G_com_val = solution.get('e_G_com', {}).get((player, t), 0)
            i_G_com_val = solution.get('i_G_com', {}).get((player, t), 0)
            coeff_hydro = i_G_com_val - e_G_com_val
            
            self.model.addConsCoeff(
                self.model.getTransformedCons(self.model.data['cons']['community_hydro_balance'][t]),
                new_var,
                coeff_hydro
            )
    def _update_lagrangian_bound(self, obj_val_list: List[float], farkas: bool):
        """
        Update Lagrangian bound
        이 문제에서 linking constraint의 right-hand-side는 전부 zero이기 때문에, subproblem들의 objective value만 합하면 됨.
        """
        if farkas:
            return
        try:
            self.lb = max(self.lb, np.sum(obj_val_list))
        except:
            print("stop")
        return

    # ===================================================================
    # Smoothing 관련 메서드 (Wentges 1997 / Pessoa et al. 2010)
    # ===================================================================

    def _price_smoothed(self, pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv, lp_obj):
        """
        Smoothed pricing: Steps 2-8 from cg_smoothing.md
        """
        # Step 2: α 계산
        Z_RM = lp_obj
        alpha = self._compute_alpha(Z_RM)

        # Step 3: π^ST 계산 (전체 vector에 대해 한 번에)
        pi_ST_elec = {t: alpha * pi_RM_elec[t] + (1 - alpha) * self.pi_bar_elec[t] for t in self.time_periods}
        pi_ST_heat = {t: alpha * pi_RM_heat[t] + (1 - alpha) * self.pi_bar_heat[t] for t in self.time_periods}
        pi_ST_hydro = {t: alpha * pi_RM_hydro[t] + (1 - alpha) * self.pi_bar_hydro[t] for t in self.time_periods}
        pi_ST_conv = {p: alpha * pi_RM_conv[p] + (1 - alpha) * self.pi_bar_conv[p] for p in self.players}

        # Step 4: π^ST로 모든 player pricing
        columns_added = 0
        st_solutions = {}
        st_obj_vals = {}
        for player in self.players:
            rc_st, sol, obj_val = self.subproblems[player].solve_pricing(
                pi_ST_elec, pi_ST_heat, pi_ST_hydro, pi_ST_conv[player])
            st_solutions[player] = sol
            st_obj_vals[player] = obj_val

        # Step 5: 각 column에 대해 π^RM 기준 reduced cost 재계산 (subproblem re-solve 없이)
        misprice = False
        for player in self.players:
            if st_solutions[player] is not None:
                rc_rm = self._recalculate_reduced_cost_wrt_pi_RM(
                    player, st_solutions[player], pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv[player])
                if rc_rm < -1e-8:
                    self._add_column(player, st_solutions[player])
                    columns_added += 1

        # Step 6: L(π^ST) 계산 및 π̄ 업데이트
        L_pi_ST = self._compute_lagrangean_bound(st_obj_vals, pi_ST_conv)
        if L_pi_ST > self.L_bar:
            self.L_bar = L_pi_ST
            self.pi_bar_elec = dict(pi_ST_elec)
            self.pi_bar_heat = dict(pi_ST_heat)
            self.pi_bar_hydro = dict(pi_ST_hydro)
            self.pi_bar_conv = dict(pi_ST_conv)

        # Also update the standard Lagrangian bound for consistency
        self.lb = max(self.lb, L_pi_ST)

        # Step 7: Misprice fallback — π^RM으로 재pricing
        if columns_added == 0:
            misprice = True
            for player in self.players:
                rc_rm, sol, obj_val = self.subproblems[player].solve_pricing(
                    pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv[player])
                if rc_rm < -1e-8:
                    self._add_column(player, sol)
                    columns_added += 1

        # Logging
        if alpha < 1.0:
            if misprice and columns_added > 0:
                misprice_str = "Y (fallback)"
            elif misprice and columns_added == 0:
                misprice_str = "Y (converged)"
            else:
                misprice_str = "N"
            print(f"Iter {self.iteration:3d} | LP Obj: {lp_obj:12.2f} | L_bar: {self.L_bar:12.2f} | "
                  f"α: {alpha:.2f} | Misprice: {misprice_str} | Cols: {columns_added}")
        else:
            status_str = "CONVERGED" if columns_added == 0 else f"Cols: {columns_added}"
            print(f"Iter {self.iteration:3d} | LP Obj: {lp_obj:12.2f} | L_bar: {self.L_bar:12.2f} | "
                  f"α: {alpha:.2f} | STANDARD CG | {status_str}")

        # Step 8: 여전히 0이면 진짜 수렴
        if columns_added == 0:
            print("\n>>> Column generation converged: No negative reduced cost found <<<\n")
            return {"result": SCIP_RESULT.SUCCESS}

        return {"result": SCIP_RESULT.SUCCESS}

    def _compute_alpha(self, Z_RM):
        """
        Pessoa et al. (2010) Section 3.2의 adaptive α 계산.
        Z_INC: incumbent (best known integer solution value)
        L_bar: best known Lagrangean lower bound
        Z_RM: current RMP objective value
        """
        base_alpha = 0.1
        if self.L_bar == -np.inf:
            return base_alpha

        gap = Z_RM - self.L_bar
        if gap < 1e-2:
            # Gap이 충분히 작으면 standard CG로 전환
            return 1.0

        if self.Z_INC < np.inf and Z_RM > self.Z_INC:
            inc_gap = self.Z_INC - self.L_bar
            if inc_gap > 1e-6:
                return base_alpha * inc_gap / gap
            else:
                return base_alpha
        else:
            return base_alpha

    def _recalculate_reduced_cost_wrt_pi_RM(self, player, solution, pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv):
        """
        Subproblem을 다시 풀지 않고, 이미 찾은 solution의 변수값으로 π^RM 기준 reduced cost를 직접 계산.

        RC = original_cost - Σ_t π^RM_elec[t] * (i_E_com - e_E_com)
                           - Σ_t π^RM_heat[t] * (i_H_com - e_H_com)
                           - Σ_t π^RM_hydro[t] * (i_G_com - e_G_com)
                           - π^RM_conv
        """
        # Original cost (same as calculate_column_cost)
        cost = calculate_column_cost(player, solution, self.subproblems[player].parameters, self.time_periods)

        # Dual 항 차감
        dual_contribution = 0.0
        for t in self.time_periods:
            # Electricity
            i_E = solution.get('i_E_com', {}).get((player, t), 0.0)
            e_E = solution.get('e_E_com', {}).get((player, t), 0.0)
            dual_contribution += pi_RM_elec[t] * (i_E - e_E)
            # Heat
            i_H = solution.get('i_H_com', {}).get((player, t), 0.0)
            e_H = solution.get('e_H_com', {}).get((player, t), 0.0)
            dual_contribution += pi_RM_heat[t] * (i_H - e_H)
            # Hydrogen
            i_G = solution.get('i_G_com', {}).get((player, t), 0.0)
            e_G = solution.get('e_G_com', {}).get((player, t), 0.0)
            dual_contribution += pi_RM_hydro[t] * (i_G - e_G)

        reduced_cost = cost - dual_contribution - pi_RM_conv
        return reduced_cost

    def _compute_lagrangean_bound(self, obj_vals, pi_conv):
        """
        L(π) = Σ_j obj_val_j
        Community balance RHS가 전부 0이므로 그 dual 항은 소멸.
        obj_val은 solve_pricing()에서 반환하는 self.model.getObjVal() (dual_convexity 빼기 전의 값).
        """
        L = 0.0
        for player in self.players:
            if obj_vals[player] is not None:
                L += obj_vals[player]
            else:
                return -np.inf
        return L
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
            # Stability centers for the reserve/peak coupling duals (adding_cons.txt).
            # Always allocated (harmless when reserve/peak disabled — never read).
            self.pi_bar_resup = {t: 0.0 for t in time_periods}
            self.pi_bar_resdn = {t: 0.0 for t in time_periods}
            self.pi_bar_peak = {t: 0.0 for t in time_periods}
            # Best Lagrangean bound found so far
            self.L_bar = -np.inf
            # Incumbent (upper bound) — will be set from outside or from init_sol
            self.Z_INC = np.inf
    @staticmethod
    def _pricing_tol(scale):
        """The one pricing tolerance, RELATIVE to the problem's own scale.

        Every threshold in this pricer used to be absolute, which is what made it
        size-dependent: `-1e-8` is `1e-12` relative on a master worth 6499, i.e. four
        orders below what double precision can resolve in a reduced cost that is a
        difference of terms of size 1e+4. The pricer was being asked to chase noise.

        It matters more that this is the SAME number everywhere than what it is.
        Convergence is declared when no column is added, so the add rule IS the stop
        rule; two different add thresholds in two branches (`-1e-8` in the main one,
        `-1e-7` in the misprice fallback) meant a reduced cost landing between them
        kept the adding branch permanently active and the stopping branch permanently
        unreachable. At 15 prosumers on the 4-hour reserve product that is exactly what
        happened: 6474 rounds, 71 494 columns of which 95% were duplicates, one column
        re-added 6148 times, `minRC` and `LB-Z` frozen to the last bit, and the 1800 s
        master limit hit — with the Lagrangian bound already equal to the LP objective
        to 3.6e-11 relative since round ~500.

        Accuracy given up: termination now allows `|rc| <= tol` per prosumer, so the
        bound is within `n * tol` of `z*` — about 1e-4 EUR at 15p, i.e. 1e-8 relative.
        Far below anything reported.
        """
        return 1e-9 * (1.0 + abs(scale))

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

        # Reserve / peak coupling duals (adding_cons.txt). These rows exist in the
        # RMP only when reserve/peak is enabled, so the dicts are non-empty exactly
        # then; otherwise the duals stay None and every downstream use is skipped,
        # preserving the flags-off invariant byte-for-byte. Duals are passed RAW
        # (no sign flip) — solve_pricing applies them with the RC = c - sum pi*a
        # convention, identical to the community-balance handling.
        dual_resup = dual_resdn = dual_peak = None
        if self.model.data['cons']['reserve_up']:
            dual_resup, dual_resdn = {}, {}
            for t in self.time_periods:
                up_cons = self.model.getTransformedCons(self.model.data['cons']['reserve_up'][t])
                dn_cons = self.model.getTransformedCons(self.model.data['cons']['reserve_dn'][t])
                if farkas:
                    dual_resup[t] = self.model.getDualfarkasLinear(up_cons)
                    dual_resdn[t] = self.model.getDualfarkasLinear(dn_cons)
                else:
                    dual_resup[t] = self.model.getDualsolLinear(up_cons)
                    dual_resdn[t] = self.model.getDualsolLinear(dn_cons)
        if self.model.data['cons']['peak']:
            dual_peak = {}
            for t in self.time_periods:
                pk_cons = self.model.getTransformedCons(self.model.data['cons']['peak'][t])
                if farkas:
                    dual_peak[t] = self.model.getDualfarkasLinear(pk_cons)
                else:
                    dual_peak[t] = self.model.getDualsolLinear(pk_cons)

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
            return self._price_smoothed(dual_elec, dual_heat, dual_hydro, dual_convexity, lp_obj,
                                        dual_resup, dual_resdn, dual_peak)

        # Solve pricing problems for each player
        columns_added = 0
        min_reduced_cost = float('inf')
        debug_sol = {}
        obj_val_list = []
        # Farkas pricing measures infeasibility, not cost, so the LP objective is not a
        # scale for it; it keeps the absolute threshold.
        tol = 1e-8 if farkas else self._pricing_tol(lp_obj)
        for player in self.players:
            reduced_cost, solution, obj_val = self.subproblems[player].solve_pricing(
                dual_elec, dual_heat, dual_hydro, dual_convexity[player],
                farkas=farkas,
                dual_resup=dual_resup, dual_resdn=dual_resdn, dual_peak=dual_peak)
            debug_sol[player] = solution
            obj_val_list.append(obj_val)
            # Add column if reduced cost is negative
            if reduced_cost < -tol:
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

        # ---- Reserve / peak coupling rows (adding_cons.txt sec.4) ----
        # Homogeneous linking rows added to the RMP exactly like community balance.
        # Column coefficients (this player's contribution at t) match the spec table:
        #   reserve up : -r_plus[u,t]      reserve dn : -r_minus[u,t]
        #   peak       : (i_E_gri - e_E_gri)[u,t]
        # Guarded by row existence so nothing is stamped when reserve/peak is off.
        if self.model.data['cons']['reserve_up']:
            for t in self.time_periods:
                r_plus_val = solution.get('r_plus', {}).get((player, t), 0.0)
                r_minus_val = solution.get('r_minus', {}).get((player, t), 0.0)
                self.model.addConsCoeff(
                    self.model.getTransformedCons(self.model.data['cons']['reserve_up'][t]),
                    new_var, -r_plus_val)
                self.model.addConsCoeff(
                    self.model.getTransformedCons(self.model.data['cons']['reserve_dn'][t]),
                    new_var, -r_minus_val)
        if self.model.data['cons']['peak']:
            for t in self.time_periods:
                i_gri = solution.get('i_E_gri', {}).get((player, t), 0.0)
                e_gri = solution.get('e_E_gri', {}).get((player, t), 0.0)
                self.model.addConsCoeff(
                    self.model.getTransformedCons(self.model.data['cons']['peak'][t]),
                    new_var, i_gri - e_gri)

    def _update_lagrangian_bound(self, obj_val_list: List[float], farkas: bool):
        """
        Update Lagrangian bound
        이 문제에서 linking constraint의 right-hand-side는 전부 zero이기 때문에, subproblem들의 objective value만 합하면 됨.

        Reserve/peak note (adding_cons.txt): the coupling rows are ALSO homogeneous
        (RHS 0), and their shared common-block variables r_sym, p live in the
        RMP as first-class (non-priced) variables. The Lagrangian dual gains a term
        min_{x0>=0}[ c0^T x0 - mu^T A0 x0 ]; because r_sym/p sit in the RMP, at
        every LP optimum where the pricer is invoked their reduced costs are >= 0,
        so that inner min is exactly 0. Hence L(mu) = sum_j obj_val_j is STILL the
        correct bound with reserve/peak on — no extra term is added here. (Adding
        the *primal* shared-var cost -|T|*pi_res*r_sym would be wrong: it is a
        different quantity and would corrupt the bound.)

        Symmetric product: r_sym appears in BOTH row families with coefficient
        +1, so its reduced cost is |T|*pi_res - sum_t (mu_plus[t] + mu_minus[t])
        and the >= 0 argument above is unchanged.
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

    def _price_smoothed(self, pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv, lp_obj,
                        dual_resup=None, dual_resdn=None, dual_peak=None):
        """
        Smoothed pricing: Steps 2-8 from cg_smoothing.md

        Reserve/peak coupling duals (adding_cons.txt) are smoothed on the SAME
        stability center as the balance duals: pi^ST = alpha*pi^RM + (1-alpha)*pi_bar,
        with their own centers pi_bar_resup/resdn/peak advanced whenever L(pi^ST)
        improves. None => disabled (skipped, flags-off invariant preserved).
        """
        # Step 2: α 계산
        Z_RM = lp_obj
        alpha = self._compute_alpha(Z_RM)

        # Step 3: π^ST 계산 (전체 vector에 대해 한 번에)
        pi_ST_elec = {t: alpha * pi_RM_elec[t] + (1 - alpha) * self.pi_bar_elec[t] for t in self.time_periods}
        pi_ST_heat = {t: alpha * pi_RM_heat[t] + (1 - alpha) * self.pi_bar_heat[t] for t in self.time_periods}
        pi_ST_hydro = {t: alpha * pi_RM_hydro[t] + (1 - alpha) * self.pi_bar_hydro[t] for t in self.time_periods}
        pi_ST_conv = {p: alpha * pi_RM_conv[p] + (1 - alpha) * self.pi_bar_conv[p] for p in self.players}
        # Reserve / peak smoothed duals (None when disabled).
        pi_ST_resup = pi_ST_resdn = pi_ST_peak = None
        if dual_resup is not None:
            pi_ST_resup = {t: alpha * dual_resup[t] + (1 - alpha) * self.pi_bar_resup[t] for t in self.time_periods}
            pi_ST_resdn = {t: alpha * dual_resdn[t] + (1 - alpha) * self.pi_bar_resdn[t] for t in self.time_periods}
        if dual_peak is not None:
            pi_ST_peak = {t: alpha * dual_peak[t] + (1 - alpha) * self.pi_bar_peak[t] for t in self.time_periods}

        # Step 6 (moved up): L(π^ST) 계산 및 π̄ 업데이트
        # π^ST로 pricing하여 Lagrangean bound를 먼저 계산
        st_solutions = {}
        st_obj_vals = {}
        for player in self.players:
            rc_st, sol, obj_val = self.subproblems[player].solve_pricing(
                pi_ST_elec, pi_ST_heat, pi_ST_hydro, pi_ST_conv[player],
                dual_resup=pi_ST_resup, dual_resdn=pi_ST_resdn, dual_peak=pi_ST_peak)
            st_solutions[player] = sol
            st_obj_vals[player] = obj_val

        L_pi_ST = self._compute_lagrangean_bound(st_obj_vals, pi_ST_conv)
        if L_pi_ST > self.L_bar:
            self.L_bar = L_pi_ST
            self.pi_bar_elec = dict(pi_ST_elec)
            self.pi_bar_heat = dict(pi_ST_heat)
            self.pi_bar_hydro = dict(pi_ST_hydro)
            self.pi_bar_conv = dict(pi_ST_conv)
            if pi_ST_resup is not None:
                self.pi_bar_resup = dict(pi_ST_resup)
                self.pi_bar_resdn = dict(pi_ST_resdn)
            if pi_ST_peak is not None:
                self.pi_bar_peak = dict(pi_ST_peak)

        # Also update the standard Lagrangian bound for consistency
        self.lb = max(self.lb, L_pi_ST)

        # Early termination: LB가 LP objective에 충분히 가까우면 column 추가 없이 종료
        tol = self._pricing_tol(Z_RM)
        if self.lb > Z_RM - tol:
            print(f"Iter {self.iteration:3d} | LP Obj: {lp_obj:12.2f} | L_bar: {self.L_bar:12.2f} | "
                  f"α: {alpha:.2f} | EARLY TERMINATION (LB ≥ Z_RM)")
            print("\n>>> Column generation converged: LB reached LP objective <<<\n")
            return {"result": SCIP_RESULT.SUCCESS}

        # Step 4→5: 각 column에 대해 π^RM 기준 reduced cost 재계산 (subproblem re-solve 없이)
        columns_added = 0
        misprice = False
        # worst reduced cost seen this round, logged only. Without it a stalled run is
        # indistinguishable from a working one: the log shows LP Obj == L_bar and
        # columns still being added every iteration, and the question of whether those
        # columns carry real reduced cost or only numerical noise cannot be answered.
        min_rc_rm = 0.0
        for player in self.players:
            if st_solutions[player] is not None:
                rc_rm = self._recalculate_reduced_cost_wrt_pi_RM(
                    player, st_solutions[player], pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv[player],
                    dual_resup, dual_resdn, dual_peak)
                min_rc_rm = min(min_rc_rm, rc_rm)
                if rc_rm < -tol:
                    self._add_column(player, st_solutions[player])
                    columns_added += 1

        # Step 7: Misprice fallback — π^RM으로 재pricing
        if columns_added == 0:
            misprice = True
            rm_solutions = {}
            rm_obj_vals = {}
            for player in self.players:
                rc_rm, sol, obj_val = self.subproblems[player].solve_pricing(
                    pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv[player],
                    dual_resup=dual_resup, dual_resdn=dual_resdn, dual_peak=dual_peak)
                if rc_rm < -tol:
                    self._add_column(player, sol)
                    columns_added += 1
                rm_solutions[player] = sol
                rm_obj_vals[player] = obj_val
            L_pi_RM = self._compute_lagrangean_bound(rm_obj_vals, pi_RM_conv)
            self.lb = max(self.lb, L_pi_RM)

        # Logging
        if alpha < 1.0:
            if misprice and columns_added > 0:
                misprice_str = "Y (fallback)"
            elif misprice and columns_added == 0:
                misprice_str = "Y (converged)"
            else:
                misprice_str = "N"
            print(f"Iter {self.iteration:3d} | LP Obj: {lp_obj:12.2f} | L_bar: {self.L_bar:12.2f} | "
                  f"α: {alpha:.2f} | Misprice: {misprice_str} | Cols: {columns_added} | "
                  f"minRC: {min_rc_rm:9.2e} | LB-Z: {self.lb - Z_RM:9.2e}")
        else:
            status_str = "CONVERGED" if columns_added == 0 else f"Cols: {columns_added}"
            print(f"Iter {self.iteration:3d} | LP Obj: {lp_obj:12.2f} | L_bar: {self.L_bar:12.2f} | "
                  f"α: {alpha:.2f} | STANDARD CG | {status_str} | "
                  f"minRC: {min_rc_rm:9.2e} | LB-Z: {self.lb - Z_RM:9.2e}")

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

    def _recalculate_reduced_cost_wrt_pi_RM(self, player, solution, pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv,
                                            pi_RM_resup=None, pi_RM_resdn=None, pi_RM_peak=None):
        """
        Subproblem을 다시 풀지 않고, 이미 찾은 solution의 변수값으로 π^RM 기준 reduced cost를 직접 계산.

        RC = original_cost - Σ_row π^RM_row * a_row(col) - π^RM_conv,
        where a_row(col) is this column's coefficient in that linking row:
          balance    : (i_com - e_com)
          reserve up : -r_plus[u,t]      reserve dn : -r_minus[u,t]
          peak       : (i_E_gri - e_E_gri)[u,t]
        (dual_contribution = Σ_row π * a; identical convention to solve_pricing.)
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
            # Reserve up/down (column coeff -r_plus / -r_minus)
            if pi_RM_resup is not None:
                r_plus = solution.get('r_plus', {}).get((player, t), 0.0)
                r_minus = solution.get('r_minus', {}).get((player, t), 0.0)
                dual_contribution += pi_RM_resup[t] * (-r_plus)
                dual_contribution += pi_RM_resdn[t] * (-r_minus)
            # Peak (column coeff i_E_gri - e_E_gri)
            if pi_RM_peak is not None:
                i_gri = solution.get('i_E_gri', {}).get((player, t), 0.0)
                e_gri = solution.get('e_E_gri', {}).get((player, t), 0.0)
                dual_contribution += pi_RM_peak[t] * (i_gri - e_gri)

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
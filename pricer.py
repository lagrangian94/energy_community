"""
Pricer for Local Energy Market column generation
"""
from pyscipopt import Pricer, SCIP_RESULT, quicksum
from typing import Dict, List


class LEMPricer(Pricer):
    """
    Pricer for Local Energy Market column generation
    """
    def __init__(self, master_model, subproblems: Dict[str, 'PlayerSubproblem'], 
                 time_periods: List[int], players: List[str], *args, **kwargs):
        """
        Initialize pricer
        
        Args:
            master_model: Master problem SCIP model
            subproblems: Dictionary of player subproblems
            time_periods: List of time periods
            players: List of player IDs
        """
        super().__init__(*args, **kwargs)
        self.subproblems = subproblems
        self.time_periods = time_periods
        self.players = players
        self.iteration = 0
        self.farkas_iteration = 0
    
    def price(self, farkas=False):
        """
        Common pricing logic for both regular and Farkas pricing
        
        Args:
            farkas: If True, use Farkas multipliers; if False, use regular duals
        """
        # Get current LP objective (only for regular pricing)
        if not farkas:
            try:
                lp_obj = self.model.getLPObjVal()
            except:
                lp_obj = 0.0
                
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
        if not farkas and self.iteration <= 3:
            print(f"\n  [Iter {self.iteration}] Dual Prices Sample:")
            sample_times = [0, 6, 12, 18] if len(self.time_periods) >= 24 else self.time_periods[:4]
            print(f"    Time  Elec      Heat      Hydro")
            for t in sample_times:
                print(f"    {t:4d}  {dual_elec[t]:8.4f}  {dual_heat[t]:8.4f}  {dual_hydro[t]:8.4f}")
            print(f"    Convexity duals: {dual_convexity}")
            
            # Additional debug: check constraint status
            if self.iteration <= 3:
                for player in ['u1']:  # Just check one player
                    conv_cons = self.model.data['cons']['convexity'][player]
                    t_conv_cons = self.model.getTransformedCons(conv_cons)
                    
                    print(f"\n    === Convexity Constraint Debug for {player} ===")
                    print(f"    Original constraint: {conv_cons}")
                    print(f"    Transformed constraint: {t_conv_cons}")
                    
                    if t_conv_cons is not None:
                        try:
                            lhs = self.model.getLhs(t_conv_cons)
                            rhs = self.model.getRhs(t_conv_cons)
                            activity = self.model.getActivity(t_conv_cons)
                            slack = rhs - activity if rhs < float('inf') else activity - lhs
                            
                            print(f"    LHS={lhs}, RHS={rhs}, Activity={activity:.6f}, Slack={slack:.6f}")
                            
                            # Get all variables in constraint
                            vals = self.model.getValsLinear(t_conv_cons)
                            print(f"    Variables in constraint: {len(vals)}")
                            for var_name, coeff in list(vals.items())[:5]:  # Show first 5
                                print(f"      {var_name}: coeff={coeff}")
                            # Print each variable and its VALUE in LP solution
                            print(f"    Variable values in LP solution:")
                            for k in range(self.iteration):
                                var = self.model.data["vars"]["u1", k]["var"]
                                val = self.model.getVal(var)
                                print(f"      {var.name}: value={val:.6f}")
                            # Try to get dual
                            try:
                                dual_method1 = self.model.getDualsolLinear(t_conv_cons)
                                print(f"    Dual (getDualsolLinear): {dual_method1}")
                            except Exception as e:
                                print(f"    Dual (getDualsolLinear): Failed - {e}")
                                
                        except Exception as e:
                            print(f"    Could not get constraint info: {e}")
                
                # Also check one community balance constraint
                for t in [0]:
                    elec_cons = self.model.data['cons']['community_elec_balance'][t]
                    t_elec_cons = self.model.getTransformedCons(elec_cons)
                    if t_elec_cons is not None:
                        try:
                            lhs = self.model.getLhs(t_elec_cons)
                            rhs = self.model.getRhs(t_elec_cons)
                            activity = self.model.getActivity(t_elec_cons)
                            slack = rhs - activity if rhs < float('inf') else activity - lhs
                            print(f"\n    Elec balance[{t}]: LHS={lhs}, RHS={rhs}, Activity={activity:.6f}, Slack={slack:.6f}")
                            
                            # Try to get dual in different ways
                            try:
                                dual_method1 = self.model.getDualsolLinear(t_elec_cons)
                                print(f"    Dual (getDualsolLinear): {dual_method1}")
                            except:
                                print(f"    Dual (getDualsolLinear): Failed")
                                
                        except Exception as e:
                            print(f"    Could not get constraint info: {e}")
        
        # Solve pricing problems for each player
        columns_added = 0
        min_reduced_cost = float('inf')
        
        print(f"iteration {self.iteration}, dual_elec: {dual_elec}")
        print(f"iteration {self.iteration}, dual_heat: {dual_heat}")
        print(f"iteration {self.iteration}, dual_hydro: {dual_hydro}")
        print(f"iteration {self.iteration}, dual_convexity: {dual_convexity}")
        if self.iteration > 5000:
            print('stop')
        debug_sol = {}
        for player in self.players:
            reduced_cost, solution = self.subproblems[player].solve_pricing(
                dual_elec, dual_heat, dual_hydro, dual_convexity[player], farkas=farkas
            )
            debug_sol[player] = solution
            min_reduced_cost = min(min_reduced_cost, reduced_cost)
            
            # Add column if reduced cost is negative
            if reduced_cost < -1e-6:
                columns_added += 1
                self._add_column(player, solution)
                if farkas:
                    print(f"  {player}: Farkas column added (RC={reduced_cost:.4f})")
        
        # Print iteration summary
        if not farkas:
            print(f"Iter {self.iteration:3d} | LP Obj: {lp_obj:12.2f} | Min RC: {min_reduced_cost:10.4f} | Columns added: {columns_added}")
        else:
            print(f"  Total Farkas columns added: {columns_added}")
        
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
        col_idx = len([k for k in self.model.data['vars'].keys() if k[0] == player])
        var_name = f"lambda_{player}_{col_idx}"
        
        # Variable is continuous in [0, 1] (for RMP)
        new_var = self.model.addVar(
            name=var_name,
            vtype="C",
            lb=0.0,
            obj=self._calculate_column_cost(player, solution),
            pricedVar=True
        )
        
        # Store variable and solution
        self.model.data['vars'][player, col_idx] = {
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
            
    def _calculate_column_cost(self, player: str, solution: Dict) -> float:
        """
        Calculate the cost coefficient for a column (extreme point)
        This is the objective value of the subproblem with original costs
        """
        cost = 0.0
        params = self.subproblems[player].parameters
        
        # Grid trading costs
        for t in self.time_periods:
            # Electricity grid
            e_E_gri = solution.get('e_E_gri', {}).get((player, t), 0)
            i_E_gri = solution.get('i_E_gri', {}).get((player, t), 0)
            cost += i_E_gri * params.get(f'pi_E_gri_import_{t}', 0)
            cost -= e_E_gri * params.get(f'pi_E_gri_export_{t}', 0)
            
            # Heat grid
            e_H_gri = solution.get('e_H_gri', {}).get((player, t), 0)
            i_H_gri = solution.get('i_H_gri', {}).get((player, t), 0)
            cost += i_H_gri * params.get(f'pi_H_gri_import_{t}', 0)
            cost -= e_H_gri * params.get(f'pi_H_gri_export_{t}', 0)
            
            # Hydrogen grid
            e_G_gri = solution.get('e_G_gri', {}).get((player, t), 0)
            i_G_gri = solution.get('i_G_gri', {}).get((player, t), 0)
            cost += i_G_gri * params.get(f'pi_G_gri_import_{t}', 0)
            cost -= e_G_gri * params.get(f'pi_G_gri_export_{t}', 0)
            
            # Production costs
            p_res = solution.get('p', {}).get((player, 'res', t), 0)
            p_hp = solution.get('p', {}).get((player, 'hp', t), 0)
            p_els = solution.get('p', {}).get((player, 'els', t), 0)
            
            cost += p_res * params.get(f'c_res_{player}', 0)
            cost += p_hp * params.get(f'c_hp_{player}', 0)
            cost += p_els * params.get(f'c_els_{player}', 0)
            
            # Startup costs
            z_su = solution.get('z_su', {}).get((player, t), 0)
            cost += z_su * params.get(f'c_su_{player}', 0)
        
        return cost
from pyscipopt import Model, quicksum, Pricer, SCIP_RESULT, SCIP_PARAMSETTING
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('/mnt/project')

from compact import LocalEnergyMarket, solve_and_extract_results

class PlayerSubproblem:
    """
    Individual player's subproblem for column generation
    Reuses LocalEnergyMarket class with single player
    """
    def __init__(self, player: str, time_periods: List[int], parameters: Dict, isLP: bool = True):
        """
        Create subproblem for a single player
        
        Args:
            player: Single player ID
            time_periods: List of time periods
            parameters: Model parameters
            isLP: Whether to use LP relaxation (True for column generation)
        """
        self.player = player
        self.time_periods = time_periods
        self.parameters = parameters
        
        # Create LocalEnergyMarket with single player and dwr=True
        # dwr=True removes community balance constraints
        self.lem = LocalEnergyMarket(
            players=[player],
            time_periods=time_periods,
            parameters=parameters,
            isLP=isLP,
            dwr=True  # This removes community balance constraints
        )
        
        self.model = self.lem.model
        # Disable output for subproblems
        self.model.hideOutput()
        # Disable output for subproblems
        self.model.hideOutput()
        
    def solve_pricing(self, dual_elec: Dict[int, float], 
                      dual_heat: Dict[int, float], 
                      dual_hydro: Dict[int, float],
                      dual_convexity: float) -> Tuple[float, Dict]:
        """
        Solve pricing problem with modified objective based on dual prices
        
        Args:
            dual_elec: Dual prices for electricity community balance constraints
            dual_heat: Dual prices for heat community balance constraints
            dual_hydro: Dual prices for hydrogen community balance constraints
            dual_convexity: Dual price for convexity constraint (sum lambda = 1)

        Returns:
            tuple: (reduced_cost, solution_dict)
        """
        # Free transform to allow objective modification
        self.model.freeTransform()
        
        # Get original objective value (we'll calculate cost from solution)
        # Build modified objective with dual prices
        new_obj = 0
        
        u = self.player
        
        # Original grid costs
        for t in self.time_periods:
            # Electricity grid costs
            if (u, t) in self.lem.i_E_gri:
                pi_import = self.parameters.get(f'pi_E_gri_import_{t}', 0)
                new_obj += pi_import * self.lem.i_E_gri[u, t]
            if (u, t) in self.lem.e_E_gri:
                pi_export = self.parameters.get(f'pi_E_gri_export_{t}', 0)
                new_obj -= pi_export * self.lem.e_E_gri[u, t]
            
            # Heat grid costs
            if (u, t) in self.lem.i_H_gri:
                pi_import = self.parameters.get(f'pi_H_gri_import_{t}', 0)
                new_obj += pi_import * self.lem.i_H_gri[u, t]
            if (u, t) in self.lem.e_H_gri:
                pi_export = self.parameters.get(f'pi_H_gri_export_{t}', 0)
                new_obj -= pi_export * self.lem.e_H_gri[u, t]
            
            # Hydrogen grid costs
            if (u, t) in self.lem.i_G_gri:
                pi_import = self.parameters.get(f'pi_G_gri_import_{t}', 0)
                new_obj += pi_import * self.lem.i_G_gri[u, t]
            if (u, t) in self.lem.e_G_gri:
                pi_export = self.parameters.get(f'pi_G_gri_export_{t}', 0)
                new_obj -= pi_export * self.lem.e_G_gri[u, t]
            
            # Production costs
            if (u, 'res', t) in self.lem.p:
                c_res = self.parameters.get(f'c_res_{u}', 0)
                new_obj += c_res * self.lem.p[u, 'res', t]
            if (u, 'hp', t) in self.lem.p:
                c_hp = self.parameters.get(f'c_hp_{u}', 0)
                new_obj += c_hp * self.lem.p[u, 'hp', t]
            if (u, 'els', t) in self.lem.p:
                c_els = self.parameters.get(f'c_els_{u}', 0)
                new_obj += c_els * self.lem.p[u, 'els', t]
            
            # Startup costs
            if (u, t) in self.lem.z_su:
                c_su = self.parameters.get(f'c_su_{u}', 0)
                new_obj += c_su * self.lem.z_su[u, t]
            
            # Community trading with dual prices
            # For reduced cost: original_cost - dual_price * coefficient
            # Electricity
            if (u, t) in self.lem.i_E_com:
                # Import from community: coefficient is +1 in balance
                new_obj -= dual_elec[t] * self.lem.i_E_com[u, t]
            if (u, t) in self.lem.e_E_com:
                # Export to community: coefficient is -1 in balance
                new_obj += dual_elec[t] * self.lem.e_E_com[u, t]
            
            # Heat
            if (u, t) in self.lem.i_H_com:
                new_obj -= dual_heat[t] * self.lem.i_H_com[u, t]
            if (u, t) in self.lem.e_H_com:
                new_obj += dual_heat[t] * self.lem.e_H_com[u, t]
            
            # Hydrogen
            if (u, t) in self.lem.i_G_com:
                new_obj -= dual_hydro[t] * self.lem.i_G_com[u, t]
            if (u, t) in self.lem.e_G_com:
                new_obj += dual_hydro[t] * self.lem.e_G_com[u, t]
        
        # Set modified objective
        self.model.setObjective(new_obj, "minimize")
        
        # Solve
        self.model.optimize()
        status = self.model.getStatus()

        if status == "optimal":
            reduced_cost = self.model.getObjVal() - dual_convexity
            # Extract solution
            _, results = solve_and_extract_results(self.model)
            
            return reduced_cost, results
        else:
            return float('inf'), None


class LEMPricer(Pricer):
    """
    Pricer for Local Energy Market column generation
    """
    def __init__(self, master_model, subproblems: Dict[str, PlayerSubproblem], 
                 time_periods: List[int], players: List[str]):
        """
        Initialize pricer
        
        Args:
            master_model: Master problem SCIP model
            subproblems: Dictionary of player subproblems
            time_periods: List of time periods
            players: List of player IDs
        """
        super().__init__()
        self.subproblems = subproblems
        self.time_periods = time_periods
        self.players = players
        self.iteration = 0
        self.farkas_iteration = 0
        
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
        # Solve pricing problem for each player
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
                            # is_active = self.model.isActive(t_conv_cons)
                            
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
        else:
            print('stop here')
        columns_added = 0
        min_reduced_cost = float('inf')
        print(f"iteration {self.iteration}, dual_elec: {dual_elec}")
        print(f"iteration {self.iteration}, dual_heat: {dual_heat}")
        print(f"iteration {self.iteration}, dual_hydro: {dual_hydro}")
        print(f"iteration {self.iteration}, dual_convexity: {dual_convexity}")
        for player in self.players:
            reduced_cost, solution = self.subproblems[player].solve_pricing(
                dual_elec, dual_heat, dual_hydro, dual_convexity[player]
            )
            
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
            pricedVar = True
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


class MasterProblem:
    """
    Restricted Master Problem for column generation
    Contains only community balance constraints
    """
    def __init__(self, players: List[str], time_periods: List[int]):
        """
        Initialize master problem
        
        Args:
            players: List of player IDs
            time_periods: List of time periods
        """
        self.players = players
        self.time_periods = time_periods
        self.model = Model("RMP_LocalEnergyMarket")
        self.model.data = {}
        # Storage for variables and constraints
        self.model.data['vars'] = {}  # {(player, col_idx): {'var': var, 'solution': dict}}
        
        # Data dictionary for storing constraints (similar to LocalEnergyMarket)
        self.model.data['cons'] = {
            'community_elec_balance': {},
            'community_heat_balance': {},
            'community_hydro_balance': {},
            'convexity': {}
        }
    
    def _create_master_constraints(self):
        """
        Create master problem constraints:
        1. Community balance constraints (linking constraints)
        2. Convexity constraints (one per player)
                """
        # Convexity constraints: sum of lambdas = 1 for each player
        for u in self.players:
            initial_var = self.model.data["vars"][u, 0]["var"]
            cons = self.model.addCons(
                quicksum([initial_var]) - 1 == 0, 
                name=f"convexity_{u}"
            )
            self.model.data['cons']['convexity'][u] = cons
        
        # Community balance constraints (no slack variables)
        # Farkas pricing will ensure feasibility
        for t in self.time_periods:
            elec_expr, heat_expr, hydro_expr = 0,0,0
            for u in self.players:
                var = self.model.data["vars"][u, 0]["var"]
                solution = self.model.data["vars"][u, 0]["solution"]
                e_E_com_val = solution.get('e_E_com', {}).get((u, t), 0)
                i_E_com_val = solution.get('i_E_com', {}).get((u, t), 0)
                elec_expr += var * (i_E_com_val - e_E_com_val)
                e_H_com_val = solution.get('e_H_com', {}).get((u, t), 0)
                i_H_com_val = solution.get('i_H_com', {}).get((u, t), 0)
                heat_expr += var * (i_H_com_val - e_H_com_val)
                e_G_com_val = solution.get('e_G_com', {}).get((u, t), 0)
                i_G_com_val = solution.get('i_G_com', {}).get((u, t), 0)
                hydro_expr += var * (i_G_com_val - e_G_com_val)
            # Electricity balance: sum(i_E_com - e_E_com) = 0
            cons = self.model.addCons(
                elec_expr == 0,
                name=f"community_elec_balance_{t}"
            )
            self.model.data['cons']['community_elec_balance'][t] = cons
            
            # Heat balance: sum(i_H_com - e_H_com) = 0
            cons = self.model.addCons(
                heat_expr == 0,
                name=f"community_heat_balance_{t}"
            )
            self.model.data['cons']['community_heat_balance'][t] = cons
            
            # Hydrogen balance: sum(i_G_com - e_G_com) = 0
            cons = self.model.addCons(
                hydro_expr == 0,
                name=f"community_hydro_balance_{t}"
            )
            self.model.data['cons']['community_hydro_balance'][t] = cons
    
    def _add_initial_columns(self, subproblems: Dict[str, PlayerSubproblem]):
        """
        Generate initial columns by solving each subproblem independently
        
        Args:
            subproblems: Dictionary of player subproblems
        """
        print("\n=== Generating Initial Columns ===")
        
        # Zero dual prices for initial solve
        zero_duals_elec = {t: 0.0 for t in self.time_periods}
        zero_duals_heat = {t: 0.0 for t in self.time_periods}
        zero_duals_hydro = {t: 0.0 for t in self.time_periods}
        zero_duals_convexity = {player: 0.0 for player in self.players}
        for player in self.players:
            print(f"Generating initial column for player {player}...")
            
            # Solve subproblem with zero dual prices
            reduced_cost, solution = subproblems[player].solve_pricing(
                zero_duals_elec, zero_duals_heat, zero_duals_hydro, zero_duals_convexity[player]
            )
            
            if solution is None:
                raise Exception(f"  WARNING: Could not generate initial column for {player}")
            col_idx = 0
            var_name = f"lambda_{player}_{col_idx}"
        
            # Calculate column cost (original objective value)
            cost = self._calculate_column_cost(player, solution, subproblems[player].parameters)
            # cost = 10**4
            # Create variable
            new_var = self.model.addVar(
                name=var_name,
                vtype="C",
                lb=0.0,
                obj=cost
            )
        
            # Store variable and solution
            self.model.data['vars'][player, col_idx] = {
                'var': new_var,
                'solution': solution
            }
            print(f" {player}: Initial column added (cost={cost:.4f})")

    def _calculate_column_cost(self, player: str, solution: Dict, params: Dict) -> float:
        """Calculate cost for a column"""
        cost = 0.0
        
        for t in self.time_periods:
            # Grid costs
            e_E_gri = solution.get('e_E_gri', {}).get((player, t), 0)
            i_E_gri = solution.get('i_E_gri', {}).get((player, t), 0)
            cost += i_E_gri * params.get(f'pi_E_gri_import_{t}', 0)
            cost -= e_E_gri * params.get(f'pi_E_gri_export_{t}', 0)
            
            e_H_gri = solution.get('e_H_gri', {}).get((player, t), 0)
            i_H_gri = solution.get('i_H_gri', {}).get((player, t), 0)
            cost += i_H_gri * params.get(f'pi_H_gri_import_{t}', 0)
            cost -= e_H_gri * params.get(f'pi_H_gri_export_{t}', 0)
            
            e_G_gri = solution.get('e_G_gri', {}).get((player, t), 0)
            i_G_gri = solution.get('i_G_gri', {}).get((player, t), 0)
            cost += i_G_gri * params.get(f'pi_G_gri_import_{t}', 0)
            cost -= e_G_gri * params.get(f'pi_G_gri_export_{t}', 0)
            
            # Production and startup costs
            p_res = solution.get('p', {}).get((player, 'res', t), 0)
            p_hp = solution.get('p', {}).get((player, 'hp', t), 0)
            p_els = solution.get('p', {}).get((player, 'els', t), 0)
            z_su = solution.get('z_su', {}).get((player, t), 0)
            
            cost += p_res * params.get(f'c_res_{player}', 0)
            cost += p_hp * params.get(f'c_hp_{player}', 0)
            cost += p_els * params.get(f'c_els_{player}', 0)
            cost += z_su * params.get(f'c_su_{player}', 0)
        
        return cost
    
    def solve(self):
        """Solve the restricted master problem"""
        self.model.optimize()
        return self.model.getStatus()
    
    def get_objective_value(self):
        """Get current objective value"""
        return self.model.getObjVal()
    
    def get_solution(self) -> Dict:
        """
        Extract solution from master problem
        Combines columns based on lambda values
        """
        solution = {}
        
        for (player, col_idx), col_data in self.model.data['vars'].items():
            lambda_val = self.model.getVal(col_data['var'])
            
            if lambda_val > 1e-6:  # Only consider active columns
                col_solution = col_data['solution']
                
                # Add weighted contribution of this column to solution
                for var_name, var_dict in col_solution.items():
                    if var_name not in solution:
                        solution[var_name] = {}
                    
                    for key, value in var_dict.items():
                        if key not in solution[var_name]:
                            solution[var_name][key] = 0.0
                        solution[var_name][key] += lambda_val * value
        
        return solution


class ColumnGenerationSolver:
    """
    Main column generation solver for Local Energy Market
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
        
        # No initial columns - will use pricerinitlp to generate them
        print("Note: Initial columns will be generated via pricerinitlp")
    
    def _create_zero_solution(self, player: str) -> Dict:
        """
        Create a zero pattern solution for a player (all variables = 0)
        This provides a feasible starting point for convexity constraints
        
        Args:
            player: Player ID
            
        Returns:
            Dictionary with all variables set to 0
        """
        solution = {}
        
        # Initialize all variable types with zeros
        var_types = [
            'e_E_gri', 'i_E_gri', 'e_E_com', 'i_E_com',
            'e_H_gri', 'i_H_gri', 'e_H_com', 'i_H_com',
            'e_G_gri', 'i_G_gri', 'e_G_com', 'i_G_com',
            's_E', 's_H', 's_G',  # Storage levels
            'z_on', 'z_su'  # Commitment and startup
        ]
        
        for var_type in var_types:
            solution[var_type] = {}
            for t in self.time_periods:
                solution[var_type][(player, t)] = 0.0
        
        # Production variables (player, technology, time)
        solution['p'] = {}
        for tech in ['res', 'hp', 'els']:
            for t in self.time_periods:
                solution['p'][(player, tech, t)] = 0.0
        
        return solution
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[str, Dict, float]:
        """
        Solve using column generation
        
        Args:
            max_iterations: Maximum number of CG iterations
            tolerance: Convergence tolerance for reduced cost
            
        Returns:
            tuple: (status, solution, objective_value)
        """
        print("\n" + "="*80)
        print("COLUMN GENERATION - STARTING ITERATIONS")
        print("="*80)
        
        # Disable presolving to allow pricer to work properly
        from pyscipopt import SCIP_PARAMSETTING
        self.master.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.master.model.setHeuristics(SCIP_PARAMSETTING.OFF)
        self.master.model.setSeparating(SCIP_PARAMSETTING.OFF)
        self.master.model.disablePropagation()
        # Generate zero pattern initial columns for each player
        # This satisfies convexity constraints (sum lambda = 1) 
        # Community balance will be achieved through pricing
        print("\n=== Generating Zero Pattern Initial Columns ===")
        self.master._add_initial_columns(self.subproblems)       
        # Create pricer
        pricer = LEMPricer(
            master_model=self.master.model,
            subproblems=self.subproblems,
            time_periods=self.time_periods,
            players=self.players        )
        # Include pricer in master problem
        self.master.model.includePricer(
            pricer,
            "LEMPricer",
            "Pricer for Local Energy Market column generation"
        )
        self.master._create_master_constraints()

        # Solve with column generation
        print("\nSolving master problem with pricing...")
        status = self.master.model.optimize()
        
        if status == "optimal":
            obj_val = self.master.get_objective_value()
            solution = self.master.get_solution()
            
            print("\n" + "="*80)
            print("COLUMN GENERATION - COMPLETED")
            print("="*80)
            print(f"Optimal objective value: {obj_val:.2f} EUR")
            print(f"Total columns generated: {len(self.master.master_vars)}")
            
            # Print column summary
            print("\nColumns per player:")
            for player in self.players:
                player_cols = [k for k in self.master.master_vars.keys() if k[0] == player]
                print(f"  {player}: {len(player_cols)} columns")
            
            # Extract and print convex hull prices (dual prices of community balance constraints)
            print("\n" + "="*80)
            print("CONVEX HULL PRICES (Shadow Prices of Community Balance)")
            print("="*80)
            
            chp_elec = {}
            chp_heat = {}
            chp_hydro = {}
            
            for t in self.time_periods:
                elec_cons = self.master.model.data['community_elec_balance'][t]
                heat_cons = self.master.model.data['community_heat_balance'][t]
                hydro_cons = self.master.model.data['community_hydro_balance'][t]
                
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


if __name__ == "__main__":
    # This will be called from compact.py's main
    print("Column Generation module loaded successfully")
    print("Use ColumnGenerationSolver class to solve LEM with Dantzig-Wolfe decomposition")
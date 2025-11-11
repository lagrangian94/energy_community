"""
Single-Node Single-Period Auction Optimization using PySCIPOpt
Author: Operations Research Implementation
"""

from pyscipopt import Model, quicksum
import numpy as np

def solve_auction_problem():
    """
    Solves the single-node single-period auction problem:
    max z = 6*d1 + 5*d2 + 4*d3 - 1*p1 - 2*p2 - 3*p3
    subject to constraints on production/consumption balance and bounds
    """
    
    # Create the model
    model = Model("Single_Node_Auction")
    
    # Decision variables: consumption (demand)
    d1 = model.addVar(vtype='C', lb=0, ub=8, name="d1")
    d2 = model.addVar(vtype='C', lb=0, ub=14, name="d2")
    d3 = model.addVar(vtype='C', lb=0, ub=5, name="d3")
    
    # Decision variables: production (supply)
    p1 = model.addVar(vtype='C', lb=0, ub=10, name="p1")
    p2 = model.addVar(vtype='C', lb=0, ub=10, name="p2")
    p3 = model.addVar(vtype='C', lb=0, ub=10, name="p3")
    
    # Set objective function (3.1a)
    # Maximize: z = 6*d1 + 5*d2 + 4*d3 - 1*p1 - 2*p2 - 3*p3
    model.setObjective(
        6*d1 + 5*d2 + 4*d3 - 1*p1 - 2*p2 - 3*p3, 
        sense="maximize"
    )
    
    # Add constraint (3.1b): Market clearing condition
    # p1 + p2 + p3 = d1 + d2 + d3
    model.addCons(
        p1 + p2 + p3 == d1 + d2 + d3, 
        name="market_clearing"
    )
    from pyscipopt import SCIP_PARAMSETTING
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    model.setSeparating(SCIP_PARAMSETTING.OFF)    
    # Solve the optimization problem
    model.optimize()
    
    # Check solution status
    if model.getStatus() == "optimal":
        print("="*60)
        print("OPTIMAL SOLUTION FOUND")
        print("="*60)
        
        # Print objective value
        print(f"\nObjective value (z): {model.getObjVal():.4f}")
        
        # Print optimal values of decision variables
        print("\n--- Consumer Consumption (Demand) ---")
        print(f"d1 = {model.getVal(d1):.4f}")
        print(f"d2 = {model.getVal(d2):.4f}")
        print(f"d3 = {model.getVal(d3):.4f}")
        print(f"Total demand: {model.getVal(d1) + model.getVal(d2) + model.getVal(d3):.4f}")
        
        print("\n--- Producer Production (Supply) ---")
        print(f"p1 = {model.getVal(p1):.4f}")
        print(f"p2 = {model.getVal(p2):.4f}")
        print(f"p3 = {model.getVal(p3):.4f}")
        print(f"Total supply: {model.getVal(p1) + model.getVal(p2) + model.getVal(p3):.4f}")
        
        # Get dual variable (λ) for market clearing constraint
        constraints = model.getConss()
        for cons in constraints:
            if cons.name == "market_clearing":
                dual_value = np.abs(model.getDualsolLinear(cons))
                print(f"\n--- Dual Variable ---")
                print(f"λ (market clearing price): {dual_value:.4f}")
        
        print("\n--- Economic Interpretation ---")
        print(f"Consumer surplus: {6*model.getVal(d1) + 5*model.getVal(d2) + 4*model.getVal(d3):.4f}")
        print(f"Producer cost: {1*model.getVal(p1) + 2*model.getVal(p2) + 3*model.getVal(p3):.4f}")
        print(f"producer1's profit: {(dual_value-1)*model.getVal(p1):.4f}")
        print(f"producer2's profit: {(dual_value-2)*model.getVal(p2):.4f}")
        print(f"producer3's profit: {(dual_value-3)*model.getVal(p3):.4f}")
        print(f"consumer1's profit: {(6-dual_value)*model.getVal(d1):.4f}")
        print(f"consumer2's profit: {(5-dual_value)*model.getVal(d2):.4f}")
        print(f"consumer3's profit: {(4-dual_value)*model.getVal(d3):.4f}")
        print(f"Social welfare: {model.getObjVal():.4f}")
        
    else:
        print(f"Optimization failed with status: {model.getStatus()}")
    
    return model

def solve_dual_problem():
    """
    Solves the dual problem:
    min 10*mu_1_p + 10*mu_2_p + 10*mu_3_p + 8*mu_1_d + 14*mu_2_d + 5*mu_3_d
    """
    
    # Define parameters
    c_1, c_2, c_3 = 1, 2, 3  # Production costs
    u_1, u_2, u_3 = 6, 5, 4  # Consumer utilities
    
    # Create the dual model
    model = Model("Dual_Auction")
    
    # Dual variables for production upper bounds (p_i <= 10)
    mu_1_p = model.addVar(vtype='C', lb=0, name="mu_1_p")
    mu_2_p = model.addVar(vtype='C', lb=0, name="mu_2_p")
    mu_3_p = model.addVar(vtype='C', lb=0, name="mu_3_p")
    
    # Dual variables for demand upper bounds (d_i <= upper_bound)
    mu_1_d = model.addVar(vtype='C', lb=0, name="mu_1_d")
    mu_2_d = model.addVar(vtype='C', lb=0, name="mu_2_d")
    mu_3_d = model.addVar(vtype='C', lb=0, name="mu_3_d")
    
    # Dual variable for market clearing constraint (unrestricted)
    lambda_var = model.addVar(vtype='C', lb=None, name="lambda")
    
    # Set objective function (minimize)
    model.setObjective(
        10*mu_1_p + 10*mu_2_p + 10*mu_3_p + 8*mu_1_d + 14*mu_2_d + 5*mu_3_d,
        sense="minimize"
    )
    
    # Add dual constraints for production variables
    model.addCons(mu_1_p - lambda_var >= -c_1, name="dual_p1")
    model.addCons(mu_2_p - lambda_var >= -c_2, name="dual_p2")
    model.addCons(mu_3_p - lambda_var >= -c_3, name="dual_p3")
    
    # Add dual constraints for demand variables
    model.addCons(mu_1_d + lambda_var >= u_1, name="dual_d1")
    model.addCons(mu_2_d + lambda_var >= u_2, name="dual_d2")
    model.addCons(mu_3_d + lambda_var >= u_3, name="dual_d3")
    
    # Solve the optimization problem
    model.optimize()
    
    # Check solution status
    if model.getStatus() == "optimal":
        print("="*60)
        print("DUAL PROBLEM - OPTIMAL SOLUTION FOUND")
        print("="*60)
        
        # Print objective value
        print(f"\nDual objective value: {model.getObjVal():.4f}")
        
        # Print optimal values of dual variables
        print("\n--- Dual Variables for Production Bounds ---")
        print(f"mu_1_p (for p1 <= 10): {model.getVal(mu_1_p):.4f}")
        print(f"mu_2_p (for p2 <= 10): {model.getVal(mu_2_p):.4f}")
        print(f"mu_3_p (for p3 <= 10): {model.getVal(mu_3_p):.4f}")
        
        print("\n--- Dual Variables for Demand Bounds ---")
        print(f"mu_1_d (for d1 <= 8): {model.getVal(mu_1_d):.4f}")
        print(f"mu_2_d (for d2 <= 14): {model.getVal(mu_2_d):.4f}")
        print(f"mu_3_d (for d3 <= 5): {model.getVal(mu_3_d):.4f}")
        
        print("\n--- Market Clearing Dual Variable ---")
        print(f"lambda (market price): {model.getVal(lambda_var):.4f}")
        
        print("\n--- Economic Interpretation ---")
        lambda_val = model.getVal(lambda_var)
        print(f"Market clearing price: {lambda_val:.4f}")
        
        # Check complementary slackness conditions
        print("\n--- Complementary Slackness Analysis ---")
        
        # For production bounds
        if model.getVal(mu_1_p) > 1e-6:
            print(f"mu_1_p > 0: p1 should be at its upper bound (10)")
        if model.getVal(mu_2_p) > 1e-6:
            print(f"mu_2_p > 0: p2 should be at its upper bound (10)")
        if model.getVal(mu_3_p) > 1e-6:
            print(f"mu_3_p > 0: p3 should be at its upper bound (10)")
            
        # For demand bounds
        if model.getVal(mu_1_d) > 1e-6:
            print(f"mu_1_d > 0: d1 should be at its upper bound (8)")
        if model.getVal(mu_2_d) > 1e-6:
            print(f"mu_2_d > 0: d2 should be at its upper bound (14)")
        if model.getVal(mu_3_d) > 1e-6:
            print(f"mu_3_d > 0: d3 should be at its upper bound (5)")
            
    else:
        print(f"Optimization failed with status: {model.getStatus()}")
    
    return model
if __name__ == "__main__":
    model = solve_auction_problem()
    dual_model = solve_dual_problem()
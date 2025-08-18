from pyscipopt import Model, quicksum
import numpy as np

def solve_energy_pricing_problem(player, time_periods, params, subprob_obj, current_node=None):
    """
    Solve the pricing problem for a single player
    
    Args:
        player: Player ID
        time_periods: List of time periods
        params: Parameters dictionary
        subprob_obj: Dictionary of dual values from RMP
        current_node: Current branch-and-bound node (for warm start if needed)
    
    Returns:
        tuple: (min_redcost, pattern, objval)
            - min_redcost: Reduced cost (objective value with duals)
            - pattern: Solution pattern (variable values)
            - objval: Original objective value (without duals)
    """
    
    # Create pricing model directly (not inheriting from LocalEnergyMarket)
    model = Model(f"PricingProblem_{player}")
    model.hideOutput()
    
    # Get player sets
    players_with_renewables = params.get('players_with_renewables', [])
    players_with_electrolyzers = params.get('players_with_electrolyzers', [])
    players_with_heatpumps = params.get('players_with_heatpumps', [])
    players_with_elec_storage = params.get('players_with_elec_storage', [])
    players_with_hydro_storage = params.get('players_with_hydro_storage', [])
    players_with_heat_storage = params.get('players_with_heat_storage', [])
    players_with_nfl_elec_demand = params.get('players_with_nfl_elec_demand', [])
    players_with_nfl_hydro_demand = params.get('players_with_nfl_hydro_demand', [])
    players_with_nfl_heat_demand = params.get('players_with_nfl_heat_demand', [])
    players_with_fl_elec_demand = params.get('players_with_fl_elec_demand', [])
    
    # Create variables for single player
    e_E_gri = {}
    i_E_gri = {}
    e_E_com = {}
    i_E_com = {}
    e_H_gri = {}
    i_H_gri = {}
    e_H_com = {}
    i_H_com = {}
    e_G_gri = {}
    i_G_gri = {}
    e_G_com = {}
    i_G_com = {}
    
    p = {}
    fl_d = {}
    nfl_d = {}
    els_d = {}
    
    b_dis_E = {}
    b_ch_E = {}
    s_E = {}
    b_dis_G = {}
    b_ch_G = {}
    s_G = {}
    b_dis_H = {}
    b_ch_H = {}
    s_H = {}
    
    z_su = {}
    z_on = {}
    z_off = {}
    z_sb = {}
    
    # Create variables for each time period
    for t in time_periods:
        # Grid/community trading variables - only if player has the capability
        if player in players_with_renewables or player in players_with_elec_storage or player in players_with_nfl_elec_demand or player in players_with_fl_elec_demand:
            e_E_gri[t] = model.addVar(vtype="C", name=f"e_E_gri_{t}", lb=0, ub=params.get(f'e_E_cap_{player}_{t}', 1000))
            i_E_gri[t] = model.addVar(vtype="C", name=f"i_E_gri_{t}", lb=0, ub=params.get(f'i_E_cap_{player}_{t}', 1000))
            e_E_com[t] = model.addVar(vtype="C", name=f"e_E_com_{t}", lb=0, ub=1000)
            i_E_com[t] = model.addVar(vtype="C", name=f"i_E_com_{t}", lb=0, ub=1000)
        
        if player in players_with_heatpumps or player in players_with_heat_storage or player in players_with_nfl_heat_demand:
            e_H_gri[t] = model.addVar(vtype="C", name=f"e_H_gri_{t}", lb=0, ub=params.get(f'e_H_cap_{player}_{t}', 500))
            i_H_gri[t] = model.addVar(vtype="C", name=f"i_H_gri_{t}", lb=0, ub=params.get(f'i_H_cap_{player}_{t}', 500))
            e_H_com[t] = model.addVar(vtype="C", name=f"e_H_com_{t}", lb=0, ub=500)
            i_H_com[t] = model.addVar(vtype="C", name=f"i_H_com_{t}", lb=0, ub=500)
        
        if player in players_with_electrolyzers or player in players_with_hydro_storage or player in players_with_nfl_hydro_demand:
            e_G_gri[t] = model.addVar(vtype="C", name=f"e_G_gri_{t}", lb=0, ub=params.get(f'e_G_cap_{player}_{t}', 100))
            i_G_gri[t] = model.addVar(vtype="C", name=f"i_G_gri_{t}", lb=0, ub=params.get(f'i_G_cap_{player}_{t}', 100))
            e_G_com[t] = model.addVar(vtype="C", name=f"e_G_com_{t}", lb=0, ub=100)
            i_G_com[t] = model.addVar(vtype="C", name=f"i_G_com_{t}", lb=0, ub=100)
        
        # Production variables
        if player in players_with_renewables:
            renewable_cap = params.get(f'renewable_cap_{player}_{t}', 200)
            p['res', t] = model.addVar(vtype="C", name=f"p_res_{t}", lb=0, ub=renewable_cap)
        
        if player in players_with_heatpumps:
            hp_cap = params.get(f'hp_cap_{player}', 100)
            p['hp', t] = model.addVar(vtype="C", name=f"p_hp_{t}", lb=0, ub=hp_cap)
        
        if player in players_with_electrolyzers:
            els_cap = params.get(f'els_cap_{player}', 1)
            p['els', t] = model.addVar(vtype="C", name=f"p_els_{t}", lb=0)
            z_su[t] = model.addVar(vtype="B", name=f"z_su_{t}")
            z_on[t] = model.addVar(vtype="B", name=f"z_on_{t}")
            z_off[t] = model.addVar(vtype="B", name=f"z_off_{t}")
            z_sb[t] = model.addVar(vtype="B", name=f"z_sb_{t}")
            els_d[t] = model.addVar(vtype="C", name=f"els_d_{t}", lb=0, ub=els_cap)
        
        # Demand variables
        if player in players_with_nfl_elec_demand:
            nfl_elec_demand = params.get(f'd_E_nfl_{player}_{t}', 0)
            nfl_d['elec', t] = model.addVar(vtype="C", name=f"nfl_d_elec_{t}", 
                                           lb=nfl_elec_demand, ub=nfl_elec_demand)
        
        if player in players_with_nfl_hydro_demand:
            nfl_hydro_demand = params.get(f'd_G_nfl_{player}_{t}', 0)
            nfl_d['hydro', t] = model.addVar(vtype="C", name=f"nfl_d_hydro_{t}", 
                                            lb=nfl_hydro_demand, ub=nfl_hydro_demand)
        
        if player in players_with_nfl_heat_demand:
            nfl_heat_demand = params.get(f'd_H_nfl_{player}_{t}', 0)
            nfl_d['heat', t] = model.addVar(vtype="C", name=f"nfl_d_heat_{t}", 
                                           lb=nfl_heat_demand, ub=nfl_heat_demand)
        
        # Flexible demand
        if player in players_with_fl_elec_demand:
            fl_elec_cap = 1 if player in players_with_electrolyzers else 0.1
            fl_d['elec', t] = model.addVar(vtype="C", name=f"fl_d_elec_{t}", lb=0, ub=fl_elec_cap)
        
        # Storage variables
        if player in players_with_elec_storage:
            storage_power = params.get('storage_power', 50)
            storage_capacity = params.get('storage_capacity', 100)
            b_dis_E[t] = model.addVar(vtype="C", name=f"b_dis_E_{t}", lb=0, ub=storage_power)
            b_ch_E[t] = model.addVar(vtype="C", name=f"b_ch_E_{t}", lb=0, ub=storage_power)
            s_E[t] = model.addVar(vtype="C", name=f"s_E_{t}", lb=0, ub=storage_capacity)
        
        if player in players_with_hydro_storage:
            storage_power = params.get('storage_power', 50)
            storage_capacity = params.get('storage_capacity', 100)
            b_dis_G[t] = model.addVar(vtype="C", name=f"b_dis_G_{t}", lb=0, ub=storage_power)
            b_ch_G[t] = model.addVar(vtype="C", name=f"b_ch_G_{t}", lb=0, ub=storage_power)
            s_G[t] = model.addVar(vtype="C", name=f"s_G_{t}", lb=0, ub=storage_capacity)
        
        if player in players_with_heat_storage:
            storage_power = params.get('storage_power', 50)
            storage_capacity = params.get('storage_capacity', 100)
            b_dis_H[t] = model.addVar(vtype="C", name=f"b_dis_H_{t}", lb=0, ub=storage_power)
            b_ch_H[t] = model.addVar(vtype="C", name=f"b_ch_H_{t}", lb=0, ub=storage_power)
            s_H[t] = model.addVar(vtype="C", name=f"s_H_{t}", lb=0, ub=storage_capacity)
    
    # Create objective with dual values
    obj_terms = []
    
    # Original costs + dual contributions
    for t in time_periods:
        # Production costs
        if ('res', t) in p:
            c_res = params.get(f'c_res_{player}', 0)
            obj_terms.append(c_res * p['res', t])
        
        if ('hp', t) in p:
            c_hp = params.get(f'c_hp_{player}', 0)
            obj_terms.append(c_hp * p['hp', t])
        
        if ('els', t) in p:
            c_els = params.get(f'c_els_{player}', 0)
            obj_terms.append(c_els * p['els', t])
        
        if t in z_su:
            c_su = params.get(f'c_su_{player}', 0)
            obj_terms.append(c_su * z_su[t])
        
        # Grid costs
        pi_E_gri_import = params.get(f'pi_E_gri_import_{t}', 0)
        pi_E_gri_export = params.get(f'pi_E_gri_export_{t}', 0)
        pi_H_gri_import = params.get(f'pi_H_gri_import_{t}', 0)
        pi_H_gri_export = params.get(f'pi_H_gri_export_{t}', 0)
        pi_G_gri_import = params.get(f'pi_G_gri_import_{t}', 0)
        pi_G_gri_export = params.get(f'pi_G_gri_export_{t}', 0)
        
        if t in i_E_gri:
            obj_terms.append(pi_E_gri_import * i_E_gri[t])
        if t in e_E_gri:
            obj_terms.append(-pi_E_gri_export * e_E_gri[t])
        if t in i_H_gri:
            obj_terms.append(pi_H_gri_import * i_H_gri[t])
        if t in e_H_gri:
            obj_terms.append(-pi_H_gri_export * e_H_gri[t])
        if t in i_G_gri:
            obj_terms.append(pi_G_gri_import * i_G_gri[t])
        if t in e_G_gri:
            obj_terms.append(-pi_G_gri_export * e_G_gri[t])
        
        # Storage costs
        c_sto = params.get('c_sto', 0.01)
        nu_ch = params.get('nu_ch', 0.9)
        nu_dis = params.get('nu_dis', 0.9)
        
        if t in b_ch_E:
            obj_terms.append(c_sto * nu_ch * b_ch_E[t])
        if t in b_dis_E:
            obj_terms.append(c_sto * (1/nu_dis) * b_dis_E[t])
        if t in b_ch_G:
            obj_terms.append(c_sto * nu_ch * b_ch_G[t])
        if t in b_dis_G:
            obj_terms.append(c_sto * (1/nu_dis) * b_dis_G[t])
        if t in b_ch_H:
            obj_terms.append(c_sto * nu_ch * b_ch_H[t])
        if t in b_dis_H:
            obj_terms.append(c_sto * (1/nu_dis) * b_dis_H[t])
    
    # Add dual contributions to objective
    dual_obj_terms = []
    
    # Community balance dual contributions
    for t in time_periods:
        # Electricity community balance dual
        if player in players_with_renewables or player in players_with_elec_storage or player in players_with_nfl_elec_demand or player in players_with_fl_elec_demand:
            pi_com_elec_export = subprob_obj.get(f"e_E_com_{t}",0)
            pi_com_elec_import = subprob_obj.get(f"i_E_com_{t}",0)
            dual_obj_terms.append(pi_com_elec_import * i_E_com[t])
            dual_obj_terms.append(pi_com_elec_export * e_E_com[t])
        
        # Heat community balance dual
        if player in players_with_heatpumps or player in players_with_heat_storage or player in players_with_nfl_heat_demand:
            pi_com_heat_export = subprob_obj.get(f"e_H_com_{t}",0)
            pi_com_heat_import = subprob_obj.get(f"i_H_com_{t}",0)
            dual_obj_terms.append(pi_com_heat_import * i_H_com[t])
            dual_obj_terms.append(pi_com_heat_export * e_H_com[t])
        
        # Hydrogen community balance dual (note: opposite sign)
        if player in players_with_electrolyzers or player in players_with_hydro_storage or player in players_with_nfl_hydro_demand:
            pi_com_hydro_export = subprob_obj.get(f"e_G_com_{t}",0)
            pi_com_hydro_import = subprob_obj.get(f"i_G_com_{t}",0)
            dual_obj_terms.append(pi_com_hydro_import * i_G_com[t])
            dual_obj_terms.append(pi_com_hydro_export * e_G_com[t])
        
        
    
    # Convexity constraint dual
    if f"cons_convexity_{player}" in subprob_obj:
        dual_obj_terms.append(subprob_obj[f"cons_convexity_{player}"])
    
    # Set objective
    model.setObjective(quicksum(obj_terms + dual_obj_terms), "minimize")
    
    # Add constraints
    # Electricity balance
    elec_bal = []
    for t in time_periods:
        if t in i_E_gri or t in e_E_gri:
            lhs = i_E_gri.get(t, 0) - e_E_gri.get(t, 0) + i_E_com.get(t, 0) - e_E_com.get(t, 0)
            lhs += p.get(('res', t), 0)
            lhs += b_dis_E.get(t, 0) - b_ch_E.get(t, 0)
            rhs = nfl_d.get(('elec', t), 0) + fl_d.get(('elec', t), 0)
            if type(lhs) != int or type(rhs) != int:
                cons = model.addCons(lhs == rhs, name=f"elec_balance_{t}")
                elec_bal.append(cons)
    
    # Heat balance
    for t in time_periods:
        if t in i_H_gri or t in e_H_gri:
            lhs = i_H_gri.get(t, 0) - e_H_gri.get(t, 0) + i_H_com.get(t, 0) - e_H_com.get(t, 0)
            lhs += p.get(('hp', t), 0)
            lhs += b_dis_H.get(t, 0) - b_ch_H.get(t, 0)
            rhs = nfl_d.get(('heat', t), 0)
            if type(lhs) != int or type(rhs) != int:
                model.addCons(lhs == rhs, name=f"heat_balance_{t}")
    
    # Hydrogen balance
    for t in time_periods:
        if t in i_G_gri or t in e_G_gri:
            lhs = i_G_gri.get(t, 0) - e_G_gri.get(t, 0) + i_G_com.get(t, 0) - e_G_com.get(t, 0)
            lhs += p.get(('els', t), 0)
            lhs += b_dis_G.get(t, 0) - b_ch_G.get(t, 0)
            rhs = nfl_d.get(('hydro', t), 0)
            if type(lhs) != int or type(rhs) != int:
                model.addCons(lhs == rhs, name=f"hydro_balance_{t}")
    
    # Heat pump coupling
    if player in players_with_heatpumps:
        for t in time_periods:
            if ('hp', t) in p and ('elec', t) in fl_d:
                nu_COP = params.get(f'nu_COP_{player}', 3.0)
                model.addCons(nu_COP * fl_d['elec', t] == p['hp', t], name=f"hp_coupling_{t}")
    
    # Electrolyzer constraints
    if player in players_with_electrolyzers:
        for t in time_periods:
            if ('els', t) in p and t in els_d:
                # Production curves
                phi1_1 = params.get(f'phi1_1_{player}', 21.12266316)
                phi0_1 = params.get(f'phi0_1_{player}', -0.37924094)
                phi1_2 = params.get(f'phi1_2_{player}', 16.66883134)
                phi0_2 = params.get(f'phi0_2_{player}', 0.87814262)
                
                model.addCons(p['els', t] <= phi1_1 * els_d[t] + phi0_1 * z_on[t], 
                            name=f"els_production_1_{t}")
                model.addCons(p['els', t] <= phi1_2 * els_d[t] + phi0_2 * z_on[t], 
                            name=f"els_production_2_{t}")
                
                # Power limits
                els_cap = params.get(f'els_cap_{player}', 1)
                C_min = params.get(f'C_min_{player}', 0.15)
                C_sb = params.get(f'C_sb_{player}', 0.01)
                
                model.addCons(els_d[t] >= C_min * els_cap * z_on[t], name=f"els_min_power_{t}")
                model.addCons(els_d[t] <= els_cap * z_on[t], name=f"els_max_power_{t}")
                
                # Power consumption coupling
                if ('elec', t) in fl_d:
                    model.addCons(fl_d['elec', t] == els_d[t] + C_sb * els_cap * z_sb[t],
                                name=f"els_power_coupling_{t}")
                
                # State constraints
                model.addCons(z_on[t] + z_off[t] + z_sb[t] == 1, name=f"els_state_{t}")
                
                # Startup logic
                if t == 6:  # Initial state
                    model.addCons(z_off[t] == 1, name=f"els_initial_off_{t}")
                    model.addCons(z_su[t] == 0, name=f"els_initial_su_{t}")
                elif t == 0:  # Wrap around
                    model.addCons(z_su[t] >= z_on[t] - z_on[23] - z_sb[t], name=f"els_startup_{t}")
                    model.addCons(z_off[23] + z_sb[t] <= 1, name=f"els_no_off_to_sb_{t}")
                else:
                    model.addCons(z_su[t] >= z_on[t] - z_on[t-1] - z_sb[t], name=f"els_startup_{t}")
                    model.addCons(z_off[t-1] + z_sb[t] <= 1, name=f"els_no_off_to_sb_{t}")
                
        # Minimum down time
        min_down = params.get('min_down_time', 1)
        for t in [tau for tau in time_periods if tau != 6]:
            down_time_idx = [tau for tau in range(t, t + min_down)]
            down_time_idx = [tau if tau < 24 else tau - 24 for tau in down_time_idx]
            
            if t != 0:
                for n in down_time_idx:
                    if n in z_off:
                        model.addCons(z_off[t] - z_off[t-1] <= z_off[n], 
                                    name=f"min_downtime_{t}_{n}")
            else:
                for n in down_time_idx:
                    if n in z_off:
                        model.addCons(z_off[t] - z_off[23] <= z_off[n], 
                                    name=f"min_downtime_{t}_{n}")
    
    # Storage SOC constraints
    if player in players_with_elec_storage:
        nu_ch = params.get('nu_ch', 0.9)
        nu_dis = params.get('nu_dis', 0.9)
        initial_soc = params.get('initial_soc', 1)
        
        if 6 in s_E:
            model.addCons(s_E[6] == initial_soc, name="initial_soc_E")
        
        for t in range(1, 24):
            if t in s_E and (t-1) in s_E:
                model.addCons(s_E[t] == s_E[t-1] + nu_ch * b_ch_E[t] - (1/nu_dis) * b_dis_E[t],
                            name=f"soc_E_{t}")
        
        if 23 in s_E and 0 in s_E:
            model.addCons(s_E[0] == s_E[23] + nu_ch * b_ch_E[0] - (1/nu_dis) * b_dis_E[0],
                        name="soc_E_wrap")
    
    if player in players_with_hydro_storage:
        nu_ch = params.get('nu_ch', 0.9)
        nu_dis = params.get('nu_dis', 0.9)
        initial_soc = 0.0
        
        if 6 in s_G:
            model.addCons(s_G[6] == initial_soc, name="initial_soc_G")
        
        for t in range(1, 24):
            if t in s_G and (t-1) in s_G:
                model.addCons(s_G[t] == s_G[t-1] + nu_ch * b_ch_G[t] - (1/nu_dis) * b_dis_G[t],
                            name=f"soc_G_{t}")
        
        if 23 in s_G and 0 in s_G:
            model.addCons(s_G[0] == s_G[23] + nu_ch * b_ch_G[0] - (1/nu_dis) * b_dis_G[0],
                        name="soc_G_wrap")
    
    if player in players_with_heat_storage:
        nu_ch = params.get('nu_ch', 0.9)
        nu_dis = params.get('nu_dis', 0.9)
        initial_soc = params.get('initial_soc', 50)
        
        if 6 in s_H:
            model.addCons(s_H[6] == initial_soc, name="initial_soc_H")
        
        for t in range(1, 24):
            if t in s_H and (t-1) in s_H:
                model.addCons(s_H[t] == s_H[t-1] + nu_ch * b_ch_H[t] - (1/nu_dis) * b_dis_H[t],
                            name=f"soc_H_{t}")
        
        if 23 in s_H and 0 in s_H:
            model.addCons(s_H[0] == s_H[23] + nu_ch * b_ch_H[0] - (1/nu_dis) * b_dis_H[0],
                        name="soc_H_wrap")
    
    # Solve
    model.optimize()
    status = model.getStatus()
    
    if status not in ["optimal", "gaplimit"]:
        print(f"Warning: Pricing problem for player {player} not optimal. Status: {status}")
        return float('inf'), {}, float('inf')
    
    # Extract solution
    min_redcost = model.getObjVal()
    if player == "u2":
        print(min_redcost)
    # Build pattern dictionary
    pattern = {}
    
    for t in time_periods:
        # Grid/community variables
        for var_dict, var_name in [(e_E_gri, 'e_E_gri'), (i_E_gri, 'i_E_gri'),
                                   (e_E_com, 'e_E_com'), (i_E_com, 'i_E_com'),
                                   (e_H_gri, 'e_H_gri'), (i_H_gri, 'i_H_gri'),
                                   (e_H_com, 'e_H_com'), (i_H_com, 'i_H_com'),
                                   (e_G_gri, 'e_G_gri'), (i_G_gri, 'i_G_gri'),
                                   (e_G_com, 'e_G_com'), (i_G_com, 'i_G_com')]:
            if t in var_dict:
                if var_name not in pattern:
                    pattern[var_name] = {}
                pattern[var_name][t] = model.getVal(var_dict[t])
        
        # Production variables
        if ('res', t) in p:
            if 'p_res' not in pattern:
                pattern['p_res'] = {}
            pattern['p_res'][t] = model.getVal(p['res', t])
        
        if ('hp', t) in p:
            if 'p_hp' not in pattern:
                pattern['p_hp'] = {}
            pattern['p_hp'][t] = model.getVal(p['hp', t])
        
        if ('els', t) in p:
            if 'p_els' not in pattern:
                pattern['p_els'] = {}
            pattern['p_els'][t] = model.getVal(p['els', t])
        
        # Binary variables
        if t in z_su:
            if 'z_su' not in pattern:
                pattern['z_su'] = {}
            pattern['z_su'][t] = model.getVal(z_su[t])
        
        if t in z_on:
            if 'z_on' not in pattern:
                pattern['z_on'] = {}
            pattern['z_on'][t] = model.getVal(z_on[t])
        
        if t in z_off:
            if 'z_off' not in pattern:
                pattern['z_off'] = {}
            pattern['z_off'][t] = model.getVal(z_off[t])
        
        if t in z_sb:
            if 'z_sb' not in pattern:
                pattern['z_sb'] = {}
            pattern['z_sb'][t] = model.getVal(z_sb[t])
        
        # Storage variables
        for var_dict, var_name in [(b_dis_E, 'b_dis_E'), (b_ch_E, 'b_ch_E'), (s_E, 's_E'),
                                   (b_dis_G, 'b_dis_G'), (b_ch_G, 'b_ch_G'), (s_G, 's_G'),
                                   (b_dis_H, 'b_dis_H'), (b_ch_H, 'b_ch_H'), (s_H, 's_H')]:
            if t in var_dict:
                if var_name not in pattern:
                    pattern[var_name] = {}
                pattern[var_name][t] = model.getVal(var_dict[t])
    
    # Calculate original objective (without duals)
    objval = calculate_original_objective(player, pattern, params)
    
    return min_redcost, pattern, objval


def calculate_original_objective(player, pattern, params):
    """
    Calculate the original objective value without dual contributions
    """
    cost = 0
    
    # Production costs
    if 'p_res' in pattern:
        c_res = params.get(f'c_res_{player}', 0)
        cost += sum(c_res * val for val in pattern['p_res'].values())
    
    if 'p_hp' in pattern:
        c_hp = params.get(f'c_hp_{player}', 0)
        cost += sum(c_hp * val for val in pattern['p_hp'].values())
    
    if 'p_els' in pattern:
        c_els = params.get(f'c_els_{player}', 0)
        cost += sum(c_els * val for val in pattern['p_els'].values())
    
    # Startup costs
    if 'z_su' in pattern:
        c_su = params.get(f'c_su_{player}', 0)
        cost += sum(c_su * val for val in pattern['z_su'].values())
    
    # Grid interaction costs
    for t in pattern.get('i_E_gri', {}).keys():
        pi_E_gri_import = params.get(f'pi_E_gri_import_{t}', 0)
        pi_E_gri_export = params.get(f'pi_E_gri_export_{t}', 0)
        
        if 'i_E_gri' in pattern and t in pattern['i_E_gri']:
            cost += pi_E_gri_import * pattern['i_E_gri'][t]
        if 'e_E_gri' in pattern and t in pattern['e_E_gri']:
            cost -= pi_E_gri_export * pattern['e_E_gri'][t]
    
    for t in pattern.get('i_H_gri', {}).keys():
        pi_H_gri_import = params.get(f'pi_H_gri_import_{t}', 0)
        pi_H_gri_export = params.get(f'pi_H_gri_export_{t}', 0)
        
        if 'i_H_gri' in pattern and t in pattern['i_H_gri']:
            cost += pi_H_gri_import * pattern['i_H_gri'][t]
        if 'e_H_gri' in pattern and t in pattern['e_H_gri']:
            cost -= pi_H_gri_export * pattern['e_H_gri'][t]
    
    for t in pattern.get('i_G_gri', {}).keys():
        pi_G_gri_import = params.get(f'pi_G_gri_import_{t}', 0)
        pi_G_gri_export = params.get(f'pi_G_gri_export_{t}', 0)
        
        if 'i_G_gri' in pattern and t in pattern['i_G_gri']:
            cost += pi_G_gri_import * pattern['i_G_gri'][t]
        if 'e_G_gri' in pattern and t in pattern['e_G_gri']:
            cost -= pi_G_gri_export * pattern['e_G_gri'][t]
    
    # Storage costs
    c_sto = params.get('c_sto', 0.01)
    nu_ch = params.get('nu_ch', 0.9)
    nu_dis = params.get('nu_dis', 0.9)
    
    for storage_type in ['E', 'G', 'H']:
        charge_var = f'b_ch_{storage_type}'
        discharge_var = f'b_dis_{storage_type}'
        
        if charge_var in pattern:
            cost += sum(c_sto * nu_ch * val for val in pattern[charge_var].values())
        if discharge_var in pattern:
            cost += sum(c_sto * (1/nu_dis) * val for val in pattern[discharge_var].values())
    
    return cost
import numpy as np

def create_initial_patterns_for_players(players, time_periods, parameters):
    """
    Create simple initial patterns for each player
    Each player satisfies their demand using only grid imports/exports
    No community trading in initial patterns
    """
    
    initial_patterns = {}
    
    for player in players:
        pattern = {}
        
        # Player u1: Renewable generator with storage
        if player == 'u1':
            pattern['i_E_gri'] = {}
            pattern['e_E_gri'] = {}
            pattern['i_E_com'] = {}
            pattern['e_E_com'] = {}
            pattern['p_res'] = {}
            pattern['b_ch_E'] = {}
            pattern['b_dis_E'] = {}
            pattern['s_E'] = {}
            
            for t in time_periods:
                # Generate renewable when available
                renewable_cap = parameters.get(f'renewable_cap_{player}_{t}', 0)
                pattern['p_res'][t] = renewable_cap * 0.8  # Use 80% of capacity
                
                # Export excess to grid (not community initially)
                pattern['e_E_gri'][t] = min(pattern['p_res'][t], parameters.get(f'e_E_cap_{player}_{t}', 0.1))
                pattern['i_E_gri'][t] = 0.0
                
                # No community trading initially
                pattern['i_E_com'][t] = 0.0
                pattern['e_E_com'][t] = 0.0
                
                # Simple storage operation
                pattern['b_ch_E'][t] = 0.0
                pattern['b_dis_E'][t] = 0.0
                pattern['s_E'][t] = parameters.get('initial_soc', 0.5)
        
        # Player u2: Electrolyzer with hydrogen storage
        elif player == 'u2':
            pattern['i_E_gri'] = {}
            pattern['e_E_gri'] = {}
            pattern['i_E_com'] = {}
            pattern['e_E_com'] = {}
            pattern['i_G_gri'] = {}
            pattern['e_G_gri'] = {}
            pattern['i_G_com'] = {}
            pattern['e_G_com'] = {}
            pattern['p_els'] = {}
            pattern['z_su'] = {}
            pattern['z_on'] = {}
            pattern['z_off'] = {}
            pattern['z_sb'] = {}
            pattern['b_ch_G'] = {}
            pattern['b_dis_G'] = {}
            pattern['s_G'] = {}
            
            for t in time_periods:
                # Simple operation: off most of the time, on during cheap hours
                if t in [0, 1, 2, 3, 4, 5]:  # Night hours - operate
                    pattern['z_on'][t] = 1.0
                    pattern['z_off'][t] = 0.0
                    pattern['z_sb'][t] = 0.0
                    pattern['p_els'][t] = 10.0  # Produce 10 kg/h hydrogen
                    pattern['i_E_gri'][t] = 0.5  # Import electricity for electrolyzer
                else:
                    pattern['z_on'][t] = 0.0
                    pattern['z_off'][t] = 1.0
                    pattern['z_sb'][t] = 0.0
                    pattern['p_els'][t] = 0.0
                    pattern['i_E_gri'][t] = 0.0
                
                # Startup detection
                if t == 0:
                    pattern['z_su'][t] = 1.0 if pattern['z_on'][t] > 0 else 0.0
                else:
                    pattern['z_su'][t] = 1.0 if (pattern['z_on'][t] > 0 and pattern['z_on'][t-1] == 0) else 0.0
                
                # Export hydrogen to grid
                pattern['e_G_gri'][t] = min(pattern['p_els'][t], parameters.get(f'e_G_cap_{player}_{t}', 50))
                pattern['i_G_gri'][t] = 0.0
                
                # No electricity export
                pattern['e_E_gri'][t] = 0.0
                
                # No community trading
                pattern['i_E_com'][t] = 0.0
                pattern['e_E_com'][t] = 0.0
                pattern['i_G_com'][t] = 0.0
                pattern['e_G_com'][t] = 0.0
                
                # No storage use initially
                pattern['b_ch_G'][t] = 0.0
                pattern['b_dis_G'][t] = 0.0
                pattern['s_G'][t] = 0.0
        
        # Player u3: Heat pump with heat storage
        elif player == 'u3':
            pattern['i_E_gri'] = {}
            pattern['e_E_gri'] = {}
            pattern['i_E_com'] = {}
            pattern['e_E_com'] = {}
            pattern['i_H_gri'] = {}
            pattern['e_H_gri'] = {}
            pattern['i_H_com'] = {}
            pattern['e_H_com'] = {}
            pattern['p_hp'] = {}
            pattern['b_ch_H'] = {}
            pattern['b_dis_H'] = {}
            pattern['s_H'] = {}
            
            for t in time_periods:
                # Simple heat pump operation
                pattern['p_hp'][t] = 0.0  # No heat production initially
                pattern['i_E_gri'][t] = 0.0  # No electricity import for heat pump
                pattern['e_E_gri'][t] = 0.0
                
                # No heat trade
                pattern['i_H_gri'][t] = 0.0
                pattern['e_H_gri'][t] = 0.0
                
                # No community trading
                pattern['i_E_com'][t] = 0.0
                pattern['e_E_com'][t] = 0.0
                pattern['i_H_com'][t] = 0.0
                pattern['e_H_com'][t] = 0.0
                
                # No storage use
                pattern['b_ch_H'][t] = 0.0
                pattern['b_dis_H'][t] = 0.0
                pattern['s_H'][t] = parameters.get('initial_soc', 50)
        
        # Player u4: Electricity consumer
        elif player == 'u4':
            pattern['i_E_gri'] = {}
            pattern['e_E_gri'] = {}
            pattern['i_E_com'] = {}
            pattern['e_E_com'] = {}
            
            for t in time_periods:
                # Import all demand from grid
                demand = parameters.get(f'd_E_nfl_{player}_{t}', 0)
                pattern['i_E_gri'][t] = demand
                pattern['e_E_gri'][t] = 0.0
                
                # No community trading
                pattern['i_E_com'][t] = 0.0
                pattern['e_E_com'][t] = 0.0
        
        # Player u5: Hydrogen consumer
        elif player == 'u5':
            pattern['i_G_gri'] = {}
            pattern['e_G_gri'] = {}
            pattern['i_G_com'] = {}
            pattern['e_G_com'] = {}
            
            for t in time_periods:
                # Import all demand from grid
                demand = parameters.get(f'd_G_nfl_{player}_{t}', 0)
                pattern['i_G_gri'][t] = demand
                pattern['e_G_gri'][t] = 0.0
                
                # No community trading
                pattern['i_G_com'][t] = 0.0
                pattern['e_G_com'][t] = 0.0
        
        # Player u6: Heat consumer
        elif player == 'u6':
            pattern['i_H_gri'] = {}
            pattern['e_H_gri'] = {}
            pattern['i_H_com'] = {}
            pattern['e_H_com'] = {}
            
            for t in time_periods:
                # Import all demand from grid
                demand = parameters.get(f'd_H_nfl_{player}_{t}', 0)
                pattern['i_H_gri'][t] = demand
                pattern['e_H_gri'][t] = 0.0
                
                # No community trading
                pattern['i_H_com'][t] = 0.0
                pattern['e_H_com'][t] = 0.0
        
        initial_patterns[player] = pattern
    
    return initial_patterns


def test_initial_patterns():
    """Test the initial pattern creation"""
    
    players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    time_periods = list(range(24))
    
    # Create minimal parameters
    parameters = {
        'initial_soc': 0.5,
    }
    
    # Add time-dependent data
    for t in time_periods:
        # Renewable capacity
        if 6 <= t <= 18:
            solar_factor = np.exp(-((t - 12) / 3.5)**2)
            parameters[f'renewable_cap_u1_{t}'] = 2 * solar_factor
        else:
            parameters[f'renewable_cap_u1_{t}'] = 0
        
        # Demands
        parameters[f'd_E_nfl_u4_{t}'] = 0.06 + 0.02 * np.sin(2 * np.pi * (t - 8) / 24)
        parameters[f'd_G_nfl_u5_{t}'] = 3 + 1 * np.sin(2 * np.pi * (t - 12) / 24)
        parameters[f'd_H_nfl_u6_{t}'] = 6 + 2 * np.cos(2 * np.pi * (t - 3) / 24)
        
        # Grid limits
        parameters[f'e_E_cap_u1_{t}'] = 0.1
        parameters[f'e_G_cap_u2_{t}'] = 50
    
    # Create patterns
    patterns = create_initial_patterns_for_players(players, time_periods, parameters)
    
    # Verify patterns
    print("Initial Patterns Created:")
    print("="*60)
    
    for player, pattern in patterns.items():
        print(f"\nPlayer {player}:")
        for var_name in pattern:
            if isinstance(pattern[var_name], dict):
                non_zero = sum(1 for v in pattern[var_name].values() if abs(v) > 1e-6)
                if non_zero > 0:
                    total = sum(pattern[var_name].values())
                    print(f"  {var_name}: {non_zero} non-zero values, total={total:.2f}")
    
    # Check feasibility
    print("\n" + "="*60)
    print("Feasibility Check:")
    print("="*60)
    
    # Check u4 electricity balance
    u4_pattern = patterns['u4']
    for t in time_periods:
        demand = parameters[f'd_E_nfl_u4_{t}']
        supply = u4_pattern['i_E_gri'][t] + u4_pattern['i_E_com'][t]
        if abs(demand - supply) > 1e-6:
            print(f"ERROR: u4 at t={t}: demand={demand:.4f}, supply={supply:.4f}")
    print("u4 electricity balance: OK")
    
    # Check u5 hydrogen balance
    u5_pattern = patterns['u5']
    for t in time_periods:
        demand = parameters[f'd_G_nfl_u5_{t}']
        supply = u5_pattern['i_G_gri'][t] + u5_pattern['i_G_com'][t]
        if abs(demand - supply) > 1e-6:
            print(f"ERROR: u5 at t={t}: demand={demand:.4f}, supply={supply:.4f}")
    print("u5 hydrogen balance: OK")
    
    # Check u6 heat balance
    u6_pattern = patterns['u6']
    for t in time_periods:
        demand = parameters[f'd_H_nfl_u6_{t}']
        supply = u6_pattern['i_H_gri'][t] + u6_pattern['i_H_com'][t]
        if abs(demand - supply) > 1e-6:
            print(f"ERROR: u6 at t={t}: demand={demand:.4f}, supply={supply:.4f}")
    print("u6 heat balance: OK")
    
    # Check community balance (should be zero for all)
    for t in time_periods:
        elec_com_balance = sum(patterns[p].get('i_E_com', {}).get(t, 0) - 
                               patterns[p].get('e_E_com', {}).get(t, 0) for p in players)
        heat_com_balance = sum(patterns[p].get('i_H_com', {}).get(t, 0) - 
                               patterns[p].get('e_H_com', {}).get(t, 0) for p in players)
        hydro_com_balance = sum(patterns[p].get('e_G_com', {}).get(t, 0) - 
                                patterns[p].get('i_G_com', {}).get(t, 0) for p in players)
        
        if abs(elec_com_balance) > 1e-6:
            print(f"ERROR: Community electricity imbalance at t={t}: {elec_com_balance:.4f}")
        if abs(heat_com_balance) > 1e-6:
            print(f"ERROR: Community heat imbalance at t={t}: {heat_com_balance:.4f}")
        if abs(hydro_com_balance) > 1e-6:
            print(f"ERROR: Community hydrogen imbalance at t={t}: {hydro_com_balance:.4f}")
    
    print("Community balance: OK")
    
    return patterns


if __name__ == "__main__":
    patterns = test_initial_patterns()
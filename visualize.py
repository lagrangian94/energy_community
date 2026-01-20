import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_community_prices_comparison(community_prices, community_prices_lp, community_prices_chp,
                                     market_prices, save_path=None):
    """
    Compare community prices across three methods: IP, LP, CHP
    
    Parameters
    ----------
    community_prices : dict
        IP (Integer Programming) prices
    community_prices_lp : dict
        LP (Linear Programming) prices
    community_prices_chp : dict
        CHP (Convex Hull Pricing) prices
    market_prices : dict
        Market prices
    save_path : str, optional
        Path to save figure
    """
    
    # Convert dictionaries to arrays
    def dict_to_array(price_dict):
        """Convert {0: val, 1: val, ...} to numpy array"""
        if not price_dict:
            return np.array([])
        indices = sorted(price_dict.keys())
        return np.array([float(price_dict[i]) for i in indices])
    def list_to_array(price_list):
        """Convert list to numpy array"""
        if not price_list:
            return np.array([])
        return np.array([float(p) for p in price_list])
    # Get all energy types (union of all three dicts)
    energy_types = set()
    energy_types.update(community_prices.keys())
    energy_types.update(community_prices_lp.keys())
    energy_types.update(community_prices_chp.keys())
    # Add energy types from market prices if provided
    if market_prices and 'import' in market_prices:
        energy_types.update(market_prices['import'].keys())
    if market_prices and 'use_tou_elec' in market_prices:
        energy_types.update(market_prices['export'].keys())
    energy_types = sorted(energy_types)
    
    # Prepare data
    data = {}
    for energy_type in energy_types:
        data[energy_type] = {
            'IP': dict_to_array(community_prices.get(energy_type, {})),
            'LP': dict_to_array(community_prices_lp.get(energy_type, {})),
            'CHP': dict_to_array(community_prices_chp.get(energy_type, {}))
        }
        # Add market import price if available
        if market_prices and 'import' in market_prices:
            if energy_type in market_prices['import']:
                data[energy_type]['Market Import'] = list_to_array(market_prices['import'][energy_type])
                data[energy_type]['Market Export'] = list_to_array(market_prices['export'][energy_type])
            else:
                data[energy_type]['Market Import'] = np.array([])
                data[energy_type]['Market Export'] = np.array([])
        else:
            data[energy_type]['Market Import'] = np.array([])
            data[energy_type]['Market Export'] = np.array([])
    
    # Create figure
    n_energy_types = len(energy_types)
    fig, axes = plt.subplots(n_energy_types, 1, figsize=(14, 4 * n_energy_types))
    if n_energy_types == 1:
        axes = [axes]
    
    # Color scheme
    colors = {
        'IP': '#2E86AB',     # Blue
        'LP': '#A23B72',     # Purple
        'CHP': '#F18F01',    # Orange
        'Market_Import': '#000000',   # Black for external market
        'Market_Export': '#444444'   # Dark gray, distinct from black
    }
    
    # Plot each energy type
    for idx, energy_type in enumerate(energy_types):
        ax_time = axes[idx]

        
        # Get data for this energy type
        ip_data = data[energy_type]['IP']
        lp_data = data[energy_type]['LP']
        chp_data = data[energy_type]['CHP']
        market_import_data = data[energy_type]['Market Import']
        market_export_data = data[energy_type]['Market Export']
        # Determine time length
        T = max(len(ip_data), len(lp_data), len(chp_data))
        hours = np.arange(T)
        
        # --- Time Series Plot ---
        if len(ip_data) > 0:
            ax_time.plot(hours[:len(ip_data)], ip_data, 'o-', 
                        color=colors['IP'], linewidth=2.5, markersize=6,
                        label='IP', alpha=0.8)
        
        if len(lp_data) > 0:
            ax_time.plot(hours[:len(lp_data)], lp_data, 's-', 
                        color=colors['LP'], linewidth=2.5, markersize=6,
                        label='LP', alpha=0.8)
        
        if len(chp_data) > 0:
            ax_time.plot(hours[:len(chp_data)], chp_data, '^-', 
                        color=colors['CHP'], linewidth=2.5, markersize=6,
                        label='CHP', alpha=0.8)
        # Plot market import price
        if len(market_import_data) > 0:
            if market_prices['use_tou_elec'] and energy_type == 'electricity':
                ax_time.plot(hours[:len(market_import_data)], market_import_data, 'x--',
                            color=colors['Market_Import'], linewidth=1.5, markersize=4,
                            label='External Market (TOU)', alpha=0.8)
            else:
                ax_time.plot(hours[:len(market_import_data)], market_import_data, 'x--',
                            color=colors['Market_Import'], linewidth=1.5, markersize=4,
                            label='External Market (Import)', alpha=0.8)
        if len(market_export_data) > 0:
            ax_time.plot(hours[:len(market_export_data)], market_export_data, 'd-.',
            color=colors['Market_Export'], linewidth=1.5, markersize=6,
            label='External Market (Export)', alpha=0.8)
        # Check if prices are all zero or nearly identical
        all_data = np.concatenate([d for d in [ip_data, lp_data, chp_data] if len(d) > 0])
        if len(all_data) > 0:
            if np.allclose(all_data, 0, atol=1e-6):
                ax_time.text(0.5, 0.5, 'All prices are zero', 
                           transform=ax_time.transAxes,
                           ha='center', va='center', fontsize=14, 
                           color='red')
            elif np.std(all_data) < 0.01:  # Very small variance
                ax_time.text(0.5, 0.9, f'Nearly constant (σ={np.std(all_data):.4f})', 
                           transform=ax_time.transAxes,
                           ha='center', va='top', fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax_time.set_xlabel('Time Period (hour)', fontsize=12)
        ax_time.set_ylabel(f'Price ({_get_unit(energy_type)})', fontsize=12)
        ax_time.set_title(f'{energy_type.capitalize()} Community Prices', 
                         fontsize=14, pad=15)
        ax_time.legend(fontsize=11, loc='best', framealpha=0.9)
        ax_time.set_xticks(hours)
        ax_time.grid(True, alpha=0.3, linestyle='--')
        
        # --- Statistics Plot ---
        methods = []
        means = []
        stds = []
        mins = []
        maxs = []
        
        for method, method_data in [('IP', ip_data), ('LP', lp_data), ('CHP', chp_data)]:
            if len(method_data) > 0:
                methods.append(method)
                means.append(np.mean(method_data))
                stds.append(np.std(method_data))
                mins.append(np.min(method_data))
                maxs.append(np.max(method_data))
        
        # if len(methods) > 0:
        #     x_pos = np.arange(len(methods))
            
        #     # Bar plot with error bars
        #     bars = ax_stats.bar(x_pos, means, 
        #                       color=[colors[m] for m in methods],
        #                       alpha=0.7, edgecolor='black', linewidth=1.5)
            
        #     ax_stats.errorbar(x_pos, means, yerr=stds, 
        #                     fmt='none', ecolor='black', capsize=5, capthick=2)
            
        #     # Add value labels
        #     for i, (bar, mean, std, mn, mx) in enumerate(zip(bars, means, stds, mins, maxs)):
        #         height = bar.get_height()
        #         ax_stats.text(bar.get_x() + bar.get_width()/2., height + std,
        #                     f'{mean:.2f}\n±{std:.2f}',
        #                     ha='center', va='bottom', fontsize=9, fontweight='bold')
                
        #         # Add min/max info
        #         ax_stats.text(bar.get_x() + bar.get_width()/2., height/2,
        #                     f'[{mn:.2f},\n{mx:.2f}]',
        #                     ha='center', va='center', fontsize=8, color='white',
        #                     fontweight='bold')
            
        #     ax_stats.set_xticks(x_pos)
        #     ax_stats.set_xticklabels(methods, fontsize=12, fontweight='bold')
        #     ax_stats.set_ylabel(f'Mean Price ({_get_unit(energy_type)})', 
        #                       fontsize=11, fontweight='bold')
        #     ax_stats.set_title('Statistics (Mean ± Std)', fontsize=12, fontweight='bold')
        #     ax_stats.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add comparison text
        if len(methods) == 3 and len(ip_data) > 0 and len(lp_data) > 0 and len(chp_data) > 0:
            ip_mean = np.mean(ip_data)
            lp_mean = np.mean(lp_data)
            chp_mean = np.mean(chp_data)
            
            # Calculate differences
            ip_lp_diff = abs(ip_mean - lp_mean)
            ip_chp_diff = abs(ip_mean - chp_mean)
            lp_chp_diff = abs(lp_mean - chp_mean)
            
            comparison_text = (
                f'IP vs LP: {ip_lp_diff:.4f}\n'
                f'IP vs CHP: {ip_chp_diff:.4f}\n'
                f'LP vs CHP: {lp_chp_diff:.4f}'
            )
            
            # # Determine if methods are equivalent
            # if max(ip_lp_diff, ip_chp_diff, lp_chp_diff) < 1e-3:
            #     comparison_text += '\n\n✓ Nearly Identical'
            #     bbox_color = 'lightgreen'
            # else:
            #     comparison_text += '\n\n⚠ Different'
            #     bbox_color = 'lightyellow'
            
            # ax_stats.text(0.98, 0.02, comparison_text,
            #             transform=ax_stats.transAxes,
            #             fontsize=9, verticalalignment='bottom',
            #             horizontalalignment='right',
            #             bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7))
    
    # Overall title
    # fig.suptitle('Community Prices Comparison: IP vs LP vs CHP', 
    #             fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(h_pad=3.0)  # h_pad로 수직 간격 조정
    # # Add legend explanation
    # legend_text = (
    #     'IP: Integer Programming\n'
    #     'LP: Linear Programming (compact formulation)\n'
    #     'CHP: Column Generation with Convex Hull Pricing'
    # )
    # fig.text(0.02, 0.02, legend_text, fontsize=10,
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.savefig('./community_prices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED PRICE COMPARISON STATISTICS")
    print("="*80)
    
    for energy_type in energy_types:
        print(f"\n{'='*80}")
        print(f"{energy_type.upper()}")
        print(f"{'='*80}")
        
        ip_data = data[energy_type]['IP']
        lp_data = data[energy_type]['LP']
        chp_data = data[energy_type]['CHP']
        
        # Print statistics table
        print(f"{'Method':<10} | {'Mean':>12} | {'Std':>12} | {'Min':>12} | {'Max':>12} | {'NonZero':>8}")
        print("-"*80)
        
        for method, method_data in [('IP', ip_data), ('LP', lp_data), ('CHP', chp_data)]:
            if len(method_data) > 0:
                mean = np.mean(method_data)
                std = np.std(method_data)
                mn = np.min(method_data)
                mx = np.max(method_data)
                nonzero = np.sum(np.abs(method_data) > 1e-6)
                
                print(f"{method:<10} | {mean:>12.4f} | {std:>12.4f} | {mn:>12.4f} | {mx:>12.4f} | {nonzero:>8}/{len(method_data)}")
            else:
                print(f"{method:<10} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>8}")
        
        # Pairwise comparison
        if len(ip_data) > 0 and len(lp_data) > 0 and len(chp_data) > 0:
            print("\nPairwise Differences (Mean Absolute Error):")
            print(f"  IP vs LP:  {np.mean(np.abs(ip_data - lp_data)):.6f}")
            print(f"  IP vs CHP: {np.mean(np.abs(ip_data[:len(chp_data)] - chp_data)):.6f}")
            print(f"  LP vs CHP: {np.mean(np.abs(lp_data[:len(chp_data)] - chp_data)):.6f}")
            
            # Correlation
            print("\nCorrelation Coefficients:")
            if len(ip_data) == len(lp_data):
                corr_ip_lp = np.corrcoef(ip_data, lp_data)[0, 1]
                print(f"  IP vs LP:  {corr_ip_lp:.6f}")
            if len(ip_data) >= len(chp_data):
                corr_ip_chp = np.corrcoef(ip_data[:len(chp_data)], chp_data)[0, 1]
                print(f"  IP vs CHP: {corr_ip_chp:.6f}")
            if len(lp_data) >= len(chp_data):
                corr_lp_chp = np.corrcoef(lp_data[:len(chp_data)], chp_data)[0, 1]
                print(f"  LP vs CHP: {corr_lp_chp:.6f}")


def _get_unit(energy_type):
    """Get appropriate unit for energy type"""
    units = {
        'electricity': 'EUR/mWh',
        'heat': 'EUR/MWh',
        'hydrogen': 'EUR/kg',
        'hydro': 'EUR/kg',  # Assuming hydro is actually hydrogen
    }
    return units.get(energy_type.lower(), 'KRW/unit')


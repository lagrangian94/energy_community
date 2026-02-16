"""
Smoothing 테스트: analysis_mip.py와 동일한 설정으로
smoothing=False vs smoothing=True 비교
"""
from data_generator import setup_lem_parameters
from compact_utility import LocalEnergyMarket, solve_and_extract_results
from chp import ColumnGenerationSolver
import time

players = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
time_periods = list(range(24))

configuration = {}
configuration["players_with_renewables"] = ['u1']
configuration["players_with_solar"] = []
configuration["players_with_wind"] = ['u1']
configuration["players_with_electrolyzers"] = ['u2']
configuration["players_with_heatpumps"] = ['u3']
configuration["players_with_elec_storage"] = ['u1']
configuration["players_with_hydro_storage"] = ['u2']
configuration["players_with_heat_storage"] = ['u3']
configuration["players_with_nfl_elec_demand"] = ['u4']
configuration["players_with_nfl_hydro_demand"] = ['u5']
configuration["players_with_nfl_heat_demand"] = ['u6']
configuration["players_with_fl_elec_demand"] = ['u2', 'u3']
configuration["players_with_fl_hydro_demand"] = []
configuration["players_with_fl_heat_demand"] = []

parameters = setup_lem_parameters(players, configuration, time_periods)

# Step 1: IP solve to get init_sol
print("=" * 80)
print("STEP 1: Solving IP to get init_sol")
print("=" * 80)
lem = LocalEnergyMarket(players, time_periods, parameters, model_type='mip')
status, results_ip, _, _, _ = lem.solve_complete_model(analyze_revenue=False)
welfare_ip = lem.model.getObjVal()
print(f"IP Objective: {welfare_ip:.2f}")
init_sol = results_ip

# Step 2: CG without smoothing
print("\n" + "=" * 80)
print("STEP 2: Column Generation WITHOUT smoothing")
print("=" * 80)
cg_no_smooth = ColumnGenerationSolver(players, time_periods, parameters, model_type='mip', init_sol=init_sol, smoothing=False)
t0 = time.time()
status_ns, results_ns, obj_ns, _ = cg_no_smooth.solve()
time_ns = time.time() - t0
print(f"Status: {status_ns} | Obj: {obj_ns:.2f} | Time: {time_ns:.2f}s | Iters: {cg_no_smooth.master.model.data.get('pricer_iters', 'N/A')}")

# Step 3: CG with smoothing
print("\n" + "=" * 80)
print("STEP 3: Column Generation WITH smoothing")
print("=" * 80)
cg_smooth = ColumnGenerationSolver(players, time_periods, parameters, model_type='mip', init_sol=init_sol, smoothing=True)
t0 = time.time()
status_s, results_s, obj_s, _ = cg_smooth.solve()
time_s = time.time() - t0
print(f"Status: {status_s} | Obj: {obj_s:.2f} | Time: {time_s:.2f}s")

# Summary
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"{'':20s} {'No Smoothing':>15s} {'Smoothing':>15s}")
print("-" * 50)
print(f"{'Objective':20s} {obj_ns:15.2f} {obj_s:15.2f}")
print(f"{'Time (s)':20s} {time_ns:15.2f} {time_s:15.2f}")
print(f"{'Obj Difference':20s} {abs(obj_ns - obj_s):15.6f}")

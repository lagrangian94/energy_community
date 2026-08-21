"""
Multi-day weak-ε-core experiment: Owen vs. Row generation (cost-of-stability),
structured like results_15p/ results_30p/ results_53/ (one row per day).

Runs, IN ORDER (sensitivity 6p scenarios → 15p → 30p), each over 31 days, T=24,
reserve+peak coupling ON (0.10 / 0.05 / 0.05 × mean import):

  6p sensitivity scenarios (players u1..u6, from results_53 / sensitivity_analysis_claude):
    baseline, low_h2_margin, full_storage, community_size_350, community_size_1000,
    export_cap_020  (per-player electricity export cap via variable ub)
  15p (large_community CONFIGURATION_15), 30p (CONFIGURATION_30)  — baseline only.

TWO-PHASE per run (so the slow/possibly-non-converging row-gen never blocks Owen):
  Phase 1  Owen for ALL 31 days   → owen.csv + cg_day<D>.json (offline-settlement ingredients)
  Phase 2  RowGen for ALL 31 days → rowgen.csv   (Gurobi separation; per-day time budget)

RESUMABLE: a (run, phase, day) already present in its CSV is skipped. Safe to kill and
relaunch; only missing pieces recompute. Row-gen for 30p is expected to hit the per-day
budget without converging — the partial CoS lower bound is stored (v* ≥ ...).

RUN GROUPS. `--groups` defaults to `core,param6p` = 9 runs x 31 days = 279 rows: the
baseline at 6/15/30 plus the five 6-prosumer parameter variations. The parameter axis is
deliberately 6p-only -- it exists to show the results are not an artefact of one
calibration, and 6p is the only size with exact ground truth to compare against, so
repeating it at 15p and 30p multiplies the most expensive instances for no extra claim.
The `reserve.txt` sec.3.1 scenario axes (channel decomposition, price sweeps, the
low-H2 x reserve interaction) are `scenario` and OPT-IN.

Usage:
  python weak_eps_experiment/run_multiday.py                       # core+param6p, both phases
  python weak_eps_experiment/run_multiday.py --phase owen          # Owen only (fast)
  python weak_eps_experiment/run_multiday.py --groups core         # 3 baselines only
  python weak_eps_experiment/run_multiday.py --groups all          # + the sec.3.1 axes
  python weak_eps_experiment/run_multiday.py --runs baseline_6p,baseline_15p
  python weak_eps_experiment/run_multiday.py --days 1,2,3 --force
  python weak_eps_experiment/run_multiday.py --phase rowgen --runs baseline_15p --budget 3600
"""
import os, sys, json, time, csv, argparse, glob
# layout: <repo root>/ieee_owen/weak_eps_experiment/. Shared model modules and
# data/ live at the repo root; large_community / reserve_metrics live in the
# paper folder, so both go on sys.path. CWD stays the repo root (data paths
# and the model's plot output are resolved relative to it).
_PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import numpy as np
from data_generator import setup_lem_parameters
from reserve_metrics import (reserve_peak_metrics, solve_standalone_r_sym,
                             flatten_for_csv, failed_checks)
from stability_check import check_allocations, flatten_for_csv as stab_for_csv
from compact_utility import LocalEnergyMarket
from chp import ColumnGenerationSolver
from core import CoreComputation
import large_community as LC

OUT = os.path.dirname(os.path.abspath(__file__))
T = list(range(24))
DAYS = list(range(1, 32))                      # 31 days, matching results_53
# Reserve/peak scenario in ABSOLUTE units (reserve.txt sec.2.1 / sec.3.1).
#   RESERVE_PRICE [EUR/MW.h]  0 = channel off, 11 = low regime (Johnsen),
#                             56 = baseline (Nordic FCR-N, DK2)
#   PEAK_PENALTY  [EUR/MW]    0 = channel off, 150-200 = Cornelusse range
RESERVE_PRICE, PEAK_PENALTY = 56.0, 150.0
# Reserve market design. 4-hour blocks match Continental Europe FCR as operated since
# 2020; the 24-hour product the code used before exists in no current FCR market and
# let a single bad hour cap the whole day. 'symmetric' is the default product.
RESERVE_BLOCK_HOURS = 4
RESERVE_PRODUCT = 'symmetric'
# Stand-alone r_sym({j}) baseline for the sec.3.3 pooling gain: one extra small
# MIP per prosumer per day. Turn off if the added solve time ever matters.
STANDALONE_BASELINE = True
SEP_SOLVER = 'gurobi'
# Budget for EACH of the two stability measurements in the Owen phase (sigma and the
# gap-corrected point), so a day costs at most twice this before it gives up and says so.
STAB_TIME_LIMIT = 3600

# 6-player config (u1..u6) — same as sensitivity_analysis_claude / analysis_mip
P6 = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
C6 = {
    "players_with_renewables": ['u1'], "players_with_solar": [], "players_with_wind": ['u1'],
    "players_with_electrolyzers": ['u2'], "players_with_heatpumps": ['u3'],
    "players_with_elec_storage": ['u1'], "players_with_hydro_storage": ['u2'],
    "players_with_heat_storage": ['u3'],
    "players_with_nfl_elec_demand": ['u4'], "players_with_nfl_hydro_demand": ['u5'],
    "players_with_nfl_heat_demand": ['u6'],
    "players_with_fl_elec_demand": ['u2', 'u3'],
    "players_with_fl_hydro_demand": [], "players_with_fl_heat_demand": [],
}
# 6p baseline candidate set (scalars) — from sensitivity_analysis_claude.BASELINE_CANDIDATES
BASE6 = {
    'use_korean_price': True, 'use_tou_elec': False, 'import_factor': 1.5, 'month': 1,
    'hp_cap': 0.8, 'els_cap': 1, 'num_households': 700, 'nu_cop': 3.28,
    'c_su_G': 50.0, 'c_su_H': 10.0, 'base_h2_price_eur': 5000 / 1500,
    'e_E_cap_ratio': 1.0, 'e_H_cap_ratio': 1.0, 'e_G_cap_ratio': 1.0,
    'eff_type': 1, 'segments': 6, 'peak_penalty_ratio': 0.0,
    'wind_el_ratio': 1.0, 'solar_el_ratio': 1.0,
    'storage_power_ratio_E': 0.25, 'storage_power_ratio_G': 0.25, 'storage_power_ratio_H': 0.25,
    'storage_capacity_ratio_E': 3.0, 'storage_capacity_ratio_G': 0.0, 'storage_capacity_ratio_H': 0.0,
    'initial_soc_ratio_E': 0.2, 'initial_soc_ratio_G': 0.2, 'initial_soc_ratio_H': 0.2,
}

def scalarize(cand):
    return {k: (v[0] if isinstance(v, list) else v) for k, v in cand.items()}

# run registry (executed in this order)
# `group` selects what a plain `--groups` default sweep covers. The parameter
# variations are a 6-prosumer story: they are there to show the results are not an
# artefact of one calibration, and 6p is the only size where the exact ground truth
# (brute force over all coalitions) is available to compare against anyway. At 15p and
# 30p the 31 days of the baseline are what Figures 1 and 1b need, and running the
# variants there would multiply the most expensive instances for no extra claim.
#
#   core      baseline at 6 / 15 / 30 -- Figures 1, 1b, and the size axis
#   param6p   6p robustness variations
#   scenario  reserve.txt sec.3.1 axes: channel decomposition, price sweeps,
#             the low-H2 x reserve interaction. OPT-IN (`--groups ...,scenario`).
RUNS = [
    dict(name='baseline_6p', group='core',   players=P6, config=C6, base=BASE6, ov={}, ovfn=None, budget=300),
    dict(name='low_h2_margin_6p',      group='param6p', players=P6, config=C6, base=BASE6,
         ov={'base_h2_price_eur': 2.0, 'import_factor': 3.0}, ovfn=None, budget=300),
    dict(name='full_storage_6p',       group='param6p', players=P6, config=C6, base=BASE6,
         ov={'storage_capacity_ratio_G': 3.0, 'storage_capacity_ratio_H': 3.0}, ovfn=None, budget=300),
    dict(name='community_size_350_6p', group='param6p', players=P6, config=C6, base=BASE6,
         ov={'num_households': 350}, ovfn=None, budget=300),
    dict(name='community_size_1000_6p',group='param6p', players=P6, config=C6, base=BASE6,
         ov={'num_households': 1000}, ovfn=None, budget=300),
    dict(name='export_cap_020_6p',     group='param6p', players=P6, config=C6, base=BASE6,
         ov={'e_E_cap_ratio': 0.2, 'e_G_cap_ratio': 0.2, 'e_H_cap_ratio': 0.2}, ovfn=None, budget=300),
    dict(name='baseline_15p', group='core', players=LC.PLAYERS_15, config=LC.CONFIGURATION_15,
         base=scalarize(LC.BASELINE_CANDIDATES_15), ov={}, ovfn=LC.apply_15player_overrides, budget=900),
    dict(name='baseline_30p', group='core', players=LC.PLAYERS_30, config=LC.CONFIGURATION_30,
         base=scalarize(LC.BASELINE_CANDIDATES_30), ov={}, ovfn=LC.apply_30player_overrides, budget=3600),

    dict(name='baseline_60p', group='core', players=LC.PLAYERS_60, config=LC.CONFIGURATION_60,
         base=scalarize(LC.BASELINE_CANDIDATES_60), ov={}, ovfn=LC.apply_60player_overrides,
         budget=3600),

    # ---- non-convex share, at fixed n (validation_plan sec.4) ----
    # The baselines hold the electrolyser count at 1/6, 3/15, 6/30, so the share of
    # members carrying commitment binaries barely moves with n and the flatness of
    # omega^LR across sizes cannot be separated from that. These raise the share while
    # changing NOTHING else: no asset is removed, only electrolysers added, to renewable
    # owners (co-located electrolysis) and to hydrogen consumers (self-supply). Heat pumps,
    # the other source of binaries, are left alone so the axis is electrolysers only.
    #
    #                 electrolysers        members with binaries
    #   baseline 15p  3                    6/15  (40%)
    #   nonconvex_15p 7                    10/15 (67%)
    #   baseline 30p  6                    10/30 (33%)
    #   nonconvex_30p 15                   19/30 (63%)
    #
    # Own group and own directories: these do not overwrite the baselines they are read
    # against.
    dict(name='nonconvex_15p', group='nonconvex', players=LC.PLAYERS_15,
         config={**LC.CONFIGURATION_15,
                 'players_with_electrolyzers': ['u1', 'u2', 'u5', 'u7', 'u8', 'u10', 'u13']},
         base=scalarize(LC.BASELINE_CANDIDATES_15), ov={},
         ovfn=LC.apply_15player_overrides, budget=900),
    dict(name='nonconvex_30p', group='nonconvex', players=LC.PLAYERS_30,
         config={**LC.CONFIGURATION_30,
                 'players_with_electrolyzers': ['u1', 'u2', 'u5', 'u7', 'u8', 'u10', 'u13',
                                                'u16', 'u17', 'u18', 'u19', 'u21', 'u22',
                                                'u27', 'u28']},
         base=scalarize(LC.BASELINE_CANDIDATES_30), ov={},
         ovfn=LC.apply_30player_overrides, budget=3600),

    # ---- reserve.txt sec.3.1 scenario axes (6p, on top of the baseline instance) ----
    # Channel toggle, for the sec.3.3 metric 1 decomposition of v(N). baseline_6p
    # IS the "+both" corner (56/150), so only the other three are needed here.
    dict(name='channel_balance_6p', group='scenario', players=P6, config=C6, base=BASE6,
         ov={'reserve_price': 0.0, 'peak_penalty': 0.0}, ovfn=None, budget=300),
    dict(name='channel_reserve_6p', group='scenario', players=P6, config=C6, base=BASE6,
         ov={'reserve_price': 56.0, 'peak_penalty': 0.0}, ovfn=None, budget=300),
    dict(name='channel_peak_6p', group='scenario', players=P6, config=C6, base=BASE6,
         ov={'reserve_price': 0.0, 'peak_penalty': 150.0}, ovfn=None, budget=300),
    # reserve_price sweep 0 / 11 / 56 at peak=150. The 0 and 56 ends are
    # channel_peak_6p and baseline_6p, so only the low regime is new.
    dict(name='reserve_low_6p', group='scenario', players=P6, config=C6, base=BASE6,
         ov={'reserve_price': 11.0, 'peak_penalty': 150.0}, ovfn=None, budget=300),
    # peak_penalty sweep 0 / 150 / 200 at reserve=56. The 0 and 150 ends are
    # channel_reserve_6p and baseline_6p, so only the top of the Cornelusse range.
    dict(name='peak_200_6p', group='scenario', players=P6, config=C6, base=BASE6,
         ov={'reserve_price': 56.0, 'peak_penalty': 200.0}, ovfn=None, budget=300),
    # sec.3.3 metric 6: reserve_price x low_h2_margin interaction -- does reserve
    # revenue substitute for the hydrogen margin and change the commitment pattern?
    # low_h2_margin_6p already covers this cell at reserve_price = 56.
    dict(name='low_h2_reserve0_6p', group='scenario', players=P6, config=C6, base=BASE6,
         ov={'base_h2_price_eur': 2.0, 'import_factor': 3.0,
             'reserve_price': 0.0, 'peak_penalty': 150.0}, ovfn=None, budget=300),
    dict(name='low_h2_reserve11_6p', group='scenario', players=P6, config=C6, base=BASE6,
         ov={'base_h2_price_eur': 2.0, 'import_factor': 3.0,
             'reserve_price': 11.0, 'peak_penalty': 150.0}, ovfn=None, budget=300),
]

# sec.3.3 metric 1/6 groupings: (label, run at that cell). Consumed by summarize.py.
CHANNEL_CELLS = {
    'balance':      'channel_balance_6p',    # reserve 0,  peak 0
    'reserve_only': 'channel_reserve_6p',    # reserve 56, peak 0
    'peak_only':    'channel_peak_6p',       # reserve 0,  peak 150
    'both':         'baseline_6p',           # reserve 56, peak 150
}
RESERVE_SWEEP = {0.0: 'channel_peak_6p', 11.0: 'reserve_low_6p', 56.0: 'baseline_6p'}
PEAK_SWEEP = {0.0: 'channel_reserve_6p', 150.0: 'baseline_6p', 200.0: 'peak_200_6p'}
LOW_H2_SWEEP = {0.0: 'low_h2_reserve0_6p', 11.0: 'low_h2_reserve11_6p', 56.0: 'low_h2_margin_6p'}

# --------------------------------------------------------------------------- helpers
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x

def _qty(results, key, players):
    d = results.get(key, {}) if isinstance(results, dict) else {}
    return {u: {int(t): float(d.get((u, t), 0.0)) for t in T} for u in players}

def build_params(run, day):
    # Module defaults first, then the run's own overrides, so a run can move
    # along the sec.3.1 reserve_price / peak_penalty axes by putting them in `ov`.
    sens = dict(run['base'])
    sens['reserve_price'] = RESERVE_PRICE
    sens['peak_penalty'] = PEAK_PENALTY
    sens['reserve_block_hours'] = RESERVE_BLOCK_HOURS
    sens['reserve_product'] = RESERVE_PRODUCT
    sens.update(run['ov'])
    sens['day'] = day
    params = setup_lem_parameters(run['players'], run['config'], T, sens)
    if run['ovfn'] is not None:
        params = run['ovfn'](params, T)
    return params

def run_dir(run):
    d = os.path.join(OUT, run['name'])
    os.makedirs(d, exist_ok=True)
    return d

def existing_days(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        with open(csv_path) as f:
            return {int(r['day']) for r in csv.DictReader(f) if r.get('day')}
    except Exception:
        return set()

def _migrate_header(csv_path, fieldnames):
    """Rewrite an existing CSV under a widened header, blank-filling new columns.

    append_row only writes a header for a brand-new file, so a CSV written before
    the reserve/peak schema was added (12 columns) would silently take 30-field
    rows underneath its old header and shift every column. Detect that and
    migrate the file first; rows already present keep their values and get empty
    cells for the new metrics.

    Returns the effective header to append under.
    """
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        old = list(reader.fieldnames or [])
        rows = list(reader)
    if not old or old == list(fieldnames):
        return list(fieldnames)
    # keep any column the old file had that the current schema dropped
    merged = list(fieldnames) + [c for c in old if c and c not in fieldnames]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=merged)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in merged})
    print(f"  [csv] migrated {os.path.basename(csv_path)} to the current schema "
          f"({len(old)} -> {len(merged)} columns)")
    return merged


def append_row(csv_path, row, fieldnames):
    """Upsert on (run, day).

    Plain append was wrong under --force: re-running a day left the stale row in place
    and added a second one, so a 31-day run came back with 35 rows and readers picking
    the first match silently got the OLD value. Replace any row with the same
    (run, day) instead.
    """
    new = not os.path.exists(csv_path)
    if new:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
        return

    fieldnames = _migrate_header(csv_path, fieldnames)
    with open(csv_path, newline='') as f:
        existing = list(csv.DictReader(f))
    key = (str(row.get('run', '')), str(row.get('day', '')))
    kept = [r for r in existing
            if (str(r.get('run', '')), str(r.get('day', ''))) != key]
    n_dropped = len(existing) - len(kept)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, '') for k in fieldnames})
        w.writerow(row)
    if n_dropped > 1:
        print(f"  [csv] replaced {n_dropped} duplicate rows for {key}")

OWEN_COLS = ['run', 'day', 'n_players', 'num_households', 'base_h2_price_eur', 'import_factor',
             # sec.3.1 scenario coordinates, so a row identifies its own cell
             'pi_res', 'pi_peak',
             'v_mip', 'v_chp', 'gap', 'eps_bound', 'time_mip_s', 'time_cg_s',
             # reserve.txt sec.3.2 headline outputs (full series stay in the JSON)
             'r_sym', 'reserve_revenue',
             # Block-era columns. `flatten_for_csv` began emitting these when the reserve
             # product went to 4-hour blocks, and DictWriter rejects a row carrying a key
             # the header does not have -- so the whole Owen phase failed on every day
             # until they were declared here. Adding the name is the fix; dropping the
             # value silently would have been worse than the crash.
             'product', 'n_blocks', 'block_hours', 'r_sym_mwh', 'r_up_mwh', 'r_dn_mwh',
             'pool_standalone_mwh', 'pool_gain_mwh', 'pool_blocks_with_gain',
             'mech_no_pooling_mwh', 'mech_full_mwh',
             'mech_gain_time_mwh', 'mech_gain_direction_mwh',
             'pool_standalone_sum', 'pool_gain_abs', 'pool_gain_ratio',
             'mech_no_pooling', 'mech_time_only', 'mech_full', 'mech_gain_time',
             'mech_gain_direction', 'mech_share_direction',
             'n_binding_up', 'n_binding_dn', 'settlement_imbalance',
             # prop:opap(i)/prop:eps measured
             'stab_method', 'stab_excess_sigma', 'stab_holds_sigma',
             'stab_excess_owen', 'stab_holds_owen', 'stab_eps_lr',
             'stab_worst_S_size', 'stab_time_s', 'stab_certified',
             'peak_value', 'peak_cost', 'sum_individual_peaks', 'coincidence_factor',
             'peak_netting_saving', 'schema_checks']
ROWGEN_COLS = ['run', 'day', 'n_players', 'converged', 'vstar_is_lower_bound',
               'cost_of_stability_vstar', 'weak_eps', 'time_rowgen_s', 'n_coalitions']

# --------------------------------------------------------------------------- Owen phase
def owen_day(run, day):
    players = run['players']
    params = build_params(run, day)
    t0 = time.time()
    lem = LocalEnergyMarket(players, T, params, model_type='mip')
    lem.model.hideOutput()
    ret = lem.solve_complete_model(analyze_revenue=False)
    results_ip = ret[1]
    v_mip = float(lem.model.getObjVal())
    t_mip = time.time() - t0

    init_priv = {k: v for k, v in results_ip.items() if isinstance(v, dict)}
    t0 = time.time()
    cg = ColumnGenerationSolver(players, T, params, model_type='mip',
                                init_sol=init_priv, smoothing=True)
    _, solution, v_chp, _ = cg.solve()
    t_cg = time.time() - t0

    owen_res = cg.compute_owen_allocation(v_mip)
    gap, eps_bound = owen_res['gap'], owen_res['eps']

    # prop:opap(i) / prop:eps measured at fixed chi (one separation MIP each)
    # 3600 s per allocation. The separation underneath is unbounded otherwise, and a
    # single hard day would hold the whole sweep.
    # mipsolver=SEP_SOLVER, the same separation oracle row generation uses. Left at the
    # default it fell through to SCIP, which is both slower here and -- until the fix
    # alongside this one -- the path on which the time limit was ignored.
    stab = check_allocations(players, T, params, owen_res['sigma'], owen_res['owen'],
                             gap, verbose=False, mipsolver=SEP_SOLVER,
                             time_limit=STAB_TIME_LIMIT)

    # reserve.txt sec.3.2-3.4 schema: primal from the MIP, duals from the master,
    # plus one small MIP per prosumer for the stand-alone pooling baseline
    solo = (solve_standalone_r_sym(players, T, params, model_type='mip')
            if STANDALONE_BASELINE and params.get('enable_reserve') else None)
    rp = reserve_peak_metrics(results_ip, params, players, T,
                              prices=solution.get('convex_hull_prices', {}),
                              standalone=solo)

    # offline CHP-settlement ingredients per (run, day)
    doc = {
        'run': run['name'], 'day': day, 'n_players': len(players), 'T': len(T),
        'reserve_peak': {'pi_res': params.get('pi_res'),
                         'pi_E_peak': params.get('pi_E_peak')},
        'v_mip': v_mip, 'v_chp': v_chp, 'gap': gap, 'eps_bound_gap_over_N': eps_bound,
        'time_mip_s': t_mip, 'time_cg_s': t_cg,
        'owen_sigma_cost': owen_res['sigma'], 'owen_alloc_cost': owen_res['owen'],
        'master_coupling_duals': solution.get('convex_hull_prices', {}),
        'mip_quantities': {k: _qty(results_ip, k, players)
                           for k in ('i_E_gri', 'e_E_gri', 'r_plus', 'r_minus')},
        'reserve_peak_metrics': rp,
        'stability_check': stab,
    }
    with open(os.path.join(run_dir(run), f"cg_day{day}.json"), 'w') as f:
        json.dump(_jsonable(doc), f, indent=2)

    row = {'run': run['name'], 'day': day, 'n_players': len(players),
           'num_households': dict(run['base'], **run['ov']).get('num_households', ''),
           'base_h2_price_eur': (dict(run['base'], **run['ov']).get('base_h2_price_eur', '')),
           'import_factor': (dict(run['base'], **run['ov']).get('import_factor', '')),
           'pi_res': params.get('pi_res'), 'pi_peak': params.get('pi_E_peak'),
           'v_mip': v_mip, 'v_chp': v_chp, 'gap': gap, 'eps_bound': eps_bound,
           'time_mip_s': round(t_mip, 2), 'time_cg_s': round(t_cg, 2)}
    row.update(flatten_for_csv(rp))
    row.update(stab_for_csv(stab))
    bad = failed_checks(rp)
    row['schema_checks'] = 'ok' if not bad else ';'.join(bad)
    # r_sym is one scalar under a 24-hour product and one value PER BLOCK under any
    # shorter one, so it cannot be formatted as a float. That is the third place the
    # block change has broken -- see convergence.md sec.3 for the warm start and
    # solver.private_dispatch for the fix that generalised there. Here the whole Owen
    # phase aborted on the log line, after the JSON had already been written, so 31 days
    # produced correct files and no CSV rows.
    _rs = rp.get('r_sym', 0.0)
    _rs_txt = (f"{min(_rs.values()):.4f}-{max(_rs.values()):.4f} over {len(_rs)} blocks"
               if isinstance(_rs, dict) else f"{_rs:.4f}")
    print(f"  [Owen] {run['name']} day {day}: v_mip={v_mip:.3f} eps_bound={eps_bound:.5f} "
          f"r_sym={_rs_txt} peak={rp.get('peak_value', 0.0):.4f} "
          f"checks={row['schema_checks']} stab={'ok' if stab['all_hold'] else 'FAIL'} "
          f"({t_mip+t_cg:.0f}s)")
    return row

# --------------------------------------------------------------------------- RowGen phase
def rowgen_day(run, day, budget):
    players = run['players']
    params = build_params(run, day)
    cc = CoreComputation(players, 'mip', T, params, mipsolver=SEP_SOLVER)
    t0 = time.time()
    _, success = cc.compute_core(max_iterations=int(1e8), tolerance=1e-6, time_limit=budget)
    t_rg = time.time() - t0
    v_star = getattr(cc, 'cost_of_stability_value', None)
    weak_eps = getattr(cc, 'weak_eps', None)
    converged = bool(getattr(cc, 'cos_converged', False))
    row = {'run': run['name'], 'day': day, 'n_players': len(players),
           'converged': converged, 'vstar_is_lower_bound': (not converged),
           'cost_of_stability_vstar': (float(v_star) if v_star is not None else ''),
           'weak_eps': (float(weak_eps) if weak_eps is not None else ''),
           'time_rowgen_s': round(t_rg, 2),
           'n_coalitions': len(cc.coalition_costs)}
    print(f"  [RowGen] {run['name']} day {day}: converged={converged} "
          f"weak_eps={weak_eps} ({t_rg:.0f}s, {len(cc.coalition_costs)} coalitions)")
    return row

# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['owen', 'rowgen', 'both'], default='both')
    ap.add_argument('--runs', default='', help='comma list of run names (overrides --groups)')
    ap.add_argument('--groups', default='core,param6p',
                    help="run groups: core | param6p | scenario | all "
                         "(default core,param6p; the sec.3.1 scenario axes are opt-in)")
    ap.add_argument('--days', default='', help='comma list of days (default: 1..31)')
    ap.add_argument('--force', action='store_true')
    # Per-day row-generation budget, overriding the per-run default. The defaults are
    # sized so a full core+param6p sweep finishes overnight; a single run reported in the
    # paper is worth more time than that, and a budget quoted in the manuscript has to be
    # the same at every size it is quoted for.
    ap.add_argument('--budget', type=int, default=0,
                    help='per-day row-generation budget in seconds (default: per-run)')
    args = ap.parse_args()

    run_filter = set(args.runs.split(',')) if args.runs else None
    groups = set(args.groups.split(',')) if args.groups else set()
    if 'all' in groups:
        groups = {r['group'] for r in RUNS}
    days = [int(d) for d in args.days.split(',')] if args.days else DAYS

    selected = [r['name'] for r in RUNS
                if (r['name'] in run_filter if run_filter else r['group'] in groups)]
    print(f"runs: {len(selected)} x {len(days)} days = {len(selected) * len(days)} rows"
          f"   groups={sorted(groups) if not run_filter else 'n/a (--runs given)'}")
    print(f"  {', '.join(selected)}")

    for run in RUNS:
        if run['name'] not in selected:
            continue
        rd = run_dir(run)
        owen_csv = os.path.join(rd, 'owen.csv')
        rowgen_csv = os.path.join(rd, 'rowgen.csv')

        # Phase 1: Owen for all days
        if args.phase in ('owen', 'both'):
            done = set() if args.force else existing_days(owen_csv)
            todo = [d for d in days if d not in done]
            print(f"\n=== [Owen phase] {run['name']}: {len(todo)} days ===")
            for d in todo:
                try:
                    append_row(owen_csv, owen_day(run, d), OWEN_COLS)
                except Exception as e:
                    print(f"  !! Owen {run['name']} day {d} FAILED: {e}")

        # Phase 2: RowGen for all days
        if args.phase in ('rowgen', 'both'):
            done = set() if args.force else existing_days(rowgen_csv)
            todo = [d for d in days if d not in done]
            budget = args.budget or run['budget']
            print(f"\n=== [RowGen phase] {run['name']}: {len(todo)} days (budget {budget}s) ===")
            for d in todo:
                try:
                    append_row(rowgen_csv, rowgen_day(run, d, budget), ROWGEN_COLS)
                except Exception as e:
                    print(f"  !! RowGen {run['name']} day {d} FAILED: {e}")

    print("\nDONE.")

if __name__ == "__main__":
    main()

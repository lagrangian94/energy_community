"""
Aggregate the multi-day Owen-vs-RowGen results into tidy CSVs + a printed table.

For each run <name>/ it merges owen.csv + rowgen.csv on `day` into <name>/summary.csv,
then writes a cross-run SUMMARY.csv (one row per run, aggregated over days) and prints it.

Owen eps = |gap|/N (upper bound, always). RowGen weak_eps = v*/n; EXACT when converged,
a LOWER BOUND when not (bracket: rowgen_weak_eps <= eps_min <= owen_eps_bound). Runs fine on
partial results — call it anytime.

Usage:  python weak_eps_experiment/summarize.py
"""
import os, glob
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

# run order (matches run_multiday.RUNS); any missing dir is skipped
ORDER = ['baseline_6p', 'low_h2_margin_6p', 'full_storage_6p',
         'community_size_350_6p', 'community_size_1000_6p', 'export_cap_020_6p',
         'baseline_15p', 'baseline_30p',
         'channel_balance_6p', 'channel_reserve_6p', 'channel_peak_6p',
         'reserve_low_6p', 'peak_200_6p',
         'low_h2_reserve0_6p', 'low_h2_reserve11_6p']

# reserve.txt sec.3.1 cells (mirrors run_multiday.CHANNEL_CELLS / *_SWEEP)
CHANNEL_CELLS = {'balance': 'channel_balance_6p', 'reserve_only': 'channel_reserve_6p',
                 'peak_only': 'channel_peak_6p', 'both': 'baseline_6p'}
RESERVE_SWEEP = {0.0: 'channel_peak_6p', 11.0: 'reserve_low_6p', 56.0: 'baseline_6p'}
PEAK_SWEEP = {0.0: 'channel_reserve_6p', 150.0: 'baseline_6p', 200.0: 'peak_200_6p'}
LOW_H2_SWEEP = {0.0: 'low_h2_reserve0_6p', 11.0: 'low_h2_reserve11_6p', 56.0: 'low_h2_margin_6p'}

def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

per_run_rows = []
for name in ORDER:
    d = os.path.join(OUT, name)
    owen, rg = _read(os.path.join(d, 'owen.csv')), _read(os.path.join(d, 'rowgen.csv'))
    if owen.empty and rg.empty:
        continue
    # merge on day
    if not owen.empty and not rg.empty:
        m = owen.merge(rg, on=['run', 'day', 'n_players'], how='outer', suffixes=('', '_rg'))
    else:
        m = owen if not owen.empty else rg
    m = m.sort_values('day')
    m.to_csv(os.path.join(d, 'summary.csv'), index=False)

    n_days = m['day'].nunique()
    row = {'run': name, 'n_players': int(m['n_players'].iloc[0]), 'n_days': n_days}
    if 'eps_bound' in m:
        row['owen_eps_bound_mean'] = round(m['eps_bound'].mean(), 5)
        row['owen_eps_bound_med'] = round(m['eps_bound'].median(), 5)
    if 'time_mip_s' in m and 'time_cg_s' in m:
        row['owen_time_mean_s'] = round((m['time_mip_s'] + m['time_cg_s']).mean(), 1)
    if 'converged' in m:
        conv = m['converged'].astype(str).str.lower().isin(['true', '1'])
        row['rowgen_converged'] = f"{int(conv.sum())}/{int(m['converged'].notna().sum())}"
    if 'weak_eps' in m:
        row['rowgen_weak_eps_mean'] = f"{pd.to_numeric(m['weak_eps'], errors='coerce').mean():.2e}"
    if 'time_rowgen_s' in m:
        row['rowgen_time_mean_s'] = round(pd.to_numeric(m['time_rowgen_s'], errors='coerce').mean(), 1)

    # reserve/peak channel metrics (reserve.txt sec.3.3), day-averaged.
    # Absent for runs produced before the schema was added -- those columns just
    # stay empty rather than breaking the table.
    for col, out, nd in [('r_sym', 'r_sym_mean', 4),
                         ('reserve_revenue', 'reserve_rev_mean', 1),
                         ('pool_standalone_sum', 'pool_standalone_mean', 4),
                         ('pool_gain_abs', 'pool_gain_mean', 4),
                         ('pool_gain_ratio', 'pool_gain_ratio_mean', 3),
                         ('mech_no_pooling', 'mech_alone_mean', 4),
                         ('mech_gain_time', 'mech_gain_time_mean', 4),
                         ('mech_gain_direction', 'mech_gain_dir_mean', 4),
                         ('mech_share_direction', 'mech_share_dir_mean', 3),
                         ('peak_value', 'peak_mean', 4),
                         ('coincidence_factor', 'coincidence_mean', 3),
                         ('peak_netting_saving', 'peak_netting_mean', 4)]:
        if col in m:
            v = pd.to_numeric(m[col], errors='coerce').mean()
            if pd.notna(v):
                row[out] = round(v, nd)
    if 'schema_checks' in m:
        ok = m['schema_checks'].astype(str).str.lower().eq('ok')
        row['schema_checks_ok'] = f"{int(ok.sum())}/{int(m['schema_checks'].notna().sum())}"
    per_run_rows.append(row)

summary = pd.DataFrame(per_run_rows)
summary.to_csv(os.path.join(OUT, 'SUMMARY.csv'), index=False)

pd.set_option('display.width', 200, 'display.max_columns', 30)
print("\n" + "=" * 90)
print("WEAK-ε-CORE EXPERIMENT SUMMARY  (Owen |gap|/N  vs  RowGen v*/n)")
print("bracket per instance:  rowgen_weak_eps  <=  eps_min  <=  owen_eps_bound")
print("=" * 90)
print(summary.to_string(index=False))
print(f"\nPer-run day-level detail: weak_eps_experiment/<run>/summary.csv")
print(f"Cross-run table:          weak_eps_experiment/SUMMARY.csv")


# --------------------------------------------------------------- sec.3.3 metrics 1 & 6
def _vn_by_day(run):
    """{day: v(N)} for a run. v_mip is a COST, so v(N) = -v_mip."""
    m = _read(os.path.join(OUT, run, 'owen.csv'))
    if m.empty or 'v_mip' not in m or 'day' not in m:
        return {}
    return {int(r['day']): -float(r['v_mip']) for _, r in m.iterrows()
            if pd.notna(r['v_mip'])}


def _paired(cells):
    """Day-aligned {label: {day: v(N)}} restricted to days present in EVERY cell.

    Returns (got, common). `got` always lists the cells that DO have data, even
    when there are too few to compare, so the caller can say what is missing.
    """
    got = {k: v for k, v in ((k, _vn_by_day(r)) for k, r in cells.items()) if v}
    if len(got) < 2:
        return got, []
    common = sorted(set.intersection(*(set(v) for v in got.values())))
    return got, common


def channel_decomposition():
    """sec.3.3 metric 1: value of each channel, as a v(N) difference on the SAME day.

    Isolating a channel needs a paired comparison, not a level: the reserve channel
    is worth v(both) - v(peak_only), and the peak channel v(both) - v(reserve_only).
    Their sum need not equal v(both) - v(balance) -- the shortfall is the interaction,
    reported as `joint - (reserve + peak)`.
    """
    got, common = _paired(CHANNEL_CELLS)
    if not common:
        missing = sorted(set(CHANNEL_CELLS) - set(got))
        print(f"\n[sec.3.3-1] channel decomposition skipped: have {sorted(got)}, "
              f"still need {missing} "
              f"(python weak_eps_experiment/run_multiday.py --runs "
              f"{','.join(CHANNEL_CELLS[k] for k in missing)})")
        return
    rows = []
    for d in common:
        v = {k: got[k][d] for k in got}
        r = {'day': d, **{f'v_{k}': round(v[k], 3) for k in v}}
        if {'both', 'peak_only'} <= v.keys():
            r['d_reserve'] = v['both'] - v['peak_only']
        if {'both', 'reserve_only'} <= v.keys():
            r['d_peak'] = v['both'] - v['reserve_only']
        if {'both', 'balance'} <= v.keys():
            r['d_joint'] = v['both'] - v['balance']
        if {'d_reserve', 'd_peak', 'd_joint'} <= r.keys():
            r['interaction'] = r['d_joint'] - r['d_reserve'] - r['d_peak']
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'CHANNEL_DECOMPOSITION.csv'), index=False)
    num = df.drop(columns=['day']).mean().round(3)
    print("\n" + "=" * 90)
    print("sec.3.3-1  CHANNEL VALUE DECOMPOSITION  (v(N), EUR/day; mean over "
          f"{len(common)} paired days)")
    print("=" * 90)
    print(num.to_string())
    print(f"-> weak_eps_experiment/CHANNEL_DECOMPOSITION.csv")


def price_sweep(cells, label, unit):
    """v(N) along a price axis, day-paired."""
    named = {str(k): v for k, v in cells.items()}
    got, common = _paired(named)
    if not common:
        missing = sorted(set(named) - set(got))
        print(f"\n[{label}] sweep skipped: have prices {sorted(got, key=float)}, "
              f"still need {missing} -> runs {[named[k] for k in missing]}")
        return
    base = min(float(k) for k in got)
    print(f"\n--- {label} sweep ({len(common)} paired days) ---")
    for k in sorted(got, key=float):
        mean_v = sum(got[k][d] for d in common) / len(common)
        delta = mean_v - sum(got[str(base)][d] for d in common) / len(common)
        print(f"    {label}={float(k):6.1f} {unit}:  v(N) = {mean_v:10.3f}   "
              f"delta vs {base:.0f} = {delta:+9.3f}")


channel_decomposition()
price_sweep(RESERVE_SWEEP, 'reserve_price', 'EUR/MW.h')
price_sweep(PEAK_SWEEP, 'peak_penalty', 'EUR/MW')
price_sweep(LOW_H2_SWEEP, 'reserve_price @low_h2', 'EUR/MW.h')

"""
Emit the LaTeX bodies of tab:results, tab:runtime and tab:oos of the uncertainty
manuscript (manuscript_ieee_uncertainty.tex) from the KL runs, and the numbers its
text quotes.

Same rule as paper_tables.py: the tables are generated, not typed. Sources:
  stochastic/main17_xi/*_main.json   main runs (run_kl_main.py), |Omega| = 20
  stochastic/bench17_xi/bench.csv    benchmarks (run_kl_bench.py)
  stochastic/bench17_xi_cykim/bench.csv   coalition generation, on a second machine
  stochastic/oos17_xi/oos_*.json     out-of-sample values (run_kl_oos.py)

CONVENTIONS.
- Values (tab:results, tab:oos) are arithmetic means over the days; run times
  (tab:runtime) are geometric means of |x|, sign restored. The values were geometric
  means too until 2026-10-15, when tab:results and tab:oos became one table and the
  signed out-of-sample quantities, which admit no geometric mean, set the rule for both.
  A cell with no data prints \\TBD{--}; one that rests on fewer days than the caption
  says prints inside \\TBD{} with the count in a comment.
- The extensive form stops at a gap of 1e-4 or 30 minutes, and g and eps rest on
  its incumbent. A day whose extensive form was re-solved later (resolve_ef.py) is
  read from 'ef_first', the 30-minute plan, so every day follows the same rule
  (kl_experiment_plan.md, sec. 8.11).
- eps(chi) is not a row: without Dinkelbach (sec. 8.12) it is exact only where all
  coalitions are valued. A single member j gains val({j}) - chi_j by leaving, which
  bounds eps(chi) from below, and eps_K bounds it from above; how often the two meet
  is counted in the comments below the tables, for the text to quote.
- tab:oos is the budget side of Proposition prop:true. 'promised' is the value the
  plan was chosen by (v^DR(N), or the sample-average value at r = 0) and 'delivered'
  the mean of the same first stage over the test sample; the budget holds on a day
  if delivered >= promised, and (delivered - promised) / n is the even shift of
  eq. (true_bound) with v(N) replaced by what the plan earns. These are signed, so
  the means are arithmetic. The cost of hedging is the paired difference of the two
  plans on the same test scenarios, with the standard error of its mean over days.

Usage:  python ieee_owen/weak_eps_experiment/kl_paper_tables.py [--out ieee_owen/kl_tables.tex]
"""
import os, glob, json, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(_HERE, 'stochastic', 'main17_xi')
BENCH = os.path.join(_HERE, 'stochastic', 'bench17_xi')
# coalition generation ran on another machine (kl_experiment_plan.md, sec. 8.14): 31 days
# at n=6, days 1-3 at n=15, nothing beyond. Its times do not compare with the other rows
# one to one (about 1.5 times faster on this method). The manuscript's table carries no
# note on machines or days (sec. 8.16): the rows print bare numbers and the comments of
# the generated file say what each rests on.
COALGEN = os.path.join(_HERE, 'stochastic', 'bench17_xi_cykim')
OOS = os.path.join(_HERE, 'stochastic', 'oos17_xi')
SIZES = (6, 15, 30, 60)
DAYS = 31


def gm(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (a != 0)]
    if not len(a):
        return None
    s = -1.0 if (a < 0).all() else 1.0
    return s * float(np.exp(np.mean(np.log(np.abs(a)))))


def num(x, digits):
    if x is None:
        return r'\TBD{--}'
    s = f'{abs(x):,.{digits}f}'.replace(',', '{,}')
    return ('$-$' if x < 0 else '') + s


def main_rows(scen):
    out = {}
    for n in SIZES:
        r = {k: [] for k in ('val', 'vdr', 'g', 'eps', 't_ef', 't_rcg', 'single', 'capped')}
        for f in glob.glob(os.path.join(MAIN, f'{n}p_*_S{scen}_*_main.json')):
            j = json.load(open(f))
            ef = j['ef_first']['ef'] if 'ef_first' in j else j['ef']
            g = ef['obj'] - j['dw']['lb']
            owen, alone = j['allocation']['owen'], j['standalone']
            r['val'].append(-ef['obj'])
            r['vdr'].append(-ef['obj'] - sum(alone.values()))
            r['g'].append(g)
            r['eps'].append(g / n)
            r['t_ef'].append(ef['time_solve'])
            r['capped'].append(ef['status'] == 'timelimit')
            r['t_rcg'].append(j['dw']['time'])
            r['single'].append(max(alone[u] - (owen[u] - g / n) for u in owen))
        out[n] = r
    return out


def bench_rows(scen, folder=BENCH):
    path = os.path.join(folder, 'bench.csv')
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines()
    head = lines[0].split(',')
    for ln in lines[1:]:
        f = dict(zip(head, ln.split(',')))
        if int(f['scen']) == scen:
            out.setdefault((f['method'], int(f['n'])), []).append(f)
    return out


def oos_rows(scen):
    """Per size, the days with both plans valued: promised and delivered value of
    each, their standard errors, and the paired cost of hedging."""
    out = {}
    for n in SIZES:
        r = {k: [] for k in ('day', 'saa_gap', 'p_saa', 'd_saa', 'se_saa', 'p_dro', 'd_dro',
                             'se_dro', 'cost', 'cost_se', 'vdr', 'test')}
        for f in glob.glob(os.path.join(OOS, f'oos_n{n}_S{scen}_day*.json')):
            j = json.load(open(f))
            pl = j['plans']
            if 'saa' not in pl or 'dro' not in pl:
                continue
            day = j['day']
            m = glob.glob(os.path.join(MAIN, f'{n}p_*_day{day}_S{scen}_*_main.json'))
            mj = json.load(open(m[0]))
            r['day'].append(day)
            r['test'].append(j['test'])
            r['saa_gap'].append(pl['saa']['ef_gap'])
            for k in ('saa', 'dro'):
                r[f'p_{k}'].append(pl[k]['promised'])
                r[f'd_{k}'].append(pl[k]['oos_mean'])
                r[f'se_{k}'].append(pl[k]['oos_se'])
            pair = pl['dro']['paired_vs_saa']            # dro - saa on the same scenarios
            r['cost'].append(-pair['mean'])
            r['cost_se'].append(pair['se'])
            r['vdr'].append(pl['dro']['promised'] - sum(mj['standalone'].values()))
        out[n] = {k: np.array(v, float) for k, v in r.items()}
    return out


def am(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def cell(vals, digits, days=DAYS, mean=gm):
    """Mean over the days (geometric unless told); inside \\TBD{} while fewer than
    `days` days are in."""
    x = mean(vals)
    if x is None:
        return r'\TBD{--}'
    return num(x, digits) if len(vals) >= days else rf'\TBD{{{num(x, digits)}}}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', type=int, default=20)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(_HERE), 'kl_tables.tex'))
    a = ap.parse_args()
    R, B = main_rows(a.scen), bench_rows(a.scen)
    L = []
    w = L.append
    w(f'% GENERATED by weak_eps_experiment/kl_paper_tables.py (|Omega| = {a.scen}) -- do not edit')
    w('% days per size: ' + ', '.join(f'n={n}: {len(R[n]["g"])}' for n in SIZES))
    w('')
    w('% ---- tab:results')
    # arithmetic means: values, not run times (CONVENTIONS)
    row = lambda name, key, d: w(f'{name} & ' + ' & '.join(cell(R[n][key], d, mean=am)
                                                            for n in SIZES) + r' \\')
    row(r'$\vdr{N}$', 'val', 0)
    row(r'$v^{\mathrm{DR}}(N)$', 'vdr', 0)
    row(r'$g_{r,K}^{\mathrm{LR}}$', 'g', 2)
    # the text defines eps_K = g_K / n; the row carries the symbol only
    row(r'$\varepsilon_{r,K}^{\mathrm{LR}}$', 'eps', 3)
    w('')
    w('% ---- tab:runtime')
    w(r'Solving $\vdr{N}$ & ' + ' & '.join(cell(R[n]['t_ef'], 0) for n in SIZES) + r' \\')
    w(r'RCG, nested master & ' + ' & '.join(cell(R[n]['t_rcg'], 0) for n in SIZES) + r' \\')
    cg, conv = [], []
    C = bench_rows(a.scen, COALGEN)
    out_below = False               # out of budget on every day run at a smaller size
    for n in SIZES:
        rows = C.get(('coalgen', n), [])
        ok = [float(f['alg_s']) for f in rows if f['status'] == 'converged']
        # a size that was not run prints as out of budget once a smaller size is, on every
        # day it ran: the separation only grows with n. The comment below says which
        # cells are measured and which are this inference.
        cg.append(num(gm(ok), 0) if ok else
                  (r'$>3{,}600$' if rows or out_below else '--'))
        conv.append(f'{len(ok)}/{len(rows)}' if rows else
                    ('not run (inferred from the smaller size)' if out_below else 'not run'))
        out_below = out_below or (bool(rows) and not ok)
    # time to the bound: first moment the Lagrangian bound is within 5% of omega of the
    # RCG's final bound (run_kl_bench.py), on the days the benchmarks ran. The RCG's
    # own time by the same measure comes from the main runs' logs.
    import run_kl_bench as RB
    here = os.getcwd()

    def to_bound(method, n):
        rows = B.get((method, n), [])
        got = [float(f['t_target']) for f in rows if f['status'] == 'target']
        return got, len(rows)

    days_run = sorted({int(f['day']) for (m, _), rs in B.items() if m in ('conic', 'flat')
                       for f in rs})
    rcg = {}
    for n in SIZES:
        for d in days_run:
            f = glob.glob(os.path.join(MAIN, f'{n}p_*_day{d}_S{a.scen}_*_main.json'))
            if f:
                j = json.load(open(f[0]))
                rcg[(n, d)] = RB.time_to(j['cg_log'], RB.target(n, a.scen, d), j['dw']['time'])
    os.chdir(here)
    w(r'\midrule')
    w(f'% time to the bound, days {days_run}: geometric mean over the days it is reached '
      f'(reached/run)')
    # the RCG's own time to the bound is the reference of the ratios below, not a row
    w('% RCG, nested master, by the same measure: '
      + ', '.join(f'n={n}: {gm([rcg[(n, d)] for d in days_run if (n, d) in rcg]):.0f}'
                  for n in SIZES))
    ratio = {}
    for name, m in ((r'CG, conic master', 'conic'), (r'RCG, single master', 'flat')):
        cells, reach = [], []
        for n in SIZES:
            rows = B.get((m, n), [])
            hit = [f for f in rows if np.isfinite(float(f['t_target']))]
            got = [float(f['t_target']) for f in hit]
            reach.append(f'n={n}: {len(got)}/{len(rows)}' if rows else f'n={n}: not run')
            if not rows:
                cells.append('--')
            elif not got:
                cells.append(r'$>3{,}600$')
            else:
                cells.append(num(gm(got), 0))
                ratio[(m, n)] = gm(got) / gm([rcg[(n, int(f['day']))] for f in hit])
        w(f'{name} & ' + ' & '.join(cells) + r' \\')
        w(f'%   bound reached on (days/run): ' + ', '.join(reach))
    os.chdir(here)
    w(r'\midrule')
    w(r'Coalition generation & ' + ' & '.join(cg) + r' \\')
    w('%   timed on another machine (stochastic/bench17_xi_cykim), about 1.5 times faster on it')
    w('% coalition generation converged on (days/run): ' + ', '.join(
        f'n={n}: {c}' for n, c in zip(SIZES, conv)))
    w('')
    w('% ---- time to the bound relative to the RCG, on the days the method reaches it')
    for m in ('conic', 'flat'):
        w(f'% {m}: ' + ', '.join(
            f'n={n}: ' + (f'{ratio[(m, n)]:.1f}x' if ratio.get((m, n)) else 'not reached / not run')
            for n in SIZES))
    O = oos_rows(a.scen)
    # since 2026-10-15 the out-of-sample block of tab:results (one table, two blocks)
    w('% ---- tab:oos (arithmetic means over the days; signed)')
    w('% days per size: ' + ', '.join(f'n={n}: {len(O[n]["day"])}' for n in SIZES)
      + '; test scenarios: ' + ', '.join(sorted({f'{int(t)}' for n in SIZES for t in O[n]['test']})))

    def tbd(n, txt):
        return txt if len(O[n]['day']) >= DAYS else rf'\TBD{{{txt}}}'

    def signed(x, d):
        return ('$-$' if x < 0 else '') + f'{abs(x):.{d}f}'
    # grouped by quantity, one row per plan named as in fig:oos (DR, SAA); the manuscript
    # adds the group headers and the indents
    plans = ((r'SAA, $r=0$', 'saa'), (rf'DR, $r={RB.Q.radius(a.scen):.3f}$', 'dro'))
    w(r'% Days the budget holds (of 31)')
    for lab, k in plans:
        w(rf'{lab} & ' + ' & '.join(
            tbd(n, f'{int((O[n][f"d_{k}"] >= O[n][f"p_{k}"]).sum())}') for n in SIZES) + r' \\')
    w(r'% Shift per member (\euro)')
    for lab, k in plans:
        w(rf'{lab} & ' + ' & '.join(
            tbd(n, signed(float(np.mean((O[n][f'd_{k}'] - O[n][f'p_{k}']) / n)), 2))
            for n in SIZES) + r' \\')
    # what the DR plan earns below the SAA plan: a row of its own under the group header
    w(r'% Cost of hedging (\% of $v^{\mathrm{DR}}(N)$)')
    w(plans[1][0] + ' & ' + ' & '.join(
        tbd(n, f'{100 * float(np.mean(O[n]["cost"] / O[n]["vdr"])):.2f}') for n in SIZES) + r' \\')
    for n in SIZES:
        o = O[n]
        D = len(o['day'])
        if not D:
            continue
        cost, cse = float(o['cost'].mean()), float(np.sqrt((o['cost_se'] ** 2).sum()) / D)
        w(f'% n={n}: promised - delivered, mean EUR: r=0 {np.mean(o["p_saa"] - o["d_saa"]):+.1f}, '
          f'KL {np.mean(o["p_dro"] - o["d_dro"]):+.1f}; cost of hedging {cost:.2f} +- {cse:.2f} '
          f'EUR/day ({100 * cost / float(np.mean(np.abs(o["d_saa"]))):.3f}% of the delivered value, '
          f'{100 * float(np.mean(o["cost"] / o["vdr"])):.3f}% of v^DR(N))')
        for lab, k in (('r=0', 'saa'), ('KL', 'dro')):
            z = (o[f'd_{k}'] - o[f'p_{k}']) / o[f'se_{k}']
            w(f'%   {lab}: budget holds on {int((z >= 0).sum())}/{D}; by more than 1.96 s.e. of the '
              f'test mean on {int((z > 1.96).sum())}, fails by more than that on '
              f'{int((z < -1.96).sum())}, within it on {int((np.abs(z) <= 1.96).sum())}; '
              f'shortfall per member on the days it fails: mean '
              + (f'{float(np.mean(((o[f"p_{k}"] - o[f"d_{k}"]) / n)[z < 0])):.2f}' if (z < 0).any() else '--')
              + f', max {float(np.max((o[f"p_{k}"] - o[f"d_{k}"]) / n)):.2f} EUR')
        cap = o['saa_gap'] > 1.0001e-4
        w(f'%   sample-average extensive form above a gap of 1e-4 on {int(cap.sum())} days'
          + (f' {sorted(int(d) for d in o["day"][cap])}' if cap.any() else ''))
    w('% ---- quoted in the text')
    g = {n: am(R[n]['g']) for n in SIZES}
    e = {n: am(R[n]['eps']) for n in SIZES}
    v = {n: am(R[n]['vdr']) for n in SIZES}
    w(f'% duality gap, arithmetic mean: ' + ', '.join(f'n={n}: {g[n]:.2f}' for n in SIZES))
    w(f'% eps_K, arithmetic mean: ' + ', '.join(f'n={n}: {e[n]:.3f}' for n in SIZES))
    w(f'% eps_K as a share of the mean surplus per member, v^DR(N) / n: '
      + ', '.join(f'n={n}: {100 * e[n] / (v[n] / n):.3f}%' for n in SIZES))
    w(f'% RCG time, geometric mean [s]: '
      + ', '.join(f'n={n}: {gm(R[n]["t_rcg"]):.0f}' for n in SIZES))
    for n in SIZES:
        s, ee = np.array(R[n]['single']), np.array(R[n]['eps'])
        w(f'% n={n}: a single member gains on {int((s > 0).sum())}/{len(s)} days; within 0.1% '
          f'of eps_K on {int((s >= 0.999 * ee).sum())}, above eps_K on {int((s > ee * (1 + 1e-9)).sum())}; '
          f'extensive form at its 30-minute cap on '
          f'{sum(R[n]["capped"])} days')
    text = '\n'.join(L) + '\n'
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(text)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()

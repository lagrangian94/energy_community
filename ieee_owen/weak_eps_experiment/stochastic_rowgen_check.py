"""
Validation of the stochastic separation and row generation (ieee_owen/stochastic_core.py).

Parts (results as JSON/CSV under stochastic/rowgen_check/):

  deact  n=6, every one of the 63 coalitions S: the separation MILP with z fixed to 1_S
         (payoffs 0) must cost exactly c(S). Deterministic: core.SeparationProblem vs
         LocalEnergyMarket(S); stochastic |Omega| in {1, 3}: StochasticSeparation vs
         solve_extensive_form(S). Both sides at a 1e-7 gap, so a difference is a
         modelling difference, not MIP tolerance. Also lists every nonzero variable of a
         non-member in the fixed separation's solution (the "fully inactive" check).
  a      |Omega| = 1 (sigmas 0): deterministic CoreComputation vs
         StochasticCoreComputation on the same instance, cost-of-stability row
         generation (omega*, allocation, coalitions added), and the weak eps of the
         deterministic Owen point (solve_deterministic) measured by both separations.
  cross  after a: each run's final allocation through both separations (with omega* = 0
         the master's optimal face is large, so the two runs may stop at different
         core points; this checks each against the other's separation).
  b      |Omega| = 3 (default sigmas, seed 0), n = 6: separation vs enumeration of all 62
         proper coalitions' extensive forms, for four allocations (raw Owen sigma,
         gap-corrected Owen E[x*], equal bill c(N)/n, equal surplus kappa + v(N)/n);
         row-generation omega* vs the OPAP LP over every coalition, and the least-core
         value (v free, nonzero here) by row generation vs over all 62 rows.
  c      size and time of the separation MILP at |Omega| in {1, 3, 5} (one solve at the
         stochastic Owen point E[x*] of that |Omega|, over every S and over the proper
         S only), plus the Dinkelbach weak eps.

  KL-DRO game (StochasticCoreComputation(kl_radius=r), KLSeparation), n=6, day 1, seed 0:
  kldeact  the KL separation with z fixed to each of the 63 coalitions vs the KL
           extensive form of the coalition alone (--radius, --tight-gap).
  klb      part b for the KL game at --radius: r = 0 on part b's four allocations,
           compared with part b; r > 0 on the robust Owen point of the command line's
           KL run (sigma and E[x*]), equal bill and equal surplus of the KL game, plus
           the command line's own enumeration (measure_eps).
  klc      size, time and tangent rounds of the KL separation at the robust Owen point
           for --radii x --scen at --n (resumable; --force recomputes).

Usage (from anywhere):
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part deact
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part a --n 6
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part a --n 15 --time-limit 7200
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part cross --n 15
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part b
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part c --n 15
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part kldeact --radius 0.5
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part klb --radius 0
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part klb --radius 0.5
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part klc --n 6 --scen 3,5
  python ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py --part klc --n 15 --scen 3 --radii 0.5
"""
import os, sys, re, csv, json, time, argparse, itertools
_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

# compact_utility plots into the CWD while solving; not wanted here
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
import matplotlib.figure as _fig
_plt.savefig = lambda *a, **k: None
_fig.Figure.savefig = lambda *a, **k: None

import numpy as np
import stochastic_extension as SE
import stochastic_core as SC
from core import CoreComputation, SeparationProblem
from compact_utility import LocalEnergyMarket, solve_mip
from run_experiment import build_instance

OUT = os.path.join(_HERE, 'stochastic', 'rowgen_check')
DAY = 1
TIGHT = 1e-7


def _jsonable(x):
    return SE._jsonable(x)


def _dump(obj, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, 'w') as f:
        json.dump(_jsonable(obj), f, indent=1)
    print(f'wrote {path}')


def _csv(rows, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path}')


def scenarios(base, players, T, k, seed=0):
    if k == 1:
        return SE.make_scenarios(base, players, T, 1, seed=seed, wind_sigma=0.0,
                                 solar_sigma=0.0, load_sigma=0.0, price_sigma=0.0)
    return SE.make_scenarios(base, players, T, k, seed=seed)


def proper_coalitions(players):
    for r in range(1, len(players)):
        for S in itertools.combinations(players, r):
            yield list(S)


class _Trace:
    """Records every row added to the OPAP master (the n singletons first)."""
    def _add_coalition_constraint(self, coalition, cost_of_stability=False):
        self.added = getattr(self, 'added', [])
        self.added.append(sorted(coalition))
        return super()._add_coalition_constraint(coalition, cost_of_stability)


class DetCore(_Trace, CoreComputation):
    pass


class StoCore(_Trace, SC.StochasticCoreComputation):
    pass


def owen_point(players, T, scen, base):
    """Owen allocation of the (stochastic) game, cost convention: raw sigma and the
    gap-corrected E[x*]. |Omega| = 1: solve_deterministic (exact CG, omega_tol off);
    otherwise the direct engine with the command line's defaults."""
    t0 = time.time()
    if len(scen) == 1:
        d = SE.solve_deterministic(players, T, base)
        return {'sigma': d['sigma'], 'x': d['owen'], 'v_mip': d['v_mip'], 'v_lr': d['v_lr'],
                'eps_LR': d['eps'], 'omega_LR': abs(d['gap']), 'time': time.time() - t0}
    # The command line's defaults, omega_tol = 2% included. With omega_tol off the
    # direct engine tails off at n=6, |Omega|=3, day 1: RMP -2565.1301 against LB
    # -2565.1360 after 7400 CG iterations (the 1e-6 CG gap is 2.6e-3), UB never finite.
    args = SE.build_parser().parse_args([])
    SE._configure_direct(args)
    ef = SE.solve_extensive_form(players, T, scen, gap=args.mip_gap, solver=args.mip_solver)
    dw, master = SE.solve_dwr_direct(players, T, scen, base, **SE._direct_kwargs(args, ef))
    al = SE.scenario_allocation(ef, dw, master)
    return {'sigma': {u: float(dw['sigma'][u]) for u in players},
            'x': {u: -float(al['Ex'][u]) for u in players},
            'v_mip': float(ef['obj']), 'v_lr': float(dw['obj']),
            'eps_LR': float(al['eps_LR']), 'omega_LR': float(al['omega_LR']),
            'budget_residual': float(al['budget_residual']),
            'duality_residual': float(al['duality_residual']), 'omega_tol': args.omega_tol,
            'cg_status': dw['status'], 'time': time.time() - t0}


def rowgen(cc, time_limit, max_iter=100):
    t0 = time.time()
    q, ok = cc.compute_core(cost_of_stability=True, time_limit=time_limit,
                            max_iterations=max_iter)
    n = len(cc.players)
    return {
        'omega_star': float(cc.cost_of_stability_value), 'weak_eps': float(cc.weak_eps),
        'converged': bool(cc.cos_converged), 'core_nonempty': bool(ok),
        'raw_p': dict(cc.cos_raw_payoffs), 'q': dict(q),
        'added': cc.added[n:], 'iterations': len(cc.added) - n + 1,
        'repeated_row_stops': getattr(cc, 'repeated_row_stops', 0),
        'kappa': {u: cc.coalition_costs[(u,)] for u in cc.players},
        'c_N': cc.coalition_costs[tuple(sorted(cc.players))],
        'time': time.time() - t0,
    }


def least_core(cc, coalitions=None, tol=1e-6, max_iter=500):
    """Least-core value min v s.t. sum p = c(N), sum_S p <= c(S) + v for every proper
    S, with v FREE (negative when the core has interior), on SCIP's LP.

    coalitions=None: row generation, separating with StochasticSeparation restricted
    to proper coalitions (min_size=1, max_size=n-1): the most violated row is the one
    with the largest excess, violated iff that excess exceeds v. Otherwise the LP over
    the given list (the all-coalition reference). A discriminating test of separation
    inside row generation when the cost of stability is 0, as it is at n=6.
    """
    from pyscipopt import Model, quicksum
    players, n = cc.players, len(cc.players)
    m = Model('least_core')
    m.hideOutput()
    p = {u: m.addVar(name=f'p_{u}', lb=None) for u in players}
    v = m.addVar(name='v', lb=None, obj=1.0)
    m.addCons(quicksum(p.values()) == cc.compute_coalition_cost(players))
    rows = set()

    def add(S):
        key = tuple(sorted(S))
        m.freeTransform()
        m.addCons(quicksum(p[u] for u in key) <= cc.compute_coalition_cost(list(key)) + v)
        rows.add(key)

    for S in (coalitions if coalitions is not None else [[u] for u in players]):
        add(S)
    tol_eff = tol * (1 + abs(cc.compute_coalition_cost(players)))
    it, t0, stop = 0, time.time(), 'all rows'
    while True:
        m.optimize()
        pv, vv = {u: m.getVal(p[u]) for u in players}, m.getVal(v)
        if coalitions is not None:
            break
        it += 1
        sep = cc.make_separation(pv)
        S, exc = sep.solve(min_size=1, max_size=n - 1)
        last_bound = sep.stats['violation_bound']
        if exc - vv <= tol_eff:
            stop = 'converged' if not sep.truncated else 'UNCERTIFIED (separation truncated)'
            break
        if tuple(sorted(S)) in rows:
            stop = f'repeated row (separation excess {exc:.6f}, v {vv:.6f})'
            break
        if it >= max_iter:
            stop = 'max_iter'
            break
        add(S)
    return {'value': vv, 'p': pv, 'rows': len(rows), 'iterations': it, 'stop': stop,
            'last_separation_bound': None if coalitions is not None else last_bound,
            'time': time.time() - t0}


def weak_eps(cc, alloc, time_limit):
    t0 = time.time()
    S, eps, imp = cc.measure_stability_violation(alloc, time_limit=time_limit)
    return {'coalition': sorted(S), 'eps': float(eps), 'is_imputation': bool(imp),
            'certified': not getattr(cc, 'last_stability_truncated', False),
            'time': time.time() - t0}


# =============================================================================
_PLAYER = re.compile(r'_(u\d+)(?:_|$)')
_FREE = ('z_off_G_', 'z_sd_G_', 'z_sd_H_', 'y_cc_')   # zero-cost, not gated on purpose


def _nonmember_nonzeros(vals, members, players):
    """Nonzero variables of players outside `members`, grouped by family."""
    out = {}
    nonm = set(players) - set(members)
    for name, x in vals.items():
        if abs(x) <= 1e-7:
            continue
        m = _PLAYER.search(name)
        if not m or m.group(1) not in nonm:
            continue
        fam = re.sub(r'_u\d+.*$', '', name)
        key = fam + (' (free by design)' if name.startswith(_FREE) else '')
        out[key] = out.get(key, 0) + 1
    return out


def part_deact(args):
    players, _, T, base, name = build_instance(6, day=DAY)
    rows, extra = [], {}
    all_S = [list(S) for r in range(1, len(players) + 1)
             for S in itertools.combinations(players, r)]

    # deterministic: SeparationProblem (z fixed) vs LocalEnergyMarket(S)
    sp = SeparationProblem(players, T, 'mip', base, {u: 0.0 for u in players},
                           mipsolver='gurobi')
    nz_det = {}
    for S in all_S:
        for u in players:
            x = 1.0 if u in S else 0.0
            sp.model.chgVarLb(sp.z[u], x)
            sp.model.chgVarUb(sp.z[u], x)
        st1, sep_c, vals = solve_mip(sp.model, 'gurobi', gap=TIGHT)
        lem = LocalEnergyMarket(S, T, base, model_type='mip', mipsolver='gurobi')
        st2, c, _ = solve_mip(lem.model, 'gurobi', gap=TIGHT)
        nz = _nonmember_nonzeros(vals, S, players)
        for k, v in nz.items():
            nz_det[k] = nz_det.get(k, 0) + v
        rows.append({'model': 'det', 'S': '+'.join(S), 'sep_cost': sep_c, 'c_S': c,
                     'diff': sep_c - c, 'status': f'{st1}/{st2}'})
    extra['det_nonmember_nonzeros'] = nz_det

    for k in (1, 3):
        scen = scenarios(base, players, T, k)
        sep = SC.StochasticSeparation(players, T, scen, {u: 0.0 for u in players})
        nz_s = {}
        for S in all_S:
            _, v = sep.solve(fix=S, gap=TIGHT)
            st1 = sep.stats['status']
            nz = _nonmember_nonzeros(sep.vals, S, players)
            for kk, vv in nz.items():
                nz_s[kk] = nz_s.get(kk, 0) + vv
            ef = SE.solve_extensive_form(S, T, scen, gap=TIGHT, solver='gurobi')
            rows.append({'model': f'sto{k}', 'S': '+'.join(S), 'sep_cost': -v,
                         'c_S': ef['obj'], 'diff': -v - ef['obj'],
                         'status': f"{st1}/{ef['status']}"})
        extra[f'sto{k}_nonmember_nonzeros'] = nz_s
        extra[f'sto{k}_size'] = {kk: sep.stats[kk] for kk in ('vars', 'bin_vars', 'rows')}

    _csv(rows, 'deact_n6.csv')
    summary = {}
    for mdl in sorted({r['model'] for r in rows}):
        d = [abs(r['diff']) for r in rows if r['model'] == mdl]
        rel = [abs(r['diff']) / max(1.0, abs(r['c_S'])) for r in rows if r['model'] == mdl]
        summary[mdl] = {'coalitions': len(d), 'max_abs_diff': max(d), 'max_rel_diff': max(rel)}
    out = {'instance': name, 'day': DAY, 'gap': TIGHT, 'summary': summary, **extra}
    _dump(out, 'deact_n6.json')
    print('\nSUMMARY deact', json.dumps(_jsonable(out), indent=1))


# =============================================================================
def part_a(args):
    n = args.n
    players, _, T, base, name = build_instance(n, day=DAY)
    scen = scenarios(base, players, T, 1)
    same = all(base.get(k) == scen[0][1].get(k) for k in set(base) | set(scen[0][1]))
    out = {'instance': name, 'n': n, 'day': DAY, 'scenario_equals_base': same,
           'time_limit': args.time_limit, 'max_iter': args.max_iter}

    owen = owen_point(players, T, scen, base)
    out['owen'] = owen

    det = DetCore(players, 'mip', T, base, mipsolver='gurobi')
    out['det_rowgen'] = rowgen(det, args.time_limit, args.max_iter)
    sto = StoCore(players, T, scen)
    out['sto_rowgen'] = rowgen(sto, args.time_limit, args.max_iter)
    out['sto_sep_log'] = sto.sep_log

    for label in ('x', 'sigma'):
        out[f'det_weak_eps_{label}'] = weak_eps(det, owen[label], args.eps_time_limit)
        out[f'sto_weak_eps_{label}'] = weak_eps(sto, owen[label], args.eps_time_limit)

    # the common coalition values both row generations saw
    common = sorted(set(det.coalition_costs) & set(sto.coalition_costs))
    out['coalition_value_diff'] = {'+'.join(S): sto.coalition_costs[S] - det.coalition_costs[S]
                                   for S in common}
    d, s = out['det_rowgen'], out['sto_rowgen']
    out['compare'] = {
        'omega_star': [d['omega_star'], s['omega_star'], s['omega_star'] - d['omega_star']],
        'c_N': [d['c_N'], s['c_N'], s['c_N'] - d['c_N']],
        'max_kappa_diff': max(abs(s['kappa'][u] - d['kappa'][u]) for u in players),
        'max_alloc_diff_q': max(abs(s['q'][u] - d['q'][u]) for u in players),
        'same_added': d['added'] == s['added'],
        'same_added_set': sorted(map(tuple, d['added'])) == sorted(map(tuple, s['added'])),
        'max_coalition_value_diff': max(abs(v) for v in out['coalition_value_diff'].values()),
        'weak_eps_x': [out['det_weak_eps_x']['eps'], out['sto_weak_eps_x']['eps']],
        'weak_eps_sigma': [out['det_weak_eps_sigma']['eps'], out['sto_weak_eps_sigma']['eps']],
    }
    _dump(out, f'part_a_n{n}.json')
    print('\nSUMMARY a', json.dumps(_jsonable(out['compare']), indent=1))


def part_cross(args):
    """After part a: each row generation's final raw allocation p (sum p = c(N)) through
    BOTH separations. With omega* = 0 the master's optimal face is large and the two
    runs may stop at different core points; this shows each is in the other's core.
    Adds 'cross' to part_a_n<n>.json."""
    n = args.n
    path = os.path.join(OUT, f'part_a_n{n}.json')
    with open(path) as f:
        out = json.load(f)
    players, _, T, base, name = build_instance(n, day=DAY)
    scen = scenarios(base, players, T, 1)
    det = CoreComputation(players, 'mip', T, base, mipsolver='gurobi')
    sto = SC.StochasticCoreComputation(players, T, scen)
    cross = {}
    for src in ('det', 'sto'):
        p = out[f'{src}_rowgen']['raw_p']
        for lab, cc in (('det', det), ('sto', sto)):
            S, v = cc.find_violated_coalition(p)
            sep = cc.last_actual_violation
            cross[f'{lab}_separation_at_{src}_p'] = {
                'coalition': sorted(S), 'violation': float(v),
                'recomputed': None if sep is None else float(sep),
                'tol_eff': 1e-6 * (1 + abs(sum(p.values())))}
    out['cross'] = cross
    _dump(out, os.path.basename(path))
    print('\nSUMMARY cross', json.dumps(_jsonable(cross), indent=1))


# =============================================================================
def enumeration_block(cc, allocs_fn, args, csv_name):
    """Separation vs enumeration of all 62 proper coalitions, OPAP and least core, for
    the game of `cc` (risk neutral or KL). Row generation runs first, then every proper
    coalition's value is computed (cached), so all comparisons read the same values.
    allocs_fn(cN, kappa) -> {label: allocation (cost convention)}."""
    players, T, scen, n = cc.players, cc.time_periods, cc.scenarios, len(cc.players)
    out = {'rowgen': rowgen(cc, args.time_limit, args.max_iter)}
    out['sep_log_rowgen'] = list(cc.sep_log)

    t0 = time.time()
    for S in proper_coalitions(players):
        cc.compute_coalition_cost(S)
    out['time_enumeration'] = time.time() - t0
    table = {'+'.join(S): cc.coalition_costs[tuple(sorted(S))] for S in proper_coalitions(players)}
    out['coalition_values'] = table
    out['ef_info'] = {'+'.join(k): v for k, v in cc.ef_info.items()}
    cN = cc.coalition_costs[tuple(sorted(players))]
    kappa = {u: cc.coalition_costs[(u,)] for u in players}
    allocs = allocs_fn(cN, kappa)
    out['allocations'] = allocs

    rows = []
    for label, a in allocs.items():
        ex = {S: sum(a[u] for u in S.split('+')) - c for S, c in table.items()}
        pc = {S: e / len(S.split('+')) for S, e in ex.items()}
        S_raw, S_pc = max(ex, key=ex.get), max(pc, key=pc.get)

        sep = cc.make_separation(a)
        S_sep, v_sep = sep.solve(min_size=1, max_size=n - 1)
        st_proper, tr_proper = dict(sep.stats), sep.truncated
        S_all, v_all = sep.solve()
        st_all, tr_all = dict(sep.stats), sep.truncated
        we = weak_eps(cc, a, args.eps_time_limit)
        enum_all = max(0.0, ex[S_raw], sum(a.values()) - cN)   # empty set and N included
        rows.append({
            'allocation': label, 'sum': sum(a.values()),
            'enum_max_excess': ex[S_raw], 'enum_argmax': S_raw,
            'sep_proper_excess': v_sep, 'sep_proper_S': '+'.join(S_sep),
            'diff_proper': v_sep - ex[S_raw],
            'sep_proper_bound': st_proper['violation_bound'],
            'sep_proper_truncated': tr_proper,
            'enum_max_excess_incl_empty_N': enum_all,
            'sep_all_excess': v_all, 'sep_all_S': '+'.join(S_all),
            'diff_all': v_all - enum_all, 'sep_all_bound': st_all['violation_bound'],
            'sep_all_truncated': tr_all,
            'enum_weak_eps': pc[S_pc], 'enum_weak_eps_S': S_pc,
            'dinkelbach_weak_eps': we['eps'], 'dinkelbach_S': '+'.join(we['coalition']),
            'dinkelbach_certified': we['certified'],
            'sep_time_proper': st_proper['time_solve'], 'sep_time_all': st_all['time_solve'],
            'sep_rounds_proper': st_proper.get('rounds'), 'sep_rounds_all': st_all.get('rounds'),
        })
    _csv(rows, csv_name)
    out['separation_vs_enumeration'] = rows

    rg = out['rowgen']
    # the all-coalition OPAP LP, on the same cached values (inherited brute force)
    q_bf, ok_bf = cc.compute_core_brute_force(cost_of_stability=True)
    bf = {'omega_star': float(cc.cost_of_stability_value), 'q': dict(q_bf)}
    p = rg['raw_p']
    worst = max(sum(p[u] for u in S.split('+')) - c for S, c in table.items())
    out['opap'] = {
        'rowgen_omega_star': rg['omega_star'], 'all_coalitions_omega_star': bf['omega_star'],
        'diff': rg['omega_star'] - bf['omega_star'],
        'rowgen_rows': len(rg['added']) + n, 'all_rows': len(table),
        'rowgen_converged': rg['converged'], 'rowgen_repeated_row_stops': rg['repeated_row_stops'],
        'rowgen_p_max_violation_over_all_62': worst,
        'max_alloc_diff_q': max(abs(rg['q'][u] - bf['q'][u]) for u in players),
        'rowgen_q': rg['q'], 'all_q': bf['q'],
    }
    lc_rg = least_core(cc)
    lc_all = least_core(cc, coalitions=list(proper_coalitions(players)))
    worst = max((sum(lc_rg['p'][u] for u in S.split('+')) - c) for S, c in table.items())
    out['least_core'] = {
        'rowgen_value': lc_rg['value'], 'all_coalitions_value': lc_all['value'],
        'diff': lc_rg['value'] - lc_all['value'], 'rowgen_rows': lc_rg['rows'],
        'rowgen_iterations': lc_rg['iterations'], 'rowgen_stop': lc_rg['stop'],
        'rowgen_last_separation_bound': lc_rg['last_separation_bound'],
        'rowgen_p_max_excess_over_all_62': worst,
        'max_alloc_diff': max(abs(lc_rg['p'][u] - lc_all['p'][u]) for u in players),
        'time': lc_rg['time'],
    }
    out['sep_log_all'] = list(cc.sep_log)
    return out


def _print_block(out, tag):
    print(f'\nSUMMARY {tag}')
    for r in out['separation_vs_enumeration']:
        print({k: r[k] for k in ('allocation', 'enum_max_excess', 'enum_argmax',
                                 'sep_proper_excess', 'sep_proper_S', 'diff_proper',
                                 'sep_proper_bound', 'enum_max_excess_incl_empty_N',
                                 'sep_all_excess', 'enum_weak_eps', 'dinkelbach_weak_eps')})
    print(json.dumps(_jsonable({k: v for k, v in out['opap'].items()
                                if k not in ('rowgen_q', 'all_q')}), indent=1))
    print(json.dumps(_jsonable(out['least_core']), indent=1))


def _standard_allocs(owen):
    def fn(cN, kappa):
        n = len(kappa)
        return {'sigma': dict(owen['sigma']), 'owen_Ex': dict(owen['x']),
                'equal_bill': {u: cN / n for u in kappa},
                'equal_surplus': {u: kappa[u] + (cN - sum(kappa.values())) / n
                                  for u in kappa}}
    return fn


def part_b(args):
    n, k = 6, 3
    players, _, T, base, name = build_instance(n, day=DAY)
    scen = scenarios(base, players, T, k, seed=0)
    out = {'instance': name, 'n': n, 'day': DAY, 'scenarios': k, 'seed': 0}
    owen = owen_point(players, T, scen, base)
    out['owen'] = owen
    cc = StoCore(players, T, scen)
    cc.output_dir = OUT
    out.update(enumeration_block(cc, _standard_allocs(owen), args,
                                 'part_b_separation_vs_enumeration.csv'))
    _dump(out, 'part_b_n6_S3.json')
    _print_block(out, 'b')


# =============================================================================
# KL-DRO game
def cli_scenarios(base, players, T, k, seed=0):
    """The scenarios stochastic_extension's command line draws: it passes the WIND
    sigma (0.25) as solar_sigma too, where make_scenarios' own default is 0.20. The
    same set as scenarios() exactly when the instance has no solar member (n=6)."""
    return SE.make_scenarios(base, players, T, k, seed=seed, wind_sigma=0.25,
                             solar_sigma=0.25)


def _same_scenarios(a, b):
    return len(a) == len(b) and all(
        pa == pb and set(sa) == set(sb) and all(sa[x] == sb[x] for x in sa)
        for (pa, sa), (pb, sb) in zip(a, b))


def kl_owen_point(n, k, radius, seed=0):
    """Robust Owen allocation of the KL game on scenarios(), cost convention: sigma
    (raw, sums to the robust v^LR) and E[x*] = sigma - omega/n (sums to the robust EF).

    Taken from the command line's own KL run (stochastic_extension run(), defaults,
    --skip-standalone) when its scenarios are ours (no solar member); otherwise the same
    code path in process on our scenarios (EF, solve_dwr_kl with the command line's
    defaults, robust_allocation) -- the command line would price a different game."""
    t0 = time.time()
    players, _, T, base, name = build_instance(n, day=DAY)
    scen = scenarios(base, players, T, k, seed=seed)
    same = _same_scenarios(scen, cli_scenarios(base, players, T, k, seed=seed))
    if same:
        argv = ['--n', str(n), '--day', str(DAY), '--scenarios', str(k), '--seed', str(seed),
                '--kl-radius', str(radius), '--skip-standalone',
                '--out', os.path.join(OUT, 'kl_cli')]
        res = SE.run(SE.build_parser().parse_args(argv))
        al = res['allocation']
        v_mip, v_lr = float(res['ef']['obj']), float(res['dw']['obj'])
    else:
        args = SE.build_parser().parse_args(['--kl-radius', str(radius)])
        SE._configure_direct(args)
        ef = SE.solve_extensive_form(players, T, scen, gap=args.mip_gap,
                                     solver=args.mip_solver, kl_radius=radius)
        dw, master = SE.solve_dwr_kl(
            players, T, scen, base, kl_radius=radius, kl_nested=args.kl_nested,
            kl_cut_frac=args.kl_cut_frac, doi_markup=args.doi_markup,
            kl_master=args.kl_master, kl_seed=None, **SE._direct_kwargs(args, ef))
        al = SE.robust_allocation(ef, dw, master)
        v_mip, v_lr = float(ef['obj']), float(dw['obj'])
    return {'sigma': {u: -float(al['owen'][u]) for u in players},
            'x': {u: -float(al['Ex'][u]) for u in players},
            'v_mip': v_mip, 'v_lr': v_lr,
            'eps_LR': float(al['eps_LR']), 'omega_LR': float(al['omega_LR']),
            'omega_interval': al.get('omega_interval'), 'rho_star': list(al.get('rho_star')),
            'budget_residual': float(al['budget_residual']),
            'source': 'command line' if same else 'in process (CLI scenarios differ)',
            'time': time.time() - t0}


def part_kldeact(args):
    """KL separation with z fixed to each of the 63 coalitions (payoffs 0) vs the
    KL extensive form of the coalition alone, both at a tight gap."""
    r, k = args.radius, 3
    players, _, T, base, name = build_instance(6, day=DAY)
    scen = scenarios(base, players, T, k, seed=0)
    sep = SC.KLSeparation(players, T, scen, {u: 0.0 for u in players}, r)
    rows = []
    for m_ in range(1, len(players) + 1):
        for S in itertools.combinations(players, m_):
            S = list(S)
            _, v = sep.solve(fix=S, gap=args.tight_gap)
            st = dict(sep.stats)
            ef = SE.solve_extensive_form(S, T, scen, gap=args.tight_gap, solver='gurobi',
                                         kl_radius=r)
            rows.append({'S': '+'.join(S), 'sep_cost': -v, 'sep_bound': -st['violation_bound'],
                         'sep_closed': st['closed'], 'sep_rounds': st['rounds'],
                         'c_KL': ef['obj'], 'c_KL_bound': ef['dual_bound'],
                         'ef_rounds': ef['kl']['solves'], 'diff': -v - ef['obj'],
                         'rel_diff': (-v - ef['obj']) / max(1.0, abs(ef['obj']))})
    _csv(rows, f'kl_deact_n6_S{k}_r{r:g}.csv')
    summ = {'radius': r, 'scenarios': k, 'gap': args.tight_gap, 'coalitions': len(rows),
            'max_abs_diff': max(abs(x['diff']) for x in rows),
            'max_rel_diff': max(abs(x['rel_diff']) for x in rows),
            'all_closed': all(x['sep_closed'] for x in rows),
            'max_rounds_sep': max(x['sep_rounds'] for x in rows),
            'max_rounds_ef': max(x['ef_rounds'] for x in rows)}
    _dump({'instance': name, 'day': DAY, 'summary': summ}, f'kl_deact_n6_S{k}_r{r:g}.json')
    print('\nSUMMARY kldeact', json.dumps(_jsonable(summ), indent=1))


def part_klb(args):
    """The KL game at radius r, n=6, |Omega|=3: separation vs enumeration, OPAP, least
    core. r = 0 uses the risk-neutral part b's four allocations and compares with it."""
    n, k, r = 6, 3, args.radius
    players, _, T, base, name = build_instance(n, day=DAY)
    scen = scenarios(base, players, T, k, seed=0)
    out = {'instance': name, 'n': n, 'day': DAY, 'scenarios': k, 'seed': 0, 'radius': r}
    if r == 0:
        with open(os.path.join(OUT, 'part_b_n6_S3.json')) as f:
            rn = json.load(f)
        if 'allocations' in rn:
            fixed = rn['allocations']
        else:
            cN0, kap0 = rn['rowgen']['c_N'], rn['rowgen']['kappa']
            fixed = _standard_allocs(rn['owen'])(cN0, kap0)
        allocs_fn = lambda cN, kappa: fixed
        out['owen'] = rn['owen']
    else:
        owen = kl_owen_point(n, k, r)
        out['owen'] = owen
        allocs_fn = _standard_allocs(owen)
    cc = StoCore(players, T, scen, kl_radius=r)
    cc.output_dir = OUT
    out.update(enumeration_block(cc, allocs_fn, args,
                                 f'kl_b_n6_S{k}_r{r:g}_separation_vs_enumeration.csv'))
    if r == 0:
        rn_rows = {x['allocation']: x for x in rn['separation_vs_enumeration']}
        out['vs_risk_neutral'] = {
            'separation': [{
                'allocation': x['allocation'],
                'sep_proper': [rn_rows[x['allocation']]['sep_proper_excess'], x['sep_proper_excess']],
                'sep_all': [rn_rows[x['allocation']]['sep_all_excess'], x['sep_all_excess']],
                'dinkelbach': [rn_rows[x['allocation']]['dinkelbach_weak_eps'],
                               x['dinkelbach_weak_eps']],
                'same_S': rn_rows[x['allocation']]['sep_proper_S'] == x['sep_proper_S']}
                for x in out['separation_vs_enumeration']],
            'opap_omega_star': [rn['opap']['rowgen_omega_star'], out['opap']['rowgen_omega_star']],
            'least_core': [rn['least_core']['rowgen_value'], out['least_core']['rowgen_value']],
            'max_coalition_value_diff': max(abs(out['coalition_values'][S] - v)
                                            for S, v in rn['coalition_values'].items()),
        }
    else:
        # the command line's own enumeration (measure_eps) of E[x*]: profit convention
        cli = SE.measure_eps(players, T, scen, {u: -owen['x'][u] for u in players},
                             time_limit=None, gap=SE.EF_GAP, solver='gurobi', kl_radius=r)
        out['measure_eps'] = {'eps': cli['eps'], 'argmax': cli['argmax'],
                              'max_value_diff_vs_cache': max(
                                  abs(-cli['coalitions'][S]['v'] - out['coalition_values'][S])
                                  for S in out['coalition_values'])}
    _dump(out, f'kl_b_n6_S{k}_r{r:g}.json')
    _print_block(out, f'klb r={r:g}')
    for key in ('vs_risk_neutral', 'measure_eps'):
        if key in out:
            print(key, json.dumps(_jsonable(out[key]), indent=1))


def part_klc(args):
    """Size, time and tangent rounds of the KL separation at the robust Owen point."""
    n = args.n
    players, _, T, base, name = build_instance(n, day=DAY)
    path = os.path.join(OUT, f'kl_c_scale_n{n}.json')
    rows = []
    if os.path.exists(path) and not args.force:
        with open(path) as f:
            rows = json.load(f)['rows']
    done = {(x['scenarios'], x['radius']) for x in rows}
    for k in [int(x) for x in args.scen.split(',')]:
        for r in [float(x) for x in args.radii.split(',')]:
            if (k, r) in done:
                continue
            scen = scenarios(base, players, T, k, seed=0)
            owen = kl_owen_point(n, k, r)
            row = {'n': n, 'scenarios': k, 'radius': r, 'eps_LR': owen['eps_LR'],
                   'omega_LR': owen['omega_LR'], 'owen_source': owen['source'],
                   'owen_time': owen['time']}
            # the raw robust Owen point must be stable outright (robust Owen, item 2)
            sep = SC.KLSeparation(players, T, scen, owen['sigma'], r)
            S0, v0 = sep.solve(time_limit=args.sep_time_limit)
            row.update({'sigma_violation': v0, 'sigma_coalition': '+'.join(S0),
                        'sigma_violation_bound': sep.stats['violation_bound'],
                        'sigma_truncated': sep.truncated})
            for tag, kw in (('all', {}), ('proper', {'min_size': 1, 'max_size': n - 1})):
                sep = SC.KLSeparation(players, T, scen, owen['x'], r)
                S, v = sep.solve(time_limit=args.sep_time_limit, **kw)
                st = sep.stats
                if tag == 'all':
                    row.update({kk: st.get(kk) for kk in (
                        'vars', 'bin_vars', 'rows', 'nonzeros', 'first_stage_vars')})
                row.update({f'{tag}_{kk}': st.get(kk) for kk in (
                    'time_solve', 'time_gurobi', 'rounds', 'nodes', 'closed', 'status',
                    'violation', 'violation_bound', 'gap')})
                row[f'{tag}_coalition'] = '+'.join(S)
                row[f'{tag}_truncated'] = sep.truncated
            if args.weak_eps:
                cc = SC.StochasticCoreComputation(players, T, scen, kl_radius=r)
                we = weak_eps(cc, owen['x'], args.eps_time_limit)
                row.update({'weak_eps': we['eps'], 'weak_eps_S': '+'.join(we['coalition']),
                            'weak_eps_certified': we['certified'], 'weak_eps_time': we['time'],
                            'weak_eps_separations': len(cc.sep_log),
                            'holds_eps_LR': we['eps'] <= owen['eps_LR'] + 1e-6 * (1 + abs(owen['v_mip']))})
            rows.append(row)
            _dump({'instance': name, 'day': DAY, 'rows': rows}, os.path.basename(path))
            print('\nROW', json.dumps(_jsonable(row)), flush=True)
    _csv(rows, f'kl_c_scale_n{n}.csv')



# =============================================================================
def part_c(args):
    n = args.n
    players, _, T, base, name = build_instance(n, day=DAY)
    rows = []
    path = os.path.join(OUT, f'part_c_scale_n{n}.json')
    for k in [int(x) for x in args.scen.split(',')]:
        scen = scenarios(base, players, T, k, seed=0)
        owen = owen_point(players, T, scen, base)
        t0 = time.time()
        sep = SC.StochasticSeparation(players, T, scen, owen['x'])
        S, v = sep.solve(time_limit=args.sep_time_limit)
        st = dict(sep.stats)
        row = {'n': n, 'scenarios': k, **{kk: st[kk] for kk in (
            'vars', 'bin_vars', 'int_vars', 'rows', 'nonzeros', 'first_stage_vars',
            'time_build', 'time_copy', 'time_solve', 'status', 'gap', 'nodes')},
            'violation': v, 'violation_bound': st['violation_bound'],
            'coalition': '+'.join(S), 'truncated': sep.truncated,
            'eps_LR': owen['eps_LR'], 'owen_time': owen['time']}
        # the max over PROPER coalitions (empty set and N excluded): a harder solve when
        # the allocation is in the core, since the empty set cannot certify it
        S2, v2 = sep.solve(time_limit=args.sep_time_limit, min_size=1, max_size=n - 1)
        row.update({'proper_violation': v2, 'proper_coalition': '+'.join(S2),
                    'proper_violation_bound': sep.stats['violation_bound'],
                    'proper_time_solve': sep.stats['time_solve'],
                    'proper_nodes': sep.stats['nodes'], 'proper_truncated': sep.truncated})
        if args.weak_eps:
            cc = SC.StochasticCoreComputation(players, T, scen)
            we = weak_eps(cc, owen['x'], args.eps_time_limit)
            row.update({'weak_eps': we['eps'], 'weak_eps_S': '+'.join(we['coalition']),
                        'weak_eps_certified': we['certified'], 'weak_eps_time': we['time'],
                        'weak_eps_separations': len(cc.sep_log),
                        'weak_eps_sep_times': [s['time_solve'] for s in cc.sep_log],
                        'holds_eps_LR': we['eps'] <= owen['eps_LR'] + 1e-6 * (1 + abs(owen['v_mip']))})
        rows.append(row)
        _dump({'instance': name, 'day': DAY, 'rows': rows}, os.path.basename(path))
        print('\nROW', json.dumps(_jsonable(row)), flush=True)
    _csv(rows, f'part_c_scale_n{n}.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parts = {'deact': part_deact, 'a': part_a, 'cross': part_cross, 'b': part_b,
             'c': part_c, 'kldeact': part_kldeact, 'klb': part_klb, 'klc': part_klc}
    ap.add_argument('--part', required=True, choices=list(parts))
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--time-limit', type=float, default=3600.0,
                    help='row generation budget (s)')
    ap.add_argument('--max-iter', type=int, default=5000,
                    help='row generation iteration cap (compute_core defaults to 100)')
    ap.add_argument('--eps-time-limit', type=float, default=3600.0,
                    help='budget of one Dinkelbach weak-eps measurement (s)')
    ap.add_argument('--sep-time-limit', type=float, default=1800.0,
                    help='part c: budget of the single separation solve (s)')
    ap.add_argument('--scen', default='1,3,5', help='part c: scenario counts')
    ap.add_argument('--weak-eps', action='store_true',
                    help='part c / klc: also run the Dinkelbach weak-eps measurement')
    ap.add_argument('--radius', type=float, default=0.5, help='kldeact / klb: KL radius r')
    ap.add_argument('--radii', default='0.1,0.5,1.0', help='klc: KL radii')
    ap.add_argument('--tight-gap', type=float, default=1e-7, help='kldeact: MIP gap')
    ap.add_argument('--force', action='store_true', help='klc: recompute existing rows')
    args = ap.parse_args()
    parts[args.part](args)


if __name__ == '__main__':
    main()

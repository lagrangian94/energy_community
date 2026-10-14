"""
The |Omega| > 20 cells of run_kl_oosn.py with plans of their own sample
(kl_experiment_plan.md, sec. 8.20): there the 20-scenario plans were only valued on
the larger sample, so the value promised is a little below that sample's optimum.
Here each extensive form is solved on the sample itself, started from that plan, for
--time seconds at most, and the better plan is measured again.

  KL     stochastic_extension.py --ef-improve: warm start from the plan, MIPFocus 1,
         tangents exact at the plan's own ratios; stops on time or after --stall seconds
         without a new incumbent
  r = 0  stochastic_extension.py --ef-start: the plan as the MIP start; stops on the gap
         (1e-4) or on time

'promised' is the extensive form's value at the better plan. If the first stage moved,
both plans are valued again on the test sample (oos_plans.py); if not, what they earn is
run_kl_oos.py's. Each record keeps the value of the 20-scenario plan ('promised_before')
and the solve ('improve').

One job at a time under the memory watchdog of run_kl_main.py; a (size, sample, day)
whose file exists is skipped. Results in <out>/oos_n<n>_S<|Omega|>_day<day>.json, laid
out as run_kl_oos.py's; the plans and solves under <out>/plans.

    python ieee_owen/weak_eps_experiment/run_kl_oosn_improve.py --n 60 --scen 100 --days 1
"""
import os, sys, glob, json, time, pickle, argparse

import run_kl_main as Q                      # chdirs to the repo root
import run_kl_oos as O
import run_kl_oosn as N

OUT = os.path.join(N.OUT, 'improve')
LABELS = ('saa', 'dro')


def plan_paths(a, n, m, day, lab):
    base = os.path.join(a.out, 'plans', f'{lab}_n{n}_S{m}_day{day}')
    return base + '_start.pkl', base + '.pkl', base + '.json'


def start_worker(a):
    """The 20-scenario plans of one (n, day) as complete solutions of the extensive
    forms of every |Omega| in --scen (the first |Omega| training scenarios each)."""
    sys.path.insert(0, os.path.join(Q._ROOT, 'ieee_owen'))
    sys.path.insert(0, Q._ROOT)
    import stochastic_extension as SE
    import e3_optimizers_curse as E3
    n, day = int(a.n), int(a.days)
    sizes = sorted(int(v) for v in a.scen.split(','))
    src = {'saa': os.path.join(O.OUT, 'saa', f'n{n}_S{N.PLAN_SCEN}_day{day}.json'),
           'dro': N.main_plan(n, day)}
    players, T, base, name = E3.instance(n, day)
    for m in sizes:
        scen = E3.scenarios(base, players, T, m, 0, day=day)
        for lab in LABELS:
            first = json.load(open(src[lab], encoding='utf-8'))['ef_first_stage']
            ef = SE.plan_extensive_form(players, T, scen, first, workers=a.workers,
                                        kl_radius=Q.radius(m) if lab == 'dro' else None)
            ef['plan'] = src[lab]
            path = plan_paths(a, n, m, day, lab)[0]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(ef, f)
            print(f'{lab} |Omega|={m}: the 20-scenario plan is worth {ef["obj"]:.4f} '
                  f'({ef["time_solve"]:.0f}s)', flush=True)


def solve(a, n, m, day, lab):
    """The extensive form of the sample from the 20-scenario plan; (record, moved)."""
    start, cache, js = plan_paths(a, n, m, day, lab)
    with open(start, 'rb') as f:
        before = pickle.load(f)
    if not os.path.exists(cache):
        cmd = [sys.executable, os.path.join('ieee_owen', 'stochastic_extension.py'),
               '--n', str(n), '--scenarios', str(m), '--day', str(day),
               '--reserve-price-scale', repr(Q.PRICE_SCALE), '--ef-nodefile-start', '3',
               '--ef-only', '--out', a.out,
               '--ef-log', os.path.join(a.out, 'logs', f'{lab}_n{n}_S{m}_day{day}_ef_gurobi.log')]
        if lab == 'dro':
            import shutil
            shutil.copy(start, cache)        # --ef-improve starts from the plan in the cache
            cmd += ['--kl-radius', repr(Q.radius(m)), '--ef-cache', cache,
                    '--ef-improve', str(a.time), '--ef-improve-stall', str(a.stall),
                    '--ef-time-cap', str(a.time)]
        else:
            cmd += ['--ef-start', start, '--ef-cache', cache, '--ef-time-cap', str(a.time)]
        rc, wall, priv, _, _, killed = Q.watched(
            cmd, os.path.join(a.out, 'logs', f'{lab}_n{n}_S{m}_day{day}.log'), a.min_free)
        print(f'  {lab}: exit {rc}, killed {int(killed)}, {wall:.0f}s, peak {priv:.1f} GB',
              flush=True)
        if rc != 0:
            if os.path.exists(cache):
                os.remove(cache)
            return None, False
    with open(cache, 'rb') as f:
        ef = pickle.load(f)
    if ef['obj'] > before['obj']:            # cost: keep the plan the solve started from
        ef = before
    first = {k: float(ef['vals'][k]) for k in ef['first_stage_names'] if abs(ef['vals'][k]) > 1e-9}
    moved = any(abs(first.get(k, 0.0) - before['vals'][k]) > 1e-6 for k in before['first_stage_names'])
    rec = {'n': n, 'scenarios': m, 'day': day, 'probs': [1.0 / m] * m,
           'kl': {'radius': Q.radius(m)} if lab == 'dro' else None,
           'ef': {k: ef[k] for k in ('status', 'obj', 'dual_bound', 'gap', 'first_cost',
                                     'scen_cost', 'time_solve')},
           'ef_first_stage': first, 'obj_before': before['obj'], 'moved': moved,
           'improve': ef.get('improve')}
    with open(js, 'w') as f:
        json.dump(json.loads(json.dumps(rec, default=float)), f, indent=1)
    return rec, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='6,15,30,60')
    ap.add_argument('--scen', default='40,100')
    ap.add_argument('--days', default='1-31')
    ap.add_argument('--time', type=float, default=1800.0, help='seconds per extensive form')
    ap.add_argument('--stall', type=float, default=600.0,
                    help='KL: stop after so many seconds without a new incumbent')
    ap.add_argument('--test', type=int, default=500)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--min-free', type=float, default=8.0, help='kill below this [GB]')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--start-worker', action='store_true', help='internal')
    a = ap.parse_args()
    if a.start_worker:
        return start_worker(a)
    sizes = [int(v) for v in a.scen.split(',')]
    os.makedirs(os.path.join(a.out, 'logs'), exist_ok=True)
    for n in [int(v) for v in a.n.split(',')]:
        for day in Q.days(a.days):
            todo = [m for m in sizes if not os.path.exists(N.result(a, n, m, day))]
            if not todo:
                continue
            if not all(os.path.exists(plan_paths(a, n, m, day, lab)[0])
                       for m in todo for lab in LABELS):
                cmd = [sys.executable, os.path.abspath(__file__), '--n', str(n), '--days',
                       str(day), '--scen', ','.join(map(str, todo)), '--workers',
                       str(a.workers), '--out', a.out, '--start-worker']
                rc, wall, *_ = Q.watched(
                    cmd, os.path.join(a.out, 'logs', f'start_n{n}_day{day}.log'), a.min_free)
                if rc != 0:
                    print(f'n={n} day {day}: valuing the 20-scenario plans failed', flush=True)
                    continue
            earned = json.load(open(os.path.join(O.OUT, f'oos_n{n}_S{N.PLAN_SCEN}_day{day}.json'),
                                    encoding='utf-8'))
            for m in todo:
                print(f'{time.strftime("%m-%d %H:%M")}  n={n} |Omega|={m} day {day}', flush=True)
                recs = {lab: solve(a, n, m, day, lab) for lab in LABELS}
                if any(r is None for r, _ in recs.values()):
                    print('  an extensive form failed, skipped', flush=True)
                    continue
                if any(mv for _, mv in recs.values()):
                    out = N.result(a, n, m, day) + '.test.json'
                    plans = ','.join(f'{lab}={plan_paths(a, n, m, day, lab)[2]}' for lab in LABELS)
                    cmd = [sys.executable, os.path.join('ieee_owen', 'oos_plans.py'), '--n',
                           str(n), '--day', str(day), '--test', str(a.test), '--plans', plans,
                           '--out', out]
                    rc, *_ = Q.watched(cmd, os.path.join(a.out, 'logs',
                                                         f'oos_n{n}_S{m}_day{day}.log'), a.min_free)
                    if rc != 0:
                        print('  test valuation failed, skipped', flush=True)
                        continue
                    test = json.load(open(out))['plans']
                else:
                    test = earned['plans']
                res = {'n': n, 'day': day, 'scenarios': m, 'plan_scenarios': N.PLAN_SCEN,
                       'test': earned['test'], 'plans': {}}
                for lab in LABELS:
                    r, mv = recs[lab]
                    res['plans'][lab] = {
                        'radius': Q.radius(m) if lab == 'dro' else 0.0,
                        'promised': -r['ef']['obj'], 'promised_before': -r['obj_before'],
                        'in_mean': -(r['ef']['first_cost'] + sum(r['ef']['scen_cost']) / m),
                        'oos_mean': test[lab]['oos_mean'], 'oos_se': test[lab]['oos_se'],
                        'moved': mv, 'ef_status': r['ef']['status'],
                        'ef_gap': r['ef']['gap'], 'improve': r['improve']}
                    res['plans'][lab]['disappoint'] = (res['plans'][lab]['promised']
                                                       - res['plans'][lab]['oos_mean'])
                with open(N.result(a, n, m, day), 'w') as f:
                    json.dump(json.loads(json.dumps(res, default=float)), f, indent=1)
                print('  ' + '  '.join(
                    f'{lab}: promised {v["promised_before"]:.2f} -> {v["promised"]:.2f}'
                    f'{" (moved)" if v["moved"] else ""} oos {v["oos_mean"]:.2f}'
                    for lab, v in res['plans'].items()), flush=True)
    print('IMPROVE DONE', flush=True)


if __name__ == '__main__':
    main()

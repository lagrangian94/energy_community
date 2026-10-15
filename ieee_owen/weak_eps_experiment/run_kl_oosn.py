"""
The out-of-sample budget of the settlement against the training sample size
(kl_experiment_plan.md, sec. 8.20): Table tab:oos of the uncertainty manuscript at
|Omega| other than the main queue's 20, with the ball (radius chi2_{1,0.95} / (2 |Omega|))
and without it (r = 0, the sample-average problem). What is compared is, as in
run_kl_oos.py, the value a plan was chosen by ('promised') and what it earns on the test
sample ('oos_mean'); their difference over n is the shift each member receives.

  |Omega| > 20   No problem is solved. The two plans of the main queue's day (the KL
        plan, and the r = 0 plan of run_kl_oos.py) are valued on the first |Omega|
        training scenarios, recourse per scenario (plan_extensive_form): 'promised' is
        the sample mean at r = 0 and the worst case over the ball with it. A plan fixed
        from the first 20 scenarios is not the best plan of the larger sample, so
        'promised' is a little below that sample's optimal value; what the plan earns is
        what run_kl_oos.py measured. One valuation at the largest |Omega| serves every
        smaller one (the samples are nested).
  |Omega| < 20   The 20-scenario plans have seen scenarios such a sample has not, so
        both extensive forms are solved on the sample itself (--cap seconds each) and
        their plans valued on the test sample (oos_plans.py).

One job at a time under the memory watchdog of run_kl_main.py; a (size, sample, day)
whose file exists is skipped. Results in <out>/oos_n<n>_S<|Omega|>_day<day>.json, laid
out as run_kl_oos.py's.

    python ieee_owen/weak_eps_experiment/run_kl_oosn.py --scen 40,100
    python ieee_owen/weak_eps_experiment/run_kl_oosn.py --scen 5
"""
import os, sys, glob, json, time, pickle, shutil, argparse

import run_kl_main as Q                      # chdirs to the repo root
import run_kl_oos as O

MAIN, OOS20 = Q.OUT, O.OUT
OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'oosn17_xi')
PLAN_SCEN = 20


def result(a, n, m, day):
    return os.path.join(a.out, f'oos_n{n}_S{m}_day{day}.json')


def main_plan(n, day):
    found = glob.glob(os.path.join(MAIN, f'{n}p_*_day{day}_S{PLAN_SCEN}_*_main.json'))
    return found[0] if len(found) == 1 else None


def reuse_worker(a):
    """The two 20-scenario plans of one (n, day) on its first max(--scen) training
    scenarios, in this process; one result file per |Omega| in --scen."""
    sys.path.insert(0, os.path.join(Q._ROOT, 'ieee_owen'))
    sys.path.insert(0, Q._ROOT)
    import numpy as np
    import stochastic_extension as SE
    import e3_optimizers_curse as E3
    n, day = int(a.n), int(a.days)
    sizes = sorted(int(v) for v in a.scen.split(','))
    earned = json.load(open(os.path.join(OOS20, f'oos_n{n}_S{PLAN_SCEN}_day{day}.json'),
                            encoding='utf-8'))
    plans = {'saa': os.path.join(OOS20, 'saa', f'n{n}_S{PLAN_SCEN}_day{day}.json'),
             'dro': main_plan(n, day)}
    players, T, base, name = E3.instance(n, day)
    scen = E3.scenarios(base, players, T, max(sizes), 0, day=day)     # the training seed
    cost = {}
    for lab, path in plans.items():
        first = json.load(open(path, encoding='utf-8'))['ef_first_stage']
        ef = SE.plan_extensive_form(players, T, scen, first, workers=a.workers)
        cost[lab] = ef['first_cost'] + np.array(ef['scen_cost'])
    for m in sizes:
        res = {'instance': name, 'n': n, 'day': day, 'scenarios': m,
               'plan_scenarios': PLAN_SCEN, 'test': earned['test'],
               'test_seed': earned['test_seed'], 'plans': {}}
        for lab, path in plans.items():
            c = cost[lab][:m]
            r = Q.radius(m) if lab == 'dro' else 0.0
            promised = -SE.kl_worst(c, np.full(m, 1.0 / m), r)[0]
            e = earned['plans'][lab]
            res['plans'][lab] = {'path': path, 'radius': r, 'promised': promised,
                                 'in_mean': -float(c.mean()), 'oos_mean': e['oos_mean'],
                                 'oos_se': e['oos_se'], 'disappoint': promised - e['oos_mean']}
        with open(result(a, n, m, day), 'w') as f:
            json.dump(res, f, indent=1)
    # the same valuation at the plans' own sample, against the values they were chosen by
    for lab in plans:
        c = cost[lab][:PLAN_SCEN]
        v = -SE.kl_worst(c, np.full(len(c), 1.0 / len(c)),
                         Q.radius(PLAN_SCEN) if lab == 'dro' else 0.0)[0]
        print(f'check {lab}: valued again on its {PLAN_SCEN} scenarios {v:.4f}, '
              f'chosen by {earned["plans"][lab]["promised"]:.4f}', flush=True)


def dro_plan(n, m, day, a):
    """The KL plan of a sample of its own, as a JSON oos_plans.load_plan reads."""
    path = os.path.join(a.out, 'dro', f'n{n}_S{m}_day{day}.json')
    if os.path.exists(path):
        return path
    cache = os.path.join(a.out, 'dro', f'n{n}_S{m}_day{day}.pkl')
    solved = os.path.join(a.solved, f'n{n}_S{m}_day{day}.pkl')        # an earlier queue's
    if not os.path.exists(cache) and os.path.exists(solved):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        shutil.copy(solved, cache)
    r = Q.radius(m)
    cmd = [sys.executable, os.path.join('ieee_owen', 'stochastic_extension.py'),
           '--n', str(n), '--scenarios', str(m), '--day', str(day), '--kl-radius', repr(r),
           '--reserve-price-scale', repr(Q.PRICE_SCALE), '--ef-time-cap', str(a.saa_cap),
           '--ef-nodefile-start', '3', '--ef-only', '--ef-cache', cache, '--out', a.out,
           '--ef-log', os.path.join(a.out, 'logs', f'dro_n{n}_S{m}_day{day}_ef_gurobi.log')]
    rc, wall, peak, _, _, killed = Q.watched(
        cmd, os.path.join(a.out, 'logs', f'dro_n{n}_S{m}_day{day}.log'), a.min_free)
    print(f'  KL n={n} day {day}: peak {peak:.1f} GB', flush=True)
    if rc != 0 or not os.path.exists(cache):
        print(f'  KL n={n} day {day}: failed (exit {rc}, killed {killed})', flush=True)
        return None
    with open(cache, 'rb') as f:
        ef = pickle.load(f)
    js = {'n': n, 'scenarios': m, 'day': day, 'probs': [1.0 / m] * m, 'kl': {'radius': r},
          'ef': {k: ef[k] for k in ('status', 'obj', 'dual_bound', 'gap', 'first_cost',
                                    'scen_cost', 'time_solve')},
          'ef_first_stage': {k: float(ef['vals'][k]) for k in ef['first_stage_names']
                             if abs(ef['vals'][k]) > 1e-9}}
    with open(path, 'w') as f:
        json.dump(json.loads(json.dumps(js, default=float)), f, indent=1)
    print(f'  KL n={n} day {day}: {ef["status"]}, {ef["time_solve"]:.0f}s, gap {ef["gap"]:.1e}',
          flush=True)
    return path


def own(n, m, day, a):
    plans = [f'{lab}={p}' for lab, p in (('saa', O.saa_plan(n, m, day, a)),
                                         ('dro', dro_plan(n, m, day, a))) if p]
    if len(plans) < 2:
        return 'a plan is missing, skipped'
    out = result(a, n, m, day)
    cmd = [sys.executable, os.path.join('ieee_owen', 'oos_plans.py'), '--n', str(n),
           '--day', str(day), '--test', str(a.test), '--plans', ','.join(plans), '--out', out]
    rc, wall, peak, _, _, killed = Q.watched(
        cmd, os.path.join(a.out, 'logs', f'oos_n{n}_S{m}_day{day}.log'), a.min_free)
    if rc != 0 or not os.path.exists(out):
        return f'failed (exit {rc}, killed {killed}, {wall:.0f}s)'
    r = json.load(open(out))['plans']
    return '  '.join(f'{k}: promised {v["promised"]:.1f} oos {v["oos_mean"]:.1f}'
                     for k, v in r.items()) + f'  {wall:.0f}s, peak {peak:.1f} GB'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='6,15,30,60')
    ap.add_argument('--scen', default='40,100')
    ap.add_argument('--days', default='1-31')
    ap.add_argument('--saa-cap', type=float, default=1800.0,
                    help='seconds for an extensive form (|Omega| < 20)')
    ap.add_argument('--test', type=int, default=500, help='test scenarios (|Omega| < 20)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--min-free', type=float, default=4.0, help='kill below this [GB]')
    ap.add_argument('--solved', default=os.path.join(
        'ieee_owen', 'weak_eps_experiment', 'stochastic', 'ubound17_xi', 'train', 'ef_cache'),
        help='folder of KL extensive forms already solved (pickles), copied if there')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--reuse-worker', action='store_true', help='internal')
    a = ap.parse_args()
    a.saa_start = None
    if a.reuse_worker:
        return reuse_worker(a)
    sizes = [int(v) for v in a.scen.split(',')]
    big, small = [m for m in sizes if m > PLAN_SCEN], [m for m in sizes if m < PLAN_SCEN]
    os.makedirs(os.path.join(a.out, 'logs'), exist_ok=True)
    for n in [int(v) for v in a.n.split(',')]:
        for day in Q.days(a.days):
            if big and not all(os.path.exists(result(a, n, m, day)) for m in big):
                print(f'{time.strftime("%m-%d %H:%M")}  plans of {PLAN_SCEN} on |Omega|={big} '
                      f'n={n} day {day}', flush=True)
                cmd = [sys.executable, os.path.abspath(__file__), '--n', str(n), '--days',
                       str(day), '--scen', ','.join(str(m) for m in big), '--workers',
                       str(a.workers), '--out', a.out, '--reuse-worker']
                rc, wall, peak, *_ = Q.watched(
                    cmd, os.path.join(a.out, 'logs', f'reuse_n{n}_day{day}.log'), a.min_free)
                print(f'  exit {rc}, {wall:.0f}s, peak {peak:.1f} GB', flush=True)
            for m in small:
                if not os.path.exists(result(a, n, m, day)):
                    print(f'{time.strftime("%m-%d %H:%M")}  own plans |Omega|={m} n={n} '
                          f'day {day}', flush=True)
                    print(f'  {own(n, m, day, a)}', flush=True)
    print('OOSN DONE', flush=True)


if __name__ == '__main__':
    main()

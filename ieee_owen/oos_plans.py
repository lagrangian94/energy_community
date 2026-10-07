"""
Out-of-sample value of first-stage plans already solved by stochastic_extension.py.

Reads the EF first stage (commitment, r_sym) of each result JSON, fixes it, and
re-optimises the recourse on an independent test sample from the same generator
(e3_optimizers_curse.scenarios, seed TEST_SEED). Every plan is valued on the same
test scenarios, so plans are compared by paired differences. Unlike
e3_optimizers_curse.py, no EF is solved, and test scenarios are built one at a time
per worker and dropped, so n=60 fits in memory.

Trade bounds follow each scenario's own data (grid_caps='bnd_size', eq:bnd_size), in
training and test alike.

Reported per plan, profit in EUR (higher is better):
  promised    the EF objective the plan was chosen by: v^DR(N) (v^SAA(N) at r = 0)
  in_mean     the sample mean at the plan over the training scenarios
  oos_mean    mean over the test scenarios, with its standard error
  disappoint  promised - oos_mean   (> 0: the plan promised more than it delivers)

    python ieee_owen/oos_plans.py --n 60 --day 3 --test 200 \
        --plans r0=path/a.json,r0.096=path/b.json --out path/oos_day3.json
"""
import os, sys, json, time, argparse

_PAPER = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import gurobipy as gp

import stochastic_extension as SE
import e3_optimizers_curse as E3
from integer_lshaped import Sub


def load_plan(path, fs_names):
    d = json.load(open(path, encoding='utf-8'))
    stored = d['ef_first_stage']                     # nonzero values only
    unknown = set(stored) - set(fs_names)
    if unknown:
        raise ValueError(f'{path}: first-stage names not in the model: {sorted(unknown)[:5]}')
    x = {n: float(stored.get(n, 0.0)) for n in fs_names}
    ef = d['ef']
    first = float(ef['first_cost'])
    return {'path': path, 'x': x, 'first_cost': first,
            'promised': -float(ef['obj']), 'ef_gap': float(ef['gap']),
            'in_mean': -(first + float(np.dot(d['probs'], ef['scen_cost']))),
            'radius': (d.get('kl') or {}).get('radius', 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--day', type=int, default=3)
    ap.add_argument('--test', type=int, default=200)
    ap.add_argument('--plans', required=True, help='label=json,label=json,...')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--gap', type=float, default=1e-5, help='recourse MIPGap')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    players, T, base, name = E3.instance(a.n, a.day)
    test = E3.scenarios(base, players, T, a.test, E3.TEST_SEED, day=a.day)
    st = SE.ScenarioStack('fs', players, T, [test[0]], dwr=False)
    fs_names = sorted(st.first_stage)
    del st
    plans = {}
    for item in a.plans.split(','):
        label, path = item.split('=', 1)
        plans[label] = load_plan(path, fs_names)
    labels = list(plans)

    from concurrent.futures import ThreadPoolExecutor
    k = max(1, min(a.workers, a.test))
    per = max(1, (os.cpu_count() or 1) // k)
    envs = [gp.Env() for _ in range(k)]
    rec = np.full((len(labels), a.test), np.nan)     # recourse cost, incumbent
    bnd = np.full((len(labels), a.test), np.nan)     # its dual bound
    t0 = time.time()

    def work(i):
        sub = Sub(players, T, test[i][1], fs_names, a.gap, 1e3, 1e-4,
                  env=envs[i % k], threads=per)
        for j, lab in enumerate(labels):
            out = sub.mip(plans[lab]['x'])
            if out is None:
                raise RuntimeError(f'recourse infeasible: plan {lab}, test scenario {i}')
            bnd[j, i], rec[j, i] = out
        sub.g.dispose(); sub.lp.dispose()
        return i

    done = 0
    with ThreadPoolExecutor(k) as pool:
        for _ in pool.map(work, range(a.test)):
            done += 1
            if done % 20 == 0:
                print(f'  {done}/{a.test} test scenarios  {time.time() - t0:.0f}s', flush=True)

    res = {'instance': name, 'n': a.n, 'day': a.day, 'test': a.test,
           'test_seed': E3.TEST_SEED, 'gap': a.gap, 'time': time.time() - t0, 'plans': {}}
    oos = {}
    for j, lab in enumerate(labels):
        p = plans[lab]
        v = -(p['first_cost'] + rec[j])              # profit per test scenario
        oos[lab] = v
        r = {kk: p[kk] for kk in ('path', 'radius', 'promised', 'in_mean', 'ef_gap')}
        r['oos_mean'] = float(v.mean())
        r['oos_se'] = float(v.std(ddof=1) / np.sqrt(len(v)))
        r['oos_mean_bound'] = float((-(p['first_cost'] + bnd[j])).mean())
        r['disappoint'] = r['promised'] - r['oos_mean']
        res['plans'][lab] = r
    ref = labels[0]
    for lab in labels[1:]:
        dlt = oos[lab] - oos[ref]
        res['plans'][lab][f'paired_vs_{ref}'] = {
            'mean': float(dlt.mean()), 'se': float(dlt.std(ddof=1) / np.sqrt(len(dlt)))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)

    print(f'\n{name}, test sample {a.test} (seed {E3.TEST_SEED}), {res["time"]:.0f}s')
    print(f'{"plan":>8} {"r":>7} {"promised":>11} {"in-sample":>11} {"out-of-sample":>19}'
          f' {"disappoint":>11} {"vs " + ref:>17}')
    for lab in labels:
        r = res['plans'][lab]
        pv = r.get(f'paired_vs_{ref}')
        print(f'{lab:>8} {r["radius"]:>7.4g} {r["promised"]:>11.2f} {r["in_mean"]:>11.2f}'
              f' {r["oos_mean"]:>11.2f} +- {r["oos_se"]:<5.2f} {r["disappoint"]:>11.2f}'
              + (f' {pv["mean"]:>8.2f} +- {pv["se"]:.2f}' if pv else ''))


if __name__ == '__main__':
    main()

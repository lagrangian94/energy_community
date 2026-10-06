"""
E3 (ieee_owen/kl_experiment_plan.md, section 5b): the optimizer's curse at the community
level. For each training sample (|Omega| scenarios, seed s) and radius r, solve the KL
extensive form (r = 0: the SAA extensive form), fix its first stage x* (electrolyzer
states and r_sym), and value x* out of sample on an independent test sample: the
recourse of every test scenario is re-optimised with x* fixed.

Reported per configuration, all in profit (EUR, higher is better):
  promised   the in-sample objective the plan is chosen by: the robust value v^DR(N)
             (the SAA value at r = 0)
  in_mean    the in-sample sample mean at x*
  oos_mean   the out-of-sample mean at x*, with its standard error
  disappoint promised - oos_mean   (> 0: the plan promised more than it delivers)

    python ieee_owen/e3_optimizers_curse.py --n 15 --day 3 --train 5,10,20 \
        --radii 0,0.05,0.2,0.5 --seeds 0,1,2 --test 50
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
from integer_lshaped import Sub

PRICE_SCALE = 17.0 / 56.0
TEST_SEED = 1000


def instance(n, day):
    sys.path.insert(0, os.path.join(_PAPER, 'weak_eps_experiment'))
    from run_experiment import build_instance
    players, _, T, base, name = build_instance(n, day=day)
    base['grid_caps'] = 'bnd_size'
    if 'pi_res_t' in base:                       # as run(): --reserve-price-scale 17/56
        base['pi_res_t'] = {t: PRICE_SCALE * v for t, v in base['pi_res_t'].items()}
        base['pi_res'] = base['pi_up'] = base['pi_dn'] = float(
            np.mean(list(base['pi_res_t'].values())))
    return players, T, base, name


def scenarios(base, players, T, m, seed):
    # the CLI's defaults (stochastic_extension.run), so the instance is the same
    return SE.make_scenarios(base, players, T, m, seed=seed, wind_sigma=0.25,
                             solar_sigma=0.20, load_sigma=0.10, price_sigma=0.15,
                             rho=0.7, price_carriers=('E',),
                             load_carriers=('E', 'H', 'G'))


class TestSet:
    """The test scenarios' recourse problems, each a Sub with the first stage fixed by
    bounds, priced in parallel (one Gurobi environment per worker)."""

    def __init__(self, players, T, test, fs_names, workers=8):
        from concurrent.futures import ThreadPoolExecutor
        k = max(1, min(workers, len(test)))
        envs = [gp.Env() for _ in range(k)]
        per = max(1, (os.cpu_count() or 1) // k)
        self.subs = [Sub(players, T, sw, fs_names, 1e-4, 1e3, 1e-4, env=envs[i % k],
                         threads=per) for i, (_, sw) in enumerate(test)]
        self.k, self.pool = k, ThreadPoolExecutor(k)
        self.groups = [list(range(i, len(test), k)) for i in range(k)]

    def recourse(self, x):
        """Q_w(x) for every test scenario (incumbent recourse cost)."""
        def work(idx):
            return {i: self.subs[i].mip(x) for i in idx}
        out = {}
        for f in [self.pool.submit(work, g) for g in self.groups]:
            out.update(f.result())
        bad = [i for i, q in out.items() if q is None]
        if bad:
            raise RuntimeError(f'recourse infeasible in test scenarios {bad}')
        return np.array([out[i][1] for i in range(len(self.subs))])


def solve_plan(players, T, train, r, time_limit):
    t0 = time.time()
    ef = SE.solve_extensive_form(players, T, train, solver='gurobi',
                                 kl_radius=(r if r > 0 else None),
                                 time_limit=time_limit)
    st = ef['stack']
    x = {n: float(ef['vals'][n]) for n in st.first_stage}
    first = float(ef['first_cost'])
    in_mean = first + float(np.mean(ef['scen_cost']))
    return {'x': x, 'first_cost': first, 'promised_cost': float(ef['obj']),
            'in_mean_cost': in_mean, 'ef_gap': float(ef['gap']), 'ef_status': str(ef['status']),
            't_ef': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=15)
    ap.add_argument('--day', type=int, default=3)
    ap.add_argument('--train', default='5,10,20')
    ap.add_argument('--radii', default='0,0.05,0.2,0.5')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--test', type=int, default=50)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ef-time-limit', type=float, default=3600)
    ap.add_argument('--out', default=os.path.join(_PAPER, 'weak_eps_experiment',
                                                  'stochastic', 'kl_e3'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    players, T, base, name = instance(a.n, a.day)
    test = scenarios(base, players, T, a.test, TEST_SEED)
    path = os.path.join(a.out, f'e3_{name}_day{a.day}_test{a.test}.json')
    done = json.load(open(path)) if os.path.exists(path) else {}
    ts = None
    for s in [int(v) for v in a.seeds.split(',')]:
        for m in [int(v) for v in a.train.split(',')]:
            train = scenarios(base, players, T, m, s)
            for r in [float(v) for v in a.radii.split(',')]:
                key = f'S{m}_seed{s}_r{r:g}'
                if key in done:
                    continue
                print(f'=== {key} {time.strftime("%H:%M")}', flush=True)
                plan = solve_plan(players, T, train, r, a.ef_time_limit)
                if ts is None:
                    ts = TestSet(players, T, test, sorted(plan['x']), a.workers)
                t0 = time.time()
                q = ts.recourse(plan['x'])
                oos = plan['first_cost'] + q             # cost per test scenario
                rec = {'train': m, 'seed': s, 'r': r,
                       'promised': -plan['promised_cost'], 'in_mean': -plan['in_mean_cost'],
                       'oos_mean': -float(oos.mean()),
                       'oos_se': float(oos.std(ddof=1) / np.sqrt(len(oos))),
                       'ef_gap': plan['ef_gap'], 'ef_status': plan['ef_status'],
                       't_ef': plan['t_ef'], 't_oos': time.time() - t0}
                rec['disappoint'] = rec['promised'] - rec['oos_mean']
                done[key] = rec
                with open(path, 'w') as f:
                    json.dump(done, f, indent=1)
                print(f'  promised {rec["promised"]:.2f}  in-sample mean {rec["in_mean"]:.2f}  '
                      f'out-of-sample {rec["oos_mean"]:.2f} +- {rec["oos_se"]:.2f}  '
                      f'disappointment {rec["disappoint"]:.2f}  '
                      f'(EF {rec["t_ef"]:.0f}s gap {rec["ef_gap"]:.1e}, oos {rec["t_oos"]:.0f}s)',
                      flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()

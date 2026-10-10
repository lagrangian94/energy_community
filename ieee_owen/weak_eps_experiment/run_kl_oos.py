"""
Out-of-sample value of the plans of the main queue (Table III of the uncertainty
manuscript; kl_experiment_plan.md sec. 8.12 (3)).

Each day of the main queue is one training sample of |Omega| scenarios, and the 31
days are 31 independent samples (scenario streams are per day). For a day, the
first stage the grand coalition committed to is fixed and its recourse re-optimised
on an independent test sample (oos_plans.py). Two plans per day:

  dro   the KL plan of the main run (radius chi2_{1,0.95} / (2 |Omega|))
  saa   the plan of the sample-average problem (r = 0), solved here as an extensive
        form with no ambiguity set, --saa-cap seconds at most; only on --saa-days

What is compared is the budget: 'promised' is the value the plan was chosen by
(v^DR(N), or v^SAA(N) at r = 0) and 'oos_mean' what it earns on the test sample. A
day keeps its promise if oos_mean >= promised. The incentive side (coalitions that
gain by deviating under the true distribution) needs every coalition's value on the
test sample and is not computed here.

One job at a time under the memory watchdog of run_kl_main.py; a day whose output
exists is skipped.

    python ieee_owen/weak_eps_experiment/run_kl_oos.py --n 6,15 --saa-days 1-31
    python ieee_owen/weak_eps_experiment/run_kl_oos.py --n 30,60 --saa-days 1-10 --test 300
"""
import os, sys, glob, json, time, pickle, argparse

import run_kl_main as Q                      # chdirs to the repo root

MAIN = Q.OUT
OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'oos17_xi')


def saa_plan(n, scen, day, a):
    """The r = 0 plan as a JSON oos_plans.load_plan reads; None if it was not solved."""
    path = os.path.join(a.out, 'saa', f'n{n}_S{scen}_day{day}.json')
    if os.path.exists(path):
        return path
    cache = os.path.join(a.out, 'saa', f'n{n}_S{scen}_day{day}.pkl')
    cmd = [sys.executable, os.path.join('ieee_owen', 'stochastic_extension.py'),
           '--n', str(n), '--scenarios', str(scen), '--day', str(day),
           '--reserve-price-scale', repr(Q.PRICE_SCALE), '--ef-time-cap', str(a.saa_cap),
           '--ef-nodefile-start', '3', '--ef-only', '--ef-cache', cache, '--out', a.out]
    rc, wall, peak, _, _, killed = Q.watched(
        cmd, os.path.join(a.out, 'logs', f'saa_n{n}_S{scen}_day{day}.log'), a.min_free)
    if rc != 0 or not os.path.exists(cache):
        print(f'  SAA n={n} day {day}: failed (exit {rc}, killed {killed})', flush=True)
        return None
    with open(cache, 'rb') as f:
        ef = pickle.load(f)
    js = {'n': n, 'scenarios': scen, 'day': day, 'probs': [1.0 / scen] * scen, 'kl': None,
          'ef': {k: ef[k] for k in ('status', 'obj', 'dual_bound', 'gap', 'first_cost',
                                    'scen_cost', 'time_solve')},
          'ef_first_stage': {k: float(ef['vals'][k]) for k in ef['first_stage_names']
                             if abs(ef['vals'][k]) > 1e-9}}
    with open(path, 'w') as f:
        json.dump(json.loads(json.dumps(js, default=float)), f, indent=1)
    print(f'  SAA n={n} day {day}: {ef["status"]}, {ef["time_solve"]:.0f}s, gap {ef["gap"]:.1e}',
          flush=True)
    return path


def run_one(n, scen, day, a, want_saa):
    out = os.path.join(a.out, f'oos_n{n}_S{scen}_day{day}.json')
    if os.path.exists(out):
        return 'skipped (done)'
    found = glob.glob(os.path.join(MAIN, f'{n}p_*_day{day}_S{scen}_*_main.json'))
    if len(found) != 1:
        return f'{len(found)} main result files, skipped'
    plans = []
    if want_saa:
        saa = saa_plan(n, scen, day, a)
        if saa:
            plans.append(f'saa={saa}')
    plans.append(f'dro={found[0]}')
    cmd = [sys.executable, os.path.join('ieee_owen', 'oos_plans.py'), '--n', str(n),
           '--day', str(day), '--test', str(a.test), '--plans', ','.join(plans), '--out', out]
    rc, wall, peak, _, min_free, killed = Q.watched(
        cmd, os.path.join(a.out, 'logs', f'oos_n{n}_S{scen}_day{day}.log'), a.min_free)
    if rc != 0 or not os.path.exists(out):
        return f'failed (exit {rc}, killed {killed}, {wall:.0f}s)'
    r = json.load(open(out))['plans']
    return '  '.join(f'{k}: promised {v["promised"]:.1f} oos {v["oos_mean"]:.1f} '
                     f'(+-{v["oos_se"]:.1f})' for k, v in r.items()) + f'  {wall:.0f}s'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='6,15,30,60')
    ap.add_argument('--scen', type=int, default=20)
    ap.add_argument('--days', default='1-31')
    ap.add_argument('--saa-days', default='', help="days that also get the r = 0 plan, e.g. 1-10")
    ap.add_argument('--saa-cap', type=float, default=600.0, help='seconds for an SAA extensive form')
    ap.add_argument('--test', type=int, default=500, help='test scenarios')
    ap.add_argument('--min-free', type=float, default=0.7, help='kill below this [GB]')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    saa_days = set(Q.days(a.saa_days)) if a.saa_days else set()
    for n in [int(v) for v in a.n.split(',')]:
        for day in Q.days(a.days):
            print(f'{time.strftime("%m-%d %H:%M")}  oos n={n} |Omega|={a.scen} day {day}', flush=True)
            print(f'  {run_one(n, a.scen, day, a, day in saa_days)}', flush=True)
    print('OOS DONE', flush=True)


if __name__ == '__main__':
    main()

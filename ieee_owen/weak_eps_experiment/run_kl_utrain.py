"""
The training side of the out-of-sample bound at other sample sizes than the main
queue's (kl_experiment_plan.md, sec. 8.18 (4)): the robust Owen solution sigma* at
|Omega| scenarios, radius chi2_{1,0.95} / (2 |Omega|), which run_kl_ubound.py and
run_kl_ubound_ws.py then hold against the reference sample.

  |Omega| > --plan-scen   column generation only. The plan is the main run's (its
        --plan-scen scenarios are the first of these, so it is a plan made from part of
        the sample), valued on the larger sample (--ef-from-plan); what it earns out of
        sample is what the main run's plan earns.
  |Omega| < --plan-scen   the main queue's own run, extensive form and all, since the
        main run's plan has seen scenarios this sample has not; then its plan is valued
        on the test sample (oos_plans.py).

One job at a time under the memory watchdog of run_kl_main.py; a run whose result file
exists is skipped. Results in <out>/train.

    python ieee_owen/weak_eps_experiment/run_kl_utrain.py --scen 40,5,100 --n 6,15 --days 1
"""
import os, sys, glob, time, argparse

import run_kl_main as Q                      # chdirs to the repo root

MAIN = Q.OUT
OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'ubound17_xi', 'train')


def result(out, n, scen, day):
    return glob.glob(os.path.join(out, f'{n}p_*_day{day}_S{scen}_seed0_*.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='40,5,100', help='training samples, in this order')
    ap.add_argument('--n', default='6,15,30,60')
    ap.add_argument('--days', default='1')
    ap.add_argument('--plan-scen', type=int, default=20, help='sample of the main run')
    ap.add_argument('--test', type=int, default=500, help='test scenarios for a plan of its own')
    ap.add_argument('--budget', type=float, default=21600.0, help='seconds of column generation')
    ap.add_argument('--min-free', type=float, default=8.0, help='kill below this [GB]')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    for scen in [int(v) for v in a.scen.split(',')]:
        for n in [int(v) for v in a.n.split(',')]:
            for day in Q.days(a.days):
                tag = f'n{n}_S{scen}_day{day}'
                print(f'{time.strftime("%m-%d %H:%M")}  train n={n} |Omega|={scen} day {day} '
                      f'r={Q.radius(scen):.4g}', flush=True)
                if not result(a.out, n, scen, day):
                    if scen > a.plan_scen:
                        plan = glob.glob(os.path.join(
                            MAIN, f'{n}p_*_day{day}_S{a.plan_scen}_*_main.json'))
                        if len(plan) != 1:
                            print(f'  {len(plan)} main result files, skipped', flush=True)
                            continue
                        cmd = [sys.executable, os.path.join('ieee_owen', 'stochastic_extension.py'),
                               '--n', str(n), '--scenarios', str(scen), '--day', str(day),
                               '--kl-radius', repr(Q.radius(scen)), '--kl-master', 'dual',
                               '--reserve-price-scale', repr(Q.PRICE_SCALE), '--stall-barrier',
                               '--ef-from-plan', plan[0], '--skip-standalone',
                               '--cg-budget', str(a.budget),
                               '--tag', f'plan{a.plan_scen}', '--out', a.out]
                    else:
                        cmd = Q.command(n, scen, day, a.out)
                    rc, wall, priv, _, free, killed = Q.watched(
                        cmd, os.path.join(a.out, 'logs', f'{tag}.log'), a.min_free)
                    print(f'  exit {rc}, killed {int(killed)}, {wall:.0f}s, peak {priv:.1f} GB',
                          flush=True)
                got = result(a.out, n, scen, day)
                oos = os.path.join(a.out, f'oos_{tag}.json')
                if scen < a.plan_scen and len(got) == 1 and not os.path.exists(oos):
                    cmd = [sys.executable, os.path.join('ieee_owen', 'oos_plans.py'),
                           '--n', str(n), '--day', str(day), '--test', str(a.test),
                           '--plans', f'dro={got[0]}', '--out', oos]
                    rc, wall, *_ = Q.watched(
                        cmd, os.path.join(a.out, 'logs', f'oos_{tag}.log'), a.min_free)
                    print(f'  out of sample: exit {rc}, {wall:.0f}s', flush=True)
    print('TRAIN DONE', flush=True)


if __name__ == '__main__':
    main()

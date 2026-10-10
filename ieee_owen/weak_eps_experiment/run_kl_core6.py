"""
eps(chi) of the robust Owen allocation at n=6 over all 62 proper coalitions
(tab:results, 'measured'; kl_experiment_plan.md sec. 8.12 (2)): the main run of each
day again with --check-core, which values every coalition's KL extensive form.
One day at a time; a day whose result exists is skipped.

    python ieee_owen/weak_eps_experiment/run_kl_core6.py [--days 1-31]
"""
import os, glob, json, time, argparse

import run_kl_main as Q                      # chdirs to the repo root

OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'core6_xi')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', default='1-31')
    ap.add_argument('--scen', type=int, default=20)
    ap.add_argument('--min-free', type=float, default=0.7)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    for day in Q.days(a.days):
        print(f'{time.strftime("%m-%d %H:%M")}  core n=6 |Omega|={a.scen} day {day}', flush=True)
        found = glob.glob(os.path.join(a.out, f'6p_*_day{day}_S{a.scen}_*_main.json'))
        if found:
            print('  skipped (done)', flush=True)
            continue
        rc, wall, *_ = Q.watched(Q.command(6, a.scen, day, a.out) + ['--check-core'],
                                 os.path.join(a.out, 'logs', f'n6_S{a.scen}_day{day}.log'),
                                 a.min_free)
        found = glob.glob(os.path.join(a.out, f'6p_*_day{day}_S{a.scen}_*_main.json'))
        if rc == 0 and found:
            js = json.load(open(found[0]))
            print(f'  eps measured {js["weak_eps"]["eps"]:.6f}  eps_LR '
                  f'{js["allocation"]["eps_LR"]:.6f}  {wall:.0f}s', flush=True)
        else:
            print(f'  failed (exit {rc}, {wall:.0f}s)', flush=True)
    print('CORE DONE', flush=True)


if __name__ == '__main__':
    main()

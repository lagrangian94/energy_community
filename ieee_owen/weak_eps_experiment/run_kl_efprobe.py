"""
How far the grand coalition's KL extensive form gets within a time cap at a given
number of scenarios (kl_experiment_plan.md, sec. 8.17): the reason the numerical
study stops at |Omega| = 20 is the extensive form at n=60, and this measures it.

Each run is the extensive form of the main queue alone (run_kl_main.command with
--ef-only), at the radius chi2_{1,0.95} / (2 |Omega|), stopped at a gap of 1e-4 or
--cap seconds. One at a time under the memory watchdog; a run whose row is in
<out>/efprobe.csv is skipped.

    python ieee_owen/weak_eps_experiment/run_kl_efprobe.py --scen 40,20 --days 1-3
"""
import os, time, pickle, argparse

import run_kl_main as Q                      # chdirs to the repo root

OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'efprobe17_xi')
HEAD = ('finished,n,scen,day,r,cap_s,status,obj,bound,gap,solve_s,build_s,wall_s,exit,killed,'
        'peak_private_gb,peak_ws_gb,min_avail_gb,n_vars')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='60')
    ap.add_argument('--scen', default='40', help='in this order')
    ap.add_argument('--days', default='1-3')
    ap.add_argument('--cap', type=float, default=3600.0, help='seconds for the extensive form')
    ap.add_argument('--nodefile', type=float, default=3.0, help='EF NodefileStart [GB]')
    ap.add_argument('--min-free', type=float, default=0.7, help='kill below this [GB]')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    path = os.path.join(a.out, 'efprobe.csv')
    for scen in [int(v) for v in a.scen.split(',')]:
        for n in [int(v) for v in a.n.split(',')]:
            for day in Q.days(a.days):
                done = set()
                if os.path.exists(path):
                    done = {tuple(ln.split(',')[1:4]) for ln in open(path).read().splitlines()[1:]}
                if (str(n), str(scen), str(day)) in done:
                    continue
                tag = f'n{n}_S{scen}_day{day}'
                print(f'{time.strftime("%m-%d %H:%M")}  EF n={n} |Omega|={scen} day {day} '
                      f'r={Q.radius(scen):.4g} cap {a.cap:.0f}s', flush=True)
                cmd = Q.command(n, scen, day, a.out, nodefile=a.nodefile) + [
                    '--ef-only', '--ef-time-cap', str(a.cap)]
                rc, wall, priv, ws, free, killed = Q.watched(
                    cmd, os.path.join(a.out, 'logs', f'{tag}.log'), a.min_free)
                status = 'failed'
                obj = bound = gap = solve = build = nv = float('nan')
                cache = os.path.join(a.out, 'ef_cache', f'{tag}.pkl')
                if rc == 0 and os.path.exists(cache):
                    with open(cache, 'rb') as f:
                        ef = pickle.load(f)
                    status, obj, bound = ef['status'], ef['obj'], ef['dual_bound']
                    gap = (obj - bound) / max(abs(obj), 1.0)
                    solve, build, nv = ef['time_solve'], ef['time_build'], ef.get('n_vars', nv)
                row = (f'{time.strftime("%Y-%m-%d %H:%M")},{n},{scen},{day},{Q.radius(scen):.6g},'
                       f'{a.cap:.0f},{status},{obj:.6f},{bound:.6f},{gap:.3e},{solve:.0f},'
                       f'{build:.0f},{wall:.0f},{rc},{int(killed)},{priv:.2f},{ws:.2f},{free:.2f},{nv}')
                fresh = not os.path.exists(path)
                with open(path, 'a') as f:
                    if fresh:
                        f.write(HEAD + '\n')
                    f.write(row + '\n')
                print(f'  {row}', flush=True)
    print('PROBE DONE', flush=True)


if __name__ == '__main__':
    main()

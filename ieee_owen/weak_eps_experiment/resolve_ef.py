"""
Re-solve the KL extensive form of chosen days of the main queue for a better plan,
and put the new omega and eps into their JSON (stochastic_extension.py --ef-improve).

The reported omega is psi(EF incumbent) - LB of the column generation, so only the
incumbent of the extensive form matters, not its bound. The days worth a re-solve
are those whose incumbent sits well above the CG lower bound: at n=60, |Omega|=20
the three days solved to the gap leave omega 1.1-1.7, and of the 28 days the
30-minute cap stopped, 3, 4, 13, 14, 15, 17, 19 leave 2.8-4.8 (2026-10-10).

Each day starts from its cached plan and runs while the incumbent keeps falling: it
stops once the plan is within --margin of the CG lower bound, once --stall seconds
have gained no more than --gain, or after --cap seconds (a safety limit; the margin
is the omega of other days, not a bound, so a day need not reach it). The first
pass capped the MILP at 60 minutes and day 3 was still gaining at the cap (4.42 ->
2.84, 0.4 EUR in its last 30 minutes). The column generation is not repeated. One row per day goes to
<out>/ef_improve.csv; the first plan stays in ef_cache/<tag>.first.pkl and under
'ef_first' in the JSON.

    python ieee_owen/weak_eps_experiment/resolve_ef.py --n 60 --days 3,4,13,14,15,17,19
"""
import os, glob, json, time, argparse

import run_kl_main as Q                      # chdirs to the repo root


def run_one(n, scen, day, a):
    tag = f'n{n}_S{scen}_day{day}'
    found = glob.glob(os.path.join(a.out, f'{n}p_*_day{day}_S{scen}_*_main.json'))
    if len(found) != 1:
        return f'{tag}: {len(found)} result files, skipped'
    with open(found[0]) as f:
        js = json.load(f)
    if 'ef_first' in js and not a.again:
        return f'{tag}: re-solved already, skipped'
    cmd = Q.command(n, scen, day, a.out,
                    ef_log=os.path.join(a.out, 'logs', f'{tag}_ef_improve_gurobi.log'))
    cmd += ['--ef-improve', str(a.cap), '--ef-improve-stall', str(a.stall),
            '--ef-improve-gain', str(a.gain),
            '--ef-improve-stop', repr(js['dw']['lb'] + a.margin)]
    rc, wall, peak_priv, _, min_free, killed = Q.watched(
        cmd, os.path.join(a.out, 'logs', f'{tag}_ef_improve.log'), a.min_free)
    with open(found[0]) as f:
        new = json.load(f)
    was = {'omega_LR': js['allocation']['omega_LR'], 'eps_LR': js['allocation']['eps_LR'],
           'ef': js['ef']}                   # before this pass (ef_first: before any)
    imp = new['ef'].get('improve', {})
    row = (f'{time.strftime("%Y-%m-%d %H:%M")},{n},{scen},{day},{rc},{int(killed)},'
           f'{wall:.0f},{peak_priv:.2f},{min_free:.2f},{imp.get("status", "")},'
           f'{imp.get("time_mip", float("nan")):.0f},{was["ef"]["obj"]:.6f},'
           f'{new["ef"]["obj"]:.6f},{was["omega_LR"]:.6f},{new["allocation"]["omega_LR"]:.6f},'
           f'{was["eps_LR"]:.6f},{new["allocation"]["eps_LR"]:.6f}')
    csv = os.path.join(a.out, 'ef_improve.csv')
    fresh = not os.path.exists(csv)
    with open(csv, 'a') as f:
        if fresh:
            f.write('finished,n,scen,day,exit,killed,wall_s,peak_private_gb,min_avail_gb,'
                    'stop,mip_s,ef_before,ef_after,omega_before,omega_after,'
                    'eps_before,eps_after\n')
        f.write(row + '\n')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='60')
    ap.add_argument('--scen', type=int, default=20)
    ap.add_argument('--days', required=True, help='e.g. 3,4,13-15')
    ap.add_argument('--cap', type=float, default=10800,
                    help='safety limit on the MILP of one day [s]')
    ap.add_argument('--stall', type=float, default=1800,
                    help='stop once this long has gained no more than --gain [s]')
    ap.add_argument('--gain', type=float, default=0.1,
                    help='EUR; 0.1 is 0.0017 of eps at n=60')
    ap.add_argument('--margin', type=float, default=1.7,
                    help='stop once psi <= CG LB + this [EUR] (omega of the solved days)')
    ap.add_argument('--min-free', type=float, default=0.7, help='kill below this [GB]')
    ap.add_argument('--again', action='store_true', help='also days re-solved before')
    ap.add_argument('--out', default=Q.OUT)
    a = ap.parse_args()
    for day in Q.days(a.days):
        for n in [int(v) for v in a.n.split(',')]:
            print(f'{time.strftime("%m-%d %H:%M")}  improve n={n} |Omega|={a.scen} day {day}',
                  flush=True)
            print(f'  {run_one(n, a.scen, day, a)}', flush=True)
    print('IMPROVE DONE', flush=True)


if __name__ == '__main__':
    main()

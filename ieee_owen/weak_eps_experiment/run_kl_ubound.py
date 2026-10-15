"""
A certified bound on the out-of-sample violation of the payments of the main queue
(kl_experiment_plan.md, sec. 8.18): no coalition gains more than U per member by
leaving, on the reference sample.

The reference sample is the first --ref scenarios of the test sample of oos_plans.py
(seed e3_optimizers_curse.TEST_SEED), valued by their plain mean: it stands for the
true distribution, so no ambiguity set goes around it. One column generation on it,
with no extensive form (--ef-from-plan: the main run's first stage, recourse solved
per scenario), gives each member's pricing bound sigma_ref at the dual point of its
Lagrangian bound, and for every coalition S the members' sigma_ref add up to at least
what S earns on the reference sample. With the payments

    sigma_out_j = sigma*_j - (sum_i sigma*_i - earned) / n,

sigma* the robust Owen solution of the main run and `earned` what its plan earns on
the reference sample (so the payments add up to exactly that),

    U = max_j (sigma_ref_j - sigma_out_j)
      = [max_j d_j - mean_j d_j] + (sum_j sigma_ref_j - earned) / n,   d = sigma_ref - sigma*.

The second term is the reference problem's duality gap plus what the plan leaves on
the table there, per member. Any dual point is valid, so a run stopped at --budget
still gives a bound, a looser one.

One job at a time under the memory watchdog of run_kl_main.py; a run whose row is in
<out>/ubound.csv is skipped.

    python ieee_owen/weak_eps_experiment/run_kl_ubound.py --n 6 --ref 100,200,500 --days 1
"""
import os, sys, glob, json, time, argparse

import run_kl_main as Q                      # chdirs to the repo root

MAIN = Q.OUT
OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'ubound17_xi')
TEST_SEED = 1000                             # e3_optimizers_curse.TEST_SEED
HEAD = ('finished,n,scen,ref,day,status,iterations,plan_s,cg_s,wall_s,exit,killed,'
        'peak_private_gb,min_avail_gb,earned,lb,ub,U,spread,gap_per_member,shift,eps_K')


def ubound(main, ref):
    """U and its parts [EUR per member] from a main-run JSON and a reference-run JSON."""
    players = list(main['dw']['sigma'])
    n = len(players)
    star = {u: -main['dw']['sigma'][u] for u in players}             # profit convention
    sref = {u: -ref['dw']['sigma_bound'][u] for u in players}
    earned = -ref['ef']['obj']
    out = {u: star[u] - (sum(star.values()) - earned) / n for u in players}
    d = [sref[u] - star[u] for u in players]
    return {'U': max(sref[u] - out[u] for u in players),
            'spread': max(d) - sum(d) / n,
            'gap_per_member': (sum(sref.values()) - earned) / n,
            'shift': (earned + main['ef']['obj']) / n,               # earned - promised
            'eps_K': main['allocation']['eps_LR'], 'earned': earned}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='6')
    ap.add_argument('--scen', type=int, default=20, help='training sample of the main run')
    ap.add_argument('--ref', default='100', help='reference scenarios, in this order')
    ap.add_argument('--days', default='1')
    ap.add_argument('--budget', type=float, default=21600.0,
                    help='seconds for the column generation on the reference sample')
    ap.add_argument('--min-free', type=float, default=4.0, help='kill below this [GB]')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    path = os.path.join(a.out, 'ubound.csv')
    for m in [int(v) for v in a.ref.split(',')]:
        for n in [int(v) for v in a.n.split(',')]:
            for day in Q.days(a.days):
                done = set()
                if os.path.exists(path):
                    done = {tuple(ln.split(',')[1:5]) for ln in open(path).read().splitlines()[1:]}
                if (str(n), str(a.scen), str(m), str(day)) in done:
                    continue
                found = glob.glob(os.path.join(MAIN, f'{n}p_*_day{day}_S{a.scen}_*_main.json'))
                if len(found) != 1:
                    print(f'n={n} day {day}: {len(found)} main result files, skipped', flush=True)
                    continue
                tag = f'n{n}_S{a.scen}_ref{m}_day{day}'
                print(f'{time.strftime("%m-%d %H:%M")}  U n={n} |Omega|={a.scen} ref {m} '
                      f'day {day}', flush=True)
                cmd = [sys.executable, os.path.join('ieee_owen', 'stochastic_extension.py'),
                       '--n', str(n), '--scenarios', str(m), '--day', str(day),
                       '--seed', str(TEST_SEED), '--reserve-price-scale', repr(Q.PRICE_SCALE),
                       '--stall-barrier', '--ef-from-plan', found[0], '--skip-standalone',
                       '--cg-budget', str(a.budget), '--tag', f'ref_S{a.scen}', '--out', a.out]
                rc, wall, priv, _, free, killed = Q.watched(
                    cmd, os.path.join(a.out, 'logs', f'{tag}.log'), a.min_free)
                got = glob.glob(os.path.join(
                    a.out, f'{n}p_*_day{day}_S{m}_seed{TEST_SEED}_*_ref_S{a.scen}.json'))
                nan = float('nan')
                status, its, plan_s, cg_s, lb, ub = 'failed', 0, nan, nan, nan, nan
                r = dict.fromkeys(('earned', 'U', 'spread', 'gap_per_member', 'shift', 'eps_K'), nan)
                if rc == 0 and len(got) == 1:
                    ref = json.load(open(got[0], encoding='utf-8'))
                    dw = ref['dw']
                    status, its, plan_s, cg_s = dw['status'], dw['iterations'], \
                        ref['ef']['time_solve'], dw['time']
                    lb, ub = dw['lb'], dw['ub']
                    r = ubound(json.load(open(found[0], encoding='utf-8')), ref)
                row = (f'{time.strftime("%Y-%m-%d %H:%M")},{n},{a.scen},{m},{day},{status},{its},'
                       f'{plan_s:.0f},{cg_s:.0f},{wall:.0f},{rc},{int(killed)},{priv:.2f},'
                       f'{free:.2f},{r["earned"]:.4f},{lb:.4f},{ub:.4f},{r["U"]:.4f},'
                       f'{r["spread"]:.4f},{r["gap_per_member"]:.4f},{r["shift"]:.4f},'
                       f'{r["eps_K"]:.4f}')
                fresh = not os.path.exists(path)
                with open(path, 'a') as f:
                    if fresh:
                        f.write(HEAD + '\n')
                    f.write(row + '\n')
                print(f'  {row}', flush=True)
    print('UBOUND DONE', flush=True)


if __name__ == '__main__':
    main()

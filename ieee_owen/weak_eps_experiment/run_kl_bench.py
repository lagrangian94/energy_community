"""
Benchmarks of the nested RCG (Table II of the uncertainty manuscript), on the runs
of the main queue (run_kl_main.py, same instances, same extensive-form cache):

  coalgen  coalition generation on the KL-DRO game (coalition_bench.py)
  flat     flat Dantzig-Wolfe: no scenario split, no MP1/MP2 (--no-split-scenarios
           --no-mp12), tangent rows generated as in the RCG
  conic    the RCG's pricing with an exponential-cone master (MOSEK) and no row
           generation (--kl-master conic)
  rcg      the nested RCG itself, timed the same way (not in the default list: on
           the PC of the main queue its time comes from the main runs' logs; on any
           other machine add it, --methods rcg,conic,flat,coalgen, since times from
           two machines do not compare)

Each run has --budget seconds (3,600) for its algorithm. The column generations
stop themselves there (--cg-budget) and still report their bound and trajectory; a
run still alive --grace seconds later is killed. One at a time, with the memory
watchdog of run_kl_main.py.

What is compared is the time to the bound, not the time to the stopping test: for
flat and conic, t_target is the first moment their Lagrangian bound is within the
RCG's own tolerance (5% of omega) of the RCG's final bound on that instance. The
stopping test is the RCG's, tuned on the nested master, and at n=6, day 2 flat DW
had the RCG's bound after ~400 s and then spent an hour 0.013 EUR short of the
test of a penalty round (RMP - LB 0.313 against 0.300). A run therefore stops as
soon as its bound is there (--cg-target-lb). Status: 'target' (bound reached, at
t_target), 'converged' (own stopping test met below the target bound), 'timeout'
(neither in the budget), 'failed'.

Days 1-3 come first for every size. A method that times out on all three at a size
is not run on the other days of that size (reported "> budget (0/3)"), and
coalition generation and flat DW are then not run at any larger size either. Days 4-5 follow,
then 6-31, so the table is complete on five days before the 31-day means move in.
A run with a row in <out>/bench.csv is skipped.

    python ieee_owen/weak_eps_experiment/run_kl_bench.py
    python ieee_owen/weak_eps_experiment/run_kl_bench.py --methods conic --n 6 --days 1-3
"""
import os, sys, glob, json, time, argparse

import run_kl_main as Q                      # chdirs to the repo root

MAIN = Q.OUT
OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'bench17_xi')
METHODS = ('conic', 'flat', 'coalgen')        # in the order they are run
SIZES = (6, 15, 30, 60)
# once out of budget at a size, not run at any larger one either
CASCADE = ('coalgen', 'flat')
HEAD = ('finished,method,n,scen,day,status,alg_s,wall_s,exit,killed,peak_private_gb,'
        'min_avail_gb,iterations,lb,obj,eps,t_target,lb_short')
OMEGA_TOL = 0.05                # the RCG's stopping tolerance, as a share of omega


def target(n, scen, day):
    """The bound a benchmark has to reach: the RCG's final LB less its tolerance."""
    found = glob.glob(os.path.join(MAIN, f'{n}p_*_day{day}_S{scen}_*_main.json'))
    js = json.load(open(found[0]))
    ef = js.get('ef_first', {'ef': js['ef']})['ef']['obj']      # the 30-minute plan
    lb = js['dw']['lb']
    return lb - OMEGA_TOL * (ef - lb)


def time_to(log, goal, total):
    """First time the logged LB is at least goal (inf if never). Logs without 't'
    (before 2026-10-10) get the LP and pricing times, scaled to the run's total."""
    if log and 't' not in log[-1]:
        cum, acc = [], 0.0
        for e in log:
            acc += e.get('t_lp', 0.0) + e.get('t_price', 0.0)
            cum.append(acc)
        ts = [c * total / acc for c in cum] if acc > 0 else cum
    else:
        ts = [e['t'] for e in log]
    for t, e in zip(ts, log):
        if e['lb'] >= goal:
            return t
    return float('inf')


def rows(a):
    """{(method, n, scen, day): status} of the runs recorded so far."""
    path = os.path.join(a.out, 'bench.csv')
    if not os.path.exists(path):
        return {}
    out = {}
    for ln in open(path).read().splitlines()[1:]:
        f = ln.split(',')
        out[(f[1], int(f[2]), int(f[3]), int(f[4]))] = f[5]
    return out


def dead(method, n, scen, done):
    """Timed out on days 1-3 at this size, or (CASCADE methods) at a smaller one."""
    def out3(m):
        return all(done.get((method, m, scen, d)) == 'timeout' for d in (1, 2, 3))
    if out3(n):
        return True
    return method in CASCADE and any(out3(m) for m in SIZES if m < n)


def command(method, n, scen, day, a):
    tag = f'n{n}_S{scen}_day{day}'
    game = ['--n', str(n), '--scenarios', str(scen), '--day', str(day),
            '--kl-radius', repr(Q.radius(scen)), '--reserve-price-scale', repr(Q.PRICE_SCALE)]
    if method == 'coalgen':
        return [sys.executable, os.path.join('ieee_owen', 'weak_eps_experiment',
                                             'coalition_bench.py'),
                '--budget', str(a.budget),
                '--result', os.path.join(a.out, f'coalgen_{tag}.json')] + game
    # the plan the main run was made with (resolve_ef.py keeps it as .first.pkl)
    cache = os.path.join(MAIN, 'ef_cache', f'{tag}.first.pkl')
    if not os.path.exists(cache):
        cache = os.path.join(MAIN, 'ef_cache', f'{tag}.pkl')
    cmd = [sys.executable, os.path.join('ieee_owen', 'stochastic_extension.py')] + game + [
        '--stall-barrier', '--mip-time-limit', '10800', '--ef-cache', cache,
        '--skip-standalone', '--tag', method, '--out', a.out, '--cg-budget', str(a.budget),
        '--cg-target-lb', repr(target(n, scen, day))]
    if method == 'flat':
        return cmd + ['--kl-master', 'dual', '--no-split-scenarios', '--no-mp12']
    if method == 'rcg':
        return cmd + ['--kl-master', 'dual']
    return cmd + ['--kl-master', 'conic']


def run_one(method, n, scen, day, a):
    tag = f'n{n}_S{scen}_day{day}'
    log = os.path.join(a.out, 'logs', f'{method}_{tag}.log')
    rc, wall, peak, _, min_free, killed = Q.watched(
        command(method, n, scen, day, a), log, a.min_free, kill_after=a.budget + a.grace)
    alg = it = lb = obj = eps = t_tar = short = float('nan')
    status = 'failed'
    if method == 'coalgen':
        path = os.path.join(a.out, f'coalgen_{tag}.json')
        if rc == 0 and os.path.exists(path):
            js = json.load(open(path))
            alg, it, eps = js['time'], js['coalitions'], js['weak_eps']
            status = 'converged' if js['converged'] else 'timeout'
    else:
        found = glob.glob(os.path.join(a.out, f'{n}p_*_day{day}_S{scen}_*_{method}.json'))
        if rc == 0 and len(found) == 1:
            js = json.load(open(found[0]))
            dw = js['dw']
            alg, it, lb, obj = dw['time'], dw['iterations'], dw['lb'], dw['obj']
            eps = js['allocation']['eps_LR']
            goal = target(n, scen, day)
            t_tar, short = time_to(js['cg_log'], goal, alg), goal - lb
            status = ('target' if t_tar <= a.budget else
                      'converged' if dw['status'] in ('optimal', 'stalled') and alg <= a.budget
                      else 'timeout')
    if killed and wall >= a.budget:
        status = 'timeout'
    row = (f'{time.strftime("%Y-%m-%d %H:%M")},{method},{n},{scen},{day},{status},{alg:.0f},'
           f'{wall:.0f},{rc},{int(killed)},{peak:.2f},{min_free:.2f},{it},{lb:.6f},{obj:.6f},'
           f'{eps:.6f},{t_tar:.0f},{short:.6f}')
    path = os.path.join(a.out, 'bench.csv')
    fresh = not os.path.exists(path)
    with open(path, 'a') as f:
        if fresh:
            f.write(HEAD + '\n')
        f.write(row + '\n')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--scen', type=int, default=20)
    ap.add_argument('--n', default=','.join(map(str, SIZES)))
    ap.add_argument('--days', default='1-31')
    ap.add_argument('--budget', type=float, default=3600.0, help='seconds per run')
    ap.add_argument('--grace', type=float, default=600.0,
                    help='seconds over budget (model building) before a run is killed')
    ap.add_argument('--min-free', type=float, default=0.7, help='kill below this [GB]')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    sizes = [int(v) for v in a.n.split(',')]
    methods = a.methods.split(',')
    want = set(Q.days(a.days))
    # days 1-3 by method and size (so the early stop can act, and the comparison that
    # matters, conic against the RCG, is in before the methods that mostly time out),
    # then 4-5, then the rest by day
    order = [(n, m, d) for m in methods for n in sizes for d in (1, 2, 3)]
    for block in ((4, 5), range(6, 32)):
        order += [(n, m, d) for d in block for n in sizes for m in methods]
    for n, m, d in order:
        if d not in want:
            continue
        done = rows(a)
        if (m, n, a.scen, d) in done:
            continue
        if dead(m, n, a.scen, done):
            continue
        print(f'{time.strftime("%m-%d %H:%M")}  {m} n={n} |Omega|={a.scen} day {d}', flush=True)
        print(f'  {run_one(m, n, a.scen, d, a)}', flush=True)
    print('BENCH DONE', flush=True)


if __name__ == '__main__':
    main()

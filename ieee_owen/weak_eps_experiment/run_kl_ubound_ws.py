"""
The scenario-wise bound on the out-of-sample violation (kl_experiment_plan.md,
sec. 8.18 (5)): U^ws, for when the joint bound U of run_kl_ubound.py does not fit.

Each reference scenario gets a column generation of its own, the one-scenario
problem, and each member the mean u_j of its pricing bounds over the scenarios. A
coalition that plans before it sees the scenario earns no more than one that plans
after, so for every coalition S the members' u_j add up to at least what S earns on
the reference sample, and

    U^ws = max_j (u_j - sigma_out_j)

bounds the violation as U does, looser by the value of knowing the scenario. The
payments sigma_out are those of run_kl_ubound.py: the main run's robust Owen solution
moved so that it adds up to what the main run's plan earns on the same scenarios.

The scenarios are independent jobs: --procs worker processes take them in strides,
each building the instance once. A scenario's row goes to
<out>/ws/n<n>_S<scen>_day<day>/w<i>of<k>.jsonl as soon as it is done, and a restart
skips the rows that are there. No extensive form is solved: the plan of the main run
is valued on the scenario (plan_extensive_form) and seeds the column generation.

    python ieee_owen/weak_eps_experiment/run_kl_ubound_ws.py --n 6 --days 1 --ref 500
"""
import os, sys, gc, glob, json, time, argparse, subprocess

import run_kl_main as Q                      # chdirs to the repo root

MAIN = Q.OUT
OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'ubound17_xi')
TEST_SEED = 1000                             # e3_optimizers_curse.TEST_SEED
HEAD = ('finished,n,scen,ref,day,done,not_optimal,wall_s,cg_s_sum,cg_s_max,earned,'
        'U_ws,spread,gap_per_member,shift,eps_K')


def main_json(n, scen, day):
    found = glob.glob(os.path.join(MAIN, f'{n}p_*_day{day}_S{scen}_*_main.json'))
    return found[0] if len(found) == 1 else None


def folder(a, n, day):
    return os.path.join(a.out, 'ws', f'n{n}_S{a.scen}_day{day}')


def rows(a, n, day):
    """{scenario: row} of every scenario done so far, whichever worker wrote it."""
    out = {}
    for f in glob.glob(os.path.join(folder(a, n, day), '*.jsonl')):
        for ln in open(f, encoding='utf-8'):
            if ln.strip():
                r = json.loads(ln)
                out[r['w']] = r
    return out


def worker(a):
    """Scenarios i, i + k, i + 2k, ... below --ref of one (n, day), in this process."""
    sys.path.insert(0, os.path.join(Q._ROOT, 'ieee_owen'))
    sys.path.insert(0, Q._ROOT)
    import stochastic_extension as SE
    import e3_optimizers_curse as E3
    n, day = int(a.n), int(a.days)
    i, k = (int(v) for v in a.worker.split('/'))
    plan = json.load(open(main_json(n, a.scen, day), encoding='utf-8'))['ef_first_stage']
    players, T, base, name = E3.instance(n, day)
    args = SE.build_parser().parse_args(['--pricing-workers', '1'])
    SE._configure_direct(args)
    path = os.path.join(folder(a, n, day), f'w{i}of{k}.jsonl')
    done = rows(a, n, day)
    for w in range(i, int(a.ref), k):
        if w in done:
            continue
        t0 = time.time()
        scen = SE.make_scenarios(base, players, T, 1, seed=TEST_SEED, day=day, first=w)
        ef = SE.plan_extensive_form(players, T, scen, plan, workers=1, threads=a.threads)
        kw = SE._direct_kwargs(args, ef)
        kw.pop('init_vals')
        master = SE.DirectMaster(players, T, scen, base, **kw)
        for s in master.subs.values():
            s.set_threads(a.threads)
        dw = master.solve(init_vals=ef['vals'])
        row = {'w': w, 'plan_cost': ef['obj'], 'status': dw['status'], 'lb': dw['lb'],
               'ub': dw['ub'], 'iterations': dw['iterations'], 'cg_s': dw['time'],
               'wall_s': time.time() - t0, 'sigma_bound': dw['sigma_bound']}
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        for s in master.subs.values():
            s.g.dispose()
        for e in master._envs:
            if e is not None:
                e.dispose()
        del master, dw, ef, scen
        gc.collect()


def uws(main, done):
    """U^ws and its parts [EUR per member] from a main-run JSON and the scenarios' rows."""
    players = list(main['dw']['sigma'])
    n, m = len(players), len(done)
    star = {u: -main['dw']['sigma'][u] for u in players}             # profit convention
    ubar = {u: -sum(r['sigma_bound'][u] for r in done.values()) / m for u in players}
    earned = -sum(r['plan_cost'] for r in done.values()) / m
    out = {u: star[u] - (sum(star.values()) - earned) / n for u in players}
    d = [ubar[u] - star[u] for u in players]
    return {'U_ws': max(ubar[u] - out[u] for u in players), 'spread': max(d) - sum(d) / n,
            'gap_per_member': (sum(ubar.values()) - earned) / n,
            'shift': (earned + main['ef']['obj']) / n,
            'eps_K': main['allocation']['eps_LR'], 'earned': earned,
            'u_bar': ubar, 'sigma_out': out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='6')
    ap.add_argument('--scen', type=int, default=20, help='training sample of the main run')
    ap.add_argument('--ref', default='500', help='reference scenarios (the first so many)')
    ap.add_argument('--days', default='1')
    ap.add_argument('--procs', type=int, default=8, help='worker processes')
    ap.add_argument('--threads', type=int, default=2, help='solver threads per worker')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--worker', default=None, help='internal: i/k, the scenarios of one worker')
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    m = int(a.ref)
    path = os.path.join(a.out, 'ubound_ws.csv')
    for n in [int(v) for v in a.n.split(',')]:
        for day in Q.days(a.days):
            src = main_json(n, a.scen, day)
            if src is None:
                print(f'n={n} day {day}: no single main result file, skipped', flush=True)
                continue
            os.makedirs(os.path.join(folder(a, n, day), 'logs'), exist_ok=True)
            print(f'{time.strftime("%m-%d %H:%M")}  U^ws n={n} |Omega|={a.scen} ref {m} '
                  f'day {day}', flush=True)
            t0 = time.time()
            if len([w for w in rows(a, n, day) if w < m]) < m:
                env = dict(os.environ, PYTHONHASHSEED='21', PYTHONUNBUFFERED='1',
                           PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
                ps = []
                for i in range(a.procs):
                    log = open(os.path.join(folder(a, n, day), 'logs', f'w{i}of{a.procs}.log'),
                               'a', encoding='utf-8')
                    ps.append(subprocess.Popen(
                        [sys.executable, os.path.abspath(__file__), '--n', str(n), '--scen',
                         str(a.scen), '--ref', str(m), '--days', str(day), '--threads',
                         str(a.threads), '--out', a.out, '--worker', f'{i}/{a.procs}'],
                        stdout=log, stderr=subprocess.STDOUT, env=env))
                rcs = [p.wait() for p in ps]
                if any(rcs):
                    print(f'  worker exits {rcs}', flush=True)
            done = {w: r for w, r in rows(a, n, day).items() if w < m}
            if not done:
                print('  no scenario done', flush=True)
                continue
            r = uws(json.load(open(src, encoding='utf-8')), done)
            with open(os.path.join(a.out, f'ws_n{n}_S{a.scen}_ref{m}_day{day}.json'), 'w') as f:
                json.dump({'n': n, 'scen': a.scen, 'ref': m, 'day': day, 'done': len(done),
                           **r}, f, indent=1)
            cg = [x['cg_s'] for x in done.values()]
            row = (f'{time.strftime("%Y-%m-%d %H:%M")},{n},{a.scen},{m},{day},{len(done)},'
                   f'{sum(x["status"] != "optimal" for x in done.values())},'
                   f'{time.time() - t0:.0f},{sum(cg):.0f},{max(cg):.0f},{r["earned"]:.4f},'
                   f'{r["U_ws"]:.4f},{r["spread"]:.4f},{r["gap_per_member"]:.4f},'
                   f'{r["shift"]:.4f},{r["eps_K"]:.4f}')
            fresh = not os.path.exists(path)
            with open(path, 'a') as f:
                if fresh:
                    f.write(HEAD + '\n')
                f.write(row + '\n')
            print(f'  {row}', flush=True)
    print('UBOUND WS DONE', flush=True)


if __name__ == '__main__':
    main()

"""
Storage and trade-direction complementarity: none vs big-M vs SOS1.

For each baseline size and day, solves the grand-coalition MILP (v^MIP) and the column
generation (v^LR) with complementarity None / 'bigm' / 'sos' on Gurobi (--solvers highs
adds None / 'bigm' on HiGHS, which does not read SOS constraints) and records
values, times and whether the solution uses simultaneous operation. The column
generation starts from the same setting's MILP solution, as run_multiday.owen_day does.

Runs one instance at a time so the timings do not compete for cores. Resumable: a
(run, day, complementarity, solver) row already in the CSV is skipped.

Usage:
  python ieee_owen/weak_eps_experiment/complementarity_bench.py --sizes 6,15,30 --days 1,2,3
"""
import os, sys, csv, time, argparse
_PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import run_multiday as RM
from compact_utility import LocalEnergyMarket
from chp import ColumnGenerationSolver

SETTINGS = [(None, 'highs'), ('bigm', 'highs'),
            (None, 'gurobi'), ('bigm', 'gurobi'), ('sos', 'gurobi')]
COLS = ['run', 'day', 'n_players', 'complementarity', 'solver',
        'n_binary', 'n_cc_binary', 'n_sos', 'mip_status', 'v_mip', 't_mip_s',
        'v_lr', 't_cg_s', 'gap', 'scd_hours', 'sim_trade_hours']
EPS = 1e-6


def simultaneous(res, players):
    """Member-hours with both charge and discharge, and with both buying and selling."""
    scd = 0
    for k in ('E', 'G', 'H'):
        ch, dis = res.get(f'b_ch_{k}', {}), res.get(f'b_dis_{k}', {})
        scd += sum(1 for key in ch if min(ch[key], dis.get(key, 0.0)) > EPS)
    sim = 0
    for k in ('E', 'G', 'H'):
        side = lambda pre: [res.get(f'{pre}_{k}_com', {}), res.get(f'{pre}_{k}_gri', {})]
        buy, sell = side('i'), side('e')
        for u in players:
            for t in RM.T:
                b = sum(d.get((u, t), 0.0) for d in buy)
                s = sum(d.get((u, t), 0.0) for d in sell)
                sim += min(b, s) > EPS
    return scd, sim


def bench(run, day, cc, solver, time_limit):
    params = RM.build_params(run, day)
    params['complementarity'] = cc
    players = run['players']
    t0 = time.time()
    lem = LocalEnergyMarket(players, RM.T, params, model_type='mip', mipsolver=solver)
    lem.model.hideOutput()
    lem.model.setParam('limits/time', float(time_limit))
    n_bin = sum(v.vtype() == 'BINARY' for v in lem.model.getVars())
    n_cc = len(getattr(lem, 'y_sto', {})) + len(getattr(lem, 'y_dir', {}))
    status = lem.solve()
    if status not in ('optimal', 'timelimit'):
        raise RuntimeError(f"{run['name']} day {day} {cc}/{solver}: MILP status {status}")
    from compact_utility import solve_and_extract_results
    res = solve_and_extract_results(lem.model)[1]
    v_mip, t_mip = float(lem.model.getObjVal()), time.time() - t0
    scd, sim = simultaneous(res, players)

    init = {k: v for k, v in res.items() if isinstance(v, dict)}
    t0 = time.time()
    cg = ColumnGenerationSolver(players, RM.T, params, model_type='mip',
                                init_sol=init, smoothing=True, mipsolver=solver)
    _, _, v_lr, _ = cg.solve()
    t_cg = time.time() - t0
    return dict(run=run['name'], day=day, n_players=len(players),
                complementarity=cc or 'none', solver=solver, n_binary=n_bin,
                n_cc_binary=n_cc, n_sos=len(lem._sos_pairs), mip_status=status,
                v_mip=v_mip, t_mip_s=round(t_mip, 2), v_lr=float(v_lr),
                t_cg_s=round(t_cg, 2), gap=v_mip - float(v_lr),
                scd_hours=scd, sim_trade_hours=sim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='6,15,30')
    ap.add_argument('--days', default='1,2,3')
    ap.add_argument('--time-limit', type=float, default=1800.0)
    ap.add_argument('--solvers', default='gurobi',
                    help="comma list of 'gurobi', 'highs' (HiGHS runs none/bigm only)")
    ap.add_argument('--out', default=os.path.join(RM.OUT, 'complementarity_bench.csv'))
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {(r['run'], int(r['day']), r['complementarity'], r['solver'])
                    for r in csv.DictReader(f)}
    new = not os.path.exists(a.out)
    for n in a.sizes.split(','):
        run = next(r for r in RM.RUNS if r['name'] == f'baseline_{n}p')
        for day in map(int, a.days.split(',')):
            for cc, solver in SETTINGS:
                if solver not in a.solvers.split(','):
                    continue
                if (run['name'], day, cc or 'none', solver) in done:
                    continue
                row = bench(run, day, cc, solver, a.time_limit)
                with open(a.out, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=COLS)
                    if new:
                        w.writeheader()
                        new = False
                    w.writerow(row)
                print(', '.join(f'{k}={row[k]}' for k in COLS), flush=True)


if __name__ == '__main__':
    main()

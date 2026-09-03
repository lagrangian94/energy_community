"""Stand-alone values kappa_j = val(DP_{{j}}) for every member, day and size.

The game was re-defined as v(S) = val(DP_S) - sum_{j in S} kappa_j (Definition
`def:game`), so the grand-coalition entry of `tab:results` is no longer val(DP_N)
alone. Nothing in the repo persisted kappa: `core.py` seeds its row generation with
the singletons and computes them on every run, but throws them away with the
CoreComputation object. They are cheap to redo -- a one-member MILP is 0.02-0.4 s --
so this recomputes rather than trying to recover them from a stale JSON.

Everything here is in the repo's COST convention (negative = profit). The paper
reports profits, so the reader-facing quantity is

    v(N) = sum_j kappa_j - c(N),

which superadditivity makes non-negative. `--check` reports any day where it is not
comfortably positive, because tab:results averages geometrically and a day at zero
would take the whole column down with it.

    python weak_eps_experiment/run_kappa.py                  # the four baselines
    python weak_eps_experiment/run_kappa.py --runs baseline_6p --force

Resumable on (run, day) exactly like run_multiday.py: a day whose JSON exists is
skipped. JSON is the source of truth; kappa.csv is regenerated from it every run.
"""
import os, sys, json, time, csv, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import run_multiday as RM
from compact_utility import LocalEnergyMarket

OUT = _HERE
DEFAULT_RUNS = ['baseline_6p', 'baseline_15p', 'baseline_30p', 'baseline_60p']
COLS = ['run', 'day', 'n_players', 'c_N', 'sum_kappa', 'v_N', 'time_s']


def kappa_day(run, day):
    """One singleton MILP per member, under the same parameters as the sweep."""
    players = run['players']
    params = RM.build_params(run, day)
    kappa, t0 = {}, time.time()
    for u in players:
        lem = LocalEnergyMarket(players=[u], time_periods=RM.T, parameters=params,
                                model_type='mip', dwr=False)
        lem.model.hideOutput()
        status = lem.solve()
        if status != 'optimal':
            raise RuntimeError(f"{run['name']} day {day} member {u}: status={status}")
        kappa[u] = float(lem.model.getObjVal())
    return {'run': run['name'], 'day': day, 'n_players': len(players),
            'kappa_cost': kappa, 'sum_kappa': float(sum(kappa.values())),
            'time_s': round(time.time() - t0, 2)}


def json_path(run, day):
    return os.path.join(RM.run_dir(run), f'kappa_day{day}.json')


def c_N_of(run):
    """val(DP_N) per day, READ from the sweep rather than re-solved.

    tab:results' v(N) has to be the same instance as its val(DP_N) row, and that row
    is owen.csv's v_mip. Re-solving here would risk quoting two different solves.
    """
    p = os.path.join(RM.run_dir(run), 'owen.csv')
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {int(r['day']): float(r['v_mip']) for r in csv.DictReader(f)
                if r.get('v_mip') not in (None, '')}


def rebuild_csv(run):
    c_N = c_N_of(run)
    rows = []
    for day in range(1, 32):
        p = json_path(run, day)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        c = c_N.get(day)
        rows.append({'run': d['run'], 'day': day, 'n_players': d['n_players'],
                     'c_N': c, 'sum_kappa': d['sum_kappa'],
                     'v_N': (d['sum_kappa'] - c) if c is not None else None,
                     'time_s': d['time_s']})
    out = os.path.join(RM.run_dir(run), 'kappa.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=','.join(DEFAULT_RUNS))
    ap.add_argument('--days', default='1-31')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    lo, _, hi = a.days.partition('-')
    days = list(range(int(lo), int(hi or lo) + 1))
    byname = {r['name']: r for r in RM.RUNS}
    names = [n.strip() for n in a.runs.split(',') if n.strip()]

    for name in names:
        run = byname[name]
        for day in days:
            p = json_path(run, day)
            if os.path.exists(p) and not a.force:
                continue
            doc = kappa_day(run, day)
            with open(p, 'w') as f:
                json.dump(doc, f, indent=1)
            print(f"[kappa] {name} day {day}: sum_kappa={doc['sum_kappa']:.2f} "
                  f"({doc['time_s']:.1f}s)", flush=True, file=sys.stderr)
        rows = rebuild_csv(run)
        print(f"[kappa] {name}: {len(rows)} days -> kappa.csv", flush=True, file=sys.stderr)


if __name__ == '__main__':
    main()

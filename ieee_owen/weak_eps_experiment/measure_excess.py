"""Per-capita excess of the executed allocation, per day and per size.

This produces the last two rows of Table `tab:results` in the draft: how many days the
proposed allocation lands in the exact core, and how far outside it lands on the rest.

WHAT IS MEASURED, AND ON WHAT. The paper's rule is the corrected Owen point
`chi^LR - eps^LR * 1`, and that is what the separation is run on -- not `chi^LR`. Running
it on `chi^LR` produces nothing usable: with no violating coalition
`core._measure_weak_eps_separation` short-circuits and returns the raw excess with the
EMPTY set attaining it, i.e. `0`, which is a floor rather than the margin. The margin of
`chi^LR` is recovered arithmetically instead, since adding a constant `c` to every share
raises the per-capita excess by exactly `c`:

    eps(chi^LR) = eps(chi^LR - eps^LR 1) - eps^LR .

WHY IT IS EXPENSIVE. `eps(chi) = max_S (v(S) - sum_S chi)/|S|` is fractional in `S`, so
each evaluation is a Dinkelbach sequence of separation MIPs rather than one solve. At
n = 30 the 30 days took 45 h in total, median 73 min, and the slowest days are the ones
whose answer is near zero -- there the optimum cannot be pruned against an incumbent, so
the search runs long. That cost is itself evidence for the `sec:complexity` argument about
the row-generation oracle.

NO COLUMN GENERATION IS REPEATED. `chi^LR` and the duality gap are read from the stored
per-day runs under `maximin_slack1e-7/`, which carry `owen_sigma_cost` and `gap`. Those
fields are unaffected by the (R2) defect that invalidated the maximin selection in the
same files (see `validation_plan.md`); only the separation is solved here.

Usage:
  python weak_eps_experiment/measure_excess.py --n 6 --days 1-30
  python weak_eps_experiment/measure_excess.py --n 30 --days 1-30 --stage 1   # sign only
  python weak_eps_experiment/measure_excess.py --n 30 --days 1-30 --stage 2   # full value

Stage 1 solves one raw separation MIP and reports only whether the allocation is in the
core; it was intended as a cheap screen but is not much cheaper than stage 2 at n = 30,
because the single MIP is the expensive part and the Dinkelbach iterations after it are
comparatively quick. Results append to `excess_<n>p.json`; a day already present is
skipped, so the sweep is resumable.
"""
import os, sys, json, time, argparse, io, contextlib

_PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from run_experiment import build_instance
from run_maximin import set_day
from core import CoreComputation

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OUT, 'maximin_slack1e-7')


def allocation_file(n, day):
    """Where the day's `chi^LR` lives.

    Days 1-30 were produced before the (R2) fix and archived under `maximin_slack1e-7/`;
    anything run since lands in the experiment directory itself. Only `owen_sigma_cost`
    and `gap` are read, and those are identical either way -- the (R2) slack affected the
    maximin selection in those files and nothing upstream of it.
    """
    for d in (SRC, OUT):
        p = os.path.join(d, f'maximin_{n}p_day{day}.json')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no stored run for n={n}, day={day}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--days', default='1-30')
    ap.add_argument('--stage', type=int, default=2, choices=(1, 2))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.days.split('-')) if '-' in a.days else (int(a.days),) * 2

    path = os.path.join(OUT, f'excess_{a.n}p.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    players, config, T, _, _ = build_instance(a.n)

    for d in range(lo, hi + 1):
        k, cur = str(d), res.get(str(d), {})
        if a.stage == 1 and 'raw' in cur:
            continue
        if a.stage == 2 and 'excess' in cur:
            continue

        j = json.load(open(allocation_file(a.n, d)))
        sigma, gap = j['owen_sigma_cost'], j['gap']
        eps = abs(gap) / a.n
        # cost convention inside the model: the correction ADDS eps to every cost share
        chi = {u: sigma[u] - gap / a.n for u in players}

        params = set_day(d, a.n, players, config, T)
        cc = CoreComputation(players, 'mip', T, params)
        buf, t0 = io.StringIO(), time.time()
        with contextlib.redirect_stdout(buf):
            for u in players:            # check_imputation indexes the singletons
                cc.compute_coalition_cost([u])
            if a.stage == 1:
                S, val = cc.find_violated_coalition(chi)
            else:
                S, val, _ = cc.measure_stability_violation(chi, brute_force=False)
        dt = time.time() - t0

        cur.update(day=d, eps=eps)
        if a.stage == 1:
            cur.update(raw=float(val), raw_S=list(S), t_stage1=dt,
                       in_core=bool(val <= 1e-6 or len(S) == 0))
            print(f"n={a.n} day {d:>2} [S1]: raw={val:+.6f} "
                  f"{'IN CORE' if cur['in_core'] else 'outside'}  |S|={len(S)}  ({dt:.0f}s)",
                  flush=True)
        else:
            cur.update(excess=float(val), coalition=list(S), t_stage2=dt)
            print(f"n={a.n} day {d:>2} [S2]: excess={val:+.6f}  eps^LR={eps:.4f}  "
                  f"|S|={len(S)}  ({dt:.0f}s)", flush=True)
        res[k] = cur
        json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()

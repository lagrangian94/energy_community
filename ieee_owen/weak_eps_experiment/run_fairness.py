"""Fairness of the duality allocation against the least-dispersed core element.

The Owen point chi^LR is derived from a dual, not chosen for fairness. This measures
what that costs, by comparing it with the core element a fairness criterion would pick:

  variance  the Variance Core, argmin_{p in core} sum_i (p_i - c(N)/n)^2. Strictly
            convex, so the minimiser is a POINT -- the same allocation whatever
            coalitions row generation happened to generate. Fioriti et al. (2025) eq.(27).
  range     Kimms MP_I, argmin_{p in core} (max_i p_i - min_i p_i). An LP, so the
            objective is determined but the allocation is not; kept to measure that.

Everything is in the repo's COST convention (negative = profit), the convention the
Owen JSONs and the core master already use. Two invariances make the comparison safe:
dispersion is unchanged by the sign flip to profits (deviations from the mean flip with
the mean), and unchanged by adding a constant to every member -- so chi^LR and the
executed chi^LR - eps^LR*1 have the same spread and the choice between them is moot.

Dispersion is measured on the SURPLUS shares y_i = p_i - kappa_i, not on the raw bills:
the game is zero-normalized (Definition `def:game`), so the equal split that egalitarianism
compares against is an equal split of v(N), not of c(N). That is a per-member shift, NOT
a constant, so neither invariance above covers it -- both V(chi^LR) and the minimiser
chi^VC move, and the numbers here supersede any produced before the normalization.

    python weak_eps_experiment/run_fairness.py --runs baseline_6p
    python weak_eps_experiment/run_fairness.py --runs baseline_15p --budget 3600
    python weak_eps_experiment/run_fairness.py --runs baseline_6p --modes variance,range

Resumable on (run, day) exactly like run_multiday.py; --force recomputes.

The Owen side is READ from <run>/cg_day<D>.json rather than recomputed, so those files
must be the same vintage as the core solves. They were not, once: the pre-4-hour-product
JSONs carry a c(N) some 550 EUR above the current one, which would have compared two
different games. The check below refuses rather than trusting the filename.
"""
import os, sys, json, time, csv, argparse, math
_PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import run_multiday as RM
from core import CoreComputation

OUT = os.path.dirname(os.path.abspath(__file__))
COLS = ['run', 'day', 'n_players', 'mode', 'converged', 'c_N', 'v_N', 'time_s', 'n_coalitions',
        'range_core', 'var_core', 'range_owen', 'var_owen',
        'range_ratio', 'var_ratio', 'max_abs_diff', 'owen_in_core', 'owen_worst_excess']

# c(N) and the Owen total must agree to this, relative, or the vintages differ.
VINTAGE_TOL = 1e-6


def dispersion(alloc, players, kappa):
    """(range, variance) of the surplus shares y_i = p_i - kappa_i."""
    y = {i: alloc[i] - kappa[i] for i in players}
    a = sum(y.values()) / len(players)
    return (max(y.values()) - min(y.values()),
            sum((y[i] - a) ** 2 for i in players))


def fairness_day(run, day, mode, budget):
    players = run['players']
    params = RM.build_params(run, day)

    cg_path = os.path.join(RM.run_dir(run), f'cg_day{day}.json')
    if not os.path.exists(cg_path):
        raise FileNotFoundError(f"{cg_path} -- run `run_multiday.py --phase owen` first")
    cg = json.load(open(cg_path))
    owen = cg['owen_alloc_cost']

    cc = CoreComputation(players, 'mip', RM.T, params, mipsolver=RM.SEP_SOLVER)
    t0 = time.time()
    alloc, ok = cc.compute_core(max_iterations=int(1e8), tolerance=1e-6,
                                time_limit=budget, egalitarian=mode)
    t = time.time() - t0
    converged = bool(getattr(cc, 'egalitarian_converged', False))
    c_N = cc.coalition_costs[tuple(sorted(players))]
    # Free: CoreComputation solves every singleton in its constructor.
    kappa = {i: cc.coalition_costs[(i,)] for i in players}
    v_N = sum(kappa.values()) - c_N          # profit convention, >= 0 by superadditivity

    # Vintage guard: the Owen allocation sums to v^MIP(N) in cost form, which is the same
    # c(N) the core master is built on. If they disagree the two sides were computed
    # under different parameters and nothing below means anything.
    owen_total = sum(owen.values())
    if abs(owen_total - c_N) > VINTAGE_TOL * max(1.0, abs(c_N)):
        raise RuntimeError(
            f"vintage mismatch on {run['name']} day {day}: the stored Owen allocation "
            f"sums to {owen_total:.4f} but c(N) is {c_N:.4f}. Re-run "
            f"`run_multiday.py --phase owen --runs {run['name']} --force`.")

    row = {'run': run['name'], 'day': day, 'n_players': len(players), 'mode': mode,
           'converged': converged, 'c_N': c_N, 'v_N': v_N, 'time_s': round(t, 2),
           'n_coalitions': len(cc.coalition_costs)}
    if converged:
        r_c, v_c = dispersion(alloc, players, kappa)
        r_o, v_o = dispersion(owen, players, kappa)
        # Is the Owen point itself in the core? Over the coalitions this run generated --
        # a subset, so a nonpositive worst excess here is necessary, not sufficient.
        worst = max(sum(owen[i] for i in S) - c
                    for S, c in cc.coalition_costs.items() if len(S) < len(players))
        row.update({'range_core': r_c, 'var_core': v_c, 'range_owen': r_o, 'var_owen': v_o,
                    'range_ratio': r_o / r_c if r_c else float('nan'),
                    'var_ratio': v_o / v_c if v_c else float('nan'),
                    'max_abs_diff': max(abs(owen[i] - alloc[i]) for i in players),
                    'owen_worst_excess': worst, 'owen_in_core': bool(worst <= 1e-6)})
        json.dump({'run': run['name'], 'day': day, 'mode': mode, 'c_N': c_N,
                   'v_N': v_N, 'kappa_cost': kappa,
                   'core_alloc_cost': alloc, 'owen_alloc_cost': owen, **{k: row[k] for k in
                   ('range_core', 'var_core', 'range_owen', 'var_owen', 'max_abs_diff')}},
                  open(os.path.join(RM.run_dir(run), f'fair_{mode}_day{day}.json'), 'w'),
                  indent=1)
        print(f"  [Fair/{mode}] {run['name']} day {day}: range {r_o:.1f} -> {r_c:.1f} "
              f"({r_o / r_c:.2f}x), var {v_o:.3e} -> {v_c:.3e} ({v_o / v_c:.2f}x), "
              f"owen_in_core={row['owen_in_core']} ({t:.0f}s)")
    else:
        print(f"  [Fair/{mode}] {run['name']} day {day}: NO CERTIFICATE ({t:.0f}s)")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='baseline_6p')
    ap.add_argument('--days', default='')
    ap.add_argument('--modes', default='variance')
    ap.add_argument('--budget', type=int, default=3600)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    days = [int(d) for d in a.days.split(',')] if a.days else RM.DAYS
    modes = a.modes.split(',')
    for name in a.runs.split(','):
        run = [r for r in RM.RUNS if r['name'] == name][0]
        for mode in modes:
            # One CSV per mode. They shared one file at first, and `RM.append_row` upserts
            # on (run, day) -- so the second mode silently overwrote the first, leaving 31
            # rows that all claimed to be `range`. The per-day JSONs were unaffected, being
            # written per mode, which is the reason nothing had to be recomputed.
            path = os.path.join(RM.run_dir(run), f'fairness_{mode}.csv')
            done = set()
            if os.path.exists(path) and not a.force:
                with open(path) as fh:
                    done = {int(r['day']) for r in csv.DictReader(fh)}
            todo = [d for d in days if d not in done]
            print(f"\n=== [Fairness/{mode}] {name}: {len(todo)} days "
                  f"(budget {a.budget}s) ===")
            for d in todo:
                try:
                    RM.append_row(path, fairness_day(run, d, mode, a.budget), COLS)
                except Exception as e:
                    print(f"  !! {name} day {d} ({mode}) FAILED: {e}")
    print("\nDONE.")


if __name__ == '__main__':
    main()

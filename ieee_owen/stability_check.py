"""
Measured verification of Proposition `prop:opap` (i) and Proposition `prop:eps`.

WHY THIS EXISTS. The harness reports `|gap|/N` and stops. That is a *bound* on the
worst coalition excess, not a measurement of it, so the central claim of
`prop:opap` (i) -- that no coalition can improve on the allocation -- has never been
checked against the allocations we actually produce. This module checks it.

TWO ALLOCATIONS, TWO CLAIMS. They are different objects and must not be conflated:

  chi^LR = sigma*            raw Owen, sums to v^CHP.
                             `prop:opap` (i): stable outright, so the measured
                             worst-case excess must be <= 0.

  chi^LR - omega^LR / n      gap-corrected, sums to v^MIP (efficient).
                             `prop:eps`: in the weak eps-core, so the measured
                             excess must be <= eps^LR = |gap| / n.

Reporting only the second, or only the bound, loses the distinction between "stable
but not efficient" and "efficient but only eps-stable" — which is the whole trade the
paper is about.

HOW — separation, not enumeration. The measurement is
`max_S (sum_{i in S} chi_i - c(S))` at a FIXED chi, which is one separation MIP: an
optimization *over* all coalitions, giving the same answer as enumerating them. Row
generation is a different thing (it searches for the optimal allocation); brute force is
a third (it enumerates `2^n - 1` coalition MIPs and does not scale past ~12 players).

Two further reasons the separation path is the right default, beyond cost:

  - it returns the raw worst-case excess, which is negative when the allocation is
    comfortably inside the core, so the margin is visible
  - `_measure_violation_brute_force` solves `min v s.t. ... , v >= 0`, so it clamps at
    zero and reports 0.0 for everything in the core — it cannot show the margin at all

`brute_force=True` is kept as a one-off cross-check that the separation implementation
agrees with enumeration at `n = 6`. It is not an experiment method.

Sign convention is the codebase's cost-minimisation one throughout: payoffs are costs,
`eps(x) = max_S (sum_{i in S} x_i - c(S)) / |S|`, and `eps(x) <= 0` means in the core.
"""
import io
import contextlib
import time


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


def check_allocations(players, time_periods, params, sigma, owen, gap,
                      mipsolver=None, brute_force=False, model_type='mip',
                      verbose=True, time_limit=3600):
    """Measure the worst-case coalition excess of both Owen points.

    sigma : raw Owen allocation (dict, cost convention), sums to v^CHP
    owen  : gap-corrected allocation, sums to v^MIP
    gap   : v^CHP - v^MIP  (negative in cost convention)
    time_limit : budget in seconds for EACH of the two measurements. The separation MIP
        underneath has no natural bound -- it ran a quarter of an hour at a 100% gap on a
        15-prosumer instance with 7 electrolysers -- so without this the whole Owen phase
        can hang on one day. On expiry `certified` is False for that allocation and its
        `holds` verdict is only trustworthy when it says VIOLATED: the coalition found is
        genuinely violating, but the search may not have reached the worst one, so a
        nonpositive excess proves nothing.

    Returns a dict with, for each allocation, the worst coalition and its per-capita
    excess, and the verdict against the corresponding proposition.
    """
    from core import CoreComputation

    n = len(players)
    eps_lr = abs(gap) / n

    cc = CoreComputation(players, model_type, time_periods, params, mipsolver=mipsolver)

    # check_imputation indexes coalition_costs[(j,)] directly, so the singletons have
    # to exist before either measurement runs.
    for j in players:
        _quiet(cc.compute_coalition_cost, [j])

    out = {'n': n, 'brute_force': brute_force, 'eps_lr': eps_lr,
           'gap': gap, 'omega': abs(gap)}

    for label, alloc, claim, limit in (
            ('sigma', sigma, 'prop:opap(i)  excess <= 0', 0.0),
            ('owen', owen, 'prop:eps      excess <= eps^LR', eps_lr)):
        t0 = time.time()
        (coalition, excess, is_imp), _log = _quiet(
            cc.measure_stability_violation, alloc, brute_force=brute_force,
            time_limit=time_limit)
        certified = not getattr(cc, 'last_stability_truncated', False)
        dt = time.time() - t0
        # tolerance: separation MIPs are solved to a relative gap, so scale with the
        # coalition value rather than using a flat epsilon
        tol = max(1e-6, 1e-5 * abs(sum(alloc.values())))
        out[label] = {
            'worst_coalition': list(coalition), 'excess': float(excess),
            'is_imputation': bool(is_imp), 'limit': limit,
            'holds': bool(excess <= limit + tol), 'slack': float(limit - excess),
            'certified': bool(certified),
            'time_s': dt, 'claim': claim,
        }
        if verbose:
            v = out[label]
            print(f"  [{label:5s}] excess = {excess:+.8f}   limit = {limit:.8f}   "
                  f"{'OK' if v['holds'] else 'VIOLATED'}   "
                  f"worst S = {coalition or '(none)'}   ({dt:.1f}s)")

    out['all_hold'] = all(out[k]['holds'] for k in ('sigma', 'owen'))
    return out


def flatten_for_csv(res, prefix=''):
    if not res:
        return {}
    return {
        f'{prefix}stab_method': 'brute' if res['brute_force'] else 'dinkelbach',
        f'{prefix}stab_excess_sigma': res['sigma']['excess'],
        f'{prefix}stab_holds_sigma': res['sigma']['holds'],
        f'{prefix}stab_excess_owen': res['owen']['excess'],
        f'{prefix}stab_holds_owen': res['owen']['holds'],
        f'{prefix}stab_eps_lr': res['eps_lr'],
        f'{prefix}stab_worst_S_size': len(res['owen']['worst_coalition']),
        # False = the measurement hit its budget, so `holds` is only meaningful when it
        # is False. Declared in OWEN_COLS; a key the header does not carry makes
        # DictWriter reject the whole row.
        f'{prefix}stab_certified': res['sigma'].get('certified', True)
                                   and res['owen'].get('certified', True),
        f'{prefix}stab_time_s': round(res['sigma']['time_s'] + res['owen']['time_s'], 2),
    }

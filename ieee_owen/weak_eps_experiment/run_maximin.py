"""
Fairness selection over Theta*  (`maximin criterion.md`; `validation_plan.md` sec.5).

Runs, on one instance: column generation to convergence, the sec.5.1 singleton screen,
and — only if the screen leaves room — the maximin cutting-plane loop. The screen comes
first on purpose: the whole subsection is vacuous if the optimal dual face is a point,
and the screen answers that in the safe direction for 2n LPs.

What is reported is the pair (t*, iteration count), not just the allocation: the paper's
claim about this construction is computational (selection over a mixed-integer game's
optimal-multiplier face using only pricing MIPs), so the count of those MIPs IS the
result. Stability of the selected point is re-measured rather than asserted, even though
(P3) gives it for free — the same discipline `stability_check.py` was written for.

Usage:
  python ieee_owen/weak_eps_experiment/run_maximin.py                  # n = 6
  python ieee_owen/weak_eps_experiment/run_maximin.py --sizes 6,15
  python ieee_owen/weak_eps_experiment/run_maximin.py --screen-only
  python ieee_owen/weak_eps_experiment/run_maximin.py --no-stability   # skip the check
"""
import os, sys, json, time, argparse

_PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from compact_utility import LocalEnergyMarket
from chp import ColumnGenerationSolver
from maximin import MaximinSelector, owen_gap_corrected
from stability_check import check_allocations
from run_experiment import (build_instance, _jsonable, RESERVE_PRICE, PEAK_PENALTY,
                            RESERVE_BLOCK_HOURS, RESERVE_PRODUCT)

OUT = os.path.dirname(os.path.abspath(__file__))

# Below this the face is a point for practical purposes and Phase 2 is skipped. Scaled
# by |z*| because sigma_j is a cost in EUR and the instances differ by orders of size.
SCREEN_REL_TOL = 1e-7


# Which prosumers may gain or lose an electrolyzer, per size, in the order they are
# taken. Two constraints on the order, both from the instance design rather than taste:
#
#  - REMOVAL must not strip an electrolyzer from a prosumer who also owns hydrogen
#    storage. CONFIGURATION_15's own comment records why: a pure hydro-storage player
#    still has to appear in `players_with_electrolyzers` (with cap 0) to get its export
#    variables. Dropping u8/u10/u18 outright would change the model in a second way and
#    confound the axis.
#  - ADDITION goes to prosumers with hydrogen demand first, who have an economic reason
#    to own one. They receive the community-default unit (`c_els`, `c_su_G`, `els_cap`
#    from setup_lem_parameters); the per-player overrides in large_community apply to
#    the original owners only, so an added unit is the generic one.
_ELS_ORDER = {
    # (removable first-out order, additive order)
    15: (['u2'], ['u5', 'u13', 'u4', 'u15']),
    30: (['u2', 'u19', 'u21'], ['u5', 'u13', 'u26', 'u27', 'u28', 'u29']),
}


def set_electrolyzer_count(k, players, config, T, params, scenario):
    """Rebuild the instance with exactly `k` electrolyzer owners, nothing else changed.

    The non-convexity in this model is the electrolyzer's commitment, so this is the
    axis that decides whether the optimal dual face has any width at all -- and it is
    the same knob `validation_plan.md` sec.1.3 needs for Figure 2, where the question is
    whether the `eps^LR` decay is a membership effect or a dilution artefact.
    """
    import copy
    from data_generator import setup_lem_parameters
    import large_community as LC

    n = len(players)
    if n not in _ELS_ORDER:
        raise ValueError(f"no electrolyzer-composition order defined for n={n}")
    removable, additive = _ELS_ORDER[n]
    base = list(config['players_with_electrolyzers'])
    lo, hi = len(base) - len(removable), len(base) + len(additive)
    if not lo <= k <= hi:
        raise ValueError(f"n={n}: electrolyzer count must be in [{lo}, {hi}], got {k}")

    owners = list(base)
    for u in removable[:len(base) - k] if k < len(base) else []:
        owners.remove(u)
    owners += additive[:k - len(base)] if k > len(base) else []

    cfg = copy.deepcopy(config)
    cfg['players_with_electrolyzers'] = owners

    sens = {key: v[0] for key, v in
            (LC.BASELINE_CANDIDATES_15 if n == 15 else LC.BASELINE_CANDIDATES_30).items()}
    sens.update(reserve_price=RESERVE_PRICE, peak_penalty=PEAK_PENALTY,
                reserve_block_hours=RESERVE_BLOCK_HOURS, reserve_product=RESERVE_PRODUCT)
    p = setup_lem_parameters(players, cfg, T, sens)
    p = (LC.apply_15player_overrides if n == 15 else LC.apply_30player_overrides)(p, T)
    print(f"  [composition] electrolyzer owners: {len(owners)} -> {owners}")
    return players, cfg, T, p, f"{scenario}_els{k}"


ASSET_KEYS = ('players_with_renewables', 'players_with_electrolyzers',
              'players_with_heatpumps', 'players_with_elec_storage',
              'players_with_hydro_storage', 'players_with_heat_storage')


def asset_owners(players, config):
    """Members owning at least one asset — the natural protected set `P`.

    §5.1c: on the absolute-level scale of (R1) a pure consumer sits at the `arg min`
    essentially always, because it only ever pays and its `chi_j` is capped from above by
    the value of its own pattern set, which no `theta` can lift. Restricting `P` is the
    knob the note already provides for this, and it costs nothing: by (P3) weak duality
    holds at every dual-feasible `theta`, so `chi_j >= v^MIP({j})` for EVERY member —
    protecting a subset can never push `N \\ P` below stand-alone.

    Preferred over rewriting (R1) on surplus, which would only be consistent if the
    characteristic function were defined on surplus in the first place.
    """
    owners = set()
    for k in ASSET_KEYS:
        owners.update(config.get(k, []) or [])
    return [u for u in players if u in owners]


def set_day(day, n, players, config, T, reserve_price=None, peak_penalty=None):
    """Rebuild the parameters for a given day of the profile set.

    `build_instance` does not take a day, and its two branches do not agree on one:
    the 6-prosumer case passes no sensitivity dict at all, so `setup_lem_parameters`
    falls through to its defaults and **day 9**, while 15p and 30p read
    `BASELINE_CANDIDATES_*`, which carry `day = 1`. Anything comparing sizes has to say
    which day it is on.

    The parameter set is assembled exactly as `run_multiday.build_params` does, and the
    module constants there are imported rather than copied, so a day-`d` instance here is
    the same object the multi-day harness would build.
    """
    from data_generator import setup_lem_parameters
    import large_community as LC
    import run_multiday as RM

    if n == 6:
        sens, ovfn = dict(RM.BASE6), None
    elif n in (15, 30):
        cand = LC.BASELINE_CANDIDATES_15 if n == 15 else LC.BASELINE_CANDIDATES_30
        sens = RM.scalarize(cand)
        ovfn = LC.apply_15player_overrides if n == 15 else LC.apply_30player_overrides
    else:
        raise ValueError(f"no day-indexed baseline for n={n}")
    sens.update(reserve_price=RESERVE_PRICE if reserve_price is None else reserve_price,
                peak_penalty=PEAK_PENALTY if peak_penalty is None else peak_penalty,
                reserve_block_hours=RESERVE_BLOCK_HOURS, reserve_product=RESERVE_PRODUCT,
                day=day)
    p = setup_lem_parameters(players, config, T, sens)
    return ovfn(p, T) if ovfn else p


def run(n, tol=1e-6, max_iterations=50, screen_only=False, stability=True,
        protected=None, screen=False, tag=None, n_electrolyzers=None, day=None,
        reserve_price=None, peak_penalty=None, r2_slack=None):
    players, config, T, params, scenario = build_instance(n)
    if n_electrolyzers is not None:
        players, config, T, params, scenario = set_electrolyzer_count(
            n_electrolyzers, players, config, T, params, scenario)
    if day is not None or reserve_price is not None or peak_penalty is not None:
        # the channel prices only reach setup_lem_parameters through the day path, so a
        # price override forces it; day defaults to the instance's own (9 at 6p).
        d = 9 if (day is None and n == 6) else (1 if day is None else day)
        params = set_day(d, n, players, config, T,
                         reserve_price=reserve_price, peak_penalty=peak_penalty)
        scenario = f"{scenario}_day{d}"
    if protected == ['assets']:
        protected = asset_owners(players, config)
        excluded = [u for u in players if u not in protected]
        print(f"  [protected] P = {len(protected)}/{len(players)} asset owners: {protected}")
        print(f"  [protected] excluded pure consumers: {excluded}")

    print(f"\n{'='*78}\n[maximin] n={n}  scenario={scenario}\n{'='*78}")
    t0 = time.time()
    lem = LocalEnergyMarket(players, T, params, model_type='mip')
    lem.model.hideOutput()
    ret = lem.solve_complete_model(analyze_revenue=False)
    results_ip = ret[1]
    v_mip = float(lem.model.getObjVal())
    t_mip = time.time() - t0
    print(f"  v_mip = {v_mip:.6f}  ({t_mip:.1f}s)")

    init_priv = {k: v for k, v in results_ip.items() if isinstance(v, dict)}
    t0 = time.time()
    cg = ColumnGenerationSolver(players, T, params, model_type='mip',
                                init_sol=init_priv, smoothing=True)
    status_cg, solution, v_chp, _ = cg.solve()
    t_cg = time.time() - t0
    if status_cg != 'optimal':
        # Not a crash but a result, and the one sec.4 warns about: (R2) is written
        # against z*, so a truncated Phase 1 would give `<= hat z < z*` — a TIGHTER
        # constraint, which breaks the inclusion hat Theta* ⊇ Theta* in the wrong
        # direction and can leave the restricted face disjoint from the real one.
        # There is no safe way to run (MM) on it.
        print(f"  CG ended {status_cg}, not optimal — Phase 1 did not converge, so (MM) "
              "cannot be run on this instance (sec.4). Recording and moving on.")
        doc = {'scenario': scenario, 'n_players': n, 'T': len(T), 'players': players,
               'v_mip': v_mip, 'cg_status': status_cg, 'time_mip_s': t_mip,
               'time_cg_s': t_cg, 'maximin': None}
        path = os.path.join(OUT, f"maximin_{n}p.json")
        with open(path, 'w') as f:
            json.dump(_jsonable(doc), f, indent=2)
        print(f"  -> saved {path}")
        return doc
    print(f"  v_chp = {v_chp:.6f}  ({t_cg:.1f}s)")

    # classical Owen, for the comparison the criterion is judged against
    owen_res = cg.compute_owen_allocation(v_mip)

    sel = MaximinSelector(cg, protected=protected, tol=tol, r2_slack=r2_slack)
    ver = sel.verify_at_optimum()
    if not ver['ok']:
        raise RuntimeError(f"pool extraction disagrees with the RMP duals: {ver}")

    doc = {
        'scenario': scenario, 'n_players': n, 'T': len(T), 'players': players,
        'v_mip': v_mip, 'v_chp': v_chp, 'gap': owen_res['gap'],
        'eps_bound_gap_over_N': owen_res['eps'],
        'time_mip_s': t_mip, 'time_cg_s': t_cg,
        'owen_sigma_cost': owen_res['sigma'],
        'pool_after_phase1': sel.pool_sizes(),
        'verify_pool': ver,
    }

    # The sec.1.5 screen is OFF by default, and it is not the gate. It answers "how far
    # can each chi_j move on its own", by swapping the objective of the very same LP
    # 2n times -- but over hat Theta* ⊇ Theta*, so only a width of exactly zero proves
    # anything. It never returns zero, because it looks in precisely the direction the
    # pool was never built to constrain: 0.396 against a true 0.0026 at 6p, 0.279
    # against 0.0048 at 15p. It cost 2n LPs and skipped no work at either size.
    #
    # The gate is the loop's own bracket instead, which is rigorous in the direction
    # that matters and available from iteration ONE: t_hat lower bounds t* at every
    # iteration (the restricted face contains the true one), classical Owen's worst-off
    # cost upper bounds it (pi* in Theta*), so their difference caps the total remaining
    # gain. At 15p a single iteration already caps it at 0.09% of omega^LR.
    if screen:
        doc['screen'] = sel.screen()
    if screen_only:
        if not screen:
            doc['screen'] = sel.screen()
        print("\n  --screen-only: diagnostic screen run, Phase 2 skipped")
    else:
        mm = sel.solve(max_iterations=max_iterations)
        chi = mm['chi']
        corrected, gap_hat, eps_hat = owen_gap_corrected(chi, v_mip, players)
        mm['gap_hat'] = gap_hat
        mm['eps_hat'] = eps_hat
        mm['chi_gap_corrected'] = corrected
        print(f"  subsidy: omega^LR = {abs(owen_res['gap']):.6f}  ->  "
              f"omega(hat pi) = {abs(gap_hat):.6f}   "
              f"(truncation cost {mm['truncation_cost']:.3e})")
        # the sec.5.1 decision, stated in the units the note asks for: what the
        # criterion moves, against the subsidy it is being compared with and against
        # the worst-off member's own position. `max_remaining_gain` bounds this even
        # when the loop stopped on the iteration limit.
        omega = abs(owen_res['gap'])
        # The worst-off cost share can be exactly 0 — an inactive member is pinned there
        # by its own single column, and in the RAW payoff convention that member is
        # always the worst off. `maximin criterion.md` sec.6.3 flags this: (R1) equalises
        # the LEVEL of chi_j, not the gain over stand-alone, so one idle member makes the
        # criterion vacuous no matter how much freedom the face has elsewhere.
        base = abs(mm['worst_off_cost_classical'])
        rel_self = f"{100 * mm['worst_off_improvement'] / base:.4f}%" if base > 1e-12 \
            else "n/a (worst-off member sits at exactly 0 — inactive)"
        print(f"  GATE: worst-off moves {mm['worst_off_improvement']:.6f} "
              f"({rel_self} of that member's payoff, "
              f"{100 * mm['worst_off_improvement'] / omega:.4f}% of omega^LR); "
              f"at most {mm['max_remaining_gain']:.6f} is available in total")
        if stability:
            # (P3) says this holds at any dual-feasible point; measure it anyway.
            mm['stability_check'] = check_allocations(players, T, params, chi, corrected,
                                                      gap_hat)
        doc['maximin'] = mm

    doc['n_electrolyzers'] = len(config['players_with_electrolyzers'])
    path = os.path.join(OUT, f"maximin_{tag or f'{n}p'}.json")
    with open(path, 'w') as f:
        json.dump(_jsonable(doc), f, indent=2)
    print(f"  -> saved {path}")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='6', help='comma-separated community sizes')
    ap.add_argument('--tol', type=float, default=1e-6, help='Phase-2 pricing tolerance eps')
    ap.add_argument('--max-iterations', type=int, default=50)
    ap.add_argument('--screen', action='store_true',
                    help='also run the sec.1.5 diagnostic screen (not the gate; see run())')
    ap.add_argument('--screen-only', action='store_true',
                    help='run the diagnostic screen and stop')
    ap.add_argument('--electrolyzers', default=None,
                    help='comma-separated electrolyzer-owner counts to sweep at fixed n')
    ap.add_argument('--days', default=None,
                    help='comma list or A-B range of days (default: the instance default '
                         '— day 9 at 6p, day 1 at 15p/30p)')
    ap.add_argument('--r2-slack', type=float, default=None,
                    help='relative (R2) tolerance; default 0 (hard equality). Only for '
                         'reproducing the diagnosis that a nonzero value IS the measured gain')
    ap.add_argument('--peak-penalty', type=float, default=None,
                    help='override delta_peak [EUR/MW]; 0 switches the peak channel off')
    ap.add_argument('--reserve-price', type=float, default=None,
                    help='override pi_res [EUR/MW.h]; 0 switches the reserve channel off')
    ap.add_argument('--no-stability', action='store_true',
                    help='skip the separation re-check of the selected allocation')
    ap.add_argument('--protected', default=None,
                    help="comma-separated protected set P, or 'assets' for every member "
                         "owning at least one asset (default: all members)")
    args = ap.parse_args()

    protected = args.protected.split(',') if args.protected else None
    counts = [int(s) for s in args.electrolyzers.split(',')] if args.electrolyzers else [None]
    if args.days and '-' in args.days:
        a, b = args.days.split('-')
        days = list(range(int(a), int(b) + 1))
    elif args.days:
        days = [int(s) for s in args.days.split(',')]
    else:
        days = [None]
    for n in [int(s) for s in args.sizes.split(',')]:
        for k in counts:
            for d in days:
                suffix = ('' if k is None else f"_els{k}") + \
                         ('' if not args.protected else f"_P-{args.protected.replace(',', '+')}") + \
                         ('' if args.peak_penalty is None else f"_pk{args.peak_penalty:g}") + \
                         ('' if args.reserve_price is None else f"_rs{args.reserve_price:g}") + \
                         ('' if d is None else f"_day{d}")
                try:
                    run(n, tol=args.tol, max_iterations=args.max_iterations,
                        screen_only=args.screen_only, stability=not args.no_stability,
                        protected=protected, screen=args.screen, n_electrolyzers=k,
                        day=d, reserve_price=args.reserve_price,
                        peak_penalty=args.peak_penalty, r2_slack=args.r2_slack,
                        tag=f"{n}p{suffix}")
                except Exception as e:
                    # one bad day must not take the sweep down with it
                    print(f"  !! n={n} els={k} day={d} FAILED: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()

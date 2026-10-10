"""
Coalition generation on the KL-DRO game of one run of the main queue: the benchmark
the Owen allocation is compared with (Table II of the uncertainty manuscript).

Row generation on the cost-of-stability LP over coalitions
(stochastic_core.StochasticCoreComputation, kl_radius=r): each pass solves one
separation MILP over the coalition's membership and its dispatch in every scenario,
and each new coalition costs a KL extensive form of its own. Takes the command line
of stochastic_extension.py, so it plays the same game on the same scenarios.

    python ieee_owen/weak_eps_experiment/coalition_bench.py --budget 3600 \
        --result path/coalgen.json  --n 6 --scenarios 20 --day 1 --kl-radius 0.096 ...
"""
import os, sys, json, time, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import stochastic_extension as SE
import stochastic_core as SC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=3600.0, help='seconds, all passes')
    ap.add_argument('--result', required=True)
    own, rest = ap.parse_known_args()
    args = SE.build_parser().parse_args(rest)
    players, T, base, name, scen = SE.instance(args)
    # the separation is exact only once its tangents are refined at the incumbent
    SE.EF_KL_SINGLE = False
    t0 = time.time()
    cc = SC.StochasticCoreComputation(players, T, scen, kl_radius=args.kl_radius)
    q, ok = cc.compute_core(cost_of_stability=True, time_limit=own.budget,
                            max_iterations=10 ** 6)
    wall = time.time() - t0
    n = len(players)
    out = {'instance': name, 'n': n, 'scenarios': len(scen), 'day': args.day,
           'kl_radius': args.kl_radius, 'budget': own.budget, 'time': wall,
           'converged': bool(cc.cos_converged) and wall <= own.budget,
           'omega_star': float(cc.cost_of_stability_value), 'weak_eps': float(cc.weak_eps),
           'coalitions': len(cc.sep_log), 'core_nonempty': bool(ok),
           'coalition_efs': len(cc.ef_info)}
    os.makedirs(os.path.dirname(os.path.abspath(own.result)), exist_ok=True)
    with open(own.result, 'w') as f:
        json.dump(SE._jsonable(out), f, indent=1)
    print(f'\ncoalition generation: converged {out["converged"]}  {wall:.1f}s  '
          f'{out["coalitions"]} coalitions  weak eps {out["weak_eps"]:.6f}')
    print(f'wrote {own.result}')


if __name__ == '__main__':
    main()

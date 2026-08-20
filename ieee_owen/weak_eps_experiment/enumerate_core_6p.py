"""Exact per-capita excess at n=6, unclamped, over all 2^n-1 coalitions.

The separation route short-circuits: with no violating coalition it returns the empty
set and a raw excess of 0, so "0" is a floor, not the margin. Enumeration gives the
actual max_{S nonempty} (sum_S chi - c(S)) / |S| for both allocations.
"""
import os, sys, json, itertools
_H='/home/user/myfolder/energy_community'
sys.path.insert(0, os.path.join(_H,'ieee_owen','weak_eps_experiment'))
sys.path.insert(0, os.path.join(_H,'ieee_owen')); sys.path.insert(0,_H); os.chdir(_H)

from run_experiment import build_instance
from core import CoreComputation

players, config, T, params, _ = build_instance(6)
cc = CoreComputation(players, 'mip', T, params)
cc.find_all_coalitions(verbose=False)
costs = {tuple(sorted(k)): v for k, v in cc.coalition_costs.items()}
print(f"### coalitions computed: {len(costs)}")

d = json.load(open('ieee_owen/weak_eps_experiment/cg_6p.json'))
sigma, owen, eps = d['owen_sigma_cost'], d['owen_alloc_cost'], d['eps_bound_gap_over_N']

def scan(chi, label):
    best_pc = (-1e18, None); best_raw = (-1e18, None)
    for r in range(1, len(players)):                      # proper coalitions only
        for S in itertools.combinations(players, r):
            c = costs[tuple(sorted(S))]
            raw = sum(chi[j] for j in S) - c
            if raw/len(S) > best_pc[0]: best_pc = (raw/len(S), S)
            if raw > best_raw[0]: best_raw = (raw, S)
    gN = sum(chi[j] for j in players) - costs[tuple(sorted(players))]
    print(f"{label:>26}: per-capita max = {best_pc[0]:+.6f} on |S|={len(best_pc[1])} {best_pc[1]}")
    print(f"{'':>26}  raw        max = {best_raw[0]:+.6f} on |S|={len(best_raw[1])} {best_raw[1]}")
    print(f"{'':>26}  grand-coalition excess = {gN:+.6f}  (= -omega for chi^LR)")

scan(sigma, "chi^LR = sigma*")
scan(owen,  "chi^LR - eps*1")
print(f"### eps^LR = {eps:.6f}")

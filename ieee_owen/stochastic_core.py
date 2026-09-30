"""
Stochastic (scenario-expanded) separation and row generation for the core of the
two-stage game of the supplement (Definition of the stochastic game).

THE GAME. A coalition deviates BEFORE the realization and is valued in expectation
(risk neutral): in the cost convention

    c^Omega(S) = min  sum_w rho_w cost_w(x_S^w)   s.t. first-stage decisions shared by w,

i.e. the extensive form (DP_S^Omega) of stochastic_extension.solve_extensive_form on the
members of S alone. kappa_j = c^Omega({j}), the scenario-expanded stand-alone value, and
the zero-normalized game is v(S) = c^Omega(S) - sum_{j in S} kappa_j exactly as in the
deterministic core.CoreComputation (cost convention throughout: payoffs chi_j are costs,
negative = profit).

SEPARATION. The most violated coalition at a fixed allocation chi,

    max_S  sum_{j in S} chi_j - c^Omega(S),

is ONE monolithic MILP: a ScenarioStack whose blocks are core.SeparationProblem's, one
per scenario. The selection binaries z_j are FIRST STAGE -- the coalition is chosen
before the realization, the same S in every scenario -- so they are created once, keep
their unscaled objective -chi_j, and every scenario block's big-M deactivation and
demand/SOC scaling reads the same z_j. Every other variable is per scenario (cost scaled
by rho_w), except the dispatch's own first-stage variables (electrolyzer states, r_sym),
which stay shared exactly as in the extensive form. At z = 1_S the model is therefore
(DP_S^Omega) plus zero-fixed copies of the non-members, and its objective is
c^Omega(S) - chi(S); minimizing over z as well gives -(max violation). The program grows
by the factor |Omega| (plus the n selection binaries, once) and stays a single MILP.
Solved on a gurobipy copy (stochastic_extension._to_gurobi), never on SCIP.

ROW GENERATION. StochasticCoreComputation is core.CoreComputation with two methods
replaced: compute_coalition_cost (the extensive form of S) and _solve_separation (the
stacked separation above). The OPAP / cost-of-stability master LP, the convergence test,
the verification of each separation against a fresh c^Omega(S), the Dinkelbach weak-eps
measurement (measure_stability_violation) and the all-coalition reference
(compute_core_brute_force) are inherited unchanged.

Validation: ieee_owen/weak_eps_experiment/stochastic_rowgen_check.py.
"""
import os, sys, io, time, contextlib
_PAPER = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from typing import Dict, List, Optional

from core import CoreComputation, SeparationProblem
import stochastic_extension as SE

# Relative MIP gaps, matching the deterministic path: SeparationProblem._solve_with_gurobi
# hard-codes 1e-4 and compact_utility.MIP_GAP (coalition values) is 1e-4.
SEP_GAP = 1e-4
EF_GAP = 1e-4


@contextlib.contextmanager
def _maybe_quiet(quiet):
    if not quiet:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def selection_name(u):
    """Name SeparationProblem gives player u's selection binary."""
    return f'z_{u}'


class StochasticSeparation:
    """max_S sum_{j in S} chi_j - c^Omega(S) as one scenario-stacked MILP.

    payoffs: chi, cost convention, one entry per player. The objective of the model is
    sum_w rho_w cost_w - sum_j chi_j z_j (minimised), so violation = -objective.
    """

    def __init__(self, players: List[str], T: List[int], scenarios, payoffs: Dict[str, float],
                 model_type: str = 'mip', quiet: bool = True):
        self.players, self.T, self.scenarios = list(players), list(T), scenarios
        self.payoffs = dict(payoffs)
        missing = [u for u in self.players if u not in self.payoffs]
        if missing:
            raise ValueError(f'payoffs missing for {missing}')
        t0 = time.time()

        def block(params, model):
            return SeparationProblem(self.players, self.T, model_type, params,
                                     self.payoffs, mipsolver='gurobi', model=model)

        with _maybe_quiet(quiet):
            self.stack = SE.ScenarioStack(
                'Sep_Omega', self.players, self.T, scenarios, dwr=False,
                model_type=model_type, block=block,
                first_stage_names={selection_name(u) for u in self.players})
        # non-anticipativity of the selection: one z_j, shared by every block
        for u in self.players:
            zs = {id(b.z[u]) for b in self.stack.blocks}
            if len(zs) != 1 or self.stack.scen_of.get(selection_name(u), 0) is not None:
                raise RuntimeError(f'selection binary of {u} is not first stage')
        self.time_build = time.time() - t0
        self.truncated = False
        self.stats = {}

    def solve(self, time_limit: Optional[float] = None, gap: float = SEP_GAP,
              fix: Optional[List[str]] = None, min_size: int = 0,
              max_size: Optional[int] = None, quiet: bool = True):
        """Solve on Gurobi. Returns (coalition, violation).

        fix: if given, z is fixed to the indicator of this coalition (a diagnostic: the
        objective is then c^Omega(fix) - chi(fix), which is how the deactivation of the
        non-members is checked against the extensive form of `fix` alone).
        min_size / max_size: bound |S| (diagnostics; the default is every S, the empty
        set and N included). min_size=1, max_size=n-1 gives the max over the proper
        coalitions, the domain an enumeration (stochastic_extension.measure_eps) covers,
        so the value is comparable even when it is negative.

        On a time limit the incumbent's coalition is returned with self.truncated set:
        its violation is real but not necessarily the largest; -ObjBound (stats
        'violation_bound') is then an upper bound on the largest.
        """
        import gurobipy as gp
        m = self.stack.model
        if m.getObjectiveSense() != 'minimize':
            raise ValueError('separation is expected to minimise')
        t0 = time.time()
        g, gv = SE._to_gurobi(m, 'Sep_Omega', time_limit, gap)
        names = list(gv)
        by_name = {v.name: v for v in m.getVars()}
        g.setAttr('Obj', [gv[n] for n in names], [by_name[n].getObj() for n in names])
        g.ObjCon = m.getObjoffset()
        if time_limit is not None:
            g.Params.TimeLimit = max(1.0, float(time_limit))
        if not quiet:
            g.Params.OutputFlag = 1
        if fix is not None:
            fs = set(fix)
            for u in self.players:
                z = gv[selection_name(u)]
                z.LB = z.UB = 1.0 if u in fs else 0.0
        zsum = gp.quicksum(gv[selection_name(u)] for u in self.players)
        if min_size:
            g.addConstr(zsum >= min_size, name='min_size')
        if max_size is not None:
            g.addConstr(zsum <= max_size, name='max_size')
        g.update()
        t_copy = time.time() - t0
        g.optimize()

        self.stats = {
            'scenarios': len(self.scenarios), 'n': len(self.players),
            'vars': g.NumVars, 'bin_vars': g.NumBinVars, 'int_vars': g.NumIntVars,
            'rows': g.NumConstrs, 'nonzeros': g.NumNZs,
            'scip_vars': m.getNVars(), 'scip_rows': m.getNConss(),
            'first_stage_vars': len(self.stack.first_stage),
            'time_build': self.time_build, 'time_copy': t_copy, 'time_solve': g.Runtime,
            'status': int(g.Status), 'gap': float(g.MIPGap) if g.SolCount else float('nan'),
            'nodes': float(g.NodeCount),
        }
        self.truncated = g.Status != gp.GRB.OPTIMAL
        if g.SolCount == 0:
            if g.Status in (gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED):
                self.stats.update(violation=0.0, violation_bound=float('inf'), coalition=[])
                return [], 0.0
            raise RuntimeError(f'stochastic separation failed with Gurobi status {g.Status}')
        coalition = [u for u in self.players if gv[selection_name(u)].X > 0.5]
        violation = -g.ObjVal
        self.stats.update(violation=violation, violation_bound=-g.ObjBound,
                          coalition=coalition)
        self.vals = {n: gv[n].X for n in names}
        return coalition, violation


class StochasticCoreComputation(CoreComputation):
    """core.CoreComputation on the scenario-expanded game (cost convention).

    Coalition values are extensive forms (expectation, first stage shared), kappa_j is
    the scenario-expanded stand-alone value, and separation is StochasticSeparation.
    Everything else -- OPAP / cost-of-stability master LP (SCIP), convergence test,
    separation verification, weak-eps measurement by Dinkelbach, brute-force reference
    -- is the deterministic class's own code.
    """

    def __init__(self, players: List[str], T: List[int], scenarios,
                 model_type: str = 'mip', ef_gap: float = EF_GAP, sep_gap: float = SEP_GAP,
                 ef_time_limit: Optional[float] = None, quiet: bool = True):
        self.scenarios = scenarios
        self.ef_gap, self.sep_gap = ef_gap, sep_gap
        self.ef_time_limit = ef_time_limit
        self.quiet = quiet
        self.ef_info = {}          # coalition tuple -> status / gap / time of its EF
        self.sep_log = []          # one stats dict per separation solve
        # CoreComputation reads self.params only to build models; the stochastic class
        # builds from the scenarios instead. Scenario 0 is kept for anything that
        # inspects it (and is THE instance when |Omega| = 1).
        super().__init__(players, model_type, T, scenarios[0][1], mipsolver='gurobi')

    def compute_coalition_cost(self, coalition: List[str]) -> float:
        key = tuple(sorted(coalition))
        if key in self.coalition_costs:
            return self.coalition_costs[key]
        t0 = time.time()
        with _maybe_quiet(self.quiet):
            ef = SE.solve_extensive_form(list(key), self.time_periods, self.scenarios,
                                         time_limit=self.ef_time_limit, gap=self.ef_gap,
                                         solver='gurobi')
        cost = float(ef['obj'])
        self.ef_info[key] = {'status': ef['status'], 'gap': float(ef['gap']),
                             'bound': float(ef['dual_bound']), 'time': time.time() - t0}
        if ef['status'] != 'optimal':
            print(f"  WARNING: extensive form of {list(key)} ended {ef['status']} "
                  f"at gap {ef['gap']:.2e}; its value is an incumbent")
        self.coalition_costs[key] = cost
        print(f"  c^Omega({list(key)}) = {cost:.4f}  ({time.time() - t0:.1f}s)")
        return cost

    def _solve_separation(self, payoffs: Dict[str, float], time_limit: Optional[float] = None):
        sep = StochasticSeparation(self.players, self.time_periods, self.scenarios, payoffs,
                                   model_type=self.model_type, quiet=self.quiet)
        coalition, violation = sep.solve(time_limit=time_limit, gap=self.sep_gap)
        self.sep_log.append(dict(sep.stats))
        print(f"  stochastic separation: S = {coalition}  violation {violation:.4f}  "
              f"({sep.stats['time_solve']:.1f}s, {sep.stats['bin_vars']} binaries"
              f"{', TRUNCATED' if sep.truncated else ''})")
        return coalition, violation, sep.truncated

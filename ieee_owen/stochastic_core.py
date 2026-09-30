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

KL-DRO GAME (robust_core.md sec. 1.1). c^KL(S) = min_x max_{KL(p||q) <= r} E_p[cost_S],
each coalition against its own worst case in a ball of fixed radius (assumption A1):
StochasticCoreComputation(kl_radius=r) values coalitions with the KL-robust extensive
form and separates with KLSeparation -- the same stack solved by
stochastic_extension._solve_extensive_kl (dual of the inner max, tangent planes refined
at the incumbent). Because z_j is first stage, -chi(S) enters every scenario's cost and
psi shifts by it, so the objective is min_{z,x} psi(cost_S(x)) - chi(S). The incumbent's
exact psi gives a valid cut; the final bound gives a certified upper bound on the max
violation, and only a closed solve may report "no violated coalition".

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

    def _restrict(self, g, gv, fix, min_size, max_size):
        """The diagnostics of solve() on the gurobipy copy: z fixed, or |S| bounded."""
        import gurobipy as gp
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
        self._restrict(g, gv, fix, min_size, max_size)
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


class KLSeparation(StochasticSeparation):
    """max_S sum_{j in S} chi_j - c^KL(S) for the KL-DRO game, as one MILP.

    c^KL(S) = min_x psi_r(cost_S(x)), psi_r(h) = max_{KL(p||q) <= r} p.h, each coalition
    against its own worst-case distribution in the fixed ball (robust_core.md sec. 1.1,
    assumption A1). The same stack as StochasticSeparation is handed to
    stochastic_extension._solve_extensive_kl, which writes psi through its dual,
    min_{lam >= 0, mu} mu + r lam + sum_w q_w lam (exp((h_w - mu)/lam) - 1), with
    h_w = first-stage cost + scenario-w cost. z_j is first stage with cost -chi_j, so it
    enters every h_w, and psi(h - chi(S) 1) = psi(h) - chi(S): the objective is
    min_{z, x} psi(cost_S(x)) - chi(S), and at fixed z its inner part is c^KL(S).

    Values and certificate. The incumbent's exact psi (kl_worst) gives violation =
    -psi, a LOWER bound on the true violation of the coalition it selects (a feasible
    dispatch for S), so a positive one is always a valid cut. Gurobi's final ObjBound
    under the tangent planes is a lower bound on the objective, hence violation_bound =
    -bound is a certified UPPER bound on the max violation. `truncated` is False only
    when the rounds closed, psi - bound <= gap max(1, |psi|); only then may an empty or
    non-positive result be read as "no violated coalition".
    """

    def __init__(self, players, T, scenarios, payoffs, radius: float,
                 model_type: str = 'mip', quiet: bool = True):
        super().__init__(players, T, scenarios, payoffs, model_type=model_type, quiet=quiet)
        self.radius = float(radius)

    def solve(self, time_limit: Optional[float] = None, gap: float = SEP_GAP,
              fix: Optional[List[str]] = None, min_size: int = 0,
              max_size: Optional[int] = None, quiet: bool = True, max_rounds: int = 30):
        """Returns (coalition, violation); time_limit bounds all tangent rounds together."""
        size = {}

        def prepare(g, gv):
            self._restrict(g, gv, fix, min_size, max_size)
            g.update()
            size.update(vars=g.NumVars, bin_vars=g.NumBinVars, int_vars=g.NumIntVars,
                        rows=g.NumConstrs, nonzeros=g.NumNZs)
            if not quiet:
                g.Params.OutputFlag = 1

        t0 = time.time()
        self.stats = {'scenarios': len(self.scenarios), 'n': len(self.players),
                      'radius': self.radius, 'first_stage_vars': len(self.stack.first_stage),
                      'time_build': self.time_build}
        try:
            res = SE._solve_extensive_kl(self.stack, self.time_build, None, gap, True,
                                         self.radius, max_rounds=max_rounds,
                                         prepare=prepare, budget=time_limit)
        except RuntimeError as e:          # a round ended without any incumbent
            self.truncated = True
            self.stats.update(size, violation=0.0, violation_bound=float('inf'),
                              coalition=[], status=str(e), rounds=None,
                              time_solve=time.time() - t0, closed=False)
            return [], 0.0
        psi, bound = float(res['obj']), float(res['dual_bound'])
        closed = psi - bound <= gap * max(1.0, abs(psi)) + 1e-9
        self.truncated = res['status'] != 'optimal' or not closed
        vals = res['vals']
        coalition = [u for u in self.players if vals[selection_name(u)] > 0.5]
        rounds = res['kl']['refinements']
        self.vals = vals
        self.stats.update(size, violation=-psi, violation_bound=-bound, coalition=coalition,
                          status=res['status'], closed=closed, rounds=len(rounds),
                          nodes=float(sum(r['nodes'] for r in rounds)),
                          time_gurobi=float(sum(r['time'] for r in rounds)),
                          time_solve=time.time() - t0, rho=res['kl']['rho'],
                          gap=(psi - bound) / max(1.0, abs(psi)))
        return coalition, -psi


class StochasticCoreComputation(CoreComputation):
    """core.CoreComputation on the scenario-expanded game (cost convention).

    Coalition values are extensive forms (expectation, first stage shared), kappa_j is
    the scenario-expanded stand-alone value, and separation is StochasticSeparation.
    Everything else -- OPAP / cost-of-stability master LP (SCIP), convergence test,
    separation verification, weak-eps measurement by Dinkelbach, brute-force reference
    -- is the deterministic class's own code.

    kl_radius=r (not None) plays the KL-DRO game instead: coalition values are the
    KL-robust extensive forms solve_extensive_form(S, ..., kl_radius=r) (each coalition
    against its own worst case in the ball), kappa_j likewise, and separation is
    KLSeparation. kl_radius=0 runs the KL code at r = 0, which is the expectation again.
    A KL separation whose tangent rounds did not close is reported as truncated, so
    compute_core never reads convergence off it; an existing row that comes back is
    judged by its recomputed violation (CoreComputation.compute_core) as in the
    risk-neutral case.
    """

    def __init__(self, players: List[str], T: List[int], scenarios,
                 model_type: str = 'mip', ef_gap: float = EF_GAP, sep_gap: float = SEP_GAP,
                 ef_time_limit: Optional[float] = None, quiet: bool = True,
                 kl_radius: Optional[float] = None):
        self.scenarios = scenarios
        self.ef_gap, self.sep_gap = ef_gap, sep_gap
        self.ef_time_limit = ef_time_limit
        self.quiet = quiet
        self.kl_radius = kl_radius
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
        kw = {} if self.kl_radius is None else {'kl_radius': self.kl_radius}
        with _maybe_quiet(self.quiet):
            ef = SE.solve_extensive_form(list(key), self.time_periods, self.scenarios,
                                         time_limit=self.ef_time_limit, gap=self.ef_gap,
                                         solver='gurobi', **kw)
        cost = float(ef['obj'])
        info = {'status': ef['status'], 'gap': float(ef['gap']),
                'bound': float(ef['dual_bound']), 'time': time.time() - t0}
        if self.kl_radius is not None:
            info.update(rounds=ef['kl']['solves'], rho=ef['kl']['rho'])
        self.ef_info[key] = info
        info['closed'] = cost - info['bound'] <= self.ef_gap * max(1.0, abs(cost)) * 1.0001
        if ef['status'] != 'optimal' or not info['closed']:
            print(f"  WARNING: extensive form of {list(key)} ended {ef['status']} "
                  f"at gap {ef['gap']:.2e}; its value is an incumbent")
        self.coalition_costs[key] = cost
        tag = 'Omega' if self.kl_radius is None else f'KL{self.kl_radius:g}'
        print(f"  c^{tag}({list(key)}) = {cost:.4f}  ({time.time() - t0:.1f}s)")
        return cost

    def make_separation(self, payoffs: Dict[str, float]):
        """The separation of this game at `payoffs` (risk neutral or KL)."""
        if self.kl_radius is None:
            return StochasticSeparation(self.players, self.time_periods, self.scenarios,
                                        payoffs, model_type=self.model_type, quiet=self.quiet)
        return KLSeparation(self.players, self.time_periods, self.scenarios, payoffs,
                            self.kl_radius, model_type=self.model_type, quiet=self.quiet)

    def _solve_separation(self, payoffs: Dict[str, float], time_limit: Optional[float] = None):
        sep = self.make_separation(payoffs)
        coalition, violation = sep.solve(time_limit=time_limit, gap=self.sep_gap)
        self.sep_log.append(dict(sep.stats))
        print(f"  stochastic separation: S = {coalition}  violation {violation:.4f}  "
              f"bound {sep.stats['violation_bound']:.4f}  "
              f"({sep.stats['time_solve']:.1f}s, {sep.stats.get('bin_vars')} binaries"
              f"{', TRUNCATED' if sep.truncated else ''})")
        return coalition, violation, sep.truncated


def deterministic_core(players: List[str], T: List[int], params: Dict,
                       model_type: str = 'mip', **kw) -> StochasticCoreComputation:
    """The deterministic game's row generation, separation and coalition values, through
    this module with one scenario: the instance as given, probability 1.

    Every IEEE harness uses this in place of core.CoreComputation, as the column
    generation uses stochastic_extension.solve_deterministic in place of chp.py: one code
    path for the deterministic and the stochastic game. At |Omega| = 1 it reproduced
    core.CoreComputation (omega*, the coalitions added at n = 6, the weak eps of the
    Owen point at n = 6 and 15; rowgen_check/part_a). core.py stays as the base class
    and for applied_energy/.
    """
    if model_type != 'mip':
        raise ValueError(f"model_type {model_type!r}: the stochastic core solves the "
                         f"mixed-integer game only")
    return StochasticCoreComputation(players, T, [(1.0, params)], model_type=model_type, **kw)

"""
Two-stage stochastic extension (supplement Sec. S-stoch, ieee_supple.txt).

Three pieces, each reusing LocalEnergyMarket unchanged, one block per scenario:

  extensive form   (DP_S^Omega)    one MILP, the deterministic equivalent
  column generation (DWR_S^Omega)  Dantzig-Wolfe master over contingent plans,
                                   one scenario-expanded MILP per prosumer as pricing
  Algorithm S1                     scenario-dependent allocation x_j*(omega) from the
                                   master duals and the terminal priced plans

STAGES. The electrolyzer states z_on/z_off/z_sb and the transitions z_su/z_sd, and the
community reserve capacities r_sym, are decided before the realization; everything
else carries a scenario index. The block builder enforces this by variable name: a
first-stage name is created once and handed back to every later block, so
non-anticipativity is by construction, not by extra rows. Rows that touch first-stage
variables only (commitment logic, minimum down time) are added once.

OBJECTIVE. LocalEnergyMarket sets costs through addVar(obj=...). A scenario block
scales them by rho_omega, a first-stage variable keeps its own, so the model objective
is exactly eq:sup_dp_obj in the cost convention (minimise; profit = -objective).

SIGNS. Master rows follow the paper, i - e on every carrier (the compact model writes
hydrogen as e - i; that is an equality and only flips a dual, which the extensive form
never reads). SCIP duals are raw, and the pricing objective is c - pi^T a exactly as in
solver.PlayerSubproblem. Row (k, t, omega) has coefficient 1 on the scenario block, so
its dual is rho_omega times a price; Algorithm S1 divides it back out.

ENGINES. --engine direct runs the loop here, on a HiGHS or Gurobi LP (--lp-solver),
following the column generation in zonal_consistency/containment_bp.py: the master is
built once and columns are added into it, so a penalty round only changes slack
bounds instead of rebuilding. Pricing is Gurobi or HiGHS (--pricing-solver), and so
are the extensive form and stand-alone MILPs (--mip-solver). SCIP builds the models
but solves none of them: --engine scip (the SCIP master with its pricer plugin, as in
chp.py, and with it --sar and --doi) and SCIP as a pricing or MIP solver now raise.
The code for them is kept only so the numbers below stay traceable.
Every combination returns the same numbers -- checked at |Omega| = 1, where all four
give v^CHP = -3039.944297 and the same Owen allocation to four decimals.

Usage (from anywhere):
  python ieee_owen/stochastic_extension.py --n 6 --scenarios 5
  python ieee_owen/stochastic_extension.py --n 6 --scenarios 3 --check-core
  python ieee_owen/stochastic_extension.py --n 6 --scenarios 1 --wind-sigma 0 \
      --load-sigma 0 --price-sigma 0          # reproduces the deterministic model
"""
import os, sys, json, time, argparse, itertools
# layout: <repo root>/ieee_owen/. Shared model modules and data/ live at the root;
# run_experiment (the instance builder) lives in weak_eps_experiment/.
_PAPER = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum
from compact_utility import LocalEnergyMarket, reserve_blocks

OUT = os.path.join(_PAPER, 'weak_eps_experiment', 'stochastic')
# Relative gaps. omega^LR = v^MIP - v^LR is a difference of two values of ~2e4 whose
# decision-independent part (the fixed non-flexible demand, ~-3.1e4 at n=60) cancels,
# so a relative 1e-4 on either side is an absolute ~2.4 against omega ~3.7. Measured at
# n=60, |Omega|=5: EF and CG at 1e-4 put omega anywhere in [1.5, 6.1]; both at 1e-6
# give 3.715 / 3.728 over two runs, and the EF still solves in ~10 s.
#   MIP_GAP  pricing MILPs (CG tightens a prosumer's gap itself when its bound is
#            what holds the Lagrangian bound back)
#   EF_GAP   extensive form and stand-alone MILPs: v^MIP enters omega directly
#   CG_GAP   column generation: stop once UB - LB <= CG_GAP (1 + |UB|)
MIP_GAP = 1e-4
EF_GAP = 1e-6
CG_GAP = 1e-6

# Names LocalEnergyMarket gives the first-stage variables (f"{prefix}{u}_{t}" and
# f"r_sym_{i}"). Heat-pump commitment is deliberately absent: it is redispatched.
FIRST_STAGE_PREFIXES = ('z_on_G_', 'z_off_G_', 'z_sb_G_', 'z_su_G_', 'z_sd_G_',
                        'r_sym_', 'r_up_', 'r_dn_')
CARRIERS = ('E', 'H', 'G')


# =============================================================================
# Scenarios
# =============================================================================
def _ar1(rng, n, rho):
    """Standard-normal AR(1) path: forecast errors are correlated across hours."""
    e = np.empty(n)
    e[0] = rng.standard_normal()
    for t in range(1, n):
        e[t] = rho * e[t - 1] + np.sqrt(1.0 - rho ** 2) * rng.standard_normal()
    return e


def make_scenarios(base, players, T, n_scen, seed=0, wind_sigma=0.25,
                   solar_sigma=0.20, load_sigma=0.10, price_sigma=0.15, rho=0.7,
                   price_carriers=('E',), load_carriers=CARRIERS):
    """Forecast-error scenarios around one deterministic instance, equiprobable.

    Every scenario is a copy of `base` with multiplicative AR(1) errors on
      renewable availability  renewable_cap_{u}_{t}   one path for wind, one for solar
                              (common weather), clipped to [0, the day's peak]
      non-flexible load       d_{k}_nfl_{u}_{t}        one path per carrier in
                                                        `load_carriers`
      market prices           pi_{k}_gri_{import,export}_{t} and u_{k}_{u}_{t},
                              one factor per carrier in `price_carriers`
    Import and export move by the same factor, so pi^imkt >= pi^emkt survives
    scenario by scenario (Assumption env(iii) as the supplement reads it), and the
    willingness to pay u moves with the import price it is defined from.

    The import bounds i_{k}_cap are raised to the largest scenario load and then
    shared by every scenario: bounds are part of the fixed support, not of the draw.
    Production costs, the peak tariff and the reserve price stay deterministic.
    """
    rng = np.random.default_rng(seed)
    wind = [u for u in base.get('players_with_wind', []) if u in players]
    solar = [u for u in base.get('players_with_solar', []) if u in players]
    nT = len(T)
    scen = []
    for _ in range(n_scen):
        p = dict(base)
        fw = 1.0 + wind_sigma * _ar1(rng, nT, rho)
        fs = 1.0 + solar_sigma * _ar1(rng, nT, rho)
        for us, f in ((wind, fw), (solar, fs)):
            for u in us:
                peak = max(base[f'renewable_cap_{u}_{t}'] for t in T)
                for i, t in enumerate(T):
                    p[f'renewable_cap_{u}_{t}'] = float(
                        np.clip(base[f'renewable_cap_{u}_{t}'] * f[i], 0.0, peak))
        for k in CARRIERS:
            f = np.maximum(0.0, 1.0 + load_sigma * _ar1(rng, nT, rho))
            if k not in load_carriers:
                continue            # drawn anyway, so the other paths do not move
            for u in players:
                for i, t in enumerate(T):
                    key = f'd_{k}_nfl_{u}_{t}'
                    if key in base:
                        p[key] = float(base[key] * f[i])
        for k in price_carriers:
            f = np.maximum(0.05, 1.0 + price_sigma * _ar1(rng, nT, rho))
            for i, t in enumerate(T):
                for side in ('import', 'export'):
                    key = f'pi_{k}_gri_{side}_{t}'
                    p[key] = float(base[key] * f[i])
                for u in players:
                    key = f'u_{k}_{u}_{t}'
                    if key in base:
                        p[key] = float(base[key] * f[i])
        scen.append(p)
    for k in CARRIERS:
        loads = [s[key] for s in scen for key in s if key.startswith(f'd_{k}_nfl_')]
        cap = max([base.get(f'i_{k}_cap', 0.0)] + loads)
        for s in scen:
            s[f'i_{k}_cap'] = cap
    return [(1.0 / n_scen, s) for s in scen]


# =============================================================================
# Scenario-stacked model
# =============================================================================
class _Block:
    """What LocalEnergyMarket sees as its model while it builds scenario omega.

    Forwards everything to the shared SCIP model, renaming per scenario, scaling
    second-stage costs by rho_omega, and sharing first-stage variables and the rows
    that involve nothing else. `data` stays per block, so each LocalEnergyMarket
    keeps its own variable dictionaries.
    """
    def __init__(self, stack, w, prob):
        self._stack, self._w, self._prob = stack, w, prob
        self._m = stack.model

    def __getattr__(self, name):
        return getattr(self._m, name)

    def addVar(self, name='', vtype='C', lb=0.0, ub=None, obj=0.0, **kw):
        st = self._stack
        if name.startswith(FIRST_STAGE_PREFIXES):
            if name not in st.vars:
                v = self._m.addVar(name=name, vtype=vtype, lb=lb, ub=ub, obj=obj, **kw)
                st._record(v, None, obj)
            return st.vars[name]
        v = self._m.addVar(name=f'{name}_s{self._w}', vtype=vtype, lb=lb, ub=ub,
                           obj=self._prob * obj, **kw)
        st._record(v, self._w, obj)
        return v

    def addCons(self, cons, name='', **kw):
        st = self._stack
        vs = [v for term in cons.expr.terms for v in term.vartuple]
        if vs and all(v.name in st.first_stage for v in vs):
            if name not in st.first_stage_cons:
                st.first_stage_cons[name] = self._m.addCons(cons, name=name, **kw)
            return st.first_stage_cons[name]
        return self._m.addCons(cons, name=f'{name}_s{self._w}', **kw)


class ScenarioStack:
    """LocalEnergyMarket(players) once per scenario, stacked into one SCIP model.

    dwr=False gives (DP_S^Omega) with every linking row imposed per scenario;
    dwr=True drops them and, with a single player, gives the pricing set
    X_j^MIP of eq:sup_Xmip.
    """
    def __init__(self, name, players, T, scenarios, dwr, model_type='mip'):
        self.players, self.T, self.scenarios = list(players), list(T), scenarios
        self.probs = [p for p, _ in scenarios]
        self.model = Model(name)
        self.vars = {}          # name -> var
        self.cost = {}          # name -> unscaled cost
        self.scen_of = {}       # name -> scenario index, None if first stage
        self.first_stage = set()
        self.first_stage_cons = {}
        self.blocks = []
        for w, (prob, params) in enumerate(scenarios):
            self.blocks.append(LocalEnergyMarket(self.players, self.T, params,
                                                 model_type=model_type, dwr=dwr,
                                                 model=_Block(self, w, prob)))

    def _record(self, v, w, obj):
        self.vars[v.name] = v
        self.cost[v.name] = float(obj)
        self.scen_of[v.name] = w
        if w is None:
            self.first_stage.add(v.name)

    def scaled_cost(self, name):
        w = self.scen_of[name]
        return self.cost[name] * (1.0 if w is None else self.probs[w])

    def values(self, sol=None):
        m = self.model
        if sol is None:
            return {n: m.getVal(v) for n, v in self.vars.items()}
        return {n: m.getSolVal(sol, v) for n, v in self.vars.items()}

    def cost_split(self, vals, names=None):
        """(first-stage cost, [unweighted second-stage cost per scenario])."""
        first, per = 0.0, [0.0] * len(self.scenarios)
        for n in (self.vars if names is None else names):
            c = self.cost[n]
            if c == 0.0:
                continue
            w = self.scen_of[n]
            if w is None:
                first += c * vals.get(n, 0.0)
            else:
                per[w] += c * vals.get(n, 0.0)
        return first, per

    # --- the linking rows of eq:sup_bal - eq:sup_peak, player u's side ---------
    def link_terms(self, u):
        """{(kind, t, w): [(var name, coef)]} for player u, kind in E,H,G,up,dn,peak."""
        rows = {}
        for w, lem in enumerate(self.blocks):
            for t in self.T:
                for k in CARRIERS:
                    terms = []
                    if (u, t) in getattr(lem, f'i_{k}_com'):
                        terms.append((getattr(lem, f'i_{k}_com')[u, t].name, 1.0))
                    if (u, t) in getattr(lem, f'e_{k}_com'):
                        terms.append((getattr(lem, f'e_{k}_com')[u, t].name, -1.0))
                    rows[(k, t, w)] = terms
                rows[('up', t, w)] = ([(lem.r_plus[u, t].name, -1.0)]
                                      if (u, t) in lem.r_plus else [])
                rows[('dn', t, w)] = ([(lem.r_minus[u, t].name, -1.0)]
                                      if (u, t) in lem.r_minus else [])
                pk = []
                if (u, t) in lem.i_E_gri:
                    pk.append((lem.i_E_gri[u, t].name, 1.0))
                if (u, t) in lem.e_E_gri:
                    pk.append((lem.e_E_gri[u, t].name, -1.0))
                rows[('peak', t, w)] = pk
        return rows

    def first_stage_values(self, vals):
        return {n: vals[n] for n in sorted(self.first_stage)}


def _set_mip_params(m, time_limit=None, gap=None, quiet=True):
    if quiet:
        m.hideOutput()
    if time_limit:
        m.setParam('limits/time', float(time_limit))
    m.setParam('limits/gap', MIP_GAP if gap is None else float(gap))


# =============================================================================
# Extensive form
# =============================================================================
def solve_extensive_form(players, T, scenarios, time_limit=None, gap=None, quiet=True,
                         solver='highs'):
    """Solve (DP_S^Omega). Returns a dict; objective in the cost convention.

    solver='highs' / 'gurobi' build the same SCIP model and solve a highspy /
    gurobipy copy of it.
    """
    if solver not in ('highs', 'gurobi'):
        raise ValueError(f"MIP solver {solver!r}: SCIP is not a supported solver here; use 'gurobi' or 'highs'")
    gap = EF_GAP if gap is None else gap
    t0 = time.time()
    st = ScenarioStack('DP_Omega', players, T, scenarios, dwr=False)
    build = time.time() - t0
    m = st.model
    if solver in ('highs', 'gurobi'):
        fn = _solve_extensive_highs if solver == 'highs' else _solve_extensive_gurobi
        return fn(st, build, time_limit, gap, quiet)
    if solver != 'scip':
        raise ValueError(f"MIP solver must be 'scip', 'highs' or 'gurobi', got {solver!r}")
    _set_mip_params(m, time_limit, gap, quiet=quiet)
    m.optimize()
    status = m.getStatus()
    if m.getNSols() == 0:
        raise RuntimeError(f'extensive form ({len(players)} players): no solution, status {status}')
    vals = st.values()
    first, per = st.cost_split(vals)
    return {
        'stack': st, 'status': status, 'obj': m.getObjVal(),
        'dual_bound': m.getDualbound(), 'gap': m.getGap(),
        'vals': vals, 'first_cost': first, 'scen_cost': per,
        # gross worth of each scenario, cost convention: w(omega) = first + c^omega
        'worth_cost': [first + c for c in per],
        'time_build': build, 'time_solve': m.getSolvingTime(),
    }


# =============================================================================
# Column generation
# =============================================================================
class Column:
    __slots__ = ('player', 'cost', 'coef', 'first', 'scen', 'fs')

    def __init__(self, player, stack, rows, names, vals):
        self.player = player
        self.cost = sum(stack.scaled_cost(n) * vals.get(n, 0.0) for n in names)
        self.coef = {}
        for key, terms in rows.items():
            a = sum(c * vals.get(n, 0.0) for n, c in terms)
            if abs(a) > 1e-12:
                self.coef[key] = a
        self.first, self.scen = stack.cost_split(vals, names)
        # the plan's commitment, kept for reporting
        self.fs = {n: round(vals.get(n, 0.0)) for n in names if n in stack.first_stage}

    @classmethod
    def combine(cls, cols, weights):
        """The convex combination sum_i w_i col_i of one prosumer's columns: a point
        of conv(X_u), so it is a column in its own right (bundle compression)."""
        new = cls.__new__(cls)
        new.player = cols[0].player
        new.cost = sum(w * c.cost for c, w in zip(cols, weights))
        new.coef = {}
        for c, w in zip(cols, weights):
            for k, a in c.coef.items():
                new.coef[k] = new.coef.get(k, 0.0) + w * a
        new.coef = {k: a for k, a in new.coef.items() if abs(a) > 1e-12}
        new.first = sum(w * c.first for c, w in zip(cols, weights))
        new.scen = [sum(w * c.scen[i] for c, w in zip(cols, weights))
                    for i in range(len(cols[0].scen))]
        new.fs = None
        return new


def _to_gurobi(scip_model, name, time_limit=None, gap=None, env=None):
    """Copy a pyscipopt model that has only linear constraints into gurobipy.

    Built from the constraint data rather than through an MPS file: LocalEnergyMarket
    reuses some constraint names, which a file round trip would have to rename.
    Variables keep their names, which is how the two models are matched.
    """
    import gurobipy as gp
    from gurobipy import GRB
    inf = scip_model.infinity()
    g = gp.Model(name, env=env) if env is not None else gp.Model(name)
    g.Params.OutputFlag = 0
    g.Params.MIPGap = MIP_GAP if gap is None else gap
    if time_limit:
        g.Params.TimeLimit = time_limit
    vt = {'CONTINUOUS': GRB.CONTINUOUS, 'BINARY': GRB.BINARY, 'INTEGER': GRB.INTEGER,
          'IMPLINT': GRB.CONTINUOUS}
    gv = {}
    for v in scip_model.getVars():
        if v.name in gv:
            raise ValueError(f'duplicate variable name {v.name}: cannot map to Gurobi')
        lb, ub = v.getLbOriginal(), v.getUbOriginal()
        gv[v.name] = g.addVar(lb=-GRB.INFINITY if lb <= -inf else lb,
                              ub=GRB.INFINITY if ub >= inf else ub,
                              vtype=vt[v.vtype()], name=v.name)
    for c in scip_model.getConss():
        if c.getConshdlrName() != 'linear':
            raise ValueError(f'constraint {c.name} is {c.getConshdlrName()}, not linear')
        expr = gp.LinExpr([(a, gv[n]) for n, a in scip_model.getValsLinear(c).items()])
        lhs, rhs = scip_model.getLhs(c), scip_model.getRhs(c)
        if lhs > -inf and rhs < inf and lhs == rhs:
            g.addLConstr(expr, GRB.EQUAL, rhs)
        else:
            if lhs > -inf:
                g.addLConstr(expr, GRB.GREATER_EQUAL, lhs)
            if rhs < inf:
                g.addLConstr(expr, GRB.LESS_EQUAL, rhs)
    g.ModelSense = GRB.MINIMIZE
    g.update()
    return g, gv


def _to_highs(scip_model, time_limit=None, gap=None):
    """Copy a pyscipopt model that has only linear constraints into highspy.

    Same contract as _to_gurobi: the columns come out in the order of
    scip_model.getVars(), which is how values are matched back by name.
    """
    import highspy
    INF, inf = highspy.kHighsInf, scip_model.infinity()
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('mip_rel_gap', MIP_GAP if gap is None else gap)
    if time_limit:
        h.setOptionValue('time_limit', float(time_limit))
    names = [v.name for v in scip_model.getVars()]
    if len(set(names)) != len(names):
        raise ValueError('duplicate variable names: cannot map to HiGHS')
    idx = {n: j for j, n in enumerate(names)}
    lo = np.array([-INF if v.getLbOriginal() <= -inf else v.getLbOriginal()
                   for v in scip_model.getVars()])
    up = np.array([INF if v.getUbOriginal() >= inf else v.getUbOriginal()
                   for v in scip_model.getVars()])
    h.addVars(len(names), lo, up)
    ints = [j for j, v in enumerate(scip_model.getVars())
            if v.vtype() in ('BINARY', 'INTEGER')]
    if ints:
        h.changeColsIntegrality(len(ints), np.array(ints, dtype=np.int32),
                                np.array([highspy.HighsVarType.kInteger] * len(ints)))
    for c in scip_model.getConss():
        if c.getConshdlrName() != 'linear':
            raise ValueError(f'constraint {c.name} is {c.getConshdlrName()}, not linear')
        row = scip_model.getValsLinear(c)
        lhs, rhs = scip_model.getLhs(c), scip_model.getRhs(c)
        h.addRow(-INF if lhs <= -inf else lhs, INF if rhs >= inf else rhs, len(row),
                 np.array([idx[n] for n in row], dtype=np.int32),
                 np.array(list(row.values())))
    return h, names


def _solve_extensive_highs(st, build, time_limit, gap, quiet):
    """solve_extensive_form on a highspy copy of the stacked SCIP model."""
    import highspy
    m = st.model
    if m.getObjectiveSense() != 'minimize':
        raise ValueError('extensive form is expected to minimise')
    h, names = _to_highs(m, time_limit, gap)
    if not quiet:
        h.setOptionValue('output_flag', True)
    by_name = {v.name: v for v in m.getVars()}
    cost = np.array([by_name[n].getObj() for n in names])
    h.changeColsCost(len(names), np.arange(len(names), dtype=np.int32), cost)
    t0 = time.time()
    h.run()
    solve = time.time() - t0
    status = h.modelStatusToString(h.getModelStatus()).lower()
    info = h.getInfo()
    if info.primal_solution_status != 2:            # kSolutionStatusFeasible
        raise RuntimeError(f'extensive form ({len(st.players)} players, highs): '
                           f'no solution, status {status}')
    x = np.array(h.getSolution().col_value)
    vals = dict(zip(names, x))
    vals = {n: vals[n] for n in st.vars}
    obj = float(cost @ x) + m.getObjoffset()
    bound = info.mip_dual_bound + m.getObjoffset() if m.getNVars() and \
        any(v.vtype() in ('BINARY', 'INTEGER') for v in m.getVars()) else obj
    first, per = st.cost_split(vals)
    return {
        'stack': st, 'status': status, 'obj': obj,
        'dual_bound': min(bound, obj), 'gap': max(info.mip_gap, 0.0) if bound != obj else 0.0,
        'vals': vals, 'first_cost': first, 'scen_cost': per,
        'worth_cost': [first + c for c in per],
        'time_build': build, 'time_solve': solve,
    }


def _solve_extensive_gurobi(st, build, time_limit, gap, quiet):
    """solve_extensive_form on a gurobipy copy of the stacked SCIP model."""
    m = st.model
    if m.getObjectiveSense() != 'minimize':
        raise ValueError('extensive form is expected to minimise')
    g, gv = _to_gurobi(m, 'DP_Omega', time_limit, gap)
    g.Params.OutputFlag = 0 if quiet else 1
    names = list(gv)
    by_name = {v.name: v for v in m.getVars()}
    gvars = [gv[n] for n in names]
    g.setAttr('Obj', gvars, [by_name[n].getObj() for n in names])
    g.ObjCon = m.getObjoffset()
    g.optimize()
    if g.SolCount == 0:
        raise RuntimeError(f'extensive form ({len(st.players)} players, gurobi): '
                           f'no solution, status {g.Status}')
    vals = dict(zip(names, g.getAttr('X', gvars)))
    vals = {n: vals[n] for n in st.vars}
    obj = g.ObjVal
    is_mip = g.IsMIP
    bound = g.ObjBound if is_mip else obj
    first, per = st.cost_split(vals)
    return {
        'stack': st, 'status': {2: 'optimal', 9: 'timelimit'}.get(g.Status, str(g.Status)),
        'obj': obj, 'dual_bound': min(bound, obj),
        'gap': g.MIPGap if is_mip else 0.0,
        'vals': vals, 'first_cost': first, 'scen_cost': per,
        'worth_cost': [first + c for c in per],
        'time_build': build, 'time_solve': g.Runtime,
    }


class PlayerPricing:
    """eq:sup_vlrj for one prosumer: a two-stage stochastic MILP of its own.

    solver='scip' solves the stacked SCIP model directly; 'gurobi' and 'highs'
    solve a copy of it (same variables, same rows) built through gurobipy or
    highspy, changing only the objective between calls.
    """
    def __init__(self, player, T, scenarios, time_limit=None, gap=None, solver='highs',
                 env=None):
        if solver not in ('highs', 'gurobi'):
            raise ValueError(f"pricing solver {solver!r}: SCIP is not a supported solver here; use 'gurobi' or 'highs'")
        self.player = player
        self.stack = ScenarioStack(f'price_{player}', [player], T, scenarios, dwr=True)
        self.model = self.stack.model
        _set_mip_params(self.model, time_limit, gap)
        self.solver = solver
        self.gap = MIP_GAP if gap is None else gap
        self.time, self.calls = 0.0, 0
        if solver == 'gurobi':
            # env: a Gurobi environment is never used from two threads at once, so
            # DirectMaster hands one per worker and prices each env's prosumers in
            # sequence. Without one, the default environment.
            self.env = env
            self.g, self.gv = _to_gurobi(self.model, f'price_{player}', time_limit, gap,
                                         env=self.env)
            self.g_names = list(self.gv)
            self.g_vars = [self.gv[n] for n in self.g_names]
        elif solver == 'highs':
            self.h, self.h_names = _to_highs(self.model, time_limit, gap)
            self.h_idx = np.arange(len(self.h_names), dtype=np.int32)
        elif solver != 'scip':
            raise ValueError("pricing solver must be 'scip', 'gurobi' or 'highs', "
                             f'got {solver!r}')
        self.rows = self.stack.link_terms(player)
        self.names = list(self.stack.vars)
        self.base = {n: self.stack.scaled_cost(n) for n in self.names}
        self.adj = sorted({n for terms in self.rows.values() for n, _ in terms})
        self.last = None        # (duals, obj, Column) of the latest call

    def _coef(self, duals, farkas):
        coef = dict.fromkeys(self.adj, 0.0) if farkas else dict(self.base)
        for key, terms in self.rows.items():
            pi = duals.get(key, 0.0)
            if pi:
                for n, c in terms:
                    coef[n] -= pi * c
        return coef

    def _objective(self, duals, farkas):
        coef = self._coef(duals, farkas)
        v = self.stack.vars
        m = self.model
        m.freeTransform()
        m.setObjective(quicksum(c * v[n] for n, c in coef.items() if c), 'minimize')

    def set_gap(self, gap):
        """Relative MIP gap of this prosumer's pricing MILP."""
        self.gap = gap
        if self.solver == 'gurobi':
            self.g.Params.MIPGap = gap
        else:
            self.h.setOptionValue('mip_rel_gap', gap)

    def set_threads(self, k):
        """Solver threads for this prosumer's MILP (None: the solver's default)."""
        if self.solver == 'gurobi':
            self.g.Params.Threads = 0 if k is None else int(k)

    def price(self, duals, farkas=False):
        """min_x c(x) - pi^T A_j x over X_j. Returns (objective, dual bound, Column)."""
        t0 = time.time()
        try:
            return self._price(duals, farkas)
        finally:
            self.time += time.time() - t0
            self.calls += 1

    def _price(self, duals, farkas=False):
        if self.solver == 'gurobi':
            obj, bound, vals = self._price_gurobi(duals, farkas)
        elif self.solver == 'highs':
            obj, bound, vals = self._price_highs(duals, farkas)
        else:
            self._objective(duals, farkas)
            m = self.model
            m.optimize()
            if m.getNSols() == 0:
                raise RuntimeError(f'pricing {self.player}: no solution, status {m.getStatus()}')
            vals = self.stack.values()
            obj, bound = m.getObjVal(), m.getDualbound()
        col = Column(self.player, self.stack, self.rows, self.names, vals)
        if not farkas:
            self.last = (dict(duals), obj, col)
        return obj, bound, col

    def _price_gurobi(self, duals, farkas):
        coef = self._coef(duals, farkas)
        g = self.g
        g.setAttr('Obj', self.g_vars, [coef.get(n, 0.0) for n in self.g_names])
        for attempt in range(5):
            try:
                g.optimize()
                break
            except Exception as e:      # WLS token renewal hiccup: wait and retry
                if 'license' not in str(e).lower() or attempt == 4:
                    raise
                time.sleep(2.0 * (attempt + 1))
        if g.SolCount == 0:
            raise RuntimeError(f'pricing {self.player} (gurobi): no solution, status {g.Status}')
        x = g.getAttr('X', self.g_vars)
        vals = dict(zip(self.g_names, x))
        # the objective is recomputed from the solution, so it and the column cost
        # agree to the last digit whatever Gurobi reports internally
        obj = sum(c * vals[n] for n, c in coef.items() if c)
        bound = min(g.ObjBound, obj)
        return obj, bound, vals

    def _price_highs(self, duals, farkas):
        import highspy
        coef = self._coef(duals, farkas)
        h = self.h
        h.changeColsCost(len(self.h_names), self.h_idx,
                         np.array([coef.get(n, 0.0) for n in self.h_names]))
        h.run()
        st = h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal:
            # HiGHS sometimes ends an LP in kUnknown (a numerical clean-up failure):
            # retry from scratch, then once more without presolve
            h.clearSolver()
            h.run()
            st = h.getModelStatus()
            if st != highspy.HighsModelStatus.kOptimal:
                h.clearSolver()
                h.setOptionValue('presolve', 'off')
                h.run()
                st = h.getModelStatus()
                h.setOptionValue('presolve', 'choose')
            self.retries = getattr(self, 'retries', 0) + 1
        if st != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f'pricing {self.player} (highs): status {st}')
        vals = dict(zip(self.h_names, h.getSolution().col_value))
        obj = sum(c * vals[n] for n, c in coef.items() if c)
        info = h.getInfo()
        bound = min(getattr(info, 'mip_dual_bound', obj) or obj, obj)
        return obj, bound, vals

    def column_from(self, ef_vals):
        """Project a grand-coalition solution onto this player's plan."""
        return Column(self.player, self.stack, self.rows, self.names, ef_vals)


class _MasterPricer(Pricer):
    def __init__(self, master):
        super().__init__()
        self.master = master

    def _call(self, farkas):
        try:
            return self.master._price(farkas)
        except Exception as e:       # SCIP swallows it otherwise; re-raised after solve
            self.master.error = e
            self.model.interruptSolve()
            return {'result': SCIP_RESULT.DIDNOTRUN}

    def pricerredcost(self):
        return self._call(False)

    def pricerfarkas(self):
        return self._call(True)


class StochasticMaster:
    """(DWR_N^Omega): convexity rows plus the linking rows of every scenario."""
    def __init__(self, players, T, scenarios, params, time_limit=None,
                 pricing_time_limit=None, pricing_gap=None, smoothing=True,
                 incumbent=None, doi=False, subs=None, families=None,
                 pricing_solver='scip', gap_tol=1e-8, penalty=None, verbose=True):
        raise RuntimeError(f"--engine scip: SCIP is not a supported solver here; use 'gurobi' or 'highs' (use --engine direct)")
        self.players, self.T, self.scenarios = list(players), list(T), scenarios
        # Wentges smoothing with the adaptive alpha of Pessoa et al. (2010), as in
        # pricer.LEMPricer; `incumbent` is the extensive-form objective when known.
        self.smoothing = smoothing
        self.incumbent = np.inf if incumbent is None else incumbent
        self.center, self.L_bar = None, -np.inf
        # (pi, {u: v_u(pi)}, {u: Column}) at the best Lagrangian bound so far. Once
        # that bound reaches the RMP value, pi with sigma_u = v_u(pi) is an optimal
        # dual of the full master (every plan prices out at pi by definition of v_u,
        # and sum_u sigma_u = z), and the Columns are the plans q_u* attaining it.
        self.best = None
        self.probs = [p for p, _ in scenarios]
        self.params = params
        self.verbose = verbose
        self.subs = subs or {u: PlayerPricing(u, T, scenarios, pricing_time_limit,
                                              pricing_gap, pricing_solver)
                             for u in self.players}
        # One relative tolerance for adding a column, for stopping on LB >= RMP and
        # for switching smoothing off (see pricer.LEMPricer._pricing_tol on why
        # these must be the same number). z is then certified to n * gap_tol * |z|.
        self.gap_tol = gap_tol
        # Dual-optimal inequalities (Ben Amor, Desrosiers & Valerio de Carvalho 2006)
        # on the balance rows, entered as the primal columns they dualize: the
        # community buying from / selling to the grid itself.
        #   y_imp  bal -1, peak +1, cost  rho P^imp   =>  p <= P^imp + peak price
        #   y_exp  bal +1, peak -1, cost -rho P^exp   =>  p >= P^exp + peak price
        # with p = -pi/rho. No member here can route grid energy to the community
        # without limit (import caps, no member with both i_gri and e_com), so the
        # inequalities are not valid a priori. They are checked a posteriori: y = 0
        # at convergence makes the RMP solution feasible for the original master, so
        # LB <= z <= RMP = LB and the duals are optimal for it. Otherwise the caller
        # drops them and resumes from the columns collected (see solve_dwr).
        self.doi = doi
        self.y = {}
        # Three-piece penalty stabilization (du Merle, Villeneuve, Desrosiers &
        # Hansen 1999; Ben Amor, Desrosiers & Frangioni 2009), combined with
        # smoothing as Pessoa et al. (2018) recommend. penalty = {center, eps,
        # delta}: each linking row gets a bounded slack on either side, priced so
        # that the dual moves freely inside [center - eps_r, center + eps_r] and
        # pays delta per unit beyond it. The slacks perturb the rows, so the RMP
        # value is only an upper bound on z once they are all zero -- which is what
        # solve_dwr_stab checks before it stops.
        self.penalty = penalty
        self.pen = {}
        self.enable_reserve = bool(params.get('enable_reserve', False))
        self.enable_peak = bool(params.get('enable_peak', False))
        self.row_keys = [(k, t, w) for w in range(len(scenarios)) for t in self.T
                         for k in CARRIERS]
        if self.enable_reserve:
            self.row_keys += [(d, t, w) for w in range(len(scenarios)) for t in self.T
                              for d in ('up', 'dn')]
        if self.enable_peak:
            self.row_keys += [('peak', t, w) for w in range(len(scenarios)) for t in self.T]
        # Rows actually present in the master. Each is a nonnegative combination of
        # original linking rows, {name: [(row key, weight)]}; the default is every
        # original row on its own. Aggregated families give the relaxed extended
        # master of dyn-SAR (Costa, Contardo, Desaulniers & Yarkony 2022), whose
        # duals gamma map back to pi_r = sum_S w_{S,r} gamma_S (their eq. 13).
        if families is None:
            families = {key: [(key, 1.0)] for key in self.row_keys}
        self.families = families
        self.fam_of = {}
        for name, members in families.items():
            for key, wgt in members:
                self.fam_of.setdefault(key, []).append((name, wgt))
        missing = set(self.row_keys) - set(self.fam_of)
        if missing:
            raise ValueError(f'{len(missing)} linking rows are in no family')
        self.m = Model('DWR_Omega')
        self.time_limit = time_limit
        self.columns = {u: [] for u in self.players}
        self.lam = {u: [] for u in self.players}
        self.rows, self.conv = {}, {}
        self.x0 = {}
        self.x0_rows = {}       # x0 key -> [(row key, coefficient)]
        self.iteration = 0
        self.lb = -np.inf
        self.log = []
        self.error = None

    # --- construction ------------------------------------------------------
    def _add_initial(self, cols):
        for col in cols:
            u = col.player
            v = self.m.addVar(name=f'lam_{u}_{len(self.lam[u])}', lb=0.0, obj=col.cost)
            self.columns[u].append(col)
            self.lam[u].append(v)

    def _build_rows(self):
        m, p = self.m, self.params
        blocks = reserve_blocks(self.T, p.get('reserve_block_hours', 24))
        block_of_t = {t: i for i, blk in enumerate(blocks) for t in blk}
        sym = p.get('reserve_product', 'symmetric') == 'symmetric'
        if self.enable_reserve:
            pi = p.get('pi_res', 0.0)
            for i, blk in enumerate(blocks):
                if sym:
                    self.x0[('r_sym', i)] = m.addVar(name=f'r_sym_{i}', lb=0.0,
                                                     obj=-len(blk) * pi)
                else:
                    self.x0[('r_up', i)] = m.addVar(name=f'r_up_{i}', lb=0.0,
                                                    obj=-len(blk) * p.get('pi_up', pi))
                    self.x0[('r_dn', i)] = m.addVar(name=f'r_dn_{i}', lb=0.0,
                                                    obj=-len(blk) * p.get('pi_dn', pi))
        if self.enable_peak:
            for w, rho in enumerate(self.probs):
                self.x0[('p', w)] = m.addVar(name=f'p_s{w}', lb=0.0,
                                             obj=rho * p.get('pi_E_peak', 0.0))

        def x0_terms(kind, t, w):
            if kind in ('up', 'dn'):
                i = block_of_t[t]
                return [(1.0, self.x0[('r_sym', i)] if sym else self.x0[(f'r_{kind}', i)])]
            if kind == 'peak':
                return [(-1.0, self.x0[('p', w)])]
            return []

        terms = {}              # original row key -> [(coef, var)]
        for key in self.row_keys:
            kind, t, w = key
            terms[key] = x0_terms(*key)
            if kind in ('up', 'dn'):
                x0k = ('r_sym', block_of_t[t]) if sym else (f'r_{kind}', block_of_t[t])
                self.x0_rows.setdefault(x0k, []).append((key, 1.0))
            elif kind == 'peak':
                self.x0_rows.setdefault(('p', w), []).append((key, -1.0))
        self.terms_x0 = {k: list(v) for k, v in terms.items()}
        for u in self.players:
            for col, v in zip(self.columns[u], self.lam[u]):
                for key, a in col.coef.items():
                    terms[key].append((a, v))
        if self.doi:
            for w, (rho, prm) in enumerate(self.scenarios):
                for t in self.T:
                    for k in CARRIERS:
                        imp = m.addVar(name=f'y_imp_{k}_{t}_s{w}', lb=0.0,
                                       obj=rho * prm[f'pi_{k}_gri_import_{t}'])
                        exp = m.addVar(name=f'y_exp_{k}_{t}_s{w}', lb=0.0,
                                       obj=-rho * prm[f'pi_{k}_gri_export_{t}'])
                        self.y[(k, t, w, 'imp')], self.y[(k, t, w, 'exp')] = imp, exp
                        pair = [(-1.0, imp), (1.0, exp)]
                        terms[(k, t, w)] += pair
                        self.terms_x0[(k, t, w)] += pair
                        if k == 'E' and self.enable_peak:
                            pair = [(1.0, imp), (-1.0, exp)]
                            terms[('peak', t, w)] += pair
                            self.terms_x0[('peak', t, w)] += pair

        if self.penalty:
            if any(len(mem) != 1 or mem[0][1] != 1.0
                   for mem in self.families.values()):
                raise ValueError('penalty stabilization expects the original rows')
            pc, pe, pd = (self.penalty['center'], self.penalty['eps'],
                          self.penalty['delta'])
            for key in self.row_keys:
                c = pc.get(key, 0.0)
                e = pe * (1.0 + abs(c))
                kind, t, w = key
                up = m.addVar(name=f'pen_up_{kind}_{t}_s{w}', lb=0.0, ub=pd, obj=c + e)
                dn = m.addVar(name=f'pen_dn_{kind}_{t}_s{w}', lb=0.0, ub=pd, obj=-c + e)
                self.pen[key] = (up, dn)
                terms[key] += [(1.0, up), (-1.0, dn)]

        # zero-coefficient placeholder, so a row no initial column touches is still
        # a linear expression SCIP accepts
        anchor = 0.0 * self.lam[self.players[0]][0]
        for name, members in self.families.items():
            coef = {}
            for key, wgt in members:
                for a, v in terms[key]:
                    coef[v.name] = (coef.get(v.name, (0.0, v))[0] + wgt * a, v)
            expr = anchor + quicksum(a * v for a, v in coef.values())
            kind = members[0][0][0]
            label = name if isinstance(name, str) else '_'.join(map(str, name))
            if kind in CARRIERS:
                self.rows[name] = m.addCons(expr == 0.0, name=label, modifiable=True)
            else:
                self.rows[name] = m.addCons(expr <= 0.0, name=label, modifiable=True)
        for u in self.players:
            self.conv[u] = m.addCons(quicksum(self.lam[u]) == 1.0, name=f'conv_{u}',
                                     modifiable=True)

    def _add_priced(self, col):
        m, u = self.m, col.player
        v = m.addVar(name=f'lam_{u}_{len(self.lam[u])}', lb=0.0, obj=col.cost,
                     pricedVar=True)
        self.columns[u].append(col)
        self.lam[u].append(v)
        m.addConsCoeff(m.getTransformedCons(self.conv[u]), v, 1.0)
        acc = {}
        for key, a in col.coef.items():
            for name, wgt in self.fam_of[key]:
                acc[name] = acc.get(name, 0.0) + wgt * a
        for name, a in acc.items():
            if a:
                m.addConsCoeff(m.getTransformedCons(self.rows[name]), v, a)

    # --- duals ---------------------------------------------------------------
    def _duals(self, farkas):
        m = self.m
        get = m.getDualfarkasLinear if farkas else m.getDualsolLinear
        gamma = {k: get(m.getTransformedCons(c)) for k, c in self.rows.items()}
        duals = {key: sum(wgt * gamma[name] for name, wgt in self.fam_of[key])
                 for key in self.row_keys}
        conv = {u: get(m.getTransformedCons(c)) for u, c in self.conv.items()}
        return duals, conv

    def violations(self, tol=1e-6):
        """Original linking rows the current master solution violates: {key: residual}."""
        m = self.m
        res = {key: sum(a * m.getVal(v) for a, v in self.terms_x0[key])
               for key in self.row_keys}
        for u in self.players:
            for col, v in zip(self.columns[u], self.lam[u]):
                lam = m.getVal(v)
                if lam > 1e-12:
                    for key, a in col.coef.items():
                        res[key] += a * lam
        return {key: r for key, r in res.items()
                if (abs(r) > tol if key[0] in CARRIERS else r > tol)}

    def _lagrangian(self, duals, bounds):
        """L(pi) = sum_j v_j(pi) + min_{x0>=0} (c0 - A0^T pi) x0.

        The second term is 0 on Theta (eq:sup_Theta) and -inf off it. At an RMP
        optimum pi is always in Theta; a smoothed pi is too when the center is,
        which is why the center starts at the first RMP duals and not at 0 (0 is
        outside Theta as soon as the reserve price is positive).
        """
        for k, rows in self.x0_rows.items():
            rc = self.x0[k].getObj() - sum(duals.get(r, 0.0) * a for r, a in rows)
            if rc < -1e-9 * (1.0 + abs(self.x0[k].getObj())):
                return -np.inf
        # inequality rows (<= 0) need nonpositive multipliers in a min problem
        if any(v > 1e-9 for (kind, _, _), v in duals.items() if kind not in CARRIERS):
            return -np.inf
        return sum(bounds)

    def _record(self, duals, res):
        L = self._lagrangian(duals, [r[1] for r in res.values()])
        if L > self.lb:
            self.lb = L
            self.best = (dict(duals), {u: r[0] for u, r in res.items()},
                         {u: r[2] for u, r in res.items()})
        if L > self.L_bar:
            self.L_bar, self.center = L, dict(duals)

    def _alpha(self, lp):
        # Smoothing stays on until the gap is within tolerance. pricer.LEMPricer
        # switches to alpha = 1 below an absolute gap of 1e-2, which on a degenerate
        # master is exactly where stabilization is needed most: the S=3 runs sat for
        # thousands of Kelley iterations at a gap of 0.005.
        base = 0.1
        gap = lp - self.L_bar
        if not np.isfinite(gap):
            return base
        if gap <= self.gap_tol * (1.0 + abs(lp)):
            return 1.0
        if lp > self.incumbent and self.incumbent - self.L_bar > 1e-6:
            return min(1.0, base * (self.incumbent - self.L_bar) / gap)
        return base

    def _rc(self, col, duals, conv):
        return col.cost - sum(duals.get(k, 0.0) * a for k, a in col.coef.items()) \
            - conv[col.player]

    def _price_all(self, duals, farkas=False):
        out = {}
        for u in self.players:
            out[u] = self.subs[u].price(duals, farkas)
        return out

    def _price(self, farkas):
        m = self.m
        duals, conv = self._duals(farkas)
        if farkas:
            added = 0
            for u, (obj, _, col) in self._price_all(duals, True).items():
                if obj - conv[u] < -1e-8:
                    self._add_priced(col)
                    added += 1
            if self.verbose:
                print(f'  Farkas: +{added} columns')
            return {'result': SCIP_RESULT.SUCCESS if added else SCIP_RESULT.DIDNOTRUN}

        lp = m.getLPObjVal()
        tol = self.gap_tol * (1.0 + abs(lp))
        self.iteration += 1
        alpha, added, min_rc, mode = 1.0, 0, 0.0, 'std'

        if self.smoothing:
            if self.center is None:
                self.center = dict(duals)
            alpha = self._alpha(lp)
            if alpha < 1.0:
                keys = set(duals) | set(self.center)
                st = {k: alpha * duals.get(k, 0.0) + (1 - alpha) * self.center.get(k, 0.0)
                      for k in keys}
                res = self._price_all(st)
                self._record(st, res)
                mode = 'smooth'
                if self.lb < lp - tol:
                    for u, (_, _, col) in res.items():
                        rc = self._rc(col, duals, conv)
                        min_rc = min(min_rc, rc)
                        if rc < -tol:
                            self._add_priced(col)
                            added += 1
                    if not added:
                        mode = 'misprice'

        # LB >= RMP stops without asking the RMP duals to price out: under dual
        # degeneracy they keep failing to while the value is already certified,
        # and self.best supplies an optimal dual instead.
        if not added and self.lb < lp - tol:
            res = self._price_all(duals)
            self._record(duals, res)
            for u, (obj, _, col) in res.items():
                rc = obj - conv[u]
                min_rc = min(min_rc, rc)
                if rc < -tol:
                    self._add_priced(col)
                    added += 1

        self.log.append({'iter': self.iteration, 'lp': lp, 'lb': self.lb, 'alpha': alpha,
                         'mode': mode, 'min_rc': min_rc, 'added': added})
        if self.verbose:
            print(f'  CG {self.iteration:3d} | RMP {lp:14.4f} | LB {self.lb:14.4f} '
                  f'| a {alpha:.2f} {mode:8s} | min rc {min_rc:11.4e} | +{added}')
        # no column added means converged: either LB reached the RMP value or no
        # plan prices out at the RMP duals
        return {'result': SCIP_RESULT.SUCCESS}

    # --- solve ---------------------------------------------------------------
    def solve(self, init_vals=None, init_cols=None, pricing=True):
        t0 = time.time()
        if init_cols is not None:
            cols = list(init_cols)
        elif init_vals is not None:
            cols = [self.subs[u].column_from(init_vals) for u in self.players]
        else:
            cols = [self.subs[u].price({})[2] for u in self.players]
        self._add_initial(cols)
        self._build_rows()
        m = self.m
        m.setPresolve(SCIP_PARAMSETTING.OFF)
        m.setHeuristics(SCIP_PARAMSETTING.OFF)
        m.setSeparating(SCIP_PARAMSETTING.OFF)
        m.disablePropagation()
        m.hideOutput()
        if self.time_limit:
            m.setParam('limits/time', float(self.time_limit))
        if pricing:
            m.includePricer(_MasterPricer(self), 'SDWPricer', 'two-stage DW pricer')
        m.optimize()
        if self.error is not None:
            raise self.error
        status = m.getStatus()
        if status != 'optimal':
            raise RuntimeError(f'DW master: status {status}')
        if not pricing:
            duals, conv = {}, {}
        elif self.best is not None and self.lb >= m.getObjVal() - \
                (len(self.players) + 1) * self.gap_tol * (1 + abs(m.getObjVal())):
            duals, conv = self.best[0], dict(self.best[1])
        else:
            raise RuntimeError('DW master: no dual attains the RMP value')
        return {
            'status': status, 'obj': m.getObjVal(), 'lb': self.lb,
            'duals': duals, 'sigma': conv,
            'lambda': {u: [m.getVal(v) for v in self.lam[u]] for u in self.players},
            'x0': {f'{k[0]}_{k[1]}': m.getVal(v) for k, v in self.x0.items()},
            'iterations': self.iteration,
            'columns': {u: len(self.columns[u]) for u in self.players},
            'y_total': sum(m.getVal(v) for v in self.y.values()),
            'penalty_slack': max((max(m.getVal(a), m.getVal(b))
                                  for a, b in self.pen.values()), default=0.0),
            'time': time.time() - t0,
        }

    def terminal_columns(self, duals):
        """q_j*: the plan attaining eq:sup_vlrj at the final duals (Algorithm S1, l.2).

        The pricer's last call is reused when it saw these duals; otherwise the
        pricing problem is solved once more at them.
        """
        out = {}
        if self.best is not None and duals is self.best[0]:
            return {u: (self.best[1][u], self.best[2][u]) for u in self.players}
        scale = 1.0 + max((abs(v) for v in duals.values()), default=0.0)
        for u, sub in self.subs.items():
            last = sub.last
            if last is not None and all(abs(last[0].get(k, 0.0) - v) <= 1e-9 * scale
                                        for k, v in duals.items()):
                out[u] = (last[1], last[2])
            else:
                obj, _, col = sub.price(duals)
                out[u] = (obj, col)
        return out


def _sar_family(kind, ts, omegas, probs):
    """One aggregated row: sum over the hours ts and scenarios omegas, rho-weighted.

    The weights are the scenario probabilities, not the plain average of Costa et
    al.: that makes the implied price pi_{k,t,omega} / rho_omega common to the
    scenarios in the set, which is the smoothness we expect at optimality.
    """
    ts, omegas = tuple(sorted(ts)), tuple(sorted(omegas))
    return (('S', kind, ts, omegas),
            [((kind, t, w), probs[w]) for t in ts for w in omegas])


def solve_dwr_sar(players, T, scenarios, params, init_vals=None, tol=1e-6,
                  max_phases=200, sar_block=1, sar_cap=1.0, **kw):
    """(DWR_N^Omega) by dyn-SAR (Costa, Contardo, Desaulniers & Yarkony 2022).

    Phase 1 keeps one row per (kind, hour block), rho-weighted over the block's
    hours and every scenario: a relaxation of the master with far fewer linking
    rows, whose duals are common to the rows inside a set. Each phase is solved by
    column generation to convergence; the original rows the solution violates are
    then grouped and added as finer aggregated rows, and the next phase starts from
    the columns already generated. It stops when no original row is violated, so
    the solution is feasible for the original master while optimal for a relaxation
    of it, and pi = sum_S w gamma_S is an optimal dual (their eq. 13) -- which is
    what Algorithm S1 reads. The Lagrangian bound and the smoothing center live in
    the space of the original rows and carry over from phase to phase.

    Sets are split gradually, (kind, block) -> (kind, block, one sign) ->
    (kind, hour, one sign) -> single row, and at most `sar_cap` of the original
    rows are added per round, largest violation first.

    MEASURED, and the reason this is off by default. On the 6-prosumer instance at
    |Omega| = 3 (Gurobi pricing, otherwise identical settings), plain column
    generation took 831 iterations; dyn-SAR with one row per (kind, hour) over the
    scenarios and every violated row added at once (sar_block=1, sar_cap=1.0, the
    defaults here) took 6256; the paper's own policy (sar_block=6, sar_cap=0.05)
    passed 17 772 without finishing, and without smoothing 35 197 while still in
    phase 3. Each phase re-converges its own column generation, and the last phase
    still pays the degenerate plateau in full -- 4132 of those 6256 iterations, 99%
    of them with the RMP value frozen. The paper's gains come from cheaper master
    reoptimizations, which cannot pay here: the master LP is 4% of the run and the
    pricing MILPs are 87%, so multiplying the pricing calls is the wrong trade.
    """
    probs = [p for p, _ in scenarios]
    S = list(range(len(scenarios)))
    proto = StochasticMaster(players, T, scenarios, params, **kw)
    blocks = reserve_blocks(T, sar_block)
    block_of_t = {t: i for i, blk in enumerate(blocks) for t in blk}
    kinds = sorted({k for k, _, _ in proto.row_keys})
    families = dict(_sar_family(k, blk, S, probs) for k in kinds for blk in blocks)
    cap = max(1, int(sar_cap * len(proto.row_keys)))
    subs = proto.subs
    master, cols, phases = None, None, []
    t0 = time.time()
    state = {'lb': -np.inf, 'L_bar': -np.inf, 'center': None, 'best': None,
             'iteration': 0, 'log': []}
    for phase in range(1, max_phases + 1):
        master = StochasticMaster(players, T, scenarios, params, subs=subs,
                                  families=dict(families), **kw)
        master.lb, master.L_bar, master.center = state['lb'], state['L_bar'], state['center']
        master.best = state['best']
        master.iteration, master.log = state['iteration'], state['log']
        res = master.solve(init_vals=init_vals if cols is None else None,
                           init_cols=cols)
        viol = master.violations(tol)
        phases.append({'phase': phase, 'rows': len(families), 'obj': res['obj'],
                       'lb': master.lb, 'iterations': master.iteration,
                       'violated': len(viol), 'time': time.time() - t0})
        if master.verbose:
            print(f'  -- SAR phase {phase}: {len(families)} rows, RMP {res["obj"]:.4f}, '
                  f'{len(viol)} original rows violated')
        state = {'lb': master.lb, 'L_bar': master.L_bar, 'center': master.center,
                 'best': master.best, 'iteration': master.iteration, 'log': master.log}
        cols = [c for u in master.players for c in master.columns[u]]
        if not viol:
            break
        # candidate sets, coarsest first; a set already present is split further
        cand = {}
        for (k, t, w), r in viol.items():
            sgn = r > 0
            for name, members in (_sar_family(k, blocks[block_of_t[t]], 
                                              [x for x in S], probs),
                                  _sar_family(k, blocks[block_of_t[t]], [w], probs),
                                  _sar_family(k, [t], [w], probs)):
                if name not in families:
                    key = (name, sgn)
                    cur = cand.get(key)
                    cand[key] = (max(cur[0], abs(r)) if cur else abs(r), name, members)
                    break
        for _, name, members in sorted(cand.values(), reverse=True)[:cap]:
            families[name] = members
    else:
        raise RuntimeError(f'dyn-SAR: rows still violated after {max_phases} phases')
    res['time'] = time.time() - t0
    res['iterations'] = master.iteration
    res['sar'] = {'phases': phases, 'final_rows': len(families),
                  'original_rows': len(proto.row_keys), 'block': sar_block,
                  'cap': cap}
    res['doi'] = {'used': False}
    return res, master


class _LP:
    """The restricted master LP, as little as column generation needs of a solver.

    Rows are created empty and columns are added into them one at a time, which is
    what lets the loop keep one model from the first iteration to the last: a
    penalty round only changes bounds and costs. Both backends use the convention
    reduced cost = c - pi^T a, with pi <= 0 on a <= row of a minimization.

    Columns are addressed by handles that survive remove_cols(): handle j sits at
    position pos[j] of the solver's column list, -1 once removed.
    """
    def __init__(self, backend='highs', name='master', method='primal', presolve='auto'):
        # method: simplex variant for the master LP. Columns only ever get added, so
        # the previous basis stays primal feasible and primal simplex warm starts
        # from it; 'auto' leaves the choice to the solver (Gurobi: concurrent).
        self.backend, self.method = backend, method
        self.pos, self.handle_at = [], []
        if backend == 'highs':
            import highspy
            self.INF = highspy.kHighsInf
            self.h = highspy.Highs()
            self.h.setOptionValue('output_flag', False)
            self._st = highspy.HighsModelStatus.kOptimal
            self.nrow = 0
        elif backend == 'gurobi':
            import gurobipy as gp
            self.gp, self.INF = gp, gp.GRB.INFINITY
            self.m = gp.Model(name)
            self.m.Params.OutputFlag = 0
            if presolve == 'off':
                self.m.Params.Presolve = 0
            self.rows, self.cols = [], []
        else:
            raise ValueError(f"lp solver must be 'highs' or 'gurobi', got {backend!r}")
        if backend == 'highs' and presolve == 'off':
            self.h.setOptionValue('presolve', 'off')
        self._apply_method()

    def _apply_method(self):
        # 'barrier': interior point without crossover, i.e. the well-centred duals of
        # primal-dual column generation (Gondzio, Gonzalez-Brevis & Munari 2013)
        if self.backend == 'highs':
            if self.method == 'barrier':
                self.h.setOptionValue('solver', 'ipm')
                self.h.setOptionValue('run_crossover', 'off')
            else:
                self.h.setOptionValue('solver', 'simplex')
                self.h.setOptionValue('simplex_strategy',
                                      {'primal': 4, 'dual': 1, 'auto': 0}[self.method])
        else:
            self.m.Params.Method = {'primal': 0, 'dual': 1, 'auto': -1,
                                    'barrier': 2}[self.method]
            self.m.Params.Crossover = 0 if self.method == 'barrier' else -1

    def add_row(self, lo, hi):
        if self.backend == 'highs':
            self.h.addRow(lo, hi, 0, np.array([], dtype=np.int32), np.array([]))
            self.nrow += 1
            return self.nrow - 1
        expr = self.gp.LinExpr()
        if lo == hi:
            c = self.m.addLConstr(expr, self.gp.GRB.EQUAL, lo)
        elif lo <= -self.INF:
            c = self.m.addLConstr(expr, self.gp.GRB.LESS_EQUAL, hi)
        else:
            c = self.m.addLConstr(expr, self.gp.GRB.GREATER_EQUAL, lo)
        self.rows.append(c)
        return len(self.rows) - 1

    def add_col(self, obj, lb, ub, rows=(), coefs=()):
        if self.backend == 'highs':
            self.h.addCol(obj, lb, ub, len(rows), np.array(rows, dtype=np.int32),
                          np.array(coefs, dtype=float))
        else:
            col = self.gp.Column(list(coefs), [self.rows[r] for r in rows])
            self.cols.append(self.m.addVar(lb=lb, ub=ub, obj=obj, column=col))
        j = len(self.pos)
        self.pos.append(len(self.handle_at))
        self.handle_at.append(j)
        return j

    def add_row_with(self, lo, hi, handles, coefs):
        """A row over existing columns (by handle), for rows added mid-run."""
        if self.backend == 'highs':
            idx = np.array([self.pos[j] for j in handles], dtype=np.int32)
            self.h.addRow(lo, hi, len(idx), idx, np.asarray(coefs, dtype=float))
            self.nrow += 1
            return self.nrow - 1
        expr = self.gp.LinExpr(list(coefs), [self.cols[self.pos[j]] for j in handles])
        if lo == hi:
            c = self.m.addLConstr(expr, self.gp.GRB.EQUAL, lo)
        elif lo <= -self.INF:
            c = self.m.addLConstr(expr, self.gp.GRB.LESS_EQUAL, hi)
        else:
            c = self.m.addLConstr(expr, self.gp.GRB.GREATER_EQUAL, lo)
        self.rows.append(c)
        return len(self.rows) - 1

    def remove_cols(self, handles):
        """Delete columns; the remaining handles stay valid."""
        drop = sorted(self.pos[j] for j in handles if self.pos[j] >= 0)
        if not drop:
            return
        if self.backend == 'highs':
            self.h.deleteCols(len(drop), np.array(drop, dtype=np.int32))
        else:
            self.m.remove([self.cols[p] for p in drop])
            gone = set(drop)
            self.cols = [v for p, v in enumerate(self.cols) if p not in gone]
        gone = set(drop)
        for p in drop:
            self.pos[self.handle_at[p]] = -1
        self.handle_at = [j for p, j in enumerate(self.handle_at) if p not in gone]
        for p, j in enumerate(self.handle_at):
            self.pos[j] = p

    def set_cols(self, js, obj=None, ub=None):
        """Batch version of set_col for costs and upper bounds (lower bounds kept)."""
        ps = [self.pos[j] for j in js]
        if self.backend == 'highs':
            idx = np.array(ps, dtype=np.int32)
            if obj is not None:
                self.h.changeColsCost(len(ps), idx, np.asarray(obj, dtype=float))
            if ub is not None:
                cur = self.h.getCols(len(ps), idx)
                self.h.changeColsBounds(len(ps), idx, np.asarray(cur[3]),
                                        np.asarray(ub, dtype=float))
            return
        vs = [self.cols[p] for p in ps]
        if obj is not None:
            self.m.setAttr('Obj', vs, list(obj))
        if ub is not None:
            self.m.setAttr('UB', vs, list(ub))

    def set_col(self, j, obj=None, lb=None, ub=None):
        p = self.pos[j]
        if self.backend == 'highs':
            if obj is not None:
                self.h.changeColCost(p, obj)
            if lb is not None or ub is not None:
                cur = self.h.getCols(1, np.array([p], dtype=np.int32))
                lo = cur[3][0] if lb is None else lb
                up = cur[4][0] if ub is None else ub
                self.h.changeColBounds(p, lo, up)
            return
        v = self.cols[p]
        if obj is not None:
            v.Obj = obj
        if lb is not None:
            v.LB = lb
        if ub is not None:
            v.UB = ub

    def solve(self, method=None):
        """Solve; `method` overrides the simplex variant for this one solve."""
        if method is not None and method != self.method:
            saved, self.method = self.method, method
            self._apply_method()
            try:
                return self.solve()
            finally:
                self.method = saved
                self._apply_method()
        if self.backend == 'highs':
            t0 = time.time()
            self.h.run()
            info = self.h.getInfo()
            self.stats = (time.time() - t0, info.simplex_iteration_count,
                          self.h.getNumCol(), self.h.getNumNz())
            if self.h.getModelStatus() != self._st:
                raise RuntimeError(f'master LP (highs): {self.h.getModelStatus()}')
            sol = self.h.getSolution()
            self._x = np.asarray(sol.col_value)
            self._rc = np.asarray(sol.col_dual)
            self._pi = np.asarray(sol.row_dual)
            return float(self.h.getObjectiveValue())
        self.m.optimize()
        self.stats = (self.m.Runtime, self.m.IterCount, self.m.NumVars, self.m.NumNZs)
        ok = self.m.Status == self.gp.GRB.OPTIMAL or (
            # barrier without crossover may stop just short of its tolerance; the
            # iterate is still usable: the Lagrangian bound holds for any dual
            self.method == 'barrier' and self.m.Status == self.gp.GRB.SUBOPTIMAL
            and self.m.SolCount > 0)
        if not ok:
            raise RuntimeError(f'master LP (gurobi): status {self.m.Status}')
        self._x = np.array(self.m.getAttr('X', self.cols))
        self._rc = np.array(self.m.getAttr('RC', self.cols))
        self._pi = np.array(self.m.getAttr('Pi', self.rows))
        return float(self.m.ObjVal)

    def x(self, j):
        return float(self._x[self.pos[j]])

    def rc(self, j):
        return float(self._rc[self.pos[j]])

    def pi(self, r):
        return float(self._pi[r])


class _TwinLP:
    """The penalized master plus an unpenalized shadow of it, for the upper bound.

    Rows and columns go into both, so indices agree; cost and bound changes (the
    penalty) go into the main LP only, so the shadow's slacks stay closed. The
    shadow keeps its own basis and warm starts from it when asked for the bound.
    """
    def __init__(self, backend, method, presolve='auto'):
        self.main = _LP(backend, method=method, presolve=presolve)
        self.shadow = _LP(backend, name='master_ub', method=method, presolve=presolve)
        self.INF = self.main.INF

    def add_row(self, lo, hi):
        r = self.main.add_row(lo, hi)
        assert self.shadow.add_row(lo, hi) == r
        return r

    def add_col(self, obj, lb, ub, rows=(), coefs=()):
        j = self.main.add_col(obj, lb, ub, rows, coefs)
        assert self.shadow.add_col(obj, lb, ub, rows, coefs) == j
        return j

    def set_cols(self, js, obj=None, ub=None):
        self.main.set_cols(js, obj=obj, ub=ub)

    def set_col(self, j, obj=None, lb=None, ub=None):
        self.main.set_col(j, obj=obj, lb=lb, ub=ub)

    def solve(self, method=None):
        return self.main.solve(method)

    def solve_shadow(self):
        return self.shadow.solve()

    def x_shadow(self, j):
        return self.shadow.x(j)

    def remove_cols(self, handles):
        self.main.remove_cols(handles)
        self.shadow.remove_cols(handles)

    def add_row_with(self, lo, hi, handles, coefs):
        r = self.main.add_row_with(lo, hi, handles, coefs)
        assert self.shadow.add_row_with(lo, hi, handles, coefs) == r
        return r

    def rc(self, j):
        return self.main.rc(j)

    @property
    def stats(self):
        return self.main.stats

    def x(self, j):
        return self.main.x(j)

    def pi(self, r):
        return self.main.pi(r)


class DirectMaster:
    """(DWR_N^Omega) with the column generation loop written out, no SCIP pricer.

    Same master as StochasticMaster -- linking rows per (kind, hour, scenario), the
    shared block x0 = (r_sym, p^omega), one convexity row per prosumer -- and the
    same stabilization: Wentges smoothing with the adaptive alpha, plus the
    three-piece penalty of du Merle et al. around the stability center. What differs
    is that the loop is ours and the LP is HiGHS or Gurobi, so the penalty rounds
    change slack bounds in place instead of rebuilding the model.
    """
    def __init__(self, players, T, scenarios, params, lp_solver='highs',
                 pricing_solver='highs', pricing_time_limit=None, pricing_gap=None,
                 smoothing=True, incumbent=None, gap_tol=CG_GAP, pen_eps=0.2,
                 pen_delta=0.2, pen_shrink=0.25, max_rounds=12, max_iter=100000,
                 subs=None, verbose=True, pricing_workers=1, round_tol=None,
                 ub_every=10, lp_method='primal', purge_every=0, purge_age=50,
                 purge_cap=40, lp_presolve='auto', sar=False, sar_block=1, sar_cap=1.0,
                 sar_exact=(), lazy_kinds=()):
        self.players, self.T, self.scenarios = list(players), list(T), scenarios
        # dyn-SAR (Costa, Contardo, Desaulniers & Yarkony 2022): the master starts
        # from aggregated linking rows and separates the original rows it violates
        self.sar, self.sar_block, self.sar_cap = sar, sar_block, sar_cap
        # row kinds that start disaggregated (one row per hour and scenario): those
        # whose prices differ across scenarios, so averaging them only gets undone
        self.sar_exact = tuple(sar_exact)
        # lazy rows: inequality kinds (peak, dn, up) left out of the master and added
        # one by one when the RMP solution violates them. Unlike dyn-SAR this never
        # averages an equality, so a row once added stays the original row.
        self.lazy_kinds = tuple(lazy_kinds)
        if any(k in CARRIERS for k in self.lazy_kinds):
            raise ValueError('only inequality rows (up, dn, peak) can be lazy')
        if {'up', 'dn'} <= set(self.lazy_kinds):
            raise ValueError('up and dn cannot both be lazy: nothing else bounds r_sym, '
                             'so the master starts unbounded')
        self.dyn_rows = sar or bool(self.lazy_kinds)
        self.lazy_added = 0
        self.sar_phases = []
        self._shadow_viol = {}
        # pricing_workers > 1 prices that many prosumers at once (threads; Gurobi
        # releases the GIL while it solves). round_tol, if set, is the column
        # admission tolerance of every penalty round but the last, which always uses
        # gap_tol: an intermediate round only has to move the center, not converge.
        self.round_tol = round_tol
        self._pool = None
        self.ub, self._pen_state = np.inf, None
        self.ub_every, self.pricing_gap_floor, self.pricing_tightened = ub_every, 1e-9, 0
        # column management: every purge_every iterations, drop the prosumer columns
        # that have been out of the RMP solution for purge_age iterations and price
        # out at the current duals (0: keep every column)
        self.purge_every, self.purge_age, self.purge_cap = purge_every, purge_age, purge_cap
        self.last_used, self.purged = {}, 0
        self.protected = set()      # the seed columns: they keep every penalty feasible
        self.probs = [p for p, _ in scenarios]
        self.params, self.verbose, self.gap_tol = params, verbose, gap_tol
        self.smoothing, self.incumbent = smoothing, np.inf if incumbent is None else incumbent
        self.pen_eps, self.pen_delta = pen_eps, pen_delta
        self.pen_shrink, self.max_rounds, self.max_iter = pen_shrink, max_rounds, max_iter
        self.pricing_workers = os.cpu_count() if pricing_workers == 0 else pricing_workers
        self.pricing_workers = max(1, min(self.pricing_workers, len(self.players)))
        # one Gurobi environment (one WLS session) per worker; prosumer u is priced
        # by worker u mod k, in sequence with that worker's other prosumers
        k = self.pricing_workers
        self._envs = [None] * k
        if subs is None and pricing_solver == 'gurobi':
            import gurobipy as gp
            for i in range(k):
                e = gp.Env(empty=True)
                e.setParam('OutputFlag', 0)
                e.start()
                self._envs[i] = e
        self._group = {u: i % k for i, u in enumerate(self.players)}
        self.subs = subs or {u: PlayerPricing(u, T, scenarios, pricing_time_limit,
                                              pricing_gap, pricing_solver,
                                              env=self._envs[self._group[u]])
                             for u in self.players}
        self.enable_reserve = bool(params.get('enable_reserve', False))
        self.enable_peak = bool(params.get('enable_peak', False))
        self.row_keys = [(k, t, w) for w in range(len(scenarios)) for t in self.T
                         for k in CARRIERS]
        if self.enable_reserve:
            self.row_keys += [(d, t, w) for w in range(len(scenarios)) for t in self.T
                              for d in ('up', 'dn')]
        if self.enable_peak:
            self.row_keys += [('peak', t, w) for w in range(len(scenarios)) for t in self.T]
        if self.pricing_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(self.pricing_workers)
            per = max(1, (os.cpu_count() or 1) // self.pricing_workers)
            for s in self.subs.values():
                s.set_threads(per)
        self.t_lp = self.t_price = 0.0
        self.t_ub_split, self.n_ub = [0.0, 0.0], 0
        # with a penalty, the upper bound comes from an unpenalized twin of the master
        self.lp_backend = lp_solver
        self.lp = (_TwinLP(lp_solver, lp_method, lp_presolve) if pen_eps > 0.0
                   else _LP(lp_solver, method=lp_method, presolve=lp_presolve))
        self.columns = {u: [] for u in self.players}
        self.col_idx = {u: [] for u in self.players}
        self.iteration, self.lb, self.L_bar = 0, -np.inf, -np.inf
        self.center, self.best, self.log = None, None, []
        self._build()

    # --- model ---------------------------------------------------------------
    def _build(self):
        """Master rows are families: nonnegative combinations of original linking
        rows, {name: [(row key, weight)]}. Without dyn-SAR each original row is its
        own family (name = key, weight 1), which is the plain master. The duals of
        the families map back to pi_r = sum_f w_{f,r} gamma_f (Costa et al. eq. 13)."""
        p, lp, INF = self.params, self.lp, self.lp.INF
        S = list(range(len(self.scenarios)))
        if self.sar:
            kinds = sorted({k for k, _, _ in self.row_keys})
            self.sar_blocks = reserve_blocks(self.T, self.sar_block)
            fams = dict(_sar_family(k, blk, S, self.probs)
                        for k in kinds if k not in self.sar_exact
                        for blk in self.sar_blocks)
            fams.update(_sar_family(k, [t], [w], self.probs)
                        for k in kinds if k in self.sar_exact
                        for t in self.T for w in S)
        else:
            fams = {key: [(key, 1.0)] for key in self.row_keys
                    if key[0] not in self.lazy_kinds}
        self.families, self.fam_of, self.row, self.pen = {}, {}, {}, {}
        # shared block x0: which original rows each x0 variable enters
        self.x0, self.x0_rows = {}, {}
        blocks = reserve_blocks(self.T, p.get('reserve_block_hours', 24))
        block_of_t = {t: i for i, blk in enumerate(blocks) for t in blk}
        sym = p.get('reserve_product', 'symmetric') == 'symmetric'
        x0_cost = {}
        if self.enable_reserve:
            pi_res = p.get('pi_res', 0.0)
            for i, blk in enumerate(blocks):
                names = [('r_sym', i)] if sym else [('r_up', i), ('r_dn', i)]
                for nm in names:
                    price = pi_res if sym else p.get(f'pi_{nm[0][2:]}', pi_res)
                    x0_cost[nm] = -len(blk) * price
                    self.x0_rows[nm] = [(k, 1.0) for k in self.row_keys
                                        if k[0] in ('up', 'dn')
                                        and block_of_t[k[1]] == i
                                        and (sym or k[0] == nm[0].split('_')[1])]
        if self.enable_peak:
            for w, rho in enumerate(self.probs):
                x0_cost[('p', w)] = rho * p.get('pi_E_peak', 0.0)
                self.x0_rows[('p', w)] = [(('peak', t, w), -1.0) for t in self.T]
        self.x0_obj = dict(x0_cost)
        self.conv = {u: lp.add_row(1.0, 1.0) for u in self.players}
        for name, members in fams.items():
            self._register_family(name, members)
            kind = members[0][0][0]
            self.row[name] = lp.add_row(0.0, 0.0) if kind in CARRIERS \
                else lp.add_row(-INF, 0.0)
        for nm, cost in x0_cost.items():
            agg = self._aggregate(dict(self.x0_rows[nm]))
            self.x0[nm] = lp.add_col(cost, 0.0, INF, [self.row[f] for f in agg],
                                     list(agg.values()))
        # penalty slacks, created disabled (ub 0)
        for name in fams:
            self._add_pen(name)

    def _register_family(self, name, members):
        self.families[name] = members
        for key, w in members:
            self.fam_of.setdefault(key, []).append((name, w))

    def _aggregate(self, coef):
        """{original row key: a} -> {family: sum_r w_{f,r} a_r}, zeros dropped."""
        out = {}
        for key, a in coef.items():
            for name, w in self.fam_of.get(key, ()):
                out[name] = out.get(name, 0.0) + w * a
        return {f: a for f, a in out.items() if a != 0.0}

    def _add_pen(self, name):
        up = self.lp.add_col(0.0, 0.0, 0.0, [self.row[name]], [1.0])
        dn = self.lp.add_col(0.0, 0.0, 0.0, [self.row[name]], [-1.0])
        self.pen[name] = (up, dn)

    def add_family(self, name, members):
        """dyn-SAR separation: a new aggregated row over every existing column."""
        self._register_family(name, members)
        mem = dict(members)
        handles, coefs = [], []
        for u in self.players:
            for h, col in zip(self.col_idx[u], self.columns[u]):
                a = sum(w * col.coef.get(k, 0.0) for k, w in mem.items())
                if a:
                    handles.append(h)
                    coefs.append(a)
        for nm, rows in self.x0_rows.items():
            a = sum(mem.get(k, 0.0) * c for k, c in rows)
            if a:
                handles.append(self.x0[nm])
                coefs.append(a)
        kind = members[0][0][0]
        lo, hi = (0.0, 0.0) if kind in CARRIERS else (-self.lp.INF, 0.0)
        self.row[name] = self.lp.add_row_with(lo, hi, handles, coefs)
        self._add_pen(name)

    def violations(self, tol=1e-5, shadow=False):
        """Original linking rows the current RMP solution violates: {key: residual}.
        shadow=True reads the unpenalized twin's solution instead. A row that is
        already a family of its own is enforced by the LP, to the LP's tolerance,
        and is not reported."""
        xval = self.lp.x_shadow if shadow else self.lp.x
        act = {}
        for nm, rows in self.x0_rows.items():
            x = xval(self.x0[nm])
            if x:
                for k, c in rows:
                    act[k] = act.get(k, 0.0) + c * x
        for u in self.players:
            for h, col in zip(self.col_idx[u], self.columns[u]):
                lam = xval(h)
                if lam > 1e-12:
                    for k, a in col.coef.items():
                        act[k] = act.get(k, 0.0) + a * lam
        return {k: r for k, r in act.items()
                if (abs(r) > tol if k[0] in CARRIERS else r > tol)
                and not self._single(k)}

    def _single(self, key):
        kind, t, w = key
        return (key in self.families
                or _sar_family(kind, [t], [w], self.probs)[0] in self.families)

    def _add_lazy(self, viol):
        """Add the violated lazy rows, each as itself. Returns how many."""
        add = [k for k in viol if k[0] in self.lazy_kinds and not self._single(k)]
        for k in add:
            self.add_family(k, [(k, 1.0)])
        if add and self._penalized():
            self._set_penalty(*self._pen_state)
        self.lazy_added += len(add)
        return len(add)

    def _separate(self, viol):
        """Costa et al.'s policy: split a violated row's family gradually --
        (kind, block, all scenarios) -> (kind, block, one scenario) -> the row
        itself -- and add at most sar_cap of the original rows per round, largest
        violation first. Returns the number of families added."""
        S = list(range(len(self.scenarios)))
        block_of_t = {t: i for i, blk in enumerate(self.sar_blocks) for t in blk}
        cand = {}
        for (k, t, w), r in viol.items():
            blk = self.sar_blocks[block_of_t[t]]
            for name, members in (_sar_family(k, blk, S, self.probs),
                                  _sar_family(k, blk, [w], self.probs),
                                  _sar_family(k, [t], [w], self.probs)):
                if name not in self.families:
                    cur = cand.get(name)
                    cand[name] = (max(cur[0], abs(r)) if cur else abs(r), name, members)
                    break
        cap = max(1, int(self.sar_cap * len(self.row_keys)))
        added = 0
        for _, name, members in sorted(cand.values(), key=lambda c: -c[0])[:cap]:
            self.add_family(name, members)
            added += 1
        if added and self._penalized():
            self._set_penalty(*self._pen_state)     # the new rows' slacks too
        return added

    def add_column(self, col):
        u = col.player
        agg = self._aggregate(col.coef)
        rows = [self.conv[u]] + [self.row[f] for f in agg]
        coefs = [1.0] + list(agg.values())
        h = self.lp.add_col(col.cost, 0.0, self.lp.INF, rows, coefs)
        self.col_idx[u].append(h)
        self.columns[u].append(col)
        self.last_used[h] = self.iteration

    def _purge(self, ref):
        """Remove prosumer columns unused for purge_age iterations with reduced cost
        above a numerical zero at the current duals. Call right after an LP solve
        and before the next one: positions shift."""
        tol = 1e-9 * (1.0 + abs(ref))
        ncol = sum(len(v) for v in self.col_idx.values())
        cap = self.purge_cap * len(self.players)
        if ncol <= cap:
            return 0
        # candidates: unused for purge_age iterations and pricing out now, the
        # longest-unused first, until the column count is back under the cap. The
        # master is primal degenerate here -- a new plan only pays off together with
        # other prosumers' plans -- so columns must be given time to find partners.
        cand = sorted(((self.last_used[h], h) for u in self.players for h in self.col_idx[u]
                       if h not in self.protected
                       and self.iteration - self.last_used[h] >= self.purge_age
                       and self.lp.rc(h) > tol))
        drop = set(h for _, h in cand[:ncol - cap])
        for u in self.players:
            keep = [(h, c) for h, c in zip(self.col_idx[u], self.columns[u]) if h not in drop]
            self.col_idx[u] = [h for h, _ in keep]
            self.columns[u] = [c for _, c in keep]
        drop = list(drop)
        if drop:
            self.lp.remove_cols(drop)
            for h in drop:
                del self.last_used[h]
            self.purged += len(drop)
        return len(drop)

    def _set_penalty(self, center, eps, delta):
        self._pen_state = (center, eps, delta)
        js, obj = [], []
        for key, (up, dn) in self.pen.items():
            # the center lives in the original rows; a family's is the least-squares
            # gamma with pi = w gamma, i.e. sum w pi / sum w^2 (pi itself for a row)
            mem = self.families[key]
            c = (sum(w * center.get(k, 0.0) for k, w in mem) / sum(w * w for _, w in mem)
                 if center else 0.0)
            e = eps * (1.0 + abs(c))
            js += [up, dn]
            obj += [c + e, -c + e]
        self.lp.set_cols(js, obj=obj, ub=[delta] * len(js))

    # --- duals and bound -----------------------------------------------------
    def _duals(self):
        gamma = {f: self.lp.pi(r) for f, r in self.row.items()}
        duals = {k: sum(w * gamma[f] for f, w in fams) for k, fams in self.fam_of.items()}
        conv = {u: self.lp.pi(r) for u, r in self.conv.items()}
        return duals, conv

    def _lagrangian(self, duals, bounds):
        for k, rows in self.x0_rows.items():
            rc = self.x0_obj[k] - sum(duals.get(r, 0.0) * a for r, a in rows)
            if rc < -1e-9 * (1.0 + abs(self.x0_obj[k])):
                return -np.inf
        if any(v > 1e-9 for (kind, _, _), v in duals.items() if kind not in CARRIERS):
            return -np.inf
        return sum(bounds)

    def _record(self, duals, res):
        L = self._lagrangian(duals, [r[1] for r in res.values()])
        if L > self.lb:
            self.lb = L
            self.best = (dict(duals), {u: r[0] for u, r in res.items()},
                         {u: r[2] for u, r in res.items()})
        if L > self.L_bar:
            self.L_bar, self.center = L, dict(duals)

    def _alpha(self, lp_obj):
        gap = lp_obj - self.L_bar
        if not np.isfinite(gap):
            return 0.1
        return 1.0 if gap <= self.gap_tol * (1.0 + abs(lp_obj)) else (
            min(1.0, 0.1 * (self.incumbent - self.L_bar) / gap)
            if lp_obj > self.incumbent and self.incumbent - self.L_bar > 1e-6 else 0.1)

    def _price_group(self, members, duals):
        return {u: self.subs[u].price(duals) for u in members}

    def _price_all(self, duals):
        t0 = time.time()
        if self._pool is None:
            res = {u: s.price(duals) for u, s in self.subs.items()}
        else:
            groups = {}
            for u in self.players:
                groups.setdefault(self._group[u], []).append(u)
            futs = [self._pool.submit(self._price_group, m, duals) for m in groups.values()]
            res = {}
            for f in futs:
                res.update(f.result())
            res = {u: res[u] for u in self.players}
        dt = time.time() - t0
        self.t_price += dt
        self._it_price += dt
        return res

    # --- bounds --------------------------------------------------------------
    def _penalized(self):
        return self._pen_state is not None and self._pen_state[2] > 0.0

    def _gap_tol(self, ref):
        return self.gap_tol * (1.0 + abs(ref))

    def _update_ub(self, lp_obj=None):
        """Upper bound on z_MP: the value of the RMP without the penalty slacks.

        Unpenalized, that is the LP just solved. Penalized, it is the twin LP that has
        every column but never the slacks: the restricted master over every column so
        far bounds z_MP from above whatever the stabilization is doing.
        """
        t0 = time.time()
        if not self._penalized():
            ub = lp_obj if lp_obj is not None else self.lp.solve()
        else:
            t1 = time.time()
            try:
                ub = self.lp.solve_shadow()
                if self.dyn_rows:
                    # the shadow has the families, not the original rows
                    self._shadow_viol = self.violations(shadow=True)
                    if self._shadow_viol:
                        ub = np.inf
                        if self.lazy_kinds:
                            self._add_lazy(self._shadow_viol)
            except RuntimeError:        # not yet feasible without the slacks
                ub = np.inf
            self.t_ub_split[0] += time.time() - t1
            self.n_ub += 1
        self.t_lp += time.time() - t0
        if ub < self.ub:
            self.ub = ub
        return ub

    def _converged(self):
        return np.isfinite(self.ub) and self.ub - self.lb <= self._gap_tol(self.ub)

    def _tighten_pricing(self, res):
        """No column prices, yet LB is short of the RMP value: only the pricing MILPs'
        own gaps can be holding the Lagrangian bound down. Tighten them tenfold for
        the prosumers whose bound is loose; False once all of those are at the floor."""
        changed = False
        for u, (obj, bound, _) in res.items():
            sub = self.subs[u]
            if obj - bound > 1e-9 * (1.0 + abs(obj)) and sub.gap > self.pricing_gap_floor:
                sub.set_gap(max(sub.gap / 10.0, self.pricing_gap_floor))
                changed = True
        if changed:
            self.pricing_tightened += 1
        return changed

    # --- the loop ------------------------------------------------------------
    def _cg(self, tag, last=True):
        """Column generation for one penalty setting.

        Returns ('done', v) as soon as the Lagrangian bound LB is within gap_tol of an
        upper bound UB on z_MP -- the unpenalized RMP value -- whatever round this is:
        the whole problem is then solved (Lagrangian-bound early termination).
        Returns ('round', v) once LB has reached this round's penalized RMP value,
        i.e. the stabilized problem is solved and the center can move. Any column
        with negative reduced cost is admitted: termination never rests on reduced
        costs, which an inexactly solved pricing MILP cannot certify, only on LB,
        which is built from the pricing MILPs' dual bounds and so stays valid.
        """
        rel = self.gap_tol if (last or self.round_tol is None) else self.round_tol
        since_ub = 0
        while True:
            t0 = time.time()
            lp_obj = self.lp.solve()
            t_lp = time.time() - t0
            self.t_lp += t_lp
            self._it_price = 0.0
            duals, conv = self._duals()
            if self.purge_every:
                for u in self.players:
                    for h in self.col_idx[u]:
                        if self.lp.x(h) > 1e-9:
                            self.last_used[h] = self.iteration
            viol = self.violations() if self.dyn_rows else {}
            if self.lazy_kinds and self._add_lazy(viol):
                # the RMP left out rows it now breaks: add them and re-solve before
                # pricing, so pricing sees their duals
                self.log.append({'iter': self.iteration, 'lp': lp_obj, 'lb': self.lb,
                                 'ub': self.ub, 'mode': 'lazy', 'added': 0,
                                 'round': tag, 't_lp': t_lp, 't_price': 0.0})
                continue
            if not self._penalized() and not viol:
                # (under dyn-SAR, only a solution of the aggregated master that
                # violates no original row is feasible, and so bounds z_MP)
                self._update_ub(lp_obj)
            tol = rel * (1.0 + abs(lp_obj))
            adm = 1e-9 * (1.0 + abs(lp_obj))        # numerical zero for reduced costs
            self.iteration += 1
            if self.iteration > self.max_iter:
                raise RuntimeError('column generation: iteration limit')
            alpha, added, min_rc, mode = 1.0, 0, 0.0, 'std'
            status = None
            if self._converged():
                status = 'done'
            elif self.lb >= lp_obj - tol:
                status = 'round'
            if status is None and self.smoothing:
                if self.center is None:
                    self.center = dict(duals)
                alpha = self._alpha(lp_obj)
                if alpha < 1.0:
                    st = {k: alpha * duals.get(k, 0.0) + (1 - alpha) * self.center.get(k, 0.0)
                          for k in set(duals) | set(self.center)}
                    res = self._price_all(st)
                    self._record(st, res)
                    mode = 'smooth'
                    for u, (_, _, col) in res.items():
                        rc = col.cost - sum(duals.get(k, 0.0) * a
                                            for k, a in col.coef.items()) - conv[u]
                        min_rc = min(min_rc, rc)
                        if rc < -adm:
                            self.add_column(col)
                            added += 1
                    if not added:
                        mode = 'misprice'
            if status is None and not added:
                if self._converged():
                    status = 'done'
                elif self.lb >= lp_obj - tol:
                    status = 'round'
                else:
                    res = self._price_all(duals)
                    self._record(duals, res)
                    for u, (obj, _, col) in res.items():
                        rc = obj - conv[u]
                        min_rc = min(min_rc, rc)
                        if rc < -adm:
                            self.add_column(col)
                            added += 1
                    if not added:
                        if self._converged():
                            status = 'done'
                        elif self.lb >= lp_obj - tol:
                            status = 'round'
                        elif self._tighten_pricing(res):
                            mode = 'tighten'
                        else:
                            status = 'stalled'
            if self.sar and status in ('round', 'stalled') and self._penalized():
                self._update_ub()
                viol = self._shadow_viol
                if not viol and self._converged():
                    status = 'done'
            if self.sar and status in ('round', 'stalled') and viol:
                # converged on the aggregated master: separate the violated rows
                n_add = self._separate(viol)
                self.sar_phases.append({'iteration': self.iteration, 'lp': lp_obj,
                                        'lb': self.lb, 'violated': len(viol),
                                        'added': n_add, 'rows': len(self.row)})
                if self.verbose:
                    kinds = {}
                    for (k, _, _) in viol:
                        kinds[k] = kinds.get(k, 0) + 1
                    print(f'  -- SAR: {len(viol)} original rows violated {kinds}, +{n_add} rows '
                          f'-> {len(self.row)} rows (RMP {lp_obj:.4f}, LB {self.lb:.4f})')
                if n_add:
                    status, mode = None, 'separate'
            since_ub += 1
            if status is None and self._penalized() and since_ub >= self.ub_every:
                self._update_ub()
                since_ub = 0
                if self._converged():
                    status = 'done'
            self.log.append({'iter': self.iteration, 'lp': lp_obj, 'lb': self.lb,
                             'ub': self.ub, 'alpha': alpha, 'mode': mode,
                             'min_rc': min_rc, 'added': added, 'round': tag,
                             't_lp': t_lp, 't_price': self._it_price,
                             'lp_stats': getattr(self.lp, 'stats', None)})
            if self.verbose and (self.iteration % 25 == 0 or status or mode == 'tighten'):
                print(f'  CG {self.iteration:4d} | RMP {lp_obj:13.4f} | LB {self.lb:13.4f} '
                      f'| UB {self.ub:13.4f} | a {alpha:.2f} {mode:8s} '
                      f'| min rc {min_rc:11.4e} | +{added}' + (f'  [{status}]' if status else '')
                      + (f' | lp {t_lp:.2f}s solver {self.lp.stats[0]:.2f}s '
                         f'{self.lp.stats[1]:.0f} it {self.lp.stats[2]} cols {self.lp.stats[3]} nz'
                         if getattr(self.lp, 'stats', None) else ''))
            if status:
                return status, lp_obj
            if self.purge_every and self.iteration % self.purge_every == 0:
                self._purge(lp_obj)         # the next pass re-solves the LP first

    def _add_seeds(self, init_vals, init_cols):
        """The first column per prosumer: its plan in the extensive-form solution."""
        cols = list(init_cols) if init_cols is not None else [
            self.subs[u].column_from(init_vals) for u in self.players]
        # The seed plans come from a MIP solved to its feasibility tolerance, so they
        # balance the carrier rows only to ~1e-7. With every prosumer on its single
        # seed column those equality rows are then feasible or not at the LP's own
        # tolerance, and Gurobi has been seen to call the plain master infeasible on
        # a re-solve. Put each row's residual on the prosumer with the largest term
        # there: community trade carries no cost, so column costs are unchanged.
        res = {}
        for c in cols:
            for k, a in c.coef.items():
                if k[0] in CARRIERS:
                    res[k] = res.get(k, 0.0) + a
        self.seed_residual = max((abs(v) for v in res.values()), default=0.0)
        if init_cols is None:
            for k, r in res.items():
                if r == 0.0:
                    continue
                c = max(cols, key=lambda c: abs(c.coef.get(k, 0.0)))
                c.coef[k] = c.coef.get(k, 0.0) - r
        if self.verbose:
            print(f'  seed columns: largest carrier-row residual {self.seed_residual:.2e}'
                  + (' (repaired)' if init_cols is None and self.seed_residual else ''))
        for c in cols:
            self.add_column(c)
        self.protected = {h for u in self.players for h in self.col_idx[u]}
        if self.lp_backend == 'gurobi' and isinstance(self.lp, _LP):
            m = self.lp.m
            m.optimize()
            if m.Status in (3, 4):          # diagnose an infeasible seed master
                m.computeIIS()
                inv = {r: k for k, r in self.row.items()}
                inv.update({r: ('conv', u) for u, r in self.conv.items()})
                rows = [inv.get(i) for i, c in enumerate(self.lp.rows) if c.IISConstr]
                bnds = sum(1 for v in self.lp.cols if v.IISLB or v.IISUB)
                print(f'  seed master infeasible; IIS rows {rows[:30]} '
                      f'({len(rows)} rows, {bnds} bounds)')
    def solve(self, init_vals=None, init_cols=None):
        t0 = time.time()
        self._add_seeds(init_vals, init_cols)
        eps, delta, rounds = self.pen_eps, self.pen_delta, []
        status = None
        for rnd in range(1, self.max_rounds + 1):
            last = rnd == self.max_rounds or eps <= 0.0
            self._set_penalty(self.center if not last else None,
                              0.0 if last else eps, 0.0 if last else delta)
            status, obj = self._cg(rnd, last)
            if status != 'done':
                self._update_ub()
                if self._converged():
                    status = 'done'
            slack = max((max(self.lp.x(a), self.lp.x(b)) for a, b in self.pen.values()),
                        default=0.0)
            rounds.append({'round': rnd, 'eps': eps, 'delta': delta, 'obj': obj,
                           'lb': self.lb, 'ub': self.ub, 'slack': slack,
                           'iterations': self.iteration, 'time': time.time() - t0,
                           'certified': status == 'done', 'status': status})
            if self.verbose:
                print(f'  -- penalty round {rnd}: eps {eps:.3g} delta {delta:.3g} '
                      f'RMP {obj:.4f} LB {self.lb:.4f} UB {self.ub:.4f} '
                      f'gap {(self.ub - self.lb) / (1 + abs(self.ub)):.2e} '
                      f'[{status}] {time.time() - t0:.0f}s')
            if status == 'done' or last:
                break
            eps, delta = eps * self.pen_shrink, delta * self.pen_shrink
            if eps < 1e-4:
                eps = delta = 0.0
        # report the unpenalized restricted master
        self._set_penalty(None, 0.0, 0.0)
        obj = self.lp.solve()
        if obj < self.ub:
            self.ub = obj
        duals, sigma = (self.best[0], dict(self.best[1])) if self.best else self._duals()
        return {'status': 'optimal' if status == 'done' else status,
                'obj': obj, 'lb': self.lb, 'ub': self.ub,
                'gap': (self.ub - self.lb) / (1.0 + abs(self.ub)), 'duals': duals,
                'sigma': sigma, 'iterations': self.iteration,
                'lambda': {u: [self.lp.x(j) for j in self.col_idx[u]] for u in self.players},
                'x0': {f'{k[0]}_{k[1]}': self.lp.x(j) for k, j in self.x0.items()},
                'columns': {u: len(self.columns[u]) for u in self.players},
                'y_total': 0.0, 'penalty': {'rounds': rounds}, 'doi': {'used': False},
                'time': time.time() - t0,
                'timing': {'lp': self.t_lp, 'pricing': self.t_price,
                           'pricing_by_player': {u: s.time for u, s in self.subs.items()},
                           'pricing_calls': sum(s.calls for s in self.subs.values()),
                           'pricing_tightened': self.pricing_tightened,
                           'purged': self.purged,
                           'seed_residual': self.seed_residual,
                           'ub_checks': self.n_ub, 'ub_solve': self.t_ub_split[0],
                           'ub_restore': self.t_ub_split[1],
                           'pricing_gap_final': {u: s.gap for u, s in self.subs.items()},
                           'workers': self.pricing_workers},
                'sar': {'phases': self.sar_phases, 'final_rows': len(self.row),
                        'original_rows': len(self.row_keys)} if self.sar else None,
                'lazy': {'kinds': list(self.lazy_kinds), 'added': self.lazy_added,
                         'final_rows': len(self.row),
                         'original_rows': len(self.row_keys)} if self.lazy_kinds else None}

    def terminal_columns(self, duals):
        if self.best is not None and duals is self.best[0]:
            return {u: (self.best[1][u], self.best[2][u]) for u in self.players}
        out = {}
        for u, sub in self.subs.items():
            obj, _, col = sub.price(duals)
            out[u] = (obj, col)
        return out


class BundleMaster(DirectMaster):
    """(DWR_N^Omega) by a disaggregated proximal bundle method on the Lagrangian dual.

    Column generation is Kelley's cutting-plane method on L(pi) = sum_u v_u(pi): every
    column is a cut, the RMP carries all of them, and at n=60, |Omega|=5 it ends with
    thousands of dense columns (about 277 nonzeros each) whose LP is most of the run.
    The bundle method keeps a bounded set of cuts and stabilizes with a proximal term
    instead of smoothing and penalty rounds (Lemarechal; Kiwiel; Frangioni, "Generalized
    bundle methods", SIAM J. Optim. 2002; Briant et al., Math. Program. 2008).

    Master, as a QP in pi:
        max  sum_u theta_u - (1/2t) ||pi - pi_hat||^2
        s.t. theta_u <= c_j - pi' a_j   (j in the bundle of u),   pi in Theta.
    It is solved in its dual form, the RMP with its linking rows made soft:
        min  c'lambda + c0'x0 - pi_hat'z + (t/2)||z||^2
        s.t. z = A lambda + A0 x0 (+ s on the <= rows, s >= 0),  sum_j lambda_uj = 1,
    and pi = pi_hat - t z, theta_u = dual of u's convexity row. The model value
    m = sum_u theta_u at pi gives the predicted increase delta = m - L(pi_hat); pi
    becomes the new center (serious step) when L(pi) >= L(pi_hat) + m1 delta, and
    otherwise only its cuts are kept (null step).

    Cuts with lambda = 0 for bundle_age iterations are dropped; the seed columns never
    are, so the plain LP over the bundle's columns (DirectMaster's LP, which holds
    exactly the same columns) is always feasible. That LP is the upper bound, and the
    run stops on the same test as DirectMaster: UB - LB <= gap_tol (1 + |UB|).
    """
    def __init__(self, *args, bundle_t=10.0, bundle_t_min=1e-2, bundle_t_max=1e4,
                 bundle_m1=0.1, bundle_age=10, bundle_cap=20, bundle_qp='primal', **kw):
        kw['pen_eps'] = 0.0             # the UB LP is the plain master
        kw['purge_every'] = 0           # the bundle manages its own columns
        super().__init__(*args, **kw)
        if self.lp_backend != 'gurobi':
            raise ValueError('the bundle master is implemented on Gurobi (QP)')
        self.t, self.t_min, self.t_max = bundle_t, bundle_t_min, bundle_t_max
        self.m1, self.age, self.cap, self.qp_method = bundle_m1, bundle_age, bundle_cap, bundle_qp
        self.aggregated = 0
        self.t_qp, self.n_serious, self.n_null, self.dropped = 0.0, 0, 0, 0
        self._build_qp()

    # --- the QP --------------------------------------------------------------
    def _build_qp(self):
        import gurobipy as gp
        self.gp = gp
        q = self.q = gp.Model('bundle_qp', env=self._envs[0]) if self._envs[0] \
            else gp.Model('bundle_qp')
        q.Params.OutputFlag = 0
        # simplex keeps a basis between solves and returns a sparse lambda (a vertex of
        # the QP's active face); barrier starts over and makes every lambda positive,
        # so no cut ever looks inactive
        q.Params.Method = {'primal': 0, 'dual': 1, 'barrier': 2}[self.qp_method]
        self.qz, self.qrow, self.qconv, self.qx0, self.qvar = {}, {}, {}, {}, {}
        self.qcost = {}
        for key in self.row_keys:
            z = q.addVar(lb=-gp.GRB.INFINITY, name=f'z_{key[0]}_{key[1]}_{key[2]}')
            expr = gp.LinExpr(1.0, z)
            if key[0] not in CARRIERS:              # <= row: z = row + s, s >= 0
                expr.add(q.addVar(lb=0.0), -1.0)
            self.qz[key] = z
            self.qrow[key] = q.addLConstr(expr, gp.GRB.EQUAL, 0.0)
        for u in self.players:
            self.qconv[u] = q.addLConstr(gp.LinExpr(), gp.GRB.EQUAL, 1.0)
        for nm, rows in self.x0_rows.items():
            self.qx0[nm] = q.addVar(lb=0.0, obj=self.x0_obj[nm],
                                    column=gp.Column([-c for _, c in rows],
                                                     [self.qrow[k] for k, _ in rows]))
        q.update()

    def add_column(self, col):
        super().add_column(col)
        h = self.col_idx[col.player][-1]
        gp = self.gp
        cons = [self.qconv[col.player]] + [self.qrow[k] for k in col.coef]
        coefs = [1.0] + [-a for a in col.coef.values()]
        self.qvar[h] = self.q.addVar(lb=0.0, obj=col.cost, column=gp.Column(coefs, cons))
        self.qcost[h] = col.cost

    def _drop(self, handles):
        drop = set(handles)
        for u in self.players:
            keep = [(h, c) for h, c in zip(self.col_idx[u], self.columns[u]) if h not in drop]
            self.col_idx[u] = [h for h, _ in keep]
            self.columns[u] = [c for _, c in keep]
        self.lp.remove_cols(list(drop))
        self.q.remove([self.qvar.pop(h) for h in drop])
        for h in drop:
            self.qcost.pop(h, None)
        for h in drop:
            self.last_used.pop(h, None)
        self.dropped += len(drop)

    def _solve_qp(self, center):
        gp, q = self.gp, self.q
        t0 = time.time()
        zs = [self.qz[k] for k in self.row_keys]
        lin = gp.LinExpr([-center.get(k, 0.0) for k in self.row_keys], zs)
        quad = gp.QuadExpr()
        quad.addTerms([0.5 * self.t] * len(zs), zs, zs)
        # the lambda/x0 costs sit in their Obj attributes; setObjective replaces them,
        # so they go back in explicitly
        hs = list(self.qvar)
        costs = gp.LinExpr([self.qcost[h] for h in hs], [self.qvar[h] for h in hs])
        costs.add(gp.LinExpr([self.x0_obj[nm] for nm in self.qx0], list(self.qx0.values())))
        q.setObjective(costs + lin + quad, gp.GRB.MINIMIZE)
        q.optimize()
        self.t_qp += time.time() - t0
        if q.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f'bundle QP: status {q.Status}')
        z = dict(zip(self.row_keys, q.getAttr('X', zs)))
        pi = {k: center.get(k, 0.0) - self.t * z[k] for k in self.row_keys}
        theta = {u: c.Pi for u, c in self.qconv.items()}
        lam = {h: v.X for h, v in self.qvar.items()}
        return pi, theta, z, lam

    def _compress(self, lam, keep):
        """Bundle compression: a prosumer with more than `cap` cuts has every cut
        outside `keep` replaced by their lambda-weighted convex combination (the
        aggregate cut), and those with lambda = 0 dropped. The QP's primal solution
        survives -- the aggregate carries the weight -- so the model loses no value at
        the current point."""
        for u in self.players:
            if len(self.col_idx[u]) <= self.cap:
                continue
            cand = [(h, c) for h, c in zip(self.col_idx[u], self.columns[u])
                    if h not in keep and h in lam]
            if len(cand) < 2:
                continue
            pos = [(h, c, lam[h]) for h, c in cand if lam[h] > 1e-12]
            self._drop([h for h, _ in cand])
            if pos:
                tot = sum(w for _, _, w in pos)
                agg = Column.combine([c for _, c, _ in pos], [w / tot for _, _, w in pos])
                self.add_column(agg)
                self.last_used[self.col_idx[u][-1]] = self.iteration
                self.aggregated += len(pos)

    # --- the loop ------------------------------------------------------------
    def solve(self, init_vals=None, init_cols=None):
        t0 = time.time()
        self._add_seeds(init_vals, init_cols)
        # first center: the seed RMP's duals, in Theta by LP duality
        lp_obj = self.lp.solve()
        self._update_ub(lp_obj)
        center, _ = self._duals()
        self._it_price = 0.0
        res = self._price_all(center)
        self._record(center, res)
        # The descent test runs on the oracle's values, sum_u of the incumbent pricing
        # objectives (Kiwiel's inexact oracle); the pricing MILPs' dual bounds, which
        # sit up to their gap below, only certify LB. With the bounds in the test a
        # step whose model is already exact reads as a failure by exactly that gap,
        # and the method stalls in null steps with t shrinking to its floor.
        L_hat = self._lagrangian(center, [r[0] for r in res.values()])
        for u, (obj, _, col) in res.items():
            self.add_column(col)
        self.center_cuts = {self.col_idx[u][-1] for u in self.players}
        log_every = 25
        status = None
        while True:
            self.iteration += 1
            if self.iteration > self.max_iter:
                raise RuntimeError('bundle: iteration limit')
            self._it_price = 0.0
            pi, theta, z, lam = self._solve_qp(center)
            model = sum(theta.values())
            delta = model - L_hat
            for h, v in lam.items():
                if v > 1e-9:
                    self.last_used[h] = self.iteration
            res = self._price_all(pi)
            self._record(pi, res)
            L_k = self._lagrangian(pi, [r[0] for r in res.values()])
            serious = np.isfinite(L_k) and L_k >= L_hat + self.m1 * delta
            added, new = 0, {}
            for u, (obj, _, col) in res.items():
                # a serious step keeps every cut at the new center, so the model there
                # is exact; a null step only the cuts that improve it
                if serious or obj < theta[u] - 1e-9 * (1.0 + abs(theta[u])):
                    self.add_column(col)
                    new[u] = self.col_idx[u][-1]
                    added += 1
            if serious:
                if L_k - L_hat >= 0.5 * delta:
                    self.t = min(2.0 * self.t, self.t_max)
                # the center's value is the model's there: sum_u min(theta_u, obj_u).
                # With an inexact oracle obj_u may exceed theta_u; taking the model
                # value keeps delta >= 0 (Kiwiel's noise-safe choice)
                L_hat = sum(min(theta[u], res[u][0]) for u in self.players)
                center = dict(pi)
                self.center_cuts = set(new.values())
                self.n_serious += 1
                self._nulls = 0
                step = 'serious'
            else:
                self._nulls = getattr(self, '_nulls', 0) + 1
                if self._nulls % 5 == 0:
                    self.t = max(0.5 * self.t, self.t_min)
                self.n_null += 1
                step = 'null'
            keep = self.protected | getattr(self, 'center_cuts', set())
            old = [h for u in self.players for h in self.col_idx[u]
                   if h not in keep and h in lam
                   and self.iteration - self.last_used.get(h, self.iteration) >= self.age]
            if old:
                self._drop(old)
            self._compress(lam, keep | set(new.values()))
            znorm = float(np.sqrt(sum(v * v for v in z.values())))
            tol = self._gap_tol(self.ub if np.isfinite(self.ub) else model)
            if self.iteration % self.ub_every == 0 or delta <= tol:
                self._update_ub(self.lp.solve())
            if self._converged():
                status = 'done'
            elif delta <= tol and not added:
                # the model is exact at pi and predicts no gain: only the pricing
                # MILPs' gaps can keep LB below UB
                if not self._tighten_pricing(res):
                    status = 'stalled'
            ncol = sum(len(v) for v in self.col_idx.values())
            self.log.append({'iter': self.iteration, 'lp': model, 'lb': self.lb,
                             'ub': self.ub, 'mode': step, 't': self.t, 'delta': delta,
                             'z': znorm, 'added': added, 'columns': ncol,
                             't_price': self._it_price, 't_lp': 0.0})
            if self.verbose and (self.iteration % log_every == 0 or status):
                print(f'  BN {self.iteration:4d} | model {model:13.4f} | L^ {L_hat:13.4f} '
                      f'| LB {self.lb:13.4f} | UB {self.ub:13.4f} | t {self.t:8.3g} '
                      f'| delta {delta:9.3e} | |z| {znorm:8.2e} | {step:7s} +{added} '
                      f'| cols {ncol}' + (f'  [{status}]' if status else ''))
            if status:
                break
        obj = self.lp.solve()
        if obj < self.ub:
            self.ub = obj
        duals, sigma = (self.best[0], dict(self.best[1])) if self.best else self._duals()
        return {'status': 'optimal' if status == 'done' else status,
                'obj': obj, 'lb': self.lb, 'ub': self.ub,
                'gap': (self.ub - self.lb) / (1.0 + abs(self.ub)), 'duals': duals,
                'sigma': sigma, 'iterations': self.iteration,
                'lambda': {u: [self.lp.x(j) for j in self.col_idx[u]] for u in self.players},
                'x0': {f'{k[0]}_{k[1]}': self.lp.x(j) for k, j in self.x0.items()},
                'columns': {u: len(self.columns[u]) for u in self.players},
                'y_total': 0.0, 'penalty': {'rounds': []}, 'doi': {'used': False},
                'time': time.time() - t0,
                'timing': {'lp': self.t_lp, 'qp': self.t_qp, 'pricing': self.t_price,
                           'pricing_by_player': {u: s.time for u, s in self.subs.items()},
                           'pricing_calls': sum(s.calls for s in self.subs.values()),
                           'pricing_tightened': self.pricing_tightened,
                           'serious': self.n_serious, 'null': self.n_null,
                           'dropped': self.dropped, 'aggregated': self.aggregated,
                           'workers': self.pricing_workers,
                           'seed_residual': self.seed_residual},
                'bundle': {'t_final': self.t, 'serious': self.n_serious,
                           'null': self.n_null}}


def solve_dwr_bundle(players, T, scenarios, params, init_vals=None, **kw):
    """(DWR_N^Omega) through BundleMaster."""
    master = BundleMaster(players, T, scenarios, params, **kw)
    return master.solve(init_vals=init_vals), master


def solve_dwr_direct(players, T, scenarios, params, init_vals=None, **kw):
    """(DWR_N^Omega) through DirectMaster: our own loop, HiGHS or Gurobi, no SCIP."""
    master = DirectMaster(players, T, scenarios, params, **kw)
    return master.solve(init_vals=init_vals), master


def solve_dwr_stab(players, T, scenarios, params, init_vals=None, pen_eps=0.1,
                   pen_delta=0.05, pen_shrink=0.25, max_rounds=12, **kw):
    """(DWR_N^Omega) with smoothing AND a three-piece penalty around the center.

    Round 0 solves the restricted master once, without pricing, to place the first
    stability center. Every round after that rebuilds the master with the penalty at
    the current center, runs column generation to convergence, then recenters on the
    dual that attains the Lagrangian bound and shrinks eps and delta. The penalty
    relaxes the linking rows, so a round only certifies z when its slacks come out at
    zero; the last round runs with no penalty at all, which always certifies.

    MEASURED, 6 prosumers, Gurobi pricing, --cg-gap 1e-8, in iterations:

        |Omega|      1      3         5
        both       194    454      1558  (5.8 s / 50 s / 343 s)
        smoothing  279    831         -  (two runs, neither converged in 50 min,
        penalty    357   1424         -   both frozen at a reduced cost of -4.8e-3)

    Neither half works on its own, which is the combination Pessoa et al. (2018)
    recommend. The penalty only bounds where the duals may go; inside the band of
    width eps_r they still jump between extreme points, and the relaxed early rounds
    price columns against a master far from z (RMP -3560 against z -2733 at
    |Omega| = 3, where smoothing holds the same round at -3094). Smoothing damps the
    jumps but cannot price the rotation among alternative optima that freezes the
    RMP value late on. Hence both, on by default.
    """
    master = StochasticMaster(players, T, scenarios, params, **kw)
    first = master.solve(init_vals=init_vals, pricing=False)
    center, _ = master._duals(False)
    cols = [c for u in master.players for c in master.columns[u]]
    subs, rounds = master.subs, []
    eps, delta = pen_eps, pen_delta
    t0 = time.time()
    state = {'lb': -np.inf, 'L_bar': -np.inf, 'center': None, 'best': None,
             'iteration': 0, 'log': []}
    for rnd in range(1, max_rounds + 1):
        last = rnd == max_rounds or eps <= 0.0
        pen = None if last else {'center': center, 'eps': eps, 'delta': delta}
        master = StochasticMaster(players, T, scenarios, params, subs=subs,
                                  penalty=pen, **kw)
        master.lb, master.L_bar, master.center = state['lb'], state['L_bar'], state['center']
        master.best = state['best']
        master.iteration, master.log = state['iteration'], state['log']
        res = master.solve(init_cols=cols)
        state = {'lb': master.lb, 'L_bar': master.L_bar, 'center': master.center,
                 'best': master.best, 'iteration': master.iteration, 'log': master.log}
        cols = [c for u in master.players for c in master.columns[u]]
        slack = res['penalty_slack']
        tol = (len(players) + 1) * master.gap_tol * (1 + abs(res['obj']))
        done = slack <= 1e-7 and master.lb >= res['obj'] - tol
        rounds.append({'round': rnd, 'eps': eps, 'delta': delta, 'obj': res['obj'],
                       'lb': master.lb, 'slack': slack, 'iterations': master.iteration,
                       'time': time.time() - t0, 'certified': bool(done)})
        if master.verbose:
            print(f'  -- penalty round {rnd}: eps {eps:.3g} delta {delta:.3g} '
                  f'RMP {res["obj"]:.4f} LB {master.lb:.4f} slack {slack:.2e}'
                  + ('  [certified]' if done else ''))
        if done:
            break
        if master.best is not None:
            center = master.best[0]
        eps, delta = eps * pen_shrink, delta * pen_shrink
        if eps < 1e-4:
            eps = delta = 0.0
    else:
        raise RuntimeError('penalty stabilization: no certified round')
    res['time'] = time.time() - t0 + first['time']
    res['iterations'] = master.iteration
    res['penalty'] = {'rounds': rounds, 'eps0': pen_eps, 'delta0': pen_delta,
                      'shrink': pen_shrink}
    res['doi'] = {'used': False}
    return res, master


def solve_dwr(players, T, scenarios, params, init_vals=None, doi=False, **kw):
    """(DWR_N^Omega) by column generation, optionally with the balance-row DOIs.

    Returns (result, master). With the DOIs on, y = 0 at convergence settles it.
    Otherwise the collected columns are tested against the original master (one LP);
    only if that fails does column generation resume without the DOIs.
    """
    master = StochasticMaster(players, T, scenarios, params, doi=doi, **kw)
    res = master.solve(init_vals=init_vals)
    res['doi'] = {'used': doi, 'y_total': res['y_total'], 'status': 'unused'}
    if not doi:
        return res, master
    if res['y_total'] <= 1e-7:
        res['doi']['status'] = 'y = 0'
        return res, master

    # y > 0. The final duals pi* price out for the augmented master and satisfy the
    # original dual constraints (a subset), so z_orig >= z_aug. Any restricted
    # original master is an upper bound; if the columns already collected reach
    # z_aug, then z_orig = z_aug and pi*, sigma* are optimal for the original too.
    cols = [c for u in master.players for c in master.columns[u]]
    z_aug = res['obj']
    tol = 1e-9 * (1.0 + abs(z_aug))
    plain = StochasticMaster(players, T, scenarios, params, doi=False,
                             subs=master.subs, **kw)
    check = plain.solve(init_cols=cols, pricing=False)
    info = {'used': True, 'y_total': res['y_total'], 'z_aug': z_aug,
            'restricted_orig': check['obj']}
    if check['obj'] <= z_aug + tol:
        info['status'] = 'certified by restricted master'
        res.update(obj=check['obj'], x0=check['x0'], y_total=0.0, doi=info,
                   time=res['time'] + check['time'])
        res['lambda'] = check['lambda']
        return res, master

    print(f'  DOI not dual-optimal here (restricted master {check["obj"]:.6f} > '
          f'{z_aug:.6f}); resuming without them')
    resume = StochasticMaster(players, T, scenarios, params, doi=False,
                              subs=master.subs, **kw)
    resume.lb, resume.L_bar = master.lb, master.L_bar   # valid: bounds on z_orig
    resume.center, resume.best = master.center, master.best
    res2 = resume.solve(init_cols=cols)
    info['status'] = 'fallback'
    info['first_pass'] = {k: res[k] for k in ('obj', 'lb', 'iterations', 'time')}
    res2['doi'] = info
    res2['iterations'] += res['iterations']
    res2['time'] += res['time'] + check['time']
    return res2, resume


# =============================================================================
# Algorithm S1
# =============================================================================
def scenario_allocation(ef, dw, master):
    """x_j*(omega) of eq:sup_resid, in the profit convention.

    kappa_j(omega) = c_j^omega - pibar^omega . a_j^omega + SU_j   (cost, per scenario)
    chi_j^Omega(omega) = -kappa_j(omega)                            eq:sup_chiscen
    g^omega = sum_j chi_j^Omega(omega) - v^Omega(N, a_hat, omega)
    x_j*(omega) = chi_j^Omega(omega) - g^omega / n
    """
    players, probs = master.players, master.probs
    S, n = len(probs), len(players)
    duals = dw['duals']
    q = master.terminal_columns(duals)
    chi = {}
    pricing_value = {}
    for u in players:
        obj, col = q[u]
        pricing_value[u] = obj
        pen = [0.0] * S
        for (kind, t, w), a in col.coef.items():
            pen[w] += duals.get((kind, t, w), 0.0) / probs[w] * a
        chi[u] = [-(col.scen[w] - pen[w] + col.first) for w in range(S)]
    v = [-c for c in ef['worth_cost']]
    g = [sum(chi[u][w] for u in players) - v[w] for w in range(S)]
    x = {u: [chi[u][w] - g[w] / n for w in range(S)] for u in players}
    E = lambda seq: float(sum(p * s for p, s in zip(probs, seq)))
    omega_lr = E(g)
    return {
        'chi_scen': chi, 'worth': v, 'g': g, 'x': x,
        'Ex': {u: E(x[u]) for u in players},
        'owen': {u: -dw['sigma'][u] for u in players},
        'omega_LR': omega_lr, 'eps_LR': omega_lr / n,
        # checks: both residuals should be ~0
        'budget_residual': max(abs(sum(x[u][w] for u in players) - v[w]) for w in range(S)),
        'duality_residual': omega_lr - (ef['obj'] - dw['obj']),
        'pricing_residual': {u: pricing_value[u] - dw['sigma'][u] for u in players},
        'terminal_commitment': {u: q[u][1].fs for u in players},
    }


def standalone_values(players, T, scenarios, **kw):
    """val(DP_{j}^Omega), profit convention: the scenario-expanded single-member problem."""
    return {u: -solve_extensive_form([u], T, scenarios, **kw)['obj'] for u in players}


def measure_eps(players, T, scenarios, payoff, **kw):
    """max_S (v(S) - sum_{j in S} E[x_j*]) / |S| over every proper coalition.

    By Lemma S(c) this is the weak-eps level of (x*, a_hat) in the stochastic game;
    Corollary eps says it is at most eps_LR. Enumerates 2^n - 2 extensive forms.
    """
    worst, arg, table = -np.inf, None, {}
    for r in range(1, len(players)):
        for S in itertools.combinations(players, r):
            vS = -solve_extensive_form(list(S), T, scenarios, **kw)['obj']
            e = (vS - sum(payoff[u] for u in S)) / len(S)
            table['+'.join(S)] = {'v': vS, 'excess': e}
            if e > worst:
                worst, arg = e, S
    return {'eps': worst, 'argmax': list(arg), 'coalitions': table}


# =============================================================================
# driver
# =============================================================================
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def run(args):
    if args.engine != 'direct':
        raise SystemExit(f"--engine {args.engine}: SCIP is not a supported solver here; use 'gurobi' or 'highs' (use --engine direct)")
    if args.doi:
        raise SystemExit('--doi exists only on the SCIP engine, which is no longer supported')
    sys.path.insert(0, os.path.join(_PAPER, 'weak_eps_experiment'))
    from run_experiment import build_instance
    players, _, T, base, name = build_instance(args.n)
    scen = make_scenarios(base, players, T, args.scenarios, seed=args.seed,
                          wind_sigma=args.wind_sigma, solar_sigma=args.wind_sigma,
                          load_sigma=args.load_sigma, price_sigma=args.price_sigma,
                          rho=args.rho,
                          price_carriers=tuple(args.price_carriers.split(',')),
                          load_carriers=tuple(args.load_carriers.split(',')))
    mip_kw = dict(time_limit=args.mip_time_limit, gap=args.mip_gap,
                  solver=args.mip_solver)
    tag = f'{name}_S{args.scenarios}_seed{args.seed}' + ('_doi' if args.doi else '') \
        + ('_sar' if args.sar else '') + ('_nosmooth' if args.no_smoothing else '') \
        + ('_grb' if args.pricing_solver == 'gurobi' else '') \
        + ('_nopen' if args.no_penalty else '') \
        + (f'_direct-{args.lp_solver}' if args.engine == 'direct' else '')         + (f'_{args.tag}' if args.tag else '')
    print(f'\n=== {tag}: n={len(players)}, |T|={len(T)}, |Omega|={len(scen)} ===')

    if args.deterministic_check:
        det = LocalEnergyMarket(players, T, scen[0][1], model_type='mip')
        det.model.hideOutput()
        det.solve()
        print(f'deterministic model on scenario 0: {det.model.getObjVal():.6f}')

    print('\n[1] extensive form (DP_N^Omega)')
    ef = solve_extensive_form(players, T, scen, **mip_kw)
    print(f'  obj {ef["obj"]:.6f}  status {ef["status"]}  gap {ef["gap"]:.2e}  '
          f'{ef["time_solve"]:.1f}s  vars {ef["stack"].model.getNVars()}')

    print('\n[2] column generation (DWR_N^Omega)')
    if args.engine == 'direct':
        solver_fn = solve_dwr_bundle if args.bundle else solve_dwr_direct
        extra_b = ({'bundle_t': args.bundle_t, 'bundle_age': args.bundle_age,
                    'bundle_t_min': args.bundle_t_min, 'bundle_cap': args.bundle_cap,
                    'bundle_qp': args.bundle_qp} if args.bundle else {})
        dw, master = solver_fn(
            players, T, scen, base,
            init_vals=None if args.cold_start else ef['vals'],
            lp_solver=args.lp_solver, pricing_solver=args.pricing_solver,
            pricing_time_limit=args.mip_time_limit,
            pricing_gap=args.pricing_gap,
            smoothing=not args.no_smoothing, incumbent=ef['obj'], gap_tol=args.cg_gap,
            pen_eps=0.0 if args.no_penalty else args.pen_eps,
            pen_delta=0.0 if args.no_penalty else args.pen_delta,
            pen_shrink=args.pen_shrink, max_rounds=args.max_rounds,
            pricing_workers=args.pricing_workers, round_tol=args.round_tol,
            ub_every=args.ub_every, lp_method=args.lp_method,
            purge_every=args.purge_every, purge_age=args.purge_age,
            purge_cap=args.purge_cap, lp_presolve=args.lp_presolve,
            sar=args.sar, sar_block=args.sar_block, sar_cap=args.sar_cap,
            sar_exact=tuple(k for k in args.sar_exact.split(',') if k),
            lazy_kinds=tuple(k for k in args.lazy_rows.split(',') if k), **extra_b)
        tm = dw['timing']
        if dw.get('bundle'):
            print(f'  bundle: {tm["serious"]} serious / {tm["null"]} null steps, '
                  f'QP {tm["qp"]:.1f}s, {tm["dropped"]} cuts dropped, '
                  f'{tm["aggregated"]} aggregated')
        print(f'  obj {dw["obj"]:.6f}  LB {dw["lb"]:.6f}  gap {dw["gap"]:.2e}  '
              f'iters {dw["iterations"]}  {dw["time"]:.1f}s  '
              f'rounds {len(dw["penalty"]["rounds"])}  status {dw["status"]}')
        if dw.get('lazy'):
            print(f'  lazy rows: {dw["lazy"]["added"]} added, master {dw["lazy"]["final_rows"]} '
                  f'of {dw["lazy"]["original_rows"]} linking rows')
        print(f'  time: master LP {tm["lp"]:.1f}s  pricing {tm["pricing"]:.1f}s '
              f'({tm["pricing_calls"]} MILPs, {tm["workers"]} worker(s))')
    elif args.sar:
        solver = solve_dwr_sar
        extra = {'sar_block': args.sar_block, 'sar_cap': args.sar_cap}
    elif not args.no_penalty:
        solver = solve_dwr_stab
        extra = {'pen_eps': args.pen_eps, 'pen_delta': args.pen_delta,
                 'pen_shrink': args.pen_shrink}
    else:
        solver = solve_dwr
        extra = {'doi': args.doi}
    if args.engine != 'direct':
        dw, master = solver(players, T, scen, base, **extra,
                            init_vals=None if args.cold_start else ef['vals'],
                            time_limit=args.cg_time_limit,
                            pricing_time_limit=args.mip_time_limit,
                            pricing_gap=args.mip_gap,
                            smoothing=not args.no_smoothing, incumbent=ef['obj'],
                            pricing_solver=args.pricing_solver, gap_tol=args.cg_gap)
        print(f'  obj {dw["obj"]:.6f}  LB {dw["lb"]:.6f}  iters {dw["iterations"]}  '
          f'{dw["time"]:.1f}s  DOI {dw["doi"]}  SAR {dw.get("sar", {}).get("phases")}  '
          f'PEN {dw.get("penalty", {}).get("rounds")}')

    print('\n[3] Algorithm S1')
    al = scenario_allocation(ef, dw, master)
    print(f'  omega_LR {al["omega_LR"]:.6f}  eps_LR {al["eps_LR"]:.6f}')
    print(f'  checks: budget {al["budget_residual"]:.2e}  duality '
          f'{al["duality_residual"]:.2e}  pricing '
          f'{max(abs(v) for v in al["pricing_residual"].values()):.2e}')

    if args.skip_standalone:
        alone = {u: float('nan') for u in players}
    else:
        print('\n[4] stand-alone values val(DP_{j}^Omega)')
        alone = standalone_values(players, T, scen, **mip_kw)
    chi_lr = {u: al['Ex'][u] - alone[u] for u in players}
    print(f'  {"player":>7} {"Owen":>12} {"E[x*]":>12} {"alone":>12} {"chi^LR":>12}')
    for u in players:
        print(f'  {u:>7} {al["owen"][u]:12.4f} {al["Ex"][u]:12.4f} '
              f'{alone[u]:12.4f} {chi_lr[u]:12.4f}')

    core = None
    if args.check_core:
        print(f'\n[5] weak eps over all {2 ** len(players) - 2} proper coalitions')
        core = measure_eps(players, T, scen, al['Ex'], **mip_kw)
        print(f'  eps measured {core["eps"]:.6f} at {core["argmax"]}  '
              f'(bound eps_LR {al["eps_LR"]:.6f})')

    first = ef['stack'].first_stage_values(ef['vals'])
    out = {
        'instance': name, 'n': len(players), 'T': len(T), 'scenarios': len(scen),
        'probs': master.probs,
        'config': {k: getattr(args, k) for k in ('seed', 'wind_sigma', 'load_sigma',
                                                   'price_sigma', 'rho', 'price_carriers',
                                                   'mip_gap', 'mip_time_limit')},
        'm_linking_rows': len(master.row_keys),
        'ef': {k: ef[k] for k in ('status', 'obj', 'dual_bound', 'gap', 'first_cost',
                                  'scen_cost', 'worth_cost', 'time_build', 'time_solve')},
        'ef_first_stage': {k: v for k, v in first.items() if abs(v) > 1e-9},
        'dw': {k: dw.get(k) for k in ('status', 'obj', 'lb', 'ub', 'gap', 'sigma', 'x0',
                                      'iterations', 'columns', 'time', 'doi', 'timing')},
        'cg_options': {k: getattr(args, k) for k in (
            'lp_solver', 'pricing_solver', 'mip_solver', 'cg_gap', 'pricing_gap',
            'no_penalty', 'no_smoothing', 'pen_eps', 'pen_delta', 'pen_shrink',
            'max_rounds', 'pricing_workers', 'round_tol', 'ub_every', 'lp_method',
            'purge_every', 'purge_age', 'purge_cap', 'lp_presolve', 'cold_start',
            'bundle', 'bundle_t', 'bundle_t_min', 'bundle_age', 'bundle_cap', 'bundle_qp')},
        'sar': dw.get('sar'),
        'penalty': dw.get('penalty'),
        'cg_log': master.log,
        'allocation': {k: al[k] for k in ('owen', 'Ex', 'x', 'g', 'worth', 'omega_LR',
                                          'eps_LR', 'budget_residual',
                                          'duality_residual', 'pricing_residual')},
        'standalone': alone, 'chi_LR': chi_lr,
        'weak_eps': core,
        'coupling_prices': {f'{k}_{t}_s{w}': d / master.probs[w]
                            for (k, t, w), d in dw['duals'].items()},
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'{tag}.json')
    with open(path, 'w') as f:
        json.dump(_jsonable(out), f, indent=1)
    try:
        print(f'\nwrote {os.path.relpath(path, _ROOT)}')
    except ValueError:              # another drive on Windows
        print(f'\nwrote {path}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=6, help='community size (6, 15, 30, 60)')
    ap.add_argument('--scenarios', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wind-sigma', type=float, default=0.25,
                    help='renewable forecast error (std of the multiplicative factor)')
    ap.add_argument('--load-sigma', type=float, default=0.10)
    ap.add_argument('--price-sigma', type=float, default=0.15)
    ap.add_argument('--rho', type=float, default=0.7, help='hour-to-hour error correlation')
    ap.add_argument('--load-carriers', default='E,H,G',
                    help='carriers whose non-flexible load is uncertain')
    ap.add_argument('--price-carriers', default='E',
                    help='carriers whose prices are uncertain, e.g. E or E,H,G')
    ap.add_argument('--mip-gap', type=float, default=EF_GAP,
                    help=f'relative gap of the extensive form and stand-alone MILPs '
                         f'(default {EF_GAP})')
    ap.add_argument('--mip-time-limit', type=float, default=None)
    ap.add_argument('--mip-solver', default='highs', choices=['highs', 'gurobi'],
                    help='extensive form and stand-alone MILPs (DP_S^Omega)')
    ap.add_argument('--cg-time-limit', type=float, default=None)
    ap.add_argument('--cold-start', action='store_true',
                    help='seed the master from zero-dual pricing instead of the EF solution')
    ap.add_argument('--doi', action='store_true',
                    help='dual-optimal inequalities on the balance rows (market price box)')
    ap.add_argument('--engine', default='direct', choices=['scip', 'direct'],
                    help="'direct': our own loop on a HiGHS or Gurobi LP; 'scip' "
                         '(SCIP master with its pricer plugin) is no longer supported')
    ap.add_argument('--lp-solver', default='highs', choices=['highs', 'gurobi'],
                    help='master LP solver for --engine direct')
    ap.add_argument('--pricing-solver', default='highs', choices=['highs', 'gurobi'])
    ap.add_argument('--pricing-gap', type=float, default=MIP_GAP,
                    help=f'relative gap of the pricing MILPs (default {MIP_GAP})')
    ap.add_argument('--pricing-workers', type=int, default=0,
                    help='prosumers priced in parallel (0: one per core)')
    ap.add_argument('--round-tol', type=float, default=None,
                    help='column admission tolerance of the penalty rounds before the '
                         'last (default: --cg-gap)')
    ap.add_argument('--max-rounds', type=int, default=12)
    ap.add_argument('--tag', default='', help='suffix for the output file name')
    ap.add_argument('--cg-gap', type=float, default=CG_GAP,
                    help='relative CG gap: stop once (UB - LB) <= cg_gap (1 + |UB|), UB the '
                         'unpenalized RMP value and LB the Lagrangian bound')
    ap.add_argument('--purge-every', type=int, default=10,
                    help='column management: purge every k iterations (0: never)')
    ap.add_argument('--purge-age', type=int, default=50,
                    help='iterations a column may sit unused before it can be purged')
    ap.add_argument('--purge-cap', type=int, default=40,
                    help='purge only while there are more than this many columns per prosumer')
    ap.add_argument('--lp-presolve', default='auto', choices=['auto', 'off'],
                    help='presolve of the master LP')
    ap.add_argument('--lp-method', default='primal',
                    choices=['primal', 'dual', 'auto', 'barrier'],
                    help='simplex variant of the master LP')
    ap.add_argument('--ub-every', type=int, default=30,
                    help='iterations between upper-bound checks inside a penalty round')
    ap.add_argument('--no-penalty', action='store_true',
                    help='smoothing only, without the three-piece dual penalty')
    ap.add_argument('--pen-eps', type=float, default=0.2,
                    help='penalty-free band, relative to |center| per row')
    ap.add_argument('--pen-delta', type=float, default=0.2,
                    help='slack bound per row, i.e. the penalty slope in the dual')
    ap.add_argument('--pen-shrink', type=float, default=0.25)
    ap.add_argument('--sar', action='store_true',
                    help='dyn-SAR: start from aggregated (block, scenario) rows')
    ap.add_argument('--sar-block', type=int, default=1,
                    help='hours per aggregated row in the first dyn-SAR phase '
                         '(6 with --sar-cap 0.05 is the policy of Costa et al.)')
    ap.add_argument('--bundle', action='store_true',
                    help='proximal bundle method on the Lagrangian dual instead of the LP '
                         'master (Gurobi only)')
    ap.add_argument('--bundle-t', type=float, default=10.0,
                    help='initial proximal parameter t of the bundle method')
    ap.add_argument('--bundle-cap', type=int, default=20,
                    help='cuts per prosumer before the bundle is compressed')
    ap.add_argument('--bundle-qp', default='primal', choices=['primal', 'dual', 'barrier'],
                    help='algorithm for the bundle QP')
    ap.add_argument('--bundle-t-min', type=float, default=1e-2,
                    help='floor of the proximal parameter t')
    ap.add_argument('--bundle-age', type=int, default=10,
                    help='iterations a cut may stay inactive before the bundle drops it')
    ap.add_argument('--lazy-rows', default='',
                    help='inequality row kinds added only when violated, e.g. peak,dn')
    ap.add_argument('--sar-exact', default='',
                    help='dyn-SAR: row kinds kept per hour and scenario from the start, '
                         'e.g. E (the carrier whose price is uncertain)')
    ap.add_argument('--sar-cap', type=float, default=1.0,
                    help='rows added per dyn-SAR round, as a share of the original rows')
    ap.add_argument('--skip-standalone', action='store_true',
                    help='stop after Algorithm S1 (no stand-alone solves)')
    ap.add_argument('--no-smoothing', action='store_true',
                    help='plain Kelley column generation (no Wentges dual smoothing)')
    ap.add_argument('--check-core', action='store_true',
                    help='enumerate every coalition (small n only)')
    ap.add_argument('--deterministic-check', action='store_true',
                    help='also solve the plain model on scenario 0')
    ap.add_argument('--out', default=OUT)
    run(ap.parse_args())


if __name__ == '__main__':
    main()

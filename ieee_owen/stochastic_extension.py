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
                   price_carriers=('E',)):
    """Forecast-error scenarios around one deterministic instance, equiprobable.

    Every scenario is a copy of `base` with multiplicative AR(1) errors on
      renewable availability  renewable_cap_{u}_{t}   one path for wind, one for solar
                              (common weather), clipped to [0, the day's peak]
      non-flexible load       d_{k}_nfl_{u}_{t}        one path per carrier
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
    if gap is not None:
        m.setParam('limits/gap', float(gap))


# =============================================================================
# Extensive form
# =============================================================================
def solve_extensive_form(players, T, scenarios, time_limit=None, gap=None, quiet=True):
    """Solve (DP_S^Omega). Returns a dict; objective in the cost convention."""
    t0 = time.time()
    st = ScenarioStack('DP_Omega', players, T, scenarios, dwr=False)
    build = time.time() - t0
    m = st.model
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


def _to_gurobi(scip_model, name, time_limit=None, gap=None, threads=1):
    """Copy a pyscipopt model that has only linear constraints into gurobipy.

    Built from the constraint data rather than through an MPS file: LocalEnergyMarket
    reuses some constraint names, which a file round trip would have to rename.
    Variables keep their names, which is how the two models are matched.
    """
    import gurobipy as gp
    from gurobipy import GRB
    inf = scip_model.infinity()
    g = gp.Model(name)
    g.Params.OutputFlag = 0
    g.Params.Threads = threads
    g.Params.MIPGap = 0.0 if gap is None else gap
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


class PlayerPricing:
    """eq:sup_vlrj for one prosumer: a two-stage stochastic MILP of its own.

    solver='scip' solves the stacked SCIP model directly; solver='gurobi' solves a
    gurobipy copy of it (same variables, same rows), changing only the objective
    between calls.
    """
    def __init__(self, player, T, scenarios, time_limit=None, gap=None, solver='scip'):
        self.player = player
        self.stack = ScenarioStack(f'price_{player}', [player], T, scenarios, dwr=True)
        self.model = self.stack.model
        _set_mip_params(self.model, time_limit, gap)
        self.solver = solver
        if solver == 'gurobi':
            self.g, self.gv = _to_gurobi(self.model, f'price_{player}', time_limit, gap)
            self.g_names = list(self.gv)
            self.g_vars = [self.gv[n] for n in self.g_names]
        elif solver != 'scip':
            raise ValueError(f"pricing solver must be 'scip' or 'gurobi', got {solver!r}")
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

    def price(self, duals, farkas=False):
        """min_x c(x) - pi^T A_j x over X_j. Returns (objective, dual bound, Column)."""
        if self.solver == 'gurobi':
            obj, bound, vals = self._price_gurobi(duals, farkas)
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
        g.optimize()
        if g.SolCount == 0:
            raise RuntimeError(f'pricing {self.player} (gurobi): no solution, status {g.Status}')
        x = g.getAttr('X', self.g_vars)
        vals = dict(zip(self.g_names, x))
        # the objective is recomputed from the solution, so it and the column cost
        # agree to the last digit whatever Gurobi reports internally
        obj = sum(c * vals[n] for n, c in coef.items() if c)
        bound = min(g.ObjBound, obj)
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
                 pricing_solver='scip', gap_tol=1e-6, verbose=True):
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
    sys.path.insert(0, os.path.join(_PAPER, 'weak_eps_experiment'))
    from run_experiment import build_instance
    players, _, T, base, name = build_instance(args.n)
    scen = make_scenarios(base, players, T, args.scenarios, seed=args.seed,
                          wind_sigma=args.wind_sigma, solar_sigma=args.wind_sigma,
                          load_sigma=args.load_sigma, price_sigma=args.price_sigma,
                          rho=args.rho,
                          price_carriers=tuple(args.price_carriers.split(',')))
    mip_kw = dict(time_limit=args.mip_time_limit, gap=args.mip_gap)
    tag = f'{name}_S{args.scenarios}_seed{args.seed}' + ('_doi' if args.doi else '') \
        + ('_sar' if args.sar else '') + ('_nosmooth' if args.no_smoothing else '') \
        + ('_grb' if args.pricing_solver == 'gurobi' else '')
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
    solver = solve_dwr_sar if args.sar else solve_dwr
    extra = ({'sar_block': args.sar_block, 'sar_cap': args.sar_cap} if args.sar
             else {'doi': args.doi})
    dw, master = solver(players, T, scen, base, **extra,
                           init_vals=None if args.cold_start else ef['vals'],
                           time_limit=args.cg_time_limit,
                           pricing_time_limit=args.mip_time_limit,
                           pricing_gap=args.mip_gap,
                           smoothing=not args.no_smoothing, incumbent=ef['obj'],
                           pricing_solver=args.pricing_solver, gap_tol=args.cg_gap)
    print(f'  obj {dw["obj"]:.6f}  LB {dw["lb"]:.6f}  iters {dw["iterations"]}  '
          f'{dw["time"]:.1f}s  DOI {dw["doi"]}  SAR {dw.get("sar", {}).get("phases")}')

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
        'dw': {k: dw[k] for k in ('status', 'obj', 'lb', 'sigma', 'x0', 'iterations',
                                  'columns', 'time', 'doi')},
        'sar': dw.get('sar'),
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
    print(f'\nwrote {os.path.relpath(path, _ROOT)}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=6, help='community size (6, 15, 30)')
    ap.add_argument('--scenarios', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wind-sigma', type=float, default=0.25,
                    help='renewable forecast error (std of the multiplicative factor)')
    ap.add_argument('--load-sigma', type=float, default=0.10)
    ap.add_argument('--price-sigma', type=float, default=0.15)
    ap.add_argument('--rho', type=float, default=0.7, help='hour-to-hour error correlation')
    ap.add_argument('--price-carriers', default='E',
                    help='carriers whose prices are uncertain, e.g. E or E,H,G')
    ap.add_argument('--mip-gap', type=float, default=None,
                    help='relative gap for every MILP (SCIP default 0)')
    ap.add_argument('--mip-time-limit', type=float, default=None)
    ap.add_argument('--cg-time-limit', type=float, default=None)
    ap.add_argument('--cold-start', action='store_true',
                    help='seed the master from zero-dual pricing instead of the EF solution')
    ap.add_argument('--doi', action='store_true',
                    help='dual-optimal inequalities on the balance rows (market price box)')
    ap.add_argument('--pricing-solver', default='scip', choices=['scip', 'gurobi'])
    ap.add_argument('--cg-gap', type=float, default=1e-6,
                    help='relative CG tolerance (column admission and LB >= RMP stop)')
    ap.add_argument('--sar', action='store_true',
                    help='dyn-SAR: start from aggregated (block, scenario) rows')
    ap.add_argument('--sar-block', type=int, default=1,
                    help='hours per aggregated row in the first dyn-SAR phase '
                         '(6 with --sar-cap 0.05 is the policy of Costa et al.)')
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

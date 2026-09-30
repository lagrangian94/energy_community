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

DEFAULTS of the direct engine, and why (measured in ieee_owen/cg_scaling.md): the
balance-row DOIs are on -- the community may trade with the grid at market prices
inside the master, which breaks the master's primal degeneracy; a bound counts only
with no grid trade, and if the master settles with some they are switched off --
and cut n=60, |Omega|=5 from 1443-1716 s to 299-313 s with the same omega^LR. Pricing
is parallel, heavy prosumers dealt out first; columns are purged; gaps are EF 1e-6,
CG 1e-6 or omega^LR to 2% (OMEGA_TOL), whichever comes first, with the pricing MILPs
on the absolute gap that needs.
Nested DW (default): a prosumer without first-stage variables is priced one scenario
at a time (one column and convexity row per prosumer and scenario, --split-scenarios);
an electrolyzer owner's plan enters as a commitment-pattern column plus one column per
scenario, tied by sum lambda = mu (MP1-2 of Maher & Muter, --mp12). conv X_j and so
v^LR are unchanged, non-anticipativity is kept; columns are 1/|Omega| as dense and
purged harder (cap 5, age 20). KL, n=60, 20 scenarios: CG 6,859 s -> 717 s.
Every combination returns the same numbers -- checked at |Omega| = 1, where all four
gave the same v^CHP and Owen allocation to four decimals (old 6p instance).

ONE SCENARIO = THE DETERMINISTIC MODEL. solve_deterministic runs the paper's
deterministic instance through this engine, and run_multiday uses it in place of
chp.ColumnGenerationSolver. Checked 2026-09-30 at 6/15/30/60p, days 1-2: the same
v^LR to CG_GAP and the same Owen point up to the degeneracy of the master duals, at
5-10x less CG time -- with omega_tol off; the 2% early stop moved omega^LR and the
Owen point by up to 2%.

KL DRO (--kl-radius r, ieee_owen/robust_core.md sec. 2.3). Worst expected cost over
KL(rho || rho_hat) <= r, on the same engine and defaults. The master (KLMaster) moves
every cost into distribution cuts, one per tilted distribution found by kl_worst in
closed form (no exponential cone); pricing is the same MILP at the cut mixture rho_bar;
LB is the Lagrangian at (rho_bar, pi), UB the worst case of the unpenalized RMP plan.
The extensive form adds the same cuts as Gurobi lazy constraints. r = 0 reproduces
the stochastic model (n=6, |Omega|=3: omega 7.118, certified [6.994, 7.125] around
the exact 7.0056); at r = 0.5 the measured weak eps over all 62 coalitions is -1.97
against the bound omega/n = 0.85.

Usage (from anywhere):
  python ieee_owen/stochastic_extension.py --n 6 --scenarios 5
  python ieee_owen/stochastic_extension.py --n 6 --scenarios 3 --check-core
  python ieee_owen/stochastic_extension.py --n 6 --scenarios 1 --wind-sigma 0 \
      --solar-sigma 0 --load-sigma 0 --price-sigma 0          # reproduces the deterministic model
"""
import os, sys, json, time, argparse, itertools, functools
# layout: <repo root>/ieee_owen/. Shared model modules and data/ live at the root;
# run_experiment (the instance builder) lives in weak_eps_experiment/.
_PAPER = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum
from compact_utility import (LocalEnergyMarket, reserve_blocks, block_reserve_payment,
                             reserve_mode, reserve_penalty_price)

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
# EF_GAP is 1e-4 since 2026-09-29 (it was 1e-6). At n=60, 20 scenarios and hourly
# FCR-N reserve the KL EF found its incumbent in ~10 min and then spent the next hour
# moving the bound from 0.04% to 0.02%, and the manuscript does not use omega to
# more than a few percent. The cost is noise in omega: an absolute ~2 on either side
# (see above). Pass --mip-gap 1e-6 where omega itself is the object of a comparison.
MIP_GAP = 1e-4
EF_GAP = 1e-4
CG_GAP = 1e-6
# OMEGA_TOL: column generation also stops once omega^LR is known to this relative
# precision (UB - LB <= OMEGA_TOL * (EF value - LB)); pricing then runs on the
# absolute gap that precision needs. 19-28% faster at 10-20 scenarios, omega 1.5-1.9%
# high (ieee_owen/cg_scaling.md, section 11). --no-omega-tol restores CG_GAP alone.
OMEGA_TOL = 0.02

# Names LocalEnergyMarket gives the first-stage variables (f"{prefix}{u}_{t}" and
# f"r_sym_{i}"). Heat-pump commitment is deliberately absent: it is redispatched.
FIRST_STAGE_PREFIXES = ('z_on_G_', 'z_off_G_', 'z_sb_G_', 'z_su_G_', 'z_sd_G_',
                        'r_sym_', 'r_up_', 'r_dn_')
CARRIERS = ('E', 'H', 'G')


def rho_key(w):
    """Key of the KL master's pricing measure rho_bar_w inside a duals dict."""
    return ('rho', 0, w)


def kl_worst(c, probs, radius):
    """psi(c) = max {rho.c : KL(rho || probs) <= radius}, the worst expected cost.

    The maximizer tilts the reference, rho_w ~ probs_w exp(c_w / eta), with eta set by
    KL = radius (bisection in log eta; KL falls monotonically in eta). If the radius
    reaches -ln probs(argmax c), rho is probs conditioned on the argmax. Returns
    (value, rho, eta): value = eta ln E exp(c/eta) + eta r is the dual bound at eta,
    so it is >= psi(c), and rho is taken on the feasible side, KL(rho) <= radius, so
    it lies in the ball. Both are exact to the last digits after the bisection.
    """
    c, p = np.asarray(c, float), np.asarray(probs, float)
    if radius <= 0.0:
        return float(p @ c), p.copy(), np.inf
    cmax = c.max()
    top = c >= cmax - 1e-12 * (1.0 + abs(cmax))
    ptop = p[top].sum()
    if -np.log(ptop) <= radius:
        return float(cmax), np.where(top, p, 0.0) / ptop, 0.0

    def tilt(eta):
        z = (c - cmax) / eta
        wt = p * np.exp(z)
        s = wt.sum()
        rho = wt / s
        kl = float(np.sum(rho[rho > 0] * (z[rho > 0] - np.log(s))))
        return rho, kl, s

    spread = cmax - c.min()
    lo, hi = np.log(spread * 1e-8), np.log(spread * 1e8)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tilt(np.exp(mid))[1] > radius:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    eta = float(np.exp(hi))
    rho, _, s = tilt(eta)
    return float(cmax + eta * np.log(s) + eta * radius), rho, eta


def kl_div(rho, probs):
    rho, p = np.asarray(rho, float), np.asarray(probs, float)
    m = rho > 0
    return float(np.sum(rho[m] * np.log(rho[m] / p[m])))


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
        if name.startswith(FIRST_STAGE_PREFIXES) or name in st.extra_first_stage:
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

    block: build each scenario with block(params, model) instead of LocalEnergyMarket
    (e.g. a core.SeparationProblem, see stochastic_core.py); it must build into
    `model`. first_stage_names: exact variable names that are first stage on top of
    FIRST_STAGE_PREFIXES (e.g. the separation's selection binaries z_j): created once,
    cost unscaled, shared by every scenario.
    """
    def __init__(self, name, players, T, scenarios, dwr, model_type='mip', block=None,
                 first_stage_names=()):
        self.players, self.T, self.scenarios = list(players), list(T), scenarios
        self.probs = [p for p, _ in scenarios]
        self.model = Model(name)
        self.vars = {}          # name -> var
        self.cost = {}          # name -> unscaled cost
        self.scen_of = {}       # name -> scenario index, None if first stage
        self.first_stage = set()
        self.extra_first_stage = frozenset(first_stage_names)
        self.first_stage_cons = {}
        self.blocks = []
        for w, (prob, params) in enumerate(scenarios):
            if block is None:
                self.blocks.append(LocalEnergyMarket(self.players, self.T, params,
                                                     model_type=model_type, dwr=dwr,
                                                     model=_Block(self, w, prob)))
            else:
                self.blocks.append(block(params, _Block(self, w, prob)))

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
                         solver='highs', kl_radius=None, log_file=None):
    """Solve (DP_S^Omega). Returns a dict; objective in the cost convention.

    solver='highs' / 'gurobi' build the same SCIP model and solve a highspy /
    gurobipy copy of it. kl_radius > 0 solves the KL-robust version instead,
    min_x max_{KL(rho || rho_hat) <= r} E_rho[cost(x)] (Gurobi only).
    log_file: with Gurobi, write its progress log (bound, incumbent, gap) there,
    appended across the KL refinement rounds; the console stays quiet.
    """
    if solver not in ('highs', 'gurobi'):
        raise ValueError(f"MIP solver {solver!r}: SCIP is not a supported solver here; use 'gurobi' or 'highs'")
    gap = EF_GAP if gap is None else gap
    t0 = time.time()
    st = ScenarioStack('DP_Omega', players, T, scenarios, dwr=False)
    build = time.time() - t0
    m = st.model
    if kl_radius is not None:
        if solver != 'gurobi':
            raise ValueError('the KL-robust extensive form needs --mip-solver gurobi '
                             '(lazy constraints)')
        return _solve_extensive_kl(st, build, time_limit, gap, quiet, kl_radius,
                                   log_file=log_file)
    if solver == 'gurobi':
        return _solve_extensive_gurobi(st, build, time_limit, gap, quiet, log_file=log_file)
    if solver == 'highs':
        return _solve_extensive_highs(st, build, time_limit, gap, quiet)
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
    __slots__ = ('player', 'cost', 'coef', 'first', 'scen', 'fs', 'extra', 'noconv', 'trade')

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
    def total(cls, player, cols):
        """The sum of a prosumer's per-scenario columns (scenario-split pricing):
        one plan over every scenario, as the unsplit pricing would return it."""
        new = cls.__new__(cls)
        new.player = player
        new.cost = sum(c.cost for c in cols)
        new.coef = {}
        for c in cols:
            for k, a in c.coef.items():
                new.coef[k] = new.coef.get(k, 0.0) + a
        new.first = sum(c.first for c in cols)
        new.scen = [sum(c.scen[i] for c in cols) for i in range(len(cols[0].scen))]
        new.fs = {}
        return new

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
            hint = ("; this engine copies rows one by one and takes linear rows only, "
                    "so use complementarity='bigm' (it measured the same as 'sos')"
                    if c.getConshdlrName() in ('SOS1', 'SOS2') else '')
            raise ValueError(f'constraint {c.name} is {c.getConshdlrName()}, not linear{hint}')
        expr = gp.LinExpr([(a, gv[n]) for n, a in scip_model.getValsLinear(c).items()])
        lhs, rhs = scip_model.getLhs(c), scip_model.getRhs(c)
        if lhs > -inf and rhs < inf and lhs == rhs:
            g.addLConstr(expr, GRB.EQUAL, rhs, name=c.name)
        else:
            if lhs > -inf:
                g.addLConstr(expr, GRB.GREATER_EQUAL, lhs,
                             name=c.name if rhs >= inf else f'{c.name}__lo')
            if rhs < inf:
                g.addLConstr(expr, GRB.LESS_EQUAL, rhs,
                             name=c.name if lhs <= -inf else f'{c.name}__hi')
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


def _gurobi_log(g, quiet, log_file):
    """Console output per `quiet`; with a log file, the full log goes there only."""
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        g.Params.OutputFlag = 1
        g.Params.LogToConsole = 0 if quiet else 1
        g.Params.LogFile = log_file
    else:
        g.Params.OutputFlag = 0 if quiet else 1


def _solve_extensive_gurobi(st, build, time_limit, gap, quiet, log_file=None):
    """solve_extensive_form on a gurobipy copy of the stacked SCIP model."""
    m = st.model
    if m.getObjectiveSense() != 'minimize':
        raise ValueError('extensive form is expected to minimise')
    g, gv = _to_gurobi(m, 'DP_Omega', time_limit, gap)
    _gurobi_log(g, quiet, log_file)
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


def _kl_tangent(s):
    """Tangent of g(h, mu, lam) = lam (exp((h - mu)/lam) - 1) at the ratio s:
    g >= a (h - mu) + b lam with a = e^s, b = e^s - 1 - s e^s, for every h, mu and
    lam >= 0 (g is the perspective of the convex e^s - 1, hence jointly convex)."""
    a = float(np.exp(s))
    return a, a - 1.0 - s * a


def _solve_extensive_kl(st, build, time_limit, gap, quiet, radius, max_rounds=30,
                        log_file=None, round_gap=1e-3, prepare=None, budget=None):
    """The KL-robust extensive form through the dual of the inner max
    (Love & Bayraksan, phi-divergence constrained two-stage programs, eq. 9):

        psi(h) = max_{KL(p || q) <= r} p.h
               = min_{lam >= 0, mu} mu + r lam + sum_w q_w lam (exp((h_w - mu)/lam) - 1)

    holds for any cost vector h, so lam and mu join the MILP's variables and the
    integers (first stage and recourse alike) stay with the MILP solver. What is not
    linear is each scenario's perspective term; it becomes a variable t_w held up by
    tangent planes (_kl_tangent), valid everywhere, so they go in as ordinary rows
    from the start: no callback, full presolve, and the root bound already sees every
    scenario (their multicut, laid down in advance). h_w is one variable per
    scenario, H_w = cost_w(x), so each tangent row has four entries.

    The tangents only underestimate, so Gurobi's bound stays a bound on the robust
    value. The incumbent's true worst case psi(h) (kl_worst) is the reported value;
    while it exceeds the bound by more than the gap, the exact tangents at the
    incumbent's ratios s_w = ln(p_w / q_w) are added and the MILP re-solved from the
    incumbent (with its exact lam, mu, t as the start).

    Each round is a full MILP solve, and the 0.25 grid alone leaves the incumbent
    ~0.03% short of psi at n=60, |Omega|=20, so a round solved to the final gap is
    usually followed by another (day 1: 80 min to 1e-4, then a restart from 0 nodes).
    Rounds therefore run at round_gap (>= gap) while the tangents at the incumbent are
    still off by more than gap; once they are exact to gap, the MIPGap drops to gap
    for the closing round(s).

    Replaces a first version that added distribution cuts theta >= p.h as lazy
    constraints at incumbents only: at n=60 it took 304 s for |Omega|=5 (stochastic
    EF: ~10 s) and had not finished |Omega|=10 after 100 minutes.

    prepare(g, gv): called on the gurobipy copy before the first round (extra rows or
    bounds, e.g. stochastic_core's KL separation fixing or bounding its z). budget:
    wall-clock seconds for ALL rounds together (time_limit caps each round); the
    last round's ObjBound is still a valid bound when it runs out.
    """
    import gurobipy as gp
    from gurobipy import GRB
    m = st.model
    if m.getObjectiveSense() != 'minimize':
        raise ValueError('extensive form is expected to minimise')
    g, gv = _to_gurobi(m, 'DP_Omega_KL', time_limit, gap)
    _gurobi_log(g, quiet, log_file)
    gap = g.Params.MIPGap
    q = np.array(st.probs)
    S = len(q)
    names = [n for n in st.vars if st.cost[n] != 0.0]
    gvars = [gv[n] for n in names]
    cost = np.array([st.cost[n] for n in names])
    sidx = np.array([-1 if st.scen_of[n] is None else st.scen_of[n] for n in names])
    first = sidx < 0

    def scen_costs(x):
        cx = cost * x
        return cx[first].sum() + np.bincount(sidx[~first], weights=cx[~first], minlength=S)

    H = [g.addVar(lb=-GRB.INFINITY, name=f'kl_h_{w}') for w in range(S)]
    t = [g.addVar(lb=-GRB.INFINITY, name=f'kl_t_{w}') for w in range(S)]
    lam = g.addVar(lb=0.0, name='kl_lambda')
    mu = g.addVar(lb=-GRB.INFINITY, name='kl_mu')
    for w in range(S):
        on = first | (sidx == w)
        g.addLConstr(gp.LinExpr(cost[on].tolist(), [v for v, o in zip(gvars, on) if o])
                     - H[w], GRB.EQUAL, 0.0, name=f'kl_cost_{w}')
    for v in g.getVars():
        v.Obj = 0.0
    g.setObjective(mu + radius * lam + gp.quicksum(q[w] * t[w] for w in range(S))
                   + m.getObjoffset(), GRB.MINIMIZE)

    def add_tangents(w, ss):
        for s in ss:
            a, b = _kl_tangent(s)
            g.addLConstr(t[w] - a * H[w] + a * mu - b * lam, GRB.GREATER_EQUAL, 0.0)

    # the grid: at the optimum s_w = ln(p*_w / q_w), so it covers ratios from a nearly
    # suppressed scenario (p/q = e^-8) to all the mass on one (p/q = 1/q_w); spacing
    # 0.25 leaves an error below 0.8% of each term, which the refinement removes
    for w in range(S):
        add_tangents(w, np.arange(-8.0, np.log(1.0 / q[w]) + 0.25, 0.25))
        g.addLConstr(t[w] + lam, GRB.GREATER_EQUAL, 0.0)      # the asymptote s -> -inf

    if prepare is not None:
        prepare(g, gv)
    rounds, t0 = [], time.time()
    loose = max(gap, round_gap)
    for k in range(max_rounds):
        g.Params.MIPGap = loose
        if budget is not None:
            left = max(1.0, budget - (time.time() - t0))
            g.Params.TimeLimit = min(left, time_limit) if time_limit else left
        g.optimize()
        if g.SolCount == 0:
            raise RuntimeError(f'KL extensive form ({len(st.players)} players): no '
                               f'solution, status {g.Status}')
        x = np.array(g.getAttr('X', gvars))
        c = scen_costs(x)
        psi, rho, eta = kl_worst(c, q, radius)
        psi += m.getObjoffset()
        bound = g.ObjBound
        rounds.append({'obj_model': g.ObjVal, 'bound': bound, 'psi': psi,
                       'time': g.Runtime, 'nodes': g.NodeCount, 'mip_gap': loose})
        tol = gap * max(1.0, abs(psi))
        if psi - bound <= tol:
            break
        if g.Status == GRB.TIME_LIMIT:
            break
        # tangents exact at the incumbent: what is left is the MIP gap, so close it
        if psi - g.ObjVal <= tol:
            loose = gap
        # exact tangents at the incumbent's own ratios, then restart from it
        pos = rho > 0
        for w in np.nonzero(pos)[0]:
            add_tangents(w, [np.log(rho[w] / q[w])])
        start = dict(zip(gvars, x))
        if not np.isfinite(eta):
            # radius 0 (kl_worst returns eta = inf): the dual optimum has lam -> inf,
            # which no start can carry; the s = 0 tangents are exact there already, so
            # the round only has to close the MIP gap. Start from x alone.
            pass
        elif eta > 0:
            m_hat = c.max() + eta * np.log(q @ np.exp((c - c.max()) / eta))
            start.update({lam: eta, mu: m_hat})
            start.update({t[w]: eta * (np.exp((c[w] - m_hat) / eta) - 1.0) for w in range(S)})
        else:                       # all the mass on the costliest scenarios
            start.update({lam: 0.0, mu: c.max()})
            start.update({t[w]: 0.0 for w in range(S)})
        start.update({H[w]: c[w] for w in range(S)})
        for v, val in start.items():
            v.Start = val
    allv = [n for n in gv]
    vals = dict(zip(allv, g.getAttr('X', [gv[n] for n in allv])))
    vals = {n: vals[n] for n in st.vars}
    fc, per = st.cost_split(vals)
    bound = min(bound, psi)
    return {
        'stack': st, 'status': {2: 'optimal', 9: 'timelimit'}.get(g.Status, str(g.Status)),
        'obj': psi, 'dual_bound': bound, 'gap': (psi - bound) / max(abs(psi), 1e-10),
        'vals': vals, 'first_cost': fc, 'scen_cost': per,
        'worth_cost': [fc + x for x in per],
        'time_build': build, 'time_solve': time.time() - t0,
        'kl': {'radius': radius, 'rho': rho.tolist(), 'eta': eta,
               'kl': kl_div(rho, q), 'expected_cost': float(q @ c) + m.getObjoffset(),
               'solves': len(rounds), 'refinements': rounds, 'lambda': lam.X, 'mu': mu.X},
    }


class PlayerPricing:
    """eq:sup_vlrj for one prosumer: a two-stage stochastic MILP of its own.

    solver='scip' solves the stacked SCIP model directly; 'gurobi' and 'highs'
    solve a copy of it (same variables, same rows) built through gurobipy or
    highspy, changing only the objective between calls.
    """
    def __init__(self, player, T, scenarios, time_limit=None, gap=None, solver='highs',
                 env=None, scen_ids=None, n_scen=None, unit=None):
        if solver not in ('highs', 'gurobi'):
            raise ValueError(f"pricing solver {solver!r}: SCIP is not a supported solver here; use 'gurobi' or 'highs'")
        self.player = player
        # scenario-split pricing: this stack holds the scenarios scen_ids of the
        # n_scen of the master; rows and costs are mapped back to those indices
        self.unit = player if unit is None else unit
        self.scen_ids = list(range(len(scenarios))) if scen_ids is None else list(scen_ids)
        self.n_scen = len(scenarios) if n_scen is None else n_scen
        self.stack = ScenarioStack(f'price_{player}', [player], T, scenarios, dwr=True)
        if scen_ids is not None and self.stack.first_stage:
            raise ValueError(f'{player} has first-stage variables: its scenarios cannot be '
                             'priced apart')
        self.model = self.stack.model
        _set_mip_params(self.model, time_limit, gap)
        self.solver = solver
        self.gap = MIP_GAP if gap is None else gap
        self.time, self.calls = 0.0, 0
        self.mip_start, self._last_x = False, None
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
        self.rows = {(k, t, self.scen_ids[w]): terms
                     for (k, t, w), terms in self.stack.link_terms(player).items()}
        self.names = list(self.stack.vars)
        self.base = {n: self.stack.scaled_cost(n) for n in self.names}
        # unscaled costs, for a pricing measure other than rho_hat (KL master)
        self._cw = [(n, self.stack.cost[n], self.stack.scen_of[n]) for n in self.names]
        self.adj = sorted({n for terms in self.rows.values() for n, _ in terms})
        self.last = None        # (duals, obj, Column) of the latest call
        # the member's four trade variables per balance row (k, t, w), in the order
        # (market import, market export, community import, community export), name
        # or None where the member has no such variable, and their upper bounds
        # (inf: unbounded). DirectMaster's grid certificate reads them.
        self.trade_vars, self.trade_ub = {}, {}
        for w, lem in enumerate(self.stack.blocks):
            for t in self.stack.T:
                for k in CARRIERS:
                    vs = [getattr(lem, f'{a}_{k}_{b}').get((player, t))
                          for a, b in (('i', 'gri'), ('e', 'gri'), ('i', 'com'), ('e', 'com'))]
                    if any(v is not None for v in vs):
                        key = (k, t, self.scen_ids[w])
                        self.trade_vars[key] = tuple(None if v is None else v.name for v in vs)
                        self.trade_ub[key] = tuple(
                            0.0 if v is None else
                            (np.inf if v.getUbOriginal() >= 1e19 else v.getUbOriginal())
                            for v in vs)

    def _coef(self, duals, farkas):
        if farkas:
            coef = dict.fromkeys(self.adj, 0.0)
        elif rho_key(0) in duals:
            # the KL master prices at its worst-case measure rho_bar, carried in the
            # duals so smoothing and the stability center treat it like any price
            rho = [duals[rho_key(g)] for g in self.scen_ids]
            coef = {n: c * (1.0 if w is None else rho[w]) for n, c, w in self._cw}
        else:
            coef = dict(self.base)
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

    def set_abs_gap(self, gap_abs):
        """Stop the pricing MILP on an absolute gap alone (incumbent minus dual bound
        <= gap_abs): the relative gap is switched off, since what the Lagrangian bound
        needs is the sum of the absolute slacks over the prosumers."""
        self.gap_abs = gap_abs
        if self.solver == 'gurobi':
            self.g.Params.MIPGap = 0.0
            self.g.Params.MIPGapAbs = gap_abs
        else:
            self.h.setOptionValue('mip_rel_gap', 0.0)
            self.h.setOptionValue('mip_abs_gap', gap_abs)

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
        col = self._column(vals)
        if not farkas:
            self.last = (dict(duals), obj, col)
        return obj, bound, col

    def _price_gurobi(self, duals, farkas):
        coef = self._coef(duals, farkas)
        g = self.g
        g.setAttr('Obj', self.g_vars, [coef.get(n, 0.0) for n in self.g_names])
        if self.mip_start and self._last_x is not None:
            # the previous plan is feasible for this prosumer whatever the duals:
            # hand it to Gurobi as a starting incumbent
            g.setAttr('Start', self.g_vars, self._last_x)
        for attempt in range(5):
            try:
                g.optimize()
                break
            except Exception as e:      # WLS token renewal hiccup: wait and retry
                # (also 'Token validation error (status 6)': it killed 2 of 30 runs at
                # n=60, 20 scenarios, ~140k pricing calls each)
                msg = str(e).lower()
                if ('license' not in msg and 'token' not in msg) or attempt == 4:
                    raise
                time.sleep(2.0 * (attempt + 1))
        if g.SolCount == 0:
            raise RuntimeError(f'pricing {self.player} (gurobi): no solution, status {g.Status}')
        x = g.getAttr('X', self.g_vars)
        self._last_x = x
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

    def _column(self, vals):
        col = Column(self.unit, self.stack, self.rows, self.names, vals)
        col.trade = {key: tuple(0.0 if n is None else vals.get(n, 0.0) for n in names)
                     for key, names in self.trade_vars.items()}
        if self.scen_ids != list(range(self.n_scen)):
            scen = [0.0] * self.n_scen
            for w, g in enumerate(self.scen_ids):
                scen[g] = col.scen[w]
            col.scen = scen
        return col

    def column_from(self, ef_vals):
        """Project a grand-coalition solution onto this player's plan."""
        if self.scen_ids == list(range(self.n_scen)):
            return self._column(ef_vals)
        # scenario block w of this stack is scenario scen_ids[w] of the EF: names end
        # in _s<w> here and _s<scen_ids[w]> there (first-stage names carry no suffix)
        vals = {}
        for n in self.names:
            w = self.stack.scen_of[n]
            if w is None:
                vals[n] = ef_vals.get(n, 0.0)
            else:
                suf = f'_s{w}'
                assert n.endswith(suf), n
                vals[n] = ef_vals.get(n[:-len(suf)] + f'_s{self.scen_ids[w]}', 0.0)
        return self._column(vals)


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
            for i, blk in enumerate(blocks):
                if sym:
                    self.x0[('r_sym', i)] = m.addVar(
                        name=f'r_sym_{i}', lb=0.0, obj=-block_reserve_payment(p, blk))
                else:
                    self.x0[('r_up', i)] = m.addVar(
                        name=f'r_up_{i}', lb=0.0,
                        obj=-block_reserve_payment(p, blk, 'pi_up'))
                    self.x0[('r_dn', i)] = m.addVar(
                        name=f'r_dn_{i}', lb=0.0,
                        obj=-block_reserve_payment(p, blk, 'pi_dn'))
        if self.enable_peak:
            for w, rho in enumerate(self.probs):
                self.x0[('p', w)] = m.addVar(name=f'p_s{w}', lb=0.0,
                                             obj=rho * p.get('pi_E_peak', 0.0))
        penalty = self.enable_reserve and reserve_mode(p) == 'penalty'
        if penalty:
            for w, rho in enumerate(self.probs):
                for t in self.T:
                    for d in ('up', 'dn'):
                        self.x0[(f's_{d}', t, w)] = m.addVar(
                            name=f's_res_{d}_{t}_s{w}', lb=0.0,
                            obj=rho * reserve_penalty_price(
                                p, t, 'pi_res' if sym else f'pi_{d}'))

        def x0_terms(kind, t, w):
            if kind in ('up', 'dn'):
                i = block_of_t[t]
                out = [(1.0, self.x0[('r_sym', i)] if sym else self.x0[(f'r_{kind}', i)])]
                if penalty:
                    out.append((-1.0, self.x0[(f's_{kind}', t, w)]))
                return out
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
                if penalty:
                    self.x0_rows.setdefault((f's_{kind}', t, w), []).append((key, -1.0))
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
                 sar_exact=(), lazy_kinds=(), doi=True, column_pool=False,
                 mip_start=False, balance_pricing=True, omega_tol=OMEGA_TOL,
                 pricing_abs=True):
        self.players, self.T, self.scenarios = list(players), list(T), scenarios
        # pricing units: a prosumer, or (prosumer, scenario) for a prosumer without
        # first-stage variables when split_scenarios is on. Such a prosumer's set is a
        # product over scenarios, so conv(X_j) is the product of the per-scenario hulls
        # and one convexity row per (j, w) gives the same master LP value, with columns
        # 1/|Omega| as dense. The electrolyzer owners' commitment ties their scenarios:
        # they stay whole.
        tied = set(params.get('players_with_electrolyzers', []))
        self.units = []
        for u in self.players:
            if self.split_scenarios and u not in tied and len(scenarios) > 1:
                self.units += [(u, w) for w in range(len(scenarios))]
            else:
                self.units.append(u)
        self.unit_player = {x: (x[0] if isinstance(x, tuple) else x) for x in self.units}
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
        # Dual-optimal inequalities (Ben Amor, Desrosiers & Valerio de Carvalho 2006)
        # on the balance rows, as the primal columns they dualize: the community
        # buying from / selling to the grid at the scenario's market prices. A new
        # plan then enters without waiting for partners, the grid covers the rest,
        # which is what the degenerate master lacks. Import caps keep them from
        # being valid a priori: a bound counts only with y = 0, and if the master
        # settles with y > 0 they are switched off and column generation goes on.
        self.doi, self.doi_active, self.y = doi, doi, {}
        # omega_tol: stop once UB - LB <= omega_tol * omega_hat as well, omega_hat =
        # incumbent - LB >= omega^LR, i.e. precision relative to the quantity reported
        # rather than to |z|. pricing_abs: pricing MILPs stop on an absolute gap of
        # (current CG tolerance) / (2 n), so their slacks never hold LB back.
        self.omega_tol, self.pricing_abs = omega_tol, pricing_abs
        self._abs_set = None
        # column pool: purged columns wait here; each iteration the pool is priced
        # first (one sparse product), and the pricing MILPs run only if it has
        # nothing with negative reduced cost
        self.pool_on, self.pool, self.pool_hits, self.pool_rounds = column_pool, [], 0, 0
        self._pool_mat = None
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
        # floor 1e-6, not 1e-9: at n=60, 20 scenarios, 4h reserve blocks (KL, day 1) the
        # tightening walked the pricing MIPGap 1e-4 -> 1e-8 in four passes without LB
        # moving and the run died in a segfault inside the pricing solver
        self.ub_every, self.pricing_gap_floor, self.pricing_tightened = ub_every, 1e-6, 0
        self._tighten_lb = None
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
        self.pricing_workers = max(1, min(self.pricing_workers, len(self.units)))
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
        # Workers take their prosumers in sequence, so an iteration lasts as long as
        # the slowest worker. The pricing MILPs with commitment binaries
        # (electrolyzers, heat pumps) are the slow ones: deal them out first, in a
        # snake order, then the rest the same way.
        heavy = (set(params.get('players_with_electrolyzers', [])) |
                 set(params.get('players_with_heatpumps', []))) if balance_pricing else set()
        order = ([x for x in self.units if self.unit_player[x] in heavy]
                 + [x for x in self.units if self.unit_player[x] not in heavy])
        self._group = {}
        for i, u in enumerate(order):
            r, c = divmod(i, k)
            self._group[u] = c if (r % 2 == 0 or not balance_pricing) else k - 1 - c
        def unit_pricing(x):
            if isinstance(x, tuple):
                return PlayerPricing(x[0], T, [scenarios[x[1]]], pricing_time_limit,
                                     pricing_gap, pricing_solver,
                                     env=self._envs[self._group[x]], scen_ids=[x[1]],
                                     n_scen=len(scenarios), unit=x)
            return PlayerPricing(x, T, scenarios, pricing_time_limit, pricing_gap,
                                 pricing_solver, env=self._envs[self._group[x]])
        self.subs = subs or {x: unit_pricing(x) for x in self.units}
        self.enable_reserve = bool(params.get('enable_reserve', False))
        self.enable_peak = bool(params.get('enable_peak', False))
        self.row_keys = [(k, t, w) for w in range(len(scenarios)) for t in self.T
                         for k in CARRIERS]
        if self.enable_reserve:
            self.row_keys += [(d, t, w) for w in range(len(scenarios)) for t in self.T
                              for d in ('up', 'dn')]
        if self.enable_peak:
            self.row_keys += [('peak', t, w) for w in range(len(scenarios)) for t in self.T]
        for sub in self.subs.values():
            sub.mip_start = mip_start
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
        self.columns = {u: [] for u in self.units}
        self.col_idx = {u: [] for u in self.units}
        self.patterns, self.mp12_patterns = {}, 0
        self.iteration, self.lb, self.L_bar = 0, -np.inf, -np.inf
        self._stall, self._stall_ref = 0, (np.inf, -np.inf)
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
        # unscaled parts of each x0 cost: first stage, and {scenario: cost}
        self.x0_first, self.x0_scen = {}, {}
        if self.enable_reserve:
            for i, blk in enumerate(blocks):
                names = [('r_sym', i)] if sym else [('r_up', i), ('r_dn', i)]
                for nm in names:
                    pay = block_reserve_payment(p, blk, 'pi_res' if sym else f'pi_{nm[0][2:]}')
                    x0_cost[nm] = -pay
                    self.x0_first[nm], self.x0_scen[nm] = -pay, {}
                    self.x0_rows[nm] = [(k, 1.0) for k in self.row_keys
                                        if k[0] in ('up', 'dn')
                                        and block_of_t[k[1]] == i
                                        and (sym or k[0] == nm[0].split('_')[1])]
        if self.enable_reserve and reserve_mode(p) == 'penalty':
            # shortfall per reserve row (t, scenario, direction): a scenario cost like
            # the peak p^w, so the KL cuts weight it by the worst-case distribution
            for w, rho in enumerate(self.probs):
                for t in self.T:
                    for d in ('up', 'dn'):
                        nm = (f's_{d}', t, w)
                        pen = reserve_penalty_price(
                            p, t, 'pi_res' if sym else f'pi_{d}')
                        x0_cost[nm] = rho * pen
                        self.x0_first[nm], self.x0_scen[nm] = 0.0, {w: pen}
                        self.x0_rows[nm] = [((d, t, w), -1.0)]
        if self.enable_peak:
            for w, rho in enumerate(self.probs):
                x0_cost[('p', w)] = rho * p.get('pi_E_peak', 0.0)
                self.x0_first[('p', w)] = 0.0
                self.x0_scen[('p', w)] = {w: p.get('pi_E_peak', 0.0)}
                self.x0_rows[('p', w)] = [(('peak', t, w), -1.0) for t in self.T]
        self.x0_obj = dict(x0_cost)
        self.conv = {u: lp.add_row(1.0, 1.0) for u in self.units}
        for name, members in fams.items():
            self._register_family(name, members)
            kind = members[0][0][0]
            self.row[name] = lp.add_row(0.0, 0.0) if kind in CARRIERS \
                else lp.add_row(-INF, 0.0)
        for nm, cost in x0_cost.items():
            agg = self._aggregate(dict(self.x0_rows[nm]))
            self.x0[nm] = lp.add_col(self._obj0(cost), 0.0, INF, [self.row[f] for f in agg],
                                     list(agg.values()))
        # penalty slacks, created disabled (ub 0)
        for name in fams:
            self._add_pen(name)
        self.y_cost = {}            # grid column -> (scenario, unscaled cost)
        if self.doi:
            for w, (rho, prm) in enumerate(self.scenarios):
                for t in self.T:
                    for k in CARRIERS:
                        for side, sgn, cost in (
                                ('imp', -1.0, rho * prm[f'pi_{k}_gri_import_{t}']),
                                ('exp', 1.0, -rho * prm[f'pi_{k}_gri_export_{t}'])):
                            if (k, side) in self.doi_skip:
                                continue
                            coef = {(k, t, w): sgn}
                            if k == 'E' and self.enable_peak:
                                coef[('peak', t, w)] = -sgn
                            cost += rho * self.doi_markup
                            agg = self._aggregate(coef)
                            self.y[(k, t, w, side)] = lp.add_col(
                                self._obj0(cost), 0.0, INF, [self.row[f] for f in agg],
                                list(agg.values()))
                            self.y_cost[(k, t, w, side)] = (w, cost / rho)

    def _obj0(self, cost):
        """Master objective of a column whose rho_hat-weighted cost is `cost`."""
        return cost

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
        for u in self.units:
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
        for u in self.units:
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

    def _col_entries(self, col):
        """(objective, rows, coefficients) of a prosumer column in the master."""
        agg = self._aggregate(col.coef)
        rows, coefs = [self.row[f] for f in agg], list(agg.values())
        if not getattr(col, 'noconv', False):
            rows.insert(0, self.conv[col.player])
            coefs.insert(0, 1.0)
        for r, a in getattr(col, 'extra', None) or ():
            rows.append(r)
            coefs.append(a)
        return col.cost, rows, coefs

    def _col_cost(self, col, duals):
        """The column's cost under the pricing measure in `duals`."""
        return col.cost

    def add_column(self, col):
        if self.mp12 and col.fs and not isinstance(col.player, tuple):
            key = tuple(sorted(col.fs.items()))
            for piece in self._mp12_pieces(col):
                h = self._add_one(piece)
                if not piece.noconv:            # the pattern column
                    self.patterns[col.player][key]['mu'] = h
        else:
            self._add_one(col)

    def _add_one(self, col):
        u = col.player
        obj, rows, coefs = self._col_entries(col)
        h = self.lp.add_col(obj, 0.0, self.lp.INF, rows, coefs)
        self.col_idx[u].append(h)
        self.columns[u].append(col)
        self.last_used[h] = self.iteration
        return h

    # MP1-2 (Maher & Muter, nested decomposition, eq. 6) for the prosumers whose
    # first-stage commitment z ties their scenarios: a plan (z, y_1..y_W) enters as
    # a pattern column mu_{j,z} (first-stage cost, the prosumer's convexity row) and
    # one scenario column per w (that scenario's cost and linking rows), tied by
    #     sum_p lambda_{j,z,w,p} - mu_{j,z} = 0          for every w,
    # rows created with the pattern. conv(X_j) = conv over z of prod_w conv Y_jw(z),
    # which is exactly this, so the master LP value is unchanged; the master can now
    # mix scenario plans of different pricing calls that share a commitment.
    # Columns still come from the full pricing MILP (its bound keeps LB valid).
    mp12 = False

    def _mp12_pieces(self, col):
        j, S = col.player, len(self.scenarios)
        key = tuple(sorted(col.fs.items()))
        pats = self.patterns.setdefault(j, {})
        pat = pats.get(key)
        pieces = []
        if pat is None:
            rows = [self.lp.add_row(0.0, 0.0) for _ in range(S)]
            pat = pats[key] = {'rows': rows, 'mu': None}
        mu = pat['mu']
        if mu is None or self._lp_pos(mu) < 0:
            m = Column.__new__(Column)
            m.player, m.cost, m.coef, m.first = j, col.first, {}, col.first
            m.scen, m.fs, m.noconv = [0.0] * S, dict(col.fs), False
            m.extra = [(r, -1.0) for r in pat['rows']]
            pieces.append(m)
            pat['mu_pending'] = True
        for w in range(S):
            c = Column.__new__(Column)
            c.player, c.first, c.fs, c.noconv = j, 0.0, None, True
            c.coef = {k: a for k, a in col.coef.items() if k[2] == w}
            c.trade = {k: v for k, v in (getattr(col, 'trade', None) or {}).items()
                       if k[2] == w}
            c.scen = [0.0] * S
            c.scen[w] = col.scen[w]
            c.cost = self.probs[w] * col.scen[w]
            c.extra = [(pat['rows'][w], 1.0)]
            pieces.append(c)
        self.mp12_patterns = sum(len(v) for v in self.patterns.values())
        return pieces

    def _lp_pos(self, h):
        lp = self.lp.main if isinstance(self.lp, _TwinLP) else self.lp
        return lp.pos[h]

    def _purge(self, ref):
        """Remove prosumer columns unused for purge_age iterations with reduced cost
        above a numerical zero at the current duals. Call right after an LP solve
        and before the next one: positions shift."""
        tol = 1e-9 * (1.0 + abs(ref))
        ncol = sum(len(v) for v in self.col_idx.values())
        cap = self.purge_cap * len(self.units)
        if ncol <= cap:
            return 0
        # candidates: unused for purge_age iterations and pricing out now, the
        # longest-unused first, until the column count is back under the cap. The
        # master is primal degenerate here -- a new plan only pays off together with
        # other prosumers' plans -- so columns must be given time to find partners.
        cand = sorted(((self.last_used[h], h) for u in self.units for h in self.col_idx[u]
                       if h not in self.protected
                       and self.iteration - self.last_used[h] >= self.purge_age
                       and self.lp.rc(h) > tol))
        drop = set(h for _, h in cand[:ncol - cap])
        self._dropped_cols = [c for u in self.units
                              for h, c in zip(self.col_idx[u], self.columns[u]) if h in drop]
        for u in self.units:
            keep = [(h, c) for h, c in zip(self.col_idx[u], self.columns[u]) if h not in drop]
            self.col_idx[u] = [h for h, _ in keep]
            self.columns[u] = [c for _, c in keep]
        drop = list(drop)
        if drop:
            if self.pool_on:
                self.pool.extend(self._dropped_cols)
                self._pool_mat = None
            self.lp.remove_cols(drop)
            for h in drop:
                del self.last_used[h]
            self.purged += len(drop)
        return len(drop)

    def _set_penalty(self, center, eps, delta):
        self._pen_state = (center, eps, delta)
        self._dual_next = True
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

    def _pool_price(self, duals, conv, adm):
        """Revive pooled columns with negative reduced cost at `duals`; returns how
        many. The pool is held as a sparse matrix over the original linking rows."""
        if not self.pool:
            return 0
        from scipy.sparse import csr_matrix
        if self._pool_mat is None:
            ridx = {k: i for i, k in enumerate(self.row_keys)}
            data, rows, cols = [], [], []
            for j, c in enumerate(self.pool):
                for k, a in c.coef.items():
                    data.append(a)
                    rows.append(j)
                    cols.append(ridx[k])
            self._pool_mat = csr_matrix((data, (rows, cols)),
                                        shape=(len(self.pool), len(self.row_keys)))
            self._pool_cost = np.array([c.cost for c in self.pool])
            pidx = {u: i for i, u in enumerate(self.units)}
            self._pool_owner = np.array([pidx[c.player] for c in self.pool])
        pi = np.array([duals.get(k, 0.0) for k in self.row_keys])
        sig = np.array([conv[u] for u in self.units])
        rc = self._pool_cost - self._pool_mat @ pi - sig[self._pool_owner]
        pick = np.nonzero(rc < -adm)[0]
        if len(pick) == 0:
            return 0
        # the most negative per prosumer, at most one each, as pricing would give
        best = {}
        for j in pick:
            u = self._pool_owner[j]
            if u not in best or rc[j] < rc[best[u]]:
                best[u] = j
        take = sorted(best.values())
        for j in take:
            self.add_column(self.pool[j])
        keep = np.ones(len(self.pool), dtype=bool)
        keep[take] = False
        self.pool = [c for c, k in zip(self.pool, keep) if k]
        self._pool_mat = None
        self.pool_hits += len(take)
        return len(take)

    def _price_group(self, members, duals):
        return {u: self.subs[u].price(duals) for u in members}

    def _price_all(self, duals):
        t0 = time.time()
        if self._pool is None:
            res = {u: s.price(duals) for u, s in self.subs.items()}
        else:
            groups = {}
            for u in self.units:
                groups.setdefault(self._group[u], []).append(u)
            futs = [self._pool.submit(self._price_group, m, duals) for m in groups.values()]
            res = {}
            for f in futs:
                res.update(f.result())
            res = {u: res[u] for u in self.units}
        dt = time.time() - t0
        self.t_price += dt
        self._it_price += dt
        return res

    # --- bounds --------------------------------------------------------------
    def _penalized(self):
        return self._pen_state is not None and self._pen_state[2] > 0.0

    def _omega_abs(self):
        if not self.omega_tol or not np.isfinite(self.lb) or not np.isfinite(self.incumbent):
            return 0.0
        return self.omega_tol * max(self.incumbent - self.lb, 0.0)

    def _gap_tol(self, ref):
        return max(self.gap_tol * (1.0 + abs(ref)), self._omega_abs())

    def _set_pricing_abs(self, tol):
        """Absolute pricing gap tol / (2 n), refreshed when it moves by 20%."""
        d = max(tol / (2.0 * len(self.units)), 1e-9)
        if self._abs_set is None or abs(d - self._abs_set) > 0.2 * self._abs_set:
            for sub in self.subs.values():
                sub.set_abs_gap(d)
            self._abs_set = d

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
                if self._y_total(shadow=True) > 1e-7:
                    ub = np.inf
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

    def _y_total(self, shadow=False):
        if not self.doi_active:
            return 0.0
        xval = self.lp.x_shadow if shadow else self.lp.x
        return sum(xval(j) for j in self.y.values())

    # fallback_purge_age: before the DOI switch-off, drop every prosumer column unused
    # for this many passes (and pricing out), whatever the column count. Measured
    # (KL master, n=60): the no-DOI master grows to ~9,900 columns and 3.4M nonzeros
    # at |Omega|=10, 10-20 s per LP, and the fallback took 66-85% of the CG time.
    # 0 keeps the ordinary purge only. Off by default: at n=60, |Omega|=5, r=0.5 it
    # dropped 1,352 of 4,572 columns and changed nothing (fallback 323 s -> 322 s);
    # the no-DOI master asks ~48 new plans a pass and is back at 6,200 columns. The
    # cost is the degeneracy, not the size.
    fallback_purge_age = 0

    # DOI taper: when the master settles with grid trade y > 0, cap the total with a
    # budget row sum(y) <= B and shrink B = Y0 * factor^k over a few column
    # generation passes, before (and ideally instead of) switching the grid columns
    # off. They stay in the master as the partners that break its degeneracy, only
    # fewer of them, so pricing is asked for the y-free combinations a bit at a time.
    # Off by default: at n=60, |Omega|=5, r=0.5 (KL) the grid trade filled every budget
    # (999 -> 0.22 over 8 steps, the RMP rising only 0.007) and no y-free upper bound
    # appeared before the last step: 441 s against 323 s for the plain switch-off.
    doi_taper_steps, doi_taper_factor = 0, 0.3

    def _taper_doi(self, rounds, t0):
        js = list(self.y.values())
        y0 = self._y_total()
        status = None
        if y0 <= 1e-7:
            return status
        twin = isinstance(self.lp, _TwinLP)
        b = self.lp.add_col(0.0, 0.0, y0, [], [])
        self.lp.add_row_with(-self.lp.INF, 0.0, js + [b], [1.0] * len(js) + [-1.0])
        self.doi_budget = b
        for k in range(1, self.doi_taper_steps + 1):
            B = y0 * self.doi_taper_factor ** k
            self.lp.set_cols([b], ub=[B])
            if twin:
                self.lp.shadow.set_cols([b], ub=[B])
            self._set_penalty(None, 0.0, 0.0)
            status, obj = self._cg(f'taper{k}', True)
            if status != 'done':
                self._update_ub()
                if self._converged():
                    status = 'done'
            rounds.append({'round': len(rounds) + 1, 'eps': 0.0, 'delta': 0.0, 'obj': obj,
                           'lb': self.lb, 'ub': self.ub, 'slack': 0.0,
                           'iterations': self.iteration, 'time': time.time() - t0,
                           'certified': status == 'done', 'status': status,
                           'doi_budget': B, 'y_total': self._y_total()})
            if self.verbose:
                print(f'  -- DOI taper {k}: sum(y) <= {B:.4g} (y {self._y_total():.4g}) '
                      f'RMP {obj:.4f} LB {self.lb:.4f} UB {self.ub:.4f} [{status}] '
                      f'{time.time() - t0:.0f}s')
            if status == 'done':
                break
        return status

    def _purge_before_fallback(self):
        if not self.fallback_purge_age or not self.purge_every:
            return 0
        saved = (self.purge_age, self.purge_cap)
        self.purge_age, self.purge_cap = self.fallback_purge_age, 0
        before = sum(len(v) for v in self.col_idx.values())
        try:
            n = self._purge(self.lb if np.isfinite(self.lb) else 0.0)
        finally:
            self.purge_age, self.purge_cap = saved
        self.fallback_purged = n
        if self.verbose:
            print(f'  -- before the DOI switch-off: purged {n} of {before} prosumer columns '
                  f'(unused for {self.fallback_purge_age}+ passes)')
        return n

    # doi_repair: before the switch-off, hand the grid trade the master settled on to
    # the members. A member that imports at the grid and sells to the community in
    # row r (or buys from the community and exports) has, per unit, exactly the
    # reduced cost of y^+_r (y^-_r), which is 0 at the settled master. Pricing every
    # member once at duals moved past the price box on those rows only,
    #     pi_r - delta rho_w   (y^+_r > 0),      pi_r + delta rho_w   (y^-_r > 0),
    # with delta = frac * import price of the row, makes that trade strictly
    # profitable, so the MILPs return plans that take it up to their own caps: the
    # y-free combinations the switched-off master otherwise waits ~1,300 s for
    # (n=60, 20 scenarios). One pricing pass per frac; the columns are ordinary
    # plans (feasible, from the MILP), and the Lagrangian bound of those passes is
    # valid as any. () = off.
    doi_repair = ()

    def _repair_doi(self):
        if not self.doi_repair:
            return 0
        t0 = time.time()
        self._set_penalty(None, 0.0, 0.0)
        self.lp.solve()
        duals, conv = self._duals()
        hot = [(key, self.lp.x(j)) for key, j in self.y.items() if self.lp.x(j) > 1e-7]
        added = 0
        for frac in self.doi_repair:
            d = dict(duals)
            for (k, t, w, side), _ in hot:
                rho = duals.get(rho_key(w), self.probs[w])
                imp = self.y_cost.get((k, t, w, 'imp'), (w, 0.0))[1]
                delta = frac * max(abs(imp), 1e-3) * rho
                d[(k, t, w)] = d.get((k, t, w), 0.0) + (-delta if side == 'imp' else delta)
            res = self._price_all(d)
            self._record(d, res)
            for u, (_, _, col) in res.items():
                self.add_column(col)
                added += 1
        self.doi_repaired = {'rows': len(hot), 'y': sum(v for _, v in hot),
                             'columns': added, 'time': time.time() - t0}
        if self.verbose:
            print(f'  -- DOI repair: {len(hot)} rows with grid trade (y {self.doi_repaired["y"]:.4g}),'
                  f' +{added} member plans at frac {list(self.doi_repair)} '
                  f'({time.time() - t0:.0f}s)')
        return added

    def _disable_doi(self, keys=None):
        """Switch grid columns off: all of them, or only `keys`."""
        self._dual_next = True
        off = getattr(self, 'y_off', set())
        keys = [k for k in (self.y if keys is None else keys) if k not in off]
        js = [self.y[k] for k in keys]
        self.lp.set_cols(js, ub=[0.0] * len(js))
        if isinstance(self.lp, _TwinLP):
            self.lp.shadow.set_cols(js, ub=[0.0] * len(js))
        self.y_off = off | set(keys)
        if len(self.y_off) == len(self.y):
            self.doi_active = False
        if self.verbose:
            print(f'  -- DOI: the master settled with y > 0; {len(keys)} grid columns '
                  f'switched off ({len(self.y) - len(self.y_off)} left)')

    # --- grid certificate ------------------------------------------------------
    # A grid column y^+_r (the community buys at the market in row r = (k, t, w)) and
    # a member trade that imports at the market and passes the unit on to the
    # community -- more community export, or less community import -- have the same
    # cost (the market price, scenario weight included), the same balance
    # coefficient and the same peak coefficient; y^-_r likewise with a member that
    # exports at the market what it takes from the community. The trade changes none
    # of the member's own constraints (its balance keeps i_gri - e_gri + i_com -
    # e_com). So if the plans in use have room for it,
    #     sum_q lambda_q cap_q(r) >= y_r        for every row r with y_r > 0,
    #     cap^+_q(r) = min(ub_ig - ig, (ub_ec - ec) + ic),
    #     cap^-_q(r) = min(ub_eg - eg, (ub_ic - ic) + ec),
    # moving the trade into those plans gives a solution of the ORIGINAL master (no
    # grid columns; every modified plan is a point of X_j with its binaries unchanged)
    # with exactly the same scenario costs, hence the same objective: the RMP value
    # itself (psi of it under KL) is then an upper bound on z_MP. Nothing is assumed
    # about duals, so it holds whatever the pricing tolerance. doi_certify=False
    # restores the y = 0 bound alone.
    doi_certify = True
    doi_switch_passes = 20
    # stall_reset: after this many passes with neither the RMP value nor LB moving,
    # one pass drops the smoothing (alpha = 1) and re-centres at the RMP duals. 0: off.
    stall_reset = 10

    def _grid_uncovered(self):
        """Grid columns with y > 0 that the plans in use cannot take over."""
        need = {k: self.lp.x(j) for k, j in self.y.items() if self.lp.x(j) > 1e-9}
        if not need:
            return []
        room = dict.fromkeys(need, 0.0)
        for u in self.units:
            ubs = self.subs[u].trade_ub
            for h, col in zip(self.col_idx[u], self.columns[u]):
                lam = self.lp.x(h)
                tr = getattr(col, 'trade', None)
                if lam <= 1e-12 or not tr:
                    continue
                for key in need:
                    v = tr.get(key[:3])
                    if v is None:
                        continue
                    ig, eg, ic, ec = v
                    Uig, Ueg, Uic, Uec = ubs[key[:3]]
                    c = (min(Uig - ig, (Uec - ec) + ic) if key[3] == 'imp'
                         else min(Ueg - eg, (Uic - ic) + ec))
                    if c > 0.0:
                        room[key] += lam * c
        return [k for k, y in need.items() if room[k] < y - 1e-7 * (1.0 + y)]

    def _grid_value(self, lp_obj):
        """Objective of the current RMP solution (the KL master: psi of it)."""
        return lp_obj

    def _certify_grid(self, lp_obj):
        """UB from an RMP solution with grid trade the plans in use can absorb."""
        t0 = time.time()
        miss = self._grid_uncovered()
        self.t_certify = getattr(self, 't_certify', 0.0) + time.time() - t0
        self.n_certify = getattr(self, 'n_certify', 0) + 1
        if not miss:
            ub = self._grid_value(lp_obj)
            self.n_certified = getattr(self, 'n_certified', 0) + 1
            if ub < self.ub:
                self.ub = ub
        return miss

    def _converged(self):
        return np.isfinite(self.ub) and self.ub - self.lb <= self._gap_tol(self.ub)

    # hooks for the KL master (KLMaster); the plain master has no distribution cuts
    kl_nested, round_frac, doi_markup = False, 1.0, 0.0
    # (carrier, side) pairs left without a grid column, e.g. {('G', 'exp')}. At n=60,
    # |Omega|=5 (KL) 81% of the grid trade left when the master settled was hydrogen
    # export, a near-tie with the electrolyzer owners exporting themselves.
    doi_skip = frozenset()
    ub_with_y = False       # KL: _update_ub can bound an RMP that trades with the grid
    # accept_doi_master: stop once the master WITH the grid columns has converged
    # (LB within tolerance of its RMP value), instead of switching them off to find
    # a y = 0 upper bound. See solve(). Off by default.
    accept_doi_master = False
    # lp_mixed: primal simplex as usual, but dual simplex for the one solve after the
    # master changed in a way that leaves the old basis dual feasible rather than
    # primal feasible: rows added (distribution cuts, tangents), penalty slacks
    # re-priced or re-bounded (a new round), grid columns switched off. Off by
    # default: n=60, 10 scenarios (KL, --accept-doi-master), those ~20 solves took
    # 3.17 s with dual simplex against 2.53 s with primal; 841 s -> 799 s overall,
    # within run-to-run noise.
    lp_mixed, _dual_next = False, False
    split_scenarios = False

    def _separate_dist(self, lp_obj):
        return 0

    def _final_ub(self, obj):
        if obj < self.ub:
            self.ub = obj

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
            use_dual = self.lp_mixed and self._dual_next
            self._dual_next = False
            lp_obj = self.lp.solve(method='dual' if use_dual else None)
            t_lp = time.time() - t0
            self.t_lp += t_lp
            self._it_price = 0.0
            duals, conv = self._duals()
            if self.purge_every:
                for u in self.units:
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
            # distribution cuts (KL master; none otherwise), before the upper bound:
            # the UB is the worst case of this solution, which separation computes
            cuts = self._separate_dist(lp_obj)
            if cuts and self.kl_nested:
                # nested: finish the row generation at these columns before pricing
                self.log.append({'iter': self.iteration, 'lp': lp_obj, 'lb': self.lb,
                                 'ub': self.ub, 'mode': 'cut', 'added': 0, 'cuts': cuts,
                                 'round': tag, 't_lp': t_lp, 't_price': 0.0})
                continue
            if not self._penalized() and not viol and self._y_total() <= 1e-7:
                # (under dyn-SAR, only a solution of the aggregated master that
                # violates no original row is feasible, and so bounds z_MP; with the
                # DOIs, only one that buys nothing from the grid)
                self._update_ub(lp_obj)
            tol = max(rel * (1.0 + abs(lp_obj)), self._omega_abs())
            if self.pricing_abs:
                self._set_pricing_abs(self._gap_tol(lp_obj))
            adm = 1e-9 * (1.0 + abs(lp_obj))        # numerical zero for reduced costs
            self.iteration += 1
            if self.iteration > self.max_iter:
                raise RuntimeError('column generation: iteration limit')
            alpha, added, min_rc, mode = 1.0, 0, 0.0, 'std'
            status = None
            if (self.doi_certify and self.doi_active and not self._penalized() and not viol
                    and not cuts and self.lb >= lp_obj - tol and self._y_total() > 1e-7):
                self._certify_grid(lp_obj)
            if self._converged():
                status = 'done'
            elif not cuts and self.lb >= lp_obj - self.round_frac * tol:
                status = 'round'
            if status is None and self.pool_on:
                added = self._pool_price(duals, conv, adm)
                if added:
                    mode = 'pool'
                    self.pool_rounds += 1
            # stall guard: count the passes in which neither the RMP value nor LB moved
            prog = 1e-3 * tol
            if (lp_obj < self._stall_ref[0] - prog) or (self.lb > self._stall_ref[1] + prog):
                self._stall = 0
            else:
                self._stall += 1
            self._stall_ref = (lp_obj, self.lb)
            if status is None and self.smoothing and not added:
                if self.center is None:
                    self.center = dict(duals)
                alpha = self._alpha(lp_obj)
                if self.stall_reset and self._stall >= self.stall_reset:
                    # Smoothing stalled: at n=60, 20 scenarios it sat ~650 passes at
                    # alpha = 0.1 (smooth, misprice, smooth, ...) with RMP and LB both
                    # frozen, until one alpha = 1 pass moved LB and ended it. Price one
                    # pass at the RMP duals alone (Kelley) and restart the center there.
                    alpha, self._stall = 1.0, 0
                    self.center = dict(duals)
                    self.n_stall_resets = getattr(self, 'n_stall_resets', 0) + 1
                if alpha < 1.0:
                    st = {k: alpha * duals.get(k, 0.0) + (1 - alpha) * self.center.get(k, 0.0)
                          for k in set(duals) | set(self.center)}
                    res = self._price_all(st)
                    self._record(st, res)
                    mode = 'smooth'
                    for u, (_, _, col) in res.items():
                        rc = self._col_cost(col, duals) - sum(
                            duals.get(k, 0.0) * a for k, a in col.coef.items()) - conv[u]
                        min_rc = min(min_rc, rc)
                        if rc < -adm:
                            self.add_column(col)
                            added += 1
                    if not added:
                        mode = 'misprice'
            if status is None and not added and mode != 'pool':
                if self._converged():
                    status = 'done'
                elif not cuts and self.lb >= lp_obj - self.round_frac * tol:
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
                    if not added and not cuts:
                        if self._converged():
                            status = 'done'
                        elif not cuts and self.lb >= lp_obj - self.round_frac * tol:
                            status = 'round'
                        if self.verbose and status is None:
                            slack = sorted(((r[0] - r[1], u) for u, r in res.items()),
                                           reverse=True)
                            print(f'    [no column] RMP-LB {lp_obj - self.lb:.4e}  '
                                  f'sum pricing slack {sum(s for s, _ in slack):.4e}  '
                                  f'top {[(u, round(s, 6)) for s, u in slack[:3]]}  '
                                  f'gaps {sorted(set(self.subs[u].gap for u in res))}')
                        if status is not None:
                            pass
                        elif (self._tighten_lb is not None
                              and self.lb <= self._tighten_lb + prog):
                            # the last tightening left LB where it was: the pricing
                            # gaps are not what holds it down, so stop tightening
                            status = 'stalled'
                        elif self._tighten_pricing(res):
                            mode = 'tighten'
                            self._tighten_lb = self.lb
                        else:
                            status = 'stalled'
                    elif (sum(r[0] - r[1] for r in res.values())
                          > 0.5 * (lp_obj - self.lb) > tol / 2):
                        # Columns still price, but at the true duals the pricing
                        # MILPs' own gaps (incumbent minus dual bound) are most of
                        # what separates LB from the RMP: at 10 or more scenarios
                        # Gurobi stops at the 1e-4 gap without proving the rest, and
                        # the tail then adds columns worth ~1e-5 while LB never
                        # moves. Tighten those prosumers now rather than waiting for
                        # a pass that prices nothing.
                        if self._tighten_pricing(res):
                            mode = 'tighten'
                    if added:
                        # the stop-tightening test is for consecutive passes that
                        # price nothing; columns in between reset it
                        self._tighten_lb = None
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
            if status is None and since_ub >= self.ub_every and (
                    self._penalized() or (self.ub_with_y and self._y_total() > 1e-7)):
                self._update_ub()
                since_ub = 0
                if self._converged():
                    status = 'done'
            self.log.append({'iter': self.iteration, 'lp': lp_obj, 'lb': self.lb,
                             'ub': self.ub, 'alpha': alpha, 'mode': mode,
                             'min_rc': min_rc, 'added': added, 'cuts': cuts, 'round': tag,
                             't_lp': t_lp, 't_price': self._it_price,
                             'lp_stats': getattr(self.lp, 'stats', None)})
            if self.verbose and (self.iteration % 25 == 0 or status or mode == 'tighten'):
                print(f'  CG {self.iteration:4d} | RMP {lp_obj:13.4f} | LB {self.lb:13.4f} '
                      f'| UB {self.ub:13.4f} | a {alpha:.2f} {mode:8s} '
                      f'| min rc {min_rc:11.4e} | +{added}' + (f' +{cuts}cut' if cuts else '')
                      + (f'  [{status}]' if status else '')
                      + (f' | lp {t_lp:.2f}s solver {self.lp.stats[0]:.2f}s '
                         f'{self.lp.stats[1]:.0f} it {self.lp.stats[2]} cols {self.lp.stats[3]} nz'
                         if getattr(self.lp, 'stats', None) else ''))
            if status:
                return status, lp_obj
            if self.purge_every and self.iteration % self.purge_every == 0:
                self._purge(lp_obj)         # the next pass re-solves the LP first

    EF_ROW = {'E': 'community_elec_balance', 'H': 'community_heat_balance',
              'G': 'community_hydro_balance', 'up': 'reserve_up_coupling',
              'dn': 'reserve_dn_coupling', 'peak': 'peak_penalty_cons'}

    def duals_from_ef(self, ef, fix_binaries=False):
        """A first dual point: the extensive form's LP duals on the linking rows.

        mode 'lp' relaxes the integrality; 'fix' fixes the integer variables at the
        EF's MIP solution first (the restricted-pricing LP). Either is a guess at the
        Lagrangian dual solution pi*, available for one LP solve. Each EF row is
        matched to its master row by name and scaled by the coefficient of one shared
        variable, pi_master = pi_EF * a_EF / a_master, so sign and the rho scaling of
        the scenario blocks come out right.
        """
        sm = ef['stack'].model
        g, gv = _to_gurobi(sm, 'ef_lp')
        # _to_gurobi copies rows and bounds only; the EF objective goes in here
        for v in sm.getVars():
            gv[v.name].Obj = v.getObj()
        g.ObjCon = sm.getObjoffset()
        if fix_binaries:
            for n, v in gv.items():
                if v.VType != 'C':
                    x = float(round(ef['vals'][n]))
                    v.LB = v.UB = x
        g.update()                      # relax() copies only what is updated
        r = g.relax()
        r.Params.OutputFlag = 0
        r.optimize()
        if r.Status != 2:
            raise RuntimeError(f'EF LP for the dual warm start: status {r.Status}')
        pi, missing = {}, 0
        for key in self.row_keys:
            kind, t, w = key
            con = r.getConstrByName(f'{self.EF_ROW[kind]}_{t}_s{w}')
            ref = next(((n, a) for u in self.players if u in self.subs
                        for n, a in self.subs[u].rows.get(key, ()) if a), None)
            if con is None or ref is None:
                missing += 1
                continue
            a_ef = r.getCoeff(con, r.getVarByName(ref[0]))
            if a_ef == 0.0:
                missing += 1
                continue
            pi[key] = con.Pi * a_ef / ref[1]
        # the <= rows need pi <= 0 (Theta); clip solver noise
        for key in pi:
            if key[0] not in CARRIERS and pi[key] > 0.0:
                pi[key] = 0.0
        self.dual_init_info = {'mode': 'fix' if fix_binaries else 'lp',
                               'lp_obj': r.ObjVal, 'missing_rows': missing}
        return pi

    def _add_seeds(self, init_vals, init_cols):
        """The first column per prosumer: its plan in the extensive-form solution."""
        cols = list(init_cols) if init_cols is not None else [
            self.subs[u].column_from(init_vals) for u in self.units]
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
        self.protected = {h for u in self.units for h in self.col_idx[u]}
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
    def solve(self, init_vals=None, init_cols=None, init_duals=None):
        t0 = time.time()
        self._add_seeds(init_vals, init_cols)
        if init_duals is not None:
            # dual warm start: price at the guess before the first round, so the
            # stability center (smoothing and penalty) starts there
            self._it_price = 0.0
            res = self._price_all(init_duals)
            self._record(init_duals, res)
            for u, (obj, _, col) in res.items():
                self.add_column(col)
            if self.verbose:
                print(f'  dual warm start ({self.dual_init_info["mode"]}): '
                      f'L = {self._lagrangian(init_duals, [r[1] for r in res.values()]):.4f}, '
                      f'EF LP {self.dual_init_info["lp_obj"]:.4f}, '
                      f'{self.dual_init_info["missing_rows"]} rows unmatched')
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
        self.doi_master_certified, self.ub_doi_master = False, None
        if (status != 'done' and self.doi_active and self.accept_doi_master
                and not self._penalized() and obj - self.lb <= self._gap_tol(obj)):
            # Converged on the master WITH the grid columns: LB is within tolerance
            # of its RMP value. LB, the dual point it came from, and with them the
            # Owen allocation, its stability and eps = (EF value - LB)/n, are all
            # certified; only the claim that LB is v^LR itself rests on the grid
            # columns being dual-optimal (at 5 and 10 scenarios the y = 0 bound found
            # by the switch-off ended 0.02-0.03 above this RMP value).
            status, self.doi_master_certified, self.ub_doi_master = 'done', True, obj
            if self.verbose:
                print(f'  -- converged on the master with grid columns: RMP {obj:.4f} '
                      f'LB {self.lb:.4f} (y {self._y_total():.4g}); no switch-off')
        if status != 'done' and self.doi_active and self.doi_taper_steps:
            status = self._taper_doi(rounds, t0)
        if status != 'done' and self.doi_active:
            self._purge_before_fallback()
            self._repair_doi()
        # Switch-off. With the certificate on, only the grid columns the members'
        # plans in use cannot take over are switched off, and column generation goes
        # on with the rest (whose trade the certificate absorbs into an exact UB);
        # repeated while some grid column is still left uncovered. Without it, or on
        # the last pass, every grid column goes.
        for sw in range(self.doi_switch_passes):
            if status == 'done' or not self.doi_active:
                break
            keys = (self._grid_uncovered() if self.doi_certify
                    and sw + 1 < self.doi_switch_passes else None)
            self._disable_doi(keys or None)
            # Without the grid columns the collected plans may no longer combine, and
            # the master is back at a degenerate vertex where unstabilized column
            # generation was seen to stall for 5000+ passes (KL master, n = 6).
            # Restart the penalty rounds, small, around the current center.
            eps = delta = (self.pen_eps * self.pen_shrink ** 2 if self.pen_eps > 0.0
                           else 0.0)
            while True:
                last = eps <= 0.0
                self._set_penalty(None if last else self.center, eps, delta)
                status, obj = self._cg(len(rounds) + 1, last)
                if status != 'done':
                    self._update_ub()
                    if self._converged():
                        status = 'done'
                rounds.append({'round': len(rounds) + 1, 'eps': eps, 'delta': delta,
                               'obj': obj, 'lb': self.lb, 'ub': self.ub, 'slack': 0.0,
                               'iterations': self.iteration, 'time': time.time() - t0,
                               'certified': status == 'done', 'status': status,
                               'doi_off': True})
                if self.verbose:
                    print(f'  -- penalty round {len(rounds)} (no DOI): eps {eps:.3g} '
                          f'RMP {obj:.4f} LB {self.lb:.4f} UB {self.ub:.4f} [{status}] '
                          f'{time.time() - t0:.0f}s')
                if status == 'done' or last:
                    break
                eps, delta = eps * self.pen_shrink, delta * self.pen_shrink
                if eps < 1e-4:
                    eps = delta = 0.0
        # report the unpenalized restricted master
        self._set_penalty(None, 0.0, 0.0)
        obj = self.lp.solve()
        self._final_ub(obj)
        duals, sigma = (self.best[0], dict(self.best[1])) if self.best else self._duals()
        if len(self.units) != len(self.players):
            unit_sigma = sigma
            sigma = {u: 0.0 for u in self.players}
            for x, v in unit_sigma.items():
                sigma[self.unit_player[x]] += v
        return {'status': 'optimal' if status == 'done' else status,
                'obj': obj, 'lb': self.lb, 'ub': self.ub,
                'gap': (self.ub - self.lb) / (1.0 + abs(self.ub)), 'duals': duals,
                'sigma': sigma, 'iterations': self.iteration,
                'lambda': {u: [self.lp.x(j) for j in self.col_idx[u]] for u in self.units},
                'x0': {f'{k[0]}_{k[1]}': self.lp.x(j) for k, j in self.x0.items()},
                'columns': {u: len(self.columns[u]) for u in self.units},
                'y_total': self._y_total(), 'penalty': {'rounds': rounds},
                'doi': {'used': self.doi, 'active_at_end': self.doi_active,
                        'master_certified': self.doi_master_certified,
                        'ub_doi_master': self.ub_doi_master},
                'time': time.time() - t0,
                'timing': {'lp': self.t_lp, 'pricing': self.t_price,
                           'pricing_by_player': {u: s.time for u, s in self.subs.items()},
                           'pricing_calls': sum(s.calls for s in self.subs.values()),
                           'pricing_tightened': self.pricing_tightened,
                           'purged': self.purged, 'pool_hits': self.pool_hits,
                           'fallback_purged': getattr(self, 'fallback_purged', 0),
                           'doi_repaired': getattr(self, 'doi_repaired', None),
                           'stall_resets': getattr(self, 'n_stall_resets', 0),
                           'grid_certify': {'calls': getattr(self, 'n_certify', 0),
                                            'certified': getattr(self, 'n_certified', 0),
                                            'time': getattr(self, 't_certify', 0.0),
                                            'switched_off': len(getattr(self, 'y_off', ()))},
                           'mp12_patterns': self.mp12_patterns,
                           'omega_tol': self.omega_tol, 'pricing_abs_final': self._abs_set,
                           'pool_rounds': self.pool_rounds,
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
            per = {x: (self.best[1][x], self.best[2][x]) for x in self.units}
        else:
            per = {}
            for x, sub in self.subs.items():
                obj, _, col = sub.price(duals)
                per[x] = (obj, col)
        if len(self.units) == len(self.players):
            return per
        out = {}
        for u in self.players:
            xs = [x for x in self.units if self.unit_player[x] == u]
            out[u] = (sum(per[x][0] for x in xs), Column.total(u, [per[x][1] for x in xs]))
        return out


class KLMaster(DirectMaster):
    """The KL-ball DRO master (ieee_owen/robust_core.md, sec. 2.3), column-and-cut.

    Cost convention: the master minimizes rho_hat . cost(lambda, x0, y) + theta, the
    worst expected community cost written as the reference expectation plus a
    premium theta, subject to one distribution cut per rho^k in D,

        sum_w (rho^k_w - rho_hat_w) cost_w(lambda, x0, y) - theta <= 0   [mu_k <= 0]

    plus the linking rows [pi^w] and the convexity rows [sigma_j] of DirectMaster.
    The objective is DirectMaster's own; the cuts carry only the cost differences
    across scenarios. D starts as {rho_hat}, whose cut is theta >= 0, so without
    further cuts this is DirectMaster exactly (radius 0 checks it). No exponential
    cone: the cuts are the tilted distributions of kl_worst.
    (A first version put every cost into the cuts and left theta alone in the
    objective: the same LP, but at n=60, |Omega|=5 each primal simplex solve took
    ~1100 iterations and 1.6 s against ~0.1 s, and the master LP was 84% of CG.)

      pricing  theta >= 0 carries rho_hat's cut as a bound, and its column forces
               sum_k (-mu_k) <= 1; rho_bar = sum_k -mu_k rho^k + (1 - sum_k -mu_k) rho_hat
               is a mixture of points of the ball, hence in it. A column's reduced
               cost is first + rho_bar . scen - pi a - sigma: the same pricing MILP
               with rho_hat replaced by rho_bar. rho_bar rides in the duals dict
               (rho_key), so Wentges smoothing mixes it like any price.
      LB       L(rho_bar, pi) from the pricing MILPs' dual bounds: valid for any
               rho_bar in the ball, since min_x max_rho >= min_x E_rho_bar.
      UB       psi(cost of the unpenalized RMP solution), the true worst case of a
               feasible plan (kl_worst's dual value); the RMP value itself bounds
               nothing, as the cuts restrict the adversary.
      cuts     each pass evaluates psi at the RMP solution and adds the tilted
               distribution if theta falls short of psi by more than kl_cut_frac of
               the CG tolerance. A cut never invalidates a column, and a column never
               a cut. Interleaved by default (cut and price in the same pass);
               kl_nested re-solves the master until no cut is violated first.

    UB - LB = (psi - z) + (z - LB), z = rho_hat . cost + theta the RMP's worst case:
    cut violation plus pricing violation.
    """
    def __init__(self, *args, kl_radius=0.0, kl_nested=False, kl_cut_frac=0.05,
                 doi_markup=0.0, kl_seed=None, **kw):
        if kw.get('sar') or kw.get('lazy_kinds') or kw.get('column_pool'):
            raise ValueError('the KL master supports neither dyn-SAR, lazy rows nor '
                             'the column pool')
        self.kl_radius, self.kl_nested, self.kl_cut_frac = kl_radius, kl_nested, kl_cut_frac
        # doi_markup [EUR per unit]: the grid columns cost this much more than the
        # market (both sides), a tie-break against settling with grid trade. Off by
        # default: at n=60, |Omega|=5, r=0 a markup of 1e-3 made penalty rounds 1-7
        # take 501 s instead of 160 s (the stochastic master: 147 s). The DOIs break
        # the master's degeneracy only when they price the grid exactly at market.
        # A settled y > 0 is left to _ub_no_grid and the stabilized DOI switch-off.
        self.doi_markup = doi_markup
        # a pass that settles a penalty round must leave room for the cut violation
        # it did not separate: UB - LB <= kl_cut_frac tol + round_frac tol < tol
        self.round_frac = 0.9
        self.cut_rows, self.cut_rho, self._cvec = [], [], {}
        self.kl_log = []
        self._psi_main = None
        super().__init__(*args, **kw)
        # kl_seed: distributions known up front (the robust EF's worst case), put in
        # as cuts before the first pass. The EF's worst case is within ~0.03 of the
        # master's final rho* at 5 and 10 scenarios; without it the master spends
        # rounds 2-4 finding it again, one cut at a time. Measured (n=60, r=0.5, A,
        # --accept-doi-master): no gain, 166/841/6859 s -> 172/886/7128 s at 5/10/20
        # scenarios, and more cuts (the worst case follows the master's h, which
        # moves a lot early on). Off by default (--kl-seed-ef).
        self.kl_seeded = 0
        for rho in (kl_seed or []):
            rho = np.asarray(rho, float)
            if kl_div(rho, self.probs) <= self.kl_radius + 1e-9 and np.abs(rho - self._ref).max() > 1e-9:
                self._refine(rho)
                self.kl_seeded += 1

    # --- model ---------------------------------------------------------------
    def _build(self):
        super()._build()
        S = len(self.scenarios)
        self._ref = np.array(self.probs)
        # x0 and grid columns: their cost per scenario, as vectors over Omega
        for nm, j in self.x0.items():
            v = np.full(S, self.x0_first[nm])
            for w, c in self.x0_scen[nm].items():
                v[w] += c
            self._cvec[j] = v
        for key, j in self.y.items():
            v = np.zeros(S)
            w, c = self.y_cost[key]
            v[w] = c                            # doi_markup included
            self._cvec[j] = v
        # theta >= 0 is rho_hat's own cut, kept as a bound rather than a row: with a
        # free theta and the row theta >= 0 the master LP was the stochastic one plus
        # a free column, yet at n=60, |Omega|=5, r=0 each primal simplex solve took
        # 3-4x the iterations. rho_hat's weight in rho_bar is theta's reduced cost.
        self.theta = self.lp.add_col(1.0, 0.0, self.lp.INF, [], [])

    def _add_cut(self, rho):
        """A distribution cut over every column so far (and theta)."""
        handles, coefs = [], []
        d = np.asarray(rho, float) - self._ref
        for j, v in self._cvec.items():        # every live column (purge prunes it)
            a = float(d @ v)
            if a:
                handles.append(j)
                coefs.append(a)
        handles.append(self.theta)
        coefs.append(-1.0)
        self.cut_rows.append(self.lp.add_row_with(-self.lp.INF, 0.0, handles, coefs))
        self.cut_rho.append(np.asarray(rho, float))

    kl_master = 'cut'

    def _model_worst(self, xval, c):
        """The RMP's own value of the worst case at a solution with scenario costs c."""
        return float(self._ref @ c) + xval(self.theta)

    def _refine(self, rho):
        self._add_cut(rho)
        self._dual_next = True

    def _n_cuts(self):
        return len(self.cut_rows) + 1       # rho_hat's cut is theta's bound

    def _col_entries(self, col):
        obj, rows, coefs = super()._col_entries(col)
        v = np.asarray(col.scen, float)         # first stage: same in every scenario
        for r, rho in zip(self.cut_rows, self.cut_rho):
            a = float((rho - self._ref) @ v)
            if a:
                rows.append(r)
                coefs.append(a)
        return obj, rows, coefs

    def _add_one(self, col):
        h = super()._add_one(col)
        self._cvec[h] = col.first + np.asarray(col.scen, float)
        return h

    def _purge(self, ref):
        n = super()._purge(ref)
        if n:
            live = set(self.x0.values()) | set(self.y.values()) | {
                h for u in self.units for h in self.col_idx[u]}
            self._cvec = {j: v for j, v in self._cvec.items() if j in live}
        return n

    # --- duals and bounds ----------------------------------------------------
    def _rho_bar(self):
        """sum_k w_k rho^k + (1 - sum_k w_k) rho_hat, w_k = -mu_k; the last weight is
        theta's reduced cost (its bound theta >= 0 is rho_hat's cut)."""
        if not self.cut_rows:
            return np.array(self.probs)
        w = np.maximum(np.array([-self.lp.pi(r) for r in self.cut_rows]), 0.0)
        if w.sum() > 1.0:
            w = w / w.sum()
        rho = w @ np.array(self.cut_rho) + (1.0 - w.sum()) * self._ref
        return rho / rho.sum()

    def _duals(self):
        duals, conv = super()._duals()
        for w, r in enumerate(self._rho_bar()):
            duals[rho_key(w)] = float(r)
        return duals, conv

    @staticmethod
    def rho_of(duals, S):
        return np.array([duals[rho_key(w)] for w in range(S)])

    def _col_cost(self, col, duals):
        return col.first + float(self.rho_of(duals, len(self.scenarios)) @ col.scen)

    def _lagrangian(self, duals, bounds):
        rho = self.rho_of(duals, len(self.scenarios))
        for k, rows in self.x0_rows.items():
            cost = self.x0_first[k] + sum(rho[w] * c for w, c in self.x0_scen[k].items())
            rc = cost - sum(duals.get(r, 0.0) * a for r, a in rows)
            if rc < -1e-9 * (1.0 + abs(cost)):
                return -np.inf
        if any(v > 1e-9 for (kind, _, _), v in duals.items()
               if kind not in CARRIERS and kind != 'rho'):
            return -np.inf
        return sum(bounds)

    def _scen_costs(self, xval):
        c = np.zeros(len(self.scenarios))
        for j, v in self._cvec.items():
            x = xval(j)
            if x > 1e-12:
                c += x * v
        return c

    def _separate_dist(self, lp_obj):
        """Evaluate psi at the RMP solution; add the tilted distribution as a cut if
        theta falls short. Caches psi for the upper bound. Returns cuts added."""
        c = self._scen_costs(self.lp.x)
        psi, rho, eta = kl_worst(c, self.probs, self.kl_radius)
        theta = self._model_worst(self.lp.x, c)         # the RMP's worst case
        self._psi_main = psi
        viol = psi - theta
        if viol > self.kl_cut_frac * self._gap_tol(lp_obj):
            self._refine(rho)
            self.kl_log.append({'iter': self.iteration, 'theta': theta, 'psi': psi,
                                'viol': viol, 'eta': eta, 'kl': kl_div(rho, self.probs),
                                'shadow': False})
            return 1
        return 0

    def _update_ub(self, lp_obj=None):
        """UB = psi(cost of an unpenalized RMP solution with no grid trade).

        An RMP solution that trades with the grid bounds nothing. Rather than switch
        the DOIs off (and fall back into the degenerate master, which at n=60 cost
        more than the rest of the run), re-solve the unpenalized twin with the grid
        columns held at 0: a feasible plan of the original master over the columns
        so far, whose worst case is an upper bound.
        """
        t0 = time.time()
        if not self._penalized() and self._y_total() <= 1e-7:
            if lp_obj is None:
                lp_obj = self.lp.solve()
                self._separate_dist(lp_obj)
            ub = self._psi_main if self._y_total() <= 1e-7 else self._ub_no_grid()
        else:
            ub = self._ub_no_grid()
            self.n_ub += 1
        self.t_lp += time.time() - t0
        if ub < self.ub:
            self.ub = ub
        return ub

    ub_with_y = True

    def _grid_value(self, lp_obj):
        return self._psi_main           # set by _separate_dist at this solution

    def _ub_no_grid(self):
        """psi at the unpenalized RMP optimum over the current columns with y = 0 (the
        twin LP when there is one, else the master itself); inf if infeasible. Adds
        the tilted distribution as a cut if the twin's theta falls short of it."""
        twin = isinstance(self.lp, _TwinLP)
        lp = self.lp.shadow if twin else self.lp
        js = list(self.y.values()) if self.doi_active else []
        if js:
            lp.set_cols(js, ub=[0.0] * len(js))
        try:
            lp.solve()
            c = self._scen_costs(lp.x)
            ub, rho, eta = kl_worst(c, self.probs, self.kl_radius)
            theta = self._model_worst(lp.x, c)
        except RuntimeError:        # not yet feasible without slacks or grid trade
            ub = np.inf
        finally:
            if js:
                lp.set_cols(js, ub=[lp.INF] * len(js))
        if np.isfinite(ub) and ub - theta > self.kl_cut_frac * self._gap_tol(ub):
            # the plan's worst case is a valid cut for the main LP too
            self._refine(rho)
            self.kl_log.append({'iter': self.iteration, 'theta': theta, 'psi': ub,
                                'viol': ub - theta, 'eta': eta,
                                'kl': kl_div(rho, self.probs), 'shadow': True})
        self.n_ub_no_grid = getattr(self, 'n_ub_no_grid', 0) + 1
        return ub

    def _final_ub(self, obj):
        if self._y_total() <= 1e-7:
            psi = kl_worst(self._scen_costs(self.lp.x), self.probs, self.kl_radius)[0]
            if psi < self.ub:
                self.ub = psi

    def solve(self, init_vals=None, init_cols=None, init_duals=None):
        res = super().solve(init_vals, init_cols, init_duals)
        S = len(self.scenarios)
        rho = self.rho_of(res['duals'], S)
        # the final (unpenalized) RMP solution: its scenario costs and worst case
        c = self._scen_costs(self.lp.x)
        psi, rho_x, _ = kl_worst(c, self.probs, self.kl_radius)
        # converged on the master with grid columns: report its value (the seed's
        # y = 0 bound, still in self.ub, is far above it)
        ub = self.ub_doi_master if self.doi_master_certified else self.ub
        res['obj'] = ub
        res['gap'] = (ub - self.lb) / (1.0 + abs(ub))
        res['kl'] = {'radius': self.kl_radius, 'nested': self.kl_nested,
                     'rho_star': rho.tolist(), 'kl_rho_star': kl_div(rho, self.probs),
                     'rho_primal': rho_x.tolist(), 'psi_final': psi,
                     'expected_cost_final': float(np.array(self.probs) @ c),
                     'theta_final': self._model_worst(self.lp.x, c),
                     'master': self.kl_master,
                     'cuts': self._n_cuts(), 'cut_log': self.kl_log,
                     'ub_no_grid_solves': getattr(self, 'n_ub_no_grid', 0),
                     'seeded': self.kl_seeded}
        return res


class KLDualMaster(KLMaster):
    """The KL master in the dual form of the inner max (Love & Bayraksan, eq. 9):

        min  mu + r kappa + sum_w rho_hat_w t_w
        s.t. sum_{j,q} lambda_jq h^q_w + (x0, y costs)_w - H_w = 0        [nu_w]
             e^s (H_w - mu) + (e^s - 1 - s e^s) kappa - t_w <= 0   for s in S_w  [xi]
             -kappa - t_w <= 0                                             (s -> -inf)
             linking rows, convexity rows as in DirectMaster;  kappa >= 0

    kappa is the multiplier of the KL constraint (lambda in the paper; renamed so it
    does not clash with the column weights). The tangent planes of the perspective
    kappa (exp((H - mu)/kappa) - 1) are valid everywhere, so a grid of them goes in at
    the start, one set per scenario: the master sees the whole ball from the first
    iteration instead of one distribution at a time. Every column carries its cost
    in the H rows only (objective 0).

      pricing  the H rows' duals give the measure: rho_bar_w = -nu_w. The dual
               constraints of H, t and mu make it rho_hat_w times a convex
               combination of the grid ratios e^s, summing to 1, and kappa's makes
               the interpolated divergence <= r, hence rho_bar lies in the ball.
      refine   at the RMP solution, psi(h) exactly (kl_worst); if the master's
               value falls short by more than kl_cut_frac of the tolerance, add the
               tangents at the worst case's own ratios s_w = ln(p_w / rho_hat_w).
    LB and UB as in KLMaster.
    """
    kl_master = 'dual'

    def _obj0(self, cost):
        return 0.0

    def _build(self):
        DirectMaster._build(self)
        lp, INF = self.lp, self.lp.INF
        S = len(self.scenarios)
        self._ref = np.array(self.probs)
        for nm, j in self.x0.items():
            v = np.full(S, self.x0_first[nm])
            for w, c in self.x0_scen[nm].items():
                v[w] += c
            self._cvec[j] = v
        for key, j in self.y.items():
            v = np.zeros(S)
            w, c = self.y_cost[key]
            v[w] = c
            self._cvec[j] = v
        self.mu = lp.add_col(1.0, -INF, INF, [], [])
        self.kappa = lp.add_col(self.kl_radius, 0.0, INF, [], [])
        self.t = [lp.add_col(self._ref[w], -INF, INF, [], []) for w in range(S)]
        self.hrow = []
        for w in range(S):
            hs = [j for j, v in self._cvec.items() if v[w]]
            self.hrow.append(lp.add_row_with(0.0, 0.0, hs, [self._cvec[j][w] for j in hs]))
        self.H = [lp.add_col(0.0, -INF, INF, [self.hrow[w]], [-1.0]) for w in range(S)]
        self.n_tangents = 0
        for w in range(S):
            lp.add_row_with(-INF, 0.0, [self.kappa, self.t[w]], [-1.0, -1.0])
            self._tangents(w, np.arange(-8.0, np.log(1.0 / self._ref[w]) + 0.25, 0.25))
        self.theta = None

    def _tangents(self, w, ss):
        for s_ in ss:
            a, b = _kl_tangent(s_)
            self.lp.add_row_with(-self.lp.INF, 0.0, [self.H[w], self.mu, self.kappa, self.t[w]],
                                 [a, -a, b, -1.0])
            self.n_tangents += 1

    def _col_entries(self, col):
        _, rows, coefs = DirectMaster._col_entries(self, col)
        v = col.first + np.asarray(col.scen, float)
        for w, r in enumerate(self.hrow):
            if v[w]:
                rows.append(r)
                coefs.append(float(v[w]))
        return 0.0, rows, coefs

    def _model_worst(self, xval, c):
        return (xval(self.mu) + self.kl_radius * xval(self.kappa)
                + float(sum(self._ref[w] * xval(self.t[w]) for w in range(len(self.t)))))

    def _refine(self, rho):
        self._dual_next = True
        for w in np.nonzero(rho > 0)[0]:
            self._tangents(w, [float(np.log(rho[w] / self._ref[w]))])

    def _n_cuts(self):
        return self.n_tangents

    def _rho_bar(self):
        rho = np.maximum(np.array([-self.lp.pi(r) for r in self.hrow]), 0.0)
        if rho.sum() <= 0.0:
            return np.array(self.probs)
        return rho / rho.sum()


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
        kw['doi'] = False               # its QP has no grid columns
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


def solve_dwr_direct(players, T, scenarios, params, init_vals=None, dual_init=None,
                     ef=None, **kw):
    """(DWR_N^Omega) through DirectMaster: our own loop, HiGHS or Gurobi, no SCIP.
    dual_init 'lp' or 'fix' warm-starts the duals from the extensive form `ef`."""
    master = DirectMaster(players, T, scenarios, params, **kw)
    duals = master.duals_from_ef(ef, fix_binaries=dual_init == 'fix') if dual_init else None
    return master.solve(init_vals=init_vals, init_duals=duals), master


def solve_dwr_kl(players, T, scenarios, params, init_vals=None, kl_radius=0.0,
                 kl_nested=False, kl_cut_frac=0.05, doi_markup=0.0, kl_master='cut',
                 kl_seed=None, **kw):
    """The KL-ball DRO master, no exp cone: distribution cuts (KLMaster, 'cut') or
    the dual form with tangent planes (KLDualMaster, 'dual')."""
    cls = {'cut': KLMaster, 'dual': KLDualMaster}[kl_master]
    master = cls(players, T, scenarios, params, kl_radius=kl_radius,
                      kl_nested=kl_nested, kl_cut_frac=kl_cut_frac,
                      doi_markup=doi_markup, kl_seed=kl_seed, **kw)
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


def robust_allocation(ef, dw, master):
    """Robust Owen solution of the KL master, profit convention.

    owen_j = -sigma_j = max_{x_j} {E_rho*[f_j] - pi*^T A_j x_j} at the dual point that
    attains LB (robust_core.md, Proposition: robust Owen), rho* = rho_bar there.
    omega^LR,rob = v^LR,rob - v^MIP,rob = EF value - sum_j sigma_j, and
    Ex_j = owen_j - omega/n is budget balanced against v^MIP,rob(N) and lies in the weak
    eps-core with eps = omega/n. The certified interval of omega is
    [EF bound - UB, EF value - LB]. No scenario-wise allocation (Algorithm S1): its
    robust (contingent) version is an open question of the notes.
    """
    players, n = master.players, len(master.players)
    sigma = dw['sigma']
    owen = {u: -sigma[u] for u in players}
    omega = ef['obj'] - sum(sigma.values())
    S = len(master.scenarios)
    rho = KLMaster.rho_of(dw['duals'], S)
    return {
        'owen': owen, 'Ex': {u: owen[u] - omega / n for u in players},
        'omega_LR': omega, 'eps_LR': omega / n,
        'omega_interval': [ef['dual_bound'] - dw['ub'], ef['obj'] - dw['lb']],
        'rho_star': rho.tolist(),
        'budget_residual': abs(sum(owen[u] - omega / n for u in players) + ef['obj']),
        'duality_residual': omega - (ef['obj'] - dw['lb']),
        'pricing_residual': {u: 0.0 for u in players},
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


def _configure_direct(args):
    """DirectMaster's class-level options, and the purge defaults that depend on the
    scenario split, from the parsed command line (mutates both)."""
    if args.fallback_purge_age is not None:
        DirectMaster.fallback_purge_age = args.fallback_purge_age
    if args.doi_taper_steps is not None:
        DirectMaster.doi_taper_steps = args.doi_taper_steps
    if args.accept_doi_master:
        DirectMaster.accept_doi_master = True
    if args.lp_mixed:
        DirectMaster.lp_mixed = True
    DirectMaster.doi_certify = args.doi_certify
    if args.stall_reset is not None:
        DirectMaster.stall_reset = args.stall_reset
    if args.doi_repair:
        DirectMaster.doi_repair = tuple(float(f) for f in args.doi_repair.split(','))
    DirectMaster.split_scenarios = args.split_scenarios
    DirectMaster.mp12 = args.mp12
    # the split units carry 1/|Omega| of a plan each: purge them harder
    if args.purge_age is None:
        args.purge_age = 20 if args.split_scenarios else 50
    if args.purge_cap is None:
        args.purge_cap = 5 if args.split_scenarios else 40
    if args.doi_skip:
        DirectMaster.doi_skip = frozenset(tuple(x.split(':')) for x in args.doi_skip.split(','))


def _direct_kwargs(args, ef):
    """The column-generation options run() hands the direct engine."""
    return dict(
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
        lazy_kinds=tuple(k for k in args.lazy_rows.split(',') if k),
        doi=args.doi and not args.bundle,
        column_pool=args.column_pool, mip_start=args.mip_start,
        omega_tol=args.omega_tol, pricing_abs=args.pricing_abs,
        balance_pricing=args.balance_pricing)


# the stochastic rows' kinds under the names chp.ColumnGenerationSolver reports them by
_PRICE_NAMES = {'E': 'electricity', 'H': 'heat', 'G': 'hydrogen',
                'up': 'reserve_up', 'dn': 'reserve_dn', 'peak': 'peak'}


def solve_deterministic(players, T, params, ef_gap=None, omega_tol=None, time_limit=None):
    """The deterministic model through this engine: one scenario, the instance as given.

    Replaces LocalEnergyMarket + chp.ColumnGenerationSolver in the multi-day sweep.
    Every option is the command line's default except two. omega_tol is off (None):
    the 2% early stop is meant for |Omega| > 1, and with it omega^LR and the Owen
    point moved by up to 2% at |Omega| = 1, while CG_GAP alone reproduces chp.py's
    v^LR and its Owen point (to the degeneracy of the master duals). ef_gap
    defaults to the command line's EF gap.

    Returns the extensive form's dispatch as solve_and_extract_results lays it out,
    v^MIP and v^LR (cost), the Owen point sigma and its gap-corrected allocation
    (cost, as chp.compute_owen_allocation), the coupling prices (as chp's
    convex_hull_prices) and the timings.
    """
    from compact_utility import extract_results_by_name
    args = build_parser().parse_args([])
    args.omega_tol = omega_tol
    args.mip_time_limit = time_limit
    args.ef_gap = args.mip_gap if ef_gap is None else ef_gap
    _configure_direct(args)
    scen = [(1.0, dict(params))]

    t0 = time.time()
    ef = solve_extensive_form(players, T, scen, time_limit=time_limit, gap=args.ef_gap,
                              solver=args.mip_solver)
    t_mip = time.time() - t0
    t0 = time.time()
    dw, master = solve_dwr_direct(players, T, scen, params, **_direct_kwargs(args, ef))
    t_cg = time.time() - t0
    al = scenario_allocation(ef, dw, master)

    v_mip, v_lr = float(ef['obj']), float(dw['obj'])
    sigma = {u: float(dw['sigma'][u]) for u in players}
    prices = {}
    for (kind, t, w), d in dw['duals'].items():
        if kind in _PRICE_NAMES:
            prices.setdefault(_PRICE_NAMES[kind], {})[t] = abs(d / master.probs[w])
    return {
        'results': extract_results_by_name(ef['stack'].blocks[0].model, ef['vals']),
        'v_mip': v_mip, 'v_lr': v_lr, 'lb': float(dw['lb']),
        'ef_status': ef['status'], 'ef_gap': ef['gap'], 'cg_status': dw['status'],
        'cg_iterations': dw['iterations'],
        'sigma': sigma, 'owen': {u: -float(al['Ex'][u]) for u in players},
        'gap': v_lr - v_mip, 'eps': abs(v_lr - v_mip) / len(players),
        'convex_hull_prices': prices, 'time_mip': t_mip, 'time_cg': t_cg,
    }


def run(args):
    if args.engine != 'direct':
        raise SystemExit(f"--engine {args.engine}: SCIP is not a supported solver here; use 'gurobi' or 'highs' (use --engine direct)")
    sys.path.insert(0, os.path.join(_PAPER, 'weak_eps_experiment'))
    from run_experiment import build_instance
    players, _, T, base, name = build_instance(args.n, day=args.day)
    base['grid_caps'] = args.grid_caps
    if args.reserve_mode is not None:
        base['reserve_mode'] = args.reserve_mode
    from run_experiment import RESERVE_BLOCK_HOURS as _BLK_DEFAULT
    if args.reserve_block_hours is not None:
        base['reserve_block_hours'] = args.reserve_block_hours
    if args.reserve_price_scale is not None and 'pi_res_t' in base:
        # the hourly series scaled to another level, shape kept (sensitivity; tag _rsc<f>)
        base['pi_res_t'] = {t: args.reserve_price_scale * v for t, v in base['pi_res_t'].items()}
        base['pi_res'] = base['pi_up'] = base['pi_dn'] = float(np.mean(list(base['pi_res_t'].values())))
    if args.reserve_price_flat is not None:
        # a flat price instead of the hourly series (sensitivity; tag _res<p>)
        base.pop('pi_res_t', None)
        base.pop('reserve_price_source', None)
        base['pi_res'] = base['pi_up'] = base['pi_dn'] = float(args.reserve_price_flat)
    if args.day is not None:
        name = f'{name}_day{args.day}'
    scen = make_scenarios(base, players, T, args.scenarios, seed=args.seed,
                          wind_sigma=args.wind_sigma, solar_sigma=args.solar_sigma,
                          load_sigma=args.load_sigma, price_sigma=args.price_sigma,
                          rho=args.rho,
                          price_carriers=tuple(args.price_carriers.split(',')),
                          load_carriers=tuple(args.load_carriers.split(',')))
    kl = args.kl_radius is not None
    mip_kw = dict(time_limit=args.mip_time_limit, gap=args.mip_gap,
                  solver=args.mip_solver)
    if kl:
        mip_kw['kl_radius'] = args.kl_radius
    tag = f'{name}_S{args.scenarios}_seed{args.seed}' + ('' if args.doi else '_nodoi') \
        + ('_sar' if args.sar else '') + ('_nosmooth' if args.no_smoothing else '') \
        + ('_grb' if args.pricing_solver == 'gurobi' else '') \
        + ('_nopen' if args.no_penalty else '') \
        + (f'_direct-{args.lp_solver}' if args.engine == 'direct' else '') \
        + (f'_kl{args.kl_radius:g}' + ('_nested' if args.kl_nested else '')
           + ('_dual' if args.kl_master == 'dual' else '') if kl else '') \
        + ('_accdoi' if args.accept_doi_master else '') \
        + ('_ndw' if args.split_scenarios and args.mp12 else
           '_split' if args.split_scenarios else '_mp12' if args.mp12 else '') \
        + ('_lpmix' if args.lp_mixed else '')         + ('_repair' if args.doi_repair else '')         + ('_legacycaps' if args.grid_caps == 'legacy' else '') \
        + ('_seed' if kl and args.kl_seed_ef else '') \
        + ('_hard' if base.get('reserve_mode') == 'hard' else '') \
        + (f'_res{args.reserve_price_flat:g}' if args.reserve_price_flat is not None else '') \
        + (f'_rsc{args.reserve_price_scale:g}' if args.reserve_price_scale is not None else '') \
\
        + (f"_blk{base['reserve_block_hours']}"
           if base.get('reserve_block_hours', _BLK_DEFAULT) != _BLK_DEFAULT else '') \
        + (f'_{args.tag}' if args.tag else '')
    print(f'\n=== {tag}: n={len(players)}, |T|={len(T)}, |Omega|={len(scen)} ===')
    _configure_direct(args)

    if args.deterministic_check:
        det = LocalEnergyMarket(players, T, scen[0][1], model_type='mip')
        det.model.hideOutput()
        det.solve()
        print(f'deterministic model on scenario 0: {det.model.getObjVal():.6f}')

    print('\n[1] extensive form (DP_N^Omega)')
    if args.ef_cache and os.path.exists(args.ef_cache):
        # the same instance's EF from an earlier run (another CG configuration):
        # everything but the SCIP stack, which nothing downstream needs
        import pickle
        with open(args.ef_cache, 'rb') as f:
            ef = pickle.load(f)
        ef['cached'] = True
        print(f'  (read from {args.ef_cache})')
    else:
        ef = solve_extensive_form(players, T, scen, log_file=args.ef_log, **mip_kw)
        ef['n_vars'] = ef['stack'].model.getNVars()
        ef['first_stage_names'] = sorted(ef['stack'].first_stage)
        if args.ef_cache:
            import pickle
            os.makedirs(os.path.dirname(os.path.abspath(args.ef_cache)), exist_ok=True)
            with open(args.ef_cache, 'wb') as f:
                pickle.dump({k: v for k, v in ef.items() if k != 'stack'}, f)
    print(f'  obj {ef["obj"]:.6f}  status {ef["status"]}  gap {ef["gap"]:.2e}  '
          f'{ef["time_solve"]:.1f}s  vars {ef["n_vars"]}')
    if kl:
        k = ef['kl']
        print(f'  KL r={k["radius"]:g}: worst case {ef["obj"]:.6f} vs expected '
              f'{k["expected_cost"]:.6f}; {k["solves"]} MILP solve(s); rho* '
              + ' '.join(f'{r:.3f}' for r in k['rho']) + f' (KL {k["kl"]:.4f})')
        for i, rd in enumerate(k['refinements']):
            print(f'    round {i + 1}: MIPGap {rd.get("mip_gap", float("nan")):.0e}  '
                  f'{rd["time"]:.1f}s  {rd["nodes"]:.0f} nodes  psi-bound '
                  f'{(rd["psi"] - rd["bound"]) / max(abs(rd["psi"]), 1.0):.2e}  '
                  f'psi-incumbent {(rd["psi"] - rd["obj_model"]) / max(abs(rd["psi"]), 1.0):.2e}')
    if args.ef_only:
        return None

    print('\n[2] column generation (DWR_N^Omega)')
    if args.engine == 'direct':
        solver_fn = solve_dwr_bundle if args.bundle else solve_dwr_direct
        if kl:
            if args.bundle or args.dual_init != 'none':
                raise SystemExit('--kl-radius does not combine with --bundle or --dual-init')
            solver_fn = functools.partial(solve_dwr_kl, kl_radius=args.kl_radius,
                                          kl_nested=args.kl_nested,
                                          kl_cut_frac=args.kl_cut_frac,
                                          doi_markup=args.doi_markup,
                                          kl_master=args.kl_master,
                                          kl_seed=[ef['kl']['rho']] if args.kl_seed_ef else None)
        extra_d = ({'dual_init': args.dual_init, 'ef': ef}
                   if args.dual_init != 'none' and not args.bundle else {})
        extra_b = ({'bundle_t': args.bundle_t, 'bundle_age': args.bundle_age,
                    'bundle_t_min': args.bundle_t_min, 'bundle_cap': args.bundle_cap,
                    'bundle_qp': args.bundle_qp} if args.bundle else {})
        dw, master = solver_fn(players, T, scen, base, **_direct_kwargs(args, ef),
                               **extra_b, **extra_d)
        tm = dw['timing']
        if dw.get('bundle'):
            print(f'  bundle: {tm["serious"]} serious / {tm["null"]} null steps, '
                  f'QP {tm["qp"]:.1f}s, {tm["dropped"]} cuts dropped, '
                  f'{tm["aggregated"]} aggregated')
        print(f'  obj {dw["obj"]:.6f}  LB {dw["lb"]:.6f}  gap {dw["gap"]:.2e}  '
              f'iters {dw["iterations"]}  {dw["time"]:.1f}s  '
              f'rounds {len(dw["penalty"]["rounds"])}  status {dw["status"]}')
        if kl:
            k = dw['kl']
            print(f'  KL r={k["radius"]:g}: {k["cuts"]} '
                  + ('tangent planes' if k['master'] == 'dual' else
                     'distribution cuts (rho_hat included)') + '; rho* '
                  + ' '.join(f'{r:.3f}' for r in k['rho_star'])
                  + f' (KL {k["kl_rho_star"]:.4f})')
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

    if kl:
        print('\n[3] robust Owen allocation')
        al = robust_allocation(ef, dw, master)
        print(f'  omega_LR,rob {al["omega_LR"]:.6f}  eps_LR {al["eps_LR"]:.6f}  '
              f'certified [{al["omega_interval"][0]:.6f}, {al["omega_interval"][1]:.6f}]')
    else:
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

    first = {n: ef['vals'][n] for n in ef['first_stage_names']}
    out = {
        'instance': name, 'n': len(players), 'T': len(T), 'scenarios': len(scen),
        'probs': master.probs,
        'config': {k: getattr(args, k) for k in ('seed', 'wind_sigma', 'solar_sigma', 'load_sigma',
                                                   'price_sigma', 'rho', 'price_carriers',
                                                   'mip_gap', 'mip_time_limit')},
        'm_linking_rows': len(master.row_keys),
        'ef': {**{k: ef[k] for k in ('status', 'obj', 'dual_bound', 'gap', 'first_cost',
                                     'scen_cost', 'worth_cost', 'time_build', 'time_solve')},
               'cached': ef.get('cached', False)},
        'ef_first_stage': {k: v for k, v in first.items() if abs(v) > 1e-9},
        'dw': {k: dw.get(k) for k in ('status', 'obj', 'lb', 'ub', 'gap', 'sigma', 'x0',
                                      'iterations', 'columns', 'time', 'doi', 'timing')},
        'cg_options': {k: getattr(args, k) for k in (
            'lp_solver', 'pricing_solver', 'mip_solver', 'cg_gap', 'pricing_gap',
            'no_penalty', 'no_smoothing', 'pen_eps', 'pen_delta', 'pen_shrink',
            'max_rounds', 'pricing_workers', 'round_tol', 'ub_every', 'lp_method',
            'purge_every', 'purge_age', 'purge_cap', 'lp_presolve', 'cold_start',
            'bundle', 'bundle_t', 'bundle_t_min', 'bundle_age', 'bundle_cap', 'bundle_qp',
            'dual_init', 'omega_tol', 'pricing_abs', 'kl_radius', 'kl_nested', 'lp_mixed',
            'split_scenarios', 'mp12',
            'kl_cut_frac', 'kl_master')},
        'sar': dw.get('sar'),
        'penalty': dw.get('penalty'),
        'cg_log': master.log,
        'allocation': {k: al.get(k) for k in ('owen', 'Ex', 'x', 'g', 'worth', 'omega_LR',
                                              'eps_LR', 'budget_residual',
                                              'duality_residual', 'pricing_residual',
                                              'omega_interval', 'rho_star')},
        'kl': {'radius': args.kl_radius, 'nested': args.kl_nested,
               'ef': ef.get('kl'), 'master': dw.get('kl')} if kl else None,
        'standalone': alone, 'chi_LR': chi_lr,
        'weak_eps': core,
        # a row's price: its dual over the pricing measure's weight of the scenario
        'coupling_prices': {f'{k}_{t}_s{w}': d / (dw['duals'][rho_key(w)] if kl
                                                  else master.probs[w])
                            for (k, t, w), d in dw['duals'].items()
                            if k != 'rho' and (not kl or dw['duals'][rho_key(w)] > 0)},
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


def build_parser():
    """The command line. solve_deterministic reads its defaults from here too, so the
    one-scenario run and the CLI cannot drift apart."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=6, help='community size (6, 15, 30, 60)')
    ap.add_argument('--scenarios', type=int, default=5)
    ap.add_argument('--day', type=int, default=None,
                    help='calendar day of the data (as run_multiday.py); default: the '
                         "instance's default day")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wind-sigma', type=float, default=0.25,
                    help='wind forecast error (std of the multiplicative factor)')
    # Until 2026-09-30 solar was drawn with --wind-sigma (0.25), not make_scenarios'
    # 0.20: every CLI run on an instance with solar members (n = 15, 30, 60) used that
    # scenario set. Result JSONs record solar_sigma from then on; one without it is
    # from before the fix.
    ap.add_argument('--solar-sigma', type=float, default=0.20,
                    help='solar forecast error (std of the multiplicative factor)')
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
    ap.add_argument('--mip-solver', default='gurobi', choices=['highs', 'gurobi'],
                    help='extensive form and stand-alone MILPs (DP_S^Omega)')
    ap.add_argument('--cg-time-limit', type=float, default=None)
    ap.add_argument('--cold-start', action='store_true',
                    help='seed the master from zero-dual pricing instead of the EF solution')
    ap.add_argument('--doi', dest='doi', action='store_true', default=True,
                    help='dual-optimal inequalities on the balance rows: the community '
                         'trading with the grid at market prices (default on)')
    ap.add_argument('--no-doi', dest='doi', action='store_false',
                    help='plain master, without the grid columns')
    ap.add_argument('--engine', default='direct', choices=['scip', 'direct'],
                    help="'direct': our own loop on a HiGHS or Gurobi LP; 'scip' "
                         '(SCIP master with its pricer plugin) is no longer supported')
    ap.add_argument('--lp-solver', default='gurobi', choices=['highs', 'gurobi'],
                    help='master LP solver for --engine direct')
    ap.add_argument('--pricing-solver', default='gurobi', choices=['highs', 'gurobi'])
    ap.add_argument('--pricing-gap', type=float, default=MIP_GAP,
                    help=f'relative gap of the pricing MILPs (default {MIP_GAP})')
    ap.add_argument('--pricing-workers', type=int, default=0,
                    help='prosumers priced in parallel (0: one per core)')
    ap.add_argument('--round-tol', type=float, default=None,
                    help='column admission tolerance of the penalty rounds before the '
                         'last (default: --cg-gap)')
    ap.add_argument('--max-rounds', type=int, default=12)
    ap.add_argument('--tag', default='', help='suffix for the output file name')
    ap.add_argument('--ef-log', default=None,
                    help='write the Gurobi log of the extensive form (bound, incumbent, '
                         'gap per node line) to this file')
    ap.add_argument('--reserve-block-hours', type=int, default=None,
                    help="reserve delivery block length in hours (a divisor of |T|); "
                         "default is run_experiment's RESERVE_BLOCK_HOURS (1, Nordic "
                         "FCR-N). Another value adds _blk<h> to the file name")
    ap.add_argument('--reserve-price-scale', type=float, default=None,
                    help='multiply the hourly reserve prices by this factor (sensitivity)')
    ap.add_argument('--reserve-price-flat', type=float, default=None,
                    help='flat reserve price [EUR/MW.h] instead of the hourly FCR-N series '
                         '(sensitivity; adds _res<p> to the file name)')
    ap.add_argument('--reserve-mode', choices=['hard', 'penalty'], default=None,
                    help="reserve shortfall treatment; default is run_experiment's "
                         "RESERVE_MODE ('penalty'). 'hard' adds _hard to the file name")
    ap.add_argument('--cg-gap', type=float, default=CG_GAP,
                    help='relative CG gap: stop once (UB - LB) <= cg_gap (1 + |UB|), UB the '
                         'unpenalized RMP value and LB the Lagrangian bound')
    ap.add_argument('--purge-every', type=int, default=10,
                    help='column management: purge every k iterations (0: never)')
    ap.add_argument('--purge-age', type=int, default=None,
                    help='iterations a column may sit unused before it can be purged '
                         '(default 20 with --split-scenarios, 50 without)')
    ap.add_argument('--purge-cap', type=int, default=None,
                    help='purge only while there are more than this many columns per '
                         'pricing unit (default 5 with --split-scenarios, 40 without)')
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
    ap.add_argument('--balance-pricing', dest='balance_pricing', action='store_true',
                    default=True,
                    help='deal the prosumers with commitment binaries out to the pricing '
                         'workers first, in a snake order (default on)')
    ap.add_argument('--no-balance-pricing', dest='balance_pricing', action='store_false')
    ap.add_argument('--mip-start', action='store_true',
                    help="start each pricing MILP from the prosumer's previous plan")
    ap.add_argument('--omega-tol', type=float, default=OMEGA_TOL,
                    help='also stop once UB - LB <= omega_tol * (EF value - LB), i.e. '
                         f'omega^LR to that relative precision (default {OMEGA_TOL})')
    ap.add_argument('--no-omega-tol', dest='omega_tol', action='store_const', const=None,
                    help='stop on --cg-gap alone')
    ap.add_argument('--pricing-abs', dest='pricing_abs', action='store_true', default=True,
                    help='pricing MILPs stop on an absolute gap of the current CG '
                         'tolerance / (2 n) (default on)')
    ap.add_argument('--no-pricing-abs', dest='pricing_abs', action='store_false',
                    help='pricing MILPs stop on the relative --pricing-gap instead')
    ap.add_argument('--column-pool', action='store_true',
                    help='keep purged columns in a pool and price the pool before the MILPs')
    ap.add_argument('--dual-init', default='none', choices=['none', 'lp', 'fix'],
                    help="dual warm start from the extensive form's LP duals: 'lp' "
                         "relaxes integrality, 'fix' fixes it at the EF solution")
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
    ap.add_argument('--kl-radius', type=float, default=None,
                    help='KL-ball DRO (robust_core.md sec. 2.3): worst expected cost over '
                         'KL(rho || rho_hat) <= r, by column-and-cut generation. 0 reproduces '
                         'the stochastic model. Needs --mip-solver gurobi')
    ap.add_argument('--kl-nested', action='store_true',
                    help='KL: finish the cut generation before each pricing pass '
                         '(default: interleaved, cut and price in the same pass)')
    ap.add_argument('--kl-cut-frac', type=float, default=0.05,
                    help='KL: add a cut once theta falls short of the worst case by this '
                         'share of the CG tolerance')
    ap.add_argument('--mp12', dest='mp12', action='store_true', default=True,
                    help='prosumers with first-stage commitment enter as a pattern column '
                         'plus one column per scenario, tied by sum lambda = mu (MP1-2; '
                         'default on)')
    ap.add_argument('--no-mp12', dest='mp12', action='store_false')
    ap.add_argument('--split-scenarios', dest='split_scenarios', action='store_true',
                    default=True,
                    help='price prosumers without first-stage variables one scenario at a '
                         'time: one column and one convexity row per (prosumer, scenario); '
                         'same master LP value, columns 1/|Omega| as dense (default on)')
    ap.add_argument('--no-split-scenarios', dest='split_scenarios', action='store_false')
    ap.add_argument('--lp-mixed', action='store_true',
                    help='master LP: dual simplex for the one solve after rows are added, '
                         'the penalty changes or the DOIs are switched off; primal otherwise')
    ap.add_argument('--kl-seed-ef', action='store_true',
                    help="KL: start the master with the robust EF's worst-case distribution "
                         'as a cut (tangents for --kl-master dual)')
    ap.add_argument('--accept-doi-master', action='store_true',
                    help='stop once the master with the grid columns has converged, '
                         'without switching them off for a y = 0 upper bound (LB, Owen '
                         'and eps stay certified; UB is then the DOI master value)')
    ap.add_argument('--doi-skip', default='',
                    help="grid columns to leave out, as carrier:side pairs, e.g. 'G:exp' "
                         "or 'G:exp,E:imp'")
    ap.add_argument('--stall-reset', type=int, default=None,
                    help='passes without progress before one unsmoothed pass (default 10; 0 off)')
    ap.add_argument('--no-doi-certify', dest='doi_certify', action='store_false',
                    help='UB only from RMP solutions without grid trade (no absorption '
                         'certificate; the whole switch-off at once)')
    ap.add_argument('--grid-caps', choices=('bnd_size', 'legacy'), default='bnd_size',
                    help="members' trade bounds: the manuscript's eq:bnd_size (default) or "
                         "the earlier shared import caps (sensitivity; tag _legacycaps)")
    ap.add_argument('--doi-repair', default='',
                    help='before the DOI switch-off, price every member once per '
                         'fraction at duals moved that fraction of the import price past '
                         "the price box on the rows with grid trade, e.g. '0.01,0.1'")
    ap.add_argument('--doi-taper-steps', type=int, default=None,
                    help='when the master settles with grid trade, shrink a budget on '
                         'it this many times before switching the DOIs off (default 0: off; did not help)')
    ap.add_argument('--fallback-purge-age', type=int, default=None,
                    help='before the DOI switch-off, purge prosumer columns unused for '
                         'this many passes (default 0: off; did not help, see DirectMaster)')
    ap.add_argument('--kl-master', default='cut', choices=['cut', 'dual'],
                    help="KL: master form. 'cut' = distribution cuts (robust_core.md "
                         "2.3); 'dual' = Love & Bayraksan's dual with tangent planes")
    ap.add_argument('--doi-markup', type=float, default=0.0,
                    help='KL: extra cost per unit on the grid (DOI) columns, a tie-break '
                         'against settling with grid trade (default 0: any markup slows '
                         'the master, see KLMaster)')
    ap.add_argument('--skip-standalone', action='store_true',
                    help='stop after Algorithm S1 (no stand-alone solves)')
    ap.add_argument('--no-smoothing', action='store_true',
                    help='plain Kelley column generation (no Wentges dual smoothing)')
    ap.add_argument('--check-core', action='store_true',
                    help='enumerate every coalition (small n only)')
    ap.add_argument('--deterministic-check', action='store_true',
                    help='also solve the plain model on scenario 0')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--ef-only', action='store_true',
                    help='solve (or read) the extensive form, write --ef-cache, and stop')
    ap.add_argument('--ef-cache', default=None,
                    help='pickle of the extensive-form result: read it if it exists, else '
                         'solve and write it (same instance only; the caller names it)')
    return ap


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()

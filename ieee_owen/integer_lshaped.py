"""
Integer L-shaped (Laporte-Louveaux) for the stochastic grand-coalition dispatch,
an EXPERIMENT against the extensive form (stochastic_extension.solve_extensive_form).

First stage: the electrolyzer states/transitions (binary) and r_sym (continuous in
the model). The L-L optimality cut needs a pure-binary first stage, so r_sym is
discretised here: r_sym_i = delta * sum_k 2^k b_ik, K bits per block, the grid
[0, r_max] with r_max from the continuous EF. The discretised EF (same grid) is the
reference the decomposition must reproduce; the continuous EF gives the loss.

The recourse keeps its integers (PWL segment binaries, the complementarity binaries
y_sto / y_dir), so Q_w is a MILP value function. Cuts, per scenario w (multi-cut):
  lp  Benders cut of the recourse LP relaxation, reduced costs of the fixed x
  sb  strengthened Benders (Zou-Ahmed-Sun): the same slope, intercept from the MILP
      min_{x in X, y} Q-cost - pi^T x with x free
  ll  the L-L cut theta_w >= (Q_w(x^) - L_w)(sum_S x - sum_notS x - |S| + 1) + L_w,
      Q_w(x^) the MILP's dual bound (valid), L_w a global dual bound
lp/sb cuts are added in a root loop on the master LP relaxation and at every
integer master solution; ll only at integer ones (lazy constraints, Gurobi).

    python ieee_owen/integer_lshaped.py --n 15 --scenarios 5 --bits 6
"""
import os, sys, time, json, argparse

_PAPER = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from stochastic_extension import (ScenarioStack, make_scenarios, _to_gurobi, EF_GAP)

TOL = 1e-6


def build(n, day, n_scen, seed):
    sys.path.insert(0, os.path.join(_PAPER, 'weak_eps_experiment'))
    from run_experiment import build_instance
    players, _, T, base, name = build_instance(n, day=day)
    base['grid_caps'] = 'bnd_size'
    scen = make_scenarios(base, players, T, n_scen, seed=seed)
    return players, T, scen, name


def stack_to_gurobi(players, T, scen, name, env=None):
    st = ScenarioStack(name, players, T, scen, dwr=False)
    g, gv = _to_gurobi(st.model, name, env=env)
    by_name = {v.name: v for v in st.model.getVars()}
    for n_, v in gv.items():
        v.Obj = by_name[n_].getObj()
    g.ObjCon = st.model.getObjoffset()
    g.update()
    return st, g, gv


def discretise(g, gv, rnames, r_max, bits):
    """r = delta * sum_k 2^k b_k, delta = r_max / (2^bits - 1). Returns {r: [(b, w)]}."""
    delta = r_max / (2 ** bits - 1)
    enc = {}
    for r in rnames:
        bs = [g.addVar(vtype=GRB.BINARY, name=f'{r}__b{k}') for k in range(bits)]
        g.addLConstr(gv[r] == gp.quicksum(delta * 2 ** k * b for k, b in enumerate(bs)),
                     name=f'{r}__enc')
        gv[r].UB = r_max
        enc[r] = [(b, delta * 2 ** k) for k, b in enumerate(bs)]
    g.update()
    return enc, delta


def solve_ef(players, T, scen, gap, tl, r_max=None, bits=None, quiet=True):
    st, g, gv = stack_to_gurobi(players, T, scen, 'EF')
    rnames = sorted(n for n in st.first_stage if n.startswith('r_sym_'))
    if r_max is not None:
        discretise(g, gv, rnames, r_max, bits)
    g.Params.OutputFlag = 0 if quiet else 1
    g.Params.MIPGap = gap
    if tl:
        g.Params.TimeLimit = tl
    t0 = time.time()
    g.optimize()
    return {'obj': g.ObjVal, 'bound': g.ObjBound, 'gap': g.MIPGap,
            'time': time.time() - t0, 'status': g.Status,
            'r': {r: gv[r].X for r in rnames},
            'x': {n: gv[n].X for n in st.first_stage}}


class Sub:
    """Scenario w's recourse: a one-scenario stack with the first stage fixed by bounds."""

    def __init__(self, players, T, scen_w, fs_names, gap, r_max, sb_gap, env=None,
                 threads=0):
        st, g, gv = stack_to_gurobi(players, T, [(1.0, scen_w)], 'sub', env=env)
        g.Params.Threads = threads
        self.fs = [n for n in fs_names]
        self.gap, self.sb_gap = gap, sb_gap
        for n in self.fs:
            gv[n].Obj = 0.0                 # first-stage cost lives in the master
            if n.startswith('r_sym_'):
                gv[n].UB = r_max            # the master's grid top
        g.Params.OutputFlag = 0
        g.Params.MIPGap = gap
        g.update()
        self.g, self.gv = g, gv
        self.lp = g.relax()
        self.lp.Params.OutputFlag = 0
        self.lp.Params.Threads = threads
        self.lpv = {n: self.lp.getVarByName(n) for n in self.fs}
        self.orig_bounds = {n: (gv[n].LB, gv[n].UB) for n in self.fs}
        self.t_mip = self.t_lp = self.t_sb = 0.0
        self.n_mip = self.n_lp = self.n_sb = 0

    def _fix(self, model, vars_, x):
        for n in self.fs:
            vars_[n].LB = vars_[n].UB = x[n]

    def _free(self, model, vars_):
        for n in self.fs:
            vars_[n].LB, vars_[n].UB = self.orig_bounds[n]

    def lower_bound(self, mode='lp'):
        """L_w, a global lower bound on Q_w: the recourse with the first stage free,
        its LP relaxation ('lp', cheap) or the MILP's dual bound ('mip')."""
        if mode == 'lp':
            for n in self.fs:
                self.lpv[n].LB, self.lpv[n].UB = self.orig_bounds[n]
            self.lp.optimize()
            return self.lp.ObjVal
        self._free(self.g, self.gv)
        self.g.Params.TimeLimit = GRB.INFINITY
        self.g.optimize()
        return self.g.ObjBound

    def enable_tied(self):
        """Evaluate the LP with x tied by rows x = x^ instead of fixed bounds, x keeping
        its original bounds. Optimal: the rows' duals are the cut slope. Infeasible
        (cut-and-project may cut a fractional x^ out of the projection): a Farkas
        feasibility cut on x."""
        self.lpt = self.lp.copy()
        self.lpt.Params.OutputFlag = 0
        self.lpt.Params.InfUnbdInfo = 1
        self.lpt.Params.DualReductions = 0
        self.lpt_by = {v.VarName: v for v in self.lpt.getVars() if v.VarName}
        self.trow = {}
        for n in self.fs:
            v = self.lpt_by[n]
            v.LB, v.UB = self.orig_bounds[n]
            self.trow[n] = self.lpt.addLConstr(v == 0.0)
        self.lpt.update()
        self.tied = True
        self.n_feas = 0

    def _farkas(self, x):
        """Feasibility cut sum_n lam_n x_n <= C from the Farkas certificate of lpt."""
        m = self.lpt
        cons = m.getConstrs()
        vs = m.getVars()
        lam = np.array(m.getAttr('FarkasDual', cons))
        sense = np.array(m.getAttr('Sense', cons))
        rhs = np.array(m.getAttr('RHS', cons))
        A = m.getA().tocsr()
        lb = np.array(m.getAttr('LB', vs), dtype=float)
        ub = np.array(m.getAttr('UB', vs), dtype=float)
        tol = 1e-9
        # orient lam so that lam^T A z >= lam^T b is implied by the rows
        if not (np.all(lam[sense == '>'] >= -tol) and np.all(lam[sense == '<'] <= tol)):
            lam = -lam
            if not (np.all(lam[sense == '>'] >= -tol) and np.all(lam[sense == '<'] <= tol)):
                return None
        d = A.T @ lam
        zmax = np.where(d > 0, ub, lb)
        if np.any(np.abs(d[np.abs(zmax) >= 1e20]) > tol):
            return None
        mx = float(np.dot(d[np.abs(d) > tol], zmax[np.abs(d) > tol]))
        tie = {self.trow[n].index: n for n in self.fs}
        rest = np.ones(len(cons), dtype=bool)
        rest[list(tie)] = False
        C = mx - float(np.dot(lam[rest], rhs[rest]))
        coef = {n: float(lam[i]) for i, n in tie.items() if abs(lam[i]) > tol}
        if sum(c * x[n] for n, c in coef.items()) <= C + 1e-7 * max(1.0, abs(C)):
            return None                          # not violated at x^: not a certificate
        return coef, C

    def lp_cut(self, x):
        """(q, pi) of an optimality cut, ('feas', coef, C) of a feasibility cut
        sum coef x <= C (tied mode, LP infeasible), or None."""
        t0 = time.time()
        if getattr(self, 'tied', False):
            for n in self.fs:
                self.trow[n].RHS = x[n]
            self.lpt.optimize()
            self.t_lp += time.time() - t0
            self.n_lp += 1
            if self.lpt.Status == GRB.INFEASIBLE:
                fc = self._farkas(x)
                if fc is None:
                    return None
                self.n_feas += 1
                return ('feas',) + fc
            if self.lpt.Status != GRB.OPTIMAL:
                return None
            return self.lpt.ObjVal, {n: self.trow[n].Pi for n in self.fs}
        self._fix(self.lp, self.lpv, x)
        self.lp.optimize()
        self.t_lp += time.time() - t0
        self.n_lp += 1
        if self.lp.Status != GRB.OPTIMAL:
            return None
        return self.lp.ObjVal, {n: self.lpv[n].RC for n in self.fs}

    # --- cut-and-project (Bodur-Dash-Gunluk-Luedtke 2017) -------------------------
    def _polyhedron(self):
        """The scenario LP relaxation over (x, y), x at its ORIGINAL bounds (the cuts
        must hold for every x), as G z >= g, bounds included as rows."""
        import scipy.sparse as sp
        lp = self.lp
        lp.update()
        vs = lp.getVars()
        A = lp.getA().tocsr()
        sense = np.array(lp.getAttr('Sense', lp.getConstrs()))
        rhs = np.array(lp.getAttr('RHS', lp.getConstrs()))
        lb = np.array(lp.getAttr('LB', vs), dtype=float)
        ub = np.array(lp.getAttr('UB', vs), dtype=float)
        pos = {v.VarName: i for i, v in enumerate(vs)}
        for n in self.fs:
            lb[pos[n]], ub[pos[n]] = self.orig_bounds[n]
        blocks, rh = [], []
        ge, le, eq = sense == '>', sense == '<', sense == '='
        blocks += [A[ge], -A[le], A[eq], -A[eq]]
        rh += [rhs[ge], -rhs[le], rhs[eq], -rhs[eq]]
        nv = len(vs)
        I = sp.identity(nv, format='csr')
        fl, fu = np.isfinite(lb) & (lb > -1e20), np.isfinite(ub) & (ub < 1e20)
        blocks += [I[fl], -I[fu]]
        rh += [lb[fl], -ub[fu]]
        G = sp.vstack(blocks).tocsr()
        return vs, pos, G, np.concatenate(rh), lb, ub

    def cut_and_project(self, x, k_max=5, frac_tol=0.02, viol_tol=1e-6):
        """Up to k_max lift-and-project cuts (Balas CGLP, trivial normalisation) on the
        scenario LP at (x, y^), y^ the LP response to x; first-stage splits first.
        The cuts go into self.lp for good. Returns the number added."""
        t0 = time.time()
        tied = getattr(self, 'tied', False)
        vs, pos, G, gr, lb, ub = self._polyhedron()
        if tied:
            # the point to cut: (x^, y^) of the tied LP; if x^ is already cut out, the
            # Farkas cut handles it
            res = self.lp_cut(x)
            if res is None or res[0] == 'feas':
                return 0
            z = np.array(self.lpt.getAttr('X', [self.lpt_by[v.VarName] for v in vs]))
        else:
            self._fix(self.lp, self.lpv, x)
            self.lp.optimize()
            if self.lp.Status != GRB.OPTIMAL:
                return 0
            z = np.array(self.lp.getAttr('X', vs))
            for n in self.fs:                        # the point, not the fixed bounds
                z[pos[n]] = x[n]
        binv = [i for i, v in enumerate(vs)
                if self.gv[v.VarName].VType == GRB.BINARY]
        fsset = set(self.fs)
        cand = [i for i in binv if frac_tol < z[i] < 1 - frac_tol]
        cand.sort(key=lambda i: (vs[i].VarName not in fsset, abs(z[i] - 0.5)))
        cand = cand[:k_max]
        if not cand:
            return 0
        m, nv = G.shape
        c = gp.Model()
        c.Params.OutputFlag = 0
        u = c.addMVar(m, lb=0.0)
        v = c.addMVar(m, lb=0.0)
        u0 = c.addVar(lb=0.0)
        v0 = c.addVar(lb=0.0)
        beta = c.addVar(lb=-GRB.INFINITY)
        GT = G.T.tocsr()
        # G^T (u - v) - (u0 + v0) e_j = 0 : row j gets the u0, v0 terms per split
        eqs = c.addConstr(GT @ u - GT @ v == np.zeros(nv))
        c.addConstr(beta - gr @ u <= 0)
        c.addConstr(beta - gr @ v - v0 <= 0)
        c.addConstr(u.sum() + v.sum() + u0 + v0 == 1)
        c.update()
        eq_rows = eqs.tolist() if hasattr(eqs, 'tolist') else list(eqs)
        Gz = G @ z                  # alpha z^ = u^T G z^ - u0 z_j
        added = 0
        for j in cand:
            c.chgCoeff(eq_rows[j], u0, -1.0)
            c.chgCoeff(eq_rows[j], v0, -1.0)
            c.setObjective(Gz @ u - z[j] * u0 - beta, GRB.MINIMIZE)
            c.optimize()
            if c.Status == GRB.OPTIMAL and c.ObjVal < -viol_tol:
                uu = u.X
                alpha = GT @ uu
                alpha[j] -= u0.X
                b = beta.X
                # drop tiny coefficients safely: relax beta by |a_i| max(|lb|,|ub|)
                keep = np.abs(alpha) > 1e-9
                for i in np.nonzero(~keep & (alpha != 0))[0]:
                    span = max(abs(lb[i]), abs(ub[i]))
                    if np.isfinite(span) and span < 1e20:
                        b -= abs(alpha[i]) * span
                    else:
                        keep[i] = True
                idx = np.nonzero(keep)[0]
                self.lp.addLConstr(gp.LinExpr(alpha[idx].tolist(), [vs[i] for i in idx]),
                                   GRB.GREATER_EQUAL, b)
                if tied:
                    self.lpt.addLConstr(gp.LinExpr(
                        alpha[idx].tolist(),
                        [self.lpt_by[vs[i].VarName] for i in idx]),
                        GRB.GREATER_EQUAL, b)
                added += 1
            c.chgCoeff(eq_rows[j], u0, 0.0)
            c.chgCoeff(eq_rows[j], v0, 0.0)
        self.lp.update()
        self.t_cp = getattr(self, 't_cp', 0.0) + time.time() - t0
        self.n_cp = getattr(self, 'n_cp', 0) + added
        return added

    def inner(self, pi):
        """min_{x in X, y} Q-cost - pi^T x, x free. Returns (dual bound, argmin x,
        Q-cost at the argmin); the bound is valid at any gap."""
        t0 = time.time()
        self._free(self.g, self.gv)
        for n in self.fs:
            self.gv[n].Obj = -pi.get(n, 0.0)
        self.g.Params.TimeLimit = GRB.INFINITY
        self.g.Params.MIPGap = self.sb_gap
        self.g.optimize()
        v = self.g.ObjBound
        xj = {n: self.gv[n].X for n in self.fs}
        cq = self.g.ObjVal + sum(pi.get(n, 0.0) * xj[n] for n in self.fs)
        self.g.Params.MIPGap = self.gap
        for n in self.fs:
            self.gv[n].Obj = 0.0
        self.t_sb += time.time() - t0
        self.n_sb += 1
        return v, xj, cq

    def sb_intercept(self, pi):
        return self.inner(pi)[0]

    def lagrangian(self, x, basis, max_it=8, rel_tol=1e-3, box=3.0):
        """Lagrangian cut at x (Chen-Luedtke restricted separation): pi = sum_i l_i
        basis_i, l maximising h(l) = v(pi(l)) + pi(l)^T x by Kelley's method in the
        box |l| <= box, started at the last basis vector (the SB cut). Returns
        (intercept v, pi) of the best cut theta >= v + pi^T x found."""
        m = len(basis)
        fs = self.fs
        lam = np.zeros(m)
        lam[-1] = 1.0
        km = gp.Model()
        km.Params.OutputFlag = 0
        lv = km.addVars(m, lb=-box, ub=box)
        eta = km.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        km.setObjective(eta, GRB.MAXIMIZE)
        best = (-np.inf, None, None)
        for it in range(max_it):
            pi = {n: sum(lam[i] * basis[i].get(n, 0.0) for i in range(m)) for n in fs}
            v, xj, cq = self.inner(pi)
            h = v + sum(pi[n] * x[n] for n in fs)
            if h > best[0]:
                best = (h, v, pi)
            # h(l) <= cq + pi(l)^T (x - xj): a cut of the concave h at l
            d = {n: x[n] - xj[n] for n in fs}
            coef = [sum(basis[i].get(n, 0.0) * d[n] for n in fs) for i in range(m)]
            km.addConstr(eta <= cq + gp.quicksum(coef[i] * lv[i] for i in range(m)))
            km.optimize()
            ub = eta.X
            if ub - best[0] <= rel_tol * max(1.0, abs(best[0])):
                break
            lam = np.array([lv[i].X for i in range(m)])
        return best[1], best[2]

    def mip(self, x):
        """(dual bound, incumbent) of Q_w(x)."""
        t0 = time.time()
        self._fix(self.g, self.gv, x)
        self.g.Params.TimeLimit = GRB.INFINITY
        self.g.optimize()
        self.t_mip += time.time() - t0
        self.n_mip += 1
        if self.g.SolCount == 0:
            return None
        return self.g.ObjBound, self.g.ObjVal


def integer_lshaped(players, T, scen, r_max, bits, cuts, sub_gap, tl, root_rounds=30,
                    quiet=True, root_cuts=('lp',), sb_gap=1e-4, keep=0, lag_basis=10,
                    lag_it=8, lw='lp', warm=0, warm_gap=1e-3, warm_tl=120, cp_k=5,
                    stall_tol=1e-4, stab=1.0, cp_rounds=30, workers=1):
    """keep > 0: partial decomposition (Crainic-Hewitt-Maggioni-Rei), scenarios
    0..keep-1 stay in the master as full MILP copies (weights p_w), the rest get theta_w.
    root_cuts: phases of the root loop in order, from lp, sb, lag; each phase runs
    until its bound stalls (or root_rounds).
    warm > 0: before branch and cut, up to `warm` rounds of solving the master MILP
    (gap warm_gap, time warm_tl) without the callback, evaluating its plan in every
    scenario and adding the L-L and callback cuts there as ordinary rows; the best
    plan becomes the MIP start."""
    t_start = time.time()
    kept = list(range(keep))
    outer = [w for w in range(len(scen)) if w not in kept]
    if kept:
        st, g, gv = stack_to_gurobi(players, T, [scen[w] for w in kept], 'master')
        fs = sorted(st.first_stage)
    else:
        # master = the first-stage part of a one-scenario model
        st, g, gv = stack_to_gurobi(players, T, scen[:1], 'master')
        fs = sorted(st.first_stage)
        fs_set = set(fs)
        for c in g.getConstrs():
            row = g.getRow(c)
            if any(row.getVar(k).VarName not in fs_set for k in range(row.size())):
                g.remove(c)
        for n_, v in list(gv.items()):
            if n_ not in fs_set:
                g.remove(v)
                del gv[n_]
        g.update()
    rnames = [n for n in fs if n.startswith('r_sym_')]
    enc, delta = discretise(g, gv, rnames, r_max, bits)
    bins = [gv[n] for n in fs if gv[n].VType == GRB.BINARY] + \
           [b for r in rnames for b, _ in enc[r]]
    probs = [p for p, _ in scen]
    # workers > 1: the scenario subproblems are solved that many at a time (threads;
    # Gurobi releases the GIL while it solves), one Gurobi environment per worker and
    # the scenarios dealt round-robin, each worker solving its own in sequence (as the
    # stochastic CG's pricing_workers)
    k = max(1, min(workers or (os.cpu_count() or 1), len(outer)))
    owner = {w: i % k for i, w in enumerate(outer)}
    envs = [None] * k
    pool = None
    if k > 1:
        from concurrent.futures import ThreadPoolExecutor
        envs = [gp.Env() for _ in range(k)]
        pool = ThreadPoolExecutor(k)
    per = max(1, (os.cpu_count() or 1) // k) if k > 1 else 0
    subs = {w: Sub(players, T, scen[w][1], fs, sub_gap, r_max, sb_gap,
                   env=envs[owner[w]], threads=per) for w in outer}

    def pmap(fn, ws):
        """{w: fn(w)} for w in ws, the work of each worker in sequence."""
        ws = list(ws)
        if pool is None or len(ws) < 2:
            return {w: fn(w) for w in ws}
        groups = {}
        for w in ws:
            groups.setdefault(owner[w], []).append(w)
        futs = [pool.submit(lambda grp: {w: fn(w) for w in grp}, grp)
                for grp in groups.values()]
        out = {}
        for f in futs:
            out.update(f.result())
        return out

    # x tied by rows: optimality cuts from their duals, Farkas feasibility cuts where a
    # fractional x^ has no LP recourse (complete recourse holds for integer x only)
    if True:
        for s_ in subs.values():
            s_.enable_tied()
    t0 = time.time()
    L = pmap(lambda w: subs[w].lower_bound(lw), outer)
    t_L = time.time() - t0
    theta = {w: g.addVar(lb=L[w], name=f'theta_{w}') for w in outer}
    g.update()
    g.setObjective(g.getObjective() + gp.quicksum(probs[w] * theta[w] for w in outer))
    g.update()
    xv = {n: gv[n] for n in fs}
    log = {'L': L, 't_L': t_L, 'root': [], 'incumbents': []}
    slopes = {w: [] for w in outer}          # Benders slopes per scenario (lag basis)

    def slope_expr(pi, vars_):
        return gp.quicksum(pi[n] * vars_[n] for n in fs if abs(pi[n]) > 1e-9)

    # --- root loop on the master LP relaxation, in phases -----------------------
    relaxed = g.relax()
    relaxed.Params.OutputFlag = 0
    rx = {n: relaxed.getVarByName(n) for n in fs}
    rth = {w: relaxed.getVarByName(f'theta_{w}') for w in outer}
    it = 0
    for phase in root_cuts:
        hist = []
        x_in, lam, flat, kelley = None, stab, 0, False
        for _ in range(cp_rounds if phase == 'cp' else root_rounds):
            relaxed.optimize()
            lb = relaxed.ObjVal
            x_out = {n: rx[n].X for n in fs}
            # in-out stabilisation (Ben-Ameur-Neto; Fischetti-Ljubic-Sinnl): separate at
            # a point between the master's x_out and a core point x_in that trails it;
            # lam = 1 is Kelley. Cuts are valid wherever they are computed; they are
            # kept only if x_out violates them.
            if lam < 1.0 and x_in is not None and not kelley:
                x = {n: lam * x_out[n] + (1 - lam) * x_in[n] for n in fs}
                x_in = {n: 0.5 * (x_in[n] + x_out[n]) for n in fs}
            else:
                x = x_out
                if x_in is None:
                    x_in = dict(x_out)
            was_kelley, kelley = kelley or lam >= 1.0, False
            added = 0

            def root_work(w):
                s = subs[w]
                if phase == 'cp':
                    s.cut_and_project(x, k_max=cp_k)
                res = s.lp_cut(x)
                if res is None or res[0] == 'feas':
                    return res, None, None
                q, pi = res
                slopes[w].append(pi)
                if phase in ('lp', 'cp'):
                    v = q - sum(pi[n] * x[n] for n in fs)
                elif phase == 'sb':
                    v = s.sb_intercept(pi)
                else:
                    v, pi = s.lagrangian(x, slopes[w][-lag_basis:], max_it=lag_it)
                return res, v, pi

            done = pmap(root_work, outer)
            for w in outer:
                th = rth[w].X
                res, v, pi = done[w]
                if res is None:
                    continue
                if res[0] == 'feas':
                    _, coef, C = res
                    if sum(c_ * x_out[n] for n, c_ in coef.items()) <= C + 1e-7 * max(1.0, abs(C)):
                        continue
                    relaxed.addConstr(gp.quicksum(c_ * rx[n] for n, c_ in coef.items()) <= C)
                    g.addConstr(gp.quicksum(c_ * xv[n] for n, c_ in coef.items()) <= C)
                    added += 1
                    continue
                if v + sum(pi[n] * x_out[n] for n in fs) > th + 1e-6 * max(1.0, abs(th)):
                    relaxed.addConstr(rth[w] >= v + slope_expr(pi, rx))
                    g.addConstr(theta[w] >= v + slope_expr(pi, xv))
                    added += 1
            log['root'].append((it, phase, lb, added, time.time() - t_start))
            if not quiet:
                print(f'  root {it:2d} [{phase}]: LB {lb:.4f}  cuts {added}  '
                      f'[{time.time()-t_start:.1f}s]', flush=True)
            it += 1
            hist.append(lb)
            if lam < 1.0:
                flat = flat + 1 if len(hist) > 1 and hist[-1] - hist[-2] < 1e-9 * max(1.0, abs(lb)) else 0
                if flat >= 5:
                    lam = 1.0                       # stalled: fall back to Kelley
                if added == 0 and not was_kelley:
                    kelley = True                   # nothing cuts x_out: one Kelley round
                    continue
            if added == 0 or (stall_tol > 0 and len(hist) > 3 and
                              hist[-1] - hist[-4] < stall_tol * max(1.0, abs(hist[-1]))):
                break
    relaxed.optimize()
    log['root_lb'] = relaxed.ObjVal
    if not quiet:
        print(f'  root LB {relaxed.ObjVal:.4f}  [{time.time()-t_start:.1f}s]', flush=True)
    g.update()
    t_root = time.time() - t_start

    # --- evaluation and cuts shared by the warm loop and the callback ------------
    cache = {}
    best = {'ub': np.inf, 'pending': None, 'sol': None}
    allv = g.getVars()
    names = [v.VarName for v in allv]
    obj_c = [v.Obj for v in allv]
    th_pos = {w: names.index(f'theta_{w}') for w in outer}
    th_set = set(th_pos.values())

    def evaluate(full):
        """(val, x, bv, qs) at a master solution; records the plan if it improves."""
        val = dict(zip(names, full))
        x = {n: val[n] for n in fs}
        for n in fs:
            if xv[n].VType == GRB.BINARY:
                x[n] = float(round(x[n]))
        for r in rnames:
            x[r] = sum(wt * round(val[b.VarName]) for b, wt in enc[r])
        bv = [val[b.VarName] for b in bins]
        key = tuple(int(round(b)) for b in bv)
        if key not in cache:
            qs = pmap(lambda w: subs[w].mip(x), outer)
            cache[key] = qs
            if all(q is not None for q in qs.values()):
                # master part (first stage + kept scenarios) at this solution
                base = sum(obj_c[i] * full[i] for i in range(len(allv)) if i not in th_set) \
                    + g.ObjCon
                ub = base + sum(probs[w] * qs[w][1] for w in outer)
                if ub < best['ub']:
                    best['ub'] = ub
                    sol = list(full)
                    for w in outer:
                        sol[th_pos[w]] = qs[w][1]
                    best['pending'] = best['sol'] = sol
                    log['incumbents'].append((time.time() - t_start, ub))
                    if not quiet:
                        print(f'  [{time.time()-t_start:7.1f}s] incumbent {ub:.4f}', flush=True)
        return val, x, bv, cache[key]

    def cuts_at(val, x, bv, qs):
        """[(lhs, rhs, sense)] of violated cuts: L-L then lp/sb per scenario, or a no-good."""
        out = []
        S = [i for i, b in enumerate(bv) if round(b) == 1]
        lin = gp.quicksum(bins[i] for i in S) - gp.quicksum(
            bins[i] for i in range(len(bins)) if round(bv[i]) == 0)
        if any(qs[w] is None for w in outer):      # infeasible recourse: no-good cut
            return [(lin, len(S) - 1, 'le')]
        viol = [w for w in outer
                if val[f'theta_{w}'] < qs[w][0] - 1e-6 * max(1.0, abs(qs[w][0]))]

        def cb_work(w):
            res = subs[w].lp_cut(x)
            if res is None or res[0] == 'feas':   # integer x: the recourse is complete
                return None
            qq, pi = res
            v = subs[w].sb_intercept(pi) if 'sb' in cuts else \
                qq - sum(pi[n] * x[n] for n in fs)
            return v, pi

        bend = pmap(cb_work, viol) if ('lp' in cuts or 'sb' in cuts) else {}
        for w in viol:
            qb = qs[w][0]
            out.append((theta[w], (qb - L[w]) * (lin - len(S) + 1) + L[w], 'ge'))
            if bend.get(w) is not None:
                v, pi = bend[w]
                out.append((theta[w], v + slope_expr(pi, xv), 'ge'))
                if 'llsb' in cuts:
                    # L-L cut on top of the SB function instead of the constant L_w:
                    # theta >= SB(x) + (Q(x^) - SB(x^)) (lin - |S| + 1). Exact at x^,
                    # <= SB(x) <= Q(x) at every other binary x; with SB(x^) >= L_w its
                    # LP relaxation is tighter than the plain cut near x^.
                    sb_hat = v + sum(pi[n] * x[n] for n in fs)
                    d = max(qb - sb_hat, 0.0)
                    out.append((theta[w], v + slope_expr(pi, xv) + d * (lin - len(S) + 1),
                                'ge'))
        return out

    # --- warm start: master MILP rounds without the callback ----------------------
    t_warm0 = time.time()
    g.Params.OutputFlag = 0
    for k in range(warm):
        g.Params.MIPGap = warm_gap
        g.Params.TimeLimit = warm_tl
        g.optimize()
        if g.SolCount == 0:
            break
        val, x, bv, qs = evaluate(g.getAttr('X', allv))
        cs = cuts_at(val, x, bv, qs)
        if not quiet:
            print(f'  warm {k}: master {g.ObjVal:.4f} (bound {g.ObjBound:.4f})  '
                  f'best plan {best["ub"]:.4f}  cuts {len(cs)}  '
                  f'[{time.time()-t_start:.1f}s]', flush=True)
        for lhs, rhs, sense in cs:
            g.addConstr(lhs <= rhs if sense == 'le' else lhs >= rhs)
        if not cs:
            break
    t_warm = time.time() - t_warm0
    if best['sol'] is not None:
        g.setAttr('Start', allv, best['sol'])
        best['pending'] = None

    # --- branch and cut with lazy L-L (and lp/sb) cuts --------------------------
    def cb(model, where):
        if where == GRB.Callback.MIPNODE and best['pending'] is not None:
            # hand the best evaluated plan back to Gurobi, theta = the recourse values
            model.cbSetSolution(allv, best['pending'])
            model.cbUseSolution()
            best['pending'] = None
            return
        if where != GRB.Callback.MIPSOL:
            return
        val, x, bv, qs = evaluate(model.cbGetSolution(allv))
        for lhs, rhs, sense in cuts_at(val, x, bv, qs):
            model.cbLazy(lhs <= rhs if sense == 'le' else lhs >= rhs)

    g.Params.LazyConstraints = 1
    g.Params.OutputFlag = 0 if quiet else 1
    g.Params.MIPGap = EF_GAP
    if tl:
        g.Params.TimeLimit = max(1.0, tl - (time.time() - t_start))
    g.optimize(cb)
    ss = subs.values()
    return {'obj': g.ObjVal if g.SolCount else None, 'bound': g.ObjBound,
            'ub_eval': best['ub'], 'status': g.Status, 'nodes': g.NodeCount,
            'time': time.time() - t_start, 't_root': t_root, 't_L': t_L, 't_warm': t_warm,
            'root_lb': log['root_lb'], 'n_eval': len(cache),
            'sub_time': {'mip': sum(s.t_mip for s in ss), 'lp': sum(s.t_lp for s in ss),
                         'inner': sum(s.t_sb for s in ss)},
            'sub_count': {'mip': sum(s.n_mip for s in ss), 'lp': sum(s.n_lp for s in ss),
                          'inner': sum(s.n_sb for s in ss)},
            'n_bin_first': len(bins), 'log': log,
            'cp': {'time': sum(getattr(s, 't_cp', 0.0) for s in ss),
                   'cuts': sum(getattr(s, 'n_cp', 0) for s in ss)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=15)
    ap.add_argument('--day', type=int, default=None)
    ap.add_argument('--scenarios', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bits', type=int, default=6)
    ap.add_argument('--r-max-factor', type=float, default=1.25,
                    help='grid top = factor * max r_sym of the continuous EF')
    ap.add_argument('--cuts', default='lp,sb', help='callback cuts besides ll: lp,sb')
    ap.add_argument('--root-cuts', default='lp', help='root phases in order: lp,sb,lag')
    ap.add_argument('--root-rounds', type=int, default=30, help='per phase')
    ap.add_argument('--cp-rounds', type=int, default=30, help='rounds of the cp phase')
    ap.add_argument('--workers', type=int, default=1,
                    help='scenario subproblems solved in parallel (0: one per CPU)')
    ap.add_argument('--stab', type=float, default=1.0,
                    help='in-out stabilisation of the root loop: weight of the master point '
                         'in the separation point (1 = Kelley, 0.2 = Fischetti et al.)')
    ap.add_argument('--stall-tol', type=float, default=1e-4,
                    help='a root phase ends when 3 rounds gain less (relative); 0: never')
    ap.add_argument('--keep', type=int, default=0,
                    help='partial decomposition: scenarios kept in the master')
    ap.add_argument('--lag-basis', type=int, default=10)
    ap.add_argument('--lag-it', type=int, default=8)
    ap.add_argument('--cp-k', type=int, default=5,
                    help='lift-and-project cuts per scenario and root round (phase cp)')
    ap.add_argument('--sub-gap', type=float, default=1e-6)
    ap.add_argument('--sb-gap', type=float, default=1e-4,
                    help='gap of the inner (SB / Lagrangian) MILPs; their dual bound is used')
    ap.add_argument('--lw', choices=('lp', 'mip'), default='lp',
                    help='global lower bound L_w of the L-L cut: recourse LP or MILP bound')
    ap.add_argument('--warm', type=int, default=0,
                    help='master-MILP warm-start rounds before branch and cut')
    ap.add_argument('--warm-gap', type=float, default=1e-3)
    ap.add_argument('--warm-tl', type=float, default=120)
    ap.add_argument('--time-limit', type=float, default=1800)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    players, T, scen, name = build(a.n, a.day, a.scenarios, a.seed)
    print(f'=== {name} n={len(players)} |Omega|={len(scen)} bits={a.bits} cuts={a.cuts} '
          f'root={a.root_cuts} keep={a.keep} ===', flush=True)
    out = os.path.join(_PAPER, 'weak_eps_experiment', 'results_lshaped')
    os.makedirs(out, exist_ok=True)
    base = f'{name}_S{a.scenarios}_seed{a.seed}_b{a.bits}'
    ef_file = os.path.join(out, f'ef_{base}.json')
    if os.path.exists(ef_file):
        with open(ef_file) as f:
            d = json.load(f)
        ef, efd = d['ef'], d['ef_disc']
        print('EF (cached)', flush=True)
    else:
        ef = solve_ef(players, T, scen, EF_GAP, a.time_limit)
        ef.pop('x')
        r_max = a.r_max_factor * max(ef['r'].values())
        efd = solve_ef(players, T, scen, EF_GAP, a.time_limit, r_max=r_max, bits=a.bits)
        efd.pop('x')
        efd['r_max'] = r_max
        with open(ef_file, 'w') as f:
            json.dump({'ef': ef, 'ef_disc': efd}, f, indent=1, default=float)
    r_max = efd['r_max']
    print(f'EF continuous : obj {ef["obj"]:.4f} bound {ef["bound"]:.4f} {ef["time"]:.1f}s',
          flush=True)
    print(f'EF discretised: obj {efd["obj"]:.4f} bound {efd["bound"]:.4f} '
          f'{efd["time"]:.1f}s  (delta {r_max/(2**a.bits-1):.4f})', flush=True)
    ils = integer_lshaped(players, T, scen, r_max, a.bits, set(a.cuts.split(',')) - {''},
                          a.sub_gap, a.time_limit, quiet=not a.verbose,
                          root_rounds=a.root_rounds, sb_gap=a.sb_gap,
                          root_cuts=tuple(a.root_cuts.split(',')), keep=a.keep,
                          lag_basis=a.lag_basis, lag_it=a.lag_it, lw=a.lw, warm=a.warm, cp_k=a.cp_k, stall_tol=a.stall_tol, stab=a.stab, cp_rounds=a.cp_rounds,
                          workers=a.workers,
                          warm_gap=a.warm_gap, warm_tl=a.warm_tl)
    print(f'int. L-shaped : obj {ils["obj"]} bound {ils["bound"]:.4f} '
          f'{ils["time"]:.1f}s  (L_w {ils["t_L"]:.1f}s, root {ils["t_root"]:.1f}s, '
          f'warm {ils["t_warm"]:.1f}s, '
          f'root LB {ils["root_lb"]:.4f})  nodes {ils["nodes"]:.0f}  '
          f'evaluated x {ils["n_eval"]}  first-stage binaries {ils["n_bin_first"]}',
          flush=True)
    print(f'  best evaluated plan {ils["ub_eval"]:.4f}  cut-and-project {ils["cp"]}')
    print(f'  subproblem time {ils["sub_time"]}  count {ils["sub_count"]}', flush=True)
    tag = f'{base}_{a.cuts.replace(",", "")}_root{a.root_cuts.replace(",", "")}' \
        + (f'_keep{a.keep}' if a.keep else '') + (f'_lw{a.lw}' if a.lw != 'lp' else '') \
        + (f'_warm{a.warm}' if a.warm else '') + (f'_stab{a.stab:g}' if a.stab < 1 else '') \
        + f'_rr{a.root_rounds}' + (f'_cpr{a.cp_rounds}' if 'cp' in a.root_cuts else '') \
        + (f'_w{a.workers}' if a.workers != 1 else '')
    with open(os.path.join(out, f'{tag}.json'), 'w') as f:
        json.dump({'args': vars(a), 'ef': ef, 'ef_disc': efd, 'lshaped': ils}, f,
                  indent=1, default=float)


if __name__ == '__main__':
    main()

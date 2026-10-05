"""
Scenario-wise Dantzig-Wolfe branch-and-price for the stochastic grand-coalition
dispatch (the extensive form of stochastic_extension), an EXPERIMENT. Exact for the
original problem: r_sym stays continuous, no discretisation (cf. integer_lshaped.md).

Dualising non-anticipativity x_w = xbar (Caroe & Schultz 1999) leaves one full-
community MILP per scenario. As a DW master over scenario plans (x_wk, y_wk):

    min  sum_w sum_k c_wk lam_wk
    s.t. sum_k x_wk lam_wk - xbar = 0      (w, first-stage n)   dual pi_wn
         sum_k lam_wk          = 1         (w)                  dual mu_w
         lam >= 0, xbar in the node's box, c_wk = p_w (c1 x_wk + c2_w y_wk)

The pricing problem of scenario w is that scenario's MILP with the first stage free,
objective p_w c_w - pi_w^T x. The master LP value at convergence is z_D, the bound of
the Lagrangian dual of non-anticipativity. Lower bound at ANY duals (valid with
artificials and inexact pricing, from the pricing MILPs' dual bounds):

    L(pi) = sum_w min_{X_w ∩ box} (p_w c_w - pi_w^T x) + min_{xbar in box} (sum_w pi_w)^T xbar

Branching on the original first-stage variables: a fractional binary of xbar (x <= 0 /
x >= 1), else a continuous r_sym on which the scenarios' columns disagree, split at
xbar (x <= xbar / x >= xbar). Both act on the pricing problems as bounds and on the
column pool as filters; nothing else changes in the pricing problem.

Primal: the first stage of a positive-weight column is feasible for every scenario
(complete recourse at integer x), so evaluating it in all scenarios gives an upper
bound.

    python ieee_owen/scenario_bnp.py --n 15 --scenarios 20 --workers 0
"""
import os, sys, time, json, argparse, heapq, itertools

_PAPER = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from integer_lshaped import build, stack_to_gurobi
from stochastic_extension import EF_GAP


class ScenarioPricer:
    """Scenario w's full-community MILP, first stage free; objective p_w c_w - pi^T x."""

    def __init__(self, players, T, scen_w, prob, r_ub, gap, env=None, threads=0):
        st, g, gv = stack_to_gurobi(players, T, [(1.0, scen_w)], 'price', env=env)
        self.fs = sorted(st.first_stage)
        self.prob = prob
        g.Params.OutputFlag = 0
        g.Params.MIPGap = gap
        g.Params.Threads = threads
        self.vars = g.getVars()
        self.base = np.array(g.getAttr('Obj', self.vars)) * prob
        g.setAttr('Obj', self.vars, self.base.tolist())
        g.ObjCon = g.ObjCon * prob
        pos = {v.VarName: i for i, v in enumerate(self.vars)}
        self.fv = [gv[n] for n in self.fs]
        self.fbase = np.array([self.base[pos[n]] for n in self.fs])
        for n in self.fs:
            if n.startswith('r_sym_'):
                gv[n].UB = min(gv[n].UB, r_ub)
        g.update()                      # attribute writes are lazy: read back only now
        self.orig = {n: (gv[n].LB, gv[n].UB) for n in self.fs}
        self.g, self.gv = g, gv
        self.t, self.n = 0.0, 0

    def set_box(self, box):
        for n, v in zip(self.fs, self.fv):
            lo, hi = box.get(n, self.orig[n])
            v.LB, v.UB = lo, hi

    def price(self, pi):
        """(x, cost, rc_incumbent, rc_bound) with pi: array over self.fs; cost = p c."""
        t0 = time.time()
        self.g.setAttr('Obj', self.fv, (self.fbase - pi).tolist())
        self.g.optimize()
        self.t += time.time() - t0
        self.n += 1
        if self.g.SolCount == 0:
            return None
        x = np.array(self.g.getAttr('X', self.fv))
        allx = np.array(self.g.getAttr('X', self.vars))
        cost = float(self.base @ allx) + self.g.ObjCon
        return x, cost, self.g.ObjVal, self.g.ObjBound

    def evaluate(self, x):
        """p_w Q_w(x) + p_w c1 x with x fixed: (incumbent, bound) or None."""
        t0 = time.time()
        self.g.setAttr('Obj', self.fv, self.fbase.tolist())
        for v, xv in zip(self.fv, x):
            v.LB = v.UB = float(xv)
        self.g.optimize()
        for n, v in zip(self.fs, self.fv):
            v.LB, v.UB = self.orig[n]
        self.t += time.time() - t0
        self.n += 1
        if self.g.SolCount == 0:
            return None
        return self.g.ObjVal, self.g.ObjBound


class BnP:
    def __init__(self, players, T, scen, workers=0, gap=EF_GAP, price_gap=1e-6,
                 r_ub=100.0, art_cost=1e6, cg_tol=1e-6, quiet=False,
                 lp_start=True, max_it=2000, pen_eps=0.2, pen_delta=0.2,
                 pen_shrink=0.25, max_rounds=12, alpha=0.1):
        self.t0 = time.time()
        self.S = len(scen)
        self.probs = [p for p, _ in scen]
        k = max(1, min(workers or (os.cpu_count() or 1), self.S))
        self.k = k
        self.owner = {w: w % k for w in range(self.S)}
        envs = [None] * k
        self.pool = None
        if k > 1:
            from concurrent.futures import ThreadPoolExecutor
            envs = [gp.Env() for _ in range(k)]
            self.pool = ThreadPoolExecutor(k)
        per = max(1, (os.cpu_count() or 1) // k) if k > 1 else 0
        self.pr = [ScenarioPricer(players, T, scen[w][1], self.probs[w], r_ub, price_gap,
                                  env=envs[self.owner[w]], threads=per)
                   for w in range(self.S)]
        self.fs = self.pr[0].fs
        self.N = len(self.fs)
        self.is_bin = np.array([self.pr[0].gv[n].VType == GRB.BINARY for n in self.fs])
        self.orig = self.pr[0].orig
        self.gap, self.cg_tol, self.quiet = gap, cg_tol, quiet
        self.art_cost = art_cost
        self.pen_eps, self.pen_delta, self.pen_shrink = pen_eps, pen_delta, pen_shrink
        self.max_rounds, self.alpha_base = max_rounds, alpha
        self.lp_start, self.max_it = lp_start, max_it
        self.t_master = self.t_price_wall = 0.0
        self._inst = (players, T, scen, r_ub)
        self.t_build = time.time() - self.t0
        # master LP
        m = gp.Model('bnp_master')
        m.Params.OutputFlag = 0
        m.Params.Method = 1                         # dual simplex: re-solves after bounds
        self.xbar = [m.addVar(lb=self.orig[n][0], ub=self.orig[n][1]) for n in self.fs]
        self.na = [[None] * self.N for _ in range(self.S)]
        self.ap = [[None] * self.N for _ in range(self.S)]
        self.am = [[None] * self.N for _ in range(self.S)]
        self.art = []
        for w in range(self.S):
            for j in range(self.N):
                ap_ = m.addVar(obj=art_cost)
                am_ = m.addVar(obj=art_cost)
                self.ap[w][j], self.am[w][j] = ap_, am_
                self.na[w][j] = m.addLConstr(-self.xbar[j] + ap_ - am_ == 0.0)
        self.conv = [m.addLConstr(gp.LinExpr() == 1.0) for w in range(self.S)]
        # convexity rows need a column before they are feasible: an artificial each
        for w in range(self.S):
            a = m.addVar(obj=art_cost, column=gp.Column([1.0], [self.conv[w]]))
            self.art.append(a)
        m.update()
        self.m = m
        self.cols = []                 # (w, x, cost, var)
        self.ub, self.best_x = np.inf, None
        self.log = []
        self.n_cols = 0

    # --- helpers ------------------------------------------------------------------
    def pmap(self, fn, ws):
        ws = list(ws)
        if self.pool is None or len(ws) < 2:
            return {w: fn(w) for w in ws}
        groups = {}
        for w in ws:
            groups.setdefault(self.owner[w], []).append(w)
        futs = [self.pool.submit(lambda grp: {w: fn(w) for w in grp}, grp)
                for grp in groups.values()]
        out = {}
        for f in futs:
            out.update(f.result())
        return out

    def elapsed(self):
        return time.time() - self.t0

    def add_column(self, w, x, cost):
        coefs = [float(v) for v in x] + [1.0]
        rows = list(self.na[w]) + [self.conv[w]]
        keep = [i for i, c in enumerate(coefs) if abs(c) > 1e-12]
        var = self.m.addVar(obj=cost, column=gp.Column([coefs[i] for i in keep],
                                                        [rows[i] for i in keep]))
        self.cols.append((w, x.copy(), cost, var))
        self.n_cols += 1

    def in_box(self, x, box):
        for j, n in enumerate(self.fs):
            if n in box:
                lo, hi = box[n]
                if x[j] < lo - 1e-6 or x[j] > hi + 1e-6:
                    return False
        return True

    def apply_box(self, box):
        lb = np.array([box.get(n, self.orig[n])[0] for n in self.fs])
        ub = np.array([box.get(n, self.orig[n])[1] for n in self.fs])
        self.m.setAttr('LB', self.xbar, lb.tolist())
        self.m.setAttr('UB', self.xbar, ub.tolist())
        for w, x, c, var in self.cols:
            var.UB = GRB.INFINITY if self.in_box(x, box) else 0.0
        for p in self.pr:
            p.set_box(box)
        return lb, ub

    # --- primal --------------------------------------------------------------------
    def try_plan(self, x, tag=''):
        """Evaluate a first stage in every scenario; update the incumbent."""
        x = np.where(self.is_bin, np.round(x), x)
        res = self.pmap(lambda w: self.pr[w].evaluate(x), range(self.S))
        if any(r is None for r in res.values()):
            return None
        # the evaluated plans are columns too (one consistent x across scenarios)
        for w, r in res.items():
            self.add_column(w, x, r[0])
        val = sum(r[0] for r in res.values())
        if val < self.ub:
            self.ub, self.best_x = val, x.copy()
            if not self.quiet:
                print(f'  [{self.elapsed():7.1f}s] incumbent {val:.4f} {tag}', flush=True)
        return val

    # --- column generation at a node --------------------------------------------------
    def lp_duals(self):
        """Non-anticipativity duals of the extensive form's LP relaxation with one copy
        of the first stage per scenario. L(pi_LP) >= z_LP, so CG starts at or above
        the LP bound instead of from pi = 0."""
        import stochastic_extension as SE
        players, T, scen, r_ub = self._inst
        t0 = time.time()
        saved = SE.FIRST_STAGE_PREFIXES
        SE.FIRST_STAGE_PREFIXES = ('\x00none',)     # every variable per scenario
        try:
            st, g, gv = stack_to_gurobi(players, T, scen, 'ef_copies')
        finally:
            SE.FIRST_STAGE_PREFIXES = saved
        lp = g.relax()
        lp.Params.OutputFlag = 0
        by = {v.VarName: v for v in lp.getVars()}
        xb = [lp.addVar(lb=self.orig[n][0], ub=self.orig[n][1]) for n in self.fs]
        rows = [[None] * self.N for _ in range(self.S)]
        for w in range(self.S):
            for j, n in enumerate(self.fs):
                v = by[f'{n}_s{w}']
                if n.startswith('r_sym_'):
                    v.UB = min(v.UB, r_ub)
                rows[w][j] = lp.addLConstr(v - xb[j] == 0.0)
        lp.optimize()
        pi = np.array([[rows[w][j].Pi for j in range(self.N)] for w in range(self.S)])
        self.t_lpduals = time.time() - t0
        if not self.quiet:
            print(f'  EF LP (copies): {lp.ObjVal:.4f}  [{self.t_lpduals:.1f}s]', flush=True)
        return pi, lp.ObjVal

    def _set_penalty(self, center, eps, delta):
        """Three-piece penalty (du Merle et al. 1999), as DirectMaster._set_penalty:
        each non-anticipativity row's slacks a+/a- are capped at delta and cost
        c + e / -c + e, e = eps (1 + |c|), so the dual moves freely in [c - e, c + e]
        and pays delta per unit beyond. center None: slacks off (delta = 0)."""
        if center is None:
            cp = cm = np.zeros((self.S, self.N))
            ub = 0.0
        else:
            e = eps * (1.0 + np.abs(center))
            cp, cm, ub = center + e, -center + e, delta
        flat_v = [v for w in range(self.S) for v in self.ap[w]] + \
                 [v for w in range(self.S) for v in self.am[w]]
        self.m.setAttr('Obj', flat_v, np.concatenate([cp.ravel(), cm.ravel()]).tolist())
        self.m.setAttr('UB', flat_v, [ub] * len(flat_v))

    def _alpha(self, obj, best_lb):
        """Adaptive smoothing weight on the RMP duals (Pessoa et al. 2010), as
        DirectMaster._alpha: base 0.1, raised when the RMP value is above the UB."""
        base = self.alpha_base
        gap = obj - best_lb
        if not np.isfinite(gap):
            return base
        if gap <= self.cg_tol * (1.0 + abs(obj)):
            return 1.0
        if obj > self.ub and self.ub - best_lb > 1e-6:
            return min(1.0, base * (self.ub - best_lb) / gap)
        return base

    def cg(self, box, ub_cut=np.inf, max_it=2000, center0=None):
        """Column generation in the node `box`, stabilised as the stochastic CG:
        Wentges smoothing around the best-bound duals and penalty rounds with
        shrinking (eps, delta), the last round unpenalised. Returns (lb, obj, its)."""
        lb_box, ub_box = self.apply_box(box)
        best_lb, center = -np.inf, None
        it = 0

        def bound_and_columns(pi, pi_rmp, conv, obj):
            """Price at pi: Lagrangian bound, and the columns with negative reduced
            cost at the RMP duals pi_rmp."""
            tp = time.time()
            res = self.pmap(lambda w: self.pr[w].price(pi[w]), range(self.S))
            self.t_price_wall += time.time() - tp
            if any(r is None for r in res.values()):
                return None, 0
            sv = pi.sum(axis=0)
            lb = sum(r[3] for r in res.values()) + \
                float(np.sum(np.minimum(sv * lb_box, sv * ub_box)))
            added = 0
            for w, (x, cost, _, _) in res.items():
                rc = cost - float(pi_rmp[w] @ x) - conv[w] if pi_rmp is not None else -1.0
                if rc < -self.cg_tol * max(1.0, abs(obj)):
                    self.add_column(w, x, cost)
                    added += 1
            return lb, added

        if center0 is not None:
            lb, _ = bound_and_columns(center0, None, None, 0.0)
            if lb is None:
                return np.inf, np.inf, 0
            best_lb, center = lb, center0.copy()
            if not self.quiet:
                print(f'    start at the given duals: LB {best_lb:.4f}', flush=True)

        eps, delta = self.pen_eps, self.pen_delta
        for rnd in range(1, self.max_rounds + 1):
            last = rnd == self.max_rounds or eps <= 0.0
            self._set_penalty(None if (last or center is None) else center, eps, delta)
            while it < max_it:
                it += 1
                tm = time.time()
                self.m.optimize()
                self.t_master += time.time() - tm
                if self.m.Status != GRB.OPTIMAL:
                    raise RuntimeError(f'master status {self.m.Status}')
                obj = self.m.ObjVal
                pi = np.array([[self.na[w][j].Pi for j in range(self.N)]
                               for w in range(self.S)])
                conv = [self.conv[w].Pi for w in range(self.S)]
                if center is None:
                    center = pi.copy()
                alpha = self._alpha(obj, best_lb)
                added, mode = 0, 'rmp'
                if alpha < 1.0:
                    sep = alpha * pi + (1 - alpha) * center
                    lb, added = bound_and_columns(sep, pi, conv, obj)
                    if lb is None:
                        return np.inf, obj, it
                    if lb > best_lb:
                        best_lb, center = lb, sep.copy()
                    mode = 'smooth' if added else 'misprice'
                if not added:
                    lb, added = bound_and_columns(pi, pi, conv, obj)
                    if lb is None:
                        return np.inf, obj, it
                    if lb > best_lb:
                        best_lb, center = lb, pi.copy()
                slack = sum(v.X for w in range(self.S) for v in self.ap[w] + self.am[w])
                self.log.append((self.elapsed(), rnd, it, obj, best_lb, added, slack, alpha))
                if not self.quiet and (it % 5 == 1 or added == 0):
                    print(f'    cg {it:4d} r{rnd}: master {obj:.4f}  LB {best_lb:.4f}  '
                          f'cols +{added} ({self.n_cols})  a {alpha:.2f} {mode:8s} '
                          f'slack {slack:.2g}  master {self.t_master:.0f}s price {self.t_price_wall:.0f}s '
                          f'[{self.elapsed():.1f}s]', flush=True)
                if best_lb >= ub_cut - self.gap * max(1.0, abs(ub_cut)):
                    self.m.optimize()
                    self.last_center = center
                    return best_lb, self.m.ObjVal, it        # prune
                if last and obj - best_lb <= self.cg_tol * max(1.0, abs(obj)):
                    break                                     # z_D reached
                if not added:
                    break                                     # round priced out
            if not self.quiet:
                print(f'  -- penalty round {rnd}: eps {eps:.3g} delta {delta:.3g}  '
                      f'master {self.m.ObjVal:.4f}  LB {best_lb:.4f}  UB {self.ub:.4f}  '
                      f'[{self.elapsed():.1f}s]', flush=True)
            if last or it >= max_it:
                break
            eps, delta = eps * self.pen_shrink, delta * self.pen_shrink
            if eps < 1e-4:
                eps = delta = 0.0
        self._set_penalty(None, 0.0, 0.0)
        self.m.optimize()                                     # leave a solved master
        self.last_center = center
        return best_lb, self.m.ObjVal, it

    def master_solution(self):
        xb = np.array(self.m.getAttr('X', self.xbar))
        used = [(w, x, var.X) for w, x, c, var in self.cols if var.X > 1e-9]
        return xb, used

    def choose_branch(self, xb, used):
        frac = [(abs(xb[j] - 0.5), j) for j in range(self.N)
                if self.is_bin[j] and 1e-6 < xb[j] < 1 - 1e-6]
        if frac:
            j = min(frac)[1]
            return j, 'bin'
        # continuous: the largest disagreement among the used columns
        best, jb = 1e-6, None
        for j in range(self.N):
            if self.is_bin[j]:
                continue
            vals = [x[j] for w, x, l in used]
            if vals and max(vals) - min(vals) > best:
                best, jb = max(vals) - min(vals), j
        if jb is not None:
            return jb, 'cont'
        return None, None

    # --- branch and price --------------------------------------------------------------
    def initial(self):
        """A consistent start: scenario 0's own optimum, evaluated in every scenario."""
        r = self.pr[0].price(np.zeros(self.N))
        self.try_plan(r[0], tag='(initial: scenario 0 optimum)')

    def solve(self, time_limit=3600):
        self.initial()
        ctr = itertools.count()
        root = {}
        pi_root = self.lp_duals()[0] if self.lp_start else None
        heap = [(-np.inf, next(ctr), root, pi_root)]
        nodes = 0
        global_lb = -np.inf
        while heap:
            if self.elapsed() > time_limit:
                break
            plb, _, box, c0 = heapq.heappop(heap)
            if plb >= self.ub - self.gap * max(1.0, abs(self.ub)):
                continue
            nodes += 1
            lb, obj, its = self.cg(box, ub_cut=self.ub, max_it=self.max_it, center0=c0)
            xb, used = self.master_solution() if np.isfinite(lb) else (None, [])
            if np.isfinite(lb):
                # primal: the first stages of the heaviest columns
                for w, x, l in sorted(used, key=lambda t: -t[2])[:2]:
                    self.try_plan(x, tag=f'(node {nodes}, column of w{w})')
                self.try_plan(xb, tag=f'(node {nodes}, rounded xbar)')
            open_lbs = [h[0] for h in heap] + ([lb] if np.isfinite(lb) else [])
            global_lb = min(open_lbs) if open_lbs else self.ub
            if not self.quiet:
                print(f'  node {nodes}: depth {len(box)}  LB {lb:.4f}  UB {self.ub:.4f}  '
                      f'global LB {global_lb:.4f}  open {len(heap)}  cg its {its}  '
                      f'[{self.elapsed():.1f}s]', flush=True)
            if not np.isfinite(lb) or lb >= self.ub - self.gap * max(1.0, abs(self.ub)):
                continue
            j, kind = self.choose_branch(xb, used)
            if j is None:
                continue                                 # xbar consistent: node solved
            n = self.fs[j]
            lo, hi = box.get(n, self.orig[n])
            if kind == 'bin':
                kids = [(lo, 0.0), (1.0, hi)]
            else:
                kids = [(lo, float(xb[j])), (float(xb[j]), hi)]
            for klo, khi in kids:
                if klo > khi + 1e-9:
                    continue
                child = dict(box)
                child[n] = (klo, khi)
                heapq.heappush(heap, (lb, next(ctr), child, self.last_center))
        open_lbs = [h[0] for h in heap]
        final_lb = min(open_lbs + [self.ub]) if heap else self.ub
        if heap and self.elapsed() <= time_limit:
            final_lb = self.ub
        return {'ub': self.ub, 'lb': final_lb, 'nodes': nodes, 'open': len(heap),
                'time': self.elapsed(), 't_build': self.t_build, 'cols': self.n_cols,
                'price_time': sum(p.t for p in self.pr), 'price_calls': sum(p.n for p in self.pr),
                'log': self.log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=15)
    ap.add_argument('--day', type=int, default=None)
    ap.add_argument('--scenarios', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=0, help='0: one per CPU')
    ap.add_argument('--gap', type=float, default=EF_GAP)
    ap.add_argument('--price-gap', type=float, default=1e-6)
    ap.add_argument('--pen-eps', type=float, default=0.2)
    ap.add_argument('--pen-delta', type=float, default=0.2)
    ap.add_argument('--pen-shrink', type=float, default=0.25)
    ap.add_argument('--max-rounds', type=int, default=12)
    ap.add_argument('--alpha', type=float, default=0.1,
                    help='Wentges smoothing base weight on the RMP duals (1: off)')
    ap.add_argument('--r-ub', type=float, default=100.0, help='bound on r_sym [MW]')
    ap.add_argument('--time-limit', type=float, default=3600)
    ap.add_argument('--root-only', action='store_true')
    ap.add_argument('--max-it', type=int, default=2000, help='CG iterations per node')
    ap.add_argument('--no-lp-start', dest='lp_start', action='store_false',
                    help='start the root CG at pi = 0 instead of the EF-LP duals')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    players, T, scen, name = build(a.n, a.day, a.scenarios, a.seed)
    print(f'=== {name} n={len(players)} |Omega|={len(scen)} workers={a.workers} '
          f'alpha={a.alpha} pen=({a.pen_eps},{a.pen_delta}) ===', flush=True)
    b = BnP(players, T, scen, workers=a.workers, gap=a.gap, price_gap=a.price_gap,
            r_ub=a.r_ub, quiet=a.quiet, lp_start=a.lp_start, max_it=a.max_it,
            pen_eps=a.pen_eps, pen_delta=a.pen_delta, pen_shrink=a.pen_shrink,
            max_rounds=a.max_rounds, alpha=a.alpha)
    print(f'build {b.t_build:.1f}s  first-stage vars {b.N} ({int(b.is_bin.sum())} binary)  '
          f'workers {b.k}', flush=True)
    if a.root_only:
        b.initial()
        pi0 = b.lp_duals()[0] if a.lp_start else None
        lb, obj, its = b.cg({}, max_it=a.max_it, center0=pi0)
        xb, used = b.master_solution()
        for w, x, l in sorted(used, key=lambda t: -t[2])[:3]:
            b.try_plan(x, tag=f'(column of w{w})')
        b.try_plan(xb, tag='(rounded xbar)')
        res = {'z_D': lb, 'master': obj, 'its': its, 'ub': b.ub, 'time': b.elapsed(),
               'cols': b.n_cols, 'price_time': sum(p.t for p in b.pr), 'log': b.log}
        print(f'root: z_D {lb:.4f}  master {obj:.4f}  UB {b.ub:.4f}  '
              f'gap {(b.ub - lb) / abs(b.ub):.3e}  {b.elapsed():.1f}s  its {its}', flush=True)
    else:
        res = b.solve(a.time_limit)
        print(f'B&P: UB {res["ub"]:.4f}  LB {res["lb"]:.4f}  '
              f'gap {(res["ub"] - res["lb"]) / abs(res["ub"]):.3e}  nodes {res["nodes"]}  '
              f'open {res["open"]}  {res["time"]:.1f}s  cols {res["cols"]}  '
              f'pricing {res["price_time"]:.1f}s over {res["price_calls"]} calls', flush=True)
    out = os.path.join(_PAPER, 'weak_eps_experiment', 'results_bnp')
    os.makedirs(out, exist_ok=True)
    tag = f'{name}_S{a.scenarios}_seed{a.seed}' + ('_root' if a.root_only else '') \
        + (f'_a{a.alpha:g}' if a.alpha != 0.1 else '') \
        + (f'_pen{a.pen_eps:g}' if a.pen_eps != 0.2 else '') + ('' if a.lp_start else '_nolp')
    with open(os.path.join(out, f'{tag}.json'), 'w') as f:
        json.dump({'args': vars(a), 'result': res}, f, indent=1, default=float)


if __name__ == '__main__':
    main()

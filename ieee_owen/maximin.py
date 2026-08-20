"""
Rawlsian maximin selection over the optimal dual face of (DWR_N).  `maximin criterion.md`.

WHAT THIS IS FOR. Column generation returns *an* Owen allocation — whichever vertex of
the optimal dual face the LP solver happened to stop at. When (DWR_N) is primal
degenerate that face is not a point, and every point of it gives an allocation that is
equally stable (P3) and equally budget-neutral (P4). Classical Owen picks among them
arbitrarily; this module picks the one that maximises the worst-off member's payoff.

SIGN CONVENTION. The codebase minimises cost, so `sigma_j` is prosumer j's cost share,
payoff is `-sigma_j`, and the Rawlsian rule

    max  min_{j in P} (payoff_j)      is        min  max_{j in P} sigma_j .

Every linking row here is homogeneous (`b_j = 0`), so `chi_j = sigma_j` with no
endowment term -- the `b_j = 0` reading of `eq:chi_lr` that `compute_owen_allocation`
already takes (`maximin criterion.md` sec.7).

THE DUAL, IN THE CODE'S SIGNS. With `pi` the master's raw coupling-row duals (as
`getDualsolLinear` reports them, i.e. `<= 0` on the `<=` rows) and `sigma_j` the
convexity dual,

    max  sum_j sigma_j
    s.t. sigma_j + pi^T a_j^q  <=  c_j^q            for every column q of j       (R3)
         A_0^T pi <= d_0                            (shared vars r_sym / p)       (R4)
         pi_row <= 0 on the reserve and peak rows,  free on the balance rows

and `sigma_j(pi) = min_q {c_j^q - pi^T a_j^q}` is exactly the objective of
`PlayerSubproblem.solve_pricing` at those duals -- which is what makes the re-pricing
step of Phase 2 free of any new modelling.

WHY THE CONVERGED POOL IS NOT ENOUGH. `sigma_j(pi)` is a min of |Q_j| affine functions;
the pool gives `hat sigma_j(pi) = min over the pool >= sigma_j(pi)`, with equality only
at `pi*`. Convergence of column generation certifies dual feasibility at that one point.
(MM) has to move `pi` off `pi*` to redistribute at all, i.e. exactly into the region
where the pool is loose, so the restricted program over-promises and the returned
allocation may be infeasible for the full one. Hence Phase 2 is a second cutting-plane
loop, not one extra LP: solve, re-price, add violated columns, repeat.

TRUNCATION IS SAFE — BUT ONLY FOR THE RE-EVALUATED ALLOCATION. Weak duality gives
`sum_{j in S} sigma_j(pi) <= c(S)` at every dual-feasible `pi`, so stopping early still
yields a stable allocation. The restricted LP's `hat sigma_j` is LARGER than the
oracle's `sigma_j(hat pi)` whenever a column is missing, and summing a set of
over-estimates can exceed `c(S)` -- so reporting `hat sigma` would forfeit exactly the
property that makes early stopping legitimate. What is reported is the oracle value.
The price of stopping early is subsidy, `z* - sum_j sigma_j(hat pi) <= n * eps` plus
the (R2) tolerance, and it is measured here rather than assumed away.
"""
import time

import numpy as np
from pyscipopt import Model, quicksum

from solver import calculate_column_cost


# --------------------------------------------------------------------------- pool extraction
def coupling_rows(master, time_periods):
    """Ordered list of the coupling rows the master actually has.

    Ordered and keyed rather than merely counted, because the dual vector `pi` is
    indexed by it.
    """
    T = list(time_periods)
    rows = [(k, t) for k in ('elec', 'heat', 'hydro') for t in T]
    if master.model.data['cons']['reserve_up']:
        rows += [('resup', t) for t in T] + [('resdn', t) for t in T]
    if master.model.data['cons']['peak']:
        rows += [('peak', t) for t in T]
    return rows


def column_coefficients(solution, u, rows):
    """`a_j^q`: the column's coefficient in every coupling row, sparse.

    Must agree exactly with `LEMPricer._add_column` / `MasterProblem._create_master_constraints`:
        balance : (i_*_com - e_*_com)      reserve : -r_plus / -r_minus
        peak    : (i_E_gri - e_E_gri)
    """
    def g(key, t):
        return float(solution.get(key, {}).get((u, t), 0.0))

    a = {}
    for kind, t in rows:
        if kind == 'elec':
            v = g('i_E_com', t) - g('e_E_com', t)
        elif kind == 'heat':
            v = g('i_H_com', t) - g('e_H_com', t)
        elif kind == 'hydro':
            v = g('i_G_com', t) - g('e_G_com', t)
        elif kind == 'resup':
            v = -g('r_plus', t)
        elif kind == 'resdn':
            v = -g('r_minus', t)
        else:
            v = g('i_E_gri', t) - g('e_E_gri', t)
        if abs(v) > 1e-12:
            a[(kind, t)] = v
    return a


def default_pi_bound(params, time_periods, floor=1e4):
    """A finite box for `pi`, needed because parts of it are unconstrained by the pool.

    A coupling row that every pooled column sits at zero in -- the heat row at an hour
    where the heat pump is off in every pattern generated so far, say -- leaves its dual
    free. (MM) does not see it in the objective and a simplex solver happily returns it
    at +-infinity. That is not a bug in the formulation: such a `pi` genuinely IS dual
    infeasible for the full problem, and Phase 2 is what exposes it. But an infinite
    dual makes the re-pricing objective unrepresentable, so the loop cannot get as far
    as exposing anything.

    The box keeps the iterate finite without changing what is being computed, provided
    it is loose enough to contain the part of Theta* that matters -- so it is scaled off
    the model's own price data (two orders of magnitude above the largest tariff) and
    whether it is ever active is reported rather than assumed away.
    """
    keys = ('pi_E_gri_import', 'pi_E_gri_export', 'pi_H_gri_import', 'pi_H_gri_export',
            'pi_G_gri_import', 'pi_G_gri_export')
    scale = [abs(float(params[f'{k}_{t}'])) for k in keys for t in time_periods
             if f'{k}_{t}' in params and np.isfinite(params[f'{k}_{t}'])]
    scale += [abs(float(params.get(k, 0.0) or 0.0)) for k in ('pi_res', 'pi_E_peak',
                                                              'pi_up', 'pi_dn')]
    return max(floor, 100.0 * (max(scale) if scale else 0.0))


def _row_duals(master, rows):
    """The master's RAW coupling duals at pi*.

    Raw, not `abs()`-ed as `solution['convex_hull_prices']` stores them: the pricing
    objective is built with the `RC = c - pi a` convention and the `<=` rows carry
    negative duals, so an absolute value would flip their sign.
    """
    m = master.model
    cons = m.data['cons']
    key = {'elec': 'community_elec_balance', 'heat': 'community_heat_balance',
           'hydro': 'community_hydro_balance', 'resup': 'reserve_up',
           'resdn': 'reserve_dn', 'peak': 'peak'}
    return {(k, t): float(m.getDualsolLinear(m.getTransformedCons(cons[key[k]][t])))
            for k, t in rows}


class MaximinSelector:
    """(MM) over the optimal dual face, with the Phase-2 re-pricing loop.

    Construct from a ColumnGenerationSolver that has already run `solve()` to
    convergence -- Phase 1 of `maximin criterion.md` sec.4. Convergence is not
    optional: it is what fixes `z*`, and (R2) is what makes the restricted face a
    superset of the true one rather than a subset (sec.4, "the accounting is exact").
    """

    def __init__(self, cg, protected=None, tol=1e-6, verbose=True, pi_bound=None,
                 r2_slack=None):
        self.cg = cg
        self.players = list(cg.players)
        self.T = [int(t) for t in cg.time_periods]
        self.params = cg.parameters
        self.master = cg.master
        self.verbose = verbose
        self.tol = tol
        self.protected = list(protected) if protected is not None else list(self.players)
        unknown = set(self.protected) - set(self.players)
        if unknown:
            raise ValueError(f"protected set contains non-members: {sorted(unknown)}")

        self.pi_bound = (float(pi_bound) if pi_bound is not None
                         else default_pi_bound(self.params, self.T))
        # (R2) tolerance. It was 1e-7 |z*|, which is far too loose: it lets the restricted
        # LP under-collect, so `t_hat` lands about `r2_slack` BELOW `t*` and the reported
        # bracket bounds `t* + O(r2_slack)` rather than `t*`. Measured at 15p, on the five
        # days where the exact improvement is numerically zero the bracket read exactly
        # `r2_slack` (ratio 1.00), and the median improvement over 30 days was only 5x it
        # — so the background level was not distinguishable from this number.
        #
        # WORSE THAN LOOSE — IT IS THE WHOLE MEASUREMENT. Re-running five 6p days at
        # 1e-9 instead of 1e-7 scaled every reported improvement by exactly 1/100, the
        # same factor as the slack, leaving `improvement / r2_slack` unchanged to four
        # significant figures (day 4: 1648.7 -> 1648.6). The "fairness gain" was the
        # slack: (R2) written as `>=  z* - slack` lets the LP lower the worst-off share
        # by UNDER-COLLECTING, which is not redistribution but extra subsidy — exactly
        # the trade `maximin criterion.md` §4 warns about, and `truncation_cost` was
        # equal to `r2_slack` in every run, which was the tell.
        #
        # So this must be 0: (R2) as a hard `sum_j sigma_j >= z*`. theta* stays feasible
        # because `verify_at_optimum` puts `sum_j hat sigma_j(pi*) - z*` at -8e-12 (6p)
        # to -1.1e-7 (15p), inside SCIP's feasibility tolerance. Overridable only to
        # reproduce the diagnosis above.
        self.r2_slack = (0.0 if r2_slack is None else
                         float(r2_slack) * max(1.0, abs(float(cg.master.model.getObjVal()))))
        self.rows = coupling_rows(self.master, self.T)
        self.z_star = float(self.master.model.getObjVal())          # v^CHP
        self.pi_star = _row_duals(self.master, self.rows)
        m = self.master.model
        self.sigma_star = {u: float(m.getDualsolLinear(
            m.getTransformedCons(m.data['cons']['convexity'][u]))) for u in self.players}

        # column pool, copied out of the master as plain data. Phase 2 grows it without
        # touching the (already solved) RMP.
        self.pool = {u: [] for u in self.players}
        for u in self.players:
            for _, col in sorted(m.data['vars'][u].items()):
                self._append_column(u, col['solution'])
        self.n_pricing_calls = 0
        self.history = []

    # ------------------------------------------------------------------ pool bookkeeping
    def _append_column(self, u, solution):
        cost = float(calculate_column_cost(u, solution, self.cg.subproblems[u].parameters, self.T))
        self.pool[u].append({'cost': cost,
                             'a': column_coefficients(solution, u, self.rows),
                             'solution': solution})

    def pool_sizes(self):
        return {u: len(self.pool[u]) for u in self.players}

    def sigma_hat(self, pi):
        """`hat sigma_j(pi) = min_{q in pool} (c_j^q - pi^T a_j^q)`, the pool's estimate."""
        out = {}
        for u in self.players:
            out[u] = min(col['cost'] - sum(pi[r] * v for r, v in col['a'].items())
                         for col in self.pool[u])
        return out

    # ------------------------------------------------------------------ the restricted LP
    def _shared_var_rows(self, mdl, pi):
        """(R4): `A_0^T pi <= d_0`, one row per shared community variable.

        r_sym (block i) sits at +1 in both reserve rows of its block with objective
        `-|blk| pi_res`, so its non-negative reduced cost reads
        `sum_{t in blk} (pi_up + pi_dn) <= -|blk| pi_res`; under the one-sided product
        the two directions separate. p sits at -1 in every peak row with objective
        `+pi_peak`, giving `sum_t pi_peak >= -pi_peak`.
        """
        mp = self.master
        if mp.enable_reserve:
            sym = mp.reserve_product == 'symmetric'
            for blk in mp.reserve_blocks:
                if sym:
                    mdl.addCons(quicksum(pi[('resup', t)] + pi[('resdn', t)] for t in blk)
                                <= -len(blk) * mp.pi_res)
                else:
                    mdl.addCons(quicksum(pi[('resup', t)] for t in blk) <= -len(blk) * mp.pi_up)
                    mdl.addCons(quicksum(pi[('resdn', t)] for t in blk) <= -len(blk) * mp.pi_dn)
        if mp.enable_peak:
            mdl.addCons(quicksum(pi[('peak', t)] for t in self.T) >= -mp.pi_peak)

    def _base_lp(self):
        """The optimal-face polyhedron restricted to the current pool: (R2)+(R3)+(R4).

        Returns (model, pi, sigma). No objective is set.
        """
        mdl = Model("maximin_face")
        mdl.hideOutput()
        pi = {}
        M = self.pi_bound
        for kind, t in self.rows:
            # <= rows carry non-positive duals in a minimisation; balance rows are
            # equalities and free. Both are boxed -- see default_pi_bound.
            ub = 0.0 if kind in ('resup', 'resdn', 'peak') else M
            pi[(kind, t)] = mdl.addVar(name=f"pi_{kind}_{t}", vtype="C", lb=-M, ub=ub)
        sigma = {u: mdl.addVar(name=f"sigma_{u}", vtype="C", lb=None) for u in self.players}

        # (R3) sigma_j + pi^T a_j^q <= c_j^q
        for u in self.players:
            for col in self.pool[u]:
                mdl.addCons(sigma[u] + quicksum(v * pi[r] for r, v in col['a'].items())
                            <= col['cost'])
        self._shared_var_rows(mdl, pi)

        # (R2) dual feasibility already gives sum_j sigma_j <= z*; forcing >= pins the
        # optimal face. Written with a tolerance for the same numerical-safety reason
        # the note writes it as an inequality. That tolerance is not free: it widens the
        # face by exactly this much, so the truncation cost at termination is bounded by
        # `r2_slack + n * tol` and in practice the slack term is the larger of the two.
        mdl.addCons(quicksum(sigma[u] for u in self.players)
                    >= self.z_star - self.r2_slack)
        return mdl, pi, sigma

    @staticmethod
    def _safe_optimize(mdl):
        """Optimize, reporting an LP failure as a status instead of an exception.

        pyscipopt raises on `error in LP solver`, which on these degenerate faces is a
        real possibility rather than a bug. Whether that is fatal depends on which stage
        raised it, so the decision belongs to the caller, not here.
        """
        try:
            mdl.optimize()
            return mdl.getStatus()
        except Exception as exc:
            return f"exception:{exc}"

    def _values(self, mdl, pi, sigma):
        return ({r: float(mdl.getVal(v)) for r, v in pi.items()},
                {u: float(mdl.getVal(v)) for u, v in sigma.items()})

    def _box_active(self, pi_val):
        """Rows whose dual sits on the artificial box — where the box, not the pool, bit.

        Expected on rows no pooled column touches, and harmless there. On a row that
        columns DO touch it would mean the box is shaping the answer, so it is counted
        and reported rather than silently tolerated.
        """
        eps = 1e-6 * self.pi_bound
        touched = set()
        for u in self.players:
            for col in self.pool[u]:
                touched.update(col['a'])
        at = [r for r, v in pi_val.items() if abs(abs(v) - self.pi_bound) <= eps]
        return {'n_at_bound': len(at),
                'n_at_bound_touched': sum(1 for r in at if r in touched),
                'rows_at_bound_touched': [f"{k}_{t}" for k, t in at if (k, t) in touched],
                'pi_bound': self.pi_bound}

    # ------------------------------------------------------------------ Phase 1.5
    def screen(self):
        """Singleton screen: is the optimal face wide enough to be worth searching?

        `2n` LPs maximising and minimising each `chi_j` over the RESTRICTED face. Since
        the restricted face contains the true one, each width is an UPPER bound on the
        true width -- so width 0 certifies `Theta* = {pi*}` and Phase 2 can be skipped,
        while a positive width certifies nothing (it may be pool truncation). A screen
        for skipping work, not a measurement of the spread.
        """
        t0 = time.time()
        widths, lo, hi = {}, {}, {}
        for u in self.players:
            for sense, store in (('minimize', lo), ('maximize', hi)):
                mdl, pi, sigma = self._base_lp()
                mdl.setObjective(sigma[u], sense)
                mdl.optimize()
                if mdl.getStatus() != 'optimal':
                    raise RuntimeError(f"screen LP for {u} ({sense}) ended {mdl.getStatus()}")
                store[u] = float(mdl.getObjVal())
            widths[u] = hi[u] - lo[u]
        res = {'width': widths, 'sigma_min': lo, 'sigma_max': hi,
               'max_width': max(widths.values()) if widths else 0.0,
               'n_lps': 2 * len(self.players), 'time_s': time.time() - t0,
               'z_star': self.z_star}
        if self.verbose:
            print("\n" + "=" * 78)
            print("PHASE 1.5 — singleton screen over the RESTRICTED optimal face")
            print("=" * 78)
            print(f"  {'player':>8} {'sigma* (CG)':>14} {'min':>14} {'max':>14} {'width':>12}")
            for u in self.players:
                print(f"  {u:>8} {self.sigma_star[u]:>14.6f} {lo[u]:>14.6f} "
                      f"{hi[u]:>14.6f} {widths[u]:>12.6f}")
            print(f"  max width = {res['max_width']:.6f}   ({res['n_lps']} LPs, "
                  f"{res['time_s']:.1f}s)")
        return res

    def _closest_optimum(self, t_hat):
        """Stage B: among the (MM) optima, the `pi` closest to `pi*` in l1.

        Two things make this necessary rather than cosmetic.

        (a) `maximin criterion.md` sec.6.2: (MM) pins `t*` but not the whole vector, so
            a simplex solver returns an arbitrary vertex of the remaining freedom. A
            documented tie-break -- stay as close as possible to the price vector column
            generation actually produced -- makes the reported allocation reproducible
            instead of solver-dependent. `chi` is then determined too, since (2d) reads
            it off the pricing oracle at `hat pi`.

        (b) Convergence. Without it the loop wanders: only `max_j sigma_j` is in the
            objective, so every iterate is free to run off to some far vertex of the
            restricted face (up against the artificial box, in the 6-prosumer instance),
            where the pool is loosest and the re-priced violation is largest. The cuts
            then arrive in no useful order and `t_hat` does not move. Anchoring at
            `pi*` is the same stabilisation idea Phase 1 already uses via Wentges
            smoothing, applied to the second loop.

        Lexicographic, not a weighted sum: `t` is fixed at its optimum first, so the
        fairness objective is never traded against the proximity term.
        """
        mdl, pi, sigma = self._base_lp()
        slack = 1e-9 * max(1.0, abs(t_hat))
        for u in self.protected:
            mdl.addCons(sigma[u] <= t_hat + slack)
        dev = {}
        for r, v in pi.items():
            d = mdl.addVar(name=f"dev_{r[0]}_{r[1]}", vtype="C", lb=0.0)
            mdl.addCons(d >= v - self.pi_star[r])
            mdl.addCons(d >= self.pi_star[r] - v)
            dev[r] = d
        mdl.setObjective(quicksum(dev.values()), "minimize")
        # Failure here costs stabilisation, not correctness: the caller falls back to
        # stage A's iterate, which is a valid point of the restricted face and yields a
        # valid cut. Only the choice among ties is lost, so the run degrades to the
        # unstabilised behaviour of sec.1.3 for that iteration rather than dying.
        if self._safe_optimize(mdl) != 'optimal':
            return None, None
        return self._values(mdl, pi, sigma)

    # ------------------------------------------------------------------ Phase 2
    def _pricing_duals(self, pi):
        """Split `pi` into the argument shape `solve_pricing` wants (raw, no sign flip)."""
        de = {t: pi[('elec', t)] for t in self.T}
        dh = {t: pi[('heat', t)] for t in self.T}
        dg = {t: pi[('hydro', t)] for t in self.T}
        has_res = self.master.enable_reserve and ('resup', self.T[0]) in pi
        has_pk = self.master.enable_peak and ('peak', self.T[0]) in pi
        dru = {t: pi[('resup', t)] for t in self.T} if has_res else None
        drd = {t: pi[('resdn', t)] for t in self.T} if has_res else None
        dpk = {t: pi[('peak', t)] for t in self.T} if has_pk else None
        return de, dh, dg, dru, drd, dpk

    def reprice(self, pi):
        """Exact `sigma_j(pi)` for every j, by the pricing oracle. n independent MIPs."""
        de, dh, dg, dru, drd, dpk = self._pricing_duals(pi)
        sig, sol = {}, {}
        for u in self.players:
            sub = self.cg.subproblems[u]
            sub.model.hideOutput()
            # the convexity dual only shifts the reduced cost by a constant; the object
            # wanted here is the subproblem objective itself, so pass 0.0.
            _, solution, obj = sub.solve_pricing(de, dh, dg, 0.0,
                                                 dual_resup=dru, dual_resdn=drd, dual_peak=dpk)
            self.n_pricing_calls += 1
            if obj is None:
                raise RuntimeError(f"pricing subproblem for {u} did not solve at hat pi")
            sig[u] = float(obj)
            sol[u] = solution
        return sig, sol

    def solve(self, max_iterations=50):
        """(MM) with the cutting-plane loop of sec.4 (2a)-(2d).

        min t  s.t.  sigma_j <= t for j in P, over the restricted face; re-price at
        `hat pi`; append the violated columns; repeat. Cost convention, so minimising
        the largest cost share IS maximising the smallest payoff.
        """
        t_start = time.time()
        converged = False
        pi_hat = sigma_hat = sigma_exact = None
        t_hat = None
        self.n_prox_fallbacks = 0
        for it in range(1, max_iterations + 1):
            # (2a)
            mdl, pi, sigma = self._base_lp()
            tvar = mdl.addVar(name="t", vtype="C", lb=None)
            for u in self.protected:
                mdl.addCons(sigma[u] <= tvar)
            mdl.setObjective(tvar, "minimize")
            st = self._safe_optimize(mdl)
            if st != 'optimal':
                # Fatal only on the first iteration, where there is no iterate to report.
                # Later, everything already computed stays valid: `t_hat` from the last
                # solved iteration is still a lower bound on `t*`, and the allocation is
                # still stable by (P3). Stop and say so rather than lose the run.
                if sigma_exact is None:
                    raise RuntimeError(f"(MM) iteration {it} ended {st}")
                if self.verbose:
                    print(f"  [MM {it:3d}] LP ended {st} — stopping early; the bracket "
                          "and the reported allocation remain valid")
                break
            t_hat = float(mdl.getObjVal())
            pi_hat, sigma_hat = self._values(mdl, pi, sigma)
            # (2a') lexicographic tie-break — see _closest_optimum
            pi_prox, sigma_prox = self._closest_optimum(t_hat)
            if pi_prox is not None:
                pi_hat, sigma_hat = pi_prox, sigma_prox
            else:
                self.n_prox_fallbacks += 1
            box = self._box_active(pi_hat)

            # (2b) re-price over ALL of N: (R3) is global, and a stale column anywhere
            # loosens the face everywhere.
            sigma_exact, sols = self.reprice(pi_hat)

            # (2c) delta_j = hat sigma_j - sigma_j(hat pi) >= 0 in the cost convention:
            # the restricted face believes j's cost share can be larger than the
            # pricing oracle allows, i.e. promises the payoff is smaller than it is.
            delta = {u: sigma_hat[u] - sigma_exact[u] for u in self.players}
            worst = max(delta, key=delta.get)
            # the fairness/subsidy trade-off, per iteration: `t_hat` is the restricted
            # LP's optimistic value, `worst_exact` the same thing re-evaluated by the
            # oracle, and `truncation` the extra subsidy that buys the difference.
            sum_exact = sum(sigma_exact.values())
            rec = {'iter': it, 't_hat': t_hat, 'max_delta': delta[worst],
                   'argmax_delta': worst, 'sum_sigma_exact': sum_exact,
                   'worst_exact': max(sigma_exact[u] for u in self.protected),
                   'truncation': self.z_star - sum_exact,
                   'pool': sum(self.pool_sizes().values()),
                   'box': box}
            self.history.append(rec)
            if self.verbose:
                warn = (f"  [box: {box['n_at_bound_touched']} PRICED rows at bound]"
                        if box['n_at_bound_touched'] else "")
                print(f"  [MM {it:3d}] t_hat = {t_hat:14.6f}   max delta = {delta[worst]:11.3e}"
                      f" ({worst})   sum sigma(hat pi) = {rec['sum_sigma_exact']:14.6f}"
                      f"   |pool| = {rec['pool']}{warn}")
            if delta[worst] <= self.tol:
                converged = True
                break
            for u in self.players:
                if delta[u] > self.tol:
                    self._append_column(u, sols[u])

        # (2d) report the RE-EVALUATED allocation. hat sigma is an over-estimate of the
        # cost share when a column is missing, so it is not a stable allocation; the
        # pricing oracle's value is.
        chi = dict(sigma_exact)
        return self._package(chi, pi_hat, t_hat, sigma_hat, converged, time.time() - t_start)

    def _package(self, chi, pi_hat, t_hat, sigma_hat, converged, elapsed):
        n = len(self.players)
        sum_chi = sum(chi.values())
        # truncation cost: z* - sum_j sigma_j(hat pi) >= 0, zero iff hat pi is on the
        # true optimal face. Extra subsidy paid for stopping early.
        truncation = self.z_star - sum_chi
        worst_now = max(chi[u] for u in self.protected)
        worst_cg = max(self.sigma_star[u] for u in self.protected)
        # A bracket on t*, valid whether or not the loop converged, and the useful
        # thing to report when it did not. The restricted face contains the true one,
        # so `t_hat` under-estimates the smallest achievable worst-off cost; and pi* is
        # itself a point of Theta*, so classical Owen's worst-off cost is achievable.
        # The width is therefore an upper bound on everything the criterion can still
        # win -- which settles the sec.5.1 gate without running to convergence.
        t_lb = self.history[-1]['t_hat'] if self.history else worst_cg
        max_gain = worst_cg - t_lb
        res = {
            'converged': converged,
            'iterations': len(self.history),
            'n_pricing_calls': self.n_pricing_calls,
            'time_s': elapsed,
            'protected': list(self.protected),
            'z_star': self.z_star,
            'pi_hat': {f"{k}_{t}": v for (k, t), v in pi_hat.items()},
            # the CG dual, kept alongside so the two price vectors can be differenced
            # offline: which coupling rows the selection actually repriced is the only
            # direct evidence of WHERE the optimal face has width.
            'pi_star': {f"{k}_{t}": v for (k, t), v in self.pi_star.items()},
            'chi': chi,                       # re-evaluated sigma_j(hat pi), cost conv.
            'sigma_hat_lp': sigma_hat,        # restricted LP's value, for the delta audit
            'sigma_star_cg': dict(self.sigma_star),
            'sum_chi': sum_chi,
            't_star_cost': t_hat,             # (MM) objective: largest protected cost share
            'worst_off_cost_maximin': worst_now,   # the same thing, re-evaluated
            'worst_off_cost_classical': worst_cg,
            # payoff convention, which is how the criterion is stated
            'worst_off_payoff_maximin': -worst_now,
            'worst_off_payoff_classical': -worst_cg,
            'worst_off_improvement': worst_cg - worst_now,
            't_star_lower_bound': t_lb,
            't_star_upper_bound': worst_cg,
            'max_remaining_gain': max_gain,
            'truncation_cost': truncation,
            # sec.4: stopping early costs `n * eps` in subsidy. The (R2) slack widens
            # the face by the same kind of amount, so the honest bound is their sum --
            # and on the instances measured so far the slack term dominates, i.e. the
            # residual truncation is LP tolerance, not early stopping.
            'truncation_bound': n * self.tol + self.r2_slack,
            'truncation_bound_n_eps': n * self.tol,
            'r2_slack': self.r2_slack,
            'box': self.history[-1]['box'] if self.history else None,
            'pool_sizes': self.pool_sizes(),
            # iterations that ran without the proximity tie-break because its LP failed.
            # Costs stabilisation only, but a large count explains slow convergence.
            'n_prox_fallbacks': getattr(self, 'n_prox_fallbacks', 0),
            'history': self.history,
        }
        if self.verbose:
            print("\n" + "=" * 78)
            print("MAXIMIN SELECTION OVER Theta*  (cost convention; payoff = -chi)")
            print("=" * 78)
            print(f"  converged = {converged}   iterations = {res['iterations']}   "
                  f"pricing MIPs = {self.n_pricing_calls}   ({elapsed:.1f}s)")
            print(f"  z* = {self.z_star:.6f}   sum_j chi_j = {sum_chi:.6f}   "
                  f"truncation cost = {truncation:.3e}")
            print(f"  worst-off payoff: classical Owen {-worst_cg:.6f}  ->  maximin "
                  f"{-worst_now:.6f}   (improvement {res['worst_off_improvement']:+.6f})")
            print(f"  t* in [{t_lb:.6f}, {worst_cg:.6f}]  =>  the criterion can win at "
                  f"most {max_gain:.6f} in worst-off cost")
            print(f"  {'player':>8} {'chi (maximin)':>16} {'sigma* (Owen)':>16} {'shift':>12}")
            for u in self.players:
                mark = ' *' if u in self.protected else '  '
                print(f"  {u:>8} {chi[u]:>16.6f} {self.sigma_star[u]:>16.6f} "
                      f"{chi[u] - self.sigma_star[u]:>12.6f}{mark}")
        return res

    # ------------------------------------------------------------------ self-test
    def verify_at_optimum(self, atol=1e-5):
        """Does the extracted pool reproduce the master's own duals at pi*?

        `hat sigma_j(pi*)` must equal the convexity dual `sigma_j*` (that identity IS
        the convergence certificate of Phase 1), and the two must sum to `z*`. A
        mismatch means the extracted costs or coupling coefficients disagree with the
        RMP, and everything downstream would be measuring the wrong polyhedron.
        """
        sh = self.sigma_hat(self.pi_star)
        err = {u: sh[u] - self.sigma_star[u] for u in self.players}
        worst = max(err, key=lambda u: abs(err[u]))
        out = {'max_abs_err': abs(err[worst]), 'argmax': worst,
               'sum_sigma_hat': sum(sh.values()), 'z_star': self.z_star,
               'sum_err': sum(sh.values()) - self.z_star,
               'ok': abs(err[worst]) <= atol * max(1.0, abs(self.z_star))}
        if self.verbose:
            print(f"  [verify] max |hat sigma_j(pi*) - sigma_j*| = {out['max_abs_err']:.3e} "
                  f"({worst});  sum - z* = {out['sum_err']:.3e};  "
                  f"{'OK' if out['ok'] else 'MISMATCH'}")
        return out


# --------------------------------------------------------------------------- convenience
def owen_gap_corrected(chi, v_mip, players):
    """Spread the residual subsidy uniformly, as `compute_owen_allocation` does.

    Kept separate from `chi` because the two answer different questions: `chi` is the
    stable-but-not-efficient point that (P3) applies to, and this is the efficient
    weak-eps-core point. The maximin criterion selects the former; the correction is
    the same afterwards either way.
    """
    n = len(players)
    gap = sum(chi.values()) - v_mip
    return {u: chi[u] - gap / n for u in players}, gap, abs(gap) / n

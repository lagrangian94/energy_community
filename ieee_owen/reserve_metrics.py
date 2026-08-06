"""
Reserve / peak output schema and derived metrics (reserve.txt sec.3.2 - 3.4).

The MIP already extracts every primal quantity we need -- solve_and_extract_results
walks model.data["vars"], so r_sym, chi_peak_E, the per-player aggregates
r_plus/r_minus and the per-asset splits r_{plus,minus}_{sto,els,hp} all land in
`results` for free. What is missing is the derived layer: the pooling-gain
decomposition, the peak coincidence factor, and the post-hoc checks. That is what
this module computes.

Duals (mu_plus, mu_minus, xi) are NOT available from the MIP -- they come from the
column-generation master, i.e. solution['convex_hull_prices'] with keys
'reserve_up' / 'reserve_dn' / 'peak'. Pass that dict as `prices` to fill in the
dual-side diagnostics and the budget-balance checks; omit it for primal-only runs.

Entry point: reserve_peak_metrics(...).
"""
import numpy as np

# Asset splits written by LocalEnergyMarket._add_reserve_headroom_cons.
ASSETS = ('sto', 'els', 'hp')


def _get(results, key):
    d = results.get(key) if isinstance(results, dict) else None
    return d if isinstance(d, dict) else {}


def _series(results, key, players, T):
    """{u: {t: val}} for a (u,t)-indexed results entry; missing entries are 0."""
    d = _get(results, key)
    return {u: {int(t): float(d.get((u, t), 0.0)) for t in T} for u in players}


def _scalar(results, key):
    v = results.get(key) if isinstance(results, dict) else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def max_headroom_at_dispatch(results, params, players, T):
    """Maximal per-asset reserve headroom implied by the solved dispatch.

    WHY THIS EXISTS. r_plus/r_minus carry no objective cost, so the solver has no
    incentive to report more headroom than r_sym needs: the optimal r_plus[j,t]
    is anywhere between the amount required and the physical maximum, and which
    value comes out is arbitrary. Reading min_t min(r+_jt, r-_jt) straight off
    the solution therefore measures solver degeneracy, not capability, and would
    make the sec.3.3 pooling gain meaningless (it reads ~0 for every prosumer).

    With the dispatch FIXED at the solved values the headroom rows decouple and
    each bound is just the row's slack, so the true maxima are available in
    closed form -- no extra solve. Same rows as
    LocalEnergyMarket._add_reserve_headroom_cons, so the nu_dis / 1/nu_ch and
    the nu_cop unit conversion carry over identically.

    Returns (up_max, dn_max) as {u: {t: MW_electric}}.
    """
    up_max = {u: {t: 0.0 for t in T} for u in players}
    dn_max = {u: {t: 0.0 for t in T} for u in players}

    sto_u, sto_d = _get(results, 'r_plus_sto'), _get(results, 'r_minus_sto')
    els_u = _get(results, 'r_plus_els')
    hp_u = _get(results, 'r_plus_hp')
    b_dis, b_ch, s_E = _get(results, 'b_dis_E'), _get(results, 'b_ch_E'), _get(results, 's_E')
    fl_d, p_out = _get(results, 'fl_d'), _get(results, 'p')
    z_on_G, z_sb_G, z_on_H = _get(results, 'z_on_G'), _get(results, 'z_sb_G'), _get(results, 'z_on_H')

    nu_ch = float(params.get('nu_ch_E', 1.0))
    nu_dis = float(params.get('nu_dis_E', 1.0))
    c_sb = float(params.get('c_sb_G', 0.0))
    c_min_G = float(params.get('c_min_G', 0.0))
    c_max_G = float(params.get('c_max_G', 0.0))
    c_min_H = float(params.get('c_min_H', 0.0))
    c_max_H = float(params.get('c_max_H', 0.0))

    for u in players:
        for t in T:
            # -- electric storage: power slack and energy (1h duration) slack
            if (u, t) in sto_u:
                pw = float(params.get(f'storage_power_E_{u}', 0.0))
                cap = float(params.get(f'storage_capacity_E_{u}', 0.0))
                bd, bc = float(b_dis.get((u, t), 0.0)), float(b_ch.get((u, t), 0.0))
                soc = float(s_E.get((u, t), 0.0))
                up_max[u][t] += max(0.0, min(pw - (bd - bc), nu_dis * soc))
                dn_max[u][t] += max(0.0, min(pw - (bc - bd), (cap - soc) / nu_ch))
            # -- electrolyzer: distance to each end of the operating window
            if (u, t) in els_u:
                cap = float(params.get(f'els_cap_{u}', params.get('els_cap', 0.0)))
                d = float(fl_d.get((u, 'elec', t), 0.0))
                on, sb = float(z_on_G.get((u, t), 0.0)), float(z_sb_G.get((u, t), 0.0))
                up_max[u][t] += max(0.0, d - c_min_G * cap * on - c_sb * cap * sb)
                dn_max[u][t] += max(0.0, c_max_G * cap * on + c_sb * cap * sb - d)
            # -- heat pump: window slack in MWth converted to MWel by nu_cop
            if (u, t) in hp_u:
                cap = float(params.get(f'hp_cap_{u}', params.get('hp_cap', 0.0)))
                nu_cop = float(params.get(f'nu_cop_{u}', 1.0))
                ph = float(p_out.get((u, 'hp', t), 0.0))
                on = float(z_on_H.get((u, t), 0.0))
                up_max[u][t] += max(0.0, (ph - c_min_H * cap * on) / nu_cop)
                dn_max[u][t] += max(0.0, (c_max_H * cap * on - ph) / nu_cop)
    return up_max, dn_max


def _pooling_decomposition(up, dn, T, players, r_sym, standalone=None):
    """sec.3.3 metric 2, as TWO separate objects -- they are not one chain.

    (1) POOLING GAIN, the economic answer: what the community can sell versus
        what its members could sell on their own,

            gain = r_sym(N)  -  sum_j r_sym({j}),

        each side at its OWN optimal dispatch. Needs the standalone solves; the
        `standalone` argument supplies them.

    (2) MECHANISM SPLIT, why pooling helps: computed at the fixed community
        dispatch so every term sits on a common footing,

            A  = sum_j min_t min(u_j(t), d_j(t))       no pooling
            T1 = min_t sum_j min(u_j(t), d_j(t))       pool ACROSS HOURS, each
                                                       prosumer still self-balances
            B  = min_t min(sum_j u_j(t), sum_j d_j(t)) + pool ACROSS DIRECTIONS

        A <= T1 <= B, so (T1 - A) is the gain from time diversity -- prosumers
        whose binding hours differ cover for each other -- and (B - T1) the gain
        from direction diversity, one prosumer's upward surplus offsetting
        another's downward shortfall at the binding hour.

    Do NOT chain (2) onto (1): their baselines differ (community dispatch vs each
    prosumer's own), so A - sum_j r_sym({j}) is not a pooling term at all -- it is
    just the community dispatch being optimized for total profit rather than for
    any one member's reserve capability, and it is routinely negative.

    u_j / d_j MUST be the maximal headroom from max_headroom_at_dispatch, not the
    raw r_plus/r_minus values -- see that function for why.

    Note the two mins commute for a single prosumer, min_t min(u,d) =
    min(min_t u, min_t d), so there is no distinct "direction-first" ordering to
    report: the mechanism split is unique, not Shapley-style order-dependent.
    """
    per_player_sym = {u: float(min(min(up[u][t], dn[u][t]) for t in T)) for u in players}
    A = float(sum(per_player_sym.values()))
    T1 = float(min(sum(min(up[u][t], dn[u][t]) for u in players) for t in T))
    sum_up = {t: float(sum(up[u][t] for u in players)) for t in T}
    sum_dn = {t: float(sum(dn[u][t] for u in players)) for t in T}
    B = float(min(min(sum_up[t], sum_dn[t]) for t in T))
    out = {
        'community_r_sym': float(r_sym),
        'mechanism': {
            'no_pooling_A': A,
            'time_pooled_T1': T1,
            'full_pooled_B': B,
            'gain_time': T1 - A,
            'gain_direction': B - T1,
            'gain_total': B - A,
            # share of the mechanism gain attributable to each channel
            'share_time': ((T1 - A) / (B - A)) if (B - A) > 1e-12 else None,
            'share_direction': ((B - T1) / (B - A)) if (B - A) > 1e-12 else None,
        },
    }
    if standalone:
        A0 = float(sum(standalone.get(u, 0.0) for u in players))
        out['standalone_sum'] = A0
        out['gain_abs'] = float(r_sym) - A0
        out['gain_ratio'] = (float(r_sym) / A0) if A0 > 1e-12 else None
        out['binding_member'] = max(players, key=lambda u: standalone.get(u, 0.0))
    return per_player_sym, sum_up, sum_dn, out


def solve_standalone_r_sym(players, time_periods, params, model_type='mip', verbose=False):
    """r_sym({j}) for each prosumer alone -- the true no-pooling baseline.

    Same construction CoreComputation uses for a coalition value (a
    LocalEnergyMarket over just that player, dwr=False), so the numbers are
    consistent with v({j}). Costs one small MIP per prosumer; call it only when
    the pooling gain is actually wanted.
    """
    from compact_utility import LocalEnergyMarket, solve_and_extract_results

    out = {}
    for u in players:
        lem = LocalEnergyMarket(players=[u], time_periods=time_periods,
                                parameters=params, model_type=model_type, dwr=False)
        lem.model.hideOutput()
        lem.model.optimize()
        status, res = solve_and_extract_results(lem.model)
        if status not in ("optimal", "gaplimit") or not res:
            if verbose:
                print(f"  [standalone] {u}: status={status}, r_sym unavailable -> 0")
            out[u] = 0.0
            continue
        out[u] = _scalar(res, 'r_sym')
        if verbose:
            print(f"  [standalone] {u}: r_sym = {out[u]:.6f} MW")
    return out


def reserve_peak_metrics(results, params, players, time_periods, prices=None,
                         standalone=None, tol=1e-6):
    """Build the reserve.txt sec.3.2 output record from one solved instance.

    results    : dict from solve_and_extract_results (grand-coalition MIP)
    params     : the LEM parameter dict (for pi_res / pi_E_peak)
    prices     : optional solution['convex_hull_prices'] for the dual-side fields
    standalone : optional {u: r_sym({u})} from solve_standalone_r_sym, giving the
                 true no-pooling baseline for the sec.3.3 gain
    """
    T = [int(t) for t in time_periods]
    pi_res = float(params.get('pi_res', 0.0) or 0.0)
    pi_peak = float(params.get('pi_E_peak', 0.0) or 0.0)
    enable_reserve = bool(params.get('enable_reserve', False))
    enable_peak = bool(params.get('enable_peak', False))
    prices = prices or {}

    out = {
        'enabled': {'reserve': enable_reserve, 'peak': enable_peak},
        'prices': {
            'pi_res_eur_per_mw_h': pi_res,
            'pi_E_peak_eur_per_mw': pi_peak,
            # r_sym is held for the whole horizon (reserve.txt sec.2.1)
            'horizon_payment_per_mw': len(T) * pi_res,
        },
        'checks': {},
    }

    # ------------------------------------------------------------------ reserve
    if enable_reserve:
        r_sym = _scalar(results, 'r_sym')
        # As OFFERED by the solver (degenerate above what r_sym needs -- these are
        # the quantities the CHP settlement prices, so they are reported as-is)
        up = _series(results, 'r_plus', players, T)
        dn = _series(results, 'r_minus', players, T)
        # As PHYSICALLY AVAILABLE at the same dispatch (used for every capability
        # metric -- see max_headroom_at_dispatch)
        up_max, dn_max = max_headroom_at_dispatch(results, params, players, T)
        per_player_sym, sum_up, sum_dn, pooling = _pooling_decomposition(
            up_max, dn_max, T, players, r_sym, standalone=standalone)

        out['r_sym'] = r_sym
        out['reserve_revenue'] = len(T) * pi_res * r_sym
        out['r_plus_by_player'] = up
        out['r_minus_by_player'] = dn
        out['r_plus_total_by_player'] = {u: float(sum(up[u].values())) for u in players}
        out['r_minus_total_by_player'] = {u: float(sum(dn[u].values())) for u in players}
        out['headroom_max_plus_by_player'] = up_max
        out['headroom_max_minus_by_player'] = dn_max
        out['community_headroom_plus_t'] = sum_up
        out['community_headroom_minus_t'] = sum_dn
        out['community_r_plus_t'] = {t: float(sum(up[u][t] for u in players)) for t in T}
        out['community_r_minus_t'] = {t: float(sum(dn[u][t] for u in players)) for t in T}
        # sec.3.2: stand-alone comparison value, the input to the pooling gain.
        # ..._at_dispatch keeps the community dispatch fixed; ..._resolved re-optimizes
        # each prosumer alone and is the defensible one (present iff `standalone` given).
        out['standalone_r_sym_by_player_at_dispatch'] = per_player_sym
        if standalone:
            out['standalone_r_sym_by_player_resolved'] = dict(standalone)
        out['pooling'] = pooling

        # per-asset decomposition (which technology actually supplies the offer)
        by_asset = {}
        for a in ASSETS:
            a_up = _series(results, f'r_plus_{a}', players, T)
            a_dn = _series(results, f'r_minus_{a}', players, T)
            by_asset[a] = {
                'up_by_player': a_up, 'dn_by_player': a_dn,
                'up_total': float(sum(sum(v.values()) for v in a_up.values())),
                'dn_total': float(sum(sum(v.values()) for v in a_dn.values())),
                # contribution at the binding hour is what actually sets r_sym
                'up_at_binding': None, 'dn_at_binding': None,
            }

        # binding hours from the primal: coupling rows tight at r_sym. The rows
        # are written on the OFFERED r_plus/r_minus, so use those, not the maxima
        # (a row can be tight on offers while physical headroom is still slack).
        off_up = out['community_r_plus_t']
        off_dn = out['community_r_minus_t']
        up_bind = [t for t in T if abs(off_up[t] - r_sym) <= tol]
        dn_bind = [t for t in T if abs(off_dn[t] - r_sym) <= tol]
        for a in ASSETS:
            if up_bind:
                by_asset[a]['up_at_binding'] = {
                    int(t): float(sum(by_asset[a]['up_by_player'][u][t] for u in players))
                    for t in up_bind}
            if dn_bind:
                by_asset[a]['dn_at_binding'] = {
                    int(t): float(sum(by_asset[a]['dn_by_player'][u][t] for u in players))
                    for t in dn_bind}
        out['r_by_asset'] = by_asset
        out['binding'] = {'up_hours_primal': up_bind, 'dn_hours_primal': dn_bind}

        # sec.3.4 post-hoc verification.
        # (a) the LP structure claim: r_sym* = min_t min over the OFFERED sums
        out['checks']['r_sym_equals_min_offered'] = bool(
            abs(r_sym - min(min(off_up[t], off_dn[t]) for t in T)) <= tol)
        # (b) the physical claim: r_sym never exceeds real headroom in any hour
        out['checks']['r_sym_le_max_headroom_all_t'] = bool(
            all(r_sym <= min(sum_up[t], sum_dn[t]) + tol for t in T))
        # (c) offers are themselves within physical headroom (validates the
        #     closed-form maxima against the model's own headroom rows)
        out['checks']['offers_le_max_headroom'] = bool(
            all(up[u][t] <= up_max[u][t] + tol and dn[u][t] <= dn_max[u][t] + tol
                for u in players for t in T))
        agg_err = 0.0
        for u in players:
            for t in T:
                agg_err = max(agg_err,
                              abs(up[u][t] - sum(by_asset[a]['up_by_player'][u][t] for a in ASSETS)),
                              abs(dn[u][t] - sum(by_asset[a]['dn_by_player'][u][t] for a in ASSETS)))
        out['checks']['asset_aggregation_max_err'] = float(agg_err)

        # dual-side diagnostics (sec.3.2: which hour and which direction is scarce)
        mu_p = {int(t): float(v) for t, v in (prices.get('reserve_up') or {}).items()}
        mu_m = {int(t): float(v) for t, v in (prices.get('reserve_dn') or {}).items()}
        if mu_p or mu_m:
            out['mu_plus_t'] = mu_p
            out['mu_minus_t'] = mu_m
            out['binding']['up_hours_dual'] = [t for t in T if mu_p.get(t, 0.0) > tol]
            out['binding']['dn_hours_dual'] = [t for t in T if mu_m.get(t, 0.0) > tol]
            # r_sym has zero reduced cost when basic => sum_t (mu+ + mu-) = |T|*pi_res
            mu_sum = float(sum(mu_p.values()) + sum(mu_m.values()))
            target = len(T) * pi_res
            out['checks']['dual_sum_mu'] = mu_sum
            out['checks']['dual_sum_mu_target'] = target
            out['checks']['budget_balance_duals_ok'] = bool(
                abs(mu_sum - target) <= 1e-6 * max(1.0, abs(target)))
            # Price x quantity settlement of the reserve rows: CHP prices (master
            # LP duals) applied to the MIP dispatch quantities.
            #
            # NOT an invariant, so NOT a check. Complementary slackness ties duals
            # to the primal of the SAME LP; here the duals come from the CG master
            # while the quantities come from the MIP, a different solution. A row
            # can therefore carry mu > 0 while the MIP leaves it slack, and the
            # settlement then differs from |T|*pi_res*r_sym by exactly
            # sum_t mu_t * (MIP row slack). That imbalance is a real quantity --
            # the reserve rows' share of the same non-convexity gap the Owen/CoS
            # machinery measures -- so report it rather than asserting it away.
            settle = {u: float(sum(mu_p.get(t, 0.0) * up[u][t] + mu_m.get(t, 0.0) * dn[u][t]
                                   for t in T)) for u in players}
            out['reserve_settlement_by_player'] = settle
            tot = float(sum(settle.values()))
            out['settlement_total'] = tot
            out['settlement_imbalance'] = tot - out['reserve_revenue']
            # where the imbalance comes from: rows priced but slack in the MIP
            out['settlement_slack_by_hour'] = {
                int(t): {'mu_plus_slack': mu_p.get(t, 0.0) * (off_up[t] - r_sym),
                         'mu_minus_slack': mu_m.get(t, 0.0) * (off_dn[t] - r_sym)}
                for t in T
                if (mu_p.get(t, 0.0) * abs(off_up[t] - r_sym)
                    + mu_m.get(t, 0.0) * abs(off_dn[t] - r_sym)) > tol
            }

    # --------------------------------------------------------------------- peak
    if enable_peak:
        p = _scalar(results, 'chi_peak_E')
        # NOTE: the peak row is written on the GRID exchange vars i_E_gri/e_E_gri
        # (compact_utility._add_peak_penalty_constraints), so the netting metric
        # uses the same quantities. reserve.txt eq:dp_peak writes i_E_mkt/e_E_mkt.
        imp = _series(results, 'i_E_gri', players, T)
        exp = _series(results, 'e_E_gri', players, T)
        net = {u: {t: imp[u][t] - exp[u][t] for t in T} for u in players}
        community_net = {t: float(sum(net[u][t] for u in players)) for t in T}
        indiv_peaks = {u: float(max(net[u][t] for t in T)) for u in players}
        # a net exporter has a negative "peak"; the coincidence factor is
        # conventionally taken over the positive parts, both are reported
        sum_indiv = float(sum(indiv_peaks.values()))
        sum_indiv_pos = float(sum(max(0.0, v) for v in indiv_peaks.values()))

        out['peak_value'] = p
        out['peak_cost'] = pi_peak * p
        out['peak_netting'] = {
            'community_peak': p,
            'community_net_t': community_net,
            'community_peak_hour': int(max(T, key=lambda t: community_net[t])),
            'individual_peaks': indiv_peaks,
            'individual_peak_hours': {u: int(max(T, key=lambda t: net[u][t])) for u in players},
            'sum_individual_peaks': sum_indiv,
            'sum_individual_peaks_pos': sum_indiv_pos,
            'coincidence_factor': (p / sum_indiv_pos) if sum_indiv_pos > 1e-12 else None,
            'netting_saving_mw': sum_indiv_pos - p,
        }
        out['checks']['peak_ge_community_net_all_t'] = bool(
            all(p >= community_net[t] - tol for t in T))

        xi = {int(t): float(v) for t, v in (prices.get('peak') or {}).items()}
        if xi:
            out['xi_t'] = xi
            xi_sum = float(sum(xi.values()))
            out['checks']['dual_sum_xi'] = xi_sum
            out['checks']['peak_dual_balance_ok'] = bool(
                abs(xi_sum - pi_peak) <= 1e-6 * max(1.0, abs(pi_peak)))

    return out


# --------------------------------------------------------------------- flat view
def flatten_for_csv(metrics, prefix=''):
    """Scalar subset of reserve_peak_metrics for a per-instance CSV row.

    The 24-long dual/quantity series and the per-player dicts stay in the JSON;
    only headline numbers go to CSV.
    """
    row = {}
    if not metrics:
        return row
    if metrics.get('enabled', {}).get('reserve'):
        pool = metrics.get('pooling', {})
        mech = pool.get('mechanism', {})
        row.update({
            f'{prefix}r_sym': metrics.get('r_sym'),
            f'{prefix}reserve_revenue': metrics.get('reserve_revenue'),
            # (1) pooling gain vs the stand-alone prosumers
            f'{prefix}pool_standalone_sum': pool.get('standalone_sum'),
            f'{prefix}pool_gain_abs': pool.get('gain_abs'),
            f'{prefix}pool_gain_ratio': pool.get('gain_ratio'),
            # (2) mechanism split at the community dispatch
            f'{prefix}mech_no_pooling': mech.get('no_pooling_A'),
            f'{prefix}mech_time_only': mech.get('time_pooled_T1'),
            f'{prefix}mech_full': mech.get('full_pooled_B'),
            f'{prefix}mech_gain_time': mech.get('gain_time'),
            f'{prefix}mech_gain_direction': mech.get('gain_direction'),
            f'{prefix}mech_share_direction': mech.get('share_direction'),
            f'{prefix}n_binding_up': len(metrics.get('binding', {}).get('up_hours_primal', [])),
            f'{prefix}n_binding_dn': len(metrics.get('binding', {}).get('dn_hours_primal', [])),
            # CHP prices x MIP quantities minus the community revenue; nonzero
            # because the two come from different solutions (see above)
            f'{prefix}settlement_imbalance': metrics.get('settlement_imbalance'),
        })
    if metrics.get('enabled', {}).get('peak'):
        pk = metrics.get('peak_netting', {})
        row.update({
            f'{prefix}peak_value': metrics.get('peak_value'),
            f'{prefix}peak_cost': metrics.get('peak_cost'),
            f'{prefix}sum_individual_peaks': pk.get('sum_individual_peaks_pos'),
            f'{prefix}coincidence_factor': pk.get('coincidence_factor'),
            f'{prefix}peak_netting_saving': pk.get('netting_saving_mw'),
        })
    return row


def print_report(metrics, v_n=None, indent='  '):
    """Console summary of the sec.3.3 headline metrics + sec.3.4 check status."""
    if not metrics:
        return
    if metrics.get('enabled', {}).get('reserve'):
        pool = metrics.get('pooling', {})
        rev = metrics.get('reserve_revenue', 0.0)
        share = f" ({100*rev/abs(v_n):.1f}% of v(N))" if v_n else ""
        print(f"{indent}r_sym = {metrics.get('r_sym', 0.0):.6f} MW   "
              f"revenue = {rev:.2f} EUR{share}")
        if 'standalone_sum' in pool:
            ratio = pool.get('gain_ratio')
            ratio_txt = f"  x{ratio:.3f}" if ratio is not None else ""
            print(f"{indent}pooling gain: community {pool['community_r_sym']:.6f} "
                  f"- standalone {pool['standalone_sum']:.6f} "
                  f"= {pool.get('gain_abs', 0.0):+.6f} MW{ratio_txt}")
        mech = pool.get('mechanism', {})
        print(f"{indent}mechanism @community dispatch: alone {mech.get('no_pooling_A', 0.0):.6f} "
              f"-> +time {mech.get('gain_time', 0.0):+.6f} "
              f"-> +direction {mech.get('gain_direction', 0.0):+.6f} "
              f"= {mech.get('full_pooled_B', 0.0):.6f} MW headroom")
        b = metrics.get('binding', {})
        print(f"{indent}binding hours: up {b.get('up_hours_primal')} "
              f"dn {b.get('dn_hours_primal')}")
    if metrics.get('enabled', {}).get('peak'):
        pk = metrics.get('peak_netting', {})
        cf = pk.get('coincidence_factor')
        cf_txt = f"{cf:.4f}" if cf is not None else "n/a"
        print(f"{indent}peak = {metrics.get('peak_value', 0.0):.6f} MW  "
              f"cost = {metrics.get('peak_cost', 0.0):.2f} EUR  "
              f"sum_indiv = {pk.get('sum_individual_peaks_pos', 0.0):.6f} MW  "
              f"coincidence = {cf_txt}")
    bad = failed_checks(metrics)
    print(f"{indent}sec.3.4 checks: {'ALL PASS' if not bad else 'FAILED -> ' + ', '.join(bad)}")


def failed_checks(metrics):
    """Names of the sec.3.4 boolean checks that did not pass (empty == all good)."""
    bad = []
    for k, v in (metrics or {}).get('checks', {}).items():
        if isinstance(v, bool) and not v:
            bad.append(k)
    err = (metrics or {}).get('checks', {}).get('asset_aggregation_max_err')
    if err is not None and err > 1e-6:
        bad.append('asset_aggregation_max_err')
    return bad

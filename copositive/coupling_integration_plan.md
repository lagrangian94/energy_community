> **IMPLEMENTATION STATUS (2026-07-04)** — Phases 0–3 implemented & validated.
> Phase 4 (copositive) and Phase 5 (monthly) deliberately NOT started (out of scope
> for this pass). Phase 3 (CHP / column generation) was the focus of this pass:
> - `pricer.py`: reserve/peak dual extraction (regular + Farkas), threaded into
>   `solve_pricing`; new-column coefficient stamping for the reserve/peak rows;
>   full wiring through the **smoothing** path (`_price_smoothed`,
>   `_recalculate_reduced_cost_wrt_pi_RM`, `pi_bar_resup/resdn/peak`).
> - `solver.py`: fixed `_solve_pricing_highs` (it now accepts & applies the
>   reserve/peak duals — previously it would crash when called with them).
> - **Lagrangian bound: intentionally NOT changed (Phase 3.7 reconsidered).** Because
>   r_up/r_dn/p are first-class RMP variables (not priced), their reduced costs are
>   ≥0 at every LP optimum where the pricer fires, so the shared-var term
>   `min_{x0≥0}[c0·x0 − μ·A0·x0]` is exactly 0 and `L(μ)=Σ_j obj_val_j` stays valid.
>   Adding the primal shared-var cost would corrupt the bound. (See `_update_lagrangian_bound` docstring.)
> - **Validation** (smoothing=True, 6-player, T=24): off obj=−2350.2325;
>   on_zero obj=−2350.2325 (|off−on_zero|=9e-13 ✓ zero-price neutrality);
>   on_pos obj=−2347.9385, r_up=0.14, community peak shaved 1.01→0.88, converged. ✓
>   Reduced-cost / dual sign convention verified consistent (raw duals, RC=c−Σπ·a).
>
> ---

# Master Plan — Integrating Reserve (Up/Down) & Peak Coupling Constraints

**Spec:** `copositive/adding_cons.txt` (two homogeneous coupling constraints for the LPG:
frequency reserve up/down, and a community peak penalty).
**Goal:** add both couplings *consistently* across the whole codebase — first the
core files outside `copositive/` (full MIP model, convex-hull pricing, core
computation, parameters), then the copositive solvers (`cop_guo.py`, `sdp_relax.py`).

This plan is the output of a four-way structural analysis of the codebase. It is
the single source of truth for the implementation; every step lists the exact
file, method, and line where code is inserted, plus the sign convention and the
invariant that keeps existing results reproducible.

---

## 0. Guiding principles

1. **Single hub — `compact_utility.py`.** `core.py` builds every coalition value
   `v(S)` by *re-instantiating* `LocalEnergyMarket` with the coalition as its
   `players` argument (`core.py:507-513`); `SeparationProblem` *subclasses* it
   (`core.py:22`). The CHP subproblems wrap it with `dwr=True` (`solver.py:40-46`).
   **⇒ Constraints added to `compact_utility.py` propagate automatically to `v(S)`
   and to separation.** Every `Σ_{j∈S}` in the spec is realized by the existing
   `for u in self.players` / `quicksum(... for u in self.players)` idiom — no
   explicit coalition handling is needed.

2. **Cost-minimization convention.** The model never calls `setObjective`; SCIP
   minimizes the sum of per-variable `obj=` coefficients set at `addVar` time.
   The spec is written in max-profit form, so signs flip:
   - `+π^up r^up`  → create `r^up` with `obj = -pi_up`   (revenue lowers cost)
   - `+π^dn r^dn`  → create `r^dn` with `obj = -pi_dn`
   - `-δ^peak p`   → penalty; under minimize it is a **positive** cost `obj = +pi_E_peak`
     (this is exactly what the existing dead-code peak var already does).

3. **`dwr` guard = master-vs-subproblem split.** Community-level coupling rows and
   the shared community variables (`r^up, r^dn, p`) sit behind `if not self.dwr:`
   (like the three community-balance blocks at `compact_utility.py:939/1009/1067`).
   The **per-prosumer headroom** variables/rows are *private* (`A_j`) and are **not**
   `dwr`-guarded — they live in every subproblem's feasible set `X_j`.

4. **INERT BY DEFAULT — the reproducibility invariant.** Reserve/peak are gated by
   explicit enable flags (`enable_reserve`, `enable_peak`), both **default False**.
   When off, *no* new variables or rows are created, so:
   - existing MIP/LP/CHP/core results are byte-identical, and
   - the copositive dimension `nc` (which counts model variables) is unchanged.
   This is non-negotiable: the copositive pipeline's `nc`/constraint counts must not
   drift unless reserve/peak is deliberately switched on.

5. **Prices default to zero.** `π^up, π^dn` are new (absent from all scenario data,
   per spec §6.2); `δ^peak = pi_E_peak` already exists but is 0 unless
   `peak_penalty_ratio > 0`. Even with the flags on, zero prices leave the optimum
   unchanged (reserve revenue = 0 ⇒ `r=0` optimal; peak price = 0 ⇒ `p` free),
   giving a clean two-stage validation (structure first, economics second).

---

## 1. Symbol → code cross-reference (authoritative)

| Spec symbol | Code object | Location |
|---|---|---|
| `i/e^{k,com}` (C0 balance) | `i_E_com,e_E_com / i_G_com,e_G_com / i_H_com,e_H_com` keyed `(u,t)` | `compact_utility.py:565-671` |
| `i/e^{E,mkt}` (C-Peak) | `i_E_gri, e_E_gri` | `compact_utility.py:565,631` |
| `b^{E,ch}, b^{E,dis}, s^E` | `b_ch_E[u,t], b_dis_E[u,t], s_E[u,t]` | `compact_utility.py:674-685` |
| `b̄^{E,ch/dis} / s̲^E / s̄^E` | `storage_power_E_{u}` / 0 / `storage_capacity_E_{u}` | params |
| `ν^{E,ch}, ν^{E,dis}` | `nu_ch_E, nu_dis_E` (global scalars) | params |
| `d̃^{E→G,sc}, z^{G,on}, z^{G,sb}, c^{G,lb}` | `fl_d[u,'elec',t], z_on_G[u,t], z_sb_G[u,t], c_sb_G*els_cap` | `compact_utility.py:598-622,1144-1153` |
| `d^{E→H,sc}, z^{H,on}, c̲^H/c̄^H, d̄^{E→H,sc}` | `p[u,'hp',t]` (heat) / `fl_d[u,'elec',t]` (elec), `z_on_H[u,t]`, `c_min_H/c_max_H`, `hp_cap_{u}` | `compact_utility.py:583-596,1263-1270` |
| `r^up, r^dn` (community, new) | to create; `obj=-pi_up, -pi_dn` | — |
| `p` (peak) | `chi_peak_E` (exists, `obj=pi_E_peak`) | `compact_utility.py:747` |
| `r_jt^{+/-}` (private, new) | `r_plus[u,t], r_minus[u,t]` (+ per-asset splits) | to create |
| `π^up, π^dn / δ^peak` | `pi_up, pi_dn` (new) / `pi_E_peak` (exists) | params |

Global-vs-per-player caveat: `nu_ch_E/nu_dis_E` and `c_min_H/c_max_H` are **global
scalars**; capacities (`storage_power_E_{u}`, `hp_cap_{u}`, `els_cap_{u}`) are
per-player. Reserve headroom rows use whichever already exists; add `_{u}` variants
only if per-prosumer efficiencies become necessary.

---

## 2. Constraint math to implement (from spec §2–§4)

**Per-prosumer headroom (private, in `X_j`), for each `u`, `t`:**

Storage:
```
(b_dis_E - b_ch_E) + r^{+,sto}  ≤  storage_power_E_u
(b_ch_E - b_dis_E) + r^{-,sto}  ≤  storage_power_E_u
r^{+,sto}  ≤  nu_dis_E * (s_E - s_min)          # s_min = 0
r^{-,sto}  ≤  (1/nu_ch_E) * (storage_capacity_E_u - s_E)
```
Electrolyzer (modify existing min/max load rows at `1144-1153`):
```
fl_d[u,'elec',t] - r^{+,els}  ≥  c_min_G*els_cap*z_on_G + c_sb_G*els_cap*z_sb_G
fl_d[u,'elec',t] + r^{-,els}  ≤  c_max_G*els_cap*z_on_G + c_sb_G*els_cap*z_sb_G
```
Heat pump (on heat output `p[u,'hp',t]`, template `1263-1270`):
```
p[u,'hp',t] - r^{+,hp}  ≥  c_min_H*hp_cap*z_on_H
p[u,'hp',t] + r^{-,hp}  ≤  c_max_H*hp_cap*z_on_H
```
Aggregation:
```
r_plus[u,t]  = r^{+,sto}+r^{+,els}+r^{+,hp}    (only the assets u owns)
r_minus[u,t] = r^{-,sto}+r^{-,els}+r^{-,hp}
```

**Community coupling rows (behind `if not self.dwr:`), for each `t`:**
```
(C-Res↑)  r^up  ≤  Σ_{u∈players} r_plus[u,t]
(C-Res↓)  r^dn  ≤  Σ_{u∈players} r_minus[u,t]
(C-Peak)  Σ_{u∈players} (i_E_gri[u,t] - e_E_gri[u,t])  ≤  p        # p = chi_peak_E
```

**Objective (via `obj=` at var creation):** `r^up:obj=-pi_up`, `r^dn:obj=-pi_dn`,
`chi_peak_E:obj=+pi_E_peak` (already set).

---

## 3. Implementation phases

Ordered so that the hub is done first (auto-propagating to MIP + core), then CHP,
then the copositive solvers. Each phase ends with a smoke test that must keep the
default-off invariant (§0.4).

### Phase 0 — parameters (`data_generator.py`)
- **δ^peak:** reuse existing `pi_E_peak` (`data_generator.py:234`, gated by
  `peak_penalty_ratio`, default 0.0). No new key.
- **π^up, π^dn:** add near line 234 (where `elec_prices` is in scope), mirroring the
  `pi_E_gri` construction, driven by new toggles `reserve_up_ratio`,
  `reserve_dn_ratio` (default 0.0) carried in the `sensitivity_analysis` dict
  (define them in the `else`-defaults at `:88-117` too). Store as scalars
  `pi_up`, `pi_dn` (and optional per-`t` series `pi_up_{t}` only if time-varying
  reserve prices are wanted). Derive from `mean(elec_prices["import"]) * ratio` so
  the values are self-justifying (spec §6.2).
- **enable flags:** add `enable_reserve` (default `reserve_up_ratio>0 or
  reserve_dn_ratio>0`) and `enable_peak` (default `peak_penalty_ratio>0`) to the
  returned params dict, so the model can gate construction with a single lookup.
- `components.py`: **no change** (it is the electrolyzer physics library, not a
  param container).
- **Smoke test:** `setup_lem_parameters` runs; `pi_up/pi_dn == 0` and
  `enable_reserve/enable_peak == False` for the default 6-player config.

### Phase 1 — full MIP hub (`compact_utility.py`)  ★ highest leverage
Read `enable_reserve = self.params.get('enable_reserve', False)` and
`enable_peak = self.params.get('enable_peak', False)` once in `__init__`.

1. **Community vars** (guard: `if not self.dwr and enable_*`):
   - `chi_peak_E` already at `747` (`obj=pi_E_peak`). Activate its method.
   - new `r_up, r_dn` = `addVar(lb=0, obj=-pi_up)` / `obj=-pi_dn`, in a new
     `_add_reserve_constraints`.
2. **Private headroom vars** `r_plus/r_minus[u,t]` (+ per-asset splits), created at
   the end of `_create_variables` (~`715`) inside `for u/for t`, gated on device
   membership and `enable_reserve`. **Not** `dwr`-guarded (private). Only for
   `model_type in ('mip','mip_fix_binaries')` where the commitment binaries exist.
3. **Private headroom rows:** new `_add_reserve_headroom_cons`, called from
   `_create_constraints` **after** the device dispatch (after `738`). Storage rows
   near the SoC block (`909-937`); electrolyzer by extending `1144-1153`; HP on
   `p[u,'hp',t]` (`1263-1270`).
4. **Community coupling rows:** C-Res↑/↓ in `_add_reserve_constraints`; C-Peak fix
   the existing rows at `750-752` — add the missing `- e_E_gri` term and sum over
   `self.players`, wrap in `if not self.dwr:`.
5. **Activation:** replace the commented call at `743` with live, flag-gated calls
   to `_add_peak_penalty_constraints` and `_add_reserve_constraints`.
6. **`_store_model_data`** (`822-867`): register `r_up, r_dn, chi_peak_E` and the
   new cons dicts (`reserve_cons`, `reserve_headroom_cons`, `peak_penalty_cons`).
7. **Smoke tests:**
   - default (flags off): run a small MIP; objective and var/constraint counts
     identical to a pre-change baseline (capture the baseline first).
   - flags on, prices 0: model builds & solves; `r=0`, objective unchanged.
   - flags on, prices > 0: reserve revenue appears, objective improves; `p` equals
     the realized net-import peak.

### Phase 2 — core computation (`core.py`)
- `v(S)` and the row-generation cut need **no structural change** (auto-propagated).
- **`SeparationProblem`:** when `enable_reserve`, the new *private* vars
  `r_plus/r_minus[u,t]` need Big-M gating so a non-selected player offers zero
  reserve — add rows mirroring `_add_bigm_constraints` (`core.py:203-216`). The
  community vars `r_up/r_dn/p` are shared singletons ⇒ no gating.
- **Smoke test:** `analysis_mip.py` on a tiny instance; with flags off the core
  allocation matches the pre-change result; with flags on it still converges.

### Phase 3 — convex-hull pricing (`chp.py`, `solver.py`, `pricer.py`)  ★ most complex
The private/shared split maps exactly onto subproblem/master:
1. **Private headroom** already handled by Phase 1 (subproblems are
   `LocalEnergyMarket(dwr=True)`); once `r_plus/r_minus` are registered in
   `model.data["vars"]`, they ride inside every column's `solution` dict for free.
2. **Shared master vars** `r_up, r_dn, p` = **first-class RMP variables** (NOT priced
   columns), added in `MasterProblem.__init__`/`_create_master_constraints`
   (`solver.py:432/434`): `obj=-pi_up, -pi_dn, +delta_peak`.
3. **Three coupling rows** in `_create_master_constraints` (after `502`),
   `modifiable=True` like the balance rows: member coeff `-r_plus/-r_minus` and
   `(i_gri-e_gri)`; shared var coeff `+1/+1/-1`. Register dicts in
   `data['cons']` (`427-432`); seed member coeffs from initial columns
   (`453-467`).
4. **Column coefficient stamping:** `LEMPricer._add_column` (`pricer.py:196-228`) —
   add `addConsCoeff` blocks for the three new rows.
5. **Dual extraction:** `LEMPricer.price` (`pricer.py:61-81`) — read
   `dual_resup/resdn/peak[t]` via `getDualsolLinear`/`getDualfarkasLinear`.
6. **Subproblem objective feedback:** `PlayerSubproblem.solve_pricing`
   (`solver.py:363-382`) — add `+= dual_resup[t]*r_plus[u,t]`,
   `+= dual_resdn[t]*r_minus[u,t]` (member coeff `-r` ⇒ reduced-cost `+dual*r`),
   and `-= dual_peak[t]*(i_E_gri-e_E_gri)`. **Mirror in three places:** the HiGHS
   fast path (`solver.py:239-266`, LP-only), the smoothed reduced-cost recompute
   (`pricer.py:376-388`), and the smoothed pi dicts (`pricer.py:35-37,255-258`).
7. **Lagrangian bound caveat:** `_update_lagrangian_bound` (`pricer.py:229-240`)
   assumes LB = Σ subproblem objectives (all linking RHS = 0). The shared-var
   master cost (`-pi_up*r_up` …) is currently outside that sum — add it so the LB
   stays valid when reserve/peak is on.
8. **`calculate_column_cost`** (`solver.py:600-661`): include any private reserve
   cost (there is none if headroom vars carry no `obj=`; keep them cost-free so the
   revenue lives only on the shared master vars — matches spec §4 `c_0`).
9. **Smoke test:** CHP on a tiny instance; flags off ⇒ identical convex-hull
   prices; flags on, prices 0 ⇒ converges, `r_up=r_dn=0`; prices > 0 ⇒ reserve
   duals nonzero and the CHP allocation stays in/near the core.

### Phase 4 — copositive solvers (`cop_guo.py`, `sdp_relax.py`)
These build `CPOPTBuilder` from the Gurobi/SCIP model, so once `compact_utility.py`
emits the reserve/peak vars and rows (flags on), the CPP lifting picks them up
mechanically. Checks:
- The new community vars `r_up,r_dn,chi_peak_E` and private `r_plus/r_minus` become
  columns of the lifted matrix; verify `nc`, the Class A/B partition, and the
  homogenization still hold (the coupling rows are homogeneous, RHS 0, so they are
  Class A first-order equalities/inequalities — consistent with the existing
  community-balance handling).
- `sdp_relax.py` needs no logic change, but the memory preflight guard (already
  added) will now reflect the larger `nc` — good, it will warn/abort if reserve/peak
  pushes the instance past the memory budget.
- **Smoke test:** `sdp_relax.py --small` with reserve/peak enabled on a tiny
  instance; confirm the DNN builds and the Burer-exact bound identity still holds
  where expected.

### Phase 5 — monthly variant (`compact_utility_monthly.py`) — OPTIONAL / follow-up
Near-verbatim duplicate (method offsets ~60-75 lines earlier), but it has **no**
peak scaffold at all. If the monthly model is in scope, mirror **all** of Phase 1
there from scratch. Not imported from `compact_utility.py`, so nothing propagates.
Deferred until the main path is validated; flagged here so it is not forgotten.

---

## 4. Risk register

| Risk | Mitigation |
|---|---|
| Breaking existing results / copositive `nc` | Enable flags default off ⇒ zero new vars/rows (§0.4); capture a baseline and diff. |
| Sign errors (max-profit spec vs min-cost code) | Single sign table (§0.2); assert reserve revenue lowers cost, peak penalty raises it. |
| UC binaries couple els/HP headroom (spec §6.3) | Expected — LPG/core exactness holds only in the LP relaxation; document, don't fight. |
| CHP dual wiring incomplete across 3 paths | Explicit checklist (Phase 3.6): SCIP + HiGHS + smoothing all updated together. |
| Lagrangian bound omits shared-var cost | Phase 3.7 fixes `_update_lagrangian_bound`. |
| HiGHS fast path is LP-only | New columns/duals handled in the HiGHS branch too; MIP subproblems use SCIP path. |
| Monthly model drift | Phase 5 mirrors it, or explicitly declare it out of scope. |

---

## 5. Validation strategy (end-to-end)

1. **Baseline capture (before any edit):** record objective, var/constraint counts,
   core allocation, and CHP prices for a small instance.
2. **Default-off diff:** after every phase, re-run and assert bit-identical to the
   baseline.
3. **Enabled, zero-price:** structure present, economics neutral ⇒ optimum
   unchanged, `r=0`. Validates construction independent of prices.
4. **Enabled, positive-price:** reserve revenue and peak penalty active ⇒ objective
   moves in the expected direction; verify core stability still holds (grand-
   coalition dual feasible for every `S`, per spec §5).
5. **Copositive tie-out:** the `--small` DNN identity `-val(DNN) == v^MIP` must still
   hold with reserve/peak on in the LP-exact regime.

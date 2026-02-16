# Dual Smoothing Implementation Guide for Column Generation

## Overview

이 문서는 `pricer.py`, `solver.py`, `chp.py`에 **Wentges (1997) dual variable smoothing**을 구현하기 위한 가이드이다.
핵심 레퍼런스: Pessoa et al. (2010) "Exact algorithm over an arc-time-indexed formulation for parallel machine scheduling problems", Section 3.2 (Algorithm 1).

## 최우선 원칙: Backward Compatibility

**모든 변경은 `smoothing: bool` 파라미터로 gating한다.**
- `smoothing=False`이면 기존 코드와 100% 동일하게 동작해야 한다.
- smoothing 관련 코드는 반드시 `if self.smoothing:` 블록 안에 있어야 한다.
- 기존 코드의 어떤 라인도 삭제하지 않는다. 분기로만 처리한다.

---

## 1. 배경: Dual Smoothing이란?

Standard CG에서 매 iteration마다 RMP의 optimal dual π^RM을 그대로 pricing에 넘기면, dual이 심하게 oscillation하면서 수렴이 느려진다. Smoothing은 pricing에 쓰는 dual을 stability center π̄와의 convex combination으로 대체한다:

```
π^ST = α · π^RM + (1 − α) · π̄
```

여기서 π̄는 지금까지 발견된 최대 Lagrangean bound L(·)를 제공하는 dual vector이다.

### Mispricing

π^ST로 pricing한 결과가 π^RM 기준으로는 유용하지 않을 수 있다. 이를 misprice라 한다.
- π^ST로 찾은 column의 reduced cost가 π^RM 기준으로 non-negative → column 추가 불가
- π^ST 기준으로 아예 negative RC column이 없음 → π^RM 기준으로는 있을 수 있음

**따라서 π^RM 기준 검증이 반드시 필요하다.**

---

## 2. 파일별 변경사항

### 2.1 `chp.py` — ColumnGenerationSolver

#### 변경 1: `__init__`에 smoothing 파라미터 추가

```python
def __init__(self, players, time_periods, parameters, model_type, init_sol=None, smoothing=False):
    # ... 기존 코드 전부 유지 ...
    self.smoothing = smoothing
```

#### 변경 2: `solve()`에서 LEMPricer 생성 시 smoothing 전달

```python
pricer = LEMPricer(
    subproblems=self.subproblems,
    time_periods=self.time_periods,
    players=self.players,
    smoothing=self.smoothing     # 추가
)
```

#### 변경 3: 외부 호출부 (sensitivity_analysis.py, analysis_mip.py 등)

기존 호출:
```python
cg_solver = ColumnGenerationSolver(players, time_periods, parameters, model_type='mip', init_sol=init_sol)
```

변경 없음 (smoothing 기본값이 False이므로). 나중에 켜고 싶으면:
```python
cg_solver = ColumnGenerationSolver(..., smoothing=True)
```

---

### 2.2 `pricer.py` — LEMPricer

이 파일이 가장 핵심적인 변경 대상이다.

#### 변경 1: `__init__`에 smoothing 상태 추가

```python
def __init__(self, subproblems, time_periods, players, smoothing=False, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.subproblems = subproblems
    self.time_periods = time_periods
    self.players = players
    self.iteration = 0
    self.farkas_iteration = 0
    self.lb = -np.inf

    # === Smoothing 관련 ===
    self.smoothing = smoothing
    if self.smoothing:
        # Stability center π̄ (전체 dual vector)
        self.pi_bar_elec = {t: 0.0 for t in time_periods}
        self.pi_bar_heat = {t: 0.0 for t in time_periods}
        self.pi_bar_hydro = {t: 0.0 for t in time_periods}
        self.pi_bar_conv = {player: 0.0 for player in players}
        # Best Lagrangean bound found so far
        self.L_bar = -np.inf
        # Incumbent (upper bound) — will be set from outside or from init_sol
        self.Z_INC = np.inf
```

#### 변경 2: `price()` 메서드 — 핵심 로직 변경

**전체 구조 (pseudocode):**

```
price(farkas):
    if farkas:
        # Farkas pricing은 smoothing 안 함 (기존 그대로)
        기존 코드 그대로 실행
        return

    # Step 1: π^RM 추출 (기존 코드)
    pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv = get_duals_from_master()

    if not self.smoothing:
        # 기존 코드 그대로: π^RM으로 pricing
        dual_elec, dual_heat, dual_hydro, dual_conv = pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv
        [기존 pricing 로직 그대로]
        return

    # === 이하 smoothing=True 전용 ===

    # Step 2: α 계산
    Z_RM = self.model.getLPObjVal()
    alpha = self._compute_alpha(Z_RM)

    # Step 3: π^ST 계산 (전체 vector에 대해 한 번에)
    pi_ST_elec = {t: alpha * pi_RM_elec[t] + (1 - alpha) * self.pi_bar_elec[t] for t in self.time_periods}
    pi_ST_heat = {t: alpha * pi_RM_heat[t] + (1 - alpha) * self.pi_bar_heat[t] for t in self.time_periods}
    pi_ST_hydro = {t: alpha * pi_RM_hydro[t] + (1 - alpha) * self.pi_bar_hydro[t] for t in self.time_periods}
    pi_ST_conv = {p: alpha * pi_RM_conv[p] + (1 - alpha) * self.pi_bar_conv[p] for p in self.players}

    # Step 4: π^ST로 모든 player pricing
    columns_added = 0
    st_solutions = {}
    st_obj_vals = {}
    for player in self.players:
        rc_st, sol, obj_val = self.subproblems[player].solve_pricing(
            pi_ST_elec, pi_ST_heat, pi_ST_hydro, pi_ST_conv[player])
        st_solutions[player] = sol
        st_obj_vals[player] = obj_val

    # Step 5: 각 column에 대해 π^RM 기준 reduced cost 재계산 (subproblem re-solve 없이)
    #
    # 주의: 이것은 π^ST로 찾은 solution이 π^RM에도 유용한지만 체크하는 것.
    # π^RM 기준 '진짜 최적' column을 찾으려면 Step 7의 재pricing이 필요하다.
    #
    # - Case A (rc_st < 0, rc_rm < 0): 여기서 잡힘 → column 추가, Step 7 스킵
    # - Case B (rc_st < 0, rc_rm ≥ 0): 여기서 걸러짐 → Step 7로 이동
    # - Case C (rc_st ≥ 0):            sol은 반환되나 대부분 rc_rm도 ≥ 0 → Step 7로 이동
    #
    # 즉 이 step의 핵심 역할은 Case A를 subproblem re-solve 없이 빠르게 처리하는 것.
    for player in self.players:
        if st_solutions[player] is not None:
            rc_rm = self._recalculate_reduced_cost_wrt_pi_RM(
                player, st_solutions[player], pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv[player])
            if rc_rm < -1e-8:
                self._add_column(player, st_solutions[player])
                columns_added += 1

    # Step 6: L(π^ST) 계산 및 π̄ 업데이트
    L_pi_ST = self._compute_lagrangean_bound(st_obj_vals, pi_ST_conv)
    if L_pi_ST > self.L_bar:
        self.L_bar = L_pi_ST
        self.pi_bar_elec = dict(pi_ST_elec)
        self.pi_bar_heat = dict(pi_ST_heat)
        self.pi_bar_hydro = dict(pi_ST_hydro)
        self.pi_bar_conv = dict(pi_ST_conv)

    # Step 7: Misprice fallback — π^RM으로 재pricing (Case B, C 처리)
    #
    # columns_added == 0이면:
    # - Case B: π^ST 최적 sol이 π^RM 기준으로 쓸모없었음 → π^RM으로 진짜 최적 column 탐색
    # - Case C: π^ST 기준으로 아예 improving column이 없었음 → π^RM 기준으로는 있을 수 있음
    #
    # 이 step에서 subproblem을 다시 풀어야 π^RM 기준 최적 column을 찾을 수 있다.
    # (Step 5의 재계산만으로는 부족 — 그건 π^ST 최적 solution의 RC만 체크한 것임)
    if columns_added == 0:
        for player in self.players:
            rc_rm, sol, obj_val = self.subproblems[player].solve_pricing(
                pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv[player])
            if rc_rm < -1e-8:
                self._add_column(player, sol)
                columns_added += 1

    # Step 8: 여전히 0이면 진짜 수렴
    if columns_added == 0:
        return {"result": SCIP_RESULT.SUCCESS}

    return {"result": SCIP_RESULT.SUCCESS}
```

#### 변경 3: helper 메서드 추가

**`_compute_alpha(self, Z_RM)`**:

```python
def _compute_alpha(self, Z_RM):
    """
    Pessoa et al. (2010) Section 3.2의 adaptive α 계산.
    Z_INC: incumbent (best known integer solution value)
    L_bar: best known Lagrangean lower bound
    Z_RM: current RMP objective value
    """
    base_alpha = 0.1
    if self.L_bar == -np.inf:
        return base_alpha

    gap = Z_RM - self.L_bar
    if gap < 1e-6:
        # Gap이 충분히 작으면 standard CG로 전환
        return 1.0

    if self.Z_INC < np.inf and Z_RM > self.Z_INC:
        inc_gap = self.Z_INC - self.L_bar
        if inc_gap > 1e-6:
            return base_alpha * inc_gap / gap
        else:
            return base_alpha
    else:
        return base_alpha
```

**`_recalculate_reduced_cost_wrt_pi_RM(self, player, solution, pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv)`**:

이 메서드는 subproblem을 다시 풀지 않고, 이미 찾은 solution의 변수값으로 π^RM 기준 reduced cost를 직접 계산한다.

```python
def _recalculate_reduced_cost_wrt_pi_RM(self, player, solution, pi_RM_elec, pi_RM_heat, pi_RM_hydro, pi_RM_conv):
    """
    RC = original_cost - Σ_t π^RM_elec[t] * (i_E_com - e_E_com)
                       - Σ_t π^RM_heat[t] * (i_H_com - e_H_com)
                       - Σ_t π^RM_hydro[t] * (i_G_com - e_G_com)
                       - π^RM_conv
    """
    # Original cost (same as calculate_column_cost)
    cost = calculate_column_cost(player, solution, self.subproblems[player].parameters, self.time_periods)

    # Dual 항 차감
    dual_contribution = 0.0
    for t in self.time_periods:
        # Electricity
        i_E = solution.get('i_E_com', {}).get((player, t), 0.0)
        e_E = solution.get('e_E_com', {}).get((player, t), 0.0)
        dual_contribution += pi_RM_elec[t] * (i_E - e_E)
        # Heat
        i_H = solution.get('i_H_com', {}).get((player, t), 0.0)
        e_H = solution.get('e_H_com', {}).get((player, t), 0.0)
        dual_contribution += pi_RM_heat[t] * (i_H - e_H)
        # Hydrogen
        i_G = solution.get('i_G_com', {}).get((player, t), 0.0)
        e_G = solution.get('e_G_com', {}).get((player, t), 0.0)
        dual_contribution += pi_RM_hydro[t] * (i_G - e_G)

    # Reduced cost = cost - dual_contribution - convexity_dual
    # 주의: 여기서 cost는 원래 objective, community balance의 dual contribution,
    #       그리고 convexity dual을 빼면 됨.
    #       (solver.py의 solve_pricing에서 reduced_cost = obj_val - dual_convexity와 동일한 구조)
    reduced_cost = cost - dual_contribution - pi_RM_conv

    return reduced_cost
```

**중요**: 이 계산이 `solver.py`의 `solve_pricing()`에서 반환하는 reduced cost와 일치하는지 반드시 검증해야 한다. Smoothing=False 상태에서 `solve_pricing()`의 RC와 `_recalculate_reduced_cost_wrt_pi_RM()`의 결과가 동일해야 한다. **구현 후 이 검증을 먼저 수행할 것.**

**`_compute_lagrangean_bound(self, obj_vals, pi_conv)`**:

```python
def _compute_lagrangean_bound(self, obj_vals, pi_conv):
    """
    L(π) = Σ_j (Z_sub_j(π) + π^conv_j)
    여기서 Z_sub_j(π)는 subproblem의 optimal objective (dual 반영 후),
    reduced_cost = Z_sub_j - π^conv_j 이므로 Z_sub_j = reduced_cost + π^conv_j.
    따라서 L(π) = Σ_j Z_sub_j = Σ_j obj_val_j.

    주의: obj_val은 solve_pricing()에서 반환하는 세 번째 값 (self.model.getObjVal()),
    이것은 dual_convexity를 빼기 전의 값임. Community balance RHS가 전부 0이므로
    그 dual 항은 소멸하고, L(π) = Σ_j obj_val_j가 맞음.

    단, π^ST를 사용했을 경우 obj_val도 π^ST 기준이어야 함.
    """
    L = 0.0
    for player in self.players:
        if obj_vals[player] is not None:
            L += obj_vals[player]
        else:
            return -np.inf  # subproblem 실패 시
    return L
```

#### 주의: Farkas pricing

Farkas pricing (`pricerfarkas`)에는 smoothing을 적용하지 않는다. Farkas pricing은 feasibility 복원 목적이므로 original Farkas multiplier를 그대로 사용해야 한다. **기존 코드를 그대로 유지한다.**

#### Z_INC 설정

`chp.py`의 `solve()` 메서드에서 pricer 생성 후, incumbent value를 설정해야 한다:

```python
if self.smoothing:
    # IP solution의 objective를 incumbent로 사용
    if self.init_sol is not None:
        # init_sol로부터 Z_INC 계산 (IP objective value)
        # 이미 master._add_initial_columns()에서 cost 계산했으므로,
        # 여기서 전체 cost를 구해 Z_INC으로 설정
        Z_INC = sum(
            calculate_column_cost(p, {k:{key:val for key,val in self.init_sol[k].items() if key[0]==p} for k in self.init_sol},
                                  self.subproblems[p].parameters, self.time_periods)
            for p in self.players
        )
        pricer.Z_INC = Z_INC
```

---

### 2.3 `solver.py` — PlayerSubproblem, MasterProblem

**이 파일은 변경하지 않는다.**

이유:
- `PlayerSubproblem.solve_pricing()`은 어떤 dual이든 받아서 pricing하는 generic 함수임. smoothing 여부와 무관하게 동일하게 동작.
- `MasterProblem`도 column 추가/제약조건 관리만 하므로 smoothing과 무관.
- `calculate_column_cost()`도 변경 없음.

---

## 3. 코드 변경 체크리스트

### 3.1 `pricer.py`
- [ ] `__init__`에 `smoothing` 파라미터 추가 (default=False)
- [ ] `smoothing=True`일 때 stability center 초기화
- [ ] `price()`에서 `if not self.smoothing:` 분기로 기존 코드 보호
- [ ] `_compute_alpha()` 메서드 추가
- [ ] `_recalculate_reduced_cost_wrt_pi_RM()` 메서드 추가
- [ ] `_compute_lagrangean_bound()` 메서드 추가
- [ ] `price()` smoothing 분기 구현 (Step 1-8)
- [ ] `pricerfarkas()`는 변경하지 않음

### 3.2 `chp.py`
- [ ] `ColumnGenerationSolver.__init__`에 `smoothing` 파라미터 추가 (default=False)
- [ ] `solve()`에서 `LEMPricer` 생성 시 smoothing 전달
- [ ] `smoothing=True`일 때 `pricer.Z_INC` 설정

### 3.3 `solver.py`
- [ ] 변경 없음

### 3.4 외부 호출부 (analysis_mip.py, sensitivity_analysis.py 등)
- [ ] 변경 없음 (smoothing=False 기본값)

---

## 4. 검증 절차

### 4.1 Backward Compatibility 검증 (최우선)
1. `smoothing=False`로 기존 테스트 케이스 실행
2. 기존 결과와 objective, convex hull prices, solve time이 동일한지 확인
3. **이 검증이 통과하지 않으면 smoothing=True 테스트로 넘어가지 않는다**

### 4.2 RC 일관성 검증
1. `smoothing=False`에서 `solve_pricing()`이 반환하는 RC와 `_recalculate_reduced_cost_wrt_pi_RM()`이 계산하는 RC가 동일한지 확인 (tolerance 1e-6 이내)
2. 이를 위해 `price()` 내에서 임시 검증 코드 추가:
```python
# 검증 코드 (나중에 제거)
if not self.smoothing:
    for player in self.players:
        rc_direct = reduced_cost_from_solve  # solve_pricing 반환값
        rc_recalc = self._recalculate_reduced_cost_wrt_pi_RM(...)
        assert abs(rc_direct - rc_recalc) < 1e-6, f"RC mismatch for {player}: {rc_direct} vs {rc_recalc}"
```

### 4.3 Smoothing 기능 검증
1. `smoothing=True`로 실행
2. 수렴 여부 확인 (같은 optimal objective에 도달하는지)
3. Iteration 수 및 시간 비교 (smoothing=False 대비)
4. Misprice 발생 횟수 로깅으로 확인

---

## 5. Smoothing 관련 로깅

디버깅과 튜닝을 위해 다음을 출력한다 (smoothing=True일 때만):

```
Iter  3 | LP Obj:  -1234.56 | L_bar:  -1500.00 | α: 0.10 | Misprice: N | Cols: 3
Iter  4 | LP Obj:  -1234.56 | L_bar:  -1400.00 | α: 0.10 | Misprice: Y (fallback) | Cols: 2
Iter  5 | LP Obj:  -1300.00 | L_bar:  -1400.00 | α: 0.10 | Misprice: N | Cols: 1
...
Iter 20 | LP Obj:  -1450.00 | L_bar:  -1449.95 | α: 1.00 | STANDARD CG | Cols: 0 → CONVERGED
```

---

## 6. 주의사항 요약

1. **smoothing=False면 기존 코드와 100% 동일 동작 보장** — 이것이 최우선 요구사항이다.
2. **Smoothing은 전체 dual vector에 대해 한 번에 수행** — player별로 개별 smoothing 절대 안 됨.
3. **Farkas pricing에는 smoothing 적용 안 함.**
4. **Community balance dual (공유)과 convexity dual (player별)을 같은 α로 동시에 smoothing.**
5. **Misprice 시 π^RM fallback으로 재pricing** — 한 `pricerredcost()` 호출 안에서 처리.
6. **`_recalculate_reduced_cost_wrt_pi_RM`과 `solve_pricing`의 RC 일치 검증 필수.**
7. **solver.py는 변경하지 않는다.**

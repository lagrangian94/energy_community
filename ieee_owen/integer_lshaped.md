# 확률적 급전의 분해법: r_sym 이산화 integer L-shaped와 시나리오 branch-and-price, 그리고 왜 접는가

Oct 1, 2026 · @Seokwoo Kim

`ieee_owen/integer_lshaped.py`로 확률적 grand-coalition 급전(stochastic_extension의 extensive form, EF)을 integer L-shaped로 풀어 본 기록이다. 결론은 네 가지다.

1. **n=15, |Ω|=20에서는 분해법이 이산 EF를 이겼다.** 최선 구성(부분 분해, Kelley 수렴, Farkas cut, 16-worker 병렬)이 90.7초, 같은 이산 문제의 EF가 145초다.
2. **그러나 비교 기준이 잘못돼 있다.** L-L cut을 쓰려면 1단계가 순수 이진이어야 해서 r_sym을 이산화했고, 이 이산화가 EF를 10배 넘게 어렵게 만들었다(n=60에서 연속 EF 349초 대 이산 EF 1시간 미종료). 분해법이 이긴 상대는 이 약해진 EF다. 원래 문제(연속 r_sym)의 EF는 n=60에서도 349초에 1e-4로 풀린다.
3. **이산화는 원래 문제도 바꾼다.** n=60에서 6비트 격자의 손실이 약 0.11%로, gap 허용치 1e-4보다 크다. 이산 문제를 정확히 풀어도 원래 문제의 해로는 정확하지 않다.
4. **방법 쪽에서 배운 것은 남는다.** root 하한이 약했던 것은 recourse 정수성 때문이 아니라 구현 버그(분수 x에서 feasibility cut 누락, root 루프의 조기 종료) 때문이었다. 고치면 Kelley가 EF LP 값으로 수렴한다. 부분 분해와 병렬 subproblem은 효과가 크고, cut-and-project는 하한을 끝까지 닫지만 비싸며, Lagrangian cut과 in-out stabilization은 효과가 없었다.

이산화가 필요 없는 정확한 방법으로 비예측성 제약을 쌍대화한 시나리오 DW(Carøe–Schultz)의 branch-and-price도 구현했다(`ieee_owen/scenario_bnp.py`, 7절). 결과는 둘로 갈린다.

5. **n=6, |Ω|=3에서는 root에서 끝난다.** stochastic CG와 같은 안정화(Wentges smoothing + 3구간 penalty)를 쓰고 EF LP의 비예측성 dual에서 출발하면, 54회 9.8초 만에 z_D = UB = 연속 EF 최적값(gap 2.6e-9)이 된다. 분기가 필요 없다.
6. **n=15, |Ω|=20에서는 쓸 수 없다.** 111회 7,204초 동안 하한이 출발점 L(π_LP) = −6698.32(EF LP 값과 거의 같음)에서 한 번도 오르지 않았다. 반복도 30~60초씩 걸린다. 연속 EF는 약 32초에 풀린다.

결론: |Ω|=20, n ≤ 60에서는 연속 EF를 유지한다. 분해법은 시나리오가 수백 개로 늘어 EF가 막힐 때 다시 볼 대상이다.

모든 수치는 Gurobi, 16코어, 기본 데이터(day 9), FCR-N 56 + penalty reserve, seed 0 기준이다.

## 1. 정식화

**1단계**는 전해조 상태와 전이(`z_on/off/sb/su/sd_G`, 이진)와 블록별 대칭 예비력 `r_sym`(연속)이다(`stochastic_extension.FIRST_STAGE_PREFIXES`). 열펌프 commitment는 없다(2026-09-30부터 선형).

**recourse**는 시나리오별 MILP다. 정수변수는 전해조 PWL 구간 이진변수와 상보성 이진변수 `y_sto`, `y_dir`(`complementarity='bigm'`)이다. 상보성 이진변수는 LP 완화가 정확하지 않다. 수출가가 음수이면 동시 충·방전이 이득이 되기 때문이다.

**r_sym 이산화.** Laporte–Louveaux(L-L) 최적성 cut은 1단계가 순수 이진이어야 유효하다. 그래서 블록마다 6비트 이진 전개를 쓴다.

r_sym_i = Δ · Σ_{k=0..5} 2^k · b_ik,  Δ = r_max / 63

- r_max는 연속 EF 해의 최대 r_sym × 1.25다(`--r-max-factor`). 해에 의존하는 실험용 편법이다. 실제로는 물리적 상한이 필요하고, 그러면 Δ가 커진다.
- 인코딩 제약은 master에만 둔다. subproblem에서 r_sym은 연속변수(상한 r_max)이고 master 값으로 고정된다.
- n=15: Δ = 0.047 MW. n=60: Δ = 0.171 MW. 1단계 이진변수는 n=15에서 504개, n=60에서 1,584개다.

## 2. 알고리즘 (`integer_lshaped.py`)

- **multi-cut**: 시나리오마다 θ_w를 둔다. Gurobi lazy-constraint branch-and-cut으로 푼다.
- **정수 master 해에서**: 시나리오 MIP를 풀고, L-L cut θ_w ≥ (Q_w(x̂) − L_w)(Σ_S x − Σ_{¬S} x − |S| + 1) + L_w를 넣는다. Q_w(x̂)는 MIP dual bound를 쓴다(유효). LP Benders cut과 strengthened Benders(SB, Zou–Ahmed–Sun) cut도 함께 넣는다.
- **L_w**: recourse의 LP 완화값을 쓴다(`--lw lp`, 기본값). MIP dual bound(`--lw mip`)는 n=15, |Ω|=20에서 467~522초가 들었고, 하한 이득은 없었다.
- **root 루프**: master LP 완화 위에서 단계별(`--root-cuts lp,sb,lag,cp`)로 진행한다.
- **x를 행으로 묶기**: subproblem LP에서 x = x̂를 bound가 아니라 행으로 고정한다. 최적이면 그 행의 dual이 cut 기울기다. infeasible이면 **Farkas feasibility cut**을 넣는다. relatively complete recourse는 정수 x에서만 성립하고, 분수 x̂에서는 recourse LP가 infeasible할 수 있다.
- **부분 분해**(`--keep k`, Crainic–Hewitt–Maggioni–Rei): 시나리오 k개를 master 안에 MILP 그대로 둔다.
- **warm-start**(`--warm 3`): 콜백 없이 master MILP를 몇 번 풀면서 해를 평가하고 cut을 넣는다. 가장 좋은 해는 MIP start로 넘긴다. 콜백에서 찾은 좋은 해도 `cbSetSolution`으로 Gurobi에 넘긴다.
- **병렬**(`--workers k`): stochastic CG의 `pricing_workers`와 같은 방식이다. worker마다 Gurobi env를 하나씩 두고 시나리오를 round-robin으로 배정한다.
- **실험적 옵션**: Lagrangian cut(`lag`, Chen–Luedtke식 제한 분리 + Kelley), SB로 조인 L-L cut(`--cuts ...,llsb`), lift-and-project cut-and-project(`cp`, Bodur–Dash–Günlük–Luedtke; Balas CGLP를 Gurobi LP로 구현), in-out stabilization(`--stab`).

## 3. 결과

### n=15, |Ω|=20 (이산 EF 145초, 최적 −6667.2536)

| 구성 | 시간 | root 하한 | 비고 |
|---|---|---|---|
| A: root LP→SB 30라운드씩, 부분 분해 없음 | 1800 s 제한, gap 0.87% | −7067 | 버그 있는 root |
| B: 시나리오 1개 master에 유지 (`--keep 1`) | 986 s | −6698.5 | L_w MIP 467 s 포함 |
| C: root LP→Lagrangian | 1500 s에 중단, gap 5.6% | −7091 | Lagrangian 740 s |
| E1: B + L_w LP + warm-start | 494 s | −6698.9 | |
| E2: E1 + SB로 조인 L-L | 732 s | −6698.3 | 노드 3.6배 |
| F1: Kelley 수렴 + Farkas, 부분 분해 없음 | 640 s | −6698.3 | 152라운드 |
| F2: F1 + `--keep 1` | 426 s | −6698.3 | |
| F3: F1 + cut-and-project 15라운드 | 948 s | −6682.2 | CGLP 1,028개, 611 s |
| **G1: F2 + 16 workers** | **90.7 s** | −6698.3 | |
| G2: G1 + cut-and-project 4라운드 | 151 s | −6684.8 | |

A~E는 root 루프에 버그가 있던 버전이다(4절). 그래서 그 root 하한은 방법의 한계가 아니다.

### n=6 / n=15, |Ω| 작을 때

- n=6, |Ω|=3: 처음 구현은 589초였다(EF 0.45초). G1 구성에 해당하는 수정 뒤로는 3초 안팎이다.
- n=15, |Ω|=5: 처음 구현은 30분에 gap 4.6%였다(이산 EF 8초).
- 시나리오가 적을수록 EF가 압도적이다. 이 규모에서 분해법이 EF를 이긴 문헌 사례도 없다.
- 일부 실행 사이에 다른 세션이 저장장치 효율 키를 고쳤다(`nu_ch_E`/`nu_dis_E`). 그래서 n=6 EF 값이 −2905에서 −2909로 바뀌었다. n=15, |Ω|=20의 EF는 수정 전후로 같은 −6667.2536이었다.

### n=60, |Ω|=20

| | 최선 해 | 하한 | gap | 시간 |
|---|---|---|---|---|
| **연속 EF** | −25468.94 | −25471.47 | 1e-4 | **349 s** |
| 이산 EF (6비트) | −25438.25 | −25449.75 | 0.045% | 3600 s 제한 |
| E1 (순차) | −25440.44 | −25452.52 | 0.047% | 3600 s 제한 |
| G1 (16 workers) | −25425.80 | −25462.78 | 0.15% | 약 2100 s에서 중단 |

- n=60에서 G1은 순차 실행보다 나아 보이지 않는다. worker당 스레드가 1개라서 큰 시나리오 MIP가 느려지는 것으로 추정하지만, 확인하지는 않았다.
- 어느 구성도 연속 EF에는 비할 바가 아니다.

## 4. 하한에 대해 배운 것

- **처음의 약한 root 하한(gap 6~10%)은 버그였다.**
  - 분수 x̂에서 recourse LP가 infeasible하면 cut 없이 넘어갔다.
  - cut이 0개면 root 루프가 끝났다.
  - 30라운드 상한과 정지 규칙도 너무 일렀다.
  - Farkas cut을 넣고 수렴까지 돌리면, Kelley는 n=6에서 250라운드(4.5초) 만에 EF LP 값 −2923.3352로 정확히 간다. n=15, |Ω|=20에서는 152라운드(42초)다.
- 부분 분해가 처음에 극적으로 보였던 것(root gap 6% → 0.5%)도 상당 부분 이 버그를 우회한 효과다. master 안의 실제 시나리오가 분수 x를 막아줬기 때문이다. 버그를 고친 뒤에도 부분 분해는 1.5배 정도 빠르다(F1 640 s → F2 426 s).
- **cut-and-project는 하한을 끝까지 닫는다.**
  - n=6에서 root 하한이 −2909.2587로, 연속 EF 최적과 같아졌다.
  - n=15에서는 15라운드로 root gap을 0.47%에서 0.22%로 줄였다.
  - 다만 cut이 쌓이면 CGLP가 커져서, 라운드 시간이 1초에서 80초까지 늘어난다. 시간 대비 효과로는 손해였다.
- **Lagrangian cut**: 본질적으로 시나리오별 Lagrangian dual을 푸는 일이다. Kelley 8회와 최근 기울기 10개의 span으로 제한한 근사는 SB와 별 차이가 없었고, root에 740초를 썼다.
- **in-out stabilization**(λ = 0.2, 0.5): 60라운드에서 순수 Kelley와 같았다. 라운드가 싸서 Kelley를 끝까지 돌리는 편이 단순하다.
- **SB로 조인 L-L cut**(상수 L_w 대신 SB 함수를 바닥으로): 유효하지만 노드가 3.6배로 늘었다.
- **L-L cut 자체**는 평가한 한 점에서만 조여지므로, 유한 수렴 보장 외에는 기여가 거의 없다. 실제 수렴은 Benders/SB cut과 분기가 만든다.

## 5. 한계

1. **이산화.** L-L cut의 정확성은 1단계가 순수 이진일 때만 성립한다. 그래서 연속 r_sym을 이산화해야 했고, 다음을 치렀다.
   - 원래 문제와의 손실이 있다. n=60에서 약 0.11%로 gap 허용치보다 크다.
   - 이진 전개의 약한 LP 완화 때문에 이산 EF가 10배 넘게 느려진다. 그래서 분해법의 비교 기준이 실제보다 약해진다.
   - master의 이진변수가 늘어서(n=60에서 1,584개) 분기 대상이 많다.
   - r_max를 연속 EF 해에서 가져왔다. 실제로 쓰려면 해와 무관한 상한이 필요하다.
2. **비교 기준.** 원래 문제의 경쟁 상대는 연속 EF다. |Ω|=20에서 n=15는 수십 초, n=60은 349초에 풀린다. 분해법이 의미를 가지려면 시나리오가 수백 개로 늘어서 EF가 메모리나 시간에서 막혀야 한다.
3. **해 품질.** n=60에서 L-shaped는 좋은 1단계 해를 늦게 찾는다. warm-start도 큰 도움이 되지 않았다.
4. **연속 1단계와 정수 recourse를 정확히 다루려면 r에 대한 공간 분기가 필요하다**(Li & Grossmann 2019, *J. Global Optim.*). Q_w가 r에 대해 볼록하지 않기 때문이다. integer L-shaped의 틀 안에서는 이를 피할 수 없다.

## 6. 재현

```bash
python ieee_owen/integer_lshaped.py --n 15 --scenarios 20 --bits 6 --keep 1 --warm 3 \
    --root-rounds 3000 --stall-tol 1e-7 --workers 0 --root-cuts lp --verbose
```

- EF(연속, 이산)는 `weak_eps_experiment/results_lshaped/ef_<instance>_S<|Ω|>_seed<s>_b<bits>.json`에 캐시된다. 모델을 바꾸면 이 파일을 지워야 한다.
- 실행마다 결과 JSON이 같은 폴더에 남는다.

## 7. 시나리오 DW의 branch-and-price (`scenario_bnp.py`)

### 정식화

비예측성 x_w = x̄를 쌍대화하면(Carøe & Schultz 1999), 시나리오마다 커뮤니티 전체 MIP 하나로 분해된다.

- **열**: 시나리오별 계획 (x_w, y_w), 비용 p_w (c1 x + c2_w y)
- **master**: Σ_k x_wk λ_wk − x̄ = 0 (행마다 dual π_wn), Σ_k λ_wk = 1
- **pricing**: 시나리오 w의 MIP를 1단계 변수까지 풀어서, 목적함수 p_w c_w − π_w·x로 푼다.
- **하한**: 어떤 π에서든 L(π) = Σ_w min(p_w c_w − π_w·x) + min_{x̄∈box} (Σ_w π_w)·x̄가 유효하다. pricing MIP의 dual bound를 쓰므로 pricing gap과 무관하게 유효하다. 수렴값이 z_D이고, Lagrangian cut을 끝까지 넣었을 때의 하한과 같다.
- **분기**: 원래 1단계 변수로 분기한다. x̄의 분수 이진변수는 0/1로, 그다음 시나리오 열들이 서로 다른 r_sym은 x̄ 값에서 구간을 나눈다. 이산화가 필요 없고 원래 문제의 전역 최적을 보장한다.
- **해(UB)**: 양의 가중치를 가진 열의 1단계 해를 모든 시나리오에서 평가한다. 정수 x에서는 complete recourse다. 평가한 계획은 그대로 열로도 넣는다.
- **병렬**: `integer_lshaped.py`와 같은 worker 풀을 쓴다.

논문의 DW와는 다른 분해다. 논문은 커뮤니티 결합 행을 쌍대화해서 프로슈머별로 블록을 나눈다. 프로슈머 열들은 결합 행에서 서로 **더해지므로** master가 쉽게 움직인다. 반면 비예측성 행은 모든 시나리오의 열 조합이 384차원(n=15)에서 **같은 x̄를 동시에** 만들어야 움직인다. 그래서 퇴화가 훨씬 심하다.

### 안정화: 무엇이 필요했나 (n=6, |Ω|=3; 연속 EF 최적 −2909.2587)

| 구성 | 결과 | 시간 |
|---|---|---|
| 안정화 없음, π = 0에서 출발 | 500회, 하한 −7720, master는 초기 해에서 정지 | 50 s |
| box-step δ=10, π = 0 | 2,000회, 하한 −2999.6 | 715 s |
| box-step δ=1, **EF LP 비예측성 dual π_LP에서 출발** | 300회, 하한 −2909.2798, gap 7e-6 | 147 s |
| box-step δ=10, π_LP | 하한 −2916.27에서 정지 | 100 s |
| **Wentges smoothing(α = 0.1, Pessoa 적응형) + 3구간 penalty(ε = δ = 0.2, 라운드마다 0.25배), π_LP** | **54회, z_D = UB = −2909.2587, gap 2.6e-9** | **9.8 s** |

- **π_LP 출발**: 시나리오별로 1단계를 복사한 EF의 LP를 푼다(`lp_duals`). 그 비예측성 dual에서는 L(π_LP) ≥ z_LP가 보장된다.
- **안정화**: smoothing과 penalty는 `stochastic_extension.DirectMaster`의 것을 그대로 옮겼다(`_alpha`, `_set_penalty`). box-step은 고정 δ에 매우 민감했다.
- **z_D의 강도**: n=6에서는 z_D가 최적값과 같아서 분기가 필요 없었다.

### n=15, |Ω|=20 (연속 EF 약 32 s, 최적 −6667.2536, EF LP −6698.3188)

| 구성 | 반복 | 하한 | 시간 |
|---|---|---|---|
| box-step δ=1, π_LP | 200 | −6698.23, master는 초기 해 −6593.69에서 정지 | 2,553 s |
| smoothing + penalty, π_LP, pricing gap 1e-6 | 80 | −6698.23 | 3,266 s |
| smoothing + penalty, π_LP, pricing gap 1e-4 | 111 | −6698.32 | 7,204 s |

- **하한은 한 번도 L(π_LP)를 넘지 못했다.** smoothing으로 탐색한 π가 모두 이보다 낮은 L을 냈다. L(π_LP)는 EF LP 값과 거의 같다. 즉 π_LP에서 시나리오 MIP들의 정수 gap이 거의 0이다.
- **z_D의 위치는 확정하지 못했다.** penalty slack이 쓰이는 동안의 master 값(마지막 −6640.5)은 z_D의 상한이 아니다. 유효한 범위는 z_D ∈ [−6698.32, −6667.25](위쪽은 최적값)뿐이다. z_D가 EF LP 값 근처라면, n=6과 달리 쌍대 분해가 root gap을 거의 닫지 못하고 분기가 많이 필요하다.
- **반복 비용**: 반복당 30~60초다. pricing은 반복당 20~40초다. 커뮤니티 전체 MIP 20개를 worker당 스레드 1개로 풀고, gap을 1e-4로 풀어도 거의 줄지 않았다. master는 행 7,680개에 1단계 성분이 촘촘한 열 2,000개 이상이 쌓여서 반복당 10~20초까지 커졌다.
- **결론**: 이 규모에서는 연속 EF(약 32초)와 비교할 수 없다. branch-and-price의 분기 단계는 n=6 이외에서는 시험하지 못했다.

### 남은 선택지 (하지 않음)

- **proximal bundle로 Lagrangian dual을 직접 풀기**: DW master 대신 cut 몇십 개짜리 QP를 쓴다.
- **dual 차원 줄이기**: 예를 들어 전이 변수 su/sd는 복사하지 않고 상태 변수만 복사한다.
- **pricing 스레드를 늘리고 worker 수를 줄이기**
- **z_D를 확정하려면** penalty 없는 마지막 라운드까지 돌리거나, bundle로 dual을 수렴시켜야 한다.

```bash
python ieee_owen/scenario_bnp.py --n 6 --scenarios 3 --root-only     # 9.8 s, gap 2.6e-9
python ieee_owen/scenario_bnp.py --n 15 --scenarios 20 --root-only --price-gap 1e-4
```

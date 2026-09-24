# 확률적 확장의 column generation: 무엇이 느리고, 무엇을 시도했나

Sep 24, 2026 · @Seokwoo Kim

`ieee_owen/stochastic_extension.py`의 direct engine(`DirectMaster`)을 n=60, |Ω|=5까지 키우면서 측정한 기록이다. 결론은 다섯 가지다.

1. **지금 기본값에서 n=60, |Ω|=5는 gap 1e-6까지 약 5분(299~313초)에 풀린다.** 시작할 때는 2시간 48분 동안 500회를 돌고도 수렴하지 못했다.
2. 결정적인 한 수는 수지 행의 **DOI**(커뮤니티가 계통과 시장가로 사고파는 컬럼, 8절)다. master의 primal degeneracy를 풀어서, DOI 없는 구성(21~29분)보다 4.8~6.9배 빠르고 ω^LR은 정밀도 안에서 같다.
3. DOI가 없을 때의 병목은 master LP였다. 원인은 행 수가 아니라 촘촘한 컬럼이 수천 개 쌓이는 것이다. 그래서 행을 줄이는 방법(dyn-SAR, lazy 행)과 master 알고리즘을 바꾸는 방법(barrier, bundle)은 모두 이득이 없었다(4절). DOI를 넣은 뒤의 병목은 pricing(약 70%)이다.
4. ω^LR을 믿을 만하게 내려면 EF와 CG 종료 gap을 1e-6으로 둬야 한다(pricing은 1e-4면 된다). 지금 기본값이 그렇게 되어 있다.
5. n=60에서는 EF의 LP 완화값이 v^LR과 사실상 같다. 그래서 LP 한 번으로 거의 최적인 dual이 나온다(9절). 그래도 CG가 필요한 것은 primal 쪽, 즉 UB를 증명하는 일 때문이다.

모든 수치는 n=60, |Ω|=5, `PYTHONHASHSEED=21`, Gurobi, 16코어 기준이다. 다르면 따로 적었다. 실행 간 시간 편차는 같은 설정에서도 약 15%다(해시 시드 21과 22에서 1443초와 1247초).

## 1. 인스턴스와 master의 모양

- linking row 720개: 6종류(E, H, G 수지; 예비력 up, dn; peak) × 24시간 × 5시나리오. 여기에 prosumer별 convexity 행 60개가 붙는다.
- 컬럼은 prosumer 하나의 5시나리오 계획 전체다. 모든 시나리오, 시간, 에너지에 걸쳐 거래하므로 컬럼 하나가 행 720개 중 평균 **277개(38%)** 를 채운다. 풍력·전해조·저장장치를 가진 prosumer는 약 450개다.
- 최종 master는 행 780개, 컬럼 약 5,500개(prosumer 컬럼 약 4,000개), 비영 원소 113만 개다. λ>0인 컬럼은 약 480개다.
- 수렴 시점의 binding 상태를 보면 peak 행은 120개 중 110개가 여유가 있고, dn 행은 51개, G 행은 60개가 dual 0이다. E와 H 행은 전부 binding이다.
- 최적 가격(π/ρ)이 시나리오 사이에 같은 시간은 E 1/24(차이 중앙값 27, 최대 184 EUR/MWh), H 20/24, G 18/24이다.

### |Ω|=1 대비 |Ω|=5 (같은 설정, gap 1e-6)

| | \|Ω\|=1 | \|Ω\|=5 | 배율 |
|---|---|---|---|
| linking row (dual 차원) | 144 | 720 | 5× |
| CG 반복 | 124 | 758 | 6.1× |
| master LP / 반복 | 0.03s | 1.24s | 41× |
| pricing MILP 하나 | 10ms | 40ms | 4× |
| CG 전체 | 13s | 1443s | 111× |

111배는 반복 수 6배와 반복당 비용 약 18배의 곱이다. 반복당 비용은 거의 master LP에서 나온다. LP가 무거워지는 이유는 세 가지가 곱해져서다.

- 행이 5배다.
- 컬럼이 4배 촘촘하다(컬럼당 비영 원소 약 52 → 196).
- 반복 동안 컬럼이 쌓인다.

반복이 느는 이유는 dual 차원이 5배이기 때문이다. CG는 Lagrangian dual의 cutting plane이다.

## 2. 효과가 있었던 것 (지금 기본값)

| 변경 | 이유와 효과 |
|---|---|
| **bound 기반 종료** (Lagrangian bound early termination) | 원래는 "reduced cost < −1e-8·\|z\|인 컬럼이 없으면 종료"였다. pricing을 gap 1e-4로 부정확하게 풀면 이 조건은 성립하지 않는다. 2시간 48분 동안 RMP가 1e-4만 움직이며 돌았다. 지금은 LB(pricing MILP dual bound의 합, Desrosiers & Lübbecke primer 식 1.31)와 UB(penalty를 뺀 RMP)의 gap으로 끝낸다. pricing이 부정확해도 LB는 유효하다. 컬럼은 reduced cost가 음수면 모두 넣는다. LB가 pricing 정밀도 때문에만 막히면 그 prosumer의 gap만 1/10로 조인다. |
| **병렬 pricing** | 60개 MILP를 순차 1스레드로 풀면 반복당 17초였다. 16개 동시에 풀면 0.3초다. Gurobi 환경(WLS 세션)은 worker당 하나다. prosumer마다 따로 열면(60개) 실행 중 라이선스가 끊겼다. |
| **primal simplex master** | Gurobi 기본(concurrent)보다 LP 시간이 절반이다. 컬럼이 추가만 되므로 이전 basis가 primal feasible하게 남는다. |
| **UB 전용 LP** | penalty slack을 닫았다 여는 방식은 UB 확인 한 번에 0.4초였다. slack 없는 쌍둥이 LP를 따로 두면 warm start가 유지돼 UB 확인 비용이 1/5이다. 확인 간격은 30회다. |
| **column purge** | 50회 이상 안 쓰이고 reduced cost가 양수인 컬럼을 컬럼 수가 prosumer당 40개를 넘을 때만 지운다. 너무 일찍 지우면 degenerate한 master가 빠져나오지 못한다(아래 3절). EF 해로 만든 초기 컬럼은 보호한다. 지우면 penalty가 줄어들 때 LP가 infeasible이 된다. |
| **초기 컬럼 수선** | EF 해는 수지를 약 1e-7만큼만 맞춘다. 모든 prosumer가 초기 컬럼 하나씩일 때 Gurobi가 LP를 실행마다 infeasible로 판정했다. 행별 잔차를 계수가 가장 큰 prosumer의 커뮤니티 거래에 흡수시킨다. 비용이 0이라 컬럼 비용은 그대로다. |

결과: purge와 UB 확인 간격 30을 쓴 설정이 gap 1e-4에서 780초, gap 1e-6에서 1443~1504초다.

## 3. 이 문제가 어려운 이유

1. **master가 primal degenerate하다.** 새 계획은 다른 prosumer의 계획과 조합돼야만 쓸모가 있다. penalty 없이 돌리면 RMP가 초기값에서 수백 회 움직이지 않는다(n=6에서 수백 회; n=60에서 275회 이상). 3구간 penalty가 이 정체를 푸는 핵심이고, smoothing이 없으면 라운드 1이 4배 느리다(270초 → 1016초).
2. **컬럼이 촘촘하고 많다.** 2단계 문제라 계획 하나가 모든 시나리오에 걸친다. 컬럼 밀도는 인스턴스 구조의 성질이고, 알고리즘이 줄여 주지 않는다.
3. **dual 차원이 시나리오 수에 비례한다.** 시나리오를 늘리면 반복 수와 master 크기가 함께 는다.
4. **평균으로 묶을 수 있는 행이 사실상 없다.** 이것은 가격(dual)의 문제가 아니라 primal의 문제다. 4.1절 참고.

## 4. 효과가 없었던 것

### 4.1 dyn-SAR (Costa, Contardo, Desaulniers & Yarkony 2022), `--sar`

시나리오를 확률 가중으로 합친 행에서 시작해 위반된 원래 행을 분리하는 방법이다. SCIP 엔진에서 `DirectMaster`로 옮겼고, penalty를 family 단위로 결합했다. penalty 없이 돌리면 마지막 단계가 n=6에서도 12,000회 넘게 멈췄다.

| | 반복 | 시간 | 최종 행 |
|---|---|---|---|
| 기존 CG | 758 | 1443s | 720 |
| dyn-SAR | 1023 | 1962s | 723 |
| dyn-SAR, E만 시나리오별로 시작 (`--sar-exact E`) | 775+ | 1300s+ | 712 |
| 열·수소 부하도 고정 (`--load-carriers E`): CG / dyn-SAR E | 641 / 960 | 1383s / 1795s | 720 / 727 |

거의 모든 행이 결국 분리된다. 첫 분리에서만 419개(58%)다. 합친 행도 남기 때문에 최종 master가 원래보다 커진다. 열 가격이 24시간 모두 시나리오 간 같은 경우에도 열 행이 분리됐다. 이유는 다음과 같다.

- 평균 제약은 "시나리오 1은 +1, 시나리오 2는 −1" 같은 **상쇄 해**를 허용하고, LP는 비용이 같은 여러 해 중 그런 해를 고를 수 있다.
- dual이 같으면 **하한(목적함수 값)은 맞지만**, v^CHP를 인증하는 데 필요한 **개별 행을 지키는 primal 해**는 저절로 나오지 않는다.

dyn-SAR이 이득을 보는 것은 집계 해가 저절로 개별 행을 만족하는 구조, 즉 bin packing 같은 set partitioning이다.

### 4.2 lazy 부등식 행, `--lazy-rows peak,dn`

최종 해에서 여유가 있는 peak와 dn 행을 빼 두고 위반될 때 넣는 방법이다. 1326초(−8%)였는데, 편차(15%) 안이라 구별할 수 없다. lazy 행 240개 중 **235개가 돌아왔다.** 최적에서 여유가 있는 행도 CG 도중에는 거의 다 한 번씩 위반된다. up과 dn을 둘 다 lazy로 두면 r_sym이 무한대로 가서 막아 두었다.

### 4.3 barrier master, `--lp-method barrier`

Gurobi `Method=2`, crossover 없이, primal-dual CG(Gondzio, González-Brevis & Munari 2013)의 중심에 가까운 dual을 노린 것이다. 라운드 1만 431초로, simplex(약 270초)보다 느렸다.

- 해가 내부점이라 λ가 전부 양수이고, purge가 한 번도 발동하지 않는다. 컬럼이 10,700개, LP가 반복당 4초까지 늘었다.
- warm start도 없다. n=6에서도 86초로 simplex의 54초보다 느렸다.

### 4.4 번들 메소드, `--bundle`

disaggregated proximal bundle이다. master QP는 max Σθ_u − (1/2t)‖π−π̂‖²이고, dual 형태로 푼다. dual 형태는 linking 잔차 z에 대한 이차 penalty가 붙은 RMP이고, π = π̂ − t z다.

구현하면서 고친 것:

- **부정확한 pricing:** step 판정은 pricing incumbent 값(Σ obj_u)으로 하고, LB 인증만 dual bound로 한다. dual bound로 판정하면 모델이 정확한 step도 gap만큼 실패로 읽혀 null step만 반복했다. 중심값은 Σ min(θ_u, obj_u)(그 점에서의 모델 값)으로 잡아 δ ≥ 0을 지킨다. serious step의 cut은 항상 넣고, 중심이 바뀔 때까지 지우지 않는다.
- **t:** 가격 크기(수십~수백)와 잔차(약 1)를 맞추려면 t≈100이 필요했다. 작으면 반복이 크게 는다.

결과:

| | 반복 | 시간 | 비고 |
|---|---|---|---|
| n=6 기존 CG | 524 | **60s** | |
| n=6 번들, barrier QP, t=100 | **473** | 85s | QP 46s |
| n=6 번들, simplex QP | 685 | 154s | |
| n=60 기존 CG | 758 | **1443s** | gap 1e-6 |
| n=60 번들, barrier QP | 275 | 23분에 gap 약 29 | 컬럼 11,590개로 증가 |
| n=60 번들, simplex QP | 600+ | 41분에 gap 약 3 | 외부에서 종료됨 |

기대한 효과 두 가지가 모두 나타나지 않았다.

1. **master가 작아지니 반복당 비용이 준다: 줄지 않았다.** 반복당 master 비용은 n=60에서 CG가 약 1.2초, 번들이 약 2.4~4초였다. 컬럼 수는 3분의 1로 줄었다(약 1,300개).
   - CG의 LP는 반복 사이에 컬럼만 추가되므로 이전 basis에서 조금만 움직이면 된다.
   - 번들의 QP는 중심 π̂과 t가 바뀌면서 목적함수가 매번 달라진다. 그래서 warm start가 약하고, QP 피벗도 LP보다 비싸다.
   - barrier로 풀면 warm start가 없고 λ가 전부 양수라 cut이 지워지지 않는다.
   - dual이 720차원이라 모델을 제대로 유지하려면 cut이 차원 수 이상 필요하다. CG의 최종 master도 λ>0인 컬럼이 약 480개였다. 줄일 여지 자체가 작다.
   - cut을 합쳐서 강제로 줄이면(compression) 조합이 필요한 이 문제에서는 정보가 사라져 멈춘다.
2. **안정화가 좋아져 반복이 준다: 오히려 늘었다.**
   - 반복의 약 80%가 null step이다(n=6에서 376/473). null step은 pricing 60개를 다 풀고도 중심을 못 옮기는 반복이다.
   - 이차 항이 π를 중심 쪽으로 당겨서 끝부분 수렴이 느리다. null step이 많아 t가 바닥(20)에서 올라가지 못했다. 600회째에도 gap이 3이었는데, CG는 LP 해가 꼭짓점이라 컬럼이 모이면 LB = UB로 정확히 끝난다.
   - cut을 지우니 UB LP의 컬럼도 적어서, UB가 약 375회까지 초기값에 머물렀다.
   - gap 1e-4의 잡음 때문에 중심값을 보수적으로 잡아야 했고, 그만큼 serious step이 어려워졌다.

같은 결론이 Briant et al.(2008)의 비교에도 있다. 번들과 안정화된 CG 사이에 분명한 우열은 없다. 이 문제에서는 3구간 penalty + smoothing이 이미 안정화를 충분히 하고 있다.

## 5. ω^LR의 정확도와 gap

- ω^LR = v^MIP − v^LR은 약 −23,600짜리 값 두 개의 차이다. 결정과 무관한 고정 수요 항(`d_*_nfl`, n=60에서 −31,213, |z|의 132%)은 양쪽에서 지워진다.
- 상대 gap은 |z|로 나누므로 1e-4는 절대 2.4를 허용한다. ω(≈3.7)의 약 65%다.
- ω가 확실히 들어가는 구간 [EF bound − CG UB, EF incumbent − CG LB]:

| EF gap | CG gap | 보고된 ω | 구간 |
|---|---|---|---|
| 1e-4 | 1e-4 | 5.36 | [2.50, 5.68] |
| 1e-4 | 1e-6 | 3.80 | [2.04, 3.80] |
| **1e-6** | **1e-6** | **3.728 / 3.715** (두 시드) | **[3.709, 3.732] / [3.707, 3.730]** |

- EF를 1e-6으로 풀어도 약 10초로 1e-4와 거의 같고, 최적성이 증명된다. pricing gap은 1e-4로 충분하다. CG가 필요할 때 스스로 조인다(실행당 한 번 발동).
- 기본값: `--mip-gap 1e-6`(EF, stand-alone), `--cg-gap 1e-6`, `--pricing-gap 1e-4`.
- `run_multiday.py`(본 실험)의 EF도 1e-4로 풀렸으니 표의 ω^LR과 ε^LR은 따로 확인이 필요하다. 아직 하지 않았다.

## 6. 남은 방향

- **시나리오 분해 DW** (Schulze, Grothey & McKinnon 2017, 확률적 unit commitment): 컬럼은 5분의 1로 희소해지지만, 1단계 결정(commitment)의 시나리오 간 일치 조건까지 완화하게 된다. convex hull을 시나리오별로 잡는 더 약한 완화라서 **v^LR과 dual 자체가 바뀐다.** 즉 논문이 정의하는 Owen 배분이 달라지므로, 속도 개선이 아니라 모델 변경이다. 쓰지 않는다.
- **pricing** (DOI 이후의 병목): 반복마다 MILP 60개(개당 약 40ms)를 16개씩 병렬로 푼다. 가장 느린 prosumer 묶음이 반복 시간을 정한다.
- **degeneracy 도구** (IPS, DCA, Row-reduced CG; Elhallaoui et al. 2005, Desrosiers, Gauthier & Lübbecke 2014, Raymond et al.): 주로 set partitioning용이다. 여기서는 degenerate basic이 551개 중 64개로 중간 수준이라 효과가 제한적일 것이다.
- **번들을 계속 판다면:** QP의 이차항은 t가 바뀔 때만 다시 쓰고, 나머지는 속성값으로만 바꿔 warm start를 살린다. t를 키우는 규칙도 개선한다(Kiwiel의 곡률 추정). 초반만 번들로 dual을 모으고 후반은 CG로 넘기는 혼합도 있다.
- **재현성:** 같은 명령이라도 EF 해가 실행마다 조금 다르다(1e-4 gap에서 −23597.65 ~ −23598.24). 파이썬 해시 순서가 모델을 만드는 순서를 바꾸는 것으로 보인다. `PYTHONHASHSEED`를 고정해야 재현된다.

## 7. 재현

```bash
# 지금의 기본 구성 (gap: EF 1e-6, CG 1e-6, pricing 1e-4; DOI, 병렬 pricing과 부하 분산,
# purge, UB 확인 30회)
PYTHONHASHSEED=21 python ieee_owen/stochastic_extension.py --n 60 --scenarios 5 \
    --lp-solver gurobi --pricing-solver gurobi --mip-solver gurobi --skip-standalone
# 끌 수 있는 기본값: --no-doi, --no-balance-pricing
# 비교한 변형 (모두 기본값에서 꺼져 있음)
#   --dual-init lp|fix [--pen-eps 0.01 --pen-delta 0.01]   dual 웜스타트
#   --column-pool / --mip-start                             컬럼 풀, pricing MIP 시작해
#   --sar [--sar-exact E]       dyn-SAR
#   --lazy-rows peak,dn         lazy 부등식 행
#   --lp-method barrier         barrier master
#   --bundle --bundle-t 100 --bundle-t-min 20 [--bundle-qp primal --bundle-cap 100000]
#   --load-carriers E           열·수소 부하 불확실성 제거
#   --no-penalty / --no-smoothing / --cg-gap 1e-4
```

참고문헌:
- [Desrosiers & Lübbecke, A Primer in Column Generation](https://homes.di.unimi.it/~trubian/primer.pdf)
- [Lübbecke, Column Generation](https://www.or.rwth-aachen.de/files/research/publications/colgen.pdf)
- [Pessoa, Sadykov, Uchoa & Vanderbeck 2018](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2017.0784)
- [Kiwiel, inexact bundle](https://link.springer.com/article/10.1007/s10107-007-0187-4)
- [Briant et al. 2008](https://link.springer.com/article/10.1007/s10107-006-0079-z)
- [Frangioni, Stabilization in CG](https://pages.di.unipi.it/frangio/papers/StabCG.pdf)
- [Gondzio, González-Brevis & Munari](https://arxiv.org/abs/1309.2168)
- [Elhallaoui et al. 2005](https://pubsonline.informs.org/doi/10.1287/opre.1050.0222)
- [Desrosiers, Gauthier & Lübbecke 2014](https://www.sciencedirect.com/science/article/abs/pii/S0377221713009922)
- [Schulze, Grothey & McKinnon 2017](https://www.sciencedirect.com/science/article/abs/pii/S0377221717301108)

## 8. DOI: 커뮤니티가 계통과 직접 거래하는 컬럼 (기본값)

수지 행 (k, t, ω)마다 컬럼 두 개를 master에 넣는다. Ben Amor, Desrosiers & Valério de Carvalho (2006), Gschwind & Irnich (2016)의 dual-optimal inequality다.

- "계통에서 산다": 행 계수 −1, 비용 ρ·수입가
- "계통에 판다": 행 계수 +1, 비용 −ρ·수출가
- 전력은 peak 행에도 반대 부호로 들어간다.

dual 쪽에서 보면 커뮤니티 가격을 [수출가, 수입가] 상자에 가두는 부등식이다. primal 쪽에서 보면 **새 계획이 다른 prosumer의 짝을 기다리지 않고 들어갈 수 있게 한다.** 빈 수지를 계통 컬럼이 채우기 때문이다. 3절의 primal degeneracy를 정면으로 푸는 장치다.

수입 한도 때문에 이 부등식이 미리 유효하다는 보장은 없다. 그래서 두 가지 안전장치를 둔다.

- UB는 계통 거래량 y = 0일 때만 인정한다. 주 LP와 UB용 LP 둘 다 그렇다.
- 수렴했는데 y > 0이면 계통 컬럼을 끄고(상한 0) CG를 이어 간다.

LB는 원래 문제의 Lagrangian bound라서 어느 경우에도 유효하다. 측정한 모든 실행에서 안전장치는 한 번도 발동하지 않았다(`doi.active_at_end`).

| 인스턴스 | DOI 없음 | **DOI** | 속도 | ω (없음 / DOI) |
|---|---|---|---|---|
| n=6, \|Ω\|=3 | 524회, 62s | 308회, 33s | 1.9× | 7.0056 / 7.0056 |
| n=30, 시드 21 | 766회, 785s | 353회, **114s** | 6.9× | 1.7296 / 1.7288 |
| n=60, 시드 21 | 758회, 1443s | 485회, **299s** | 4.8× | 3.728 / 3.729 |
| n=60, 시드 22 | 951회, 1716s | 491회, **313s** | 5.5× | 3.689 / 3.727 |

n=60에서 master LP가 941초에서 76초로 줄었다(반복당 1~3초 → 0.07~0.15초). 이제 pricing이 약 70%다. ω는 시드 사이에서 3.69~3.73으로 움직이는데, gap 1e-6의 인증 구간(약 ±0.024) 안이다.

DOI 위에서 pricing과 컬럼 관리를 더 시험했다(n=60, 시드 21, DOI만 쓰면 299s).

| 추가 | 시간 | 결과 |
|---|---|---|
| 부하 분산 (`--balance-pricing`, 기본값) | 287s | pricing 215 → 201s. 해가 없어서 켜 둔다 |
| pricing MIP 시작해 (`--mip-start`) | 292s | 차이 없음 |
| 컬럼 풀 (`--column-pool`) | 393s | 더 나쁨. 풀에서 되살린 반복이 LB를 못 올려 반복이 늘고(577회), 풀 행렬을 다시 만드는 비용도 약 90초 든다 |
| 셋 다 | 343s | 더 나쁨 |

## 9. dual 웜스타트와 "LP 완화 ≈ Lagrangian 완화"

`--dual-init lp`는 EF의 LP 완화에서 linking 행의 dual을 읽어(0.2초) 첫 안정화 중심으로 쓴다. `fix`는 정수 변수를 EF 해에 고정한 LP다. EF 행은 master 행과 이름으로 짝지어진다. 부호와 ρ 배율은 공유 변수의 계수 비로 맞춘다.

- n=60에서 L(π_LP) = −23601.9535이다. CG가 인증한 v^LR 구간 [−23601.974, −23601.951] 안에 있다. 즉 **EF의 LP 완화가 Lagrangian 완화만큼 강하고, LB는 처음부터 최적이다.** n=6에서는 둘이 0.41 다르다(−2733.85 vs −2733.44). 큰 n에서만 성립하는 성질로 보인다. 논문의 O(|T|/n) 논의와 관련이 있을 수 있는데, 확인하지는 않았다.
- **n=6에서는 좋았다.** 웜스타트에 작은 초기 penalty 0.01을 주면 223회/25초(DOI 없음 기준 524회/59초)였다. 여기에 DOI를 더하면 183회/19초였다.
- **n=60에서는 오히려 나빴다.** penalty 0.02로 27분에 1,550회를 돌고도 라운드 1이었다. DOI를 더해도 마지막 라운드에서 3,000회 넘게 정체했다. dual을 최적 근처에 붙잡아 두면 pricing이 최적 면 근처의 비슷한 컬럼만 만든다. 그래서 primal 해(UB)를 만드는 조합이 잘 생기지 않는다. dual 안정화가 primal 수렴을 늦추는 전형적인 모습이다. 그래서 끈 채로 둔다.

## 10. 외부 종료

긴 실행 몇 개(n=60의 dyn-SAR E, bundle simplex, dual 웜스타트 0.01)가 traceback 없이 도중에 끝났다(exit 127). 같은 기계의 다른 세션이 Python 프로세스를 멈춘 것으로 보이지만 확인하지 못했다. 해당 표의 "중단" 표시는 이것이다.

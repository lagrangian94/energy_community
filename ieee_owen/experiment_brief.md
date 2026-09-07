# Experiment brief — finding instances with an empty core

## Goal

현재 계산된 모든 instance에서 `violation_bf = 0`, 즉 core가 비지 않았다.
논문은 "Assumption `as:pool` 하에서도 core가 빌 수 있다"를 주장하려 하는데
(`prop:core` 삭제 근거), 반례가 stylized $n=3$ 예제 하나뿐이다.
**본 모델(24기간, 저장, 예비력, 피크, 시장거래)에서 $\omega^{*}>0$인 instance를
최소 하나 확보하는 것**이 목표다. 실패해도 결과다 — "이 조건들을 다 걸어도
core가 유지된다"는 것 자체가 보고할 만하다.

## 배경 (왜 이 방향인가)

Stylized 분석에서 두 무차원 수가 나왔다.

- $\kappa := \underline d^{G} / e$ — 전해조 최소부하를 1인당 가용 잉여전력으로 나눈 값.
  "문턱을 낭비 없이 채우는 데 필요한 인원수".
- $\mu := V / (\pi^{E,imkt}\,\underline d^{G})$ — 수소 가치를 전량 매수 운전비로 나눈 값.
  $\mu \ge 1$이면 누구나 사서 켤 수 있어 문턱이 무의미해진다.

Core가 비려면 대략 다음이 동시에 걸려야 한다.

1. $\mu < 1$ — 문턱이 실제로 구속
2. $\kappa \le n-1$ — 진부분집합이 문턱을 채울 수 있음
3. $\kappa \nmid n$ — 대연합이 그 배수가 아니어서 낭비가 생김
4. 낭비가 흡수되지 않음 — 저장 부족, 잉여의 외부 판로 부족

**주의.** 위는 단일기간 균질 모델에서 나온 것이다. 24기간 + 저장이 붙으면
정수 구조가 시간축으로 번져 흐려진다. 예측이 아니라 **탐색 방향**으로 쓸 것.

## 사전 작업 (sweep 전에)

### T0. 진단 지표 함수 추가

대칭 instance(모든 prosumer가 동일 자산 구성)에서는 core 여부를 $n$번의 solve로
정확히 판정할 수 있다. $\vmip(S)$가 $|S|$에만 의존하므로 $v(s) := \vmip(S)$,
$|S|=s$로 쓰면

$$\alpha^{*} = n\max_{s} \frac{v(s)}{s}, \qquad
\omega^{*} = n\max_s \frac{v(s)}{s} - v(n)$$

즉 **1인당 가치가 $s=n$에서 최대인지**만 보면 된다.
(근거: OPAP의 실행가능집합이 볼록이고 좌표 치환에 불변이므로 치환평균을 취해
균등배분만 보면 되고, $\chi_j \equiv c$의 실행가능성은 $sc \ge v(s)\ \forall s$.)

구현할 것:

- `symmetric_omega_star(instance) -> (omega_star, s_star, per_capita_curve)`
  - $v(1), \dots, v(n)$을 각각 한 번씩 푼다 ($2^n$ 아님, row generation 아님)
  - $s^{*} := \arg\max_s v(s)/s$ 반환
  - `s_star != n` 이면 core가 빈 것
- **비대칭 instance에는 쓰면 안 된다.** 대칭성 검사(모든 prosumer의 자산
  파라미터가 동일한지)를 함수 안에 assert로 넣을 것.
- 기존 `violation_bf` / row generation 경로와 최소 3개 instance에서 값이
  일치하는지 교차검증.

### T1. $\kappa$, $\mu$ 계산기

파라미터에서 직접 계산해 로그에 남긴다. 눈대중으로 파라미터를 밀지 말고
$\kappa$가 목표값이 되도록 역산해서 격자를 잡을 것.

- $e$ = 1인당 가용 잉여전력. 정의를 결정해야 함. 후보:
  (a) 재생발전 설비용량 × 대표 capacity factor,
  (b) 24시간 평균 순잉여(발전 − 자기부하),
  (c) 잉여가 가장 큰 시간대의 순잉여.
  **(c)를 권장.** 문턱은 시점별로 걸리므로 피크 잉여가 관련 척도다.
  세 가지 다 계산해 로그에 남기고 어느 것이 $\omega^{*}$와 상관이 높은지 볼 것.
- $\underline d^{G}$: `els_cap` × 최소부하 비율. 코드에서 실제 값 확인.
- $V$: 수소 1단위 가치. `base_h2_price_eur` × 전해조 효율 반영.
- $\pi^{E,imkt}$: `import_factor` 반영된 실제 매수가. TOU면 대표값(평균/피크) 병기.

### T2. 모델–코드 정합성 확인 (독립적으로 필요한 작업)

- `eq:dp`에 시장수출 상한이 없는데 코드에는 `e_E_cap_ratio`,
  `e_G_cap_ratio`, `e_H_cap_ratio`가 있다. 코드가 거는 제약의 정확한 형태를
  확인해서 보고할 것 (변수별 상한인지, 합계 상한인지, 시점별인지).
- 재생발전 curtailment가 가능한지 확인. `eq:bnd_pd`의 $\underline p^{E}_{jt}$가
  0이면 가능, 강제 인수면 불가능. **이게 4번 조건(낭비 흡수)에 직결된다.**
- 효율곡선 세그먼트 기울기 $\phi^1_l$이 실제로 $l$에 대해 감소하는지
  (= 곡선이 concave인지) 확인. `eff_type=1`이 무엇인지 명시.

## Sweep 설계

### S1. 주 sweep — $\kappa$ 격자

대칭 배치 유지. $n$ 고정(우선 $n=6$).

- $\kappa \in \{1.0,\ 1.5,\ 2.0,\ 2.5,\ 3.0,\ 4.0,\ 5.0\}$
- 조작 변수는 **`els_cap`을 우선** (최소부하가 정격 비율이면 $\underline d^{G}$가
  따라 오르므로 $\kappa$를 직접 움직인다).
  보조로 `wind_el_ratio` / `solar_el_ratio`를 낮춰 $e$를 줄이는 경로도.
  두 경로가 같은 $\kappa$에서 같은 결과를 주는지 비교하면 $\kappa$가 실제
  설명변수인지 검증된다.
- 나머지 고정: `storage_*_ratio` 최소, export cap 낮게, `import_factor` 기본.

기대: $\kappa \in [2, n-1]$이고 $\kappa \nmid n$인 구간에서 $s^{*} \ne n$.
$n=6$이면 $\kappa = 4, 5$가 유력. $\kappa = 2, 3, 6$은 배수라 빠져나갈 것.

### S2. $\mu$ sweep — 문턱이 구속되는 구간 찾기

- `base_h2_price_eur`를 낮추거나 `import_factor`를 높여 $\mu$를 조정
- $\mu \in \{0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.5\}$
- **가동률 확인 필수.** $\mu$를 너무 낮추면 전해조가 아예 안 켜져 게임이
  자명해진다. 전해조 가동 시간수를 함께 기록하고, 0이 되는 구간은 버릴 것.
- S1의 최선 $\kappa$에서 실행.

### S3. 흡수 요인 제거

낭비가 흡수되면 core가 살아난다. 각각 단독으로 그리고 조합으로.

- `storage_capacity_ratio_E` ↓ (0에 가깝게). 근거: `full_storage`의
  `violation_ip`가 다른 시나리오의 약 1/7이다.
- `e_E_cap_ratio`, `e_G_cap_ratio` ↓ (0.2 이하). 잉여의 외부 판로 차단.
- curtailment 불가 설정이 가능하면 켤 것 (T2에서 확인).

### S4. 재생 프로파일 뾰족하게

24기간이라 잉여가 시간축으로 퍼지면 문턱 미스매치가 상쇄된다.

- 잉여가 소수 시간대에 집중된 날 선택 (기존 `day` 파라미터 활용)
- 또는 `wind_el_ratio` / `solar_el_ratio` 비중 조정으로 프로파일 첨도 변경
- 각 day에 대해 "잉여의 시간축 집중도"(예: 상위 4시간 잉여 비중)를 계산해
  $\omega^{*}$와의 상관을 볼 것

### S5. 대조군 — 비대칭 배치

**반직관적이지만 비대칭은 core를 보호한다.** 전해조 보유자가 소수면
비보유자끼리의 그룹이 무가치해져 겹치는 coalition family가 힘을 못 쓴다.

- S1의 최선 설정에서 전해조 보유자 수를 $n, n-1, \dots, 1$로 줄여가며
  $\omega^{*}$ 측정 (여기서는 대칭이 아니므로 row generation 필요)
- 기대: 보유자가 줄수록 $\omega^{*}$ 감소

## 기록할 것

instance마다 다음을 CSV에 남긴다.

| 열 | 내용 |
|---|---|
| `kappa_peak`, `kappa_mean`, `kappa_cap` | T1의 세 정의 |
| `mu` | $V/(\pi^{E,imkt}\underline d^{G})$ |
| `els_on_hours` | 전해조 가동 시간수 (자명한 instance 걸러내기) |
| `v_1` ... `v_n` | 대칭 instance의 $v(s)$ |
| `s_star` | $\arg\max_s v(s)/s$ |
| `omega_star` | $n\max_s v(s)/s - v(n)$ |
| `omega_lr` | $\mathrm{val}(\mathrm{DWR}_N) - \vmip(N)$ (기존) |
| `surplus_top4_share` | S4용 집중도 지표 |
| `is_symmetric` | 대칭 여부 (T0 함수 적용 가능 여부) |

## 판정

- **$s^{*} \ne n$인 instance가 하나라도 나오면 성공.** 그 instance의
  $\kappa$, $\mu$, 저장·export 설정을 기록하고, 주변 격자를 조밀하게 다시 훑어
  경계를 찾을 것.
- 전 격자에서 $s^{*}=n$이면, $\kappa$–$\mu$ 평면에서 $\max_s v(s)/s - v(n)/n$
  (음수여도)의 등고선을 그려 **가장 근접한 지점**을 보고할 것. 0에 얼마나
  가까이 갔는지가 정보다.

## 하지 말 것

- $\ssum_j \gamma_j(\bar u_j)$의 구성원별 실측. 상한의 상한이고 $n$이 작아
  의미가 없다.
- $(m+1)\bar\gamma / n$ 보고. $n=6$, $m=144$면 $\approx 24\bar\gamma$로 공허하다.
- 비대칭 instance에 T0의 대칭 공식 적용.
- 파라미터를 $\kappa$ 계산 없이 눈대중으로 미는 것.

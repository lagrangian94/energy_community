# Robust core와 DW Owen: robust optimization 판

Sep 24, 2026 · @Seokwoo Kim

> [Claude Docs 문서](https://claude.ai/code/artifact/fd931bec-e11b-40de-a9ad-b2393bf781b0)의 "RO 판" 탭 스냅샷(2026-09-24). 살아 있는 판은 문서다. 본문의 "유한 시나리오 판"(유한 판)은 [robust_core.md](robust_core.md)다.

불확실성이 연속 집합 U 위에 있는 pure RO 판이다. Robust core의 정의, 가정 (A1), robust Owen 명제, 협력 이득의 원리는 유한 시나리오 판 1절을 그대로 쓰고, 여기서는 연속 U에서 달라지는 것만 쓴다. 결론은 둘이다. 고정 Ω의 전역 ball-box를 쓰면 (A1)과 분산 효과가 함께 성립하고(2절), balance 등식을 참여 규칙으로 흡수하면 master가 n과 무관한 LP로 남는다(4절). 수치로 확인한 것은 2절의 ball Monte Carlo, 3절의 budget 반례, 6절의 √n 반례다.

## 1. 유한 시나리오 판에서 달라지는 것

유한 판의 명제와 증명(weak duality 한 줄)은 그대로다. 바뀌는 것은 adversary 집합과 dual의 모양이다.

- **Adversary:** P = Δ(U)다. 이익이 ξ에 선형이면 U 위의 min과 Δ(U) 위의 min이 같으므로 pure RO와 같다.
- **Epigraph row:** “θ ≤ ξ별 커뮤니티 이익” row가 ξ마다 하나씩인 semi-infinite 행이 되고, dual ρ는 U 위의 측도다. 최적 dual은 N의 최악점 ξ\*에 몰린다(4절).
- **(A1):** 연합은 전역 U의 투영 U\_S = proj\_S U를 쓴다. N의 worst case를 S로 자른 것이 U\_S 안에 있으므로 (A1)이 자동으로 성립한다. 연합 의존성은 투영(데이터)에만 있다.
- **분산 효과:** 유한 판에서 밀도비 통제가 하던 역할을 set의 모양(ball)이 한다.
- **Endowment:** 불균형 요건(4절)은 멤버가 가져오는 음의 endowment라 χ\_j에 π\*ᵀb\_j 항이 붙는다.

| 집합과 크기 규칙 | (A1) | 투영 일관성 | 분산 효과 |
| --- | --- | --- | --- |
| Ball-box, 고정 Ω | 성립 | 성립 | 있음 (√\|S\|) |
| Budget, 상수 Γ | 성립 | 깨짐 (소연합 과보호) | \|S\| > Γ인 연합에만 |
| Budget, Γ\_S ∝ √\|S\| | 깨짐 | 깨짐 | 있음 |

## 2. 모델: 전역 U와 투영

모든 멤버 좌표 위에 고정된 U 하나를 두고 연합은 그 투영 U\_S = proj\_S U를 쓴다. P = Δ(U)이고, N의 worst case를 S로 자른 것은 proj\_S U의 원소라 (A1)이 자동으로 성립한다. 고정 Ω ball-box면 (A1), 연합마다 같은 신뢰도, 분산 효과가 함께 성립한다.

```latex
U=\{\xi:\ \|\xi\|_\infty\le1,\ \|L^{-1}\xi\|_2\le\Omega\},\qquad h_U(w_S)=\max_{\xi\in U}\sum_{i\in S}w_i\xi_i=\min_z\big\{\|w_S-z\|_1+\Omega\|L^\top z\|_2\big\}
```

- L은 시간 상관(AR(1))의 Cholesky이고 멤버 간은 독립이다. L = I이면 ball-box이고 h\_U(w\_S) ≤ min(‖w\_S‖₁, Ω‖w\_S‖₂)다.
- Box가 비활성이면 h\_U(w\_S) = Ω‖Lᵀw\_S‖₂라 표준편차를 그대로 따라간다. L = I에서 최악점은 ξ\_i = Ωw\_i/‖w\_S‖₂(Cauchy–Schwarz)이고, w ≡ 1이면 h = min(|S|, Ω√|S|)다. √|S|는 손으로 넣은 것이 아니라 고차원 ball의 모양에서 나온다.
- Ω를 모든 연합에 고정해도 되는 이유: 독립이고 평균 0이며 |ξ\_i| ≤ 1인 교란에 대한 Ben-Tal–Nemirovski 보장에는 차원도 가중치도 없다. ε\_v = 5%이면 Ω ≈ 2.45다.

```latex
\Pr\Big(\sum_{i\in S}w_i\xi_i>\Omega\|w_S\|_2\Big)\le e^{-\Omega^2/2}\quad\forall S,\qquad \Omega=\sqrt{2\ln(1/\varepsilon_v)}
```

**예 (w = (5, 5, 1), Ω = 1, 독립 Uniform\[−1, 1\], Monte Carlo 400만).** 보호량이 ‖w\_S‖₂를 따라가고 위반확률이 연합과 무관하다.

| 연합 | ‖w\_S‖₂ | 보호량 | 위반확률 |
| --- | --- | --- | --- |
| {3} | 1 | 1 | 0 |
| {1,2} | 7.07 | 7.07 | 4.3% |
| N | 7.14 | 7.14 | 4.3% |

**Owen(Euler) 부담.** 오차가 linking row에 들어오면 멤버 j의 부담은 N의 최악점 ξ\*에서 읽는다. L = I이고 box가 비활성이면 r\_j = w\_jξ\*\_j = Ωw\_j²/‖w‖₂로, 분산 기여에 비례한다.

- 예에서 부담은 (3.50, 3.50, 0.14)다. 작은 멤버 3은 혼자일 때의 1 대신 0.14를 낸다.
- 모든 연합에서 r(S) = Ω‖w\_S‖₂²/‖w‖₂ ≤ Ω‖w\_S‖₂라 이탈 이유가 없다. 예: {1,2}는 7.00 ≤ 7.07, {1,3}은 3.64 ≤ 5.10.

**계산.** Static robust counterpart는 linking row마다 SOC 하나(Ω‖·‖₂와 box 항)라 master가 SOCP가 되고 Owen은 conic dual에서 읽는다. LP가 필요하면 Ben-Tal–Nemirovski polyhedral 근사를 쓰고, 근사 set도 전역이라 (A1)은 유지된다. 다만 exposure가 데이터이면 SOC가 RHS 상수로 바뀌어 LP로 남는다(4절).

**원칙.** 가격·공통 날씨처럼 모두가 공유하는 불확실성은 이 parameter set으로 다뤘도 된다. 멤버 소유 독립 불확실성은 전역 ball(이 절)이나 유한 시나리오 판의 DRO로 다룬다. Budget set은 피한다(3절).

## 3. Budget set을 쓸 때

Budget(Bertsimas–Sim)은 신뢰도 보장에 |S|가 들어가서, 연합마다 같은 신뢰도를 주려면 Γ\_S를 바꿔야 하고 그 순간 (A1)이 깨진다. Γ를 고정하면 (A1)은 살지만 신뢰도가 연합마다 달라진다. 둘 다 필요하면 ball(2절)을 쓴다.

```latex
U_S=\{\|\xi_S\|_\infty\le 1,\ \|\xi_S\|_1\le\Gamma_S\},\qquad \Pr(\text{위반})\lesssim e^{-\Gamma_S^2/(2|S|)},\qquad \mathrm{proj}_S\,U_T\subseteq U_S\iff\Gamma_S\ge\min(|S|,\Gamma_T)\ \ \forall S\subseteq T
```

- 마지막 식이 (A1)이고 superadditivity에도 같은 조건이 필요하다. 싱글턴은 Γ\_{j} ≥ 1이면 항상 안전하다.
- 상수 Γ(N의 신뢰도에 맞춘 Γ ≈ √(2n ln(1/ε\_v)), n = 6, 30, 60이면 6.0, 13.4, 18.9)는 안전하지만 소연합을 과보호하고(위반확률 약 exp(−Γ²/(2|S|))), n = 6에서는 Γ ≥ n이라 분산 이득이 없다. 이탈자를 비관해서 산 안정성이고, 소연합이 전체 크기의 set을 상정할 이유가 없어 해석이 약하다.
- Γ\_S = c√|S|는 노출이 이질적이면 정수 변수 없는 볼록 게임에서도 무너진다. 결정 없이 손실만 있는 게임 v(S) = −max\_{ξ∈U\_S} Σ\_{i∈S} w\_iξ\_i로 확인했다(least core는 HiGHS LP).

| 성질 | 결과 | 반례 |
| --- | --- | --- |
| Owen 배분의 안정성 | 깨짐 (core가 있어도) | n = 16, w ≡ 1, c = 2. 균등 배분 −0.5는 core에 든다. 꼭짓점 dual ξ\* = (1×8, 0×8)이 주는 χ는 앞 8명 연합에게 2.34만큼 막힌다(χ(S) = −8, v(S) = −2√8 = −5.66). DW의 simplex dual은 보통 꼭짓점이다 |
| Superadditivity | 깨짐 | n = 3, w = (M, M, 1), c = 1. v(N) − v({1,2}) − v({3}) = 1 − (√3 − √2)M < 0 (M > 3.15) |
| Core가 비어 있지 않음 | 깨짐 | 같은 예, M = 5에서 cost of stability ω\* = 0.589 |
| ε rate O(1/n) | Θ(1/√n)로 느려짐 | 큰 멤버 n/2명(M = 5)과 작은 멤버 n/2명, c = 2. 분할 {큰, 작은}만으로 ε\* ≥ 1.515/√n (n = 16…4096). LP에도 남는 구조적 항이다 |
| 유한 판 3.4(a) H(S,T) ≥ 0 | 깨짐 | 하나의 P를 전제로 한 증명 |
| 개인 합리성, 유한 판 3.5의 p\_j, h\_j ≥ 0 | 유지 | 싱글턴 투영 \[−1, 1\] ⊆ U\_{j} (c ≥ 1) |

원인은 Γ\_S가 adversary의 자원이라는 데 있다. 작은 멤버 3이 들어오면 Γ가 √2에서 √3으로 늘고, adversary는 그 증분을 큰 멤버 1, 2에 쓴다. 노출이 균일하면(w ≡ 1) 비용 min(|S|, c√|S|)가 concave라 convex game이 되고 core가 남는다. 같은 예(w = (5, 5, 1))에서 budget은 위험도 잘못 잰다.

| 연합 | ‖w\_S‖₂ | √\|S\| budget (c = 1): 보호량 / 위반확률 | 고정 Ω = 1 ball: 보호량 / 위반확률 |
| --- | --- | --- | --- |
| {1,2} | 7.07 | 7.07 / 4.3% | 7.07 / 4.3% |
| N | 7.14 | 8.66 / 1.1% | 7.14 / 4.3% |

멤버 3이 들어오면 실제 표준편차는 1% 늘지만 budget 보호량은 22% 늘고 N의 신뢰도만 4배 엄격해진다. 가짜 합병 손실은 여기서 나온다. 같은 신뢰도 해석은 고정 Ω ball이 (A1)과 함께 달성한다.

- Budget을 써도 되는 경우: 노출이 거의 균일하면 √|S| budget도 신뢰도를 고르게 주고 core도 산다. 그래도 DW dual에서 읽은 Owen이 core에 든다는 보장은 없어 대칭 dual을 고르거나 --check-core로 확인해야 한다.
- LP master나 유한 꼭짓점(C&CG의 유한 수렴)이 필요하면 상수 Γ budget이나 ball의 polyhedral 근사를 쓴다. 신뢰도 면에서는 후자가 낫다.

## 4. Balance 등식 흡수와 master LP

실시간 balance 등식을 참여 규칙으로 흡수하면, master는 결정론 master에 흡수 용량 행(시간당 2개, 배터리 에너지까지 4개)을 더한 LP가 되고 pricing은 시나리오 하나 크기의 MILP다. 행 수는 n과 무관하다.

**설정.** 멤버 j의 시간 t 실제 순주입은 y\_{j,t}(ξ) + g\_{j,t} + e\_{j,t}ξ\_{j,t}다. y는 유연 자산(배터리, 전해조, heat pump)의 순주입으로, ξ를 보고 반응하는 recourse다. y⁰는 그 명목값(ξ = 0일 때의 계획, DW column 안의 first-stage 결정)이다. g는 조절할 수 없는 부분(부하, 재생)의 예측이고 e\_{j,t}ξ\_{j,t}는 그 오차다(e는 오차 규모, 데이터). 커뮤니티 순불균형은 I\_t(ξ) = Σ\_j e\_{j,t}ξ\_{j,t}다. 정수 결정은 모두 first-stage(heat pump commitment 포함)이고 가격은 확정이다. 멤버 여유 R^±\_{j,t}는 y가 명목 출력 주변에서 움직일 수 있는 범위이고 private다.

등식 흡수는 네 단계로 유도한다. 클래식 affine decision rule에서 시작해 participation factor로 제한하고, 그 제한이 여유 조건에 대해 손실이 없음을 보인 뒤 β를 소거한다. 아래에서 시간 t는 생략한다.

**① Affine decision rule (Ben-Tal et al. 2004).** y\_j(ξ) = y⁰\_j + Y\_jξ로 두면 balance 등식은 ξ에 대한 항등식이어야 하므로 계수를 맞춘다.

```latex
\sum_j y^0_j+\sum_j g_j=\text{(nominal balance)},\qquad \sum_j Y_{j,i}+e_i=0\ \ \forall i,\qquad h_U(Y_j^\top)\le R^+_j,\ \ h_U(-Y_j^\top)\le R^-_j\ \ \forall j
```

상수항은 결정론 linking rows(시간당 6) 그대로다. ξ 계수 조건은 n개 등식이고, 여유 조건은 결정변수 Y\_j에 대한 SOC다. 이대로면 exposure가 결정에 의존해 master가 SOCP가 되고 ε가 Θ(1/√n)이 된다(6절).

**② Participation factor로 제한 (모델링 선택).** Y\_j를 합산 불균형에만 반응하는 rank-one 형태로 제한한다. 멤버 j는 오차의 출처를 보지 않고 I(ξ)만 보며, 떠안는 몫 β\_j는 출처와 무관하다. AGC participation factor이고, Bienstock–Chertkov–Harnett(2014), Jabr(2013)의 affine policy와 같은 구조다.

```latex
Y_{j,i}=-\beta_j e_i\ \Longrightarrow\ y_j(\xi)=y^0_j-\beta_j I(\xi),\qquad \Big(\sum_j Y_{j,i}+e_i=0\ \forall i\Big)\iff\sum_j\beta_j=1,\qquad h_U(\pm Y_j^\top)=\beta_j\,h_U(e)
```

n개의 계수 조건이 Σβ = 1 하나로 줄고, 여유 조건은 β\_j·h\_U(e) ≤ R^±\_j로 선형이다. h\_U(e)는 데이터라 상수다. 배제되는 것은 “멤버 1의 배터리는 멤버 2의 오차에만 반응한다” 같은 규칙이다.

**③ 제한은 여유 조건에 대해 손실이 없다 (명제).** 대칭 U에서 한 시간·한 carrier의 여유 조건만 보면, 일반 affine rule과 β 규칙과 fully adaptive recourse가 모두 같은 조건 ΣR ≥ h\_U(e)를 준다.

```latex
\sum_jR_j\ \ge\ \sum_j h_U(Y_j^\top)\ \ge\ h_U\Big(\sum_jY_j^\top\Big)=h_U(-e)=h_U(e)
```

둘째 부등호는 support function의 subadditivity다. β 규칙은 Σ\_j β\_j h\_U(e) = h\_U(e)로 이 하한을 정확히 달성한다. Fully adaptive recourse에서도 최악점 ξ\*의 불균형 h\_U(e)를 누군가 메워야 하므로 이 조건은 필요조건이다. 시간에 걸친 제약(배터리 SoC, 아래)이나 멤버별 반응 비용(merit order)이 들어오면 이 동치는 깨질 수 있다.

**④ β 소거 (Fourier–Motzkin).** β는 보조변수라 투영해 없애고, master에는 합산 행만 남긴다.

```latex
\exists\beta\ge0,\ \sum_j\beta_{j,t}=1,\ \beta_{j,t}\,b_t\le R^{\pm}_{j,t}\ \forall j\quad\iff\quad \sum_jR^{\pm}_{j,t}\ \ge\ b_t:=h_U(e_{\cdot,t})
```

(⇒)는 멤버별 조건을 더하면 되고, (⇐)는 β\_j = R\_j/ΣR이 witness다. 즉 pro rata β는 모델링 가정이 아니라 이 존재 증명의 해다.

- 남는 행은 시간당 2개(상·하향)이고 기존 reserve 행(r\_sym ≤ Σ r⁺)과 모양이 같다. TSO에 파는 reserve와 함께 쓰면 r^±\_{j,t} + R^±\_{j,t} ≤ 여유이고, 유연성이 없는 멤버는 R = 0이다.
- 등식을 하루 전으로 옮기는 것이 아니다. 하루 전에 정하는 것은 명목 schedule뿐이고 실시간 편차는 커뮤니티 안의 유연 자산이 메운다(내부 netting 유지).
- 정보 면에서 멤버는 I(ξ) 신호 하나만 알면 된다. 일반 Y는 멤버 간 오차 전체 공유가 필요하고, 자기 오차에만 반응하는 대각 Y는 아래 표의 self-balancing이라 pooling이 없다.

| 방식 | 규칙 | 결합 행 | 필요한 여유 합 |
| --- | --- | --- | --- |
| Self-balancing (= stand-alone) | y\_j = y⁰\_j − e\_jξ\_j | 없음 (각자 R\_j ≥ e\_j) | Σ\_j e\_j (n에 비례) |
| Pooling (기본) | y\_j = y⁰\_j − β\_j I | 여유 행 하나 | h\_U(e) ≈ Ω√n·e |
| 혼합 | y\_j = y⁰\_j − α\_je\_jξ\_j − β\_j I | 남은 불균형 기준 | 중간 (비교에서 제외) |

싱글턴의 투영은 box라서 self-balancing이 곧 stand-alone이다. 두 방식의 차이 Σ\_j e\_j − h\_U(e)가 여유 공유의 협력 이득이고 n과 함께 커진다. 멤버별 배터리면 한 멤버가 요건(e\_j)과 여유(R\_j) 양쪽에 선다.

**Master와 pricing.** Column q는 멤버 j의 명목 plan이고 명목 이익 w, linking 기여 a, 여유 R^±를 갖는다.

```latex
\begin{aligned}
\max\ &\sum_{j,q}w_j^q\lambda_{jq}+c_0^\top x_0\quad\text{s.t.}\quad \sum_{j,q}a_j^q\lambda_{jq}+A_0x_0\le 0\ [\theta],\quad \sum_{j,q}R^{\pm,q}_{j,t}\lambda_{jq}+R^{\pm}_{0,t}\ge b_t\ [\eta^{\pm}_t],\quad \sum_q\lambda_{jq}=1\ [\sigma_j]\\
\mathrm{rc}_j&=\max_{x_j\in X_j}\Big\{w_j-\theta^\top a_j+\sum_t\big(\eta^+_tR^+_{j,t}+\eta^-_tR^-_{j,t}\big)\Big\}-\sigma_j
\end{aligned}
```

- e가 데이터라 b\_t는 상수다. 타원은 RHS에만 들어가고 master에 SOC가 없다.
- R\_0는 grid를 최후의 balancing 제공자로 두는 공유 변수(비용과 연결 용량 한도)라 master가 항상 feasible하다.
- Pricing에는 ξ가 없다. 여유 R은 reserve의 r±처럼 가격 η를 받으므로 기존 결정론 DW(|Ω| = 1)에 행 두 종류만 추가하면 된다.
- 목적함수는 확정이다. 반응 에너지를 커뮤니티 내부 가격으로 정산하면 내부 이전은 상쇄되고 grid 흐름은 명목에서 변하지 않는다는 가정이다.

**Owen 배분과 안정성.** 요건 RHS b\_{S,t} = h\_{U\_S}(e\_{S,t})는 연합마다 다르고 가법적이지 않다. 그러나 semi-infinite로 쓰면 ξ마다 멤버별로 가법적이고 U는 전역이라 (A1)이 성립하며, dual은 N의 최악점 ξ\*에 몰린다.

```latex
\sum_{j\in S}R^{+}_{j,t}\ \ge\ \sum_{j\in S}e_{j,t}\,\xi_{j,t}\ \ \forall\xi\in U,\qquad \chi_j=\sigma_j^*-\sum_t\big(\eta^{+*}_t+\eta^{-*}_t\big)s_{j,t},\qquad s_{j,t}=e_{j,t}\,\xi^*_{j,t}\ \big(=\Omega e_{j,t}^2/\|e_{\cdot,t}\|_2\ \text{if }L=I\big)
```

- 원고는 b\_j = 0이라 Owen이 σ\*뿐이었다. 여기서는 멤버가 음의 endowment −s\_{j,t}(불균형 요건)를 가져오므로 그 몫이 빠진다. 대칭 U에서는 하향 행의 최악점이 −ξ\*라 몫이 같다.
- 안정성: ξ\*를 S로 자른 것은 proj\_S U에 속하므로 Σ\_S s ≤ b\_S이고(요건 몫은 비용게임 c(S) = h\_{U\_S}(e\_S)의 core), weak duality로 Σ\_S χ ≥ v(S)다. 효율성 초과분은 원고와 같은 duality gap뿐이다.
- 해석: 멤버 j의 순수령액은 여유 제공 수입(σ\* 안의 η\*R\_j)에서 불균형 요건 몫을 뺀 것이다. 오차는 작고 배터리가 큰 멤버는 순수령, 반대는 순지불이고, 유연성 없는 residential 멤버도 자기 분산 기여만큼 낸다.

**배터리 에너지(SoC).** 반응이 시간에 걸쳐 누적되므로 에너지 요건이 시간당 2행 추가된다(RHS 상수, LP 유지). AR(1) 양의 상관이면 누적 불균형이 빨리 커져 L이 중요해진다.

```latex
\mathrm{SoC}_{j,t}(\xi)=\mathrm{SoC}^0_{j,t}-\beta_j\sum_{\tau\le t}I_\tau(\xi),\qquad \sum_jE^{\pm}_{j,t}\ \ge\ h_U\Big(\sum_{\tau\le t}e_{\cdot,\tau}\Big)\quad\forall t
```

출력 비율과 에너지 비율을 같은 β로 쓰려면 멤버들의 R/E 비율이 맞아야 한다. 맞지 않으면 β를 master 변수로 두어야 해서 bilinear가 된다. 보수적 대안은 둘 중 빡빡한 쪽 비율로 반응하는 것이고, β가 시간마다 바뀔 때의 누적도 정리가 필요하다.

| 항목 | 유한 판 (DRO) | RO 판 (이 모델) |
| --- | --- | --- |
| Recourse | 시나리오별 복사, 정수 recourse 가능 | 비례 참여 규칙, 정수는 first-stage |
| Master | LP, 시간당 6·\|Ω\|행 (+cut) | LP, 시간당 8–10행 |
| Pricing | scenario-expanded MILP | 시나리오 하나 크기 MILP |
| 실시간 netting | 시나리오별 balance | 흡수 여유 공유 |
| 분산 효과 | ρ̂의 독립성 | 타원 모양 |
| 배분 | 시나리오별 (S1) | 확정: 여유 수입 − 요건 몫 |

**가정과 한계.**

- 비례 배분이라 더 싼 멤버부터 부르는 merit order는 표현하지 못한다.
- 반응 에너지 비용을 목적함수에 넣으면 exposure가 결정에 의존해 master가 SOCP가 되고 ε가 Θ(1/√n)으로 느려진다(6절).
- 가격 불확실성을 넣으면 가격 오차 × 반응량이 되어 ξ에 대해 2차가 되고, S-lemma를 거쳐 SDP가 필요하다.
- 재생 출력은 실시간 출력 제한 없이 그대로 나온다고 가정했다. Private 제약 안의 불확실성은 5절에서 다룬다.
- Grid(R\_0)의 연결 용량과 peak 과금의 상호작용은 아직 정리하지 않았다.

## 5. Private 불확실성과 ADR

Private 제약 안의 불확실성은 볼록화 뒤 계수로 옮겨 가고 hull이 ξ에 대해 불연속이라, 표준 RO 트릭이 깨진다. 불확실성이 가격이나 linking RHS처럼 master 쪽에만 있으면 column이 ξ와 무관해 문제가 없다(4절이 이 경우다).

예: 최소 부하 m, 자기 풍력 ξ로만 도는 전해조.

```latex
X(\xi)=\{u\in\{0,1\},\ m\,u\le y\le\bar Yu,\ y\le\xi\},\qquad \mathrm{conv}\,X(\xi)=\begin{cases}\{0\le u\le1,\ m\,u\le y\le\min(\bar Y,\xi)\,u\}&\xi\ge m\\\{(0,0)\}&\xi<m\end{cases}
```

- (a) RHS의 ξ가 hull에서는 u에 곱해진다(y ≤ ξ·u). DW가 LP relaxation보다 좋은 bound를 주는 바로 그 이유로 ξ-의존성이 affine이 아니게 되어, inner max를 dualize해 한 단계로 만드는 트릭을 쓸 수 없다.
- (b) Hull이 ξ = m에서 점프한다. 이익이 p·y − c·u면 LR recourse 가치 Q(ξ)가 점프하고 concave도 연속도 아니어서, 최악의 ξ는 m 바로 아래의 내부점일 수 있고 극점 탐색이 통하지 않는다.
- (c) Column이 정책 y\_j(·)이 된다. 볼록화 대상은 ξ별 conv X\_j(ξ)의 곱이 아니라 정책 집합의 hull이고(first-stage 정수가 ξ들을 묶음), 연속 U에서는 무한차원이다. C&CG(Zeng–Zhao)로 ξ를 추가하면 기존 column이 무효가 되고(relatively complete recourse 필요), 루프는 C&CG, adversary, DW의 세 겹이다. 정수 recourse가 있으면 nested C&CG가 필요하다.
- 쉬운 경우에도 한계가 있다. Recourse가 LP면 recourse 가치가 ξ에 concave라 최악점은 U의 극점이지만, ball의 극점은 무한히 많아 열거할 수 없다. C&CG의 유한 수렴이 필요하면 polyhedral 근사를 쓴다. “Active support가 유한하다”는 LP recourse에서만 성립한다.
- Static robust에서 분산 효과는 합산되는 row에서만 생긴다. Private 제약은 자기 좌표의 box \[−1, 1\]만 보므로 각자의 최악을 맞는다. 분산 효과를 얻으려면 private 불확실성이 recourse를 통해 linking row로 전파되어야 한다.

**절충안: ADR.** y\_j(ξ) = y\_j⁰ + Y\_jξ로 두면 column (first-stage 정수, y\_j⁰, Y\_j)은 다시 유한 차원이고 ξ와 무관하다. Private 제약은 자기 좌표의 box 위에서만 보면 돼 선형이고 pricing은 MILP로 남는다. Linking row는 ball 위의 robust counterpart라 SOC가 되고(또는 polyhedral 근사), Owen은 conic dual에서 읽는다.

```latex
\sum_{j,p}\lambda_{jp}a_{jp}^0+\Omega\,\Big\|\sum_{j,p}\lambda_{jp}Y_{jp}-B\Big\|_2+(\text{box 항})\ \le\ b^0
```

대가는 세 가지다. 완전 적응보다 보수적이고, 정수 recourse는 first-stage로 올려야 하며, exposure Y\_j가 결정에 의존해 ε가 Θ(1/√n)으로 느려진다(6절).

## 6. ε bound

Rate는 exposure(멤버가 robust 항에 기여하는 양)가 결정에 의존하는지에 달렸다. 데이터면 O(1/n) 그대로이고, 결정에 의존하면 Θ(1/√n)로 느려지며 이 차수는 타이트하다.

SF가 세는 것은 robust 항이 의존하는 멤버 합산 벡터의 차원이다. Ball 위의 robust 항은 h\_U(d)이고 d = (d\_j)의 좌표 j에는 멤버 j만 기여하므로 n차원이다.

| Exposure | SF 차원 | ε rate |
| --- | --- | --- |
| 결정과 무관 (예측오차가 balance에 가법적, 4절) | m+1 | O(1/n) |
| 공통 불확실성 k차원 (가격, 공통 날씨) | m + k·(robust row 수) | O(1/n) |
| 결정에 의존 (commitment가 exposure를 켬, ADR의 Y\_j) | 사실상 n | Θ(1/√n) |

**반례 (결정 의존 exposure, 수치 확인).** 멤버 j는 u\_j ∈ {0,1}(커밋하면 exposure w·u\_j)와 y\_j ∈ \[0, Y\]를 고른다. Linking row는 Σu\_j ≤ κn(슬롯)과 Σy\_j + Ωw‖u‖₂ ≤ bn(용량)이고 목적함수는 pΣu + qΣy다.

- MIP는 κn명이 커밋해 여유분 Ωw√(κn)을 남긴다. DW는 모든 멤버가 u\_j = κ로 나눠 커밋해 여유분이 Ωwκ√n으로 줄어든다. 실제로는 없는 “분수 분산”이다.
- Gap = qΩw√n(√κ − κ) = Θ(√n)이라 ε = Θ(1/√n)이다. κ = 1/4, Ω = 2, w = q = 1에서 n = 16, 64, 256, 1024, 4096의 gap은 2, 4, 8, 16, 32로 닫힌 형태와 같다(relaxation은 Gurobi, MIP는 대칭성, n = 16에서 Gurobi MIP로 교차 확인).
- 이 게임의 MIP core는 비어 있지 않다(대칭이고 1인당 가치가 |S|에 증가). 잃는 것은 Owen 방법의 효율이고 안정성은 (A1) 덕에 그대로다.

**상한 유도 (확인 필요).** DW 해의 linear row와 목적함수에만 SF를 쓰면 분수 멤버는 m+1명 이하다. 대신 모든 멤버의 exposure가 바뀔 수 있는데, h\_U는 subadditive이고 h\_U(Δ) ≤ Ω‖Δ‖₂라 robust 항의 증가는 ΩR√n 이하다(R은 멤버 한 명의 exposure 범위). 위반을 단가 π̄ 이하로 메울 수 있으면(eq:bnd\_trade와 같은 가정) 아래가 되고, N\_R은 robust row 수다. 반례가 √n 차수가 타이트함을 보인다.

```latex
\omega^{\mathrm{LR,rob}}\le(m+1)\,\bar\gamma^{\mathrm{rob}}+N_R\,\bar\pi\,\Omega R\sqrt n,\qquad \varepsilon^{\mathrm{LR,rob}}=O\Big(\frac{m}{n}\Big)+O\Big(\frac{N_R\,\bar\pi\,\Omega R}{\sqrt n}\Big)
```

- 4절의 모델은 exposure가 데이터라 첫 행이고 원고의 rate O(|T|/n)이 그대로다. γ̄에는 reserve처럼 여유 부족분 항이 들어간다.
- Private 불확실성의 분산 효과를 얻으려고 ADR(5절)을 쓰거나 반응 에너지 비용을 넣으면 셋째 행이 된다. 그래도 1인당 hedging 이득은 Θ(1)이라 유한 판 3.4(b)의 메시지 방향은 유지된다.

## 7. 논문 블록 (RO 판을 붙일 때)

유한 판의 블록(유한 시나리오 판 3절)에 아래를 더한다. 아직 확정한 것은 없다.

- **ε bound:** “결정 무관 exposure면 원고 rate O(|T|/n)이 유지된다”를 명제로, 결정 의존 exposure의 √n 반례를 remark나 example로 둔다(6절).
- **Hedging 이득:** 유한 판 3.4의 명제는 (A1)만 가정하므로 고정 Ω ball에서도 성립하고, √|S| budget에서는 성립하지 않는다(3절).

## 8. 열린 질문과 다음 단계

1. 배터리: 출력 비율과 에너지 비율이 다를 때의 반응 규칙(4절).
2. 연속 U에서 private 불확실성을 다룰 때 adversary 문제의 tractability(5절).
3. 결정 의존 exposure의 ε 상한 유도(6절).

- [ ] 시제품: 결정론 DW에 흡수 용량 행을 더해 유한 판과 가치·반복 수를 비교한다.

## 9. 참고문헌

기억에 의존한 목록이다. 인용 전에 서지를 확인해야 한다. 공통 문헌(Owen, Geoffrion, Sion 등)은 유한 판 5절에 있다.

- Bertsimas & Sim (2004), The price of robustness, OR.
- Ben-Tal & Nemirovski (2000), Robust solutions of LPs contaminated with uncertain data, Math. Programming. Ben-Tal, El Ghaoui & Nemirovski (2009), Robust Optimization.
- Ben-Tal, Goryashko, Guslitzer & Nemirovski (2004), Adjustable robust solutions of uncertain linear programs, Math. Programming.
- Zeng & Zhao (2013), C&CG for two-stage robust optimization, OR Letters. Zhao & Zeng (2012), 정수 recourse의 nested C&CG (서지 확인 필요).
- Bienstock, Chertkov & Harnett (2014), Chance-constrained optimal power flow, SIAM Review (서지 확인 필요).
- Jabr (2013), Adjustable robust OPF with renewable energy, IEEE Trans. Power Systems (서지 확인 필요).

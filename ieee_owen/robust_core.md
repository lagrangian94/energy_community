# Robust core와 Dantzig-Wolfe Owen 확장

Sep 23, 2026 · @Seokwoo Kim

Ambiguity set P가 연합과 무관하게 고정되어 있으면, grand coalition DW master 하나의 dual (ρ\*, π\*, μ\*)에서 읽은 Owen 배분이 robust core에 들어간다. 효율성 손실은 robust integrality gap이고, 핵심은 epigraph row도 linking row로 보고 함께 dualize하는 것이다. 아래는 손으로 따라간 논증이며 코드로 검증하지 않았다.

## 1. 설정과 기호

Stochastic extension의 구조를 그대로 두고, 확률 ρ̂ 하나 대신 확률들의 집합 P를 쓴다.

- 멤버 N = {1, …, n}, 연합 S ⊆ N.
- 시나리오 집합 Ω는 유한하다. stochastic\_extension.py의 시나리오를 그대로 쓴다.
- X\_j: 멤버 j의 scenario-expanded, non-anticipative 운영집합(MILP, bounded). 논문 eq:Xmip 형태이고 Remark stoch와 같다.
- f\_j^ω(x\_j): 시나리오 ω에서 멤버 j의 이익. x\_j에 대해 선형이다.
- Linking rows: 시나리오마다 Σ\_{j∈S} A\_j^ω x\_j ≤ b\_S^ω (carrier balance, reserve, peak). RHS는 가법적이어서 b\_S^ω = Σ\_{j∈S} b\_j^ω. 현재 코드에서는 nonflex load가 column 안에 있어 b\_j = 0이다.
- 공유 master 변수 x0 (r\_sym, p^ω)는 표기에서 생략한다. 5절에서 따로 다룬다.

Ambiguity set은 확률 simplex 안의 공집합이 아닌 polytope이다.

```latex
P=\{\rho\in\Delta(\Omega):\ G\rho\ge h\},\qquad F_S(x,\rho)=\sum_{\omega\in\Omega}\rho_\omega\sum_{j\in S}f_j^\omega(x_j)
```

| P의 선택 | 의미 |
| --- | --- |
| {ρ̂} | 현재 stochastic extension 그대로 |
| Δ(Ω) | 최악 시나리오 하나에 대한 robust |
| CVaR\_β: 0 ≤ ρ\_ω ≤ ρ̂\_ω/(1−β) | β=0이면 stochastic, β→1이면 worst-scenario |
| TV ball: ‖ρ − ρ̂‖₁ ≤ r | r로 보수성 조절 |

고정된 ρ에 대한 stochastic game은 v\_ρ(S) = max\_{x∈X\_S} F\_S(x, ρ)이다. 여기서 X\_S는 x\_j ∈ X\_j와 S의 linking rows를 만족하는 plan의 집합이다.

## 2. Robust core의 정의

채택할 정의는 “각 연합이 자기 worst case로 평가된다”는 정의 1이다. 정의 2는 거의 항상 비어 있고, 정의 3은 열린 문제로 남긴다.

먼저 연합의 robust 가치를 정의한다. 연합이 plan(first-stage와 contingent recourse)을 먼저 정하고, 자연이 그 연합에게 가장 불리한 분포를 고른다.

```latex
v^{\mathrm{MIP,rob}}(S)=\max_{x\in X_S}\ \min_{\rho\in P}\ F_S(x,\rho)
```

**정의 1 (Robust core, 채택).** 다음을 만족하는 확정 배분 χ ∈ R^n의 집합이다.

```latex
\mathcal{C}^{\mathrm{rob}}=\Big\{\chi\in\mathbb{R}^n:\ \chi(N)=v^{\mathrm{MIP,rob}}(N),\ \ \chi(S)\ge v^{\mathrm{MIP,rob}}(S)\ \ \forall S\subseteq N\Big\}
```

- 배분은 ex-ante 확정 금액이다.
- 이탈하는 연합도, 커뮤니티 자신도 각자의 worst case로 평가된다. 비관의 기준이 양쪽에 일관된다.
- 연합 S의 worst case ρ\*\_S는 N의 worst case ρ\*\_N과 다를 수 있다.
- P = {ρ̂}이면 Remark stoch의 ex-ante core와 같다.
- 논문의 weak ε-core는 이 게임에 그대로 적용한다.

**정의 2 (Uniform robust core, 기각).** 연합은 어떤 ρ에서든 이탈할 수 있다고 본다. 즉 χ(S) ≥ max\_{ρ∈P} v\_ρ(S)를 요구한다. S = N에 넣으면 v^rob(N) ≥ max\_ρ v\_ρ(N) ≥ v^rob(N)이 필요하다. 따라서 P가 사실상 한 점이 아니면 비어 있다. 커뮤니티는 비관하고 이탈자는 낙관하는 비대칭이 원인이다.

**정의 3 (Contingent robust core, 열린 문제).** 시나리오별 배분 χ\_j(ω)를 허용한다. Stochastic Algorithm S1의 x\_j\*(ω)에 대응한다.

```latex
\sum_{j\in N}\chi_j(\omega)=v(N,\omega)\ \ \forall\omega,\qquad \min_{\rho\in P}\sum_{\omega}\rho_\omega\,\chi(S,\omega)\ \ge\ v^{\mathrm{MIP,rob}}(S)\ \ \forall S
```

첫 식은 ex-post budget balance이고, 둘째 식은 각 연합이 자기 배분 흐름을 자기 worst case로 평가한다는 뜻이다. 상수 배분은 ex-post balance를 깨므로 정의 1과 정의 3은 실제로 다른 개념이다. 상태는 7절에 있다.

## 3. Lagrangian relaxation과 minimax swap

Epigraph row를 linking row와 함께 dualize하면 v^LR,rob(S) = min\_{ρ∈P} v^LR\_ρ(S)가 min 두 개를 중첩한 것으로 바로 나온다. 스왑은 Geoffrion 정리 안에 들어 있다.

**Epigraph 형태.** 안쪽 min\_ρ는 LP이므로 그 dual로 바꾼다. P가 비어 있지 않으면 다음과 같다.

```latex
\min_{\rho\in P}\sum_\omega\rho_\omega f^\omega=\max_{\theta,\ \alpha\ge 0}\Big\{\theta+h^\top\alpha:\ \theta+(G^\top\alpha)_\omega\le f^\omega\ \ \forall\omega\Big\}
```

따라서 robust MILP는 하나의 최대화 문제가 된다.

```latex
\begin{aligned}
v^{\mathrm{MIP,rob}}(S)=\max\ &\theta+h^\top\alpha\\
\text{s.t. }&\theta+(G^\top\alpha)_\omega\le\sum_{j\in S}f_j^\omega(x_j)\quad\forall\omega\qquad[\rho_\omega\ge 0]\\
&\sum_{j\in S}A_j^\omega x_j\le b_S^\omega\quad\forall\omega\qquad[\pi^\omega\ge 0]\\
&x_j\in X_j,\ \ \alpha\ge 0
\end{aligned}
```

**관찰.** Epigraph row도 멤버들을 묶는 row다. 그러니 linking row의 일종이다.

**두 row family를 함께 dualize.** θ에 대한 sup이 유한하려면 Σρ\_ω = 1이어야 한다. α ≥ 0에 대한 sup이 유한하려면 Gρ ≥ h여야 한다. 그래서 multiplier ρ는 자동으로 P 안에 들어간다.

```latex
v^{\mathrm{LR,rob}}(S)=\min_{\rho\in P,\ \pi\ge 0}\ \sum_{j\in S}\Big[\phi_j(\rho,\pi)+\sum_\omega\pi^{\omega\top}b_j^\omega\Big],\qquad \phi_j(\rho,\pi)=\max_{x_j\in X_j}\sum_\omega\Big(\rho_\omega f_j^\omega(x_j)-\pi^{\omega\top}A_j^\omega x_j\Big)
```

목적함수는 멤버별로 separable하다. φ\_j는 지금의 stochastic pricing MILP에서 ρ̂을 ρ로 바꾼 것이다.

**보제 1 (중첩 min).** 다음이 성립한다.

```latex
v^{\mathrm{LR,rob}}(S)=\min_{\rho\in P}\ \underbrace{\min_{\pi\ge 0}\sum_{j\in S}\Big[\phi_j(\rho,\pi)+\sum_\omega\pi^{\omega\top}b_j^\omega\Big]}_{=\ v^{\mathrm{LR}}_\rho(S)}
```

안쪽 min\_π는 확률 ρ를 쓴 stochastic Lagrangian dual, 즉 Remark stoch의 v^LR이다. 스왑이 필요 없다.

**보제 2 (primal 표현).** C\_S를 x\_j ∈ conv X\_j와 S의 linking rows로 정의한다. DW master의 feasible set이다.

```latex
v^{\mathrm{LR,rob}}(S)=\max_{x\in C_S}\ \min_{\rho\in P}F_S(x,\rho)=\min_{\rho\in P}\ \max_{x\in C_S}F_S(x,\rho)
```

- 첫 등호는 Geoffrion이다. 목적함수와 dualize한 row가 선형이면 Lagrangian dual은 남긴 제약을 convex hull로 바꾼 LP와 같다.
- 둘째 등호는 Sion(또는 von Neumann)이다. F\_S는 bilinear이고 C\_S와 P는 compact convex polytope이다.
- 결과적으로 X\_j ⊆ conv X\_j이므로 v^MIP,rob(S) ≤ v^LR,rob(S)다.

**MILP에서는 스왑이 안 된다.** X\_S가 비볼록이면 일반적으로 부등호만 남는다.

```latex
\max_{x\in X_S}\min_{\rho\in P}F_S\ \le\ \min_{\rho\in P}\max_{x\in X_S}F_S=\min_{\rho\in P}v^{\mathrm{MIP}}_\rho(S)
```

즉 robust MILP에는 integrality gap 외에 minimax gap이 하나 더 있다. DW 볼록화는 두 gap을 한꺼번에 닫는다. 논문의 duality-based 관점이 robust에서 더 필수적이라는 스토리가 여기서 나온다.

**함정.** Linking row만 dualize하고 epigraph를 row로 풀지 않으면 min\_ρ가 안쪽에 남는다. 그러면 멤버가 결합되어 pricing이 멤버별로 쪼개지지 않는다.

## 4. 주정리: robust Owen 배분

Grand coalition의 minimizer (ρ\*, π\*)는 모든 연합의 dual에서 feasible하다. 그래서 weak duality 한 줄로 core 조건이 나온다.

**가정.**

- (A1) P는 연합과 무관한, 비어 있지 않은 polytope다.
- (A2) X\_j는 bounded이고, f\_j^ω와 A\_j^ω x\_j는 x\_j에 대해 선형이다.

**명제 (Robust Owen).** (ρ\*, π\*)를 S = N에 대한 보제 1의 minimizer라 하자. 즉 grand coalition DW master의 optimal dual이다. 배분을 다음과 같이 정의한다.

```latex
\chi_j=\phi_j(\rho^*,\pi^*)+\sum_\omega\pi^{*\omega\top}b_j^\omega
```

그러면 다음이 성립한다.

1. Σ\_{j∈N} χ\_j = v^LR,rob(N).
2. 모든 S에 대해 Σ\_{j∈S} χ\_j ≥ v^LR,rob(S) ≥ v^MIP,rob(S).
3. 논문의 weak ε 배분 x\_j = χ\_j − g/n은 정의 1의 ε-core에 든다. 여기서 g = v^LR,rob(N) − v^MIP,rob(N), 즉 robust integrality gap이고 ε = g/n이다.

**증명.**

- 1번은 minimizer의 정의 그대로다.
- 2번: (A1)에 의해 S의 dual feasible region P × R₊는 N의 것과 같다. 따라서 (ρ\*, π\*)는 S의 min에서도 feasible하다. 목적함수가 separable하니 그 점에서의 값은 Σ\_{j∈S} χ\_j이다. min은 어떤 feasible 점의 값보다도 작다. 마지막 부등호는 보제 2다.
- 3번은 논문의 deterministic 증명과 같다.

증명에는 스왑이 쓰이지 않는다. 보제 2는 v^LR,rob이 “볼록화된 커뮤니티의 robust 가치”라는 해석과, MIP 가치를 위에서 누른다는 사실에만 필요하다.

**Remark 1 (멤버 소유 불확실성의 진입과 이탈).** 멤버가 들어오면 그 멤버의 불확실성이 생기고, 나가면 빠진다. 이는 모델에 이미 들어 있고 (A1)을 깨지 않는다.

- 시나리오 ω는 모든 멤버 불확실성의 결합 실현값이다. Ω와 P는 이 결합 공간 위에 한 번 정의되고 모든 연합이 공유한다.
- 연합 S의 문제에는 S 멤버의 f\_j^ω, A\_j^ω, b\_j^ω만 등장한다. 예를 들어 residential 멤버 r이 S에 없으면 r의 수요 오차 좌표는 S의 문제 어디에도 없다.
- 따라서 S가 실제로 마주하는 것은 P의 S-marginal이다. 유효 ambiguity set은 연합마다 다르다.
- 그래도 증명은 그대로다. 필요한 것은 ρ\*\_N이 S의 문제에서 feasible하다는 것뿐이고, ρ\*\_N은 결합 분포로서 P에 속한다. S는 그 marginal로 평가할 뿐이다.
- Remark stoch도 같은 방식으로 처리되고 있었다.

구분할 것은 연합 의존성이 어디에 있느냐이다. 데이터(marginal)에 있으면 허용된다. adversary의 힘(P 자체)에 있으면 (A1)이 깨진다. 후자는 6절에서 다룬다.

**해석.**

- χ\_j는 커뮤니티의 worst-case 분포 ρ*와 가격 π* 아래에서 평가한 멤버 j의 가치다. ρ\*는 내생적으로 정해지는 pricing measure다.
- 커뮤니티에 나쁜 시나리오(ρ\*가 크게 잡힌 ω)에서 돈을 버는 멤버가 더 받는다. 보험료로 해석할 수 있다.
- P = {ρ̂}이면 Remark stoch와 정확히 같다. P = Δ(Ω)이면 ρ\*는 N의 최악 시나리오들 위에 모인다.
- Stand-alone κ\_j = v^MIP,rob({j})는 멤버 자신의 worst case라 훨씬 비관적이다. 그래서 협력 이득에 자원 pooling 외에 hedging(분산) 가치가 더해진다. eq:v0의 Γ^MIP 정규화도 이 κ\_j로 한다.

**ε bound.** Shapley-Folkman에서 세는 linking row 수가 epigraph row |Ω|개만큼 늘어난다.

```latex
m'=(2|K|+3)\,|T|\,|\Omega|+|\Omega|
```

O(1/n) rate는 그대로고 상수만 조금 커진다. γ̄는 확인이 필요하다. 목적함수가 고정 ρ̂이 아니라 P 위의 ρ로 들어오므로, eq:gammatilde의 bound가 max\_{ρ∈P}로 바뀔 가능성이 크다.

## 5. Dantzig-Wolfe 계산

Master에 θ, α column과 ω-row |Ω|개가 추가되고, pricing은 같은 MILP에서 ρ̂만 ρ\*로 바뀐다. Reserve의 r\_sym이 이미 “max-min over t”를 master epigraph 변수로 처리하는데, θ는 같은 장치를 ω에 쓴 것이다.

**Master (S = N).** Column p는 멤버 j의 contingent plan이고, 시나리오별 이익 f\_jp^ω와 linking 계수 a\_jp^ω를 가진다.

```latex
\begin{aligned}
\max\ &\theta+h^\top\alpha\\
\text{s.t. }&\theta+(G^\top\alpha)_\omega-\sum_j\sum_p\lambda_{jp}f_{jp}^\omega\le 0&&[\rho_\omega]\quad\forall\omega\\
&\sum_j\sum_p\lambda_{jp}a_{jp}^\omega\le b_N^\omega&&[\pi^\omega]\quad\forall\omega\\
&\sum_p\lambda_{jp}=1&&[\mu_j]\quad\forall j\\
&\lambda\ge 0,\ \alpha\ge 0,\ \theta\ \text{free}
\end{aligned}
```

- ω-row의 dual이 worst-case 분포 ρ*다. θ column에서 Σρ = 1이, α column에서 Gρ ≥ h가 나오므로 ρ*는 자동으로 P 안에 있다.
- Linking row는 ρ로 scale하지 않는다. π\*^ω에 확률 가중이 이미 들어 있다.
- 종료 시점에 μ\_j = φ\_j(ρ\*, π\*)이므로 χ\_j = μ\_j + π\*ᵀb\_j이다. 코드에서는 b\_j = 0이라 χ\_j = μ\_j이고, 현재의 owen = −sigma와 같은 모양이다.

**Pricing (멤버 j).** 현재 PlayerPricing과 같은 MILP이고 가중치만 다르다.

```latex
\max_{x_j\in X_j}\ \sum_\omega\rho^*_\omega f_j^\omega(x_j)-\sum_\omega\pi^{*\omega\top}A_j^\omega x_j-\mu_j
```

**공유 변수 x0.** r\_sym의 비용은 first-stage라 모든 ω에 같은 값으로 들어간다. Σρ = 1이므로 목적함수에 직접 두어도 같다. p^ω처럼 시나리오별 비용은 해당 ω-row에 넣는다. 이 변수들의 dual 제약은 연합과 무관하므로 (A1)을 깨지 않는다.

**코드 변경점 (ieee\_owen/stochastic\_extension.py).**

1. DirectMaster: θ, α column과 ω-row를 추가한다. Column의 ω-row 계수는 Column.scen\[w\] + Column.first(비용 convention, unscaled)이다. 목적함수의 ρ̂-scaled cost는 빼야 한다.
2. PlayerPricing: base cost가 생성 시점에 stack.scaled\_cost로 ρ̂에 고정되어 있다. 호출마다 ρ를 받도록 바꾼다.
3. scenario\_allocation (S1): pen\[w\]가 dual을 probs\[w\]로 나눈다. ρ\*\_ω = 0인 시나리오가 생기면 0으로 나누게 되므로, 나누지 않는 형태로 다시 쓴다.
4. Stabilization(Wentges smoothing, du Merle penalty)은 그대로 적용된다. ρ는 dual vector의 일부일 뿐이다.
5. 검증: CVaR β = 0(P = {ρ̂})에서 현재 v^CHP와 Owen 배분이 소수점 넷째 자리까지 같아야 한다.

## 6. 성립 조건과 깨지는 지점

모델링에서 가장 중요한 선택은 (A1), 즉 uncertainty set이 연합에 따라 달라지는가이다. 증명에 실제로 필요한 것은 “grand coalition의 worst case가 모든 S에서도 허용된다”는 것뿐이다.

```latex
\rho^*_N\in P_S\quad\forall S\qquad(\text{충분조건: } P_N\subseteq P_S\ \ \forall S)
```

### 6.1 Budget 방식: 분산 효과와 (A1)의 충돌

Budget으로 짜면 (A1)과 분산 효과 중 하나를 골라야 한다. 멤버별 forecast 오차 ζ\_i ∈ \[−1, 1\]에 Σ\_{i∈S} |ζ\_i| ≤ Γ\_S를 건다고 하자. N의 worst case를 S로 제한하면 budget을 최대 min(|S|, Γ\_N)까지 쓴다. 따라서 다음이 성립한다.

```latex
(\mathrm{A1})\iff\Gamma_S\ge\min(|S|,\Gamma_N)\quad\forall S\subseteq N
```

| 모델 | (A1) | 문제점 |
| --- | --- | --- |
| Rectangular (×\_j U\_j, Γ\_S = \|S\|) | 성립 | 멤버 간 분산 효과가 없다 |
| 공통 budget (Γ\_S = min(\|S\|, Γ)) | 성립 | 멤버가 빠지면 adversary가 남은 멤버에게 budget을 몰아준다. 작은 연합일수록 과하게 비관적이다 |
| 통계적 budget (Γ\_S ∝ √\|S\|) | 깨짐 | 독립 오차에 가장 자연스러운데 Owen 논증이 끊긴다 |

Residential 수요처럼 서로 독립인 오차에 가장 자연스러운 모델이 바로 (A1)을 깨는 모델이다. 이 때는 LR robust game의 core 자체가 비어 있을 수도 있다.

### 6.2 해법: 분산 효과를 기준 분포에서 얻는다

√|S| 스케일링이 흉내 내려던 것은 중심극한정리식 상쇄다. 독립 오차 |S|개를 더하면 표준편차는 |S|가 아니라 √|S|로 큰다. 이 상쇄를 adversary의 budget이 아니라 기준 분포 ρ̂에 넣는다.

1. ρ̂: 멤버별 오차가 서로 독립인 결합 분포다. Ω는 그 결합 표본이고 연합과 무관하다.
2. P: ρ̂ 주변의 ambiguity set이다. CVaR\_β 집합이면 아래와 같다.
3. Adversary의 힘은 β 하나다. 각 시나리오의 가중치를 최대 1/(1−β)배까지만 올릴 수 있다. β는 연합과 무관하므로 (A1)이 성립한다.

```latex
P_\beta=\Big\{\rho\in\Delta(\Omega):\ 0\le\rho_\omega\le\frac{\hat\rho_\omega}{1-\beta}\Big\},\qquad \min_{\rho\in P_\beta}\mathbb{E}_\rho\big[-L_S\big]=-\mathrm{CVaR}_\beta(L_S),\quad L_S=\sum_{i\in S}L_i
```

**왜 이걸로 분산 효과가 나오는가.** Adversary는 시나리오를 새로 만들 수 없고, ρ̂에 있는 시나리오의 가중치만 바꾼다. 그래서 ρ̂에서 드문 사건은 adversary도 크게 키울 수 없다.

- |S|명의 오차가 동시에 최악인 시나리오는 독립인 ρ̂에서 지수적으로 드물다.
- 그래서 S의 tail(최악 1−β 비율)에서도 멤버 오차는 부분적으로 상쇄되고, tail 손실은 √|S|로 큰다.
- Budget 방식은 adversary가 오차 벡터를 직접 고른다. 이 상쇄가 없으니 √|S|를 Γ\_S로 손으로 넣어야 했다.

한 줄로 말하면, 연합 의존성이 adversary의 힘(Γ\_S)에서 데이터(ρ̂의 S-marginal)로 옮겨 간다. Remark 1에 의해 후자는 허용된다.

**예제 (i.i.d. Gaussian).** 멤버 손실 L\_i \~ N(0, σ²)가 독립이면 다음과 같다.

```latex
\mathrm{CVaR}_\beta(L_S)=k_\beta\,\sigma\sqrt{|S|},\qquad k_\beta=\frac{\varphi\big(\Phi^{-1}(\beta)\big)}{1-\beta}\qquad(k_{0.9}\approx 1.755)
```

β = 0.9, n = 30이면 개별 CVaR의 합은 약 52.7σ이고 공동 CVaR는 약 9.6σ이다. adversary의 힘 β는 같은데 tail 손실은 약 5.5배 작다.

**Parameter-set robust로는 안 되는 이유.** 독립성은 product measure ⊗\_j ρ\_j로만 표현된다. Product measure들의 집합은 볼록하지 않아서 보제 2의 minimax swap이 깨진다. P\_β처럼 결합 공간의 볼록 집합을 쓰면 이 문제가 없다. P\_β 안의 ρ는 tail 시나리오에 가중치를 몰아 멤버 오차 사이에 상관을 만들 수 있다. 이는 adversary가 β 한도 안에서 가장 나쁜 상관을 고르는 것이라 모델의 의도와 맞다.

**정리.** 가격이나 날씨처럼 모두가 공유하는 불확실성은 parameter set U로 robust하게 다뤄도 된다. Residential 수요처럼 멤버가 소유한 독립 불확실성은 기준 분포 주변의 DRO로 다룬다.

**코드 주의.** ieee\_owen/stochastic\_extension.py의 make\_scenarios는 부하 오차 path를 carrier마다 하나만 뽑아 모든 player가 공유한다. 지금의 수요 불확실성은 멤버 소유가 아니라 완전히 상관된 공통 충격이다. 멤버 소유 불확실성을 쓰려면 player별 독립 path가 필요하다. 또 결합 분포를 표본으로 표현하려면 |Ω|를 n과 함께 키워야 한다.

### 6.3 분산 효과는 협력 이득이다

DRO에서는 멤버 오차의 상쇄 자체가 협력 이득의 세 번째 원천이 된다. CVaR는 coherent risk measure라 subadditive하기 때문이다. 멤버들이 stand-alone plan을 그대로 쓰기만 해도 아래만큼 이득이고, 공동 dispatch는 그 위에 더한다.

```latex
v^{\mathrm{rob}}(S)-\sum_{i\in S}v^{\mathrm{rob}}(\{i\})\ \ge\ \sum_{i\in S}\mathrm{CVaR}_\beta(L_i)-\mathrm{CVaR}_\beta\Big(\sum_{i\in S}L_i\Big)\ \ge\ 0
```

여기서 L\_i는 멤버 i의 stand-alone plan 아래 손실이다.

| 협력 이득의 원천 | Stochastic (β = 0) | DRO (β > 0) |
| --- | --- | --- |
| 자원 pooling (공동 dispatch, linking rows) | 있음 | 있음 |
| Netting (import와 export 가격 차이 때문에 반대 방향 imbalance가 상쇄) | 있음 | 있음 |
| Tail 분산 (CVaR subadditivity) | 없음 (기댓값은 가법적) | 있음 |

**예제 계속 (i.i.d. Gaussian).** 분산 이득은 k\_β σ (|S| − √|S|)이다. β = 0.9, n = 30이면 약 52.7σ − 9.6σ = 43.1σ다. 커뮤니티가 커질수록 이득이 거의 선형으로 큰다.

**Owen 배분은 이 이득을 어떻게 나누는가.** P\_β에서 ρ\*는 커뮤니티 전체 손실 L\_N의 최악 1−β tail 위의 균등 분포다. 따라서 멤버 j가 부담하는 위험 비용은 커뮤니티 tail에서의 조건부 기대 손실이다.

```latex
\chi_j^{\mathrm{risk}}=-\,\mathbb{E}\big[L_j\ \big|\ L_N\ \text{in its worst }(1-\beta)\text{ tail}\big]
```

- Risk capital allocation의 Euler(CVaR contribution) 배분과 같은 형태다 (Tasche, Denault).
- i.i.d. Gaussian이면 각 멤버는 k\_β σ/√n을 부담한다. stand-alone k\_β σ보다 √n배 작다.
- Core 확인: 연합 S의 부담 합 |S| k\_β σ/√n은 CVaR\_β(L\_S) = k\_β σ √|S|보다 작다. √|S| ≤ √n이기 때문이다. 명제와 일치한다.
- 커뮤니티 tail에서 오히려 이익을 내는 멤버(예: 커뮤니티가 부족할 때 돈을 버는 flexible 자산)는 음의 부담, 즉 보험금을 받는다.

**논문 메시지 후보.** Stochastic 모델의 협력 이득은 pooling과 netting뿐이다. DRO는 tail 분산을 세 번째 원천으로 더하고, 그 이득은 커뮤니티 크기와 함께 커진다. 반면 안정성의 비용인 1인당 ε는 O(1/n)으로 줄어든다. 단, 두의 비교는 γ̄ 확인(7절) 뒤에만 말할 수 있다.

### 6.4 연속 U: private 제약 안 불확실성의 어려움

Private 제약 안의 불확실성은 볼록화를 거치면 RHS에서 계수 행렬로 옮겨 가고, convex hull 자체가 ξ에 대해 불연속으로 바뀐다. 그래서 master 쪽 불확실성에 통하던 표준 트릭이 모두 깨진다.

**쉬운 경우: master 쪽 불확실성.** 가격이나 linking RHS만 ξ에 의존하면 column(plan x\_j ∈ X\_j)은 ξ와 무관한 객체다. ξ는 column 계수와 master RHS에 affine으로만 들어간다. Robust counterpart는 master 안에서 처리되고 pricing은 그대로다. 2-stage라도 recourse가 LP면 recourse 가치는 ξ에 대해 concave라서, 최악의 ξ는 U의 꼭짓점에서 나온다.

**예제.** 최소 부하가 m이고 자기 풍력 ξ로만 도는 전해조를 본다.

```latex
X(\xi)=\{(u,y):\ u\in\{0,1\},\ m\,u\le y\le \bar Y u,\ y\le\xi\},\qquad \mathrm{conv}\,X(\xi)=\begin{cases}\{0\le u\le 1,\ m\,u\le y\le\min(\bar Y,\xi)\,u\} & \xi\ge m\\ \{(0,0)\} & \xi<m\end{cases}
```

**(a) RHS 불확실성이 계수 불확실성이 된다.** 원래 RHS에 있던 ξ가 convex hull에서는 u에 곱해진다(y ≤ ξ·u). LP relaxation에서는 y ≤ ξ로 RHS에 남는다. DW가 LP relaxation보다 좋은 bound를 주는 바로 그 이유 때문에 ξ-의존성이 affine이 아니게 된다. 그래서 inner max를 dualize해 한 단계 문제로 만드는 표준 트릭을 쓸 수 없다.

**(b) Convex hull이 ξ에 대해 불연속이다.** ξ가 m을 넘는 순간 한 점에서 2차원 polytope로 커진다. 이익이 p·y − c·u면 LR recourse 가치 Q(ξ)는 ξ = m에서 점프하고, concave도 연속도 아니다. 최악의 ξ는 꼭짓점이 아니라 m 바로 아래의 내부 점일 수 있어서, 꼭짓점 탐색이 통하지 않는다.

**(c) Column이 정책이 된다.** Recourse가 ξ에 반응해야 하므로 column은 정책 y\_j(·): U → X\_j(·)다.

- 볼록화 대상은 “ξ마다 conv X\_j(ξ)를 취한 것들의 곱”이 아니라 정책 집합의 convex hull이다. First-stage 정수(commitment)가 여러 ξ를 묶기 때문에 둘은 다르다.
- U가 연속이면 이 hull은 무한차원이다.
- C&CG(Zeng-Zhao)로 ξ를 하나씩 추가하면, 기존 column에는 새 ξ에서의 recourse가 없어 무효가 된다. First-stage를 고정하고 멤버 recourse를 풀어 확장하려면 relatively complete recourse 가정이 필요하다.

결국 adversary 문제는 “ξ에 대한 min 안에 DW 또는 MILP 가치가 들어간” bilevel이다. 루프는 C&CG 바깥, adversary, 그 안의 DW로 세 겹이 된다. Stage 2에 정수 recourse가 있으면(현재 코드의 heat pump commitment) Zhao-Zeng식 nested C&CG가 필요하다.

**ε bound에 대한 정정.** 이전 판에서 “active support는 유한하고 |Ω|는 |Ξ\*|로 바뀐다”고 썼다. 이는 LP recourse에서만 보장된다. Private 불확실성이나 정수 recourse가 있으면 성립한다고 말할 수 없다.

### 6.5 Pure robust에서의 설계: 전역 ball-box와 ADR

모든 멤버 좌표 위에 고정된 U 하나를 두고 연합은 그 투영을 쓰게 하면 (A1)은 자동으로 성립한다. 분산 효과는 그 U를 반경이 고정된 ball-box로 잡으면 얻는다.

**원칙: 전역 U와 투영.** U\_S := proj\_S U로 둔다. N의 worst case를 S로 제한한 것은 곧 proj\_S U의 원소이므로 (A1)이 성립한다. Remark 1의 robust 판이다. 연합 의존성이 adversary의 힘이 아니라 투영(데이터)에만 있다.

**분산 효과는 U의 모양이 정한다.** 연합 S의 합산 최악 편차는 support function이다.

```latex
h_U(\mathbf 1_S)=\max_{\xi\in U}\sum_{i\in S}\xi_i
```

| 전역 U | h\_U(1\_S) | 분산 효과 |
| --- | --- | --- |
| Box: ‖ξ‖∞ ≤ 1 | \|S\| | 없음 |
| Bertsimas-Sim: ‖ξ‖∞ ≤ 1, ‖ξ‖₁ ≤ Γ | min(\|S\|, Γ) | 없음. 연합이 커지면 상한이 멈춤 |
| Ball-box: ‖ξ‖∞ ≤ 1, ‖ξ‖₂ ≤ Ω | min(\|S\|, Ω√\|S\|) | 있음 |

**Ω를 모든 연합에 고정해도 되는 이유.** 확률 보장이 차원과 무관하기 때문이다. 독립이고 평균 0이며 |ξ\_i| ≤ 1인 교란에 대해 다음이 성립한다.

```latex
\text{Ball (Ben-Tal--Nemirovski)}:\ \ \Pr\Big(\sum_i a_i\xi_i>\Omega\|a\|_2\Big)\le e^{-\Omega^2/2},\qquad \text{Budget (Bertsimas--Sim)}:\ \ \Pr(\text{위반})\lesssim e^{-\Gamma^2/(2|S|)}
```

- Ball은 멤버 수와 상관없이 같은 Ω가 같은 신뢰도를 준다.
- Budget은 같은 신뢰도를 유지하려면 Γ\_S ∝ √|S|로 연합마다 바꿔야 하고, 그 순간 (A1)이 깨진다.
- 즉 6.1의 충돌은 robust 자체의 문제가 아니라 차원 의존적인 set(budget)을 쓴 탓이다.

**주의 1: 분산 효과는 합산되는 곳에서만 생긴다.** Private 제약 y\_j ≤ ξ\_j는 자기 좌표만 본다. Ball-box를 j 좌표로 투영하면 box \[−1, 1\]이다. 그래서 static robust에서 private 제약은 멤버 각자의 최악을 그대로 맞고 분산 효과가 없다. 분산 효과를 얻으려면 private 불확실성이 recourse를 통해 linking row(커뮤니티 balance)로 전파되어야 한다. 예를 들어 풍력 부족분을 커뮤니티 import로 메우는 경우다. 그러려면 adjustable robust가 필요하고, 6.4의 어려움으로 돌아간다.

**주의 2: 절충안은 affine decision rule(ADR)이다.** Continuous recourse를 다음과 같이 제한한다.

```latex
y_j(\xi)=y_j^0+Y_j\,\xi,\qquad \text{master row: }\ \sum_{j,p}\lambda_{jp}a_{jp}^0+\Omega\,\Big\|\sum_{j,p}\lambda_{jp}Y_{jp}-B\Big\|_2+(\text{box 항})\le b^0
```

- Column은 (first-stage 정수, y\_j⁰, Y\_j)로 다시 유한 차원이고 ξ와 무관한 객체다. 6.4의 (b), (c)가 사라진다.
- Private 제약은 자기 좌표의 box 위에서만 보면 되므로 robust counterpart가 선형이다. Pricing은 MILP로 남는다.
- Linking row는 ball 위의 robust counterpart라 SOC가 되고 master는 SOCP다. 분산 효과는 여기서 나온다.
- Owen은 conic dual에서 읽는다. U가 고정이므로 (A1)은 유지된다.
- Master를 LP로 유지하려면 ball 대신 Ben-Tal-Nemirovski의 polyhedral 근사를 쓴다.

대가는 두 가지다. ADR의 보수성 때문에 완전 적응보다 가치가 낮다. 또 정수 recourse(heat pump commitment)는 first stage로 올리거나 고정해야 한다.

### 6.6 MILP 게임 자체의 core

**MILP 게임 자체의 core.** v^MIP,rob의 core가 비어 있는지는 이 논증이 다루지 않는다. copositive(Sec. 4.3) 방법이 max-min 구조로 확장되는지도 열려 있다.

## 7. 열린 질문과 다음 단계

가장 큰 열린 질문은 정의 3(contingent)이고, 가장 싼 다음 단계는 finite Ω 위의 CVaR DRO 구현이다.

**열린 질문.**

1. 정의 3: ρ*를 쓴 Algorithm S1의 x\_j*(ω)가 min\_{ρ∈P} E\_ρ\[x(S, ·)\] ≥ v^MIP,rob(S)를 만족하는가? 보장되는 것은 E\_{ρ\*}\[χ(S, ·)\] ≥ v(S)뿐이다. 다른 ρ에서는 더 낮을 수 있다. 반례를 찾거나, 성립하는 P의 범위를 찾아야 한다.
2. γ̄: eq:gammatilde를 P 아래에서 다시 확인해야 한다. max\_{ρ∈P}로 바뀜다고 예상한다.
3. 논문 프레이밍: uncertainty set U를 쓰는 robust로 갈지, Ω 위의 분포 집합 P를 쓰는 DRO로 갈지 정해야 한다. 후자는 현재 코드로 거의 바로 된다.
4. P의 선택: CVaR\_β가 가장 자연스럽다. β 하나로 stochastic과 worst-scenario 사이를 잇는다.
5. 연속 U에서 column 확장과 adversary 문제의 tractability (6절).

**다음 단계.**

- [ ] stochastic\_extension.py에 --dro cvar --beta 옵션을 추가한다 (5절 변경점 1–3).
- [ ] β = 0에서 현재 v^CHP와 Owen 배분이 재현되는지 확인한다.
- [ ] make\_scenarios에 player별 독립 부하 오차 옵션을 추가하고, 분산 이득이 보이는 |Ω|를 정한다 (6.2).
- [ ] β sweep으로 ε, integrality gap, 협력 이득의 세 원천(6.3 표)을 분해해 본다.
- [ ] 작은 n에서 --check-core로 명제 2번(모든 S에 대한 stability)을 수치로 확인한다.
- [ ] 정의 3의 반례를 작은 예제(n = 2, |Ω| = 2)로 찾아본다.

## 8. 참고문헌

기억에 의존한 목록이고 웹에서 확인하지 않았다. 인용 전에 서지를 확인해야 한다.

- Owen (1975), On the core of linear production games, Mathematical Programming.
- Kalai & Zemel (1982), Totally balanced games and games of flow, Mathematics of Operations Research. (명제 1·2가 이 틀에서 바로 나온다)
- Geoffrion (1974), Lagrangean relaxation for integer programming, Mathematical Programming Study.
- Sion (1958), On general minimax theorems, Pacific Journal of Mathematics.
- Bertsimas & Sim (2004), The price of robustness, Operations Research.
- Ben-Tal & Nemirovski (2000), Robust solutions of linear programming problems contaminated with uncertain data, Mathematical Programming. (ball의 차원 무관 확률 보장, 6.5)
- Ben-Tal, El Ghaoui & Nemirovski (2009), Robust Optimization, Princeton University Press. (ball-box와 polyhedral 근사)
- Ben-Tal, Goryashko, Guslitzer & Nemirovski (2004), Adjustable robust solutions of uncertain linear programs, Mathematical Programming. (ADR, 6.5)
- Rockafellar & Uryasev (2000), Optimization of conditional value-at-risk, Journal of Risk.
- Delbaen (2000), Coherent risk measures on general probability spaces. (worst-case 측도 배분이 core에 든다는 결과, 서지 확인 필요)
- Denault (2001), Coherent allocation of risk capital, Journal of Risk. (6.3의 배분과 같은 형태)
- Tasche (1999), Risk contributions and performance measurement. (Euler 배분, 서지 확인 필요)
- Zeng & Zhao (2013), Solving two-stage robust optimization problems using a column-and-constraint generation method, Operations Research Letters.
- Zhao & Zeng (2012), 정수 recourse가 있는 two-stage robust의 nested C&CG. (제목과 출처 확인 필요, 6.4)
- Bauso & Timmer (2009), Robust dynamic cooperative games, International Journal of Game Theory. (서지 확인 필요)
- Doan & Nguyen, Robust stable payoff distribution in stochastic cooperative games. (저널과 연도 확인 필요. 정의 3과 가장 가까울 가능성이 있다)

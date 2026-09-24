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

- χ\_j는 커뮤니티의 worst-case 분포 ρ\*와 가격 π\* 아래에서 평가한 멤버 j의 가치다. ρ\*는 내생적으로 정해지는 pricing measure다.
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

### 5.1 KL ball을 LP로: column-and-cut generation 전체 절차

Exponential cone 없이 LP 하나에서 두 oracle이 번갈아 돈다. 멤버 pricing(MILP)은 column을, KL separation(닫힌 형태)은 분포 cut(row)을 추가하고, 두 oracle의 위반량을 더한 것이 곧 상한과 하한의 차이다.

**Restricted master (RMP).** D는 지금까지 만든 분포들이고, g\_ω는 시나리오별 커뮤니티 이익이다.

```latex
\begin{aligned}
\max\ &\theta\\
\text{s.t. }&\theta-\sum_\omega\rho^k_\omega\,g_\omega(\lambda,x_0)\le 0&&[\mu_k]\quad\forall k\in D\\
&\sum_{j,q}a^\omega_{jq}\lambda_{jq}+A^\omega_0x_0\le b^\omega&&[\pi^\omega]\quad\forall\omega\\
&\sum_q\lambda_{jq}=1&&[\sigma_j]\quad\forall j\\
&\lambda,\ x_0\ge 0,\ \theta\ \text{free},\qquad g_\omega(\lambda,x_0)=\sum_{j,q}w^\omega_{jq}\lambda_{jq}+c_0^{\omega\top}x_0
\end{aligned}
```

- θ column에서 Σ\_k μ\_k = 1이 나온다. 집계 분포 ρ̄ = Σ\_k μ\_k ρ^k는 conv D ⊆ P에 있다.
- Column q의 cut k 계수는 −Σ\_ω ρ^k\_ω w\_jq^ω이다. Column.scen(시나리오별 비용)에서 바로 계산하고, 새 cut을 넣을 때 기존 column 전부의 계수를 내적으로 채운다.

**Oracle 1: pricing (멤버별 MILP).** 기존 PlayerPricing과 같은 MILP이고 가중치만 ρ̂에서 ρ̄로 바뀐다. rc\_j > tol이면 column을 추가한다.

```latex
\mathrm{rc}_j=\max_{x_j\in X_j}\Big\{\mathbb{E}_{\bar\rho}\big[f_j(x_j)\big]-\sum_\omega\pi^{\omega\top}A_j^\omega x_j\Big\}-\sigma_j
```

**Oracle 2: KL separation (닫힌 형태).** 현재 RMP 해의 시나리오별 이익 g\*로 worst-case 분포를 구한다.

```latex
\phi(g)=\min_{\mathrm{KL}(\rho\|\hat\rho)\le r}\rho^\top g,\qquad \rho_\omega(\eta)\propto\hat\rho_\omega\,e^{-g_\omega/\eta},\qquad \mathrm{KL}\big(\rho(\eta)\|\hat\rho\big)=r
```

- η는 이분법으로 찾는다. KL은 η에 대해 단조 감소한다. η → ∞면 ρ̂이고, η → 0이면 argmin g 위에 몰린다. r ≥ −ln ρ̂(argmin g)이면 해는 argmin 위의 ρ̂ 조건부 분포다.
- φ(g\*) < θ\* − tol이면 ρ(g\*)를 D에 cut으로 추가한다. 비용은 O(|Ω| × 이분법 횟수)라 MILP pricing에 비하면 무시할 만하다.

**상한과 하한.**

- 하한 LB = φ(g\*): 현재 RMP 해는 참 master에서도 feasible하고, 그 참 robust 가치가 φ(g\*)다.
- 상한 UB = z\_RMP + Σ\_j rc\_j: (ρ̄, π\*)가 참 Lagrangian dual에서 feasible(ρ̄ ∈ P)이므로 유효하다. Pricing을 MIP gap으로 풀면 rc\_j 대신 pricing의 dual bound를 쓴다.
- UB − LB = (θ\* − φ(g\*)) + Σ\_j rc\_j다. 첫 항이 cut 위반량, 둘째 항이 pricing 위반량이다.
- RMP 값 자체는 어느 쪽 bound도 아니다. Column 제한은 값을 낮추고, 분포 제한(conv D ⊆ P)은 값을 높인다.

**루프.**

1. 초기화: D = {ρ̂}, column은 같은 Ω의 stochastic 실행 terminal column으로 warm start한다. 이 상태의 RMP가 곧 현재 stochastic master다.
2. RMP를 푼다. HiGHS LP이고 column·row 추가 후 warm start한다.
3. Separation: 위반 cut이 있으면 추가하고 2로 간다. 싸므로 매 반복 먼저 한다.
4. Pricing: (ρ̄, π\*)(smoothing을 쓰면 안정화 중심과의 볼록결합)으로 멤버 MILP를 풀고 rc\_j > tol인 column을 추가한다.
5. LB와 UB를 갱신한다. UB − LB ≤ tol이면 종료하고, 아니면 2로 간다.
6. 관리: 오래 μ\_k = 0인 cut과 오래 쓰이지 않은 column을 purge한다. 기존 column purge 규칙을 cut에도 적용한다.

**Column을 추가해도 기존 cut은 그대로 유효하다.** Cut은 KL ball 안의 분포 하나(내부 근사 conv D ⊆ P의 꼭짓점)이므로 column 집합과 무관하다. 그래서 column이 늘어도 cut을 버리거나 다시 만들 필요가 없고, 다음 row generation은 기존 cut들을 들고 warm start한다. C&CG에서 새 ξ가 기존 column을 무효로 만드는 것(6.4)과 반대이고, 이 절차가 가벼운 이유다.

**Nested와 interleaved.** 논문 설명은 nested로 한다. 현재 column으로 exp master를 row generation으로 끝까지 푼 뒤 그 dual (π\*, ρ̄)로 pricing하는 방식이고 표준 DW와 같다. 구현에서는 interleaved(매 반복 cut 몇 개만 추가하고 바로 pricing)도 시도해볼 만하다. 두 방식 모두 종료 시 위반 cut과 양의 rc가 없는지 확인하므로 결과는 같다.

**안정화.** Wentges smoothing은 (π, ρ̄)에 그대로 적용된다. ρ의 볼록결합은 P 안에 머물므로 ρ̄ ∈ P가 유지된다. du Merle penalty는 π에만 걸면 된다. 분포 cut은 매끄러운 concave 함수 φ의 Kelley 근사라 느리게 수렴할 수 있다.

**Owen 배분 (종료 후).**

```latex
\rho^*=\bar\rho=\sum_k\mu_k\rho^k,\qquad \chi_j=\sigma_j+\mathrm{rc}_j\ \big(=\phi_j(\rho^*,\pi^*)\ \text{또는 그 상한}\big)
```

- 안정성은 tolerance와 무관하게 정확히 성립한다. χ\_j ≥ φ\_j(ρ\*, π\*)이고 (ρ\*, π\*) ∈ P × Θ이기 때문이다(9.2).
- 효율성 쪽 초과분은 robust duality gap에 종료 gap(UB − LB)을 더한 것이다. 균등 차감한 ε에 종료 gap/n이 더해진다.

**검증 계획.**

- r = 0: separation이 ρ̂만 돌려주므로 현재 stochastic 결과(v^CHP, Owen)가 재현돼야 한다.
- r을 키우면 v^LR,rob은 단조 감소하고 ρ\*는 나쁜 시나리오로 이동해야 한다.
- 작은 n에서 --check-core로 안정성을 확인하고, exponential cone(MOSEK이나 Gurobi) 직접 풀이와 값을 비교한다.

**코드 변경점 (stochastic\_extension.py의 DirectMaster).**

1. θ column과 cut row 집합을 두고, 새 cut의 column 계수를 채운다.
2. PlayerPricing의 가중치를 호출마다 받는다(5절 변경점 2와 같다).
3. Separation 함수(tilting과 이분법)를 추가한다.
4. LB와 UB 계산을 위 식으로 교체한다.
5. S1 배분의 probs를 ρ\*로 바꾼다(5절 변경점 3과 같다).

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

### 6.7 Ambiguity set의 선택: 투영 일관성, 밀도비 통제, KL

본 모델의 ambiguity set은 두 조건을 만족해야 한다. Robust game으로 제시하려면 고정 반경 KL ball이 가장 자연스럽고, 분해를 쓰면 LP로 풀 수 있다.

**두 조건.**

1. 투영 일관성: 집합 규칙이 marginal화와 교환되어야 한다. 즉 {ρ의 S-marginal : ρ ∈ P(ρ̂)} = P(ρ̂\_S)다. 이것이 (A1)을 주고, “소연합이 같은 규칙으로 스스로 골랐을 set이 공 투영과 같다”는 해석을 보장한다. Law-invariance는 이 성질을 위험척도 쪽에서 본 모습이다.
2. 밀도비 통제: ρ̂에서 드문 사건에 질량을 싸게 올릴 수 없어야 한다. 이것이 분산 효과를 보장한다.

확률 p인 사건에 질량 δ를 올리는 비용은 TV에서 약 δ, KL에서 약 δ·log(δ/p), χ²에서 약 δ²/p다. 따라서 φ가 초선형인 divergence(KL, χ²)는 드문 결합 극단을 막고, 선형인 TV는 못 막는다. TV ball의 worst case는 아래와 같고, 둘째 항은 유계 손실이면 |S|에 선형이며 표본에서는 |Ω|에 의존한다.

```latex
\max_{\rho:\ \mathrm{TV}(\rho,\hat\rho)\le\delta}\mathbb{E}_\rho[L]=(1-\delta)\,\mathrm{CVaR}_\delta(L)+\delta\max L
```

| 집합과 크기 규칙 | (A1) | 투영 일관성 | 분산 효과 |
| --- | --- | --- | --- |
| Budget, 신뢰도에 맞춘 Γ\_S ∝ √\|S\| | 깨짐 | 성립 | 있음 |
| Budget, 상수 Γ (N에 맞춤) | 성립 | 깨짐 (소연합 과보호) | \|S\| > Γ인 연합에만 |
| Ball-box, 고정 Ω | 성립 | 성립 | 있음 (√\|S\|) |
| CVaR\_β, 고정 β | 성립 | 성립 | 있음 (√\|S\|) |
| KL ball, 고정 r | 성립 | 성립 | 있음 (√\|S\|) |
| TV ball, 고정 δ | 성립 | 성립 | 부분적 |
| Wasserstein/KL, 표본 수·차원으로 보정한 반경 | 투영을 쓰면 성립 | 깨짐 | 설정에 따라 다름 |

**Budget set의 Γ (pure robust).** (A1)은 min(|S|, Γ\_N) ≤ Γ\_S ≤ |S|와 같다. 분산 이득이 가장 큰 선택은 상수 Γ를 N의 신뢰도에 맞추는 것이다. Bertsimas-Sim 상한을 쓰면 Γ ≈ √(2n ln(1/ε\_v))이고, ε\_v = 5%에서 n = 6, 30, 60이면 Γ ≈ 6.0, 13.4, 18.9다. n = 6에서는 Γ ≥ n이라 분산 이득이 없다. 작은 연합은 위반 확률이 약 exp(−Γ²/(2|S|))로 더 작아 과보호된다. 이탈 연합을 보수적으로 평가하는 것이라 안정성에는 문제가 없지만, 소연합이 굳이 전체 크기의 set을 상정할 이유가 없어 해석이 약하다.

**KL 반경을 표본 수 없이 잡는 법.** 세 기준 모두 차원과 무관하다. KL의 chain rule 때문에 결합 KL ball의 투영은 같은 반경의 KL ball이다.

1. 꼬리 사건: r = ln(1/p). 확률 P(A)인 사건으로 조건부를 걸면 KL 비용이 정확히 −ln P(A)다. 그래서 r은 “확률 p 이상인 어떤 사건도 일어났다고 가정하고 대비한다”는 뜻이다.
2. σ 단위: r = k²/2. Gaussian이면 worst-case 평균 이동이 σ\_S√(2r)이다. √|S|는 σ\_S에서 나오므로 r 자체는 연합과 무관하다.
3. CVaR와 비교: r = ln(1/(1−β))이면 P\_β ⊆ KL ball이다. P\_β의 원소는 밀도비가 1/(1−β) 이하라 KL도 그 이하이기 때문이다. β = 0.9이면 r ≈ 2.30이다.

**주의.**

- 공통 편향은 빠진다. 모든 멤버의 예측이 같은 방향으로 틀리는 경우 product ρ̂ 기준 KL은 |S|·KL\_i라, r을 고정하면 큰 연합에서 ball 밖이다. 분산 이득은 “공통 오지정이 비싸다”는 가정에서 나온다. 공통 날씨 모델 오차처럼 실제로 걱정되는 공통 요인은 ρ̂에 명시해야 하고, 그 요인에 대해서는 분산 효과가 정직하게 사라진다. CVaR도 같다.
- 유한 표본에서는 r < ln|Ω|여야 의미가 있다. 표본 하나에 질량을 몰아도 KL이 ln|Ω|이기 때문이다. tail에 실질적으로 남는 표본 수 |Ω|e^(−r)이 수십 개는 되게 잡는다.

**포장: risk game이 아니라 robust game.** KL ball은 “기준 예측오차 모형의 오지정에 대한 robustness”로 읽힌다(Hansen-Sargent 계열의 entropy 제약 robustness). CVaR도 형식상 DRO(밀도비 ≤ 1/(1−β)인 재가중)지만 위험척도로 읽히기 쉽다. 그래서 본 모델은 KL로 두고 CVaR는 비교용으로 쓰는 방향을 검토 중이다.

**계산: KL을 LP로 푸는 분해.** KL-DRO를 그대로 쓰면 master가 exponential cone이지만, worst-case 분포가 closed form이라 cut 생성으로 LP를 유지할 수 있다.

```latex
\phi(g)=\min_{\rho:\ \mathrm{KL}(\rho\|\hat\rho)\le r}\rho^\top g,\qquad \rho_\omega(g)\propto\hat\rho_\omega\,e^{-g_\omega/\lambda},\quad \lambda>0:\ \mathrm{KL}\big(\rho(g)\|\hat\rho\big)=r
```

- g는 현재 master 해의 시나리오별 커뮤니티 이익이다. φ는 concave이고, tilted 분포 ρ(g)가 그 supergradient다. λ는 단조인 1차원 근 찾기로 정해진다.
- Master는 exponential cone 대신 분포 cut θ ≤ Σ\_ω ρ^k\_ω g\_ω(λ)를 쓴다(k는 지금까지 만든 분포). LP이고, 멤버 column 생성과 분포 cut(row) 생성이 같은 LP에서 번갈아 돈다. 새 cut이 위반되지 않으면(φ(g\*) ≥ θ\* − tol) 종료한다.
- Cut들은 P를 안쪽에서 근사한다(conv{ρ^k} ⊆ P). 그래서 중간 master 값은 참값보다 크고, 종료 시 tol 이내로 같아진다.
- Owen: cut row의 dual μ\_k(Σμ\_k = 1)로 ρ\* = Σ μ\_k ρ^k를 만든다. P가 볼록이므로 ρ\* ∈ P이고, 9.2의 안정성 증명에 필요한 것은 (ρ\*, π\*) ∈ P × Θ뿐이라 그대로 성립한다. 초과분은 robust duality gap에 tol이 더해진다.
- 비용: 루프가 두 겹(분포 cut, 멤버 column)이지만 같은 LP 안이라 DirectMaster 구조를 재사용하고 HiGHS로 푼다. Kelley형 cut은 느리게 수렴할 수 있어 stabilization이 필요할 수 있다. 대안은 Gurobi나 MOSEK의 exponential cone을 직접 쓰고 conic dual로 Owen을 읽는 것이다.

## 7. 열린 질문과 다음 단계

가장 큰 열린 질문은 정의 3(contingent)이고, 가장 싼 다음 단계는 finite Ω 위의 CVaR DRO 구현이다.

**열린 질문.**

1. 정의 3: ρ\*를 쓴 Algorithm S1의 x\_j\*(ω)가 min\_{ρ∈P} E\_ρ\[x(S, ·)\] ≥ v^MIP,rob(S)를 만족하는가? 보장되는 것은 E\_{ρ\*}\[χ(S, ·)\] ≥ v(S)뿐이다. 다른 ρ에서는 더 낮을 수 있다. 반례를 찾거나, 성립하는 P의 범위를 찾아야 한다.
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

## 9. 논문 블록 정리 (논의 중)

논문에 넣을 핵심 블록 다섯 개의 메시지와 진술 초안이다. 기호는 원고(lem:lpg, prop:opap, cor:eps, prop:eps, κ\_j 정규화)에 맞춘다. 아직 확정한 것은 없고, 모두 손 유도라 9.3 상수와 9.5 분해는 검증이 필요하다.

### 9.1 Definition: robust game과 robust core

**메시지.** 각 연합은 자기 worst case로 평가받고, 커뮤니티 자신도 마찬가지다. 비관의 기준이 양쪽에서 같다.

**진술.**

```latex
v^{\mathrm{rob}}(S)=\max_{x\in X_S}\ \min_{\rho\in P}\ \sum_{\omega}\rho_\omega f_S^\omega(x)
```

- Core와 weak ε-core는 원고 eq:weakeps를 그대로 쓴다.
- 가정 (A1)을 여기서 선언한다. P ⊆ Δ(Ω)는 연합과 무관한 polytope다.
- P = {ρ̂}이면 Remark stoch의 게임이다.

**논의할 점.** Remark 1(멤버 소유 불확실성은 marginal로 들어오므로 (A1)을 깨지 않는다)을 정의 바로 뒤에 붙이는 안. 리뷰어가 가장 먼저 물을 질문이다.

### 9.2 Lemma: robust Owen solution

**메시지.** Robust화는 행을 추가할 뿐 구조를 바꾸지 않는다. Epigraph row도 linking row이므로 sub-game은 여전히 linear production game이고, worst-case 분포 ρ\*는 같은 master의 dual 변수로 나온다.

**진술.** lem:lpg, prop:opap, cor:eps의 robust 판이다.

- (a) Lagrangian sub-game v̄(S) = min\_{(ρ,θ)∈P×Θ} v^LR(S; ρ, θ)는 val(DWR^rob\_S)와 같다. Epigraph 변수 θ와 α는 x0처럼 master 소유다. Epigraph row의 RHS는 0이라 endowment는 원고와 같다.
- (b) Owen 해는 아래와 같다. ρ\*와 θ\*는 grand coalition master의 dual이다.
- (c) 모든 S에서 χ(S) ≥ v^rob(S)이고, 초과분 ω^LR,rob은 robust duality gap이다. 균등 차감하면 weak ε-core에 들고 ε = ω^LR,rob/n이다.

```latex
\sigma_j^*=\max_{x_j\in X_j}\Big\{\mathbb{E}_{\rho^*}\big[f_j(x_j)\big]-\theta^{*\top}A_jx_j\Big\}
```

**논의할 점.** 원고의 “MILP 게임은 이 구조가 없다”는 대비를 강화할 수 있다. Robust MILP에는 integrality gap 외에 minimax gap이 하나 더 있고, DW 볼록화가 둘을 동시에 닫는다. 이 문장을 lemma 뒤 본문에 둘지 remark로 나눌지 정해야 한다.

### 9.3 Proposition: ε bound

**메시지.** Robust화의 비용은 행 수와 상수에만 들어가고, rate는 O(1/n) 그대로다.

**진술 (손 유도, 확인 필요).**

```latex
\varepsilon^{\mathrm{LR,rob}}\ \le\ \frac{(m+|\Omega|)\,\bar\gamma^{\mathrm{rob}}}{n}
```

- Shapley-Folkman 차원: 원고는 m+1(linking rows와 목적함수)이다. Robust에서는 목적함수 한 줄이 epigraph row |Ω|개로 바뀌어 m+|Ω|가 된다. m 자체가 이미 (2|K|+3)|T||Ω|라 차이는 작다.
- γ̄^rob: eq:gammatilde의 목적함수 손실 항 (w̄\_j − w\_j)⁺를 max\_{ρ∈P} E\_ρ\[w̄\_j^ω − w\_j^ω\]⁺로 바꾼다. 새 dispatch의 robust 가치가 θ̄ − Σ\_{j∈J} max\_ρ E\_ρ\[손실\_j\] 이상이라는 점에서 나온다.
- P\_β이면 γ̄^rob ≤ γ̄^stoch/(1−β)라는 명시적 상한이 된다. ρ ≤ ρ̂/(1−β)이고 손실이 0 이상이기 때문이다.

**KL ball에서의 γ̄^rob (반경 r).** γ̄^rob는 여전히 n과 무관하게 유계라 rate는 그대로다. 다만 CVaR처럼 γ̄^stoch의 몇 배라는 곡셈형 상한은 없고, 손실 범위가 들어간 덧셈형 상한이 나온다.

- P와 무관한 부분: SF로 고른 멤버 j ∈ J가 시나리오 ω에서 내는 되돌리기 손실을 ℓ\_j^ω ≥ 0(목적함수 손실과 peak 증가분)이라 하면, 되돌린 dispatch의 robust 가치는 θ̄ − Σ\_{j∈J} Φ(ℓ\_j) 이상이다. Φ(ℓ) := max\_{ρ∈P} E\_ρ\[ℓ\]이고, min(a − b) ≥ min a − max b와 max의 subadditivity에서 나온다.
- 그래서 γ̃\_j^rob = sup\_{x̄\_j} min\_{x∈X\_j(ā\_j)} { Π^res·(reserve 부족분) + Φ(ℓ\_j(x)) }이다. KL로 바뀌는 것은 Φ뿐이다.

```latex
\Phi_r(\ell)=\sup_{\mathrm{KL}(\rho\|\hat\rho)\le r}\mathbb{E}_\rho[\ell]=\inf_{\eta>0}\Big\{\eta\log\mathbb{E}_{\hat\rho}\big[e^{\ell/\eta}\big]+\eta\,r\Big\}
```

닫힌 상한은 세 가지다. ρ̂ 기준으로 μ는 평균, σ²는 분산, R은 범위 max ℓ − min ℓ, M은 max ℓ이다.

| 상한 | 식 | 쓰임 |
| --- | --- | --- |
| Pinsker | μ + R·√(r/2) | 가장 단순. 범위만 필요 |
| Bernstein | μ + √(2rσ²) + R·r/3 | r과 분산이 작을 때 더 조임 |
| 평균·최댓값 기준 최적 | M·kl⁻¹(μ/M, r) | 평균과 최댓값만 알 때 가장 조임(두 점 분포가 극값, KL-UCB와 같은 함수) |

Pinsker를 쓰고 안쪽 min의 x를 stochastic 최적 x로 고정하면 다음이 된다. R̄는 멤버 하나의 시나리오별 되돌리기 손실의 최대 범위이고, eq:bnd\_trade로 유계이며 n과 무관하다.

```latex
\bar\gamma^{\mathrm{rob}}\le\bar\gamma^{\mathrm{stoch}}+\sqrt{r/2}\,\bar R,\qquad \varepsilon^{\mathrm{LR,rob}}\le\frac{(m+|\Omega|)\big(\bar\gamma^{\mathrm{stoch}}+\sqrt{r/2}\,\bar R\big)}{n}
```

- 곡셈형 상한이 없는 이유: 확률 p인 사건에서만 손실 M이 나면 CVaR는 Φ ≤ Mp/(1−β)지만 KL은 Φ = M·q\*이다. q\*는 kl(q\*‖p) = r의 해라 p → 0에서도 대략 r/ln(1/p) 수준이고, q\*/p → ∞다. KL은 드문 사건의 우도비를 로그 비용만 받고 키워 주므로, γ̄는 드물지만 큰 되돌리기 손실(R̄)에 민감하다.
- r = ln(1/(1−β))이면 P\_β ⊆ KL ball이라 KL의 γ̄는 CVaR의 γ̄ 이상이다.
- 분산 효과(6.2)와는 충돌하지 않는다. 그쪽은 많은 멤버의 동시 극단 비용 문제이고, 여기는 멤버 한 명의 손실 분포다.
- 개선 가능성(추측): subadditivity 단계는 J의 손실이 동시에 최악이라고 가정하는 셈이라 느슨하다. 합 Σ\_J ℓ\_j에 Bernstein을 직접 쓰고 손실들이 약하게만 상관되면, robust 할증 항이 (m+|Ω|)가 아니라 √(m+|Ω|)로 커진다. ℓ\_j들이 같은 결합 시나리오의 함수라 독립 가정의 정당성은 확인이 필요하다.

**논의할 점.** SF를 m+1 차원으로 줄이는 방법(ρ\*에서 평가)은 부등호 방향이 반대라 안 된다고 판단했다. 새 dispatch의 가치는 min\_ρ로 평가되는데, ρ\*에서의 값은 그 상한일 뿐이다. 그래서 시나리오별 총이익 벡터 전체를 보존해야 하고 m+|Ω| 차원이 필요하다. 이 판단이 맞는지 확인이 필요하다. 4절의 m' 표기도 이에 맞춰 정리해야 한다.

### 9.4 Proposition 후보: 분산 효과는 협력 이득이다

후보는 둘이다. 현재 추천은 (a)를 명제로, (b)를 corollary나 example로 두는 것이고, 결정은 보류했다.

**(a) 일반 명제: 합병 이득의 분해.** 서로소인 S, T에 대해 다음이 성립한다. g\_S는 S의 최적 robust plan이 시나리오별로 내는 이익이다.

```latex
v^{\mathrm{rob}}(S\cup T)-v^{\mathrm{rob}}(S)-v^{\mathrm{rob}}(T)\ \ge\ H(S,T)\ \ge\ 0,\qquad H(S,T)=\min_{\rho\in P}\mathbb{E}_\rho[g_S+g_T]-\min_{\rho\in P}\mathbb{E}_\rho[g_S]-\min_{\rho\in P}\mathbb{E}_\rho[g_T]
```

- H = 0일 필요충분조건은 S와 T가 공통의 worst-case 분포를 갖는 것이다.
- 증명은 세 줄이다. S∪T는 두 plan을 동시에 돌릴 수 있고(linking row가 가법적, as:pool), min\_ρ는 선형 함수들의 min이라 superadditive하다.
- Stochastic(P가 한 점)에서는 H ≡ 0이라 합병 이득이 전부 pooling이다. Robust에서는 worst case가 다른 연합끼리 합칠수록 hedging 이득이 더해진다.
- 가정은 (A1)뿐이고 분포 가정이 필요 없다. 그래서 명제로 적합하다.

**(b) 규모 법칙: corollary나 example.** 멤버 손실이 독립이고 분산이 유한하며 P = P\_β이면, CLT에 의해 1인당 hedging 이득은 Θ(1)로 k\_β σ에 수렴한다. 반면 1인당 안정성 비용 ε는 O(1/n)이다.

- 논문 메시지로는 가장 강하다. 커뮤니티가 커질수록 1인당 협력 가치는 유지되고 불안정성은 사라진다.
- 다만 CLT와 독립성 가정, 9.3의 γ̄^rob 확정이 필요하다. 멤버 간 상관이 있으면 분산이 다시 |S|에 비례해 효과가 줄어든다(9.6).

### 9.5 Corollary 후보: Owen 배분의 pooling/hedging 분해

**메시지.** 각 멤버의 몫은 pooling 몫과 hedging 몫으로 쪼개지고, 둘 다 음수가 아니다.

**진술.** κ\_j(ρ\*)는 멤버 j가 커뮤니티의 worst-case 분포 ρ\* 아래에서 혼자 낼 수 있는 가치다.

```latex
\chi_j^{\mathrm{OW}}=\sigma_j^*-\kappa_j=p_j+h_j,\qquad p_j:=\sigma_j^*-\kappa_j(\rho^*)\ge 0,\qquad h_j:=\kappa_j(\rho^*)-\kappa_j\ge 0
```

- p\_j ≥ 0 (pooling 몫): 싱글턴 {j}에 eq:weakdual을 (ρ\*, θ\*)에서 적용하면 나온다.
- h\_j ≥ 0 (hedging 몫): κ\_j = max\_x min\_ρ ≤ min\_ρ max\_x ≤ κ\_j(ρ\*)이다.
- h\_j = 0일 필요충분조건은 j 자신의 worst case가 커뮤니티의 worst case와 일치하는 것이다. 커뮤니티가 나쁨 때 같이 나쁜 멤버는 hedging 몫이 없고, 반대로 움직이는 멤버는 보험료를 받는다.
- 위험만 있는 예시에서 h\_j는 Euler(CVaR contribution) 배분이 된다(Tasche, Denault). i.i.d. Gaussian이면 h\_j = k\_β σ (1 − 1/√n)이다.

**논의할 점.** 이 분해는 κ\_j 정규화를 쓰는 원고 eq:chi\_ow에 바로 얹히므로 corollary가 자연스럽다. 9.4 명제의 part (ii)로 합치는 안도 있다. p\_j ≥ 0 유도에서 공유 변수 x0(r\_sym, p)의 Lagrangian 항이 dual feasible한 θ\*에서 0 이하인지도 확인해야 한다.

### 9.6 댓글 스레드에서 정리된 점과 열린 결정

**6.2 예제에 대한 댓글 논의.**

- β가 정하는 것은 adversary가 시나리오 가중치를 최대 1/(1−β)배까지만 올릴 수 있다는 것뿐이다. 독립인 ρ̂에서는 |S|명이 동시에 최악인 시나리오의 질량이 지수적으로 작아서 tail은 일부 멤버만 나쁜 시나리오로 채워진다.
- “분산 효과가 연합 수와 무관하다”는 틀린 표현이다. 효과의 크기(|S| 대비 √|S|)는 연합 크기에 따라 커지고, 연합과 무관한 것은 adversary의 힘 β다.
- √|S|의 근거: Gaussian이면 독립 합의 표준편차가 σ√|S|이고 CVaR가 positively homogeneous라 정확하다. 일반 분포에서는 CLT 근사로 CVaR\_β(L\_S) ≈ |S|μ + k\_β σ√|S|이고, 평균 항은 가법적이라 협력 이득에 영향이 없다.
- 깨지는 조건: 멤버 간 상관 ρ\_c > 0이면 분산이 σ²(|S| + |S|(|S|−1)ρ\_c)라서 큰 |S|에서 다시 |S|에 비례한다. 작은 |S|, heavy tail, 작은 표본 Ω에서도 근사가 나빠진다.
- Projection: S의 문제에는 S 멤버 데이터만 들어오므로 S가 마주하는 것은 P의 S-marginal이다. P\_β이면 이는 ρ̂의 S-marginal에 같은 β로 CVaR를 건 것과 정확히 같다. 다만 증명은 projection 없이 ρ\*\_N ∈ P만으로 충분하다.

**보류한 결정.**

1. 9.4: (a)를 명제로, (b)를 corollary나 example로 둘지.
2. 9.5: 독립 corollary로 둘지, 9.4 명제의 part (ii)로 합칠지.
3. 9.3: SF 차원을 m+|Ω|로 두는 판단과 γ̄^rob의 정의.
4. 9.2: minimax gap 문장을 본문에 둘지 remark로 둘지.
5. 9.1: Remark 1을 정의 바로 뒤에 둘지.
6. 본 모델 ambiguity set: KL(robust game 포장, cut 생성으로 LP) vs CVaR(LP 직접, risk game으로 읽힘). 검토 중인 방향은 KL (6.7). KL이면 9.3의 γ̄^rob 상한도 P\_β 대신 KL ball로 다시 써야 한다.

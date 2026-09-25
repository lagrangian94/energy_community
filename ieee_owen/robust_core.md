# Robust core와 DW Owen: 유한 시나리오 판

Sep 24, 2026 · @Seokwoo Kim

> [Claude Docs 문서](https://claude.ai/code/artifact/fd931bec-e11b-40de-a9ad-b2393bf781b0)의 "유한 시나리오 판" 탭 스냅샷(2026-09-25). 살아 있는 판은 문서다.

유한 시나리오 Ω 위에서 robust core를 정의하고(1절), DRO 모델과 계산을 유도한다(2절). 결론은 하나다. Ambiguity set의 크기가 연합과 무관하게 고정되면 grand coalition DW master의 dual에서 읽은 Owen 배분이 robust core에 들고, 효율 손실은 minimax gap과 integrality gap의 합이다(1.2 Remark). 이론은 손 유도다.

## 1. Robust core

### 1.1 설정과 정의

- 멤버 N = {1, …, n}, 연합 S ⊆ N. X\_j는 멤버 j의 MILP 운영집합(bounded)이고, 이익 f\_j와 linking 기여 A\_jx\_j는 선형이며, linking rows는 멤버에 대해 가법적이고 b\_j = 0이다.
- 불확실성은 유한 시나리오 집합 Ω 위에 있다(stochastic\_extension.py의 시나리오). 시나리오 하나는 모든 멤버 불확실성의 결합 실현값이고, 아래 식의 ξ는 그 실현값이다.
- Adversary는 분포 ρ ∈ P ⊆ Δ(Ω)를 고른다. P는 기준 분포 ρ̂ 주변의 ambiguity set이다.

```latex
v^{\mathrm{rob}}(S)=\max_{x\in X_S}\ \min_{\rho\in P}\ \mathbb{E}_\rho\big[f_S(x,\xi)\big],\qquad \mathcal{C}^{\mathrm{rob}}=\big\{\chi:\ \chi(N)=v^{\mathrm{rob}}(N),\ \chi(S)\ge v^{\mathrm{rob}}(S)\ \forall S\big\}
```

- **채택한 정의:** 이탈 연합도 커뮤니티도 각자의 worst case로 평가한다(일관된 비관). Weak ε-core는 원고 eq:weakeps 그대로다. P = {ρ̂}이면 현재 stochastic extension이다.
- **기각한 정의:** uniform robust core(χ(S) ≥ max\_ρ v\_ρ(S))는 S = N에서 모순이라 P가 한 점이 아니면 빈다. 커뮤니티는 비관하고 이탈자는 낙관하는 비대칭이 원인이다.
- **열린 정의:** contingent robust core. 실현별 배분에 ex-post budget balance와 min\_ρ E\_ρ\[χ(S, ·)\] ≥ v^rob(S)를 요구한다. 상수 배분은 ex-post balance를 깨므로 채택한 정의와는 실제로 다른 개념이다.

**가정 (A1)과 원칙.** Adversary의 힘(P의 크기: β, r 등)은 연합과 무관하게 고정하고, 연합 의존성은 데이터(ρ̂의 marginal)에만 둔다. 증명에 실제로 필요한 것은 “grand coalition의 worst case가 모든 S에서도 허용된다”는 것뿐이다. 즉 모든 S에서 ρ\*\_N ∈ P\_S이고, 충분조건은 P\_N ⊆ P\_S다.

**Remark (멤버 소유 불확실성).** 멤버가 들어오면 그 불확실성이 생기고 나가면 빠지는 것은 이미 모델에 있다. S의 문제에는 S 멤버 데이터만 들어오므로 S가 마주하는 것은 P의 S-marginal이다. 연합 의존성이 데이터에 있으면 (A1)과 충돌하지 않고, adversary의 힘에 있으면 깨진다(예: 1.3 표의 보정 반경, √|S|로 커지는 budget).

### 1.2 주정리: robust Owen

**핵심 관찰.** Robust MILP를 epigraph로 쓰면 “θ ≤ ξ별 커뮤니티 이익” row도 멤버를 묶는 linking row다. Linking row와 함께 dualize하면 multiplier ρ가 자동으로 P 안에 들어가고 목적함수는 멤버별로 separable하다. 이 row는 시나리오마다 하나씩 |Ω|개다(2.1).

```latex
v^{\mathrm{LR,rob}}(S)=\min_{\rho\in P,\ \pi\ge 0}\sum_{j\in S}\phi_j(\rho,\pi)=\min_{\rho\in P}v^{\mathrm{LR}}_\rho(S),\qquad \phi_j(\rho,\pi)=\max_{x_j\in X_j}\Big\{\mathbb{E}_\rho\big[f_j(x_j,\xi)\big]-\pi^\top A_jx_j\Big\}
```

- 안쪽 min\_π는 확률 ρ를 쓴 stochastic Lagrangian dual(Remark stoch의 v^LR)이라, v^LR,rob = min\_ρ v^LR\_ρ는 min 두 개의 중첩일 뿐이고 스왑이 필요 없다.
- Primal로 쓰면 max\_{x∈C\_S} min\_ρ F\_S = min\_ρ max\_{x∈C\_S} F\_S다(Geoffrion과 Sion, C\_S는 conv X\_j와 linking rows). X\_j ⊆ conv X\_j이므로 v^MIP,rob ≤ v^LR,rob이다.
- MILP에서는 max-min이 min-max보다 작을 수 있다(minimax gap). DW 볼록화는 integrality gap과 함께 이 gap도 닫는다.
- 함정: linking row만 dualize하고 epigraph를 row로 풀지 않으면 min\_ρ가 안쪽에 남아 pricing이 멤버별로 쪼개지지 않는다.

**명제 (Robust Owen).** (A1) P는 연합과 무관하다. (A2) X\_j는 bounded이고 f, A는 선형이다. (ρ\*, π\*)를 grand coalition master의 optimal dual로 두고 χ\_j = φ\_j(ρ\*, π\*)라 하면 다음이 성립한다.

1. Σ\_{j∈N} χ\_j = v^LR,rob(N).
2. 모든 S에서 Σ\_{j∈S} χ\_j ≥ v^LR,rob(S) ≥ v^MIP,rob(S).
3. χ\_j − g/n은 weak ε-core에 든다. g는 효율 손실(아래 Remark의 minimax gap과 integrality gap의 합)이고 ε = g/n이다.

**증명.** (A1)에 의해 S의 dual feasible region P × R₊는 N의 것과 같다. 따라서 (ρ\*, π\*)는 S에서도 feasible하고, 목적함수가 separable하므로 weak duality 한 줄로 2번이 나온다. 스왑은 쓰이지 않는다. b\_j ≠ 0이면χ\_j에 π\*ᵀb\_j가 더해진다.

**해석.**

- ρ\*는 커뮤니티가 내생적으로 정하는 pricing measure다. 커뮤니티에 나쁜 실현에서 돈을 버는 멤버가 더 받는다(보험료).
- Stand-alone κ\_j = v^MIP,rob({j})는 멤버 자신의 worst case라 비관적이다. 그래서 협력 이득에 자원 pooling 외에 hedging 가치가 더해진다. 원고 eq:v0의 Γ^MIP 정규화도 이 κ\_j로 한다.
- 포지셔닝: 명제의 구조는 Owen(1975), Kalai–Zemel(1982), risk capital의 worst-case 측도 배분(Delbaen, Denault)과 같다. 새로운 점은 MILP dispatch에서 LR을 거쳐 살아남는다는 것, epigraph를 linking으로 보는 계산 설계, robust ε bound(2.4)다.

**Remark (Geoffrion과 Sion의 역할).** 명제 자체는 weak duality 한 줄로 끝나고 스왑은 쓰이지 않는다. 두 정리는 계산한 값과 ρ\*가 무엇을 뜻하는지를 말해 준다.

1. Weak duality(정리): (A1)로 (ρ\*, π\*)가 모든 S의 Lagrangian dual에서 feasible하므로 안정성이 나온다.
2. Geoffrion(값의 의미): v^LR,rob(S) = max\_{x∈C\_S} min\_ρ F\_S, 즉 X를 convex hull로 바꾼 커뮤니티의 robust 가치다. 그래서 v^MIP,rob ≤ v^LR,rob이고(LR 게임에서 안정이면 MIP 게임에서도 안정), 효율 손실은 볼록화된 해를 정수해로 되돌릴 때 잃는 양이다. Shapley–Folkman(2.4)은 이 primal 형태 위에서 작동한다.
3. Sion(ρ\*의 의미): C\_S가 볼록이라 max-min = min-max = min\_ρ v^LR\_ρ(S)이고 (x\*, ρ\*)는 saddle point다. ρ\*는 볼록화된 plan에 대한 진짜 최악 분포이고, robust Owen은 “ρ\*에서 평가한 stochastic Owen”과 같다. Worst case를 먼저 구한 뒤 Owen을 계산하는 2단계가 필요 없고, master 하나에서 ρ\*와 π\*가 함께 나온다.

```latex
v^{\mathrm{MIP,rob}}(S)=\max_{x\in X_S}\min_{\rho}F_S\ \underbrace{\le}_{\text{minimax gap}}\ \min_{\rho}\max_{x\in X_S}F_S\ \underbrace{\le}_{\text{integrality gap}}\ \min_{\rho}\max_{x\in C_S}F_S=v^{\mathrm{LR,rob}}(S)
```

비볼록이면 효율 손실 g는 minimax gap과 integrality gap의 합이다. 볼록화된 세계에서는 두 gap이 모두 0이라 Owen 구조가 살아나고, DW 한 번이 두 gap을 동시에 닫으며 대가는 그 합의 1/n이다. 원고의 duality-based 관점이 robust에서 더 필수적이라는 논문 메시지가 여기서 나온다.

### 1.3 분산 효과와 협력 이득의 원리

분산 효과를 내면서 (A1)을 지키는 adversary 집합은 두 조건을 만족해야 한다. KL·CVaR가 둘을 만족한다.

1. 투영 일관성: 집합 규칙이 marginal화(투영)와 교환된다. 이것이 (A1)을 주고, “소연합이 같은 규칙으로 고를 집합이 공 투영”이라는 해석을 보장한다. Law-invariance는 이 성질을 위험척도 쪽에서 본 모습이다.
2. 밀도비 통제: ρ̂에서 드문 사건(여러 멤버의 동시 최악)에 질량을 싸게 올릴 수 없다. 이것이 분산 효과를 보장한다. 확률 p 사건에 질량 δ를 올리는 비용은 TV ≈ δ, KL ≈ δ·log(δ/p), χ² ≈ δ²/p다.

| 집합과 크기 규칙 | (A1) | 투영 일관성 | 분산 효과 |
| --- | --- | --- | --- |
| KL ball, 고정 r | 성립 | 성립 | 있음 (√\|S\|) |
| CVaR\_β, 고정 β | 성립 | 성립 | 있음 (√\|S\|) |
| TV ball, 고정 δ | 성립 | 성립 | 부분적: (1−δ)CVaR\_δ + δ·max |
| 표본 수·차원으로 보정한 반경 (Wasserstein, KL) | 투영을 쓰면 성립 | 깨짐 | 설정에 따라 다름 |

**분산 효과는 협력 이득이다.** 최악 기댓값(위험척도)은 superadditive라서 stand-alone plan을 그대로 써도 아래만큼 이득이다. 공동 dispatch는 그 위에 더한다. 식은 CVaR로 썼지만 어떤 볼록 P에서도 같다.

```latex
v^{\mathrm{rob}}(S)-\sum_{i\in S}v^{\mathrm{rob}}(\{i\})\ \ge\ \sum_{i\in S}\mathrm{CVaR}_\beta(L_i)-\mathrm{CVaR}_\beta\Big(\sum_{i\in S}L_i\Big)\ \ge\ 0
```

| 협력 이득의 원천 | Stochastic | Robust (DRO) |
| --- | --- | --- |
| 자원 pooling (공동 dispatch) | 있음 | 있음 |
| Netting (import·export 가격 차이) | 있음 | 있음 |
| Tail 분산 (hedging) | 없음 (기댓값은 가법적) | 있음 |

Owen 배분은 이 이득을 Euler 배분으로 나눈다. 각 멤버는 커뮤니티 worst case에 대한 자기 기여만큼 부담하고, 커뮤니티가 나쁨 때 오히려 버는 멤버는 음의 부담(보험금)을 받는다. 구체적인 식은 2.2(CVaR)에 있다.

## 2. DRO 모델과 계산

### 2.1 모델과 유도

Ω는 유한하고(stochastic\_extension.py의 시나리오), X\_j는 scenario-expanded, non-anticipative MILP 운영집합이다. Linking rows는 시나리오마다 있다. Ambiguity set은 P = {ρ ∈ Δ(Ω): Gρ ≥ h} 같은 볼록 집합이고, P = {ρ̂}이면 현재 stochastic extension이며 P = Δ(Ω)이면 최악 시나리오 robust다.

안쪽 min을 LP dual로 바꾸면 robust MILP가 하나의 max 문제가 된다. 대괄호는 함께 dualize할 row의 multiplier다.

```latex
\min_{\rho\in P}\sum_\omega\rho_\omega f^\omega=\max_{\theta,\ \alpha\ge0}\Big\{\theta+h^\top\alpha:\ \theta+(G^\top\alpha)_\omega\le f^\omega\ \forall\omega\Big\},\qquad v^{\mathrm{MIP,rob}}(S)=\max\Big\{\theta+h^\top\alpha:\ \theta+(G^\top\alpha)_\omega\le\sum_{j\in S}f_j^\omega\ [\rho_\omega],\ \sum_{j\in S}A_j^\omega x_j\le 0\ [\pi^\omega],\ x_j\in X_j,\ \alpha\ge0\Big\}
```

- θ에 대한 sup이 유한하려면 Σρ = 1, α ≥ 0에 대한 sup이 유한하려면 Gρ ≥ h여야 한다. 그래서 multiplier ρ는 자동으로 P 안에 있다.
- φ\_j(ρ, π) = max\_{x\_j∈X\_j} Σ\_ω (ρ\_ω f\_j^ω − π^ωᵀA\_j^ω x\_j)는 현재 stochastic pricing MILP에서 ρ̂을 ρ로 바꾼 것이다.
- P = Δ(Ω)이면 ρ\*는 N의 최악 시나리오들 위에 모인다.

### 2.2 Ambiguity set 선택과 분산 효과

분산 효과는 adversary의 힘이 아니라 기준 분포 ρ̂에서 나온다. ρ̂는 멤버별 오차가 독립인 결합 분포이고, adversary는 ρ̂에 있는 시나리오의 가중치만 바꿀 수 있다. |S|명이 동시에 최악인 시나리오는 지수적으로 드물어서 tail 손실은 √|S|로 큰다.

**CVaR.**

```latex
P_\beta=\Big\{\rho\in\Delta(\Omega):\ 0\le\rho_\omega\le\frac{\hat\rho_\omega}{1-\beta}\Big\},\qquad \min_{\rho\in P_\beta}\mathbb{E}_\rho[-L_S]=-\mathrm{CVaR}_\beta(L_S),\quad L_S=\sum_{i\in S}L_i
```

- β=0이면 stochastic, β→1이면 최악 시나리오다. S가 마주하는 것은 ρ̂의 S-marginal에 같은 β로 CVaR를 건 것과 정확히 같다(law-invariant). 증명에는 ρ\*\_N ∈ P만 필요하다.
- √|S|의 근거: Gaussian이면 정확하고 CVaR\_β(L\_S) = k\_βσ√|S|(k\_0.9 ≈ 1.755)다. 일반 분포는 CLT로 ≈ |S|μ + k\_βσ√|S|이고, 평균 항은 가법적이라 협력 이득과 무관하다. 멤버 간 상관 ρ\_c > 0이면 분산이 σ²(|S| + |S|(|S|−1)ρ\_c)라 큰 |S|에서 다시 |S|에 비례한다. 작은 |S|, heavy tail, 작은 표본에서도 근사가 나빠진다.
- 예: β = 0.9, n = 30이면 개별 CVaR 합 52.7σ가 공동 9.6σ로 줄어든다.
- Owen(Euler) 부담: ρ\*는 L\_N의 최악 (1−β) tail 위의 균등 분포라 χ\_j^risk = −E\[L\_j | L\_N의 최악 tail\]이다. i.i.d. Gaussian이면 멤버당 k\_βσ/√n이고, 연합 S의 부담 합 |S|k\_βσ/√n은 CVaR\_β(L\_S) = k\_βσ√|S| 이하라 core에 든다(Tasche, Denault의 CVaR contribution과 같은 형태).
- 독립성을 parameter-set robust로 직접 표현하려면 product measure 집합이 필요한데 볼록이 아니라 minimax swap이 깨진다. P\_β는 tail에 가중치를 몰아 멤버 간 상관을 만들 수 있고, 이는 β 한도 안의 가장 나쁜 상관이라 의도에 맞다.

**KL ball (본 모델 후보).** φ가 초선형이라 드문 결합 극단을 막고, KL의 chain rule 때문에 결합 KL ball의 투영은 같은 반경의 KL ball이다. Gaussian이면 worst-case 평균 이동이 σ\_S√(2r)라 √|S|로 큰다. 반경은 표본 수 없이 선호로 잡는다.

- r = ln(1/p): 확률 p 이상인 어떤 사건도 일어났다고 보고 대비한다(사건 A로 조건부를 걸면 KL 비용이 정확히 −ln P(A)).
- r = k²/2: Gaussian에서 합산 손실이 k-sigma 밀리는 것까지 대비한다.
- r = ln(1/(1−β)): P\_β ⊆ KL ball(밀도비가 1/(1−β) 이하라 KL도 그 이하)이라 CVaR\_β보다 보수적이다. β = 0.9이면 r ≈ 2.30이다.
- 주의: 공통 편향은 product ρ̂ 기준 KL이 |S|에 비례해 큰 연합에서 ball 밖이다. 공통 날씨 오차처럼 걱정되는 공통 요인은 ρ̂에 명시하고, 그 요인에 대해서는 분산 효과가 정직하게 사라진다.
- 유한 표본에서는 r < ln|Ω|이고, tail 유효 표본 |Ω|e^(−r)이 수십 개는 되어야 한다.

**TV는 부적합하다.** worst case가 (1−δ)CVaR\_δ(L) + δ·max L이고, 둘째 항은 유계 손실이면 |S|에 선형, 표본에서는 |Ω|에 의존한다. TV는 질량만 제한하고 밀도비는 제한하지 않기 때문이다.

**포장.** KL ball은 기준 예측오차 모형의 오지정에 대한 robustness(Hansen–Sargent 계열)로 읽혀 robust game으로 제시하기 좋다. CVaR는 risk game으로 읽히기 쉬워 비교용이다.

**코드 주의.** make\_scenarios는 부하 오차 path를 carrier마다 하나만 뽑아 모든 player가 공유한다(완전 상관). 멤버 소유 불확실성을 쓰려면 player별 독립 path가 필요하고 |Ω|를 n과 함께 키워야 한다.

### 2.3 계산: DW와 KL column-and-cut

Master는 기존 DW에 epigraph row를 더한 것이고, pricing은 같은 MILP에서 ρ̂만 worst-case 분포로 바뀐다. KL은 closed-form tilting으로 분포 cut을 만들어 exponential cone 없이 LP로 푼다.

**Polytope P (예: CVaR).** Master에 θ, α column과 ω-row |Ω|개를 더한다. ω-row의 dual이 곧 ρ\*이고 자동으로 P 안에 있다. Reserve의 r\_sym이 t에 대해 하던 max-min을 ω에 대해 하는 것과 같은 장치다.

```latex
\begin{aligned}
\max\ &\theta+h^\top\alpha\\
\text{s.t. }&\theta+(G^\top\alpha)_\omega-\sum_{j,p}\lambda_{jp}f_{jp}^\omega\le 0&&[\rho_\omega]\quad\forall\omega\\
&\sum_{j,p}\lambda_{jp}a_{jp}^\omega\le b_N^\omega&&[\pi^\omega]\quad\forall\omega\\
&\sum_p\lambda_{jp}=1&&[\mu_j]\quad\forall j,\qquad \lambda,\alpha\ge 0,\ \theta\ \text{free}
\end{aligned}
```

- Linking row는 ρ로 scale하지 않는다. π\*^ω에 확률 가중이 이미 들어 있다. 종료 시 μ\_j = φ\_j(ρ\*, π\*)이고 b\_j = 0이라 χ\_j = μ\_j로, 현재의 owen = −sigma와 같은 모양이다.
- Pricing은 max\_{x\_j} Σ\_ω ρ\*\_ω f\_j^ω − Σ\_ω π\*^ωᵀA\_j^ωx\_j − μ\_j로, 현재 PlayerPricing에서 ρ̂만 ρ\*로 바뀐다.
- 공유 변수 x0: r\_sym의 비용은 first-stage라 모든 ω에 같고 Σρ = 1이므로 목적함수에 직접 둔다. p^ω처럼 시나리오별 비용은 해당 ω-row에 넣는다. 이 변수들의 dual 제약은 연합과 무관해 (A1)을 깨지 않는다.
- Stabilization(Wentges smoothing, du Merle penalty)은 그대로 적용된다.

**KL ball: restricted master.** D는 지금까지 만든 분포들이고, g\_ω는 시나리오별 커뮤니티 이익이다.

```latex
\begin{aligned}
\max\ &\theta\\
\text{s.t. }&\theta-\sum_\omega\rho^k_\omega\,g_\omega(\lambda,x_0)\le 0&&[\mu_k]\quad\forall k\in D\\
&\sum_{j,q}a^\omega_{jq}\lambda_{jq}+A^\omega_0x_0\le b^\omega&&[\pi^\omega]\quad\forall\omega\\
&\sum_q\lambda_{jq}=1&&[\sigma_j]\quad\forall j
\end{aligned}
```

- Pricing(멤버별 MILP): rc\_j = max\_{x\_j} {E\_ρ̄\[f\_j\] − Σ\_ω π^ωᵀA\_j^ω x\_j} − σ\_j. 가중치는 ρ̄ = Σ\_k μ\_k ρ^k ∈ P다.
- KL separation(닫힌 형태): 현재 해의 g\*로 아래 분포를 만들고, φ(g\*) < θ\* − tol이면 cut으로 추가한다.

```latex
\phi(g)=\min_{\mathrm{KL}(\rho\|\hat\rho)\le r}\rho^\top g,\qquad \rho_\omega\propto\hat\rho_\omega e^{-g_\omega/\eta},\quad \mathrm{KL}=r;\qquad \mathrm{UB}-\mathrm{LB}=\big(\theta^*-\phi(g^*)\big)+\sum_j\mathrm{rc}_j
```

- LB = φ(g\*)는 현재 해의 참 robust 가치이고, UB = z\_RMP + Σ rc\_j는 (ρ̄, π\*)의 Lagrangian bound다. 차이가 cut 위반량과 pricing 위반량으로 정확히 나뉜다. RMP 값 자체는 어느 쪽 bound도 아니다(column 제한은 낮추고 분포 제한 conv D ⊆ P는 높인다). Pricing을 MIP gap으로 풀면 rc\_j 대신 pricing의 dual bound를 쓴다.
- 루프: D = {ρ̂}와 같은 Ω의 stochastic column으로 시작(이 상태가 곷 현재 stochastic master) → RMP → separation(매 반복 먼저) → pricing → UB − LB ≤ tol이면 종료.
- **Column을 추가해도 기존 cut은 그대로 유효하다.** Cut은 KL ball 안의 분포 하나라 column과 무관하고, row generation은 항상 warm start된다. C&CG에서 새 ξ가 기존 column을 무효로 만드는 것과 반대다.
- 논문 설명은 nested(exp master를 row generation으로 끝까지 푼 뒤 그 dual로 pricing)로 하고, 구현에서는 interleaved(매 반복 cut 몇 개만 더하고 바로 pricing)도 시도해볼 만하다.
- Owen: ρ\* = ρ̄, χ\_j = σ\_j + rc\_j. 안정성은 tolerance와 무관하게 정확하고(ρ̄ ∈ P), 종료 gap은 ε에 gap/n으로만 더해진다.

**KL 세부.**

- η 찾기: KL(ρ(η)‖ρ̂)은 η에 대해 단조 감소한다. η → ∞이면 ρ̂, η → 0이면 argmin g 위로 몰린다. r ≥ −ln ρ̂(argmin g)이면 해는 argmin 위의 ρ̂ 조건부 분포다. φ는 concave이고 tilted 분포가 그 supergradient다. 비용은 O(|Ω| × 이분법 횟수)다.
- Cut 계수: column q의 cut k 계수는 −Σ\_ω ρ^k\_ω w\_jq^ω다. 새 cut을 넣을 때 기존 column 전부의 계수를 내적으로 채운다.
- 안정화: Wentges smoothing은 (π, ρ̄)에 그대로 적용된다(ρ의 볼록결합은 P 안에 머문다). du Merle penalty는 π에만 건다. 분포 cut은 매끄러운 concave 함수의 Kelley 근사라 느리게 수렴할 수 있다.
- 관리: 오래 μ\_k = 0인 cut과 오래 쓰이지 않은 column은 기존 column purge 규칙으로 정리한다.
- 대안: Gurobi나 MOSEK의 exponential cone을 직접 쓰고 conic dual에서 Owen을 읽는다. 검증용으로도 쓴다.

**구현과 검증 (stochastic\_extension.py).**

- DirectMaster: θ(와 polytope면 α) column, ω-row 또는 cut row를 추가한다. 계수는 시나리오별 unscaled 비용 Column.scen\[w\] + Column.first에서 계산하고, 목적함수의 ρ̂-scaled cost는 뺀다. LB와 UB는 위 식으로 바꾼다.
- PlayerPricing: base cost가 생성 시점에 stack.scaled\_cost로 ρ̂에 고정되어 있다. 호출마다 ρ를 받도록 바꾼다. Tilting separation 함수를 추가한다.
- scenario\_allocation(S1): pen\[w\]가 dual을 probs\[w\]로 나눈다. ρ\*\_ω = 0이면 0으로 나누게 되므로 나누지 않는 형태로 다시 쓴다.
- 검증: r = 0(또는 β = 0)에서 v^CHP와 Owen이 소수점 넷째 자리까지 같아야 하고, r을 키우면 v^LR,rob은 단조 감소하고 ρ\*는 나쁜 시나리오로 이동해야 한다. 작은 n에서 --check-core와 exponential cone 직접 풀이로 확인한다.

### 2.4 ε bound

Robust화의 비용은 행 수와 상수에만 들어가고 rate는 O(1/n) 그대로다(확인 필요).

```latex
\varepsilon^{\mathrm{LR,rob}}\le\frac{(m+|\Omega|)\,\bar\gamma^{\mathrm{rob}}}{n},\qquad \bar\gamma^{\mathrm{rob}}\le\begin{cases}\bar\gamma^{\mathrm{stoch}}/(1-\beta)&\text{CVaR}_\beta\\ \bar\gamma^{\mathrm{stoch}}+\sqrt{r/2}\,\bar R&\text{KL ball (Pinsker)}\end{cases}
```

- SF 차원: 원고는 m+1(linking rows와 목적함수)이다. 여기서는 목적함수 1행이 epigraph |Ω|행으로 바뀌어 m+|Ω|다. m = (2|K|+3)|T||Ω|이라 차이는 작다. ρ\*에서 평가해 m+1로 줄이는 것은 부등호 방향이 반대라 안 된다고 판단했다. 새 dispatch의 가치는 min\_ρ로 평가되는데 ρ\*에서의 값은 그 상한일 뿐이라, 시나리오별 총이익 벡터 전체를 보존해야 한다.

**γ̄^rob의 정의.** SF로 고른 멤버 j ∈ J가 되돌리기에서 내는 시나리오별 손실을 ℓ\_j^ω ≥ 0(목적함수 손실과 peak 증가분)이라 하면, 되돌린 dispatch의 robust 가치는 θ̄ − Σ\_{j∈J} Φ(ℓ\_j) 이상이다. min(a − b) ≥ min a − max b와 max의 subadditivity에서 나온다.

```latex
\tilde\gamma_j^{\mathrm{rob}}=\sup_{\bar x_j}\ \min_{x\in X_j(\bar a_j)}\Big\{\Pi^{\mathrm{res}}\cdot(\text{reserve 부족분})+\Phi\big(\ell_j(x)\big)\Big\},\qquad \Phi(\ell)=\max_{\rho\in P}\mathbb{E}_\rho[\ell],\qquad \bar\gamma^{\mathrm{rob}}=\max_j\tilde\gamma_j^{\mathrm{rob}}
```

- CVaR: ρ ≤ ρ̂/(1−β)이고 손실이 0 이상이라 γ̄^rob ≤ γ̄^stoch/(1−β)다.
- KL: Φ\_r(ℓ) = inf\_η {η log E\_ρ̂ e^(ℓ/η) + ηr}(Donsker–Varadhan)이고 닫힌 상한은 아래 세 가지다. μ, σ², R, M은 ρ̂ 기준 평균, 분산, 범위, 최댓값이고, R̄는 멤버 하나의 시나리오별 되돌리기 손실의 최대 범위(eq:bnd\_trade로 유계, n 무관)다.

| 상한 | 식 | 쓰임 |
| --- | --- | --- |
| Pinsker | μ + R·√(r/2) | 범위만 필요. 위 bound에 쓴 것 |
| Bernstein | μ + √(2rσ²) + R·r/3 | r과 분산이 작을 때 더 조임 |
| 평균·최댓값 기준 최적 | M·kl⁻¹(μ/M, r) | 두 점 분포가 극값(KL-UCB와 같은 함수) |

- KL에는 γ̄^stoch의 곡셈형 상한이 없다. 확률 p 사건에만 손실 M이 나면 CVaR는 Φ ≤ Mp/(1−β)지만 KL은 Φ = M·q\*이고, q\*(kl(q\*‖p) = r의 해)는 p → 0에서도 약 r/ln(1/p)라 q\*/p → ∞다. 그래서 KL의 γ̄는 드물지만 큰 되돌리기 손실(R̄)에 민감하다.
- r = ln(1/(1−β))이면 P\_β ⊆ KL ball이라 KL의 γ̄는 CVaR의 γ̄ 이상이다. 분산 효과(2.2)와는 충돌하지 않는다. 그쪽은 여러 멤버의 동시 극단이고 여기는 한 멤버의 손실 분포다.
- 개선 가능성(추측): subadditivity 대신 Σ\_J ℓ\_j에 Bernstein을 직접 쓰면, 손실이 약하게만 상관될 때 robust 할증이 (m+|Ω|)가 아니라 √(m+|Ω|)로 커진다.
- 주의: 멤버 소유 독립 오차를 표현하려고 |Ω|를 n과 함께 키우면 m+|Ω|도 n과 함께 커져 bound가 나빠진다. 실제 gap이 Θ(√n)인지는 확인하지 않았다.

### 2.5 행 수 문제와 모델링 선택

Stochastic master는 시간당 6|Ω|행(carrier balance 3, reserve 2, peak 1)이라 CG가 느려진다. 6인에서 |Ω| = 1, 2, 3, 5일 때 약 175, 328, 454, 1558번 반복했다(엔진 혼재). |Ω| = 5에서는 멤버 하나(u2)가 거의 매 반복 column을 추가했다.

- Worst-case 분포가 sparse해도(TV, CVaR) feasibility 행은 줄지 않는다. 가중치 0인 시나리오의 dual은 degenerate해져 오히려 진동을 키울 수 있다. Ambiguity가 더하는 행은 cut |D|개뿐이다.
- 행이 |Ω|에 비례하는 것은 “실시간 recourse를 커뮤니티가 공동 정산한다”는 설계의 결과다. 계량값 기준 사후 배분이라 현실적이고 netting과 peak 분산 이득을 잡는다.

| 설계 | 시간당 행 | 잃는 것 |
| --- | --- | --- |
| A. 전부 실시간 공동 정산 (현재) | 6·\|Ω\| | 없음 |
| C. 전기만 실시간, H·G는 first-stage, reserve는 멤버별 first-stage 분담 | 약 \|Ω\| + 2–3 | reserve 재조정 유연성. peak까지 분담하면 peak 분산 이득도 |
| B. Two-settlement (커뮤니티는 day-ahead만, 편차는 각자 grid) | 6 (\|Ω\| 무관) | 실시간 netting. DRO의 tail 분산은 남는다 |

- 이 선택은 ε bound에도 들어간다. Remark stoch의 “불확실성이 허용 편차를 키운다”는 m에 |Ω|가 곱해지기 때문이라, B면 결정론과 같은 O(|T|/n)이다.
- 같은 모델에서 시도할 것: 시나리오별 결정론 dual(ρ̂\_ω·π^det\_ω)로 warm start와 안정화 중심, 시나리오별·extensive form 해의 commitment로 column seeding, barrier RMP(crossover 없이), Gurobi solution pool로 멤버당 여러 column, scenario reduction.

## 3. 논문 블록

기호는 원고(lem:lpg, prop:opap, cor:eps, prop:eps, κ\_j 정규화)에 맞춘다. 이 탭의 1절이 본문의 공통 결과이고 2절의 DRO가 그 모델이다. 아직 확정한 것은 없다.

### 3.1 Definition: robust game과 robust core

메시지: 각 연합과 커뮤니티가 모두 자기 worst case로 평가된다. 1.1의 v^rob, core, 가정 (A1)을 선언하고 멤버 소유 불확실성 Remark를 바로 뒤에 둔다(리뷰어가 가장 먼저 물을 질문).

### 3.2 Lemma: robust Owen solution

메시지: robust화는 행을 추가할 뿐 구조를 바꾸지 않는다. (a) Lagrangian sub-game은 epigraph row를 linking으로 포함한 DWR^rob이고 θ, α는 x0처럼 master 소유다(epigraph row의 RHS는 0이라 endowment는 원고와 같다). (b) Owen 해는 σ\_j\* = max {E\_ρ\*\[f\_j\] − θ\*ᵀA\_jx\_j}다. (c) 모든 S에서 안정이고 ε = ω^LR,rob/n이다. Minimax gap 문장의 위치는 미정이다.

### 3.3 Proposition: ε bound

메시지: robust화의 비용은 행 수와 상수에 들어간다. 유한 시나리오에서는 (m+|Ω|)γ̄^rob/n이다(2.4).

### 3.4 Proposition 후보: hedging 이득

(a) 서로소인 S, T에 대해 합병 이득은 H(S,T) 이상이다. g\_S는 S의 최적 robust plan이 실현별로 내는 이익이다.

```latex
v^{\mathrm{rob}}(S\cup T)-v^{\mathrm{rob}}(S)-v^{\mathrm{rob}}(T)\ \ge\ H(S,T)=\min_{\rho\in P}\mathbb{E}_\rho[g_S+g_T]-\min_{\rho\in P}\mathbb{E}_\rho[g_S]-\min_{\rho\in P}\mathbb{E}_\rho[g_T]\ \ge\ 0
```

증명: S∪T는 두 plan을 동시에 돌릴 수 있고(linking row가 가법적, as:pool), min\_ρ는 선형 함수들의 min이라 superadditive하다. 등호는 두 함수의 minimizer가 공통일 때만 성립한다. Stochastic(P가 한 점)에서는 H ≡ 0이라 합병 이득이 전부 pooling이고, robust에서는 worst case가 다른 연합끼리 합칠수록 hedging 이득이 더해진다. 가정은 (A1)뿐이고, √|S| budget처럼 (A1)이 깨지면 성립하지 않는다.

(b) 규모 법칙(corollary나 example): 독립 손실이면 1인당 hedging 이득은 Θ(1)(Gaussian이면 k\_βσ에 수렴)이고 1인당 ε는 0으로 간다. 멤버 간 상관이 있으면 효과가 줄어든다.

### 3.5 Corollary 후보: pooling/hedging 분해

메시지: 각 멤버의 몫은 pooling 몫과 hedging 몫으로 쪼개지고 둘 다 음수가 아니다. κ\_j(ρ\*)는 멤버 j가 커뮤니티의 worst-case 분포 아래에서 혼자 낼 수 있는 가치다. 원고 eq:chi\_ow에 바로 얹힌다.

```latex
\chi_j^{\mathrm{OW}}=\sigma_j^*-\kappa_j=\underbrace{\sigma_j^*-\kappa_j(\rho^*)}_{p_j\ \ge\ 0}+\underbrace{\kappa_j(\rho^*)-\kappa_j}_{h_j\ \ge\ 0}
```

- p\_j ≥ 0은 싱글턴에 eq:weakdual을 (ρ\*, θ\*)에서 적용한 것이고, h\_j ≥ 0은 κ\_j = max-min ≤ min-max ≤ κ\_j(ρ\*)다. h\_j = 0이면 j의 worst case가 커뮤니티와 같다.
- 위험만 있는 예시에서 h\_j는 Euler(CVaR contribution) 배분이고, i.i.d. Gaussian이면 h\_j = k\_βσ(1 − 1/√n)이다. 커뮤니티가 나쁨 때 같이 나쁜 멤버는 hedging 몫이 없고, 반대로 움직이는 멤버는 보험료를 받는다.
- 확인할 점: p\_j ≥ 0에서 공유 변수 x0(r\_sym, p)의 Lagrangian 항이 θ\*에서 0 이하인가.

## 4. 열린 질문, 보류한 결정, 다음 단계

**열린 질문.**

1. Contingent robust core: ρ\*를 쓴 실현별 배분(S1)이 min\_ρ E\_ρ\[x(S, ·)\] ≥ v^rob(S)를 만족하는가. 보장되는 것은 ρ\*에서의 부등식뿐이다.
2. 2.4의 SF 차원 판단(m+|Ω|)과 γ̄^rob의 최종 형태.
3. MILP 게임 자체(v^MIP,rob)의 core. copositive(Sec. 4.3) 방법이 max-min으로 확장되는가.

**보류한 결정.**

1. 본 모델: 이 탭의 유한 Ω DRO(KL).
2. Ambiguity set: KL(robust game, cut 생성 LP) vs CVaR(LP 직접, risk game으로 읽힘). 현재 방향은 KL이다.
3. 실시간 공동 정산 범위(2.5): 열·수소를 실시간으로 공동 정산하는가(물리 네트워크 가정), reserve를 멤버별 first-stage 분담으로 둘 수 있는가.
4. 3.4: (a)를 명제, (b)를 corollary나 example로. 3.5는 독립 corollary로 둘지 3.4에 합칠지.
5. 3.2의 minimax gap 문장을 본문에 둘지 remark로 둘지.

**다음 단계.**

- [ ] stochastic\_extension.py에 KL(또는 CVaR) DRO 옵션을 넣고 r = 0에서 현재 결과를 재현한다.
- [ ] make\_scenarios에 player별 독립 부하 오차 옵션을 넣는다.
- [ ] r(또는 β) sweep으로 ε, integrality gap, 협력 이득의 세 원천을 분해한다.
- [ ] 작은 n에서 --check-core로 명제를 수치 확인하고, contingent core의 반례를 n = 2, |Ω| = 2로 찾는다.
- [ ] Stochastic CG 가속: 시나리오별 dual warm start와 barrier RMP를 6인, |Ω| = 5에서 비교한다.

## 5. 참고문헌

기억에 의존한 목록이다. 인용 전에 서지를 확인해야 한다.

- Owen (1975), On the core of linear production games, Math. Programming.
- Kalai & Zemel (1982), Totally balanced games and games of flow, Math. of OR.
- Geoffrion (1974), Lagrangean relaxation for integer programming, Math. Programming Study.
- Sion (1958), On general minimax theorems, Pacific J. Math.
- Delbaen (2000), Coherent risk measures on general probability spaces. Denault (2001), Coherent allocation of risk capital, J. Risk. Tasche (1999), Risk contributions and performance measurement.
- Rockafellar & Uryasev (2000), Optimization of conditional value-at-risk, J. Risk.
- Hansen & Sargent, entropy 제약 robustness (구체 문헌 확인 필요).
- Bauso & Timmer (2009), Robust dynamic cooperative games, IJGT. Doan & Nguyen, Robust stable payoff distribution in stochastic cooperative games (서지 확인 필요).

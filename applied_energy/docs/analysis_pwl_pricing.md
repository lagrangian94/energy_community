# Convex Hull Pricing at the Kink: How CHP Captures Piecewise-Linear Non-Convexity

## 1. Setup

We compare IP pricing and convex hull pricing (CHP) for a local energy community with an electrolyzer (player u2). The electrolyzer's hydrogen production is modeled as a 6-segment piecewise-linear (PWL) function of electricity input:

$$H_2 = a_s \cdot p + b_s, \quad p \in [p_s, p_{s+1}], \quad s = 1, \dots, 6$$

Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $a_1, a_2$ | 21.12, 18.82 | Slope of segments 1, 2 (kgH₂/MWh) |
| $b_1, b_2$ | −0.379, +0.270 | Intercept of segments 1, 2 |
| $p_{\text{val}}^{(1)}$ | 0.2823 | Kink point (boundary between seg 1 and seg 2) |
| $P_{\min}$ | 0.15 | Minimum operating point when ON |
| $c_{su}$ | 0.0 | Startup cost (zero in this scenario) |

The slope decreases across segments ($a_1 > a_2 > \cdots$), reflecting diminishing marginal efficiency at higher loads. The negative intercept $b_1 = -0.379$ represents a fixed efficiency loss incurred whenever the electrolyzer is ON.

## 2. Observation: Price Differences Only When $z_{on}$ Differs

Comparing IP and CHP across all 24 hours reveals a clear pattern:

| Hours | IP $z_{on}$ | CHP $z_{on}$ | $z_{els,d}$ differs? | Price difference? |
|-------|------------|-------------|----------------------|-------------------|
| t=0,1,6,21–23 | 0 | 0 | No | **No** |
| t=2–5,7–8 | 0 | 0.09–0.36 | Yes | **Yes** (elec +7 to +12) |
| t=9 | 1.0 | 0.921 | Yes | **Yes** (elec +4.8, H₂ +0.47) |
| t=10–20 | 1.0 | 1.0 | Yes | **No** |

The critical control group is **t=10–20**: despite substantially different segment allocations (e.g., t=19: IP uses $[s_1\!=\!1]$, CHP uses $[s_1\!=\!0.26, s_2\!=\!0.74]$), prices are identical when $z_{on}$ agrees. This rules out PWL segment selection alone as the driver of price differences, and points to the **ON/OFF commitment variable** $z_{on}$ as the mediating factor.

## 3. The Mechanism at t=9

### 3.1 Dispatch comparison

|  | IP | CHP |
|--|-----|------|
| Electricity input ($fl_d$) | 0.2614 | 0.2600 |
| $z_{on}$ | 1.000 | 0.921 |
| $z_{els,d}$ | [1, 0, 0, 0, 0, 0] | [0.16, 0.76, 0, 0, 0, 0] |
| H₂ production | 5.142 | 5.142 |
| Electricity price | 71.20 | **76.00** (+4.80) |
| Hydrogen price | 3.42 | **3.89** (+0.47) |

Both methods achieve effectively the same dispatch (electricity ~0.260, H₂ = 5.142), but CHP assigns higher prices.

### 3.2 CHP's convex decomposition

CHP represents this dispatch as a convex combination of extreme points (columns):

- **92.1%** weight on ON columns ($z_{on} = 1$)
- **7.9%** weight on OFF columns ($z_{on} = 0$, zero consumption)

A key numerical finding: the average electricity consumption of the ON columns is

$$\frac{fl_d}{z_{on}} = \frac{0.2600}{0.9209} = 0.2823 = p_{\text{val}}^{(1)}$$

**All ON columns operate at the kink point.** Among these, 16.2% are labeled segment 1 (approaching the kink from below) and 75.9% are labeled segment 2 (approaching from above). Solving for the per-segment consumption confirms both cluster at $p_{\text{val}}^{(1)}$ within numerical tolerance ($< 10^{-4}$).

### 3.3 Why OFF columns are necessary

The kink point lies at $p_{\text{val}}^{(1)} = 0.2823$, but the optimal consumption is only 0.260. Since ON columns operate at 0.2823, mixing with OFF columns (consuming 0) is the only way to achieve a weighted average of 0.260:

$$0.921 \times 0.2823 + 0.079 \times 0 = 0.260 \checkmark$$

This is not merely a numerical artifact. The convex combination is strictly more efficient than operating directly at 0.260 in segment 1:

| Strategy | Electricity | H₂ produced |
|----------|------------|-------------|
| 100% ON @ 0.260 (segment 1) | 0.260 | $21.12 \times 0.260 - 0.379 = 5.113$ |
| 92.1% ON @ kink + 7.9% OFF | 0.260 | $0.921 \times 5.584 = 5.142$ |

The mix produces **0.58% more hydrogen** for the same electricity. This arises from the negative intercept $b_1 = -0.379$: a fixed efficiency loss incurred whenever the electrolyzer is ON. Operating at higher power for a shorter effective duration amortizes this fixed loss more efficiently than operating at lower power continuously.

### 3.4 Pricing at the subdifferential

At the kink point, the PWL production function is non-differentiable. The subdifferential is the interval $[a_2, a_1] = [18.82, 21.12]$, meaning the marginal hydrogen yield per unit electricity is ambiguous.

- **IP** fixes $z_{els,d} = [1, 0, \dots]$, locking the pricing LP into segment 1. The marginal efficiency is $a_1 = 21.12$ — a single, overly optimistic slope.
- **CHP** has columns from both segments simultaneously in the master LP basis. The dual variable must be consistent with both $a_1 = 21.12$ and $a_2 = 18.82$. Since 75.9% of the weight is on segment-2 columns (with the lower slope), the resulting price reflects the **diminishing returns** beyond the kink.

This yields higher CHP prices: the electricity price rises from 71.20 to 76.00 because the electrolyzer is a less efficient marginal consumer than IP assumes, and the hydrogen price rises from 3.42 to 3.89 because the marginal conversion cost is higher.

## 4. Addressing the "Unrealistic Operation" Concern

The fractional $z_{on} = 0.921$ and the ON+OFF mixing represent a **convex hull relaxation**, not an implementable operating schedule. In practice, at t=9, the electrolyzer is either fully ON or fully OFF.

However, this is by design. In the CHP framework:

- The **dispatch** (actual operation) follows the IP solution: $z_{on} = 1$, operating in segment 1 at 0.2614.
- Only the **prices** are derived from the convex hull relaxation.

The convex hull prices have a well-defined economic interpretation: they are the tightest linear (anonymous, uniform) prices that support the efficient allocation while **minimizing uplift** (side payments). The "unrealistic" convex combination is the mathematical device through which the opportunity cost of the ON/OFF non-convexity and the PWL kink are internalized into market-clearing prices.

## 5. Summary

The price difference between IP and CHP at t=9 arises from their differing treatment of the PWL kink at $p_{\text{val}}^{(1)} = 0.2823$:

1. **CHP concentrates all ON columns at the kink point**, where the production function has a non-smooth transition from slope 21.12 to 18.82.
2. **OFF columns are mixed in** to reach the target consumption of 0.260, exploiting the negative-intercept structure for higher effective efficiency.
3. **The dual price reflects the subdifferential** $[18.82, 21.12]$ at the kink, rather than the single slope $a_1 = 21.12$ that IP sees.
4. The control group (t=10–20) confirms that **segment allocation alone does not cause price differences** when $z_{on}$ is identical — the ON/OFF mixing at the kink is the essential mechanism.

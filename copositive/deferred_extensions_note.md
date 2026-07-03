# Deferred Coupling Extensions for CPP Reformulation

These coupling constraints from `note revised.txt` are deferred from the initial CPP implementation. Both are **homogeneous** (RHS = 0) and preserve the LPG structure.

## 1. Reserve Adequacy

**Constraint (per period t, up and down):**
```
r^sym - Σ_{i∈S} Σ_k r^{k,+}_{i,t} ≤ 0
r^sym - Σ_{i∈S} Σ_k r^{k,-}_{i,t} ≤ 0
```

**Common variable:** x_0 = (r^sym) — symmetric reserve quantity, controlled by whichever coalition operates. Not a player; receives no allocation.

**New coupling variables per player:** r^{k,±}_{i,t} (storage headroom offers) join x_i^cpl. These are bounded by existing storage constraints:
- r^{k,+}_{i,t} ≤ P_max^k - b^{dis,k}_{i,t}  (upward headroom)
- r^{k,-}_{i,t} ≤ P_max^k - b^{ch,k}_{i,t}   (downward headroom)

**Objective addition:** w_0^T x_0 = δ^res · r^sym (reserve revenue).

**CPP impact:**
- A_0 matrix (common variable coefficient) enters coupling: Σ_i A_i^cpl x_i^cpl + A_0 x_0 = 0
- Lifted block Y_0 for common variables (no binaries, polyhedral RLT)
- Cross-block vanishing extends: prosumer-common blocks X_{i0} vanish by the same finiteness mechanism (note revised, eq. 28-29)
- Owen allocation defined over prosumers only; x_0 contributes zero endowment

## 2. Peak Power Penalty

**Constraint (per period t, electricity only):**
```
Σ_{i∈S} (1/ΔT)(i^{E,mkt}_{i,t} - e^{E,mkt}_{i,t}) - p ≤ 0
```

**Common variable:** p — community peak import power. Not a player.

**New coupling variables per player:** Grid exchange variables i^{E,mkt}_{i,t}, e^{E,mkt}_{i,t} (or equivalently, i^{E,gri}_{i,t}, e^{E,gri}_{i,t} in the code) join x_i^cpl.

**Objective addition:** -δ^peak · p (peak penalty cost).

**CPP impact:** Same as reserve — A_0 gains a column for p, coupling gains rows, cross-block vanishing extends.

## 3. Implementation Priority

1. **Current:** Community balance only (3 carriers × T periods = 72 constraints)
2. **Next:** Reserve adequacy (adds ~2T constraints + common variable block)
3. **Later:** Peak penalty (adds T constraints + common variable)

Both extensions are homogeneous, so they slot into the existing framework mechanically. The key code changes would be:
- Add r^{k,±}_{i,t} variables to storage players (new coupling vars)
- Add x_0 = (r^sym, p) as a shared block
- Extend the A^cpl matrix with reserve/peak rows
- Extend the lifted matrix Y_S with the common-variable block Y_0

---
name: COP implementation progress
description: Copositive programming implementation for energy community core allocation - current status, issues, and decisions
type: project
---

## Project: Copositive Core Allocation for Energy Community

### Goal
Implement cutting-plane algorithms (Guo et al. baseline + Gabl & Anstreicher enhanced) to solve the copositive dual (COPOPT) of the energy community CPP, recovering core allocations for the mixed-integer cooperative game.

### Key Files
- `energy_community/copositive/cpp_reformulation.tex` — Complete LaTeX document: CPP reformulation + algorithm design (Guo baseline + G&A enhanced, with Class A/B constraint partition)
- `energy_community/copositive/note revised.txt` — Theoretical framework (note revised)
- `energy_community/copositive/cop_guo.py` — Guo et al. implementation (~870 lines). S-explicit master + MILPCOP separation. Runs but cuts don't converge.
- `energy_community/copositive/cop_ga.py` — Gabl & Anstreicher version (NOT YET CREATED)
- `energy_community/copositive/deferred_extensions_note.md` — Reserve/peak extensions (future work)

### Environment
- Virtual env: `/home/user/myfolder/.venv/` (PySCIPOpt 5.2.1, numpy, scipy, gurobipy)
- **SCIP**: import OK, model creation OK, BUT `optimize()` segfaults (LP solver issue). Use ONLY for model creation + MPS export.
- **Gurobi**: WLS academic license (ID 926906). Used for ALL optimization (LP, MILP, QCP). SOCP via quadratic constraints (not native cone API).
- Working dir: `/home/user/energy_community`
- Run: `source /home/user/myfolder/.venv/bin/activate && python -u copositive/cop_guo.py`
- `data_generator.setup_lem_parameters(players, config, time_periods)` — argument order: config before time_periods

### 6-Player MIP Facts
- Players: u1(wind+ESS_E), u2(electrolyzer+ESS_G), u3(HP+ESS_H), u4(D_E), u5(D_G), u6(D_H)
- 1272 variables (936 continuous, 336 binary), 1250 constraints
- Gurobi solves in 0.38s: obj = -2349.66 (cost min) → v^MIP = 2349.66 (profit max)
- After slacking: n_slacked=2591, nc=2592 (dimension of Y)
- Equality rows: 1994 (Class A), CPOPT constraints total: 4324 (1994 A + 1994 RLT + 336 binary diag)

### Design Decisions
1. **Technology-modular CPP**: per-technology reformulation, any player combination works mechanically.
2. **Profit maximization** convention (matching note revised)
3. **Community balance signs unified**: all carriers Σ(i-e)=0
4. **Only community balance coupling** (reserve/peak deferred)
5. **Gabl & Anstreicher** as primary algorithm (MILPCOP(K) with Au=0)
6. **Class A vs B partition**: K = first-order linear equalities ONLY. RLT, binary diagonal, squared coupling are Class B (CPOPT only, NOT in K).

### cop_guo.py Architecture (current state)

**CPOPTBuilder** (lines 58-460):
- Step 1: Build SCIP model → export MPS → read in Gurobi
- Step 2: Solve MIP with Gurobi (v^MIP = 2349.66)
- Step 3: Extract variable/constraint structure from Gurobi model
- Step 4: Build equality form (add inequality slacks, ub slacks, binary complements)
- Step 5: Identify coupling constraints (community balance by name pattern)
- Step 6: Homogenize (ā = (-b; a) for each a^T x = b)
- Step 7-9: Build objective, C matrix, CPOPT constraint matrices (B_l)

**GUOSolver** (lines 465-):
- **Master**: S-explicit with Gurobi. 3.36M S variables (upper triangle of 2592×2592) + 4325 v variables. Linking: S = C - v_norm·E00 - Σ v_l·B_l. Box bounds |S_{ij}| ≤ 5000. Diagonal S_{jj} ≥ 0.
- **Separation**: MILPCOP with nc=2592 binary variables. ~30s per solve.
- **Cuts**: u^T S u ≥ 0 added as linear constraints on S variables.
- **Complementary slackness**: Tr(x* x*^T S) ≥ 0 from MIP solution (Guo Sec 5.1).

### Sign Convention Issue (RESOLVED)
- CPOPT minimises C•Y where C = -W_S. At optimum, v_norm = val(CPOPT) = -v^MIP.
- So optimal v_norm = -2349.66 (negative). COPOPT maximises v_norm.
- Gap = (v_norm - (-v^MIP)) / |v^MIP|.

### Current Results

**After external review fix**: comp_slack removed (it was pinning obj and preventing master from exploring S).

**Small instance (u1+u4, T=4, nc=61, no binaries), 30 iterations, 0.7s:**
```
Iter 1:  obj=161,538  γ=2484
Iter 5:  obj=154,430  γ=4974
Iter 10: obj=150,692  γ=906
Iter 20: obj=143,493  γ=907
Iter 30: obj=140,749  γ=907   (target=-282, gap=500x)
```
- obj is DECREASING from above (correct Guo behavior!)
- γ drops from 2484 → 906 then stagnates (possible duplicate cuts despite filter)
- Convergence is very slow — expected limitation of Guo's method
- This is the baseline behavior that G&A's Au=0 should improve

**Root cause of previous stagnation**: comp_slack cut Tr(x*x*^T S) ≥ 0 was pinning v_norm = -v^MIP from iter 1, preventing master from moving S. Without it, master correctly starts at S_BOUND and descends.

### Review Outcome (2026-06-30)
- **Formulation is correct.** Sign convention, B_l matrices, separation MILP all verified.
- **Bug found**: comp_slack cut was pinning obj, preventing master from exploring. FIXED by removing it.
- **Remaining issues**: slow convergence (Guo baseline limitation), γ stagnation (duplicate cuts not fully filtered). These are expected and motivate G&A's Au=0 approach.

### Next Steps
1. **External review** of cop_guo.py (correctness of formulation + separation)
2. **Create cop_ga.py**: same CPOPTBuilder, MILPCOP(K) with Au=0 in separation
3. **Compare**: does Au=0 improve γ convergence?
4. **Debug**: check if duplicate cuts are being generated; add cut deduplication

### Theoretical References
- Guo, Bodur & Taylor (2024): baseline cutting-plane (MILPCOP, SOCP strengthening)
- Gabl & Anstreicher (2025, Math Prog B): set-copositive MILPCOP(K) with equality structure
- Bomze & Gabl (2023): Burer's theorem (Thm 4), key assumptions, dual attainment
- Cifuentes, Dey & Xu (2025): strong duality for bounded MBQPs
- Liu, Qi & Xu (2016): Lagrangian relaxation bound for cooperative games

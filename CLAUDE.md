# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying **core-selecting mechanism design for local energy communities (LEM)** using cooperative game theory and optimization. The system models multi-energy-carrier communities (electricity, hydrogen, heat) with players owning different devices (renewables, electrolyzers, heat pumps, storage, demand).

## Environment Setup

```bash
# Virtual environment is in parent directory
source ../.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:** `pyscipopt` (SCIP solver), `numpy`, `pandas`, `matplotlib`, `scipy`

## Running

```bash
# Full IP analysis with core computation
python analysis_mip.py

# LP relaxation analysis
python analysis_lp.py

# Column generation (convex hull pricing)
python chp.py

# Sensitivity analysis
python sensitivity_analysis.py
```

## Architecture

### Optimization Model
- **compact_utility.py** — `LocalEnergyMarket` class: complete LEM formulation with energy balance constraints, grid/community trading, device commitments (binary variables), piecewise-linear efficiency approximations. Uses PySCIPOpt as the solver.

### Solution Methods (three pricing approaches compared)
- **core.py** — `CoreComputation`: row generation algorithm for core allocation in cooperative games. `SeparationProblem` finds most violated coalitions; iterates until no coalition can improve by deviating.
- **chp.py** — `ColumnGenerationSolver`: Dantzig-Wolfe decomposition for convex hull pricing. Decomposes the community problem into per-player subproblems.
- **solver.py** — `PlayerSubproblem` and `MasterProblem`: column generation components. Subproblems solve individual player optimization with dual prices; master problem manages convex combinations of extreme points.
- **pricer.py** — SCIP pricer plugin for automated column generation.

### Data Generation
- **data_generator.py** — `setup_lem_parameters()`: configurable parameter setup supporting sensitivity analysis.
- **ElecGen.py / HeatGen.py / HydroGen.py / ElsGen.py** — Load profiles, market prices, and device parameter generators. Supports Korean (Jeju) and Danish (DK2) market data from CSV files in `data/`.

### Analysis & Visualization
- **analysis_mip.py / analysis_lp.py** — Entry points that run the model, compute core allocations, and generate results.
- **sensitivity_analysis.py** — Parametric sweeps over HP/electrolyzer/storage capacities, import prices, etc.
- **visualize.py** — Plotting utilities.

### Player Configuration
Standard 6-player setup: u1 (wind + elec storage), u2 (electrolyzer + H2 storage), u3 (heat pump + heat storage), u4 (elec demand), u5 (H2 demand), u6 (heat demand). 24-hour time horizon.

## Key Concepts

- **Core stability**: payoff allocation where no coalition prefers to deviate from the grand coalition.
- **Row generation**: iteratively adds coalition constraints to find core allocations.
- **Column generation / Dantzig-Wolfe**: decomposes community problem into player subproblems; master problem finds convex hull prices.
- **Three pricing methods compared**: IP (integer programming), CHP (convex hull pricing), LP (linear relaxation).

## Output Artifacts

Results are saved as `.pkl` (pickled objects), `.json`, `.csv`, `.png` (plots), and `.tex` (LaTeX tables). Result directories include `results_53/` and `with_ess/`.

## Configuration

Model parameters are stored in `parameters.json`. Sensitivity analysis parameters (capacity ratios, price factors, seasons) are configured in `data_generator.py`.

"""
Guo et al.'s cutting-plane algorithm for solving the copositive dual (COPOPT)
of the energy community completely positive program (CPP).

Reference:
    Guo, C., Bodur, M., & Taylor, J. A. (2021).
    Copositive duality for discrete energy markets.

PRIMAL vs. DUAL -- who builds what (this is the key distinction):
    * CPOPTBuilder builds the PRIMAL CPP data only.  Its functions
      (_build_equality_form, _homogenize, _build_C_matrix,
      _build_cpopt_constraints) assemble the coefficients of the primal CPP
      (see cpp_reformulation.tex, eq:cpp):

          min  C . Y   s.t.  B_l . Y = b_l  (l=1..m),  Y_00 = 1,  Y in CP^{1+n}

      i.e. C, {B_l, b_l}, the homogenised A, and the normalisation.  The
      equality form + RLT identities exist to produce the PRIMAL constraints
      B_l . Y = b_l; the builder never constructs a dual.  This primal data is
      shared: DNNRelaxationSolver (sdp_relax.py) consumes the SAME C, {B_l, b_l}
      and just relaxes the CP cone to DNN.

    * GUOSolver does NOT build data -- it takes the builder's primal C, {B_l, b_l}
      and forms + solves the LAGRANGIAN DUAL (COPOPT) on the fly inside solve().
      The dual coefficients ARE the primal C and B_l, so no separate "dual
      builder" is needed.  The dual (see solve(), the master-model section) is

          max  v_norm   s.t.  S = C - v_norm * e0 e0^T - sum_l v_l B_l  in COP^{1+n}

      where v_norm dualises Y_00 = 1, v_l dualises B_l . Y = b_l, and COP is the
      dual cone of CP.  S in COP is enforced by outer-approximation cutting
      planes (MILP separation).

Strategy (CPOPTBuilder pipeline):
    1. Build the SCIP model via LocalEnergyMarket (do NOT call optimize -- segfaults).
    2. Export to MPS, read in Gurobi.
    3. From the Gurobi model extract variable/constraint structure.
    4. Convert to equality form (slacks for inequalities, binary complements).
    5. Homogenize and build the PRIMAL CPP data structures (C, {B_l, b_l}).
    6. (GUOSolver) Form the COPOPT dual and solve it via outer-approximation
       cutting planes using Gurobi for both the LP master and the MILP separation.

Classes:
    CPOPTBuilder  -- Builds the PRIMAL CPP data matrices (C, {B_l, b_l}) from the
                     MPS-extracted Gurobi model
    GUOSolver     -- Forms the COPOPT dual from that primal data and solves it
                     with an outer-approximation cutting-plane algorithm
"""

import sys
import os
import time as _time
import tempfile
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Ensure parent directory is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/user/myfolder/energy_community")

import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sparse_outer(a, b):
    """Return sparse matrix a @ b^T for 1-D arrays a, b."""
    a_col = sp.csc_matrix(a.reshape(-1, 1))
    b_col = sp.csc_matrix(b.reshape(-1, 1))
    return a_col @ b_col.T


def _sym_outer(a, b):
    """0.5 * (a b^T + b a^T) as sparse."""
    return 0.5 * (_sparse_outer(a, b) + _sparse_outer(b, a))


# ===================================================================
# CPOPTBuilder
# ===================================================================
class CPOPTBuilder:
    """
    Build CPOPT data from the LEM model via MPS export.

    After construction provides:
        nc              -- dimension of Y  (1 + n_slacked)
        n_orig          -- number of original MIP variables
        n_slacked       -- total variables in the slacked equality form
        n_binary        -- number of binary variables
        binary_indices  -- indices of binary vars in the slacked vector (1-indexed in y)
        A_homo          -- full homogenised equality matrix (p x nc, sparse)
        A_rows          -- list of homogenised rows as dense (nc,) vectors
        coupling_row_indices -- which rows of A_homo are community-balance coupling
        w_profit        -- profit objective vector (nc,) with w[0]=0
        C               -- objective matrix for CPOPT (nc x nc, sparse)
        vmip            -- v^MIP solved with Gurobi
        gurobi_model    -- the Gurobi MIP model (for reference)
    """

    def __init__(self, players, time_periods, params, config, lift_slacks=True):
        self.players = players
        self.T = time_periods
        self.params = params
        self.config = config
        # lift_slacks=True  -> exact CPP-eq lifting: slacks/complements are
        #   lifted, all inequalities become equalities, every row gets an RLT
        #   identity (nc = 1 + n_slacked).  Required by GUOSolver.
        # lift_slacks=False -> relaxed CPP-ineq lifting (Zhao et al. 2026,
        #   "ADMM-based decomposed DNN+RLT", Sec. 4.1): lift ONLY the original
        #   variables (nc = 1 + n_orig, much smaller); keep inequalities and
        #   variable bounds as linear/diagonal inequalities on Y, and keep RLT
        #   only for the equality rows.  Looser but far more tractable; used by
        #   the DNN solver, NOT by GUOSolver.
        self.lift_slacks = lift_slacks

        # Step 1: Build SCIP model, export MPS, read in Gurobi
        self._build_gurobi_model()

        # Step 2: Solve with Gurobi to get vmip
        self._solve_mip()

        # Step 3: Extract model structure
        self._extract_structure()

        # Step 4: Build the lifting form (equality- or inequality-based)
        if self.lift_slacks:
            self._build_equality_form()
        else:
            self._build_inequality_form()

        # Step 5: Identify coupling constraints
        self._identify_coupling()

        # Step 6: Homogenize the equality rows (and, in CPP-ineq, the
        # inequality rows too)
        self._homogenize()
        if not self.lift_slacks:
            self._homogenize_ineq()

        # Step 7: Build objective
        self._build_objective()

        # Step 8: Build C matrix
        self._build_C_matrix()

        # Step 9: Build CPOPT equality constraint matrices (Class A / RLT /
        # binary diag).  In CPP-ineq mode, also the inequality matrices.
        self._build_cpopt_constraints()
        if not self.lift_slacks:
            self._build_cpopt_inequalities()

    # ------------------------------------------------------------------
    # Step 1: Build SCIP model, export MPS, read into Gurobi
    # ------------------------------------------------------------------
    def _build_gurobi_model(self):
        from compact_utility import LocalEnergyMarket

        print("[CPOPTBuilder] Building SCIP model ...")
        lem = LocalEnergyMarket(self.players, self.T, self.params, model_type="mip")

        mps_path = os.path.join(tempfile.gettempdir(), "lem_cop.mps")
        lem.model.writeProblem(mps_path)
        print(f"[CPOPTBuilder] Exported MPS to {mps_path}")

        # Read into Gurobi
        self.gurobi_model = gp.read(mps_path)
        print(f"[CPOPTBuilder] Gurobi model: {self.gurobi_model.NumVars} vars, "
              f"{self.gurobi_model.NumConstrs} constrs")

    # ------------------------------------------------------------------
    # Step 2: Solve MIP with Gurobi
    # ------------------------------------------------------------------
    def _solve_mip(self):
        print("[CPOPTBuilder] Solving MIP with Gurobi ...")
        self.gurobi_model.setParam("OutputFlag", 1)
        self.gurobi_model.optimize()

        if self.gurobi_model.Status == GRB.OPTIMAL:
            # SCIP/MPS model minimises cost; profit = -cost
            self.vmip = -self.gurobi_model.ObjVal
            print(f"[CPOPTBuilder] Gurobi obj = {self.gurobi_model.ObjVal:.6f}, "
                  f"v^MIP = {self.vmip:.6f}")
        else:
            print(f"[CPOPTBuilder] Gurobi status = {self.gurobi_model.Status}")
            self.vmip = None

    # ------------------------------------------------------------------
    # Step 3: Extract variable/constraint structure from Gurobi
    # ------------------------------------------------------------------
    def _extract_structure(self):
        mdl = self.gurobi_model
        self.vars_list = mdl.getVars()
        self.n_orig = len(self.vars_list)

        # Variable info arrays
        self.var_names_orig = [v.VarName for v in self.vars_list]
        self.var_lbs = np.array([v.LB for v in self.vars_list])
        self.var_ubs = np.array([v.UB for v in self.vars_list])
        self.var_objs = np.array([v.Obj for v in self.vars_list])
        self.var_types = [v.VType for v in self.vars_list]

        # Name -> original index
        self.name_to_orig = {v.VarName: i for i, v in enumerate(self.vars_list)}

        # Binary variable original indices
        self.binary_orig = [i for i, vt in enumerate(self.var_types)
                            if vt == GRB.BINARY]

        # Constraint info
        self.constrs = mdl.getConstrs()
        self.n_constrs = len(self.constrs)
        self.constr_names = [c.ConstrName for c in self.constrs]
        self.constr_senses = [c.Sense for c in self.constrs]
        self.constr_rhs = np.array([c.RHS for c in self.constrs])

        # Build constraint matrix rows (as sparse)
        # Each row: coefficients for original variables
        self.A_orig_rows = []
        for c in self.constrs:
            row = mdl.getRow(c)
            indices = []
            vals = []
            for k in range(row.size()):
                var = row.getVar(k)
                col_idx = self.name_to_orig[var.VarName]
                indices.append(col_idx)
                vals.append(row.getCoeff(k))
            sparse_row = sp.csr_matrix(
                (vals, ([0] * len(indices), indices)),
                shape=(1, self.n_orig)
            )
            self.A_orig_rows.append(sparse_row)

        print(f"[CPOPTBuilder] Original: {self.n_orig} vars "
              f"({len(self.binary_orig)} binary), {self.n_constrs} constrs")

    # ------------------------------------------------------------------
    # Step 4: Build equality form
    # ------------------------------------------------------------------
    def _build_equality_form(self):
        """
        Convert to equality form:
        - For each '<' constraint: add slack s >= 0, a^T x + s = b
        - For each '>' constraint: add slack s >= 0, a^T x - s = b
        - '=' constraints stay as-is.
        - For each variable with finite ub (not binary): add x_j + sigma_j = ub_j
        - For each binary variable: add z_j + sigma_j = 1  (complement)
        - Shift variables with lb > 0: x' = x - lb, adjust constraints.

        All variables in the slacked form have lb=0.
        """
        n_orig = self.n_orig

        # ---- Determine additional variables needed ----
        # Slack vars for inequality constraints
        ineq_slacks = []
        for i, sense in enumerate(self.constr_senses):
            if sense == '<' or sense == '>':
                ineq_slacks.append(i)

        # Upper-bound slacks for non-binary variables with finite ub
        ub_slacks = []
        for j in range(n_orig):
            if j in set(self.binary_orig):
                continue  # handled separately
            ub = self.var_ubs[j]
            if np.isfinite(ub) and ub > 0:
                ub_slacks.append(j)

        # Binary complements
        bin_complements = list(self.binary_orig)

        n_ineq_slack = len(ineq_slacks)
        n_ub_slack = len(ub_slacks)
        n_bin_compl = len(bin_complements)

        self.n_slacked = n_orig + n_ineq_slack + n_ub_slack + n_bin_compl
        self.nc = 1 + self.n_slacked  # y = (y0=1; x_slacked)
        self.n_binary = len(self.binary_orig)

        print(f"[CPOPTBuilder] Slacked form: n_slacked={self.n_slacked}, nc={self.nc}")
        print(f"  ineq_slacks={n_ineq_slack}, ub_slacks={n_ub_slack}, "
              f"bin_complements={n_bin_compl}")

        # ---- Map: position in the slacked x vector ----
        # Original variables: indices 0..n_orig-1  (in x-space, 1..n_orig in y-space)
        # Inequality slacks: n_orig .. n_orig + n_ineq_slack - 1
        # UB slacks: ...
        # Binary complements: ...

        slack_offset = n_orig
        self.ineq_slack_map = {}  # constr_idx -> slacked_x_idx
        for k, ci in enumerate(ineq_slacks):
            self.ineq_slack_map[ci] = slack_offset + k

        ub_offset = slack_offset + n_ineq_slack
        self.ub_slack_map = {}  # orig_var_idx -> slacked_x_idx
        for k, vi in enumerate(ub_slacks):
            self.ub_slack_map[vi] = ub_offset + k

        compl_offset = ub_offset + n_ub_slack
        self.bin_compl_map = {}  # orig_var_idx -> slacked_x_idx
        for k, bi in enumerate(bin_complements):
            self.bin_compl_map[bi] = compl_offset + k

        # Binary indices in y-space (1-indexed)
        self.binary_indices = [bi + 1 for bi in self.binary_orig]

        # ---- Handle variable lower bounds ----
        # Shift x' = x - lb for variables with lb > 0.
        # New ub' = ub - lb.  Constraints: a^T (x'+lb) sense b => a^T x' sense (b - a^T lb)
        self.lb_shift = np.copy(self.var_lbs)
        # Set lb_shift to 0 for vars with lb=0 or lb=-inf
        self.lb_shift[self.lb_shift <= 0] = 0.0
        self.lb_shift[~np.isfinite(self.lb_shift)] = 0.0

        # ---- Build equality rows in the slacked system ----
        # Each row is a dense vector of length n_slacked.
        # RHS is also stored for homogenization.

        eq_rows = []   # list of (coeffs_array[n_slacked], rhs_value, label)

        # (a) Original constraints (converted to equality)
        for i in range(self.n_constrs):
            row_orig = np.array(self.A_orig_rows[i].todense()).flatten()
            # Adjust RHS for lower-bound shift: b' = b - a^T lb_shift
            rhs = self.constr_rhs[i] - np.dot(row_orig, self.lb_shift)

            coeffs = np.zeros(self.n_slacked)
            coeffs[:n_orig] = row_orig

            sense = self.constr_senses[i]
            if sense == '<':
                # a^T x' + s = rhs  (s >= 0)
                coeffs[self.ineq_slack_map[i]] = 1.0
            elif sense == '>':
                # a^T x' - s = rhs  (s >= 0)
                coeffs[self.ineq_slack_map[i]] = -1.0
            # else: '=' already

            eq_rows.append((coeffs, rhs, self.constr_names[i]))

        # (b) Upper-bound slacks: x'_j + sigma_j = ub'_j
        for vi, si in self.ub_slack_map.items():
            coeffs = np.zeros(self.n_slacked)
            coeffs[vi] = 1.0
            coeffs[si] = 1.0
            ub_shifted = self.var_ubs[vi] - self.lb_shift[vi]
            eq_rows.append((coeffs, ub_shifted, f"ub_slack_{self.var_names_orig[vi]}"))

        # (c) Binary complements: z_j + sigma_j = 1
        for bi, ci in self.bin_compl_map.items():
            coeffs = np.zeros(self.n_slacked)
            coeffs[bi] = 1.0
            coeffs[ci] = 1.0
            eq_rows.append((coeffs, 1.0, f"bin_compl_{self.var_names_orig[bi]}"))

        self.eq_rows = eq_rows
        self.n_equalities = len(eq_rows)
        print(f"[CPOPTBuilder] Total equalities: {self.n_equalities}")

    # ------------------------------------------------------------------
    # Step 4' (CPP-ineq): inequality form -- lift only original variables
    # ------------------------------------------------------------------
    def _build_inequality_form(self):
        """Relaxed CPP-ineq lifting (Zhao et al. 2026, Sec. 4.1).

        Lift ONLY the original variables (no slacks, no binary complements),
        so nc = 1 + n_orig instead of 1 + n_slacked.  Constraints:
          * '=' rows          -> kept as equalities (first-order + RLT later).
          * '<' / '>' rows     -> kept as LINEAR inequalities a^T x <= b on Y
                                  (no slack, no RLT).
          * finite upper bound -> linear  x_j <= ub  plus the lifted diagonal
                                  bound  X_jj <= ub^2  (continuous vars only;
                                  binaries already get X_jj = x_j).
          * binaries           -> diagonal identity X_jj = x_j (as before).

        This is a valid relaxation of the MILP: any MILP-feasible x with
        X = x x^T satisfies all of the above, so -val(DNN) stays an upper
        bound on v^MIP (just looser than the exact CPP-eq lift).
        """
        n_orig = self.n_orig
        self.n_slacked = n_orig          # nothing extra is lifted
        self.nc = 1 + n_orig
        self.n_binary = len(self.binary_orig)

        print(f"[CPOPTBuilder] CPP-ineq form: n_orig={n_orig}, nc={self.nc} "
              f"(no slacks lifted)")

        # Lower-bound shift x' = x - lb for vars with lb > 0 (same convention
        # as the equality form; vars with lb<=0 or -inf are left as-is).
        self.lb_shift = np.copy(self.var_lbs)
        self.lb_shift[self.lb_shift <= 0] = 0.0
        self.lb_shift[~np.isfinite(self.lb_shift)] = 0.0

        # Binary indices in y-space (1-indexed)
        self.binary_indices = [bi + 1 for bi in self.binary_orig]
        binset = set(self.binary_orig)

        eq_rows = []      # (coeffs[n_orig], rhs, label)          -- '=' rows
        ineq_rows = []    # (coeffs[n_orig], rhs, label): a^T x <= rhs
        diag_bounds = []  # (orig_var_idx, ub_shifted^2, label): X_jj <= ub^2

        # (a) original constraints
        for i in range(self.n_constrs):
            row_orig = np.array(self.A_orig_rows[i].todense()).flatten()
            rhs = self.constr_rhs[i] - np.dot(row_orig, self.lb_shift)
            sense = self.constr_senses[i]
            label = self.constr_names[i]
            if sense == '=':
                eq_rows.append((row_orig.copy(), rhs, label))
            elif sense == '<':
                ineq_rows.append((row_orig.copy(), rhs, label))
            elif sense == '>':
                # a^T x >= rhs  <=>  (-a)^T x <= -rhs
                ineq_rows.append((-row_orig, -rhs, label))

        # (b) variable upper bounds
        for j in range(n_orig):
            ub = self.var_ubs[j]
            if not np.isfinite(ub):
                continue
            ub_sh = ub - self.lb_shift[j]
            coeffs = np.zeros(n_orig)
            coeffs[j] = 1.0
            name = self.var_names_orig[j]
            if j in binset:
                # z_j <= 1 (X_jj = z_j is added as the binary diagonal)
                ineq_rows.append((coeffs, ub_sh, f"ub_{name}"))
            elif ub_sh > 0:
                ineq_rows.append((coeffs, ub_sh, f"ub_{name}"))
                diag_bounds.append((j, ub_sh * ub_sh, f"diag_{name}"))

        self.eq_rows = eq_rows
        self.ineq_rows = ineq_rows
        self.diag_bounds = diag_bounds
        self.n_equalities = len(eq_rows)
        print(f"[CPOPTBuilder] CPP-ineq: {self.n_equalities} equalities, "
              f"{len(ineq_rows)} linear inequalities, "
              f"{len(diag_bounds)} diagonal bounds")

    # ------------------------------------------------------------------
    # Step 5: Identify coupling constraints
    # ------------------------------------------------------------------
    def _identify_coupling(self):
        """Find community balance constraints by name pattern."""
        self.coupling_row_indices = []
        for idx, (_, _, label) in enumerate(self.eq_rows):
            if ("community_elec_balance" in label or
                    "community_heat_balance" in label or
                    "community_hydro_balance" in label):
                self.coupling_row_indices.append(idx)

        print(f"[CPOPTBuilder] Coupling constraints: {len(self.coupling_row_indices)}")

    # ------------------------------------------------------------------
    # Step 6: Homogenize
    # ------------------------------------------------------------------
    def _homogenize(self):
        """
        For each equality a^T x = b, create homogenised row:
            a_bar = (-b, a_1, ..., a_n)
        so that a_bar^T y = 0  where y = (1; x).
        """
        nc = self.nc
        rows_dense = []
        self.A_rows = []
        self._eq_labels = []

        for coeffs, rhs, label in self.eq_rows:
            a_bar = np.zeros(nc)
            a_bar[0] = -rhs
            a_bar[1:] = coeffs
            rows_dense.append(a_bar)
            self.A_rows.append(a_bar)
            self._eq_labels.append(label)

        if rows_dense:
            self.A_homo = sp.csr_matrix(np.array(rows_dense))
        else:
            self.A_homo = sp.csr_matrix((0, nc))

        print(f"[CPOPTBuilder] Homogenised A: {self.A_homo.shape}")

    # ------------------------------------------------------------------
    # Step 6' (CPP-ineq): homogenize the linear inequality rows
    # ------------------------------------------------------------------
    def _homogenize_ineq(self):
        """Homogenize each  a^T x <= b  into  a_bar^T y <= 0  with
        a_bar = (-b, a) and y = (1; x)."""
        nc = self.nc
        self.Aineq_rows = []
        self._ineq_labels = []
        for coeffs, rhs, label in self.ineq_rows:
            a_bar = np.zeros(nc)
            a_bar[0] = -rhs
            a_bar[1:] = coeffs
            self.Aineq_rows.append(a_bar)
            self._ineq_labels.append(label)
        print(f"[CPOPTBuilder] Homogenised inequalities: {len(self.Aineq_rows)}")

    # ------------------------------------------------------------------
    # Step 7: Build objective vector
    # ------------------------------------------------------------------
    def _build_objective(self):
        """
        Build w_profit in y-space.
        MIP minimises cost, so profit coeff = -obj_coeff.
        Only original variables have nonzero profit coefficients.
        Adjust for lb shift:  w^T x = w^T (x' + lb) = w^T x' + w^T lb.
        The constant w^T lb shifts vmip but doesn't appear in y since
        we absorb it in the normalization.
        """
        nc = self.nc
        self.w_profit = np.zeros(nc)
        # w_profit[0] = 0  (for y_0)
        for j in range(self.n_orig):
            # profit = -cost_coeff
            self.w_profit[j + 1] = -self.var_objs[j]

        # The constant from lb shift: sum_j (-obj_j) * lb_shift_j
        # This is already accounted for in vmip since Gurobi solves the
        # original (unshifted) problem.

        nz = np.count_nonzero(self.w_profit)
        print(f"[CPOPTBuilder] Objective w_profit: {nz} nonzeros, "
              f"range=[{self.w_profit.min():.6f}, {self.w_profit.max():.6f}]")

    # ------------------------------------------------------------------
    # Step 8: Build C matrix
    # ------------------------------------------------------------------
    def _build_C_matrix(self):
        """
        C = -W_S where W_S = 0.5 * (e_0 w^T + w e_0^T)
        So C[0,j] = C[j,0] = -0.5 * w_j  and C[i,j]=0 for i,j >= 1.
        """
        nc = self.nc
        e0 = np.zeros(nc)
        e0[0] = 1.0
        w = self.w_profit
        self.C = -0.5 * (_sparse_outer(e0, w) + _sparse_outer(w, e0))
        self.C = sp.csr_matrix(self.C)

    # ------------------------------------------------------------------
    # Step 9: Build CPOPT constraint matrices (B_l, b_l)
    # ------------------------------------------------------------------
    def _build_cpopt_constraints(self):
        """
        Build all CPOPT constraint matrices:
          Class A (first-order):  B_l = 0.5*(a_bar e_0^T + e_0 a_bar^T), b_l = 0
          Class B RLT:            B_l = a_bar a_bar^T, b_l = 0
          Class B binary diag:    B_l = e_m e_m^T - 0.5*(e_m e_0^T + e_0 e_m^T), b_l = 0
          Class B coupling^2:     (already included in RLT for coupling rows)
        """
        nc = self.nc
        e0 = np.zeros(nc)
        e0[0] = 1.0

        self.B_list = []     # list of sparse (nc, nc) matrices
        self.b_list = []     # list of floats
        self.bl_labels = []

        # --- Class A: first-order from each equality row ---
        for r in range(self.n_equalities):
            a_bar = self.A_rows[r]
            B_l = _sym_outer(a_bar, e0)
            self.B_list.append(B_l)
            self.b_list.append(0.0)
            self.bl_labels.append(f"A_{self._eq_labels[r]}")

        n_class_a = len(self.B_list)

        # --- Class B RLT: a_bar a_bar^T for each equality row ---
        for r in range(self.n_equalities):
            a_bar = self.A_rows[r]
            B_l = _sparse_outer(a_bar, a_bar)
            self.B_list.append(B_l)
            self.b_list.append(0.0)
            self.bl_labels.append(f"B_RLT_{self._eq_labels[r]}")

        n_class_b_rlt = len(self.B_list) - n_class_a

        # --- Class B binary diagonal: e_m e_m^T - 0.5*(e_m e_0^T + e_0 e_m^T) ---
        for m in self.binary_indices:
            em = np.zeros(nc)
            em[m] = 1.0
            B_l = _sparse_outer(em, em) - _sym_outer(em, e0)
            self.B_list.append(B_l)
            self.b_list.append(0.0)
            self.bl_labels.append(f"B_bindiag_{m}")

        n_class_b_bin = len(self.B_list) - n_class_a - n_class_b_rlt

        n_total = len(self.B_list)
        print(f"[CPOPTBuilder] CPOPT constraints: {n_total}")
        print(f"  Class A (first-order): {n_class_a}")
        print(f"  Class B RLT: {n_class_b_rlt}")
        print(f"  Class B binary diag: {n_class_b_bin}")

    # ------------------------------------------------------------------
    # Step 9' (CPP-ineq): inequality constraint matrices
    # ------------------------------------------------------------------
    def _build_cpopt_inequalities(self):
        """Build the inequality constraint data for the CPP-ineq DNN model:
          * linear inequalities  a_bar^T y <= 0  ->  <0.5(a_bar e0^T + e0 a_bar^T), Y> <= 0
          * diagonal bounds      X_jj <= ub^2    ->  <e_m e_m^T, Y> <= ub^2  (m = j+1)
        Stored as (Bineq_list, bineq_list) with the '<=' sense implicit.
        """
        nc = self.nc
        e0 = np.zeros(nc)
        e0[0] = 1.0

        self.Bineq_list = []
        self.bineq_list = []
        self.bineq_labels = []

        # --- linear inequalities (first-order, <= 0) ---
        for a_bar, label in zip(self.Aineq_rows, self._ineq_labels):
            B_l = _sym_outer(a_bar, e0)
            self.Bineq_list.append(B_l)
            self.bineq_list.append(0.0)
            self.bineq_labels.append(f"ineq_{label}")

        n_lin = len(self.Bineq_list)

        # --- diagonal bounds  X_mm <= ub^2 ---
        for (j, ub_sq, label) in self.diag_bounds:
            em = np.zeros(nc)
            em[j + 1] = 1.0
            B_l = _sparse_outer(em, em)
            self.Bineq_list.append(B_l)
            self.bineq_list.append(float(ub_sq))
            self.bineq_labels.append(f"diagbnd_{label}")

        n_diag = len(self.Bineq_list) - n_lin
        print(f"[CPOPTBuilder] CPOPT inequalities: {len(self.Bineq_list)} "
              f"(linear {n_lin}, diagonal bounds {n_diag})")


# ===================================================================
# GUOSolver
# ===================================================================
class GUOSolver:
    """
    Guo et al.'s outer-approximation cutting-plane algorithm for COPOPT.

    Master problem:  max v_norm
        s.t.  S = C - v_norm * e_0 e_0^T - sum_l v_l B_l
              diag(S) >= 0            (initial outer approx)
              u_k^T S u_k >= 0       (accumulated cuts)

    Separation:  MILP to find u with u^T S u < 0 (proving S not COP).
    """

    def __init__(self, builder: CPOPTBuilder):
        if not getattr(builder, "lift_slacks", True):
            raise ValueError(
                "GUOSolver requires the exact CPP-eq lifting "
                "(lift_slacks=True). The CPP-ineq builder carries inequality "
                "constraints that GUOSolver's COPOPT dual does not model; use "
                "DNNRelaxationSolver for CPP-ineq data.")
        self.builder = builder
        self.nc = builder.nc

    # ------------------------------------------------------------------
    def solve(self, max_iter=100, time_limit=3600, eps_gamma=1e-6,
              S_BOUND=5000.0, verbose=True):
        """
        Run Guo et al.'s cutting-plane algorithm with S as explicit variables.

        Returns dict with keys: obj, iterations, gap, converged, cuts, time, history
        """
        nc = self.nc
        builder = self.builder
        num_bl = len(builder.B_list)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Guo Cutting-Plane Algorithm (S-explicit)")
            print(f"  nc={nc}, CPOPT constraints={num_bl}, S_BOUND={S_BOUND}")
            print(f"  max_iter={max_iter}, time_limit={time_limit}")
            print(f"{'='*70}\n")

        t_start = _time.time()

        C_dense = builder.C.toarray() if sp.issparse(builder.C) else np.array(builder.C)

        # ----------------------------------------------------------------
        # Pre-compute nonzero entries of each B_l in COO format
        # ----------------------------------------------------------------
        if verbose:
            print("Pre-computing B_l nonzero structure ...")

        # B_l stored as list of (rows, cols, vals) tuples for fast iteration
        B_coo_list = []
        total_nnz = 0
        for l_idx in range(num_bl):
            B_l = builder.B_list[l_idx]
            if sp.issparse(B_l):
                coo = sp.triu(B_l).tocoo()  # upper triangle only (symmetric)
            else:
                coo = sp.triu(sp.csr_matrix(B_l)).tocoo()
            B_coo_list.append((coo.row, coo.col, coo.data))
            total_nnz += len(coo.data)

        if verbose:
            print(f"  Total B_l nonzeros (upper tri): {total_nnz}")

        # ----------------------------------------------------------------
        # Build master model with S as explicit variables
        # ----------------------------------------------------------------
        if verbose:
            print("Building Gurobi master with S-explicit variables ...")

        master = gp.Model("COPOPT_Master")
        master.setParam("OutputFlag", 0)
        master.setParam("Method", 1)  # dual simplex for LP

        # -- S variables (upper triangle only, S symmetric) --
        # S[i,j] for i <= j, with box bounds
        S_vars = {}
        for i in range(nc):
            for j in range(i, nc):
                if i == j:
                    # Diagonal: S[i,i] >= 0, bounded above
                    S_vars[i, j] = master.addVar(lb=0.0, ub=S_BOUND,
                                                  name=f"S_{i}_{j}")
                else:
                    S_vars[i, j] = master.addVar(lb=-S_BOUND, ub=S_BOUND,
                                                  name=f"S_{i}_{j}")

        # -- v variables (dual multipliers) --
        v_vars = master.addVars(num_bl, lb=-GRB.INFINITY, name="v")

        # -- v_norm (dual of Y_00=1, the objective) --
        v_norm = master.addVar(lb=-GRB.INFINITY, name="v_norm")

        master.update()

        if verbose:
            n_S = len(S_vars)
            print(f"  S variables: {n_S} (upper triangle of {nc}x{nc})")
            print(f"  v variables: {num_bl} + 1 (v_norm)")

        # -- Linking constraints: S[i,j] = C[i,j] - v_norm*delta - sum_l v_l * B_l[i,j] --
        if verbose:
            print("Building linking constraints S = C - v_norm*E00 - sum v_l*B_l ...")

        # First, build a map: for each (i,j) upper-tri, which (l, val) contribute
        link_data = {(i, j): [] for i in range(nc) for j in range(i, nc)}
        for l_idx in range(num_bl):
            rows, cols, vals = B_coo_list[l_idx]
            for r, c, val in zip(rows, cols, vals):
                ii, jj = (r, c) if r <= c else (c, r)
                link_data[ii, jj].append((l_idx, val))

        for i in range(nc):
            for j in range(i, nc):
                rhs_const = C_dense[i, j]
                lhs = gp.LinExpr()
                lhs.add(S_vars[i, j], 1.0)

                # v_norm contribution (only for (0,0))
                if i == 0 and j == 0:
                    lhs.add(v_norm, 1.0)

                # B_l contributions
                for l_idx, val in link_data[i, j]:
                    lhs.add(v_vars[l_idx], val)

                master.addConstr(lhs == rhs_const, name=f"link_{i}_{j}")

        # -- Complementary slackness cut from MIP solution (Guo et al. Sec 5.1) --
        # Tr(x* x*^T Omega) >= 0 is always valid for COP
        # We add Tr(x* x*^T S) >= 0 using the MIP optimal solution
        # NOTE: Complementary slackness cut Tr(x* x*^T S) >= 0 is DISABLED.
        # It was pinning v_norm to -vmip from iter 1, preventing master from
        # exploring different S matrices. Without it, obj should decrease
        # from S_BOUND towards -vmip as cuts accumulate.
        # Can be re-enabled as acceleration AFTER basic convergence is verified.

        # -- Objective: max v_norm --
        master.setObjective(v_norm, GRB.MAXIMIZE)
        master.update()

        if verbose:
            print(f"Master built: {master.NumVars} vars, {master.NumConstrs} constrs")
            print(f"Build time: {_time.time()-t_start:.1f}s")

        # ----------------------------------------------------------------
        # Cutting-plane loop
        # ----------------------------------------------------------------
        cuts_added = 0
        best_obj = -1e30
        converged = False
        history = []
        v_current = None
        cut_directions = []   # for duplicate cut detection
        n_dup_skipped = 0     # consecutive duplicate counter

        for iteration in range(1, max_iter + 1):
            elapsed = _time.time() - t_start
            if elapsed > time_limit:
                if verbose:
                    print(f"Time limit reached at iteration {iteration}")
                break

            # Solve master
            master.optimize()

            if master.Status == GRB.INFEASIBLE:
                if verbose:
                    print(f"  Iter {iteration}: Master infeasible")
                break
            elif master.Status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
                if verbose:
                    print(f"  Iter {iteration}: Master unbounded (should not happen with S bounds)")
                break
            elif master.Status != GRB.OPTIMAL:
                if verbose:
                    print(f"  Iter {iteration}: Master status={master.Status}")
                break

            obj_val = master.ObjVal
            best_obj = obj_val

            # Extract S matrix from solution
            S_bar = np.zeros((nc, nc))
            for (i, j), var in S_vars.items():
                S_bar[i, j] = var.X
                if i != j:
                    S_bar[j, i] = var.X  # symmetric

            if verbose:
                S_min_diag = np.min(np.diag(S_bar))
                target = -builder.vmip if builder.vmip is not None else float('nan')
                print(f"  Iter {iteration}: obj={obj_val:.4f}, "
                      f"target={target:.4f}, "
                      f"min_diag(S)={S_min_diag:.2e}, "
                      f"cuts={cuts_added}, time={elapsed:.1f}s")

            # Separation: ALWAYS test if S_bar ∈ COP_{nc}
            gamma_val, u_cut = self._solve_separation(S_bar)

            if verbose:
                print(f"    Separation: gamma={gamma_val:.6f}")

            if gamma_val <= eps_gamma:
                # S is copositive — verified by separation oracle
                if verbose:
                    print(f"  Converged: S ∈ COP (gamma={gamma_val:.6f} <= eps={eps_gamma})")
                converged = True
                break

            # Normalize u_cut for consistent cut direction
            u_norm = np.linalg.norm(u_cut)
            if u_norm > 1e-15:
                u_cut = u_cut / u_norm

            # Duplicate cut filter: skip if cosine similarity > 0.999
            # with any existing cut direction
            is_duplicate = False
            for prev_u in cut_directions:
                cos_sim = np.dot(u_cut, prev_u)
                if cos_sim > 0.999:
                    is_duplicate = True
                    break
            if is_duplicate:
                if verbose:
                    print(f"    Skipping duplicate cut (cosine > 0.999)")
                # Still record iteration but don't add cut
                history.append({
                    "iteration": iteration, "obj": obj_val,
                    "gamma": gamma_val, "time": elapsed,
                })
                n_dup_skipped += 1
                if n_dup_skipped >= 5:
                    if verbose:
                        print(f"  Stopping: 5 consecutive duplicate cuts")
                    break
                continue
            n_dup_skipped = 0
            cut_directions.append(u_cut.copy())

            # Add cut: u^T S u >= 0 → sum_{i<=j} coeff * S[i,j] >= 0
            cut_lhs = gp.LinExpr()
            nz_u = np.where(np.abs(u_cut) > 1e-15)[0]
            for ii_idx, i in enumerate(nz_u):
                for j in nz_u[ii_idx:]:
                    coeff = u_cut[i] * u_cut[j]
                    if i < j:
                        coeff *= 2.0
                    if abs(coeff) > 1e-15:
                        ij = (min(i,j), max(i,j))
                        cut_lhs.add(S_vars[ij], coeff)
            master.addConstr(cut_lhs >= 0, name=f"cut_{cuts_added}")
            cuts_added += 1

            history.append({
                "iteration": iteration,
                "obj": obj_val,
                "gamma": gamma_val,
                "time": elapsed,
            })

        # ---- Final report ----
        elapsed = _time.time() - t_start
        # COPOPT obj = v_norm converges to -vmip from above.
        # gap = (obj - (-vmip)) / |vmip| = (obj + vmip) / |vmip|
        gap = 0.0
        if builder.vmip is not None and abs(builder.vmip) > 1e-12:
            gap = (best_obj - (-builder.vmip)) / abs(builder.vmip)

        if verbose:
            print(f"\nAlgorithm finished: {cuts_added} cuts, {elapsed:.1f}s")
            print(f"COPOPT bound (v_norm) = {best_obj:.6f}")
            print(f"COPOPT target (-vmip) = {-builder.vmip:.6f}")
            if builder.vmip is not None:
                print(f"v^MIP                 = {builder.vmip:.6f}")
                print(f"Relative gap          = {gap:.6f}")

        return {
            "obj": best_obj,
            "iterations": len(history),
            "gap": gap,
            "converged": converged,
            "cuts": cuts_added,
            "time": elapsed,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Separation MILP
    # ------------------------------------------------------------------
    def _solve_separation(self, S_bar):
        """
        Solve MILPCOP: test if S_bar is in COP_{nc}.

        max gamma
        s.t. S_bar u <= -gamma e + nu .* (e - z)
             0 <= u <= z
             z in {0,1}^nc
             e^T z >= 2
             nu_j = 1 + sum_{k!=j} max(0, S_bar[j,k])

        Returns (gamma_val, u_vector).
        gamma_val = 0 means S_bar is copositive (within tolerance).
        """
        nc = self.nc

        # Compute big-M coefficients nu
        nu = np.zeros(nc)
        for j in range(nc):
            nu[j] = 1.0 + np.sum(np.maximum(0.0, np.delete(S_bar[j, :], j)))

        sep = gp.Model("Separation")
        sep.setParam("OutputFlag", 0)
        sep.setParam("TimeLimit", 300)
        sep.setParam("MIPGap", 1e-4)

        gamma = sep.addVar(lb=0.0, name="gamma")
        u_vars = sep.addVars(nc, lb=0.0, ub=1.0, name="u")
        z_vars = sep.addVars(nc, vtype=GRB.BINARY, name="z")

        # S_bar u <= -gamma e + nu * (e - z)
        for j in range(nc):
            lhs = gp.quicksum(S_bar[j, k] * u_vars[k] for k in range(nc))
            sep.addConstr(lhs <= -gamma + nu[j] * (1.0 - z_vars[j]),
                          name=f"sep_{j}")

        # 0 <= u <= z
        for j in range(nc):
            sep.addConstr(u_vars[j] <= z_vars[j], name=f"uz_{j}")

        # e^T z >= 2
        sep.addConstr(gp.quicksum(z_vars[j] for j in range(nc)) >= 2, name="card")

        sep.setObjective(gamma, GRB.MAXIMIZE)
        sep.optimize()

        if sep.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            if sep.SolCount > 0:
                gamma_val = gamma.X
                u_vec = np.array([u_vars[j].X for j in range(nc)])
                return gamma_val, u_vec
            else:
                return 0.0, np.zeros(nc)
        else:
            return 0.0, np.zeros(nc)


# ===================================================================
# Main test script
# ===================================================================
if __name__ == "__main__":
    from data_generator import setup_lem_parameters

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true",
                        help="Use small instance (u1+u4, T=4) for debugging")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--s-bound", type=float, default=5000.0)
    args = parser.parse_args()

    if args.small:
        # Small instance: 2 players, T=4, no binaries
        players = ["u1", "u4"]
        time_periods = list(range(4))
        config = {
            "players_with_renewables": ["u1"],
            "players_with_solar": [],
            "players_with_wind": ["u1"],
            "players_with_electrolyzers": [],
            "players_with_heatpumps": [],
            "players_with_elec_storage": ["u1"],
            "players_with_hydro_storage": [],
            "players_with_heat_storage": [],
            "players_with_nfl_elec_demand": ["u4"],
            "players_with_nfl_hydro_demand": [],
            "players_with_nfl_heat_demand": [],
            "players_with_fl_elec_demand": [],
            "players_with_fl_hydro_demand": [],
            "players_with_fl_heat_demand": [],
        }
        desc = "SMALL instance (u1+u4, T=4, no binaries)"
    else:
        # Full 6-player instance
        players = ["u1", "u2", "u3", "u4", "u5", "u6"]
        time_periods = list(range(24))
        config = {
            "players_with_renewables": ["u1"],
            "players_with_solar": [],
            "players_with_wind": ["u1"],
            "players_with_electrolyzers": ["u2"],
            "players_with_heatpumps": ["u3"],
            "players_with_elec_storage": ["u1"],
            "players_with_hydro_storage": ["u2"],
            "players_with_heat_storage": ["u3"],
            "players_with_nfl_elec_demand": ["u4"],
            "players_with_nfl_hydro_demand": ["u5"],
            "players_with_nfl_heat_demand": ["u6"],
            "players_with_fl_elec_demand": ["u2", "u3"],
            "players_with_fl_hydro_demand": [],
            "players_with_fl_heat_demand": [],
        }
        desc = "FULL 6-player instance (T=24)"

    params = setup_lem_parameters(players, config, time_periods)

    print("=" * 70)
    print(f"Building CPOPT: {desc}")
    print("=" * 70)

    builder = CPOPTBuilder(players, time_periods, params, config)
    print(f"\nnc={builder.nc}, n_orig={builder.n_orig}, "
          f"n_slacked={builder.n_slacked}")
    print(f"Binaries: {builder.n_binary}")
    print(f"Equality rows: {builder.A_homo.shape[0]}")
    print(f"v^MIP = {builder.vmip:.4f}")

    print("\n" + "=" * 70)
    print("Starting Guo cutting-plane algorithm ...")
    print("=" * 70)

    solver = GUOSolver(builder)
    result = solver.solve(max_iter=args.max_iter, S_BOUND=args.s_bound,
                          verbose=True)

    print(f"\n{'='*70}")
    print(f"Final Results:")
    print(f"  COPOPT bound  = {result['obj']:.4f}")
    if builder.vmip is not None:
        print(f"  v^MIP         = {builder.vmip:.4f}")
    print(f"  Relative gap  = {result['gap']:.6f}")
    print(f"  Iterations    = {result['iterations']}")
    print(f"  Cuts added    = {result['cuts']}")
    print(f"  Converged     = {result['converged']}")
    print(f"  Time          = {result['time']:.2f}s")
    print(f"{'='*70}")

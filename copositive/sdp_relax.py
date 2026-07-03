"""
Doubly-nonnegative (DNN) relaxation of the energy-community completely
positive program (CPOPT), solved with MOSEK.

Background
----------
The exact reformulation (see ``cpp_reformulation.tex``) is

    CPOPT:   min  C . Y   s.t.  B_l . Y = b_l  (l=1..m),  Y_00 = 1,  Y in CP^{1+n}

where ``CP`` is the cone of completely positive matrices
(``Y = sum_k u_k u_k^T`` with every ``u_k >= 0``).  The CP cone is
NP-hard to optimise over, so here we replace it by its most common
tractable outer approximation, the *doubly-nonnegative* (DNN) cone

    DNN = { Y : Y is symmetric positive semidefinite } cap { Y : Y >= 0 elementwise }.

Because ``CP subset DNN``, enlarging the feasible cone can only lower the
minimum, so

    val(DNN) <= val(CPOPT) = -v^MIP        (C = -W_S, so val(CPOPT) = -v^MIP)

i.e. ``-val(DNN) >= v^MIP`` is a valid *upper bound* on the coalition
value.  The gap ``-val(DNN) - v^MIP >= 0`` measures how much the DNN cone
over-relaxes the completely positive cone for this instance.

This file:
    * reuses ``CPOPTBuilder`` from ``cop_guo.py`` to assemble C, {B_l, b_l}
      and the normalisation, exactly as for the Guo / Gabl-Anstreicher
      cutting-plane solvers;
    * builds the DNN relaxation as a single semidefinite program with the
      MOSEK Fusion API and solves it in one shot (no cutting planes needed
      -- MOSEK handles the PSD + nonnegativity cones natively).

Reference:
    Burer, S. (2009). On the copositive representation of binary and
    continuous nonconvex quadratic programs. Math. Programming.

Environment:
    Run with the project virtualenv (has mosek, gurobipy, numpy, scipy,
    pyscipopt installed).  The device generators read ``./data/*.csv`` via
    RELATIVE paths, so launch from the repo root ``energy_community/`` (not
    from ``copositive/``)::

        cd /home/user/myfolder/energy_community
        /home/user/myfolder/.venv/bin/python copositive/sdp_relax.py           # 6-player
        /home/user/myfolder/.venv/bin/python copositive/sdp_relax.py --small   # LP debug case

Debugging / validation strategy (do this before trusting any bound):
    (1) PURE-LP SANITY CHECK -- run ``--small`` (players u1+u4, T=4) which has
        NO binary variables, so the MIP is really a plain LP.  For a pure LP,
        Burer exactness is trivial (no z-z^2 constraints, condition (a) of
        Prop. exact is vacuous), so the completely-positive program is exact
        AND its DNN relaxation is TIGHT: there is no CP-vs-DNN gap because a
        continuous LP feasible set already lifts to a rank-1 CP matrix.
        Therefore we must get

            -val(DNN) == v^MIP   (gap ~ 0, up to solver tolerance).

        Confirming this identity end-to-end validates the whole pipeline --
        MPS export, equality form, homogenisation, C and {B_l, b_l}
        assembly, and the MOSEK model -- WITHOUT the confounding factor of a
        relaxation gap.  Any nonzero gap on ``--small`` is a construction bug,
        not a property of the relaxation.  (Status: verified -- objectives
        match.)
    (2) Only after (1) passes, run the full 6-player instance (below), where a
        genuine, nonnegative CP-vs-DNN gap is expected because of the 336
        binaries (electrolyzer/heat-pump commitment).  There the gap MEASURES
        how loose DNN is, rather than signalling a bug.
"""

import os
import sys
import time as _time

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# MOSEK license location.
# MOSEK looks for the license in $MOSEKLM_LICENSE_FILE (or ~/mosek/mosek.lic).
# On this machine the license lives on the Windows side; point to it unless
# the user has already configured one.
# ---------------------------------------------------------------------------
_DEFAULT_LIC = "/mnt/c/Users/user/mosek/mosek.lic"
if "MOSEKLM_LICENSE_FILE" not in os.environ and os.path.exists(_DEFAULT_LIC):
    os.environ["MOSEKLM_LICENSE_FILE"] = _DEFAULT_LIC

# Make the copositive package importable and pull in the shared builder.
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/user/myfolder/energy_community")

from cop_guo import CPOPTBuilder  # noqa: E402  (reuse the exact CPOPT data)

import mosek.fusion as mf  # noqa: E402
from mosek.fusion import Domain, Expr, Matrix, ObjectiveSense  # noqa: E402


# ===================================================================
# Helpers
# ===================================================================
def _to_mosek_matrix(spmat, nc):
    """Convert a scipy sparse (nc x nc) matrix into a MOSEK Fusion sparse
    Matrix, keeping *all* stored entries (both triangles).  The Frobenius
    inner product ``Expr.dot(A, Y)`` uses every entry of ``A``, so the full
    symmetric matrix must be supplied (not just the upper triangle)."""
    coo = spmat.tocoo()
    rows = coo.row.astype(np.int32).tolist()
    cols = coo.col.astype(np.int32).tolist()
    vals = coo.data.astype(float).tolist()
    return Matrix.sparse(nc, nc, rows, cols, vals)


class _Tee:
    """Minimal stream fan-out: forward every write to several underlying
    streams and flush each one immediately, so a log file stays up to date
    in real time (useful for ``tail -f`` on the MOSEK solver log while the
    solve is still running)."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ===================================================================
# Memory preflight guard
# ===================================================================
# The monolithic DNN blows up MOSEK's memory silently: the interior-point
# method forms and factorises a DENSE Schur complement whose side is the
# number of linear constraints coupling the PSD block.  Imposing ``Y >= 0``
# elementwise turns every one of the nc*nc matrix entries into its own linear
# constraint, so that side grows like nc^2 and the Schur complement like nc^4
# -- for nc=1273 that is ~21 TB, far past any RAM, and the process is
# SIGKILLed with no traceback (see run_6player_cppineq.out, which froze right
# after "Presolve terminated").  The estimator below flags that BEFORE MOSEK
# is even built, so we fail loudly and cleanly instead of being killed.

def _fmt_bytes(n):
    """Human-readable byte count (base-1024)."""
    if n is None:
        return "unknown"
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < step:
            return f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} EB"


def _available_memory_bytes():
    """Kernel's MemAvailable (bytes) from /proc/meminfo -- its own estimate
    of memory obtainable for a new workload without swapping, which is the
    right yardstick for a preflight check.  Returns None if unreadable
    (e.g. non-Linux), in which case the guard degrades to a warning only."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # value is in kB
    except (OSError, ValueError, IndexError):
        return None
    return None


def estimate_peak_memory(nc, n_eq, n_ineq, use_psd, use_nonneg):
    """Conservative estimate (bytes) of MOSEK's peak memory for this DNN.

    Dominant term: the DENSE Schur complement of side ``m`` = number of
    linear constraints hitting the PSD block, stored/factorised as O(m^2)
    doubles.  Elementwise ``Y >= 0`` contributes nc^2 of those constraints,
    so it -- not the equalities -- is what drives the blow-up.  We add the
    PSD variable's own dense primal+dual storage.  The estimate is meant as a
    safety envelope (order-of-magnitude), not a MOSEK-exact figure: solvable
    instances land in the MB range, impossible ones in the TB range, so the
    two are cleanly separated even with a coarse model.

    Returns a dict with the component byte counts and 'total'.
    """
    DBL = 8
    # Scalar linear constraints coupling the matrix variable.
    m = n_eq + n_ineq + 1  # +1 for the Y_00 = 1 normalisation
    if use_nonneg:
        m += nc * nc       # elementwise Y >= 0  -- the dominant contributor
    # Dense Schur complement (m x m). Halve for symmetry, but MOSEK's working
    # set and ordering carry overhead, so we do NOT shave it further.
    schur = (m * m // 2) * DBL if use_psd else 0
    # PSD variable: primal + dual dense symmetric storage.
    psd = 2 * (nc * (nc + 1) // 2) * DBL if use_psd else 0
    total = schur + psd
    return {"m": m, "schur": schur, "psd": psd, "total": total}


# ===================================================================
# DNN relaxation solver
# ===================================================================
class DNNRelaxationSolver:
    """Solve the doubly-nonnegative relaxation of CPOPT with MOSEK Fusion.

    Model:
        min   C . Y
        s.t.  B_l . Y = b_l      for l = 1..m   (Class A first-order,
                                                 Class B RLT / binary diag)
              Y_00 = 1                          (normalisation)
              Y >= 0               (elementwise nonnegativity)
              Y in PSD cone        (positive semidefiniteness)

    The last two rows together are ``Y in DNN`` -- the doubly-nonnegative
    cone that replaces the completely positive cone ``CP`` of the exact CPP.

    SCALABILITY -- the monolithic PSD cone is the bottleneck.
    -----------------------------------------------------------------------
    This solver imposes ONE dense PSD cone on the whole (nc x nc) matrix Y.
    For the full 6-player, T=24 instance nc ~ 2592, so the PSD variable has
    ~3.36M scalar entries and MOSEK runs out of memory (rescode.err_space).
    This is NOT a bug -- it is inherent to the monolithic lifting.  Even Guo
    et al. (2026, "Pricing Discrete and Nonlinear Markets with Semidefinite
    Relaxations") never solve such a monolithic cone: on a 350GB cluster they
    still relax  Y in PSD  by a BLOCK-DIAGONAL PSD relaxation, decomposing the
    cone BY TIME PERIOD -- for each t they impose PSD only on the submatrix of
    Y spanned by the variables of period t (nonnegativity + RLT stay global).
    Dropping the cross-period PSD coupling loosens the bound, but they recover
    tightness with cheap cuts (triangle inequalities, X_ii <= U_i^2) and
    report it stays tight in practice (avg gap ~2%).

    TODO (time-block DNN): add a ``block_by_time=True`` mode that replaces the
    single ``Domain.inPSDCone(nc)`` below with one PSD cone per period on the
    sub-block of indices {0} U {vars whose name ends in "_t"} (CPOPTBuilder
    already carries the Gurobi var names, so the periods can be parsed from
    them).  Strong duality -- hence the pricing duals -- is preserved by any
    such relaxation, so the DNN prices remain well defined; only the bound
    loosens.  See also the per-PLAYER block variant, which is the CHP lift of
    cpp_reformulation.tex Sec. 6 (Y_i in CP_i per player) -- the same idea on
    the player axis instead of the time axis.  An alternative to the single
    monolithic solve is the ADMM decomposition of the DNN+RLT relaxation
    (see "ADMM-based decomposed DNN_RLT relaxations.txt" in this folder).
    """

    def __init__(self, builder: CPOPTBuilder):
        self.builder = builder
        self.nc = builder.nc

    # ------------------------------------------------------------------
    def solve(self, use_psd=True, use_nonneg=True, time_limit=3600.0,
              verbose=True, log_stream=None, max_mem_frac=0.8, force=False):
        """Build and solve the DNN relaxation.

        Parameters
        ----------
        use_psd : bool
            Impose ``Y in PSD``.  (Turning this off leaves the pure
            nonnegative/LP relaxation -- only useful for diagnostics.)
        use_nonneg : bool
            Impose ``Y >= 0`` elementwise.  (Turning this off leaves the
            Shor / SDP relaxation without the completely-positive
            nonnegativity, a weaker bound.)
        max_mem_frac : float
            Memory preflight budget as a fraction of MemAvailable.  If the
            estimated peak (see ``estimate_peak_memory``) exceeds this, the
            solve is aborted BEFORE MOSEK is built -- returning cleanly with
            status ``"SKIPPED_MEMORY_GUARD"`` instead of being OOM-killed.
        force : bool
            Bypass the memory guard and attempt the solve regardless (the
            estimate is still printed).  Use when you know the machine can
            take it or want to observe the failure.

        Returns
        -------
        dict with keys:
            obj        -- val(DNN) = C . Y*   (a lower bound on val(CPOPT))
            bound      -- -val(DNN)           (an upper bound on v^MIP)
            vmip       -- builder.vmip
            gap        -- (bound - vmip) / |vmip|   (relaxation gap, >= 0)
            x          -- recovered first-column solution  Y*[1:, 0]
            Y          -- full solution matrix (nc x nc numpy array)
            status     -- MOSEK solution status string
            time       -- wall-clock seconds
        """
        nc = self.nc
        builder = self.builder
        num_bl = len(builder.B_list)
        n_ineq_dbg = len(getattr(builder, "Bineq_list", []))

        if verbose:
            cone = "DNN (PSD + nonneg)"
            if use_psd and not use_nonneg:
                cone = "PSD only (Shor)"
            elif use_nonneg and not use_psd:
                cone = "nonneg only (LP)"
            print(f"\n{'='*70}")
            print("DNN relaxation of CPOPT via MOSEK Fusion")
            print(f"  nc={nc}, CPOPT constraints={num_bl}, cone={cone}")
            print(f"{'='*70}\n")
            # -- debug: report the problem dimensions we are about to hand MOSEK
            psd_entries = nc * nc
            print("[debug] problem dimensions:")
            print(f"[debug]   nc (matrix side)      = {nc}")
            print(f"[debug]   PSD variable entries  = {psd_entries:,}"
                  f"  ({'ON' if use_psd else 'OFF'})")
            print(f"[debug]   nonneg constraint     = {'ON' if use_nonneg else 'OFF'}")
            print(f"[debug]   equality  rows B_l=b_l = {num_bl}")
            print(f"[debug]   inequality rows B_l<=b = {n_ineq_dbg}")
            print(f"[debug]   C nnz                  = {builder.C.nnz}")
            print(f"[debug]   time limit             = {time_limit:.0f}s")
            sys.stdout.flush()

        # -- Memory preflight guard ------------------------------------
        # Estimate MOSEK's peak memory and compare against what the kernel
        # says is available, so we fail loudly here instead of getting
        # SIGKILLed mid-solve with no traceback.
        est = estimate_peak_memory(nc, num_bl, n_ineq_dbg, use_psd, use_nonneg)
        avail = _available_memory_bytes()
        budget = avail * max_mem_frac if avail is not None else None
        if verbose:
            print("[preflight] estimated MOSEK peak memory:")
            print(f"[preflight]   dense Schur complement (side m={est['m']:,}) "
                  f"= {_fmt_bytes(est['schur'])}")
            print(f"[preflight]   PSD variable storage           "
                  f"= {_fmt_bytes(est['psd'])}")
            print(f"[preflight]   estimated total                "
                  f"= {_fmt_bytes(est['total'])}")
            if avail is not None:
                print(f"[preflight]   MemAvailable                   "
                      f"= {_fmt_bytes(avail)}  (budget {max_mem_frac:.0%} = "
                      f"{_fmt_bytes(budget)})")
            sys.stdout.flush()

        over_budget = budget is not None and est["total"] > budget
        if over_budget and not force:
            print(f"\n{'!'*70}")
            print("[preflight] ABORTED -- estimated memory exceeds the budget.")
            print(f"[preflight]   need ~{_fmt_bytes(est['total'])} but only "
                  f"{_fmt_bytes(budget)} is safely available.")
            print("[preflight] The elementwise Y>=0 constraint makes the Schur")
            print(f"[preflight]   complement side grow like nc^2 (nc={nc}); this")
            print("[preflight]   monolithic DNN cannot fit. Options:")
            print("[preflight]     * --small          validate the pipeline")
            print("[preflight]     * block/ADMM decomposition (see TODO + "
                  "ADMM note in this folder)")
            print("[preflight]     * --force           attempt anyway (may be "
                  "OOM-killed)")
            print(f"{'!'*70}\n")
            sys.stdout.flush()
            return {
                "obj": None, "bound": None, "vmip": builder.vmip, "gap": None,
                "x": None, "Y": None, "status": "SKIPPED_MEMORY_GUARD",
                "time": 0.0, "mem_estimate": est, "mem_available": avail,
            }
        if over_budget and force:
            print("[preflight] over budget but --force given; attempting solve "
                  "anyway (watch for OOM).")
            sys.stdout.flush()

        t_start = _time.time()

        with mf.Model("DNN_relaxation") as M:
            # -- Symmetric matrix variable Y (nc x nc) --
            if use_psd:
                # PSD cone variable is symmetric by construction.
                Y = M.variable("Y", Domain.inPSDCone(nc))
            else:
                # Symmetric variable without the PSD requirement.
                Y = M.variable("Y", [nc, nc], Domain.unbounded())
                M.constraint("sym", Expr.sub(Y, Y.transpose()),
                             Domain.equalsTo(0.0))

            # -- Elementwise nonnegativity (the "completely positive" part) --
            if use_nonneg:
                M.constraint("nonneg", Y, Domain.greaterThan(0.0))

            # -- Normalisation  Y_00 = 1 --
            M.constraint("norm", Y.index(0, 0), Domain.equalsTo(1.0))

            # -- CPOPT linear constraints  B_l . Y = b_l --
            if verbose:
                print(f"Adding {num_bl} constraints  B_l . Y = b_l ...")
            _t_eq = _time.time()
            for l_idx in range(num_bl):
                B_l = builder.B_list[l_idx]
                b_l = builder.b_list[l_idx]
                Bmat = _to_mosek_matrix(B_l, nc)
                M.constraint(f"c_{l_idx}", Expr.dot(Bmat, Y),
                             Domain.equalsTo(float(b_l)))
                if verbose and num_bl >= 500 and (l_idx + 1) % 500 == 0:
                    print(f"[debug]   ... {l_idx+1}/{num_bl} equalities added "
                          f"({_time.time()-_t_eq:.1f}s)")
                    sys.stdout.flush()

            # -- CPP-ineq inequality constraints  B_l . Y <= b_l  --
            # Present only when the builder used lift_slacks=False.  These are
            # the linear inequalities and diagonal bounds X_ii <= U_i^2 that
            # replace the (unlifted) slack variables.
            n_ineq = len(getattr(builder, "Bineq_list", []))
            if n_ineq:
                if verbose:
                    print(f"Adding {n_ineq} inequalities  B_l . Y <= b_l ...")
                for l_idx in range(n_ineq):
                    B_l = builder.Bineq_list[l_idx]
                    b_l = builder.bineq_list[l_idx]
                    Bmat = _to_mosek_matrix(B_l, nc)
                    M.constraint(f"ineq_{l_idx}", Expr.dot(Bmat, Y),
                                 Domain.lessThan(float(b_l)))

            # -- Objective:  min C . Y --
            Cmat = _to_mosek_matrix(builder.C, nc)
            M.objective(ObjectiveSense.Minimize, Expr.dot(Cmat, Y))

            if verbose:
                print(f"Model built in {_time.time()-t_start:.1f}s. Solving ...")
                # Route the live MOSEK solver log to stdout AND (if given) a
                # log file, flushing on every write so it can be tail -f'd.
                M.setLogHandler(_Tee(sys.stdout, log_stream))
                sys.stdout.flush()

            M.setSolverParam("optimizerMaxTime", float(time_limit))
            M.solve()

            status = str(M.getPrimalSolutionStatus())
            elapsed = _time.time() - t_start

            obj = None
            Y_val = None
            x_val = None
            if M.getPrimalSolutionStatus() in (
                mf.SolutionStatus.Optimal, mf.SolutionStatus.Feasible):
                obj = M.primalObjValue()
                Y_flat = Y.level()
                Y_val = np.array(Y_flat).reshape(nc, nc)
                x_val = Y_val[1:, 0].copy()

        # ---- Report ----
        bound = -obj if obj is not None else None
        gap = None
        if (obj is not None and builder.vmip is not None
                and abs(builder.vmip) > 1e-12):
            gap = (bound - builder.vmip) / abs(builder.vmip)

        if verbose:
            print(f"\n{'-'*70}")
            print(f"MOSEK status          : {status}")
            print(f"Solve time            : {elapsed:.2f}s")
            if obj is not None:
                print(f"val(DNN) = C.Y*       : {obj:.6f}")
                print(f"DNN bound (-val(DNN)) : {bound:.6f}   (upper bound on v^MIP)")
                if builder.vmip is not None:
                    print(f"v^MIP                 : {builder.vmip:.6f}")
                    print(f"Relaxation gap        : {gap:.6f}"
                          f"  ({100*gap:.4f}% of |v^MIP|)")
            print(f"{'-'*70}")

        return {
            "obj": obj,
            "bound": bound,
            "vmip": builder.vmip,
            "gap": gap,
            "x": x_val,
            "Y": Y_val,
            "status": status,
            "time": elapsed,
        }


# ===================================================================
# Main test script
# ===================================================================
if __name__ == "__main__":
    from data_generator import setup_lem_parameters
    import argparse

    parser = argparse.ArgumentParser(
        description="DNN relaxation of the community CPP, solved with MOSEK.")
    parser.add_argument("--small", action="store_true",
                        help="Small instance (u1+u4, T=4, no binaries).")
    parser.add_argument("--no-psd", action="store_true",
                        help="Drop the PSD constraint (LP/nonneg only).")
    parser.add_argument("--no-nonneg", action="store_true",
                        help="Drop elementwise nonnegativity (Shor SDP only).")
    parser.add_argument("--cpp-ineq", action="store_true",
                        help="Relaxed CPP-ineq lifting: lift only original "
                             "vars (nc=1+n_orig), keep bounds/inequalities as "
                             "linear/diagonal cuts. Much smaller, looser bound.")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--log-file", type=str, default=None,
                        help="Write the live MOSEK solver log to this file "
                             "(flushed on every line, so tail -f works).")
    parser.add_argument("--max-mem-frac", type=float, default=0.8,
                        help="Memory preflight budget as a fraction of "
                             "MemAvailable (default 0.8). Solve is skipped if "
                             "the estimated peak exceeds this.")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the memory preflight guard and attempt "
                             "the solve regardless (may be OOM-killed).")
    args = parser.parse_args()

    if args.small:
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

    builder = CPOPTBuilder(players, time_periods, params, config,
                           lift_slacks=not args.cpp_ineq)
    print(f"\nnc={builder.nc}, n_orig={builder.n_orig}, "
          f"n_slacked={builder.n_slacked}")
    print(f"Binaries: {builder.n_binary}")
    print(f"Equality rows: {builder.A_homo.shape[0]}")
    print(f"v^MIP = {builder.vmip:.4f}")

    if not args.small and not args.cpp_ineq:
        print("\n[warning] The full CPP-eq instance has nc ~ 2592, i.e. a PSD")
        print("          cone of ~3.4M matrix entries and thousands of")
        print("          equalities. The DNN SDP is very large and MOSEK may")
        print("          run out of memory. Try --cpp-ineq (nc ~ 1+n_orig) or")
        print("          --small to validate the pipeline first.")

    solver = DNNRelaxationSolver(builder)

    log_fh = None
    if args.log_file:
        log_fh = open(args.log_file, "w", buffering=1)  # line-buffered
        print(f"[debug] MOSEK solver log -> {args.log_file} "
              f"(tail -f to watch live)")

    try:
        result = solver.solve(use_psd=not args.no_psd,
                              use_nonneg=not args.no_nonneg,
                              time_limit=args.time_limit,
                              verbose=True,
                              log_stream=log_fh,
                              max_mem_frac=args.max_mem_frac,
                              force=args.force)
    finally:
        if log_fh is not None:
            log_fh.flush()
            log_fh.close()

    print(f"\n{'='*70}")
    print("Final Results (DNN relaxation):")
    if result["obj"] is not None:
        print(f"  val(DNN)      = {result['obj']:.4f}")
        print(f"  DNN bound     = {result['bound']:.4f}  (>= v^MIP)")
        print(f"  v^MIP         = {result['vmip']:.4f}")
        print(f"  Relaxation gap= {result['gap']:.6f}")
    else:
        print(f"  No solution recovered (status={result['status']}).")
    print(f"  Status        = {result['status']}")
    print(f"  Time          = {result['time']:.2f}s")
    print(f"{'='*70}")

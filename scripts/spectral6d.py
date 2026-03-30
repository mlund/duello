"""
6D spectral gap analysis for unified translational/rotational diffusion.

Builds a 6D Smoluchowski generator on (R, Ω) within a cell model, where R is
the center-of-mass separation and Ω = (θ_A, φ_A, θ_B, φ_B, ω) the angular
degrees of freedom.  The spectral gap of this operator gives the slowest
relaxation mode — translational, rotational, or coupled — and its eigenvalue
ratio λ₁/λ₁_free yields D/D⁰.

The cell model is built into the operator: equilibrium π ∝ R² exp(-βU(R,Ω))
naturally downweights high-energy states at close separation.

Reads exported matrices from `duello diffusion --export-matrices <dir>`.

Usage:
    python spectral6d.py <matrix_dir> --cmin 3.5e-5 --cmax 1.4e-3 --cn 10

Output:
    CSV to stdout with columns per concentration.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg

log = logging.getLogger(__name__)

AVOGADRO = 6.02214076e23
ZERO_THRESHOLD = 1e-6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metadata(directory: Path) -> pd.DataFrame:
    path = directory / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {directory}")
    return pd.read_csv(path)


def load_coords(directory: Path, R: float) -> pd.DataFrame | None:
    """Load compact_index → (vi, vj, oi, energy) mapping for an R-slice."""
    path = directory / f"coords_R{R:.1f}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_icosphere_neighbors(directory: Path, n_v: int) -> list[list[int]] | None:
    path = directory / f"icosphere_nv{n_v}.csv"
    if not path.exists():
        return None
    neighbors: list[list[int]] = []
    with open(path) as f:
        next(f)
        for line in f:
            _, nbrs = line.strip().split(",", 1)
            neighbors.append([int(x) for x in nbrs.split()])
    return neighbors


def r_cell_from_molarity(molarity: float) -> float:
    """Cell radius in Å from molar concentration."""
    c_m3 = molarity * 1000.0
    volume = 3.0 / (4.0 * np.pi * c_m3 * AVOGADRO)
    return volume ** (1.0 / 3.0) * 1e10


# ---------------------------------------------------------------------------
# State space: 6D indexing across R-slices
# ---------------------------------------------------------------------------


def build_state_space(
    meta: pd.DataFrame,
    r_cell: float,
    directory: Path,
    r_max: float | None = None,
    pmf_threshold: float = 0.05,
) -> dict | None:
    """Build 6D state space from exported R-slices within [R_min, min(R_cell, R_max)].

    Auto-truncates R where |PMF/kT| < pmf_threshold (potential negligible).
    Use r_max to override the upper R cutoff.

    Returns dict with slices, total_states, n_v, n_omega, or None if insufficient data.
    """
    r_upper = min(r_cell, r_max) if r_max else r_cell
    in_cell = meta[meta["R"] <= r_upper].copy()
    if in_cell.empty:
        return None

    # Auto-truncate: drop trailing R-slices where PMF is negligible
    if "boltzmann_weight" in in_cell.columns and r_max is None:
        pmf = -np.log(np.clip(in_cell["boltzmann_weight"].values, 1e-30, None))
        # Find last R where |PMF| > threshold
        significant = np.where(np.abs(pmf) > pmf_threshold)[0]
        if len(significant) > 0:
            last_sig = significant[-1]
            # Keep a few extra slices beyond the last significant one
            cutoff = min(last_sig + 3, len(in_cell))
            in_cell = in_cell.iloc[:cutoff]
            log.info("  Auto-truncated at R=%.1f Å (|PMF/kT| < %.2f beyond)",
                     in_cell["R"].iloc[-1], pmf_threshold)

    # Restrict to dominant mesh level
    dominant_nv = in_cell["n_v"].mode().iloc[0]
    in_cell = in_cell[in_cell["n_v"] == dominant_nv]
    if in_cell.empty:
        return None

    slices = []
    offset = 0
    for _, row in in_cell.iterrows():
        R = float(row["R"])
        n_v = int(row["n_v"])
        n_omega = int(row["n_omega"])

        coords = load_coords(directory, R)
        if coords is None or "energy" not in coords.columns:
            log.warning("R=%.1f: missing coords or energy column, skipping", R)
            continue

        mtx_path = directory / f"generator_R{R:.1f}.mtx"
        if not mtx_path.exists():
            continue

        gen = scipy.io.mmread(mtx_path).tocsc()
        ci_arr = coords["compact"].values
        vi_arr = coords["vi"].values
        vj_arr = coords["vj"].values
        oi_arr = coords["oi"].values
        energies = coords["energy"].values.astype(np.float64)

        # Build (vi,vj,oi) → compact_index lookup for radial matching
        key_to_ci = {}
        for ci, vi, vj, oi in zip(ci_arr, vi_arr, vj_arr, oi_arr):
            key_to_ci[(int(vi), int(vj), int(oi))] = int(ci)

        slices.append({
            "R": R,
            "ci": ci_arr,
            "vi": vi_arr,
            "vj": vj_arr,
            "oi": oi_arr,
            "energies": energies,
            "key_to_ci": key_to_ci,
            "n_active": len(coords),
            "n_v": n_v,
            "n_omega": n_omega,
            "offset": offset,
            "generator": gen,
        })
        offset += len(coords)

    if not slices:
        return None

    return {
        "slices": slices,
        "total_states": offset,
        "n_v": int(dominant_nv),
        "n_omega": int(in_cell["n_omega"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# Generator construction
# ---------------------------------------------------------------------------


def build_6d_generator(
    state_space: dict,
    beta: float,
    radial_D: np.ndarray | None = None,
    angular_D: np.ndarray | None = None,
) -> tuple[scipy.sparse.csc_matrix, np.ndarray]:
    """Build the symmetrized 6D Smoluchowski generator.

    Returns (L, pi) where L is the generator matrix (Rust convention:
    off-diagonal = 1.0, diagonal = -Σ exp(-βΔU/2)) and
    pi[i] is the equilibrium weight with π_i ∝ R² exp(-βU).

    radial_D: D_tt(R)/D_tt^0 per slice (RPY hook, default 1.0)
    angular_D: D_rr(R)/D_rr^0 per slice (RPY hook, default 1.0)
    """
    slices = state_space["slices"]
    n_total = state_space["total_states"]
    n_slices = len(slices)

    if radial_D is None:
        radial_D = np.ones(n_slices)
    if angular_D is None:
        angular_D = np.ones(n_slices)

    # Equilibrium weights: π_i ∝ R² exp(-βU_i)
    log_pi = np.empty(n_total)
    for s in slices:
        off = s["offset"]
        n = s["n_active"]
        log_pi[off : off + n] = -beta * s["energies"] + 2.0 * np.log(s["R"])
    log_pi -= log_pi.max()
    pi = np.exp(log_pi)
    pi /= pi.sum()

    rows, cols, vals = [], [], []

    # Angular edges: include ALL entries from per-R generators (including diagonal).
    # The Rust convention: off-diagonal = 1.0, diagonal = -Σ exp(-β(U_j-U_i)/2).
    for si, s in enumerate(slices):
        off = s["offset"]
        d_scale = angular_D[si]
        coo = s["generator"].tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            rows.append(off + i)
            cols.append(off + j)
            vals.append(v * d_scale)

    # Radial edges: same convention as Rust generator.
    # Off-diagonal = 1.0, diagonal contribution = -exp(-β(U_j-U_i)/2)
    for si in range(n_slices - 1):
        s0, s1 = slices[si], slices[si + 1]
        d_scale = radial_D[si]

        for ci1, vi, vj, oi in zip(s1["ci"], s1["vi"], s1["vj"], s1["oi"]):
            ci0 = s0["key_to_ci"].get((int(vi), int(vj), int(oi)))
            if ci0 is None:
                continue

            gi = s0["offset"] + ci0
            gj = s1["offset"] + int(ci1)
            du = s1["energies"][int(ci1)] - s0["energies"][ci0]

            # Off-diagonal: 1.0 (adjacency)
            rows.extend([gi, gj])
            cols.extend([gj, gi])
            vals.extend([d_scale, d_scale])

            # Diagonal: -exp(-βΔU/2) for each direction
            half_du = 0.5 * beta * du
            rows.extend([gi, gj])
            cols.extend([gi, gj])
            vals.extend([-np.exp(-half_du) * d_scale,
                         -np.exp(half_du) * d_scale])

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    vals = np.array(vals, dtype=np.float64)

    L = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n_total, n_total)).tocsc()

    return L, pi


# ---------------------------------------------------------------------------
# Eigensolvers
# ---------------------------------------------------------------------------


def count_null_space(A_sparse) -> int:
    n_components, _ = scipy.sparse.csgraph.connected_components(
        A_sparse, directed=False
    )
    return n_components


def solve_dense(A_dense: np.ndarray, k: int, n_components: int):
    n = A_dense.shape[0]
    k_req = min(n_components + k, n)
    lo = max(0, n - k_req)
    vals, vecs = scipy.linalg.eigh(A_dense, subset_by_index=[lo, n - 1])
    return vals[::-1], vecs[:, ::-1], "dense"


def solve_arpack(A_sparse, k: int, n_components: int):
    """ARPACK eigsh with which='LA' finds largest algebraic eigenvalues (nearest zero)."""
    k_req = min(n_components + k, A_sparse.shape[0] - 2)
    vals, vecs = scipy.sparse.linalg.eigsh(
        A_sparse, k=k_req, which="LA", tol=1e-10, maxiter=2000,
    )
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order], "arpack"


def solve_eigenproblem(L_sym, k: int, dense_threshold: int = 10000):
    n = L_sym.shape[0]
    n_components = count_null_space(L_sym)
    log.info("  n=%d, components=%d", n, n_components)

    if n <= dense_threshold:
        return solve_dense(L_sym.toarray(), k, n_components)
    try:
        return solve_arpack(L_sym, k, n_components)
    except Exception as exc:
        log.warning("ARPACK failed (%s), falling back to dense", exc)
        return solve_dense(L_sym.toarray(), k, n_components)


# ---------------------------------------------------------------------------
# Coordinate projection: 4-way (R, mol_A, mol_B, ω)
# ---------------------------------------------------------------------------


def coordinate_projection_6d(
    eigvec: np.ndarray,
    state_space: dict,
    neighbors: list[list[int]],
) -> tuple[float, float, float, float]:
    """Decompose eigenvector into R / mol_A / mol_B / ω fractions via edge variance."""
    slices = state_space["slices"]
    n_v = state_space["n_v"]
    n_omega = state_space["n_omega"]

    var_r = var_a = var_b = var_omega = 0.0
    edges_r = edges_a = edges_b = edges_omega = 0

    # R-edges: matched states between adjacent slices
    for si in range(len(slices) - 1):
        s0, s1 = slices[si], slices[si + 1]
        for ci1, vi, vj, oi in zip(s1["ci"], s1["vi"], s1["vj"], s1["oi"]):
            ci0 = s0["key_to_ci"].get((int(vi), int(vj), int(oi)))
            if ci0 is None:
                continue
            gi = s0["offset"] + ci0
            gj = s1["offset"] + int(ci1)
            var_r += (eigvec[gi] - eigvec[gj]) ** 2
            edges_r += 1

    # Angular edges: within each R-slice
    for s in slices:
        off = s["offset"]
        n_total_slice = n_v * n_v * n_omega
        compact = np.full(n_total_slice, -1, dtype=np.int64)
        for ci, vi, vj, oi in zip(s["ci"], s["vi"], s["vj"], s["oi"]):
            compact[vi * n_v * n_omega + vj * n_omega + oi] = ci

        for ci, vi, vj, oi in zip(s["ci"], s["vi"], s["vj"], s["oi"]):
            psi_i = eigvec[off + ci]

            for ni in neighbors[vi]:
                cj = compact[ni * n_v * n_omega + vj * n_omega + oi]
                if cj >= 0:
                    var_a += (psi_i - eigvec[off + cj]) ** 2
                    edges_a += 1

            for nj in neighbors[vj]:
                cj = compact[vi * n_v * n_omega + nj * n_omega + oi]
                if cj >= 0:
                    var_b += (psi_i - eigvec[off + cj]) ** 2
                    edges_b += 1

            for oi_n in [(oi - 1) % n_omega, (oi + 1) % n_omega]:
                cj = compact[vi * n_v * n_omega + vj * n_omega + oi_n]
                if cj >= 0:
                    var_omega += (psi_i - eigvec[off + cj]) ** 2
                    edges_omega += 1

    norm_r = var_r / edges_r if edges_r > 0 else 0.0
    norm_a = var_a / edges_a if edges_a > 0 else 0.0
    norm_b = var_b / edges_b if edges_b > 0 else 0.0
    norm_omega = var_omega / edges_omega if edges_omega > 0 else 0.0

    total = norm_r + norm_a + norm_b + norm_omega
    if total < 1e-30:
        return (0.25, 0.25, 0.25, 0.25)
    return (norm_r / total, norm_a / total, norm_b / total, norm_omega / total)


# ---------------------------------------------------------------------------
# Free normalization: product spectrum of 1D chain × angular graphs
# ---------------------------------------------------------------------------


def icosphere_laplacian_evals(neighbors: list[list[int]]) -> np.ndarray:
    n_v = len(neighbors)
    lap = np.zeros((n_v, n_v))
    for i, nbrs in enumerate(neighbors):
        lap[i, i] = len(nbrs)
        for j in nbrs:
            lap[i, j] = -1.0
    return np.sort(np.maximum(np.linalg.eigvalsh(lap), 0.0))


def free_eigenvalues_6d(
    n_R: int,
    neighbors: list[list[int]],
    n_omega: int,
) -> float:
    """Spectral gap of the free 6D product graph (smallest non-zero eigenvalue)."""
    chain_evals = [2.0 * (1.0 - np.cos(np.pi * k / n_R)) for k in range(n_R)]
    ico_evals = icosphere_laplacian_evals(neighbors)
    ring_evals = [2.0 * (1.0 - np.cos(2.0 * np.pi * m / n_omega)) for m in range(n_omega)]

    min_nz = float("inf")
    for lr in chain_evals:
        for la in ico_evals:
            for lb in ico_evals:
                for lw in ring_evals:
                    total = lr + la + lb + lw
                    if total > 1e-10:
                        min_nz = min(min_nz, total)
    return min_nz


def free_coordinate_fractions_6d(
    n_R: int,
    neighbors: list[list[int]],
    n_omega: int,
) -> tuple[float, float, float, float]:
    """Coordinate fractions (R, A, B, ω) for the free-diffusion spectral gap mode."""
    chain_evals = sorted(2.0 * (1.0 - np.cos(np.pi * k / n_R)) for k in range(n_R))
    ico_evals = icosphere_laplacian_evals(neighbors)
    ring_evals = sorted(2.0 * (1.0 - np.cos(2.0 * np.pi * m / n_omega)) for m in range(n_omega))

    best = (float("inf"), 0.0, 0.0, 0.0, 0.0)
    for lr in chain_evals:
        for la in ico_evals:
            for lb in ico_evals:
                for lw in ring_evals:
                    total = lr + la + lb + lw
                    if total > 1e-10 and total < best[0]:
                        best = (total, lr / total, la / total, lb / total, lw / total)
    return best[1:]


# ---------------------------------------------------------------------------
# Concentration scan
# ---------------------------------------------------------------------------


def diffusion_vs_concentration(
    directory: Path,
    molarities: np.ndarray,
    beta: float,
    k: int = 4,
    dense_threshold: int = 500,
    r_max: float | None = None,
    pmf_threshold: float = 0.05,
) -> list[dict]:
    """Compute 6D spectral gap at each concentration."""
    meta = load_metadata(directory)
    results = []

    n_v_dom = int(meta["n_v"].mode().iloc[0])
    n_omega = int(meta["n_omega"].iloc[0])
    neighbors = load_icosphere_neighbors(directory, n_v_dom)
    if neighbors is None:
        log.error("No icosphere neighbors found for n_v=%d", n_v_dom)
        return results

    for c in molarities:
        r_cell = r_cell_from_molarity(c)
        log.info("c=%.4e M, R_cell=%.1f Å", c, r_cell)

        ss = build_state_space(meta, r_cell, directory, r_max, pmf_threshold)
        if ss is None or len(ss["slices"]) < 2:
            log.warning("  Too few R-slices within R_cell=%.1f, skipping", r_cell)
            continue

        n_R = len(ss["slices"])
        t0 = time.perf_counter()

        L_sym, pi = build_6d_generator(ss, beta)
        vals, vecs, method = solve_eigenproblem(L_sym, k, dense_threshold)
        elapsed = time.perf_counter() - t0

        non_null = np.abs(vals) > ZERO_THRESHOLD
        vals_nn = vals[non_null]
        vecs_nn = vecs[:, non_null]

        if len(vals_nn) == 0:
            log.warning("  No non-null eigenvalues found")
            continue

        lambda1_free = free_eigenvalues_6d(n_R, neighbors, n_omega)
        free_fracs = free_coordinate_fractions_6d(n_R, neighbors, n_omega)

        spectral_gap = vals_nn[0]
        ratio = abs(spectral_gap) / lambda1_free if lambda1_free > 0 else float("nan")

        frac_r, frac_a, frac_b, frac_w = coordinate_projection_6d(
            vecs_nn[:, 0], ss, neighbors,
        )

        fr0, fa0, fb0, fw0 = free_fracs
        # Free spectral gap is often purely translational (fr0≈1, others≈0).
        # Normalized fractions are only meaningful when free fraction > 0.
        norm_fr = frac_r / fr0 if fr0 > 0.01 else frac_r
        norm_fa = frac_a / fa0 if fa0 > 0.01 else frac_a
        norm_fb = frac_b / fb0 if fb0 > 0.01 else frac_b
        norm_fw = frac_w / fw0 if fw0 > 0.01 else frac_w

        log.info(
            "  n=%d  λ₁/λ_free=%.4f  frac(R=%.2f A=%.2f B=%.2f ω=%.2f)  %.2fs [%s]",
            ss["total_states"], ratio,
            frac_r, frac_a, frac_b, frac_w,
            elapsed, method,
        )

        row = {
            "c/M": c,
            "R_cell/Å": r_cell,
            "n_R": n_R,
            "n_states": ss["total_states"],
            "D/D⁰": ratio,
            "frac_R": frac_r,
            "frac_A": frac_a,
            "frac_B": frac_b,
            "frac_ω": frac_w,
            "frac_R_norm": norm_fr,
            "frac_A_norm": norm_fa,
            "frac_B_norm": norm_fb,
            "frac_ω_norm": norm_fw,
            "spectral_gap": spectral_gap,
            "lambda1_free": lambda1_free,
            "method": method,
        }
        for i, v in enumerate(vals_nn[:k]):
            row[f"lambda{i}"] = v

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory", type=Path,
        help="Directory with generator_R*.mtx, coords_R*.csv, metadata.csv",
    )
    parser.add_argument("--cmin", type=float, required=True, help="Min concentration (M)")
    parser.add_argument("--cmax", type=float, required=True, help="Max concentration (M)")
    parser.add_argument("--cn", type=int, default=10, help="Number of concentration points")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature (K)")
    parser.add_argument("--rmax", type=float, default=None,
                        help="Max R (Å) to include; auto-truncates by PMF if omitted")
    parser.add_argument("--pmf-threshold", type=float, default=0.05,
                        help="Auto-truncate R where |PMF/kT| < this (default: 0.05)")
    parser.add_argument("--k", type=int, default=4, help="Number of eigenvalues")
    parser.add_argument(
        "--dense-threshold", type=int, default=10000,
        help="Use dense solver below this state count (default: 10000)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    import warnings
    warnings.filterwarnings("ignore", message=".*not reaching the requested tolerance.*")
    warnings.filterwarnings("ignore", message=".*Exited postprocessing.*")

    kB = 1.380649e-23  # J/K
    beta = 1.0 / (kB * args.temperature * AVOGADRO / 1000.0)  # 1/(kJ/mol)

    molarities = np.linspace(args.cmin, args.cmax, args.cn)

    results = diffusion_vs_concentration(
        args.directory.resolve(), molarities, beta, args.k, args.dense_threshold,
        args.rmax, args.pmf_threshold,
    )

    if not results:
        log.error("No results produced")
        sys.exit(1)

    lambda_cols = [f"lambda{i}" for i in range(args.k)]
    fieldnames = [
        "c/M", "R_cell/Å", "n_R", "n_states", "D/D⁰",
        "frac_R", "frac_A", "frac_B", "frac_ω",
        "frac_R_norm", "frac_A_norm", "frac_B_norm", "frac_ω_norm",
        "spectral_gap", "lambda1_free",
    ] + lambda_cols + ["method"]

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

    log.info("Done. %d concentration points.", len(results))


if __name__ == "__main__":
    main()

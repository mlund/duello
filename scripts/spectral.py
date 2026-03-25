"""
Compute spectral gaps and coordinate projections from exported generator matrices.

Reads Matrix Market files produced by `duello` and finds the k eigenvalues
nearest to zero for each R-slice.  The spectral gap λ₁ (first non-zero
eigenvalue) determines the association/dissociation rate.

Since all generator eigenvalues are ≤ 0, eigenvalues nearest zero are the
*largest*, so LOBPCG (which finds extremal eigenvalues) works directly —
no shift-invert factorization needed.

When coordinate maps and icosphere neighbors are available (exported by
duello alongside the matrices), eigenvector decomposition into mol A,
mol B, and ω contributions is computed via edge-variance projection.

Usage:
    python spectral.py <matrix_dir> [--k 4] [--dense-threshold 500]

Output:
    CSV to stdout with columns: R, n_active, lambda1_over_free, spectral_gap,
    frac_a, frac_b, frac_omega, lambda0..N, method
"""

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import scipy.sparse.linalg

log = logging.getLogger(__name__)

# Eigenvalues below this magnitude are treated as the null eigenvalue (λ₀ ≈ 0)
ZERO_THRESHOLD = 1e-6


def load_metadata(directory: Path) -> pd.DataFrame:
    path = directory / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {directory}")
    df = pd.read_csv(path)
    df = df.set_index("R")
    log.info("Loaded metadata: %d R-slices, λ₁_free = %.6g", len(df), df["lambda1_free"].iloc[0])
    return df


def count_null_space(A_sparse) -> int:
    """Count connected components via the graph Laplacian null space.

    Each disconnected component contributes one zero eigenvalue.
    We detect this cheaply from the sparsity pattern using scipy.
    """
    import scipy.sparse.csgraph
    # Treat as undirected adjacency (ignore diagonal / signs)
    n_components, _ = scipy.sparse.csgraph.connected_components(A_sparse, directed=False)
    return n_components


def load_coords(directory: Path, R: float) -> pd.DataFrame | None:
    """Load compact_index → (vi, vj, oi) mapping for an R-slice."""
    path = directory / f"coords_R{R:.1f}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_icosphere_neighbors(directory: Path, n_v: int) -> list[list[int]] | None:
    """Load icosphere adjacency list for a given vertex count."""
    path = directory / f"icosphere_nv{n_v}.csv"
    if not path.exists():
        return None
    neighbors: list[list[int]] = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            _, nbrs = line.strip().split(",", 1)
            neighbors.append([int(x) for x in nbrs.split()])
    return neighbors


def coordinate_projection(
    eigvec: np.ndarray,
    coords: pd.DataFrame,
    neighbors: list[list[int]],
    n_v: int,
    n_omega: int,
) -> tuple[float, float, float]:
    """Decompose eigenvector into mol A / mol B / ω fractions via edge variance.

    Matches the Rust `coordinate_projection()` in diffusion.rs: for each
    directed edge in each coordinate direction, accumulate [ψ(i) - ψ(j)]²,
    normalize by edge count per direction, then report fractions.
    """
    # Build compact index lookup: full_index → compact_index (or -1)
    n_total = n_v * n_v * n_omega
    compact = np.full(n_total, -1, dtype=np.int64)
    ci_arr = coords["compact"].values
    vi_arr = coords["vi"].values
    vj_arr = coords["vj"].values
    oi_arr = coords["oi"].values
    for ci, vi, vj, oi in zip(ci_arr, vi_arr, vj_arr, oi_arr):
        full_idx = vi * n_v * n_omega + vj * n_omega + oi
        compact[full_idx] = ci

    var_a = var_b = var_omega = 0.0
    edges_a = edges_b = edges_omega = 0

    for ci, vi, vj, oi in zip(ci_arr, vi_arr, vj_arr, oi_arr):
        psi_i = eigvec[ci]

        # Mol A edges: vary vi, keep vj and oi fixed
        for ni in neighbors[vi]:
            cj = compact[ni * n_v * n_omega + vj * n_omega + oi]
            if cj >= 0:
                var_a += (psi_i - eigvec[cj]) ** 2
                edges_a += 1

        # Mol B edges: vary vj, keep vi and oi fixed
        for nj in neighbors[vj]:
            cj = compact[vi * n_v * n_omega + nj * n_omega + oi]
            if cj >= 0:
                var_b += (psi_i - eigvec[cj]) ** 2
                edges_b += 1

        # Omega edges: cyclic ring neighbors
        oi_prev = (oi - 1) % n_omega
        oi_next = (oi + 1) % n_omega
        for oi_n in (oi_prev, oi_next):
            cj = compact[vi * n_v * n_omega + vj * n_omega + oi_n]
            if cj >= 0:
                var_omega += (psi_i - eigvec[cj]) ** 2
                edges_omega += 1

    norm_a = var_a / edges_a if edges_a > 0 else 0.0
    norm_b = var_b / edges_b if edges_b > 0 else 0.0
    norm_omega = var_omega / edges_omega if edges_omega > 0 else 0.0

    total = norm_a + norm_b + norm_omega
    if total < 1e-30:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (norm_a / total, norm_b / total, norm_omega / total)


def solve_dense(A_dense: np.ndarray, k: int, n_components: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Full symmetric eigensolver, requesting enough eigenvalues to skip the null space."""
    n = A_dense.shape[0]
    k_req = min(n_components + k, n)
    lo = max(0, n - k_req)
    vals, vecs = scipy.linalg.eigh(A_dense, subset_by_index=[lo, n - 1])
    # Reverse so largest (nearest zero) come first
    return vals[::-1], vecs[:, ::-1], "dense"


def solve_lobpcg(A_sparse, k: int, n_components: int) -> tuple[np.ndarray, np.ndarray, str]:
    """LOBPCG for largest eigenvalues (nearest zero) of the generator.

    All generator eigenvalues are ≤ 0, so the k eigenvalues nearest zero
    are the k largest.  We request extra vectors to cover the null space
    from disconnected components.
    """
    n = A_sparse.shape[0]
    k_req = min(n_components + k, n - 1)
    rng = np.random.default_rng(seed=42)
    X0 = rng.standard_normal((n, k_req))
    vals, vecs = scipy.sparse.linalg.lobpcg(
        A_sparse, X0, largest=True, tol=1e-8, maxiter=500, verbosityLevel=0,
    )
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order], "lobpcg"


def process_matrix(path: Path, meta_row: pd.Series, k: int, dense_threshold: int,
                   directory: Path) -> dict:
    R = float(meta_row.name)
    n_active = int(meta_row["n_active"])
    lambda1_free = float(meta_row["lambda1_free"])
    n_v = int(meta_row["n_v"])
    n_omega = int(meta_row["n_omega"])

    A = scipy.io.mmread(path).tocsc()
    n_components = count_null_space(A)

    t0 = time.perf_counter()
    if n_active <= dense_threshold:
        vals, vecs, method = solve_dense(A.toarray(), k, n_components)
    else:
        try:
            vals, vecs, method = solve_lobpcg(A, k, n_components)
        except Exception as exc:
            log.warning("R=%.1f: LOBPCG failed (%s), falling back to dense", R, exc)
            vals, vecs, method = solve_dense(A.toarray(), k, n_components)
    elapsed = time.perf_counter() - t0

    # Keep only the k eigenvalues/vectors past the null space
    non_null_mask = np.abs(vals) > ZERO_THRESHOLD
    vals_nonnull = vals[non_null_mask]
    vecs_nonnull = vecs[:, non_null_mask]

    vals_out = list(vals_nonnull[:k])
    vals_out += [float("nan")] * (k - len(vals_out))

    # Spectral gap: first non-null eigenvalue (all are ≤ 0).
    spectral_gap = vals_out[0] if vals_out and np.isfinite(vals_out[0]) else float("nan")
    ratio = abs(spectral_gap) / lambda1_free if np.isfinite(spectral_gap) else float("nan")

    # Coordinate projection for the spectral-gap eigenvector
    frac_a = frac_b = frac_omega = float("nan")
    if np.isfinite(spectral_gap) and vecs_nonnull.shape[1] > 0:
        coords = load_coords(directory, R)
        neighbors = load_icosphere_neighbors(directory, n_v)
        if coords is not None and neighbors is not None:
            frac_a, frac_b, frac_omega = coordinate_projection(
                vecs_nonnull[:, 0], coords, neighbors, n_v, n_omega,
            )

    log.info(
        "R=%5.1f Å  n=%5d  comp=%3d  λ₁/λ_free=%.4f  "
        "frac(A=%.2f B=%.2f ω=%.2f)  %.3fs [%s]",
        R, n_active, n_components, ratio, frac_a, frac_b, frac_omega, elapsed, method,
    )

    row = {"R": R, "n_active": n_active, "n_components": n_components,
           "method": method, "spectral_gap": spectral_gap, "lambda1_over_free": ratio,
           "frac_a": frac_a, "frac_b": frac_b, "frac_omega": frac_omega}
    for i, v in enumerate(vals_out):
        row[f"lambda{i}"] = v
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Directory containing generator_R*.mtx and metadata.csv")
    parser.add_argument("--k", type=int, default=4, help="Number of eigenvalues to compute (default: 4)")
    parser.add_argument("--dense-threshold", type=int, default=500,
                        help="Use dense solver for n_active ≤ this (default: 500)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    directory = args.directory.resolve()
    if not directory.is_dir():
        log.error("Not a directory: %s", directory)
        sys.exit(1)

    meta = load_metadata(directory)

    mtx_files = sorted(
        directory.glob("generator_R*.mtx"),
        key=lambda p: float(re.search(r"R([\d.]+?)\.mtx", p.name).group(1)),
    )
    if not mtx_files:
        log.error("No generator_R*.mtx files found in %s", directory)
        sys.exit(1)
    log.info("Found %d matrix files", len(mtx_files))

    import warnings
    warnings.filterwarnings("ignore", message=".*not reaching the requested tolerance.*")
    warnings.filterwarnings("ignore", message=".*Exited postprocessing.*")

    results = []
    t_total = time.perf_counter()
    for path in mtx_files:
        R = float(re.search(r"R([\d.]+?)\.mtx", path.name).group(1))
        if R not in meta.index:
            log.warning("No metadata for R=%.1f, skipping", R)
            continue
        results.append(process_matrix(path, meta.loc[R], args.k, args.dense_threshold, directory))
    t_total = time.perf_counter() - t_total

    if not results:
        log.error("No results produced")
        sys.exit(1)

    # Write CSV to stdout
    lambda_cols = [f"lambda{i}" for i in range(args.k)]
    fieldnames = ["R", "n_active", "n_components", "lambda1_over_free", "spectral_gap",
                  "frac_a", "frac_b", "frac_omega"] + lambda_cols + ["method"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

    log.info("Done. %d R-slices in %.1fs.", len(results), t_total)


if __name__ == "__main__":
    main()

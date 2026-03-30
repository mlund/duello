"""
Mean first passage time (MFPT) analysis for translational diffusion.

Computes D_t/D_t^0 from the MFPT across the cell in 1D (PMF-based)
and from the full 6D spectral decomposition (captures trans/rot coupling).

1D MFPT (Kramers/Szabo):
    τ = (1/D₀) ∫ exp(βw(x)) [∫ exp(-βw(y)) dy] dx
    D_eff/D₀ = τ_free / τ

6D spectral MFPT:
    τ = Σ_k (1/|λ_k|) × |⟨ψ_k|δR⟩|²
    where δR is a radial displacement observable and ψ_k are eigenmodes.
    D_eff/D₀ = τ_free / τ

Usage:
    python mfpt6d.py <matrix_dir> [--homo] [--cmin 3.5e-5] [--cmax 1.4e-3]
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

AVOGADRO = 6.02214076e23


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metadata(directory: Path) -> pd.DataFrame:
    return pd.read_csv(directory / "metadata.csv")


def load_coords(directory: Path, R: float) -> pd.DataFrame | None:
    path = directory / f"coords_R{R:.1f}.csv"
    return pd.read_csv(path) if path.exists() else None


def r_cell_from_molarity(molarity: float) -> float:
    c_m3 = molarity * 1000.0
    return (3.0 / (4.0 * np.pi * c_m3 * AVOGADRO)) ** (1.0 / 3.0) * 1e10


def symmetrize_exchange(energies, n_v, n_omega):
    for vi in range(n_v):
        for vj in range(vi, n_v):
            for oi in range(n_omega):
                oi_swap = (n_omega - oi) % n_omega
                idx_fwd = vi * n_v * n_omega + vj * n_omega + oi
                idx_rev = vj * n_v * n_omega + vi * n_omega + oi_swap
                if idx_fwd == idx_rev:
                    continue
                u_fwd, u_rev = energies[idx_fwd], energies[idx_rev]
                if np.isfinite(u_fwd) and np.isfinite(u_rev):
                    avg = 0.5 * (u_fwd + u_rev)
                    energies[idx_fwd] = avg
                    energies[idx_rev] = avg


# ---------------------------------------------------------------------------
# 1D MFPT on PMF
# ---------------------------------------------------------------------------


def mfpt_1d(pmf_kt, dr, n_free):
    """MFPT across [R_min, R_cell] in a 1D potential w(R).

    τ = Σ_i exp(βw_i) × Σ_{j≤i} exp(-βw_j) × dr²
    τ_free = same with w=0.

    pmf_kt: w(R)/kT array for table region.
    n_free: number of free slices (w=0) beyond table up to R_cell.

    Returns D_eff/D₀ = τ_free/τ.
    """
    n_tab = len(pmf_kt)
    n_total = n_tab + n_free

    if n_total < 2:
        return 1.0

    # Full PMF array including free tail
    w = np.zeros(n_total)
    w[:n_tab] = pmf_kt

    # τ = Σ_i exp(w_i) × [Σ_{j=0}^{i} exp(-w_j)] × dr²
    # Use cumulative sum for the inner integral
    exp_minus = np.exp(-w)
    exp_plus = np.exp(w)
    cumsum_minus = np.cumsum(exp_minus)
    tau = np.sum(exp_plus * cumsum_minus) * dr * dr

    # Free MFPT: w=0 everywhere → τ_free = Σ_i 1 × (i+1) × dr²
    tau_free = np.sum(np.arange(1, n_total + 1, dtype=float)) * dr * dr

    return tau_free / tau if tau > 0 else 1.0


# ---------------------------------------------------------------------------
# 6D spectral MFPT
# ---------------------------------------------------------------------------


def mfpt_6d_spectral(directory, meta, beta, r_cell, homo_dimer=False):
    """D_eff/D₀ from 6D spectral decomposition of the MFPT.

    τ = Σ_k (1/|λ_k|) × |⟨ψ_k|δR⟩|²
    where δR_i = R_i - ⟨R⟩_eq is the radial displacement observable
    projected onto the equilibrium-weighted 6D states.

    For the symmetrized generator, ψ_k are orthonormal eigenvectors
    and λ_k are eigenvalues (≤ 0). The sum skips λ₀ = 0 (equilibrium).
    """
    import scipy.io
    import scipy.sparse

    dr = float(meta["R"].diff().iloc[1]) if len(meta) > 1 else 1.0

    # Restrict to dominant mesh level within cell
    in_cell = meta[meta["R"] <= r_cell].copy()
    if in_cell.empty:
        return 1.0, {}
    dominant_nv = int(in_cell["n_v"].mode().iloc[0])
    in_cell = in_cell[in_cell["n_v"] == dominant_nv]

    # Load slices
    slices = []
    offset = 0
    for _, row in in_cell.iterrows():
        R = float(row["R"])
        n_v = int(row["n_v"])
        n_omega = int(row["n_omega"])
        coords = load_coords(directory, R)
        if coords is None or "energy" not in coords.columns:
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

        if homo_dimer:
            n_total_slice = n_v * n_v * n_omega
            full_e = np.full(n_total_slice, np.inf)
            for ci, vi, vj, oi, e in zip(ci_arr, vi_arr, vj_arr, oi_arr, energies):
                full_e[vi * n_v * n_omega + vj * n_omega + oi] = e
            symmetrize_exchange(full_e, n_v, n_omega)
            for idx, (ci, vi, vj, oi) in enumerate(zip(ci_arr, vi_arr, vj_arr, oi_arr)):
                energies[idx] = full_e[vi * n_v * n_omega + vj * n_omega + oi]

        key_to_ci = {}
        for ci, vi, vj, oi in zip(ci_arr, vi_arr, vj_arr, oi_arr):
            key_to_ci[(int(vi), int(vj), int(oi))] = int(ci)

        slices.append({
            "R": R, "ci": ci_arr, "vi": vi_arr, "vj": vj_arr, "oi": oi_arr,
            "energies": energies, "key_to_ci": key_to_ci,
            "n_active": len(coords), "offset": offset, "generator": gen,
        })
        offset += len(coords)

    n_total = offset
    n_slices = len(slices)
    if n_slices < 2 or n_total < 10:
        return 1.0, {}

    # Build 6D generator (Rust convention)
    rows, cols, vals = [], [], []

    for s in slices:
        off = s["offset"]
        coo = s["generator"].tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            rows.append(off + i)
            cols.append(off + j)
            vals.append(v)

    for si in range(n_slices - 1):
        s0, s1 = slices[si], slices[si + 1]
        for ci1, vi, vj, oi in zip(s1["ci"], s1["vi"], s1["vj"], s1["oi"]):
            ci0 = s0["key_to_ci"].get((int(vi), int(vj), int(oi)))
            if ci0 is None:
                continue
            gi = s0["offset"] + ci0
            gj = s1["offset"] + int(ci1)
            du = s1["energies"][int(ci1)] - s0["energies"][ci0]
            half_du = 0.5 * beta * du
            rows.extend([gi, gj, gi, gj])
            cols.extend([gj, gi, gi, gj])
            vals.extend([1.0, 1.0, -np.exp(-half_du), -np.exp(half_du)])

    L = scipy.sparse.coo_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(n_total, n_total),
    ).tocsc()

    # Dense eigen (only feasible for moderate n_total)
    if n_total > 50000:
        log.warning("  n=%d too large for dense MFPT, skipping 6D spectral", n_total)
        return float("nan"), {}

    log.info("  Dense eigen on %d×%d matrix...", n_total, n_total)
    import scipy.linalg
    evals, evecs = scipy.linalg.eigh(L.toarray())
    # Sort descending (largest = 0, then negative)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Build radial displacement observable: δR_i = R_i - ⟨R⟩_eq
    R_vec = np.empty(n_total)
    for s in slices:
        R_vec[s["offset"]:s["offset"] + s["n_active"]] = s["R"]

    # Equilibrium weights: π ∝ R² exp(-βU) — recover from generator null vector
    # The null eigenvector of L_sym is proportional to sqrt(π)
    # But our matrix is NOT L_sym, it's the Rust-convention matrix.
    # Use the Boltzmann weights directly.
    log_pi = np.empty(n_total)
    for s in slices:
        off = s["offset"]
        log_pi[off:off + s["n_active"]] = -beta * s["energies"] + 2.0 * np.log(s["R"])
    log_pi -= log_pi.max()
    pi = np.exp(log_pi)
    pi /= pi.sum()

    R_mean = np.sum(pi * R_vec)
    delta_R = R_vec - R_mean

    # MFPT from spectral decomposition:
    # τ = Σ_{k: λ_k < 0} |⟨ψ_k|δR⟩_π|² / |λ_k|
    # where ⟨ψ_k|δR⟩_π = Σ_i ψ_k(i) × δR(i) × π(i) (π-weighted projection)
    #
    # But the eigenvectors of the Rust-convention matrix are NOT the same as
    # those of L_sym. We need to be careful about the weighting.
    #
    # For simplicity, use the direct approach: compute the π-weighted
    # correlation function ⟨δR(0) δR(t)⟩ decay from the eigenmodes.
    # C(t) = Σ_k |c_k|² exp(λ_k t), c_k = Σ_i ψ_k(i) δR(i) sqrt(π_i)
    # τ = ∫₀^∞ C(t)/C(0) dt = Σ_k |c_k|² / |λ_k| / C(0)

    sqrt_pi = np.sqrt(pi)
    delta_R_weighted = delta_R * sqrt_pi

    tau = 0.0
    tau_contributions = []
    for k in range(n_total):
        if evals[k] > -1e-10:
            continue  # skip null eigenvalues
        ck = np.dot(evecs[:, k], delta_R_weighted)
        contrib = ck * ck / abs(evals[k])
        tau += contrib
        tau_contributions.append((evals[k], ck * ck, contrib))

    # Free MFPT: same but with uniform potential (all evals from free Laplacian)
    # For simplicity, compute numerically: same matrix with U=0
    R_arr = np.array([s["R"] for s in slices])
    n_R = len(R_arr)
    pi_free = np.ones(n_total) / n_total
    R_mean_free = np.mean(R_vec)
    delta_R_free = R_vec - R_mean_free
    sqrt_pi_free = np.sqrt(pi_free)
    delta_R_free_w = delta_R_free * sqrt_pi_free

    # Free generator: same graph, U=0 → off-diag=1, diag=-degree
    L_free_rows, L_free_cols, L_free_vals = [], [], []
    for s in slices:
        off = s["offset"]
        coo = s["generator"].tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i != j:
                L_free_rows.extend([off + i, off + i])
                L_free_cols.extend([off + j, off + i])
                L_free_vals.extend([1.0, -1.0])

    for si in range(n_slices - 1):
        s0, s1 = slices[si], slices[si + 1]
        for ci1, vi, vj, oi in zip(s1["ci"], s1["vi"], s1["vj"], s1["oi"]):
            ci0 = s0["key_to_ci"].get((int(vi), int(vj), int(oi)))
            if ci0 is None:
                continue
            gi = s0["offset"] + ci0
            gj = s1["offset"] + int(ci1)
            L_free_rows.extend([gi, gj, gi, gj])
            L_free_cols.extend([gj, gi, gi, gj])
            L_free_vals.extend([1.0, 1.0, -1.0, -1.0])

    L_free = scipy.sparse.coo_matrix(
        (np.array(L_free_vals),
         (np.array(L_free_rows, dtype=np.int64), np.array(L_free_cols, dtype=np.int64))),
        shape=(n_total, n_total),
    ).tocsc()

    evals_f, evecs_f = scipy.linalg.eigh(L_free.toarray())
    order_f = np.argsort(evals_f)[::-1]
    evals_f = evals_f[order_f]
    evecs_f = evecs_f[:, order_f]

    tau_free = 0.0
    for k in range(n_total):
        if evals_f[k] > -1e-10:
            continue
        ck = np.dot(evecs_f[:, k], delta_R_free_w)
        tau_free += ck * ck / abs(evals_f[k])

    d_ratio = tau_free / tau if tau > 0 else 1.0

    info = {
        "tau": tau,
        "tau_free": tau_free,
        "n_modes": len(tau_contributions),
        "top_modes": sorted(tau_contributions, key=lambda x: -x[2])[:5],
    }
    return d_ratio, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--homo", action="store_true")
    parser.add_argument("--cmin", type=float, default=0.035e-3)
    parser.add_argument("--cmax", type=float, default=1.398e-3)
    parser.add_argument("--cn", type=int, default=15)
    parser.add_argument("--rmax", type=float, default=None,
                        help="Limit R range for 6D spectral (reduces matrix size)")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    kB = 1.380649e-23
    beta = 1.0 / (kB * args.temperature * AVOGADRO / 1000.0)

    directory = args.directory.resolve()
    meta = load_metadata(directory)
    if args.rmax:
        meta = meta[meta["R"] <= args.rmax]

    dr = float(meta["R"].diff().iloc[1]) if len(meta) > 1 else 1.0
    bw = meta["boltzmann_weight"].values if "boltzmann_weight" in meta.columns else np.ones(len(meta))
    pmf_kt = -np.log(np.clip(bw, 1e-30, None))

    molarities = np.linspace(args.cmin, args.cmax, args.cn)
    dt_1d = np.empty(args.cn)
    dt_6d = np.empty(args.cn)

    for i, c in enumerate(molarities):
        r_cell = r_cell_from_molarity(c)
        r_max_tab = meta["R"].max()
        n_free = max(0, int((r_cell - r_max_tab) / dr))

        # 1D MFPT
        mask = meta["R"].values <= r_cell
        dt_1d[i] = mfpt_1d(pmf_kt[mask], dr, n_free)

        # 6D spectral MFPT
        dt_6d[i], info = mfpt_6d_spectral(directory, meta, beta, r_cell, args.homo)

        log.info("c=%.3e M  R_cell=%6.1f  D_t(1D)=%.4f  D_t(6D)=%.4f",
                 c, r_cell, dt_1d[i], dt_6d[i])
        if info.get("top_modes"):
            for lam, ck2, contrib in info["top_modes"][:3]:
                log.info("    λ=%.4e  |c_k|²=%.4e  contrib=%.4e", lam, ck2, contrib)

    # --- Plot ---
    c_mM = molarities * 1e3
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.plot(meta["R"].values, pmf_kt, "k-", lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("w(R) / kT")
    ax.set_title("PMF")

    ax = axes[1]
    ax.plot(c_mM, dt_1d, "o-", label="D_t/D_t⁰ (1D MFPT)", ms=4, lw=1.5)
    valid = np.isfinite(dt_6d)
    if valid.any():
        ax.plot(c_mM[valid], dt_6d[valid], "s-", label="D_t/D_t⁰ (6D spectral MFPT)",
                ms=4, lw=1.5)
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("c (mM)")
    ax.set_ylabel("D_t / D_t⁰")
    ax.set_title("Translational diffusion (MFPT)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("mfpt6d_results.png", dpi=150, bbox_inches="tight")
    log.info("Saved mfpt6d_results.png")
    plt.show()


if __name__ == "__main__":
    main()

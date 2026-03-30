"""
6D Zwanzig diffusion coefficients vs protein concentration.

Extends the 5D per-R angular Zwanzig to include R as a 4th coordinate.
Marginal projections give D_R (translational), D_A, D_B, D_ω (rotational),
and separability = D_full / (D_R × D_A × D_B × D_ω).

Cell model: within [R_min, R_cell], Boltzmann-weighted; free beyond table.

Usage:
    python zwanzig6d.py <matrix_dir> [--homo] [--cmin 3.5e-5] [--cmax 1.4e-3]
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
    """Cell radius in Å from molar concentration."""
    c_m3 = molarity * 1000.0
    return (3.0 / (4.0 * np.pi * c_m3 * AVOGADRO)) ** (1.0 / 3.0) * 1e10


# ---------------------------------------------------------------------------
# Exchange symmetrization for homodimers
# ---------------------------------------------------------------------------


def symmetrize_exchange(energies, n_v, n_omega):
    """U_sym(vi,vj,oi) = ½[U(vi,vj,oi) + U(vj,vi,-oi)]."""
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
# Zwanzig helpers (mirrors diffusion.rs)
# ---------------------------------------------------------------------------


def zwanzig(energies, beta):
    """Zwanzig D/D₀ = 1/[⟨exp(βU)⟩ × ⟨exp(-βU)⟩] in log-space.

    Also returns ⟨exp(-βU)⟩ (Boltzmann weight for cell model).
    """
    finite = energies[np.isfinite(energies)]
    if len(finite) < 2:
        return 1.0, 1.0
    u_min = finite.min()
    shifted = finite - u_min
    n = len(finite)
    log_avg_minus = np.log(np.sum(np.exp(-beta * shifted))) - np.log(n)
    max_plus = (beta * shifted).max()
    log_avg_plus = max_plus + np.log(np.sum(np.exp(beta * shifted - max_plus))) - np.log(n)
    d_ratio = np.exp(-log_avg_minus - log_avg_plus)
    bw = np.exp(log_avg_minus - beta * u_min)  # true ⟨exp(-βU)⟩
    return d_ratio, bw


def marginal_pmf(n_bins, n_inner, beta, energy_at):
    """Marginal PMF: w(k) = -kT ln Σ_j exp(-βU(k,j))."""
    pmf = np.empty(n_bins)
    for k in range(n_bins):
        terms = []
        for j in range(n_inner):
            u = energy_at(k, j)
            if np.isfinite(u):
                terms.append(-beta * u)
        if not terms:
            pmf[k] = np.inf
        else:
            mx = max(terms)
            pmf[k] = -(mx + np.log(sum(np.exp(t - mx) for t in terms))) / beta
    return pmf


def zwanzig_1d(pmf, beta):
    return zwanzig(pmf, beta)[0]


# ---------------------------------------------------------------------------
# 6D Zwanzig with marginal projections
# ---------------------------------------------------------------------------


def zwanzig_6d(directory, meta, beta, r_cell, homo_dimer=False):
    """Compute full 6D and marginal Zwanzig within cell, with free tail.

    Returns dict with D_full, D_R, D_A, D_B, D_omega, separability.
    """
    dr = meta["R"].diff().iloc[1] if len(meta) > 1 else 1.0
    in_cell = meta[meta["R"] <= r_cell]

    # Restrict to dominant mesh level
    dominant_nv = int(in_cell["n_v"].mode().iloc[0])
    in_cell = in_cell[in_cell["n_v"] == dominant_nv]

    slices = []
    for _, row in in_cell.iterrows():
        R = float(row["R"])
        n_v = int(row["n_v"])
        n_omega = int(row["n_omega"])
        coords = load_coords(directory, R)
        if coords is None or "energy" not in coords.columns:
            continue

        # Reconstruct full energy array (inf for inactive states)
        n_total = n_v * n_v * n_omega
        energies = np.full(n_total, np.inf)
        for _, c in coords.iterrows():
            idx = int(c["vi"]) * n_v * n_omega + int(c["vj"]) * n_omega + int(c["oi"])
            energies[idx] = c["energy"]

        if homo_dimer:
            symmetrize_exchange(energies, n_v, n_omega)

        slices.append({"R": R, "energies": energies, "n_v": n_v, "n_omega": n_omega})

    if not slices:
        return {k: 1.0 for k in ["D_full", "D_R", "D_A", "D_B", "D_omega", "separability"]}

    n_v = slices[0]["n_v"]
    n_omega = slices[0]["n_omega"]
    n_ang = n_v * n_v * n_omega
    n_R_tab = len(slices)

    # Free slices beyond table
    r_max = slices[-1]["R"]
    n_free = max(0, int((r_cell - r_max) / dr))

    # --- Full 6D Zwanzig ---
    # Flatten all energies into one array: [R0_ang0, R0_ang1, ..., R1_ang0, ...]
    all_energies = []
    for s in slices:
        all_energies.append(s["energies"])
    # Free slices: energy = 0
    for _ in range(n_free):
        all_energies.append(np.zeros(n_ang))
    all_e = np.concatenate(all_energies)
    D_full = zwanzig(all_e, beta)[0]

    n_R_total = n_R_tab + n_free

    # Helper: energy at 6D index (ri, vi, vj, oi) → flat index
    def e6d(ri, ang_idx):
        if ri < n_R_tab:
            return slices[ri]["energies"][ang_idx]
        return 0.0  # free region

    # --- Marginal PMF along R: w(R) = -kT ln Σ_Ω exp(-βU(R,Ω)) ---
    pmf_R = marginal_pmf(n_R_total, n_ang, beta,
                         lambda ri, j: e6d(ri, j))
    D_R = zwanzig_1d(pmf_R, beta)

    # --- Marginal along mol A (vi): marginalize over R, vj, oi ---
    n_inner_a = n_R_total * n_v * n_omega
    def energy_a(vi, j):
        ri = j // (n_v * n_omega)
        rem = j % (n_v * n_omega)
        vj = rem // n_omega
        oi = rem % n_omega
        return e6d(ri, vi * n_v * n_omega + vj * n_omega + oi)
    pmf_A = marginal_pmf(n_v, n_inner_a, beta, energy_a)
    D_A = zwanzig_1d(pmf_A, beta)

    # --- Marginal along mol B (vj) ---
    def energy_b(vj, j):
        ri = j // (n_v * n_omega)
        rem = j % (n_v * n_omega)
        vi = rem // n_omega
        oi = rem % n_omega
        return e6d(ri, vi * n_v * n_omega + vj * n_omega + oi)
    pmf_B = marginal_pmf(n_v, n_inner_a, beta, energy_b)
    D_B = zwanzig_1d(pmf_B, beta)

    # --- Marginal along ω ---
    n_inner_w = n_R_total * n_v * n_v
    def energy_w(oi, j):
        ri = j // (n_v * n_v)
        rem = j % (n_v * n_v)
        vi = rem // n_v
        vj = rem % n_v
        return e6d(ri, vi * n_v * n_omega + vj * n_omega + oi)
    pmf_omega = marginal_pmf(n_omega, n_inner_w, beta, energy_w)
    D_omega = zwanzig_1d(pmf_omega, beta)

    product = D_R * D_A * D_B * D_omega
    separability = D_full / product if product > 0 else 0.0

    return {
        "D_full": D_full,
        "D_R": D_R,
        "D_A": D_A,
        "D_B": D_B,
        "D_omega": D_omega,
        "separability": separability,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--homo", action="store_true", help="Apply exchange symmetrization")
    parser.add_argument("--cmin", type=float, default=0.035e-3)
    parser.add_argument("--cmax", type=float, default=1.398e-3)
    parser.add_argument("--cn", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    kB = 1.380649e-23
    beta = 1.0 / (kB * args.temperature * AVOGADRO / 1000.0)

    directory = args.directory.resolve()
    meta = load_metadata(directory)

    molarities = np.linspace(args.cmin, args.cmax, args.cn)
    results = {k: [] for k in ["D_full", "D_R", "D_A", "D_B", "D_omega", "separability"]}

    for c in molarities:
        r_cell = r_cell_from_molarity(c)
        res = zwanzig_6d(directory, meta, beta, r_cell, args.homo)
        for k in results:
            results[k].append(res[k])
        log.info("c=%.3e M  R_cell=%6.1f Å  D_R=%.4f  D_A=%.4f  D_ω=%.4f  sep=%.4f",
                 c, r_cell, res["D_R"], res["D_A"], res["D_omega"], res["separability"])

    for k in results:
        results[k] = np.array(results[k])

    # --- PMF for context ---
    R_all = meta["R"].values
    bw = meta["boltzmann_weight"].values if "boltzmann_weight" in meta.columns else np.ones(len(meta))
    pmf = -np.log(np.clip(bw, 1e-30, None))

    # --- Plot ---
    c_mM = molarities * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: PMF
    ax = axes[0]
    ax.plot(R_all, pmf, "k-", lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("w(R) / kT")
    ax.set_title("PMF")

    # Panel 2: D/D⁰ vs concentration
    ax = axes[1]
    ax.plot(c_mM, results["D_R"], "s-", label="D_R (trans)", ms=4, lw=1.5)
    ax.plot(c_mM, results["D_A"], "^-", label="D_A (rot A)", ms=4, lw=1.5)
    ax.plot(c_mM, results["D_B"], "v-", label="D_B (rot B)", ms=4, lw=1.5)
    ax.plot(c_mM, results["D_omega"], "D-", label="D_ω (dihedral)", ms=4, lw=1.5)
    ax.plot(c_mM, results["D_full"], "o-", label="D_full (6D)", ms=4, lw=1.5, color="k")
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("c (mM)")
    ax.set_ylabel("D / D⁰")
    ax.set_title("6D Zwanzig vs concentration")
    ax.legend(fontsize=7)

    # Panel 3: separability
    ax = axes[2]
    ax.plot(c_mM, results["separability"], "o-", color="C3", ms=4, lw=1.5)
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("c (mM)")
    ax.set_ylabel("D_full / (D_R × D_A × D_B × D_ω)")
    ax.set_title("Separability")

    fig.suptitle(f"Lysozyme — 6D Zwanzig cell model {'(homo)' if args.homo else ''}", fontsize=12)
    fig.tight_layout()
    fig.savefig("spectral6d_results.png", dpi=150, bbox_inches="tight")
    log.info("Saved spectral6d_results.png")
    plt.show()


if __name__ == "__main__":
    main()

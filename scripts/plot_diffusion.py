#!/usr/bin/env python3
"""Plot rotational diffusion analysis from duello CSV output.

Usage:
    python plot_diffusion.py diffusion.csv [--output prefix]

Generates three figures:
1. Isotropic D/D⁰ and λ₁/λ₁_free
2. Per-coordinate Zwanzig (D_A, D_B, D_ω)
3. Eigenvalue spectrum colored by coordinate character
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def plot_isotropic(df: pd.DataFrame, ax: plt.Axes):
    """D/D⁰ and λ₁/λ₁_free on the same axes."""
    r = df["R"]
    ax.plot(r, df["D/D⁰"], "o-", color="C0", label="$D_r/D_r^0$ (Zwanzig)", markersize=3)

    if "λ1" in df.columns and "λ1_free" in df.columns:
        mask = df["λ1"].notna() & df["λ1_free"].notna() & (df["λ1_free"] > 0)
        if mask.any():
            ratio = df.loc[mask, "λ1"] / df.loc[mask, "λ1_free"]
            ax.plot(
                df.loc[mask, "R"],
                ratio,
                "s--",
                color="C1",
                label="$\\lambda_1 / \\lambda_1^{\\mathrm{free}}$",
                markersize=3,
            )

    ax.set_xlabel("R (Å)")
    ax.set_ylabel("Normalized diffusion / relaxation")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend()
    ax.set_title("Isotropic rotational diffusion")


def plot_anisotropy_zwanzig(df: pd.DataFrame, ax: plt.Axes):
    """Per-coordinate Zwanzig decomposition with coupling index."""
    r = df["R"]
    ax.plot(r, df["D_A/D_A⁰"], "o-", color="C0", label="$D_A/D_A^0$ (mol A)", markersize=3)
    ax.plot(r, df["D_B/D_B⁰"], "s-", color="C1", label="$D_B/D_B^0$ (mol B)", markersize=3)
    ax.plot(r, df["D_ω/D_ω⁰"], "^-", color="C2", label="$D_\\omega/D_\\omega^0$ (dihedral)", markersize=3)
    ax.plot(r, df["D/D⁰"], "k--", alpha=0.4, label="$D/D^0$ (full 5D)", linewidth=1)

    if "separability" in df.columns:
        ax.plot(r, df["separability"], "x-", color="gray", alpha=0.5, markersize=3,
                linewidth=0.8, label="separability")

    ax.set_xlabel("R (Å)")
    ax.set_ylabel("$D/D^0$")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Anisotropic diffusion (Zwanzig per coordinate)")


def frac_to_color(fa, fb, fw):
    """Map (f_A, f_B, f_ω) fractions to an RGB color.

    f_A → blue, f_B → orange, f_ω → green. Mixed modes get blended colors."""
    c_a = np.array(mcolors.to_rgb("C0"))  # blue
    c_b = np.array(mcolors.to_rgb("C1"))  # orange
    c_w = np.array(mcolors.to_rgb("C2"))  # green
    return tuple(np.clip(fa * c_a + fb * c_b + fw * c_w, 0, 1))


def plot_eigenvalue_spectrum(df: pd.DataFrame, axes, hetero: bool = False):
    """One subplot per eigenmode (slowest on top), scatter colored by dominant coordinate."""
    from matplotlib.lines import Line2D

    n_modes = 0
    for i in range(1, 6):
        if f"λ{i}" in df.columns and f"λ{i}_free" in df.columns:
            n_modes = i

    for idx, i in enumerate(range(1, n_modes + 1)):
        ax = axes[idx]
        col_ev, col_free = f"λ{i}", f"λ{i}_free"
        col_fa, col_fb, col_fw = f"f_A{i}", f"f_B{i}", f"f_ω{i}"

        mask = df[col_ev].notna() & df[col_free].notna() & (df[col_free] > 0)
        if not mask.any():
            ax.set_visible(False)
            continue

        ratio = df.loc[mask, col_ev] / df.loc[mask, col_free]
        valid = ratio.between(0, 2.0)
        r = df.loc[mask, "R"][valid].values
        y = ratio[valid].values
        # Normalize so that the long-range limit is exactly 1
        if len(y) > 0 and y[-1] > 0:
            y = y / y[-1]
        fa = df.loc[mask, col_fa][valid].values if col_fa in df.columns else np.zeros_like(r)
        fb = df.loc[mask, col_fb][valid].values if col_fb in df.columns else np.zeros_like(r)
        fw = df.loc[mask, col_fw][valid].values if col_fw in df.columns else np.zeros_like(r)

        for j in range(len(r)):
            if hetero:
                dominant = np.argmax([fa[j], fb[j], fw[j]])
                color = ["C0", "C1", "C2"][dominant]
            else:
                color = "C2" if fw[j] > fa[j] + fb[j] else "C0"
            ax.plot(r[j], y[j], "o", color=color, markersize=3, alpha=0.8)
        if len(r) > 1:
            ax.plot(r, y, "-", color="gray", alpha=0.2, linewidth=0.5)

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_ylim(0, 1.5)
        ax.set_ylabel(f"$\\lambda_{i}/\\lambda_{i}^{{free}}$", fontsize=8)
        ax.tick_params(labelsize=7)

        if idx == 0:
            ax.text(0.02, 0.82, "slow", transform=ax.transAxes, fontsize=7, color="gray")
            if hetero:
                legend_el = [
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", label="mol A", markersize=5),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="C1", label="mol B", markersize=5),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="C2", label="dihedral", markersize=5),
                ]
            else:
                legend_el = [
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", label="molecular", markersize=5),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="C2", label="dihedral", markersize=5),
                ]
            ax.legend(handles=legend_el, fontsize=6, loc="upper right", ncol=3)
        if idx == n_modes - 1:
            ax.text(0.02, 0.82, "fast", transform=ax.transAxes, fontsize=7, color="gray")
            ax.set_xlabel("R (Å)")
        else:
            ax.set_xticklabels([])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Path to diffusion.csv from duello diffusion")
    parser.add_argument("--output", "-o", default="diffusion", help="Output filename prefix (default: diffusion)")
    parser.add_argument("--homo", action="store_true", help="Homo-dimer: merge mol A/B into single 'molecular' color (default: separate A/B/dihedral)")
    args = parser.parse_args()

    df = load_csv(args.csv)

    # Count eigenmode columns
    n_modes = 0
    for i in range(1, 6):
        if f"λ{i}" in df.columns and f"λ{i}_free" in df.columns:
            n_modes = i
    n_modes = max(n_modes, 1)

    height = max(6, n_modes * 1.5)
    fig = plt.figure(figsize=(14, height), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 0.6])

    # Left: isotropic + Zwanzig (stacked vertically)
    left_axes = subfigs[0].subplots(2, 1)
    plot_isotropic(df, left_axes[0])
    plot_anisotropy_zwanzig(df, left_axes[1])

    # Right: stacked eigenmode subplots
    eigen_axes = subfigs[1].subplots(n_modes, 1)
    if n_modes == 1:
        eigen_axes = [eigen_axes]
    subfigs[1].suptitle("Relaxation eigenmodes", fontsize=10)
    plot_eigenvalue_spectrum(df, eigen_axes, hetero=not args.homo)

    outpath = f"{args.output}.png"
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.show()


if __name__ == "__main__":
    main()

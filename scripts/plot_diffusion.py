#!/usr/bin/env python3
"""Plot rotational diffusion analysis from duello CSV output.

Usage:
    python plot_diffusion.py diffusion.csv [--output prefix]

Generates three figures:
1. Isotropic D/D⁰ and λ₁/λ₁_free vs R
2. Per-coordinate Zwanzig (D_A, D_B, D_ω) vs R
3. Eigenvalue spectrum λ_k/λ_k_free vs R
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip any whitespace from column names
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
    """Per-coordinate Zwanzig decomposition."""
    r = df["R"]
    ax.plot(r, df["D_A/D_A⁰"], "o-", color="C0", label="$D_A/D_A^0$ (mol A)", markersize=3)
    ax.plot(r, df["D_B/D_B⁰"], "s-", color="C1", label="$D_B/D_B^0$ (mol B)", markersize=3)
    ax.plot(r, df["D_ω/D_ω⁰"], "^-", color="C2", label="$D_\\omega/D_\\omega^0$ (dihedral)", markersize=3)
    ax.plot(r, df["D/D⁰"], "k--", alpha=0.4, label="$D/D^0$ (full 5D)", linewidth=1)

    ax.set_xlabel("R (Å)")
    ax.set_ylabel("$D/D^0$")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend()
    ax.set_title("Anisotropic diffusion (Zwanzig per coordinate)")


def plot_eigenvalue_spectrum(df: pd.DataFrame, ax: plt.Axes):
    """Eigenvalue ratios λ_k/λ_k_free vs R."""
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

    for i in range(1, 6):
        col = f"λ{i}"
        col_free = f"λ{i}_free"
        if col not in df.columns or col_free not in df.columns:
            break
        mask = df[col].notna() & df[col_free].notna() & (df[col_free] > 0)
        if not mask.any():
            continue
        ratio = df.loc[mask, col] / df.loc[mask, col_free]
        # Filter out unstable ratios from close-range Lanczos artifacts
        valid = ratio.between(0, 2.0)
        ax.plot(
            df.loc[mask, "R"][valid],
            ratio[valid],
            "o-",
            color=colors[i - 1],
            label=f"$\\lambda_{i}/\\lambda_{i}^{{\\mathrm{{free}}}}$",
            markersize=3,
        )

    ax.set_xlabel("R (Å)")
    ax.set_ylabel("$\\lambda_k / \\lambda_k^{\\mathrm{free}}$")
    ax.set_ylim(-0.05, 1.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.set_title("Eigenvalue spectrum (relaxation anisotropy)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Path to diffusion.csv from duello diffusion")
    parser.add_argument("--output", "-o", default="diffusion", help="Output filename prefix (default: diffusion)")
    args = parser.parse_args()

    df = load_csv(args.csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    plot_isotropic(df, axes[0])
    plot_anisotropy_zwanzig(df, axes[1])
    plot_eigenvalue_spectrum(df, axes[2])

    outpath = f"{args.output}.png"
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.show()


if __name__ == "__main__":
    main()

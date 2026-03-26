#!/usr/bin/env python3
"""Plot rotational diffusion analysis from duello CSV output.

Usage:
    python plot_diffusion.py diffusion.csv [--output prefix] [--homo]

Generates two side-by-side panels:
1. Zwanzig D/D⁰ — isotropic and per-coordinate decomposition
2. Spectral analysis — λ₁/λ₁_free and eigenmode coordinate fractions
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def count_modes(df: pd.DataFrame) -> int:
    n = 0
    for i in range(1, 6):
        if f"λ{i}" in df.columns and f"λ{i}_free" in df.columns:
            n = i
    return n


def plot_zwanzig(df: pd.DataFrame, ax: plt.Axes):
    """Isotropic and per-coordinate Zwanzig D/D⁰."""
    r = df["R"]
    ax.plot(r, df["D/D⁰"], "k-", linewidth=1.5, label="$D/D^0$ (full)", zorder=5)
    ax.plot(r, df["D_A/D_A⁰"], "o-", color="C0", label="$D_A/D_A^0$", markersize=3)
    ax.plot(r, df["D_B/D_B⁰"], "s-", color="C1", label="$D_B/D_B^0$", markersize=3)
    ax.plot(
        r,
        df["D_ω/D_ω⁰"],
        "^-",
        color="C2",
        label="$D_\\omega/D_\\omega^0$",
        markersize=3,
    )

    ax.set_xlabel("R (Å)")
    ax.set_ylabel("$D/D^0$")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Zwanzig diffusion coefficients")


def plot_spectral(df: pd.DataFrame, axes, homo: bool):
    """Top: λ_k/λ_k_free ratios. Bottom: coordinate fractions of slowest mode."""
    ax_ratio, ax_frac = axes
    n_modes = count_modes(df)

    # --- Top: eigenvalue ratios ---
    for i in range(1, n_modes + 1):
        col_ev, col_free = f"λ{i}", f"λ{i}_free"
        mask = df[col_ev].notna() & df[col_free].notna() & (df[col_free] > 0)
        if not mask.any():
            continue
        ratio = df.loc[mask, col_ev] / df.loc[mask, col_free]
        valid = ratio.between(0, 10.0)
        r = df.loc[mask, "R"][valid]
        y = ratio[valid]
        alpha = 1.0 if i == 1 else 0.4
        lw = 1.5 if i == 1 else 0.8
        ax_ratio.plot(
            r,
            y,
            "o-",
            markersize=3,
            alpha=alpha,
            linewidth=lw,
            label=f"$\\lambda_{i}/\\lambda_{i}^{{free}}$",
        )

    ax_ratio.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax_ratio.set_ylim(-0.05, 1.15)
    ax_ratio.set_ylabel("$\\lambda_k / \\lambda_k^{\\mathrm{free}}$")
    ax_ratio.set_xticklabels([])
    ax_ratio.legend(fontsize=9, frameon=False, ncol=3)
    ax_ratio.set_title("Spectral analysis")

    # --- Bottom: coordinate fractions of spectral-gap eigenvector ---
    cols = ["f_A1", "f_B1", "f_ω1"]
    if not all(c in df.columns for c in cols):
        ax_frac.set_visible(False)
        return

    mask = df["λ1"].notna()
    r = df.loc[mask, "R"]
    fa = df.loc[mask, "f_A1"]
    fb = df.loc[mask, "f_B1"]
    fw = df.loc[mask, "f_ω1"]

    if homo:
        ax_frac.plot(
            r, fa + fb, "o-", color="C0", markersize=3, label="molecular (A+B)"
        )
        ax_frac.plot(r, fw, "^-", color="C2", markersize=3, label="dihedral")
    else:
        ax_frac.plot(r, fa, "o-", color="C0", markersize=3, label="mol A")
        ax_frac.plot(r, fb, "s-", color="C1", markersize=3, label="mol B")
        ax_frac.plot(r, fw, "^-", color="C2", markersize=3, label="dihedral")

    ax_frac.set_ylim(-0.05, 1.05)
    ax_frac.set_xlabel("R (Å)")
    ax_frac.set_ylabel("Fraction of $\\lambda_1$ eigenvector")
    ax_frac.legend(fontsize=9, frameon=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", help="Path to diffusion.csv from duello diffusion")
    parser.add_argument(
        "--output",
        "-o",
        default="diffusion",
        help="Output filename prefix (default: diffusion)",
    )
    parser.add_argument(
        "--homo", action="store_true", help="Homo-dimer: merge mol A/B fractions"
    )
    args = parser.parse_args()

    df = load_csv(args.csv)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    plot_zwanzig(df, axes[0])
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    plot_spectral(df, axes[1:], homo=args.homo)

    outpath = f"{args.output}.png"
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.show()


if __name__ == "__main__":
    main()

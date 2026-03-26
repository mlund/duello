#!/usr/bin/env python3
"""Plot concentration-dependent rotational diffusion from cell model.

Usage:
    python plot_cell_diffusion.py cell_diffusion.csv [--output prefix] [--homo]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", help="Path to cell_diffusion.csv")
    parser.add_argument(
        "--output", "-o", default="cell_diffusion", help="Output filename prefix"
    )
    parser.add_argument(
        "--homo", action="store_true", help="Homo-dimer: merge mol A/B labels"
    )
    args = parser.parse_args()

    df = load_csv(args.csv)
    c = df["c/M"] * 1e3  # convert to mM for display

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True,
                             sharex=True, sharey=True)

    # Panel 1: Isotropic D/D⁰ vs concentration
    ax = axes[0]
    ax.plot(
        c,
        df["⟨D/D⁰⟩"],
        "o-",
        color="C0",
        label="$\\langle D_r/D_r^0 \\rangle$ (Zwanzig)",
        markersize=4,
    )
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("$\\langle D/D^0 \\rangle$")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Isotropic $\\langle D_r/D_r^0 \\rangle$")

    # Panel 2: Per-coordinate decomposition
    ax = axes[1]
    if args.homo:
        ax.plot(
            c,
            df["⟨D_A/D_A⁰⟩"],
            "o-",
            color="C0",
            label="$\\langle D_{A,B}/D^0 \\rangle$ (molecular)",
            markersize=4,
        )
    else:
        ax.plot(
            c,
            df["⟨D_A/D_A⁰⟩"],
            "o-",
            color="C0",
            label="$\\langle D_A/D_A^0 \\rangle$",
            markersize=4,
        )
        ax.plot(
            c,
            df["⟨D_B/D_B⁰⟩"],
            "s-",
            color="C1",
            label="$\\langle D_B/D_B^0 \\rangle$",
            markersize=4,
        )
    ax.plot(
        c,
        df["⟨D_ω/D_ω⁰⟩"],
        "^-",
        color="C2",
        label="$\\langle D_\\omega/D_\\omega^0 \\rangle$",
        markersize=4,
    )
    ax.plot(
        c,
        df["⟨D/D⁰⟩"],
        "k--",
        alpha=0.4,
        label="$\\langle D/D^0 \\rangle$ (full)",
        linewidth=1,
    )
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("$\\langle D/D^0 \\rangle$")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Anisotropic decomposition")

    outpath = f"{args.output}.png"
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.show()


if __name__ == "__main__":
    main()

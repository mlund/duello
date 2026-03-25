#!/usr/bin/env python3
"""Plot cell-model D_rot/D_rot⁰ vs concentration alongside experimental data.

Usage:
    python plot_cell_vs_experiment.py \
        --sim cell_100mM.csv cell_950mM.csv \
        --sim-labels "I=0.1 M" "I=0.95 M" \
        --exp takahashi2008_rotational_diffusion.csv \
        [--output diffusion_vs_experiment]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sim", nargs="+", required=True, help="Cell-model CSV files")
    parser.add_argument("--sim-labels", nargs="+", help="Labels for each sim file")
    parser.add_argument("--exp", required=True, help="Experimental CSV file")
    parser.add_argument("--output", "-o", default="diffusion_vs_experiment", help="Output prefix")
    args = parser.parse_args()

    sim_labels = args.sim_labels or [f"sim {i+1}" for i in range(len(args.sim))]

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    # Plot simulation cell-model curves
    sim_colors = ["C0", "C1", "C3", "C4"]
    for i, (path, label) in enumerate(zip(args.sim, sim_labels)):
        df = pd.read_csv(path)
        c_mM = df["c/M"] * 1e3
        ax.plot(c_mM, df["⟨D/D⁰⟩"], "-", color=sim_colors[i % len(sim_colors)],
                linewidth=1.5, label=f"Duello ({label})")

    # Plot experimental data
    exp = pd.read_csv(args.exp)
    exp_markers = ["o", "s", "D", "^"]
    ionic_strengths = sorted(exp["I_M"].unique())
    for j, I in enumerate(ionic_strengths):
        subset = exp[exp["I_M"] == I]
        ax.plot(subset["lysozyme_mM"], subset["D_rot/D_rot0"],
                exp_markers[j % len(exp_markers)],
                color=sim_colors[j % len(sim_colors)] if j < len(args.sim) else f"C{j+2}",
                markersize=6, markerfacecolor="none", linewidth=0,
                label=f"Takahashi (I={I} M)")

    ax.set_xlabel("Lysozyme concentration (mM)")
    ax.set_ylabel("$D_{rot} / D_{rot}^0$")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.set_title("Rotational diffusion: cell model vs experiment")

    outpath = f"{args.output}.png"
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.show()


if __name__ == "__main__":
    main()

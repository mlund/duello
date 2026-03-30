"""Plot 6D spectral gap results: D/D⁰ and coordinate fractions vs R_max."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from spectral6d import (
    build_6d_generator,
    build_state_space,
    coordinate_projection_6d,
    free_eigenvalues_6d,
    free_coordinate_fractions_6d,
    load_icosphere_neighbors,
    load_metadata,
    solve_eigenproblem,
)

ZERO_THRESHOLD = 1e-6


def main():
    import argparse, logging

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--rmin-plot", type=float, default=30)
    parser.add_argument("--rmax-plot", type=float, default=50)
    parser.add_argument("--dr-plot", type=float, default=2)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    log = logging.getLogger(__name__)

    AVOGADRO = 6.02214076e23
    kB = 1.380649e-23
    beta = 1.0 / (kB * args.temperature * AVOGADRO / 1000.0)

    directory = args.directory.resolve()
    meta = load_metadata(directory)
    n_v_dom = int(meta["n_v"].mode().iloc[0])
    n_omega = int(meta["n_omega"].iloc[0])
    neighbors = load_icosphere_neighbors(directory, n_v_dom)

    r_cell = 999.0
    rmax_values = np.arange(args.rmin_plot, args.rmax_plot + 0.1, args.dr_plot)

    rmax_out, d_d0, frac_R, frac_A, frac_B, frac_w = [], [], [], [], [], []

    for rmax in rmax_values:
        ss = build_state_space(meta, r_cell, directory, r_max=rmax, pmf_threshold=0.0)
        if ss is None or len(ss["slices"]) < 2:
            continue

        n_R = len(ss["slices"])
        log.info("R_max=%.0f: %d slices, %d states", rmax, n_R, ss["total_states"])

        L, pi = build_6d_generator(ss, beta)
        vals, vecs, method = solve_eigenproblem(L, args.k)

        non_null = np.abs(vals) > ZERO_THRESHOLD
        vals_nn, vecs_nn = vals[non_null], vecs[:, non_null]
        if len(vals_nn) == 0:
            continue

        lam1_free = free_eigenvalues_6d(n_R, neighbors, n_omega)
        ratio = abs(vals_nn[0]) / lam1_free if lam1_free > 0 else float("nan")
        fr, fa, fb, fw = coordinate_projection_6d(vecs_nn[:, 0], ss, neighbors)

        log.info("  D/D⁰=%.4f frac(R=%.2f A=%.2f B=%.2f ω=%.2f) [%s]",
                 ratio, fr, fa, fb, fw, method)

        rmax_out.append(rmax)
        d_d0.append(ratio)
        frac_R.append(fr)
        frac_A.append(fa)
        frac_B.append(fb)
        frac_w.append(fw)

    if not rmax_out:
        log.error("No results")
        sys.exit(1)

    rmax_out = np.array(rmax_out)

    # PMF
    bw = meta["boltzmann_weight"].values if "boltzmann_weight" in meta.columns else np.ones(len(meta))
    pmf = -np.log(np.clip(bw, 1e-30, None))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.plot(meta["R"].values, pmf, "k-", lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("w(R) / kT")
    ax.set_title("PMF")

    ax = axes[1]
    ax.plot(rmax_out, d_d0, "o-", color="C0", lw=1.5, ms=5)
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("R_max (Å)")
    ax.set_ylabel("D / D⁰")
    ax.set_title("6D spectral gap ratio")

    ax = axes[2]
    ax.plot(rmax_out, frac_R, "s-", label="R (trans)", lw=1.5, ms=5)
    ax.plot(rmax_out, frac_A, "^-", label="mol A", lw=1.5, ms=5)
    ax.plot(rmax_out, frac_B, "v-", label="mol B", lw=1.5, ms=5)
    ax.plot(rmax_out, frac_w, "D-", label="ω (dihedral)", lw=1.5, ms=5)
    ax.set_xlabel("R_max (Å)")
    ax.set_ylabel("Fraction")
    ax.set_title("Slowest mode decomposition")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    fig.suptitle("6D spectral analysis", fontsize=12)
    fig.tight_layout()
    fig.savefig("spectral6d_results.png", dpi=150, bbox_inches="tight")
    log.info("Saved spectral6d_results.png")
    plt.show()


if __name__ == "__main__":
    main()

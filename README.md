<p align="center">
  <img src="https://raw.githubusercontent.com/mlund/duello/main/assets/duello-logo.png" alt="Duello logo" height="300">
</p>
<p align="center">
    <a href="https://doi.org/10/p5d4">
        <img src="https://img.shields.io/badge/doi-10.26434%2Fchemrxiv--2025--0bfhd-lightgrey" alt="DOI:10.26434/chemrxiv-2025-0bfhd">
    </a>
    <a href="https://doi.org/10.5281/zenodo.15772003">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15772003.svg" alt="Zenodo">
    </a>
    <a href="https://colab.research.google.com/github/mlund/duello/blob/master/scripts/colab.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
    </a>
    <a href="https://github.com/mlund/duello/actions/workflows/rust.yml">
        <img src="https://github.com/mlund/duello/actions/workflows/rust.yml/badge.svg">
    </a>
</p>

-----

<p align = "center">
<b>Duello</b><br>
<i>Virial Coefficient and Dissociation Constant Estimation for Rigid Macromolecules</i>
</p>

-----

# Introduction

Duello is a tool to calculate the potential of mean force (PMF) between two rigid bodies, performing a
statistical mechanical average over inter-molecular orientations using subdivided icosahedrons.
For each mass center separation, _R_, the static contribution to the partition function,
$\mathcal{Z}(R) = \sum_{\mathbf{\Omega}} e^{-V(R,\mathbf{\Omega})/k_BT}$, is explicitly
evaluated to obtain the potential of mean force,
$w(R) = -k_BT \ln \mathcal{Z}(R)$
and the thermally averaged energy,

$$
U(R) = \frac{\sum V(R,\mathbf{\Omega}) e^{-V(R,\mathbf{\Omega})/k_BT}} {\mathcal{Z}(R)}
$$

where $V(R,\mathbf{\Omega})$ is the total inter-body interaction energy and $\mathbf{\Omega}$ represents a 5D angular space (_e.g._ two spherical coordinates for each body plus a dihedral angle around the connection line).

The osmotic second virial coefficient, which has dimensions of _volume_, reports on exactly two-body interactions:

$$
\begin{align}
B_2 & = -\frac{1}{16\pi^2} \int_{\mathbf{\Omega}} \int_0^{\infty}
\left (
  e^{-V(R,\mathbf{\Omega})/k_BT} - 1
\right )
R^2 dR d\mathbf{\Omega}\\
& =  -2\pi \int_0^{\infty} \left ( e^{-w(R)/k_BT} -1 \right )R^2 dR \\
& = B_2^{hs} -2\pi \int_{\sigma}^{\infty} \left ( e^{-w(R)/k_BT} -1 \right )R^2 dR\\
\end{align}
$$

where $B_2^{hs} = 2\pi\sigma^3/3$ is the hard-sphere contribution and $\sigma$ is a distance
of closest approach where $w(R\lt \sigma)=\infty$ is assumed.
For systems with net attractive interactions, the dissociation constant, $K_d$, can be estimated by,

$$
K_d^{-1} = 2 N_A\left (B_2^{hs} - B_2\right )
$$

<p align="center">
  <img height="200" alt="Image" src="https://github.com/user-attachments/assets/8d68eb85-6aa1-49e9-9d8d-75f91cdf1687" />
</p>


# Installation

Binary packages are available through PyPI.org:

```console
pip install duello
```

If you have a [Rust toolchain](https://www.rust-lang.org/learn/get-started) installed,
you may alternatively build and install from source:

```sh
cargo install --git https://github.com/mlund/duello
```

# Usage

The command-line tool `duello` does the 6D scanning and calculates
the angularly averaged potential of mean force, _w(R)_, which
is used to derive the 2nd virial coefficient and two-body dissociation constant, $K_d$.
The two input structures should be in `.xyz` format and all particle names must
be defined in the topology file under `atoms`.
The topology also defines the particular pair-potential to use, see below.
Note that a Coulomb/Yukawa potential is automatically added and should
hence _not_ be specified in the topology.
Coulomb is evaluated analytically (no cutoff) in all backends, while
short-range potentials (e.g. AshbaughHatch, WCA) are splined for GPU/SIMD backends.

```sh
duello scan \
    --mol1 cppm-p18.xyz \
    --mol2 cppm-p18.xyz \
    --rmin 37 --rmax 50 --dr 0.5 \
    --top topology.yaml \
    --max-ndiv 3 \
    --cutoff 50 \
    --molarity 0.05 \
    --temperature 298.15 \
    --backend auto \
    --grid "type=powerlaw2,size=500,shift=true"
```

### Angular resolution

The `--max-ndiv` option controls the icosphere subdivision level, which determines
the angular mesh density:

| `--max-ndiv` | Vertices | Approx. angular spacing |
|--------------|----------|-------------------------|
| 0            | 12       | 1.0 rad                 |
| 1            | 42       | 0.55 rad                |
| 2            | 92       | 0.37 rad                |
| 3 (default)  | 162      | 0.28 rad                |

The dihedral angle (&omega;) bin width is derived from the vertex spacing.
Higher subdivision levels give more accurate angular integration at the cost of
more energy evaluations per R-slice.

### Adaptive resolution

The table generator uses adaptive resolution: at each radial distance, the angular
gradient is evaluated and slabs are classified into tiers:

- **Repulsive** &mdash; all Boltzmann weights negligible (exp(&minus;&beta;U) &approx; 0).
  Detected automatically from the temperature; zero storage, lookup returns infinity.
- **Scalar** &mdash; nearly isotropic energy surface collapsed to a single value.
- **Nearest-vertex** &mdash; smooth surface; lookup uses nearest vertex without interpolation.
- **Interpolated** &mdash; full barycentric interpolation on the icosphere face.

The `--gradient-threshold` (default: 10.0 kJ/mol/rad) controls when resolution
is reduced. A summary of slab classifications is printed after generation.

## Spline Grid Options

The `--grid` option controls interpolation of short-range pair potentials (GPU/SIMD backends).
The `--cutoff` sets the spline range in angstroms; it should cover the SR potential
(e.g. AshbaughHatch cutoff) but does not affect the analytical Coulomb/Yukawa evaluation.

| Key          | Values               | Default     | Description                              |
|--------------|----------------------|-------------|------------------------------------------|
| `type`       | `powerlaw2`, `invr2` | `powerlaw2` | Grid spacing (invr2 avoids sqrt in eval) |
| `size`       | integer              | `500`       | Number of grid points                    |
| `shift`      | `true`, `false`      | `true`      | Shift energy to zero at cutoff           |
| `energy_cap` | float or `none`      | `none`      | Cap repulsive wall (kJ/mol) for f32 precision |

Example: `--grid "type=invr2,size=1000,shift=false,energy_cap=50"`

## Backend Performance

The program is written in Rust and attempts to use either GPU or all available CPU cores.
The following backends are available, with performance measured on the Calvados3 lysozyme example
(2.4M poses, 128 atoms per molecule, Apple M4):

| Backend     | Description                    | Poses/ms | Speedup |
|-------------|--------------------------------|----------|---------|
| `reference` | Exact potentials (validation)  |       48 |    1.0x |
| `simd`      | NEON (aarch64) / AVX2 (x86)    |      131 |    2.7x |
| `gpu`       | wgpu compute shaders           |     1065 |     22x |

The `auto` backend (default) selects GPU if available, otherwise SIMD.
GPU and SIMD backends use cubic Hermite splines for short-range potentials
and evaluate Coulomb/Yukawa analytically without cutoff.

## Atom Scan

For simulations involving a rigid body and mobile atoms (e.g. ions), the full 6D scan is unnecessary.
The `atom-scan` subcommand computes a 3D table (R, theta, phi) of interaction energies between a
rigid body and a single test atom type, using Yukawa electrostatics and short-range pair potentials
from the topology:

```sh
duello atom-scan \
    --mol1 4lzt.xyz \
    --atom Na \
    --rmin 2.0 --rmax 60 --dr 0.5 \
    --top topology.yaml \
    --resolution 0.5 \
    --cutoff 150 \
    --molarity 0.02 \
    --output atom_table.bin.gz
```

The output is a flat binary table (`Table3DFlat`) that can be loaded for fast lookup
with barycentric interpolation on the icosphere mesh.

## Rotational Diffusion Analysis

The `diffusion` subcommand estimates the normalized rotational diffusion coefficient
D_r/D_r^0 from a saved 6D energy table, without re-scanning.
This quantifies how much the inter-molecular energy landscape hinders rotational
diffusion compared to free rotation at each mass center separation R.

```sh
duello diffusion \
    --table table.bin.gz \
    --temperature 300 \
    --output diffusion.csv
```

The temperature can differ from the scan temperature since energies are stored in absolute
units (kJ/mol). If `--temperature` is omitted, the table's generation temperature is used.

### Output columns

| Column | Description |
|--------|-------------|
| `R` | Mass center separation (Å) |
| `D/D⁰` | Normalized rotational diffusion (Zwanzig, full 5D). 1 = free rotation, 0 = locked. |
| `D_A/D_A⁰` | Per-molecule-A Zwanzig (marginalized over B and dihedral) |
| `D_B/D_B⁰` | Per-molecule-B Zwanzig (marginalized over A and dihedral) |
| `D_ω/D_ω⁰` | Per-dihedral Zwanzig (marginalized over both molecules) |
| `separability` | D/D⁰ ÷ (D_A × D_B × D_ω). 1 = coordinates are independent, 0 = strongly coupled |
| `λk` | Eigenvalue magnitude of mode k |
| `f_Ak, f_Bk, f_ωk` | Coordinate fractions: how much of mode k is mol A / mol B / dihedral rotation |
| `λk_free, ...` | Corresponding free-diffusion eigenmodes for normalization |
| `n_active` | Number of finite-energy grid points at this R |

### Interpretation

**D/D⁰** is computed via the Zwanzig formula for diffusion in a rough potential:

$$
D_r / D_r^0 = \frac{1}{\langle e^{\beta U} \rangle \cdot \langle e^{-\beta U} \rangle}
$$

where the averages are over the 5D angular grid (molecule A orientation, molecule B
orientation, dihedral angle).

**D_A/D_A⁰, D_B/D_B⁰, D_ω/D_ω⁰** decompose the diffusion into per-coordinate contributions.
Each is computed by marginalizing the energy landscape over the other two coordinates
to get a 1D potential of mean force, then applying Zwanzig to it.
For symmetric molecules (mol1 = mol2), D_A ≈ D_B at all separations.

**λk with (f_A, f_B, f_ω)** are eigenvalue magnitudes of the symmetrized generator
with coordinate decomposition. The fractions f_A + f_B + f_ω = 1 show whether
mode k is primarily molecule A tumbling (high f_A), molecule B tumbling (high f_B),
or dihedral rotation (high f_ω). Mixed values indicate coupled modes.
For homo-dimers (identical molecules), the table is exchange-symmetrized
automatically so that D_A = D_B and f_A ≈ f_B.

## Preparing PDB files

The following uses `pdb2xyz` to create a coarse grained XYZ file and Calvados topology for Duello:

```sh
pip install pdb2xyz
pdb2xyz -i 4lzt.pdb -o 4lzt.xyz --pH 7.0 --sidechains
duello scan \
  -1 4lzt.xyz -2 4lzt.xyz \
  --rmin 24 --rmax 80 --dr 0.5 \
  --max-ndiv 2 \
  --top topology.yaml \
  --molarity 0.05
```

If `pdb2xyz` gives errors, you may be able to correct your PDB file with
[pdbfixer](https://github.com/openmm/pdbfixer?tab=readme-ov-file).

## Examples

Ready-to-run script examples are provided in the `scripts/` directory:

Command                | Description
---------------------- | ------------------------------------------------------------
`scripts/cppm.sh`      | Spherical, multipolar particles using the CPPM model
`scripts/calvados3.sh` | Two coarse grained lysozyme molecules w. Calvados3 interactions


## Interaction models

Each macromolecule is represented by a rigid constellation of beads with
properties defined under `atoms` in the topology file.
The inter-molecular energy, $V(R,\Omega)$ is calculated by summing all pairwise interactions
between beads using a customizable pair potential, $u_{ij}$.
If needed, different pair-potentials can be explicitly defined for
specific atom pairs.

The provided examples illustrate the following schemes:

- Screened `Coulomb` + `AshbaughHatch`, for the Calvados model.
- Screened `Coulomb` + `WeeksChandlerAndersen` for the CPPM model.

Many more pair-potentials are available through the
[`interatomic`](https://docs.rs/interatomic/latest/interatomic/twobody/index.html) library,
_e.g._ `LennardJones`, `HardSphere` etc.

__Warning:__ The electrostatic term (Coulomb/Yukawa) is
always automatically added and should therefore _not_ be specified in the topology.

# Development

This is for development purposes only and details how to create and publish a
binary package on pypi.org.

## Create `pip` package using Maturin via a Docker image:

Run this on MacOS, linux (x86 and arm) to get all architectures:

```sh
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin:v1.11.5 publish -u __token__ -p PYPI_TOKEN
```


For local Maturin installs, follow the steps below.

```sh
pip install ziglang pipx
pipx install maturin # on ubuntu; then restart shell
maturin publish -u __token__ -p PYPI_TOKEN --target=x86_64-unknown-linux-gnu --zig
```

MacOS targets can be generated without `--zig` using the targets
`x86_64-apple-darwin` and `aarch64-apple-darwin`.

```sh
rustup target list
rustup target add x86_64-apple-darwin
```


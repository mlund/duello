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

# Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Overview](#overview)
- [Commands](#commands)
  - [`duello scan` -- Two-Body Scan](#duello-scan----two-body-scan)
  - [`duello atom-scan` -- Atom Scan](#duello-atom-scan----atom-scan)
  - [`duello diffusion` -- Rotational Diffusion](#duello-diffusion----rotational-diffusion)
- [Input Formats](#input-formats)
  - [PDB / mmCIF (recommended)](#pdb--mmcif-recommended)
  - [Pre-generated XYZ](#pre-generated-xyz)
  - [Topology File](#topology-file)
- [Tuning and Options](#tuning-and-options)
  - [Angular Resolution](#angular-resolution)
  - [Adaptive Resolution](#adaptive-resolution)
  - [Cutoff Settings](#cutoff-settings)
  - [Spline Grid](#spline-grid)
  - [Compute Backends](#compute-backends)
- [Interaction Models](#interaction-models)
- [Theory](#theory)
- [Examples](#examples)
- [Development](#development)

# Quick Start

Install via pip:

```sh
pip install duello
```

Run a two-body scan between two lysozyme molecules, starting directly from a PDB file:

```sh
duello scan \
    --mol1 4lzt.pdb \
    --mol2 4lzt.pdb \
    --rmin 24 --rmax 80 --dr 0.5 \
    --molarity 0.05
```

This computes the potential of mean force _w(R)_ and reports the second virial coefficient _B2_ and dissociation constant _Kd_.
The output is written to `pmf.dat` by default.

Duello automatically coarse-grains PDB/CIF structures using the Calvados 3 force field at pH 7.0.
To override these defaults:

```sh
duello scan \
    --mol1 4lzt.pdb --mol2 4lzt.pdb \
    --rmin 24 --rmax 80 --dr 0.5 \
    --molarity 0.05 \
    --ph 4.5 --model calvados3 --cg multi
```

# Installation

Binary packages are available through PyPI:

```sh
pip install duello
```

Alternatively, build from source with a [Rust toolchain](https://www.rust-lang.org/learn/get-started):

```sh
cargo install --git https://github.com/mlund/duello
```

# Overview

Duello calculates the potential of mean force (PMF) between two rigid bodies by explicitly evaluating the partition function over inter-molecular orientations on a subdivided icosphere mesh.

For each mass center separation _R_, the angular partition function is:

$$
\mathcal{Z}(R) = \sum_{\mathbf{\Omega}} e^{-V(R,\mathbf{\Omega})/k_BT}
$$

yielding the potential of mean force:

$$
w(R) = -k_BT \ln \mathcal{Z}(R)
$$

and the thermally averaged energy:

$$
U(R) = \frac{\sum V(R,\mathbf{\Omega}) e^{-V(R,\mathbf{\Omega})/k_BT}} {\mathcal{Z}(R)}
$$

where $V(R,\mathbf{\Omega})$ is the total inter-body interaction energy and $\mathbf{\Omega}$ represents the 5D angular space (two spherical coordinates per body plus a dihedral angle around the inter-molecular axis).

<p align="center">
  <img height="200" alt="Image" src="https://github.com/user-attachments/assets/8d68eb85-6aa1-49e9-9d8d-75f91cdf1687" />
</p>

From _w(R)_, duello derives the osmotic second virial coefficient _B2_ and, for net-attractive systems, the dissociation constant _Kd_. See [Theory](#theory) for the full expressions.

# Commands

Duello provides three subcommands:

| Command           | Purpose                                                     |
|-------------------|-------------------------------------------------------------|
| `duello scan`     | 6D scan between two rigid bodies; computes _w(R)_, _B2_, _Kd_ |
| `duello atom-scan`| 3D scan between a rigid body and a single atom type         |
| `duello diffusion`| Rotational diffusion analysis from a saved 6D energy table  |

## `duello scan` -- Two-Body Scan

The primary command. Scans all inter-molecular orientations at each mass center separation to compute _w(R)_, _B2_, and _Kd_.

**From PDB/CIF (simplest):**

```sh
duello scan \
    --mol1 4lzt.pdb --mol2 4lzt.pdb \
    --rmin 24 --rmax 80 --dr 0.5 \
    --molarity 0.05
```

**From pre-generated XYZ (full control):**

```sh
duello scan \
    --mol1 cppm-p18.xyz --mol2 cppm-p18.xyz \
    --rmin 37 --rmax 50 --dr 0.5 \
    --top topology.yaml \
    --max-ndiv 3 \
    --cutoff 30 \
    --molarity 0.05 \
    --temperature 298.15 \
    --backend auto \
    --grid "type=powerlaw2,size=200,shift=true"
```

### Key Options

| Flag                     | Default       | Description                                           |
|--------------------------|---------------|-------------------------------------------------------|
| `-1, --mol1`             | (required)    | First structure (PDB, CIF, or XYZ)                    |
| `-2, --mol2`             | (required)    | Second structure (PDB, CIF, or XYZ)                   |
| `--rmin`, `--rmax`, `--dr` | (required) | Radial scan range and step size (angstrom)            |
| `-a, --top`              | --            | Topology YAML (required for XYZ, auto-generated for PDB/CIF) |
| `-M, --molarity`         | 0.1           | 1:1 salt concentration (mol/L)                        |
| `-T, --temperature`      | 298.15        | Temperature (K)                                       |
| `--cutoff`               | 30.0          | Short-range spline cutoff (angstrom)                  |
| `--sr-cutoff`            | = `--cutoff`  | Fine-tuned SR potential range (angstrom)              |
| `--max-ndiv`             | 3             | Icosphere subdivision level (see [Angular Resolution](#angular-resolution)) |
| `--gradient-threshold`   | 0.5           | Adaptive resolution threshold (1/rad)                 |
| `--backend`              | auto          | Compute backend: `auto`, `gpu`, `simd`, `reference`   |
| `--grid`                 | (see below)   | Spline grid settings (see [Spline Grid](#spline-grid)) |
| `--pmf`                  | pmf.dat       | Output PMF file                                       |
| `--savetable`            | --            | Save 6D binary table (`.bin.gz` for compression)      |
| `--fixed-dielectric`     | --            | Override dielectric constant                          |

**PDB/CIF-only options:**

| Flag                     | Default       | Description                                           |
|--------------------------|---------------|-------------------------------------------------------|
| `--ph`                   | 7.0           | pH for coarse-graining                                |
| `--model`                | calvados3     | Force field model                                     |
| `--cg`                   | single        | Coarse-graining policy: `single` or `multi`           |
| `--scale-hydrophobic`    | --            | Hydrophobic scaling, e.g. `lambda:1.2` or `epsilon:0.8` |
| `--chain`                | (all)         | Keep only these chain IDs (repeatable: `--chain A --chain B`) |

## `duello atom-scan` -- Atom Scan

Computes a 3D energy table (_R_, _theta_, _phi_) between a rigid body and a single test atom type. Useful for rigid body + mobile ion simulations where the full 6D scan is unnecessary.

```sh
duello atom-scan \
    --mol1 4lzt.xyz \
    --atom Na \
    --rmin 2.0 --rmax 60 --dr 0.5 \
    --top topology.yaml \
    --molarity 0.02 \
    --cutoff 150
```

The output is a flat binary table (`Table3DFlat`) with barycentric interpolation on the icosphere mesh.

| Flag            | Default            | Description                              |
|-----------------|--------------------|------------------------------------------|
| `--mol1`        | (required)         | Rigid body structure                     |
| `--atom`        | (required)         | Atom type name (from topology)           |
| `-o, --output`  | atom_table.bin.gz  | Output binary table                      |
| `--pmf`         | pmf.dat            | Output PMF file                          |

All other radial, topology, backend, and PDB/CIF flags are shared with `duello scan`.

## `duello diffusion` -- Rotational Diffusion

> **Note:** This subcommand is experimental. Output format and methods may change in future releases.

Estimates the normalized rotational diffusion coefficient _D_r / D_r^0_ from a previously saved 6D energy table, without re-scanning. Two methods are used:

1. **Zwanzig formula** -- transport coefficient from energy landscape roughness
2. **Spectral gap** -- slowest relaxation rate from sparse Lanczos eigendecomposition

```sh
duello diffusion table.bin.gz \
    -T 300 \
    -o diffusion.csv \
    -j 4
```

The temperature can differ from the scan temperature since energies are stored in absolute units (kJ/mol).

| Flag            | Default          | Description                              |
|-----------------|------------------|------------------------------------------|
| `<TABLE>`       | (required)       | Path to 6D binary table (.bin or .bin.gz)|
| `-T`            | 298.15           | Temperature (K)                          |
| `-o`            | diffusion.csv    | Output CSV                               |
| `-j`            | 0 (all cores)    | Thread count (fewer = less memory)       |
| `--cmin/--cmax` | --               | Concentration scan range (mol/L)         |
| `--cn`          | 20               | Number of concentration points           |
| `--cell-output` | cell_diffusion.csv | Output CSV for cell-model D(c)         |
| `--export-matrices` | --           | Export generator matrices (Matrix Market)|

### Output Columns

| Column          | Description |
|-----------------|-------------|
| `R`             | Mass center separation (angstrom) |
| `D/D^0`         | Normalized rotational diffusion (Zwanzig, full 5D). 1 = free, 0 = locked |
| `D_A/D_A^0`     | Per-molecule-A Zwanzig (marginalized over B and dihedral) |
| `D_B/D_B^0`     | Per-molecule-B Zwanzig (marginalized over A and dihedral) |
| `D_w/D_w^0`     | Per-dihedral Zwanzig (marginalized over both molecules) |
| `separability`  | D/D^0 / (D_A x D_B x D_w). 1 = independent, 0 = coupled |
| `lambda_k`      | Eigenvalue magnitude of mode _k_ |
| `f_Ak, f_Bk, f_wk` | Coordinate fractions for mode _k_ (mol A / mol B / dihedral) |
| `n_active`      | Number of finite-energy grid points at this _R_ |

**Interpretation:** D/D^0 near 1 means nearly free rotation; values near 0 indicate a rugged energy landscape that locks orientations. The separability column reveals whether molecular rotations are independent or coupled. For homo-dimers, D_A ~ D_B by symmetry.

# Input Formats

## PDB / mmCIF (recommended)

Pass PDB or CIF files directly. Duello coarse-grains them on the fly using [cgkitten](https://github.com/mlund/cgkitten), generating `.xyz` and `_topology.yaml` files alongside the input. Charges are computed using Monte Carlo titration.

If your PDB has missing atoms or residues, pre-process with [pdbfixer](https://github.com/openmm/pdbfixer):

```sh
pip install pdbfixer
pdbfixer 4lzt.pdb --add-atoms=all --add-residues
```

## Pre-generated XYZ

For XYZ input, a topology file (`--top`) must be provided. You can generate XYZ and topology files from PDB using [cgkitten](https://github.com/mlund/cgkitten):

```sh
cargo install cgkitten
cgkitten 4lzt.pdb convert --ph 7.0
```

## Topology File

The topology is a YAML file defining atom properties and the pair potential. Example:

```yaml
atoms:
  - {name: ALA, charge: 0.0, mass: 71.07, σ: 5.04, ε: 0.8368, hydrophobicity: !Lambda 0.34}
  - {name: LYS, charge: 1.0, mass: 128.17, σ: 6.36, ε: 0.8368, hydrophobicity: !Lambda 0.14}
  # ... one entry per bead type

system:
  energy:
    nonbonded:
      # Coulomb/Yukawa is always added automatically -- do NOT specify it here
      default:
        - !AshbaughHatch {mixing: arithmetic, cutoff: 20.0}
```

Each atom entry defines charge, mass, Lennard-Jones-like size (sigma) and well depth (epsilon), and model-specific properties.
The `nonbonded` section specifies the short-range pair potential; different potentials can be assigned to specific atom pairs if needed.

**Warning:** The electrostatic Coulomb/Yukawa term is always added automatically. Do _not_ include it in the topology.

# Tuning and Options

## Angular Resolution

The `--max-ndiv` flag sets the icosphere subdivision level, controlling angular mesh density:

| `--max-ndiv` | Vertices | Approx. angular spacing |
|--------------|----------|-------------------------|
| 0            | 12       | 1.0 rad                 |
| 1            | 42       | 0.55 rad                |
| 2            | 92       | 0.37 rad                |
| 3 (default)  | 162      | 0.28 rad                |

Higher levels give more accurate angular integration at the cost of more energy evaluations per radial slice.

## Adaptive Resolution

Duello automatically classifies each radial slab into tiers based on the angular energy gradient:

| Tier            | Description                                              |
|-----------------|----------------------------------------------------------|
| **Repulsive**   | All Boltzmann weights negligible; zero storage, returns infinity |
| **Scalar**      | Nearly isotropic surface; collapsed to a single value    |
| **Nearest-vertex** | Smooth surface; nearest-vertex lookup, no interpolation |
| **Interpolated** | Full barycentric interpolation on icosphere faces       |

The `--gradient-threshold` flag (default: 0.5 1/rad) controls when resolution is reduced. A summary is printed after table generation.

## Cutoff Settings

The `--cutoff` flag (default 30 angstrom) sets the spline range for short-range pair potentials in GPU/SIMD backends. Beyond this distance the spline returns zero.
It does **not** affect the analytical Coulomb/Yukawa evaluation, which has no cutoff.

Important rules:
- `--cutoff` must be >= the largest cutoff defined in the topology (otherwise the spline silently truncates the potential).
- Setting `--cutoff` much larger than needed wastes spline points but is otherwise harmless.
- Use `--sr-cutoff` to match the actual SR potential range for best accuracy (e.g. `--sr-cutoff 20` when AshbaughHatch has cutoff 20).
- The `reference` backend ignores `--cutoff` and evaluates potentials with their native cutoffs.

## Spline Grid

The `--grid` flag controls spline interpolation of short-range potentials (GPU/SIMD backends):

| Key          | Values               | Default     | Description                                   |
|--------------|----------------------|-------------|-----------------------------------------------|
| `type`       | `powerlaw2`, `invr2` | `powerlaw2` | Grid spacing (`invr2` avoids sqrt in lookup)  |
| `size`       | integer              | `200`       | Number of grid points                         |
| `shift`      | `true`, `false`      | `true`      | Shift energy to zero at cutoff                |
| `energy_cap` | float or `none`      | `none`      | Cap repulsive wall (kJ/mol) for f32 precision |

Example: `--grid "type=invr2,size=1000,shift=false,energy_cap=50"`

## Compute Backends

Duello automatically selects the best available backend. Performance on the Calvados3 lysozyme benchmark (2.4M poses, 128 atoms/molecule, Apple M4):

| Backend     | Description                    | Poses/ms | Speedup |
|-------------|--------------------------------|----------|---------|
| `reference` | Exact potentials (validation)  |       48 |    1.0x |
| `simd`      | NEON (aarch64) / AVX2 (x86)   |      131 |    2.7x |
| `gpu`       | wgpu compute shaders           |     1065 |     22x |

`auto` (default) selects GPU if available, otherwise SIMD.
GPU and SIMD backends use cubic Hermite splines for short-range potentials and evaluate Coulomb/Yukawa analytically.

# Interaction Models

Each macromolecule is represented by a rigid constellation of beads with properties defined in the topology.
The inter-molecular energy $V(R,\Omega)$ is the sum of all pairwise bead-bead interactions.

Provided examples use:
- **Calvados 3:** Screened Coulomb + AshbaughHatch
- **CPPM:** Screened Coulomb + WeeksChandlerAndersen

Many additional pair potentials are available through the [`interatomic`](https://docs.rs/interatomic/latest/interatomic/twobody/index.html) library (LennardJones, HardSphere, etc.).
Different pair potentials can be assigned to specific atom pairs in the topology.

# Theory

The osmotic second virial coefficient reports on two-body interactions:

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

where $B_2^{hs} = 2\pi\sigma^3/3$ is the hard-sphere contribution and $\sigma$ is the distance of closest approach where $w(R\lt \sigma)=\infty$.

For systems with net attractive interactions, the dissociation constant is estimated by:

$$
K_d^{-1} = 2 N_A\left (B_2^{hs} - B_2\right )
$$

The rotational diffusion coefficient is estimated via the Zwanzig formula:

$$
D_r / D_r^0 = \frac{1}{\langle e^{\beta U} \rangle \cdot \langle e^{-\beta U} \rangle}
$$

where the averages are over the 5D angular grid.

# Examples

Ready-to-run examples are provided in `scripts/` and `examples/`:

| Path                     | Description                                            |
|--------------------------|--------------------------------------------------------|
| `scripts/cppm.sh`        | Spherical, multipolar particles using the CPPM model   |
| `scripts/calvados3.sh`   | Two coarse-grained lysozyme molecules with Calvados 3  |

A Google Colab notebook is also available: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlund/duello/blob/master/scripts/colab.ipynb)

# Development

### Publishing to PyPI via Docker

Run on macOS, Linux (x86 and arm) to cover all architectures:

```sh
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin:v1.11.5 publish -u __token__ -p PYPI_TOKEN
```

### Local Maturin build

```sh
pip install ziglang pipx
pipx install maturin
maturin publish -u __token__ -p PYPI_TOKEN --target=x86_64-unknown-linux-gnu --zig
```

macOS targets can be built without `--zig`:

```sh
rustup target add x86_64-apple-darwin aarch64-apple-darwin
maturin publish -u __token__ -p PYPI_TOKEN --target=aarch64-apple-darwin
```

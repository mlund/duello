<p align="center">
  <img src="assets/duello-logo.png" alt="crates.io", height="300">
</p>
<p align="center">
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
<b>Duello</b></br>
<i>Virial Coefficient and Dissociation Constant Estimation for Rigid Macromolecules</i>
</p>

-----

# Introduction

Duello is a tool to calculate the potential of mean force (PMF) between two ridig bodies, performing a
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
  <img src="assets/illustration.png" alt="crates.io", height="200">
</p>

# Installation

## Using pip (linux x86)

```console
pip install duello
```

## Using Cargo (all platforms)

This requires prior installation of the [Rust](https://www.rust-lang.org/learn/get-started) toolchain.

```sh
cargo install duello
```

Alternatively you may compile and run directly from the source code:

```sh
git clone https://github.com/mlund/duello
cd duello/
cargo run --release -- <args...>
```

# Usage

The command-line tool `duello` does the 6D scanning and calculates
the angularly averaged potential of mean force, _A(R)_ which
is used to derive the 2nd virial coefficient and twobody dissociation constant, $K_d$.
The two input structures should be in `.xyz` format and all particle names must
be defined in the topology file under `atoms`.
The topology also defines the particular pair-potential to use.
Note that currently, a coulomb potential is automatically added and should
hence _not_ be specified in the topology.
The program is written in Rust and attempts to use all available CPU cores.

```sh
duello scan \
    --icotable \
    --mol1 cppm-p18.xyz \
    --mol2 cppm-p18.xyz \
    --rmin 37 --rmax 50 --dr 0.5 \
    --top topology.yaml \
    --resolution 0.8 \
    --cutoff 1000 \
    --molarity 0.05 \
    --temperature 298.15
```

## Examples

Ready run scripts examples are provided in the `scripts/` directory:

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
[`interatomic`](https://crates.io/crates/interatomic) library,
_e.g._ `LennardJones`, `HardSphere` etc.

__Warning:__ The electrostatic term, `Coulomb` is
always automatically added and should therefore _not_ be specified in the topology.

# Development

This is for development purposes only.

## Create `pip` package using Maturin

```sh
pip install ziglang pipx
pipx install maturin # on ubuntu; then restart shell
maturin publish --target=x86_64-unknown-linux-gnu --zig
```

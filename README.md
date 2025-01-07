<p align="center">
  <img src="assets/duello-logo.png" alt="crates.io", height="300">
</p>
<p align="center">
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
    </a>
</p>

-----

<p align = "center">
<b>Duello</b></br>
<i>Virial Coefficient and Dissociation Constant Estimation for Rigid Macromolecules</i>
</p>

-----

# Introduction

This iterates over all intermolecular poses between two rigid molecules using a regular grid in angular space using subdivided icosahedrons.
For each mass center separation, _R_, the partition function,
$Q(r) = \sum e^{-V(R)/k_BT}$, is explicitly
evaluated to obtain the free energy, $A(R) = -k_BT \ln \langle e^{-V(R)/k_BT} \rangle$ and
the thermally averaged energy,
$$
U(R) = \frac{\sum V(R) e^{-V(R)/k_BT}} {Q}
$$
where $V(R)$ is the inter-body interaction energy averaged over angular space.

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

Command               | Description
--------------------- | ------------------------------------------------------------
`scripts/cppm.sh`     | Spherical, multipolar particles using the CPPM model
`scripts/lysozyme.sh` | Two coarse grained lysozyme molecules w. Calvados3 interactions

# Development

This is for development purposes only.

## Create `pip` package using Maturin

```sh
pip install ziglang pipx
pipx install maturin # on ubuntu; then restart shell
maturin publish --target=x86_64-unknown-linux-gnu --zig
```

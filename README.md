<p align="center">
  <img src="assets/duello-logo.png" alt="crates.io", height="200">
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

This iterates over all intermolecular poses between two rigid molecules.
For each pose, defined by two quaternions and a mass center separation, the
intermolecular interaction energy is calculated.

For each mass center separation, _r_, the partition function,
$Q(r) = \sum e^{-\beta u(r)}$, is explicitly
evaluated, whereby we can obtain the free energy, $w(r) = -kT \ln \langle e^{-\beta u(r)} \rangle$ and
the thermally averaged energy, $u(r) = \sum u(r)e^{-\beta u(r)} / Q$.

![Angular Scan](assets/illustration.png)

# Installation

```console
pip install duello
```

# Usage

The command-line tool `duello` does the 6D scanning and calculates
the potential of mean force, _w(r)_ which
is used to derive the 2nd virial coefficient and twobody dissociation constant.
The two input structures should be in `.xyz` format and all particle names must
be defined in the topology file.
The topology also defines the particular pair-potential to use.
Note that currently, a coulomb potential is automatically added and should
hence _not_ be specified in the topology.

```console
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

The program is written in Rust and attempts to use all available CPU cores.

## Examples

Ready run scripts examples are provided in the `scripts/` directory:

Command               | Description
--------------------- | ------------------------------------------------------------
`scripts/cppm.sh`     | Spherical, multipolar particles using the CPPM model
`scripts/lysozyme.sh` | Two coarse grained lysozyme molecules w. Calvados3 interactions

# Development

This is for development purposes only.

## Create `pip` package using Maturin

```console
pip install ziglang pipx
pipx install maturin # on ubuntu; then restart shell
maturin publish --target=x86_64-unknown-linux-gnu --zig
```

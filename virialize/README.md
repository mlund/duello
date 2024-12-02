# Angular Scan

This iterates over all intermolecular poses between two rigid molecules.
For each pose, defined by two quaternions and a mass center separation, the
intermolecular interaction energy is calculated.

For each mass center separation, _r_, the partition function,
$Q(r) = \sum e^{-\beta u(r)}$, is explicitly
evaluated, whereby we can obtain the free energy, $w(r) = -kT \ln \langle e^{-\beta u(r)} \rangle$ and
the thermally averaged energy, $u(r) = \sum u(r)e^{-\beta u(r)} / Q$.

![Angular Scan](assets/illustration.png)

## Usage

The command-line tool `virialize` does the 6D scanning and calculates the potential of mean force, w(r) which
is used to derive the 2nd virial coefficient and twobody dissociation constant.
Two input structures are requires (`.xyz` format) and all particle types must be defined in the topology file.
The topology files also defines the particular pair-potential to use. Note that currently, a coulomb potential
is automatically added and should hence not be specified in the topology.

```console
virialize scan --icotable -1 cppm-p18.xyz -2 cppm-p00.xyz --rmin 40.5 --rmax 60 --dr 1.0 --top topology.yaml --resolution 0.6 --molarity 0.05
```


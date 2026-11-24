# Benchmarks

Validation and timing of the compute backends. All timings are on an Apple M4.

## Grid backend against exact pairs

`--backend gpu` evaluates every bead pair with the same radial splines, so it is the reference for `--backend grid`.

### Lysozyme

The benchmark command, from `examples/calvados3/`:

~~~
duello scan -1 2CDS.xyz -2 2CDS.xyz --rmin 26 --rmax 60 --dr 0.5 --top topology.yaml --molarity 0.02
~~~

Measured before the topology set each pair's spline range (then `--cutoff 150`);
not comparable with the command above until re-run.

| Backend                         | B₂ (Å³) | ΔB₂     | Scan time |
|---------------------------------|---------|---------|-----------|
| `gpu`                           | 367 543 |   —     | 68 s      |
| `grid --grid-spacing 0.5`       | 367 543 | +0.00 % |  9 s      |
| `grid` (default, 1.0 Å)         | 367 543 | +0.00 % |  7 s      |

### Chymotrypsinogen A (7KTZ)

A pair whose B₂ is set by contact attraction, for both force fields and bead policies.
Coarse-graining titrates charges by Monte Carlo, so each run from the PDB differs slightly; every row therefore scans the same generated `.xyz` and topology:

~~~
duello scan -1 7KTZ.xyz -2 7KTZ.xyz --top 7KTZ_topology.yaml --rmin 32 --rmax 60 --dr 1 \
    --molarity 0.015 --temperature 300 --gradient-threshold 0
~~~

Kim–Hummer rows predate cgkitten 0.6, whose model is cut and shifted at 30 Å.

| `--model`, `--cg`     | `gpu` B₂ (Å³) | `grid` 1.0 Å − `gpu` | `grid` 0.5 Å − `gpu` |
|-----------------------|---------------|----------------------|----------------------|
| `calvados3`, `single` |    10 914     |   −1                 |   −1                 |
| `calvados3`, `multi`  |  −150 591     |   −5                 |   −7                 |
| `kimhummer`, `single` |  −154 384     | −303 (0.20 %)        |  −50                 |
| `kimhummer`, `multi`  |  −184 126     | −125 (0.07 %)        |  −14                 |

At 2 Å the difference reaches 1.2 % of B₂ (Kim–Hummer, single bead).
On these systems B₂ did not change for `--grid-cap` from 30 to 1000 kJ/mol; a cap of 10 kJ/mol shifted it.

### Reproducing

From the package root, the grid's accuracy and speed on lysozyme, and its contact energies on 7KTZ against `--backend gpu`:

~~~
cargo test --release -p duello --lib grid_scan -- --ignored --nocapture
~~~

# Grid backend: method

`--backend grid` precomputes molecule A's short-range field on a Cartesian map, one map per atom type in B, as in AutoDock's affinity grids ([Goodsell & Olson 1990](https://doi.org/10.1002/prot.340080302); [Morris et al. 2009](https://doi.org/10.1002/jcc.21256)).
A pose then costs one map lookup and a short exact sum per B bead, plus an exact screened-Coulomb (Yukawa) sum over charged beads.
Electrostatics are not gridded, so low ionic strength doesn't grow the maps.

## Near/far split

Trilinear interpolation smears steep walls and narrow contact wells.
So each pair energy U(r) is split by a smooth switch S(r), as in multilevel summation ([Tameling et al. 2014](https://doi.org/10.1063/1.4857735)):

- the near part S·U, which contains every wall and well, is summed exactly over A's beads within r₂ of each B bead;
- only the far part (1 − S)·U is mapped.

S is a quintic smoothstep (C²) from 1 at r₁ to 0 at r₂ = r₁ + 3 Å.
r₁ lies 1 Å past the outermost well bottom (or, for a pair without a well, the end of its wall) among the type pairs present, so the far part contains no wall or well.
For CALVADOS 3 and Kim–Hummer, r₂ is about 12 Å.
The near beads are found with a cell list of A, with cell edge r₂.

## Prefiltered far map

The far part's attractive tail is curved, so plain trilinear lookups would be biased in one direction, and each B bead sums many such pairs.
Map nodes therefore hold f − h²/12·∇²f, where h is the spacing and ∇²f = f″ + 2f′/r is computed per pair by central differences (step 0.1 Å).
Linear interpolation errs by h²/12·f″ on average over a cell, so this cancels the leading error on average where f is smooth.
The error at any single point is still O(h²), and B₂ is not guaranteed to be unbiased.
A short `--cutoff` leaves a kink where the stencil crosses it; there the correction is right only on average over node positions, so check convergence when using one.

## Overlaps

Each B bead's near energy is capped at `--grid-cap`, and a pose with a bead at the cap counts at least that much.
The cap acts on a bead's net near energy, so a bead pressed into one A bead but held in the wells of others keeps its summed energy.

## Memory

Maps are stored as f16 in 2×2×2 bricks, which keeps lookups cache-friendly.
They take about 2 bytes × nodes × B atom types of GPU memory, about 200 MB for lysozyme (2CDS) at 0.5 Å.
The box covers A plus the range where the short-range potentials are nonzero, independent of `--cutoff` and `--rmax`.

Accuracy and speed comparisons are in [BENCHMARKS.md](../BENCHMARKS.md).

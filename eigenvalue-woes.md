# Eigenvalue Analysis: Challenges and Solutions

## Problem

Computing the spectral gap of the symmetrized Smoluchowski generator on a 5D
angular grid (n_v² × n_omega states, up to 144K) for macromolecular energy
landscapes with large energy variations across orientations.

## Three regimes

| R range | Landscape character | n_active | Difficulty |
|---------|-------------------|----------|------------|
| Close (< σ) | Few accessible orientations, deep steric clashes | 10–1000 | Easy (small dense matrix) |
| Intermediate (σ to ~2σ) | Many states with moderate-to-large energy range | 5K–100K | **Hard** (extreme condition number) |
| Long range (> 2σ) | Nearly flat, all rates ≈ 1 | ~144K | Easy (well-conditioned) |

The intermediate region is the most physically interesting (bounding-sphere
overlap, funneling effect) and the hardest numerically.

## Root causes

### 1. Downhill exit rate overflow (fixed)

The Smoluchowski rate `exp(-β(U_j - U_i)/2)` overflows for downhill transitions
from high-energy states. A state at U = +800 kT with a neighbor at U = -10 kT
produces `exp(505) = Inf`.

**Fix:** Symmetric overflow guard `|half_delta| > 50` skips both directions.

### 2. Extreme condition number

Even after the overflow fix, the generator diagonal spans many orders of
magnitude. A state at +100 kT has exit rate ≈ Σ exp(+50) ≈ 10²², while a
state at the energy minimum has exit rate ≈ n_neighbors. The condition number
of the matrix can reach 10²⁰.

No iterative eigensolver (Lanczos, Arnoldi) can resolve the spectral gap (eigenvalue
near zero) when the extreme eigenvalues are at -10²². The Krylov subspace converges
to the extremes first, and 200 iterations cannot bridge 20 orders of magnitude.

### 3. Energy threshold filtering (partial fix)

Excluding states with β(U - U_min) > threshold reduces the matrix size but
doesn't solve the condition number problem. With threshold = 30 kT, exit rates
still span exp(-15) to exp(+15) — 13 orders of magnitude.

## Current solution: hybrid dense/Lanczos

1. **Energy filter**: Exclude thermally inaccessible states (β ΔU > 30 kT)
   from the generator, remapping to a compact n_active × n_active matrix.

2. **Dense eigensolver** (n_active ≤ 10K): nalgebra `SymmetricEigen` gives
   exact eigenvalues regardless of condition number. Handles the intermediate
   region reliably. Cost: O(n³), ~seconds for n ≤ 10K.

3. **`lanczos` crate** (n_active > 10K): Uses `Order::Largest` to target
   eigenvalues near zero. Works for the long-range regime where the landscape
   is well-conditioned.

## Results on lysozyme (4lzt) at I = 0.1 M

- **Before fixes:** 12/62 R-slices had eigenvalues (only long-range)
- **After fixes:** 46/62 R-slices (covers the overlap region)
- λ₁/λ₁_free = 300–900× at R = 31–38 Å (funneling effect)
- Gap: R = 26–30 Å (n_active = 100–5000, dense works, but eigenvalues
  fail the plausibility check due to extreme values)

## Remaining limitations

- **R = 26–30 Å**: Active state count is small enough for dense, but the
  eigenvalues are physically very large (>1000× free) and get rejected by
  `MAX_EIGENVALUE_RATIO`. These may be genuine physics (extreme funneling)
  or numerical artifacts from the energy threshold boundary.

- **Large grids without energy filtering** (e.g., n_v = 162, n_omega = 32,
  840K states): Neither dense nor Lanczos is feasible. Would require
  shift-invert Lanczos (sparse LU factorization near σ = 0) or a different
  approach entirely.

- **The `lanczos` crate uses nalgebra 0.33**, while duello uses 0.34. This
  causes two nalgebra versions in the dependency tree. The eigenvector types
  from the Lanczos path need conversion via `.iter().copied().collect()`.

## Physical interpretation of λ₁/λ₁_free > 1

A ratio exceeding 1 means the energy landscape **accelerates equilibration**
compared to free diffusion. This is physical, not an artifact:

- At short R, most orientations are sterically forbidden
- The few accessible orientations are connected by steep energy gradients
- The system quickly funnels into these orientations
- The effective angular state space is smaller → faster equilibration

This is complementary to the Zwanzig D/D₀, which measures long-time
transport (always ≤ 1). Both can be true simultaneously: fast local
equilibration but slow long-range rotational drift.

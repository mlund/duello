# Eigenvalue Analysis Review

## Current Implementation

Sparse Lanczos eigensolver for spectral gap analysis, with dense fallback for small matrices.
No external eigensolver crate — own Lanczos with full re-orthogonalization.

### Dependencies

- `nalgebra = "0.34"` — dense matrices, `SymmetricEigen`
- `nalgebra-sparse = "0.11"` — `CooMatrix`, `CscMatrix` (matches nalgebra 0.34)

### Architecture

```
diffusion_at_r(table, ri, beta, ...)
  ├── zwanzig()              — D/D⁰ transport coefficient
  ├── marginal_zwanzig()     — per-coordinate D_A, D_B, D_ω
  ├── build_generator()      — CooMatrix → CscMatrix (sparse)
  ├── count_components()     — BFS on sparsity pattern
  └── if D/D⁰ > 0.01:       — skip steric overlap regime
      ├── n ≤ 500:  dense_eigenmodes()   — nalgebra SymmetricEigen
      └── n > 500:  sparse_eigenmodes()  — own Lanczos iteration
          ├── lanczos_largest()          — O(m²n) with full re-orth
          └── coordinate_projection()   — edge-variance decomposition
```

### Key Design Decisions

1. **Own Lanczos over crate**: `lanczos 0.2` uses nalgebra 0.33 (version mismatch).
   Our implementation matches the crate's approach (full re-orth, tridiagonal projection)
   with ~80 lines of code.

2. **`Order::Largest` not shift-invert**: All generator eigenvalues are ≤ 0, so the
   spectral gap is the *largest* non-trivial eigenvalue. Standard Krylov iteration
   targets it directly — no sparse LU needed.

3. **Component-based null space skipping**: BFS counts disconnected components at each R.
   The Lanczos result skips `n_components` eigenvalues nearest zero, which is more robust
   than a magnitude threshold (ghost eigenvalues from loss of orthogonality can mimic null space).

4. **Steric overlap filter**: Eigenmodes suppressed when Zwanzig D/D⁰ < 0.01. At short R,
   steep energy cliffs create artificially fast modes (λ₁/λ₁_free >> 1) that don't
   reflect physical diffusion.

5. **`DENSE_THRESHOLD = 500`**: Dense O(n³) is ~0.01s at n=500. Above that, Lanczos with
   m=100 iterations is faster. The m²n re-orth cost (10K×n) is negligible vs m sparse
   matvecs (~12n each).

### Performance

Lysozyme homo-dimer at 100 mM NaCl, n_v=92, n_omega=17 (143,888 states/slice):

| Metric | Value |
|--------|-------|
| R-slices | 32 |
| Wall time (8 cores) | 9 s |
| Wall time (2 cores) | 23 s |
| Peak memory (8 cores) | 3.4 GB |
| Peak memory (2 cores) | 1.7 GB |
| Python LOBPCG reference | 14–26 s |

Memory scales linearly with threads: each Lanczos instance allocates an n×m basis matrix
(143K × 100 × 8B = 115 MB).

### Sparse LU Fill-in (Why Shift-Invert Fails)

The generator is a Laplacian on a product graph (icosphere × icosphere × ring).
Sparse LU produces O(n^{4/3}) fill-in: 117× at n=5K, 454× at n=16K.
LU factors become effectively dense with sparse bookkeeping overhead.
This ruled out shift-invert approaches entirely.

### Disconnected Components at Short Range

Energy filtering creates disconnected subgraphs at close range:

| R (Å) | n_active | n_components | D/D⁰ |
|--------|----------|-------------|-------|
| 27 | 143,888 | 43 | 0.000 |
| 30 | 143,888 | 167 | 0.000 |
| 35 | 143,888 | 46 | 0.258 |
| 38 | 143,888 | 7 | 0.890 |
| 41 | 143,888 | 1 | 0.989 |

Peak patchiness (167 components) occurs at R≈30 Å, just inside the bounding sphere.
Components merge as R increases, reaching full connectivity at R≈41 Å.

### Patchiness Measures

The eigenvalue spectrum provides multiple patchiness indicators:

- **n_components**: completely disconnected orientational patches
- **Slow eigenvalue count**: n_k with λ_k/λ_k_free < 0.5 (nearly disconnected patches)
- **λ₁/λ₂ ratio**: close to 1 = multiple equivalent bottlenecks; close to 0 = single dominant bottleneck
- **Eigenvector fractions**: (f_A, f_B, f_ω) show whether bottleneck is molecular tumbling or dihedral rotation

### Output Columns

CSV output includes per R-slice:

| Column | Description |
|--------|-------------|
| `n_active` | Finite-energy grid points |
| `n_components` | Disconnected components (patchiness) |
| `λk, f_Ak, f_Bk, f_ωk` | Eigenmodes with coordinate decomposition |
| `λk_free, ...` | Free-diffusion reference |
| `D/D⁰` | Zwanzig transport coefficient |
| `D_A/D_A⁰, D_B/D_B⁰, D_ω/D_ω⁰` | Per-coordinate Zwanzig |

### Files

| File | Role |
|------|------|
| `src/diffusion.rs` | Zwanzig, Lanczos, eigenmodes, cell model |
| `src/main.rs` | CLI with `-j` thread control, CSV output |
| `scripts/spectral.py` | Python LOBPCG reference implementation |
| `scripts/plot_diffusion.py` | Three-panel plot (Zwanzig, spectral, patchiness) |

### Matrix Export for Postprocessing

`--export-matrices <dir>` saves per R-slice:
- `generator_R{R}.mtx` — symmetric generator (Matrix Market COO)
- `coords_R{R}.csv` — compact_index → (vi, vj, oi)
- `metadata.csv` — R, n_active, n_total, n_v, n_omega, lambda1_free
- `icosphere_nv{n}.csv` — vertex adjacency list

Python `scripts/spectral.py` loads these for independent validation via scipy LOBPCG.

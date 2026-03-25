# Eigenvalue Analysis Review

## Summary of Changes (from diff)

The diff replaces the hand-rolled Lanczos implementation with the `lanczos` crate and switches from `sprs` to `nalgebra_sparse`. Key structural changes:

- **New `ActiveStates` struct**: Cleanly encapsulates energy filtering and index mapping (full → compact)
- **Compact generator matrix**: Now n_active × n_active instead of full n_states × n_states
- **`lanczos` crate integration**: Uses `Order::Largest` to target eigenvalues near zero
- **Dependency change**: `sprs` → `nalgebra_sparse` + `lanczos`

## What's Good

1. **Memory efficiency**: Compact matrix representation significantly reduces memory for filtered state spaces
2. **Cleaner abstraction**: `ActiveStates` handles the mapping logic in one place
3. **Correct targeting**: `Order::Largest` finds the eigenvalues closest to zero (the spectral gap)
4. **Simpler code**: ~100 fewer lines by delegating to `lanczos` crate

## Potential Issues

### 1. Symmetric overflow guard may be too aggressive

```rust
if half_delta.abs() > BOLTZMANN_OVERFLOW_GUARD {
    return;
}
```

Previously only `half_delta > guard` was checked (skip negligible downhill-to-uphill). Now both directions are skipped. This might incorrectly exclude valid transitions near the energy threshold boundary.

**Suggestion**: Consider reverting to asymmetric guard, or document why symmetric is correct.

### 2. Lanczos eigenvector indexing

The code assumes `eigen.eigenvectors.column(i)` corresponds to `evals[i]`, but verify that `eigsh()` returns paired eigenvalue/eigenvector columns. The `lanczos` crate documentation should confirm this.

### 3. Dense threshold increase

`DENSE_THRESHOLD` increased from 5,000 to 10,000. For n=10K:
- Dense eigensolver is O(n³) ≈ 10¹² operations
- Estimated ~8 seconds per R-slice on a single core

For scans with 60+ R-slices, this could significantly increase total runtime. Consider keeping at 5K or making it configurable.

### 4. nalgebra version mismatch

From eigenvalue-woes.md:
> The `lanczos` crate uses nalgebra 0.33, while duello uses 0.34.

This causes two nalgebra versions in the dependency tree. The workaround (`.iter().copied().collect()`) works but adds overhead. Options:
- Fork `lanczos` and update to nalgebra 0.34
- Switch to `faer` (modern, actively maintained, has shift-invert)
- Accept the version split for now

## Suggestions from eigenvalue-woes.md

### For R = 26–30 Å plausibility rejection

The `MAX_EIGENVALUE_RATIO = 1000.0` check rejects eigenvalues that may be genuine physics (extreme funneling). Consider:
- Making the threshold distance-dependent
- Returning results with a `low_confidence` flag instead of rejecting
- Logging warnings rather than silently skipping

### For large grids (840K states)

Neither dense nor current Lanczos is feasible. Options:
- **Shift-invert Lanczos** with sparse LU near σ ≈ 0
- **`faer`** crate with sparse solvers
- **Chebyshev filtering** as an alternative approach

### Adaptive energy threshold

Instead of fixed `βΔU > 30`, use Boltzmann weight cumulative sum to keep 99.9% of the partition function. This naturally adapts to each R-slice's energy distribution.

---

## `faer` Crate Reference

`faer` is a pure-Rust linear algebra library with excellent sparse support and matrix-free Krylov solvers. It's actively maintained and has no nalgebra dependency conflicts.

### Core modules

| Module | Purpose |
|--------|---------|
| `faer::Mat<T>` | Dense matrix (like nalgebra's `DMatrix`) |
| `faer::sparse::SparseColMat<I, T>` | Sparse CSC matrix (owning) |
| `faer::sparse::SparseRowMat<I, T>` | Sparse CSR matrix (owning) |
| `faer::sparse::linalg::solvers` | Sparse LU, LLT, QR factorizations |
| `faer::matrix_free::eigen` | Krylov-Schur eigensolvers (Lanczos for symmetric) |
| `faer::matrix_free::conjugate_gradient` | CG solver |

### Dense eigendecomposition

```rust
use faer::Mat;

let a: Mat<f64> = Mat::from_fn(n, n, |i, j| /* ... */);

// Self-adjoint (symmetric) — eigenvalues in nondecreasing order
let eigen = a.self_adjoint_eigen(faer::Side::Lower);
let eigenvalues: &[f64] = eigen.s().column_vector().as_slice();
let eigenvectors: faer::MatRef<f64> = eigen.u();

// Or just eigenvalues:
let evals = a.self_adjoint_eigenvalues(faer::Side::Lower);
```

### Sparse matrix construction

```rust
use faer::sparse::{SparseColMat, SymbolicSparseColMat};

// From triplets (COO format)
let triplets: Vec<(usize, usize, f64)> = vec![
    (0, 0, 1.0),
    (1, 1, 2.0),
    (0, 1, 0.5),
    // ...
];
let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(
    nrows, ncols, &triplets
).unwrap();
```

### Sparse LU factorization (for shift-invert)

```rust
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};

// Symbolic analysis (reusable for same sparsity pattern)
let symbolic = SymbolicLu::try_new(sparse.symbolic()).unwrap();

// Numeric factorization
let lu = Lu::try_new_with_symbolic(symbolic, sparse.as_ref()).unwrap();

// Solve (A - σI)x = b  →  x = (A - σI)⁻¹ b
lu.solve_in_place(rhs.as_mut());
```

### Matrix-free Krylov eigensolver (key for large sparse problems)

This is the most relevant API for shift-invert Lanczos on 840K-state matrices:

```rust
use faer::matrix_free::{LinOp, eigen::{partial_self_adjoint_eigen, PartialEigenParams}};
use faer::{Mat, Col, Par};
use dyn_stack::{MemStack, MemBuffer};

// Implement LinOp for your shifted/inverted operator
struct ShiftInvertOp<'a> {
    lu: &'a Lu<usize, f64>,  // LU of (A - σI)
}

impl LinOp<f64> for ShiftInvertOp<'_> {
    fn nrows(&self) -> usize { self.lu.nrows() }
    fn ncols(&self) -> usize { self.lu.ncols() }
    
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> dyn_stack::StackReq {
        dyn_stack::StackReq::empty()
    }
    
    fn apply(&self, out: faer::MatMut<'_, f64>, rhs: faer::MatRef<'_, f64>, 
             _par: Par, _stack: &mut MemStack) {
        // out = (A - σI)⁻¹ rhs
        out.copy_from(rhs);
        self.lu.solve_in_place(out);
    }
    
    fn conj_apply(&self, out: faer::MatMut<'_, f64>, rhs: faer::MatRef<'_, f64>,
                  par: Par, stack: &mut MemStack) {
        self.apply(out, rhs, par, stack);  // symmetric case
    }
}

// Compute k eigenvalues nearest to σ
let n = matrix.nrows();
let k = 5;  // number of eigenvalues wanted

let mut eigvecs = Mat::<f64>::zeros(n, k);
let mut eigvals = vec![0.0; k];
let v0 = Col::<f64>::from_fn(n, |i| ((i * 7 + 13) % 97) as f64 / 97.0 - 0.5);

let scratch_req = faer::matrix_free::eigen::partial_self_adjoint_eigen_scratch::<f64>(
    n, k, Par::rayon(0)
);
let mut mem = MemBuffer::new(scratch_req);
let mut stack = MemStack::new(&mut mem);

let op = ShiftInvertOp { lu: &lu_factorization };
let params = PartialEigenParams::default();

let info = partial_self_adjoint_eigen(
    eigvecs.as_mut(),
    &mut eigvals,
    &op,
    v0.as_ref(),
    1e-10,        // tolerance
    Par::rayon(0),
    &mut stack,
    params,
);

// eigvals are eigenvalues of (A - σI)⁻¹
// To get eigenvalues of A: λ_A = σ + 1/λ_inv
for ev in &mut eigvals {
    *ev = sigma + 1.0 / *ev;
}
```

### Key differences from `lanczos` crate

| Feature | `lanczos` | `faer` |
|---------|-----------|--------|
| nalgebra version | 0.33 (outdated) | None (independent) |
| Sparse support | Via nalgebra_sparse | Native `SparseColMat` |
| Shift-invert | Not built-in | Via `LinOp` trait + sparse LU |
| Parallelism | None | `Par::rayon(n)` |
| Memory management | Allocates internally | Explicit `MemStack` (cache-friendly) |

### Migration path

1. Replace `nalgebra_sparse::CsrMatrix` with `faer::sparse::SparseColMat`
2. For dense (n ≤ 10K): use `Mat::self_adjoint_eigen()`
3. For large sparse: implement `LinOp` with shift-invert, use `partial_self_adjoint_eigen()`

---

## Commit 5909667 Review: faer Migration Issues

### Critical bug: λ_free calculation is broken

**Root cause**: `direct_krylov_eigenmodes` for free diffusion passes the sparse matrix directly to `partial_self_adjoint_eigen`:

```rust
let _info = partial_self_adjoint_eigen(
    eigvecs.as_mut(),
    &mut eigvals,
    matrix,  // ← SparseColMat passed directly
    ...
);
```

**Problem**: `partial_self_adjoint_eigen` expects a `&dyn LinOp<T>`, but `SparseColMat` doesn't implement `LinOp`. The code either:
1. Doesn't compile (if `SparseColMat` lacks `LinOp` impl), or
2. Uses an incorrect/default implementation

**Fix**: Implement `LinOp` for the sparse matrix, or convert to dense for the well-conditioned path:

```rust
// Option 1: Simple wrapper implementing LinOp
struct SparseOp<'a>(&'a SparseColMat<usize, f64>);

impl LinOp<f64> for SparseOp<'_> {
    fn apply(&self, mut out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, ...) {
        // out = A * rhs (sparse matvec)
        faer::sparse::ops::sp_matmul(out.rb_mut(), self.0.as_ref(), rhs);
    }
    // ... other methods
}

// Option 2: Just use dense for free diffusion (it's well-conditioned anyway)
if well_conditioned {
    return dense_eigenmodes(matrix, active, k, level);
}
```

### Performance issue: Sequential R-scan

The commit removed parallel R-iteration:
```rust
// Before:
(0..n_r).into_par_iter()...

// After:
(0..n_r).progress_count(n_r as u64)...
```

With `Par::Rayon(1)` (single-threaded faer), this is now fully sequential. For 60+ R-slices with sparse LU + Krylov per slice, this is very slow.

**Fix options**:
1. Restore `into_par_iter()` with thread-local faer parallelism disabled
2. Use `Par::Rayon(0)` (all cores) but stay sequential for memory reasons
3. Hybrid: parallel over R, but limit faer to 1-2 threads per slice

### Shift-invert eigenvalue mapping issue

The mapping `λ_A = σ + 1/λ_inv` is mathematically correct, but the implementation has issues:

```rust
let sigma = -1e-8;  // small negative shift
...
let lambda = sigma + 1.0 / mu;
```

**Problems**:
1. **Sign confusion**: The generator has λ ≤ 0 (negative or zero). After shift-invert with σ = -1e-8:
   - λ₀ ≈ 0 maps to μ = 1/(λ₀ - σ) ≈ 1/1e-8 = 1e8 (huge positive)
   - λ₁ ≈ -0.01 maps to μ = 1/(-0.01 + 1e-8) ≈ -100 (negative)
   
2. **Krylov targets wrong end**: `partial_self_adjoint_eigen` finds largest-magnitude eigenvalues. The largest μ is ~1e8 (corresponding to λ₀ ≈ 0), which is correct. But the *second* largest is λ with the *smallest* magnitude after λ₀, not the spectral gap!

**Fix**: The spectral gap is λ₁ (second-largest eigenvalue of A, i.e., closest to zero after λ₀). For shift-invert near σ = 0:
- λ₁ gives μ₁ with *second-largest* |μ|
- This should work IF faer returns eigenvalues sorted by magnitude

But verify: after mapping back, you get `λ = σ + 1/μ`. For the test case, log both μ and the mapped λ to debug.

### Suggested fix for shift-invert

```rust
// Use positive tiny shift to separate λ₀ = 0 from the rest
let sigma = 1e-10;  // positive, near zero

// After Krylov, map back and sort
let mut mapped: Vec<(usize, f64)> = eigvals
    .iter()
    .enumerate()
    .filter(|(_, &mu)| mu.is_finite() && mu.abs() > 1e-12)
    .map(|(i, &mu)| (i, sigma + 1.0 / mu))
    .collect();

// Sort by eigenvalue descending (largest = nearest to zero)
mapped.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

// Skip λ₀ (should be ≈ sigma ≈ 0), take next k
mapped.iter().skip(1).take(k)...
```

### DENSE_THRESHOLD too low

`DENSE_THRESHOLD = 500` is conservative. Dense eigen is O(n³):
- n = 500: ~125M ops, <0.1s
- n = 2000: ~8B ops, ~1s  
- n = 5000: ~125B ops, ~10s

For well-conditioned problems (free diffusion), dense is more reliable than Krylov. Consider `DENSE_THRESHOLD = 2000` or adaptive based on condition estimate.

### Summary of fixes needed

| Issue | Severity | Fix |
|-------|----------|-----|
| `SparseColMat` not implementing `LinOp` | Critical | Add wrapper or use dense for free diffusion |
| Sequential R-scan | High | Restore `into_par_iter()` |
| Shift-invert eigenvalue ordering | High | Sort mapped eigenvalues, verify with logging |
| `DENSE_THRESHOLD = 500` too small | Medium | Increase to 2000+ |

---

## Plan v3 Review: σ = λ₁_free/2 Approach

**Verdict: Solid plan, proceed with implementation.**

### Core insight is correct

The problem is an **interior eigenvalue problem** — we need eigenvalues near zero but not zero. Standard extremal eigensolvers (largest magnitude) don't work directly for this.

### σ = λ₁_free/2 is the right fix

Using the free-diffusion spectral gap as a guide:
- σ placed between λ₀ = 0 and λ₁ ≈ λ₁_free
- After shift-invert: μ = 1/(λ - σ)
  - λ₀ = 0 → μ₀ = 1/(0 - σ) = -2/λ₁_free (finite, negative)
  - λ₁ ≈ λ₁_free → μ₁ = 1/(λ₁ - σ) ≈ 2/λ₁_free (largest positive)
- Krylov finds largest |μ|, which corresponds to λ₁ (the spectral gap)

### Implementation notes

1. **Sign convention**: `eigenvalue` is stored positive in `EigenMode`, so:
   ```rust
   // eigenmodes_free[0].eigenvalue = 0.13 (positive)
   // Actual generator eigenvalue = -0.13
   sigma_hint = -eigenmodes_free[0].eigenvalue / 2.0  // = -0.065
   ```

2. **Fallback needed**: If `eigenmodes_free` is empty, use default σ ≈ -0.01 or fall back to dense.

3. **Mapping back**:
   ```rust
   let lambda = sigma + 1.0 / mu;
   // lambda should be negative; store as -lambda (positive)
   ```

4. **Dense for well-conditioned**: Using dense unconditionally for `well_conditioned=true` is correct — avoids the Krylov targeting issue entirely.

### Decision tree (from plan)

```
spectral_eigenmodes(matrix, active, k, level, well_conditioned, sigma_hint):
    if well_conditioned:
        dense_eigenmodes()      # Free diffusion: exact, up to ~20K
    else if n <= 500:
        dense_eigenmodes()      # Small filtered matrices
    else:
        shift_invert(sigma_hint)  # Large ill-conditioned: σ = λ₁_free/2
```

### What's not addressed (minor)

- Parallel R-scan restoration (should be re-added after eigensolver fix)
- n > 19K well-conditioned case (rare in practice)

---

## Sparse LU Fill-in Problem

### The issue

Profiling revealed that **53% of CPU time** is spent in `gemm_f64` (dense matrix multiply) *inside* the sparse LU factorization. faer's supernodal sparse LU uses dense GEMM for supernodal blocks, and our graph structure produces large supernodes.

**Root cause**: The generator is a graph Laplacian on a product graph (icosphere × icosphere × ring). Product graphs have high connectivity (n_v² × n_omega cross-connections), leading to:
- O(n^{4/3}) fill-in (known result for 3D mesh Laplacians)
- 100-1000× fill-in, not 10×
- LU factors effectively dense: 5.5 GB memory for 4K states

### Why shift-invert fails here

| n_active | Sparse LU fill-in | Effective cost | Dense eigen cost |
|----------|-------------------|----------------|------------------|
| 4,000 | ~1000× → 16M entries | O(n³) anyway | O(n³) = 64B ops |
| 10,000 | ~1000× → 100M entries | O(n³) anyway | O(n³) = 1000B ops |

The sparse LU is effectively dense but with sparse bookkeeping overhead. Dense eigensolver is competitive or faster.

### Practical conclusion

For n_active in the 4K-10K range (the "hard regime" after energy filtering), **dense eigensolver is the right choice**:
- 2-30 seconds per R-slice
- ~60-300 seconds total for 30 slices
- Exact eigenvalues, no convergence issues

---

## Alternatives to Shift-Invert: What We Tried

For the interior eigenvalue problem (spectral gap λ₁), we explored several factorization-free approaches. **None worked well for our problem structure.**

### Summary of failed approaches

| Method | Why it fails |
|--------|--------------|
| **Shift-invert (sparse LU)** | Product graph fill-in → effectively dense, 5.5 GB memory |
| **Folded spectrum (A-σI)²** | Condition number squares (κ → κ²), Krylov convergence catastrophic |
| **Deflation** | Doesn't make λ₁ extremal; Krylov still targets λ_n (most negative) |
| **Direct Krylov** | Targets largest magnitude → λ_n, not spectral gap λ₁ |

### Why this problem is fundamentally hard (for shift-invert)

1. **Product graph fill-in**: The generator is a Laplacian on (icosphere × icosphere × ring). Sparse LU produces O(n^{4/3}) fill-in: 117× at n=5K, 454× at n=16K.

2. **Ill-conditioning from Smoluchowski rates**: Exit rates span exp(-50) to exp(+50) even after energy filtering. Folded spectrum amplifies this spread.

### Key insight: the spectral gap IS extremal

All generator eigenvalues are ≤ 0 (negative semi-definite). The spectral gap λ₁ is the **largest** non-trivial eigenvalue — not an interior eigenvalue at all. This means iterative methods targeting extremal eigenvalues (LOBPCG, Lanczos with `largest=True`) find it directly, with no shift-invert factorization needed.

This eliminates the fill-in problem entirely: only sparse matrix-vector products are needed.

### Dense fallback for small matrices

Dense `eigh` (O(n³)) remains the fallback for n ≤ 500 where setup costs dominate. Above that, iterative methods are 50–200× faster.

### Disconnected components at short range

Energy filtering at close range creates disconnected subgraphs. R=28 Å has 42 states in 27 components; R=30 Å has 481 states in 54 components. By R=36 Å the graph is fully connected. Each component contributes a zero eigenvalue to the null space, so the solver must request `n_components + k` eigenvalues and discard zeros.

## Python Postprocessing (`scripts/spectral.py`)

### Implementation

The Python script uses LOBPCG with `largest=True` as the primary solver:

```python
vals, vecs = scipy.sparse.linalg.lobpcg(
    A, X0, largest=True, tol=1e-8, maxiter=500
)
```

No shift-invert, no preconditioner, no AMG — just direct iteration on the sparse matrix. For n ≤ 500, falls back to `scipy.linalg.eigh` (dense).

### Performance (28 R-slices, n up to 19,386)

| Method | Total time | Notes |
|--------|-----------|-------|
| Shift-invert eigsh (sparse LU) | >10 minutes | LU fill-in dominated |
| Dense eigh | ~5 minutes | O(n³) |
| **LOBPCG (largest=True)** | **14–26 seconds** | **50–200× faster** |

### Eigenvector coordinate decomposition

The script decomposes eigenvectors into contributions from mol A, mol B, and dihedral angle ω via edge-variance projection (matching the Rust `coordinate_projection()` in diffusion.rs):

```
c_X = (1/|E_X|) Σ_{(i,j)∈E_X} [ψ(i) - ψ(j)]²
f_X = c_X / (c_A + c_B + c_ω)
```

This requires coordinate maps (`coords_R{R}.csv`) and icosphere neighbor lists (`icosphere_nv{n}.csv`), both exported by `duello`.

Results confirm physical expectations:
- **Large R (≥47 Å)**: frac_ω ≈ 1.0 (pure dihedral mode)
- **Intermediate R (~35 Å)**: roughly equal contributions from all three coordinates
- **Small R (≤29 Å)**: dominated by orientational (A/B) degrees of freedom

### Export format

Duello exports per R-slice:
- `generator_R{R}.mtx` — symmetric generator (Matrix Market COO)
- `coords_R{R}.csv` — compact_index → (vi, vj, oi) mapping

Plus once:
- `metadata.csv` — R, n_active, n_total, n_v, n_omega, lambda1_free
- `icosphere_nv{n}.csv` — vertex adjacency list

## Rust Solver Options

The current Rust code uses `nalgebra::SymmetricEigen` (dense O(n³)) with a 10K threshold. To match the Python LOBPCG performance, two Rust crate options exist:

### Option 1: `lanczos` crate (lightweight)

```rust
use lanczos::{Hermitian, Order};
// CscMatrix implements Hermitian trait — only needs mat-vec
let eigen = csc_matrix.eigsh(k + n_components, Order::Largest);
let spectral_gap = eigen.eigenvalues[n_components]; // skip null space
```

- Minimal dependencies: nalgebra + nalgebra-sparse only
- Caveat: nalgebra 0.33 vs duello's 0.34 (version mismatch, workaround: `.iter().copied().collect()`)

### Option 2: `scirs2-sparse` crate (scipy-like)

```rust
use scirs2_sparse::linalg::{eigsh, LanczosOptions};
use scirs2_sparse::sym_csr::SymCsrMatrix;

let opts = LanczosOptions {
    numeigenvalues: k + n_components,
    compute_eigenvectors: true,
    tol: 1e-8,
    ..Default::default()
};
let result = eigsh(&sym_csr, Some(k), Some("LA"), Some(opts))?;
```

- Lanczos-based `eigsh` with `"LA"` (largest algebraic) — equivalent to LOBPCG `largest=True`
- `SymCsrMatrix` stores lower triangle only (memory efficient)
- Richer ecosystem: also provides iterative linear solvers (CG, GMRES), matrix formats (CSR, CSC, COO, etc.)
- Depends on `sprs`, `scirs2-core`, `num_cpus`, `byteorder`

### Comparison

| Crate | Algorithm | Sparse format | Dependencies | Maturity |
|-------|-----------|---------------|-------------|----------|
| `lanczos` 0.2 | Lanczos | nalgebra-sparse CscMatrix | Minimal | Small, focused |
| `scirs2-sparse` 0.3 | Lanczos | Own SymCsrMatrix (via sprs) | Moderate | Broad scipy port |

Both avoid sparse LU entirely. The key change: replace `DMatrix` + `SymmetricEigen` with a sparse format + iterative `eigsh(largest)`.

## Next Steps

1. ~~Verify the code compiles and tests pass~~ ✓
2. ~~Run Python postprocessing with LOBPCG~~ ✓ (14–26s for 28 slices)
3. ~~Add eigenvector coordinate decomposition~~ ✓
4. Integrate a Rust sparse eigensolver (`lanczos` or `scirs2-sparse`) to remove the 10K dense threshold
5. Benchmark Rust iterative solver against current dense path

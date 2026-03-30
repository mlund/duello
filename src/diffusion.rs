// Copyright 2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Rotational diffusion analysis from 6D energy tables.
//!
//! Two complementary approaches:
//!
//! 1. **Zwanzig formula** — normalized transport coefficient D_r/D_r⁰:
//!    `D/D₀ = 1 / [⟨exp(βU)⟩ × ⟨exp(-βU)⟩]`
//!    Exact for 1D periodic potentials, good approximation for higher D.
//!
//! 2. **Spectral gap** — relaxation rate from the transition rate matrix.
//!    The spectral gap λ₁ is the slowest non-equilibrium eigenmode of the
//!    symmetrized generator, computed via sparse Lanczos iteration.

use anyhow::Result;
use icotable::adaptive::MeshLevel;
use indicatif::ParallelProgressIterator;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use rayon::prelude::*;

/// Number of non-trivial eigenmodes to extract from the generator spectrum.
const NUM_EIGENMODES: usize = 5;

/// Eigenvalue with coordinate decomposition showing which angular
/// degree of freedom dominates the mode.
#[derive(Debug, Clone)]
pub struct EigenMode {
    /// Eigenvalue magnitude |λ|.
    pub eigenvalue: f64,
    /// Fraction of mode from molecule A rotation [0,1].
    pub frac_mol_a: f64,
    /// Fraction from molecule B rotation [0,1].
    pub frac_mol_b: f64,
    /// Fraction from dihedral rotation [0,1].
    pub frac_omega: f64,
}

/// Result of diffusion analysis at one R-slice.
#[derive(Debug, Clone)]
pub struct DiffusionResult {
    /// Radial distance (Å).
    pub r: f64,
    /// Normalized diffusion D_r/D_r⁰ (Zwanzig formula, full 5D).
    pub dr_normalized: f64,
    /// Per-molecule-A Zwanzig D_A/D_A⁰ (marginalized over vj, oi).
    pub dr_mol_a: f64,
    /// Per-molecule-B Zwanzig D_B/D_B⁰ (marginalized over vi, oi).
    pub dr_mol_b: f64,
    /// Per-dihedral Zwanzig D_ω/D_ω⁰ (marginalized over vi, vj).
    pub dr_omega: f64,
    /// Angular Boltzmann average ⟨exp(-βU)⟩ ∝ exp(-βw(R)). Used as PMF weight.
    pub boltzmann_weight: f64,
    /// First non-trivial eigenmodes with coordinate decomposition.
    pub eigenmodes: Vec<EigenMode>,
    /// Free-diffusion eigenmodes for normalization.
    pub eigenmodes_free: Vec<EigenMode>,
    /// Number of active states (finite-energy grid points).
    pub n_active: usize,
    /// Number of disconnected components in the generator graph.
    pub n_components: usize,
}

/// Flatten a 5D state (vi, vj, oi) into a linear index.
fn state_index(vi: usize, vj: usize, oi: usize, n_v: usize, n_omega: usize) -> usize {
    vi * n_v * n_omega + vj * n_omega + oi
}

/// Invoke `f` for each neighbor index of state (vi, vj, oi) without allocating.
fn for_each_neighbor(
    vi: usize,
    vj: usize,
    oi: usize,
    level: &MeshLevel,
    n_v: usize,
    n_omega: usize,
    mut f: impl FnMut(usize),
) {
    for &ni in &level.neighbors[vi] {
        f(state_index(ni as usize, vj, oi, n_v, n_omega));
    }
    for &nj in &level.neighbors[vj] {
        f(state_index(vi, nj as usize, oi, n_v, n_omega));
    }
    let oi_prev = if oi == 0 { n_omega - 1 } else { oi - 1 };
    f(state_index(vi, vj, oi_prev, n_v, n_omega));
    f(state_index(vi, vj, (oi + 1) % n_omega, n_v, n_omega));
}

/// Compute coordinate fractions for an eigenvector by measuring how much
/// the mode varies along each angular coordinate.
///
/// Uses squared differences Σ[ψ(i) - ψ(j)]² per edge, normalized by the
/// number of edges in each direction to remove topology bias (icosphere
/// vertices have ~5 neighbors vs 2 for the periodic dihedral ring).
fn coordinate_projection(
    eigvec: &[f64],
    active: &ActiveStates,
    level: &MeshLevel,
) -> (f64, f64, f64) {
    let n_v = active.n_v;
    let n_omega = active.n_omega;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    let mut var_omega = 0.0;
    let mut edges_a = 0u64;
    let mut edges_b = 0u64;
    let mut edges_omega = 0u64;

    for (ci, &(vi, vj, oi)) in active.coords.iter().enumerate() {
        let psi_i = eigvec[ci];

        for &ni in &level.neighbors[vi] {
            let cj = active.compact[state_index(ni as usize, vj, oi, n_v, n_omega)];
            if cj != usize::MAX {
                var_a += (psi_i - eigvec[cj]).powi(2);
                edges_a += 1;
            }
        }
        for &nj in &level.neighbors[vj] {
            let cj = active.compact[state_index(vi, nj as usize, oi, n_v, n_omega)];
            if cj != usize::MAX {
                var_b += (psi_i - eigvec[cj]).powi(2);
                edges_b += 1;
            }
        }
        let oi_prev = if oi == 0 { n_omega - 1 } else { oi - 1 };
        for &oi_n in &[oi_prev, (oi + 1) % n_omega] {
            let cj = active.compact[state_index(vi, vj, oi_n, n_v, n_omega)];
            if cj != usize::MAX {
                var_omega += (psi_i - eigvec[cj]).powi(2);
                edges_omega += 1;
            }
        }
    }

    let safe_div = |v: f64, e: u64| if e > 0 { v / e as f64 } else { 0.0 };
    let (na, nb, nw) = (
        safe_div(var_a, edges_a),
        safe_div(var_b, edges_b),
        safe_div(var_omega, edges_omega),
    );
    let total = na + nb + nw;
    if total < 1e-30 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }
    (na / total, nb / total, nw / total)
}

/// Compact index mapping from full state space to active subset.
struct ActiveStates {
    /// Maps full index → compact index (usize::MAX if inactive).
    compact: Vec<usize>,
    /// Maps compact index → (vi, vj, oi) for eigenvector projection.
    coords: Vec<(usize, usize, usize)>,
    /// Number of active states.
    n_active: usize,
    /// Grid dimensions (needed for eigenvector projection).
    n_v: usize,
    n_omega: usize,
}

impl ActiveStates {
    /// Build active set: all states for free diffusion, energy-filtered for potential.
    ///
    /// For potentials, keeps states covering PARTITION_FRACTION (99.9%) of the
    /// partition function. This adaptively selects the thermally relevant states.
    fn new(energies: Option<(&[f64], f64)>, n_v: usize, n_omega: usize) -> Self {
        let n_states = n_v * n_v * n_omega;
        let mut compact = vec![usize::MAX; n_states];
        let mut coords = Vec::new();

        // Compute adaptive energy cutoff from partition function
        let cutoff = energies.map(|(e, beta)| {
            let u_min = e
                .iter()
                .copied()
                .filter(|u| u.is_finite())
                .fold(f64::INFINITY, f64::min);
            // Sort finite shifted energies, accumulate Boltzmann weight
            let mut shifted: Vec<f64> = e
                .iter()
                .copied()
                .filter(|u| u.is_finite())
                .map(|u| u - u_min)
                .collect();
            shifted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let z_total: f64 = shifted.iter().map(|&u| (-beta * u).exp()).sum();
            let target = PARTITION_FRACTION * z_total;
            let mut z_accum = 0.0;
            let mut energy_cutoff = f64::INFINITY;
            for &u in &shifted {
                z_accum += (-beta * u).exp();
                if z_accum >= target {
                    energy_cutoff = u;
                    break;
                }
            }
            (u_min, energy_cutoff)
        });

        for vi in 0..n_v {
            for vj in 0..n_v {
                for oi in 0..n_omega {
                    let idx = state_index(vi, vj, oi, n_v, n_omega);
                    let active = match (energies, cutoff) {
                        (Some((e, _)), Some((u_min, threshold))) => {
                            e[idx].is_finite() && (e[idx] - u_min) <= threshold
                        }
                        _ => true,
                    };
                    if active {
                        compact[idx] = coords.len();
                        coords.push((vi, vj, oi));
                    }
                }
            }
        }

        let n_active = coords.len();
        Self {
            compact,
            coords,
            n_active,
            n_v,
            n_omega,
        }
    }
}

/// Build a sparse CscMatrix generator over the compact active state space.
fn build_generator(
    active: &ActiveStates,
    level: &MeshLevel,
    potential: Option<(&[f64], f64)>,
) -> CscMatrix<f64> {
    let n_v = level.n_vertices;
    let n_omega = active.n_omega;
    let n = active.n_active;

    let u_min = potential.map(|(energies, _)| {
        energies
            .iter()
            .copied()
            .filter(|u| u.is_finite())
            .fold(f64::INFINITY, f64::min)
    });

    let mut coo = CooMatrix::new(n, n);

    for (ci, &(vi, vj, oi)) in active.coords.iter().enumerate() {
        let full_idx = state_index(vi, vj, oi, n_v, n_omega);
        let u_i = potential.map_or(0.0, |(e, _)| e[full_idx] - u_min.unwrap());

        let mut exit_rate = 0.0;
        for_each_neighbor(vi, vj, oi, level, n_v, n_omega, |full_j| {
            let cj = active.compact[full_j];
            if cj == usize::MAX {
                return;
            }
            if let Some((energies, beta)) = potential {
                let half_delta = beta * (energies[full_j] - u_min.unwrap() - u_i) / 2.0;
                if half_delta.abs() > BOLTZMANN_OVERFLOW_GUARD {
                    return;
                }
                exit_rate += (-half_delta).exp();
            } else {
                exit_rate += 1.0;
            }
            coo.push(ci, cj, 1.0);
        });
        coo.push(ci, ci, -exit_rate);
    }

    CscMatrix::from(&coo)
}

/// Dense O(n³) is ~0.01s at n=500; above this Lanczos with m=100 wins.
const DENSE_THRESHOLD: usize = 500;

/// Safety net for dense solver; sparse path uses component counting instead.
const ZERO_THRESHOLD: f64 = 1e-6;

/// Count connected components via BFS on the active state neighbor graph.
///
/// Each disconnected component contributes one zero eigenvalue to the null space.
/// Works directly on ActiveStates + MeshLevel without building a sparse matrix.
fn count_components(active: &ActiveStates, level: &MeshLevel) -> usize {
    let n = active.n_active;
    let mut visited = vec![false; n];
    let mut n_components = 0;
    let mut queue = std::collections::VecDeque::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        n_components += 1;
        visited[start] = true;
        queue.push_back(start);
        while let Some(ci) = queue.pop_front() {
            let (vi, vj, oi) = active.coords[ci];
            for_each_neighbor(vi, vj, oi, level, active.n_v, active.n_omega, |full_j| {
                let cj = active.compact[full_j];
                if cj != usize::MAX && !visited[cj] {
                    visited[cj] = true;
                    queue.push_back(cj);
                }
            });
        }
    }
    n_components
}

/// Convert CscMatrix to nalgebra DMatrix for dense eigensolver.
fn csc_to_dense(matrix: &CscMatrix<f64>) -> DMatrix<f64> {
    let n = matrix.nrows();
    let mut dense = DMatrix::zeros(n, n);
    for j in 0..n {
        let col = matrix.col(j);
        for (&i, &v) in col.row_indices().iter().zip(col.values()) {
            dense[(i, j)] = v;
        }
    }
    dense
}

/// Build an EigenMode from an eigenvalue and its eigenvector via edge-variance projection.
fn eigenmode_from_eigvec(
    eigenvalue: f64,
    eigvec: &[f64],
    active: &ActiveStates,
    level: &MeshLevel,
) -> EigenMode {
    let (fa, fb, fw) = coordinate_projection(eigvec, active, level);
    EigenMode {
        eigenvalue,
        frac_mol_a: fa,
        frac_mol_b: fb,
        frac_omega: fw,
    }
}

/// Extract first `k` non-trivial eigenmodes using dense eigensolver.
fn dense_eigenmodes(
    matrix: &CscMatrix<f64>,
    active: &ActiveStates,
    k: usize,
    level: &MeshLevel,
) -> Vec<EigenMode> {
    let n = active.n_active;
    if n < 3 {
        return Vec::new();
    }

    let dense = csc_to_dense(matrix);
    let eigen = SymmetricEigen::new(dense);
    let evals = &eigen.eigenvalues;

    // Sort descending: evals[0] ≈ 0 (equilibrium), evals[1] = spectral gap
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    indices
        .iter()
        .filter(|&&i| evals[i].abs() > ZERO_THRESHOLD && evals[i] < 0.0)
        .take(k)
        .map(|&i| {
            let eigvec: Vec<f64> = eigen.eigenvectors.column(i).iter().copied().collect();
            eigenmode_from_eigvec(-evals[i], &eigvec, active, level)
        })
        .collect()
}

/// Lanczos with full re-orthogonalization for the m largest eigenvalues.
///
/// Cost is O(m²n) but m is fixed (~100), so the m² factor is negligible
/// next to the m sparse matvecs at ~12n each.
/// Returns (eigenvalues, eigenvectors) sorted descending (largest first).
fn lanczos_largest(matrix: &CscMatrix<f64>, m: usize) -> (Vec<f64>, Vec<DVector<f64>>) {
    let n = matrix.nrows();
    let m = m.min(n);

    // Basis stored as columns of a dense matrix for efficient Ritz vector recovery
    let mut vs = DMatrix::zeros(n, m);
    let mut alpha = DVector::zeros(m);
    let mut beta = Vec::with_capacity(m);

    // Golden-ratio hash gives a deterministic, non-uniform start vector.
    // A uniform vector would lie entirely in the null space (equilibrium).
    let q0 = DVector::from_fn(n, |i, _| {
        let x = (i as f64 + 1.0) * 0.6180339887;
        x - x.floor() - 0.5
    })
    .normalize();
    vs.set_column(0, &q0);

    let w_prime: DVector<f64> = matrix * &q0;
    alpha[0] = w_prime.dot(&q0);
    let mut w: DVector<f64> = &w_prime - &q0 * alpha[0];

    // CscMatrix * DVector requires an owned vector, not a column view.
    // Reuse one buffer to avoid 143K-element allocation per iteration.
    let mut vi_buf = DVector::zeros(n);

    let mut m_actual = m;
    for i in 1..m {
        let b = w.norm();
        beta.push(b);
        if b < 1e-12 {
            m_actual = i;
            break;
        }
        vs.set_column(i, &w.normalize());

        vi_buf.copy_from(&vs.column(i));
        let w_prime: DVector<f64> = matrix * &vi_buf;
        alpha[i] = w_prime.dot(&vi_buf);
        w = &w_prime - &vi_buf * alpha[i] - vs.column(i - 1) * b;

        // Without re-orth, Lanczos produces ghost (duplicate) eigenvalues
        // near the spectral gap — indistinguishable from the real null space.
        for j in 0..=i {
            let proj = w.dot(&vs.column(j));
            w -= vs.column(j) * proj;
        }
    }

    // Build and solve tridiagonal eigenproblem
    let tri = DMatrix::from_fn(m_actual, m_actual, |i, j| {
        if i == j {
            alpha[i]
        } else if i + 1 == j && i < beta.len() {
            beta[i]
        } else if j + 1 == i && j < beta.len() {
            beta[j]
        } else {
            0.0
        }
    });

    let eigen = SymmetricEigen::new(tri);
    let mut indices: Vec<usize> = (0..m_actual).collect();
    indices.sort_by(|&a, &b| {
        eigen.eigenvalues[b]
            .partial_cmp(&eigen.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Ritz vectors: V * y (single matrix multiply is faster than column-by-column)
    let vs_active = vs.columns(0, m_actual);
    let evals: Vec<f64> = indices.iter().map(|&i| eigen.eigenvalues[i]).collect();
    let evecs: Vec<DVector<f64>> = indices
        .iter()
        .map(|&i| vs_active * eigen.eigenvectors.column(i))
        .collect();

    (evals, evecs)
}

/// Extract first `k` non-trivial eigenmodes using sparse Lanczos iteration.
///
/// All generator eigenvalues are ≤ 0, so eigenvalues nearest zero are the
/// *largest*. We request extra to cover the null space from disconnected components.
fn sparse_eigenmodes(
    matrix: &CscMatrix<f64>,
    active: &ActiveStates,
    k: usize,
    level: &MeshLevel,
    n_components: usize,
) -> Vec<EigenMode> {
    let n = active.n_active;
    if n < 3 {
        return Vec::new();
    }

    let n_eigs = (n_components + k).min(n - 1);
    let m = n_eigs.max(100).min(n - 1);

    let (evals, evecs) = lanczos_largest(matrix, m);

    // Eigenvalues are sorted descending (nearest zero first). Each disconnected
    // component contributes one λ ≈ 0. We skip exactly n_components of these before
    // extracting the physical modes. Positive eigenvalues are numerical noise.
    let mut modes = Vec::new();
    let mut null_skipped = 0;
    for (ev, evec) in evals.iter().zip(&evecs) {
        if *ev >= 0.0 {
            continue;
        }
        if null_skipped < n_components {
            null_skipped += 1;
            continue;
        }
        modes.push(eigenmode_from_eigvec(-ev, evec.as_slice(), active, level));
        if modes.len() >= k {
            break;
        }
    }
    modes
}

/// Compute free-diffusion eigenmodes analytically from graph structure.
///
/// The product graph (icosphere_A × icosphere_B × ring_ω) has eigenvalues
/// that are sums: λ_{k,l,m} = λ_k^A + λ_l^B + λ_m^ω.
/// We compute the icosphere and ring eigenvalues separately (small problems),
/// then combine the smallest non-trivial sums.
fn free_eigenmodes_analytical(level: &MeshLevel, n_omega: usize, k: usize) -> Vec<EigenMode> {
    // Icosphere graph Laplacian eigenvalues (n_v × n_v dense — at most 162×162)
    let ico_evals = icosphere_eigenvalues(level);

    // Periodic ring eigenvalues: λ_m = 2(1 - cos(2πm/n_ω)) for m = 0..n_ω-1
    let mut ring_evals: Vec<f64> = (0..n_omega)
        .map(|m| 2.0 * (1.0 - (2.0 * std::f64::consts::PI * m as f64 / n_omega as f64).cos()))
        .collect();
    ring_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Product eigenvalues: λ = λ_A + λ_B + λ_ω (all combinations)
    // Collect the k+1 smallest non-negative sums, then skip the zero
    let mut product_evals: Vec<(f64, f64, f64, f64)> = Vec::new(); // (total, frac_A, frac_B, frac_ω)
    for &la in &ico_evals {
        for &lb in &ico_evals {
            for &lw in &ring_evals {
                let total = la + lb + lw;
                if total > 1e-10 {
                    // Coordinate fractions: which component dominates this mode
                    let sum = la + lb + lw;
                    product_evals.push((total, la / sum, lb / sum, lw / sum));
                }
            }
        }
    }
    product_evals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    product_evals.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10);

    product_evals
        .iter()
        .take(k)
        .map(|&(ev, fa, fb, fw)| EigenMode {
            eigenvalue: ev,
            frac_mol_a: fa,
            frac_mol_b: fb,
            frac_omega: fw,
        })
        .collect()
}

/// Compute eigenvalues of the icosphere graph Laplacian (small dense problem).
fn icosphere_eigenvalues(level: &MeshLevel) -> Vec<f64> {
    let n = level.n_vertices;
    let mut lap = DMatrix::zeros(n, n);
    for i in 0..n {
        let deg = level.neighbors[i].len();
        lap[(i, i)] = deg as f64;
        for &j in &level.neighbors[i] {
            lap[(i, j as usize)] = -1.0;
        }
    }
    let eigen = SymmetricEigen::new(lap);
    let mut vals: Vec<f64> = eigen.eigenvalues.iter().map(|&e| e.max(0.0)).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals
}

/// Boltzmann exponent cutoff: exp(-x) for |x| > this is treated as zero to avoid overflow.
const BOLTZMANN_OVERFLOW_GUARD: f64 = 50.0;

/// Fraction of the partition function to retain in the active state set.
const PARTITION_FRACTION: f64 = 0.999;

/// Skip spectral gap when fewer than this many active states exist.
const MIN_ACTIVE_STATES: usize = 100;

/// Skip spectral gap when active fraction is below this threshold.
const MIN_ACTIVE_FRACTION: f64 = 0.01;

/// Reject eigenvalues exceeding this multiple of the free-diffusion value.
const MAX_EIGENVALUE_RATIO: f64 = 1000.0;

/// Zwanzig formula: D/D₀ = 1 / [⟨exp(-βU)⟩ x ⟨exp(βU)⟩]
///
/// Computed in log-space with streaming sums to avoid intermediate allocations.
fn zwanzig(energies: &[f64], beta: f64) -> Option<(f64, f64, usize)> {
    // First pass: find u_min and count finite values
    let mut u_min = f64::INFINITY;
    let mut n = 0usize;
    for &u in energies {
        if u.is_finite() {
            u_min = u_min.min(u);
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }

    // The "minus" exponents (-β·(U-U_min)) are all ≤ 0 so no overflow possible.
    // The "plus" exponents (β·(U-U_min)) can overflow, so we need their max for log-sum-exp.
    let mut max_plus = f64::NEG_INFINITY;
    for &u in energies {
        if u.is_finite() {
            max_plus = max_plus.max(beta * (u - u_min));
        }
    }

    // Second pass: accumulate both log-sum-exp values
    let mut sum_minus = 0.0f64;
    let mut sum_plus = 0.0f64;
    for &u in energies {
        if u.is_finite() {
            let shifted = u - u_min;
            sum_minus += (-beta * shifted).exp();
            sum_plus += (beta * shifted - max_plus).exp();
        }
    }

    let ln_n = (n as f64).ln();
    let log_avg_minus = sum_minus.ln() - ln_n;
    let log_avg_plus = max_plus + sum_plus.ln() - ln_n;

    let d_ratio = (-log_avg_minus - log_avg_plus).exp();

    // Undo the U_min shift to get the true ⟨exp(-βU)⟩ (needed for PMF weights).
    // log_avg_minus = ln⟨exp(-β(U-U_min))⟩ = βU_min + ln⟨exp(-βU)⟩
    let true_log_avg_minus = log_avg_minus - beta * u_min;
    Some((d_ratio, true_log_avg_minus.exp(), n))
}

/// Symmetrize energies for homo-dimer exchange: U_sym(vi,vj,oi) = ½[U(vi,vj,oi) + U(vj,vi,-oi)].
///
/// The inverse_orient decomposition is asymmetric, so the raw table data doesn't
/// satisfy exchange symmetry even for identical molecules. This restores it by
/// averaging each pair of exchanged states, matching faunus's double-lookup approach.
fn symmetrize_exchange(energies: &mut [f64], n_v: usize, n_omega: usize) {
    for vi in 0..n_v {
        for vj in vi..n_v {
            for oi in 0..n_omega {
                // Periodic negation: -oi mod n_omega (oi=0 maps to 0, oi=1 maps to n-1, etc.)
                let oi_swap = (n_omega - oi) % n_omega;
                let idx_fwd = state_index(vi, vj, oi, n_v, n_omega);
                let idx_rev = state_index(vj, vi, oi_swap, n_v, n_omega);
                if idx_fwd == idx_rev {
                    continue;
                }
                let u_fwd = energies[idx_fwd];
                let u_rev = energies[idx_rev];
                // Average in Boltzmann space when both are finite
                if u_fwd.is_finite() && u_rev.is_finite() {
                    let avg = 0.5 * (u_fwd + u_rev);
                    energies[idx_fwd] = avg;
                    energies[idx_rev] = avg;
                }
            }
        }
    }
}

/// Compute 1D Zwanzig D/D⁰ from a marginal PMF array (in energy units).
///
/// Returns 1.0 if the PMF is uniform or has fewer than 2 finite values.
fn zwanzig_1d(pmf: &[f64], beta: f64) -> f64 {
    zwanzig(pmf, beta).map_or(1.0, |(ratio, _, _)| ratio)
}

/// Marginal PMF along one coordinate: w(k) = -kT ln Σ_other exp(-βU).
///
/// `n_bins`: number of bins for the marginalized coordinate.
/// `energy_at`: maps (bin_index, inner_index) → energy value.
/// `n_inner`: number of states to sum over for each bin.
fn marginal_pmf(
    n_bins: usize,
    n_inner: usize,
    beta: f64,
    energy_at: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    (0..n_bins)
        .map(|k| {
            let mut max_val = f64::NEG_INFINITY;
            let mut terms = Vec::new();
            for j in 0..n_inner {
                let u = energy_at(k, j);
                if u.is_finite() {
                    let x = -beta * u;
                    max_val = max_val.max(x);
                    terms.push(x);
                }
            }
            if terms.is_empty() {
                return f64::INFINITY;
            }
            let lse = max_val + terms.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln();
            -lse / beta
        })
        .collect()
}

/// Compute per-coordinate Zwanzig by marginalizing over the other coordinates.
///
/// For each coordinate (vi, vj, oi), integrates out the other two to get a
/// 1D potential of mean force, then applies Zwanzig independently.
///
/// Note: marginalization smooths cross-correlations, so D_A × D_B × D_ω ≥ D/D⁰.
/// The ratio D/D⁰ / (D_A × D_B × D_ω) measures coordinate separability.
fn marginal_zwanzig(energies: &[f64], n_v: usize, n_omega: usize, beta: f64) -> (f64, f64, f64) {
    let n_inner = n_v * n_omega;
    let pmf_a = marginal_pmf(n_v, n_inner, beta, |vi, j| {
        let (vj, oi) = (j / n_omega, j % n_omega);
        energies[state_index(vi, vj, oi, n_v, n_omega)]
    });
    let pmf_b = marginal_pmf(n_v, n_inner, beta, |vj, j| {
        let (vi, oi) = (j / n_omega, j % n_omega);
        energies[state_index(vi, vj, oi, n_v, n_omega)]
    });
    let pmf_omega = marginal_pmf(n_omega, n_v * n_v, beta, |oi, j| {
        let (vi, vj) = (j / n_v, j % n_v);
        energies[state_index(vi, vj, oi, n_v, n_omega)]
    });
    (
        zwanzig_1d(&pmf_a, beta),
        zwanzig_1d(&pmf_b, beta),
        zwanzig_1d(&pmf_omega, beta),
    )
}

/// Compute diffusion analysis at a single R-slice.
///
/// `free_evals` maps n_vertices → pre-computed free-diffusion eigenmodes.
/// `homo_dimer`: apply exchange symmetrization for identical molecules.
fn diffusion_at_r(
    table: &icotable::Table6DAdaptive<f32>,
    ri: usize,
    beta: f64,
    free_evals: &std::collections::HashMap<usize, Vec<EigenMode>>,
    homo_dimer: bool,
) -> Result<DiffusionResult> {
    let r = table.rmin + ri as f64 * table.dr;
    let n_omega = table.n_omega;

    let (mut energies, level) = table
        .energies_at_r(ri)
        .ok_or_else(|| anyhow::anyhow!("R-slice {ri} is fully repulsive or out of range"))?;

    let n_v = level.n_vertices;
    let n_total = n_v * n_v * n_omega;

    if homo_dimer {
        symmetrize_exchange(&mut energies, n_v, n_omega);
    }

    let (dr_zwanzig, boltzmann_weight, n_active) = zwanzig(&energies, beta)
        .ok_or_else(|| anyhow::anyhow!("No finite energies at R={r:.1}"))?;

    let (dr_mol_a, dr_mol_b, dr_omega) = marginal_zwanzig(&energies, n_v, n_omega, beta);

    // Full 5D Zwanzig underflows when steric clashes dominate ⟨exp(βU)⟩.
    // The product of marginals is a better estimate in the bounding-sphere overlap region.
    let dr_normalized = dr_zwanzig.max(dr_mol_a * dr_mol_b * dr_omega);

    let eigenmodes_free = free_evals
        .get(&level.n_vertices)
        .cloned()
        .unwrap_or_default();

    // BFS on the neighbor graph is O(n_active) and avoids building the
    // full CscMatrix, which is only needed when eigenmodes are computed.
    let active = ActiveStates::new(Some((&energies, beta)), n_v, n_omega);
    let n_components = count_components(&active, level);
    debug!(
        "R={r:.1} Å: n_active={}/{n_total}, components={n_components}",
        active.n_active
    );

    // At short R, steep energy cliffs create artificially fast eigenmodes
    // (λ₁/λ₁_free >> 1) that reflect barrier topology, not physical diffusion.
    let eigenmodes = if n_active > MIN_ACTIVE_STATES
        && (n_active as f64 / n_total as f64) > MIN_ACTIVE_FRACTION
        && dr_normalized > 0.01
    {
        let gen = build_generator(&active, level, Some((&energies, beta)));
        let modes = if active.n_active <= DENSE_THRESHOLD {
            dense_eigenmodes(&gen, &active, NUM_EIGENMODES, level)
        } else {
            sparse_eigenmodes(&gen, &active, NUM_EIGENMODES, level, n_components)
        };
        // Reject if largest eigenvalue exceeds free value by too much
        let max_plausible = eigenmodes_free
            .first()
            .map_or(f64::INFINITY, |m| m.eigenvalue * MAX_EIGENVALUE_RATIO);
        if modes.first().is_some_and(|m| m.eigenvalue < max_plausible) {
            modes
        } else {
            debug!(
                "R={r:.1} Å: rejecting eigenmodes (λ₁={:.2e}, max={max_plausible:.2e})",
                modes.first().map_or(0.0, |m| m.eigenvalue)
            );
            Vec::new()
        }
    } else {
        Vec::new()
    };

    debug!(
        "R={r:.1} Å: D_r/D_r⁰={dr_normalized:.6}, D_A={dr_mol_a:.4}, D_B={dr_mol_b:.4}, D_ω={dr_omega:.4}, λ₁={}",
        eigenmodes.first().map_or("N/A".into(), |m| format!("{:.4e}", m.eigenvalue))
    );

    Ok(DiffusionResult {
        r,
        dr_normalized,
        dr_mol_a,
        dr_mol_b,
        dr_omega,
        boltzmann_weight,
        eigenmodes,
        eigenmodes_free,
        n_active,
        n_components,
    })
}

/// Scan all R-slices in parallel at a given temperature.
///
/// `homo_dimer`: apply exchange symmetrization U(vi,vj,ω) ↔ U(vj,vi,−ω).
pub fn diffusion_scan(
    table: &icotable::Table6DAdaptive<f32>,
    beta: f64,
    homo_dimer: bool,
) -> Vec<DiffusionResult> {
    if homo_dimer {
        info!("Applying exchange symmetrization (homo-dimer)");
    }
    // One set per mesh level (depends only on graph topology)
    let free_evals: std::collections::HashMap<usize, Vec<EigenMode>> = table
        .levels
        .iter()
        .map(|level| {
            let n_v = level.n_vertices;
            let n_omega = table.n_omega;
            let modes = free_eigenmodes_analytical(level, n_omega, NUM_EIGENMODES);
            info!(
                "Free-diffusion eigenmodes (n_v={n_v}, n_omega={n_omega}): λ₁={:.4e}",
                modes.first().map_or(0.0, |m| m.eigenvalue)
            );
            (n_v, modes)
        })
        .collect();

    let n_r = table.n_r;
    // Parallel over R-slices: folded spectrum uses only sparse matvec (low memory per thread)
    let mut results: Vec<DiffusionResult> = (0..n_r)
        .into_par_iter()
        .progress_count(n_r as u64)
        .filter_map(|ri| diffusion_at_r(table, ri, beta, &free_evals, homo_dimer).ok())
        .collect();

    results.sort_by(|a, b| a.r.partial_cmp(&b.r).unwrap());
    results
}

/// Export generator matrices as Matrix Market files for Python postprocessing.
///
/// Saves per R-slice:
/// - `generator_R{R}.mtx` — symmetric generator in Matrix Market COO format
/// - `coords_R{R}.csv` — compact index → (vi, vj, oi) mapping for eigenvector projection
///
/// Saves once:
/// - `metadata.csv` — R, n_active, n_total, n_v, n_omega, lambda1_free
/// - `icosphere.csv` — vertex adjacency list for the icosphere graph
///
/// The matrices can be loaded in Python with `scipy.io.mmread()`.
pub fn export_generator_matrices(
    table: &icotable::Table6DAdaptive<f32>,
    beta: f64,
    homo_dimer: bool,
    dir: &std::path::Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    use std::io::Write;

    let free_evals: std::collections::HashMap<usize, Vec<EigenMode>> = table
        .levels
        .iter()
        .map(|level| {
            let n_v = level.n_vertices;
            let n_omega = table.n_omega;
            let modes = free_eigenmodes_analytical(level, n_omega, NUM_EIGENMODES);
            (n_v, modes)
        })
        .collect();

    // Export icosphere neighbor lists (one file per distinct mesh level)
    for level in &table.levels {
        let path = dir.join(format!("icosphere_nv{}.csv", level.n_vertices));
        if !path.exists() {
            let mut f = std::fs::File::create(&path)?;
            writeln!(f, "vertex,neighbors")?;
            for (i, nbrs) in level.neighbors.iter().enumerate() {
                let nbr_str: Vec<String> = nbrs.iter().map(|n| n.to_string()).collect();
                writeln!(f, "{},{}", i, nbr_str.join(" "))?;
            }
            info!(
                "Exported icosphere adjacency (n_v={}) to {}",
                level.n_vertices,
                path.display()
            );
        }
    }

    let mut meta = std::fs::File::create(dir.join("metadata.csv"))?;
    writeln!(meta, "R,n_active,n_total,n_v,n_omega,lambda1_free,boltzmann_weight")?;

    for ri in 0..table.n_r {
        let r = table.rmin + ri as f64 * table.dr;
        let n_omega = table.n_omega;

        let Some((mut energies, level)) = table.energies_at_r(ri) else {
            continue;
        };
        let n_v = level.n_vertices;
        let n_total = n_v * n_v * n_omega;

        if homo_dimer {
            symmetrize_exchange(&mut energies, n_v, n_omega);
        }

        let active = ActiveStates::new(Some((&energies, beta)), n_v, n_omega);
        let mat = build_generator(&active, level, Some((&energies, beta)));
        let n = active.n_active;
        if n < 3 {
            continue;
        }

        let lambda1_free = free_evals
            .get(&n_v)
            .and_then(|m| m.first())
            .map_or(0.0, |m| m.eigenvalue);

        // Write coordinate mapping and per-state energies for eigenvector projection
        let coords_path = dir.join(format!("coords_R{:.1}.csv", r));
        let mut cf = std::fs::File::create(&coords_path)?;
        writeln!(cf, "compact,vi,vj,oi,energy")?;
        for (ci, &(vi, vj, oi)) in active.coords.iter().enumerate() {
            let full_idx = state_index(vi, vj, oi, n_v, n_omega);
            writeln!(cf, "{ci},{vi},{vj},{oi},{:.6e}", energies[full_idx])?;
        }

        // Write Matrix Market format (COO) from sparse CscMatrix
        let path = dir.join(format!("generator_R{:.1}.mtx", r));
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "%%MatrixMarket matrix coordinate real symmetric")?;
        // Count upper-triangle non-zeros
        let mut nnz = 0usize;
        for j in 0..n {
            let col = mat.col(j);
            for (&i, &v) in col.row_indices().iter().zip(col.values()) {
                if i <= j && v != 0.0 {
                    nnz += 1;
                }
            }
        }
        writeln!(f, "{n} {n} {nnz}")?;
        for j in 0..n {
            let col = mat.col(j);
            for (&i, &v) in col.row_indices().iter().zip(col.values()) {
                if i <= j && v != 0.0 {
                    writeln!(f, "{} {} {:.15e}", i + 1, j + 1, v)?; // 1-indexed
                }
            }
        }

        // ⟨exp(-βU)⟩ = (1/n_total) Σ exp(-βU_i) over all states (infinite U → 0 contribution)
        let u_min = energies
            .iter()
            .copied()
            .filter(|u| u.is_finite())
            .fold(f64::INFINITY, f64::min);
        let boltzmann_weight = if u_min.is_finite() {
            let sum: f64 = energies
                .iter()
                .filter(|u| u.is_finite())
                .map(|u| (-beta * (u - u_min)).exp())
                .sum();
            (sum / n_total as f64) * (-beta * u_min).exp()
        } else {
            0.0
        };

        writeln!(
            meta,
            "{r:.2},{n},{n_total},{n_v},{n_omega},{lambda1_free:.6e},{boltzmann_weight:.6e}"
        )?;
        info!(
            "R={r:.1} Å: exported {n}×{n} generator to {}",
            path.display()
        );
    }

    info!("Exported matrices to {}", dir.display());
    Ok(())
}

/// Cell-model result at one concentration.
#[derive(Debug, Clone)]
pub struct CellModelResult {
    /// Molar concentration (mol/L).
    pub molarity: f64,
    /// Cell radius (Å).
    pub r_cell: f64,
    /// PMF-weighted ⟨D_r/D_r⁰⟩.
    pub dr_normalized: f64,
    /// PMF-weighted ⟨D_A/D_A⁰⟩.
    pub dr_mol_a: f64,
    /// PMF-weighted ⟨D_B/D_B⁰⟩.
    pub dr_mol_b: f64,
    /// PMF-weighted ⟨D_ω/D_ω⁰⟩.
    pub dr_omega: f64,
    /// PMF-weighted ⟨separability⟩.
    pub separability: f64,
    /// PMF-weighted ⟨λ₁/λ₁_free⟩.
    pub spectral_ratio: f64,
}

/// Compute cell radius from molar concentration.
/// R_cell = (3 / (4π · c · N_A))^{1/3} in Å.
fn r_cell_from_molarity(molarity: f64) -> f64 {
    const AVOGADRO: f64 = physical_constants::AVOGADRO_CONSTANT;
    // c in mol/m³ = 1000 * molarity; result in m, convert to Å
    let c_m3 = molarity * 1000.0;
    let volume = 3.0 / (4.0 * std::f64::consts::PI * c_m3 * AVOGADRO);
    volume.cbrt() * 1e10
}

/// PMF-weighted average of `f(R)` within [R_min, R_cell].
///
/// weight(R) = boltzmann_weight(R) · R²  ∝  exp(-βw(R)) · R²
/// Beyond the table (R > R_max), assumes f(R) = 1 and w(R) = 0,
/// contributing an analytical (R³_cell - R³_max)/3 term.
fn cell_model_average(
    results: &[DiffusionResult],
    r_cell: f64,
    f: impl Fn(&DiffusionResult) -> f64,
) -> f64 {
    if results.is_empty() {
        return 1.0;
    }
    let dr = if results.len() > 1 {
        results[1].r - results[0].r
    } else {
        1.0
    };

    let mut num_tab = 0.0;
    let mut den_tab = 0.0;
    for res in results {
        if res.r > r_cell {
            break;
        }
        let w = res.boltzmann_weight * res.r * res.r;
        num_tab += f(res) * w;
        den_tab += w;
    }
    // Trapezoidal: sum × dr
    num_tab *= dr;
    den_tab *= dr;

    // Beyond table: f=1, w=0 → ∫R²dR = (R³_cell - R³_max)/3
    let r_max = results.last().map_or(0.0, |r| r.r);
    let free_vol = (r_cell.max(r_max).powi(3) - r_max.powi(3)) / 3.0;

    let den = den_tab + free_vol;
    if den > 0.0 {
        (num_tab + free_vol) / den
    } else {
        1.0
    }
}

/// Scan cell-model averaged diffusion over a range of concentrations.
pub fn cell_model_scan(results: &[DiffusionResult], molarities: &[f64]) -> Vec<CellModelResult> {
    let r_min = results.first().map_or(0.0, |r| r.r);

    molarities
        .iter()
        .map(|&c| {
            let r_cell = r_cell_from_molarity(c);
            if r_cell < r_min {
                warn!(
                    "c={c:.4e} M: R_cell={r_cell:.1} Å < table R_min={r_min:.1} Å, \
                     result is free-diffusion limit"
                );
            }
            let dr_normalized = cell_model_average(results, r_cell, |r| r.dr_normalized);
            let dr_mol_a = cell_model_average(results, r_cell, |r| r.dr_mol_a);
            let dr_mol_b = cell_model_average(results, r_cell, |r| r.dr_mol_b);
            let dr_omega = cell_model_average(results, r_cell, |r| r.dr_omega);
            let separability = cell_model_average(results, r_cell, |r| {
                let product = r.dr_mol_a * r.dr_mol_b * r.dr_omega;
                if product > 0.0 {
                    r.dr_normalized / product
                } else {
                    0.0
                }
            });
            let spectral_ratio = cell_model_average(results, r_cell, |r| {
                match (r.eigenmodes.first(), r.eigenmodes_free.first()) {
                    (Some(m), Some(mf)) if mf.eigenvalue > 0.0 => m.eigenvalue / mf.eigenvalue,
                    _ => 1.0,
                }
            });
            CellModelResult {
                molarity: c,
                r_cell,
                dr_normalized,
                dr_mol_a,
                dr_mol_b,
                dr_omega,
                separability,
                spectral_ratio,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use icotable::adaptive::MeshLevel;

    // --- Zwanzig tests ---

    #[test]
    fn test_zwanzig_uniform() {
        let energies = vec![0.0; 100];
        let (ratio, _, _) = zwanzig(&energies, 1.0).unwrap();
        assert_relative_eq!(ratio, 1.0, epsilon = 1e-10);
    }

    /// Validate against Lifson-Jackson: D/D₀ = 1/[I₀(βU₀)]²
    #[test]
    fn test_zwanzig_cosine() {
        let n = 1000;
        let u0 = 1.5;
        let beta = 1.0;

        let energies: Vec<f64> = (0..n)
            .map(|i| u0 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect();

        let (ratio, _, _) = zwanzig(&energies, beta).unwrap();
        let i0 = bessel_i0(beta * u0);
        let analytical = 1.0 / (i0 * i0);
        assert_relative_eq!(ratio, analytical, epsilon = 1e-4);
    }

    #[test]
    fn test_zwanzig_roughness_monotonic() {
        let n = 200;
        let beta = 1.0;
        let mut prev_ratio = 1.0;

        for &amplitude in &[0.5, 1.0, 2.0, 4.0] {
            let energies: Vec<f64> = (0..n)
                .map(|i| amplitude * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
                .collect();
            let (ratio, _, _) = zwanzig(&energies, beta).unwrap();
            assert!(ratio < prev_ratio, "D/D₀ should decrease with roughness");
            prev_ratio = ratio;
        }
    }

    #[test]
    fn test_zwanzig_high_temperature() {
        let n = 200;
        let energies: Vec<f64> = (0..n)
            .map(|i| 2.0 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect();
        let (ratio, _, _) = zwanzig(&energies, 0.01).unwrap();
        assert_relative_eq!(ratio, 1.0, epsilon = 1e-3);
    }

    // --- Marginal Zwanzig tests ---

    /// Uniform energy: all marginals = 1.
    #[test]
    fn test_marginal_uniform() {
        let n_v = 12;
        let n_omega = 8;
        let energies = vec![0.0; n_v * n_v * n_omega];
        let (da, db, dw) = marginal_zwanzig(&energies, n_v, n_omega, 1.0);
        assert_relative_eq!(da, 1.0, epsilon = 1e-10);
        assert_relative_eq!(db, 1.0, epsilon = 1e-10);
        assert_relative_eq!(dw, 1.0, epsilon = 1e-10);
    }

    /// Potential depending only on vi: D_A < 1, D_B = 1, D_ω = 1.
    #[test]
    fn test_marginal_vi_only() {
        let level = MeshLevel::new(0);
        let n_v = level.n_vertices;
        let n_omega = 8;
        let u0 = 2.0;
        let beta = 1.0;

        // U = u0 * cos(θ_A) where θ_A is the polar angle of vertex vi
        let energies: Vec<f64> = (0..n_v * n_v * n_omega)
            .map(|idx| {
                let vi = idx / (n_v * n_omega);
                u0 * level.vertices[vi][2] // z-component ≈ cos(θ)
            })
            .collect();

        let (da, db, dw) = marginal_zwanzig(&energies, n_v, n_omega, beta);
        assert!(da < 0.99, "D_A should be hindered, got {da}");
        assert_relative_eq!(db, 1.0, epsilon = 1e-6);
        assert_relative_eq!(dw, 1.0, epsilon = 1e-6);
    }

    /// Dihedral-only cosine: D_A = D_B = 1, D_ω = 1/[I₀(βU₀)]².
    #[test]
    fn test_marginal_dihedral_only() {
        let n_v = 12;
        let n_omega = 64;
        let u0 = 1.5;
        let beta = 1.0;

        let energies: Vec<f64> = (0..n_v * n_v * n_omega)
            .map(|idx| {
                let oi = idx % n_omega;
                u0 * (2.0 * std::f64::consts::PI * oi as f64 / n_omega as f64).cos()
            })
            .collect();

        let (da, db, dw) = marginal_zwanzig(&energies, n_v, n_omega, beta);
        let i0 = bessel_i0(beta * u0);
        let analytical = 1.0 / (i0 * i0);

        assert_relative_eq!(da, 1.0, epsilon = 1e-6);
        assert_relative_eq!(db, 1.0, epsilon = 1e-6);
        assert_relative_eq!(dw, analytical, epsilon = 0.01);
    }

    // --- Spectral eigenvalue tests ---

    // --- Eigenmode / projection tests ---

    /// Uniform energy: eigenvalues match between potential and free.
    #[test]
    fn test_eigenmodes_uniform() {
        let level = MeshLevel::new(0);
        let n_v = level.n_vertices;
        let n_omega = 4;
        let n_states = n_v * n_v * n_omega;

        let energies = vec![0.0; n_states];
        let active = ActiveStates::new(Some((&energies, 1.0)), n_v, n_omega);
        let gen = build_generator(&active, &level, Some((&energies, 1.0)));
        let modes = dense_eigenmodes(&gen, &active, 3, &level);

        let free_active = ActiveStates::new(None, n_v, n_omega);
        let free_gen = build_generator(&free_active, &level, None);
        let free_modes = dense_eigenmodes(&free_gen, &free_active, 3, &level);

        assert!(!modes.is_empty());
        for (m, mf) in modes.iter().zip(&free_modes) {
            assert_relative_eq!(m.eigenvalue / mf.eigenvalue, 1.0, epsilon = 1e-6);
        }
    }

    /// Free diffusion: fractions always sum to 1.
    #[test]
    fn test_projection_normalization() {
        let level = MeshLevel::new(0);
        let n_v = level.n_vertices;
        let n_omega = 8;
        let n_states = n_v * n_v * n_omega;

        let free_active = ActiveStates::new(None, n_v, n_omega);
        let free_gen = build_generator(&free_active, &level, None);
        let modes = dense_eigenmodes(&free_gen, &free_active, 5, &level);

        for m in &modes {
            let sum = m.frac_mol_a + m.frac_mol_b + m.frac_omega;
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    /// vi-only potential: fractions sum to 1 even with asymmetric landscape.
    #[test]
    fn test_projection_with_potential() {
        let level = MeshLevel::new(0);
        let n_v = level.n_vertices;
        let n_omega = 4;
        let n_states = n_v * n_v * n_omega;

        let energies: Vec<f64> = (0..n_states)
            .map(|idx| {
                let vi = idx / (n_v * n_omega);
                2.0 * level.vertices[vi][2]
            })
            .collect();

        let active = ActiveStates::new(Some((&energies, 1.0)), n_v, n_omega);
        let gen = build_generator(&active, &level, Some((&energies, 1.0)));
        let modes = dense_eigenmodes(&gen, &active, 5, &level);

        for m in &modes {
            let sum = m.frac_mol_a + m.frac_mol_b + m.frac_omega;
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    /// Icosphere eigenvalues: sorted nondecreasing, first ≈ 0, rest positive.
    #[test]
    fn test_icosphere_eigenvalues() {
        let level = MeshLevel::new(0);
        let evals = icosphere_eigenvalues(&level);

        assert_eq!(evals.len(), 12);
        assert_relative_eq!(evals[0], 0.0, epsilon = 1e-10);
        // All positive and sorted
        for i in 1..evals.len() {
            assert!(evals[i] > 0.0);
            assert!(evals[i] >= evals[i - 1] - 1e-10);
        }
        // Spectral gap should be reasonable (> 0, < max degree)
        assert!(evals[1] > 0.1);
        assert!(evals[1] < 12.0);
    }

    /// Free-diffusion analytical eigenmodes: spectral gap matches ring eigenvalue
    /// when the ring has fewer bins than icosphere vertices.
    #[test]
    fn test_free_eigenmodes_analytical() {
        let level = MeshLevel::new(0); // 12 vertices
        let n_omega = 8;
        let modes = free_eigenmodes_analytical(&level, n_omega, 3);

        // Ring spectral gap: 2(1 - cos(2π/8)) = 2(1 - cos(π/4)) = 2(1 - √2/2) ≈ 0.5858
        let ring_gap = 2.0 * (1.0 - (2.0 * std::f64::consts::PI / n_omega as f64).cos());
        // Icosphere spectral gap: 5 - √5 ≈ 2.764
        // Product spectral gap = min(ring_gap, ico_gap) = ring_gap ≈ 0.5858
        assert!(!modes.is_empty());
        assert_relative_eq!(modes[0].eigenvalue, ring_gap, epsilon = 1e-6);
        // First mode should be dihedral-dominated (f_ω ≈ 1)
        assert!(modes[0].frac_omega > 0.9);
    }

    /// Sparse Lanczos matches dense eigensolver on free diffusion.
    #[test]
    fn test_sparse_eigenmodes_matches_dense() {
        let level = MeshLevel::new(0);
        let n_v = level.n_vertices;
        let n_omega = 4;

        let active = ActiveStates::new(None, n_v, n_omega);
        let gen = build_generator(&active, &level, None);
        let n_components = count_components(&active, &level);
        let dense_modes = dense_eigenmodes(&gen, &active, 3, &level);
        let sparse_modes = sparse_eigenmodes(&gen, &active, 3, &level, n_components);

        assert_eq!(dense_modes.len(), sparse_modes.len());
        for (dm, sm) in dense_modes.iter().zip(&sparse_modes) {
            assert_relative_eq!(dm.eigenvalue, sm.eigenvalue, epsilon = 1e-4);
        }
    }

    /// Connected components are counted correctly.
    #[test]
    fn test_count_components() {
        let level = MeshLevel::new(0);
        let n_v = level.n_vertices;
        let n_omega = 4;

        // Fully connected free diffusion: 1 component
        let active = ActiveStates::new(None, n_v, n_omega);
        assert_eq!(count_components(&active, &level), 1);
    }

    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..50 {
            term *= (x / (2.0 * k as f64)).powi(2);
            sum += term;
            if term < 1e-15 * sum {
                break;
            }
        }
        sum
    }
}

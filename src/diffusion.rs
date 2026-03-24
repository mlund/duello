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
use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;
use sprs::{CsMat, TriMat};

/// Result of diffusion analysis at one R-slice.
#[derive(Debug, Clone)]
pub struct DiffusionResult {
    /// Radial distance (Å).
    pub r: f64,
    /// Normalized diffusion D_r/D_r⁰ (Zwanzig formula).
    pub dr_normalized: f64,
    /// ⟨exp(-βU)⟩ spatial average.
    pub avg_exp_minus: f64,
    /// ⟨exp(+βU)⟩ spatial average.
    pub avg_exp_plus: f64,
    /// Spectral gap of the symmetrized generator (if computed).
    pub spectral_gap: Option<f64>,
    /// Spectral gap for free diffusion (if computed).
    pub spectral_gap_free: Option<f64>,
    /// Number of active states (finite-energy grid points).
    pub n_active: usize,
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

/// Build a sparse CSR generator matrix over the (vi, vj, oi) state space.
///
/// When `potential` is `Some((energies, beta))`, builds the symmetrized Smoluchowski
/// generator: off-diagonals are 1 for finite-energy neighbors, and the diagonal
/// exit rate uses exp(-β ΔU / 2). States with non-finite energy are skipped.
///
/// When `potential` is `None`, builds the free-diffusion generator where every
/// state is active and the exit rate equals the neighbor count.
fn build_sparse_generator(
    level: &MeshLevel,
    n_omega: usize,
    potential: Option<(&[f64], f64)>,
) -> CsMat<f64> {
    let n_v = level.n_vertices;
    let n_states = n_v * n_v * n_omega;

    // Shift energies so all finite values are ≥ 0, preventing Boltzmann overflow
    let u_min = potential.map(|(energies, _)| {
        energies
            .iter()
            .copied()
            .filter(|u| u.is_finite())
            .fold(f64::INFINITY, f64::min)
    });

    let mut triplets = TriMat::new((n_states, n_states));

    for vi in 0..n_v {
        for vj in 0..n_v {
            for oi in 0..n_omega {
                let idx_i = state_index(vi, vj, oi, n_v, n_omega);

                let u_i = match potential {
                    Some((energies, _)) => {
                        if !energies[idx_i].is_finite() {
                            continue;
                        }
                        energies[idx_i] - u_min.unwrap()
                    }
                    None => 0.0,
                };

                let mut exit_rate = 0.0;
                for_each_neighbor(vi, vj, oi, level, n_v, n_omega, |idx_j| {
                    if let Some((energies, beta)) = potential {
                        if !energies[idx_j].is_finite() {
                            return;
                        }
                        let half_delta = beta * (energies[idx_j] - u_min.unwrap() - u_i) / 2.0;
                        if half_delta <= BOLTZMANN_OVERFLOW_GUARD {
                            exit_rate += (-half_delta).exp();
                        }
                    } else {
                        exit_rate += 1.0;
                    }
                    // Off-diagonal of L_sym is always 1 after Smoluchowski symmetrization
                    triplets.add_triplet(idx_i, idx_j, 1.0);
                });
                // Row sum = 0 ensures probability conservation
                triplets.add_triplet(idx_i, idx_i, -exit_rate);
            }
        }
    }

    triplets.to_csr()
}

/// Sparse matrix-vector product: y = A * x
fn sparse_matvec(a: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.rows()];
    sprs::prod::mul_acc_mat_vec_csr(a.view(), x, &mut y);
    y
}

/// Dense spectral gap for small matrices (convert sparse → dense).
fn dense_spectral_gap(matrix: &CsMat<f64>, n: usize) -> f64 {
    let mut dense = DMatrix::zeros(n, n);
    for (row_idx, row) in matrix.outer_iterator().enumerate() {
        for (col_idx, &val) in row.iter() {
            dense[(row_idx, col_idx)] = val;
        }
    }
    let eigen = SymmetricEigen::new(dense);
    let mut evals: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    // Descending sort: evals[0] ≈ 0 (equilibrium), evals[1] = spectral gap
    evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if evals.len() < 2 {
        0.0
    } else {
        -evals[1]
    }
}

/// Threshold for switching from dense to Lanczos eigensolver.
const DENSE_THRESHOLD: usize = 5_000;

/// Maximum Lanczos basis vectors. Memory per thread ~ n_states × m × 8B.
/// For 143K states: 143K × 60 × 8B ≈ 69 MB per thread.
const MAX_LANCZOS_VECTORS: usize = 60;

/// Lanczos convergence tolerance for the residual norm.
const LANCZOS_TOLERANCE: f64 = 1e-14;

/// Boltzmann exponent cutoff: exp(-x) for x > this is treated as zero to avoid overflow.
const BOLTZMANN_OVERFLOW_GUARD: f64 = 50.0;

/// Skip spectral gap when fewer than this many active states exist.
const MIN_ACTIVE_STATES: usize = 100;

/// Skip spectral gap when active fraction is below this threshold.
const MIN_ACTIVE_FRACTION: f64 = 0.01;

/// Reject spectral gaps above this as numerically unstable.
const MAX_PLAUSIBLE_SPECTRAL_GAP: f64 = 1e10;

/// Find the spectral gap of a sparse symmetric matrix.
///
/// Dense eigensolver for small matrices (exact, O(n³) but n is small).
/// Lanczos with full reorthogonalization for larger ones (O(m²·n)).
fn sparse_spectral_gap(matrix: &CsMat<f64>, n_states: usize) -> f64 {
    if n_states <= DENSE_THRESHOLD {
        return dense_spectral_gap(matrix, n_states);
    }
    lanczos_spectral_gap(matrix, n_states)
}

/// Lanczos iteration with full reorthogonalization for the spectral gap.
///
/// Full reorthogonalization prevents loss of orthogonality that causes
/// spurious eigenvalues in the standard Lanczos algorithm.
fn lanczos_spectral_gap(matrix: &CsMat<f64>, n_states: usize) -> f64 {
    let m = MAX_LANCZOS_VECTORS.min(n_states);


    // Deterministic pseudo-random start to ensure reproducible eigenvalues
    let mut v: Vec<f64> = (0..n_states)
        .map(|i| ((i * 7 + 13) % 97) as f64 / 97.0 - 0.5)
        .collect();
    let norm = vec_norm(&v);
    v.iter_mut().for_each(|x| *x /= norm);

    let mut alpha = Vec::with_capacity(m);
    let mut beta_vec = Vec::with_capacity(m);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(m);
    basis.push(v.clone());

    for _j in 0..m {
        let mut w = sparse_matvec(matrix, basis.last().unwrap());

        let a_j: f64 = w.iter().zip(basis.last().unwrap()).map(|(a, b)| a * b).sum();
        alpha.push(a_j);

        let v_cur = basis.last().unwrap();
        for k in 0..w.len() {
            w[k] -= a_j * v_cur[k];
        }
        if basis.len() >= 2 {
            let beta_prev = *beta_vec.last().unwrap();
            let v_prev = &basis[basis.len() - 2];
            for k in 0..w.len() {
                w[k] -= beta_prev * v_prev[k];
            }
        }

        // Prevent spurious eigenvalues from loss of orthogonality
        for v_old in &basis {
            let proj: f64 = w.iter().zip(v_old).map(|(a, b)| a * b).sum();
            for k in 0..w.len() {
                w[k] -= proj * v_old[k];
            }
        }

        let beta_j = vec_norm(&w);
        if beta_j < LANCZOS_TOLERANCE {
            break;
        }
        beta_vec.push(beta_j);

        w.iter_mut().for_each(|x| *x /= beta_j);
        basis.push(w);
    }

    let k = alpha.len();
    let mut tridiag = DMatrix::zeros(k, k);
    for i in 0..k {
        tridiag[(i, i)] = alpha[i];
        if i + 1 < k && i < beta_vec.len() {
            tridiag[(i, i + 1)] = beta_vec[i];
            tridiag[(i + 1, i)] = beta_vec[i];
        }
    }
    let eigen = SymmetricEigen::new(tridiag);
    let mut evals: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    // Descending sort; NaN from Lanczos instability treated as most negative
    evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    if evals.len() < 2 || !evals[1].is_finite() {
        0.0
    } else {
        -evals[1]
    }
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Zwanzig formula: D/D₀ = 1 / [⟨exp(-βU)⟩ x ⟨exp(βU)⟩]
///
/// Computed in log-space with streaming sums to avoid intermediate allocations.
fn zwanzig(energies: &[f64], beta: f64) -> Option<(f64, f64, f64, usize)> {
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
    Some((d_ratio, log_avg_minus.exp(), log_avg_plus.exp(), n))
}

/// Compute diffusion analysis at a single R-slice.
///
/// `free_gaps` maps n_vertices → pre-computed free-diffusion spectral gap.
fn diffusion_at_r(
    table: &icotable::Table6DAdaptive<f32>,
    ri: usize,
    beta: f64,
    free_gaps: &std::collections::HashMap<usize, f64>,
) -> Result<DiffusionResult> {
    let r = table.rmin + ri as f64 * table.dr;
    let n_omega = table.n_omega;

    let (energies, level) = table
        .energies_at_r(ri)
        .ok_or_else(|| anyhow::anyhow!("R-slice {ri} is fully repulsive or out of range"))?;

    let n_v = level.n_vertices;
    let n_total = n_v * n_v * n_omega;

    let (dr_normalized, avg_exp_minus, avg_exp_plus, n_active) =
        zwanzig(&energies, beta).ok_or_else(|| anyhow::anyhow!("No finite energies at R={r:.1}"))?;

    let spectral_gap =
        if n_active > MIN_ACTIVE_STATES && (n_active as f64 / n_total as f64) > MIN_ACTIVE_FRACTION
        {
            let gen = build_sparse_generator(level, n_omega, Some((&energies, beta)));
            let gap = sparse_spectral_gap(&gen, n_total);
            if gap.is_finite() && gap > 0.0 && gap < MAX_PLAUSIBLE_SPECTRAL_GAP {
                Some(gap)
            } else {
                debug!("R={r:.1} Å: rejecting unstable spectral gap {gap:.2e}");
                None
            }
        } else {
            debug!("R={r:.1} Å: skipping spectral gap (n_active={n_active}/{n_total})");
            None
        };

    debug!(
        "R={r:.1} Å: D_r/D_r⁰={dr_normalized:.6}, λ₁={}",
        spectral_gap.map_or("N/A".into(), |g| format!("{g:.4e}"))
    );

    Ok(DiffusionResult {
        r,
        dr_normalized,
        avg_exp_minus,
        avg_exp_plus,
        spectral_gap,
        spectral_gap_free: free_gaps.get(&level.n_vertices).copied(),
        n_active,
    })
}

/// Scan all R-slices in parallel at a given temperature.
pub fn diffusion_scan(
    table: &icotable::Table6DAdaptive<f32>,
    beta: f64,
) -> Vec<DiffusionResult> {
    // Free-diffusion gap depends only on graph topology — one per mesh level
    let free_gaps: std::collections::HashMap<usize, f64> = table
        .levels
        .iter()
        .map(|level| {
            let n_omega = table.n_omega;
            let n_states = level.n_vertices * level.n_vertices * n_omega;
            info!(
                "Computing free-diffusion spectral gap (n_v={}, n_omega={n_omega})",
                level.n_vertices
            );
            let free_gen = build_sparse_generator(level, n_omega, None);
            (level.n_vertices, sparse_spectral_gap(&free_gen, n_states))
        })
        .collect();

    let n_r = table.n_r;
    let mut results: Vec<DiffusionResult> = (0..n_r)
        .into_par_iter()
        .progress_count(n_r as u64)
        .filter_map(|ri| diffusion_at_r(table, ri, beta, &free_gaps).ok())
        .collect();

    results.sort_by(|a, b| a.r.partial_cmp(&b.r).unwrap());
    results
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
        let (ratio, _, _, _) = zwanzig(&energies, 1.0).unwrap();
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

        let (ratio, _, _, _) = zwanzig(&energies, beta).unwrap();
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
            let (ratio, _, _, _) = zwanzig(&energies, beta).unwrap();
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
        let (ratio, _, _, _) = zwanzig(&energies, 0.01).unwrap();
        assert_relative_eq!(ratio, 1.0, epsilon = 1e-3);
    }

    // --- Spectral gap tests ---

    /// Uniform energy: sparse spectral gap ratio = 1.
    #[test]
    fn test_spectral_uniform_sparse() {
        let level = MeshLevel::new(0); // 12 vertices
        let n_v = level.n_vertices;
        let n_omega = 4;
        let n_states = n_v * n_v * n_omega;

        let energies = vec![0.0; n_states];
        let gen = build_sparse_generator(&level, n_omega, Some((&energies, 1.0)));
        let gap = sparse_spectral_gap(&gen, n_states);

        let free_gen = build_sparse_generator(&level, n_omega, None);
        let gap_free = sparse_spectral_gap(&free_gen, n_states);

        assert_relative_eq!(gap / gap_free, 1.0, epsilon = 1e-6);
    }

    /// Verify sparse Lanczos agrees with dense eigensolver on a small ring.
    #[test]
    fn test_sparse_vs_dense_ring() {
        let n = 32;
        // Dense free ring
        let mut dense = DMatrix::zeros(n, n);
        let mut triplets = TriMat::new((n, n));
        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = (i + 1) % n;
            dense[(i, prev)] = 1.0;
            dense[(i, next)] = 1.0;
            dense[(i, i)] = -2.0;
            triplets.add_triplet(i, prev, 1.0);
            triplets.add_triplet(i, next, 1.0);
            triplets.add_triplet(i, i, -2.0);
        }
        let sparse = triplets.to_csr();

        let dense_eigen = SymmetricEigen::new(dense);
        let mut dense_evals: Vec<f64> = dense_eigen.eigenvalues.iter().copied().collect();
        dense_evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let dense_gap = -dense_evals[1];

        let sparse_gap = sparse_spectral_gap(&sparse, n);
        assert_relative_eq!(sparse_gap, dense_gap, epsilon = 1e-8);
    }

    /// Spectral gap of a ring should agree with the analytical result.
    #[test]
    fn test_spectral_gap_ring_analytical() {
        let n = 128;
        let mut triplets = TriMat::new((n, n));
        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = (i + 1) % n;
            triplets.add_triplet(i, prev, 1.0);
            triplets.add_triplet(i, next, 1.0);
            triplets.add_triplet(i, i, -2.0);
        }
        let sparse = triplets.to_csr();

        // Analytical: λ_1 = -2(1 - cos(2π/n))
        let analytical_gap = 2.0 * (1.0 - (2.0 * std::f64::consts::PI / n as f64).cos());
        let gap = sparse_spectral_gap(&sparse, n);

        assert_relative_eq!(gap, analytical_gap, epsilon = 1e-8);
    }

    /// Lanczos with a potential should give the same result as dense eigensolver.
    #[test]
    fn test_lanczos_vs_dense_with_potential() {
        let n = 64;
        let u0 = 1.5;
        let beta = 1.0;
        let energies: Vec<f64> = (0..n)
            .map(|i| u0 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect();
        let u_min = energies.iter().copied().fold(f64::INFINITY, f64::min);

        let mut triplets = TriMat::new((n, n));
        for i in 0..n {
            let u_i = energies[i] - u_min;
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = (i + 1) % n;
            triplets.add_triplet(i, prev, 1.0);
            triplets.add_triplet(i, next, 1.0);
            let mut exit = 0.0;
            for &j in &[prev, next] {
                exit += (-beta * (energies[j] - u_min - u_i) / 2.0).exp();
            }
            triplets.add_triplet(i, i, -exit);
        }
        let sparse = triplets.to_csr();

        let dense_gap = dense_spectral_gap(&sparse, n);
        let lanczos_gap = lanczos_spectral_gap(&sparse, n);

        assert_relative_eq!(lanczos_gap, dense_gap, epsilon = 1e-8);
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

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
use lanczos::{Hermitian, Order};
use nalgebra::{DMatrix, SymmetricEigen};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
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
    n_v: usize,
    n_omega: usize,
) -> (f64, f64, f64) {
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

    let norm_a = if edges_a > 0 { var_a / edges_a as f64 } else { 0.0 };
    let norm_b = if edges_b > 0 { var_b / edges_b as f64 } else { 0.0 };
    let norm_omega = if edges_omega > 0 { var_omega / edges_omega as f64 } else { 0.0 };

    let total = norm_a + norm_b + norm_omega;
    if total < 1e-30 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }
    (norm_a / total, norm_b / total, norm_omega / total)
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
    fn new(
        energies: Option<(&[f64], f64)>,
        n_v: usize,
        n_omega: usize,
    ) -> Self {
        let n_states = n_v * n_v * n_omega;
        let mut compact = vec![usize::MAX; n_states];
        let mut coords = Vec::new();

        let u_min = energies.map(|(e, _)| {
            e.iter().copied().filter(|u| u.is_finite()).fold(f64::INFINITY, f64::min)
        });

        for vi in 0..n_v {
            for vj in 0..n_v {
                for oi in 0..n_omega {
                    let idx = state_index(vi, vj, oi, n_v, n_omega);
                    let active = match energies {
                        Some((e, beta)) => {
                            e[idx].is_finite()
                                && beta * (e[idx] - u_min.unwrap()) <= SPECTRAL_ENERGY_THRESHOLD
                        }
                        None => true,
                    };
                    if active {
                        compact[idx] = coords.len();
                        coords.push((vi, vj, oi));
                    }
                }
            }
        }

        let n_active = coords.len();
        Self { compact, coords, n_active, n_v, n_omega }
    }
}

/// Build a sparse CSR generator over the compact active state space.
///
/// Returns the generator matrix (n_active × n_active) and the index mapping.
/// For potential: Smoluchowski rate with energy threshold filtering.
/// For free diffusion (None): all states active.
fn build_sparse_generator(
    level: &MeshLevel,
    n_omega: usize,
    potential: Option<(&[f64], f64)>,
) -> (CsrMatrix<f64>, ActiveStates) {
    let n_v = level.n_vertices;

    let active = ActiveStates::new(potential, n_v, n_omega);
    let n = active.n_active;
    let mut coo = CooMatrix::new(n, n);

    let u_min = potential.map(|(energies, _)| {
        energies
            .iter()
            .copied()
            .filter(|u| u.is_finite())
            .fold(f64::INFINITY, f64::min)
    });

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

    (CsrMatrix::from(&coo), active)
}

/// Extract first `k` non-trivial eigenmodes.
///
/// Dense eigensolver for n_active ≤ DENSE_THRESHOLD (exact, reliable).
/// `lanczos` crate with `Order::Largest` for larger matrices (targets near-zero eigenvalues).
const DENSE_THRESHOLD: usize = 10_000;

fn spectral_eigenmodes(
    matrix: &CsrMatrix<f64>,
    active: &ActiveStates,
    k: usize,
    level: &MeshLevel,
) -> Vec<EigenMode> {
    let n = active.n_active;
    if n < 3 {
        return Vec::new();
    }

    if n <= DENSE_THRESHOLD {
        return dense_eigenmodes(matrix, active, k, level);
    }

    // Lanczos via `lanczos` crate for large matrices
    let iterations = n.min(300);
    let eigen = matrix.eigsh(iterations, Order::Largest);

    // Convert lanczos crate types (nalgebra 0.33) to our types (0.34)
    let evals: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    evals
        .iter()
        .enumerate()
        .skip(1)
        .take(k)
        .filter(|(_, &e)| e.is_finite() && e < 0.0)
        .map(|(i, &e)| {
            let eigvec: Vec<f64> = eigen.eigenvectors.column(i).iter().copied().collect();
            let (fa, fb, fw) =
                coordinate_projection(&eigvec, active, level, active.n_v, active.n_omega);
            EigenMode {
                eigenvalue: -e,
                frac_mol_a: fa,
                frac_mol_b: fb,
                frac_omega: fw,
            }
        })
        .collect()
}

/// Dense eigensolver: convert sparse → dense, use nalgebra SymmetricEigen.
fn dense_eigenmodes(
    matrix: &CsrMatrix<f64>,
    active: &ActiveStates,
    k: usize,
    level: &MeshLevel,
) -> Vec<EigenMode> {
    let n = active.n_active;
    let mut dense = DMatrix::zeros(n, n);
    for (i, j, &v) in matrix.triplet_iter() {
        dense[(i, j)] = v;
    }
    let eigen = SymmetricEigen::new(dense);
    // Sort descending (largest = λ₀ ≈ 0 first)
    let mut indices: Vec<usize> = (0..n).collect();
    let evals = &eigen.eigenvalues;
    indices.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
        .iter()
        .skip(1)
        .take(k)
        .filter(|&&i| evals[i].is_finite() && evals[i] < 0.0)
        .map(|&i| {
            let eigvec: Vec<f64> = eigen.eigenvectors.column(i).iter().copied().collect();
            let (fa, fb, fw) =
                coordinate_projection(&eigvec, active, level, active.n_v, active.n_omega);
            EigenMode {
                eigenvalue: -evals[i],
                frac_mol_a: fa,
                frac_mol_b: fb,
                frac_omega: fw,
            }
        })
        .collect()
}

/// Boltzmann exponent cutoff: exp(-x) for |x| > this is treated as zero to avoid overflow.
const BOLTZMANN_OVERFLOW_GUARD: f64 = 50.0;

/// Energy threshold for spectral analysis: states with β(U - U_min) above this
/// are excluded from the generator to improve conditioning.
const SPECTRAL_ENERGY_THRESHOLD: f64 = 30.0;

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

/// Compute per-coordinate Zwanzig by marginalizing over the other coordinates.
///
/// For each coordinate (vi, vj, oi), integrates out the other two to get a
/// 1D potential of mean force, then applies Zwanzig independently.
/// Returns (D_A/D⁰, D_B/D⁰, D_ω/D⁰).
///
/// Note: marginalization smooths cross-correlations, so D_A × D_B × D_ω ≥ D/D⁰.
/// The ratio D/D⁰ / (D_A × D_B × D_ω) measures coordinate separability.
fn marginal_zwanzig(energies: &[f64], n_v: usize, n_omega: usize, beta: f64) -> (f64, f64, f64) {
    // w_A(vi) = -kT ln Σ_{vj,oi} exp(-βU) = -(1/β) · log_sum_exp over (vj, oi)
    let pmf_a: Vec<f64> = (0..n_v)
        .map(|vi| {
            let mut max_val = f64::NEG_INFINITY;
            let mut terms = Vec::new();
            for vj in 0..n_v {
                for oi in 0..n_omega {
                    let u = energies[state_index(vi, vj, oi, n_v, n_omega)];
                    if u.is_finite() {
                        let x = -beta * u;
                        max_val = max_val.max(x);
                        terms.push(x);
                    }
                }
            }
            if terms.is_empty() {
                return f64::INFINITY;
            }
            // w_A = -(1/β) · (max + ln Σ exp(x - max))
            let lse = max_val + terms.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln();
            -lse / beta
        })
        .collect();

    // w_B(vj) = -kT ln Σ_{vi,oi} exp(-βU)
    let pmf_b: Vec<f64> = (0..n_v)
        .map(|vj| {
            let mut max_val = f64::NEG_INFINITY;
            let mut terms = Vec::new();
            for vi in 0..n_v {
                for oi in 0..n_omega {
                    let u = energies[state_index(vi, vj, oi, n_v, n_omega)];
                    if u.is_finite() {
                        let x = -beta * u;
                        max_val = max_val.max(x);
                        terms.push(x);
                    }
                }
            }
            if terms.is_empty() {
                return f64::INFINITY;
            }
            let lse = max_val + terms.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln();
            -lse / beta
        })
        .collect();

    // w_ω(oi) = -kT ln Σ_{vi,vj} exp(-βU)
    let pmf_omega: Vec<f64> = (0..n_omega)
        .map(|oi| {
            let mut max_val = f64::NEG_INFINITY;
            let mut terms = Vec::new();
            for vi in 0..n_v {
                for vj in 0..n_v {
                    let u = energies[state_index(vi, vj, oi, n_v, n_omega)];
                    if u.is_finite() {
                        let x = -beta * u;
                        max_val = max_val.max(x);
                        terms.push(x);
                    }
                }
            }
            if terms.is_empty() {
                return f64::INFINITY;
            }
            let lse = max_val + terms.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln();
            -lse / beta
        })
        .collect();

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

    let eigenmodes = if n_active > MIN_ACTIVE_STATES
        && (n_active as f64 / n_total as f64) > MIN_ACTIVE_FRACTION
    {
        let (gen, active) = build_sparse_generator(level, n_omega, Some((&energies, beta)));
        debug!("R={r:.1} Å: spectral active states: {}/{n_total}", active.n_active);
        let modes = spectral_eigenmodes(&gen, &active, NUM_EIGENMODES, level);
        // Reject if largest eigenvalue exceeds free value by too much (Lanczos instability)
        let max_plausible = eigenmodes_free
            .first()
            .map_or(f64::INFINITY, |m| m.eigenvalue * MAX_EIGENVALUE_RATIO);
        if modes
            .first()
            .is_some_and(|m| m.eigenvalue < max_plausible)
        {
            modes
        } else {
            debug!(
                "R={r:.1} Å: rejecting eigenmodes (λ₁={:.2e}, max={max_plausible:.2e})",
                modes.first().map_or(0.0, |m| m.eigenvalue)
            );
            Vec::new()
        }
    } else {
        debug!("R={r:.1} Å: skipping eigenmodes (n_active={n_active}/{n_total})");
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
    // Free-diffusion eigenmodes depend only on graph topology — one set per mesh level
    let free_evals: std::collections::HashMap<usize, Vec<EigenMode>> = table
        .levels
        .iter()
        .map(|level| {
            let n_v = level.n_vertices;
            let n_omega = table.n_omega;
            let n_states = n_v * n_v * n_omega;
            info!("Computing free-diffusion eigenmodes (n_v={n_v}, n_omega={n_omega})");
            let (free_gen, free_active) = build_sparse_generator(level, n_omega, None);
            (
                n_v,
                spectral_eigenmodes(&free_gen, &free_active, NUM_EIGENMODES, level),
            )
        })
        .collect();

    let n_r = table.n_r;
    let mut results: Vec<DiffusionResult> = (0..n_r)
        .into_par_iter()
        .progress_count(n_r as u64)
        .filter_map(|ri| diffusion_at_r(table, ri, beta, &free_evals, homo_dimer).ok())
        .collect();

    results.sort_by(|a, b| a.r.partial_cmp(&b.r).unwrap());
    results
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
        let (gen, active) = build_sparse_generator(&level, n_omega, Some((&energies, 1.0)));
        let modes = spectral_eigenmodes(&gen, &active, 3, &level);

        let (free_gen, free_active) = build_sparse_generator(&level, n_omega, None);
        let free_modes = spectral_eigenmodes(&free_gen, &free_active, 3, &level);

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

        let (free_gen, free_active) = build_sparse_generator(&level, n_omega, None);
        let modes = spectral_eigenmodes(&free_gen, &free_active, 5, &level);

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

        let (gen, active) = build_sparse_generator(&level, n_omega, Some((&energies, 1.0)));
        let modes = spectral_eigenmodes(&gen, &active, 5, &level);

        for m in &modes {
            let sum = m.frac_mol_a + m.frac_mol_b + m.frac_omega;
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
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

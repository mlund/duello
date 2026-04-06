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

use crate::{
    backend::{EnergyBackend, PoseParams},
    structure::Structure,
    Sample, Vector3,
};
use icotable::{AdaptiveBuilder, Table6DAdaptive};
use std::f64::consts::PI;

#[cfg(feature = "cli")]
use icotable::{TableMetadata, TailCorrectionTerm};
#[cfg(feature = "cli")]
use std::path::PathBuf;

/// Configuration for a 6D icoscan (I/O-free parameters only)
pub struct ScanParams {
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    pub temperature: f64,
    /// Net charges of the two molecules in elementary charges.
    pub charges: [f64; 2],
    /// Dipole moment magnitudes in e·Å.
    pub dipole_moments: [f64; 2],
    /// Debye screening parameter (1/Å), if known.
    pub kappa: Option<f64>,
    /// Relative permittivity of the medium.
    pub permittivity: f64,
    /// Maximum icosphere subdivision level (0=12 vertices, 3=162 vertices).
    pub max_n_div: usize,
    /// Angular gradient threshold for adaptive resolution reduction.
    pub gradient_threshold: f64,
    /// True when mol1 and mol2 are identical (homo-dimer).
    pub homo_dimer: bool,
}

/// CLI-specific scan configuration extending ScanParams with file paths.
#[cfg(feature = "cli")]
pub struct ScanConfig {
    pub params: ScanParams,
    pub pmf_file: PathBuf,
    /// Save binary 6D table for Faunus lookup (.gz suffix enables gzip compression)
    pub save_table: Option<PathBuf>,
}

/// Results from a completed scan (no I/O).
pub struct ScanResult {
    pub samples: Vec<(Vector3, Sample)>,
    pub distances: Vec<f64>,
    pub table: Table6DAdaptive<f32>,
}

/// Run the 6D scan computation, returning raw results without I/O.
pub fn compute_scan<B: EnergyBackend + Sync>(
    params: &ScanParams,
    backend: &B,
) -> anyhow::Result<ScanResult> {
    // Derive dihedral angle bin width from the finest icosphere mesh so that
    // the ω resolution matches the angular vertex spacing.
    let max_n_vertices = 10 * (params.max_n_div + 1).pow(2) + 2;
    let omega_step = (4.0 * PI / max_n_vertices as f64).sqrt();

    let thermal_energy =
        physical_constants::MOLAR_GAS_CONSTANT * params.temperature * 1e-3; // kJ/mol
    let beta = 1.0 / thermal_energy;

    let mut builder = AdaptiveBuilder::new(
        params.rmin,
        params.rmax,
        params.dr,
        omega_step,
        params.max_n_div,
        params.gradient_threshold,
        beta,
    );

    let n_r = builder.n_r();
    let n_omega = builder.n_omega();
    info!(
        "Adaptive 6D table: {} R-bins x {} omega-bins, max {} vertices (n_div={})",
        n_r, n_omega, max_n_vertices, params.max_n_div,
    );

    let mut partition_samples: Vec<Sample> = Vec::with_capacity(n_r);

    #[cfg(feature = "cli")]
    let r_iter = {
        use indicatif::ProgressIterator;
        (0..n_r).progress_count(n_r as u64)
    };
    #[cfg(not(feature = "cli"))]
    let r_iter = 0..n_r;

    for ri in r_iter {
        let r = builder.r_value(ri);
        let level = builder.current_level();
        let n_v = builder.current_n_vertices();
        let verts = builder.vertex_directions(level);

        let slab_data: Vec<Vec<f64>> = if backend.prefers_batch() {
            compute_slab_batch(backend, &builder, n_omega, n_v, r, verts)
        } else {
            compute_slab_serial(backend, &builder, n_omega, n_v, r, verts)
        };

        let weights = builder.vertex_weights(level);
        let r_sample = accumulate_r_sample(&slab_data, weights, n_v, params.temperature);

        for (oi, data) in slab_data.iter().enumerate() {
            builder.set_slab(ri, oi, data);
        }
        let n_div_before = builder.current_n_div();
        let max_grad = builder.finish_r_slice(ri);
        let n_div_after = builder.current_n_div();

        if n_div_after < n_div_before {
            let new_n_v = builder.current_n_vertices();
            log::debug!(
                "R={:.1} Å: resolution reduced n_div {} → {} ({} vertices)",
                r,
                n_div_before,
                n_div_after,
                new_n_v
            );
        }
        log::debug!(
            "r={:.1} (n_div={}, max_grad={:.2e}): free_energy={:.4e}",
            r,
            n_div_after,
            max_grad,
            r_sample.free_energy()
        );
        partition_samples.push(r_sample);
    }

    let distances: Vec<f64> = (0..n_r).map(|ri| builder.r_value(ri)).collect();
    let samples: Vec<(Vector3, Sample)> = distances
        .iter()
        .zip(partition_samples)
        .map(|(r, s)| (Vector3::new(0.0, 0.0, *r), s))
        .collect();

    let table = builder.build();
    log_resolution_summary(&table);

    Ok(ScanResult {
        samples,
        distances,
        table,
    })
}

/// Async version of compute_scan for WASM where GPU readback must be async.
///
/// The optional `progress` callback is called with `(current_r, total_r)` after each R-bin.
/// If `cancel` is set to true, the scan aborts early with an error.
pub async fn compute_scan_async(
    params: &ScanParams,
    backend: &crate::backend::GpuBackend,
    progress: Option<&dyn Fn(usize, usize)>,
    cancel: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> anyhow::Result<ScanResult> {
    let max_n_vertices = 10 * (params.max_n_div + 1).pow(2) + 2;
    let omega_step = (4.0 * PI / max_n_vertices as f64).sqrt();
    let thermal_energy =
        physical_constants::MOLAR_GAS_CONSTANT * params.temperature * 1e-3;
    let beta = 1.0 / thermal_energy;

    let mut builder = AdaptiveBuilder::new(
        params.rmin, params.rmax, params.dr, omega_step,
        params.max_n_div, params.gradient_threshold, beta,
    );

    let n_r = builder.n_r();
    let n_omega = builder.n_omega();
    info!(
        "Adaptive 6D table: {} R-bins x {} omega-bins, max {} vertices (n_div={})",
        n_r, n_omega, max_n_vertices, params.max_n_div,
    );

    let mut partition_samples: Vec<Sample> = Vec::with_capacity(n_r);

    for ri in 0..n_r {
        let r = builder.r_value(ri);
        let level = builder.current_level();
        let n_v = builder.current_n_vertices();
        let verts = builder.vertex_directions(level);

        let poses = build_slab_poses(&builder, n_omega, n_v, r, verts);
        let energies = backend.compute_energies_async(&poses).await;
        let slab_data = energies_to_slabs(&energies, n_v);

        let weights = builder.vertex_weights(level);
        let r_sample = accumulate_r_sample(&slab_data, weights, n_v, params.temperature);

        for (oi, data) in slab_data.iter().enumerate() {
            builder.set_slab(ri, oi, data);
        }
        builder.finish_r_slice(ri);
        partition_samples.push(r_sample);

        if let Some(cb) = &progress {
            cb(ri + 1, n_r);
        }

        if let Some(flag) = &cancel {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                anyhow::bail!("Scan cancelled");
            }
        }
    }

    let distances: Vec<f64> = (0..n_r).map(|ri| builder.r_value(ri)).collect();
    let samples: Vec<(Vector3, Sample)> = distances
        .iter()
        .zip(partition_samples)
        .map(|(r, s)| (Vector3::new(0.0, 0.0, *r), s))
        .collect();

    let table = builder.build();
    log_resolution_summary(&table);

    Ok(ScanResult { samples, distances, table })
}

/// Build all poses for one R-bin across all omega and vertex pairs.
fn build_slab_poses(
    builder: &AdaptiveBuilder,
    n_omega: usize,
    n_v: usize,
    r: f64,
    verts: &[[f64; 3]],
) -> Vec<PoseParams> {
    let mut poses = Vec::with_capacity(n_omega * n_v * n_v);
    for oi in 0..n_omega {
        let omega = builder.omega_value(oi);
        for vi in 0..n_v {
            for vj in 0..n_v {
                poses.push(PoseParams {
                    r,
                    omega,
                    vertex_i: Vector3::from(verts[vi]),
                    vertex_j: Vector3::from(verts[vj]),
                });
            }
        }
    }
    poses
}

/// Accumulate partition function sample from slab data using Voronoi quadrature weights.
fn accumulate_r_sample(slab_data: &[Vec<f64>], weights: &[f64], n_v: usize, temperature: f64) -> Sample {
    let mut r_sample = Sample::default();
    for data in slab_data {
        for vi in 0..n_v {
            for vj in 0..n_v {
                let degeneracy = weights[vi] * weights[vj];
                r_sample += Sample::new(data[vi * n_v + vj], temperature, degeneracy);
            }
        }
    }
    r_sample
}

/// Split flat energies into per-omega slab chunks.
fn energies_to_slabs(energies: &[f64], n_v: usize) -> Vec<Vec<f64>> {
    energies.chunks_exact(n_v * n_v).map(|c| c.to_vec()).collect()
}

/// GPU: batch all (omega, vi, vj) for this R into one dispatch.
fn compute_slab_batch<B: EnergyBackend>(
    backend: &B,
    builder: &AdaptiveBuilder,
    n_omega: usize,
    n_v: usize,
    r: f64,
    verts: &[[f64; 3]],
) -> Vec<Vec<f64>> {
    let poses = build_slab_poses(builder, n_omega, n_v, r, verts);
    energies_to_slabs(&backend.compute_energies(&poses), n_v)
}

/// CPU: parallelize over omega slabs (serial fallback without rayon).
fn compute_slab_serial<B: EnergyBackend + Sync>(
    backend: &B,
    builder: &AdaptiveBuilder,
    n_omega: usize,
    n_v: usize,
    r: f64,
    verts: &[[f64; 3]],
) -> Vec<Vec<f64>> {
    let compute_one = |oi: usize| -> Vec<f64> {
        let omega = builder.omega_value(oi);
        let mut data = vec![0.0; n_v * n_v];
        for vi in 0..n_v {
            for vj in 0..n_v {
                data[vi * n_v + vj] = backend.compute_energy(&PoseParams {
                    r,
                    omega,
                    vertex_i: Vector3::from(verts[vi]),
                    vertex_j: Vector3::from(verts[vj]),
                });
            }
        }
        data
    };

    #[cfg(feature = "cli")]
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        (0..n_omega).into_par_iter().map(compute_one).collect()
    }
    #[cfg(not(feature = "cli"))]
    {
        (0..n_omega).map(compute_one).collect()
    }
}

/// CLI wrapper: runs compute_scan + file I/O + reporting.
#[cfg(feature = "cli")]
pub fn do_icoscan<B: EnergyBackend + Sync>(
    config: &ScanConfig,
    backend: &B,
) -> anyhow::Result<()> {
    let start_time = std::time::Instant::now();
    let mut result = compute_scan(&config.params, backend)?;

    let elapsed = start_time.elapsed();
    info!("Finished computing energies in {:.2}s", elapsed.as_secs_f64());

    if let Some(path) = &config.save_table {
        log::info!("Saving adaptive 6D table to {}", path.display());
        let free_energies: Vec<f64> =
            result.samples.iter().map(|(_, s)| s.free_energy()).collect();
        let electric_prefactor =
            faunus::interatomic::ELECTRIC_PREFACTOR / config.params.permittivity;
        let tail_terms = fit_tail_correction(
            &result.distances,
            &free_energies,
            config.params.charges,
            config.params.dipole_moments,
            config.params.kappa,
            electric_prefactor,
        );
        let point_group = if config.params.homo_dimer {
            icotable::PointGroup::Exchange
        } else {
            icotable::PointGroup::Asymmetric
        };
        result.table.metadata = Some(TableMetadata {
            tail_terms,
            charges: Some(config.params.charges),
            dipole_moments: Some(config.params.dipole_moments),
            temperature: Some(config.params.temperature),
            electric_prefactor: Some(electric_prefactor),
            point_group,
        });
        result.table.save(path)?;
    }

    let masses = (
        backend.ref_a().total_mass(),
        backend.ref_b().total_mass(),
    );
    crate::report::report_pmf(&result.samples, &config.pmf_file, Some(masses))?;
    Ok(())
}

/// Orient two reference structures to a given 6D point.
///
/// Structure A is kept at origin; structure B is rotated and translated.
pub(crate) fn orient_structures(
    r: f64,
    omega: f64,
    vertex_i: &Vector3,
    vertex_j: &Vector3,
    ref_a: &Structure,
    ref_b: &Structure,
) -> (Structure, Structure) {
    let (_, q_b, separation) = icotable::orient(r, omega, vertex_i, vertex_j);
    let mut mol_b = ref_b.clone();
    mol_b.transform(|pos| q_b.transform_vector(&pos) + separation);
    (ref_a.clone(), mol_b)
}

/// Build tail correction from the angularly averaged free energy at outer radial bins.
///
/// The ion-ion (Yukawa) coefficient is the charge product `z₁·z₂`; the Coulomb
/// prefactor `e²/(4��ε₀εᵣ)` is stored separately in [`TableMetadata::electric_prefactor`].
/// Higher-order terms are fitted from the residual using log-space linear regression.
#[cfg(feature = "cli")]
fn fit_tail_correction(
    distances: &[f64],
    free_energies: &[f64],
    charges: [f64; 2],
    dipole_moments: [f64; 2],
    kappa_hint: Option<f64>,
    electric_prefactor: f64,
) -> Vec<TailCorrectionTerm> {
    let mut terms = Vec::new();
    let n = distances.len();
    if n < 5 {
        return terms;
    }
    let start = n.saturating_sub(5);
    let epsilon = 1e-10;

    let mut residuals: Vec<f64> = free_energies.to_vec();

    // Ion-ion (Yukawa): coefficient = z₁·z₂ (charge product only)
    let has_charges = charges[0].abs() > epsilon && charges[1].abs() > epsilon;
    if has_charges {
        if let Some(kappa) = kappa_hint {
            let coefficient = charges[0] * charges[1];
            for (res, r) in residuals.iter_mut().zip(distances) {
                *res -= electric_prefactor * coefficient * (-kappa * r).exp() / r;
            }
            info!(
                "Tail charge product = {:.4} e², κ = {:.4} 1/Å",
                coefficient, kappa
            );
            terms.push(TailCorrectionTerm {
                coefficient,
                kappa,
                power: 1,
            });
        }
    }

    // Ion-dipole: fit residual with R⁴ denominator
    let has_dipole_coupling = (charges[0].abs() > epsilon && dipole_moments[1] > epsilon)
        || (charges[1].abs() > epsilon && dipole_moments[0] > epsilon);

    if has_dipole_coupling {
        if let Some(mut term) = fit_screened_term(distances, &residuals, start, 4, kappa_hint) {
            // Fitted coefficient is in absolute units; divide by prefactor
            // so that tail_energy() (which multiplies all terms by prefactor) is correct
            term.coefficient /= electric_prefactor;
            info!(
                "Tail fit ion-dipole: C={:.4e}, κ={:.4} 1/Å",
                term.coefficient, term.kappa
            );
            terms.push(term);
        }
    }
    terms
}

/// Fit a single `C·exp(-κR)/R^power` term via log-space linear regression.
///
/// Linearizes `ln|u(R)·R^power| = ln|C| - κ·R` and fits slope/intercept.
/// Only uses bins with the majority sign to avoid fitting across zero crossings.
#[cfg(feature = "cli")]
fn fit_screened_term(
    distances: &[f64],
    energies: &[f64],
    start: usize,
    power: u32,
    kappa_hint: Option<f64>,
) -> Option<TailCorrectionTerm> {
    let epsilon = 1e-10;

    // Exclude minority-sign bins to avoid log-space fitting across zero crossings
    let sign_sum: i64 = energies[start..]
        .iter()
        .filter(|u| u.abs() > epsilon)
        .map(|u| if *u > 0.0 { 1i64 } else { -1 })
        .sum();
    let majority_positive = sign_sum >= 0;

    let (xs, ys): (Vec<f64>, Vec<f64>) = distances[start..]
        .iter()
        .zip(&energies[start..])
        .filter(|(_, u)| u.abs() >= epsilon && (u.is_sign_positive() == majority_positive))
        .map(|(r, u)| (*r, (u.abs() * r.powi(power as i32)).ln()))
        .unzip();
    if xs.len() < 2 {
        return None;
    }

    // Linear regression: y = a + b*x → b = -κ, a = ln|C|
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < epsilon {
        return None;
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;

    // Accept fitted κ if close to hint; otherwise force hint κ and refit intercept,
    // because too few bins can give a wildly wrong slope
    let (kappa, ln_c) = if let Some(hint) = kappa_hint {
        let fitted = -slope;
        if fitted > 0.0 && (fitted - hint).abs() / hint < 1.0 {
            (fitted, intercept)
        } else {
            let ln_c = xs.iter().zip(&ys).map(|(r, y)| y + hint * r).sum::<f64>() / n;
            (hint, ln_c)
        }
    } else {
        let fitted = -slope;
        if fitted > 0.0 {
            (fitted, intercept)
        } else {
            return None;
        }
    };

    let sign = if majority_positive { 1.0 } else { -1.0 };
    let coefficient = sign * ln_c.exp();

    Some(TailCorrectionTerm {
        coefficient,
        kappa,
        power,
    })
}

/// Log a per-R summary of slab resolution tiers.
fn log_resolution_summary(table: &icotable::Table6DAdaptive<f32>) {
    use icotable::SlabResolution;
    log::debug!("Slab resolution summary (per R-slice across all ω bins):");
    log::debug!(
        "{:>8} {:>10} {:>8} {:>8} {:>8} {:>6}",
        "R (Å)",
        "repulsive",
        "scalar",
        "nearest",
        "interp",
        "n_div"
    );
    let mut tot = [0u32; 4]; // [repulsive, scalar, nearest, interp]
    for ri in 0..table.n_r {
        let r = table.rmin + ri as f64 * table.dr;
        let slabs = &table.slab_res[ri * table.n_omega..(ri + 1) * table.n_omega];
        let mut row = [0u32; 4];
        let mut level = None;
        for s in slabs {
            match s {
                SlabResolution::Repulsive => row[0] += 1,
                SlabResolution::Scalar(_) => row[1] += 1,
                SlabResolution::Mesh {
                    level: l,
                    interpolate: false,
                } => {
                    row[2] += 1;
                    level = Some(*l);
                }
                SlabResolution::Mesh {
                    level: l,
                    interpolate: true,
                } => {
                    row[3] += 1;
                    level = Some(*l);
                }
            }
        }
        for i in 0..4 {
            tot[i] += row[i];
        }
        let n_div_str = level.map_or("-".to_string(), |l| {
            table.levels[l as usize].n_div.to_string()
        });
        log::debug!(
            "{:>8.1} {:>10} {:>8} {:>8} {:>8} {:>6}",
            r,
            row[0],
            row[1],
            row[2],
            row[3],
            n_div_str
        );
    }
    let total: u32 = tot.iter().sum();
    info!(
        "Total slabs: {} (repulsive: {}, scalar: {}, nearest: {}, interp: {})",
        total, tot[0], tot[1], tot[2], tot[3]
    );
}

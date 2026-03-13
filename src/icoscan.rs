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
    report::report_pmf,
    structure::Structure,
    Sample, Table6D, Vector3,
};
use icotable::{TableMetadata, TailCorrectionTerm};
use faunus::auxiliary::open_compressed;
use get_size::GetSize;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use iter_num_tools::arange;
use itertools::Itertools;
use molly::{Frame, XTCWriter};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    f64::consts::PI,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

/// Configuration for a 6D icoscan
pub struct ScanConfig {
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    pub angle_resolution: f64,
    pub temperature: f64,
    pub pmf_file: PathBuf,
    pub xtcfile: Option<PathBuf>,
    /// Save binary 6D table for Faunus lookup (.gz suffix enables gzip compression)
    pub save_table: Option<PathBuf>,
    /// Net charges of the two molecules in elementary charges.
    pub charges: [f64; 2],
    /// Dipole moment magnitudes in e·Å.
    pub dipole_moments: [f64; 2],
    /// Debye screening parameter (1/Å), if known.
    pub kappa: Option<f64>,
    /// Relative permittivity of the medium.
    pub permittivity: f64,
}

/// Build tail correction from the angularly averaged free energy at outer radial bins.
///
/// The ion-ion (Yukawa) coefficient is the charge product `z₁·z₂`; the Coulomb
/// prefactor `e²/(4πε₀εᵣ)` is stored separately in [`TableMetadata::electric_prefactor`].
/// Higher-order terms are fitted from the residual using log-space linear regression.
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
            terms.push(TailCorrectionTerm { coefficient, kappa, power: 1 });
        }
    }

    // Ion-dipole: fit residual with R⁴ denominator
    let has_dipole_coupling =
        (charges[0].abs() > epsilon && dipole_moments[1] > epsilon)
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
        if fitted > 0.0 { (fitted, intercept) } else { return None; }
    };

    let sign = if majority_positive { 1.0 } else { -1.0 };
    let coefficient = sign * ln_c.exp();

    Some(TailCorrectionTerm { coefficient, kappa, power })
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

pub fn do_icoscan<B: EnergyBackend>(config: &ScanConfig, backend: &B) -> anyhow::Result<()> {
    let ref_a = backend.ref_a();
    let ref_b = backend.ref_b();
    let table =
        Table6D::from_resolution(config.rmin, config.rmax, config.dr, config.angle_resolution)?;
    let n_vertices = table.get(config.rmin)?.get(0.0)?.len();
    // Average angular spacing between icosphere vertices
    let angle_resolution = (4.0 * PI / n_vertices as f64).sqrt();
    let dihedral_angles = arange(0.0..2.0 * PI, angle_resolution).collect_vec();
    let distances = arange(config.rmin..config.rmax, config.dr).collect_vec();
    let n_total = distances.len() * dihedral_angles.len() * n_vertices * n_vertices;

    info!(
        "6D table: 𝑅({}) x 𝜔({}) x 𝜃𝜑({}) x 𝜃𝜑({}) = {} poses 💃🕺 ({:.1} MB)",
        distances.len(),
        dihedral_angles.len(),
        n_vertices,
        n_vertices,
        n_total,
        table.get_size() as f64 / 1e6
    );

    // Pair all mass center separations (r) and dihedral angles (omega)
    let r_and_omega = distances
        .iter()
        .copied()
        .cartesian_product(dihedral_angles.iter().copied())
        .collect_vec();

    // Populate 6D table with inter-particle energies
    let start_time = std::time::Instant::now();
    if backend.prefers_batch() {
        // Batched processing (for GPU): batch by distance for progress while keeping large batches
        for r in distances.iter().progress_count(distances.len() as u64) {
            // Collect all poses for this distance (all omega and vertex combinations)
            let mut poses: Vec<PoseParams> = Vec::new();
            let mut data_refs: Vec<_> = Vec::new();

            for omega in &dihedral_angles {
                table
                    .get_icospheres(*r, *omega)
                    .expect("invalid (r, omega) value")
                    .flat_iter()
                    .for_each(|(pos_a, pos_b, data_b)| {
                        poses.push(PoseParams {
                            r: *r,
                            omega: *omega,
                            vertex_i: *pos_a,
                            vertex_j: *pos_b,
                        });
                        data_refs.push(data_b);
                    });
            }

            // Compute all energies for this distance in one GPU batch
            let energies = backend.compute_energies(&poses);

            // Store results
            for (data_b, energy) in data_refs.into_iter().zip(energies) {
                data_b.set(energy).expect("Energy already calculated");
            }
        }
    } else {
        // Per-pose processing with rayon parallelization (for CPU)
        info!(
            "Computing {} (r,omega) pairs using CPU backend...",
            r_and_omega.len()
        );
        let calc_energy = |r: f64, omega: f64| {
            table
                .get_icospheres(r, omega)
                .expect("invalid (r, omega) value")
                .flat_iter()
                .for_each(|(pos_a, pos_b, data_b)| {
                    let pose = PoseParams {
                        r,
                        omega,
                        vertex_i: *pos_a,
                        vertex_j: *pos_b,
                    };
                    let energy = backend.compute_energy(&pose);
                    data_b.set(energy).expect("Energy already calculated");
                });
        };

        r_and_omega
            .par_iter()
            .progress_count(r_and_omega.len() as u64)
            .for_each(|(r, omega)| {
                calc_energy(*r, *omega);
            });
    }
    let elapsed = start_time.elapsed();
    let poses_per_ms = n_total as f64 / elapsed.as_millis() as f64;
    info!(
        "Finished computing energies in {:.2}s ({:.1} poses/ms)",
        elapsed.as_secs_f64(),
        poses_per_ms
    );

    if let Some(xtcfile) = &config.xtcfile {
        write_trajectory(xtcfile, &table, &r_and_omega, ref_a, ref_b)?;
    }

    // Partition function contribution for single (r, omega) point
    // i.e. averaged over 4D angular space
    let calc_partition_func = |r: f64, omega: f64| {
        table.get_icospheres(r, omega).unwrap().flat_iter().fold(
            Sample::default(),
            |sum, (vertex_i, vertex_j, data_b)| {
                let degeneracy = vertex_i.norm() * vertex_j.norm();
                let energy = data_b.get().unwrap(); // kJ/mol
                sum + Sample::new(*energy, config.temperature, degeneracy)
            },
        )
    };

    // Calculate partition function as function of r only
    let samples: Vec<(Vector3, Sample)> = distances
        .iter()
        .map(|r| {
            let partition_func: Sample = dihedral_angles
                .iter()
                .map(|omega| calc_partition_func(*r, *omega))
                .sum();
            log::debug!(
                "r={:.1}: exp_energy={:.4e}, mean_energy={:.4e}, free_energy={:.4e}",
                r,
                partition_func.exp_energy(),
                partition_func.mean_energy(),
                partition_func.free_energy()
            );
            (Vector3::new(0.0, 0.0, *r), partition_func)
        })
        .collect();

    // Save binary 6D table for Faunus lookup, with tail correction metadata
    if let Some(path) = &config.save_table {
        log::info!("Saving binary 6D table to {}", path.display());
        let mut flat = icotable::Table6DFlat::<icotable::f16>::try_from(&table)?;

        let free_energies: Vec<f64> = samples.iter().map(|(_, s)| s.free_energy()).collect();
        let electric_prefactor =
            faunus::interatomic::ELECTRIC_PREFACTOR / config.permittivity;
        let tail_terms = fit_tail_correction(
            &distances,
            &free_energies,
            config.charges,
            config.dipole_moments,
            config.kappa,
            electric_prefactor,
        );
        flat.metadata = Some(TableMetadata {
            tail_terms,
            charges: Some(config.charges),
            dipole_moments: Some(config.dipole_moments),
            temperature: Some(config.temperature),
            electric_prefactor: Some(electric_prefactor),
        });
        flat.save(path)?;
    }

    let masses = (ref_a.total_mass(), ref_b.total_mass());

    report_pmf(&samples, &config.pmf_file, Some(masses))?;
    Ok(())
}

/// Write oriented structures and energies to XTC trajectory and companion energy file.
fn write_trajectory(
    xtcfile: &Path,
    table: &Table6D,
    r_and_omega: &[(f64, f64)],
    ref_a: &Structure,
    ref_b: &Structure,
) -> anyhow::Result<()> {
    info!("Writing trajectory file {}", xtcfile.display());
    let mut traj = XTCWriter::create(xtcfile)?;
    let mut energy_file =
        BufWriter::new(open_compressed(&xtcfile.with_extension("energy.dat.gz"))?);
    writeln!(energy_file, "# Energy (kJ/mol)").expect("Failed to write header");
    let mut frame_cnt: u32 = 0;
    let mut frame = Frame {
        precision: 1000.0,
        ..Default::default()
    };

    let mut write_frame = |oriented_a: &Structure, oriented_b: &Structure, energy| {
        frame.step = frame_cnt;
        frame.time = frame_cnt as f32;
        frame_cnt += 1;
        frame.positions = oriented_a
            .pos
            .iter()
            .chain(oriented_b.pos.iter())
            .flat_map(|&p| [p.x as f32 * 0.1, p.y as f32 * 0.1, p.z as f32 * 0.1])
            .collect();
        traj.write_frame(&frame).expect("Failed to write XTC frame");
        writeln!(energy_file, "{energy:.6}").expect("Failed to write energy to file");
    };

    let n = r_and_omega.len();
    r_and_omega
        .iter()
        .progress_count(n as u64)
        .for_each(|(r, omega)| {
            table
                .get_icospheres(*r, *omega)
                .expect("invalid (r, omega) value")
                .flat_iter()
                .for_each(|(pos_a, pos_b, data_b)| {
                    let (oriented_a, oriented_b) =
                        orient_structures(*r, *omega, pos_a, pos_b, ref_a, ref_b);
                    write_frame(&oriented_a, &oriented_b, data_b.get().unwrap());
                });
        });
    info!("Wrote {frame_cnt} frames to trajectory file");
    Ok(())
}

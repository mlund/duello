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

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use duello::{
    backend::{self, GpuBackend, SimdBackend},
    energy, icoscan, report,
    structure::{pqr_write_atom, Structure},
    Adaptive3DBuilder, IcoTable2D, Sample, SphericalCoord, UnitQuaternion, Vector3,
};
use faunus::interatomic::coulomb::{pairwise::Plain, permittivity, DebyeLength, Medium, Salt};
use faunus::interatomic::twobody::{GridTrim, GridType, SplineConfig};
use faunus::{
    energy::NonbondedMatrix,
    topology::{FindByName, Topology},
};
use std::process::ExitCode;
use std::{f64::consts::PI, fs::File, io::Write, ops::Neg, path::PathBuf};
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
use iter_num_tools::arange;
use rand::Rng;

/// Spline grid type selection
#[derive(Clone, Copy, Debug, Default)]
pub enum SplineGrid {
    /// PowerLaw2 grid: denser at short range, uses 2 sqrts for index (default)
    #[default]
    Powerlaw2,
    /// InverseRsq grid: denser at short range, uses only division for index
    InvR2,
}

impl From<SplineGrid> for GridType {
    fn from(grid: SplineGrid) -> Self {
        match grid {
            SplineGrid::Powerlaw2 => Self::PowerLaw2,
            SplineGrid::InvR2 => Self::InverseRsq,
        }
    }
}

/// Grid interpolation options parsed from "type=X,size=Y,shift=bool" format
#[derive(Clone, Copy, Debug)]
pub struct GridOptions {
    pub grid_type: SplineGrid,
    pub size: usize,
    pub shift: bool,
    /// If set, cap |U(r_min)| to reduce f32 precision loss on GPU/SIMD
    pub energy_cap: Option<f64>,
}

impl Default for GridOptions {
    fn default() -> Self {
        Self {
            grid_type: SplineGrid::Powerlaw2,
            size: 500,
            shift: true,
            energy_cap: None,
        }
    }
}

impl std::str::FromStr for GridOptions {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut opts = Self::default();
        for part in s.split(',') {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("expected key=value, got '{part}'"))?;
            match key.trim() {
                "type" => {
                    opts.grid_type = match value.trim().to_lowercase().as_str() {
                        "powerlaw2" => SplineGrid::Powerlaw2,
                        "invr2" | "inversersq" => SplineGrid::InvR2,
                        _ => return Err(format!("unknown grid type '{value}'")),
                    };
                }
                "size" => {
                    opts.size = value
                        .trim()
                        .parse()
                        .map_err(|_| format!("invalid size '{value}'"))?;
                }
                "shift" => {
                    opts.shift = match value.trim().to_lowercase().as_str() {
                        "true" | "1" | "yes" => true,
                        "false" | "0" | "no" => false,
                        _ => return Err(format!("invalid shift value '{value}'")),
                    };
                }
                "energy_cap" => {
                    let v = value.trim();
                    opts.energy_cap = match v.to_lowercase().as_str() {
                        "none" | "off" | "false" => None,
                        _ => Some(v.parse().map_err(|_| format!("invalid energy_cap '{v}'"))?),
                    };
                }
                _ => return Err(format!("unknown option '{key}'")),
            }
        }
        Ok(opts)
    }
}

/// Compute backend selection
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum Backend {
    /// Auto-detect: GPU if available, otherwise SIMD
    #[default]
    Auto,
    /// Reference backend using exact (non-splined) pair potentials
    Reference,
    /// GPU backend using wgpu compute shaders
    Gpu,
    /// SIMD backend (AVX2 on x86_64, NEON on aarch64)
    Simd,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Dipole {
        /// Path to first XYZ file
        #[arg(short = 'o', long)]
        output: PathBuf,
        /// Dipole moment strength
        #[arg(short = 'm')]
        mu: f64,
        /// Angular resolution in radians
        #[arg(short = 'r', long, default_value = "0.1")]
        resolution: f64,
        /// Minimum mass center distance
        #[arg(long)]
        rmin: f64,
        /// Maximum mass center distance
        #[arg(long)]
        rmax: f64,
        /// Mass center distance step
        #[arg(long)]
        dr: f64,
    },

    Potential {
        /// Path to first XYZ file
        #[arg(short = '1', long)]
        mol1: PathBuf,
        /// Angular resolution in radians
        #[arg(short = 'r', long, default_value = "0.1")]
        resolution: f64,
        /// Radius around center of mass to scan to calc. potentil (angstroms)
        #[arg(long)]
        radius: f64,
        /// YAML file with atom definitions (names, charges, etc.)
        #[arg(short = 'p', long = "top")]
        topology: PathBuf,
        /// 1:1 salt molarity in mol/l
        #[arg(short = 'M', long, default_value = "0.1")]
        molarity: f64,
        /// Cutoff distance for pair-wise interactions (angstroms)
        #[arg(long, default_value = "50.0")]
        cutoff: f64,
        /// Temperature in K
        #[arg(short = 'T', long, default_value = "298.15")]
        temperature: f64,
    },

    /// Scan angles and tabulate energy between a rigid body and a single atom
    AtomScan {
        /// Path to rigid body XYZ file
        #[arg(short = '1', long)]
        mol1: PathBuf,
        /// Name of atom type to scan (looked up in topology)
        #[arg(long)]
        atom: String,
        /// Max icosphere subdivision level (0=12, 1=42, 2=92, 3=162 vertices)
        #[arg(long, default_value = "3")]
        max_ndiv: usize,
        /// Boltzmann-weight gradient threshold for adaptive resolution (1/rad)
        #[arg(long, default_value = "0.5")]
        gradient_threshold: f64,
        /// Minimum mass center distance
        #[arg(long)]
        rmin: f64,
        /// Maximum mass center distance
        #[arg(long)]
        rmax: f64,
        /// Mass center distance step
        #[arg(long)]
        dr: f64,
        /// YAML file with atom definitions (names, charges, etc.)
        #[arg(short = 'a', long = "top")]
        topology: PathBuf,
        /// 1:1 salt molarity in mol/l
        #[arg(short = 'M', long, default_value = "0.1")]
        molarity: f64,
        /// Cutoff distance for pair-wise interactions (angstroms)
        #[arg(long, default_value = "50.0")]
        cutoff: f64,
        /// Temperature in K
        #[arg(short = 'T', long, default_value = "298.15")]
        temperature: f64,
        /// Output binary table path (.gz suffix enables gzip compression)
        #[arg(short = 'o', long, default_value = "atom_table.bin.gz")]
        output: PathBuf,
        /// Output file for PMF
        #[arg(long = "pmf", default_value = "pmf.dat")]
        pmf_file: PathBuf,
    },

    /// Scan angles and tabulate energy between two rigid bodies
    Scan {
        /// Path to first XYZ file
        #[arg(short = '1', long)]
        mol1: PathBuf,
        /// Path to second XYZ file
        #[arg(short = '2', long)]
        mol2: PathBuf,
        /// Minimum mass center distance
        #[arg(long)]
        rmin: f64,
        /// Maximum mass center distance
        #[arg(long)]
        rmax: f64,
        /// Mass center distance step
        #[arg(long)]
        dr: f64,
        /// YAML file with atom definitions (names, charges, etc.)
        #[arg(short = 'a', long = "top")]
        topology: PathBuf,
        /// 1:1 salt molarity in mol/l
        #[arg(short = 'M', long, default_value = "0.1")]
        molarity: f64,
        /// Cutoff distance for pair-wise interactions (angstroms)
        #[arg(long, default_value = "50.0")]
        cutoff: f64,
        /// Temperature in K
        #[arg(short = 'T', long, default_value = "298.15")]
        temperature: f64,
        /// Optionally use fixed dielectric constant
        #[arg(long)]
        fixed_dielectric: Option<f64>,
        /// Output file for PMF
        #[arg(long = "pmf", default_value = "pmf.dat")]
        pmf_file: PathBuf,
        /// Save binary 6D table for Faunus lookup (.gz suffix enables gzip compression)
        #[arg(long)]
        savetable: Option<PathBuf>,
        /// Max icosphere subdivision level (0=12, 1=42, 2=92, 3=162 vertices)
        #[arg(long, default_value = "3")]
        max_ndiv: usize,
        /// Boltzmann-weight gradient threshold for adaptive resolution (1/rad)
        #[arg(long, default_value = "0.5")]
        gradient_threshold: f64,
        /// Compute backend
        #[arg(long, value_enum, default_value = "auto")]
        backend: Backend,
        /// Grid interpolation: type=powerlaw2|invr2,size=N,shift=bool
        #[arg(
            long,
            default_value = "type=powerlaw2,size=500,shift=true,energy_cap=none"
        )]
        grid: GridOptions,
        /// Short-range spline cutoff for GPU/SIMD split path (angstroms).
        /// Defaults to --cutoff if not set. Set to SR potential range for best accuracy.
        #[arg(long)]
        sr_cutoff: Option<f64>,
    },

    /// Estimate rotational diffusion from a saved 6D energy table
    Diffusion {
        /// Path to saved Table6DAdaptive (.bin or .bin.gz)
        table: PathBuf,
        /// Temperature in K
        #[arg(short = 'T', long, default_value = "298.15")]
        temperature: f64,
        /// Output CSV file for diffusion results
        #[arg(short = 'o', long, default_value = "diffusion.csv")]
        output: PathBuf,
        /// Homo-dimer: apply exchange symmetrization U(A,B) ↔ U(B,A)
        #[arg(long)]
        symmetric: bool,
    },
}

/// Calculate energy of all two-body poses
fn do_scan(cmd: &Commands) -> Result<()> {
    let Commands::Scan {
        mol1,
        mol2,
        rmin,
        rmax,
        dr,
        topology: top_file,
        molarity,
        cutoff,
        temperature,
        fixed_dielectric,
        pmf_file,
        savetable,
        max_ndiv,
        gradient_threshold,
        backend: backend_type,
        grid,
        sr_cutoff,
    } = cmd
    else {
        bail!("Unknown command");
    };
    assert!(rmin < rmax);

    let mut topology = Topology::from_file_partial(top_file)?;
    topology.finalize_atoms()?;
    topology.finalize_molecules()?;
    faunus::topology::set_missing_epsilon(topology.atomkinds_mut(), 2.479);

    // Either use fixed dielectric constant or calculate it from the medium
    let medium = fixed_dielectric.map_or_else(
        || Medium::salt_water(*temperature, Salt::SodiumChloride, *molarity),
        |dielectric_const| {
            Medium::new(
                *temperature,
                permittivity::Permittivity::Fixed(dielectric_const),
                Some((Salt::SodiumChloride, *molarity)),
            )
        },
    );

    let nonbonded = NonbondedMatrix::from_file(top_file, &topology, Some(medium.clone()))?;

    let ref_a = Structure::from_xyz(mol1, topology.atomkinds())?;
    let ref_b = Structure::from_xyz(mol2, topology.atomkinds())?;

    info!("{medium}");
    info!(
        "Molecular net-charges:    [{:.2}e, {:.2}e]",
        ref_a.net_charge(),
        ref_b.net_charge(),
    );

    const ELECTRON_ANGSTROM_TO_DEBYE: f64 = 4.803_204_25;
    info! {
        "Molecular dipole moments: [{:.2} D, {:.2} D]",
        ref_a.dipole_moment().norm() * ELECTRON_ANGSTROM_TO_DEBYE,
        ref_b.dipole_moment().norm() * ELECTRON_ANGSTROM_TO_DEBYE,
    };

    info!(
        "Molecular masses (g/mol): [{:.2}, {:.2}]",
        ref_a.total_mass(),
        ref_b.total_mass(),
    );

    info!("COM range: [{rmin:.1}, {rmax:.1}) in {dr:.1} Å steps 🐾");

    // Auto-detect backend: try GPU first, fall back to SIMD
    let backend_type = match backend_type {
        Backend::Auto => {
            if GpuBackend::is_available() {
                info!("Auto-detected GPU backend");
                Backend::Gpu
            } else {
                info!("No GPU available, using SIMD backend");
                Backend::Simd
            }
        }
        other => *other,
    };

    let scan_config = icoscan::ScanConfig {
        rmin: *rmin,
        rmax: *rmax,
        dr: *dr,
        temperature: *temperature,
        pmf_file: pmf_file.clone(),
        save_table: savetable.clone(),
        charges: [ref_a.net_charge(), ref_b.net_charge()],
        dipole_moments: [ref_a.dipole_moment().norm(), ref_b.dipole_moment().norm()],
        kappa: medium.debye_length().map(|dl| 1.0 / dl),
        permittivity: medium.permittivity(),
        max_n_div: *max_ndiv,
        gradient_threshold: *gradient_threshold,
    };

    // energy_cap only applies to SR-only splines (GPU/SIMD split path).
    // The combined spline (CPU) includes Coulomb which diverges as 1/r,
    // so capping it would corrupt the potential.
    let sr_grid_trim = grid
        .energy_cap
        .map_or(GridTrim::NoTrim, |cap| GridTrim::RepulsiveDecay {
            energy_cap: cap,
        });
    let base_spline_config = SplineConfig {
        n_points: grid.size,
        shift_energy: grid.shift,
        grid_type: grid.grid_type.into(),
        ..Default::default()
    };

    let pair_matrix = energy::PairMatrix::new(nonbonded, topology.atomkinds(), &medium);

    match backend_type {
        Backend::Auto => unreachable!(), // Already resolved above
        Backend::Reference => {
            let backend = backend::cpu::CpuBackend::new(ref_a, ref_b, pair_matrix);
            icoscan::do_icoscan(&scan_config, &backend)
        }
        Backend::Gpu | Backend::Simd => {
            let sr_cut = sr_cutoff.unwrap_or(*cutoff);
            info!("Split path: SR spline (cutoff={sr_cut:.1} Å) + analytical Coulomb");
            let sr_spline_config = SplineConfig {
                grid_trim: sr_grid_trim,
                ..base_spline_config
            };
            let splined = pair_matrix.create_sr_splines(sr_cut, sr_spline_config);
            let coulomb = pair_matrix.coulomb_params();

            match backend_type {
                Backend::Gpu => {
                    let gpu_backend = GpuBackend::new(ref_a, ref_b, &splined, coulomb)?;
                    icoscan::do_icoscan(&scan_config, &gpu_backend)
                }
                Backend::Simd => {
                    let simd_backend = SimdBackend::new(ref_a, ref_b, &splined, coulomb);
                    icoscan::do_icoscan(&scan_config, &simd_backend)
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Compute radial+angular energy table between a rigid body and a single test atom
fn do_atom_scan(cmd: &Commands) -> Result<()> {
    let Commands::AtomScan {
        mol1,
        atom,
        max_ndiv,
        gradient_threshold,
        rmin,
        rmax,
        dr,
        topology: top_file,
        molarity,
        cutoff: _,
        temperature,
        output,
        pmf_file,
    } = cmd
    else {
        bail!("Unknown command");
    };
    anyhow::ensure!(rmin < rmax, "rmin ({rmin}) must be less than rmax ({rmax})");

    let mut topology = Topology::from_file_partial(top_file)?;
    topology.finalize_atoms()?;
    topology.finalize_molecules()?;
    faunus::topology::set_missing_epsilon(topology.atomkinds_mut(), 2.479);

    let medium = Medium::salt_water(*temperature, Salt::SodiumChloride, *molarity);
    let nonbonded = NonbondedMatrix::from_file(top_file, &topology, Some(medium.clone()))?;

    let pair_matrix = energy::PairMatrix::new(nonbonded, topology.atomkinds(), &medium);

    let ref_a = Structure::from_xyz(mol1, topology.atomkinds())?;
    let test_atom_id = topology
        .atomkinds()
        .find_name(atom)
        .ok_or_else(|| anyhow::anyhow!("Unknown atom type: {atom}"))?
        .id();

    info!("{medium}");
    info!("Rigid body net-charge: {:.2}e", ref_a.net_charge());
    info!("Test atom: {atom} (id={test_atom_id})");

    // β = 1/kT in mol/kJ — needed by the adaptive builder to detect
    // repulsive slabs where all Boltzmann weights are negligible
    let thermal_energy = physical_constants::MOLAR_GAS_CONSTANT * temperature * 1e-3;
    let beta = 1.0 / thermal_energy;

    let mut builder =
        Adaptive3DBuilder::new(*rmin, *rmax, *dr, *max_ndiv, *gradient_threshold, beta);

    let n_r = builder.n_r();
    let max_vertices = 10 * (*max_ndiv + 1).pow(2) + 2;
    info!(
        "Adaptive 3D: {} R-bins, max {} vertices (n_div={})",
        n_r, max_vertices, max_ndiv
    );
    info!("COM range: [{rmin:.1}, {rmax:.1}) in {dr:.1} Å steps");

    // Scan from short range outward; finish_r_slice may lower the mesh
    // level when gradients are small, so later (larger-R) slabs use fewer vertices
    let mut energies = Vec::with_capacity(max_vertices);
    let mut samples = Vec::with_capacity(n_r);
    for ri in 0..n_r {
        let r = builder.r_value(ri);
        let level = builder.current_level();
        let verts = builder.vertex_directions(level);
        let weights = builder.vertex_weights(level);

        energies.clear();
        energies.extend(verts.iter().map(|v| {
            let test_pos = Vector3::from(*v).scale(r);
            pair_matrix.energy_with_atom(&ref_a, test_atom_id, &test_pos)
        }));

        // Accumulate Boltzmann-weighted samples for PMF using quadrature weights
        let r_sample: Sample = energies
            .iter()
            .zip(weights)
            .map(|(&e, &w)| Sample::new(e, *temperature, w))
            .sum();
        samples.push((Vector3::new(r, 0.0, 0.0), r_sample));

        builder.set_slab(ri, &energies);
        builder.finish_r_slice(ri);
    }

    let table = builder.build();
    let n_data = table.data.len();
    table.save(output)?;
    info!(
        "Saved adaptive 3D table ({} R-bins, {} data values) to {}",
        n_r,
        n_data,
        output.display()
    );

    report::report_pmf(&samples, pmf_file, None)?;
    Ok(())
}

fn do_dipole(cmd: &Commands) -> Result<()> {
    let Commands::Dipole {
        output,
        mu: dipole_moment,
        resolution,
        rmin,
        rmax,
        dr,
    } = cmd
    else {
        panic!("Unexpected command");
    };
    let distances: Vec<f64> = iter_num_tools::arange(*rmin..*rmax, *dr).collect();
    let n_points = (4.0 * PI / resolution.powi(2)).round() as usize;
    let mut icotable = IcoTable2D::<f64>::from_min_points(n_points)?;
    let resolution = (4.0 * PI / icotable.len() as f64).sqrt();
    log::info!(
        "Requested {} points on a sphere; got {} -> new resolution = {:.3}",
        n_points,
        icotable.len(),
        resolution
    );

    let mut dipole_file = File::create(output)?;
    writeln!(dipole_file, "# R/Å w_vertex w_exact w_interpolated")?;

    let charge = 1.0;
    let bjerrum_len = 7.0;

    // for each ion-dipole separation, calculate the partition function and free energy
    for radius in distances {
        // exact exp. energy at a given point, exp(-βu)
        let exact_exp_energy = |_, p: &Vector3| {
            let (_r, theta, _phi) = SphericalCoord::from_cartesian(*p).into();
            let field = bjerrum_len * charge / radius.powi(2);
            let energy_in_kt = field * dipole_moment * theta.cos();
            energy_in_kt.neg().exp()
        };
        icotable.clear_vertex_data();
        icotable.set_vertex_data(exact_exp_energy)?;

        // Q summed from exact data at each vertex
        let partition_function = icotable.vertex_data().sum::<f64>() / icotable.len() as f64;

        // analytical solution to angular average of exp(-βu)
        let field = -bjerrum_len * charge / radius.powi(2);
        let exact_free_energy = ((field * dipole_moment).sinh() / (field * dipole_moment))
            .ln()
            .neg();

        // rotations to apply to vertices of a new icosphere used for sampling interpolated points
        let mut rng = rand::thread_rng();
        let quaternions: Vec<UnitQuaternion> = (0..20)
            .map(|_| {
                let point = faunus::transform::random_unit_vector(&mut rng);
                UnitQuaternion::from_axis_angle(
                    &nalgebra::Unit::new_normalize(point),
                    rng.gen_range(0.0..PI),
                )
            })
            .collect();

        // Sample interpolated points using a randomly rotate icospheres
        let mut rotated_icosphere = IcoTable2D::<f64>::from_min_points(1000)?;
        let mut partition_func_interpolated = 0.0;

        for q in &quaternions {
            rotated_icosphere.transform_vertex_positions(|v| q.transform_vector(v));
            partition_func_interpolated += rotated_icosphere
                .iter()
                .map(|(pos, _)| icotable.interpolate(pos))
                .sum::<f64>()
                / rotated_icosphere.len() as f64;
        }
        partition_func_interpolated /= quaternions.len() as f64;

        writeln!(
            dipole_file,
            "{:.5} {:.5} {:.5} {:.5}",
            radius,
            partition_function.ln().neg(),
            exact_free_energy,
            partition_func_interpolated.ln().neg(),
        )?;
    }
    Ok(())
}

// Calculate electric potential at points on a sphere around a molecule
fn do_potential(cmd: &Commands) -> Result<()> {
    let Commands::Potential {
        mol1,
        resolution,
        radius,
        topology,
        molarity,
        cutoff: _,
        temperature,
    } = cmd
    else {
        panic!("Unexpected command");
    };
    let mut topology = Topology::from_file_partial(topology)?;
    faunus::topology::set_missing_epsilon(topology.atomkinds_mut(), 2.479);

    let structure = Structure::from_xyz(mol1, topology.atomkinds())?;

    let n_points = (4.0 * PI / resolution.powi(2)).round() as usize;
    let vertices = duello::make_icosphere_vertices(n_points)?;
    let resolution = (4.0 * PI / vertices.len() as f64).sqrt();
    log::info!(
        "Requested {} points on a sphere; got {} -> new resolution = {:.2}",
        n_points,
        vertices.len(),
        resolution
    );

    // Electrolyte background
    let medium = Medium::salt_water(*temperature, Salt::SodiumChloride, *molarity);
    let multipole = Plain::new(f64::INFINITY, medium.debye_length());

    let icotable = IcoTable2D::<f64>::from_min_points(n_points)?;
    icotable.set_vertex_data(|_, v| {
        energy::electric_potential(&structure, &v.scale(*radius), &multipole)
    })?;

    File::create("pot_at_vertices.dat")?.write_fmt(format_args!("{icotable}"))?;

    // Make PQR file illustrating the electric potential at each vertex
    let mut pqr_file = File::create("potential.pqr")?;
    for (vertex_pos, data) in std::iter::zip(icotable.iter_positions(), icotable.vertex_data()) {
        pqr_write_atom(&mut pqr_file, 1, &vertex_pos.scale(*radius), *data, 2.0)?;
    }

    // Compare interpolated and exact potential linearly in angular space
    let mut pot_angles_file = File::create("pot_at_angles.dat")?;
    let mut pqr_file = File::create("potential_angles.pqr")?;
    writeln!(pot_angles_file, "# theta phi interpolated exact relerr")?;
    for theta in arange(0.0001..PI, resolution) {
        for phi in arange(0.0001..2.0 * PI, resolution) {
            let point: Vector3 = SphericalCoord::new(1.0, theta, phi).into();
            let interpolated = icotable.interpolate(&point);
            let exact = energy::electric_potential(&structure, &point.scale(*radius), &multipole);
            pqr_write_atom(&mut pqr_file, 1, &point.scale(*radius), exact, 2.0)?;
            let rel_err = (interpolated - exact) / exact;
            let abs_err = (interpolated - exact).abs();
            if abs_err > 0.05 {
                log::debug!(
                    "Potential at theta={theta:.3} phi={phi:.3} is {interpolated:.4} (exact: {exact:.4}) abs. error {abs_err:.4}"
                );
                let face = icotable.nearest_face(&point);
                let bary = icotable.naive_barycentric(&point, &face);
                log::debug!("Face: {face:?} Barycentric: {bary:?}\n");
            }
            writeln!(
                pot_angles_file,
                "{theta:.3} {phi:.3} {interpolated:.4} {exact:.4} {rel_err:.4}"
            )?;
        }
    }
    Ok(())
}

fn do_diffusion(cmd: &Commands) -> Result<()> {
    let Commands::Diffusion {
        table: table_path,
        temperature,
        output,
        symmetric,
    } = cmd
    else {
        bail!("expected Diffusion command");
    };

    info!("Loading table from {}", table_path.display());
    let table = icotable::Table6DAdaptive::<f32>::load(table_path)?;

    let temp = *temperature;

    let beta = 1.0 / (physical_constants::MOLAR_GAS_CONSTANT * 1e-3 * temp);
    info!(
        "T = {temp:.2} K, β = {beta:.4} mol/kJ, R range = {:.1}–{:.1} Å, dr = {:.2} Å",
        table.rmin, table.rmax, table.dr
    );

    let results = duello::diffusion::diffusion_scan(&table, beta, *symmetric);

    if results.is_empty() {
        bail!("No R-slices yielded diffusion results");
    }

    let mut file = File::create(output)?;

    // Build header with dynamic eigenvalue columns
    let max_evals = results.iter().map(|r| r.eigenvalues.len()).max().unwrap_or(0);
    let max_evals_free = results
        .iter()
        .map(|r| r.eigenvalues_free.len())
        .max()
        .unwrap_or(0);
    let mut header =
        "R,D/D⁰,D_A/D_A⁰,D_B/D_B⁰,D_ω/D_ω⁰".to_string();
    for i in 1..=max_evals {
        header.push_str(&format!(",λ{i}"));
    }
    for i in 1..=max_evals_free {
        header.push_str(&format!(",λ{i}_free"));
    }
    header.push_str(",n_active");
    writeln!(file, "{header}")?;

    for r in &results {
        write!(
            file,
            "{:.2},{:.6e},{:.6e},{:.6e},{:.6e}",
            r.r, r.dr_normalized, r.dr_mol_a, r.dr_mol_b, r.dr_omega
        )?;
        for i in 0..max_evals {
            if let Some(&e) = r.eigenvalues.get(i) {
                write!(file, ",{e:.6e}")?;
            } else {
                write!(file, ",")?;
            }
        }
        for i in 0..max_evals_free {
            if let Some(&e) = r.eigenvalues_free.get(i) {
                write!(file, ",{e:.6e}")?;
            } else {
                write!(file, ",")?;
            }
        }
        writeln!(file, ",{}", r.n_active)?;
    }

    info!("Wrote {} R-slices to {}", results.len(), output.display());
    Ok(())
}

// Wrapper for main function to handle errors
fn do_main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        Some(cmd) => match cmd {
            Commands::AtomScan { .. } => do_atom_scan(&cmd)?,
            Commands::Diffusion { .. } => do_diffusion(&cmd)?,
            Commands::Dipole { .. } => do_dipole(&cmd)?,
            Commands::Scan { .. } => do_scan(&cmd)?,
            Commands::Potential { .. } => do_potential(&cmd)?,
        },
        None => {
            bail!("No command given");
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    if let Err(err) = do_main() {
        eprintln!("Error: {}", &err);
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

//! High-level API for running scans from in-memory PDB data.
//!
//! Hides faunus internals so that external crates (duello-web) don't need
//! to depend on faunus types directly.

use crate::backend::{EnergyBackend, GpuBackend};
use crate::energy::{CoulombParams, PairMatrix, SplinedPotentials};
use crate::icoscan::{compute_scan, compute_scan_async, ScanParams};
use crate::loader::{load_molecule_from_pdb_bytes, CgOptions, LoadedMoleculeInMemory};
use crate::report::compute_pmf;
use crate::VirialCoeff;
use faunus::energy::NonbondedMatrix;
use faunus::interatomic::coulomb::{DebyeLength as _, Medium, Salt};
use faunus::interatomic::twobody::SplineConfig;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Parameters for a web-initiated scan (no file paths).
pub struct WebScanRequest {
    pub mol1_pdb: Vec<u8>,
    pub mol2_pdb: Vec<u8>,
    pub ph: f64,
    pub model: String,
    pub single_bead: bool,
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    pub molarity: f64,
    pub temperature: f64,
    pub max_ndiv: usize,
    pub gradient_threshold: f64,
    pub cutoff: f64,
    pub grid_size: usize,
    pub homo_dimer: bool,
}

/// Result of a web scan.
pub struct WebScanResult {
    pub pmf_data: Vec<(f32, f32)>,
    pub mean_energy_data: Vec<(f32, f32)>,
    pub virial: VirialCoeff,
    /// Molecular weights in g/mol (mol1, mol2)
    pub molar_masses: (f64, f64),
    /// Net charges in elementary charges (mol1, mol2)
    pub net_charges: (f64, f64),
    /// CG XYZ file contents (mol1, mol2)
    pub cg_xyz: (String, String),
    /// Topology YAML string
    pub topology_yaml: String,
}

struct PreparedScan {
    sr_splines: SplinedPotentials,
    coulomb: CoulombParams,
    mol1: LoadedMoleculeInMemory,
    mol2: LoadedMoleculeInMemory,
    medium: Medium,
}

fn prepare_scan(req: &WebScanRequest) -> anyhow::Result<PreparedScan> {
    let cg_opts = CgOptions {
        ph: req.ph,
        model: req.model.clone(),
        single_bead: req.single_bead,
        ..Default::default()
    };

    log::info!("Coarse-graining molecule 1...");
    let mol1 = load_molecule_from_pdb_bytes(&req.mol1_pdb, &cg_opts)?;
    log::info!("Coarse-graining molecule 2...");
    let mol2 = load_molecule_from_pdb_bytes(&req.mol2_pdb, &cg_opts)?;

    let medium = Medium::salt_water(req.temperature, Salt::SodiumChloride, req.molarity);

    let nonbonded =
        NonbondedMatrix::from_str(&mol1.topology_yaml, &mol1.topology, Some(medium.clone()))?;

    let pair_matrix = PairMatrix::new(nonbonded, mol1.topology.atomkinds(), &medium);

    let spline_config = SplineConfig {
        n_points: req.grid_size,
        shift_energy: true,
        grid_type: faunus::interatomic::twobody::GridType::PowerLaw2,
        ..Default::default()
    };
    let sr_splines = pair_matrix.create_sr_splines(req.cutoff, spline_config);
    let coulomb = pair_matrix.coulomb_params().clone();

    Ok(PreparedScan {
        sr_splines,
        coulomb,
        mol1,
        mol2,
        medium,
    })
}

fn build_scan_params(req: &WebScanRequest, prep: &PreparedScan) -> ScanParams {
    ScanParams {
        rmin: req.rmin,
        rmax: req.rmax,
        dr: req.dr,
        temperature: req.temperature,
        charges: [
            prep.mol1.structure.net_charge(),
            prep.mol2.structure.net_charge(),
        ],
        dipole_moments: [
            prep.mol1.structure.dipole_moment().norm(),
            prep.mol2.structure.dipole_moment().norm(),
        ],
        kappa: prep.medium.debye_length().map(|dl| 1.0 / dl),
        permittivity: prep.medium.permittivity(),
        max_n_div: req.max_ndiv,
        gradient_threshold: req.gradient_threshold,
        homo_dimer: req.homo_dimer,
    }
}

fn structure_to_xyz(mol: &LoadedMoleculeInMemory) -> String {
    let mut buf = Vec::new();
    mol.structure
        .to_xyz(&mut buf, mol.topology.atomkinds())
        .ok();
    String::from_utf8(buf).unwrap_or_default()
}

fn build_result(prep: &PreparedScan, pmf_result: crate::report::PmfResult) -> WebScanResult {
    WebScanResult {
        pmf_data: pmf_result.pmf_data,
        mean_energy_data: pmf_result.mean_energy_data,
        virial: pmf_result.virial,
        molar_masses: (
            prep.mol1.structure.masses.iter().sum(),
            prep.mol2.structure.masses.iter().sum(),
        ),
        net_charges: (
            prep.mol1.structure.net_charge(),
            prep.mol2.structure.net_charge(),
        ),
        cg_xyz: (structure_to_xyz(&prep.mol1), structure_to_xyz(&prep.mol2)),
        topology_yaml: prep.mol1.topology_yaml.clone(),
    }
}

fn finish_scan(
    req: &WebScanRequest,
    prep: &PreparedScan,
    backend: &(impl EnergyBackend + Sync),
) -> anyhow::Result<WebScanResult> {
    let scan_params = build_scan_params(req, prep);

    log::info!("Running 6D scan...");
    let scan_result = compute_scan(&scan_params, backend)?;
    let pmf_result = compute_pmf(&scan_result.samples)?;
    Ok(build_result(prep, pmf_result))
}

/// Run the complete scan pipeline asynchronously (required for WASM).
///
/// The optional `progress` callback is called with `(current_r, total_r)` after each R-bin.
pub async fn run_scan_async(
    req: WebScanRequest,
    progress: Option<&dyn Fn(usize, usize)>,
    cancel: Option<&Arc<AtomicBool>>,
) -> anyhow::Result<WebScanResult> {
    let prep = prepare_scan(&req)?;

    log::info!("Initializing GPU backend...");
    let backend = GpuBackend::new_async(
        prep.mol1.structure.clone(),
        prep.mol2.structure.clone(),
        &prep.sr_splines,
        &prep.coulomb,
    )
    .await?;

    let scan_params = build_scan_params(&req, &prep);

    log::info!("Running 6D scan (async)...");
    let scan_result = compute_scan_async(&scan_params, &backend, progress, cancel).await?;

    let pmf_result = compute_pmf(&scan_result.samples)?;
    Ok(build_result(&prep, pmf_result))
}

/// Run the complete scan pipeline from PDB bytes to PMF results.
///
/// This is the primary entry point for the native desktop UI.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_scan(req: WebScanRequest) -> anyhow::Result<WebScanResult> {
    let prep = prepare_scan(&req)?;

    log::info!("Initializing GPU backend...");
    let backend = GpuBackend::new(
        prep.mol1.structure.clone(),
        prep.mol2.structure.clone(),
        &prep.sr_splines,
        &prep.coulomb,
    )?;

    finish_scan(&req, &prep, &backend)
}

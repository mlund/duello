//! High-level API for running scans from in-memory PDB data.
//!
//! Hides faunus internals so that external crates (duello-web) don't need
//! to depend on faunus types directly.

#[cfg(not(target_arch = "wasm32"))]
use crate::backend::EnergyBackend;
use crate::backend::GpuBackend;
use crate::energy::SplinedPotentials;
#[cfg(not(target_arch = "wasm32"))]
use crate::icoscan::compute_scan;
use crate::icoscan::{compute_scan_async, ScanParams};
pub use crate::icoscan::{OnProgress, ScanProgress};
use crate::report::compute_pmf;
pub use crate::report::PmfPoint;
use crate::structure::Structure;
use crate::system::{CgOptions, PairSpec, PairSystem, Source};
use crate::VirialCoeff;
use faunus::interatomic::coulomb::{DebyeLength as _, Medium, Salt};
use faunus::interatomic::twobody::SplineConfig;
use faunus::topology::Topology;

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
    /// Same points as streamed through [`ScanProgress`] during the scan.
    pub points: Vec<PmfPoint>,
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
    system: PairSystem,
    medium: Medium,
}

fn prepare_scan(req: &WebScanRequest) -> anyhow::Result<PreparedScan> {
    if req.homo_dimer {
        log::info!("Homo-dimer: reusing molecule 1 coarse-graining for molecule 2");
    }
    log::info!("Coarse-graining...");
    let medium = Medium::salt_water(req.temperature, Salt::SodiumChloride, req.molarity);
    let system = PairSystem::load(PairSpec {
        a: Source::PdbBytes(&req.mol1_pdb),
        b: (!req.homo_dimer).then_some(Source::PdbBytes(&req.mol2_pdb)),
        top: None,
        cg: CgOptions {
            ph: req.ph,
            model: req.model.clone(),
            single_bead: req.single_bead,
            ..Default::default()
        },
        medium: &medium,
    })?;

    let spline_config = SplineConfig {
        n_points: req.grid_size,
        shift_energy: true,
        grid_type: faunus::interatomic::twobody::GridType::PowerLaw2,
        ..Default::default()
    };
    let sr_splines = system
        .pair_matrix
        .create_sr_splines(req.cutoff, spline_config);

    Ok(PreparedScan {
        sr_splines,
        system,
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
            prep.system.ref_a.net_charge(),
            prep.system.ref_b.net_charge(),
        ],
        dipole_moments: [
            prep.system.ref_a.dipole_moment().norm(),
            prep.system.ref_b.dipole_moment().norm(),
        ],
        kappa: prep.medium.debye_length().map(|dl| 1.0 / dl),
        permittivity: prep.medium.permittivity(),
        max_n_div: req.max_ndiv,
        gradient_threshold: req.gradient_threshold,
        subdivision: icotable::Subdivision::default(),
        homo_dimer: req.homo_dimer,
    }
}

fn structure_to_xyz(structure: &Structure, topology: &Topology) -> String {
    let mut buf = Vec::new();
    structure.to_xyz(&mut buf, topology.atomkinds()).ok();
    String::from_utf8(buf).unwrap_or_default()
}

fn build_result(prep: &PreparedScan, pmf_result: crate::report::PmfResult) -> WebScanResult {
    WebScanResult {
        points: pmf_result.points,
        virial: pmf_result.virial,
        molar_masses: (
            prep.system.ref_a.masses.iter().sum(),
            prep.system.ref_b.masses.iter().sum(),
        ),
        net_charges: (
            prep.system.ref_a.net_charge(),
            prep.system.ref_b.net_charge(),
        ),
        cg_xyz: (
            structure_to_xyz(&prep.system.ref_a, &prep.system.topology),
            structure_to_xyz(&prep.system.ref_b, &prep.system.topology),
        ),
        topology_yaml: prep
            .system
            .generated
            .as_ref()
            .map(|cg| cg.topology.clone())
            .unwrap_or_default(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn finish_scan(
    req: &WebScanRequest,
    prep: &PreparedScan,
    backend: &(impl EnergyBackend + Sync),
    progress: OnProgress,
) -> anyhow::Result<WebScanResult> {
    let scan_params = build_scan_params(req, prep);

    log::info!("Running 6D scan...");
    let scan_result = compute_scan(&scan_params, backend, progress)?;
    let pmf_result = compute_pmf(&scan_result.samples)?;
    Ok(build_result(prep, pmf_result))
}

/// Run the complete scan pipeline asynchronously (required for WASM).
pub async fn run_scan_async(
    req: WebScanRequest,
    progress: OnProgress<'_>,
) -> anyhow::Result<WebScanResult> {
    let prep = prepare_scan(&req)?;

    log::info!("Initializing GPU backend...");
    let backend = GpuBackend::new_async(
        prep.system.ref_a.clone(),
        prep.system.ref_b.clone(),
        &prep.sr_splines,
        prep.system.pair_matrix.coulomb_params(),
    )
    .await?;

    let scan_params = build_scan_params(&req, &prep);

    log::info!("Running 6D scan (async)...");
    let scan_result = compute_scan_async(&scan_params, &backend, progress).await?;

    let pmf_result = compute_pmf(&scan_result.samples)?;
    Ok(build_result(&prep, pmf_result))
}

/// Run the complete scan pipeline from PDB bytes to PMF results.
///
/// This is the primary entry point for the native desktop UI.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_scan(req: WebScanRequest, progress: OnProgress) -> anyhow::Result<WebScanResult> {
    let prep = prepare_scan(&req)?;

    log::info!("Initializing GPU backend...");
    let backend = GpuBackend::new(
        prep.system.ref_a.clone(),
        prep.system.ref_b.clone(),
        &prep.sr_splines,
        prep.system.pair_matrix.coulomb_params(),
    )?;

    finish_scan(&req, &prep, &backend, progress)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed-column PDB with backbone + CB per residue; residues are spaced along x.
    fn pdb(residues: &[&str]) -> Vec<u8> {
        let atoms = [
            ("N", -1.2),
            ("CA", 0.0),
            ("C", 1.2),
            ("O", 1.8),
            ("CB", 0.4),
        ];
        let mut out = String::new();
        let mut serial = 1;
        for (i, res) in residues.iter().enumerate() {
            for (name, dx) in atoms {
                let (x, y) = (3.8 * i as f64 + dx, if name == "CB" { 1.5 } else { 0.0 });
                let element = &name[..1];
                out += &format!(
                    "ATOM  {serial:>5}  {name:<3} {res:>3} A{seq:>4}    {x:>8.3}{y:>8.3}{z:>8.3}  1.00  0.00          {element:>2}\n",
                    seq = i + 1,
                    z = 0.0,
                );
                serial += 1;
            }
        }
        out.into_bytes()
    }

    fn request(mol1: &[&str], mol2: &[&str]) -> WebScanRequest {
        WebScanRequest {
            mol1_pdb: pdb(mol1),
            mol2_pdb: pdb(mol2),
            ph: 7.0,
            model: "calvados3".into(),
            single_bead: true,
            rmin: 20.0,
            rmax: 24.0,
            dr: 2.0,
            molarity: 0.05,
            temperature: 298.15,
            max_ndiv: 0,
            gradient_threshold: 0.5,
            cutoff: 30.0,
            grid_size: 200,
            homo_dimer: mol1 == mol2,
        }
    }

    #[test]
    fn homo_dimer_reuses_molecule_1() {
        let peptide = ["LEU", "PHE", "LYS", "GLU"];
        let prep = prepare_scan(&request(&peptide, &peptide)).unwrap();
        assert_eq!(prep.system.ref_b.atom_ids, prep.system.ref_a.atom_ids);
        assert_eq!(prep.system.ref_b.pos, prep.system.ref_a.pos);
    }

    /// Molecule 2 has ALA, absent from molecule 1; both must index one kind table
    /// with B's beads resolving to their own kinds (gh #39).
    #[test]
    fn hetero_dimer_shares_kind_table() {
        let prep = prepare_scan(&request(&["LEU", "LYS"], &["ALA", "LYS"])).unwrap();
        let kinds = prep.system.topology.atomkinds();
        let names: Vec<_> = prep
            .system
            .ref_b
            .atom_ids
            .iter()
            .map(|&id| kinds[id].name())
            .collect();
        assert!(names.contains(&"ALA"), "{names:?}");
        assert!(!names.contains(&"LEU"), "{names:?}");
    }

    /// Kim-Hummer lists pairs explicitly, so a kind only B has (ALA) must still meet
    /// A's kinds in the pair table, or that contact would cost nothing.
    #[test]
    fn hetero_dimer_cross_pairs_interact() {
        use faunus::interatomic::twobody::IsotropicTwobodyEnergy;
        use faunus::topology::FindByName;
        let mut req = request(&["LEU", "LYS"], &["ALA", "LYS"]);
        req.model = "kimhummer".into();
        let prep = prepare_scan(&req).unwrap();
        let kinds = prep.system.topology.atomkinds();
        let id = |name| kinds.find_name(name).unwrap().id();
        let u = prep.sr_splines.get(id("LEU"), id("ALA"));
        assert!(u.isotropic_twobody_energy(36.0) != 0.0);
    }
}

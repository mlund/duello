//! High-level API for running scans from in-memory PDB data.
//!
//! Hides faunus internals so that external crates (duello-web) don't need
//! to depend on faunus types directly.

pub use crate::archive::{CgBody, ScanArchive, ScanSettings};
#[cfg(not(target_arch = "wasm32"))]
use crate::backend::EnergyBackend;
use crate::backend::GpuBackend;
use crate::energy::SplinedPotentials;
#[cfg(not(target_arch = "wasm32"))]
use crate::icoscan::compute_scan;
use crate::icoscan::{compute_scan_async, ScanParams, ScanResult};
pub use crate::icoscan::{OnProgress, ScanProgress};
pub use crate::report::PmfPoint;
use crate::system::{CgOptions, PairSpec, PairSystem, Source};
use faunus::interatomic::coulomb::{DebyeLength as _, Medium, Salt};
use faunus::interatomic::twobody::SplineConfig;

/// Parameters for a web-initiated scan (no file paths). The settings are the
/// same type the archive records, so nothing needs copying field by field.
pub struct WebScanRequest {
    pub mol1_pdb: Vec<u8>,
    pub mol2_pdb: Vec<u8>,
    pub settings: ScanSettings,
}

/// A web scan's result is the exportable archive itself, so what the UI shows
/// and what it exports can't drift apart.
pub type WebScanResult = ScanArchive;

struct PreparedScan {
    sr_splines: SplinedPotentials,
    system: PairSystem,
    medium: Medium,
}

fn prepare_scan(req: &WebScanRequest) -> anyhow::Result<PreparedScan> {
    let settings = &req.settings;
    if settings.homo_dimer {
        log::info!("Homo-dimer: reusing molecule 1 coarse-graining for molecule 2");
    }
    log::info!("Coarse-graining...");
    let medium = Medium::salt_water(
        settings.temperature,
        Salt::SodiumChloride,
        settings.molarity,
    );
    let system = PairSystem::load(PairSpec {
        a: Source::PdbBytes(&req.mol1_pdb),
        b: (!settings.homo_dimer).then_some(Source::PdbBytes(&req.mol2_pdb)),
        top: None,
        cg: CgOptions {
            ph: settings.ph,
            model: settings.model.clone(),
            single_bead: settings.single_bead,
            ..Default::default()
        },
        medium: &medium,
    })?;

    let spline_config = SplineConfig {
        n_points: settings.grid_size,
        shift_energy: true,
        grid_type: faunus::interatomic::twobody::GridType::PowerLaw2,
        ..Default::default()
    };
    let sr_splines = system
        .pair_matrix
        .create_sr_splines(settings.cutoff, spline_config);

    Ok(PreparedScan {
        sr_splines,
        system,
        medium,
    })
}

fn build_scan_params(req: &WebScanRequest, prep: &PreparedScan) -> ScanParams {
    let settings = &req.settings;
    ScanParams {
        rmin: settings.rmin,
        rmax: settings.rmax,
        dr: settings.dr,
        temperature: settings.temperature,
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
        max_n_div: settings.max_ndiv,
        gradient_threshold: settings.gradient_threshold,
        subdivision: settings.subdivision,
        homo_dimer: settings.homo_dimer,
    }
}

fn build_result(
    req: &WebScanRequest,
    prep: &PreparedScan,
    params: &ScanParams,
    scan: ScanResult,
) -> anyhow::Result<ScanArchive> {
    let system = &prep.system;
    let topology_yaml = system.generated.as_ref().map(|cg| cg.topology.clone());
    ScanArchive::from_scan(
        req.settings.clone(),
        params,
        scan,
        [&system.ref_a, &system.ref_b],
        system.topology.atomkinds(),
        topology_yaml.unwrap_or_default(),
    )
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
    build_result(req, prep, &scan_params, scan_result)
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
    build_result(&req, &prep, &scan_params, scan_result)
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
    use crate::Subdivision;

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
            settings: ScanSettings {
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
                subdivision: Subdivision::default(),
            },
        }
    }

    /// Runs the scan on the CPU so tests need no GPU adapter.
    fn cpu_scan(req: &WebScanRequest) -> ScanArchive {
        let prep = prepare_scan(req).unwrap();
        let energy = prepare_scan(req).unwrap().system;
        let backend =
            crate::backend::cpu::CpuBackend::new(energy.ref_a, energy.ref_b, energy.pair_matrix);
        finish_scan(req, &prep, &backend, None).unwrap()
    }

    #[test]
    fn scan_result_is_a_self_contained_archive() {
        let peptide = ["LYS", "LYS", "LYS"];
        let mut req = request(&peptide, &peptide);
        req.settings.rmax = 40.0; // the tail fit needs at least five R bins
        let archive = cpu_scan(&req);

        assert!(archive.settings.homo_dimer);
        assert_eq!(archive.settings.ph, 7.0);
        // Single-bead CG adds separate terminal beads (N, O).
        assert_eq!(archive.bodies[0].kinds, ["LYS", "N", "LYS", "LYS", "O"]);
        assert_eq!(archive.bodies[0].positions.len(), 5);

        let meta = archive.table.metadata.as_ref().expect("tail metadata");
        assert_eq!(meta.temperature, Some(298.15));
        let yukawa = &meta.tail_terms[0];
        let z = archive.net_charges().0;
        approx::assert_relative_eq!(yukawa.coefficient, z * z, max_relative = 1e-9);
        // Debye length at 50 mM NaCl, 298.15 K is about 13.6 Å.
        approx::assert_relative_eq!(yukawa.kappa, 1.0 / 13.6, max_relative = 0.02);

        let restored = ScanArchive::from_bytes(&archive.to_bytes().unwrap()).unwrap();
        assert_eq!(restored.bodies, archive.bodies);
    }

    #[test]
    fn scan_uses_and_records_the_requested_subdivision() {
        let peptide = ["LYS", "LYS", "LYS"];
        let mut req = request(&peptide, &peptide);
        req.settings.rmax = 40.0;
        let geodesic = cpu_scan(&req);
        assert_eq!(geodesic.table.levels[0].subdivision(), Subdivision::Geodesic);
        req.settings.subdivision = Subdivision::Lattice;
        let archive = cpu_scan(&req);
        let restored = ScanArchive::from_bytes(&archive.to_bytes().unwrap()).unwrap();
        assert_eq!(restored.settings.subdivision, Subdivision::Lattice);
        // The table must really be built on the lattice, not just labelled so.
        let level = &restored.table.levels[0];
        assert_eq!(level.subdivision(), Subdivision::Lattice);
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
        req.settings.model = "kimhummer".into();
        let prep = prepare_scan(&req).unwrap();
        let kinds = prep.system.topology.atomkinds();
        let id = |name| kinds.find_name(name).unwrap().id();
        let u = prep.sr_splines.get(id("LEU"), id("ALA"));
        assert!(u.isotropic_twobody_energy(36.0) != 0.0);
    }
}

//! Shared fixtures for backend tests: lysozyme pairs scored by the exact CPU path.

use super::{cpu::CpuBackend, PoseParams};
use crate::energy::{CoulombParams, PairMatrix, SplinedPotentials};
use crate::structure::Structure;
use crate::system::{CgOptions, PairSpec, PairSystem, Source};
use crate::Vector3;
use faunus::interatomic::coulomb::{Medium, Salt};
use faunus::interatomic::twobody::{GridType, SplineConfig};
use std::path::{Path, PathBuf};

const TEMPERATURE: f64 = 298.15;

/// kJ/mol at the fixture temperature.
pub fn kt() -> f64 {
    crate::thermal_energy(TEMPERATURE)
}

/// A homodimer with its splines and an exact reference.
pub struct TestPair {
    pub ref_a: Structure,
    pub ref_b: Structure,
    pub splines: SplinedPotentials,
    pub coulomb: CoulombParams,
    pair_matrix: PairMatrix,
}

impl TestPair {
    /// Exact (unsplined) reference for any structures typed by this force field.
    pub fn into_exact(self, a: Structure, b: Structure) -> CpuBackend {
        CpuBackend::new(a, b, self.pair_matrix)
    }

    /// Atom types in A, split by whether they carry charge.
    pub fn types_by_charge(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.splines.n_types();
        self.ref_a
            .unique_atom_types()
            .into_iter()
            .partition(|&t| self.coulomb.prefactors[t * n + t] == 0.0)
    }
}

/// GPU tests skip with a note on machines without an adapter.
pub fn no_gpu() -> bool {
    let missing = !crate::backend::gpu_context::GpuContext::is_available();
    if missing {
        eprintln!("no GPU adapter; skipping");
    }
    missing
}

/// One bead of type `id` at the origin.
pub fn bead(id: usize) -> Structure {
    Structure {
        pos: vec![Vector3::zeros()],
        masses: vec![1.0],
        charges: vec![0.0],
        radii: vec![1.0],
        atom_ids: vec![id],
    }
}

fn calvados_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/calvados3")
}

/// The CLAUDE.md benchmark system: 2CDS × 2CDS, 20 mM NaCl.
pub fn lysozyme() -> TestPair {
    lysozyme_with_topology(&calvados_dir().join("topology.yaml"))
}

/// Lysozyme with its Ashbaugh–Hatch pairs cut at `range` (Å) instead of 20 Å.
pub fn lysozyme_with_range(range: f64) -> TestPair {
    let yaml =
        std::fs::read_to_string(calvados_dir().join("topology.yaml")).expect("topology reads");
    assert!(yaml.contains("cutoff: 20.0"), "fixture's AH range moved");
    let top = std::env::temp_dir().join(format!(
        "duello-lysozyme-{range}-{}.yaml",
        std::process::id()
    ));
    std::fs::write(
        &top,
        yaml.replace("cutoff: 20.0", &format!("cutoff: {range}")),
    )
    .unwrap();
    let pair = lysozyme_with_topology(&top);
    let _ = std::fs::remove_file(top);
    pair
}

fn lysozyme_with_topology(top: &Path) -> TestPair {
    let xyz = calvados_dir().join("2CDS.xyz");
    let medium = Medium::salt_water(TEMPERATURE, Salt::SodiumChloride, 0.02);
    let spec = PairSpec {
        a: Source::File(&xyz),
        b: None,
        top: Some(top),
        chains: [&[], &[]],
        cg: CgOptions::default(),
        medium: &medium,
    };
    load(spec)
}

/// Temperature (K) of the chymotrypsinogen fixture.
pub const CGN_TEMPERATURE: f64 = 300.0;

/// Chymotrypsinogen A (7KTZ) at pH 7, 15 mM, coarse-grained
/// with `model`: a pair whose B₂ is set by contact attraction, unlike
/// lysozyme's excluded volume.
pub fn chymotrypsinogen(model: &str, single_bead: bool) -> TestPair {
    let pdb = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/calvados3/7KTZ.pdb");
    let medium = Medium::salt_water(CGN_TEMPERATURE, Salt::SodiumChloride, 0.015);
    let cg = CgOptions {
        model: model.to_string(),
        single_bead,
        ..Default::default()
    };
    let spec = PairSpec {
        a: Source::File(&pdb),
        b: None,
        top: None,
        chains: [&[], &[]],
        cg,
        medium: &medium,
    };
    load(spec)
}

fn load(spec: PairSpec) -> TestPair {
    let system = PairSystem::load(spec).expect("example loads");
    TestPair {
        splines: sr_splines(&system.pair_matrix),
        coulomb: system.pair_matrix.coulomb_params().clone(),
        ref_a: system.ref_a,
        ref_b: system.ref_b,
        pair_matrix: system.pair_matrix,
    }
}

/// Short-range splines as the CLI builds them by default.
pub fn sr_splines(pair_matrix: &PairMatrix) -> SplinedPotentials {
    let config = SplineConfig {
        n_points: 200,
        grid_type: GridType::PowerLaw2,
        ..Default::default()
    };
    pair_matrix
        .create_sr_splines(config)
        .expect("splines build")
}

/// 21 poses (3 R × 7 orientations) spanning contact (30 Å), mid (40 Å) and tail (55 Å) separations.
pub fn pose_list() -> Vec<PoseParams> {
    let orientations = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0),
        ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], 0.7),
        ([0.3, 0.7, 0.5], [-0.5, 0.2, 0.8], 2.1),
        ([-0.6, 0.1, -0.4], [0.9, -0.3, 0.1], 4.0),
        ([0.2, -0.9, 0.3], [0.1, 0.1, 0.95], 5.5),
        ([-0.4, -0.4, 0.8], [-0.7, 0.6, -0.2], 1.3),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 3.1),
    ];
    let v = |a: [f64; 3]| Vector3::new(a[0], a[1], a[2]);
    [30.0, 40.0, 55.0]
        .into_iter()
        .flat_map(|r| {
            orientations.iter().map(move |&(vi, vj, omega)| PoseParams {
                r,
                omega,
                vertex_i: v(vi),
                vertex_j: v(vj),
            })
        })
        .collect()
}

/// A vertex vector uniform in the unit ball, so its direction is uniform on
/// the sphere (`PoseParams` normalises it); never near zero.
pub fn random_direction(rng: &mut rand_pcg::Pcg64) -> Vector3 {
    use rand::Rng;
    loop {
        let v = Vector3::from_fn(|_, _| rng.gen_range(-1.0..1.0));
        if (1e-6..1.0).contains(&v.norm_squared()) {
            return v;
        }
    }
}

/// Random orientations at R = `r_min` … `r_min` + 3 Å; seeded, so runs agree.
pub fn contact_poses(n: usize, r_min: f64) -> Vec<PoseParams> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_pcg::Pcg64::seed_from_u64(12345);
    let direction = random_direction;
    (0..n)
        .map(|i| PoseParams {
            r: r_min + (i % 4) as f64,
            omega: rng.gen_range(0.0..std::f64::consts::TAU),
            vertex_i: direction(&mut rng),
            vertex_j: direction(&mut rng),
        })
        .collect()
}

/// Kernel changes must keep agreeing with the exact pair sum. Poses that carry
/// Boltzmann weight match to the spline error (1e-3, with a floor for f32 noise
/// near U = 0). Overlaps differ by the spline's linear extrapolation below
/// r_min, so they only need to stay repulsive.
pub fn assert_matches_exact(label: &str, poses: &[PoseParams], got: &[f64], exact: &CpuBackend) {
    use super::EnergyBackend;
    let u_weighted = 10.0 * kt();
    let mut n_weighted = 0;
    for (pose, &u) in poses.iter().zip(got) {
        let u_exact = exact.compute_energy(pose);
        let weighted = u_exact < u_weighted;
        n_weighted += usize::from(weighted);
        let ok = if weighted {
            (u - u_exact).abs() < 1e-3 * u_exact.abs() + 0.05
        } else {
            u > u_weighted
        };
        assert!(
            ok,
            "R={} ω={}: {label} {u:.4} vs exact {u_exact:.4} kJ/mol",
            pose.r, pose.omega
        );
    }
    // Guards against a pose list that only exercises the repulsive branch.
    assert!(n_weighted >= 14, "only {n_weighted} weighted poses");
}

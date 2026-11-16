//! Shared fixtures for backend tests: lysozyme pairs scored by the exact CPU path.

use super::{cpu::CpuBackend, PoseParams};
use crate::energy::{CoulombParams, SplinedPotentials};
use crate::system::{CgOptions, PairSpec, PairSystem, Source};
use crate::Vector3;
use faunus::interatomic::coulomb::{Medium, Salt};
use faunus::interatomic::twobody::{GridType, SplineConfig};
use std::path::PathBuf;

const TEMPERATURE: f64 = 298.15;

/// kJ/mol at the fixture temperature.
pub fn kt() -> f64 {
    crate::thermal_energy(TEMPERATURE)
}

/// The CLAUDE.md benchmark system: 2CDS × 2CDS, 20 mM NaCl, 150 Å cutoff.
pub struct Lysozyme {
    pub splines: SplinedPotentials,
    pub coulomb: CoulombParams,
    /// Exact (unsplined) reference; also owns the reference structures.
    pub exact: CpuBackend,
}

pub fn lysozyme() -> Lysozyme {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/calvados3");
    let (xyz, top) = (dir.join("2CDS.xyz"), dir.join("topology.yaml"));
    let medium = Medium::salt_water(TEMPERATURE, Salt::SodiumChloride, 0.02);
    let system = PairSystem::load(PairSpec {
        a: Source::File(&xyz),
        b: None,
        top: Some(&top),
        chains: [&[], &[]],
        cg: CgOptions::default(),
        medium: &medium,
    })
    .expect("2CDS example loads");
    let config = SplineConfig {
        n_points: 200,
        shift_energy: true,
        grid_type: GridType::PowerLaw2,
        ..Default::default()
    };
    let splines = system.pair_matrix.create_sr_splines(150.0, config);
    let coulomb = system.pair_matrix.coulomb_params().clone();
    Lysozyme {
        splines,
        coulomb,
        exact: CpuBackend::new(system.ref_a, system.ref_b, system.pair_matrix),
    }
}

/// About 20 poses spanning contact (30 Å), mid (40 Å) and tail (55 Å) separations.
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

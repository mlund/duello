//! Analytic 6D tables for tests, so expected energies can be worked by hand.
//! All are generated at 298.15 K, where kT = 2.4789570 kJ/mol.

use super::Pose;
use crate::{AdaptiveBuilder, Table6DAdaptive, Vector3};
use icotable::TableMetadata;

pub const RMIN: f64 = 5.0;
pub const RMAX: f64 = 8.0;
pub const KT: f64 = 2.4789570;

/// Table over R in [5, 8) Å with energy `u(R, dir_a, dir_b)` in kJ/mol.
pub fn table(u: impl Fn(f64, &Vector3, &Vector3) -> f64) -> Table6DAdaptive<f32> {
    table_in_omega(4, 1, |r, _, a, b| u(r, a, b))
}

/// Table filled the way a scan does: each entry is the energy of the two
/// poses `icotable::orient` builds for it, so both lookup orders describe
/// one physical configuration. Finer than `table`, since the reverse
/// lookup lands between grid points.
pub fn physical_table(energy: impl Fn(&Pose, &Pose) -> f64) -> Table6DAdaptive<f32> {
    table_in_omega(64, 2, |r, omega, vi, vj| {
        let (q_a, q_b, separation) = icotable::orient(r, omega, vi, vj);
        let a = Pose {
            position: Vector3::zeros(),
            orientation: q_a,
        };
        let b = Pose {
            position: separation,
            orientation: q_b,
        };
        energy(&a, &b)
    })
}

fn table_in_omega(
    omega_bins: usize,
    max_n_div: usize,
    u: impl Fn(f64, f64, &Vector3, &Vector3) -> f64,
) -> Table6DAdaptive<f32> {
    let omega_step = std::f64::consts::TAU / omega_bins as f64;
    let mut builder = AdaptiveBuilder::new(RMIN, RMAX, 1.0, omega_step, max_n_div, 1e-9, 0.4);
    for ri in 0..builder.n_r() {
        let r = RMIN + ri as f64;
        let verts: Vec<Vector3> = builder
            .vertex_directions(builder.current_level())
            .iter()
            .map(|&v| Vector3::from(v))
            .collect();
        for oi in 0..builder.n_omega() {
            let omega = oi as f64 * omega_step; // bin centre
            let data: Vec<f64> = verts
                .iter()
                .flat_map(|a| verts.iter().map(|b| u(r, omega, a, b)).collect::<Vec<_>>())
                .collect();
            builder.set_slab(ri, oi, &data);
        }
        builder.finish_r_slice(ri);
    }
    let mut table = builder.build();
    table.metadata = Some(TableMetadata {
        tail_terms: vec![],
        charges: None,
        dipole_moments: None,
        temperature: Some(298.15),
        electric_prefactor: None,
        point_group: icotable::PointGroup::Exchange,
    });
    table
}

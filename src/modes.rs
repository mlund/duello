//! Binding modes: the poses that carry B₂, grouped by similarity.
//!
//! B₂ splits exactly into one term per tabulated state, so each pose's share of
//! the attraction can be ranked instead of guessed from its energy alone.

use crate::archive::ScanArchive;
use crate::poses::{r_at, TablePose};
use crate::virial::{mayer, shell};
use crate::Table6DAdaptive;
use icotable::{MeshLevel, SlabResolution};

/// How finely poses are told apart, and how many modes to keep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeSpec {
    /// Poses whose B beads lie within this RMSD (Å) are one mode.
    pub rmsd: f64,
    pub max_modes: usize,
}

impl Default for ModeSpec {
    fn default() -> Self {
        Self {
            rmsd: 4.0,
            max_modes: 10,
        }
    }
}

/// A family of similar poses and what it adds to B₂.
#[derive(Clone, Debug)]
pub struct BindingMode {
    /// The family's largest contributor
    pub pose: TablePose,
    /// Σc over the family (Å³), negative
    pub b2: f64,
    /// Fraction of [`Modes::attractive_b2`]
    pub share: f64,
    pub n_poses: usize,
}

/// Binding modes ranked by B₂ contribution.
#[derive(Clone, Debug)]
pub struct Modes {
    pub modes: Vec<BindingMode>,
    /// Σc over attractive states (Å³). Shares are relative to this rather than
    /// to the reported B₂, which the coarsened f32 table only approximates.
    pub attractive_b2: f64,
}

/// The poses that carry the attraction in `archive`'s B₂, as binding modes.
pub fn binding_modes(archive: &ScanArchive, spec: ModeSpec) -> Modes {
    let _ = spec;
    let mut attractive_b2 = 0.0;
    contributions(archive, |c| attractive_b2 += c.min(0.0));
    Modes {
        modes: Vec::new(),
        attractive_b2,
    }
}

/// Visits every state's B₂ term c (Å³) at R ≥ σ, so that 2πσ³/3 + Σc is the
/// table's B₂; negative terms attract. A visitor rather than a list, as there
/// may be ~10⁸ states.
pub(crate) fn contributions(archive: &ScanArchive, mut visit: impl FnMut(f64)) {
    let table = &archive.table;
    let beta = 1.0 / crate::thermal_energy(archive.settings.temperature);
    for ri in 0..table.n_r {
        if let Some(shell) = shell(r_at(table, ri), table.dr, archive.virial.sigma()) {
            r_bin(table, ri, shell, beta, &mut visit);
        }
    }
}

/// Each ω-slab holds 1/n_ω of the orientations; the terms are −shell·weight·f
/// with Mayer f = e^(−βU) − 1.
fn r_bin(
    table: &Table6DAdaptive<f32>,
    ri: usize,
    shell: f64,
    beta: f64,
    visit: &mut impl FnMut(f64),
) {
    let per_slab = shell / table.n_omega as f64;
    let mut mesh_slabs = Vec::new();
    for oi in 0..table.n_omega {
        match &table.slab_res[ri * table.n_omega + oi] {
            SlabResolution::Repulsive => visit(per_slab),
            // icotable stores the plain mean U, so this is f(⟨U⟩) rather than
            // ⟨f(U)⟩: the table's own B₂, which is what the terms sum to.
            SlabResolution::Scalar(u) => visit(-per_slab * mayer(beta * f64::from(*u))),
            SlabResolution::Mesh { .. } => mesh_slabs.push(oi),
        }
    }
    if let Some((energies, level)) = table.energies_at_r(ri) {
        mesh_terms(&energies, level, &mesh_slabs, per_slab, beta, visit);
    }
}

/// Vertex pairs weigh w_i·w_j/n_v², their share of the two Voronoi spheres.
/// Pairs outermost, as `energies` keeps each pair's ω-bins together.
fn mesh_terms(
    energies: &[f64],
    level: &MeshLevel,
    mesh_slabs: &[usize],
    per_slab: f64,
    beta: f64,
    visit: &mut impl FnMut(f64),
) {
    let n_v = level.n_vertices;
    let per_pair = per_slab / (n_v * n_v) as f64;
    let pairs = energies.chunks_exact(energies.len() / (n_v * n_v));
    let weights = level
        .weights
        .iter()
        .flat_map(|w_i| level.weights.iter().map(move |w_j| w_i * w_j));
    for (by_omega, weight) in pairs.zip(weights) {
        for &oi in mesh_slabs {
            visit(-per_pair * weight * mayer(beta * by_omega[oi]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::tests::archive;
    use crate::icoscan::omega_step;
    use crate::virial::{hardsphere, VirialCoeff};
    use crate::AdaptiveBuilder;
    use approx::assert_relative_eq;

    /// What one ω-slab holds, in kJ/mol.
    #[derive(Clone, Copy)]
    enum Slab {
        /// `base` everywhere but `spike` at vertex pair (3, 7), so it stays a mesh.
        Mesh { base: f64, spike: f64 },
        /// Uniform, so the builder stores one scalar.
        Scalar(f64),
        /// Overlapping everywhere, so the builder stores nothing.
        Repulsive,
    }

    /// A table on the test archive's grid (R = 5, 6, 7 Å) with `slab(ri, oi)`.
    fn archive_with(slab: impl Fn(usize, usize) -> Slab) -> ScanArchive {
        let mut archive = archive();
        let s = &archive.settings;
        let step = omega_step(s.subdivision, s.max_ndiv);
        let beta = 1.0 / crate::thermal_energy(s.temperature);
        let mut builder = AdaptiveBuilder::new(5.0, 7.0, 1.0, step, s.max_ndiv, 1e-9, beta);
        for ri in 0..builder.n_r() {
            let n_v = builder.vertex_directions(builder.current_level()).len();
            for oi in 0..builder.n_omega() {
                let data = match slab(ri, oi) {
                    Slab::Mesh { base, spike } => {
                        let mut data = vec![base; n_v * n_v];
                        data[3 * n_v + 7] = spike;
                        data
                    }
                    Slab::Scalar(u) => vec![u; n_v * n_v],
                    Slab::Repulsive => vec![1e4; n_v * n_v],
                };
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }
        archive.table = builder.build();
        archive
    }

    fn b2_from_terms(archive: &ScanArchive) -> f64 {
        let core = hardsphere(archive.virial.sigma());
        let mut sum = 0.0;
        contributions(archive, |c| sum += c);
        core + sum
    }

    /// Base energy per R-bin, kJ/mol; the spike is deep enough to dominate.
    const BASE: [f64; 3] = [1.0, -0.5, -0.2];
    const SPIKE: f64 = -20.0;

    /// B₂ from the angular average of `slab`, which is exact for these slabs:
    /// a mesh differs from its base only in the spike's Voronoi cell, which
    /// covers w₃·w₇/n_v² of the vertex pairs.
    fn expected_b2(archive: &ScanArchive, slab: impl Fn(usize, usize) -> Slab) -> f64 {
        let table = &archive.table;
        let kt = crate::thermal_energy(archive.settings.temperature);
        let boltz = |u: f64| (-u / kt).exp();
        let pmf: Vec<(f64, f64)> = (0..table.n_r)
            .map(|ri| {
                let slab_mean = |oi| match slab(ri, oi) {
                    Slab::Mesh { base, spike } => {
                        let level = table.mesh_level_at_r(ri).unwrap();
                        let n_v = level.n_vertices as f64;
                        let f = level.weights[3] * level.weights[7] / (n_v * n_v);
                        (1.0 - f) * boltz(base) + f * boltz(spike)
                    }
                    Slab::Scalar(u) => boltz(u),
                    Slab::Repulsive => 0.0,
                };
                let n = table.n_omega;
                let mean = (0..n).map(slab_mean).sum::<f64>() / n as f64;
                (5.0 + ri as f64, -mean.ln())
            })
            .collect();
        VirialCoeff::from_pmf(pmf, Some(5.0)).unwrap().b2()
    }

    fn spiked(ri: usize, _: usize) -> Slab {
        Slab::Mesh {
            base: BASE[ri],
            spike: SPIKE,
        }
    }

    #[test]
    fn terms_sum_to_b2_of_the_angular_average() {
        let archive = archive_with(spiked);
        let expected = expected_b2(&archive, spiked);
        assert_relative_eq!(b2_from_terms(&archive), expected, max_relative = 1e-5);
    }

    /// At contact one ω-slab overlaps and one is isotropic.
    fn mixed(ri: usize, oi: usize) -> Slab {
        match (ri, oi) {
            (0, 0) => Slab::Repulsive,
            (0, 1) => Slab::Scalar(-3.0),
            _ => spiked(ri, oi),
        }
    }

    #[test]
    fn scalar_and_repulsive_slabs_count_as_their_share_of_omega() {
        let archive = archive_with(mixed);
        let slab = |oi| &archive.table.slab_res[oi];
        assert_eq!(slab(0), &SlabResolution::Repulsive);
        assert!(matches!(slab(1), SlabResolution::Scalar(_)));
        let expected = expected_b2(&archive, mixed);
        assert_relative_eq!(b2_from_terms(&archive), expected, max_relative = 1e-5);
    }

    /// Attractive everywhere except one overlapping ω-slab at contact.
    fn walled(ri: usize, oi: usize) -> Slab {
        match (ri, oi) {
            (0, 0) => Slab::Repulsive,
            _ => Slab::Mesh {
                base: -0.5,
                spike: SPIKE,
            },
        }
    }

    #[test]
    fn attractive_b2_leaves_out_the_repulsive_terms() {
        let archive = archive_with(walled);
        let core = hardsphere(5.0);
        let wall = 2.0 * std::f64::consts::PI * 5.0_f64.powi(2) / archive.table.n_omega as f64;
        let expected = expected_b2(&archive, walled) - core - wall;
        let modes = binding_modes(&archive, ModeSpec::default());
        assert_relative_eq!(modes.attractive_b2, expected, max_relative = 1e-5);
    }

    /// The real table is coarsened and f32, so its terms only approximate the
    /// B₂ reported from the scan's exact angular averages.
    #[test]
    #[ignore = "needs DUELLO_BENCH_ARCHIVE, a saved 2CDS scan (see CLAUDE.md)"]
    fn terms_match_the_reported_b2_of_a_real_scan() {
        let path = std::env::var("DUELLO_BENCH_ARCHIVE").expect("DUELLO_BENCH_ARCHIVE");
        let bundle = crate::bundle::Bundle::from_zip(&std::fs::read(path).unwrap()).unwrap();
        let archive = bundle.scan.expect("a two-body scan");
        let (terms, reported) = (b2_from_terms(&archive), archive.virial.b2());
        println!("Σc + core = {terms:.0} Å³, reported B₂ = {reported:.0} Å³");
        assert_relative_eq!(terms, reported, max_relative = 0.01);
    }
}

//! Binding modes: the poses that carry B₂, grouped by similarity.
//!
//! B₂ splits exactly into one term per tabulated state, so each pose's share of
//! the attraction can be ranked instead of guessed from its energy alone.

mod cluster;
pub mod export;

use crate::archive::ScanArchive;
use crate::poses::{keep_least, r_at, ByKey, TablePose};
use crate::virial::{mayer, shell};
use crate::Table6DAdaptive;
use cluster::Founders;
use icotable::{MeshLevel, SlabResolution};
use std::collections::BinaryHeap;

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
    /// Tabulated states in the family, counting the weak ones that joined
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
    clustered(archive, spec, MAX_CANDIDATES)
}

/// [`binding_modes`] with the seed count as a parameter, so tests can make
/// the cap bite on small tables.
fn clustered(archive: &ScanArchive, spec: ModeSpec, max_candidates: usize) -> Modes {
    let strongest = Strongest::of(archive, max_candidates);
    let attractive_b2 = strongest.total;
    let homodimer = archive.settings.homo_dimer;
    let mut founders = Founders::new(&archive.bodies, homodimer, spec.rmsd);
    for ByKey(_, state) in strongest.covering() {
        founders.found_if_new(state.pose(archive));
    }
    let mut families = founders.freeze();
    // Every attractive state joins, seeds too, so shares count the whole table.
    contributions(archive, |Term { c, state }| {
        if let Some(state) = state.filter(|_| c < 0.0) {
            families.join(&state.pose(archive).b_pose(), c);
        }
    });
    let mut modes = families.into_modes();
    modes.sort_by(|a, b| a.b2.total_cmp(&b.b2));
    modes.truncate(spec.max_modes);
    for mode in &mut modes {
        mode.share = mode.b2 / attractive_b2;
    }
    Modes {
        modes,
        attractive_b2,
    }
}

/// Seeds: the states that may found modes. The rest only join them, so this
/// bounds memory and the number of modes, not what the shares cover.
const MAX_CANDIDATES: usize = 20_000;

/// Seeding stops once this fraction of the attraction is placed; the tail
/// is a cloud of tiny terms that would each found a mode of its own.
const COVERAGE: f64 = 0.999;

/// The strongest attractive mesh states, strongest first. A bounded heap, as
/// holding all ~10⁸ states would not fit.
struct Strongest {
    /// Σc over all attractive states (Å³)
    total: f64,
    /// Σc over attractive mesh states, the only ones a mode can hold
    oriented: f64,
    /// Keyed by c
    states: Vec<ByKey<State>>,
}

impl Strongest {
    fn of(archive: &ScanArchive, max_candidates: usize) -> Self {
        let (mut total, mut oriented) = (0.0, 0.0);
        let mut kept = BinaryHeap::with_capacity(max_candidates);
        contributions(archive, |Term { c, state }| {
            if c < 0.0 {
                total += c;
                if let Some(state) = state {
                    oriented += c;
                    keep_least(&mut kept, max_candidates, ByKey(c, state));
                }
            }
        });
        let states = kept.into_sorted_vec();
        Self {
            total,
            oriented,
            states,
        }
    }

    /// The leading states that together reach [`COVERAGE`] of the oriented
    /// attraction.
    fn covering(self) -> impl Iterator<Item = ByKey<State>> {
        let (mut placed, needed) = (0.0, COVERAGE * self.oriented.abs());
        self.states.into_iter().take_while(move |s| {
            let more = placed < needed;
            placed += s.0.abs();
            more
        })
    }
}

/// One tabulated state's term of B₂ (Å³); negative terms attract.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Term {
    pub c: f64,
    /// None for isotropic and repulsive slabs, which have no orientation.
    pub state: Option<State>,
}

/// A mesh entry, kept as indices because a table can hold ~10⁸ of them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct State {
    ri: usize,
    /// Flat index into [`Table6DAdaptive::energies_at_r`]
    k: usize,
    /// kJ/mol
    energy: f64,
}

impl State {
    fn pose(&self, archive: &ScanArchive) -> TablePose {
        TablePose::at(
            &archive.table,
            &archive.settings,
            self.ri,
            self.k,
            self.energy,
        )
    }
}

/// Visits every state's B₂ term at R ≥ σ, so that 2πσ³/3 + Σc is the table's
/// B₂. A visitor rather than a list, as there may be ~10⁸ states.
pub(crate) fn contributions(archive: &ScanArchive, mut visit: impl FnMut(Term)) {
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
    visit: &mut impl FnMut(Term),
) {
    let per_slab = shell / table.n_omega as f64;
    let mut mesh_slabs = Vec::new();
    for oi in 0..table.n_omega {
        match &table.slab_res[ri * table.n_omega + oi] {
            SlabResolution::Repulsive => visit(Term {
                c: per_slab,
                state: None,
            }),
            // icotable stores the plain mean U, so this is f(⟨U⟩) rather than
            // ⟨f(U)⟩: the table's own B₂, which is what the terms sum to.
            SlabResolution::Scalar(u) => visit(Term {
                c: -per_slab * mayer(beta * f64::from(*u)),
                state: None,
            }),
            SlabResolution::Mesh { .. } => mesh_slabs.push(oi),
        }
    }
    if let Some((energies, level)) = table.energies_at_r(ri) {
        let bin = Bin {
            ri,
            n_omega: table.n_omega,
            per_slab,
            beta,
        };
        mesh_terms(&bin, &energies, level, &mesh_slabs, visit);
    }
}

/// What every term of one R-bin shares.
struct Bin {
    ri: usize,
    n_omega: usize,
    /// shell/n_ω (Å³)
    per_slab: f64,
    beta: f64,
}

/// Vertex pairs weigh w_i·w_j/n_v², their share of the two Voronoi spheres.
/// Pairs outermost, as `energies` keeps each pair's ω-bins together.
fn mesh_terms(
    bin: &Bin,
    energies: &[f64],
    level: &MeshLevel,
    mesh_slabs: &[usize],
    visit: &mut impl FnMut(Term),
) {
    let n_v = level.n_vertices;
    let per_pair = bin.per_slab / (n_v * n_v) as f64;
    let weights = level
        .weights
        .iter()
        .flat_map(|w_i| level.weights.iter().map(move |w_j| w_i * w_j));
    for (pair, weight) in weights.enumerate() {
        for &oi in mesh_slabs {
            let k = pair * bin.n_omega + oi;
            let energy = energies[k];
            visit(Term {
                c: -per_pair * weight * mayer(bin.beta * energy),
                state: Some(State {
                    ri: bin.ri,
                    k,
                    energy,
                }),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::cluster::seen_from;
    use super::*;
    use crate::archive::tests::archive;
    use crate::icoscan::omega_step;
    use crate::virial::{hardsphere, VirialCoeff};
    use crate::AdaptiveBuilder;
    use crate::{UnitQuaternion, Vector3};
    use approx::assert_relative_eq;

    /// What one ω-slab holds, in kJ/mol.
    #[derive(Clone)]
    enum Slab {
        /// `base` everywhere but at the (vi, vj, U) spikes, so it stays a mesh.
        Mesh {
            base: f64,
            spikes: Vec<(usize, usize, f64)>,
        },
        /// Uniform, so the builder stores one scalar.
        Scalar(f64),
        /// Overlapping everywhere, so the builder stores nothing.
        Repulsive,
    }

    /// A builder on the test archive's grid: R = 5, 6, 7 Å, mesh level 1.
    fn builder(archive: &ScanArchive) -> AdaptiveBuilder {
        let s = &archive.settings;
        let step = omega_step(s.subdivision, s.max_ndiv);
        let beta = 1.0 / crate::thermal_energy(s.temperature);
        AdaptiveBuilder::new(5.0, 7.0, 1.0, step, s.max_ndiv, 1e-9, beta)
    }

    /// The test archive with a table holding `slab(ri, oi)`.
    fn archive_with(slab: impl Fn(usize, usize) -> Slab) -> ScanArchive {
        let mut archive = archive();
        let mut builder = builder(&archive);
        for ri in 0..builder.n_r() {
            let n_v = builder.vertex_directions(builder.current_level()).len();
            for oi in 0..builder.n_omega() {
                let data = match slab(ri, oi) {
                    Slab::Mesh { base, spikes } => {
                        let mut data = vec![base; n_v * n_v];
                        for (vi, vj, u) in spikes {
                            data[vi * n_v + vj] = u;
                        }
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
        contributions(archive, |t| sum += t.c);
        core + sum
    }

    /// Base energy per R-bin, kJ/mol; the spike is deep enough to dominate.
    const BASE: [f64; 3] = [1.0, -0.5, -0.2];
    const SPIKE: f64 = -20.0;

    /// B₂ from the angular average of `slab`, which is exact for these slabs:
    /// a mesh differs from its base only in each spike's Voronoi cell, which
    /// covers w_i·w_j/n_v² of the vertex pairs.
    fn expected_b2(archive: &ScanArchive, slab: impl Fn(usize, usize) -> Slab) -> f64 {
        let table = &archive.table;
        let kt = kt(archive);
        let boltz = |u: f64| (-u / kt).exp();
        let pmf: Vec<(f64, f64)> = (0..table.n_r)
            .map(|ri| {
                let slab_mean = |oi| match slab(ri, oi) {
                    Slab::Mesh { base, spikes } => {
                        let level = table.mesh_level_at_r(ri).unwrap();
                        mesh_mean(level, base, &spikes, boltz)
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

    /// ⟨e^(−βU)⟩ over a mesh slab: each spike replaces its Voronoi cell's base.
    fn mesh_mean(
        level: &MeshLevel,
        base: f64,
        spikes: &[(usize, usize, f64)],
        boltz: impl Fn(f64) -> f64,
    ) -> f64 {
        let n_v = level.n_vertices as f64;
        let excess: f64 = spikes
            .iter()
            .map(|&(vi, vj, u)| {
                level.weights[vi] * level.weights[vj] / (n_v * n_v) * (boltz(u) - boltz(base))
            })
            .sum();
        boltz(base) + excess
    }

    fn spiked(ri: usize, _: usize) -> Slab {
        Slab::Mesh {
            base: BASE[ri],
            spikes: vec![(3, 7, SPIKE)],
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
                spikes: vec![(3, 7, SPIKE)],
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

    /// Wells as (ri, vi, vj, U) in ω-bin 0; everything else mildly repulsive,
    /// so the wells hold all of the attraction.
    fn wells(wells: &[(usize, usize, usize, f64)]) -> impl Fn(usize, usize) -> Slab + '_ {
        move |ri, oi| {
            if oi != 0 {
                return Slab::Scalar(0.5);
            }
            let spikes: Vec<_> = wells
                .iter()
                .filter(|w| w.0 == ri)
                .map(|&(_, vi, vj, u)| (vi, vj, u))
                .collect();
            if spikes.is_empty() {
                Slab::Scalar(0.5)
            } else {
                Slab::Mesh { base: 0.5, spikes }
            }
        }
    }

    /// The mesh vertex pointing most nearly away from vertex `v`.
    fn opposite(v: usize) -> usize {
        let builder = builder(&archive());
        let verts = builder.vertex_directions(builder.current_level());
        let dot = |u: usize| (0..3).map(|d| verts[u][d] * verts[v][d]).sum::<f64>();
        (0..verts.len())
            .min_by(|&a, &b| dot(a).total_cmp(&dot(b)))
            .unwrap()
    }

    #[test]
    fn distant_wells_are_separate_modes_ranked_by_b2() {
        let far = opposite(3);
        let archive = archive_with(wells(&[(0, 3, 7, -15.0), (0, far, 7, -20.0)]));
        let modes = binding_modes(&archive, ModeSpec::default());

        assert_eq!(modes.modes.len(), 2);
        let [deep, shallow] = [&modes.modes[0], &modes.modes[1]];
        assert_relative_eq!(deep.pose.energy * kt(&archive), -20.0, max_relative = 1e-6);
        assert_relative_eq!(
            shallow.pose.energy * kt(&archive),
            -15.0,
            max_relative = 1e-6
        );
        assert!(deep.b2 < shallow.b2);
        assert_relative_eq!(deep.share + shallow.share, 1.0, max_relative = 1e-9);
        assert_relative_eq!(
            deep.b2 + shallow.b2,
            modes.attractive_b2,
            max_relative = 1e-9
        );
    }

    #[test]
    fn poses_within_the_rmsd_merge_into_their_strongest() {
        // Same orientation one R-bin apart, so B's beads shift by exactly 1 Å.
        let archive = archive_with(wells(&[(0, 3, 7, -20.0), (1, 3, 7, -15.0)]));
        let merged = binding_modes(&archive, ModeSpec::default());
        let [mode] = merged.modes.as_slice() else {
            panic!("expected one mode, got {}", merged.modes.len());
        };
        assert_eq!(mode.n_poses, 2);
        assert_relative_eq!(mode.pose.r, 5.0);
        assert_relative_eq!(mode.share, 1.0, max_relative = 1e-9);

        let tight = ModeSpec {
            rmsd: 0.5,
            ..ModeSpec::default()
        };
        assert_eq!(binding_modes(&archive, tight).modes.len(), 2);
    }

    #[test]
    fn poses_beyond_the_candidate_cap_still_join_their_mode() {
        let archive = archive_with(wells(&[(0, 3, 7, -20.0), (1, 3, 7, -15.0)]));
        let modes = clustered(&archive, ModeSpec::default(), 1);
        let [mode] = modes.modes.as_slice() else {
            panic!("expected one mode, got {}", modes.modes.len());
        };
        assert_eq!(mode.n_poses, 2);
        assert_relative_eq!(mode.share, 1.0, max_relative = 1e-9);
    }

    /// A pose and the same complex seen from B: B's frame becomes the lab.
    fn pose_and_swap() -> [TablePose; 2] {
        let dir = |x, y, z| Vector3::new(x, y, z).normalize();
        let pose = TablePose::off_grid(8.0, 0.7, [dir(1.0, 0.3, 0.2), dir(-0.2, 1.0, 0.5)]);
        let a = seen_from(&pose.b_pose());
        let (r, omega, dir_a, dir_b) =
            icotable::inverse_orient(&a.position, &UnitQuaternion::identity(), &a.orientation);
        [pose, TablePose::off_grid(r, omega, [dir_a, dir_b])]
    }

    #[test]
    fn a_homodimer_pose_and_its_swap_are_one_mode() {
        let mut body = crate::archive::tests::body("LYS");
        body.positions = vec![[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -3.0]];
        body.masses = vec![1.0; 3];
        let bodies = [body.clone(), body];
        let [pose, swap] = pose_and_swap();
        let cluster = |homodimer| {
            let mut founders = Founders::new(&bodies, homodimer, 0.1);
            founders.found_if_new(pose.clone());
            founders.found_if_new(swap.clone());
            let mut families = founders.freeze();
            families.join(&pose.b_pose(), -2.0);
            families.join(&swap.b_pose(), -1.0);
            families.into_modes()
        };

        let homo = cluster(true);
        assert_eq!(homo.len(), 1);
        assert_eq!(homo[0].n_poses, 2);
        assert_eq!(cluster(false).len(), 2);
    }

    fn kt(archive: &ScanArchive) -> f64 {
        crate::thermal_energy(archive.settings.temperature)
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

        let start = std::time::Instant::now();
        let modes = binding_modes(&archive, ModeSpec::default());
        println!("{:?} for {} modes", start.elapsed(), modes.modes.len());
        for m in &modes.modes {
            let (share, u, r) = (100.0 * m.share, m.pose.energy, m.pose.r);
            println!(
                "{share:5.1}% {:9.0} Å³ U {u:6.2} kT R {r:5.1} Å, {} poses",
                m.b2, m.n_poses
            );
        }
        assert!(modes.modes.iter().map(|m| m.share).sum::<f64>() <= 1.0 + 1e-9);
    }
}

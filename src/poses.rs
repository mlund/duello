//! The lowest-energy states of a scan's 6D table, placed as bead coordinates.

use crate::archive::{CgBody, ScanArchive, ScanSettings};
use crate::icoscan::omega_step;
use crate::nbody::{trajectory::expand, Pose};
use crate::{Table6DAdaptive, Vector3};
use icotable::{AngularMesh, SlabResolution};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One stored (R, ω, vertex i, vertex j) state of the table.
#[derive(Clone, Debug, PartialEq)]
pub struct TablePose {
    /// Mass-centre separation (Å)
    pub r: f64,
    /// Dihedral angle (radians)
    pub omega: f64,
    /// Pair energy (kT)
    pub energy: f64,
    /// Mesh directions of the two vertices; an index alone means nothing
    /// without the mesh level of its R-bin.
    directions: [Vector3; 2],
}

/// The `n` lowest-energy states, sorted by energy. Repulsive states store no
/// energy and are never returned.
pub fn lowest(archive: &ScanArchive, n: usize) -> Vec<TablePose> {
    let table = &archive.table;
    let mut kept = BinaryHeap::new();
    for ri in 0..table.n_r {
        // Scalar-only and repulsive R-bins have no mesh, so no pose is defined.
        let Some((energies, _)) = table.energies_at_r(ri) else {
            continue;
        };
        let n_v = table.mesh_level_at_r(ri).map_or(0, |l| l.n_vertices);
        for oi in 0..table.n_omega {
            // A scalar slab is copied to every vertex pair; only its first is a distinct state.
            let pairs = if is_scalar(table, ri, oi) {
                1
            } else {
                n_v * n_v
            };
            for k in (0..pairs).map(|pair| pair * table.n_omega + oi) {
                let candidate = Candidate {
                    energy: energies[k],
                    ri,
                    k,
                };
                keep_if_lower(&mut kept, n, candidate);
            }
        }
    }
    kept.into_sorted_vec()
        .into_iter()
        .map(|c| c.to_pose(table, &archive.settings))
        .collect()
}

fn is_scalar(table: &Table6DAdaptive<f32>, ri: usize, oi: usize) -> bool {
    matches!(
        table.slab_res[ri * table.n_omega + oi],
        SlabResolution::Scalar(_)
    )
}

/// A bounded max-heap: the worst kept state is on top, ready to be replaced.
fn keep_if_lower(kept: &mut BinaryHeap<Candidate>, n: usize, candidate: Candidate) {
    if !candidate.energy.is_finite() {
        return;
    }
    if kept.len() < n {
        kept.push(candidate);
    } else if let Some(mut top) = kept.peek_mut() {
        if candidate < *top {
            *top = candidate;
        }
    }
}

/// A table entry by its R-bin and flat index in [`Table6DAdaptive::energies_at_r`].
struct Candidate {
    /// kJ/mol, as tabulated
    energy: f64,
    ri: usize,
    k: usize,
}

impl Candidate {
    /// Decodes the layout `vi·n_v·n_ω + vj·n_ω + oi` and converts to kT the way
    /// [`crate::Sample`] does.
    fn to_pose(&self, table: &Table6DAdaptive<f32>, settings: &ScanSettings) -> TablePose {
        let level = table
            .mesh_level_at_r(self.ri)
            .expect("the energies came from this R-bin's mesh");
        let (n_v, n_omega) = (level.n_vertices, table.n_omega);
        let (vi, vj, oi) = (
            self.k / (n_v * n_omega),
            self.k / n_omega % n_v,
            self.k % n_omega,
        );
        TablePose {
            r: (self.ri as f64).mul_add(table.dr, table.rmin),
            // The table keeps only the bin count; τ/n_ω can be off by half a bin.
            omega: oi as f64 * omega_step(settings.subdivision, settings.max_ndiv),
            energy: self.energy / crate::thermal_energy(settings.temperature),
            directions: [level.direction(vi), level.direction(vj)],
        }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.energy.total_cmp(&other.energy)
    }
}

impl TablePose {
    /// Bead positions (Å) of both bodies: A as stored, B rotated and shifted.
    pub fn place(&self, bodies: &[CgBody; 2]) -> [Vec<[f64; 3]>; 2] {
        let [dir_i, dir_j] = &self.directions;
        let (q_a, q_b, separation) = icotable::orient(self.r, self.omega, dir_i, dir_j);
        let placed = |body: &CgBody, position, orientation| {
            expand(
                body,
                &Pose {
                    position,
                    orientation,
                },
            )
            .into_iter()
            .map(Into::into)
            .collect()
        };
        [
            placed(&bodies[0], Vector3::zeros(), q_a),
            placed(&bodies[1], separation, q_b),
        ]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::archive::tests::archive;
    use crate::structure::mass_center;
    use crate::{AdaptiveBuilder, Table6DAdaptive};
    use approx::assert_relative_eq;

    const PLANTED: f64 = -20.0; // kJ/mol
    pub(crate) const ISOTROPIC: f64 = -30.0; // kJ/mol

    /// Smooth energies with one deep state at R = 6 Å, ω-bin 2, vertices (3, 7).
    fn planted_table(archive: &ScanArchive) -> Table6DAdaptive<f32> {
        let s = &archive.settings;
        let step = omega_step(s.subdivision, s.max_ndiv);
        let beta = 1.0 / crate::thermal_energy(s.temperature);
        let mut builder = AdaptiveBuilder::new(5.0, 7.0, 1.0, step, s.max_ndiv, 1e-9, beta);
        for ri in 0..builder.n_r() {
            let verts = builder.vertex_directions(builder.current_level()).to_vec();
            let n_v = verts.len();
            for oi in 0..builder.n_omega() {
                let mut data: Vec<f64> = verts
                    .iter()
                    .flat_map(|a| verts.iter().map(move |b| 5.0 + a[0] - 2.0 * b[1]))
                    .collect();
                if (ri, oi) == (1, 2) {
                    data[3 * n_v + 7] = PLANTED;
                }
                // Isotropic, so stored as one scalar for every vertex pair.
                if (ri, oi) == (0, 0) {
                    data.fill(ISOTROPIC);
                }
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }
        builder.build()
    }

    pub(crate) fn planted_archive() -> ScanArchive {
        let mut archive = archive();
        archive.table = planted_table(&archive);
        archive
    }

    #[test]
    fn an_isotropic_slab_is_one_pose_not_one_per_vertex_pair() {
        let archive = planted_archive();
        let kt = crate::thermal_energy(archive.settings.temperature);
        let energies: Vec<f64> = lowest(&archive, 2).iter().map(|p| p.energy * kt).collect();
        assert_relative_eq!(energies[0], ISOTROPIC, max_relative = 1e-6);
        assert_relative_eq!(energies[1], PLANTED, max_relative = 1e-6);
    }

    #[test]
    fn placed_bodies_sit_r_apart_with_a_unmoved() {
        let archive = planted_archive();
        let pose = &lowest(&archive, 2)[1];
        let [a, b] = pose.place(&archive.bodies);

        let [body_a, body_b] = &archive.bodies;
        assert_eq!(a, body_a.positions);
        assert_eq!(b.len(), body_b.positions.len());
        let centre = |beads: &[[f64; 3]], masses: &[f64]| {
            mass_center(beads.iter().map(|&p| p.into()), masses.iter().copied())
        };
        let separation = centre(&b, &body_b.masses) - centre(&a, &body_a.masses);
        assert_relative_eq!(separation.norm(), 6.0, epsilon = 1e-9);
    }

    #[test]
    fn lowest_finds_the_planted_minimum_and_sorts_ascending() {
        let archive = planted_archive();
        let poses = lowest(&archive, 5);

        assert_eq!(poses.len(), 5);
        let kt = crate::thermal_energy(archive.settings.temperature);
        let planted = &poses[1]; // after the isotropic slab
        assert_relative_eq!(planted.energy, PLANTED / kt, max_relative = 1e-6);
        assert_relative_eq!(planted.r, 6.0);
        let step = omega_step(archive.settings.subdivision, 1);
        assert_relative_eq!(planted.omega, 2.0 * step);
        assert!(poses.windows(2).all(|w| w[0].energy <= w[1].energy));
        assert!(poses[2].energy > planted.energy);
    }
}

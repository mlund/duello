//! Pair energy between two rigid bodies, looked up in a 6D table.

use super::Pose;
use crate::Table6DAdaptive;
use anyhow::{anyhow, ensure, Result};
use icotable::lookup::{Interpolated, Lookup6D};
use icotable::PointGroup;

/// Energies of one pair in kT, or a configuration the table forbids. The free
/// energy decides acceptance; the mean energy is what ⟨U⟩ sums.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairValue {
    Finite(Interpolated),
    Overlap,
}

/// Pair energies for identical rigid bodies in a cubic periodic box.
pub struct PairEnergy {
    table: Table6DAdaptive<f32>,
    /// 1/kT in mol/kJ, at the temperature the table was generated for.
    beta: f64,
}

impl PairEnergy {
    /// The table must be for identical bodies (exchange symmetry), and record
    /// the temperature it was made at.
    pub fn new(table: Table6DAdaptive<f32>) -> Result<Self> {
        let meta = table
            .metadata
            .as_ref()
            .ok_or_else(|| anyhow!("table has no metadata"))?;
        let temperature = meta
            .temperature
            .ok_or_else(|| anyhow!("table has no generation temperature"))?;
        ensure!(
            meta.point_group == PointGroup::Exchange,
            "table is not for identical bodies (point group {:?})",
            meta.point_group
        );
        let beta = 1.0 / crate::thermal_energy(temperature);
        Ok(Self { table, beta })
    }

    /// Below this the minimum image folds pairs back into the table's range.
    pub fn min_box_length(&self) -> f64 {
        2.0 * self.table.rmax
    }

    /// Separation is `b − a`: icotable's `orient` places B there relative to A.
    /// The table is exchange symmetric, so this is the same energy whichever
    /// body comes first, as detailed balance needs (see
    /// [`icotable::lookup::Lookup6D::pair_energy`]).
    pub fn energy(&self, a: &Pose, b: &Pose, box_length: f64) -> PairValue {
        debug_assert!(
            box_length >= self.min_box_length(),
            "box {box_length} Å is below twice the table range"
        );
        let separation = minimum_image(b.position - a.position, box_length);
        let pair = self
            .table
            .pair_energy(&separation, &a.orientation, &b.orientation, self.beta);
        to_pair_value(Interpolated {
            energy: pair.energy * self.beta,
            mean_energy: pair.mean_energy * self.beta,
        })
    }
}

pub(super) fn minimum_image(d: crate::Vector3, box_length: f64) -> crate::Vector3 {
    d.map(|x| x - box_length * (x / box_length).round())
}

/// Repulsive slabs and contact give +∞; callers get a variant to branch on.
/// NaN would mean a broken table, and must not pass as an ordinary rejection.
fn to_pair_value(u: Interpolated) -> PairValue {
    debug_assert!(!u.energy.is_nan(), "pair energy is NaN");
    if u.energy == f64::INFINITY {
        PairValue::Overlap
    } else {
        PairValue::Finite(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbody::test_tables::{physical_table, table, KT};
    use crate::{UnitQuaternion, Vector3};

    fn pose(x: f64, y: f64, z: f64) -> Pose {
        Pose {
            position: Vector3::new(x, y, z),
            orientation: UnitQuaternion::from_euler_angles(0.3, -1.1, 2.0),
        }
    }

    const BIG_BOX: f64 = 1000.0;

    #[test]
    fn energy_is_the_table_value_in_kt() {
        let energy = PairEnergy::new(table(|r, _, _| 2.0 * r)).unwrap();
        // R = 6.2 Å snaps to the R = 6 bin: 12 kJ/mol = 4.8407456 kT.
        let u = energy.energy(&pose(0.0, 0.0, 0.0), &pose(6.2, 0.0, 0.0), BIG_BOX);
        approx::assert_relative_eq!(finite(u).energy, 4.8407456, max_relative = 1e-6);
    }

    /// Pins the frame convention: B placed with `icotable::orient` must read
    /// back the energy stored for those mesh vertices. The symmetry tests
    /// can't catch a mirrored separation, as a flip leaves them unchanged.
    #[test]
    fn body_placed_on_mesh_vertices_reads_their_energy() {
        // Odd in each direction, so a mirrored lookup gives a different value.
        let g = |v: &Vector3| v.x + 2.0 * v.y + 3.0 * v.z;
        // Symmetric under exchange: each body sees the other along a direction
        // in its own frame, from b − a.
        let geometric = |a: &Pose, b: &Pose| {
            let d = b.position - a.position;
            let r = d.norm();
            let towards_b = a.orientation.inverse() * d / r;
            let towards_a = b.orientation.inverse() * -d / r;
            r + g(&towards_b) + g(&towards_a)
        };
        let t = physical_table(geometric);
        let vertices = &t.levels.last().unwrap().vertices;
        let (vi, vj) = (Vector3::from(vertices[3]), Vector3::from(vertices[17]));
        let energy = PairEnergy::new(t.clone()).unwrap();

        // orient puts B along vi from A, with B's vertex vj pointing back at A.
        let (q_a, q_b, separation) = icotable::orient(6.0, 0.0, &vi, &vj);
        let a = Pose {
            position: Vector3::zeros(),
            orientation: q_a,
        };
        let b = Pose {
            position: separation,
            orientation: q_b,
        };
        let expected = (6.0 + g(&vi) + g(&vj)) / 2.4789570;
        // The forward lookup is exact; the reverse one falls between dihedral
        // bins and vertices, so allow table-resolution error. A mirrored
        // separation is off by far more (g changes sign).
        approx::assert_relative_eq!(
            finite(energy.energy(&a, &b, BIG_BOX)).energy,
            expected,
            max_relative = 0.02
        );
    }

    /// Exchange symmetric; steep in directions and in the dihedral, so coarse
    /// dihedral bins make the two lookup orders disagree by many kT.
    fn steep(a: &Pose, b: &Pose) -> f64 {
        let g = |v: &Vector3| v.x + 2.0 * v.y + 3.0 * v.z;
        let d = b.position - a.position;
        let r = d.norm();
        let towards_b = a.orientation.inverse() * d / r;
        let towards_a = b.orientation.inverse() * -d / r;
        let twist = (a.orientation * Vector3::x()).dot(&(b.orientation * Vector3::x()));
        KT * (2.0 * (g(&towards_b) + g(&towards_a)) + 12.0 * twist)
    }

    /// Uniform random poses with the mass centres about 6 Å apart.
    fn random_poses(n: usize) -> Vec<(Pose, Pose)> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
        (0..n)
            .map(|_| {
                let a = Pose {
                    position: Vector3::zeros(),
                    orientation: crate::random_orientation(&mut rng),
                };
                let dir = crate::random_orientation(&mut rng) * Vector3::z();
                let b = Pose {
                    position: dir * (6.0 + rng.gen_range(-0.1..0.1)),
                    orientation: crate::random_orientation(&mut rng),
                };
                (a, b)
            })
            .collect()
    }

    /// Random poses at fixed R must average exp(−u) to the exact value, as the
    /// scan's PMF does, so how the two lookup orders are combined matters.
    #[test]
    fn random_poses_average_the_exact_boltzmann_factor() {
        let energy = PairEnergy::new(physical_table(steep)).unwrap();
        let (mut table_sum, mut exact_sum) = (0.0, 0.0);
        for (a, b) in random_poses(20_000) {
            table_sum += (-finite(energy.energy(&a, &b, BIG_BOX)).energy).exp();
            exact_sum += (-steep(&a, &b) / KT).exp();
        }
        // Averaging the two orders' energies instead reads 0.81 of the exact value.
        approx::assert_relative_eq!(table_sum, exact_sum, max_relative = 0.05);
    }

    /// Sampling with the free energy w, the mean energy must give the exact
    /// ⟨u⟩. The free energy can't: it exceeds the mean wherever the stencil
    /// or the two lookup orders disagree.
    #[test]
    fn boltzmann_weighted_mean_energy_is_the_exact_mean_energy() {
        let energy = PairEnergy::new(physical_table(steep)).unwrap();
        let (mut weight, mut mean, mut exact_weight, mut exact_mean) = (0.0, 0.0, 0.0, 0.0);
        for (a, b) in random_poses(20_000) {
            let e = finite(energy.energy(&a, &b, BIG_BOX));
            let x = (-e.energy).exp();
            weight += x;
            mean += x * e.mean_energy;
            let u = steep(&a, &b) / KT;
            exact_weight += (-u).exp();
            exact_mean += (-u).exp() * u;
        }
        // Within 0.05 kT here; the free energy reads 0.9 kT high.
        approx::assert_abs_diff_eq!(mean / weight, exact_mean / exact_weight, epsilon = 0.2);
    }

    /// Averaging both lookup orders is only valid for identical bodies.
    #[test]
    fn table_for_different_bodies_is_rejected() {
        let mut t = table(|r, _, _| r);
        t.metadata.as_mut().unwrap().point_group = icotable::PointGroup::Asymmetric;
        assert!(PairEnergy::new(t).is_err());
    }

    /// A NaN distance would pass the range checks and read the contact bin.
    #[test]
    #[should_panic(expected = "finite")]
    fn non_finite_separation_is_a_bug_not_an_energy() {
        let energy = PairEnergy::new(table(|r, _, _| r)).unwrap();
        energy.energy(&pose(0.0, 0.0, 0.0), &pose(6.0, 0.0, 0.0), f64::INFINITY);
    }

    /// Below 2·rmax the minimum image would fold pairs back inside the table.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "box")]
    fn box_narrower_than_twice_the_table_range_is_a_bug() {
        let energy = PairEnergy::new(table(|r, _, _| r)).unwrap();
        assert_eq!(
            energy.min_box_length(),
            2.0 * crate::nbody::test_tables::RMAX
        );
        energy.energy(&pose(0.0, 0.0, 0.0), &pose(6.0, 0.0, 0.0), 10.0);
    }

    /// Depends on both orientations, and is deliberately not symmetric under
    /// exchange, as interpolation on different grid points makes real tables.
    fn anisotropic(r: f64, a: &Vector3, b: &Vector3) -> f64 {
        r + 3.0 * a.x - 2.0 * b.y + a.z * b.z
    }

    fn finite(u: PairValue) -> Interpolated {
        let PairValue::Finite(u) = u else {
            panic!("expected finite, got {u:?}")
        };
        u
    }

    #[test]
    fn moving_and_rotating_the_pair_together_changes_nothing() {
        let energy = PairEnergy::new(table(anisotropic)).unwrap();
        let (a, b) = (pose(1.0, 2.0, 3.0), pose(4.0, 6.5, 4.0));
        let turn = UnitQuaternion::from_euler_angles(1.2, 0.4, -0.7);
        let shift = Vector3::new(-30.0, 11.0, 7.5);
        let moved = |p: &Pose| Pose {
            position: turn * p.position + shift,
            orientation: turn * p.orientation,
        };
        let before = finite(energy.energy(&a, &b, BIG_BOX));
        let after = finite(energy.energy(&moved(&a), &moved(&b), BIG_BOX));
        approx::assert_relative_eq!(before.energy, after.energy, max_relative = 1e-9);
        approx::assert_relative_eq!(before.mean_energy, after.mean_energy, max_relative = 1e-9);
    }

    /// Detailed balance needs one energy per pair, whichever body is "first".
    #[test]
    fn swapping_the_bodies_gives_the_same_energy() {
        let energy = PairEnergy::new(table(anisotropic)).unwrap();
        let a = pose(1.0, 2.0, 3.0);
        let b = Pose {
            position: Vector3::new(4.0, 6.5, 4.0),
            orientation: UnitQuaternion::from_euler_angles(-0.5, 2.2, 0.9),
        };
        assert_eq!(
            energy.energy(&a, &b, BIG_BOX),
            energy.energy(&b, &a, BIG_BOX)
        );
    }

    #[test]
    fn nearest_periodic_image_is_used() {
        let energy = PairEnergy::new(table(|r, _, _| 2.0 * r)).unwrap();
        // 13.8 Å apart in a 20 Å box: the nearest image is 6.2 Å away.
        let u = energy.energy(&pose(0.5, 0.0, 0.0), &pose(14.3, 0.0, 0.0), 20.0);
        approx::assert_relative_eq!(finite(u).energy, 4.8407456, max_relative = 1e-6);
    }

    /// Without it there is no kT to convert to.
    #[test]
    fn table_without_a_temperature_is_rejected() {
        let mut t = table(|r, _, _| r);
        t.metadata = None;
        assert!(PairEnergy::new(t).is_err());
    }

    #[test]
    fn closer_than_the_table_is_an_overlap() {
        let energy = PairEnergy::new(table(|r, _, _| 2.0 * r)).unwrap();
        let u = energy.energy(&pose(0.0, 0.0, 0.0), &pose(0.0, 4.0, 0.0), BIG_BOX);
        assert_eq!(u, PairValue::Overlap);
    }

    /// The table marks slabs where every orientation is negligible in Boltzmann
    /// weight as repulsive; they must reject, not return a huge number.
    #[test]
    fn repulsive_slab_is_an_overlap() {
        let energy = PairEnergy::new(table(|r, _, _| if r < 5.5 { 1e5 } else { 0.0 })).unwrap();
        let u = energy.energy(&pose(0.0, 0.0, 0.0), &pose(5.1, 0.0, 0.0), BIG_BOX);
        assert_eq!(u, PairValue::Overlap);
    }

    #[test]
    fn beyond_the_table_is_the_tail_correction() {
        let mut t = table(|_, _, _| 0.0);
        let meta = t.metadata.as_mut().unwrap();
        meta.electric_prefactor = Some(5.0);
        meta.tail_terms = vec![icotable::TailCorrectionTerm {
            coefficient: 2.0,
            kappa: 0.1,
            power: 1,
        }];
        let energy = PairEnergy::new(t).unwrap();
        // 5 · 2 · exp(-0.1 · 10) / 10 = 0.3678794 kJ/mol = 0.1484010 kT
        let u = energy.energy(&pose(0.0, 0.0, 0.0), &pose(0.0, 0.0, 10.0), BIG_BOX);
        approx::assert_relative_eq!(finite(u).energy, 0.1484010, max_relative = 1e-5);
    }
}

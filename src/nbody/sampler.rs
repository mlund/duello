//! Native Metropolis sampler: single-body translate and rotate moves.

use super::pair_cache::PairCache;
use super::{PairEnergy, PairValue, Pose};
use crate::{UnitQuaternion, Vector3};
use anyhow::{bail, ensure, Result};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::sync::Arc;

/// What to simulate and how far each move reaches.
#[derive(Clone, Debug)]
pub struct SamplerSpec {
    pub n_bodies: usize,
    /// Cubic box side in Å.
    pub box_length: f64,
    /// Maximum displacement per axis in Å.
    pub translation: f64,
    /// Maximum rotation angle in radians.
    pub rotation: f64,
}

/// Snapshot of one copy of the system; the only thing consumers see.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub copy: usize,
    /// Sweeps completed, where one sweep is `n_bodies` attempted moves.
    pub sweep: u64,
    pub box_length: f64,
    pub poses: Vec<Pose>,
}

/// Tallies for one move type since the stats were last taken.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MoveCounter {
    pub attempts: u64,
    pub accepted: u64,
    /// Sum of squared displacements of accepted moves (Å² or rad²).
    pub sum_square_displacement: f64,
}

/// Both ratios are `None` without attempts, rather than a NaN that would slip
/// through comparisons in a UI or step tuner.
impl MoveCounter {
    pub fn acceptance(&self) -> Option<f64> {
        self.per_attempt(self.accepted as f64)
    }

    /// Per attempted move, so rejections count as zero: the sampling
    /// efficiency measure, unlike the mean over accepted moves only.
    pub fn mean_square_displacement(&self) -> Option<f64> {
        self.per_attempt(self.sum_square_displacement)
    }

    fn per_attempt(&self, total: f64) -> Option<f64> {
        (self.attempts > 0).then(|| total / self.attempts as f64)
    }

    fn record(&mut self, accepted: bool, square_displacement: f64) {
        self.attempts += 1;
        if accepted {
            self.accepted += 1;
            self.sum_square_displacement += square_displacement;
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MoveStats {
    pub translate: MoveCounter,
    pub rotate: MoveCounter,
}

impl MoveStats {
    /// Adds another copy's or period's tallies, so runs can pool copies.
    pub fn merge(&mut self, other: &MoveStats) {
        for (mine, theirs) in [
            (&mut self.translate, &other.translate),
            (&mut self.rotate, &other.rotate),
        ] {
            mine.attempts += theirs.attempts;
            mine.accepted += theirs.accepted;
            mine.sum_square_displacement += theirs.sum_square_displacement;
        }
    }
}

/// Steps beyond L/2 or π add nothing: the minimum image and the rotation
/// group wrap them back.
fn check_steps(translation: f64, rotation: f64, box_length: f64) -> Result<()> {
    let valid = |step: f64, max: f64| step.is_finite() && (0.0..=max).contains(&step);
    ensure!(
        valid(translation, box_length / 2.0) && valid(rotation, std::f64::consts::PI),
        "move steps must lie in [0, L/2] Å and [0, π] rad; got {translation}, {rotation}"
    );
    ensure!(
        translation > 0.0 || rotation > 0.0,
        "at least one move type needs a non-zero step"
    );
    Ok(())
}

/// One copy of the system, advanced by sequential Metropolis moves.
pub struct Sampler {
    /// Shared, so copies and observers don't each load the table.
    energy: Arc<PairEnergy>,
    spec: SamplerSpec,
    /// Change only through `accept`, which keeps `pairs` in step.
    poses: Vec<Pose>,
    pairs: PairCache,
    sweep: u64,
    /// Pcg64 is portable, so a seed gives the same run natively and in WASM.
    rng: Pcg64,
    stats: MoveStats,
}

impl Sampler {
    /// Inserts bodies at random non-overlapping poses. A zero step switches
    /// that move type off, e.g. rotations of spherical bodies.
    pub fn new(energy: Arc<PairEnergy>, spec: SamplerSpec, seed: u64) -> Result<Self> {
        ensure!(
            spec.box_length >= energy.min_box_length(),
            "box of {:.1} Å is below the {:.1} Å the table needs; use more bodies",
            spec.box_length,
            energy.min_box_length()
        );
        check_steps(spec.translation, spec.rotation, spec.box_length)?;
        let mut sampler = Self {
            energy,
            poses: Vec::with_capacity(spec.n_bodies),
            pairs: PairCache::new(0, |_, _| 0.0),
            spec,
            sweep: 0,
            rng: Pcg64::seed_from_u64(seed),
            stats: MoveStats::default(),
        };
        sampler.insert_all()?;
        let (poses, l) = (&sampler.poses, sampler.spec.box_length);
        let pair =
            |i: usize, j: usize| finite_or_zero(sampler.energy.energy(&poses[i], &poses[j], l), j);
        sampler.pairs = PairCache::new(poses.len(), pair);
        Ok(sampler)
    }

    /// One sweep is `n_bodies` attempted moves.
    pub fn advance(&mut self, sweeps: u64) {
        for _ in 0..sweeps {
            for _ in 0..self.spec.n_bodies {
                self.attempt_move();
            }
            self.sweep += 1;
        }
    }

    /// For step tuning; the same rules as at construction.
    pub fn set_steps(&mut self, translation: f64, rotation: f64) -> Result<()> {
        check_steps(translation, rotation, self.spec.box_length)?;
        self.spec.translation = translation;
        self.spec.rotation = rotation;
        Ok(())
    }

    /// Returns the stats since the last call and starts new ones, so a caller
    /// can leave out equilibration.
    pub fn take_move_stats(&mut self) -> MoveStats {
        std::mem::take(&mut self.stats)
    }

    /// Total energy (kT) of the current poses, kept by the pair cache
    /// instead of N²/2 table lookups.
    pub fn total_energy(&self) -> f64 {
        self.pairs.total()
    }

    pub fn frame(&self) -> Frame {
        Frame {
            copy: 0,
            sweep: self.sweep,
            box_length: self.spec.box_length,
            poses: self.poses.clone(),
        }
    }

    fn insert_all(&mut self) -> Result<()> {
        const MAX_TRIES: usize = 10_000;
        for i in 0..self.spec.n_bodies {
            let fits = (0..MAX_TRIES).find_map(|_| {
                let pose = self.random_pose();
                (!self.overlaps_any(&pose)).then_some(pose)
            });
            let Some(pose) = fits else {
                bail!("could not insert body {i} without overlap; lower the concentration");
            };
            self.poses.push(pose);
        }
        Ok(())
    }

    fn accept(&mut self, i: usize, trial: Pose, du: f64) {
        self.poses[i] = trial;
        self.pairs.commit(i, du);
    }

    fn overlaps_any(&self, pose: &Pose) -> bool {
        let l = self.spec.box_length;
        self.poses
            .iter()
            .any(|other| self.energy.energy(pose, other, l) == PairValue::Overlap)
    }

    fn attempt_move(&mut self) {
        let i = self.rng.gen_range(0..self.poses.len());
        let current = self.poses[i];
        let translate = self.choose_translation();
        let (trial, square_displacement) = if translate {
            self.translated(&current)
        } else {
            self.rotated(&current)
        };
        let du = self.energy_change(i, &trial);
        let accepted = self.metropolis(du);
        if let (true, Some(du)) = (accepted, du) {
            self.accept(i, trial, du);
        }
        let counter = match translate {
            true => &mut self.stats.translate,
            false => &mut self.stats.rotate,
        };
        counter.record(accepted, square_displacement);
    }

    /// Even odds, unless one move type is switched off by a zero step.
    fn choose_translation(&mut self) -> bool {
        match (self.spec.translation > 0.0, self.spec.rotation > 0.0) {
            (true, true) => self.rng.gen(),
            (translate, _) => translate,
        }
    }

    /// `du` is `None` for an overlap.
    fn metropolis(&mut self, du: Option<f64>) -> bool {
        match du {
            None => false, // overlap
            Some(du) => du <= 0.0 || self.rng.gen::<f64>() < (-du).exp(),
        }
    }

    /// Change in kT if body `i` moved to `trial`, staged in the pair cache
    /// for `accept`; `None` if it would overlap.
    fn energy_change(&mut self, i: usize, trial: &Pose) -> Option<f64> {
        let l = self.spec.box_length;
        let mut du = 0.0;
        for (j, other) in self.poses.iter().enumerate().filter(|&(j, _)| j != i) {
            let PairValue::Finite(new) = self.energy.energy(trial, other, l) else {
                return None;
            };
            du += self.pairs.stage(i, j, new);
        }
        Some(du)
    }

    /// Trial pose and its squared step; the step, not the wrapped difference.
    fn translated(&mut self, pose: &Pose) -> (Pose, f64) {
        let d = self.spec.translation;
        let step = Vector3::from_fn(|_, _| self.rng.gen_range(-d..d));
        let l = self.spec.box_length;
        let trial = Pose {
            position: (pose.position + step).map(|x| wrap(x, l)),
            orientation: pose.orientation,
        };
        (trial, step.norm_squared())
    }

    /// Uniform axis and symmetric angle, so the proposal is its own reverse.
    fn rotated(&mut self, pose: &Pose) -> (Pose, f64) {
        let angle = self.rng.gen_range(-self.spec.rotation..self.spec.rotation);
        let axis = nalgebra::Unit::new_unchecked(self.random_unit_vector());
        let trial = Pose {
            position: pose.position,
            orientation: UnitQuaternion::from_axis_angle(&axis, angle) * pose.orientation,
        };
        (trial, angle * angle)
    }

    fn random_pose(&mut self) -> Pose {
        let l = self.spec.box_length;
        Pose {
            position: Vector3::from_fn(|_, _| self.rng.gen_range(0.0..l)),
            orientation: self.random_orientation(),
        }
    }

    fn random_unit_vector(&mut self) -> Vector3 {
        let z: f64 = self.rng.gen_range(-1.0..1.0);
        let phi = self.rng.gen_range(0.0..std::f64::consts::TAU);
        let s = (1.0 - z * z).sqrt();
        Vector3::new(s * phi.cos(), s * phi.sin(), z)
    }

    fn random_orientation(&mut self) -> UnitQuaternion {
        crate::random_orientation(&mut self.rng)
    }
}

/// Into [0, l); `rem_euclid` can return `l` itself for tiny negative inputs.
fn wrap(x: f64, l: f64) -> f64 {
    let y = x.rem_euclid(l);
    if y >= l {
        0.0
    } else {
        y
    }
}

/// The current state never overlaps, as overlapping moves are always rejected.
fn finite_or_zero(u: PairValue, j: usize) -> f64 {
    match u {
        PairValue::Finite(u) => u,
        PairValue::Overlap => unreachable!("accepted state overlaps body {j}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbody::test_tables::table;

    fn ideal(n_bodies: usize, box_length: f64) -> Sampler {
        let energy = PairEnergy::new(table(|_, _, _| 0.0)).unwrap();
        let spec = SamplerSpec {
            n_bodies,
            box_length,
            translation: 3.0,
            rotation: 0.5,
        };
        Sampler::new(energy.into(), spec, 7).unwrap()
    }

    /// The cached total must be what a full pair sum gives, as energy
    /// statistics read it instead of recomputing every pair.
    #[test]
    fn total_energy_matches_a_full_pair_sum() {
        // A well that varies with distance, so pairs differ.
        let energy = Arc::new(PairEnergy::new(table(|r, _, _| r - 9.0)).unwrap());
        let spec = SamplerSpec {
            n_bodies: 20,
            box_length: 20.0,
            translation: 3.0,
            rotation: 0.5,
        };
        let mut sampler = Sampler::new(energy.clone(), spec, 3).unwrap();
        sampler.advance(50);
        let full = crate::nbody::EnergyStats::new(energy)
            .frame_energy(&sampler.frame())
            .unwrap();
        assert!(full.abs() > 1.0, "test needs interacting bodies: {full}");
        approx::assert_relative_eq!(sampler.total_energy(), full, max_relative = 1e-5);
    }

    #[test]
    fn frames_report_the_system_and_keep_bodies_in_the_box() {
        let mut sampler = ideal(10, 40.0);
        sampler.advance(5);
        let frame = sampler.frame();
        assert_eq!((frame.copy, frame.sweep, frame.box_length), (0, 5, 40.0));
        assert_eq!(frame.poses.len(), 10);
        let inside = |x: f64| (0.0..40.0).contains(&x);
        assert!(frame
            .poses
            .iter()
            .all(|p| p.position.iter().all(|&x| inside(x))));
    }

    #[test]
    fn same_seed_gives_the_same_run() {
        let run = || {
            let mut sampler = ideal(10, 40.0);
            sampler.advance(20);
            sampler.frame()
        };
        assert_eq!(run(), run());
    }

    /// One body has no partners, so every move is accepted, and the means are
    /// those of the proposals: uniform in ±d per axis gives ⟨Δr²⟩ = d², and a
    /// uniform angle in ±θ gives ⟨Δθ²⟩ = θ²/3.
    #[test]
    fn move_stats_report_acceptance_and_mean_square_displacement() {
        let mut sampler = ideal(1, 40.0); // translation 3 Å, rotation 0.5 rad
        sampler.advance(40_000);
        let stats = sampler.take_move_stats();
        for counter in [&stats.translate, &stats.rotate] {
            assert_eq!(counter.acceptance(), Some(1.0));
        }
        let (attempts, total) = (stats.translate.attempts + stats.rotate.attempts, 40_000);
        assert_eq!(attempts, total);
        approx::assert_relative_eq!(
            stats.translate.mean_square_displacement().unwrap(),
            9.0,
            max_relative = 0.03
        );
        approx::assert_relative_eq!(
            stats.rotate.mean_square_displacement().unwrap(),
            0.25 / 3.0,
            max_relative = 0.03
        );

        // Taking the stats resets them, so equilibration can be left out.
        assert_eq!(sampler.take_move_stats().translate.attempts, 0);
    }

    /// New steps act on the next moves: ⟨Δr²⟩ = d², ⟨Δθ²⟩ = θ²/3.
    #[test]
    fn set_steps_changes_the_next_moves() {
        let mut sampler = ideal(1, 40.0);
        sampler.set_steps(6.0, 1.2).unwrap();
        sampler.advance(40_000);
        let stats = sampler.take_move_stats();
        approx::assert_relative_eq!(
            stats.translate.mean_square_displacement().unwrap(),
            36.0,
            max_relative = 0.03
        );
        approx::assert_relative_eq!(
            stats.rotate.mean_square_displacement().unwrap(),
            1.44 / 3.0,
            max_relative = 0.03
        );
    }

    /// Beyond L/2 or π a larger step adds nothing.
    #[test]
    fn steps_out_of_range_are_errors() {
        let mut sampler = ideal(1, 40.0);
        for (d, theta) in [
            (-1.0, 0.1),
            (f64::NAN, 0.1),
            (20.5, 0.1),
            (1.0, 3.2),
            (0.0, 0.0),
        ] {
            assert!(sampler.set_steps(d, theta).is_err(), "{d}, {theta}");
        }
        assert!(sampler.set_steps(20.0, std::f64::consts::PI).is_ok());
    }

    /// Averages over one sampled body after every move.
    fn sample_single_body(sweeps: u64, mut observe: impl FnMut(&Pose)) {
        let mut sampler = ideal(1, 40.0);
        for _ in 0..sweeps {
            sampler.advance(1);
            observe(&sampler.frame().poses[0]);
        }
    }

    /// With nothing to interact with, the box and SO(3) are sampled uniformly:
    /// ⟨x⟩ = L/2, ⟨x²⟩ = L²/3, and a body axis has ⟨z⟩ = 0, ⟨z²⟩ = 1/3.
    #[test]
    fn free_body_samples_box_and_orientations_uniformly() {
        let (n, l) = (200_000, 40.0);
        let (mut x, mut x2, mut z, mut z2) = (0.0, 0.0, 0.0, 0.0);
        sample_single_body(n, |pose| {
            x += pose.position.x;
            x2 += pose.position.x.powi(2);
            let axis_z = (pose.orientation * Vector3::z()).z;
            z += axis_z;
            z2 += axis_z * axis_z;
        });
        let n = n as f64;
        approx::assert_relative_eq!(x / n, l / 2.0, max_relative = 0.03);
        approx::assert_relative_eq!(x2 / n, l * l / 3.0, max_relative = 0.03);
        approx::assert_abs_diff_eq!(z / n, 0.0, epsilon = 0.02);
        approx::assert_relative_eq!(z2 / n, 1.0 / 3.0, max_relative = 0.03);
    }

    /// Detailed balance: two bodies must sample P(R) ∝ R² exp(-βV(R)).
    /// Lookups snap R to the nearest bin (centres 5, 6, 7 Å; the last bin
    /// reaches rmax), and V = 0 beyond rmax as the table has no tail. For
    /// R < L/2 the whole shell is inside the box, so each interval [a, b) of
    /// constant V has weight exp(-βV)·(b³ - a³).
    #[test]
    fn two_bodies_sample_the_boltzmann_distance_distribution() {
        use crate::nbody::test_tables::KT;
        let v = |r: f64| match r {
            r if r < 5.5 => 2.0,
            r if r < 6.5 => -3.0,
            _ => 1.0,
        };
        let energy = PairEnergy::new(table(move |r, _, _| v(r))).unwrap();
        let spec = SamplerSpec {
            n_bodies: 2,
            box_length: 20.0,
            translation: 5.0,
            rotation: 0.1,
        };
        let mut sampler = Sampler::new(energy.into(), spec, 3).unwrap();

        let edges = [5.0, 5.5, 6.5, 8.0, 10.0];
        let energies = [2.0, -3.0, 1.0, 0.0]; // kJ/mol per interval
        let mut counts = [0usize; 4];
        // Enough that the rarest interval (~2%) has a few thousand samples.
        for _ in 0..400_000 {
            sampler.advance(1);
            let poses = sampler.frame().poses;
            let d = (poses[1].position - poses[0].position).map(|x| x - 20.0 * (x / 20.0).round());
            if let Some(k) = edges
                .windows(2)
                .position(|w| (w[0]..w[1]).contains(&d.norm()))
            {
                counts[k] += 1;
            }
        }
        let weights: Vec<f64> = (0..4)
            .map(|k| (-energies[k] / KT).exp() * (edges[k + 1].powi(3) - edges[k].powi(3)))
            .collect();
        let (total_w, total_n) = (weights.iter().sum::<f64>(), counts.iter().sum::<usize>());
        for k in 0..4 {
            let (expected, observed) = (weights[k] / total_w, counts[k] as f64 / total_n as f64);
            approx::assert_relative_eq!(observed, expected, max_relative = 0.05);
        }
    }

    /// Orientations: with U = ε(cos θ_a + cos θ_b), θ being the angle between a
    /// body's z axis and the direction to its partner, each direction is
    /// Boltzmann-distributed on the sphere, so inside the table's range
    /// ⟨cos θ⟩ = -L(βε), the Langevin function: -(coth 1 - 1) for βε = 1.
    #[test]
    fn two_bodies_sample_boltzmann_orientations() {
        use crate::nbody::test_tables::{physical_table, KT, RMAX, RMIN};
        let towards = |from: &Pose, d: &Vector3| (from.orientation.inverse() * d).normalize();
        let energy = PairEnergy::new(physical_table(|a, b| {
            let d = b.position - a.position;
            KT * (towards(a, &d).z + towards(b, &-d).z)
        }))
        .unwrap();
        let spec = SamplerSpec {
            n_bodies: 2,
            box_length: 16.0,
            translation: 4.0,
            rotation: 1.0,
        };
        let mut sampler = Sampler::new(energy.into(), spec, 5).unwrap();

        let (mut sum, mut n) = (0.0, 0);
        for _ in 0..200_000 {
            sampler.advance(1);
            let poses = sampler.frame().poses;
            let d = (poses[1].position - poses[0].position).map(|x| x - 16.0 * (x / 16.0).round());
            if (RMIN..RMAX).contains(&d.norm()) {
                sum += towards(&poses[0], &d).z + towards(&poses[1], &-d).z;
                n += 2;
            }
        }
        let langevin = 1.0 / 1.0_f64.tanh() - 1.0;
        approx::assert_abs_diff_eq!(sum / n as f64, -langevin, epsilon = 0.02);
    }

    /// A zero step switches that move off, e.g. rotations of spherical bodies.
    #[test]
    fn zero_step_switches_that_move_off() {
        let energy = PairEnergy::new(table(|_, _, _| 0.0)).unwrap();
        let spec = SamplerSpec {
            n_bodies: 1,
            box_length: 40.0,
            translation: 3.0,
            rotation: 0.0,
        };
        let mut sampler = Sampler::new(energy.into(), spec, 1).unwrap();
        sampler.advance(100);
        let stats = sampler.take_move_stats();
        assert_eq!((stats.translate.attempts, stats.rotate.attempts), (100, 0));
        assert_eq!(stats.rotate.acceptance(), None);
    }

    #[test]
    fn no_moves_at_all_is_an_error() {
        let energy = PairEnergy::new(table(|_, _, _| 0.0)).unwrap();
        let spec = SamplerSpec {
            n_bodies: 1,
            box_length: 40.0,
            translation: 0.0,
            rotation: 0.0,
        };
        assert!(Sampler::new(energy.into(), spec, 1).is_err());
    }

    /// Below 2·rmax the minimum image would fold pairs back into the table.
    #[test]
    fn box_too_small_for_the_table_is_an_error() {
        let energy = PairEnergy::new(table(|_, _, _| 0.0)).unwrap();
        let spec = SamplerSpec {
            n_bodies: 2,
            box_length: 15.0,
            translation: 1.0,
            rotation: 0.1,
        };
        let error = Sampler::new(energy.into(), spec, 1).err().unwrap();
        assert!(error.to_string().contains("box"), "{error}");
    }

    /// Hard cores of radius ≥ rmin/2 = 2.5 Å can't fill a 16 Å box 1000 times.
    #[test]
    fn impossible_insertion_is_an_error() {
        let energy = PairEnergy::new(table(|_, _, _| 0.0)).unwrap();
        let spec = SamplerSpec {
            n_bodies: 1000,
            box_length: 16.0,
            translation: 1.0,
            rotation: 0.1,
        };
        let error = Sampler::new(energy.into(), spec, 1).err().unwrap();
        assert!(error.to_string().contains("insert"), "{error}");
    }
}

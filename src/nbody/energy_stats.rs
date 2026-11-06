//! Total-energy statistics computed from frames.
//!
//! Recomputing from frames rather than keeping a running total avoids drift
//! (notably in f32 on the GPU), works the same for every sampler, and works
//! offline on saved trajectories.

use super::{Frame, PairEnergy, PairValue};
use anyhow::{bail, ensure, Result};
use std::sync::Arc;

/// Mean and population variance of the total energy, in kT and kT².
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnergySummary {
    pub frames: usize,
    pub mean: f64,
    pub variance: f64,
}

pub struct EnergyStats {
    /// Shared with the sampler, so the table is loaded once.
    energy: Arc<PairEnergy>,
    /// Welford's running mean and sum of squared deviations, stable for
    /// long runs where a naive Σu² − (Σu)²/n cancels badly.
    frames: usize,
    mean: f64,
    sum_squared_deviation: f64,
}

impl EnergyStats {
    pub fn new(energy: Arc<PairEnergy>) -> Self {
        Self {
            energy,
            frames: 0,
            mean: 0.0,
            sum_squared_deviation: 0.0,
        }
    }

    /// Adds the frame and returns its total energy in kT, which equilibration
    /// checks also need. Fails for a frame no valid sampler produces but a
    /// damaged or mismatched trajectory might: overlapping bodies, or a box
    /// below 2·rmax, where the minimum image would fold pairs into the table.
    pub fn observe(&mut self, frame: &Frame) -> Result<f64> {
        ensure!(
            frame.box_length >= self.energy.min_box_length(),
            "frame box of {:.1} Å is below the {:.1} Å the table needs",
            frame.box_length,
            self.energy.min_box_length()
        );
        let u = self.total_energy(frame)?;
        self.frames += 1;
        let delta = u - self.mean;
        self.mean += delta / self.frames as f64;
        self.sum_squared_deviation += delta * (u - self.mean);
        Ok(u)
    }

    pub fn summary(&self) -> EnergySummary {
        let variance = match self.frames {
            0 => 0.0,
            n => self.sum_squared_deviation / n as f64,
        };
        EnergySummary {
            frames: self.frames,
            mean: self.mean,
            variance,
        }
    }

    fn total_energy(&self, frame: &Frame) -> Result<f64> {
        let mut total = 0.0;
        for (i, a) in frame.poses.iter().enumerate() {
            for (j, b) in frame.poses.iter().enumerate().skip(i + 1) {
                match self.energy.energy(a, b, frame.box_length) {
                    PairValue::Finite(u) => total += u,
                    PairValue::Overlap => {
                        bail!("bodies {i} and {j} overlap in frame {}", frame.sweep)
                    }
                }
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbody::test_tables::{table, KT};
    use crate::nbody::Pose;
    use crate::{UnitQuaternion, Vector3};

    fn frame(positions: &[[f64; 3]]) -> Frame {
        let poses = positions
            .iter()
            .map(|&[x, y, z]| Pose {
                position: Vector3::new(x, y, z),
                orientation: UnitQuaternion::identity(),
            })
            .collect();
        Frame {
            copy: 0,
            sweep: 0,
            box_length: 100.0,
            poses,
        }
    }

    fn linear_in_r() -> EnergyStats {
        EnergyStats::new(PairEnergy::new(table(|r, _, _| 2.0 * r)).unwrap().into())
    }

    #[test]
    fn total_energy_is_the_sum_over_pairs() {
        let mut stats = linear_in_r();
        // 6.2 Å → bin 6 → 12 kJ/mol; 7.1 Å → bin 7 → 14 kJ/mol; the third
        // pair is 9.4 Å apart, beyond rmax with no tail: 0.
        stats
            .observe(&frame(&[[0.0, 0.0, 0.0], [6.2, 0.0, 0.0], [0.0, 7.1, 0.0]]))
            .unwrap();
        approx::assert_relative_eq!(stats.summary().mean, 26.0 / KT, max_relative = 1e-6);
    }

    /// A saved trajectory can be damaged or from another table; below 2·rmax
    /// the minimum image would silently fold pairs back into the table.
    #[test]
    fn frame_box_too_small_for_the_table_is_an_error() {
        let mut stats = linear_in_r();
        let mut small = frame(&[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]);
        small.box_length = 12.0;
        assert!(stats.observe(&small).is_err());
    }

    #[test]
    fn mean_and_population_variance_over_frames() {
        let mut stats = linear_in_r();
        for r in [5.0, 6.0, 7.0] {
            stats
                .observe(&frame(&[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]))
                .unwrap();
        }
        // 10, 12, 14 kJ/mol: mean 12, population variance 8/3.
        let summary = stats.summary();
        assert_eq!(summary.frames, 3);
        approx::assert_relative_eq!(summary.mean, 12.0 / KT, max_relative = 1e-6);
        approx::assert_relative_eq!(summary.variance, 8.0 / 3.0 / (KT * KT), max_relative = 1e-6);
    }

    /// Sampled two-body energies against exact integration over the box: the
    /// separation is uniform in the box volume, weighted by exp(-βU). Shells of
    /// constant V (nearest-bin lookup, see the sampler's distance test) add
    /// exp(-βV)·(4π/3)(b³ - a³); the rest of the box has U = 0.
    #[test]
    fn sampled_mean_and_variance_match_exact_integration() {
        use crate::nbody::{Sampler, SamplerSpec};
        use std::f64::consts::PI;
        let v = |r: f64| match r {
            r if r < 5.5 => 2.0,
            r if r < 6.5 => -3.0,
            _ => 1.0,
        };
        let pair = || PairEnergy::new(table(move |r, _, _| v(r))).unwrap();
        let l = 20.0;
        let spec = SamplerSpec {
            n_bodies: 2,
            box_length: l,
            translation: 5.0,
            rotation: 0.1,
        };
        let mut sampler = Sampler::new(pair().into(), spec, 11).unwrap();
        let mut stats = EnergyStats::new(pair().into());
        for _ in 0..400_000 {
            sampler.advance(1);
            stats.observe(&sampler.frame()).unwrap();
        }

        let shells = [(5.0, 5.5, 2.0), (5.5, 6.5, -3.0), (6.5, 8.0, 1.0)];
        let shell_volume = |a: f64, b: f64| 4.0 * PI / 3.0 * (b.powi(3) - a.powi(3));
        let outside = l.powi(3) - 4.0 * PI / 3.0 * 8.0_f64.powi(3);
        let (mut z, mut u1, mut u2) = (outside, 0.0, 0.0);
        for (a, b, energy) in shells {
            let (bu, w) = (energy / KT, (-energy / KT).exp() * shell_volume(a, b));
            (z, u1, u2) = (z + w, u1 + w * bu, u2 + w * bu * bu);
        }
        let (mean, variance) = (u1 / z, u2 / z - (u1 / z).powi(2));

        let summary = stats.summary();
        approx::assert_relative_eq!(summary.mean, mean, max_relative = 0.03);
        approx::assert_relative_eq!(summary.variance, variance, max_relative = 0.05);
    }
}

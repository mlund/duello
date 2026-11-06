//! Centre-of-mass radial distribution function, g(r).

use super::energy::minimum_image;
use super::scattering::jackknife_error;
use super::Frame;

/// g(r) at bin centres; `error` is the jackknife spread over copies, `None`
/// with fewer than two observed copies.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RdfCurve {
    pub r: Vec<f64>,
    pub g: Vec<f64>,
    pub error: Option<Vec<f64>>,
}

impl RdfCurve {
    /// One row per bin; the error is NaN without enough copies for one.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("r/Å,g,g_error\n");
        let error = self.error.as_deref().unwrap_or_default();
        for k in 0..self.r.len() {
            let e = error.get(k).copied().unwrap_or(f64::NAN);
            csv += &format!("{:.2},{:.5},{:.5}\n", self.r[k], self.g[k], e);
        }
        csv
    }
}

/// Bin width in Å.
const DR: f64 = 0.5;

/// Mass-centre pair distances binned per copy, out to L/2 where the minimum
/// image is still unique. Copies stay apart so their spread gives the error.
pub struct PairCorrelation {
    n_bodies: usize,
    box_length: f64,
    /// Pairs an uncorrelated fluid puts in each bin per frame; empty for
    /// fewer than two bodies, which have no pairs.
    ideal: Vec<f64>,
    copies: Vec<CopyCounts>,
}

#[derive(Clone)]
struct CopyCounts {
    counts: Vec<f64>,
    frames: u64,
}

impl PairCorrelation {
    pub fn new(n_bodies: usize, box_length: f64) -> Self {
        let bins = if n_bodies < 2 {
            0
        } else {
            (box_length / 2.0 / DR) as usize
        };
        Self {
            n_bodies,
            box_length,
            ideal: (0..bins)
                .map(|k| ideal_pairs(n_bodies, box_length, k))
                .collect(),
            copies: Vec::new(),
        }
    }

    /// Frames must come from this run's box and body count, which are fixed.
    pub fn observe(&mut self, frame: &Frame) {
        debug_assert_eq!(frame.box_length, self.box_length);
        debug_assert_eq!(frame.poses.len(), self.n_bodies);
        let (box_length, r_max) = (self.box_length, self.ideal.len() as f64 * DR);
        let copy = self.copy_mut(frame.copy);
        for (i, a) in frame.poses.iter().enumerate() {
            for b in &frame.poses[i + 1..] {
                let d2 = minimum_image(b.position - a.position, box_length).norm_squared();
                // Most pairs lie beyond the last bin; skip their square root.
                if d2 < r_max * r_max {
                    copy.counts[(d2.sqrt() / DR) as usize] += 1.0;
                }
            }
        }
        copy.frames += 1;
    }

    fn copy_mut(&mut self, copy: usize) -> &mut CopyCounts {
        if self.copies.len() <= copy {
            let empty = CopyCounts {
                counts: vec![0.0; self.ideal.len()],
                frames: 0,
            };
            self.copies.resize(copy + 1, empty);
        }
        &mut self.copies[copy]
    }

    pub fn reset(&mut self) {
        self.copies.clear();
    }

    /// Empty until a frame is observed, and for fewer than two bodies.
    pub fn curve(&self) -> RdfCurve {
        let observed: Vec<&CopyCounts> = self.copies.iter().filter(|c| c.frames > 0).collect();
        if observed.is_empty() || self.ideal.is_empty() {
            return RdfCurve::default();
        }
        let bins = 0..self.ideal.len();
        let r = bins.clone().map(|k| (k as f64 + 0.5) * DR).collect();
        let g = bins.clone().map(|k| self.pooled(&observed, k)).collect();
        let error = (observed.len() > 1).then(|| {
            bins.map(|k| jackknife_error(observed.iter().map(|c| self.ratio_terms(c, k))))
                .collect()
        });
        RdfCurve { r, g, error }
    }

    fn pooled(&self, copies: &[&CopyCounts], k: usize) -> f64 {
        let (counts, expected) = copies
            .iter()
            .map(|c| self.ratio_terms(c, k))
            .fold((0.0, 0.0), |(n, e), (cn, ce)| (n + cn, e + ce));
        counts / expected
    }

    /// (pairs counted, pairs an uncorrelated fluid would give) in bin `k`.
    fn ratio_terms(&self, copy: &CopyCounts, k: usize) -> (f64, f64) {
        (copy.counts[k], copy.frames as f64 * self.ideal[k])
    }
}

/// N(N−1)/2 pairs, each in bin `k`'s shell with probability shell volume / V.
/// Using N − 1, not N, makes g → 1 exactly at any N.
fn ideal_pairs(n_bodies: usize, box_length: f64, k: usize) -> f64 {
    let n = n_bodies as f64;
    let (inner, outer) = (k as f64 * DR, (k + 1) as f64 * DR);
    let shell = 4.0 / 3.0 * std::f64::consts::PI * (outer.powi(3) - inner.powi(3));
    n * (n - 1.0) / 2.0 * shell / box_length.powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbody::Pose;
    use crate::{UnitQuaternion, Vector3};

    fn frame(copy: usize, positions: &[[f64; 3]], box_length: f64) -> Frame {
        let poses = positions
            .iter()
            .map(|&p| Pose {
                position: Vector3::from(p),
                orientation: UnitQuaternion::identity(),
            })
            .collect();
        Frame {
            copy,
            sweep: 0,
            box_length,
            poses,
        }
    }

    /// Two bodies `d` Å apart along x in the L = 100 Å box most tests use.
    fn pair_at(copy: usize, d: f64) -> Frame {
        frame(copy, &[[0.0, 0.0, 0.0], [d, 0.0, 0.0]], 100.0)
    }

    fn two_bodies() -> PairCorrelation {
        PairCorrelation::new(2, 100.0)
    }

    /// (bin centre, value) where `values` is non-zero.
    fn nonzero(r: &[f64], values: &[f64]) -> Vec<(f64, f64)> {
        r.iter()
            .zip(values)
            .filter(|(_, v)| **v != 0.0)
            .map(|(&r, &v)| (r, v))
            .collect()
    }

    /// Non-zero bins as (bin centre, g).
    fn filled(curve: &RdfCurve) -> Vec<(f64, f64)> {
        nonzero(&curve.r, &curve.g)
    }

    /// One pair in V = 10⁶ Å³, so the ideal-gas count in [10, 10.5) is
    /// 4π/3·(10.5³ − 10³)/V = 660.258/10⁶, and g = 10⁶/660.258 = 1514.559.
    #[test]
    fn one_pair_fills_the_bin_of_its_separation() {
        let mut rdf = two_bodies();
        rdf.observe(&pair_at(0, 10.0));
        let filled = filled(&rdf.curve());
        assert_eq!(filled.len(), 1, "{filled:?}");
        let (r, g) = filled[0];
        assert_eq!(r, 10.25);
        approx::assert_relative_eq!(g, 1514.559, max_relative = 1e-6);
    }

    /// x = 2 and x = 96 in L = 100 are 6 Å apart through the boundary; as
    /// above, g = 10⁶/(4π/3·(6.5³ − 6³)) = 4072.195 in [6, 6.5).
    #[test]
    fn pairs_count_at_their_minimum_image_distance() {
        let mut rdf = two_bodies();
        rdf.observe(&frame(0, &[[2.0, 0.0, 0.0], [96.0, 0.0, 0.0]], 100.0));
        let filled = filled(&rdf.curve());
        assert_eq!(filled.len(), 1, "{filled:?}");
        let (r, g) = filled[0];
        assert_eq!(r, 6.25);
        approx::assert_relative_eq!(g, 4072.195, max_relative = 1e-6);
    }

    /// g averages frames: repeating one changes nothing, and two different
    /// frames each fill their bin at half the one-frame value (1514.559/2).
    #[test]
    fn g_is_the_average_over_frames() {
        let (near, far) = (pair_at(0, 10.0), pair_at(0, 20.0));
        let mut once = two_bodies();
        once.observe(&near);
        let mut twice = two_bodies();
        twice.observe(&near);
        twice.observe(&near);
        assert_eq!(once.curve(), twice.curve());

        let mut mixed = two_bodies();
        mixed.observe(&near);
        mixed.observe(&far);
        let filled = filled(&mixed.curve());
        assert_eq!(filled[0].0, 10.25);
        approx::assert_relative_eq!(filled[0].1, 757.2797, max_relative = 1e-6);
    }

    /// An uncorrelated fluid has g = 1. At r ≥ 10 Å each bin expects over
    /// 1300 of the 2000·1999/2 pairs, so noise stays near 3%.
    #[test]
    fn uniform_points_give_g_of_one() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
        let positions: Vec<[f64; 3]> = (0..2000)
            .map(|_| [0; 3].map(|_| rng.gen_range(0.0..100.0)))
            .collect();
        let mut rdf = PairCorrelation::new(2000, 100.0);
        rdf.observe(&frame(0, &positions, 100.0));
        let curve = rdf.curve();
        for (r, g) in curve.r.iter().zip(&curve.g).filter(|(r, _)| **r >= 10.0) {
            assert!((g - 1.0).abs() < 0.1, "g({r}) = {g}");
        }
        assert_eq!(curve.r.last(), Some(&49.75), "bins reach L/2");
    }

    /// Copy 0 has its pair at 10 Å, copy 1 at 20 Å. Leaving out either
    /// copy gives g(10.25) = 0 or 1514.559, so the jackknife error is
    /// √(½·Σ(gᵢ − 757.28)²) = 757.28; bins neither copy fills have none.
    #[test]
    fn copies_disagreeing_give_error_only_where_they_differ() {
        let mut rdf = two_bodies();
        rdf.observe(&pair_at(0, 10.0));
        rdf.observe(&pair_at(1, 20.0));
        let curve = rdf.curve();
        let error = curve.error.expect("two copies give an error");
        let bins: Vec<f64> = nonzero(&curve.r, &error).iter().map(|(r, _)| *r).collect();
        assert_eq!(bins, [10.25, 20.25]);
        approx::assert_relative_eq!(error[20], 757.2797, max_relative = 1e-6);
    }

    #[test]
    fn one_copy_has_no_error() {
        let mut rdf = two_bodies();
        rdf.observe(&pair_at(0, 10.0));
        assert_eq!(rdf.curve().error, None);
    }

    #[test]
    fn reset_forgets_every_frame() {
        let mut rdf = two_bodies();
        rdf.observe(&pair_at(0, 10.0));
        rdf.reset();
        assert_eq!(rdf.curve(), RdfCurve::default());
    }

    /// One body has no pairs, so there is no g(r) to give, not NaN.
    #[test]
    fn a_single_body_has_no_curve() {
        let mut rdf = PairCorrelation::new(1, 100.0);
        rdf.observe(&frame(0, &[[0.0, 0.0, 0.0]], 100.0));
        assert_eq!(rdf.curve(), RdfCurve::default());
    }

    #[test]
    fn csv_has_a_header_and_one_row_per_bin() {
        let mut rdf = two_bodies();
        rdf.observe(&pair_at(0, 10.0));
        let csv = rdf.curve().to_csv();
        let mut lines = csv.lines();
        assert_eq!(lines.next(), Some("r/Å,g,g_error"));
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), 100, "0.5 Å bins to L/2 = 50 Å");
        assert_eq!(rows[20], "10.25,1514.55933,NaN");
    }
}

//! Scattering intensity from many-body frames, via pripps' box scatterer.

use super::Frame;
use crate::archive::CgBody;
use crate::Vector3;
use anyhow::{ensure, Result};

/// Frame-averaged scattering per molecule. `s_eff = intensity / form_factor`
/// is a ratio of averages, which is what an experiment measures.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScatteringCurve {
    /// Å⁻¹. q-points where the form factor is zero are left out, as S_eff is
    /// undefined there.
    pub q: Vec<f64>,
    /// ⟨|Σ_m F_m|²⟩ / N
    pub intensity: Vec<f64>,
    /// ⟨Σ_m |F_m|²⟩ / N, the at-concentration form factor
    pub form_factor: Vec<f64>,
    pub s_eff: Vec<f64>,
    /// Jackknife standard error of `s_eff` over copies; `None` with fewer
    /// than two copies.
    pub s_eff_error: Option<Vec<f64>>,
    pub frames: usize,
}

impl ScatteringCurve {
    /// One row per q; the error is NaN without enough copies for one.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("q/Å⁻¹,I,P,S_eff,S_eff_error\n");
        for k in 0..self.q.len() {
            let error = self.s_eff_error.as_ref().map_or(f64::NAN, |e| e[k]);
            csv += &format!(
                "{:.5},{:.6e},{:.6e},{:.5},{:.5}\n",
                self.q[k], self.intensity[k], self.form_factor[k], self.s_eff[k], error
            );
        }
        csv
    }
}

/// Where the scattering sums run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Cpu,
    /// Phases in f32; fails if no GPU adapter is available.
    Gpu,
}

/// Sums over the frames of one copy; copies are kept apart so their spread
/// gives an error estimate that frame-to-frame correlation can't fool.
#[derive(Clone, Debug, Default)]
struct CopySums {
    frames: usize,
    total: Vec<f64>,
    self_term: Vec<f64>,
}

/// Far above any run; a larger index means a damaged or foreign frame.
const MAX_COPIES: usize = 1 << 16;

pub struct Scattering {
    reference: CgBody,
    n_bodies: usize,
    /// q-vectors depend on the box, so every frame must have this one.
    box_length: f64,
    scatterer: pripps::BoxScatterer,
    q: Vec<f64>,
    copies: Vec<CopySums>,
    /// Reused bead positions, body after body as the scatterer expects.
    positions: Vec<[f64; 3]>,
}

impl Scattering {
    /// `q_max` (Å⁻¹) bounds the sampled q-vectors; it must lie inside the form
    /// factors' range (0.75 Å⁻¹ for pripps' residue form factors). Async as
    /// GPU setup is.
    pub async fn new(
        reference: &CgBody,
        n_bodies: usize,
        box_length: f64,
        q_max: f64,
        formfactors_yaml: &str,
        device: Device,
    ) -> Result<Self> {
        let p = p_max(box_length, q_max);
        ensure!(
            p > 0,
            "q_max {q_max} Å⁻¹ is below the shortest [111] q-vector of a {box_length} Å box"
        );
        let kinds = &reference.kinds;
        let scatterer = match device {
            Device::Cpu => pripps::BoxScatterer::new_cpu(formfactors_yaml, kinds, n_bodies, p)?,
            Device::Gpu => {
                pripps::BoxScatterer::new_gpu(formfactors_yaml, kinds, n_bodies, p).await?
            }
        };
        Ok(Self {
            reference: reference.clone(),
            n_bodies,
            box_length,
            scatterer,
            q: Vec::new(),
            copies: Vec::new(),
            positions: Vec::with_capacity(n_bodies * reference.positions.len()),
        })
    }

    pub async fn observe(&mut self, frame: &Frame) -> Result<()> {
        self.check_frame(frame)?;
        self.fill_positions(frame);
        let result = self
            .scatterer
            .evaluate(&self.positions, [self.box_length; 3])
            .await?;
        if self.q.is_empty() {
            self.q = result.q;
        }
        ensure!(
            result.total.len() == self.q.len() && result.self_term.len() == self.q.len(),
            "scattering returned {} q-points, expected {}",
            result.total.len(),
            self.q.len()
        );
        self.accumulate(frame.copy, &result.total, &result.self_term);
        Ok(())
    }

    fn check_frame(&self, frame: &Frame) -> Result<()> {
        ensure!(
            frame.box_length == self.box_length,
            "frame box {} Å differs from the {} Å the q-vectors were set up for",
            frame.box_length,
            self.box_length
        );
        ensure!(
            frame.poses.len() == self.n_bodies,
            "frame has {} bodies, expected {}",
            frame.poses.len(),
            self.n_bodies
        );
        ensure!(
            frame.copy < MAX_COPIES,
            "implausible copy index {}",
            frame.copy
        );
        Ok(())
    }

    /// R + q·r for every bead, without a temporary per body.
    fn fill_positions(&mut self, frame: &Frame) {
        self.positions.clear();
        for pose in &frame.poses {
            for &r in &self.reference.positions {
                let p = pose.position + pose.orientation * Vector3::from(r);
                self.positions.push([p.x, p.y, p.z]);
            }
        }
    }

    fn accumulate(&mut self, copy: usize, total: &[f64], self_term: &[f64]) {
        if self.copies.len() <= copy {
            self.copies.resize(copy + 1, CopySums::default());
        }
        let sums = &mut self.copies[copy];
        if sums.frames == 0 {
            sums.total = vec![0.0; total.len()];
            sums.self_term = vec![0.0; self_term.len()];
        }
        sums.frames += 1;
        add_into(&mut sums.total, total);
        add_into(&mut sums.self_term, self_term);
    }

    /// Forgets all frames; the q-grid and form factors are kept.
    pub fn reset(&mut self) {
        self.copies.clear();
    }

    pub fn curve(&self) -> ScatteringCurve {
        let observed: Vec<&CopySums> = self.copies.iter().filter(|c| c.frames > 0).collect();
        let frames: usize = observed.iter().map(|c| c.frames).sum();
        if frames == 0 {
            return ScatteringCurve::default();
        }
        let sum = |k: usize, field: fn(&CopySums) -> &Vec<f64>| -> f64 {
            observed.iter().map(|c| field(c)[k]).sum()
        };
        let per_molecule = (frames * self.n_bodies) as f64;
        let mut curve = ScatteringCurve {
            frames,
            s_eff_error: (observed.len() > 1).then(Vec::new),
            ..Default::default()
        };
        for (k, &q) in self.q.iter().enumerate() {
            let (total, self_term) = (sum(k, |c| &c.total), sum(k, |c| &c.self_term));
            if self_term == 0.0 {
                continue;
            }
            curve.q.push(q);
            curve.intensity.push(total / per_molecule);
            curve.form_factor.push(self_term / per_molecule);
            curve.s_eff.push(total / self_term);
            if let Some(errors) = &mut curve.s_eff_error {
                errors.push(jackknife_error(&observed, k, total, self_term));
            }
        }
        curve
    }
}

fn add_into(sum: &mut [f64], values: &[f64]) {
    sum.iter_mut().zip(values).for_each(|(s, v)| *s += v);
}

/// Leave-one-copy-out jackknife on the pooled ratio Σtotal/Σself: it weights
/// copies by their frames, as the pooled S_eff does.
fn jackknife_error(copies: &[&CopySums], k: usize, total: f64, self_term: f64) -> f64 {
    let n = copies.len() as f64;
    let leave_out: Vec<f64> = copies
        .iter()
        .map(|c| (total - c.total[k]) / (self_term - c.self_term[k]))
        .collect();
    let mean = leave_out.iter().sum::<f64>() / n;
    let spread = leave_out.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
    ((n - 1.0) / n * spread).sqrt()
}

/// Largest multiple p of the shortest q-vector such that every sampled
/// vector, the longest being p·(2π/L)·√3 along [111], stays within `q_max`.
fn p_max(box_length: f64, q_max: f64) -> usize {
    let longest_unit = std::f64::consts::TAU / box_length * 3f64.sqrt();
    // Small tolerance: a q_max chosen at exactly a shell should include it.
    (q_max / longest_unit + 1e-9).floor() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbody::Pose;
    use crate::UnitQuaternion;

    const UNIT: &str = "X: { formfactor: !Constant 1.0 }\nZ: { formfactor: !Constant 0.0 }\n";

    fn body(beads: &[[f64; 3]], kind: &str) -> CgBody {
        let n = beads.len();
        CgBody {
            positions: beads.to_vec(),
            masses: vec![1.0; n],
            charges: vec![0.0; n],
            radii: vec![1.0; n],
            kinds: vec![kind.into(); n],
        }
    }

    fn bead(kind: &str) -> CgBody {
        body(&[[0.0; 3]], kind)
    }

    fn scattering(reference: &CgBody, n: usize, l: f64, q_max: f64) -> Scattering {
        pollster::block_on(Scattering::new(reference, n, l, q_max, UNIT, Device::Cpu)).unwrap()
    }

    fn observe(scattering: &mut Scattering, frame: &Frame) {
        pollster::block_on(scattering.observe(frame)).unwrap();
    }

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

    /// The longest vector at p is p·(2π/60)·√3 = 0.1814·p: p = 2 fits 0.4,
    /// p = 3 (0.544) does not.
    #[test]
    fn p_max_keeps_all_vectors_inside_q_max() {
        assert_eq!(p_max(60.0, 0.4), 2);
        assert_eq!(p_max(60.0, 0.55), 3);
    }

    /// Two unit scatterers L/4 apart (see pripps' BoxScatterer test): box
    /// intensity 10/3, 8/3, 2 per Miller shell, self term 2; per molecule
    /// halved, and S_eff = total/self.
    #[test]
    fn two_bodies_give_the_worked_curve() {
        let mut s = scattering(&bead("X"), 2, 40.0, 0.3);
        observe(
            &mut s,
            &frame(0, &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], 40.0),
        );
        let curve = s.curve();
        let expected = [10.0 / 3.0, 8.0 / 3.0, 2.0];
        assert_eq!(curve.frames, 1);
        for k in 0..3 {
            approx::assert_relative_eq!(
                curve.intensity[k],
                expected[k] / 2.0,
                max_relative = 1e-12
            );
            approx::assert_relative_eq!(curve.form_factor[k], 1.0, max_relative = 1e-12);
            approx::assert_relative_eq!(curve.s_eff[k], expected[k] / 2.0, max_relative = 1e-12);
        }
        assert_eq!(curve.s_eff_error, None);
    }

    /// Beads must be grouped by body: each body is two unit beads L/4 apart,
    /// so its own |F|² is the worked 10/3, 8/3, 2 per shell. Pairing beads
    /// across bodies would give a different self term.
    #[test]
    fn self_term_groups_beads_by_body() {
        let pair = body(&[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], "X");
        let mut s = scattering(&pair, 2, 40.0, 0.3);
        observe(
            &mut s,
            &frame(0, &[[1.0, 3.0, 5.0], [21.0, 17.0, 11.0]], 40.0),
        );
        let expected = [10.0 / 3.0, 8.0 / 3.0, 2.0];
        for (p, e) in s.curve().form_factor.iter().zip(expected) {
            approx::assert_relative_eq!(*p, e, max_relative = 1e-12);
        }
    }

    /// Averaged over orientations, the form factor of a rigid body of unit
    /// scatterers is Debye's Σᵢⱼ sin(q rᵢⱼ)/(q rᵢⱼ), independently of the
    /// few q-vector directions sampled per frame.
    #[test]
    fn orientation_averaged_form_factor_is_debye() {
        use rand::{Rng, SeedableRng};
        let triangle = body(&[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 7.0, 0.0]], "X");
        let l = 100.0;
        let mut s = scattering(&triangle, 1, l, 0.4);
        let mut rng = rand_pcg::Pcg64::seed_from_u64(2);
        for _ in 0..3000 {
            let axis = Vector3::from_fn(|_, _| rng.gen_range(-1.0..1.0));
            let pose = Pose {
                position: Vector3::new(50.0, 50.0, 50.0),
                orientation: UnitQuaternion::from_scaled_axis(axis * 3.0),
            };
            let frame = Frame {
                copy: 0,
                sweep: 0,
                box_length: l,
                poses: vec![pose],
            };
            observe(&mut s, &frame);
        }
        let r = |i: usize, j: usize| {
            (Vector3::from(triangle.positions[i]) - Vector3::from(triangle.positions[j])).norm()
        };
        let debye = |q: f64| {
            let mut sum = 3.0; // i = j terms
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                sum += 2.0 * (q * r(i, j)).sin() / (q * r(i, j));
            }
            sum
        };
        let curve = s.curve();
        for (q, p) in curve.q.iter().zip(&curve.form_factor) {
            approx::assert_relative_eq!(*p, debye(*q), max_relative = 0.03);
        }
    }

    /// A bead without a form factor would silently scatter nothing.
    #[test]
    fn bead_kind_without_form_factor_is_an_error() {
        let result = pollster::block_on(Scattering::new(
            &bead("NTR"),
            2,
            40.0,
            0.3,
            UNIT,
            Device::Cpu,
        ));
        let error = result.err().unwrap();
        assert!(error.to_string().contains("NTR"), "{error}");
    }

    /// S_eff is undefined where nothing scatters; those points are left out
    /// rather than reported as NaN.
    #[test]
    fn points_without_scattering_are_left_out() {
        let mut s = scattering(&bead("Z"), 2, 40.0, 0.3);
        observe(&mut s, &frame(0, &[[0.0; 3], [10.0, 0.0, 0.0]], 40.0));
        let curve = s.curve();
        assert_eq!(curve.frames, 1);
        assert!(curve.q.is_empty() && curve.s_eff.is_empty());
    }

    #[test]
    fn frame_with_the_wrong_body_count_is_an_error() {
        let mut s = scattering(&bead("X"), 2, 40.0, 0.3);
        let result = pollster::block_on(s.observe(&frame(0, &[[0.0; 3]], 40.0)));
        assert!(result.err().unwrap().to_string().contains("bodies"));
    }

    /// Without interactions the cross terms average out: S_eff = 1 within
    /// the jackknife error over copies.
    #[test]
    fn ideal_gas_has_unit_structure_factor() {
        use rand::{Rng, SeedableRng};
        let (n, l) = (20, 50.0);
        let mut rng = rand_pcg::Pcg64::seed_from_u64(9);
        let mut s = scattering(&bead("X"), n, l, 0.4);
        for copy in 0..4 {
            for _ in 0..500 {
                let positions: Vec<[f64; 3]> = (0..n)
                    .map(|_| [0; 3].map(|_| rng.gen_range(0.0..l)))
                    .collect();
                observe(&mut s, &frame(copy, &positions, l));
            }
        }
        let curve = s.curve();
        let errors = curve.s_eff_error.unwrap();
        for (s, e) in curve.s_eff.iter().zip(&errors) {
            assert!((s - 1.0).abs() < 4.0 * e, "S_eff = {s} ± {e}");
        }
    }
}

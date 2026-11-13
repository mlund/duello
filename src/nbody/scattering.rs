//! Scattering intensity from many-body frames, via pripps' box scatterer.

use super::pending::{self, PendingFrames};
use super::Frame;
use crate::archive::CgBody;
use crate::Vector3;
use anyhow::{ensure, Result};
use std::future::Future;
use std::sync::Arc;

/// Frame-averaged scattering per molecule. `s_eff = intensity / form_factor`
/// is a ratio of averages, which is what an experiment measures.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Analytic S(q) of the equivalent charged sphere; empty unless filled
    /// by `NBodyRun`, which knows the protein and solution.
    pub s_hsy: Vec<f64>,
    /// Mass-centre ⟨|Σ e^{iq·R}|²⟩ / N: what point-particle theories (such as
    /// the HSY reference) predict, free of orientation–form-factor coupling.
    pub s_com: Vec<f64>,
    /// Jackknife error of `s_com`, as for `s_eff_error`.
    pub s_com_error: Option<Vec<f64>>,
    pub frames: usize,
}

impl ScatteringCurve {
    /// One row per q; the error is NaN without enough copies for one.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("q/Å⁻¹,I,P,S_eff,S_eff_error,S_com,S_com_error,S_HSY\n");
        // Bead columns are empty for mass-centre-only curves.
        let at = |values: &[f64], k: usize| values.get(k).copied().unwrap_or(f64::NAN);
        let s_eff_error = self.s_eff_error.as_deref().unwrap_or_default();
        let s_com_error = self.s_com_error.as_deref().unwrap_or_default();
        for k in 0..self.q.len() {
            csv += &format!(
                "{:.5},{:.6e},{:.6e},{:.5},{:.5},{:.5},{:.5},{:.5}\n",
                self.q[k],
                at(&self.intensity, k),
                at(&self.form_factor, k),
                at(&self.s_eff, k),
                at(s_eff_error, k),
                at(&self.s_com, k),
                at(s_com_error, k),
                at(&self.s_hsy, k),
            );
        }
        csv
    }
}

/// Where the scattering sums run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Device {
    #[default]
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
    centres: Vec<f64>,
}

/// Far above any run; a larger index means a damaged or foreign frame.
const MAX_COPIES: usize = 1 << 16;

pub struct Scattering {
    n_bodies: usize,
    /// q-vectors depend on the box, so every frame must have this one.
    box_length: f64,
    /// `None` for mass-centre-only runs, which skip the costly bead sums.
    beads: Option<Beads>,
    /// One unit point per body at its mass centre, on the same q-grid.
    centres: pripps::BoxScatterer,
    q: Vec<f64>,
    copies: Vec<CopySums>,
    centre_positions: Vec<[f64; 3]>,
    /// Bead sums not yet back, each with its frame's mass-centre sum.
    pending: PendingFrames<pripps::BoxIntensity>,
}

/// One frame's sums; the bead ones are empty without beads.
struct Terms<'a> {
    total: &'a [f64],
    self_term: &'a [f64],
    centres: &'a [f64],
}

struct Beads {
    reference: CgBody,
    /// Shared with in-flight sums, which outlive the `observe` call.
    scatterer: Arc<pripps::BoxScatterer>,
}

const CENTRE_KIND: &str = "COM";
const CENTRE_FORM_FACTOR: &str = "COM: { formfactor: !Constant 1.0 }\n";
/// Mass-centre sums are cheap, so bins keep up to this many lattice vectors.
pub(super) const CENTRE_VECTORS_PER_BIN: usize = 500;

impl Scattering {
    /// `q_max` (Å⁻¹) bounds the sampled lattice vectors; it must lie inside
    /// the form factors' range (0.75 Å⁻¹ for pripps' residue form factors).
    /// Bead sums cost per vector, so `vectors_per_bin` caps them per |q| bin.
    /// Async as GPU setup is.
    pub async fn new(
        reference: &CgBody,
        n_bodies: usize,
        box_length: f64,
        q_max: f64,
        vectors_per_bin: usize,
        formfactors_yaml: &str,
        device: Device,
    ) -> Result<Self> {
        let mut scattering = Self::mass_centres(n_bodies, box_length, q_max)?;
        let grid = pripps::QGrid::lattice([box_length; 3], q_max, vectors_per_bin)?;
        let kinds = &reference.kinds;
        let scatterer = match device {
            Device::Cpu => pripps::BoxScatterer::new_cpu(formfactors_yaml, kinds, n_bodies, grid)?,
            Device::Gpu => {
                pripps::BoxScatterer::new_gpu(formfactors_yaml, kinds, n_bodies, grid).await?
            }
        };
        scattering.beads = Some(Beads {
            reference: reference.clone(),
            scatterer: Arc::new(scatterer),
        });
        Ok(scattering)
    }

    /// Mass-centre S(q) only: about a bead count cheaper per frame.
    pub fn mass_centres(n_bodies: usize, box_length: f64, q_max: f64) -> Result<Self> {
        let grid = pripps::QGrid::lattice([box_length; 3], q_max, CENTRE_VECTORS_PER_BIN)?;
        // Known up front, so a run can solve q-dependent references at start.
        let q = grid.q().to_vec();
        Ok(Self {
            n_bodies,
            box_length,
            beads: None,
            centres: pripps::BoxScatterer::new_cpu(
                CENTRE_FORM_FACTOR,
                &[CENTRE_KIND],
                n_bodies,
                grid,
            )?,
            q,
            copies: Vec::new(),
            centre_positions: Vec::with_capacity(n_bodies),
            pending: PendingFrames::new(),
        })
    }

    /// Never blocks: bead sums that aren't back yet (GPU) are folded in by
    /// a later `observe` or `settle`, so `curve` may lag by those frames.
    pub fn observe(&mut self, frame: &Frame) -> Result<()> {
        self.check_frame(frame)?;
        let centres = self.centre_sums(frame)?;
        let Some(beads) = &self.beads else {
            return self.fold(frame.copy, &centres, None);
        };
        let job = beads.evaluate(frame, [self.box_length; 3]);
        self.pending.push(frame.copy, centres, job);
        self.settle()?;
        Ok(())
    }

    /// Folds in finished bead sums; returns how many frames are still out.
    /// A failed frame doesn't discard the good ones finished with it.
    pub fn settle(&mut self) -> Result<usize> {
        let mut first_error = None;
        for finished in self.pending.drain_ready() {
            if let Err(e) =
                finished.and_then(|(copy, centres, beads)| self.fold(copy, &centres, Some(&beads)))
            {
                first_error.get_or_insert(e);
            }
        }
        first_error.map_or(Ok(self.pending.len()), Err)
    }

    /// `beads` is `None` for mass-centre-only runs.
    fn fold(
        &mut self,
        copy: usize,
        centres: &[f64],
        beads: Option<&pripps::BoxIntensity>,
    ) -> Result<()> {
        if let Some(beads) = beads {
            self.check_grid(beads, "bead")?;
        }
        let (total, self_term) = beads.map_or((&[][..], &[][..]), |b| (&b.total, &b.self_term));
        let terms = Terms {
            total,
            self_term,
            centres,
        };
        self.accumulate(copy, terms);
        Ok(())
    }

    /// Always on the CPU, so it is ready at once.
    fn centre_sums(&mut self, frame: &Frame) -> Result<Vec<f64>> {
        self.centre_positions.clear();
        self.centre_positions.extend(
            frame
                .poses
                .iter()
                .map(|p| [p.position.x, p.position.y, p.position.z]),
        );
        let sums = self
            .centres
            .evaluate(&self.centre_positions, [self.box_length; 3]);
        let centres = pending::ready_now(sums)?;
        self.check_grid(&centres, "mass-centre")?;
        Ok(centres.total)
    }

    /// Sums are added per q-point, so every grid must be the first one.
    fn check_grid(&self, result: &pripps::BoxIntensity, what: &str) -> Result<()> {
        let n = self.q.len();
        ensure!(
            result.q == self.q && result.total.len() == n && result.self_term.len() == n,
            "{what} scattering returned {} q-points, expected {n}",
            result.q.len()
        );
        Ok(())
    }

    /// q-vectors and bead counts are fixed at setup, so a run must match.
    pub(super) fn check_system(&self, n_bodies: usize, box_length: f64) -> Result<()> {
        ensure!(
            box_length == self.box_length,
            "box {box_length} Å differs from the {} Å the q-vectors were set up for",
            self.box_length
        );
        ensure!(
            n_bodies == self.n_bodies,
            "{n_bodies} bodies, but scattering was set up for {}",
            self.n_bodies
        );
        Ok(())
    }

    fn check_frame(&self, frame: &Frame) -> Result<()> {
        self.check_system(frame.poses.len(), frame.box_length)?;
        ensure!(
            frame.copy < MAX_COPIES,
            "implausible copy index {}",
            frame.copy
        );
        Ok(())
    }

    fn accumulate(&mut self, copy: usize, terms: Terms) {
        if self.copies.len() <= copy {
            self.copies.resize(copy + 1, CopySums::default());
        }
        let sums = &mut self.copies[copy];
        if sums.frames == 0 {
            sums.total = vec![0.0; terms.total.len()];
            sums.self_term = vec![0.0; terms.self_term.len()];
            sums.centres = vec![0.0; terms.centres.len()];
        }
        sums.frames += 1;
        add_into(&mut sums.total, terms.total);
        add_into(&mut sums.self_term, terms.self_term);
        add_into(&mut sums.centres, terms.centres);
    }

    /// Every |q| bin, including any `curve` leaves out.
    pub fn q(&self) -> &[f64] {
        &self.q
    }

    /// Forgets all frames; the q-grid and form factors are kept.
    pub fn reset(&mut self) {
        self.copies.clear();
        self.pending.clear();
    }

    pub fn curve(&self) -> ScatteringCurve {
        self.curve_with(&[])
    }

    /// `reference` holds one value per bin of `q()` (or is empty); it is
    /// filtered with the bins, so it stays aligned with `curve.q`.
    pub fn curve_with(&self, reference: &[f64]) -> ScatteringCurve {
        let observed: Vec<&CopySums> = self.copies.iter().filter(|c| c.frames > 0).collect();
        let frames: usize = observed.iter().map(|c| c.frames).sum();
        if frames == 0 {
            return ScatteringCurve::default();
        }
        let several = observed.len() > 1;
        let beads = self.beads.is_some();
        let mut curve = ScatteringCurve {
            frames,
            s_eff_error: (several && beads).then(Vec::new),
            s_com_error: several.then(Vec::new),
            ..Default::default()
        };
        let per_molecule = (frames * self.n_bodies) as f64;
        for (k, &q) in self.q.iter().enumerate() {
            let sums = Shell {
                copies: &observed,
                k,
            };
            // S_eff is undefined where the form factor vanishes.
            if beads && sums.of(|c| &c.self_term) == 0.0 {
                continue;
            }
            curve.q.push(q);
            curve.s_hsy.extend(reference.get(k));
            if beads {
                sums.push_beads(&mut curve, per_molecule);
            }
            sums.push_centres(&mut curve, per_molecule, self.n_bodies);
        }
        curve
    }
}

impl Beads {
    /// Owns its inputs, so it can outlive the frame and the caller.
    fn evaluate(
        &self,
        frame: &Frame,
        box_sides: [f64; 3],
    ) -> impl Future<Output = Result<pripps::BoxIntensity>> + 'static {
        // R + q·r for every bead, body after body as the scatterer expects.
        let positions: Vec<[f64; 3]> = frame
            .poses
            .iter()
            .flat_map(|pose| {
                self.reference.positions.iter().map(move |&r| {
                    let p = pose.position + pose.orientation * Vector3::from(r);
                    [p.x, p.y, p.z]
                })
            })
            .collect();
        let scatterer = self.scatterer.clone();
        async move { Ok(scatterer.evaluate(&positions, box_sides).await?) }
    }
}

/// The copies' sums at one q-point.
struct Shell<'a> {
    copies: &'a [&'a CopySums],
    k: usize,
}

impl Shell<'_> {
    fn of(&self, field: fn(&CopySums) -> &Vec<f64>) -> f64 {
        self.copies.iter().map(|c| field(c)[self.k]).sum()
    }

    fn push_beads(&self, curve: &mut ScatteringCurve, per_molecule: f64) {
        let (total, self_term) = (self.of(|c| &c.total), self.of(|c| &c.self_term));
        curve.intensity.push(total / per_molecule);
        curve.form_factor.push(self_term / per_molecule);
        curve.s_eff.push(total / self_term);
        let k = self.k;
        if let Some(errors) = &mut curve.s_eff_error {
            errors.push(jackknife_error(
                self.copies.iter().map(|c| (c.total[k], c.self_term[k])),
            ));
        }
    }

    /// A copy's mass-centre self term is N per frame.
    fn push_centres(&self, curve: &mut ScatteringCurve, per_molecule: f64, n_bodies: usize) {
        curve.s_com.push(self.of(|c| &c.centres) / per_molecule);
        let k = self.k;
        if let Some(errors) = &mut curve.s_com_error {
            errors.push(jackknife_error(
                self.copies
                    .iter()
                    .map(|c| (c.centres[k], (c.frames * n_bodies) as f64)),
            ));
        }
    }
}

pub(super) fn add_into(sum: &mut [f64], values: &[f64]) {
    sum.iter_mut().zip(values).for_each(|(s, v)| *s += v);
}

/// Leave-one-copy-out jackknife on a pooled ratio Σnumerator/Σdenominator
/// over per-copy sums: it weights copies by their frames, as the pooled
/// ratio does.
pub(super) fn jackknife_error(copies: impl Iterator<Item = (f64, f64)> + Clone) -> f64 {
    let (top, bottom) = copies
        .clone()
        .fold((0.0, 0.0), |(t, b), (x, y)| (t + x, b + y));
    let leave_out: Vec<f64> = copies.map(|(x, y)| (top - x) / (bottom - y)).collect();
    let n = leave_out.len() as f64;
    let mean = leave_out.iter().sum::<f64>() / n;
    let spread = leave_out.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
    ((n - 1.0) / n * spread).sqrt()
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
        pollster::block_on(Scattering::new(
            reference,
            n,
            l,
            q_max,
            usize::MAX,
            UNIT,
            Device::Cpu,
        ))
        .unwrap()
    }

    fn observe(scattering: &mut Scattering, frame: &Frame) {
        scattering.observe(frame).unwrap();
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

    /// Two unit scatterers L/4 apart on the lattice grid (L = 40, q ≤ 0.3):
    /// bin 1 is n = (1,0,0)- and (1,1,0)-type, 26/9 as in pripps' worked
    /// test; bin 2 only the four (±1,±1,±1), h = ±1, so 2. Self term 2; per
    /// molecule halved, and S_eff = total/self.
    #[test]
    fn two_bodies_give_the_worked_curve() {
        let mut s = scattering(&bead("X"), 2, 40.0, 0.3);
        observe(
            &mut s,
            &frame(0, &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], 40.0),
        );
        let curve = s.curve();
        let expected = [26.0 / 9.0, 2.0];
        assert_eq!(curve.frames, 1);
        assert_eq!(curve.q.len(), 2);
        for k in 0..2 {
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

    /// CPU bead sums finish within `observe`, so the curve never lags.
    #[test]
    fn cpu_frames_are_folded_in_at_once() {
        let mut s = scattering(&bead("X"), 2, 40.0, 0.3);
        s.observe(&frame(0, &[[0.0; 3], [10.0, 0.0, 0.0]], 40.0))
            .unwrap();
        assert_eq!(s.curve().frames, 1);
        assert_eq!(s.settle().unwrap(), 0);
    }

    /// Mass centres L/4 apart give the worked two-point curve halved,
    /// 13/9 and 1 per bin, however the beads are arranged around them.
    #[test]
    fn mass_centre_structure_factor_ignores_bead_layout() {
        let dumbbell = body(&[[-3.0, -4.0, 1.0], [3.0, 4.0, -1.0]], "X");
        let mut s = scattering(&dumbbell, 2, 40.0, 0.3);
        observe(
            &mut s,
            &frame(0, &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], 40.0),
        );
        let curve = s.curve();
        for (k, expected) in [13.0 / 9.0, 1.0].into_iter().enumerate() {
            approx::assert_relative_eq!(curve.s_com[k], expected, max_relative = 1e-12);
        }
    }

    /// Without beads, only S_com is computed, and every q is kept.
    #[test]
    fn mass_centres_alone_give_only_s_com() {
        let mut s = Scattering::mass_centres(2, 40.0, 0.3).unwrap();
        observe(
            &mut s,
            &frame(0, &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], 40.0),
        );
        let curve = s.curve();
        assert_eq!(curve.q.len(), 2);
        for (k, expected) in [13.0 / 9.0, 1.0].into_iter().enumerate() {
            approx::assert_relative_eq!(curve.s_com[k], expected, max_relative = 1e-12);
        }
        assert!(curve.intensity.is_empty() && curve.s_eff.is_empty());
        assert_eq!(curve.to_csv().lines().count(), 3);
    }

    /// Bin 1 (nine vectors). Copy 0 has centres L/4 apart: box sum 26/9.
    /// Copy 1 has them L/2 apart, |1 + e^{iπh}|² = 0 for odd h, 4 for h = 0:
    /// (0,1,0), (0,0,1), (0,1,±1) give 4 each, so 16/9. Pooled S_com =
    /// (26/9 + 16/9)/4 = 7/6; leave-one-out values 8/9 and 13/9 give the
    /// error √(½·2·(5/18)²) = 5/18.
    #[test]
    fn s_com_error_is_the_jackknife_over_copies() {
        let mut s = Scattering::mass_centres(2, 40.0, 0.3).unwrap();
        observe(&mut s, &frame(0, &[[0.0; 3], [10.0, 0.0, 0.0]], 40.0));
        observe(&mut s, &frame(1, &[[0.0; 3], [20.0, 0.0, 0.0]], 40.0));
        let curve = s.curve();
        approx::assert_relative_eq!(curve.s_com[0], 7.0 / 6.0, max_relative = 1e-12);
        approx::assert_relative_eq!(
            curve.s_com_error.unwrap()[0],
            5.0 / 18.0,
            max_relative = 1e-12
        );
    }

    /// Beads must be grouped by body: each body is two unit beads L/4 apart,
    /// so its own |F|² is the worked 26/9 and 2 per bin. Pairing beads
    /// across bodies would give a different self term.
    #[test]
    fn self_term_groups_beads_by_body() {
        let pair = body(&[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], "X");
        let mut s = scattering(&pair, 2, 40.0, 0.3);
        observe(
            &mut s,
            &frame(0, &[[1.0, 3.0, 5.0], [21.0, 17.0, 11.0]], 40.0),
        );
        let expected = [26.0 / 9.0, 2.0];
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
            16,
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
        let result = s.observe(&frame(0, &[[0.0; 3]], 40.0));
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

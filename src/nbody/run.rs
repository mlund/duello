//! A whole many-body run: equilibration, then production of I(q), energy
//! and move statistics, and a trajectory, from a homo-dimer scan archive.
//!
//! Synchronous: callers advance it in small steps (e.g. one per UI frame),
//! which keeps a browser responsive without async.

use super::trajectory::{TrajectoryHeader, TrajectoryWriter};
use super::{
    box_length, Criteria, Device, EnergyStats, EnergySummary, Equilibration, Frame, HsyFluid,
    MoveStats, MsaOverrides, PairCorrelation, PairEnergy, RdfCurve, Sampler, SamplerSpec,
    Scattering, ScatteringCurve, Status, StepTuner,
};
use crate::archive::{CgBody, ScanArchive};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::future::Future;
use std::sync::Arc;

/// What to run. `RunSpec::new` fills in defaults for everything but the
/// system. Saved in `.duello` sessions: a field change needs an archive
/// `FORMAT_VERSION` bump.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSpec {
    pub n_bodies: usize,
    /// Protein concentration in mol/L.
    pub molarity: f64,
    /// Independent copies; their spread gives the error bars.
    pub copies: usize,
    pub seed: u64,
    /// Largest q (Å⁻¹) in the scattering curve.
    pub q_max: f64,
    /// Lattice q-vectors kept per |q| bin for the bead sums, whose cost is
    /// per vector; mass centres always use many more.
    pub vectors_per_bin: usize,
    /// Sweeps between production frames.
    pub sweeps_per_frame: u64,
    /// Sweeps between equilibration energy checks.
    pub sweeps_per_check: u64,
    pub criteria: Criteria,
    /// Maximum translation per axis (Å) and rotation angle (rad): where
    /// tuning starts, or the steps used throughout without it.
    pub translation: f64,
    pub rotation: f64,
    /// Tune both steps during equilibration for the largest displacement
    /// per attempted move, then freeze them for production.
    pub tune_steps: bool,
    /// Replaces the reference sphere's derived diameter or screening.
    pub msa: MsaOverrides,
    /// Off by default: a trajectory grows by MBs per minute and the curve
    /// doesn't need it.
    pub record_trajectory: bool,
    /// Recording stops beyond this; the run and curve carry on.
    pub max_trajectory_bytes: usize,
    /// Mass-centre S(q) only: frames become cheap enough to take at every
    /// check, but I, P and S_eff are not computed.
    pub centres_only: bool,
    /// Where bead sums run; on a GPU they are folded in as they come back.
    /// Not saved: a runtime choice, not an input.
    #[serde(skip)]
    pub device: Device,
}

impl RunSpec {
    pub fn new(n_bodies: usize, molarity: f64) -> Self {
        Self {
            n_bodies,
            molarity,
            copies: 4,
            seed: 1,
            q_max: 0.5,
            vectors_per_bin: 16,
            sweeps_per_frame: 50,
            sweeps_per_check: 10,
            criteria: Criteria {
                min_sweeps: 200,
                max_sweeps: 20_000,
                tolerance: 2.0,
                max_drift: 0.5,
            },
            translation: 3.0,
            rotation: 0.2,
            tune_steps: true,
            msa: MsaOverrides::default(),
            record_trajectory: false,
            max_trajectory_bytes: 256 << 20,
            centres_only: false,
            device: Device::Cpu,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stage {
    Equilibrating,
    /// `equilibrated` is false if the sweep cap was reached first.
    Production {
        equilibrated: bool,
    },
    Stopped,
}

/// Where a run is, for a progress display.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progress {
    pub stage: Stage,
    /// Sweeps per copy so far.
    pub sweep: u64,
    /// Copy-averaged energy per body (kT) at the last check or frame; 0
    /// before the first check.
    pub energy_per_body: f64,
    /// Production frames taken, summed over copies; `curve().frames` lags
    /// it while GPU sums are out.
    pub frames: usize,
    /// False if recording is off or the trajectory reached its size cap.
    pub recording: bool,
}

/// Production statistics, pooled over copies.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunStats {
    pub moves: MoveStats,
    pub energy: EnergySummary,
    /// Translation (Å) and rotation (rad) steps now in use.
    pub steps: (f64, f64),
}

pub struct NBodyRun {
    spec: RunSpec,
    archive_settings: crate::archive::ScanSettings,
    reference: CgBody,
    samplers: Vec<Sampler>,
    stage: Stage,
    sweep: u64,
    equilibration: Equilibration,
    /// Production energy statistics; also gives per-frame energies during
    /// equilibration, which are not added to it.
    energy: EnergyStats,
    moves: MoveStats,
    scattering: Scattering,
    pair_correlation: PairCorrelation,
    /// Charged-sphere S(q) to compare S_eff against.
    msa: MsaReference,
    /// Created when production starts, as the header records how
    /// equilibration ended.
    trajectory: Option<TrajectoryWriter<Vec<u8>>>,
    /// The trajectory reached `max_trajectory_bytes`.
    capped: bool,
    /// How equilibration ended; kept apart from `stage`, which forgets it on
    /// `stop`.
    equilibrated: bool,
    equilibration_sweeps: u64,
    frames: usize,
    energy_per_body: f64,
    /// (sweep, U/N) at each equilibration check, for plotting convergence.
    energy_trace: Vec<[f64; 2]>,
    /// Translation and rotation tuners; `None` once frozen or when off.
    tuners: Option<[StepTuner; 2]>,
    steps: (f64, f64),
    /// Energy converged at some check, while the tuners may still search.
    converged: bool,
}

impl NBodyRun {
    /// Fails for hetero-dimer archives: every body is the archive's first.
    /// Blocks on scattering setup, so a browser wanting `Device::Gpu` must
    /// await `scattering` and call `start_with` instead.
    pub fn start(archive: &ScanArchive, spec: RunSpec, formfactors_yaml: &str) -> Result<Self> {
        let scattering = pollster::block_on(Self::scattering(archive, &spec, formfactors_yaml))?;
        Self::start_with(archive, spec, scattering)
    }

    /// The scattering `start_with` needs. Owns its inputs, so a browser can
    /// await it on the event loop, as GPU setup requires.
    pub fn scattering(
        archive: &ScanArchive,
        spec: &RunSpec,
        formfactors_yaml: &str,
    ) -> impl Future<Output = Result<Scattering>> + 'static {
        let checked = check_inputs(archive, spec);
        let reference = archive.bodies[0].clone();
        let yaml = with_missing_kinds(formfactors_yaml, &reference.kinds);
        let l = box_length(spec.n_bodies, spec.molarity);
        let spec = spec.clone();
        async move {
            checked?;
            if spec.centres_only {
                return Scattering::mass_centres(spec.n_bodies, l, spec.q_max);
            }
            let (n, q_max, per_bin) = (spec.n_bodies, spec.q_max, spec.vectors_per_bin);
            Scattering::new(&reference, n, l, q_max, per_bin, &yaml, spec.device).await
        }
    }

    /// `scattering` must come from `NBodyRun::scattering` with this spec.
    pub fn start_with(
        archive: &ScanArchive,
        spec: RunSpec,
        scattering: Scattering,
    ) -> Result<Self> {
        check_inputs(archive, &spec)?;
        let energy = Arc::new(PairEnergy::new(archive.table.clone())?);
        let l = box_length(spec.n_bodies, spec.molarity);
        // Larger steps add nothing, so a start beyond them is clamped.
        let sampler_spec = SamplerSpec {
            n_bodies: spec.n_bodies,
            box_length: l,
            translation: spec.translation.min(l / 2.0),
            rotation: spec.rotation.min(std::f64::consts::PI),
        };
        let spec_tuners = spec.tune_steps.then(|| {
            [
                StepTuner::new(sampler_spec.translation, l / 2.0),
                StepTuner::new(sampler_spec.rotation, std::f64::consts::PI),
            ]
        });
        let samplers = (0..spec.copies)
            .map(|c| {
                Sampler::new(
                    energy.clone(),
                    sampler_spec.clone(),
                    spec.seed.wrapping_add(c as u64),
                )
            })
            .collect::<Result<_>>()?;
        scattering.check_system(spec.n_bodies, l)?;
        let reference = archive.bodies[0].clone();
        let msa = MsaReference::new(
            HsyFluid::for_scan(archive, spec.molarity),
            spec.msa,
            scattering.q(),
        );
        Ok(Self {
            equilibration: Equilibration::new(spec.criteria),
            spec,
            archive_settings: archive.settings.clone(),
            reference,
            samplers,
            stage: Stage::Equilibrating,
            sweep: 0,
            energy: EnergyStats::new(energy),
            moves: MoveStats::default(),
            scattering,
            pair_correlation: PairCorrelation::new(sampler_spec.n_bodies, sampler_spec.box_length),
            msa,
            trajectory: None,
            capped: false,
            equilibrated: false,
            equilibration_sweeps: 0,
            frames: 0,
            energy_per_body: 0.0,
            energy_trace: Vec::new(),
            tuners: spec_tuners,
            steps: (sampler_spec.translation, sampler_spec.rotation),
            converged: false,
        })
    }

    /// Advances every copy by exactly `sweeps`, so a UI can step in small
    /// pieces; energy checks and frames still fall on their intervals. An
    /// error stops the run, as copies may then be out of step.
    pub fn step(&mut self, sweeps: u64) -> Result<Progress> {
        let stepped = self.settle().and_then(|_| self.sweep_for(sweeps));
        self.stop_on_error(stepped).map(|()| self.progress())
    }

    /// Folds in frames whose GPU sums came back since the last call; true
    /// while some are still out. `step` does this too, but a paused or
    /// stopped UI must call it for the curve to catch up.
    pub fn settle(&mut self) -> Result<bool> {
        let pending = self.scattering.settle();
        self.stop_on_error(pending).map(|left| left > 0)
    }

    fn stop_on_error<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() {
            self.stage = Stage::Stopped;
        }
        result
    }

    fn sweep_for(&mut self, sweeps: u64) -> Result<()> {
        let mut done = 0;
        while done < sweeps {
            let left = sweeps - done;
            done += match self.stage {
                Stage::Equilibrating => self.equilibration_step(left)?,
                Stage::Production { .. } => self.production_step(left)?,
                Stage::Stopped => break,
            };
        }
        Ok(())
    }

    fn progress(&self) -> Progress {
        Progress {
            stage: self.stage,
            sweep: self.sweep,
            energy_per_body: self.energy_per_body,
            frames: self.frames,
            recording: self.spec.record_trajectory && !self.capped,
        }
    }

    fn advance(&mut self, sweeps: u64) {
        self.sweep += sweeps;
        self.samplers.iter_mut().for_each(|s| s.advance(sweeps));
    }

    /// (sweep, U/N) at each equilibration check; empty before the first.
    pub fn energy_trace(&self) -> &[[f64; 2]] {
        &self.energy_trace
    }

    /// The latest configuration of every copy, at any stage; for viewers.
    /// After a failed step stops the run, copies may be sweeps apart.
    pub fn frames(&self) -> Vec<Frame> {
        (0..self.samplers.len()).map(|c| self.frame_of(c)).collect()
    }

    /// One copy's latest configuration; cheaper than `frames` when only one
    /// is shown.
    pub fn frame(&self, copy: usize) -> Result<Frame> {
        ensure!(
            copy < self.samplers.len(),
            "copy {copy} out of range: the run has {} copies",
            self.samplers.len()
        );
        Ok(self.frame_of(copy))
    }

    /// The rigid body every pose places; beads follow via `trajectory::expand`.
    pub fn reference(&self) -> &CgBody {
        &self.reference
    }

    /// Samplers don't know their copy index, so it is set here.
    fn frame_of(&self, copy: usize) -> Frame {
        Frame {
            copy,
            ..self.samplers[copy].frame()
        }
    }

    /// Up to `most` sweeps, stopping at the next check, which runs only once
    /// its interval is complete; a UI can then step in small pieces.
    fn equilibration_step(&mut self, most: u64) -> Result<u64> {
        let check = self.spec.sweeps_per_check;
        let sweeps = (check - self.sweep % check).min(most);
        self.advance(sweeps);
        if !self.sweep.is_multiple_of(check) {
            return Ok(sweeps);
        }
        let frames = self.frames();
        let energies = frames
            .iter()
            .map(|f| self.energy.frame_energy(f))
            .collect::<Result<Vec<_>>>()?;
        self.energy_per_body = mean(&energies) / self.spec.n_bodies as f64;
        self.energy_trace
            .push([self.sweep as f64, self.energy_per_body]);
        let tuned = self.tune_steps()?;
        let status = self.equilibration.push(self.sweep, &energies);
        // The verdict sticks while the tuners finish, so a later noisy
        // window can't relabel a converged run.
        self.converged |= status == Status::Converged;
        let capped = status == Status::CapReached || self.sweep >= self.spec.criteria.max_sweeps;
        if (self.converged && tuned) || capped {
            self.begin_production(self.converged)?;
        }
        Ok(sweeps)
    }

    /// One tuning window over all copies; true once both tuners settled
    /// (or tuning is off).
    fn tune_steps(&mut self) -> Result<bool> {
        let Some([translate, rotate]) = &mut self.tuners else {
            return Ok(true);
        };
        let mut window = MoveStats::default();
        for sampler in &mut self.samplers {
            window.merge(&sampler.take_move_stats());
        }
        let steps = (
            translate.observe(&window.translate),
            rotate.observe(&window.rotate),
        );
        let settled = translate.settled() && rotate.settled();
        self.apply_steps(steps)?;
        Ok(settled)
    }

    fn apply_steps(&mut self, steps: (f64, f64)) -> Result<()> {
        for sampler in &mut self.samplers {
            sampler.set_steps(steps.0, steps.1)?;
        }
        self.steps = steps;
        Ok(())
    }

    /// Statistics restart here, so equilibration never enters them.
    fn begin_production(&mut self, equilibrated: bool) -> Result<()> {
        if let Some([translate, rotate]) = self.tuners.take() {
            self.apply_steps((translate.best(), rotate.best()))?;
        }
        for sampler in &mut self.samplers {
            sampler.take_move_stats();
        }
        if self.spec.record_trajectory {
            let header = self.header(equilibrated);
            self.trajectory = Some(TrajectoryWriter::new(&header, Vec::new())?);
        }
        self.stage = Stage::Production { equilibrated };
        self.equilibrated = equilibrated;
        self.equilibration_sweeps = self.sweep;
        Ok(())
    }

    fn header(&self, equilibrated: bool) -> TrajectoryHeader {
        TrajectoryHeader {
            n_bodies: self.spec.n_bodies,
            protein_molarity: self.spec.molarity,
            sweeps_per_frame: self.spec.sweeps_per_frame,
            equilibration_sweeps: self.sweep,
            equilibrated,
            reference: self.reference.clone(),
            settings: self.archive_settings.clone(),
            translation: Some(self.steps.0),
            rotation: Some(self.steps.1),
        }
    }

    /// Up to `most` sweeps, stopping at the next due frame. Each copy takes a
    /// frame once per interval, but copies are staggered so
    /// their scattering costs spread over steps instead of landing together.
    fn production_step(&mut self, most: u64) -> Result<u64> {
        let produced = self.sweep - self.equilibration_sweeps;
        let interval = self.spec.sweeps_per_frame;
        let to_frame: Vec<u64> = (0..self.spec.copies)
            .map(|copy| interval - (produced + self.phase(copy)) % interval)
            .collect();
        let next = to_frame.iter().copied().min().unwrap_or(interval);
        let sweeps = next.min(most);
        self.advance(sweeps);
        // Only copies due a frame pay for one.
        let due: Vec<Frame> = (to_frame.iter().enumerate())
            .filter(|&(_, &left)| left == sweeps)
            .map(|(copy, _)| self.frame_of(copy))
            .collect();
        if !due.is_empty() {
            self.observe(&due)?;
        }
        Ok(sweeps)
    }

    /// Sweeps by which copy `copy`'s frames come early.
    fn phase(&self, copy: usize) -> u64 {
        copy as u64 * self.spec.sweeps_per_frame / self.spec.copies as u64
    }

    fn observe(&mut self, frames: &[Frame]) -> Result<()> {
        let mut energies = Vec::with_capacity(frames.len());
        for frame in frames {
            self.scattering.observe(frame)?;
            self.pair_correlation.observe(frame);
            energies.push(self.energy.observe(frame)?);
            self.record(frame)?;
            // Only up to a copy's latest frame, so moves and frames agree.
            let moves = self.samplers[frame.copy].take_move_stats();
            self.moves.merge(&moves);
        }
        self.frames += frames.len();
        self.energy_per_body = mean(&energies) / self.spec.n_bodies as f64;
        Ok(())
    }

    fn record(&mut self, frame: &Frame) -> Result<()> {
        let Some(writer) = self.trajectory.as_mut() else {
            return Ok(());
        };
        // Sticky, so a later smaller frame can't leave a gap.
        self.capped |= writer.sink().len() + writer.record_bytes() > self.spec.max_trajectory_bytes;
        if !self.capped {
            writer.append(frame)?;
        }
        Ok(())
    }

    pub fn curve(&self) -> ScatteringCurve {
        self.scattering.curve_with(&self.msa.curve)
    }

    /// Mass-centre g(r) over production frames; empty before the first.
    pub fn pair_correlation(&self) -> RdfCurve {
        self.pair_correlation.curve()
    }

    pub fn stats(&self) -> RunStats {
        RunStats {
            moves: self.moves.clone(),
            energy: self.energy.summary(),
            steps: self.steps,
        }
    }

    /// Theory only, so it may change at any stage without touching the
    /// simulation; a refused change keeps the current reference.
    pub fn set_msa(&mut self, overrides: MsaOverrides) -> Result<()> {
        check_msa(&overrides)?;
        self.msa = MsaReference::new(self.msa.derived.clone(), overrides, self.scattering.q());
        Ok(())
    }

    pub fn msa(&self) -> &MsaReference {
        &self.msa
    }

    /// Discards production samples so far (curve, energy and move statistics,
    /// trajectory) and keeps sampling from the current state. The trajectory
    /// header then counts the discarded sweeps as equilibration.
    pub fn reset_statistics(&mut self) -> Result<Progress> {
        ensure!(
            matches!(self.stage, Stage::Production { .. }),
            "statistics exist only during production"
        );
        self.scattering.reset();
        self.pair_correlation.reset();
        self.energy.reset();
        self.moves = MoveStats::default();
        self.frames = 0;
        self.capped = false;
        self.begin_production(self.equilibrated)?;
        Ok(self.progress())
    }

    /// Curve and statistics as CSV.
    pub fn report_csv(&self) -> String {
        let mut csv = self.stats_comments();
        csv += &self.curve().to_csv();
        csv
    }

    fn stats_comments(&self) -> String {
        let stats = self.stats();
        let n = self.spec.n_bodies as f64;
        let show = |x: Option<f64>| x.map_or("NaN".to_string(), |v| format!("{v:.5}"));
        let produced = stats.energy.frames > 0;
        let per_body = |x: f64| show(produced.then_some(x / n));
        let moves = &stats.moves;
        [
            format!("bodies: {}", self.spec.n_bodies),
            format!("protein molarity: {}", self.spec.molarity),
            format!("centres only: {}", self.spec.centres_only),
            format!("equilibrated: {}", self.equilibrated),
            format!("equilibration sweeps: {}", self.equilibration_sweeps),
            format!("frames: {}", stats.energy.frames),
            format!(
                "translate acceptance: {}",
                show(moves.translate.acceptance())
            ),
            format!("rotate acceptance: {}", show(moves.rotate.acceptance())),
            format!("translation step/Å: {:.4}", stats.steps.0),
            format!("rotation step/rad: {:.4}", stats.steps.1),
            format!(
                "translate msd/Å²: {}",
                show(moves.translate.mean_square_displacement())
            ),
            format!(
                "rotate msd/rad²: {}",
                show(moves.rotate.mean_square_displacement())
            ),
            format!("U/N mean: {} kT", per_body(stats.energy.mean)),
            format!("Var(U)/N: {} kT²", per_body(stats.energy.variance)),
            self.msa.sphere().describe(),
        ]
        .iter()
        .map(|line| format!("# {line}\n"))
        .collect()
    }

    /// The trajectory so far, borrowed as it can be large; header only if
    /// production hasn't started. An error if the spec didn't record one.
    pub fn trajectory_bytes(&self) -> Result<Cow<'_, [u8]>> {
        ensure!(
            self.spec.record_trajectory,
            "trajectory not recorded: recording was off for this run"
        );
        match &self.trajectory {
            Some(writer) => Ok(Cow::Borrowed(writer.sink())),
            None => Ok(Cow::Owned(
                TrajectoryWriter::new(&self.header(false), Vec::new())?.into_inner(),
            )),
        }
    }

    /// Ends the run; results so far stay available.
    pub fn stop(&mut self) {
        self.stage = Stage::Stopped;
    }
}

/// Before any setup, so a bad archive or spec never builds GPU buffers.
fn check_inputs(archive: &ScanArchive, spec: &RunSpec) -> Result<()> {
    ensure!(
        archive.settings.homo_dimer,
        "many-body runs need a scan of identical molecules"
    );
    validate(spec)
}

/// Checks that would otherwise surface late: a zero interval hangs `step`,
/// and an oversized N fails only when production starts.
fn validate(spec: &RunSpec) -> Result<()> {
    ensure!(spec.copies > 0, "a run needs at least one copy");
    ensure!(
        spec.n_bodies > 0 && spec.n_bodies <= super::trajectory::MAX_BODIES,
        "number of bodies {} is out of range",
        spec.n_bodies
    );
    ensure!(
        spec.sweeps_per_frame > 0 && spec.sweeps_per_check > 0,
        "sweep intervals must be positive"
    );
    ensure!(
        spec.vectors_per_bin > 0,
        "vectors per q-bin must be positive"
    );
    check_msa(&spec.msa)
}

fn check_msa(overrides: &MsaOverrides) -> Result<()> {
    let MsaOverrides {
        diameter,
        debye_length,
    } = *overrides;
    // `> 0` is false for NaN too.
    ensure!(
        [diameter, debye_length].iter().flatten().all(|&x| x > 0.0),
        "MSA diameter and Debye length must be positive"
    );
    Ok(())
}

/// A run's charged sphere and its S(q) on every bin of the fixed q-grid,
/// solved once per change as the curve is redrawn every UI frame.
pub struct MsaReference {
    /// From the scan and concentration, before overrides.
    derived: HsyFluid,
    overrides: MsaOverrides,
    fluid: HsyFluid,
    /// Empty without a solution.
    curve: Vec<f64>,
}

impl MsaReference {
    fn new(derived: HsyFluid, overrides: MsaOverrides, q: &[f64]) -> Self {
        let fluid = derived.clone().overridden(&overrides);
        let curve = fluid.structure_factor(q).unwrap_or_default();
        Self {
            derived,
            overrides,
            fluid,
            curve,
        }
    }

    /// The defaults the overrides replace.
    pub fn derived(&self) -> &HsyFluid {
        &self.derived
    }

    pub fn overrides(&self) -> MsaOverrides {
        self.overrides
    }

    /// The sphere actually solved for.
    pub fn sphere(&self) -> &HsyFluid {
        &self.fluid
    }

    /// False when the rescaling found no physical solution.
    pub fn solved(&self) -> bool {
        !self.curve.is_empty()
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Residue form-factor files lack the terminal charge beads (N, O) and
/// disulfide cysteines (CYX). N and O mark small charged groups next to
/// their residue's bead, so they scatter nothing here; CYX scatters as CYS.
/// Added only when absent, so a file that defines them wins.
pub(super) fn with_missing_kinds(yaml: &str, kinds: &[String]) -> String {
    let defined = |kind: &str| {
        yaml.lines()
            .any(|line| line.starts_with(&format!("{kind}:")))
    };
    let supplements = [
        ("N", "{ formfactor: !Constant 0.0 }"),
        ("O", "{ formfactor: !Constant 0.0 }"),
        ("CYX", "{ formfactor: !Alias CYS }"),
    ];
    let mut yaml = yaml.to_string();
    for (kind, spec) in supplements {
        if kinds.iter().any(|k| k == kind) && !defined(kind) {
            yaml += &format!("\n{kind}: {spec}\n");
        }
    }
    yaml
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::{CgBody, ScanSettings};
    use crate::nbody::test_tables::table;
    use crate::nbody::trajectory::TrajectoryReader;

    const UNIT: &str = "X: { formfactor: !Constant 1.0 }\nCYS: { formfactor: !Constant 1.0 }\n";

    fn archive(homo_dimer: bool, kinds: &[&str]) -> ScanArchive {
        archive_with(homo_dimer, kinds, 0.0)
    }

    /// `well` (kJ/mol) is the energy at every in-table separation.
    fn archive_with(homo_dimer: bool, kinds: &[&str], well: f64) -> ScanArchive {
        let n = kinds.len();
        let body = CgBody {
            positions: (0..n).map(|i| [i as f64, 0.0, 0.0]).collect(),
            masses: vec![1.0; n],
            charges: vec![0.0; n],
            radii: vec![1.0; n],
            kinds: kinds.iter().map(|k| k.to_string()).collect(),
        };
        ScanArchive {
            settings: ScanSettings {
                ph: 7.0,
                model: "calvados3".into(),
                single_bead: true,
                rmin: 5.0,
                rmax: 8.0,
                dr: 1.0,
                ionic_strength: 0.05,
                temperature: 298.15,
                max_ndiv: 1,
                gradient_threshold: 0.5,
                cutoff: 30.0,
                grid_size: 200,
                homo_dimer,
                subdivision: Default::default(),
            },
            table: table(move |_, _, _| well),
            bodies: [body.clone(), body],
            points: vec![],
            virial: crate::VirialCoeff::from_raw_parts(0.0, 5.0),
            topology_yaml: String::new(),
            session: Default::default(),
        }
    }

    /// Ten bodies in a 20 Å box (ρ = 1.25e-3 Å⁻³), small test criteria,
    /// recording a trajectory.
    fn spec() -> RunSpec {
        let molarity = 10.0 / 8000.0 / (physical_constants::AVOGADRO_CONSTANT * 1e-27);
        let mut spec = RunSpec::new(10, molarity);
        spec.copies = 2;
        spec.q_max = 0.6;
        spec.sweeps_per_frame = 5;
        spec.criteria.min_sweeps = 20;
        spec.criteria.max_sweeps = 200;
        spec.record_trajectory = true;
        spec
    }

    #[test]
    fn without_recording_frames_reach_only_the_curve() {
        let spec = RunSpec {
            record_trajectory: false,
            ..spec()
        };
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        let progress = run.step(10).unwrap();
        assert!(progress.frames >= 4, "{progress:?}");
        assert_eq!(run.curve().frames, progress.frames);
        assert!(!progress.recording);
        let error = run.trajectory_bytes().unwrap_err();
        assert!(error.to_string().contains("not recorded"), "{error}");
    }

    fn run_to(run: &mut NBodyRun, stage: fn(&Stage) -> bool) -> Progress {
        for _ in 0..1000 {
            let progress = run.step(10).unwrap();
            if stage(&progress.stage) {
                return progress;
            }
        }
        panic!("stage not reached");
    }

    fn frames_in(bytes: &[u8]) -> (TrajectoryHeader, usize) {
        let reader = TrajectoryReader::new(bytes).unwrap();
        let header = reader.header().clone();
        (header, reader.count())
    }

    #[test]
    fn the_reference_sphere_takes_charge_from_the_scan() {
        let mut scan = archive(true, &["X", "X"]);
        scan.bodies[0].charges = vec![1.0, 0.5];
        let fluid = HsyFluid::for_scan(&scan, 1e-3);
        assert_eq!(fluid.charge, 1.5);
        // 1 mM protein, per Å³.
        approx::assert_relative_eq!(fluid.number_density, 6.0221e-7, max_relative = 1e-4);
        // The scan's 50 mM only: κ² = 4π × 7.13 Å × 2 × 3.011e-5 Å⁻³.
        approx::assert_relative_eq!(fluid.debye_length, 13.6, max_relative = 0.005);
    }

    #[test]
    fn hetero_dimer_archives_are_rejected() {
        assert!(NBodyRun::start(&archive(false, &["X"]), spec(), UNIT).is_err());
    }

    /// Equilibration feeds only the energy check: no curve and no frames.
    #[test]
    fn equilibration_produces_no_curve_or_frames() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        let progress = run.step(10).unwrap();
        assert_eq!(progress.stage, Stage::Equilibrating);
        assert_eq!(run.curve().frames, 0);
        let (header, frames) = frames_in(&run.trajectory_bytes().unwrap());
        assert_eq!((frames, header.equilibrated), (0, false));
    }

    /// After convergence, frames from every copy reach the curve and the
    /// trajectory, and move statistics start afresh.
    #[test]
    fn production_frames_reach_curve_and_trajectory() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        let progress = run.step(10).unwrap();
        assert_eq!(progress.stage, Stage::Production { equilibrated: true });
        assert!(progress.frames >= 4, "{progress:?}");
        assert_eq!(run.curve().frames, progress.frames);
        let (header, frames) = frames_in(&run.trajectory_bytes().unwrap());
        assert!(header.equilibrated);
        assert_eq!(frames, progress.frames);
        let stats = run.stats();
        let attempts = stats.moves.translate.attempts + stats.moves.rotate.attempts;
        // Production only, up to each copy's last frame: one move per body
        // per sweep, 10 bodies.
        let bytes = run.trajectory_bytes().unwrap();
        let reader = TrajectoryReader::new(&bytes[..]).unwrap();
        let mut last = [header.equilibration_sweeps; 2];
        reader.for_each(|f| {
            let f = f.unwrap();
            last[f.copy] = last[f.copy].max(f.sweep);
        });
        let sweeps: u64 = last.iter().map(|s| s - header.equilibration_sweeps).sum();
        assert_eq!(attempts, 10 * sweeps);
        assert_eq!(stats.energy.frames, progress.frames);
    }

    /// A browser builds the scattering asynchronously and starts with it;
    /// the run must not depend on how it was built.
    #[test]
    fn starting_with_prebuilt_scattering_matches_start() {
        let archive = archive(true, &["X"]);
        let mut direct = NBodyRun::start(&archive, spec(), UNIT).unwrap();
        let scattering = pollster::block_on(NBodyRun::scattering(&archive, &spec(), UNIT));
        let mut prebuilt = NBodyRun::start_with(&archive, spec(), scattering.unwrap()).unwrap();
        for run in [&mut direct, &mut prebuilt] {
            run_to(run, |s| matches!(s, Stage::Production { .. }));
            run.step(20).unwrap();
        }
        assert!(direct.curve().frames > 0);
        assert_eq!(direct.curve(), prebuilt.curve());
    }

    /// A paused or stopped UI keeps settling; on the CPU nothing is ever
    /// outstanding, and a stopped run still reports where it is.
    #[test]
    fn cpu_runs_have_nothing_left_to_settle() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        run.step(10).unwrap();
        assert!(!run.settle().unwrap());
        run.stop();
        let progress = run.step(0).unwrap();
        assert_eq!(progress.stage, Stage::Stopped);
        assert_eq!(run.curve().frames, progress.frames);
        assert!(!run.settle().unwrap());
    }

    /// Sampling never depends on where the sums run, so GPU and CPU see
    /// the same frames; only f32 phases separate their curves.
    #[test]
    #[ignore = "needs a GPU adapter: run in release with --ignored"]
    fn gpu_bead_sums_match_the_cpu() {
        let archive = archive(true, &["X", "X", "X"]);
        let mut gpu_spec = spec();
        gpu_spec.device = Device::Gpu;
        let scattering = pollster::block_on(NBodyRun::scattering(&archive, &gpu_spec, UNIT));
        let scattering = match scattering {
            Ok(scattering) => scattering,
            Err(e) => return eprintln!("skipping: no GPU scattering ({e})"),
        };
        let mut gpu = NBodyRun::start_with(&archive, gpu_spec, scattering).unwrap();
        let mut cpu = NBodyRun::start(&archive, spec(), UNIT).unwrap();
        for run in [&mut cpu, &mut gpu] {
            run_to(run, |s| matches!(s, Stage::Production { .. }));
            run.step(100).unwrap();
        }
        while gpu.settle().unwrap() {}
        let (cpu, gpu) = (cpu.curve(), gpu.curve());
        assert!(cpu.frames >= 40, "{}", cpu.frames);
        assert_eq!((cpu.frames, &cpu.q), (gpu.frames, &gpu.q));
        for (c, g) in cpu.s_eff.iter().zip(&gpu.s_eff) {
            approx::assert_relative_eq!(c, g, max_relative = 1e-3);
        }
    }

    /// A UI budgets wall time per step, so equilibration takes the sweeps
    /// asked for; its energy trace gains a point only per completed check.
    #[test]
    fn equilibration_traces_energy_at_each_check() {
        let mut run = NBodyRun::start(&archive_with(true, &["X"], -0.5), spec(), UNIT).unwrap();
        let first = run.step(1).unwrap();
        assert_eq!((first.stage, first.sweep), (Stage::Equilibrating, 1));
        assert!(run.energy_trace().is_empty());
        let progress = run.step(19).unwrap();
        let trace = run.energy_trace();
        let sweeps: Vec<f64> = trace.iter().map(|p| p[0]).collect();
        assert_eq!(sweeps, [10.0, 20.0]);
        assert_eq!(trace[1][1], progress.energy_per_body);
    }

    /// Small steps only split the work: the samplers draw the same random
    /// numbers, and checks, tuning, the start of production and staggered
    /// frames fall on the same sweeps however a total is split.
    #[test]
    fn small_steps_end_where_one_large_step_does() {
        let scan = archive_with(true, &["X"], -0.5);
        let mut whole = NBodyRun::start(&scan, spec(), UNIT).unwrap();
        let mut pieces = NBodyRun::start(&scan, spec(), UNIT).unwrap();
        // Past the 200-sweep cap, so production has started and framed.
        let once = whole.step(230).unwrap();
        let mut split = [3, 7, 13, 1, 26].iter().cycle();
        let mut last = pieces.progress();
        while last.sweep < 230 {
            let n = (*split.next().unwrap()).min(230 - last.sweep);
            last = pieces.step(n).unwrap();
        }
        assert!(matches!(once.stage, Stage::Production { .. }), "{once:?}");
        assert_eq!(once, last);
        assert_eq!(whole.stats().steps, pieces.stats().steps);
        assert_eq!(whole.frames(), pieces.frames());
        assert_eq!(whole.curve(), pieces.curve());
    }

    /// A UI budgets wall time per step, so production takes the sweeps
    /// asked for; frames still land on their interval.
    #[test]
    fn production_steps_take_the_sweeps_asked_for() {
        let mut spec = spec();
        spec.sweeps_per_frame = 50;
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        let start = run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        let first = run.step(1).unwrap();
        assert_eq!((first.sweep - start.sweep, first.frames), (1, start.frames));
        let framed = run.step(49).unwrap();
        assert_eq!(framed.sweep - start.sweep, 50);
        assert_eq!(framed.frames, start.frames + 2);
    }

    /// Copies take frames at staggered sweeps, so a UI step never pays for
    /// every copy's scattering at once; each still gets one per interval.
    #[test]
    fn copies_take_frames_at_staggered_sweeps() {
        let mut spec = spec();
        spec.copies = 4;
        spec.sweeps_per_frame = 40;
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        let start = run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        let frames: Vec<usize> = (0..8).map(|_| run.step(10).unwrap().frames).collect();
        let expected: Vec<usize> = (1..=8).map(|k| start.frames + k).collect();
        assert_eq!(frames, expected);
        let bytes = run.trajectory_bytes().unwrap();
        let reader = TrajectoryReader::new(&bytes[..]).unwrap();
        let mut copies: Vec<usize> = reader.map(|f| f.unwrap().copy).collect();
        copies.sort();
        assert_eq!(copies, [0, 0, 1, 1, 2, 2, 3, 3]);
    }

    /// Resetting discards production samples so far, e.g. once a plot shows
    /// the early frames were still relaxing, and production carries on.
    #[test]
    fn reset_statistics_restarts_production_samples() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        assert!(run.reset_statistics().is_err(), "nothing to reset yet");
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        let before = run.step(20).unwrap();
        assert!(before.frames > 0);
        let reset = run.reset_statistics().unwrap();
        assert_eq!((reset.frames, reset.sweep), (0, before.sweep));
        assert_eq!(run.curve().frames, 0);
        let stats = run.stats();
        assert_eq!(stats.energy.frames, 0);
        assert_eq!(
            stats.moves.translate.attempts + stats.moves.rotate.attempts,
            0
        );
        let (header, frames) = frames_in(&run.trajectory_bytes().unwrap());
        assert_eq!((frames, header.equilibration_sweeps), (0, before.sweep));
        assert!(header.equilibrated);

        let after = run.step(20).unwrap();
        assert_eq!(after.stage, Stage::Production { equilibrated: true });
        assert!(after.frames > 0);
        assert_eq!(run.curve().frames, after.frames);
    }

    /// Hard spheres (σ = 5 Å, the fixture's rmin) never come closer than σ,
    /// so g(r) is zero below it; like the curve it holds production only.
    #[test]
    fn pair_correlation_follows_production() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run.step(10).unwrap();
        assert_eq!(run.pair_correlation(), RdfCurve::default());
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        run.step(50).unwrap();
        let rdf = run.pair_correlation();
        assert_eq!(rdf.r.last(), Some(&9.75), "out to L/2 of the 20 Å box");
        for (r, g) in rdf.r.iter().zip(&rdf.g).filter(|(r, _)| **r < 5.0) {
            assert_eq!(*g, 0.0, "overlap at r = {r}");
        }
        assert!(rdf.g.iter().any(|&g| g > 0.0));
        run.reset_statistics().unwrap();
        assert_eq!(run.pair_correlation(), RdfCurve::default());
    }

    /// A viewer needs every copy's current configuration, labelled by copy.
    #[test]
    fn frames_give_each_copy_its_configuration() {
        let mut spec = spec();
        spec.copies = 3;
        let run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        let frames = run.frames();
        let copies: Vec<usize> = frames.iter().map(|f| f.copy).collect();
        assert_eq!(copies, [0, 1, 2]);
        assert!(frames.iter().all(|f| f.poses.len() == 10));
        assert_eq!(run.frame(1).unwrap(), frames[1]);
        assert!(run.frame(3).is_err());
    }

    /// Every move is accepted on a zero table, so frames show the motion.
    #[test]
    fn frames_follow_the_simulation() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        let before = run.frames();
        run.step(10).unwrap();
        let after = run.frames();
        assert_eq!((before.len(), after.len()), (2, 2));
        for (b, a) in before.iter().zip(&after) {
            assert_ne!(b.poses, a.poses, "copy {}", b.copy);
        }
    }

    #[test]
    fn reference_is_the_scanned_body() {
        let scan = archive(true, &["X", "CYS"]);
        let run = NBodyRun::start(&scan, spec(), UNIT).unwrap();
        assert_eq!(run.reference().kinds, ["X", "CYS"]);
        assert_eq!(
            run.reference().positions,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        );
    }

    /// A run that never meets the drift test still produces, flagged as such.
    #[test]
    fn sweep_cap_moves_to_production_unequilibrated() {
        let mut spec = spec();
        spec.criteria.max_drift = -1.0; // unattainable once energies fluctuate
        let attractive = archive_with(true, &["X"], -2.0);
        let mut run = NBodyRun::start(&attractive, spec, UNIT).unwrap();
        let progress = run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        assert_eq!(
            progress.stage,
            Stage::Production {
                equilibrated: false
            }
        );
        assert!(progress.sweep >= 200);
        let (header, _) = frames_in(&run.trajectory_bytes().unwrap());
        assert!(!header.equilibrated);
        assert!(run.report_csv().contains("# equilibrated: false\n"));
    }

    /// Statistics as `#` comments, so plotting tools skip them, then one
    /// row per q.
    #[test]
    fn report_csv_has_stats_then_the_curve() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        run.step(10).unwrap();
        let csv = run.report_csv();
        assert!(csv.contains("# equilibrated: true\n"), "{csv}");
        assert!(csv.contains("# translate acceptance: "), "{csv}");
        assert!(csv.contains("# U/N mean: "), "{csv}");
        let mut table = csv.lines().skip_while(|l| l.starts_with('#'));
        assert_eq!(
            table.next(),
            Some("q/Å⁻¹,I,P,S_eff,S_eff_error,S_com,S_com_error,S_HSY")
        );
        assert_eq!(table.count(), run.curve().q.len());
    }

    /// Every move is accepted on a zero table, so the largest steps win:
    /// L/2 = 10 Å (20 Å box) and π.
    #[test]
    fn tuning_takes_ideal_fluid_steps_to_their_bounds() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        assert_eq!(run.stats().steps, (10.0, std::f64::consts::PI));
    }

    #[test]
    fn tuned_steps_stay_frozen_and_are_reported() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        let frozen = run.stats().steps;
        for _ in 0..20 {
            run.step(10).unwrap();
        }
        assert_eq!(run.stats().steps, frozen);
        let csv = run.report_csv();
        assert!(csv.contains("# translation step/Å: 10.0000\n"), "{csv}");
        assert!(csv.contains("# rotation step/rad: 3.1416\n"), "{csv}");
        let header = frames_in(&run.trajectory_bytes().unwrap()).0;
        assert_eq!(
            (header.translation, header.rotation),
            (Some(frozen.0), Some(frozen.1))
        );
    }

    /// Dense hard spheres (σ = 5 Å, the fixture's rmin; φ = 0.3) reject
    /// large moves, so the tuned translation lies inside its bounds and the
    /// sampled structure still matches Percus–Yevick.
    #[test]
    #[ignore = "slow: run in release with --ignored"]
    fn tuned_hard_spheres_match_percus_yevick() {
        use crate::nbody::acceptance::{hard_sphere_density, percus_yevick};
        let (sigma, phi) = (5.0, 0.3);
        let density = hard_sphere_density(sigma, phi);
        let mut spec = RunSpec::new(
            300,
            density / (physical_constants::AVOGADRO_CONSTANT * 1e-27),
        );
        spec.q_max = 1.45;
        spec.vectors_per_bin = 500;
        spec.sweeps_per_frame = 5;
        spec.criteria.min_sweeps = 300;
        spec.criteria.max_sweeps = 3000;
        let half_box = box_length(spec.n_bodies, spec.molarity) / 2.0;
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        for _ in 0..300 {
            run.step(10).unwrap();
        }
        let stats = run.stats();
        assert!(
            stats.steps.0 > 0.1 && stats.steps.0 < half_box,
            "{:?}",
            stats.steps
        );
        let acceptance = stats.moves.translate.acceptance().unwrap();
        assert!((0.1..0.9).contains(&acceptance), "acceptance {acceptance}");
        let curve = run.curve();
        let errors = curve.s_com_error.as_ref().unwrap();
        for ((q, s), e) in curve.q.iter().zip(&curve.s_com).zip(errors) {
            let expected = percus_yevick(*q, sigma, phi);
            assert!(
                (s - expected).abs() < 4.0 * e + 0.05 * expected,
                "q = {q:.3}: S = {s:.3} ± {e:.3}, PY {expected:.3}"
            );
        }
    }

    /// Two check windows are too few for the tuners to settle; the cap
    /// must still end equilibration.
    #[test]
    fn the_sweep_cap_ends_equilibration_while_tuning() {
        let mut spec = spec();
        spec.criteria.min_sweeps = 10;
        spec.criteria.max_sweeps = 20;
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        for _ in 0..3 {
            run.step(10).unwrap();
        }
        assert!(matches!(run.progress().stage, Stage::Production { .. }));
    }

    /// A step beyond L/2 is only a start: it is clamped, not an error.
    #[test]
    fn starting_steps_beyond_the_box_are_clamped() {
        let mut spec = spec();
        spec.translation = 500.0;
        spec.tune_steps = false;
        let run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        assert_eq!(run.stats().steps.0, 10.0);
    }

    #[test]
    fn fixed_steps_stay_as_given() {
        let mut spec = spec();
        spec.tune_steps = false;
        spec.translation = 2.0;
        spec.rotation = 0.1;
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        assert_eq!(run.stats().steps, (2.0, 0.1));
    }

    #[test]
    fn centres_only_runs_skip_bead_scattering() {
        let mut spec = spec();
        spec.centres_only = true;
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        run.step(10).unwrap();
        let curve = run.curve();
        assert!(!curve.s_com.is_empty());
        assert!(curve.s_eff.is_empty() && curve.intensity.is_empty());
        assert!(run.report_csv().contains("# centres only: true\n"));
    }

    #[test]
    fn curve_carries_the_charged_sphere_reference() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        run.step(10).unwrap();
        let curve = run.curve();
        assert_eq!(curve.s_hsy.len(), curve.q.len());
        // One 1 g/mol neutral bead: a 1.3 Å sphere at φ ≈ 1.5e-3, so S ≈ 1.
        assert!(
            curve.s_hsy.iter().all(|s| (s - 1.0).abs() < 0.02),
            "{:?}",
            curve.s_hsy
        );
        assert!(
            run.report_csv().contains("# HSY sphere: σ = 1.3 Å"),
            "{}",
            run.report_csv()
        );
    }

    #[test]
    fn non_positive_msa_parameters_are_refused() {
        for bad in [0.0, -1.0, f64::NAN] {
            let (mut wide, mut screened) = (spec(), spec());
            wide.msa.diameter = Some(bad);
            screened.msa.debye_length = Some(bad);
            for spec in [wide, screened] {
                assert!(NBodyRun::start(&archive(true, &["X"]), spec, UNIT).is_err());
            }
        }
    }

    /// Only the theory changes: the simulation and its frames carry on.
    #[test]
    fn msa_parameters_can_change_mid_run() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        run.step(10).unwrap();
        let frames = run.curve().frames;
        let wide = MsaOverrides {
            diameter: Some(5.0),
            ..Default::default()
        };
        let derived = run.msa().derived().diameter;
        run.set_msa(wide).unwrap();
        let msa = run.msa();
        assert_eq!(msa.overrides(), wide);
        assert_eq!(
            (msa.sphere().diameter, msa.derived().diameter),
            (5.0, derived)
        );
        assert!(msa.solved());
        // As in the test below: φ = 0.082 pulls S(0) well under 1.
        assert!(run.curve().s_hsy[0] < 0.9, "{:?}", run.curve().s_hsy);
        assert_eq!(run.curve().frames, frames);
        assert!(run.report_csv().contains("σ = 5.0 Å"));
        let bad = MsaOverrides {
            debye_length: Some(-1.0),
            ..Default::default()
        };
        assert!(run.set_msa(bad).is_err());
        assert_eq!(run.msa().overrides(), wide);
        assert!(
            run.report_csv().contains("σ = 5.0 Å"),
            "a refused change keeps the old"
        );
    }

    #[test]
    fn chosen_msa_parameters_replace_the_derived_sphere() {
        let mut spec = spec();
        spec.msa = MsaOverrides {
            diameter: Some(5.0),
            debye_length: Some(3.0),
        };
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        run.step(10).unwrap();
        // φ = 1.25e-3 Å⁻³ × π 5³/6 = 0.082: hard-sphere S(0) ≈ 0.53, well below 1.
        assert!(run.curve().s_hsy[0] < 0.9, "{:?}", run.curve().s_hsy);
        let csv = run.report_csv();
        assert!(
            csv.contains("σ = 5.0 Å") && csv.contains("Debye length 3.0 Å"),
            "{csv}"
        );
    }

    #[test]
    fn recording_stops_at_its_cap_while_the_curve_continues() {
        let mut spec = spec();
        spec.max_trajectory_bytes = 3000; // header plus a few frames
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec, UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        let progress = (0..20).map(|_| run.step(10).unwrap()).last().unwrap();
        assert!(!progress.recording);
        let bytes = run.trajectory_bytes().unwrap();
        assert!(bytes.len() <= 3000);
        assert!(frames_in(&bytes).1 < progress.frames);
        assert_eq!(run.curve().frames, progress.frames);
    }

    #[test]
    fn stopping_keeps_results_and_ends_stepping() {
        let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
        run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
        let before = run.step(10).unwrap();
        run.stop();
        let after = run.step(10).unwrap();
        assert_eq!(after.stage, Stage::Stopped);
        assert_eq!(after.frames, before.frames);
        assert_eq!(run.curve().frames, before.frames);
        assert_eq!(frames_in(&run.trajectory_bytes().unwrap()).1, before.frames);
    }

    /// A zero interval would make `step` loop forever.
    #[test]
    fn zero_sweep_intervals_are_rejected() {
        let mut spec = spec();
        spec.sweeps_per_frame = 0;
        assert!(NBodyRun::start(&archive(true, &["X"]), spec, UNIT).is_err());
    }

    /// Kinds already defined, however the key is written, are not added again.
    #[test]
    fn defined_kinds_are_not_supplied_twice() {
        let yaml = format!("{UNIT}\"N\": {{ formfactor: !Constant 2.0 }}\n");
        assert!(NBodyRun::start(&archive(true, &["N", "X"]), spec(), &yaml).is_ok());
    }

    /// Terminal beads (N, O) and disulfide cysteines (CYX) lack residue form
    /// factors; the run supplies them rather than failing.
    #[test]
    fn beads_without_form_factors_are_supplied() {
        assert!(NBodyRun::start(&archive(true, &["N", "CYX", "O"]), spec(), UNIT).is_ok());
    }
}

//! A whole many-body run: equilibration, then production of I(q), energy
//! and move statistics, and a trajectory, from a homo-dimer scan archive.
//!
//! Synchronous: callers advance it in small steps (e.g. one per UI frame),
//! which keeps a browser responsive without async.

use super::trajectory::{TrajectoryHeader, TrajectoryWriter};
use super::{
    box_length, Criteria, Device, EnergyStats, EnergySummary, Equilibration, Frame, MoveStats,
    PairEnergy, Sampler, SamplerSpec, Scattering, ScatteringCurve, Status,
};
use crate::archive::ScanArchive;
use anyhow::{ensure, Result};
use std::borrow::Cow;
use std::sync::Arc;

/// What to run. `RunSpec::new` fills in defaults for everything but the
/// system.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub n_bodies: usize,
    /// Protein concentration in mol/L.
    pub molarity: f64,
    /// Independent copies; their spread gives the error bars.
    pub copies: usize,
    pub seed: u64,
    /// Largest q (Å⁻¹) in the scattering curve.
    pub q_max: f64,
    /// Sweeps between production frames.
    pub sweeps_per_frame: u64,
    /// Sweeps between equilibration energy checks.
    pub sweeps_per_check: u64,
    pub criteria: Criteria,
    /// Maximum translation per axis (Å) and rotation angle (rad).
    pub translation: f64,
    pub rotation: f64,
    /// Recording stops beyond this; the run and curve carry on.
    pub max_trajectory_bytes: usize,
}

impl RunSpec {
    pub fn new(n_bodies: usize, molarity: f64) -> Self {
        Self {
            n_bodies,
            molarity,
            copies: 4,
            seed: 1,
            q_max: 0.5,
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
            max_trajectory_bytes: 256 << 20,
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
    /// Latest copy-averaged energy per body (kT).
    pub energy_per_body: f64,
    /// Production frames, summed over copies.
    pub frames: usize,
    /// False once the trajectory reached its size cap.
    pub recording: bool,
}

/// Production statistics, pooled over copies.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunStats {
    pub moves: MoveStats,
    pub energy: EnergySummary,
}

pub struct NBodyRun {
    spec: RunSpec,
    archive_settings: crate::archive::ScanSettings,
    reference: crate::archive::CgBody,
    samplers: Vec<Sampler>,
    stage: Stage,
    sweep: u64,
    equilibration: Equilibration,
    /// Production energy statistics; also gives per-frame energies during
    /// equilibration, which are not added to it.
    energy: EnergyStats,
    moves: MoveStats,
    scattering: Scattering,
    /// Created when production starts, as the header records how
    /// equilibration ended.
    trajectory: Option<TrajectoryWriter<Vec<u8>>>,
    recording: bool,
    frames: usize,
    energy_per_body: f64,
}

impl NBodyRun {
    /// Fails for hetero-dimer archives: every body is the archive's first.
    pub fn start(archive: &ScanArchive, spec: RunSpec, formfactors_yaml: &str) -> Result<Self> {
        ensure!(
            archive.settings.homo_dimer,
            "many-body runs need a scan of identical molecules"
        );
        validate(&spec)?;
        let energy = Arc::new(PairEnergy::new(archive.table.clone())?);
        let sampler_spec = SamplerSpec {
            n_bodies: spec.n_bodies,
            box_length: box_length(spec.n_bodies, spec.molarity),
            translation: spec.translation,
            rotation: spec.rotation,
        };
        let samplers = (0..spec.copies)
            .map(|c| {
                Sampler::new(
                    energy.clone(),
                    sampler_spec.clone(),
                    spec.seed.wrapping_add(c as u64),
                )
            })
            .collect::<Result<_>>()?;
        let reference = archive.bodies[0].clone();
        let yaml = with_missing_kinds(formfactors_yaml, &reference.kinds);
        let scattering = pollster::block_on(Scattering::new(
            &reference,
            spec.n_bodies,
            sampler_spec.box_length,
            spec.q_max,
            &yaml,
            Device::Cpu,
        ))?;
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
            trajectory: None,
            recording: true,
            frames: 0,
            energy_per_body: 0.0,
        })
    }

    /// Advances every copy by about `sweeps`, in whole check or frame steps.
    /// An error stops the run, as copies may then be out of step.
    pub fn step(&mut self, sweeps: u64) -> Result<Progress> {
        let mut done = 0;
        while done < sweeps {
            let advanced = match self.stage {
                Stage::Equilibrating => self.equilibration_step(),
                Stage::Production { .. } => self.production_step(),
                Stage::Stopped => break,
            };
            match advanced {
                Ok(sweeps) => done += sweeps,
                Err(e) => {
                    self.stage = Stage::Stopped;
                    return Err(e);
                }
            }
        }
        Ok(self.progress())
    }

    fn progress(&self) -> Progress {
        Progress {
            stage: self.stage,
            sweep: self.sweep,
            energy_per_body: self.energy_per_body,
            frames: self.frames,
            recording: self.recording,
        }
    }

    fn advance(&mut self, sweeps: u64) -> Vec<Frame> {
        self.sweep += sweeps;
        self.samplers
            .iter_mut()
            .enumerate()
            .map(|(copy, sampler)| {
                sampler.advance(sweeps);
                Frame {
                    copy,
                    ..sampler.frame()
                }
            })
            .collect()
    }

    fn equilibration_step(&mut self) -> Result<u64> {
        let sweeps = self.spec.sweeps_per_check;
        let frames = self.advance(sweeps);
        let energies = frames
            .iter()
            .map(|f| self.energy.frame_energy(f))
            .collect::<Result<Vec<_>>>()?;
        self.energy_per_body = mean(&energies) / self.spec.n_bodies as f64;
        match self.equilibration.push(self.sweep, &energies) {
            Status::Running => {}
            Status::Converged => self.begin_production(true)?,
            Status::CapReached => self.begin_production(false)?,
        }
        Ok(sweeps)
    }

    /// Statistics restart here, so equilibration never enters them.
    fn begin_production(&mut self, equilibrated: bool) -> Result<()> {
        self.samplers
            .iter_mut()
            .for_each(|s| drop(s.take_move_stats()));
        let header = self.header(equilibrated);
        self.trajectory = Some(TrajectoryWriter::new(&header, Vec::new())?);
        self.stage = Stage::Production { equilibrated };
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
        }
    }

    fn production_step(&mut self) -> Result<u64> {
        let sweeps = self.spec.sweeps_per_frame;
        let frames = self.advance(sweeps);
        let mut energies = Vec::with_capacity(frames.len());
        for frame in &frames {
            pollster::block_on(self.scattering.observe(frame))?;
            energies.push(self.energy.observe(frame)?);
            self.record(frame)?;
        }
        for sampler in &mut self.samplers {
            self.moves.merge(&sampler.take_move_stats());
        }
        self.frames += frames.len();
        self.energy_per_body = mean(&energies) / self.spec.n_bodies as f64;
        Ok(sweeps)
    }

    fn record(&mut self, frame: &Frame) -> Result<()> {
        let Some(writer) = self.trajectory.as_mut() else {
            return Ok(());
        };
        if writer.sink().len() + writer.record_bytes() > self.spec.max_trajectory_bytes {
            self.recording = false;
        }
        if self.recording {
            writer.append(frame)?;
        }
        Ok(())
    }

    pub fn curve(&self) -> ScatteringCurve {
        self.scattering.curve()
    }

    pub fn stats(&self) -> RunStats {
        RunStats {
            moves: self.moves.clone(),
            energy: self.energy.summary(),
        }
    }

    /// The trajectory so far, borrowed as it can be large; header only if
    /// production hasn't started.
    pub fn trajectory_bytes(&self) -> Result<Cow<'_, [u8]>> {
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
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Residue form-factor files lack the terminal charge beads (N, O) and
/// disulfide cysteines (CYX). N and O mark small charged groups next to
/// their residue's bead, so they scatter nothing here; CYX scatters as CYS.
/// Added only when absent, so a file that defines them wins.
fn with_missing_kinds(yaml: &str, kinds: &[String]) -> String {
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
                molarity: 0.05,
                temperature: 298.15,
                max_ndiv: 1,
                gradient_threshold: 0.5,
                cutoff: 30.0,
                grid_size: 200,
                homo_dimer,
            },
            table: table(move |_, _, _| well),
            bodies: [body.clone(), body],
            points: vec![],
            virial: crate::VirialCoeff::from_raw_parts(0.0, 5.0),
            topology_yaml: String::new(),
        }
    }

    /// Ten bodies in a 20 Å box (ρ = 1.25e-3 Å⁻³), small test criteria.
    fn spec() -> RunSpec {
        let molarity = 10.0 / 8000.0 / (physical_constants::AVOGADRO_CONSTANT * 1e-27);
        let mut spec = RunSpec::new(10, molarity);
        spec.copies = 2;
        spec.q_max = 0.6;
        spec.sweeps_per_frame = 5;
        spec.criteria.min_sweeps = 20;
        spec.criteria.max_sweeps = 200;
        spec
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
        // Production only: 2 copies × 10 bodies × the production sweeps.
        assert_eq!(attempts, 2 * 10 * (progress.frames as u64 / 2) * 5);
        assert_eq!(stats.energy.frames, progress.frames);
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

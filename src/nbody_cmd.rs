//! `duello nbody` and `duello nbody-expand`: the many-body run from a saved
//! scan, as in the web tool's Manybody tab, but unattended.

use anyhow::{ensure, Context, Result};
use duello::archive::ScanArchive;
use duello::nbody::{
    trajectory, MsaOverrides, NBodyRun, Progress, RunSpec, Stage, RESIDUE_FORM_FACTORS,
};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(clap::Args)]
pub struct NBodyArgs {
    /// Homo-dimer scan archive from `duello scan --savescan`
    #[arg(long)]
    scan: PathBuf,
    /// Number of proteins
    #[arg(short = 'n', long)]
    bodies: usize,
    /// Protein concentration (mol/L); salt is fixed by the scan
    #[arg(long)]
    protein_molarity: f64,
    /// Independent copies, whose spread gives the error bars
    #[arg(long, default_value = "4")]
    copies: usize,
    /// Production sweeps per copy, after equilibration
    #[arg(long, default_value = "10000")]
    sweeps: u64,
    /// Sweeps between I(q) frames [default: 50, or 10 with --centres-only]
    #[arg(long)]
    frame_interval: Option<u64>,
    /// Largest q (Å⁻¹)
    #[arg(long, default_value = "0.5")]
    qmax: f64,
    #[arg(long, default_value = "1")]
    seed: u64,
    /// Lattice q-vectors per |q| bin for the bead sums; more cost more
    /// per frame but give smoother I(q) and S_eff
    #[arg(long, default_value = "16")]
    vectors_per_bin: usize,
    /// Maximum translation per axis (Å): the starting step, or the step
    /// throughout with --fixed-steps [default: the run's, 3 Å]
    #[arg(long)]
    translation: Option<f64>,
    /// Maximum rotation angle (rad), as for --translation [default: 0.2]
    #[arg(long)]
    rotation: Option<f64>,
    /// Keep the steps as given instead of tuning them during equilibration
    #[arg(long)]
    fixed_steps: bool,
    /// Diameter of the MSA reference sphere (Å) [default: from molar mass]
    #[arg(long)]
    msa_diameter: Option<f64>,
    /// Debye length of the MSA reference (Å) [default: salt plus counterions]
    #[arg(long)]
    msa_debye_length: Option<f64>,
    /// Mass-centre S(q) only; far cheaper per frame, so frames are taken
    /// more often by default
    #[arg(long)]
    centres_only: bool,
    /// Output prefix: writes PREFIX.csv, PREFIX.rdf.csv (g(r)) and PREFIX.traj
    #[arg(short = 'o', long, default_value = "nbody")]
    output: PathBuf,
}

impl NBodyArgs {
    /// Mass-centre frames cost little next to the sweeps between them.
    fn frame_interval(&self) -> u64 {
        self.frame_interval
            .unwrap_or(if self.centres_only { 10 } else { 50 })
    }
}

#[derive(clap::Args)]
pub struct ExpandArgs {
    /// Trajectory from `duello nbody`
    trajectory: PathBuf,
    /// Bead XTC; an XYZ topology is written next to it
    #[arg(short = 'o', long, default_value = "nbody.xtc")]
    output: PathBuf,
    /// Copy to keep; copies are independent runs, so a viewer needs one
    #[arg(long, default_value = "0", conflicts_with = "all_copies")]
    copy: usize,
    /// Keep every copy, e.g. for averaging I(q); frames interleave
    #[arg(long)]
    all_copies: bool,
}

pub fn run_nbody(args: &NBodyArgs) -> Result<()> {
    ensure!(
        args.sweeps > 0 && args.frame_interval() > 0,
        "--sweeps and --frame-interval must be positive"
    );
    let bytes = std::fs::read(&args.scan).with_context(|| args.scan.display().to_string())?;
    let archive = ScanArchive::from_bytes(&bytes)?;
    let mut run = NBodyRun::start(&archive, spec(args), RESIDUE_FORM_FACTORS)?;
    let progress = drive(&mut run, args)?;
    if !progress.recording {
        log::warn!("Trajectory reached its size cap; later frames are not recorded");
    }
    std::fs::write(with_suffix(&args.output, ".csv"), run.report_csv())?;
    // Its own file: a second table would break readers of the first.
    std::fs::write(
        with_suffix(&args.output, ".rdf.csv"),
        run.pair_correlation().to_csv(),
    )?;
    std::fs::write(with_suffix(&args.output, ".traj"), run.trajectory_bytes()?)?;
    Ok(())
}

fn spec(args: &NBodyArgs) -> RunSpec {
    let mut spec = RunSpec::new(args.bodies, args.protein_molarity);
    spec.copies = args.copies;
    spec.seed = args.seed;
    spec.q_max = args.qmax;
    spec.sweeps_per_frame = args.frame_interval();
    spec.centres_only = args.centres_only;
    spec.vectors_per_bin = args.vectors_per_bin;
    spec.translation = args.translation.unwrap_or(spec.translation);
    spec.rotation = args.rotation.unwrap_or(spec.rotation);
    spec.tune_steps = !args.fixed_steps;
    spec.msa = MsaOverrides {
        diameter: args.msa_diameter,
        debye_length: args.msa_debye_length,
    };
    spec
}

/// Appends rather than `with_extension`, which would cut a prefix such as
/// `lys_0.05M` at its dot.
fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut name = prefix.as_os_str().to_owned();
    name.push(suffix);
    name.into()
}

/// Steps until each copy has its production frames. Counting frames rather
/// than sweeps is exact even when a step crosses into production. Copies take
/// frames at staggered sweeps, so single sweeps stop right at the last one.
fn drive(run: &mut NBodyRun, args: &NBodyArgs) -> Result<Progress> {
    let target = args.sweeps.div_ceil(args.frame_interval()) as usize * args.copies;
    let mut announced = false;
    loop {
        let progress = run.step(1)?;
        if let (Stage::Production { equilibrated }, false) = (progress.stage, announced) {
            log_equilibration(equilibrated, run.stats().steps);
            announced = true;
        }
        if progress.frames >= target {
            return Ok(progress);
        }
    }
}

fn log_equilibration(equilibrated: bool, (translation, rotation): (f64, f64)) {
    log::info!("Move steps: {translation:.2} Å, {rotation:.3} rad");
    if equilibrated {
        log::info!("Equilibrated; production started");
    } else {
        log::warn!("Energy not converged at the sweep cap; producing anyway");
    }
}

pub fn run_expand(args: &ExpandArgs) -> Result<()> {
    let source = std::fs::File::open(&args.trajectory)
        .with_context(|| args.trajectory.display().to_string())?;
    let topology = args.output.with_extension("xyz");
    check_outputs(args, &topology)?;
    let copy = (!args.all_copies).then_some(args.copy);
    let summary = write_xtc(BufReader::new(source), &args.output, copy).inspect_err(|_| {
        std::fs::remove_file(&args.output).ok();
    })?;
    std::fs::write(&topology, summary.topology_xyz)?;
    log::info!(
        "Wrote {} frames to {} with topology {}",
        summary.frames,
        args.output.display(),
        topology.display()
    );
    Ok(())
}

/// Guards the input and keeps the topology from overwriting the XTC.
fn check_outputs(args: &ExpandArgs, topology: &Path) -> Result<()> {
    ensure!(
        args.output.extension().is_some_and(|e| e == "xtc"),
        "output must end in .xtc"
    );
    let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    ensure!(
        !same(&args.output, &args.trajectory) && !same(topology, &args.trajectory),
        "output would overwrite the input trajectory"
    );
    Ok(())
}

fn write_xtc(
    source: impl std::io::Read,
    path: &Path,
    copy: Option<usize>,
) -> Result<trajectory::XtcSummary> {
    let mut xtc = BufWriter::new(std::fs::File::create(path)?);
    let summary = trajectory::to_xtc(source, &mut xtc, copy)?;
    // Explicit, as a flush on drop would hide a full disk.
    xtc.flush()?;
    Ok(summary)
}

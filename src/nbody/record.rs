//! A many-body run's inputs and results, detached from the running engine so
//! they can be saved, reloaded and shown without a scan or GPU.

use super::{HsyFluid, NBodyRun, RdfCurve, RunSpec, RunStats, ScatteringCurve};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Bound to the curve and stats types, so changing any of them changes the
/// session format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub spec: RunSpec,
    pub curve: ScatteringCurve,
    pub rdf: RdfCurve,
    /// (sweep, U/N in kT), as [`NBodyRun::energy_trace`].
    pub energy: Vec<[f64; 2]>,
    pub stats: RunStats,
    /// False if the sweep cap was reached first.
    pub equilibrated: bool,
    pub equilibration_sweeps: u64,
    /// Sweeps per copy, equilibration included.
    pub sweeps: u64,
    /// The charged sphere behind `curve.s_hsy`.
    pub sphere: HsyFluid,
    /// Saved apart, as it can reach hundreds of MB.
    #[serde(skip)]
    pub trajectory: Option<Vec<u8>>,
}

impl RunRecord {
    /// Copies the trajectory too, when the run recorded one.
    pub fn from_run(run: &NBodyRun) -> Result<Self> {
        let trajectory = run
            .spec()
            .record_trajectory
            .then(|| run.trajectory_bytes().map(|b| b.into_owned()))
            .transpose()?;
        Ok(run.snapshot(trajectory))
    }

    /// Curve code zips these vectors, so an opened file must agree on lengths.
    /// S(q) columns may be empty instead: bead sums are off for mass-centre
    /// runs, and the HSY curve when MSA has no solution.
    pub(crate) fn check_consistency(&self) -> Result<()> {
        let rdf = &self.rdf;
        let n = rdf.r.len();
        ensure!(
            rdf.g.len() == n && rdf.error.as_ref().is_none_or(|e| e.len() == n),
            "run's g(r) has inconsistent lengths"
        );
        let c = &self.curve;
        let columns = [&c.intensity, &c.form_factor, &c.s_eff, &c.s_hsy, &c.s_com];
        let errors = [&c.s_eff_error, &c.s_com_error].into_iter().flatten();
        let fits = |v: &Vec<f64>| v.is_empty() || v.len() == c.q.len();
        ensure!(
            columns.into_iter().chain(errors).all(fits),
            "run's S(q) has inconsistent lengths"
        );
        Ok(())
    }

    /// Curve and statistics as CSV, with the statistics as `#` comments.
    pub fn report_csv(&self) -> String {
        let mut csv = self.stats_comments();
        csv += &self.curve.to_csv();
        csv
    }

    fn stats_comments(&self) -> String {
        let stats = &self.stats;
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
            // Var(U)/(N (kT)²): the excess heat capacity per molecule in k_B.
            format!("Cv_ex/(N kB): {}", per_body(stats.energy.variance)),
            self.sphere.describe(),
        ]
        .iter()
        .map(|line| format!("# {line}\n"))
        .collect()
    }

    /// (sweep, U/N) as CSV.
    pub fn energy_csv(&self) -> String {
        let mut csv = String::from("sweep,U/N/kT\n");
        for [sweep, u] in &self.energy {
            csv += &format!("{sweep},{u:.6}\n");
        }
        csv
    }
}

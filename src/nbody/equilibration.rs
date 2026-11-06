//! Decides when energy has stopped drifting, from a series of per-copy energies.

/// When to start and stop testing for drift.
#[derive(Clone, Copy, Debug)]
pub struct Criteria {
    /// No verdict before this many sweeps.
    pub min_sweeps: u64,
    /// Give up after this many sweeps.
    pub max_sweeps: u64,
    /// Standard errors added to the measured drift, so the verdict needs the
    /// drift to be shown small, not merely hidden by noise.
    pub tolerance: f64,
    /// Largest acceptable drift between the last two quarters, as a fraction
    /// of the frame-to-frame energy spread; scale-free, so it suits any
    /// system size or interaction strength.
    pub max_drift: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Status {
    Running,
    Converged,
    CapReached,
}

/// Compares the last two quarters of the history, so the windows grow with
/// the run and early relaxation drops out of both.
pub struct Equilibration {
    criteria: Criteria,
    /// Per-copy energies at each push.
    history: Vec<Vec<f64>>,
}

impl Equilibration {
    pub fn new(criteria: Criteria) -> Self {
        Self {
            criteria,
            history: Vec::new(),
        }
    }

    /// Adds the total energy of each copy at `sweep`.
    /// The number of copies must not change between pushes.
    pub fn push(&mut self, sweep: u64, copy_energies: &[f64]) -> Status {
        let copies = self.history.first().map_or(copy_energies.len(), Vec::len);
        assert!(
            !copy_energies.is_empty() && copy_energies.len() == copies,
            "expected energies for {copies} copies, got {}",
            copy_energies.len()
        );
        self.history.push(copy_energies.to_vec());
        if sweep < self.criteria.min_sweeps {
            Status::Running
        } else if self.drift_is_negligible() {
            Status::Converged
        } else if sweep >= self.criteria.max_sweeps {
            Status::CapReached
        } else {
            Status::Running
        }
    }

    fn drift_is_negligible(&self) -> bool {
        let n = self.history.len();
        if n < 8 {
            return false; // too few points per window for a spread
        }
        let (early, late) = (&self.history[n / 2..3 * n / 4], &self.history[3 * n / 4..]);
        let (drift, error) = match self.history[0].len() {
            1 => within_series(early, late),
            _ => between_copies(early, late),
        };
        // Equivalence test: the drift plus its uncertainty must fit inside
        // the margin. Testing only "drift < k·error" passes whenever slow,
        // copy-to-copy fluctuations make the error large.
        let spread = variance(&late.concat()).sqrt();
        drift.abs() + self.criteria.tolerance * error <= self.criteria.max_drift * spread
    }
}

/// Copies are independent, so the spread of their drifts gives an honest
/// error even though each copy's series is autocorrelated.
fn between_copies(early: &[Vec<f64>], late: &[Vec<f64>]) -> (f64, f64) {
    let copies = early[0].len();
    let drifts: Vec<f64> = (0..copies)
        .map(|c| mean(late.iter().map(|e| e[c])) - mean(early.iter().map(|e| e[c])))
        .collect();
    let error = (variance(&drifts) / copies as f64).sqrt();
    (mean(drifts.iter().copied()), error)
}

/// Fallback for one copy. Ignores autocorrelation, so it underestimates the
/// error and errs towards running longer.
fn within_series(early: &[Vec<f64>], late: &[Vec<f64>]) -> (f64, f64) {
    let (a, b): (Vec<f64>, Vec<f64>) = (
        early.iter().map(|e| e[0]).collect(),
        late.iter().map(|e| e[0]).collect(),
    );
    let error = (variance(&a) / a.len() as f64 + variance(&b) / b.len() as f64).sqrt();
    (mean(b.iter().copied()) - mean(a.iter().copied()), error)
}

fn mean(values: impl ExactSizeIterator<Item = f64>) -> f64 {
    let n = values.len() as f64;
    values.sum::<f64>() / n
}

/// Sample variance (n - 1).
fn variance(values: &[f64]) -> f64 {
    let m = mean(values.iter().copied());
    values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    const CRITERIA: Criteria = Criteria {
        min_sweeps: 500,
        max_sweeps: 20_000,
        tolerance: 2.0,
        max_drift: 0.5,
    };

    fn run(copies: usize, signal: impl Fn(f64) -> f64) -> (u64, Status) {
        run_noisy(copies, 1.0, 0.0, signal)
    }

    /// Pushes `signal(sweep)` plus noise for `copies` copies, every 10 sweeps,
    /// and returns the first sweep with a final status. The noise has standard
    /// deviation `noise_std` and correlation time `tau` sweeps (AR(1); 0 for
    /// white noise), as slow fluctuations like clustering produce.
    fn run_noisy(
        copies: usize,
        noise_std: f64,
        tau: f64,
        signal: impl Fn(f64) -> f64,
    ) -> (u64, Status) {
        let mut rng = Pcg64::seed_from_u64(17);
        let mut equilibration = Equilibration::new(CRITERIA);
        let a = if tau > 0.0 { (-10.0 / tau).exp() } else { 0.0 };
        // Uniform on ±√3 has unit variance.
        let mut kick = || rng.gen_range(-3f64.sqrt()..3f64.sqrt());
        let mut state: Vec<f64> = (0..copies).map(|_| noise_std * kick()).collect();
        for sweep in (10..).step_by(10) {
            for x in state.iter_mut() {
                *x = a * *x + (1.0 - a * a).sqrt() * noise_std * kick();
            }
            let energies: Vec<f64> = state.iter().map(|x| signal(sweep as f64) + x).collect();
            let status = equilibration.push(sweep, &energies);
            if status != Status::Running {
                return (sweep, status);
            }
        }
        unreachable!()
    }

    #[test]
    fn flat_series_converges_once_the_minimum_has_passed() {
        let (sweep, status) = run(8, |_| -40.0);
        assert_eq!(status, Status::Converged);
        assert!((500..1000).contains(&sweep), "converged at {sweep}");
    }

    /// 50 kT decaying with τ = 200 sweeps: still 1 kT at ~4τ, far above the
    /// noise in a window mean, so no verdict before then.
    #[test]
    fn relaxing_series_converges_only_after_the_decay() {
        let (sweep, status) = run(8, |t| -40.0 + 50.0 * (-t / 200.0).exp());
        assert_eq!(status, Status::Converged);
        assert!((800..5000).contains(&sweep), "converged at {sweep}");
    }

    /// Noise that hides the drift is not evidence of no drift. With 60 kT
    /// relaxing (τ = 300) under slow noise (σ = 10, correlation 200 sweeps),
    /// the verdict must wait until the remaining offset is within the drift
    /// margin, max_drift·σ (20% slack for the 10-sweep grid). Testing only
    /// "drift < 2 errors" stopped at sweep 510, with 11 kT still to go.
    #[test]
    fn noisy_slow_relaxation_is_not_declared_converged_early() {
        let offset = |t: f64| 60.0 * (-t / 300.0).exp();
        let (sweep, status) = run_noisy(4, 10.0, 200.0, |t| -40.0 + offset(t));
        assert_eq!(status, Status::Converged);
        let margin = 1.2 * CRITERIA.max_drift * 10.0;
        let remaining = offset(sweep as f64);
        assert!(
            remaining <= margin,
            "converged at {sweep} with {remaining:.1} kT to go"
        );
    }

    #[test]
    fn drifting_series_never_converges() {
        let (sweep, status) = run(8, |t| 0.01 * t);
        assert_eq!((sweep, status), (CRITERIA.max_sweeps, Status::CapReached));
    }

    #[test]
    #[should_panic(expected = "copies")]
    fn changing_the_number_of_copies_is_a_bug() {
        let mut equilibration = Equilibration::new(CRITERIA);
        equilibration.push(10, &[1.0, 2.0]);
        equilibration.push(20, &[1.0, 2.0, 3.0]);
    }

    /// One copy has no spread between copies; the error comes from the series.
    #[test]
    fn single_copy_uses_the_spread_within_its_series() {
        let (sweep, status) = run(1, |_| -40.0);
        assert_eq!(status, Status::Converged);
        assert!((500..10_000).contains(&sweep), "converged at {sweep}");
    }
}

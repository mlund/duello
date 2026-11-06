//! Chooses a Monte Carlo step size during equilibration: the one that
//! moves bodies furthest per attempted move, acceptance × ⟨Δ²⟩.

use super::MoveCounter;

/// Searches one move type's step; feed it one window of tallies at a time.
///
/// A hill-climb in log step: keep going while the objective rises, turn
/// back with a narrower factor when it falls, and settle on the best step
/// once the factor is too small to matter.
#[derive(Clone, Debug)]
pub struct StepTuner {
    step: f64,
    max: f64,
    factor: f64,
    upward: bool,
    previous: Option<f64>,
    best: (f64, f64),
    /// Windows in a row observed at `max`.
    at_bound: u32,
    /// Sticky: once set, window noise can't restart the search.
    settled: bool,
}

/// Below this the steps tried differ by less than window noise.
const SETTLED_FACTOR: f64 = 1.05;

impl StepTuner {
    /// `max` bounds the step (L/2 for translations, π for rotations). A
    /// zero step means the move type is off and stays so.
    pub fn new(initial: f64, max: f64) -> Self {
        let step = initial.min(max);
        Self {
            step,
            max,
            factor: 2.0,
            upward: true,
            previous: None,
            best: (step, f64::NEG_INFINITY),
            at_bound: 0,
            settled: initial == 0.0,
        }
    }

    /// Takes the tallies of a window run at the current step; returns the
    /// step for the next window.
    pub fn observe(&mut self, window: &MoveCounter) -> f64 {
        let Some(objective) = window.mean_square_displacement() else {
            return self.step;
        };
        if self.settled {
            return self.step;
        }
        self.at_bound = if self.step == self.max {
            self.at_bound + 1
        } else {
            0
        };
        if objective > self.best.1 {
            self.best = (self.step, objective);
        }
        if objective == 0.0 {
            // All rejected: only smaller steps can help, at full stride.
            self.upward = false;
        } else if self.previous.is_some_and(|previous| objective < previous) {
            self.upward = !self.upward;
            self.factor = self.factor.sqrt();
        }
        self.previous = Some(objective);
        self.settled = self.factor < SETTLED_FACTOR || self.at_bound >= 2;
        self.step = if self.settled {
            self.best.0
        } else {
            let scale = if self.upward {
                self.factor
            } else {
                1.0 / self.factor
            };
            (self.step * scale).min(self.max)
        };
        self.step
    }

    /// Whether further windows would change little: the search has narrowed
    /// to noise level, or the step has stayed at the bound.
    pub fn settled(&self) -> bool {
        self.settled
    }

    /// The best step seen, to freeze for production.
    pub fn best(&self) -> f64 {
        self.best.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D0: f64 = 2.0;

    /// Tallies of a cube proposal (⟨Δr²⟩ = d² per accepted move) with
    /// acceptance `a(d)`.
    fn window(d: f64, acceptance: impl Fn(f64) -> f64) -> MoveCounter {
        let attempts = 1_000_000;
        let accepted = (acceptance(d) * attempts as f64).round() as u64;
        MoveCounter {
            attempts,
            accepted,
            sum_square_displacement: accepted as f64 * d * d,
        }
    }

    fn tune(start: f64, max: f64, acceptance: impl Fn(f64) -> f64, windows: usize) -> f64 {
        let mut tuner = StepTuner::new(start, max);
        let mut step = start;
        for _ in 0..windows {
            step = tuner.observe(&window(step, &acceptance));
        }
        tuner.best()
    }

    /// d²·e^{−d/d₀} peaks at d = 2d₀.
    #[test]
    fn finds_the_largest_displacement_from_below() {
        let best = tune(D0 / 4.0, 100.0, |d| (-d / D0).exp(), 30);
        approx::assert_relative_eq!(best, 2.0 * D0, max_relative = 0.1);
    }

    #[test]
    fn finds_the_largest_displacement_from_above() {
        let best = tune(10.0 * D0, 100.0, |d| (-d / D0).exp(), 30);
        approx::assert_relative_eq!(best, 2.0 * D0, max_relative = 0.1);
    }

    /// Every move accepted: larger is always better, up to the bound.
    #[test]
    fn always_accepted_moves_climb_to_the_bound() {
        assert_eq!(tune(0.1, 7.5, |_| 1.0, 30), 7.5);
    }

    /// 5% multiplicative noise on each window's displacement sum.
    #[test]
    fn noise_does_not_derail_the_search() {
        let mut seed = 7u64;
        let mut noise = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1.0 + 0.1 * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        };
        let mut tuner = StepTuner::new(D0 / 4.0, 100.0);
        let mut step = D0 / 4.0;
        for _ in 0..40 {
            let mut tallies = window(step, |d| (-d / D0).exp());
            tallies.sum_square_displacement *= noise();
            step = tuner.observe(&tallies);
        }
        approx::assert_relative_eq!(tuner.best(), 2.0 * D0, max_relative = 0.2);
    }

    #[test]
    fn empty_windows_and_disabled_moves_keep_their_step() {
        let mut tuner = StepTuner::new(1.5, 10.0);
        assert_eq!(tuner.observe(&MoveCounter::default()), 1.5);
        let mut off = StepTuner::new(0.0, 10.0);
        assert_eq!(off.observe(&window(0.0, |_| 1.0)), 0.0);
        assert_eq!(off.best(), 0.0);
    }

    /// Settled once the factor is spent, or once held at the bound.
    #[test]
    fn settles_after_narrowing_or_at_the_bound() {
        let mut tuner = StepTuner::new(D0 / 4.0, 100.0);
        let mut step = D0 / 4.0;
        let mut windows = 0;
        while !tuner.settled() && windows < 50 {
            step = tuner.observe(&window(step, |d| (-d / D0).exp()));
            windows += 1;
        }
        assert!(tuner.settled() && windows < 30, "{windows} windows");

        let mut bounded = StepTuner::new(1.0, 3.0);
        let mut step = 1.0;
        for _ in 0..4 {
            assert!(!bounded.settled());
            step = bounded.observe(&window(step, |_| 1.0));
        }
        assert!(bounded.settled());
        assert!(StepTuner::new(0.0, 3.0).settled());
    }

    /// Once settled, noise must not restart the search.
    #[test]
    fn settled_stays_settled() {
        let mut tuner = StepTuner::new(1.0, 3.0);
        let mut step = 1.0;
        while !tuner.settled() {
            step = tuner.observe(&window(step, |_| 1.0));
        }
        let mut worse = window(step, |_| 1.0);
        worse.sum_square_displacement *= 0.9;
        assert_eq!(tuner.observe(&worse), 3.0);
        assert!(tuner.settled());
    }

    /// Everything rejected above 0.5: the search must head down from a
    /// too-large start, not climb on equal zeros.
    #[test]
    fn rejected_moves_turn_the_search_down() {
        let best = tune(2.0, 10.0, |d| if d < 0.5 { 1.0 } else { 0.0 }, 30);
        assert!((0.25..0.5).contains(&best), "{best}");
    }
}

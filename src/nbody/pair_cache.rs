//! Pair energies of a sampler's current poses, so a move evaluates only its
//! trial pose: half the table lookups.

use icotable::lookup::Interpolated;

/// Per pair (kT, row-major n × n): the free energy, whose changes decide
/// acceptance, and the mean energy, whose total is reported as ⟨U⟩. Also the
/// trial pose's row of each. f32 halves the memory (32 MB per copy at
/// N = 2000); the table is f32 anyway.
pub(super) struct PairCache {
    n: usize,
    free: Vec<f32>,
    mean: Vec<f32>,
    trial_free: Vec<f32>,
    trial_mean: Vec<f32>,
    /// Σ_{i<j} mean energy, kept by `commit`, so reading it costs nothing.
    mean_total: f64,
}

impl PairCache {
    /// Filled from `pair(i, j)` for every j < i.
    pub fn new(n: usize, mut pair: impl FnMut(usize, usize) -> Interpolated) -> Self {
        let mut cache = Self {
            n,
            free: vec![0.0; n * n],
            mean: vec![0.0; n * n],
            trial_free: vec![0.0; n],
            trial_mean: vec![0.0; n],
            mean_total: 0.0,
        };
        for i in 0..n {
            for j in 0..i {
                let u = pair(i, j);
                let (free, mean) = (u.energy as f32, u.mean_energy as f32);
                cache.free[i * n + j] = free;
                cache.free[j * n + i] = free;
                cache.mean[i * n + j] = mean;
                cache.mean[j * n + i] = mean;
                cache.mean_total += f64::from(mean);
            }
        }
        cache
    }

    /// Records the pair energies for body i's trial pose; returns the change
    /// in free energy, for acceptance. Rounded as stored, so a move and its
    /// reverse cancel exactly.
    pub fn stage(&mut self, i: usize, j: usize, u: Interpolated) -> f64 {
        let free = u.energy as f32;
        self.trial_free[j] = free;
        self.trial_mean[j] = u.mean_energy as f32;
        f64::from(free) - f64::from(self.free[i * self.n + j])
    }

    /// Body i's staged row becomes current.
    pub fn commit(&mut self, i: usize) {
        let n = self.n;
        // Never staged for i itself, but left over from other moves.
        self.trial_free[i] = 0.0;
        self.trial_mean[i] = 0.0;
        let row = i * n..(i + 1) * n;
        let change: f64 = self
            .trial_mean
            .iter()
            .zip(&self.mean[row.clone()])
            .map(|(&new, &old)| f64::from(new) - f64::from(old))
            .sum();
        self.mean_total += change;
        self.free[row.clone()].copy_from_slice(&self.trial_free);
        self.mean[row].copy_from_slice(&self.trial_mean);
        for j in 0..n {
            self.free[j * n + i] = self.trial_free[j];
            self.mean[j * n + i] = self.trial_mean[j];
        }
    }

    /// Σ_{i<j} mean energy (kT): the total to report as ⟨U⟩.
    pub fn mean_total(&self) -> f64 {
        self.mean_total
    }
}

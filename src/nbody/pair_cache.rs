//! Pair energies of a sampler's current poses, so a move evaluates only its
//! trial pose: half the table lookups.

/// u_ij (kT) row-major n × n, the trial pose's row, and their total. f32
/// halves the memory (16 MB per copy at N = 2000); the table is f32 anyway.
pub(super) struct PairCache {
    n: usize,
    u: Vec<f32>,
    trial: Vec<f32>,
    /// Σ_{i<j} u_ij, kept by `commit`, so reading it costs nothing.
    total: f64,
}

impl PairCache {
    /// Filled from `pair(i, j)` for every j < i.
    pub fn new(n: usize, mut pair: impl FnMut(usize, usize) -> f64) -> Self {
        let mut cache = Self {
            n,
            u: vec![0.0; n * n],
            trial: vec![0.0; n],
            total: 0.0,
        };
        for i in 0..n {
            for j in 0..i {
                let u = pair(i, j) as f32;
                cache.u[i * n + j] = u;
                cache.u[j * n + i] = u;
                cache.total += f64::from(u);
            }
        }
        cache
    }

    /// Records u_ij for body i's trial pose; returns the change from the
    /// current u_ij. Rounded as stored, so a move and its reverse cancel
    /// exactly.
    pub fn stage(&mut self, i: usize, j: usize, u: f64) -> f64 {
        let u = u as f32;
        self.trial[j] = u;
        f64::from(u) - f64::from(self.u[i * self.n + j])
    }

    /// Body i's staged row becomes current; `du` is the sum of its `stage`
    /// changes.
    pub fn commit(&mut self, i: usize, du: f64) {
        let n = self.n;
        // Never staged for i itself, but left over from other moves.
        self.trial[i] = 0.0;
        self.u[i * n..(i + 1) * n].copy_from_slice(&self.trial);
        for (j, &u) in self.trial.iter().enumerate() {
            self.u[j * n + i] = u;
        }
        self.total += du;
    }

    pub fn total(&self) -> f64 {
        self.total
    }
}

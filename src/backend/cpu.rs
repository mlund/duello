// Copyright 2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! CPU backend using splined pair potentials.

use super::{EnergyBackend, PoseParams};
use crate::{energy::PairMatrix, icoscan::orient_structures, structure::Structure};

/// CPU backend for energy calculations using splined pair potentials.
///
/// Uses the existing `PairMatrix` implementation with rayon parallelization
/// happening at the caller level (in `do_icoscan`).
pub struct CpuBackend {
    /// Reference structure for molecule A (centered at origin)
    ref_a: Structure,
    /// Reference structure for molecule B (centered at origin)
    ref_b: Structure,
    /// Pair potential matrix with splined interpolation
    pair_matrix: PairMatrix,
}

impl CpuBackend {
    /// Create a new CPU backend.
    ///
    /// # Arguments
    /// * `ref_a` - Reference structure for molecule A (should be centered at origin)
    /// * `ref_b` - Reference structure for molecule B (should be centered at origin)
    /// * `pair_matrix` - Pair potential matrix for computing energies
    pub fn new(ref_a: Structure, ref_b: Structure, pair_matrix: PairMatrix) -> Self {
        Self {
            ref_a,
            ref_b,
            pair_matrix,
        }
    }
}

impl EnergyBackend for CpuBackend {
    fn compute_energy(&self, pose: &PoseParams) -> f64 {
        let (oriented_a, oriented_b) = orient_structures(
            pose.r,
            pose.omega,
            pose.vertex_i,
            pose.vertex_j,
            &self.ref_a,
            &self.ref_b,
        );
        self.pair_matrix.sum_energy(&oriented_a, &oriented_b)
    }

    fn ref_a(&self) -> &Structure {
        &self.ref_a
    }

    fn ref_b(&self) -> &Structure {
        &self.ref_b
    }
}

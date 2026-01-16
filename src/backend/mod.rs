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

//! Backend abstraction for energy calculations.
//!
//! Provides a trait for computing pairwise energies between molecular poses,
//! with implementations for CPU (using splined potentials) and GPU (using wgpu).

mod cpu;
mod gpu;
mod simd;

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;
pub use simd::SimdBackend;

use crate::{structure::Structure, Vector3};

/// Parameters for a single pose in the 6D configurational space.
#[derive(Clone, Copy, Debug)]
pub struct PoseParams {
    /// Mass center separation distance
    pub r: f64,
    /// Dihedral angle around the separation axis
    pub omega: f64,
    /// Direction vector for molecule A (will be normalized)
    pub vertex_i: Vector3,
    /// Direction vector for molecule B (will be normalized)
    pub vertex_j: Vector3,
}

/// Backend trait for computing pairwise energies between molecular poses.
///
/// Implementations can use different computational strategies:
/// - `CpuBackend`: Uses splined potentials with rayon parallelization
/// - `GpuBackend`: Uses wgpu compute shaders
pub trait EnergyBackend: Send + Sync {
    /// Compute energy for a single pose.
    ///
    /// Returns the interaction energy in kJ/mol between the two molecules
    /// when molecule B is positioned at the given pose relative to molecule A.
    fn compute_energy(&self, pose: &PoseParams) -> f64;

    /// Compute energies for a batch of poses.
    ///
    /// Default implementation calls `compute_energy` for each pose.
    /// GPU backends can override this to batch GPU dispatches.
    fn compute_energies(&self, poses: &[PoseParams]) -> Vec<f64> {
        poses.iter().map(|p| self.compute_energy(p)).collect()
    }

    /// Returns true if this backend prefers batched processing.
    ///
    /// GPU backends return true because they benefit from processing many
    /// poses at once. CPU backends return false because they use rayon
    /// for parallelization at a higher level.
    fn prefers_batch(&self) -> bool {
        false
    }

    /// Get reference to molecule A structure (at origin).
    fn ref_a(&self) -> &Structure;

    /// Get reference to molecule B structure (at origin).
    fn ref_b(&self) -> &Structure;
}

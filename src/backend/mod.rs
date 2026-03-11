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
//! with implementations for CPU, GPU (using wgpu), and SIMD (using AVX2).

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
/// - `CpuBackend`: Uses pair potentials with rayon parallelization
/// - `GpuBackend`: Uses wgpu compute shaders
/// - `SimdBackend`: Uses SIMD vectorization (AVX2 on x86_64, NEON on aarch64)
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

#[cfg(test)]
mod tests {
    use super::simd::orient_position_f32;
    use super::*;
    use crate::icoscan::orient_structures;
    use crate::structure::Structure;

    /// Verify that the f32/glam orientation (SIMD) agrees with the f64/nalgebra
    /// orientation (CPU) to within f32 precision. Both paths must produce the
    /// same rotation sequence to keep energy results consistent across backends.
    #[test]
    fn test_simd_cpu_orientation_agreement() {
        let positions = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(0.0, 0.0, 3.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];
        let ref_b = Structure {
            pos: positions,
            masses: vec![1.0; 4],
            charges: vec![0.0; 4],
            radii: vec![1.0; 4],
            atom_ids: vec![0; 4],
        };
        let ref_a = Structure {
            pos: vec![Vector3::zeros()],
            masses: vec![1.0],
            charges: vec![0.0],
            radii: vec![1.0],
            atom_ids: vec![0],
        };

        // Test several representative poses
        let poses = [
            PoseParams {
                r: 10.0,
                omega: 0.5,
                vertex_i: Vector3::new(1.0, 0.0, 0.0),
                vertex_j: Vector3::new(0.0, 1.0, 0.0),
            },
            PoseParams {
                r: 25.0,
                omega: 2.1,
                vertex_i: Vector3::new(0.3, 0.7, 0.5),
                vertex_j: Vector3::new(-0.5, 0.2, 0.8),
            },
            PoseParams {
                r: 5.0,
                omega: 0.0,
                vertex_i: Vector3::new(0.0, 0.0, 1.0),
                vertex_j: Vector3::new(0.0, 0.0, -1.0),
            },
        ];

        for pose in &poses {
            let (_, oriented_b) = orient_structures(
                pose.r,
                pose.omega,
                &pose.vertex_i,
                &pose.vertex_j,
                &ref_a,
                &ref_b,
            );

            for (i, ref_pos) in ref_b.pos.iter().enumerate() {
                let glam_pos =
                    glam::Vec3::new(ref_pos.x as f32, ref_pos.y as f32, ref_pos.z as f32);
                let simd_result = orient_position_f32(glam_pos, pose);
                let cpu_result = &oriented_b.pos[i];

                let tol = 1e-3; // f32 precision with accumulated rotation error
                assert!(
                    (simd_result.x as f64 - cpu_result.x).abs() < tol
                        && (simd_result.y as f64 - cpu_result.y).abs() < tol
                        && (simd_result.z as f64 - cpu_result.z).abs() < tol,
                    "Orientation mismatch at atom {i} for pose r={}, omega={}: \
                     SIMD=({:.4}, {:.4}, {:.4}), CPU=({:.6}, {:.6}, {:.6})",
                    pose.r,
                    pose.omega,
                    simd_result.x,
                    simd_result.y,
                    simd_result.z,
                    cpu_result.x,
                    cpu_result.y,
                    cpu_result.z,
                );
            }
        }
    }
}

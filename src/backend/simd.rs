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

//! SIMD backend using the `wide` crate for AVX2 vectorization.
//!
//! This backend vectorizes across atom pairs within a single pose,
//! processing 8 atom pairs simultaneously using f32x8 (AVX2).

use super::{EnergyBackend, PoseParams};
use crate::structure::Structure;
use faunus::energy::NonbondedMatrixSplined;
use wide::f32x8;

/// Spline parameters for a single pair type.
#[derive(Clone, Copy, Debug)]
struct SimdSplineParams {
    r_min: f32,
    r_max: f32,
    n_coeffs: u32,
    coeff_offset: u32,
}

/// Extracted spline data for SIMD evaluation.
struct SimdSplineData {
    /// Flattened spline coefficients: [pair0_coeff0, pair0_coeff1, ..., pair1_coeff0, ...]
    /// Each coefficient is [a0, a1, a2, a3]
    coefficients: Vec<[f32; 4]>,
    /// Parameters for each pair type (n_types × n_types)
    params: Vec<SimdSplineParams>,
    /// Number of atom types
    n_types: usize,
}

impl SimdSplineData {
    /// Extract spline data from NonbondedMatrixSplined.
    fn from_splined_matrix(matrix: &NonbondedMatrixSplined) -> Self {
        use interatomic::twobody::GridType;

        let potentials = matrix.get_potentials();
        let shape = potentials.shape();
        let n_types = shape[0];

        let mut coefficients = Vec::new();
        let mut params = Vec::with_capacity(n_types * n_types);

        for i in 0..n_types {
            for j in 0..n_types {
                let spline = &potentials[(i, j)];
                let stats = spline.stats();
                let coeffs = spline.coefficients();

                let coeff_offset = coefficients.len() as u32;

                // Extract energy coefficients (u[0..4]) for each interval
                for c in coeffs {
                    coefficients.push([c.u[0] as f32, c.u[1] as f32, c.u[2] as f32, c.u[3] as f32]);
                }

                // Verify power-law grid type
                match stats.grid_type {
                    GridType::PowerLaw2 => {}
                    GridType::PowerLaw(p) if (p - 2.0).abs() < 1e-6 => {}
                    _ => panic!("SIMD backend requires PowerLaw2 or PowerLaw(2.0) grid type"),
                };

                params.push(SimdSplineParams {
                    r_min: stats.r_min as f32,
                    r_max: stats.r_max as f32,
                    n_coeffs: coeffs.len() as u32,
                    coeff_offset,
                });
            }
        }

        Self {
            coefficients,
            params,
            n_types,
        }
    }
}

/// SIMD backend for energy calculations using AVX2 vectorization.
pub struct SimdBackend {
    /// Reference structure for molecule A (centered at origin)
    ref_a: Structure,
    /// Reference structure for molecule B (centered at origin)
    ref_b: Structure,
    /// Reference positions for A (SoA layout)
    pos_a_x: Vec<f32>,
    pos_a_y: Vec<f32>,
    pos_a_z: Vec<f32>,
    /// Reference positions for B (SoA layout)
    pos_b_x: Vec<f32>,
    pos_b_y: Vec<f32>,
    pos_b_z: Vec<f32>,
    /// Atom type IDs for A and B
    atom_ids_a: Vec<u32>,
    atom_ids_b: Vec<u32>,
    /// Spline data
    spline_data: SimdSplineData,
}

impl SimdBackend {
    /// Create a new SIMD backend.
    pub fn new(
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &NonbondedMatrixSplined,
    ) -> Self {
        // Convert positions to SoA layout
        let pos_a_x: Vec<f32> = ref_a.pos.iter().map(|p| p.x as f32).collect();
        let pos_a_y: Vec<f32> = ref_a.pos.iter().map(|p| p.y as f32).collect();
        let pos_a_z: Vec<f32> = ref_a.pos.iter().map(|p| p.z as f32).collect();

        let pos_b_x: Vec<f32> = ref_b.pos.iter().map(|p| p.x as f32).collect();
        let pos_b_y: Vec<f32> = ref_b.pos.iter().map(|p| p.y as f32).collect();
        let pos_b_z: Vec<f32> = ref_b.pos.iter().map(|p| p.z as f32).collect();

        let atom_ids_a: Vec<u32> = ref_a.atom_ids.iter().map(|&id| id as u32).collect();
        let atom_ids_b: Vec<u32> = ref_b.atom_ids.iter().map(|&id| id as u32).collect();

        let spline_data = SimdSplineData::from_splined_matrix(splined_matrix);

        log::info!(
            "SIMD backend initialized: {} spline coefficients, {} atom types, {} + {} atoms",
            spline_data.coefficients.len(),
            spline_data.n_types,
            ref_a.pos.len(),
            ref_b.pos.len()
        );

        Self {
            ref_a,
            ref_b,
            pos_a_x,
            pos_a_y,
            pos_a_z,
            pos_b_x,
            pos_b_y,
            pos_b_z,
            atom_ids_a,
            atom_ids_b,
            spline_data,
        }
    }

    /// Evaluate spline energy for a single (r_sq, pair_idx) - scalar fallback.
    #[inline]
    fn eval_spline_scalar(&self, r_sq: f32, pair_idx: u32) -> f32 {
        let params = &self.spline_data.params[pair_idx as usize];
        let r = r_sq.sqrt();

        // Check cutoff
        if r >= params.r_max {
            return 0.0;
        }

        // Clamp to minimum
        let r_clamped = r.max(params.r_min);

        // Inverse power-law mapping with p=2: x = sqrt((r - r_min) / (r_max - r_min))
        let r_range = params.r_max - params.r_min;
        let x = ((r_clamped - params.r_min) / r_range).sqrt();

        // Grid index and fraction: t = x * (n - 1)
        let t = x * (params.n_coeffs - 1) as f32;
        let idx = (t as u32).min(params.n_coeffs - 2);
        let frac = t - idx as f32;

        // Fetch coefficients
        let c = self.spline_data.coefficients[(params.coeff_offset + idx) as usize];

        // Horner's method: a0 + frac*(a1 + frac*(a2 + frac*a3))
        c[0] + frac * (c[1] + frac * (c[2] + frac * c[3]))
    }

    /// Compute energy for a single pose using SIMD.
    fn compute_energy_simd(&self, pose: &PoseParams) -> f32 {
        let n_a = self.pos_a_x.len();
        let n_b = self.pos_b_x.len();
        let n_types = self.spline_data.n_types as u32;

        // Build transformation quaternions
        let zaxis = glam::Vec3::new(0.0005, 0.0005, 1.0).normalize();
        let neg_zaxis = -zaxis;

        let vertex_i = glam::Vec3::new(
            pose.vertex_i.x as f32,
            pose.vertex_i.y as f32,
            pose.vertex_i.z as f32,
        )
        .normalize();
        let vertex_j = glam::Vec3::new(
            pose.vertex_j.x as f32,
            pose.vertex_j.y as f32,
            pose.vertex_j.z as f32,
        )
        .normalize();

        // q1: rotate vertex_j to -z axis
        let q1 = glam::Quat::from_rotation_arc(vertex_j, neg_zaxis);
        // q2: rotate around z by omega
        let q2 = glam::Quat::from_axis_angle(zaxis, pose.omega as f32);
        // q3: rotate z axis to vertex_i
        let q3 = glam::Quat::from_rotation_arc(zaxis, vertex_i);

        // Combined rotation q12 = q1 * q2
        let q12 = q1 * q2;

        // Translation vector
        let r_vec = glam::Vec3::new(0.0, 0.0, pose.r as f32);

        // Transform all B positions
        let mut trans_b_x = Vec::with_capacity(n_b);
        let mut trans_b_y = Vec::with_capacity(n_b);
        let mut trans_b_z = Vec::with_capacity(n_b);

        for j in 0..n_b {
            let pos_b = glam::Vec3::new(self.pos_b_x[j], self.pos_b_y[j], self.pos_b_z[j]);
            // Transform: q3 * (q12 * pos_b + r_vec)
            let rotated = q12 * pos_b;
            let translated = rotated + r_vec;
            let final_pos = q3 * translated;

            trans_b_x.push(final_pos.x);
            trans_b_y.push(final_pos.y);
            trans_b_z.push(final_pos.z);
        }

        // Compute pairwise energies using SIMD
        let mut total_energy = f32x8::ZERO;
        let simd_width = 8;

        for i in 0..n_a {
            let pos_a_x = f32x8::splat(self.pos_a_x[i]);
            let pos_a_y = f32x8::splat(self.pos_a_y[i]);
            let pos_a_z = f32x8::splat(self.pos_a_z[i]);
            let id_a = self.atom_ids_a[i];

            // Process 8 B atoms at a time
            let mut j = 0;
            while j + simd_width <= n_b {
                // Load 8 B positions
                let bx = f32x8::new([
                    trans_b_x[j],
                    trans_b_x[j + 1],
                    trans_b_x[j + 2],
                    trans_b_x[j + 3],
                    trans_b_x[j + 4],
                    trans_b_x[j + 5],
                    trans_b_x[j + 6],
                    trans_b_x[j + 7],
                ]);
                let by = f32x8::new([
                    trans_b_y[j],
                    trans_b_y[j + 1],
                    trans_b_y[j + 2],
                    trans_b_y[j + 3],
                    trans_b_y[j + 4],
                    trans_b_y[j + 5],
                    trans_b_y[j + 6],
                    trans_b_y[j + 7],
                ]);
                let bz = f32x8::new([
                    trans_b_z[j],
                    trans_b_z[j + 1],
                    trans_b_z[j + 2],
                    trans_b_z[j + 3],
                    trans_b_z[j + 4],
                    trans_b_z[j + 5],
                    trans_b_z[j + 6],
                    trans_b_z[j + 7],
                ]);

                // Compute squared distances
                let dx = pos_a_x - bx;
                let dy = pos_a_y - by;
                let dz = pos_a_z - bz;
                let r_sq = dx * dx + dy * dy + dz * dz;

                // Evaluate splines for each lane (scalar gather due to heterogeneous splines)
                let r_sq_arr: [f32; 8] = r_sq.into();
                let mut energies = [0.0f32; 8];
                for lane in 0..8 {
                    let id_b = self.atom_ids_b[j + lane];
                    let pair_idx = id_a * n_types + id_b;
                    energies[lane] = self.eval_spline_scalar(r_sq_arr[lane], pair_idx);
                }

                total_energy += f32x8::new(energies);
                j += simd_width;
            }

            // Handle remainder atoms (scalar)
            while j < n_b {
                let dx = self.pos_a_x[i] - trans_b_x[j];
                let dy = self.pos_a_y[i] - trans_b_y[j];
                let dz = self.pos_a_z[i] - trans_b_z[j];
                let r_sq = dx * dx + dy * dy + dz * dz;

                let id_b = self.atom_ids_b[j];
                let pair_idx = id_a * n_types + id_b;
                let energy = self.eval_spline_scalar(r_sq, pair_idx);

                // Add to total via scalar accumulator (will be combined at end)
                let mut arr: [f32; 8] = total_energy.into();
                arr[0] += energy;
                total_energy = f32x8::new(arr);
                j += 1;
            }
        }

        // Horizontal sum
        let arr: [f32; 8] = total_energy.into();
        arr.iter().sum()
    }
}

impl EnergyBackend for SimdBackend {
    fn compute_energy(&self, pose: &PoseParams) -> f64 {
        self.compute_energy_simd(pose) as f64
    }

    fn ref_a(&self) -> &Structure {
        &self.ref_a
    }

    fn ref_b(&self) -> &Structure {
        &self.ref_b
    }
}

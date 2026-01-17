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

//! SIMD backend using the `wide` crate for vectorization.
//!
//! This backend vectorizes across atom pairs within a single pose.
//! The SIMD width is selected at compile time based on target architecture:
//! - x86_64: f32x8 (AVX2, 256-bit, 8 lanes)
//! - aarch64: f32x4 (NEON, 128-bit, 4 lanes)
//!
//! Pairs are grouped by atom type combination at initialization time,
//! allowing fully vectorized spline evaluation without scalar gather.

use super::{EnergyBackend, PoseParams};
use crate::structure::Structure;
use faunus::energy::NonbondedMatrixSplined;
use std::collections::HashMap;
use wide::CmpLt;

// Compile-time SIMD configuration based on target architecture
#[cfg(target_arch = "aarch64")]
mod simd_config {
    pub use wide::f32x4 as SimdFloat;
    pub const LANES: usize = 4;
    pub type SimdArray = [f32; 4];

    #[inline]
    pub fn simd_new(arr: SimdArray) -> SimdFloat {
        SimdFloat::new(arr)
    }

    #[inline]
    pub fn simd_to_array(v: SimdFloat) -> SimdArray {
        v.into()
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod simd_config {
    pub use wide::f32x8 as SimdFloat;
    pub const LANES: usize = 8;
    pub type SimdArray = [f32; 8];

    #[inline]
    pub fn simd_new(arr: SimdArray) -> SimdFloat {
        SimdFloat::new(arr)
    }

    #[inline]
    pub fn simd_to_array(v: SimdFloat) -> SimdArray {
        v.into()
    }
}

use simd_config::{simd_new, simd_to_array, SimdArray, SimdFloat, LANES};

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

/// A group of atom pairs that share the same pair type (same spline parameters).
struct PairGroup {
    /// Atom indices: (index_in_A, index_in_B)
    pairs: Vec<(u32, u32)>,
    /// The pair type index into spline_data.params
    pair_idx: u32,
}

/// Precomputed pair groups, organized by pair type for efficient SIMD processing.
struct PairGroups {
    groups: Vec<PairGroup>,
}

impl PairGroups {
    /// Build pair groups from atom type IDs.
    /// Groups pairs by their atom type combination for uniform SIMD processing.
    fn new(atom_ids_a: &[u32], atom_ids_b: &[u32], n_types: u32) -> Self {
        let mut groups_map: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();

        for (i, &id_a) in atom_ids_a.iter().enumerate() {
            for (j, &id_b) in atom_ids_b.iter().enumerate() {
                let pair_idx = id_a * n_types + id_b;
                groups_map
                    .entry(pair_idx)
                    .or_default()
                    .push((i as u32, j as u32));
            }
        }

        let groups: Vec<PairGroup> = groups_map
            .into_iter()
            .map(|(pair_idx, pairs)| PairGroup { pairs, pair_idx })
            .collect();

        Self { groups }
    }
}

/// SIMD backend for energy calculations.
///
/// Uses compile-time selected SIMD width: 8 lanes (AVX2) on x86_64,
/// 4 lanes (NEON) on aarch64.
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
    /// Spline data
    spline_data: SimdSplineData,
    /// Precomputed pair groups (pairs grouped by atom type combination)
    pair_groups: PairGroups,
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
        let pair_groups = PairGroups::new(&atom_ids_a, &atom_ids_b, spline_data.n_types as u32);

        let n_unique_pairs: usize = pair_groups.groups.len();
        let total_pairs: usize = pair_groups.groups.iter().map(|g| g.pairs.len()).sum();

        #[cfg(target_arch = "aarch64")]
        let simd_type = "NEON (f32x4)";
        #[cfg(not(target_arch = "aarch64"))]
        let simd_type = "AVX2 (f32x8)";

        log::info!(
            "SIMD backend initialized ({}): {} atoms, {} unique pair types, {} total pairs",
            simd_type,
            ref_a.pos.len() + ref_b.pos.len(),
            n_unique_pairs,
            total_pairs
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
            spline_data,
            pair_groups,
        }
    }

    /// Evaluate spline energy for LANES squared distances with uniform spline parameters.
    /// All lanes use the same spline params, enabling full SIMD vectorization.
    #[inline]
    fn eval_spline_simd(&self, r_sq: SimdFloat, params: &SimdSplineParams) -> SimdFloat {
        let r = r_sq.sqrt();

        // Cutoff mask: r < r_max
        let r_max = SimdFloat::splat(params.r_max);
        let cutoff_mask = r.cmp_lt(r_max);

        // Early exit if all distances are beyond cutoff
        if cutoff_mask.none() {
            return SimdFloat::ZERO;
        }

        // Clamp to minimum
        let r_min = SimdFloat::splat(params.r_min);
        let r_clamped = r.max(r_min);

        // Inverse power-law mapping with p=2: x = sqrt((r - r_min) / (r_max - r_min))
        let r_range = SimdFloat::splat(params.r_max - params.r_min);
        let x = ((r_clamped - r_min) / r_range).sqrt();

        // Grid index and fraction: t = x * (n - 1)
        let n_coeffs_minus_1 = SimdFloat::splat((params.n_coeffs - 1) as f32);
        let t = x * n_coeffs_minus_1;

        // Convert to integer indices and fractions
        let t_arr: SimdArray = simd_to_array(t);
        let max_idx = params.n_coeffs - 2;

        let mut energies: SimdArray = [0.0f32; LANES];
        for lane in 0..LANES {
            let idx = (t_arr[lane] as u32).min(max_idx);
            let frac = t_arr[lane] - idx as f32;

            // Fetch coefficients for this interval
            let c = self.spline_data.coefficients[(params.coeff_offset + idx) as usize];

            // Horner's method: a0 + frac*(a1 + frac*(a2 + frac*a3))
            energies[lane] = c[0] + frac * (c[1] + frac * (c[2] + frac * c[3]));
        }

        // Apply cutoff mask
        cutoff_mask.blend(simd_new(energies), SimdFloat::ZERO)
    }

    /// Evaluate spline energy for a single squared distance (scalar fallback).
    #[inline]
    fn eval_spline_scalar(&self, r_sq: f32, params: &SimdSplineParams) -> f32 {
        let r = r_sq.sqrt();

        if r >= params.r_max {
            return 0.0;
        }

        let r_clamped = r.max(params.r_min);
        let r_range = params.r_max - params.r_min;
        let x = ((r_clamped - params.r_min) / r_range).sqrt();

        let t = x * (params.n_coeffs - 1) as f32;
        let idx = (t as u32).min(params.n_coeffs - 2);
        let frac = t - idx as f32;

        let c = self.spline_data.coefficients[(params.coeff_offset + idx) as usize];
        c[0] + frac * (c[1] + frac * (c[2] + frac * c[3]))
    }

    /// Compute energy for a single pose using SIMD with pair grouping.
    fn compute_energy_simd(&self, pose: &PoseParams) -> f32 {
        let n_b = self.pos_b_x.len();

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
            let rotated = q12 * pos_b;
            let translated = rotated + r_vec;
            let final_pos = q3 * translated;

            trans_b_x.push(final_pos.x);
            trans_b_y.push(final_pos.y);
            trans_b_z.push(final_pos.z);
        }

        // Process each pair group with uniform spline parameters
        let mut total_energy = 0.0f32;

        for group in &self.pair_groups.groups {
            let params = &self.spline_data.params[group.pair_idx as usize];
            let pairs = &group.pairs;
            let n_pairs = pairs.len();

            // Process LANES pairs at a time with full SIMD
            let mut p = 0;
            while p + LANES <= n_pairs {
                // Gather positions for LANES pairs
                let mut ax: SimdArray = [0.0f32; LANES];
                let mut ay: SimdArray = [0.0f32; LANES];
                let mut az: SimdArray = [0.0f32; LANES];
                let mut bx: SimdArray = [0.0f32; LANES];
                let mut by: SimdArray = [0.0f32; LANES];
                let mut bz: SimdArray = [0.0f32; LANES];

                for lane in 0..LANES {
                    let (i, j) = pairs[p + lane];
                    ax[lane] = self.pos_a_x[i as usize];
                    ay[lane] = self.pos_a_y[i as usize];
                    az[lane] = self.pos_a_z[i as usize];
                    bx[lane] = trans_b_x[j as usize];
                    by[lane] = trans_b_y[j as usize];
                    bz[lane] = trans_b_z[j as usize];
                }

                // Compute squared distances
                let dx = simd_new(ax) - simd_new(bx);
                let dy = simd_new(ay) - simd_new(by);
                let dz = simd_new(az) - simd_new(bz);
                let r_sq = dx * dx + dy * dy + dz * dz;

                // Evaluate splines with uniform parameters (no gather!)
                let energies = self.eval_spline_simd(r_sq, params);

                // Horizontal sum
                let arr: SimdArray = simd_to_array(energies);
                total_energy += arr.iter().sum::<f32>();

                p += LANES;
            }

            // Handle remainder pairs (scalar)
            while p < n_pairs {
                let (i, j) = pairs[p];
                let dx = self.pos_a_x[i as usize] - trans_b_x[j as usize];
                let dy = self.pos_a_y[i as usize] - trans_b_y[j as usize];
                let dz = self.pos_a_z[i as usize] - trans_b_z[j as usize];
                let r_sq = dx * dx + dy * dy + dz * dz;

                total_energy += self.eval_spline_scalar(r_sq, params);
                p += 1;
            }
        }

        total_energy
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

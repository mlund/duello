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
//! allowing efficient SIMD evaluation per pair type.
//!
//! Spline evaluation uses `interatomic::twobody::SplineTableSimdF32`.

use super::{EnergyBackend, PoseParams};
use crate::structure::Structure;
use faunus::energy::NonbondedMatrixSplined;
use interatomic::twobody::{simd_f32_from_array, simd_f32_to_array, GridType, SimdArrayF32, SimdF32, SplineTableSimdF32, LANES_F32};
use std::collections::HashMap;

/// A group of atom pairs that share the same pair type (same spline parameters).
struct PairGroup {
    /// Atom indices: (index_in_A, index_in_B)
    pairs: Vec<(u32, u32)>,
    /// The pair type index into spline_tables
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
/// 4 lanes (NEON) on aarch64. Spline evaluation delegated to interatomic.
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
    /// Spline tables from interatomic (one per pair type, flattened n_types × n_types)
    spline_tables: Vec<SplineTableSimdF32>,
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

        // Convert splined potentials to f32 SIMD tables
        let potentials = splined_matrix.get_potentials();
        let shape = potentials.shape();
        let n_types = shape[0];

        let mut spline_tables = Vec::with_capacity(n_types * n_types);
        for i in 0..n_types {
            for j in 0..n_types {
                let spline = &potentials[(i, j)];

                // Verify power-law grid type (required for efficient SIMD)
                let stats = spline.stats();
                match stats.grid_type {
                    GridType::PowerLaw2 => {}
                    GridType::PowerLaw(p) if (p - 2.0).abs() < 1e-6 => {}
                    _ => panic!("SIMD backend requires PowerLaw2 or PowerLaw(2.0) grid type"),
                };

                spline_tables.push(spline.to_simd_f32());
            }
        }

        let pair_groups = PairGroups::new(&atom_ids_a, &atom_ids_b, n_types as u32);

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
            spline_tables,
            pair_groups,
        }
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
            let spline = &self.spline_tables[group.pair_idx as usize];
            let pairs = &group.pairs;
            let n_pairs = pairs.len();

            // Process LANES_F32 pairs at a time with full SIMD
            let mut p = 0;
            while p + LANES_F32 <= n_pairs {
                // Gather positions for LANES_F32 pairs
                let mut ax: SimdArrayF32 = [0.0f32; LANES_F32];
                let mut ay: SimdArrayF32 = [0.0f32; LANES_F32];
                let mut az: SimdArrayF32 = [0.0f32; LANES_F32];
                let mut bx: SimdArrayF32 = [0.0f32; LANES_F32];
                let mut by: SimdArrayF32 = [0.0f32; LANES_F32];
                let mut bz: SimdArrayF32 = [0.0f32; LANES_F32];

                for lane in 0..LANES_F32 {
                    let (i, j) = pairs[p + lane];
                    ax[lane] = self.pos_a_x[i as usize];
                    ay[lane] = self.pos_a_y[i as usize];
                    az[lane] = self.pos_a_z[i as usize];
                    bx[lane] = trans_b_x[j as usize];
                    by[lane] = trans_b_y[j as usize];
                    bz[lane] = trans_b_z[j as usize];
                }

                // Compute squared distances
                let dx = simd_f32_from_array(ax) - simd_f32_from_array(bx);
                let dy = simd_f32_from_array(ay) - simd_f32_from_array(by);
                let dz = simd_f32_from_array(az) - simd_f32_from_array(bz);
                let r_sq: SimdF32 = dx * dx + dy * dy + dz * dz;

                // Evaluate spline using interatomic's optimized PowerLaw2 implementation
                let energies = spline.energy_simd_powerlaw2(r_sq);

                // Horizontal sum
                let arr = simd_f32_to_array(energies);
                total_energy += arr.iter().sum::<f32>();

                p += LANES_F32;
            }

            // Handle remainder pairs (scalar)
            while p < n_pairs {
                let (i, j) = pairs[p];
                let dx = self.pos_a_x[i as usize] - trans_b_x[j as usize];
                let dy = self.pos_a_y[i as usize] - trans_b_y[j as usize];
                let dz = self.pos_a_z[i as usize] - trans_b_z[j as usize];
                let r_sq = dx * dx + dy * dy + dz * dz;

                total_energy += spline.energy(r_sq);
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

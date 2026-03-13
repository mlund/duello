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
//! Spline evaluation uses `SplineTableSimdF32`.

use super::{EnergyBackend, PoseParams};
use crate::energy::{CoulombParams, SplinedPotentials};
use crate::structure::Structure;
use faunus::interatomic::twobody::{GridType, SplineTableSimdF32};
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

/// Apply the same orientation transform as `orient_structures`, but in f32/glam.
///
/// Returns the final position of molecule B's atom after rotation and translation.
/// The sequence is: rotate by q1*q2, translate along z by r, then rotate by q3.
pub(crate) fn orient_position_f32(pos: glam::Vec3, pose: &PoseParams) -> glam::Vec3 {
    let zaxis = glam::Vec3::Z;
    let neg_zaxis = glam::Vec3::NEG_Z;

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

    // glam::from_rotation_arc handles anti-parallel vectors correctly
    let q1 = glam::Quat::from_rotation_arc(vertex_j, neg_zaxis);
    let q2 = glam::Quat::from_axis_angle(zaxis, pose.omega as f32);
    let q3 = glam::Quat::from_rotation_arc(zaxis, vertex_i);
    let q12 = q1 * q2;
    let r_vec = glam::Vec3::new(0.0, 0.0, pose.r as f32);

    let rotated = q12 * pos;
    let translated = rotated + r_vec;
    q3 * translated
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
    /// Coulomb prefactors per pair type (n_types²), kJ/mol·Å
    coulomb_prefactors: Vec<f32>,
    /// Inverse Debye screening length κ (1/Å), 0.0 for unscreened
    kappa: f32,
}

impl SimdBackend {
    /// Create a new SIMD backend.
    pub fn new(
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &SplinedPotentials,
        coulomb: &CoulombParams,
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
        let potentials = splined_matrix.inner().get_potentials();
        let shape = potentials.shape();
        let n_types = shape[0];

        let mut spline_tables = Vec::with_capacity(n_types * n_types);
        for i in 0..n_types {
            for j in 0..n_types {
                let spline = &potentials[(i, j)];

                // Verify supported grid type (PowerLaw2 or InverseRsq for efficient SIMD)
                let stats = spline.stats();
                match stats.grid_type {
                    GridType::PowerLaw2 | GridType::InverseRsq => {}
                    GridType::PowerLaw(p) if (p - 2.0).abs() < 1e-6 => {}
                    _ => panic!(
                        "SIMD backend requires PowerLaw2, InverseRsq, or PowerLaw(2.0) grid type, got {:?}",
                        stats.grid_type
                    ),
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
            coulomb_prefactors: coulomb.prefactors.clone(),
            kappa: coulomb.kappa,
        }
    }

    /// Compute energy for a single pose using SIMD with pair grouping.
    ///
    /// The quaternion orientation sequence here mirrors `orient_structures` in icoscan.rs
    /// but uses f32/glam instead of f64/nalgebra to enable SIMD vectorization.
    /// Both must produce equivalent rotations — see `test_simd_cpu_orientation_agreement`.
    ///
    /// Note: spline and Coulomb energies are accumulated separately (batched SIMD vs scalar),
    /// so f32 summation order differs slightly from the GPU backend. This is expected and
    /// not worth fixing as it would degrade performance or code clarity.
    fn compute_energy_simd(&self, pose: &PoseParams) -> f32 {
        let n_b = self.pos_b_x.len();

        // Transform all B positions using the same rotation sequence as orient_structures
        let mut trans_b_x = Vec::with_capacity(n_b);
        let mut trans_b_y = Vec::with_capacity(n_b);
        let mut trans_b_z = Vec::with_capacity(n_b);

        for j in 0..n_b {
            let pos_b = glam::Vec3::new(self.pos_b_x[j], self.pos_b_y[j], self.pos_b_z[j]);
            let final_pos = orient_position_f32(pos_b, pose);
            trans_b_x.push(final_pos.x);
            trans_b_y.push(final_pos.y);
            trans_b_z.push(final_pos.z);
        }

        // Process each pair group with uniform spline parameters
        let mut total_energy = 0.0f32;

        for group in &self.pair_groups.groups {
            let spline = &self.spline_tables[group.pair_idx as usize];
            let pairs = &group.pairs;

            // Compute all r² values for this group
            let rsq_vec: Vec<f32> = pairs
                .iter()
                .map(|&(i, j)| {
                    let dx = self.pos_a_x[i as usize] - trans_b_x[j as usize];
                    let dy = self.pos_a_y[i as usize] - trans_b_y[j as usize];
                    let dz = self.pos_a_z[i as usize] - trans_b_z[j as usize];
                    dz.mul_add(dz, dx.mul_add(dx, dy * dy))
                })
                .collect();

            total_energy += spline.energy_batch(&rsq_vec);

            // Analytical Coulomb/Yukawa (not splined to avoid ringing from SR cutoff)
            let prefactor = self.coulomb_prefactors[group.pair_idx as usize];
            if prefactor.abs() > 1e-12 { // skip uncharged pair types
                let kappa = self.kappa;
                for &rsq in &rsq_vec {
                    let r = rsq.sqrt();
                    total_energy += prefactor * (-kappa * r).exp() / r;
                }
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

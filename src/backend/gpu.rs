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

//! GPU backend using wgpu compute shaders.

use super::gpu_context::{Batch, Binding, GpuContext, Limits};
use super::{EnergyBackend, PoseParams};
use crate::energy::{CoulombParams, SplinedPotentials};
use crate::structure::Structure;
use bytemuck::{Pod, Zeroable};
use faunus::interatomic::gpu::{GpuGridType, GpuSplineData, InverseRsq, PowerLaw2};
use faunus::interatomic::twobody::GridType;

/// GPU-compatible pose parameters.
/// Note: WGSL struct alignment requires vec4 to be 16-byte aligned,
/// so we add padding after omega.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPoseParams {
    pub(crate) r: f32,
    pub(crate) omega: f32,
    pub(crate) _pad: [f32; 2],     // Padding to align vertex_i to 16 bytes
    pub(crate) vertex_i: [f32; 4], // vec4<f32>
    pub(crate) vertex_j: [f32; 4], // vec4<f32>
}

impl From<&PoseParams> for GpuPoseParams {
    fn from(p: &PoseParams) -> Self {
        let v = |v: &crate::Vector3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
        Self {
            r: p.r as f32,
            omega: p.omega as f32,
            _pad: [0.0, 0.0],
            vertex_i: v(&p.vertex_i),
            vertex_j: v(&p.vertex_j),
        }
    }
}

/// Poses per dispatch; bounds each batch's pose and output buffers.
const MAX_BATCH: usize = 100_000;

/// A one-thread-per-pose pipeline. Owns batching, the pose and output buffers
/// and readback, so a backend supplies only its bind group.
pub(crate) struct PoseKernel {
    pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

impl PoseKernel {
    pub fn new(gpu: &GpuContext, label: &str, source: &str, bindings: &[Binding]) -> Self {
        let (pipeline, layout) = gpu.pipeline(label, source, bindings);
        Self { pipeline, layout }
    }

    /// `bind(poses, energies, n_poses)` builds one batch's bind group.
    pub fn energies<B>(&self, gpu: &GpuContext, poses: &[PoseParams], bind: B) -> Vec<f64>
    where
        B: Fn(&wgpu::Buffer, &wgpu::Buffer, u32) -> wgpu::BindGroup,
    {
        let batches = poses.chunks(MAX_BATCH).map(|c| Self::batch(gpu, c, &bind));
        widen(gpu.dispatch_read(&self.pipeline, batches))
    }

    /// As [`Self::energies`], for WASM.
    pub async fn energies_async<B>(
        &self,
        gpu: &GpuContext,
        poses: &[PoseParams],
        bind: B,
    ) -> Vec<f64>
    where
        B: Fn(&wgpu::Buffer, &wgpu::Buffer, u32) -> wgpu::BindGroup,
    {
        let batches = poses.chunks(MAX_BATCH).map(|c| Self::batch(gpu, c, &bind));
        widen(gpu.dispatch_read_async(&self.pipeline, batches).await)
    }

    /// Built lazily as the dispatcher pulls it, so few batches' buffers live at once.
    fn batch<B>(gpu: &GpuContext, chunk: &[PoseParams], bind: &B) -> Batch
    where
        B: Fn(&wgpu::Buffer, &wgpu::Buffer, u32) -> wgpu::BindGroup,
    {
        let gpu_poses: Vec<GpuPoseParams> = chunk.iter().map(GpuPoseParams::from).collect();
        let pose_buffer = gpu.storage("Poses", bytemuck::cast_slice(&gpu_poses));
        let output = gpu.output("Energies", chunk.len());
        Batch {
            group: bind(&pose_buffer, &output, chunk.len() as u32),
            workgroups: workgroups(chunk.len()),
            output,
        }
    }
}

fn widen(energies: Vec<f32>) -> Vec<f64> {
    energies.into_iter().map(f64::from).collect()
}

/// One thread per pose, 64 per workgroup as the kernels declare.
fn workgroups(n_poses: usize) -> [u32; 2] {
    [(n_poses as u32).div_ceil(64), 1]
}

/// Spline tables plus the WGSL that indexes them, for whichever grid type
/// the potentials were built on.
pub(crate) struct SplineTables {
    pub coeffs: Vec<u8>,
    pub params: Vec<u8>,
    pub wgsl: &'static str,
}

impl SplineTables {
    pub fn new(splines: &SplinedPotentials) -> anyhow::Result<Self> {
        let grid_type = splines
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty spline matrix"))?
            .grid_type();
        match grid_type {
            GridType::PowerLaw2 => Ok(Self::typed::<PowerLaw2>(splines)),
            GridType::InverseRsq => Ok(Self::typed::<InverseRsq>(splines)),
            other => {
                anyhow::bail!("GPU backend requires PowerLaw2 or InverseRsq grid, got {other:?}")
            }
        }
    }

    fn typed<G: GpuGridType>(splines: &SplinedPotentials) -> Self {
        let data = GpuSplineData::<G>::from_potentials(splines.iter());
        Self {
            coeffs: data.coefficients_as_bytes().to_vec(),
            params: data.params_as_bytes().to_vec(),
            wgsl: G::SPLINE_WGSL,
        }
    }
}

/// A position as the `vec4<f32>` WGSL arrays expect, with `w` in the fourth lane.
pub(crate) fn pack_bead(p: &crate::Vector3, w: f32) -> [f32; 4] {
    [p.x as f32, p.y as f32, p.z as f32, w]
}

/// Positions with `w` unused.
pub(crate) fn pack_positions(s: &Structure) -> Vec<[f32; 4]> {
    s.pos.iter().map(|p| pack_bead(p, 0.0)).collect()
}

/// GPU-compatible uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuUniforms {
    n_atoms_a: u32,
    n_atoms_b: u32,
    n_atom_types: u32,
    n_poses: u32,
    kappa: f32,
    _pad: [u32; 3],
}

/// GPU backend for energy calculations using wgpu compute shaders.
pub struct GpuBackend {
    gpu: GpuContext,
    kernel: PoseKernel,
    /// Bindings 0–4, uploaded once: spline coeffs, spline params, A, B, atom ids.
    constant: [wgpu::Buffer; 5],
    coulomb_prefactors: wgpu::Buffer,
    ref_a: Structure,
    ref_b: Structure,
    uniforms: GpuUniforms,
}

// SAFETY: wasm32 is single-threaded; see `GpuContext`.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for GpuBackend {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for GpuBackend {}

/// Binding order of `energy.wgsl`.
const BINDINGS: [Binding; 9] = [
    Binding::Read,      // spline coefficients
    Binding::Read,      // spline params
    Binding::Read,      // reference positions A
    Binding::Read,      // reference positions B
    Binding::Read,      // atom ids, A then B
    Binding::Read,      // poses
    Binding::ReadWrite, // energies
    Binding::Uniform,   // uniforms
    Binding::Read,      // Coulomb prefactors
];

impl GpuBackend {
    /// Check if a GPU is available for compute (not available on WASM — use async path).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_available() -> bool {
        GpuContext::is_available()
    }

    /// Create a new GPU backend, auto-detecting grid type from the splined matrix (not available on WASM).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &SplinedPotentials,
        coulomb: &CoulombParams,
    ) -> anyhow::Result<Self> {
        let gpu = GpuContext::new(Limits::Default)?;
        Self::setup(gpu, ref_a, ref_b, splined_matrix, coulomb)
    }

    /// Create a new GPU backend asynchronously (required for WASM).
    pub async fn new_async(
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &SplinedPotentials,
        coulomb: &CoulombParams,
    ) -> anyhow::Result<Self> {
        let gpu = GpuContext::new_async(Limits::Default).await?;
        Self::setup(gpu, ref_a, ref_b, splined_matrix, coulomb)
    }

    fn setup(
        gpu: GpuContext,
        ref_a: Structure,
        ref_b: Structure,
        splines: &SplinedPotentials,
        coulomb: &CoulombParams,
    ) -> anyhow::Result<Self> {
        let tables = SplineTables::new(splines)?;
        let source = format!(
            "{}\n{}\n{}\n{}",
            tables.wgsl,
            include_str!("../shaders/common.wgsl"),
            include_str!("../shaders/spline_eval.wgsl"),
            include_str!("../shaders/energy.wgsl")
        );
        let kernel = PoseKernel::new(&gpu, "Energy", &source, &BINDINGS);
        let atom_ids: Vec<u32> = ref_a
            .atom_ids
            .iter()
            .chain(&ref_b.atom_ids)
            .map(|&id| id as u32)
            .collect();
        let constant = [
            gpu.storage("Spline Coefficients", &tables.coeffs),
            gpu.storage("Spline Params", &tables.params),
            gpu.storage("Positions A", bytemuck::cast_slice(&pack_positions(&ref_a))),
            gpu.storage("Positions B", bytemuck::cast_slice(&pack_positions(&ref_b))),
            gpu.storage("Atom IDs", bytemuck::cast_slice(&atom_ids)),
        ];
        let coulomb_prefactors = gpu.storage(
            "Coulomb Prefactors",
            bytemuck::cast_slice(&coulomb.prefactors),
        );
        let uniforms = GpuUniforms {
            n_atoms_a: ref_a.pos.len() as u32,
            n_atoms_b: ref_b.pos.len() as u32,
            n_atom_types: splines.n_types() as u32,
            n_poses: 0,
            kappa: coulomb.kappa,
            _pad: [0; 3],
        };
        log::info!(
            "GPU backend: {} atom types, {} + {} atoms",
            uniforms.n_atom_types,
            uniforms.n_atoms_a,
            uniforms.n_atoms_b
        );
        Ok(Self {
            gpu,
            kernel,
            constant,
            coulomb_prefactors,
            ref_a,
            ref_b,
            uniforms,
        })
    }

    fn bind(&self, poses: &wgpu::Buffer, energies: &wgpu::Buffer, n_poses: u32) -> wgpu::BindGroup {
        let uniforms = GpuUniforms {
            n_poses,
            ..self.uniforms
        };
        let uniform_buffer = self.gpu.uniform("Uniforms", bytemuck::bytes_of(&uniforms));
        let [c0, c1, c2, c3, c4] = &self.constant;
        let buffers = [
            c0,
            c1,
            c2,
            c3,
            c4,
            poses,
            energies,
            &uniform_buffer,
            &self.coulomb_prefactors,
        ];
        self.gpu.bind_group(&self.kernel.layout, &buffers)
    }
}

impl EnergyBackend for GpuBackend {
    fn compute_energy(&self, pose: &PoseParams) -> f64 {
        self.compute_energies(std::slice::from_ref(pose))[0]
    }

    fn compute_energies(&self, poses: &[PoseParams]) -> Vec<f64> {
        let bind = |p: &_, e: &_, n| self.bind(p, e, n);
        self.kernel.energies(&self.gpu, poses, bind)
    }

    fn prefers_batch(&self) -> bool {
        true
    }

    fn kind(&self) -> super::ScanBackend {
        super::ScanBackend::Gpu
    }

    fn ref_a(&self) -> &Structure {
        &self.ref_a
    }

    fn ref_b(&self) -> &Structure {
        &self.ref_b
    }
}

impl super::AsyncEnergyBackend for GpuBackend {
    async fn compute_energies_async(&self, poses: &[PoseParams]) -> Vec<f64> {
        let bind = |p: &_, e: &_, n| self.bind(p, e, n);
        self.kernel.energies_async(&self.gpu, poses, bind).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_support::{kt, lysozyme, no_gpu, pose_list};

    /// Kernel changes must keep agreeing with the exact pair sum. Poses that
    /// carry Boltzmann weight match to the spline error (1e-3, with a floor for
    /// f32 noise near U = 0). Overlaps differ by the spline's linear
    /// extrapolation below r_min, so they only need to stay repulsive.
    #[test]
    fn gpu_matches_exact_cpu_on_lysozyme() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (a, b) = (sys.ref_a.clone(), sys.ref_b.clone());
        let gpu = GpuBackend::new(a.clone(), b.clone(), &sys.splines, &sys.coulomb).unwrap();
        let exact = sys.into_exact(a, b);
        let u_weighted = 10.0 * kt();
        let poses = pose_list();
        let got = gpu.compute_energies(&poses);
        let mut n_weighted = 0;
        for (pose, u_gpu) in poses.iter().zip(got) {
            let u_exact = exact.compute_energy(pose);
            let weighted = u_exact < u_weighted;
            n_weighted += usize::from(weighted);
            let ok = if weighted {
                (u_gpu - u_exact).abs() < 1e-3 * u_exact.abs() + 0.05
            } else {
                u_gpu > u_weighted
            };
            assert!(
                ok,
                "R={} ω={}: gpu {u_gpu:.4} vs exact {u_exact:.4} kJ/mol",
                pose.r, pose.omega
            );
        }
        // Guards against a pose list that only exercises the repulsive branch.
        assert!(n_weighted >= 14, "only {n_weighted} weighted poses");
    }
}

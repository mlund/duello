//! Short-range maps of A, one per B atom type, built once on the GPU.

use super::tables::SlotTable;
use super::{GridDomain, GridSpec};
use crate::backend::gpu::{pack_positions, SplineTables};
use crate::backend::gpu_context::{f32_bytes, Binding, GpuContext};
use crate::energy::SplinedPotentials;
use crate::structure::Structure;
use bytemuck::{Pod, Zeroable};

/// `Grid` in grid_common.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GridUniform {
    origin: [f32; 4],
    /// Nodes along x, y, z; w = number of slots.
    dims: [u32; 4],
    spacing: f32,
    n_atom_types: u32,
    energy_cap: f32,
    n_atoms_a: u32,
    reach: f32,
    _pad: [u32; 3],
}

impl GridUniform {
    fn new(
        domain: &GridDomain,
        slots: &SlotTable,
        ref_a: &Structure,
        n_types: usize,
        energy_cap: f64,
        reach: f64,
    ) -> Self {
        let o = domain.origin();
        let [nx, ny, nz] = domain.dims();
        Self {
            origin: [o.x as f32, o.y as f32, o.z as f32, 0.0],
            dims: [nx, ny, nz, slots.types.len() as u32],
            spacing: domain.spacing() as f32,
            n_atom_types: n_types as u32,
            energy_cap: energy_cap as f32,
            n_atoms_a: ref_a.pos.len() as u32,
            reach: reach as f32,
            _pad: [0; 3],
        }
    }
}

/// A's short-range maps on the GPU and the lattice uniform that indexes them.
pub(super) struct Maps {
    pub values: wgpu::Buffer,
    pub grid_uniform: wgpu::Buffer,
}

const BUILD_BINDINGS: [Binding; 8] = [
    Binding::Read,      // spline coefficients
    Binding::Read,      // spline params
    Binding::Read,      // positions A
    Binding::Read,      // atom ids A
    Binding::Read,      // slot → B type
    Binding::ReadWrite, // maps
    Binding::Uniform,   // grid
    Binding::Uniform,   // first voxel of this dispatch
];

/// Voxels per build dispatch: keeps each one well under GPU watchdog limits
/// (~2 s on Windows/Linux drivers and in browsers).
const VOXELS_PER_DISPATCH: usize = 1 << 20;

/// Allocate the maps and fill them on the GPU; queue order puts the build
/// ahead of any pose dispatch, so nothing waits here.
pub(super) async fn build_maps(
    gpu: &GpuContext,
    ref_a: &Structure,
    splines: &SplinedPotentials,
    slots: &SlotTable,
    domain: &GridDomain,
    spec: GridSpec,
    reach: f64,
) -> anyhow::Result<Maps> {
    let bytes = f32_bytes(domain.n_voxels() * slots.types.len());
    check_fits(gpu, bytes)?;
    // Device limits don't account for free VRAM, so catch any allocation here.
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let maps = fill_maps(gpu, ref_a, splines, slots, domain, spec, reach);
    if let Some(err) = scope.pop().await {
        return Err(too_big(bytes, format!("free GPU memory ({err})")));
    }
    maps
}

fn fill_maps(
    gpu: &GpuContext,
    ref_a: &Structure,
    splines: &SplinedPotentials,
    slots: &SlotTable,
    domain: &GridDomain,
    spec: GridSpec,
    reach: f64,
) -> anyhow::Result<Maps> {
    let n_vox = domain.n_voxels();
    let grid = GridUniform::new(
        domain,
        slots,
        ref_a,
        splines.n_types(),
        spec.energy_cap,
        reach,
    );
    let grid_uniform = gpu.uniform("Grid", bytemuck::bytes_of(&grid));
    let values = gpu.output("Grid maps", n_vox * slots.types.len());
    let tables = SplineTables::new(splines)?;
    let source = format!(
        "{}\nconst MAX_SLOTS: u32 = {}u;\n{}\n{}\n{}",
        tables.wgsl,
        slots.types.len(),
        include_str!("../../shaders/spline_eval.wgsl"),
        include_str!("../../shaders/grid_common.wgsl"),
        include_str!("../../shaders/grid_build.wgsl")
    );
    let (pipeline, layout) = gpu.pipeline("Grid build", &source, &BUILD_BINDINGS);
    let ids_a: Vec<u32> = ref_a.atom_ids.iter().map(|&t| t as u32).collect();
    let fixed = [
        gpu.storage("Spline Coefficients", &tables.coeffs),
        gpu.storage("Spline Params", &tables.params),
        gpu.storage("Positions A", bytemuck::cast_slice(&pack_positions(ref_a))),
        gpu.storage("Atom IDs A", bytemuck::cast_slice(&ids_a)),
        gpu.storage("Slot types", bytemuck::cast_slice(&slots.types)),
    ];
    let shared: Vec<&wgpu::Buffer> = fixed.iter().chain([&values, &grid_uniform]).collect();
    for first in (0..n_vox).step_by(VOXELS_PER_DISPATCH) {
        let offset = gpu.uniform("Voxel offset", bytemuck::bytes_of(&[first as u32, 0, 0, 0]));
        let group = gpu.bind_group(&layout, &[&shared[..], &[&offset]].concat());
        let n = VOXELS_PER_DISPATCH.min(n_vox - first);
        gpu.dispatch(&pipeline, &group, [(n as u32).div_ceil(64), 1]);
    }
    Ok(Maps {
        values,
        grid_uniform,
    })
}

fn too_big(bytes: u64, limit: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!(
        "grid maps need {} MiB, more than {limit}; use a coarser grid spacing",
        bytes >> 20
    )
}

/// Kernels index maps with u32, so the value count is capped there too.
fn check_fits(gpu: &GpuContext, bytes: u64) -> anyhow::Result<()> {
    let limits = gpu.device.limits();
    let max = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size)
        .min(f32_bytes(u32::MAX as usize));
    if bytes > max {
        return Err(too_big(
            bytes,
            format!("the {} MiB the GPU allows", max >> 20),
        ));
    }
    Ok(())
}

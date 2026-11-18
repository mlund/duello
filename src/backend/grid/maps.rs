//! Short-range maps of A, one per B atom type, built once on the GPU.

use super::tables::SlotTable;
use super::{GridDomain, GridSpec};
use crate::backend::gpu::{pack_positions, SplineTables};
use crate::backend::gpu_context::{Binding, GpuContext};
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

const BUILD_BINDINGS: [Binding; 9] = [
    Binding::Read,      // spline coefficients
    Binding::Read,      // spline params
    Binding::Read,      // positions A
    Binding::Read,      // atom ids A
    Binding::Read,      // slot → B type
    Binding::ReadWrite, // maps
    Binding::Uniform,   // grid
    Binding::Uniform,   // node-pair range of this dispatch
    Binding::Read,      // pair minima
];

/// Node pairs (2 voxels each) per build dispatch: keeps each one well under GPU watchdog limits
/// (~2 s on Windows/Linux drivers and in browsers).
const PAIRS_PER_DISPATCH: usize = 1 << 19;

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
    let bytes = BYTES_PER_VALUE * map_values(domain, slots) as u64;
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
    // One build thread per x-pair of nodes; see grid_build.wgsl.
    let [nx, ny, nz] = domain.dims();
    let n_pairs = (nx.div_ceil(2) * ny * nz) as usize;
    let grid = GridUniform::new(
        domain,
        slots,
        ref_a,
        splines.n_types(),
        spec.energy_cap,
        reach,
    );
    let grid_uniform = gpu.uniform("Grid", bytemuck::bytes_of(&grid));
    // f16 pairs packed in u32 words; `map_values` is even since BRICK is.
    let values = gpu.output("Grid maps", map_values(domain, slots) / 2);
    let tables = SplineTables::new(splines)?;
    let source = format!(
        "{}\nconst MAX_SLOTS: u32 = {}u;\n{}{}\n{}\n{}",
        tables.wgsl,
        slots.types.len(),
        layout_consts(),
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
    let minima = super::tables::pair_minima(splines, ref_a, &slots.types);
    let minima = gpu.storage("Pair minima", bytemuck::cast_slice(&minima));
    let shared: Vec<&wgpu::Buffer> = fixed.iter().chain([&values, &grid_uniform]).collect();
    for first in (0..n_pairs).step_by(PAIRS_PER_DISPATCH) {
        let range = [first as u32, n_pairs as u32, 0, 0];
        let pair_range = gpu.uniform("Pair range", bytemuck::bytes_of(&range));
        let group = gpu.bind_group(&layout, &[&shared[..], &[&pair_range, &minima]].concat());
        let n = PAIRS_PER_DISPATCH.min(n_pairs - first);
        gpu.dispatch(&pipeline, &group, [(n as u32).div_ceil(64), 1]);
    }
    Ok(Maps {
        values,
        grid_uniform,
    })
}

/// Maps hold f16; `map_index` counts values, so u32 caps values, not bytes.
const BYTES_PER_VALUE: u64 = 2;

/// Map brick edge in nodes; see `map_index` in grid_common.wgsl.
const BRICK: u32 = 2;

/// Largest finite f16, the range of stored map values.
pub(super) const F16_MAX: f64 = 65504.0;

/// WGSL constants of the map layout, prepended to both grid kernels so each
/// has a single definition.
pub(super) fn layout_consts() -> String {
    format!("const BRICK: u32 = {BRICK}u;\nconst F16_MAX: f32 = {F16_MAX:.1};\n")
}

/// Stored values: every slot holds the lattice padded to whole bricks (see
/// `map_index` in grid_common.wgsl).
fn map_values(domain: &GridDomain, slots: &SlotTable) -> usize {
    let padded: usize = domain
        .dims()
        .iter()
        .map(|&n| (n.div_ceil(BRICK) * BRICK) as usize)
        .product();
    // A repulsive and an attractive map per slot.
    2 * padded * slots.types.len()
}

fn too_big(bytes: u64, limit: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!(
        "grid maps need {} MiB, more than {limit}; use a coarser grid spacing",
        bytes >> 20
    )
}

/// Kernels index map values with u32, so their count is capped there too.
fn check_fits(gpu: &GpuContext, bytes: u64) -> anyhow::Result<()> {
    let limits = gpu.device.limits();
    let max = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size)
        .min(BYTES_PER_VALUE * u64::from(u32::MAX));
    if bytes > max {
        return Err(too_big(
            bytes,
            format!("the {} MiB the GPU allows", max >> 20),
        ));
    }
    Ok(())
}

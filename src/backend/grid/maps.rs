//! Short-range maps of A, one per B atom type, built once on the GPU.

use super::switch::Switch;
use super::tables::SlotTable;
use super::{GridDomain, GridSpec};
use crate::backend::gpu::{pack_positions, SplineTables};
use crate::backend::gpu_context::{Binding, GpuContext};
use crate::structure::Structure;
use bytemuck::{Pod, Zeroable};

/// What the maps are built from.
pub(super) struct Field<'a> {
    pub ref_a: &'a Structure,
    pub n_types: usize,
    pub slots: &'a SlotTable,
    pub domain: GridDomain,
    pub spec: GridSpec,
    /// The far map is nonzero within this distance (Å) of an A bead: where the
    /// short-range energy vanishes, plus the prefilter's stencil.
    pub support: f64,
    pub switch: Switch,
}

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
    support: f32,
    near_r1: f32,
    near_r2: f32,
    _pad: u32,
}

impl GridUniform {
    fn new(field: &Field) -> Self {
        let domain = &field.domain;
        let o = domain.origin();
        let [nx, ny, nz] = domain.dims();
        Self {
            origin: [o.x as f32, o.y as f32, o.z as f32, 0.0],
            dims: [nx, ny, nz, field.slots.types.len() as u32],
            spacing: domain.spacing() as f32,
            n_atom_types: field.n_types as u32,
            energy_cap: field.spec.energy_cap as f32,
            n_atoms_a: field.ref_a.pos.len() as u32,
            support: field.support as f32,
            near_r1: field.switch.r1 as f32,
            near_r2: field.switch.r2 as f32,
            _pad: 0,
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
    Binding::Uniform,   // node-pair range of this dispatch
];

/// Node pairs (2 voxels each) per build dispatch: keeps each one well under GPU watchdog limits
/// (~2 s on Windows/Linux drivers and in browsers).
const PAIRS_PER_DISPATCH: usize = 1 << 19;

/// Allocate the maps and fill them on the GPU; queue order puts the build
/// ahead of any pose dispatch, so nothing waits here. `splines` are the
/// buffers of `tables`, shared with the pose kernel.
pub(super) async fn build_maps(
    gpu: &GpuContext,
    field: &Field<'_>,
    tables: &SplineTables,
    splines: &[wgpu::Buffer; 2],
) -> anyhow::Result<Maps> {
    let bytes = BYTES_PER_VALUE * map_values(&field.domain, field.slots) as u64;
    check_fits(gpu, bytes)?;
    // Device limits don't account for free VRAM, so catch any allocation here.
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let maps = fill_maps(gpu, field, tables, splines);
    if let Some(err) = scope.pop().await {
        return Err(too_big(bytes, format!("free GPU memory ({err})")));
    }
    Ok(maps)
}

fn fill_maps(
    gpu: &GpuContext,
    field: &Field<'_>,
    tables: &SplineTables,
    splines: &[wgpu::Buffer; 2],
) -> Maps {
    // One build thread per x-pair of nodes; see grid_build.wgsl.
    let [nx, ny, nz] = field.domain.dims();
    let n_pairs = (nx.div_ceil(2) * ny * nz) as usize;
    let grid_uniform = gpu.uniform("Grid", bytemuck::bytes_of(&GridUniform::new(field)));
    // f16 pairs packed in u32 words; `map_values` is even since BRICK is.
    let values = gpu.output("Grid maps", map_values(&field.domain, field.slots) / 2);
    let source = format!(
        "{}\nconst MAX_SLOTS: u32 = {}u;\nconst LAPLACIAN_STEP: f32 = {LAPLACIAN_STEP};\n{}{}\n{}\n{}\n{}",
        tables.wgsl,
        field.slots.types.len(),
        layout_consts(),
        include_str!("../../shaders/spline_eval.wgsl"),
        include_str!("../../shaders/grid_common.wgsl"),
        super::switch::WGSL,
        include_str!("../../shaders/grid_build.wgsl")
    );
    let (pipeline, layout) = gpu.pipeline("Grid build", &source, &BUILD_BINDINGS);
    let ref_a = field.ref_a;
    let ids_a: Vec<u32> = ref_a.atom_ids.iter().map(|&t| t as u32).collect();
    let fixed = [
        gpu.storage("Positions A", bytemuck::cast_slice(&pack_positions(ref_a))),
        gpu.storage("Atom IDs A", bytemuck::cast_slice(&ids_a)),
        gpu.storage("Slot types", bytemuck::cast_slice(&field.slots.types)),
    ];
    let shared: Vec<&wgpu::Buffer> = splines
        .iter()
        .chain(&fixed)
        .chain([&values, &grid_uniform])
        .collect();
    for first in (0..n_pairs).step_by(PAIRS_PER_DISPATCH) {
        let range = [first as u32, n_pairs as u32, 0, 0];
        let pair_range = gpu.uniform("Pair range", bytemuck::bytes_of(&range));
        let group = gpu.bind_group(&layout, &[&shared[..], &[&pair_range]].concat());
        let n = PAIRS_PER_DISPATCH.min(n_pairs - first);
        gpu.dispatch(&pipeline, &group, [(n as u32).div_ceil(64), 1]);
    }
    Maps {
        values,
        grid_uniform,
    }
}

/// Radial step (Å) of the build's finite-difference Laplacian: wide enough
/// that f32 cancellation stays ~1e-5 relative, narrow enough that the
/// truncation error on an r⁻⁶ tail is ~1e-3 of f″.
pub(super) const LAPLACIAN_STEP: f64 = 0.1;

/// Maps hold f16; `map_index` counts values, so u32 caps values, not bytes.
const BYTES_PER_VALUE: u64 = 2;

/// Map brick edge in nodes; see `map_index` in grid_common.wgsl.
const BRICK: u32 = 2;

/// Largest finite f16; the cap clamps stored values, so it must fit.
pub(super) const F16_MAX: f64 = 65504.0;

/// WGSL constants of the map layout, prepended to both grid kernels so each
/// has a single definition.
pub(super) fn layout_consts() -> String {
    format!("const BRICK: u32 = {BRICK}u;\n")
}

/// Stored values: every slot holds the lattice padded to whole bricks (see
/// `map_index` in grid_common.wgsl).
fn map_values(domain: &GridDomain, slots: &SlotTable) -> usize {
    let padded: usize = domain
        .dims()
        .iter()
        .map(|&n| (n.div_ceil(BRICK) * BRICK) as usize)
        .product();
    padded * slots.types.len()
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

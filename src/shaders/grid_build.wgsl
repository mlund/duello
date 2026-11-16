// Map build: one thread per voxel sums A's short-range energy for every B type.
// Each node acts as AutoDock's probe atom, "sequentially placed at each grid
// point" with "an affinity grid [...] calculated for each type of atom"
// (Goodsell & Olson 1990, p. 196).
//
// Prepended at runtime: the grid-type-specific spline WGSL, spline_eval.wgsl,
// grid_common.wgsl and `const MAX_SLOTS`, sized to the actual slot count so
// the per-thread accumulator array stays small.

@group(0) @binding(0) var<storage, read> spline_coeffs: array<SplineCoeffs>;
@group(0) @binding(1) var<storage, read> spline_params: array<SplineParams>;
@group(0) @binding(2) var<storage, read> pos_a: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ids_a: array<u32>;
// B atom type held by each map slot.
@group(0) @binding(4) var<storage, read> slot_types: array<u32>;
@group(0) @binding(5) var<storage, read_write> maps: array<f32>;
@group(0) @binding(6) var<uniform> grid: Grid;
// The build is split into dispatches; x = first voxel of this one.
@group(0) @binding(7) var<uniform> voxel_offset: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = voxel_offset.x + gid.x;
    let n_vox = grid.dims.x * grid.dims.y * grid.dims.z;
    if v >= n_vox {
        return;
    }
    let nx = grid.dims.x;
    let ny = grid.dims.y;
    let node = vec3<u32>(v % nx, (v / nx) % ny, v / (nx * ny));
    let p = grid.origin.xyz + grid.spacing * vec3<f32>(node);

    // A outer, slots inner: one distance per A bead serves every slot.
    var u: array<f32, MAX_SLOTS>;
    let n_slots = grid.dims.w;
    for (var i = 0u; i < grid.n_atoms_a; i++) {
        let r = distance(pos_a[i].xyz, p);
        if r >= grid.reach {
            continue; // most of the box is out of reach of any one bead
        }
        let row = ids_a[i] * grid.n_atom_types;
        for (var s = 0u; s < n_slots; s++) {
            u[s] += spline_energy(row + slot_types[s], r);
        }
    }
    // A flat wall keeps trilinear weights from smearing huge overlap energies
    // into neighbouring cells.
    for (var s = 0u; s < n_slots; s++) {
        maps[s * n_vox + v] = min(u[s], grid.energy_cap);
    }
}

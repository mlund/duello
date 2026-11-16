// Map build: each thread sums A's short-range energy for every B type at an
// x-pair of nodes, which share one u32 of f16 maps (BRICK is even), so no two
// threads write the same word.
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
// Two f16 per word: half the memory and bandwidth of f32. See `map_value`.
@group(0) @binding(5) var<storage, read_write> maps: array<u32>;
@group(0) @binding(6) var<uniform> grid: Grid;
// The build is split into dispatches: x = first node pair of this one,
// y = total node pairs.
@group(0) @binding(7) var<uniform> pairs: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = pairs.x + gid.x;
    if q >= pairs.y {
        return;
    }
    let n_pairs_x = (grid.dims.x + 1u) / 2u;
    let ny = grid.dims.y;
    let node = vec3<u32>(2u * (q % n_pairs_x), (q / n_pairs_x) % ny, q / (n_pairs_x * ny));
    let p0 = grid.origin.xyz + grid.spacing * vec3<f32>(node);
    // May fall in the x padding: stored but never read.
    let p1 = p0 + vec3<f32>(grid.spacing, 0.0, 0.0);

    // A outer, slots inner: one distance per A bead serves every slot.
    var u: array<vec2<f32>, MAX_SLOTS>;
    let n_slots = grid.dims.w;
    for (var i = 0u; i < grid.n_atoms_a; i++) {
        let a = pos_a[i].xyz;
        let r = vec2<f32>(distance(a, p0), distance(a, p1));
        let near = r < vec2<f32>(grid.reach);
        if !any(near) {
            continue; // most of the box is out of reach of any one bead
        }
        let row = ids_a[i] * grid.n_atom_types;
        for (var s = 0u; s < n_slots; s++) {
            let pair = row + slot_types[s];
            let e = vec2<f32>(spline_energy(pair, r.x), spline_energy(pair, r.y));
            u[s] += select(vec2<f32>(0.0), e, near);
        }
    }
    // A flat wall keeps trilinear weights from smearing huge overlap energies
    // into neighbouring cells; the floor keeps deep attraction inside f16
    // range, where pack is defined. The cap is validated to fit already.
    for (var s = 0u; s < n_slots; s++) {
        let stored = clamp(u[s], vec2<f32>(-F16_MAX), vec2<f32>(grid.energy_cap));
        maps[map_index(s, node) / 2u] = pack2x16float(stored);
    }
}

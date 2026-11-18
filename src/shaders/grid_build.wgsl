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
// (r_m, U_m) per type pair: where each potential splits into repulsion and attraction.
@group(0) @binding(8) var<storage, read> pair_min: array<vec2<f32>>;

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
    var rep: array<vec2<f32>, MAX_SLOTS>;
    var att: array<vec2<f32>, MAX_SLOTS>;
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
            let m = pair_min[pair];
            let inside = r < vec2<f32>(m.x);
            // f32 splines can dip a hair below the CPU-found U_m near r_m.
            let e_rep = select(vec2<f32>(0.0), max(e - m.y, vec2<f32>(0.0)), inside);
            let e_att = select(e, vec2<f32>(m.y), inside);
            rep[s] += select(vec2<f32>(0.0), e_rep, near);
            att[s] += select(vec2<f32>(0.0), e_att, near);
        }
    }
    // A flat wall keeps trilinear weights from smearing huge overlap energies
    // into neighbouring cells; attraction is kept inside f16 range, where pack
    // is defined, and may be positive past r_m (e.g. a barrier).
    for (var s = 0u; s < n_slots; s++) {
        // Repulsion as its fourth root, which interpolates nearly linearly up
        // a steep wall (Minh 2018); attraction plain, as it is smooth.
        let r4 = sqrt(sqrt(min(rep[s], vec2<f32>(grid.energy_cap))));
        maps[map_index(s, node) / 2u] = pack2x16float(r4);
        let a = clamp(att[s], vec2<f32>(-F16_MAX), vec2<f32>(F16_MAX));
        maps[map_index(attractive_slot(s), node) / 2u] = pack2x16float(a);
    }
}

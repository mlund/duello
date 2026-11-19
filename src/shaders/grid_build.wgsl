// Map build: each thread sums A's short-range energy for every B type at an
// x-pair of nodes, which share one u32 of f16 maps (BRICK is even), so no two
// threads write the same word.
// Each node acts as AutoDock's probe atom, "sequentially placed at each grid
// point" with "an affinity grid [...] calculated for each type of atom"
// (Goodsell & Olson 1990, p. 196).
//
// Only the far part (1 − S)·U is stored, prefiltered so trilinear lookups are
// unbiased: near pairs are summed exactly per pose, so walls and wells never
// reach the map.
//
// Prepended at runtime: the grid-type-specific spline WGSL, spline_eval.wgsl,
// grid_common.wgsl, grid_switch.wgsl, `const MAX_SLOTS`, sized to the
// actual slot count so the per-thread accumulator array stays small.

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

// Radii r − d, r, r + d (d = LAPLACIAN_STEP) and the far weights 1 − S there;
// they depend on r only, so one serves every slot.
struct Stencil {
    r: vec3<f32>,
    w: vec3<f32>,
}

fn stencil(r: f32) -> Stencil {
    let rs = r + LAPLACIAN_STEP * vec3<f32>(-1.0, 0.0, 1.0);
    let s = vec3<f32>(near_weight(rs.x), near_weight(rs.y), near_weight(rs.z));
    return Stencil(rs, 1.0 - s);
}

// f − h²/12·∇²f of the far part f = (1 − S)·U, with ∇²f = f″ + 2f′/r; see
// the grid module docs. `h2_12` is h²/12.
fn prefiltered_far(pair: u32, st: Stencil, h2_12: f32) -> f32 {
    let u = vec3<f32>(spline_energy(pair, st.r.x), spline_energy(pair, st.r.y), spline_energy(pair, st.r.z));
    let f = st.w * u;
    let d = LAPLACIAN_STEP;
    let laplacian = (f.z - 2.0 * f.y + f.x) / (d * d) + (f.z - f.x) / (d * st.r.y);
    return f.y - h2_12 * laplacian;
}

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
    var far: array<vec2<f32>, MAX_SLOTS>;
    let h2_12 = grid.spacing * grid.spacing / 12.0;
    let n_slots = grid.dims.w;
    for (var i = 0u; i < grid.n_atoms_a; i++) {
        let a = pos_a[i].xyz;
        let r = vec2<f32>(distance(a, p0), distance(a, p1));
        // The far part is zero inside r1 and beyond reach.
        let used = r > vec2<f32>(grid.near_r1) & r < vec2<f32>(grid.reach);
        if !any(used) {
            continue; // most of the box is out of reach of any one bead
        }
        let row = ids_a[i] * grid.n_atom_types;
        let st = array<Stencil, 2>(stencil(r.x), stencil(r.y));
        for (var s = 0u; s < n_slots; s++) {
            let pair = row + slot_types[s];
            let e = vec2<f32>(prefiltered_far(pair, st[0], h2_12), prefiltered_far(pair, st[1], h2_12));
            far[s] += select(vec2<f32>(0.0), e, used);
        }
    }
    // Far values stay far below the cap; clamping only keeps pack defined.
    let cap = vec2<f32>(grid.energy_cap);
    for (var s = 0u; s < n_slots; s++) {
        maps[map_index(s, node) / 2u] = pack2x16float(clamp(far[s], -cap, cap));
    }
}

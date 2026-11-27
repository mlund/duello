// Pose energies: one thread per pose. Per B bead, the far map by trilinear
// lookup plus the near part summed exactly over A; electrostatics exact.
//
// Prepended at runtime: the grid-type-specific spline WGSL, common.wgsl,
// spline_eval.wgsl, grid_common.wgsl, grid_switch.wgsl and grid_near.wgsl.

struct Batch {
    n_poses: u32,
    n_atoms_b: u32,
    // Beyond this R no B bead can reach A's short-range field.
    sr_range: f32,
    kappa: f32,
    n_charged_a: u32,
    n_charged_b: u32,
    // Where B's beads start in `beads`.
    b_offset: u32,
    // Yukawa truncation; f32 max and 0 when untruncated.
    coulomb_cutoff_sq: f32,
    coulomb_shift: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    cells: Cells,
}

// f16 pairs; see grid_build.wgsl.
@group(0) @binding(0) var<storage, read> maps: array<u32>;
// One list, to stay within WebGPU's default of 8 storage buffers per stage:
// A beads (xyz, type) in cell order, then the charged ones again from
// grid.n_atoms_a; B beads (xyz, slot·2¹⁶ + type) from batch.b_offset, then
// its charged ones (xyz, type).
@group(0) @binding(1) var<storage, read> beads: array<vec4<f32>>;
// Start of each cell's A beads; see grid_near.wgsl.
@group(0) @binding(2) var<storage, read> cell_start: array<u32>;

// The four lists in `beads`; only these know the layout.
fn bead_a(i: u32) -> vec4<f32> {
    return beads[i];
}
fn charged_a(i: u32) -> vec4<f32> {
    return beads[grid.n_atoms_a + i];
}
fn bead_b(j: u32) -> vec4<f32> {
    return beads[batch.b_offset + j];
}
fn charged_b(j: u32) -> vec4<f32> {
    return beads[batch.b_offset + batch.n_atoms_b + j];
}
@group(0) @binding(3) var<storage, read> poses: array<PoseParams>;
@group(0) @binding(4) var<storage, read_write> energies: array<f32>;
@group(0) @binding(5) var<uniform> grid: Grid;
@group(0) @binding(6) var<uniform> batch: Batch;
@group(0) @binding(7) var<storage, read> coulomb_prefactors: array<f32>;
@group(0) @binding(8) var<storage, read> spline_coeffs: array<SplineCoeffs>;
@group(0) @binding(9) var<storage, read> spline_params: array<SplineParams>;

fn corner(slot: u32, node: vec3<u32>) -> f32 {
    let i = map_index(slot, node);
    let pair = unpack2x16float(maps[i / 2u]);
    return select(pair.x, pair.y, i % 2u == 1u);
}

// The far map of `slot` at `p` by "trilinear interpolation of the affinity
// values of the eight grid points surrounding the atom position" (Goodsell &
// Olson 1990, p. 196). The cell index is clamped to dims - 2 so a point on
// the top face uses the last cell with weight 1.
fn far_energy(slot: u32, p: vec3<f32>) -> f32 {
    let g = (p - grid.origin.xyz) / grid.spacing;
    let top = vec3<f32>(grid.dims.xyz - 1u);
    if any(g < vec3<f32>(0.0)) || any(g > top) {
        return 0.0; // outside the box: beyond reach of every A bead
    }
    let cell = min(vec3<u32>(floor(g)), grid.dims.xyz - 2u);
    let t = g - vec3<f32>(cell);
    let c00 = mix(corner(slot, cell + vec3<u32>(0u, 0u, 0u)), corner(slot, cell + vec3<u32>(1u, 0u, 0u)), t.x);
    let c10 = mix(corner(slot, cell + vec3<u32>(0u, 1u, 0u)), corner(slot, cell + vec3<u32>(1u, 1u, 0u)), t.x);
    let c01 = mix(corner(slot, cell + vec3<u32>(0u, 0u, 1u)), corner(slot, cell + vec3<u32>(1u, 0u, 1u)), t.x);
    let c11 = mix(corner(slot, cell + vec3<u32>(0u, 1u, 1u)), corner(slot, cell + vec3<u32>(1u, 1u, 1u)), t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

struct Near {
    u: f32,
    // The bead's near energy reached the cap: the pose is an overlap.
    in_wall: bool,
    // A pair sits inside its table start: an overlap too (see inside_table).
    inside: bool,
}

// S·U of A beads [span.x, span.y) within near_r2 of `p`, uncapped, and
// whether one sits inside its table start (`inside`). Walls are finite (splines
// extrapolate linearly); one big enough to swamp the f32 sum also puts its
// bead over the cap, so the lost digits never matter.
fn span_energy(span: vec2<u32>, type_b: u32, p: vec3<f32>, r2_max: f32) -> Near {
    var near = Near(0.0, false, false);
    for (var i = span.x; i < span.y; i++) {
        let a = bead_a(i);
        let d = a.xyz - p;
        // A cell box is ~6× the range sphere: test r² before paying for the sqrt.
        let d2 = dot(d, d);
        if d2 < r2_max {
            let r = sqrt(d2);
            let pair = u32(a.w) * grid.n_atom_types + type_b;
            let params = spline_params[pair];
            near.u += near_weight(r) * spline_energy_at(params, r);
            near.inside = near.inside || inside_table(params, r);
        }
    }
    return near;
}

// S·U over A beads within near_r2 of `p`, then capped as a whole: a bead
// whose net near energy reaches the cap is an overlap, while a wall pair
// offset by attraction from other A beads stays as summed.
fn near_energy(type_b: u32, p: vec3<f32>) -> Near {
    var near = Near(0.0, false, false);
    let r2_max = grid.near_r2 * grid.near_r2;
    let box = near_box(batch.cells, p);
    if !box.empty {
        for (var z = box.lo.z; z <= box.hi.z; z++) {
            for (var y = box.lo.y; y <= box.hi.y; y++) {
                let span = span_energy(row_span(batch.cells, box, y, z), type_b, p, r2_max);
                near.u += span.u;
                near.inside = near.inside || span.inside;
            }
        }
    }
    return Near(min(near.u, grid.energy_cap), near.u >= grid.energy_cap, near.inside);
}

// Exact screened Coulomb over charged pairs; not gridded, so it holds at any
// spacing and Debye length.
fn yukawa(tf: RigidTransform) -> f32 {
    var u: f32 = 0.0;
    for (var j = 0u; j < batch.n_charged_b; j++) {
        let b = charged_b(j);
        let pb = quat_rotate(tf.q, b.xyz) + tf.c;
        for (var i = 0u; i < batch.n_charged_a; i++) {
            let a = charged_a(i);
            let r = distance(a.xyz, pb);
            let prefactor = coulomb_prefactors[u32(a.w) * grid.n_atom_types + u32(b.w)];
            // A kept bead can still be neutral towards this type; skipping avoids
            // 0·inf = NaN when two beads coincide.
            if prefactor != 0.0 && r * r < batch.coulomb_cutoff_sq {
                u += prefactor * (exp(-batch.kappa * r) / r - batch.coulomb_shift);
            }
        }
    }
    return u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pose_idx = gid.x;
    if pose_idx >= batch.n_poses {
        return;
    }
    let pose = poses[pose_idx];
    let tf = pose_transform(pose);
    var u: f32 = 0.0;
    var in_wall = false;
    var inside = false;
    if pose.r <= batch.sr_range {
        for (var j = 0u; j < batch.n_atoms_b; j++) {
            let b = bead_b(j);
            let p = quat_rotate(tf.q, b.xyz) + tf.c;
            let code = u32(b.w);
            let near = near_energy(code & 0xffffu, p);
            in_wall = in_wall || near.in_wall;
            inside = inside || near.inside;
            u += far_energy(code >> 16u, p) + near.u;
        }
    }
    u += yukawa(tf);
    // A bead in the wall is an overlap, whatever the rest attract; without
    // this a low cap would let attraction outweigh it.
    if in_wall {
        u = max(u, grid.energy_cap);
    }
    if inside {
        u = OVERLAP_ENERGY;
    }
    energies[pose_idx] = u;
}

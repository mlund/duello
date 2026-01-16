// Compute shader for pairwise energy calculation
// Each workgroup thread processes one pose

struct SplineParams {
    r_min: f32,
    inv_delta: f32,
    n_coeffs: u32,
    coeff_offset: u32,
}

struct PoseParams {
    r: f32,
    omega: f32,
    _pad: vec2<f32>,       // Padding to align vertex_i to 16 bytes
    vertex_i: vec4<f32>,
    vertex_j: vec4<f32>,
}

struct Uniforms {
    n_atoms_a: u32,
    n_atoms_b: u32,
    n_atom_types: u32,
    n_poses: u32,
}

@group(0) @binding(0) var<storage, read> spline_coeffs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> spline_params: array<SplineParams>;
@group(0) @binding(2) var<storage, read> ref_pos_a: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ref_pos_b: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> atom_ids: array<u32>;
@group(0) @binding(5) var<storage, read> poses: array<PoseParams>;
@group(0) @binding(6) var<storage, read_write> energies: array<f32>;
@group(0) @binding(7) var<uniform> uniforms: Uniforms;

// Quaternion multiplication: q1 * q2
// Quaternion format: (x, y, z, w) where w is scalar
fn quat_mul(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// Rotate vector by quaternion: q * v * q^-1
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + 2.0 * (q.w * uv + uuv);
}

// Create quaternion that rotates from_vec to to_vec
// Matches nalgebra's UnitQuaternion::rotation_between behavior
fn quat_rotation_between(from_vec: vec3<f32>, to_vec: vec3<f32>) -> vec4<f32> {
    let from_n = normalize(from_vec);
    let to_n = normalize(to_vec);
    let c = cross(from_n, to_n);
    let d = dot(from_n, to_n);

    let c_len_sq = dot(c, c);

    // Handle nearly parallel vectors (identity rotation)
    if c_len_sq < 1e-12 && d > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // Handle nearly anti-parallel vectors (180 degree rotation)
    // Match nalgebra's perpendicular axis choice: (-y-z, x, x) or (z, z, -x-y)
    if c_len_sq < 1e-12 && d < 0.0 {
        // Use nalgebra's perpendicular axis: (-from.y - from.z, from.x, from.x)
        var perp = vec3<f32>(-from_n.y - from_n.z, from_n.x, from_n.x);
        let perp_len_sq = dot(perp, perp);
        if perp_len_sq < 1e-12 {
            // Fallback: (from.z, from.z, -from.x - from.y)
            perp = vec3<f32>(from_n.z, from_n.z, -from_n.x - from_n.y);
        }
        perp = normalize(perp);
        // 180 degree rotation: q = (axis, 0)
        return vec4<f32>(perp.x, perp.y, perp.z, 0.0);
    }

    // Standard case using the half-angle formula for numerical stability
    // This is equivalent to nalgebra's implementation
    let denom = 1.0 + d;

    // Check for near-antiparallel case that passed the cross product test
    // but still has small denom (happens when vectors are nearly opposite)
    if denom < 1e-6 {
        // Use the same perpendicular axis choice as above
        var perp = vec3<f32>(-from_n.y - from_n.z, from_n.x, from_n.x);
        let perp_len_sq = dot(perp, perp);
        if perp_len_sq < 1e-12 {
            perp = vec3<f32>(from_n.z, from_n.z, -from_n.x - from_n.y);
        }
        perp = normalize(perp);
        return vec4<f32>(perp.x, perp.y, perp.z, 0.0);
    }

    // cos(θ/2) = sqrt((1+d)/2), sin(θ/2) = sqrt((1-d)/2)
    // Scale factor: sin(θ/2) / |c| = sqrt((1-d)/2) / sqrt(1-d²) = sqrt((1-d)/2) / sqrt((1-d)(1+d))
    //             = 1 / sqrt(2(1+d))
    let scale = 1.0 / sqrt(2.0 * denom);
    let w = sqrt(denom * 0.5);
    return vec4<f32>(c.x * scale, c.y * scale, c.z * scale, w);
}

// Create quaternion for rotation around axis by angle
fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let half_angle = angle * 0.5;
    let s = sin(half_angle);
    let axis_n = normalize(axis);
    return vec4<f32>(axis_n.x * s, axis_n.y * s, axis_n.z * s, cos(half_angle));
}

// Evaluate spline energy using Horner's method
fn spline_energy(pair_idx: u32, r_sq: f32) -> f32 {
    let params = spline_params[pair_idx];
    let r = sqrt(r_sq);

    // Check cutoff
    let r_max = params.r_min + f32(params.n_coeffs - 1u) / params.inv_delta;
    if r >= r_max {
        return 0.0;
    }
    if r < params.r_min {
        // Clamp to minimum
        let c = spline_coeffs[params.coeff_offset];
        return c.x;
    }

    // Calculate grid index and fraction
    let t = (r - params.r_min) * params.inv_delta;
    let idx = min(u32(t), params.n_coeffs - 2u);
    let frac = t - f32(idx);

    // Fetch coefficients
    let c = spline_coeffs[params.coeff_offset + idx];

    // Horner's method: a0 + frac*(a1 + frac*(a2 + frac*a3))
    return c.x + frac * (c.y + frac * (c.z + frac * c.w));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pose_idx = gid.x;
    if pose_idx >= uniforms.n_poses {
        return;
    }

    let pose = poses[pose_idx];

    // Z axis (slightly offset to avoid singularity when vertex is at pole)
    // Using same offset as CPU: (0.0005, 0.0005, 1.0) normalized
    let zaxis = normalize(vec3<f32>(0.0005, 0.0005, 1.0));
    let neg_zaxis = -zaxis;

    // Normalize vertex directions
    let vertex_i = normalize(pose.vertex_i.xyz);
    let vertex_j = normalize(pose.vertex_j.xyz);

    // Build quaternions for structure B transformation:
    // q1: rotate vertex_j to -z axis
    // q2: rotate around z by omega
    // q3: rotate z axis to vertex_i
    let q1 = quat_rotation_between(vertex_j, neg_zaxis);
    let q2 = quat_from_axis_angle(zaxis, pose.omega);
    let q3 = quat_rotation_between(zaxis, vertex_i);

    // Combined rotation for initial orientation: q1 * q2
    // Note: quat_mul(a, b) computes a * b, so for q1 * q2 we need quat_mul(q1, q2)
    let q12 = quat_mul(q1, q2);

    // Translation vector
    let r_vec = vec3<f32>(0.0, 0.0, pose.r);

    var total_energy: f32 = 0.0;

    for (var i: u32 = 0u; i < uniforms.n_atoms_a; i++) {
        let pos_a = ref_pos_a[i].xyz;
        let id_a = atom_ids[i];

        for (var j: u32 = 0u; j < uniforms.n_atoms_b; j++) {
            // Transform pos_b:
            // 1. Rotate by q12 = q2 * q1
            // 2. Translate by r_vec
            // 3. Rotate by q3
            var pos_b = ref_pos_b[j].xyz;
            pos_b = quat_rotate(q12, pos_b);
            pos_b = pos_b + r_vec;
            pos_b = quat_rotate(q3, pos_b);

            let id_b = atom_ids[uniforms.n_atoms_a + j];
            let pair_idx = id_a * uniforms.n_atom_types + id_b;

            let diff = pos_a - pos_b;
            let r_sq = dot(diff, diff);

            total_energy += spline_energy(pair_idx, r_sq);
        }
    }

    energies[pose_idx] = total_energy;
}

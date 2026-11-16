// Pose struct and quaternion helpers shared by the compute kernels.
// Quaternion format: (x, y, z, w) where w is scalar.

struct PoseParams {
    r: f32,
    omega: f32,
    _pad: vec2<f32>,       // Padding to align vertex_i to 16 bytes
    vertex_i: vec4<f32>,
    vertex_j: vec4<f32>,
}

// Quaternion multiplication: q1 * q2
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

// Create quaternion for rotation around axis by angle
fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let half_angle = angle * 0.5;
    let s = sin(half_angle);
    let axis_n = normalize(axis);
    return vec4<f32>(axis_n.x * s, axis_n.y * s, axis_n.z * s, cos(half_angle));
}

// Rotation taking unit `u` to unit `v`; port of icotable's
// robust_rotation_between so GPU poses land where the CPU puts them.
fn robust_rotation_between(u: vec3<f32>, v: vec3<f32>) -> vec4<f32> {
    let d = dot(u, v);
    if d < -1.0 + 1e-6 {
        var perp = vec3<f32>(0.0, 1.0, 0.0);
        if abs(u.x) < 0.9 {
            perp = vec3<f32>(1.0, 0.0, 0.0);
        }
        return vec4<f32>(normalize(cross(u, perp)), 0.0);
    }
    return normalize(vec4<f32>(cross(u, v), 1.0 + d));
}

struct RigidTransform {
    q: vec4<f32>,
    c: vec3<f32>,
}

// B bead b lands at q·b + c (icotable::orient).
fn pose_transform(pose: PoseParams) -> RigidTransform {
    let z = vec3<f32>(0.0, 0.0, 1.0);
    let q1 = robust_rotation_between(normalize(pose.vertex_j.xyz), -z);
    let q2 = quat_from_axis_angle(z, pose.omega);
    let q3 = robust_rotation_between(z, normalize(pose.vertex_i.xyz));
    let q = quat_mul(q3, quat_mul(q1, q2));
    return RigidTransform(q, quat_rotate(q3, vec3<f32>(0.0, 0.0, pose.r)));
}


// Compute shader for pairwise energy calculation
// Each workgroup thread processes one pose
//
// Prepended at runtime: the grid-type-specific spline WGSL (interatomic
// crate), common.wgsl and spline_eval.wgsl.

struct Uniforms {
    n_atoms_a: u32,
    n_atoms_b: u32,
    n_atom_types: u32,
    n_poses: u32,
    kappa: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> spline_coeffs: array<SplineCoeffs>;
@group(0) @binding(1) var<storage, read> spline_params: array<SplineParams>;
@group(0) @binding(2) var<storage, read> ref_pos_a: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ref_pos_b: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> atom_ids: array<u32>;
@group(0) @binding(5) var<storage, read> poses: array<PoseParams>;
@group(0) @binding(6) var<storage, read_write> energies: array<f32>;
@group(0) @binding(7) var<uniform> uniforms: Uniforms;
@group(0) @binding(8) var<storage, read> coulomb_prefactors: array<f32>;

// Analytical Coulomb/Yukawa energy: prefactor * exp(-κr) / r
// Evaluated analytically (not splined) to avoid polynomial ringing from the
// SR potential's hard cutoff discontinuity inside the spline grid.
// No distance cutoff — Yukawa decays exponentially; exp() is single-cycle on GPU.
fn coulomb_energy(pair_idx: u32, r: f32) -> f32 {
    let prefactor = coulomb_prefactors[pair_idx];
    if prefactor == 0.0 { return 0.0; } // skip uncharged pairs
    return prefactor * exp(-uniforms.kappa * r) / r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pose_idx = gid.x;
    if pose_idx >= uniforms.n_poses {
        return;
    }

    let tf = pose_transform(poses[pose_idx]);

    // Transform each B bead once, then loop A inside: N_B rotations per pose.
    var total_energy: f32 = 0.0;

    for (var j: u32 = 0u; j < uniforms.n_atoms_b; j++) {
        let pos_b = quat_rotate(tf.q, ref_pos_b[j].xyz) + tf.c;
        let id_b = atom_ids[uniforms.n_atoms_a + j];

        for (var i: u32 = 0u; i < uniforms.n_atoms_a; i++) {
            let pair_idx = atom_ids[i] * uniforms.n_atom_types + id_b;
            let diff = ref_pos_a[i].xyz - pos_b;
            let r = sqrt(dot(diff, diff));
            total_energy += spline_energy(pair_idx, r) + coulomb_energy(pair_idx, r);
        }
    }

    energies[pose_idx] = total_energy;
}

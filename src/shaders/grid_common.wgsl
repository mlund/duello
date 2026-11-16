// Lattice uniforms shared by the grid build and pose kernels.

struct Grid {
    origin: vec4<f32>,
    dims: vec4<u32>,   // nodes along x, y, z; w = number of map slots
    spacing: f32,
    n_atom_types: u32,
    energy_cap: f32,
    n_atoms_a: u32,
    // Short-range energy is zero beyond this distance from an A bead.
    reach: f32,
    // Scalars, not vec3: a vec3 would align to 16 and grow the struct to 80 bytes.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

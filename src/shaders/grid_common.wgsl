// Lattice uniforms shared by the grid build and pose kernels.
// Prepended at runtime: `const BRICK` and `const F16_MAX` (maps::layout_consts).

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

// Maps are stored in bricks of BRICK³ nodes, each axis padded to a multiple of
// BRICK, so a lookup's eight corners mostly share one brick. At BRICK = 2 (32
// bytes) a cell starting on even nodes is a single read; measured faster than
// 4 or 8 on the lysozyme scan. The padding is never read.

fn bricks_per_axis() -> vec3<u32> {
    return (grid.dims.xyz + BRICK - 1u) / BRICK;
}

fn map_index(slot: u32, node: vec3<u32>) -> u32 {
    let nb = bricks_per_axis();
    let brick = node / BRICK;
    let local = node % BRICK;
    let volume = BRICK * BRICK * BRICK;
    let brick_id = (brick.z * nb.y + brick.y) * nb.x + brick.x;
    let offset = local.x + BRICK * (local.y + BRICK * local.z);
    return (slot * nb.x * nb.y * nb.z + brick_id) * volume + offset;
}

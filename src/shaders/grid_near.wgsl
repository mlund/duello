// A's cell list (see near.rs): every A bead within the near range of a point
// lies in the 3×3×3 cells around it. Beads are sorted by cell, x fastest, so
// each row of up to three cells is one span of the bead list.
//
// Needs `cell_start: array<u32>` bound by the including kernel.

struct Cells {
    origin_edge: vec4<f32>, // lattice origin, cell edge
    dims: vec4<u32>,        // cells along x, y, z
}

// The 3×3×3 cells around `p`, clamped to the lattice, inclusive; empty when
// `p` is beyond range of every cell. Measured faster than padding the
// lattice with empty cells to avoid the clamp.
struct CellBox {
    lo: vec3<u32>,
    hi: vec3<u32>,
    empty: bool,
}

fn near_box(cells: Cells, p: vec3<f32>) -> CellBox {
    let c = vec3<i32>(floor((p - cells.origin_edge.xyz) / cells.origin_edge.w));
    let lo = max(c - 1, vec3<i32>(0));
    let hi = min(c + 1, vec3<i32>(cells.dims.xyz) - 1);
    // hi < 0 only for an empty box; clamped so the cast stays defined.
    return CellBox(vec3<u32>(lo), vec3<u32>(max(hi, vec3<i32>(0))), any(lo > hi));
}

// Bead-list span [x, y) of row (y, z) of `box`.
fn row_span(cells: Cells, box: CellBox, y: u32, z: u32) -> vec2<u32> {
    let row = (z * cells.dims.y + y) * cells.dims.x;
    return vec2<u32>(cell_start[row + box.lo.x], cell_start[row + box.hi.x + 1u]);
}

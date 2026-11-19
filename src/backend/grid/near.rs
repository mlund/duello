//! A's cell list: which A beads can lie within the near range of a point.

use super::domain::bounds;
use crate::steric::counting_sort;
use crate::Vector3;
use bytemuck::{Pod, Zeroable};

/// `near_box` and `row_span` in WGSL; the kernel binds `cell_start`.
pub(super) const WGSL: &str = include_str!("../../shaders/grid_near.wgsl");

/// `Cells` in grid_near.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CellsUniform {
    /// Lattice origin in xyz, cell edge in w.
    origin_edge: [f32; 4],
    /// Cells along x, y, z, padding included; w unused.
    dims: [u32; 4],
}

/// Edge slack (Å), so that f32 rounding of a point's cell on the GPU can't
/// leave a bead within range two cells away.
const SLACK: f64 = 0.01;

/// A's beads binned into cubic cells of edge ≥ the near range, so every bead
/// within range of a point lies in the 3×3×3 cells around it.
pub(super) struct NearCells {
    origin: Vector3,
    edge: f64,
    dims: [u32; 3],
    /// Bead indices sorted by cell; x-neighbours are adjacent, so each row
    /// of three cells is one contiguous span.
    order: Vec<u32>,
    /// Start of each cell in `order`, plus the end.
    start: Vec<u32>,
}

impl NearCells {
    /// # Panics
    /// If `points` is empty or `range` is not positive.
    pub fn new(points: &[Vector3], range: f64) -> Self {
        assert!(!points.is_empty() && range > 0.0);
        let edge = range + SLACK;
        let (lo, hi) = bounds(points);
        let dims = std::array::from_fn(|k| ((hi[k] - lo[k]) / edge).floor() as u32 + 1);
        let mut cells = Self {
            origin: lo,
            edge,
            dims,
            order: Vec::new(),
            start: Vec::new(),
        };
        let keys: Vec<usize> = points.iter().map(|p| cells.flat(cells.cell(p))).collect();
        let (start, order) = counting_sort(&keys, dims.iter().map(|&n| n as usize).product());
        cells.start = start.into_iter().map(|i| i as u32).collect();
        cells.order = order.into_iter().map(|i| i as u32).collect();
        cells
    }

    /// Point indices in cell order: upload the points in this order.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Where each cell starts in [`Self::order`], plus the end.
    pub fn starts(&self) -> &[u32] {
        &self.start
    }

    pub fn uniform(&self) -> CellsUniform {
        let o = self.origin;
        let [nx, ny, nz] = self.dims;
        CellsUniform {
            origin_edge: [o.x as f32, o.y as f32, o.z as f32, self.edge as f32],
            dims: [nx, ny, nz, 0],
        }
    }

    /// Cell coordinates of `p`, unclamped: may lie outside the lattice.
    fn cell(&self, p: &Vector3) -> [i64; 3] {
        std::array::from_fn(|k| ((p[k] - self.origin[k]) / self.edge).floor() as i64)
    }

    fn flat(&self, [x, y, z]: [i64; 3]) -> usize {
        let [nx, ny, _] = self.dims.map(|n| n as usize);
        (z as usize * ny + y as usize) * nx + x as usize
    }

    /// Indices of every point within range of `p`, and some beyond; as
    /// `near_box` and `row_span` in WGSL.
    #[cfg(test)]
    pub fn beads_near(&self, p: &Vector3) -> Box<dyn Iterator<Item = usize> + '_> {
        let c = self.cell(p);
        let lo: [i64; 3] = std::array::from_fn(|k| (c[k] - 1).max(0));
        let hi: [i64; 3] = std::array::from_fn(|k| (c[k] + 1).min(i64::from(self.dims[k]) - 1));
        if (0..3).any(|k| lo[k] > hi[k]) {
            return Box::new(std::iter::empty()); // beyond range of every cell
        }
        let rows = itertools::iproduct!(lo[2]..=hi[2], lo[1]..=hi[1]);
        Box::new(rows.flat_map(move |(z, y)| {
            let first = self.start[self.flat([lo[0], y, z])] as usize;
            let end = self.start[self.flat([hi[0], y, z]) + 1] as usize;
            self.order[first..end].iter().map(|&i| i as usize)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_support::lysozyme;
    use rand::{Rng, SeedableRng};

    /// Every bead within range is listed, at random points in and around A,
    /// including well outside its cells.
    #[test]
    fn lists_every_bead_within_range() {
        const RANGE: f64 = 11.6;
        let a = lysozyme().ref_a.pos;
        let cells = NearCells::new(&a, RANGE);
        let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
        let (mut listed, mut within) = (0, 0);
        for _ in 0..2000 {
            let p = Vector3::from_fn(|_, _| rng.gen_range(-45.0..45.0));
            let mut near = std::collections::HashSet::new();
            for i in cells.beads_near(&p) {
                assert!(near.insert(i), "bead {i} listed twice at {p}");
            }
            for (i, q) in a.iter().enumerate() {
                if (q - p).norm() < RANGE {
                    within += 1;
                    assert!(near.contains(&i), "bead {i} missing at {p}");
                }
            }
            listed += near.len();
        }
        assert!(within > 1000, "too few points near A: {within}");
        // Cells must actually cull: the 27-cell box is ~6× the range sphere.
        assert!(
            listed < 10 * within,
            "listed {listed} for {within} within range"
        );
    }
}

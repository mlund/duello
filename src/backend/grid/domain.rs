//! Box of grid nodes covering everything within reach of molecule A.

use crate::Vector3;

/// Axis-aligned node lattice on multiples of the spacing; shared by map
/// building and lookup so both index alike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridDomain {
    origin: Vector3,
    dims: [u32; 3],
    spacing: f64,
}

impl GridDomain {
    /// Smallest lattice at `spacing` covering every point within `reach` of `points`.
    ///
    /// # Panics
    /// If `points` is empty or `spacing` is not positive.
    pub fn covering(points: &[Vector3], reach: f64, spacing: f64) -> Self {
        assert!(!points.is_empty(), "grid needs at least one point");
        assert!(
            spacing > 0.0,
            "grid spacing must be positive, got {spacing}"
        );
        let pad = Vector3::repeat(reach);
        let (lo, hi) = points.iter().fold(
            (Vector3::repeat(f64::MAX), Vector3::repeat(f64::MIN)),
            |(lo, hi), p| (lo.inf(p), hi.sup(p)),
        );
        // Snapping to multiples of `spacing` makes node positions independent of
        // the point set, so lattices at one spacing line up.
        let lo = (lo - pad).map(|x| (x / spacing).floor() * spacing);
        let hi = hi + pad;
        let dims = std::array::from_fn(|k| cells_spanning(lo[k], hi[k], spacing) + 1);
        Self {
            origin: lo,
            dims,
            spacing,
        }
    }

    /// Node count along x, y, z.
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// Position of node (0, 0, 0).
    pub fn origin(&self) -> Vector3 {
        self.origin
    }

    /// Total node count.
    pub fn n_voxels(&self) -> usize {
        self.dims.iter().map(|&n| n as usize).product()
    }

    pub fn spacing(&self) -> f64 {
        self.spacing
    }

    /// True if `p` lies inside the node lattice. The top face counts, so a
    /// trilinear lookup must clamp its cell index to `dims - 2` there.
    pub fn contains(&self, p: &Vector3) -> bool {
        (0..3).all(|k| (self.origin[k]..=self.top(k)).contains(&p[k]))
    }

    /// Coordinate of the last node along axis `k`.
    fn top(&self, k: usize) -> f64 {
        self.origin[k] + self.spacing * f64::from(self.dims[k] - 1)
    }
}

/// Whole cells from `lo` reaching at least `hi`. Checked the way the top node
/// is computed, since `ceil` alone can leave it an ulp short of `hi`.
fn cells_spanning(lo: f64, hi: f64, spacing: f64) -> u32 {
    let cells = ((hi - lo) / spacing).ceil();
    let cells = if lo + cells * spacing < hi {
        cells + 1.0
    } else {
        cells
    };
    cells as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point_box() {
        let d = GridDomain::covering(&[Vector3::zeros()], 2.0, 1.0);
        assert_eq!(d.dims(), [5, 5, 5]);
        assert_eq!(d.origin(), Vector3::new(-2.0, -2.0, -2.0));
    }

    #[test]
    fn everything_within_reach_is_contained() {
        let points = [
            Vector3::new(1.0, -3.0, 0.5),
            Vector3::new(-4.2, 2.0, 7.0),
            Vector3::new(0.3, 0.3, -6.1),
        ];
        let reach = 2.5;
        let d = GridDomain::covering(&points, reach, 0.7);
        let (x, y, z) = (Vector3::x(), Vector3::y(), Vector3::z());
        let offsets = [x, -x, y, -y, z, -z].map(|u| reach * u);
        for p in points.iter().flat_map(|a| offsets.map(|o| a + o)) {
            assert!(d.contains(&p), "{p:?} not in {d:?}");
        }
    }

    #[test]
    fn partial_cells_round_up() {
        let points = [Vector3::zeros(), Vector3::new(4.2, 0.0, 0.0)];
        let d = GridDomain::covering(&points, 0.0, 1.0);
        assert_eq!(d.dims(), [6, 1, 1]);
        assert!(d.contains(&points[1]));
    }

    /// Lattices at one spacing share nodes, so a point at k·h is always a node.
    #[test]
    fn nodes_sit_on_multiples_of_spacing() {
        let d = GridDomain::covering(&[Vector3::new(0.3, 0.0, 0.0)], 1.0, 0.5);
        assert_eq!(d.origin(), Vector3::new(-1.0, -1.0, -1.0));
        assert!(d.contains(&Vector3::new(1.3, 0.0, 0.0)));
    }
}

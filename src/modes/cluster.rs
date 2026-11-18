//! Greedy clustering of poses into binding modes by B-bead RMSD.
//!
//! Every attractive state of a fine table (~10⁷) must find its mode, so the
//! RMSD comes from rigid-body moments in O(1) rather than from the beads, and
//! modes are found through a grid of their centroids.

use super::BindingMode;
use crate::archive::CgBody;
use crate::nbody::Pose;
use crate::poses::TablePose;
use crate::Vector3;
use nalgebra::Matrix3;
use std::collections::HashMap;

/// Founds modes from the strongest states. Only founding happens here, so
/// the one pass that later adds every state counts each exactly once.
pub(super) struct Founders {
    index: Index,
    poses: Vec<TablePose>,
}

impl Founders {
    pub fn new(bodies: &[CgBody; 2], homodimer: bool, rmsd: f64) -> Self {
        Self {
            index: Index::new(&bodies[1], homodimer, rmsd),
            poses: Vec::new(),
        }
    }

    /// For poses taken strongest first: founds a mode unless one is within
    /// the RMSD, so each representative is its family's largest term.
    pub fn found_if_new(&mut self, pose: TablePose) {
        let b = pose.b_pose();
        let view = View::new(&b, &self.index.shape);
        let near = self.index.near(view.centroid, self.index.rmsd);
        if self.index.first_within(&view, &near).is_none() {
            self.index.insert(self.poses.len(), &b, view);
            self.poses.push(pose);
        }
    }

    /// Fixes the modes, so that states can join them but not found more.
    pub fn freeze(self) -> Families {
        let modes = self.poses.into_iter().map(|pose| BindingMode {
            pose,
            b2: 0.0,
            share: 0.0,
            n_poses: 0,
        });
        Families {
            index: self.index,
            modes: modes.collect(),
            near: None,
        }
    }
}

/// Fixed modes that states join.
pub(super) struct Families {
    index: Index,
    modes: Vec<BindingMode>,
    /// Views near the last B position joined, which every orientation at the
    /// same (R, vertex) shares
    near: Option<(Vector3, Vec<usize>)>,
}

impl Families {
    /// Adds the state at `b` to the first mode within the RMSD, if any; a
    /// state like none is attraction no mode explains.
    pub fn join(&mut self, b: &Pose, c: f64) {
        let view = View::new(b, &self.index.shape);
        let near = match self.near.take() {
            Some((at, near)) if at == b.position => near,
            _ => self.index.near_position(b.position),
        };
        if let Some(i) = self.index.first_within(&view, &near) {
            self.modes[i].b2 += c;
            self.modes[i].n_poses += 1;
        }
        self.near = Some((b.position, near));
    }

    pub fn into_modes(self) -> Vec<BindingMode> {
        self.modes
    }
}

/// Where the modes' views sit, by the cell of their centroids.
struct Index {
    shape: Shape,
    homodimer: bool,
    rmsd: f64,
    /// (mode index, view) in mode order; a homodimer mode has two
    views: Vec<(usize, View)>,
    /// Indices into `views`
    grid: HashMap<[i64; 3], Vec<usize>>,
}

impl Index {
    fn new(body: &CgBody, homodimer: bool, rmsd: f64) -> Self {
        Self {
            shape: Shape::of(body),
            homodimer,
            rmsd,
            views: Vec::new(),
            grid: HashMap::new(),
        }
    }

    /// The views of mode `mode`, whose B sits at `b`.
    fn insert(&mut self, mode: usize, b: &Pose, view: View) {
        self.insert_view(mode, view);
        if self.homodimer {
            // A as B sees it: swapping identical molecules is the same complex,
            // so B's shape stands for A's too.
            self.insert_view(mode, View::new(&seen_from(b), &self.shape));
        }
    }

    fn insert_view(&mut self, mode: usize, view: View) {
        let cell = self.cell(view.centroid);
        self.grid.entry(cell).or_default().push(self.views.len());
        self.views.push((mode, view));
    }

    /// The strongest mode with a view within the RMSD of `view`. `near` is
    /// in view order, which is mode order, so the first match is the one.
    fn first_within(&self, view: &View, near: &[usize]) -> Option<usize> {
        let max_msd = self.rmsd * self.rmsd;
        near.iter()
            .map(|&v| &self.views[v])
            // The RMSD is at least the centroid distance, which costs far less.
            .filter(|(_, other)| (view.centroid - other.centroid).norm_squared() <= max_msd)
            .find(|(_, other)| view.msd(other, &self.shape) <= max_msd)
            .map(|&(mode, _)| mode)
    }

    /// Views that any orientation of B at `position` may be within the RMSD
    /// of: its placed centroid lies within |centroid| of `position`.
    fn near_position(&self, position: Vector3) -> Vec<usize> {
        let reach = self.rmsd + self.shape.centroid.norm();
        let mut near = self.near(position, reach);
        near.retain(|&v| (self.views[v].1.centroid - position).norm() <= reach);
        near
    }

    /// Views whose centroid may lie within `radius` of `point`, in view order.
    fn near(&self, point: Vector3, radius: f64) -> Vec<usize> {
        let reach = (radius / self.cell_size()).ceil() as i64;
        let [x, y, z] = self.cell(point);
        let mut near: Vec<usize> = offsets(reach)
            .filter_map(|[dx, dy, dz]| self.grid.get(&[x + dx, y + dy, z + dz]))
            .flatten()
            .copied()
            .collect();
        near.sort_unstable();
        near
    }

    fn cell(&self, p: Vector3) -> [i64; 3] {
        p.map(|x| (x / self.cell_size()).floor() as i64).into()
    }

    /// One RMSD, so a lookup spans few cells; bounded below for tiny RMSDs.
    fn cell_size(&self) -> f64 {
        self.rmsd.max(0.5)
    }
}

/// Every cell offset within `reach` along each axis.
fn offsets(reach: i64) -> impl Iterator<Item = [i64; 3]> {
    let axis = move || -reach..=reach;
    axis().flat_map(move |x| axis().flat_map(move |y| axis().map(move |z| [x, y, z])))
}

/// The partner's placement in the frame of a body placed at `pose`.
pub(super) fn seen_from(pose: &Pose) -> Pose {
    let orientation = pose.orientation.inverse();
    Pose {
        position: orientation * -pose.position,
        orientation,
    }
}

/// What the RMSD between two placements of one rigid body depends on.
struct Shape {
    /// Mean bead position (Å)
    centroid: Vector3,
    /// Mean of x·xᵀ over beads (Å²)
    second: Matrix3<f64>,
}

impl Shape {
    fn of(body: &CgBody) -> Self {
        let n = body.positions.len() as f64;
        let beads = || body.positions.iter().map(|&p| Vector3::from(p));
        Self {
            centroid: beads().sum::<Vector3>() / n,
            second: beads().map(|x| x * x.transpose()).sum::<Matrix3<f64>>() / n,
        }
    }
}

/// A placement of the body, with its placed centroid for the grid.
struct View {
    rotation: Matrix3<f64>,
    position: Vector3,
    centroid: Vector3,
}

impl View {
    fn new(pose: &Pose, shape: &Shape) -> Self {
        let rotation = pose.orientation.to_rotation_matrix().into_inner();
        Self {
            rotation,
            position: pose.position,
            centroid: rotation * shape.centroid + pose.position,
        }
    }

    /// Mean of |(R₁x + s₁) − (R₂x + s₂)|² over the beads x, exactly:
    /// |Δs|² + 2Δs·ΔR·x̄ + tr(ΔRᵀΔR·⟨xxᵀ⟩).
    fn msd(&self, other: &Self, shape: &Shape) -> f64 {
        let dr = self.rotation - other.rotation;
        let ds = self.position - other.position;
        ds.norm_squared()
            + 2.0 * ds.dot(&(dr * shape.centroid))
            + (dr.transpose() * dr * shape.second).trace()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbody::trajectory::expand;
    use crate::UnitQuaternion;
    use rand::{Rng, SeedableRng};

    /// The closed form against the beads it stands for, on rotated poses of
    /// an asymmetric body whose bead mean is off the origin.
    #[test]
    fn the_moment_msd_is_the_bead_msd() {
        let mut body = crate::archive::tests::body("LYS");
        body.positions = vec![
            [2.0, 0.5, 0.0],
            [0.0, 1.0, -1.0],
            [0.3, 0.0, -3.0],
            [4.0, 4.0, 1.0],
        ];
        let shape = Shape::of(&body);
        let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
        let mut pose = || Pose {
            position: Vector3::from_fn(|_, _| rng.gen_range(-5.0..5.0)),
            orientation: UnitQuaternion::from_scaled_axis(Vector3::from_fn(|_, _| {
                rng.gen_range(-3.0..3.0)
            })),
        };
        for _ in 0..20 {
            let (a, b) = (pose(), pose());
            let beads = expand(&body, &a).into_iter().zip(expand(&body, &b));
            let expected = beads.map(|(x, y)| (x - y).norm_squared()).sum::<f64>() / 4.0;
            let found = View::new(&a, &shape).msd(&View::new(&b, &shape), &shape);
            approx::assert_relative_eq!(found, expected, max_relative = 1e-9);
        }
    }
}

//! Many-body Metropolis simulation of rigid bodies interacting through a 6D table.

mod energy;

pub use energy::{PairEnergy, PairValue};

use crate::{UnitQuaternion, Vector3};

/// Position and orientation of one rigid body; its beads follow from the
/// reference structure, which keeps the state to seven numbers per body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: Vector3,
    pub orientation: UnitQuaternion,
}

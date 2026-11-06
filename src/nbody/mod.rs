//! Many-body Metropolis simulation of rigid bodies interacting through a 6D table.

#[cfg(test)]
mod acceptance;
mod energy;
mod energy_stats;
mod equilibration;
mod sampler;
mod scattering;
#[cfg(test)]
mod test_tables;
pub mod trajectory;

pub use energy::{PairEnergy, PairValue};
pub use energy_stats::{EnergyStats, EnergySummary};
pub use equilibration::{Criteria, Equilibration, Status};
pub use sampler::{Frame, MoveCounter, MoveStats, Sampler, SamplerSpec};
pub use scattering::{Device, Scattering, ScatteringCurve};

use crate::{UnitQuaternion, Vector3};

/// Position and orientation of one rigid body; its beads follow from the
/// reference structure, which keeps the state to seven numbers per body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: Vector3,
    pub orientation: UnitQuaternion,
}

/// Side length in Å of the cubic box holding `n_bodies` at `molarity` (mol/L).
pub fn box_length(n_bodies: usize, molarity: f64) -> f64 {
    let density =
        molarity * physical_constants::AVOGADRO_CONSTANT * crate::LITER_PER_CUBIC_ANGSTROM;
    (n_bodies as f64 / density).cbrt()
}

#[cfg(test)]
mod tests {
    #[test]
    fn box_holds_the_requested_concentration() {
        // 200 bodies at 5 mM: ρ = 3.0110704e-6 Å⁻³, so L = (200/ρ)^(1/3).
        approx::assert_relative_eq!(super::box_length(200, 0.005), 404.98, max_relative = 1e-4);
    }
}

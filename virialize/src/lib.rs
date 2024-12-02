pub use nalgebra::{Matrix3, UnitQuaternion, Vector3};
pub mod anglescan;
pub mod energy;
pub mod icoscan;
pub mod icotable;
pub mod report;
pub mod structure;
pub mod table;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
use physical_constants::MOLAR_GAS_CONSTANT;
use std::f64::consts::PI;
use std::iter::Sum;
use std::ops::{Add, Neg};

extern crate flate2;

pub use anglescan::{
    make_fibonacci_sphere, make_icosphere, make_icosphere_vertices, TwobodyAngles,
};

/// RMSD angle between two quaternion rotations
///
/// The root-mean-square deviation (RMSD) between two quaternion rotations is
/// defined as the square of the angle between the two quaternions.
///
/// - <https://fr.mathworks.com/matlabcentral/answers/415936-angle-between-2-quaternions>
/// - <https://github.com/charnley/rmsd>
/// - <https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.20296>
/// - <https://www.ams.stonybrook.edu/~coutsias/papers/2004-rmsd.pdf>
pub fn rmsd_angle(q1: &UnitQuaternion<f64>, q2: &UnitQuaternion<f64>) -> f64 {
    // let q = q1 * q2.inverse();
    // q.angle().powi(2)
    q1.angle_to(q2).powi(2)
}

#[allow(non_snake_case)]
pub fn rmsd2(Q: &UnitQuaternion<f64>, inertia: &Matrix3<f64>, total_mass: f64) -> f64 {
    let q = Q.vector();
    4.0 / total_mass * (q.transpose() * inertia * q)[0]
}

/// Structure to store energy samples
#[derive(Debug, Default, Clone)]
pub struct Sample {
    /// Number of samples
    n: u64,
    /// Thermal energy, RT in kJ/mol
    pub thermal_energy: f64,
    /// Boltzmann weighted energy, U * exp(-U/kT)
    pub mean_energy: f64,
    /// Boltzmann factored energy, exp(-U/kT)
    pub exp_energy: f64,
}

impl Sample {
    /// New from energy in kJ/mol and temperature in K
    pub fn new(energy: f64, temperature: f64) -> Self {
        let thermal_energy = MOLAR_GAS_CONSTANT * temperature * 1e-3; // kJ/mol
        let exp_energy = (-energy / thermal_energy).exp();
        Self {
            n: 1,
            thermal_energy,
            mean_energy: energy * exp_energy,
            exp_energy,
        }
    }
    /// Mean energy (kJ/mol)
    pub fn mean_energy(&self) -> f64 {
        self.mean_energy / self.exp_energy
    }
    /// Free energy (kJ / mol)
    pub fn free_energy(&self) -> f64 {
        (self.exp_energy / self.n as f64).ln().neg() * self.thermal_energy
    }
}

impl Sum for Sample {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Sample::default(), |sum, s| sum + s)
    }
}

impl Add for Sample {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            n: self.n + other.n,
            thermal_energy: f64::max(self.thermal_energy, other.thermal_energy),
            mean_energy: self.mean_energy + other.mean_energy,
            exp_energy: self.exp_energy + other.exp_energy,
        }
    }
}

/// Converts Cartesian coordinates to spherical coordinates (r, theta, phi)
/// where:
/// - r is the radius
/// - theta is the polar angle (0..pi)
/// - phi is the azimuthal angle (0..2pi)
pub fn to_spherical(cartesian: &Vector3<f64>) -> (f64, f64, f64) {
    let r = cartesian.norm();
    let theta = (cartesian.z / r).acos();
    let phi = cartesian.y.atan2(cartesian.x);
    if phi < 0.0 {
        (r, theta, phi + 2.0 * PI)
    } else {
        (r, theta, phi)
    }
}

/// Converts spherical coordinates (r, theta, phi) to Cartesian coordinates
/// where:
/// - r is the radius
/// - theta is the polar angle (0..pi)
/// - phi is the azimuthal angle (0..2pi)
pub fn to_cartesian(r: f64, theta: f64, phi: f64) -> Vector3<f64> {
    let (theta_sin, theta_cos) = theta.sin_cos();
    let (phi_sin, phi_cos) = phi.sin_cos();
    Vector3::new(
        r * theta_sin * phi_cos,
        r * theta_sin * phi_sin,
        r * theta_cos,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use iter_num_tools::arange;

    #[test]
    fn test_spherical_cartesian_conversion() {
        const ANGLE_TOL: f64 = 1e-6;
        // Skip theta = 0 as phi is undefined
        for theta in arange(0.00001..PI, 0.01) {
            for phi in arange(0.0..2.0 * PI, 0.01) {
                let cartesian = to_cartesian(1.0, theta, phi);
                let (_, theta_converted, phi_converted) = to_spherical(&cartesian);
                assert_relative_eq!(theta, theta_converted, epsilon = ANGLE_TOL);
                assert_relative_eq!(phi, phi_converted, epsilon = ANGLE_TOL);
            }
        }
    }
}

// Copyright 2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

pub mod energy;
mod fibonacci;
pub mod icoscan;
mod icosphere;
pub mod icotable;
pub mod report;
mod sample;
pub mod structure;
pub mod table;
mod vertex;
mod virial;
pub use fibonacci::make_fibonacci_sphere;
pub use sample::Sample;
pub use vertex::*;
pub use virial::VirialCoeff;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
use std::f64::consts::PI;

extern crate flate2;

pub type IcoSphere = hexasphere::Subdivided<(), hexasphere::shapes::IcoSphereBase>;
pub type Matrix3 = nalgebra::Matrix3<f64>;
pub type Vector3 = nalgebra::Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;

pub use icosphere::*;

/// RMSD angle between two quaternion rotations
///
/// The root-mean-square deviation (RMSD) between two quaternion rotations is
/// defined as the square of the angle between the two quaternions.
///
/// - <https://fr.mathworks.com/matlabcentral/answers/415936-angle-between-2-quaternions>
/// - <https://github.com/charnley/rmsd>
/// - <https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.20296>
/// - <https://www.ams.stonybrook.edu/~coutsias/papers/2004-rmsd.pdf>
pub(crate) fn _rmsd_angle(q1: &UnitQuaternion, q2: &UnitQuaternion) -> f64 {
    // let q = q1 * q2.inverse();
    // q.angle().powi(2)
    q1.angle_to(q2).powi(2)
}

#[allow(non_snake_case)]
pub(crate) fn _rmsd2(Q: &UnitQuaternion, inertia: &Matrix3, total_mass: f64) -> f64 {
    let q = Q.vector();
    4.0 / total_mass * (q.transpose() * inertia * q)[0]
}

/// Converts Cartesian coordinates to spherical coordinates (r, theta, phi)
/// where:
/// - r is the radius
/// - theta is the polar angle (0..pi)
/// - phi is the azimuthal angle (0..2pi)
pub fn to_spherical(cartesian: &Vector3) -> (f64, f64, f64) {
    let r = cartesian.norm();
    let theta = (cartesian.z / r).acos();
    let phi = cartesian.y.atan2(cartesian.x);
    // Ensure phi is in the range [0..2pi)
    let phi = (phi + 2.0 * PI) % (2.0 * PI);
    (r, theta, phi)
}

/// Converts spherical coordinates (r, theta, phi) to Cartesian coordinates
/// where:
/// - r is the radius
/// - theta is the polar angle (0..pi)
/// - phi is the azimuthal angle (0..2pi)
pub fn to_cartesian(r: f64, theta: f64, phi: f64) -> Vector3 {
    let (theta_sin, theta_cos) = theta.sin_cos();
    let (phi_sin, phi_cos) = phi.sin_cos();
    Vector3::new(theta_sin * phi_cos, theta_sin * phi_sin, theta_cos).scale(r)
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

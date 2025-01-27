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
mod spherical;
pub mod structure;
pub mod table;
mod vertex;
mod virial;
pub use fibonacci::make_fibonacci_sphere;
pub use sample::Sample;
pub use spherical::SphericalCoord;
pub use vertex::*;
pub use virial::VirialCoeff;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;

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

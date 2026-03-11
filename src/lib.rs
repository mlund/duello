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

pub mod backend;
pub mod energy;
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

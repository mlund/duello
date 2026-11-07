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

pub mod archive;
pub mod backend;
#[cfg(feature = "cli")]
pub mod diffusion;
pub mod energy;
pub mod icoscan;
pub mod molecule;
pub mod nbody;
pub mod report;
mod sample;
pub mod solution;
pub mod structure;
pub mod system;
#[cfg(feature = "cli")]
pub mod trajectory;
mod virial;
pub mod web_api;

pub use sample::Sample;
/// Shown by front-ends so a result can be traced to the library that produced it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub use virial::VirialCoeff;

/// Converts number densities in Å⁻³ to and from mol/L.
pub(crate) const LITER_PER_CUBIC_ANGSTROM: f64 = 1e-27;

/// Particles per Å³ at `molarity` mol/L.
pub(crate) fn number_density(molarity: f64) -> f64 {
    molarity * physical_constants::AVOGADRO_CONSTANT * LITER_PER_CUBIC_ANGSTROM
}

/// kT in kJ/mol, the energy unit of all tables.
pub fn thermal_energy(temperature: f64) -> f64 {
    physical_constants::MOLAR_GAS_CONSTANT * temperature * 1e-3
}

#[macro_use]
extern crate log;

// Icosphere table types, coordinate transforms, and nalgebra aliases live in
// the shared `icotable` crate to avoid duplication with Faunus.
pub use icotable::{
    make_icosphere_vertices, Adaptive3DBuilder, AdaptiveBuilder, IcoTable2D, PaddedTable,
    SphericalCoord, Subdivision, Table3DAdaptive, Table6D, Table6DAdaptive, UnitQuaternion,
    Vector3,
};

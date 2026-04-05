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
pub mod diffusion;
pub mod energy;
pub mod icoscan;
pub mod loader;
pub mod report;
mod sample;
pub mod structure;
mod virial;

pub use sample::Sample;
pub use virial::VirialCoeff;

#[macro_use]
extern crate log;

// Icosphere table types, coordinate transforms, and nalgebra aliases live in
// the shared `icotable` crate to avoid duplication with Faunus.
pub use icotable::{
    make_icosphere_vertices, Adaptive3DBuilder, AdaptiveBuilder, IcoTable2D, PaddedTable,
    SphericalCoord, Table3DAdaptive, Table6D, Table6DAdaptive, UnitQuaternion, Vector3,
};

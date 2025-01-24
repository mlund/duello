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

use crate::{IcoSphere, Vector3};
use get_size::GetSize;
use hexasphere::AdjacencyBuilder;
use itertools::Itertools;
use std::sync::OnceLock;

/// Structure for storing vertex positions and neighbors
#[derive(Clone, GetSize, Debug)]
pub struct Vertices {
    /// 3D coordinates of the vertex on a unit sphere
    #[get_size(size = 24)]
    pub pos: Vector3,
    /// Indices of neighboring vertices
    pub neighbors: Vec<u16>,
}

/// Extract vertices and neightbourlists from an icosphere
pub fn make_vertices(icosphere: &IcoSphere) -> Vec<Vertices> {
    let vertex_positions = icosphere
        .raw_points()
        .iter()
        .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64));

    // Get neighborlist for each vertex
    let indices = icosphere.get_all_indices();
    let mut builder = AdjacencyBuilder::new(icosphere.raw_points().len());
    builder.add_indices(indices.as_slice());
    let neighbors = builder.finish().iter().map(|i| i.to_vec()).collect_vec();

    assert!(vertex_positions.len() == neighbors.len());

    vertex_positions
        .zip(neighbors)
        .map(|(pos, neighbors)| Vertices {
            pos,
            neighbors: neighbors.iter().map(|i| *i as u16).collect_vec(),
        })
        .collect()
}

/// Struct representing data stored at vertices on an icosphere
///
/// Interior mutability of vertex associated data is enabled using `std::sync::OnceLock`.
/// This allows for data to be set once and then read multiple times.
#[derive(Clone, GetSize)]
pub struct DataOnVertex<T: Clone + GetSize> {
    /// Data associated with the vertex
    #[get_size(size_fn = oncelock_size_helper)]
    pub data: OnceLock<T>,
}

fn oncelock_size_helper<T: GetSize>(value: &OnceLock<T>) -> usize {
    std::mem::size_of::<OnceLock<T>>() + value.get().map(|v| v.get_heap_size()).unwrap_or(0)
}

impl<T: Clone + GetSize> DataOnVertex<T> {
    /// Construct a new vertex where data is *locked* to fixed value
    pub fn from(data: T) -> Self {
        Self {
            data: OnceLock::from(data),
        }
    }
    /// Construct new uninitialized data
    pub fn uninitialized() -> Self {
        Self {
            data: OnceLock::new(),
        }
    }
}

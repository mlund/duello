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

use super::anglescan::*;
use get_size::GetSize;
use hexasphere::{shapes::IcoSphereBase, AdjacencyBuilder, Subdivided};
use itertools::Itertools;
use std::sync::OnceLock;

/// Structure for storing vertex positions and neighbors
#[derive(Clone, GetSize)]
pub struct VertexPosAndNeighbors {
    /// 3D coordinates of the vertex on a unit sphere
    #[get_size(size = 24)]
    pub pos: Vector3,
    /// Indices of neighboring vertices
    pub neighbors: Vec<u16>,
}

pub fn make_vertex_vec(icosphere: &Subdivided<(), IcoSphereBase>) -> Vec<VertexPosAndNeighbors> {
    let indices = icosphere.get_all_indices();
    let mut builder = AdjacencyBuilder::new(icosphere.raw_points().len());
    builder.add_indices(indices.as_slice());
    let neighbors = builder.finish().iter().map(|i| i.to_vec()).collect_vec();
    let vertex_positions = icosphere
        .raw_points()
        .iter()
        .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64));

    assert!(vertex_positions.len() == neighbors.len());

    vertex_positions
        .zip(neighbors)
        .map(|(pos, neighbors)| VertexPosAndNeighbors {
            pos,
            neighbors: neighbors.iter().map(|i| *i as u16).collect_vec(),
        })
        .collect()
}

/// Struct representing a vertex in the icosphere
///
/// Interior mutability of vertex associated data is enabled using `std::sync::OnceLock`.
#[derive(Clone, GetSize)]
pub struct DataOnVertex<T: Clone + GetSize> {
    /// 3D coordinates of the vertex on a unit sphere
    #[get_size(size = 24)]
    pub pos: Vector3,
    /// Data associated with the vertex
    #[get_size(size_fn = oncelock_size_helper)]
    pub data: OnceLock<T>,
    /// Indices of neighboring vertices
    pub neighbors: Vec<u16>,
}

fn oncelock_size_helper<T: GetSize>(value: &OnceLock<T>) -> usize {
    std::mem::size_of::<OnceLock<T>>() + value.get().map(|v| v.get_heap_size()).unwrap_or(0)
}

impl<T: Clone + GetSize> DataOnVertex<T> {
    /// Construct a new vertex where data is *locked* to fixed value
    pub fn with_fixed_data(pos: Vector3, data: T, neighbors: Vec<u16>) -> Self {
        let vertex = Self::without_data(pos, neighbors);
        let _ = vertex.data.set(data);
        vertex
    }

    /// Construct a new vertex; write-once data is left empty and can/should be set later
    pub fn without_data(pos: Vector3, neighbors: Vec<u16>) -> Self {
        assert!(matches!(neighbors.len(), 5 | 6));
        Self {
            pos,
            data: OnceLock::<T>::new(),
            neighbors,
        }
    }
}

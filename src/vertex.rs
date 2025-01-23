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
use std::sync::OnceLock;

/// Struct representing a vertex in the icosphere
///
/// Interior mutability of vertex associated data is enabled using `std::sync::OnceLock`.
#[derive(Clone, GetSize)]
pub struct Vertex<T: Clone + GetSize> {
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

impl<T: Clone + GetSize> Vertex<T> {
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

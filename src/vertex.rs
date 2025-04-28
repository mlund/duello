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

/// Structure for storing vertex positions and neighbors
#[derive(Clone, GetSize, Debug)]
pub struct Vertex {
    /// 3D coordinates of the vertex on a unit sphere
    #[get_size(size = 24)]
    pub pos: Vector3,
    /// Indices of neighboring vertices
    pub neighbors: Vec<u16>,
}

/// Extract vertices and neightbourlists from an icosphere
pub fn make_vertices(icosphere: &IcoSphere) -> Vec<Vertex> {
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

    let vertices: Vec<_> = vertex_positions
        .zip(neighbors)
        .map(|(pos, neighbors)| Vertex {
            pos,
            neighbors: neighbors.iter().map(|i| *i as u16).collect_vec(),
        })
        .collect();

    // Create list of faces
    let _faces: Vec<[usize; 3]> = icosphere
        .get_all_indices()
        .chunks(3)
        .map(|i| [i[0] as usize, i[1] as usize, i[2] as usize])
        .collect::<Vec<_>>();

    vertices
}

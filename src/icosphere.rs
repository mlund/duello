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

#[cfg(test)]
extern crate approx;
use std::f64::consts::PI;

use crate::{IcoSphere, Vector3};
use anyhow::{Context, Result};
use glam::f32::Vec3A;
use hexasphere::AdjacencyBuilder;

/// Make icosphere with at least `min_points` surface points (vertices).
///
/// This is done by iteratively subdividing the faces of an icosahedron
/// until at least `min_points` vertices are achieved.
/// The number of vertices on the icosphere is _N_ = 10 × (_n_divisions_ + 1)² + 2
/// whereby 0, 1, 2, ... subdivisions give 12, 42, 92, ... vertices, respectively.
///
///
/// ## Further reading
///
/// - <https://en.wikipedia.org/wiki/Loop_subdivision_surface>
/// - <https://danielsieger.com/blog/2021/03/27/generating-spheres.html>
/// - <https://danielsieger.com/blog/2021/01/03/generating-platonic-solids.html>
///
/// ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Loop_Subdivision_Icosahedron.svg/300px-Loop_Subdivision_Icosahedron.svg.png)
///
pub fn make_icosphere(min_points: usize) -> Result<IcoSphere> {
    let points_per_division = |n_div: usize| 10 * (n_div + 1) * (n_div + 1) + 2;
    let n_points = (0..200).map(points_per_division);

    // Number of divisions to achieve at least `min_points` vertices
    let n_divisions = n_points
        .enumerate()
        .find(|(_, n)| *n >= min_points)
        .map(|(n_div, _)| n_div)
        .context("too many vertices")?;

    debug!(
        "Creating icosphere with {} divisions, {} vertices",
        n_divisions,
        points_per_division(n_divisions)
    );

    Ok(IcoSphere::new(n_divisions, |_| ()))
}

/// Make icosphere vertices as 3D vectors
///
/// ## Examples
/// ~~~
/// let vertices = duello::make_icosphere_vertices(20).unwrap();
/// assert_eq!(vertices.len(), 42);
/// ~~~
pub fn make_icosphere_vertices(min_points: usize) -> Result<Vec<Vector3>> {
    let icosphere = make_icosphere(min_points)?;
    let vertices = extract_vertices(&icosphere);
    Ok(vertices)
}

/// Get the icosphere vertices as a vector of 3D vectors.
pub fn extract_vertices(icosphere: &IcoSphere) -> Vec<Vector3> {
    icosphere
        .raw_points()
        .iter()
        .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect()
}

/// Make weights for each vertex in the icosphere based on the average area of adjacent faces.
pub fn make_weights(icosphere: &IcoSphere) -> Vec<f64> {
    let indices = icosphere.get_all_indices();
    let vertices = icosphere.raw_points();

    // flat (euclidean) face area
    let _flat_face_area = |a: usize, b: usize, c: usize| {
        let a = vertices[a];
        let b = vertices[b];
        let c = vertices[c];
        let ab = b - a;
        let ac = c - a;
        0.5 * ab.cross(ac).length()
    };

    // spherical face area
    #[allow(non_snake_case)]
    let spherical_face_area = |a: usize, b: usize, c: usize| {
        let a = &vertices[a].normalize();
        let b = &vertices[b].normalize();
        let c = &vertices[c].normalize();

        let angle = |u: &Vec3A, v: &Vec3A, w: &Vec3A| {
            let vu = (u - v * v.dot(*u)).normalize();
            let vw = (w - v * v.dot(*w)).normalize();
            vu.angle_between(vw)
        };

        let A = angle(b, a, c);
        let B = angle(c, b, a);
        let C = angle(a, c, b);

        (A + B + C) as f64 - PI
    };

    let mut weights = Vec::with_capacity(vertices.len());
    let mut adjency = AdjacencyBuilder::new(vertices.len());
    adjency.add_indices(&indices);

    for (i, nb) in adjency.finish().iter().enumerate() {
        let mut areas = Vec::with_capacity(nb.len());
        areas.push(spherical_face_area(
            i,
            *nb.first().unwrap(),
            *nb.last().unwrap(),
        ));
        for j in 0..nb.len() - 1 {
            let area = spherical_face_area(i, nb[j], nb[j + 1]);
            areas.push(area);
        }
        let avg_area = areas.iter().sum::<f64>() / areas.len() as f64;
        weights.push(avg_area);
    }
    assert_eq!(weights.len(), vertices.len());

    let scale = vertices.len() as f64 / weights.iter().sum::<f64>();
    weights.iter_mut().for_each(|w| *w *= scale);
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_icosphere() {
        let points = make_icosphere_vertices(1).unwrap();
        assert_eq!(points.len(), 12);
        let points = make_icosphere_vertices(10).unwrap();
        assert_eq!(points.len(), 12);
        let points = make_icosphere_vertices(13).unwrap();
        assert_eq!(points.len(), 42);
        let points = make_icosphere_vertices(42).unwrap();
        assert_eq!(points.len(), 42);
        let points = make_icosphere_vertices(43).unwrap();
        assert_eq!(points.len(), 92);
        let _ = make_icosphere_vertices(400003).is_err();

        let samples = 1000;
        let points = make_icosphere_vertices(samples).unwrap();
        let mut center: Vector3 = Vector3::zeros();
        assert_eq!(points.len(), 1002);
        for point in points {
            assert_relative_eq!((point.norm() - 1.0).abs(), 0.0, epsilon = 1e-6);
            center += point;
        }
        assert_relative_eq!(center.norm(), 0.0, epsilon = 1e-1);

        let icosphere = make_icosphere(1).unwrap();
        let weights = make_weights(&icosphere);
        let min_weight = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_weight = weights.iter().cloned().fold(0.0, f64::max);
        assert_relative_eq!(min_weight, 1.0, epsilon = 1e-6);
        assert_relative_eq!(max_weight, 1.0, epsilon = 1e-6);

        let icosphere = make_icosphere(42).unwrap();
        let weights = make_weights(&icosphere);
        let min_weight = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_weight = weights.iter().cloned().fold(0.0, f64::max);
        let mean_weight = weights.iter().sum::<f64>() / weights.len() as f64;
        let weight_stddev = weights
            .iter()
            .map(|w| (w - mean_weight).powi(2))
            .sum::<f64>()
            .sqrt()
            / (weights.len() as f64).sqrt();
        assert_relative_eq!(min_weight, 0.9538671732382897, epsilon = 1e-6);
        assert_relative_eq!(max_weight, 1.0184529616689328, epsilon = 1e-6);
        assert_relative_eq!(mean_weight, 0.9999999999999999, epsilon = 1e-6);
        assert_relative_eq!(weight_stddev, 0.0291766407712738, epsilon = 1e-6);

        let icosphere = make_icosphere(92).unwrap();
        let weights = make_weights(&icosphere);
        let min_weight = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_weight = weights.iter().cloned().fold(0.0, f64::max);
        let mean_weight = weights.iter().sum::<f64>() / weights.len() as f64;
        let weight_stddev = weights
            .iter()
            .map(|w| (w - mean_weight).powi(2))
            .sum::<f64>()
            .sqrt()
            / (weights.len() as f64).sqrt();
        assert_relative_eq!(min_weight, 0.9399785391170831, epsilon = 1e-6);
        assert_relative_eq!(max_weight, 1.0320120385450426, epsilon = 1e-5);
        assert_relative_eq!(mean_weight, 0.9999999999999999, epsilon = 1e-6);
        assert_relative_eq!(weight_stddev, 0.028536625575550475, epsilon = 1e-6);
    }
}

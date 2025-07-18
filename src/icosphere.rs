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

/// Surface area of a unit sphere.
const UNIT_SPHERE_AREA: f64 = 4.0 * PI;

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
    let mut weights = Vec::with_capacity(vertices.len());
    let mut adjency = AdjacencyBuilder::new(vertices.len());
    adjency.add_indices(&indices);

    // Loop over each neighboring face to i'th vertex
    for (i, neighbors) in adjency.finish().iter().enumerate() {
        let mut area = 0.0;
        // Handle the face made of i with the first and last neighboring vertices
        area += spherical_face_area(
            &vertices[i],
            &vertices[*neighbors.first().unwrap()],
            &vertices[*neighbors.last().unwrap()],
        );
        // Handle the faces made of i with each remaining pairs of neighboring vertices
        for j in 0..neighbors.len() - 1 {
            area += spherical_face_area(
                &vertices[i],
                &vertices[neighbors[j]],
                &vertices[neighbors[j + 1]],
            );
        }
        // Faces contribute w. 1/3 of their area to a single vertex weight
        weights.push(area / 3.0);
    }
    debug_assert_eq!(weights.len(), vertices.len());

    // The sum of all vertex contributions should add up to 4π,
    // the surface area of a unit sphere
    let total_area = weights.iter().sum::<f64>();
    approx::assert_relative_eq!(total_area, UNIT_SPHERE_AREA, epsilon = 1e-4);

    // Normalize the weights so that they fluctuate around 1
    // (this has no effect on final results)
    let ideal_vertex_area = UNIT_SPHERE_AREA / vertices.len() as f64;
    weights.iter_mut().for_each(|w| *w /= ideal_vertex_area);
    weights
}

/// Calculate the spherical face area of a triangle defined by three vertices
/// See <https://en.wikipedia.org/wiki/Spherical_trigonometry>
#[allow(non_snake_case)]
fn spherical_face_area(a: &Vec3A, b: &Vec3A, c: &Vec3A) -> f64 {
    debug_assert!(a.is_normalized());
    debug_assert!(b.is_normalized());
    debug_assert!(c.is_normalized());

    let angle = |u: &Vec3A, v: &Vec3A, w: &Vec3A| {
        let vu = u - v * v.dot(*u);
        let vw = w - v * v.dot(*w);
        vu.angle_between(vw)
    };
    let A = angle(b, a, c);
    let B = angle(c, b, a);
    let C = angle(a, c, b);
    (A + B + C) as f64 - PI // Spherical excess
}

/// Calculate the euclidian (flat) face area of a triangle defined by three vertices
#[allow(unused)]
fn flat_face_area(a: &Vec3A, b: &Vec3A, c: &Vec3A) -> f64 {
    debug_assert!(a.is_normalized());
    debug_assert!(b.is_normalized());
    debug_assert!(c.is_normalized());
    let ab = b - a;
    let ac = c - a;
    0.5 * ab.cross(ac).length() as f64
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
        assert_relative_eq!(min_weight, 0.8327130552002484, epsilon = 1e-6);
        assert_relative_eq!(max_weight, 1.0669150648124996, epsilon = 1e-6);
        assert_relative_eq!(mean_weight, 0.9999999999999999, epsilon = 1e-6);
        assert_relative_eq!(weight_stddev, 0.10580141129416724, epsilon = 1e-6);

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
        assert_relative_eq!(min_weight, 0.7996549501752724, epsilon = 1e-6);
        assert_relative_eq!(max_weight, 1.053539416842339, epsilon = 1e-5);
        assert_relative_eq!(mean_weight, 1.0, epsilon = 1e-3);
        assert_relative_eq!(weight_stddev, 0.07941105694522466, epsilon = 1e-6);
    }

    #[test]
    fn test_spherical_face_area() {
        // Equilateral triangle on the unit sphere (1/8 of a unit sphere)
        let [a, b, c] = [
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(0.0, 0.0, 1.0),
        ];
        let area = spherical_face_area(&a, &b, &c);
        assert_relative_eq!(area, 0.5 * PI, epsilon = 1e-6);
    }
    #[test]
    fn test_icosahedron_face_areas() {
        // Sum face areas of a regular icosahedron - should be 4π
        let icosahedron = IcoSphere::new(0, |_| ()); // no subdivision
        let vertices = icosahedron.raw_points();
        let to_area = |face: &[u32]| {
            let [a, b, c] = [
                &vertices[face[0] as usize],
                &vertices[face[1] as usize],
                &vertices[face[2] as usize],
            ];
            spherical_face_area(a, b, c)
        };
        let total_area: f64 = icosahedron.get_all_indices().chunks(3).map(to_area).sum();
        assert_eq!(vertices.len(), 12);
        assert_relative_eq!(total_area, UNIT_SPHERE_AREA, epsilon = 1e-5);
    }
    #[test]
    fn test_subdivided_icosahedron_face_areas() {
        // Sum face areas of a double subdivided icosahedron - should be 4π
        let icosahedron = IcoSphere::new(2, |_| ()); // 2 subdivisions
        let vertices = icosahedron.raw_points();
        let to_area = |face: &[u32]| {
            let [a, b, c] = [
                &vertices[face[0] as usize],
                &vertices[face[1] as usize],
                &vertices[face[2] as usize],
            ];
            spherical_face_area(a, b, c)
        };
        let total_area: f64 = icosahedron.get_all_indices().chunks(3).map(to_area).sum();
        assert_eq!(vertices.len(), 92);
        assert_relative_eq!(total_area, UNIT_SPHERE_AREA, epsilon = 1e-4);
    }
}

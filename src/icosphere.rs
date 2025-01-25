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
use crate::{IcoSphere, Vector3};
use anyhow::{Context, Result};

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
    let vertex_positions = make_icosphere(min_points)?
        .raw_points()
        .iter()
        .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect();
    Ok(vertex_positions)
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
    }
}

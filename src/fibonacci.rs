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

use crate::Vector3;
use std::f64::consts::PI;

/// Generates n points uniformly distributed on a unit sphere
///
/// Related information:
/// - <https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere>
/// - <https://en.wikipedia.org/wiki/Geodesic_polyhedron>
/// - c++: <https://github.com/caosdoar/spheres>
pub fn make_fibonacci_sphere(n_points: usize) -> Vec<Vector3> {
    assert!(n_points > 1, "n_points must be greater than 1");
    let phi = PI * (3.0 - f64::sqrt(5.0)); // golden angle in radians
    let make_point = |i: usize| -> Vector3 {
        let y = 1.0 - 2.0 * (i as f64 / (n_points - 1) as f64); // y goes from 1 to -1
        let radius = (1.0 - y * y).sqrt(); // radius at y
        let theta = phi * i as f64; // golden angle increment
        let (z, x) = theta.sin_cos();
        Vector3::new(x * radius, y, z * radius).normalize()
    };
    (0..n_points).map(make_point).collect()
}

#[test]
fn test_fibonacci_sphere() {
    use approx::assert_relative_eq;
    let samples = 1000;
    let points_on_sphere = make_fibonacci_sphere(samples);
    let mut center: Vector3 = Vector3::zeros();
    assert_eq!(points_on_sphere.len(), samples);
    for point in points_on_sphere {
        assert_relative_eq!((point.norm() - 1.0).abs(), 0.0, epsilon = 1e-10);
        center += point;
    }
    assert_relative_eq!(center.norm(), 0.0, epsilon = 1e-1);
}

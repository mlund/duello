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

/// Spherical coordinates (r, theta, phi) i.e. radius, polar angle, azimuthal angle.
#[derive(Debug, Clone)]
pub struct SphericalCoord {
    /// Radius, r >=0
    r: f64,
    /// Polar angle or inclination (0..pi)
    theta: f64,
    /// Azimuthal angle (0..2pi)
    phi: f64,
}

impl SphericalCoord {
    #[inline(always)]
    /// Radius
    pub const fn radius(&self) -> f64 {
        self.r
    }
    /// Polar angle, theta (0..pi)
    #[inline(always)]
    pub const fn theta(&self) -> f64 {
        self.theta
    }
    /// Polar angle, theta (0..pi)
    #[inline(always)]
    pub const fn polar_angle(&self) -> f64 {
        self.theta
    }
    /// Azimuthal angle, phi (0..2pi)
    #[inline(always)]
    pub const fn phi(&self) -> f64 {
        self.phi
    }
    /// Azimuthal angle, phi (0..2pi)
    #[inline(always)]
    pub const fn azimuthal_angle(&self) -> f64 {
        self.phi
    }
    pub const fn new(r: f64, theta: f64, phi: f64) -> Self {
        // Ensure phi is in the range [0..2pi)
        let phi = (phi + 2.0 * PI) % (2.0 * PI);
        // Ensure theta is in the range [0..pi]
        let theta = (theta + PI) % PI;
        Self { r, theta, phi }
    }
    /// Create spherical coordinates from cartesian coordinates
    pub fn from_cartesian(cartesian: Vector3) -> Self {
        let r = cartesian.norm();
        let theta = (cartesian.z / r).acos();
        let phi = cartesian.y.atan2(cartesian.x);
        // Ensure phi is in the range [0..2pi)
        let phi = (phi + 2.0 * PI) % (2.0 * PI);
        Self::new(r, theta, phi)
    }
    /// Create cartesian coordinates from spherical coordinates
    pub fn to_cartesian(&self) -> Vector3 {
        let (theta_sin, theta_cos) = self.theta.sin_cos();
        let (phi_sin, phi_cos) = self.phi.sin_cos();
        Vector3::new(theta_sin * phi_cos, theta_sin * phi_sin, theta_cos).scale(self.r)
    }
}

/// Convert to tuple (r, theta, phi)
impl From<SphericalCoord> for (f64, f64, f64) {
    fn from(spherical: SphericalCoord) -> Self {
        (spherical.r, spherical.theta, spherical.phi)
    }
}

/// Get cartesian coordinates from spherical coordinates
impl From<SphericalCoord> for Vector3 {
    fn from(spherical: SphericalCoord) -> Self {
        spherical.to_cartesian()
    }
}

/// Get spherical coordinates from cartesian coordinates
impl From<Vector3> for SphericalCoord {
    fn from(cartesian: Vector3) -> Self {
        SphericalCoord::from_cartesian(cartesian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use iter_num_tools::arange;

    #[test]
    fn test_spherical_cartesian_conversion() {
        const TOL: f64 = 1e-6;
        // Skip theta = 0 as phi is undefined
        for theta in arange(0.0000001..PI, 0.01) {
            for phi in arange(-2.0..2.0 * PI, 0.01) {
                // round-trip spherical -> cartesian -> spherical
                let spherical1 = SphericalCoord::new(1.0, theta, phi);
                let cartesian = Vector3::from(spherical1.clone()).scale(2.0);
                let spherical2 = SphericalCoord::from(cartesian);
                assert_relative_eq!(spherical1.theta(), spherical2.theta(), epsilon = TOL);
                assert_relative_eq!(spherical1.phi(), spherical2.phi(), epsilon = TOL);
                assert_relative_eq!(
                    spherical1.radius() * 2.0,
                    spherical2.radius(),
                    epsilon = TOL
                );
            }
        }
    }
}

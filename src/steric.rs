//! Excluded volume of two bead sets, the steric reference for B₂₂* and K_a.

use crate::{random_orientation, structure::Structure, UnitQuaternion, Vector3};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::f64::consts::PI;

/// Second virial coefficient of the pair as hard beads (topology radii),
/// orientation-averaged: B₂₂ʳᵉᶠ = 2π∫f_ov(R)R²dR. Needs no σ.
pub struct Steric {
    b2: f64,
    profile: Vec<(f64, f64)>,
}

impl Steric {
    pub fn new(a: &Structure, b: &Structure) -> Self {
        let covered: Vec<_> = orientations()
            .map(|(n, q)| overlap_intervals(a, &rotated(b, &q), &b.radii, &n))
            .collect();
        let volume: f64 = covered
            .iter()
            .flatten()
            .map(|(lo, hi)| (hi.powi(3) - lo.powi(3)) / 3.0)
            .sum();
        Self {
            b2: 2.0 * PI * volume / ORIENTATIONS as f64,
            profile: profile(&covered),
        }
    }

    /// B₂₂ʳᵉᶠ (Å³)
    pub fn b2(&self) -> f64 {
        self.b2
    }

    /// (R, f_ov) on a PROFILE_DR grid from 0 to past the furthest contact.
    pub fn overlap(&self) -> &[(f64, f64)] {
        &self.profile
    }
}

/// Grid of the overlap profile (Å); B₂₂ʳᵉᶠ itself is exact per orientation.
const PROFILE_DR: f64 = 0.25;

/// Fraction of orientations overlapping at each grid R.
fn profile(covered: &[Vec<(f64, f64)>]) -> Vec<(f64, f64)> {
    let reach = covered.iter().flatten().map(|s| s.1).fold(0.0, f64::max);
    let inside = |r: f64, spans: &[(f64, f64)]| spans.iter().any(|&(lo, hi)| lo <= r && r < hi);
    (0..=(reach / PROFILE_DR).ceil() as usize)
        .map(|i| {
            let r = i as f64 * PROFILE_DR;
            let n = covered.iter().filter(|spans| inside(r, spans)).count();
            (r, n as f64 / covered.len() as f64)
        })
        .collect()
}

/// Binomial error in f_ov stays below ~1% of B₂₂ʳᵉᶠ.
const ORIENTATIONS: usize = 2000;

/// Separation direction and rotation of B: together they cover every relative pose.
/// Directions on a Fibonacci sphere, as random ones leave ~2% noise for elongated bodies;
/// rotations from a fixed seed, so a structure pair always gets the same reference.
fn orientations() -> impl Iterator<Item = (Vector3, UnitQuaternion)> {
    let mut rng = Pcg64::seed_from_u64(0x5e71c);
    (0..ORIENTATIONS).map(move |i| (fibonacci_direction(i), random_orientation(&mut rng)))
}

/// i-th of ORIENTATIONS near-uniform points on the unit sphere.
fn fibonacci_direction(i: usize) -> Vector3 {
    let golden_angle = PI * (3.0 - 5f64.sqrt());
    let z = 1.0 - (2.0 * i as f64 + 1.0) / ORIENTATIONS as f64;
    let s = (1.0 - z * z).sqrt();
    let phi = golden_angle * i as f64;
    Vector3::new(s * phi.cos(), s * phi.sin(), z)
}

/// B's positions turned about its mass centre (the origin).
fn rotated(b: &Structure, q: &UnitQuaternion) -> Vec<Vector3> {
    b.pos.iter().map(|p| q * p).collect()
}

/// R ≥ 0 at which B, its mass centre at R·n, overlaps A, as disjoint sorted intervals.
/// Per bead pair |R·n − d| < s is a quadratic in R, so each overlap is one exact interval.
fn overlap_intervals(
    a: &Structure,
    b_pos: &[Vector3],
    b_radii: &[f64],
    n: &Vector3,
) -> Vec<(f64, f64)> {
    let mut spans: Vec<(f64, f64)> = a
        .pos
        .iter()
        .zip(&a.radii)
        .flat_map(|(pa, ra)| {
            b_pos.iter().zip(b_radii).filter_map(move |(pb, rb)| {
                let d = pa - pb;
                let along = n.dot(&d);
                let disc = along * along - d.norm_squared() + (ra + rb).powi(2);
                let half = (disc > 0.0).then(|| disc.sqrt())?;
                let hi = along + half;
                (hi > 0.0).then_some(((along - half).max(0.0), hi))
            })
        })
        .collect();
    spans.sort_by(|x, y| x.0.total_cmp(&y.0));
    merge(spans)
}

/// Sorted intervals to their disjoint union.
fn merge(sorted: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut union: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
    for (lo, hi) in sorted {
        match union.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => union.push((lo, hi)),
        }
    }
    union
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{structure::Structure, Vector3};
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn beads(beads: &[([f64; 3], f64)]) -> Structure {
        Structure {
            pos: beads.iter().map(|(p, _)| Vector3::from(*p)).collect(),
            masses: vec![1.0; beads.len()],
            charges: vec![0.0; beads.len()],
            radii: beads.iter().map(|(_, r)| *r).collect(),
            atom_ids: vec![0; beads.len()],
        }
    }

    #[test]
    fn two_hard_spheres() {
        let steric = Steric::new(&beads(&[([0.0; 3], 2.0)]), &beads(&[([0.0; 3], 3.0)]));
        assert_relative_eq!(steric.b2(), 2.0 * PI * 125.0 / 3.0, max_relative = 0.01);
    }

    /// Tangent dimer (beads d/2 at ±d/2) against a bead d/2, d = 4 Å.
    fn dimer_and_sphere() -> (Structure, Structure) {
        let dimer = beads(&[([0.0, 0.0, 2.0], 2.0), ([0.0, 0.0, -2.0], 2.0)]);
        (dimer, beads(&[([0.0; 3], 2.0)]))
    }

    /// Excluded volume is two radius-d spheres d apart: V = 2·4πd³/3 − lens 5πd³/12
    /// = 9πd³/4, so B₂ = V/2 = 9πd³/8.
    #[test]
    fn sphere_against_a_tangent_dimer() {
        let (dimer, sphere) = dimer_and_sphere();
        let b2 = 9.0 * PI * 64.0 / 8.0;
        assert_relative_eq!(Steric::new(&dimer, &sphere).b2(), b2, max_relative = 0.02);
    }

    /// Every orientation overlaps while √(R² + d²/4) < d (side-on), none from 3d/2 (end-on).
    #[test]
    fn overlap_profile_of_a_tangent_dimer() {
        let (dimer, sphere) = dimer_and_sphere();
        let steric = Steric::new(&dimer, &sphere);
        let side_on = 3f64.sqrt() / 2.0 * 4.0;
        for &(r, f) in steric.overlap() {
            match r {
                r if r < side_on => assert_eq!(f, 1.0, "R = {r}"),
                r if r >= 6.0 => assert_eq!(f, 0.0, "R = {r}"),
                _ => assert!(f > 0.0 && f <= 1.0, "R = {r}, f = {f}"),
            }
        }
        let (r_last, _) = *steric.overlap().last().unwrap();
        assert!(r_last >= 6.0, "profile reaches past contact, {r_last}");
    }
}

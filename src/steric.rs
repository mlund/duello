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
        let covered = covered(a, b, 1.0);
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

/// Radii scale for the core: soft beads at full contact cost only ~ε, so the pair
/// is excluded only well inside it.
pub const CORE_RADIUS_SCALE: f64 = 0.85;

/// Largest R at which every sampled orientation overlaps, with radii scaled by
/// [`CORE_RADIUS_SCALE`] (Å). Below it e^{−w/kT} ≈ 0, so a scan may start there.
pub fn core_distance(a: &Structure, b: &Structure) -> f64 {
    covered(a, b, CORE_RADIUS_SCALE)
        .iter()
        .map(|spans| match spans.first() {
            Some(&(0.0, hi)) => hi,
            _ => 0.0, // open at R = 0, e.g. B's centre in a cavity of A
        })
        .fold(f64::INFINITY, f64::min)
}

/// Scan start: [`core_distance`] snapped down onto the `dr` grid, so σ is a round grid value.
pub fn auto_rmin(a: &Structure, b: &Structure, dr: f64) -> f64 {
    // The epsilon keeps a core on a grid value from dropping a bin to rounding.
    (core_distance(a, b) / dr + 1e-9).floor() * dr
}

/// The scan's R_min: `rmin` if given, else [`auto_rmin`]. The one rule CLI and web share.
pub fn resolve_rmin(
    rmin: Option<f64>,
    [a, b]: [&Structure; 2],
    dr: f64,
    rmax: f64,
) -> anyhow::Result<f64> {
    anyhow::ensure!(dr > 0.0, "ΔR must be positive, not {dr}");
    let rmin = match rmin {
        Some(rmin) => rmin,
        None => {
            let auto = auto_rmin(a, b, dr);
            anyhow::ensure!(
                auto > 0.0,
                "No R at which every orientation overlaps, e.g. an open or hollow shape; set R_min by hand"
            );
            log::info!("R_min = {auto:.1} Å from bead overlap");
            auto
        }
    };
    anyhow::ensure!(
        rmin < rmax,
        "R_min ({rmin:.1} Å) must be below R_max ({rmax:.1} Å)"
    );
    Ok(rmin)
}

/// Each orientation's overlap intervals, radii scaled by `scale`.
fn covered(a: &Structure, b: &Structure, scale: f64) -> Vec<Vec<(f64, f64)>> {
    orientations()
        .map(|(n, q)| overlap_intervals(a, &rotated(b, &q), &b.radii, &n, scale))
        .collect()
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
    scale: f64,
) -> Vec<(f64, f64)> {
    let mut spans: Vec<(f64, f64)> = a
        .pos
        .iter()
        .zip(&a.radii)
        .flat_map(|(pa, ra)| {
            b_pos.iter().zip(b_radii).filter_map(move |(pb, rb)| {
                let d = pa - pb;
                let along = n.dot(&d);
                let disc = along * along - d.norm_squared() + (scale * (ra + rb)).powi(2);
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

    #[test]
    fn core_of_two_spheres_is_the_scaled_contact() {
        let core = core_distance(&beads(&[([0.0; 3], 2.0)]), &beads(&[([0.0; 3], 3.0)]));
        assert_relative_eq!(core, 0.85 * 5.0, epsilon = 0.25);
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

    /// Core 4.25 Å snaps down to the 0.5 Å grid.
    #[test]
    fn auto_rmin_snaps_down_onto_the_dr_grid() {
        let (a, b) = (beads(&[([0.0; 3], 2.0)]), beads(&[([0.0; 3], 3.0)]));
        assert_eq!(auto_rmin(&a, &b, 0.5), 4.0);
    }

    /// Side-on is the worst direction: every orientation overlaps while
    /// √(R² + d²/4) < 0.85d, so the core is d·√(0.85² − ¼); end-on alone would give 0.85d + d/2.
    #[test]
    fn core_of_a_tangent_dimer_is_set_side_on() {
        let (dimer, sphere) = dimer_and_sphere();
        let side_on = 4.0 * (0.85f64.powi(2) - 0.25).sqrt();
        assert_relative_eq!(core_distance(&dimer, &sphere), side_on, epsilon = 0.01);
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

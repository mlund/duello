//! Near/far split of each pair energy: S·U summed exactly, (1 − S)·U gridded.

use crate::energy::SplinedPotentials;
use crate::structure::Structure;
use faunus::interatomic::twobody::IsotropicTwobodyEnergy;

/// `near_weight` in WGSL; reads `grid.near_r1` and `grid.near_r2`.
pub(super) const WGSL: &str = include_str!("../../shaders/grid_switch.wgsl");

/// Smooth step S(r) from 1 at `r1` to 0 at `r2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Switch {
    pub r1: f64,
    pub r2: f64,
}

impl Switch {
    /// Past every used pair's wall and well by [`MARGIN`], so the far part is
    /// smooth for all of them.
    pub fn new(splines: &SplinedPotentials, a: &Structure, b: &Structure) -> Self {
        let r1 = itertools::iproduct!(a.unique_atom_types(), b.unique_atom_types())
            .map(|(i, j)| wall_end(splines, i, j))
            .fold(0.0, f64::max)
            + MARGIN;
        Self { r1, r2: r1 + WIDTH }
    }

    /// Weight of the exactly summed part of a pair at distance `r`; the
    /// kernels use the WGSL twin.
    #[cfg(test)]
    pub fn near_weight(&self, r: f64) -> f64 {
        // Quintic smoothstep: S′ and S″ vanish at both ends, as in grid_switch.wgsl.
        let t = ((r - self.r1) / (self.r2 - self.r1)).clamp(0.0, 1.0);
        1.0 - t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    }
}

/// Gap (Å) between the outermost wall or well and the switch's start.
const MARGIN: f64 = 1.0;

/// Width (Å) of the switch: wider makes the far part smoother, but adds near pairs.
const WIDTH: f64 = 3.0;

/// Slope (kJ/mol/Å) below which a well-less pair's wall has ended.
const FLAT: f64 = 0.1;

/// Where `(i, j)`'s steep part ends: the well's bottom, or for a well-less
/// pair (e.g. a repulsive Kim–Hummer tail) the outermost point steeper than
/// [`FLAT`]. Scanned inwards, so a plateau near `r_min` can't hide a wall.
fn wall_end(splines: &SplinedPotentials, i: usize, j: usize) -> f64 {
    const STEP: f64 = 0.005;
    if let Some((r_m, _)) = splines.minimum(i, j) {
        return r_m;
    }
    let spline = splines.get(i, j);
    let u = |r: f64| spline.isotropic_twobody_energy(r * r);
    let end = splines.vanishes_beyond(i, j);
    (0..)
        .map(|k| end - k as f64 * STEP)
        .take_while(|&r| r > spline.r_min())
        .find(|&r| u(r - STEP) - u(r) > FLAT * STEP)
        .unwrap_or(spline.r_min())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_support::{chymotrypsinogen, kt, lysozyme, TestPair, CGN_TEMPERATURE};

    const SWITCH: Switch = Switch { r1: 8.0, r2: 11.0 };

    fn type_pairs(sys: &TestPair) -> impl Iterator<Item = (usize, usize)> {
        let types = sys.ref_a.unique_atom_types();
        itertools::iproduct!(types.clone(), types)
    }

    /// Name, system and its kT.
    fn fixtures() -> [(&'static str, TestPair, f64); 2] {
        [
            ("lysozyme", lysozyme(), kt()),
            (
                "7KTZ KH",
                chymotrypsinogen("kimhummer", false),
                crate::thermal_energy(CGN_TEMPERATURE),
            ),
        ]
    }

    /// The exact near part must hold every well whole.
    #[test]
    fn r1_lies_past_every_well() {
        for (name, sys, _) in fixtures() {
            let switch = Switch::new(&sys.splines, &sys.ref_a, &sys.ref_b);
            let mut wells = 0;
            for (i, j) in type_pairs(&sys) {
                if let Some((r_m, _)) = sys.splines.minimum(i, j) {
                    wells += 1;
                    assert!(
                        r_m < switch.r1,
                        "{name} ({i}, {j}): r_m {r_m} ≥ r1 {}",
                        switch.r1
                    );
                }
            }
            assert!(wells > 0, "{name}");
        }
    }

    /// Wells and walls stay in the near part, so the far part trilinear sees
    /// is gentle: at 1 Å spacing, h²/8·|U″| stays below 0.02 kT per pair,
    /// where a gridded wall reaches hundreds of kJ/mol.
    /// U″ changes sign across the switch, so pair errors partly cancel.
    #[test]
    fn far_part_is_smooth() {
        const D: f64 = 0.01;
        for (name, sys, kt) in fixtures() {
            let switch = Switch::new(&sys.splines, &sys.ref_a, &sys.ref_b);
            for (i, j) in type_pairs(&sys) {
                let spline = sys.splines.get(i, j);
                let far =
                    |r: f64| (1.0 - switch.near_weight(r)) * spline.isotropic_twobody_energy(r * r);
                let worst = (1..2000)
                    .map(|k| 0.01 * f64::from(k))
                    .map(|r| ((far(r + D) - 2.0 * far(r) + far(r - D)) / (D * D)).abs())
                    .fold(0.0, f64::max);
                assert!(
                    worst / 8.0 < 0.02 * kt,
                    "{name} ({i}, {j}): |U″| {worst:.3}"
                );
            }
        }
    }

    /// Central differences: S′, S″ at `r`.
    fn derivatives(r: f64) -> [f64; 2] {
        const D: f64 = 1e-4;
        let s = |r| SWITCH.near_weight(r);
        [
            (s(r + D) - s(r - D)) / (2.0 * D),
            (s(r + D) - 2.0 * s(r) + s(r - D)) / (D * D),
        ]
    }

    #[test]
    fn steps_from_one_to_zero() {
        assert_eq!(SWITCH.near_weight(0.0), 1.0);
        assert_eq!(SWITCH.near_weight(8.0), 1.0);
        assert_eq!(SWITCH.near_weight(9.5), 0.5);
        assert_eq!(SWITCH.near_weight(11.0), 0.0);
        assert_eq!(SWITCH.near_weight(30.0), 0.0);
    }

    /// C² at both ends, so the far part carries no kink for trilinear to smear.
    #[test]
    fn first_and_second_derivatives_vanish_at_both_ends() {
        for r in [SWITCH.r1, SWITCH.r2] {
            let [d1, d2] = derivatives(r);
            assert!(d1.abs() < 1e-6, "S′({r}) = {d1}");
            assert!(d2.abs() < 1e-3, "S″({r}) = {d2}");
        }
    }

    #[test]
    fn decreases_monotonically() {
        let s: Vec<f64> = (0..=60)
            .map(|k| 7.0 + 0.1 * f64::from(k))
            .map(|r| SWITCH.near_weight(r))
            .collect();
        assert!(s.windows(2).all(|w| w[1] <= w[0]), "{s:?}");
    }
}

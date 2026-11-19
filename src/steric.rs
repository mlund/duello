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
/// Orientations are independent, so the CLI spreads them over threads; the
/// ordered collect keeps every sum, and so the result, as serial.
fn covered(a: &Structure, b: &Structure, scale: f64) -> Vec<Vec<(f64, f64)>> {
    let spans = |(n, q): (Vector3, UnitQuaternion)| {
        overlap_intervals(a, &rotated(b, &q), &b.radii, &n, scale)
    };
    crate::par_map(orientations().collect(), spans)
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
    let max_radius = |radii: &[f64]| radii.iter().copied().fold(0.0, f64::max);
    let reach = scale * (max_radius(&a.radii) + max_radius(b_radii));
    let grid = PlaneGrid::new(b_pos, b_radii, n, reach);
    let mut spans = Vec::new();
    for (pa, ra) in a.pos.iter().zip(&a.radii) {
        let contact = |(pb, rb): &(Vector3, f64)| pair_interval(pa - pb, n, scale * (ra + rb));
        grid.for_each_candidates(pa, |beads| spans.extend(beads.iter().filter_map(contact)));
    }
    spans.sort_unstable_by(|x, y| x.0.total_cmp(&y.0)); // union ignores tie order
    merge(spans)
}

/// R ≥ 0 with |R·n − d| < s, if any: the roots of a quadratic in R.
fn pair_interval(d: Vector3, n: &Vector3, s: f64) -> Option<(f64, f64)> {
    let along = n.dot(&d);
    let disc = along * along - d.norm_squared() + s * s;
    let half = (disc > 0.0).then(|| disc.sqrt())?;
    let hi = along + half;
    (hi > 0.0).then_some(((along - half).max(0.0), hi))
}

/// Beads binned by their projection onto the plane perpendicular to `n`.
///
/// Sliding B along n can bring a bead pair into contact only if their
/// separation across n is below the contact distance. With cells that wide,
/// such partners lie in the 3×3 cells around a point, so a lookup checks a
/// handful of beads instead of all of B.
struct PlaneGrid {
    /// In-plane axes.
    u: Vector3,
    v: Vector3,
    /// Projected lower corner and cell width.
    corner: (f64, f64),
    width: f64,
    dims: (usize, usize),
    /// Beads of cell c are `beads[starts[c]..starts[c + 1]]`.
    starts: Vec<usize>,
    /// (position, radius) in cell order, so a row of cells is one slice.
    beads: Vec<(Vector3, f64)>,
}

impl PlaneGrid {
    /// Bin beads in cells at least `reach` wide. A zero reach means no pair
    /// can touch, so the grid stays empty and offers no candidates.
    fn new(pos: &[Vector3], radii: &[f64], n: &Vector3, reach: f64) -> Self {
        let (u, v) = plane_axes(n);
        let xy: Vec<(f64, f64)> = pos.iter().map(|p| (p.dot(&u), p.dot(&v))).collect();
        let (lo, hi) = xy.iter().fold(
            (
                (f64::INFINITY, f64::INFINITY),
                (f64::NEG_INFINITY, f64::NEG_INFINITY),
            ),
            |(lo, hi), &(x, y)| ((lo.0.min(x), lo.1.min(y)), (hi.0.max(x), hi.1.max(y))),
        );
        let extent = (hi.0 - lo.0).max(hi.1 - lo.1);
        // Padded so rounding can't push a touching pair two cells apart, and
        // never so fine that sparse input allocates a huge grid.
        let width = (reach * (1.0 + 1e-9)).max(extent / MAX_CELLS_PER_AXIS);
        let span = |extent: f64| (extent / width) as usize + 1;
        let dims = if pos.is_empty() || reach <= 0.0 {
            (0, 0)
        } else {
            (span(hi.0 - lo.0), span(hi.1 - lo.1))
        };
        let cell = |(x, y): &(f64, f64)| {
            let ix = (((x - lo.0) / width) as usize).min(dims.0 - 1);
            let iy = (((y - lo.1) / width) as usize).min(dims.1 - 1);
            ix + dims.0 * iy
        };
        let cells: Vec<usize> = if dims.0 == 0 {
            vec![]
        } else {
            xy.iter().map(cell).collect()
        };
        let (starts, order) = counting_sort(&cells, dims.0 * dims.1);
        Self {
            u,
            v,
            corner: lo,
            width,
            dims,
            starts,
            beads: order.into_iter().map(|i| (pos[i], radii[i])).collect(),
        }
    }

    /// Calls `f` with slices of beads holding every bead that may lie within
    /// `reach` of `p` across the plane: one slice per row of the 3×3 cells.
    fn for_each_candidates(&self, p: &Vector3, mut f: impl FnMut(&[(Vector3, f64)])) {
        let xs = self.neighbours(p.dot(&self.u) - self.corner.0, self.dims.0);
        for iy in self.neighbours(p.dot(&self.v) - self.corner.1, self.dims.1) {
            let row = self.dims.0 * iy;
            f(&self.beads[self.starts[row + xs.start]..self.starts[row + xs.end]]);
        }
    }

    /// Cells along one axis at most one away from offset `x`, within `0..n`.
    fn neighbours(&self, x: f64, n: usize) -> std::ops::Range<usize> {
        let cell = (x / self.width).floor() as i64;
        let lo = (cell - 1).clamp(0, n as i64) as usize;
        let hi = (cell + 2).clamp(0, n as i64) as usize;
        lo..hi
    }
}

/// Stable counting sort of indices by key in `0..n_keys`: the start of each
/// key's run (plus a final end) and the indices in key order.
pub(crate) fn counting_sort(keys: &[usize], n_keys: usize) -> (Vec<usize>, Vec<usize>) {
    let mut starts = vec![0; n_keys + 1];
    for &k in keys {
        starts[k + 1] += 1;
    }
    for k in 0..n_keys {
        starts[k + 1] += starts[k];
    }
    let mut next = starts.clone();
    let mut order = vec![0; keys.len()];
    for (i, &k) in keys.iter().enumerate() {
        order[next[k]] = i;
        next[k] += 1;
    }
    (starts, order)
}

/// Cells beyond this per axis buy nothing for proteins and could exhaust memory
/// for sparse input (e.g. far-apart chains).
const MAX_CELLS_PER_AXIS: f64 = 64.0;

/// Orthonormal axes spanning the plane perpendicular to unit `n`.
fn plane_axes(n: &Vector3) -> (Vector3, Vector3) {
    let helper = if n.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let u = n.cross(&helper).normalize();
    (u, n.cross(&u))
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

    /// Every bead pair, as before the neighbour grid: the oracle it must match.
    fn all_pairs(
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

    /// A random compact cloud of `n` beads, radii 1.5–4 Å, mass-centred-ish.
    fn cloud(rng: &mut impl rand::Rng, n: usize, extent: f64) -> Structure {
        let mut s = beads(&[]);
        for _ in 0..n {
            let p = Vector3::from_fn(|_, _| rng.gen_range(-extent..extent));
            s.pos.push(p);
            s.radii.push(rng.gen_range(1.5..4.0));
        }
        s
    }

    /// The neighbour grid only skips pairs that can't touch, so the overlap
    /// intervals must be exactly those of the all-pairs search.
    #[test]
    fn neighbour_grid_matches_all_pairs() {
        let mut rng = Pcg64::seed_from_u64(7);
        for _ in 0..4 {
            let (a, b) = (cloud(&mut rng, 60, 15.0), cloud(&mut rng, 40, 10.0));
            for (n, q) in orientations().take(200) {
                let b_pos = rotated(&b, &q);
                for scale in [CORE_RADIUS_SCALE, 1.0] {
                    let got = overlap_intervals(&a, &b_pos, &b.radii, &n, scale);
                    assert_eq!(got, all_pairs(&a, &b_pos, &b.radii, &n, scale));
                }
            }
        }
    }

    /// Far-apart beads would ask for millions of cells; the cap keeps the grid
    /// small and the result exact.
    #[test]
    fn sparse_input_stays_exact_and_small() {
        let a = beads(&[([0.0; 3], 3.0), ([1.0e4, 0.0, 0.0], 3.0)]);
        let b = beads(&[([0.0; 3], 2.0), ([0.0, 1.0e4, 0.0], 2.0)]);
        for (n, q) in orientations().take(50) {
            let b_pos = rotated(&b, &q);
            let grid = PlaneGrid::new(&b_pos, &b.radii, &n, 5.0);
            assert!(grid.dims.0 <= 65 && grid.dims.1 <= 65, "{:?}", grid.dims);
            let got = overlap_intervals(&a, &b_pos, &b.radii, &n, 1.0);
            assert_eq!(got, all_pairs(&a, &b_pos, &b.radii, &n, 1.0));
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

    /// Both steric passes on the CLAUDE.md lysozyme pair; they scale as the
    /// bead count squared without a neighbour search.
    #[test]
    #[ignore]
    fn steric_benchmark() {
        let sys = crate::backend::test_support::lysozyme();
        let (a, b) = (&sys.ref_a, &sys.ref_b);
        let t = std::time::Instant::now();
        let rmin = auto_rmin(a, b, 0.5);
        let t_core = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let b2 = Steric::new(a, b).b2();
        let t_ref = t.elapsed().as_secs_f64();
        println!("R_min {rmin:.2} Å in {t_core:.3} s; B₂ʳᵉᶠ {b2:.1} Å³ in {t_ref:.3} s");

        // Scaling: all pairs grows as N², the grid roughly as N.
        let mut rng = Pcg64::seed_from_u64(1);
        for (n_beads, extent) in [(160, 15.0), (1000, 28.0)] {
            let (a, b) = (
                cloud(&mut rng, n_beads, extent),
                cloud(&mut rng, n_beads, extent),
            );
            let time = |f: &dyn Fn(&[Vector3], &Vector3)| {
                let t = std::time::Instant::now();
                for (n, q) in orientations().take(200) {
                    f(&rotated(&b, &q), &n);
                }
                t.elapsed().as_secs_f64() * 1e3
            };
            let grid = time(&|p, n| drop(overlap_intervals(&a, p, &b.radii, n, 1.0)));
            let pairs = time(&|p, n| drop(all_pairs(&a, p, &b.radii, n, 1.0)));
            println!(
                "{n_beads} beads, 200 orientations: grid {grid:.1} ms, all pairs {pairs:.1} ms"
            );
        }
    }
}

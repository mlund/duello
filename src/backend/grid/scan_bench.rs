//! Scan-level checks of the grid backend against the spline-exact GPU pair
//! kernel. Slow, so `#[ignore]`d; run with
//! `cargo test --release -p duello --lib grid_scan -- --ignored --nocapture`.

use super::{GridBackend, GridSpec};
use crate::backend::test_support::{lysozyme, no_gpu, sr_splines, Lysozyme};
use crate::backend::{EnergyBackend, GpuBackend, PoseParams};
use crate::icoscan::{compute_scan, ScanParams};
use crate::report::{compute_pmf, PmfResult};
use std::time::Instant;

/// The CLAUDE.md benchmark settings over `[rmin, rmax)` in steps of `dr`.
fn params(sys: &Lysozyme, rmin: f64, rmax: f64, dr: f64) -> ScanParams {
    ScanParams {
        rmin,
        rmax,
        dr,
        temperature: 298.15,
        charges: [sys.ref_a.net_charge(), sys.ref_b.net_charge()],
        dipole_moments: [
            sys.ref_a.dipole_moment().norm(),
            sys.ref_b.dipole_moment().norm(),
        ],
        kappa: Some(f64::from(sys.coulomb.kappa)),
        permittivity: 78.4, // only feeds table metadata, which these tests don't read
        max_n_div: 3,
        gradient_threshold: 0.5,
        subdivision: Default::default(),
        homo_dimer: true,
    }
}

/// PMF and wall time of one scan.
fn scan<B: EnergyBackend + Sync>(params: &ScanParams, backend: &B) -> (PmfResult, f64) {
    let start = Instant::now();
    let result = compute_scan(params, backend, None).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let pmf = compute_pmf(&result.samples, [backend.ref_a(), backend.ref_b()]).unwrap();
    (pmf, seconds)
}

fn gpu_backend(sys: &Lysozyme) -> GpuBackend {
    let (a, b) = (sys.ref_a.clone(), sys.ref_b.clone());
    GpuBackend::new(a, b, &sys.splines, &sys.coulomb).unwrap()
}

fn grid_backend(sys: &Lysozyme, spacing: f64) -> GridBackend {
    let (a, b) = (sys.ref_a.clone(), sys.ref_b.clone());
    let spec = GridSpec {
        spacing,
        ..Default::default()
    };
    GridBackend::new(a, b, &sys.splines, &sys.coulomb, spec).unwrap()
}

/// Worst |ΔF| (kT) between two PMFs over bins where `reference` is below 5 kT,
/// i.e. where the PMF carries weight.
fn worst_weighted_error(reference: &PmfResult, other: &PmfResult) -> f32 {
    reference
        .points
        .iter()
        .zip(&other.points)
        .inspect(|(e, g)| assert_eq!(e.r, g.r))
        .filter(|(e, _)| e.free_energy < 5.0)
        .map(|(e, g)| (g.free_energy - e.free_energy).abs())
        .fold(0.0, f32::max)
}

/// Interpolation error scales as h²: about 0.01 kT at 0.5 Å on 2CDS. So the
/// bar is set by h², and halving h² must roughly halve the error. A bug
/// wouldn't converge that way.
#[test]
#[ignore]
fn grid_scan_matches_gpu_free_energy() {
    if no_gpu() {
        return;
    }
    let sys = lysozyme();
    let params = params(&sys, 26.0, 40.0, 1.0);
    let (expected, _) = scan(&params, &gpu_backend(&sys));
    let (coarse, fine) = (
        grid_backend(&sys, 0.5),
        grid_backend(&sys, 0.5 / 2f64.sqrt()),
    );
    let err_coarse = worst_weighted_error(&expected, &scan(&params, &coarse).0);
    let err_fine = worst_weighted_error(&expected, &scan(&params, &fine).0);
    println!("worst |ΔF|: {err_coarse:.3} kT at 0.5 Å, {err_fine:.3} kT at 0.35 Å");
    assert!(err_coarse < 0.3, "|ΔF| {err_coarse} kT at 0.5 Å");
    assert!(err_fine < 0.65 * err_coarse, "error doesn't scale as h²");
}

/// The CLAUDE.md benchmark. Asserts accuracy only, since timings vary by machine.
#[test]
#[ignore]
fn grid_scan_benchmark() {
    if no_gpu() {
        return;
    }
    let sys = lysozyme();
    let params = params(&sys, 26.0, 60.0, 0.5);
    let (expected, t_gpu) = scan(&params, &gpu_backend(&sys));
    let b2 = expected.virial.b2();
    println!("gpu: B₂ {b2:.0} Å³ in {t_gpu:.1} s");
    for spacing in [0.5, 1.0] {
        let start = Instant::now();
        let grid = grid_backend(&sys, spacing);
        let (got, t_grid) = scan(&params, &grid);
        let total = start.elapsed().as_secs_f64();
        let rel = got.virial.b2() / b2 - 1.0;
        println!(
            "grid {spacing} Å: B₂ {:.0} Å³ ({:+.2}%) in {t_grid:.1} s, {total:.1} s with setup; {:.1}× faster",
            got.virial.b2(),
            100.0 * rel,
            t_gpu / total
        );
        assert!(
            rel.abs() < 0.02,
            "B₂ off by {:.1}% at {spacing} Å",
            100.0 * rel
        );
    }
}

/// Grid-only CLAUDE.md scan, timed: the kernel is bound by scattered map
/// reads, so only real scan ordering shows layout changes honestly.
#[test]
#[ignore]
fn grid_kernel_benchmark() {
    if no_gpu() {
        return;
    }
    let sys = lysozyme();
    let params = params(&sys, 26.0, 60.0, 0.5);
    for spacing in [1.0, 0.5] {
        let (pmf, seconds) = scan(&params, &grid_backend(&sys, spacing));
        println!(
            "grid {spacing} Å: {seconds:.2} s, B₂ {:.0} Å³",
            pmf.virial.b2()
        );
    }
}

/// Chymotrypsinogen A (7KTZ), multi-bead, pH 7, 15 mM, at this temperature
/// (K): a pair whose B₂ is set by contact attraction, unlike lysozyme's
/// excluded volume. Same settings as the README's grid comparison.
const CGN_TEMPERATURE: f64 = 300.0;

fn chymotrypsinogen() -> (GridBackend, GpuBackend) {
    use crate::system::{CgOptions, PairSpec, PairSystem, Source};
    use faunus::interatomic::coulomb::{Medium, Salt};
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pdb = dir.join("examples/calvados3/7KTZ.pdb");
    let medium = Medium::salt_water(CGN_TEMPERATURE, Salt::SodiumChloride, 0.015);
    let cg = CgOptions {
        single_bead: false,
        ..Default::default()
    };
    let spec = PairSpec {
        a: Source::File(&pdb),
        b: None,
        top: None,
        chains: [&[], &[]],
        cg,
        medium: &medium,
    };
    let sys = PairSystem::load(spec).expect("7KTZ example loads");
    let splines = sr_splines(&sys.pair_matrix, 30.0); // the CLI's default --cutoff
    let coulomb = sys.pair_matrix.coulomb_params();
    let (a, b) = (sys.ref_a, sys.ref_b);
    let grid = GridBackend::new(a.clone(), b.clone(), &splines, coulomb, GridSpec::default());
    (
        grid.unwrap(),
        GpuBackend::new(a, b, &splines, coulomb).unwrap(),
    )
}

/// Random orientations at contact, R = 33–36 Å; seeded, so runs agree.
/// Vertex vectors are uniform in the unit ball, so their directions are
/// uniform on the sphere (`PoseParams` normalises them).
fn contact_poses(n: usize) -> Vec<PoseParams> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_pcg::Pcg64::seed_from_u64(12345);
    let direction = |rng: &mut rand_pcg::Pcg64| loop {
        let v = crate::Vector3::from_fn(|_, _| rng.gen_range(-1.0..1.0));
        if (1e-6..1.0).contains(&v.norm_squared()) {
            return v;
        }
    };
    (0..n)
        .map(|i| PoseParams {
            r: 33.0 + (i % 4) as f64,
            omega: rng.gen_range(0.0..std::f64::consts::TAU),
            vertex_i: direction(&mut rng),
            vertex_j: direction(&mut rng),
        })
        .collect()
}

/// Pose by pose against exact pairs where B₂ is decided: attractive contacts.
/// Plain trilinear maps failed this at 1 Å (mean βΔU +2.1, contact Boltzmann
/// weight 0.5), which gave B₂ the wrong sign; the split, power-transformed
/// maps pass.
#[test]
#[ignore]
fn grid_scan_contact_energies_cgn() {
    if no_gpu() {
        return;
    }
    let (grid, exact) = chymotrypsinogen();
    let poses = contact_poses(40_000);
    let (got, want) = (
        grid.compute_energies(&poses),
        exact.compute_energies(&poses),
    );
    let kt = crate::thermal_energy(CGN_TEMPERATURE);
    let attractive: Vec<f64> = (got.iter().zip(&want))
        .filter(|(_, w)| **w < 0.0)
        .map(|(g, w)| (g - w) / kt)
        .collect();
    assert!(
        !attractive.is_empty(),
        "no attractive contact poses sampled"
    );
    let bias = attractive.iter().sum::<f64>() / attractive.len() as f64;
    let weight = |u: &[f64]| u.iter().map(|u| (-u / kt).exp()).sum::<f64>();
    let ratio = weight(&got) / weight(&want);
    println!(
        "{} attractive poses: mean βΔU {bias:+.2}; Σe^(−βU) grid/exact {ratio:.2}",
        attractive.len()
    );
    assert!(bias.abs() < 0.2, "contact bias {bias:+.2} kT");
    assert!(
        (ratio - 1.0).abs() < 0.1,
        "contact Boltzmann weight ratio {ratio:.2}"
    );
}

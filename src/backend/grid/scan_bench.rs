//! Scan-level checks of the grid backend against the spline-exact GPU pair
//! kernel. Slow, so `#[ignore]`d; run with
//! `cargo test --release -p duello --lib grid_scan -- --ignored --nocapture`.

use super::{GridBackend, GridSpec, DEFAULT_GRID_SPACING};
use crate::backend::test_support::{
    chymotrypsinogen, contact_poses, lysozyme, no_gpu, TestPair, CGN_TEMPERATURE,
};
use crate::backend::{EnergyBackend, GpuBackend, PoseParams};
use crate::icoscan::{compute_scan, ScanParams};
use crate::report::{compute_pmf, PmfResult};
use std::time::Instant;

/// The CLAUDE.md benchmark settings over `[rmin, rmax)` in steps of `dr`.
fn params(sys: &TestPair, rmin: f64, rmax: f64, dr: f64) -> ScanParams {
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

fn gpu_backend(sys: &TestPair) -> GpuBackend {
    let (a, b) = (sys.ref_a.clone(), sys.ref_b.clone());
    GpuBackend::new(a, b, &sys.splines, &sys.coulomb).unwrap()
}

fn grid_backend(sys: &TestPair, spacing: f64) -> GridBackend {
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

/// Only the prefiltered far map is interpolated: ~1e-3 kT at 2 Å on 2CDS,
/// halving with each halving of h² down to 1 Å. A bug wouldn't converge
/// that way.
#[test]
#[ignore]
fn grid_scan_matches_gpu_free_energy() {
    if no_gpu() {
        return;
    }
    let sys = lysozyme();
    let params = params(&sys, 26.0, 40.0, 1.0);
    let (expected, _) = scan(&params, &gpu_backend(&sys));
    let (coarse, fine) = (grid_backend(&sys, 2.0), grid_backend(&sys, 2f64.sqrt()));
    let err_coarse = worst_weighted_error(&expected, &scan(&params, &coarse).0);
    let err_fine = worst_weighted_error(&expected, &scan(&params, &fine).0);
    println!("worst |ΔF|: {err_coarse:.4} kT at 2 Å, {err_fine:.4} kT at 1.41 Å");
    assert!(err_coarse < 5e-3, "|ΔF| {err_coarse} kT at 2 Å");
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

/// 7KTZ coarse-grained with `model`, as grid and pair kernel.
fn cgn_backends(model: &str, single_bead: bool) -> (GridBackend, GpuBackend) {
    let sys = chymotrypsinogen(model, single_bead);
    (grid_backend(&sys, DEFAULT_GRID_SPACING), gpu_backend(&sys))
}

/// Pose by pose against exact pairs where B₂ is decided: attractive contacts,
/// for each model and bead policy. Each B bead sees ~100 far pairs, so any
/// bias of the far map shows here.
#[test]
#[ignore]
fn grid_scan_contact_energies_cgn() {
    if no_gpu() {
        return;
    }
    let poses = contact_poses(40_000, 33.0);
    for (model, single_bead) in itertools::iproduct!(["calvados3", "kimhummer"], [true, false]) {
        let (bias, ratio, n) = contact_errors(model, single_bead, &poses);
        let tag = format!("{model} {}", if single_bead { "single" } else { "multi" });
        println!(
            "{tag}: {n} attractive poses, mean βΔU {bias:+.4}; Σe^(−βU) grid/exact {ratio:.4}"
        );
        assert!(bias.abs() < 2e-3, "{tag}: contact bias {bias:+.4} kT");
        assert!((ratio - 1.0).abs() < 3e-3, "{tag}: weight ratio {ratio:.4}");
    }
}

/// Mean βΔU over attractive contact poses, the grid/exact ratio of their
/// summed Boltzmann weights, and the number of attractive poses.
fn contact_errors(model: &str, single_bead: bool, poses: &[PoseParams]) -> (f64, f64, usize) {
    let (grid, exact) = cgn_backends(model, single_bead);
    let (got, want) = (grid.compute_energies(poses), exact.compute_energies(poses));
    let kt = crate::thermal_energy(CGN_TEMPERATURE);
    let attractive: Vec<f64> = (got.iter().zip(&want))
        .filter(|(_, w)| **w < 0.0)
        .map(|(g, w)| (g - w) / kt)
        .collect();
    assert!(
        !attractive.is_empty(),
        "{model}: no attractive contact poses sampled"
    );
    let bias = attractive.iter().sum::<f64>() / attractive.len() as f64;
    let weight = |u: &[f64]| u.iter().map(|u| (-u / kt).exp()).sum::<f64>();
    (bias, weight(&got) / weight(&want), attractive.len())
}

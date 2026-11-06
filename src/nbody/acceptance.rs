//! Physics acceptance tests: whole runs against analytic liquid-state results.
//!
//! The slow ones are ignored by default; run them with
//! `cargo test --release -p duello --lib acceptance -- --ignored`.
//! Seeds are fixed, so outcomes are deterministic for unchanged code.

use super::test_tables::{table, KT};
use super::{
    Criteria, Device, EnergyStats, Equilibration, Frame, PairEnergy, Sampler, SamplerSpec,
    Scattering, ScatteringCurve, Status,
};
use crate::archive::CgBody;
use std::f64::consts::PI;
use std::sync::Arc;

const UNIT: &str = "X: { formfactor: !Constant 1.0 }\n";

fn point_body() -> CgBody {
    CgBody {
        positions: vec![[0.0; 3]],
        masses: vec![1.0],
        charges: vec![0.0],
        radii: vec![1.0],
        kinds: vec!["X".into()],
    }
}

/// 4π ∫ₐᵇ f(r) r² sin(qr)/(qr) dr by the midpoint rule; sin(qr)/(qr) → 1 at q = 0.
fn radial_fourier(f: impl Fn(f64) -> f64, a: f64, b: f64, q: f64) -> f64 {
    let steps = 2000;
    let dr = (b - a) / steps as f64;
    let sinc = |x: f64| if x == 0.0 { 1.0 } else { x.sin() / x };
    (0..steps)
        .map(|i| {
            let r = a + (i as f64 + 0.5) * dr;
            f(r) * r * r * sinc(q * r) * dr
        })
        .sum::<f64>()
        * 4.0
        * PI
}

/// Number density of spheres of diameter σ at packing fraction φ.
fn hard_sphere_density(sigma: f64, phi: f64) -> f64 {
    6.0 * phi / (PI * sigma.powi(3))
}

fn spec(n_bodies: usize, density: f64, translation: f64) -> SamplerSpec {
    SamplerSpec {
        n_bodies,
        box_length: (n_bodies as f64 / density).cbrt(),
        translation,
        rotation: 0.0, // point bodies: orientation is irrelevant
    }
}

/// Runs independent copies and returns the centre-of-mass S(q), which is
/// S_eff for single-bead bodies with a unit form factor.
fn structure_factor(
    energy: Arc<PairEnergy>,
    spec: &SamplerSpec,
    q_max: f64,
    copies: usize,
    (equilibration, production, every): (u64, u64, u64),
) -> ScatteringCurve {
    let mut scattering = pollster::block_on(Scattering::new(
        &point_body(),
        spec.n_bodies,
        spec.box_length,
        q_max,
        // Point bodies cost no more than mass centres.
        super::scattering::CENTRE_VECTORS_PER_BIN,
        UNIT,
        Device::Cpu,
    ))
    .unwrap();
    for copy in 0..copies {
        let mut sampler = Sampler::new(energy.clone(), spec.clone(), copy as u64 + 1).unwrap();
        sampler.advance(equilibration);
        for _ in 0..production / every {
            sampler.advance(every);
            let frame = Frame {
                copy,
                ..sampler.frame()
            };
            pollster::block_on(scattering.observe(&frame)).unwrap();
        }
    }
    scattering.curve()
}

/// Percus–Yevick S(q) for hard spheres of diameter σ at packing fraction φ:
/// S = 1/(1 − ρ ĉ(q)), with the analytic PY direct correlation function
/// c(r) = −λ₁ − 6φλ₂ (r/σ) − ½φλ₁ (r/σ)³ inside the core and zero outside.
fn percus_yevick(q: f64, sigma: f64, phi: f64) -> f64 {
    let lambda1 = (1.0 + 2.0 * phi).powi(2) / (1.0 - phi).powi(4);
    let lambda2 = -(1.0 + 0.5 * phi).powi(2) / (1.0 - phi).powi(4);
    let c = |r: f64| {
        let x = r / sigma;
        -lambda1 - 6.0 * phi * lambda2 * x - 0.5 * phi * lambda1 * x.powi(3)
    };
    1.0 / (1.0 - hard_sphere_density(sigma, phi) * radial_fourier(c, 0.0, sigma, q))
}

/// Hard core below 5 Å, a −1.5 kT well to rmax = 8 Å, nothing beyond.
const WELL: f64 = -1.5;

/// ∫ f(r) sin(qr)/(qr) 4πr² dr for the well potential, Mayer f = exp(−βu) − 1.
fn well_mayer_integral(q: f64) -> f64 {
    radial_fourier(|_| -1.0, 0.0, 5.0, q) + radial_fourier(|_| (-WELL).exp_m1(), 5.0, 8.0, q)
}

/// The analytic PY solution: S(0) = (1 − φ)⁴ / (1 + 2φ)², which the
/// quadrature must reproduce before it can judge a simulation.
#[test]
fn percus_yevick_quadrature_reproduces_the_compressibility() {
    let phi = 0.2;
    let s0 = (1.0f64 - phi).powi(4) / (1.0 + 2.0 * phi).powi(2);
    approx::assert_relative_eq!(percus_yevick(0.0, 5.0, phi), s0, max_relative = 1e-5);
}

/// duello's B₂ (from a PMF, as scans report it) and the Mayer integral the
/// dilute test checks against must agree: B₂ = −½ ∫ f d³r. This ties the
/// sampled S(q → 0) = 1 − 2B₂ρ to the B₂ that scans report.
#[test]
fn duello_b2_is_half_the_negative_mayer_integral() {
    let h = 0.001;
    let pmf = (0..3000).map(|i| (5.0 + (i as f64 + 0.5) * h, WELL));
    let b2 = crate::VirialCoeff::from_pmf(pmf, Some(5.0)).unwrap().b2();
    approx::assert_relative_eq!(b2, -0.5 * well_mayer_integral(0.0), max_relative = 1e-5);
}

/// Low-density limit, exact to first order in ρ at any q:
/// S(q) = 1 + ρ ∫ f(r) sin(qr)/(qr) 4πr² dr, whose q → 0 limit is 1 − 2B₂ρ.
/// Checks both each q-point and the pooled mean, whose smaller error would
/// catch a mis-scaled well (half the well gives S − 1 ≈ 0.008, not 0.03).
#[test]
#[ignore = "slow: run in release with --ignored"]
fn dilute_attractive_fluid_matches_the_second_virial_order() {
    // ρ ∫ f = 0.03: second-order terms, (ρ f̂)² plus three-body ones, stay
    // near 0.002, well below the 0.03 first-order signal.
    let density = 0.03 / well_mayer_integral(0.0);
    let spec = spec(100, density, 20.0);
    let energy = Arc::new(PairEnergy::new(table(|_, _, _| WELL * KT)).unwrap());
    let curve = structure_factor(energy, &spec, 0.3, 16, (200, 8000, 4));
    let errors = curve.s_eff_error.as_ref().unwrap();
    let mut deviations = Vec::new();
    for ((q, s), e) in curve.q.iter().zip(&curve.s_eff).zip(errors) {
        let expected = 1.0 + density * well_mayer_integral(*q);
        assert!(
            (s - expected).abs() < 4.0 * e + 0.005,
            "q = {q:.3}: S = {s:.4} ± {e:.4}, first order {expected:.4}"
        );
        deviations.push(s - expected);
    }
    let pooled = deviations.iter().sum::<f64>() / deviations.len() as f64;
    assert!(pooled.abs() < 0.005, "pooled deviation {pooled:.4}");
}

/// A table with zero energy beyond rmin is a hard-sphere fluid of diameter
/// σ = rmin. PY is accurate to a few per cent at φ = 0.2, including around
/// the main peak (qσ ≈ 7), which probes the structure at contact.
#[test]
#[ignore = "slow: run in release with --ignored"]
fn hard_spheres_match_percus_yevick() {
    let (sigma, phi) = (5.0, 0.2);
    let spec = spec(100, hard_sphere_density(sigma, phi), 1.5);
    let energy = Arc::new(PairEnergy::new(table(|_, _, _| 0.0)).unwrap());
    let curve = structure_factor(energy, &spec, 1.45, 8, (300, 3000, 5));
    assert!(curve.q.last().unwrap() * sigma > 6.5, "peak not covered");
    let errors = curve.s_eff_error.as_ref().unwrap();
    for ((q, s), e) in curve.q.iter().zip(&curve.s_eff).zip(errors) {
        let expected = percus_yevick(*q, sigma, phi);
        assert!(
            (s - expected).abs() < 4.0 * e + 0.03 * expected,
            "q = {q:.3}: S = {s:.3} ± {e:.3}, PY {expected:.3}"
        );
    }
}

/// From random insertion, the drift test must declare convergence, and the
/// mean energy after it must match a run three times as long: otherwise the
/// thresholds would let relaxation leak into production.
#[test]
#[ignore = "slow: run in release with --ignored"]
fn equilibration_verdict_leaves_no_drift_in_production() {
    let copies = 4;
    let spec = spec(40, 2.0e-3, 2.0);
    let energy = Arc::new(PairEnergy::new(table(|_, _, _| WELL * KT)).unwrap());
    let new_sampler = |seed: u64| Sampler::new(energy.clone(), spec.clone(), seed).unwrap();
    let mut samplers: Vec<Sampler> = (0..copies).map(|c| new_sampler(c as u64 + 1)).collect();
    let criteria = Criteria {
        min_sweeps: 100,
        max_sweeps: 20_000,
        tolerance: 2.0,
        max_drift: 0.5,
    };
    let mut equilibration = Equilibration::new(criteria);
    let mut stats = EnergyStats::new(energy.clone());
    let sweep = |samplers: &mut Vec<Sampler>, stats: &mut EnergyStats| -> Vec<f64> {
        samplers
            .iter_mut()
            .map(|s| {
                s.advance(5);
                stats.observe(&s.frame()).unwrap()
            })
            .collect()
    };
    let converged_at = loop {
        let energies = sweep(&mut samplers, &mut stats);
        match equilibration.push(samplers[0].frame().sweep, &energies) {
            Status::Running => continue,
            Status::Converged => break samplers[0].frame().sweep,
            Status::CapReached => panic!("no convergence within {} sweeps", criteria.max_sweeps),
        }
    };
    let production = |samplers: &mut Vec<Sampler>, sweeps: u64| {
        let mut stats = EnergyStats::new(energy.clone());
        for _ in 0..sweeps / 5 {
            sweep(samplers, &mut stats);
        }
        stats.summary()
    };
    let short = production(&mut samplers, converged_at);
    let mut reference: Vec<Sampler> = (0..copies).map(|c| new_sampler(100 + c as u64)).collect();
    reference
        .iter_mut()
        .for_each(|s| s.advance(3 * converged_at));
    let long = production(&mut reference, 3 * converged_at);
    let error = (short.variance / short.frames as f64).sqrt() * 10.0; // frames are correlated
    assert!(
        (short.mean - long.mean).abs() < error + 0.02 * long.mean.abs(),
        "after convergence at sweep {converged_at}: {:.3} vs reference {:.3}",
        short.mean,
        long.mean
    );
}

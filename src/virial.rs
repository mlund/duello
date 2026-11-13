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

use crate::report::PmfPoint;
use anyhow::ensure;
use itertools::Itertools;
use num_traits::Inv;
use physical_constants::AVOGADRO_CONSTANT;
use serde::ser::SerializeStruct;
use std::{f64::consts::PI, ops::Neg};

/// Struct for handling osmotic second virial coefficients, B2
#[derive(Debug, Clone)]
pub struct VirialCoeff {
    /// Osmotic second virial coefficient, B2 (Å³)
    b2: f64,
    /// Distance of closest approach, σ (Å)
    sigma: f64,
    /// Steric B₂₂ʳᵉᶠ (Å³) that B₂₂* and K_a are relative to; 2πσ³/3 until replaced.
    reference: f64,
}

impl VirialCoeff {
    /// Calculates B2 from PMF data, pairs of (r, w(r)).
    /// w(r) should be in units of kT; r should be equidistant and in Å.
    /// 𝐵₂ = -½ ∫ [ exp(-𝛽𝑤(𝑟) ) - 1 ] 4π𝑟² d𝑟
    /// If σ is not provided, it is assumed to be the first distance in the PMF.
    pub fn from_pmf(
        pomf: impl IntoIterator<Item = (impl Into<f64>, impl Into<f64>)>,
        sigma: Option<f64>,
    ) -> anyhow::Result<Self> {
        let pomf = pomf
            .into_iter()
            .map(|(r, w)| (r.into(), w.into()))
            .collect_vec();
        Self::from_pmf_slice(pomf.as_slice(), sigma)
    }

    /// Calculates B2 from PMF data, pairs of (r, w(r)).
    /// w(r) should be in units of kT; r should be equidistant and in Å.
    /// 𝐵₂ = -½ ∫ [ exp(-𝛽𝑤(𝑟) ) - 1 ] 4π𝑟² d𝑟
    /// If σ is not provided, it is assumed to be the first distance in the PMF.
    pub fn from_pmf_slice(pomf: &[(f64, f64)], sigma: Option<f64>) -> anyhow::Result<Self> {
        ensure!(pomf.len() >= 2, "PMF must have at least two points");
        ensure!(
            pomf.iter().map(|(r, _)| *r).is_sorted(),
            "PMF distances must be in ascending order"
        );
        let sigma = sigma.unwrap_or(pomf[0].0); // closest distance, "σ"
        let b2 = shells(pomf, |(r, _)| *r, sigma)
            .expect("two points checked above")
            .map(|((_, w), weight)| mayer(*w) * weight)
            .sum::<f64>();
        Ok(Self::from_raw_parts(hardsphere(sigma) - b2, sigma))
    }

    /// B₂₂ integrated out to each R ≥ σ, to judge convergence; the last value is [`Self::b2`].
    pub fn running(pmf: &[PmfPoint], sigma: f64) -> Vec<(f64, f64)> {
        shells(pmf, |p| p.r.into(), sigma)
            .into_iter()
            .flatten()
            .scan(hardsphere(sigma), |b2, (p, weight)| {
                *b2 -= mayer(p.free_energy.into()) * weight;
                Some((p.r.into(), *b2))
            })
            .collect()
    }

    /// B₂₂ as if the scan had started at each R′ ≥ σ, with hard spheres below R′.
    /// A plateau towards σ means R_min is small enough; the first value is [`Self::b2`]
    /// when σ lies on the grid, as it does by default.
    pub fn by_start(pmf: &[PmfPoint], sigma: f64) -> Vec<(f64, f64)> {
        let shells = shells(pmf, |p| p.r.into(), sigma)
            .into_iter()
            .flatten()
            .collect_vec();
        // tail = Σ_{R≥R′} Mayer shells
        let mut by_start = shells
            .iter()
            .rev()
            .scan(0.0, |tail, (p, weight)| {
                *tail += mayer(p.free_energy.into()) * weight;
                let r = f64::from(p.r);
                Some((r, hardsphere(r) - *tail))
            })
            .collect_vec();
        by_start.reverse();
        by_start
    }

    /// T·∂B₂₂/∂T (Å³) at a fixed pair potential, so the T dependence of εᵣ, κ and
    /// titration is left out; `None` below two points.
    pub fn temperature_slope(pmf: &[PmfPoint], sigma: f64) -> Option<f64> {
        // Gibbs–Helmholtz: ∂(βw)/∂β = ⟨U⟩, so ∂(e^{−βw} − 1)/∂β = −e^{−βw}·⟨U⟩.
        let sum = shells(pmf, |p| p.r.into(), sigma)?
            .map(|(p, weight)| boltzmann_mean_energy(p) * weight)
            .sum::<f64>();
        Some(-sum)
    }

    /// Constructs a new VirialCoeff from raw parts, with the hard-sphere reference at σ
    pub fn from_raw_parts(b2: f64, sigma: f64) -> Self {
        Self {
            b2,
            sigma,
            reference: hardsphere(sigma),
        }
    }

    /// Takes B₂₂* and K_a relative to `b2_ref` (Å³), e.g. from [`crate::steric::Steric`].
    pub fn with_reference(self, b2_ref: f64) -> Self {
        Self {
            reference: b2_ref,
            ..self
        }
    }

    /// Virial coefficient, B2 (Å³)
    pub const fn b2(&self) -> f64 {
        self.b2
    }
    /// Distance of closest approach, σ (Å)
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }
    /// Reference B₂₂ʳᵉᶠ (Å³)
    pub const fn reference(&self) -> f64 {
        self.reference
    }
    /// Diameter of hard spheres with the reference's B₂₂, (3B₂₂ʳᵉᶠ/2π)^⅓ (Å)
    pub fn sigma_eff(&self) -> f64 {
        (3.0 * self.reference / (2.0 * PI)).cbrt()
    }
    /// Reduced second virial coefficient, B2 / B2ref
    pub fn reduced(&self) -> f64 {
        self.b2 / self.reference
    }
    /// Association constant, 𝐾𝑑⁻¹ = -2(𝐵₂ - 𝐵₂ʳᵉᶠ)
    /// See "Colloidal Domain" by Evans and Wennerström, 2nd Ed, p. 408
    pub fn association_const(&self) -> Option<f64> {
        Some(
            -2.0 * (self.b2 - self.reference) * crate::LITER_PER_CUBIC_ANGSTROM * AVOGADRO_CONSTANT,
        )
        .filter(|x| x.is_sign_positive())
    }
    /// Dissociation constant, 𝐾𝑑
    /// See "Colloidal Domain" by Evans and Wennerström, 2nd Ed, p. 408
    pub fn dissociation_const(&self) -> Option<f64> {
        self.association_const().map(Inv::inv)
    }
    /// Virial coefficient, B2 in mol⋅ml/g². Molar weights in g/mol.
    pub fn mol_ml_per_gram2(&self, mw1: f64, mw2: f64) -> f64 {
        const ML_PER_ANGSTROM3: f64 = 1e-24;
        self.b2 * ML_PER_ANGSTROM3 / (mw1 * mw2) * AVOGADRO_CONSTANT
    }
}

/// Association defined from B₂₂ alone, K_a = −2(B₂₂ − B₂₂ʳᵉᶠ), so it needs no
/// contact criterion. Energies are in kT at a 1 M standard state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Association {
    /// K_a (1/M)
    pub ka: f64,
    /// ΔU° = −∂ln K_a/∂β at a fixed pair potential
    pub delta_u: f64,
}

impl Association {
    /// `None` where K_a ≤ 0, i.e. no net attraction beyond the reference.
    /// `pmf` is the one `virial` was computed from.
    pub fn new(virial: &VirialCoeff, pmf: &[PmfPoint]) -> Option<Self> {
        let ka = virial.association_const().filter(|&ka| ka > 0.0)?;
        // K_a ∝ B₂₂ʳᵉᶠ − B₂₂, and hard beads don't depend on T, so
        // −∂ln K_a/∂β = T·∂B₂₂/∂T / (B₂₂ − B₂₂ʳᵉᶠ) in kT.
        let excess = virial.b2() - virial.reference();
        let delta_u = VirialCoeff::temperature_slope(pmf, virial.sigma())? / excess;
        Some(Self { ka, delta_u })
    }

    /// ΔA° = −ln(K_a·1 M)
    pub fn delta_a(&self) -> f64 {
        -self.ka.ln()
    }

    /// ΔA°, ΔU° and ΔS° at `temperature` (K), in the units people quote.
    pub fn molar(&self, temperature: f64) -> MolarAssociation {
        let kt = crate::thermal_energy(temperature); // kJ/mol
        MolarAssociation {
            delta_a: self.delta_a() * kt,
            delta_u: self.delta_u * kt,
            // ΔS°/k = ΔU° − ΔA° in kT, so R times it is J/(K·mol).
            delta_s: (self.delta_u - self.delta_a()) * kt * 1e3 / temperature,
        }
    }
}

/// [`Association`] energetics at a 1 M standard state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MolarAssociation {
    /// kJ/mol
    pub delta_a: f64,
    /// kJ/mol
    pub delta_u: f64,
    /// J/(K·mol)
    pub delta_s: f64,
}

/// Above this e^{−w/kT} at σ, the pair still meets there, so B₂₂ depends on R_min.
pub const OPEN_AT_SIGMA: f64 = 0.01;

/// Whether a [`weight_at_sigma`] is large enough that B₂₂ depends on R_min;
/// the one rule CLI and GUI share.
pub fn open_at_sigma(weight: f64) -> bool {
    weight >= OPEN_AT_SIGMA
}

/// e^{−w/kT} at σ. The hard-sphere fill below σ is exact only if this is ≈ 0.
pub fn weight_at_sigma(pmf: &[PmfPoint], sigma: f64) -> Option<f64> {
    let (p, _) = shells(pmf, |p| p.r.into(), sigma)?.next()?;
    Some(f64::from(p.free_energy).neg().exp())
}

/// e^{−βw} − 1, with w in kT.
fn mayer(w: f64) -> f64 {
    w.neg().exp_m1()
}

/// e^{−βw}·β⟨U⟩, the integrand of every β-derivative of B₂₂.
fn boltzmann_mean_energy(p: &PmfPoint) -> f64 {
    f64::from(p.free_energy).neg().exp() * f64::from(p.mean_energy)
}

/// 2π σ³/3 (Å³)
fn hardsphere(sigma: f64) -> f64 {
    2.0 * PI / 3.0 * sigma.powi(3)
}

/// Each point at r ≥ σ with its quadrature weight 2π·dr·r², the one rule every
/// B₂₂-derived quantity integrates with so they cannot drift apart. dr comes
/// from the first two points, so the grid must be equidistant; `None` below two.
fn shells<P>(
    points: &[P],
    r: impl Fn(&P) -> f64 + Copy,
    sigma: f64,
) -> Option<impl Iterator<Item = (&P, f64)>> {
    let dr = r(points.get(1)?) - r(&points[0]);
    // PMF radii are f32, so σ given in f64 (e.g. 26.3) can sit just above its own point.
    let from = sigma - 1e-3 * dr;
    Some(
        points
            .iter()
            .filter(move |p| r(p) >= from)
            .map(move |p| (p, 2.0 * PI * dr * r(p).powi(2))),
    )
}

// Implement serialize which writes B2, hardsphere, and dissociation constant
impl serde::Serialize for VirialCoeff {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("VirialCoeff", 6)?;
        state.serialize_field("B2", &self.b2)?;
        state.serialize_field("B2_ref", &self.reference)?;
        state.serialize_field("sigma_eff", &self.sigma_eff())?;
        state.serialize_field("B2_reduced", &self.reduced())?;
        state.serialize_field("sigma", &self.sigma())?;
        if let Some(kd) = self.dissociation_const() {
            state.serialize_field("kD", &kd)?;
        }
        state.end()
    }
}

impl From<VirialCoeff> for f64 {
    fn from(v: VirialCoeff) -> Self {
        v.b2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn point(r: f64, free_energy: f64, mean_energy: f64) -> PmfPoint {
        PmfPoint {
            r: r as f32,
            free_energy: free_energy as f32,
            mean_energy: mean_energy as f32,
            energy_variance: 0.0,
        }
    }

    /// w = ⟨U⟩ = −ε for σ ≤ r < 13 Å and 0 beyond; σ = 10 Å, dr = 1 Å.
    fn square_well(epsilon: f64) -> Vec<PmfPoint> {
        (10..20)
            .map(|r| {
                let u = if r < 13 { -epsilon } else { 0.0 };
                point(r as f64, u, u)
            })
            .collect()
    }

    fn virial_of(pmf: &[PmfPoint]) -> VirialCoeff {
        VirialCoeff::from_pmf(pmf.iter().map(|p| (p.r, p.free_energy)), Some(10.0)).unwrap()
    }

    #[test]
    fn temperature_slope_of_a_square_well() {
        let epsilon = 1.5;
        let pmf = square_well(epsilon);
        let slope = VirialCoeff::temperature_slope(&pmf, 10.0).unwrap();

        // −2π·dr·Σ e^{−βw}·β⟨U⟩·r² over the well
        let by_hand = 2.0 * PI * epsilon * epsilon.exp() * (100.0 + 121.0 + 144.0);
        assert_relative_eq!(slope, by_hand, max_relative = 1e-6);

        // T∂B/∂T = −β∂B/∂β, and βw scales with β when ⟨U⟩ = w.
        let b2_at = |scale: f64| {
            let scaled = pmf.iter().map(|p| (p.r, f64::from(p.free_energy) * scale));
            VirialCoeff::from_pmf(scaled, Some(10.0)).unwrap().b2()
        };
        let delta = 1e-3;
        let finite_difference = -(b2_at(1.0 + delta) - b2_at(1.0 - delta)) / (2.0 * delta);
        assert_relative_eq!(slope, finite_difference, max_relative = 1e-4);
    }

    #[test]
    fn association_of_a_square_well_and_none_when_repulsive() {
        let epsilon = 1.5;
        let pmf = square_well(epsilon);
        let association = Association::new(&virial_of(&pmf), &pmf).unwrap();

        let shells = 100.0 + 121.0 + 144.0;
        let ka_cubic_angstrom = 4.0 * PI * (epsilon.exp() - 1.0) * shells;
        let ka = ka_cubic_angstrom * 1e-27 * AVOGADRO_CONSTANT;
        // ΔU° = 4π∫e^{−βw}⟨U⟩r²dr / K_a; the −1 in K_a's integrand gives e^ε − 1.
        let delta_u = -epsilon * epsilon.exp() / (epsilon.exp() - 1.0);
        assert_relative_eq!(association.ka, ka, max_relative = 1e-6);
        assert_relative_eq!(association.delta_a(), -ka.ln(), max_relative = 1e-6);
        assert_relative_eq!(association.delta_u, delta_u, max_relative = 1e-6);

        let repulsive: Vec<_> = (10..20).map(|r| point(r as f64, 2.0, 2.0)).collect();
        assert_eq!(Association::new(&virial_of(&repulsive), &repulsive), None);
    }

    /// K_a = e² 1/M and ΔU° = −5 kT at 300 K, where kT = 2.494339 kJ/mol:
    /// ΔA° = −2 kT, and ΔS° = (ΔU° − ΔA°)/T = −3R.
    #[test]
    fn association_energetics_in_molar_units() {
        let association = Association {
            ka: 2f64.exp(),
            delta_u: -5.0,
        };
        let molar = association.molar(300.0);
        assert_relative_eq!(molar.delta_a, -4.988678, max_relative = 1e-6);
        assert_relative_eq!(molar.delta_u, -12.471694, max_relative = 1e-6);
        assert_relative_eq!(molar.delta_s, -24.943388, max_relative = 1e-6);
    }

    #[test]
    fn weight_at_sigma_is_boltzmann_factor_of_first_point() {
        let pmf = [
            point(25.5, 0.0, 0.0),
            point(26.0, 10f64.ln(), 0.0),
            point(26.5, 0.0, 0.0),
        ];
        assert_relative_eq!(
            weight_at_sigma(&pmf, 26.0).unwrap(),
            0.1,
            max_relative = 1e-6
        );
    }

    /// Hard core of diameter 10 Å scanned from σ = 5 Å: any start inside the core gives
    /// the same B₂₂, the hard sphere 2π·10³/3.
    #[test]
    fn b2_by_start_plateaus_inside_a_hard_core() {
        let pmf: Vec<_> = (0..150)
            .map(|k| point(5.0 + 0.1 * k as f64, if k < 50 { 50.0 } else { 0.0 }, 0.0))
            .collect();
        let by_start = VirialCoeff::by_start(&pmf, 5.0);

        let inside: Vec<_> = by_start.iter().filter(|(r, _)| *r <= 10.0).collect();
        assert_eq!(inside.len(), 51);
        for (_, b2) in inside {
            assert_relative_eq!(*b2, 2094.4, max_relative = 0.02);
        }
        let pairs = pmf.iter().map(|p| (p.r, p.free_energy));
        let b2 = VirialCoeff::from_pmf(pairs, Some(5.0)).unwrap().b2();
        assert_relative_eq!(by_start[0].1, b2, max_relative = 1e-9);
    }

    /// σ = 10 Å would give a hard-sphere value of 2094 Å³; the reference replaces it.
    #[test]
    fn ratios_use_the_reference_not_sigma() {
        let virial = VirialCoeff::from_raw_parts(-1000.0, 10.0).with_reference(500.0);
        assert_eq!(virial.reference(), 500.0);
        assert_relative_eq!(virial.reduced(), -2.0);
        let ka = 3000.0 * 1e-27 * AVOGADRO_CONSTANT;
        assert_relative_eq!(
            virial.association_const().unwrap(),
            ka,
            max_relative = 1e-12
        );
    }

    /// 26.3f32 widens to 26.2999992, below σ = 26.3 in f64.
    #[test]
    fn sigma_in_f64_keeps_its_own_f32_point() {
        let pmf = [point(26.3, -1.0, 0.0), point(26.8, 0.0, 0.0)];
        assert_eq!(VirialCoeff::running(&pmf, 26.3).len(), 2);
    }

    #[test]
    fn running_b2_adds_one_shell_per_point_and_ends_at_b2() {
        let pmf = [
            point(10.0, 1.0, 0.0),
            point(11.0, -0.5, 0.0),
            point(12.0, 0.0, 0.0),
        ];
        let running = VirialCoeff::running(&pmf, 10.0);

        let hardsphere = 2.0 * PI / 3.0 * 1000.0;
        let at_10 = hardsphere - 2.0 * PI * ((-1.0f64).exp() - 1.0) * 100.0;
        let at_11 = at_10 - 2.0 * PI * (0.5f64.exp() - 1.0) * 121.0;
        let rs: Vec<f64> = running.iter().map(|p| p.0).collect();
        assert_eq!(rs, [10.0, 11.0, 12.0]);
        assert_relative_eq!(running[0].1, at_10, max_relative = 1e-6);
        assert_relative_eq!(running[1].1, at_11, max_relative = 1e-6);
        assert_relative_eq!(running[2].1, at_11, max_relative = 1e-6);

        let pairs: Vec<_> = pmf.iter().map(|p| (p.r, p.free_energy)).collect();
        let b2 = VirialCoeff::from_pmf(pairs, Some(10.0)).unwrap().b2();
        assert_relative_eq!(running[2].1, b2, max_relative = 1e-6);
    }
    #[test]
    fn test_virial_coeff() {
        let pmf = vec![
            (37.0, 20.0772),
            (38.0, 10.3099),
            (39.0, 4.8785),
            (40.0, 1.6420),
            (41.0, -0.2038),
            (42.0, -0.8156),
            (43.0, -0.8042),
            (44.0, -0.6059),
            (45.0, -0.3888),
            (46.0, -0.2398),
            (47.0, -0.1417),
            (48.0, -0.0774),
            (49.0, -0.0356),
        ];
        let virial = VirialCoeff::from_pmf(pmf.iter().cloned(), None).unwrap();
        assert_relative_eq!(virial.b2(), 87041.72463623626, epsilon = 1e-3);
        assert_relative_eq!(virial.reference(), 106087.39512152252);
        assert_relative_eq!(virial.sigma(), 37.0);
        assert_relative_eq!(virial.reduced(), 0.820471881098885);
        assert_relative_eq!(virial.dissociation_const().unwrap(), 0.04359361011881143);

        let virial = VirialCoeff::from_pmf(pmf.iter().cloned(), Some(40.0)).unwrap();
        assert_relative_eq!(virial.b2(), 87837.30466559599);
        assert_relative_eq!(virial.reference(), 134041.2865531645);
        assert_relative_eq!(virial.sigma(), 40.0);
        assert_relative_eq!(virial.reduced(), 0.6553003699405502);
        assert_relative_eq!(virial.dissociation_const().unwrap(), 0.017969653256450453);

        let virial = VirialCoeff::from_raw_parts(16760.0, 20.0);
        assert!(virial.association_const().is_none());
    }
}

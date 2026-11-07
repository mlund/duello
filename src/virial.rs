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
        Ok(Self {
            b2: hardsphere(sigma) - b2,
            sigma,
        })
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

    /// T·∂B₂₂/∂T (Å³) at a fixed pair potential, so the T dependence of εᵣ, κ and
    /// titration is left out; `None` below two points.
    pub fn temperature_slope(pmf: &[PmfPoint], sigma: f64) -> Option<f64> {
        // Gibbs–Helmholtz: ∂(βw)/∂β = ⟨U⟩, so ∂(e^{−βw} − 1)/∂β = −e^{−βw}·⟨U⟩.
        let sum = shells(pmf, |p| p.r.into(), sigma)?
            .map(|(p, weight)| boltzmann_mean_energy(p) * weight)
            .sum::<f64>();
        Some(-sum)
    }

    /// Constructs a new VirialCoeff from raw parts
    pub const fn from_raw_parts(b2: f64, sigma: f64) -> Self {
        Self { b2, sigma }
    }

    /// Virial coefficient, B2 (Å³)
    pub const fn b2(&self) -> f64 {
        self.b2
    }
    /// Distance of closest approach, σ (Å)
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }
    /// Hard sphere contribution to second virial coefficient, B2hs (Å³)
    pub fn hardsphere(&self) -> f64 {
        hardsphere(self.sigma)
    }
    /// Reduced second virial coefficient, B2 / B2hs
    pub fn reduced(&self) -> f64 {
        self.b2 / self.hardsphere()
    }
    /// Association constant, 𝐾𝑑⁻¹ = -2(𝐵₂ - 𝐵₂hs)
    /// See "Colloidal Domain" by Evans and Wennerström, 2nd Ed, p. 408
    pub fn association_const(&self) -> Option<f64> {
        Some(
            -2.0 * (self.b2 - self.hardsphere())
                * crate::LITER_PER_CUBIC_ANGSTROM
                * AVOGADRO_CONSTANT,
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

/// Association defined from B₂₂ alone, K_a = −2(B₂₂ − B₂₂ʰˢ), so it needs no
/// contact criterion. Energies are in kT at a 1 M standard state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Association {
    /// K_a (1/M)
    pub ka: f64,
    /// ΔU° = −∂ln K_a/∂β at a fixed pair potential
    pub delta_u: f64,
}

impl Association {
    /// `None` where K_a ≤ 0, i.e. no net attraction beyond hard spheres.
    pub fn from_pmf(pmf: &[PmfPoint], sigma: f64) -> Option<Self> {
        let pairs = pmf.iter().map(|p| (p.r, p.free_energy));
        let virial = VirialCoeff::from_pmf(pairs, Some(sigma)).ok()?;
        let ka = virial.association_const().filter(|&ka| ka > 0.0)?;
        // K_a ∝ B₂₂ʰˢ − B₂₂, so −∂ln K_a/∂β = T·∂B₂₂/∂T / (B₂₂ − B₂₂ʰˢ) in kT.
        let excess = virial.b2() - virial.hardsphere();
        let delta_u = VirialCoeff::temperature_slope(pmf, sigma)? / excess;
        Some(Self { ka, delta_u })
    }

    /// ΔA° = −ln(K_a·1 M)
    pub fn delta_a(&self) -> f64 {
        -self.ka.ln()
    }

    /// −TΔS° = ΔA° − ΔU°
    pub fn minus_t_delta_s(&self) -> f64 {
        self.delta_a() - self.delta_u
    }
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
        let mut state = serializer.serialize_struct("VirialCoeff", 4)?;
        state.serialize_field("B2", &self.b2)?;
        state.serialize_field("B2_hs", &self.hardsphere())?;
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
        let association = Association::from_pmf(&square_well(epsilon), 10.0).unwrap();

        let shells = 100.0 + 121.0 + 144.0;
        let ka_cubic_angstrom = 4.0 * PI * (epsilon.exp() - 1.0) * shells;
        let ka = ka_cubic_angstrom * 1e-27 * AVOGADRO_CONSTANT;
        // ΔU° = 4π∫e^{−βw}⟨U⟩r²dr / K_a; the −1 in K_a's integrand gives e^ε − 1.
        let delta_u = -epsilon * epsilon.exp() / (epsilon.exp() - 1.0);
        assert_relative_eq!(association.ka, ka, max_relative = 1e-6);
        assert_relative_eq!(association.delta_a(), -ka.ln(), max_relative = 1e-6);
        assert_relative_eq!(association.delta_u, delta_u, max_relative = 1e-6);
        assert_relative_eq!(
            association.minus_t_delta_s(),
            -ka.ln() - delta_u,
            max_relative = 1e-6
        );

        let repulsive: Vec<_> = (10..20).map(|r| point(r as f64, 2.0, 2.0)).collect();
        assert_eq!(Association::from_pmf(&repulsive, 10.0), None);
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
        assert_relative_eq!(virial.hardsphere(), 106087.39512152252);
        assert_relative_eq!(virial.sigma(), 37.0);
        assert_relative_eq!(virial.reduced(), 0.820471881098885);
        assert_relative_eq!(virial.dissociation_const().unwrap(), 0.04359361011881143);

        let virial = VirialCoeff::from_pmf(pmf.iter().cloned(), Some(40.0)).unwrap();
        assert_relative_eq!(virial.b2(), 87837.30466559599);
        assert_relative_eq!(virial.hardsphere(), 134041.2865531645);
        assert_relative_eq!(virial.sigma(), 40.0);
        assert_relative_eq!(virial.reduced(), 0.6553003699405502);
        assert_relative_eq!(virial.dissociation_const().unwrap(), 0.017969653256450453);

        let virial = VirialCoeff::from_raw_parts(16760.0, 20.0);
        assert!(virial.association_const().is_none());
    }
}

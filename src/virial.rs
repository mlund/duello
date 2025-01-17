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

use itertools::Itertools;
use num_traits::Inv;
use physical_constants::AVOGADRO_CONSTANT;
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
        // use first two distances to calculate dr and assume it's constant
        let (r0, r1) = pomf
            .iter()
            .map(|pair| pair.0)
            .take(2)
            .collect_tuple()
            .ok_or_else(|| anyhow::anyhow!("Error calculating PMF dr"))?;
        let dr = r1 - r0;
        if dr <= 0.0 {
            anyhow::bail!("Negative dr in PMF");
        }
        let sigma = sigma.unwrap_or(r0); // closest distance, "σ"
        let b2_hardsphere = 2.0 * PI / 3.0 * sigma.powi(3);
        // integrate
        let b2 = pomf
            .iter()
            .filter(|(r, _)| *r >= sigma)
            .map(|(r, w)| w.neg().exp_m1() * r * r)
            .sum::<f64>()
            .mul_add(-2.0 * PI * dr, b2_hardsphere);
        Ok(Self { b2, sigma })
    }

    /// Constructs a new VirialCoeff from raw parts
    pub fn from_raw_parts(b2: f64, sigma: f64) -> Self {
        Self { b2, sigma }
    }

    /// Virial coefficient, B2 (Å³)
    pub fn b2(&self) -> f64 {
        self.b2
    }
    /// Distance of closest approach, σ (Å)
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    /// Hard sphere contribution to second virial coefficient, B2hs (Å³)
    pub fn hardsphere(&self) -> f64 {
        2.0 * PI / 3.0 * self.sigma.powi(3)
    }
    /// Reduced second virial coefficient, B2 / B2hs
    pub fn reduced(&self) -> f64 {
        self.b2 / self.hardsphere()
    }
    /// Association constant, 𝐾𝑑⁻¹ = -2(𝐵₂ - 𝐵₂hs)
    /// See "Colloidal Domain" by Evans and Wennerström, 2nd Ed, p. 408
    pub fn association_const(&self) -> Option<f64> {
        const LITER_PER_CUBIC_ANGSTROM: f64 = 1e-27;
        let association_const =
            -2.0 * (self.b2 - self.hardsphere()) * LITER_PER_CUBIC_ANGSTROM * AVOGADRO_CONSTANT;
        if association_const.is_sign_positive() {
            Some(association_const)
        } else {
            None
        }
    }
    /// Dissociation constant, 𝐾𝑑
    /// See "Colloidal Domain" by Evans and Wennerström, 2nd Ed, p. 408
    pub fn dissociation_const(&self) -> Option<f64> {
        self.association_const().map(|k| k.inv())
    }
    /// Virial coefficient, B2 in mol⋅ml/g². Molar weights in g/mol.
    pub fn mol_ml_per_gram2(&self, mw1: f64, mw2: f64) -> f64 {
        const ML_PER_ANGSTROM3: f64 = 1e-24;
        self.b2 * ML_PER_ANGSTROM3 / (mw1 * mw2) * AVOGADRO_CONSTANT
    }
}

impl From<VirialCoeff> for f64 {
    fn from(v: VirialCoeff) -> f64 {
        v.b2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
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
        assert_relative_eq!(virial.b2(), 87041.72463623626);
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
    }
}

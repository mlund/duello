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

use crate::structure::Structure;
use crate::Vector3;
use faunus::interatomic::coulomb::pairwise::{MultipolePotential as _, Plain};
use faunus::interatomic::coulomb::permittivity::ConstantPermittivity;
use faunus::interatomic::coulomb::{DebyeLength as _, Medium};
use faunus::interatomic::twobody::{IsotropicTwobodyEnergy, SplineConfig, SplinedPotential};
use faunus::topology::CustomProperty;
use faunus::{energy::NonbondedMatrix, topology::AtomKind};

/// Get excess polarizability (Å³) from the custom "alpha" property of an atom kind.
fn get_alpha(atom: &AtomKind) -> f64 {
    atom.get_property("alpha").map_or(0.0, |v| {
        f64::try_from(v).expect("Failed to convert alpha to f64")
    })
}

/// Check if an atom pair requires ion-induced dipole interaction.
fn needs_polarization(alpha1: f64, charge1: f64, alpha2: f64, charge2: f64) -> bool {
    (alpha1 * charge2).abs() > 1e-6 || (alpha2 * charge1).abs() > 1e-6
}

/// Coulomb parameters for analytical evaluation in GPU/SIMD backends.
///
/// Splining the combined SR+Coulomb potential causes polynomial ringing when the
/// SR hard cutoff (~20Å) falls inside the spline grid. Evaluating Coulomb/Yukawa
/// analytically as `prefactor * exp(-kappa * r) / r` avoids this entirely.
pub struct CoulombParams {
    /// Per-pair prefactors in kJ/mol·Å, indexed as `[id_a * n_types + id_b]`.
    /// Each entry = `ELECTRIC_PREFACTOR * z_i * z_j / ε_r`.
    /// Zero prefactor means no electrostatic interaction for that pair.
    /// Stored as f32 because GPU/SIMD backends consume them directly.
    pub prefactors: Vec<f32>,
    /// Inverse Debye screening length κ (1/Å). 0.0 for unscreened Coulomb.
    pub kappa: f32,
    /// Cached to avoid `isqrt(prefactors.len())` on every pair lookup.
    n_types: usize,
}

impl CoulombParams {
    /// Create zero-valued Coulomb params (no analytical Coulomb contribution).
    pub fn zero(n_types: usize) -> Self {
        Self {
            prefactors: vec![0.0; n_types * n_types],
            kappa: 0.0,
            n_types,
        }
    }
}

/// Extract Coulomb prefactors for analytical evaluation.
///
/// Returns `None` if any pair requires ion-induced dipole interaction (IonIonPolar),
/// whose energy formula is too complex for the simple `prefactor * exp(-κr) / r`
/// analytical path.
pub fn extract_coulomb_params(
    atomkinds: &[AtomKind],
    permittivity: ConstantPermittivity,
    kappa: Option<f64>,
) -> Option<CoulombParams> {
    let n_types = atomkinds.len();
    let eps_r = f64::from(permittivity);
    let to_kjmol = faunus::interatomic::ELECTRIC_PREFACTOR / eps_r;

    let alphas: Vec<f64> = atomkinds.iter().map(get_alpha).collect();
    let charges: Vec<f64> = atomkinds.iter().map(AtomKind::charge).collect();

    let mut prefactors = vec![0.0f32; n_types * n_types];
    for i in 0..n_types {
        for j in 0..n_types {
            if needs_polarization(alphas[i], charges[i], alphas[j], charges[j]) {
                log::warn!(
                    "Ion-induced dipole detected for pair ({i}, {j}); \
                     analytical Coulomb not supported"
                );
                return None;
            }
            prefactors[i * n_types + j] = (to_kjmol * charges[i] * charges[j]) as f32;
        }
    }

    let params = CoulombParams {
        prefactors,
        kappa: kappa.unwrap_or(0.0) as f32,
        n_types,
    };
    debug_assert_eq!(params.prefactors.len(), n_types * n_types);
    Some(params)
}

/// Splined pair potentials for GPU/SIMD backends.
///
/// Created via [`PairMatrix::create_sr_splines`] and consumed by backend constructors.
/// Owns the data directly rather than wrapping faunus's `NonbondedMatrixSplined`,
/// so Duello is decoupled from faunus's internal matrix layout.
pub struct SplinedPotentials {
    potentials: Vec<SplinedPotential>,
    n_types: usize,
}

impl SplinedPotentials {
    /// Spline potentials from `NonbondedMatrix`, treating entries opaquely
    /// via [`IsotropicTwobodyEnergy`].
    fn from_nonbonded(nonbonded: &NonbondedMatrix, cutoff: f64, config: SplineConfig) -> Self {
        let source = nonbonded.get_potentials();
        let n_types = source.shape()[0];
        let mut potentials = Vec::with_capacity(n_types * n_types);
        for i in 0..n_types {
            for j in 0..n_types {
                potentials.push(SplinedPotential::with_cutoff(
                    &source[(i, j)],
                    cutoff,
                    config.clone(),
                ));
            }
        }
        debug_assert_eq!(potentials.len(), n_types * n_types);
        Self {
            potentials,
            n_types,
        }
    }

    /// Number of atom types (matrix side length).
    pub fn n_types(&self) -> usize {
        self.n_types
    }

    /// Get the splined potential for atom type pair `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> &SplinedPotential {
        &self.potentials[i * self.n_types + j]
    }

    /// Iterate over all splined potentials in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &SplinedPotential> {
        self.potentials.iter()
    }
}

/// Pair-matrix of twobody energies combining SR potentials and analytical Coulomb.
///
/// Single entry point for all energy evaluation in Duello. Internally stores
/// SR potentials from the faunus `NonbondedMatrix` and precomputed Coulomb
/// prefactors. Callers don't need to know whether energy comes from a spline,
/// exact potential, or analytical formula.
pub struct PairMatrix {
    nonbonded: NonbondedMatrix,
    coulomb_params: CoulombParams,
}

impl PairMatrix {
    /// Build from SR potentials + medium (determines Coulomb parameters).
    ///
    /// If ion-induced dipole interactions are detected, Coulomb is set to zero
    /// with a warning (polarization not supported in the split-path design).
    pub fn new(nonbonded: NonbondedMatrix, atomkinds: &[AtomKind], medium: &Medium) -> Self {
        // κ = 1/λ_D where λ_D is the Debye screening length from the electrolyte
        let kappa = medium.debye_length().map(|dl| 1.0 / dl);
        let coulomb_params = extract_coulomb_params(atomkinds, medium.permittivity().into(), kappa)
            .unwrap_or_else(|| {
                log::warn!("Ion-induced dipole detected; Coulomb contribution will be zero");
                CoulombParams::zero(atomkinds.len())
            });
        Self {
            nonbonded,
            coulomb_params,
        }
    }

    /// Sum energy between two structures (kJ/mol): SR + analytical Coulomb.
    pub fn sum_energy(&self, a: &Structure, b: &Structure) -> f64 {
        let sr = compute_pairwise_energy(a, b, self.nonbonded.get_potentials());
        let coulomb = pairwise_analytical_coulomb(a, b, &self.coulomb_params);
        let energy = sr + coulomb;
        trace!("molecule-molecule energy: {energy:.2} kJ/mol");
        energy
    }

    /// Sum energy between all atoms in a structure and a single test atom (kJ/mol).
    pub fn energy_with_atom(&self, structure: &Structure, atom_id: usize, pos: &Vector3) -> f64 {
        let sr = single_atom_energy(structure, atom_id, pos, self.nonbonded.get_potentials());
        let coulomb = single_atom_analytical_coulomb(structure, atom_id, pos, &self.coulomb_params);
        sr + coulomb
    }

    /// Create splined SR potentials for GPU/SIMD backends.
    pub fn create_sr_splines(&self, cutoff: f64, config: SplineConfig) -> SplinedPotentials {
        log::info!("Creating SR-only spline with cutoff {cutoff:.1} Å");
        SplinedPotentials::from_nonbonded(&self.nonbonded, cutoff, config)
    }

    /// Coulomb parameters for analytical evaluation in backends.
    pub const fn coulomb_params(&self) -> &CoulombParams {
        &self.coulomb_params
    }
}

/// Compute pairwise energy between two structures using a potential matrix.
fn compute_pairwise_energy<P, T>(a: &Structure, b: &Structure, potentials: &P) -> f64
where
    P: std::ops::Index<(usize, usize), Output = T>,
    T: IsotropicTwobodyEnergy,
{
    itertools::iproduct!(a.pos.iter().zip(&a.atom_ids), b.pos.iter().zip(&b.atom_ids))
        .map(|((pa, &ida), (pb, &idb))| {
            potentials[(ida, idb)].isotropic_twobody_energy((pa - pb).norm_squared())
        })
        .sum()
}

/// Screened Coulomb (Yukawa) energy for a single pair.
/// Skips uncharged pairs to avoid the exp() call and division by zero at r=0.
#[inline]
fn yukawa_energy(prefactor: f32, kappa: f64, r: f64) -> f64 {
    if prefactor.abs() > 1e-12 {
        f64::from(prefactor) * (-kappa * r).exp() / r
    } else {
        0.0
    }
}

/// Analytical Coulomb/Yukawa energy between two structures.
fn pairwise_analytical_coulomb(a: &Structure, b: &Structure, params: &CoulombParams) -> f64 {
    let n = params.n_types;
    let kappa = f64::from(params.kappa);
    itertools::iproduct!(a.pos.iter().zip(&a.atom_ids), b.pos.iter().zip(&b.atom_ids))
        .map(|((pa, &ida), (pb, &idb))| {
            yukawa_energy(params.prefactors[ida * n + idb], kappa, (pa - pb).norm())
        })
        .sum()
}

/// Compute energy between all atoms in a structure and a single test atom.
fn single_atom_energy<P, T>(
    structure: &Structure,
    atom_id: usize,
    pos: &Vector3,
    potentials: &P,
) -> f64
where
    P: std::ops::Index<(usize, usize), Output = T>,
    T: IsotropicTwobodyEnergy,
{
    structure
        .pos
        .iter()
        .zip(structure.atom_ids.iter())
        .map(|(atom_pos, &aid)| {
            let dist_sq = (atom_pos - pos).norm_squared();
            potentials[(aid, atom_id)].isotropic_twobody_energy(dist_sq)
        })
        .sum()
}

/// Analytical Coulomb energy between all atoms in a structure and a single test atom.
fn single_atom_analytical_coulomb(
    structure: &Structure,
    atom_id: usize,
    pos: &Vector3,
    params: &CoulombParams,
) -> f64 {
    let n = params.n_types;
    let kappa = f64::from(params.kappa);
    structure
        .pos
        .iter()
        .zip(structure.atom_ids.iter())
        .map(|(atom_pos, &aid)| {
            let idx = aid * n + atom_id;
            let r = (atom_pos - pos).norm();
            yukawa_energy(params.prefactors[idx], kappa, r)
        })
        .sum()
}

/// Calculate accumulated electric potential at point `r` due to charges in `structure`
pub fn electric_potential(structure: &Structure, r: &Vector3, multipole: &Plain) -> f64 {
    std::iter::zip(structure.pos.iter(), structure.charges.iter())
        .map(|(pos, charge)| multipole.ion_potential(*charge, (pos - r).norm()))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_coulomb_params_zero() {
        let params = CoulombParams::zero(3);
        assert_eq!(params.prefactors.len(), 9);
        assert!(params.prefactors.iter().all(|&p| p == 0.0));
        assert_eq!(params.kappa, 0.0);
        assert_eq!(params.n_types, 3);
    }

    /// Helper: single-atom structure at `pos` with given atom type id.
    fn atom(pos: Vector3, atom_id: usize) -> Structure {
        Structure {
            pos: vec![pos],
            masses: vec![1.0],
            charges: vec![0.0],
            radii: vec![1.0],
            atom_ids: vec![atom_id],
        }
    }

    #[test]
    fn test_analytical_coulomb_symmetry() {
        let a = atom(Vector3::new(0.0, 0.0, 0.0), 0);
        let b = atom(Vector3::new(5.0, 0.0, 0.0), 0);
        let params = CoulombParams {
            prefactors: vec![10.0],
            kappa: 0.0,
            n_types: 1,
        };
        let e_ab = pairwise_analytical_coulomb(&a, &b, &params);
        let e_ba = pairwise_analytical_coulomb(&b, &a, &params);
        assert_abs_diff_eq!(e_ab, e_ba, epsilon = 1e-12);
        // At r=5, unscreened: prefactor/r = 10/5 = 2
        assert_abs_diff_eq!(e_ab, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_analytical_coulomb_screening() {
        let a = atom(Vector3::zeros(), 0);
        let b = atom(Vector3::new(10.0, 0.0, 0.0), 0);
        let params = CoulombParams {
            prefactors: vec![7.0],
            kappa: 0.1,
            n_types: 1,
        };
        let e = pairwise_analytical_coulomb(&a, &b, &params);
        let expected = 7.0 * (-1.0_f64).exp() / 10.0;
        assert_abs_diff_eq!(e, expected, epsilon = 1e-6);
    }
}

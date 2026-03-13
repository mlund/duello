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
use faunus::topology::CustomProperty;
use faunus::{
    energy::{NonbondedMatrix, NonbondedMatrixSplined},
    topology::AtomKind,
};
use faunus::interatomic::coulomb::pairwise::{MultipoleEnergy, MultipolePotential, Plain};
use faunus::interatomic::coulomb::permittivity::ConstantPermittivity;
use faunus::interatomic::twobody::{IonIon, IonIonPolar, IsotropicTwobodyEnergy, SplineConfig};
use std::{cmp::PartialEq, fmt::Debug};

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
    pub prefactors: Vec<f32>,
    /// Inverse Debye screening length κ (1/Å). 0.0 for unscreened Coulomb.
    pub kappa: f32,
}

impl CoulombParams {
    /// Create zero-valued Coulomb params (no analytical Coulomb contribution).
    pub fn zero(n_types: usize) -> Self {
        Self {
            prefactors: vec![0.0; n_types * n_types],
            kappa: 0.0,
        }
    }
}

/// Extract Coulomb prefactors for analytical evaluation.
///
/// Returns `None` if any pair requires ion-induced dipole interaction (IonIonPolar),
/// whose energy formula is too complex for the simple `prefactor * exp(-κr) / r`
/// analytical path. In that case the caller falls back to the combined spline.
pub fn extract_coulomb_params(
    atomkinds: &[AtomKind],
    permittivity: ConstantPermittivity,
    kappa: Option<f64>,
) -> Option<CoulombParams> {
    let n_types = atomkinds.len();
    let eps_r = f64::from(permittivity);
    let to_kjmol = faunus::interatomic::ELECTRIC_PREFACTOR / eps_r;

    // Pre-compute per-type alpha and charge to avoid repeated property lookups
    let alphas: Vec<f64> = atomkinds.iter().map(get_alpha).collect();
    let charges: Vec<f64> = atomkinds.iter().map(AtomKind::charge).collect();

    let mut prefactors = vec![0.0f32; n_types * n_types];
    for i in 0..n_types {
        for j in 0..n_types {
            if needs_polarization(alphas[i], charges[i], alphas[j], charges[j]) {
                log::warn!(
                    "Ion-induced dipole detected for pair ({i}, {j}); \
                     analytical Coulomb not supported, using combined spline"
                );
                return None;
            }
            prefactors[i * n_types + j] = (to_kjmol * charges[i] * charges[j]) as f32;
        }
    }

    Some(CoulombParams {
        prefactors,
        kappa: kappa.unwrap_or(0.0) as f32,
    })
}

/// Wrapper around splined pair potentials, hiding the upstream faunus type.
///
/// Created via `PairMatrix::create_splined_potentials` and consumed by backend constructors.
pub struct SplinedPotentials(NonbondedMatrixSplined);

impl SplinedPotentials {
    /// Access the inner splined matrix (for backend initialization).
    pub(crate) const fn inner(&self) -> &NonbondedMatrixSplined {
        &self.0
    }
}

/// Storage for either standard or splined pair potentials
enum PotentialStorage {
    Standard(NonbondedMatrix),
    Splined(NonbondedMatrixSplined),
}

/// Pair-matrix of twobody energies for pairs of atom ids
pub struct PairMatrix {
    storage: PotentialStorage,
}

impl PairMatrix {
    /// Add Coulomb potential to a nonbonded matrix
    fn add_coulomb_to_matrix<
        T: MultipoleEnergy + Clone + Send + Sync + Debug + PartialEq + 'static,
    >(
        mut nonbonded: NonbondedMatrix,
        atomkinds: &[AtomKind],
        permittivity: ConstantPermittivity,
        coulomb_method: &T,
    ) -> NonbondedMatrix {
        log::info!("Adding Coulomb potential to nonbonded matrix");
        nonbonded
            .get_potentials_mut()
            .indexed_iter_mut()
            .for_each(|((i, j), pairpot)| {
                let alpha1 = get_alpha(&atomkinds[i]);
                let alpha2 = get_alpha(&atomkinds[j]);
                let charge1 = atomkinds[i].charge();
                let charge2 = atomkinds[j].charge();
                let charge_product = charge1 * charge2;
                let use_polarization =
                    needs_polarization(alpha1, charge1, alpha2, charge2);

                if use_polarization {
                    log::debug!(
                        "Adding ion-induced dipole term for atom pair ({i}, {j}). Alphas: {alpha1}, {alpha2}"
                    );
                }
                let coulomb =
                    IonIon::<T>::new(charge_product, permittivity, coulomb_method.clone());
                let coulomb_polar = Box::new(IonIonPolar::<T>::new(
                    coulomb.clone(),
                    (charge1, charge2),
                    (alpha1, alpha2),
                )) as Box<dyn IsotropicTwobodyEnergy>;
                let combined = match use_polarization {
                    true => coulomb_polar + Box::new(pairpot.clone()),
                    false => {
                        Box::new(coulomb) as Box<dyn IsotropicTwobodyEnergy>
                            + Box::new(pairpot.clone())
                    }
                };
                *pairpot = faunus::interatomic::twobody::ArcPotential(std::sync::Arc::new(combined));
            });
        nonbonded
    }

    /// Create a new pair matrix with added Coulomb potential
    pub fn new_with_coulomb<
        T: MultipoleEnergy + Clone + Send + Sync + Debug + PartialEq + 'static,
    >(
        nonbonded: NonbondedMatrix,
        atomkinds: &[AtomKind],
        permittivity: ConstantPermittivity,
        coulomb_method: &T,
    ) -> Self {
        let nonbonded =
            Self::add_coulomb_to_matrix(nonbonded, atomkinds, permittivity, coulomb_method);
        Self {
            storage: PotentialStorage::Standard(nonbonded),
        }
    }

    /// Create a new pair matrix with Coulomb potential and splined interpolation
    pub fn new_with_coulomb_splined<
        T: MultipoleEnergy + Clone + Send + Sync + Debug + PartialEq + 'static,
    >(
        nonbonded: NonbondedMatrix,
        atomkinds: &[AtomKind],
        permittivity: ConstantPermittivity,
        coulomb_method: &T,
        cutoff: f64,
        spline_config: SplineConfig,
    ) -> Self {
        let splined = Self::create_splined_potentials(
            nonbonded,
            atomkinds,
            permittivity,
            coulomb_method,
            cutoff,
            spline_config,
        );
        Self::from_splined(splined)
    }

    /// Create splined potentials with Coulomb potential added.
    ///
    /// Returns a `SplinedPotentials` wrapper that can be shared between backends
    /// or converted into a `PairMatrix` via `from_splined`.
    pub fn create_splined_potentials<
        T: MultipoleEnergy + Clone + Send + Sync + Debug + PartialEq + 'static,
    >(
        nonbonded: NonbondedMatrix,
        atomkinds: &[AtomKind],
        permittivity: ConstantPermittivity,
        coulomb_method: &T,
        cutoff: f64,
        spline_config: SplineConfig,
    ) -> SplinedPotentials {
        let nonbonded =
            Self::add_coulomb_to_matrix(nonbonded, atomkinds, permittivity, coulomb_method);
        SplinedPotentials(NonbondedMatrixSplined::from_nonbonded(
            &nonbonded,
            cutoff,
            Some(spline_config),
        ))
    }

    /// Create SR-only splined potentials (without Coulomb).
    ///
    /// The SR potential (e.g. Ashbaugh-Hatch) has a finite cutoff that creates a
    /// discontinuity when combined with Coulomb inside a single spline. Splining
    /// SR alone avoids this; Coulomb is added analytically by GPU/SIMD backends.
    pub fn create_sr_splined_potentials(
        nonbonded: NonbondedMatrix,
        cutoff: f64,
        spline_config: SplineConfig,
    ) -> SplinedPotentials {
        log::info!("Creating SR-only spline (no Coulomb) with cutoff {cutoff:.1} Å");
        SplinedPotentials(NonbondedMatrixSplined::from_nonbonded(
            &nonbonded,
            cutoff,
            Some(spline_config),
        ))
    }

    /// Create a pair matrix from splined potentials.
    pub fn from_splined(splined: SplinedPotentials) -> Self {
        Self {
            storage: PotentialStorage::Splined(splined.0),
        }
    }

    /// Sum energy between two set of atomic structures (kJ/mol)
    pub fn sum_energy(&self, a: &Structure, b: &Structure) -> f64 {
        let energy = match &self.storage {
            PotentialStorage::Standard(nb) => compute_pairwise_energy(a, b, nb.get_potentials()),
            PotentialStorage::Splined(sp) => compute_pairwise_energy(a, b, sp.get_potentials()),
        };
        trace!("molecule-molecule energy: {energy:.2} kJ/mol");
        energy
    }

    /// Sum energy between all atoms in a structure and a single test atom (kJ/mol)
    pub fn energy_with_atom(&self, structure: &Structure, atom_id: usize, pos: &Vector3) -> f64 {
        match &self.storage {
            PotentialStorage::Standard(nb) => {
                single_atom_energy(structure, atom_id, pos, nb.get_potentials())
            }
            PotentialStorage::Splined(sp) => {
                single_atom_energy(structure, atom_id, pos, sp.get_potentials())
            }
        }
    }
}

/// Compute pairwise energy between two structures using a potential matrix
fn compute_pairwise_energy<P, T>(a: &Structure, b: &Structure, potentials: &P) -> f64
where
    P: std::ops::Index<(usize, usize), Output = T>,
    T: IsotropicTwobodyEnergy,
{
    let mut energy = 0.0;
    for i in 0..a.pos.len() {
        for j in 0..b.pos.len() {
            let distance_sq = (a.pos[i] - b.pos[j]).norm_squared();
            let atom_a = a.atom_ids[i];
            let atom_b = b.atom_ids[j];
            energy += potentials[(atom_a, atom_b)].isotropic_twobody_energy(distance_sq);
        }
    }
    energy
}

/// Compute energy between all atoms in a structure and a single test atom
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

/// Calculate accumulated electric potential at point `r` due to charges in `structure`
pub fn electric_potential(structure: &Structure, r: &Vector3, multipole: &Plain) -> f64 {
    std::iter::zip(structure.pos.iter(), structure.charges.iter())
        .map(|(pos, charge)| multipole.ion_potential(*charge, (pos - r).norm()))
        .sum()
}

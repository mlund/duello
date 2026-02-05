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
use coulomb::pairwise::{MultipoleEnergy, MultipolePotential, Plain};
use coulomb::permittivity::ConstantPermittivity;
use faunus::topology::CustomProperty;
use faunus::{
    energy::{NonbondedMatrix, NonbondedMatrixSplined},
    topology::AtomKind,
};
use interatomic::{
    twobody::{GridType, IonIon, IonIonPolar, IsotropicTwobodyEnergy, SplineConfig},
    Vector3,
};
use std::{cmp::PartialEq, fmt::Debug};

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
                // Fetch excess polarizability for the two atom kinds from the
                // custom "alpha" field, if it exists. Add to topology atoms like this:
                // `custom: {alpha: 50.0}`
                let get_alpha = |atom: &AtomKind| {
                    atom.get_property("alpha").map_or(0.0, |v| {
                        f64::try_from(v).expect("Failed to convert alpha to f64")
                    })
                };
                let alpha1 = get_alpha(&atomkinds[i]);
                let alpha2 = get_alpha(&atomkinds[j]);
                let charge1 = atomkinds[i].charge();
                let charge2 = atomkinds[j].charge();
                let charge_product = charge1 * charge2;
                let use_polarization =
                    (alpha1 * charge2).abs() > 1e-6 || (alpha2 * charge1).abs() > 1e-6;

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
                *pairpot = interatomic::twobody::ArcPotential(std::sync::Arc::new(combined));
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
        n_points: Option<usize>,
        grid_type: GridType,
        shift_energy: bool,
    ) -> Self {
        let splined = Self::create_splined_matrix(
            nonbonded,
            atomkinds,
            permittivity,
            coulomb_method,
            cutoff,
            n_points,
            grid_type,
            shift_energy,
        );
        Self::from_splined(splined)
    }

    /// Create a splined matrix with Coulomb potential added.
    ///
    /// This is useful when the splined matrix needs to be shared between backends.
    pub fn create_splined_matrix<
        T: MultipoleEnergy + Clone + Send + Sync + Debug + PartialEq + 'static,
    >(
        nonbonded: NonbondedMatrix,
        atomkinds: &[AtomKind],
        permittivity: ConstantPermittivity,
        coulomb_method: &T,
        cutoff: f64,
        n_points: Option<usize>,
        grid_type: GridType,
        shift_energy: bool,
    ) -> NonbondedMatrixSplined {
        let nonbonded =
            Self::add_coulomb_to_matrix(nonbonded, atomkinds, permittivity, coulomb_method);
        let config = Some(SplineConfig {
            n_points: n_points.unwrap_or(2000),
            shift_energy,
            grid_type,
            ..Default::default()
        });
        NonbondedMatrixSplined::new(&nonbonded, cutoff, config)
    }

    /// Create a pair matrix from an existing splined matrix.
    pub const fn from_splined(splined: NonbondedMatrixSplined) -> Self {
        Self {
            storage: PotentialStorage::Splined(splined),
        }
    }

    /// Sum energy between two set of atomic structures (kJ/mol)
    pub fn sum_energy(&self, a: &Structure, b: &Structure) -> f64 {
        let energy = match &self.storage {
            PotentialStorage::Standard(nonbonded) => {
                compute_pairwise_energy(a, b, nonbonded.get_potentials())
            }
            PotentialStorage::Splined(splined) => {
                compute_pairwise_energy(a, b, splined.get_potentials())
            }
        };
        trace!("molecule-molecule energy: {energy:.2} kJ/mol");
        energy
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

/// Calculate accumulated electric potential at point `r` due to charges in `structure`
pub fn electric_potential(structure: &Structure, r: &Vector3, multipole: &Plain) -> f64 {
    std::iter::zip(structure.pos.iter(), structure.charges.iter())
        .map(|(pos, charge)| multipole.ion_potential(*charge, (pos - r).norm()))
        .sum()
}

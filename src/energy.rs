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
use faunus::{energy::NonbondedMatrix, topology::AtomKind};
use interatomic::{
    twobody::{IonIon, IonIonPolar, IsotropicTwobodyEnergy},
    Vector3,
};
use std::{cmp::PartialEq, fmt::Debug};

/// Pair-matrix of twobody energies for pairs of atom ids
pub struct PairMatrix {
    nonbonded: NonbondedMatrix,
}

impl PairMatrix {
    /// Create a new pair matrix with added Coulomb potential
    pub fn new_with_coulomb<
        T: MultipoleEnergy + Clone + Send + Sync + Debug + PartialEq + 'static,
    >(
        mut nonbonded: NonbondedMatrix,
        atomkinds: &[AtomKind],
        permittivity: ConstantPermittivity,
        coulomb_method: &T,
    ) -> Self {
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
                        "Adding ion-induced dipole term for atom pair ({}, {}). Alphas: {}, {}",
                        i,
                        j,
                        alpha1,
                        alpha2
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
                *pairpot = std::sync::Arc::new(combined);
            });
        Self { nonbonded }
    }

    // Sum energy between two set of atomic structures (kJ/mol)
    pub fn sum_energy(&self, a: &Structure, b: &Structure) -> f64 {
        let potentials = self.nonbonded.get_potentials();
        let mut energy = 0.0;
        for i in 0..a.pos.len() {
            for j in 0..b.pos.len() {
                let distance_sq = (a.pos[i] - b.pos[j]).norm_squared();
                let a = a.atom_ids[i];
                let b = b.atom_ids[j];
                energy += potentials[(a, b)].isotropic_twobody_energy(distance_sq);
            }
        }
        trace!("molecule-molecule energy: {:.2} kJ/mol", energy);
        energy
    }
}

/// Calculate accumulated electric potential at point `r` due to charges in `structure`
pub fn electric_potential(structure: &Structure, r: &Vector3, multipole: &Plain) -> f64 {
    std::iter::zip(structure.pos.iter(), structure.charges.iter())
        .map(|(pos, charge)| multipole.ion_potential(*charge, (pos - r).norm()))
        .sum()
}

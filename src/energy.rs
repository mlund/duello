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
use faunus::interatomic::Cutoff;
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
#[derive(Clone)]
pub struct CoulombParams {
    /// Per-pair prefactors in kJ/mol·Å, indexed as `[id_a * n_types + id_b]`.
    /// Each entry = `ELECTRIC_PREFACTOR * z_i * z_j / ε_r`.
    /// Zero prefactor means no electrostatic interaction for that pair.
    /// Stored as f32 because GPU/SIMD backends consume them directly.
    pub prefactors: Vec<f32>,
    /// Inverse Debye screening length κ (1/Å). 0.0 for unscreened Coulomb.
    pub kappa: f32,
    /// Pairs at or beyond this distance (Å) don't interact; ∞ = untruncated.
    pub cutoff: f32,
    /// e^{−κ·cutoff}/cutoff, subtracted so each pair's energy reaches 0 at the cutoff.
    pub shift: f32,
    /// Cached to avoid `isqrt(prefactors.len())` on every pair lookup.
    n_types: usize,
}

impl CoulombParams {
    /// Create zero-valued Coulomb params (no analytical Coulomb contribution).
    pub fn zero(n_types: usize) -> Self {
        Self {
            prefactors: vec![0.0; n_types * n_types],
            kappa: 0.0,
            cutoff: f32::INFINITY,
            shift: 0.0,
            n_types,
        }
    }

    /// Truncated at `cutoff` (Å) and shifted to vanish there. Set together here
    /// so `shift` can't go stale.
    fn with_cutoff(self, cutoff: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(
            cutoff.is_finite() && cutoff > 0.0,
            "Coulomb cutoff must be finite and positive, got {cutoff}"
        );
        // From the stored f32 cutoff, so the shift cancels where backends truncate.
        let cutoff = f64::from(cutoff as f32);
        let kappa = f64::from(self.kappa);
        Ok(Self {
            cutoff: cutoff as f32,
            shift: ((-kappa * cutoff).exp() / cutoff) as f32,
            ..self
        })
    }

    /// Squared cutoff for f32 kernels; f32::MAX rather than ∞ when untruncated,
    /// as GPU fast-math needn't honour ∞ in comparisons.
    pub fn cutoff_sq(&self) -> f32 {
        (self.cutoff * self.cutoff).min(f32::MAX)
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
        ..CoulombParams::zero(n_types)
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
    fn from_nonbonded(nonbonded: &NonbondedMatrix, config: SplineConfig) -> anyhow::Result<Self> {
        let source = nonbonded.get_potentials();
        let n_types = source.order();
        let mut potentials = Vec::with_capacity(n_types * n_types);
        for i in 0..n_types {
            for j in 0..n_types {
                potentials.push(SplinedPotential::new(&source[(i, j)], config.clone())?);
            }
        }
        debug_assert_eq!(potentials.len(), n_types * n_types);
        Ok(Self {
            potentials,
            n_types,
        })
    }

    /// Number of atom types (matrix side length).
    pub fn n_types(&self) -> usize {
        self.n_types
    }

    /// Get the splined potential for atom type pair `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> &SplinedPotential {
        &self.potentials[i * self.n_types + j]
    }

    /// Distance beyond which the `(i, j)` energy is negligible (< 1 µJ/mol).
    ///
    /// A pair can vanish well inside its nominal range (e.g. Ashbaugh–Hatch with
    /// λ ≈ 0), so callers that size or cull by range should use this rather
    /// than the pair's range or the table's extent.
    pub fn vanishes_beyond(&self, i: usize, j: usize) -> f64 {
        const STEP: f64 = 0.05;
        const NEGLIGIBLE: f64 = 1e-6; // kJ/mol
        let spline = self.get(i, j);
        // Where the pair acts: its range, which a shorter table truncates. The
        // table's extent alone overstates an absent pair.
        let range = spline.cutoff().min(spline.r_max());
        let mut r = range;
        while r > STEP && spline.isotropic_twobody_energy(r * r).abs() < NEGLIGIBLE {
            r -= STEP;
        }
        (r + STEP).min(range)
    }

    /// The `(i, j)` well's bottom (r, U) in (Å, kJ/mol), sampled over the
    /// spline's own range; None for a purely repulsive pair (U ≥ 0).
    pub fn minimum(&self, i: usize, j: usize) -> Option<(f64, f64)> {
        const STEP: f64 = 0.005;
        let spline = self.get(i, j);
        let end = self.vanishes_beyond(i, j);
        (0..)
            .map(|k| spline.r_min() + k as f64 * STEP)
            .take_while(|&r| r < end)
            .map(|r| (r, spline.isotropic_twobody_energy(r * r)))
            .filter(|&(_, u)| u < 0.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
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
    /// Fails unless every pair is short-ranged only: duello adds electrostatics
    /// from the medium itself, and each pair's own range sizes its spline.
    /// If ion-induced dipole interactions are detected, Coulomb is set to zero
    /// with a warning (polarization not supported in the split-path design).
    pub fn new(
        nonbonded: NonbondedMatrix,
        atomkinds: &[AtomKind],
        medium: &Medium,
    ) -> anyhow::Result<Self> {
        validate_pairs(&nonbonded, atomkinds)?;
        // κ = 1/λ_D where λ_D is the Debye screening length from the electrolyte
        let kappa = medium.debye_length().map(|dl| 1.0 / dl);
        let coulomb_params = extract_coulomb_params(atomkinds, medium.permittivity().into(), kappa)
            .unwrap_or_else(|| {
                log::warn!("Ion-induced dipole detected; Coulomb contribution will be zero");
                CoulombParams::zero(atomkinds.len())
            });
        Ok(Self {
            nonbonded,
            coulomb_params,
        })
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
    ///
    /// Each pair is tabulated over its own range, unshifted, so splines and the
    /// exact reference evaluate the same model.
    pub fn create_sr_splines(&self, config: SplineConfig) -> anyhow::Result<SplinedPotentials> {
        let config = SplineConfig {
            shift_energy: false,
            ..config
        };
        SplinedPotentials::from_nonbonded(&self.nonbonded, config)
    }

    /// Truncate Yukawa at `cutoff` (Å), shifted to vanish there, as a cutoff MC
    /// reference does. None keeps it untruncated.
    pub fn with_coulomb_cutoff(mut self, cutoff: Option<f64>) -> anyhow::Result<Self> {
        if let Some(cutoff) = cutoff {
            self.coulomb_params = self.coulomb_params.with_cutoff(cutoff)?;
        }
        Ok(self)
    }

    /// Mass-centre separation (Å) at and beyond which `a` and `b` don't interact:
    /// the longest pair range plus both radii. ∞ while Yukawa is untruncated.
    pub fn reach(&self, a: &Structure, b: &Structure) -> f64 {
        let pairs = self.nonbonded.get_potentials();
        let short_range = itertools::iproduct!(a.unique_atom_types(), b.unique_atom_types())
            .map(|(i, j)| pairs[(i, j)].cutoff())
            .fold(0.0, f64::max);
        let range = short_range.max(f64::from(self.coulomb_params.cutoff));
        range + a.radius() + b.radius()
    }

    /// Coulomb parameters for analytical evaluation in backends.
    pub const fn coulomb_params(&self) -> &CoulombParams {
        &self.coulomb_params
    }
}

/// Every pair must be Coulomb-free with a finite range; the error names the pair.
fn validate_pairs(nonbonded: &NonbondedMatrix, atomkinds: &[AtomKind]) -> anyhow::Result<()> {
    let pairs = nonbonded.get_potentials();
    for (i, j) in itertools::iproduct!(0..pairs.order(), 0..pairs.order()) {
        let names = (atomkinds[i].name(), atomkinds[j].name());
        anyhow::ensure!(
            !pairs[(i, j)].has_coulomb(),
            "pair {names:?}: the topology has a Coulomb term; remove it, \
             duello adds electrostatics from the medium"
        );
        anyhow::ensure!(
            pairs[(i, j)].cutoff().is_finite(),
            "pair {names:?} has unbounded range; give it a cutoff, e.g. Lennard-Jones \
             cut and shifted as !AshbaughHatch {{mixing: …, cutoff: rc}} with !Lambda 1.0"
        );
    }
    Ok(())
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
fn yukawa_energy(prefactor: f32, params: &CoulombParams, r_sq: f64) -> f64 {
    if prefactor.abs() <= 1e-12 || r_sq >= f64::from(params.cutoff).powi(2) {
        return 0.0;
    }
    let (r, kappa) = (r_sq.sqrt(), f64::from(params.kappa));
    f64::from(prefactor) * ((-kappa * r).exp() / r - f64::from(params.shift))
}

/// Analytical Coulomb/Yukawa energy between two structures.
fn pairwise_analytical_coulomb(a: &Structure, b: &Structure, params: &CoulombParams) -> f64 {
    let n = params.n_types;
    itertools::iproduct!(a.pos.iter().zip(&a.atom_ids), b.pos.iter().zip(&b.atom_ids))
        .map(|((pa, &ida), (pb, &idb))| {
            yukawa_energy(
                params.prefactors[ida * n + idb],
                params,
                (pa - pb).norm_squared(),
            )
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
    structure
        .pos
        .iter()
        .zip(structure.atom_ids.iter())
        .map(|(atom_pos, &aid)| {
            let idx = aid * n + atom_id;
            yukawa_energy(
                params.prefactors[idx],
                params,
                (atom_pos - pos).norm_squared(),
            )
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
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    /// Two bead kinds, A (σ = 4) and B (σ = 6), whose every pair is `terms`.
    fn pair_matrix(terms: &str) -> anyhow::Result<PairMatrix> {
        let yaml = format!(
            "atoms:\n  \
             - {{name: A, mass: 1.0, charge: 1.0, σ: 4.0, ε: 1.0, hydrophobicity: !Lambda 0.5}}\n  \
             - {{name: B, mass: 1.0, charge: -1.0, σ: 6.0, ε: 1.0, hydrophobicity: !Lambda 0.5}}\n\
             system:\n  energy:\n    - !Nonbonded\n      default: [{terms}]\n"
        );
        let mut topology = faunus::topology::Topology::from_str_partial(&yaml)?;
        topology.finalize_atoms()?;
        let medium = Medium::salt_water(
            298.15,
            faunus::interatomic::coulomb::Salt::SodiumChloride,
            0.02,
        );
        let nonbonded = NonbondedMatrix::from_str(&yaml, &topology, Some(medium.clone()))?;
        PairMatrix::new(nonbonded, topology.atomkinds(), &medium)
    }

    fn error_of(terms: &str) -> String {
        format!(
            "{:#}",
            pair_matrix(terms).err().expect("topology is refused")
        )
    }

    /// An unbounded pair can't size its spline; the error names it and the bounded form.
    #[test]
    fn unbounded_pair_is_refused() {
        let err = error_of("!LennardJones {mixing: LB}");
        assert!(
            err.contains("\"A\"") && err.contains("!AshbaughHatch") && err.contains("!Lambda 1"),
            "{err}"
        );
        assert!(pair_matrix("!AshbaughHatch {mixing: arithmetic, cutoff: 20.0}").is_ok());
    }

    /// Each pair's spline spans exactly its own range: exact inside, zero beyond.
    #[test]
    fn spline_range_is_the_pair_range() {
        let matrix = pair_matrix("!AshbaughHatch {mixing: arithmetic, cutoff: 12.0}").unwrap();
        let splines = matrix.create_sr_splines(SplineConfig::default()).unwrap();
        let (spline, exact) = (
            splines.get(0, 1),
            &matrix.nonbonded.get_potentials()[(0, 1)],
        );
        assert_eq!(spline.r_max(), 12.0);
        let u = |p: &dyn IsotropicTwobodyEnergy, r: f64| p.isotropic_twobody_energy(r * r);
        assert_abs_diff_eq!(u(spline, 11.9), u(exact, 11.9), epsilon = 1e-6);
        assert_eq!(u(spline, 12.1), 0.0);
    }

    /// A pair without any term is exactly zero at every distance.
    #[test]
    fn absent_pair_splines_to_zero() {
        let splines = pair_matrix("")
            .unwrap()
            .create_sr_splines(SplineConfig::default())
            .unwrap();
        for r in [1.0, 5.0, 50.0] {
            assert_eq!(
                splines.get(0, 1).isotropic_twobody_energy(r * r),
                0.0,
                "r = {r}"
            );
        }
        assert_eq!(splines.get(0, 1).cutoff(), 0.0);
        assert_eq!(splines.vanishes_beyond(0, 1), 0.0);
    }

    /// Electrostatics belong to duello, not the topology: a Coulomb term is refused by name.
    #[test]
    fn coulomb_in_topology_is_refused() {
        let err =
            error_of("!AshbaughHatch {mixing: arithmetic, cutoff: 20.0}, !Coulomb {cutoff: 50.0}");
        assert!(err.contains("A") && err.contains("Coulomb"), "{err}");
    }

    /// A(e^{−κr}/r − e^{−κr_el}/r_el) inside r_el, continuous to 0 there, 0 beyond.
    #[test]
    fn yukawa_is_shifted_and_truncated_at_the_coulomb_cutoff() {
        let matrix = pair_matrix("")
            .unwrap()
            .with_coulomb_cutoff(Some(15.0))
            .unwrap();
        let (a, b) = (
            matrix.coulomb_params().prefactors[1],
            matrix.coulomb_params().kappa,
        );
        let (a, kappa) = (f64::from(a), f64::from(b));
        let u = |r: f64| matrix.sum_energy(&atom(Vector3::zeros(), 0), &atom(Vector3::x() * r, 1));
        let yukawa = |r: f64| (-kappa * r).exp() / r;
        assert_relative_eq!(
            u(5.0),
            a * (yukawa(5.0) - yukawa(15.0)),
            max_relative = 1e-6
        );
        assert_abs_diff_eq!(u(15.0 - 1e-6), 0.0, epsilon = 1e-6);
        assert_eq!(u(15.0), 0.0);
        assert_eq!(u(20.0), 0.0);
    }

    /// Charged bead 10 Å from the mass centre, balanced by a neutral one opposite.
    fn off_center(atom_id: usize) -> Structure {
        Structure {
            pos: vec![Vector3::x() * 10.0, -Vector3::x() * 10.0],
            masses: vec![1.0, 1.0],
            charges: vec![0.0; 2],
            radii: vec![1.0; 2],
            atom_ids: vec![atom_id, atom_id],
        }
    }

    /// Beyond R* = max(r_SR, r_el) + R_a + R_b no bead pair interacts.
    #[test]
    fn reach_adds_both_radii_to_the_longest_range() {
        let (a, b) = (off_center(0), off_center(1));
        let untruncated = pair_matrix("").unwrap();
        assert_eq!(untruncated.reach(&a, &b), f64::INFINITY);
        let coulomb_only = untruncated.with_coulomb_cutoff(Some(15.0)).unwrap();
        assert_eq!(coulomb_only.reach(&a, &b), 35.0);
        let sr = pair_matrix("!AshbaughHatch {mixing: arithmetic, cutoff: 20.0}").unwrap();
        assert_eq!(
            sr.with_coulomb_cutoff(Some(15.0)).unwrap().reach(&a, &b),
            40.0
        );
    }

    #[test]
    fn coulomb_cutoff_must_be_finite_and_positive() {
        for bad in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let result = pair_matrix("").unwrap().with_coulomb_cutoff(Some(bad));
            assert!(result.is_err(), "{bad} accepted");
        }
    }

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
            ..CoulombParams::zero(1)
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
            ..CoulombParams::zero(1)
        };
        let e = pairwise_analytical_coulomb(&a, &b, &params);
        let expected = 7.0 * (-1.0_f64).exp() / 10.0;
        assert_abs_diff_eq!(e, expected, epsilon = 1e-6);
    }
}

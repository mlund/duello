//! One structure, coarse-grained and titrated on its own, for per-molecule analysis.
//! Titration is incremental so a single-threaded front-end can stay responsive.
//! Kinds and parameters come from the same topology step as a scan; pKa overrides,
//! chain filters and hydrophobic scaling aren't offered here.

use crate::archive::CgBody;
use crate::system::{cg_policy, Loaded, MERGE_TOLERANCE};
use anyhow::{Context, Result};
use cgkitten::forcefield::{self, HydrophobicScaling};
use cgkitten::{Bead, ChargeCalc, ChargeResult, Multipole, Titration};

/// What to build and under which conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct MoleculeSpec {
    /// Force field model name, e.g. "calvados3".
    pub model: String,
    pub single_bead: bool,
    pub ph: f64,
    /// mol/L.
    pub ionic_strength: f64,
    /// K.
    pub temperature: f64,
    /// Monte Carlo sweeps; 0 = Henderson–Hasselbalch.
    pub mc_sweeps: usize,
}

impl Default for MoleculeSpec {
    fn default() -> Self {
        Self {
            model: "calvados3".to_string(),
            single_bead: true,
            ph: 7.0,
            ionic_strength: 0.1,
            temperature: 298.15,
            mc_sweeps: 0,
        }
    }
}

impl MoleculeSpec {
    fn charge_calc(&self) -> ChargeCalc {
        ChargeCalc {
            ph: self.ph,
            ionic_strength: self.ionic_strength,
            temperature: self.temperature,
            mc: self.mc_sweeps,
            ..Default::default()
        }
    }
}

/// A titrated molecule.
#[derive(Clone, Debug)]
pub struct Molecule {
    /// As a scan would see it: merged kind charges, mass centre at origin.
    pub body: CgBody,
    /// One per bead, in `body` order.
    pub beads: Vec<BeadRow>,
    pub multipole: Multipole,
}

/// Per-bead properties for tables.
#[derive(Clone, Debug, PartialEq)]
pub struct BeadRow {
    pub chain: String,
    pub residue: i32,
    /// Atom kind name.
    pub kind: String,
    /// Titrated mean charge, before kinds merge it with similar beads.
    pub charge: f64,
    /// `None` for beads that don't titrate.
    pub charge_variance: Option<f64>,
    /// g/mol.
    pub mass: f64,
    /// Å; `None` where the model has no such parameter.
    pub sigma: Option<f64>,
    /// kJ/mol; models without ε get the same 2.479 fallback as a scan.
    pub epsilon: Option<f64>,
    /// Ashbaugh–Hatch hydrophobicity.
    pub lambda: Option<f64>,
}

/// A titration in progress; see [`Titrating::advance`].
pub struct Titrating {
    beads: Vec<Bead>,
    titration: Titration,
    ff: Option<Box<dyn forcefield::ForceField>>,
}

/// Outcome of [`Titrating::advance`]: a type, not a flag, so a finished titration
/// can't be advanced again.
pub enum Step {
    Running(Titrating),
    Done(Box<Molecule>),
}

impl Titrating {
    /// Coarse-grains `pdb` (PDB or mmCIF); `seed` makes MC runs reproducible.
    pub fn start(pdb: &[u8], spec: MoleculeSpec, seed: u64) -> Result<Self> {
        let ff = forcefield::from_name(&spec.model, HydrophobicScaling::NoScale)
            .map_err(anyhow::Error::msg)?;
        let beads = coarse_grain(pdb, &spec)?;
        let titration = spec.charge_calc().start(&beads, seed);
        Ok(Self {
            beads,
            titration,
            ff,
        })
    }

    /// Runs at most `max_sweeps` more sweeps.
    pub fn advance(mut self, max_sweeps: usize) -> Result<Step> {
        if !self.titration.advance(max_sweeps) {
            return Ok(Step::Running(self));
        }
        let result = self.titration.finish();
        let molecule = build(&self.beads, &result, self.ff.as_deref())?;
        Ok(Step::Done(Box::new(molecule)))
    }

    /// Fraction completed, 0–1.
    pub fn progress(&self) -> f64 {
        self.titration.progress()
    }
}

/// One point of a titration curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvePoint {
    pub ph: f64,
    /// Net charge ⟨Z⟩.
    pub charge: f64,
    /// ⟨Z²⟩ − ⟨Z⟩².
    pub capacitance: f64,
}

/// Henderson–Hasselbalch net charge over `ph`; fast enough for a whole curve,
/// which MC is not. `spec.ph` and `spec.mc_sweeps` are ignored.
pub fn titration_curve(pdb: &[u8], spec: &MoleculeSpec, ph: &[f64]) -> Result<Vec<CurvePoint>> {
    let beads = coarse_grain(pdb, spec)?;
    let point = |&ph| {
        let at_ph = MoleculeSpec {
            ph,
            mc_sweeps: 0,
            ..spec.clone()
        };
        let multipole = at_ph.charge_calc().start(&beads, 0).finish().multipole;
        CurvePoint {
            ph,
            charge: multipole.charge,
            capacitance: multipole.capacitance(),
        }
    };
    Ok(ph.iter().map(point).collect())
}

fn coarse_grain(pdb: &[u8], spec: &MoleculeSpec) -> Result<Vec<Bead>> {
    let beads = cgkitten::coarse_grain_bytes(pdb, cg_policy(spec.single_bead));
    anyhow::ensure!(!beads.is_empty(), "no residues to coarse-grain");
    Ok(beads)
}

/// Goes through the scan's YAML → faunus topology route, so σ, ε, λ and merged
/// charges are the ones a scan of this structure would use.
fn build(
    beads: &[Bead],
    result: &ChargeResult,
    ff: Option<&dyn forcefield::ForceField>,
) -> Result<Molecule> {
    let charged = result.apply(beads);
    let loaded = Loaded::from_generated(cg_system(&charged, ff)?)?;
    let structure = loaded.structures.first().context("no structure built")?;
    // Rows pair cgkitten beads with structure beads by index.
    anyhow::ensure!(
        structure.atom_ids.len() == charged.len(),
        "{} beads became {} in the topology",
        charged.len(),
        structure.atom_ids.len()
    );
    let kinds = loaded.topology.atomkinds();
    Ok(Molecule {
        body: CgBody::new(structure, kinds),
        beads: bead_rows(&charged, result, structure, kinds),
        multipole: result.multipole.clone(),
    })
}

/// The topology step of `cgkitten::coarse_grain_jointly`, which offers no way to
/// supply already-titrated beads.
fn cg_system(
    charged: &[Bead],
    ff: Option<&dyn forcefield::ForceField>,
) -> Result<cgkitten::CgSystem> {
    let topo = cgkitten::topology::Topology::new(charged, MERGE_TOLERANCE);
    let multi_bead = charged
        .iter()
        .any(|b| b.bead_type == cgkitten::BeadType::Virtual);
    let xyz = cgkitten::format_xyz(charged, topo.bead_names(), "generated by cgkitten")
        .map_err(anyhow::Error::msg)?;
    Ok(cgkitten::CgSystem {
        xyz: vec![xyz],
        topology: cgkitten::format_topology(&topo, ff, multi_bead),
    })
}

fn bead_rows(
    beads: &[Bead],
    result: &ChargeResult,
    structure: &crate::structure::Structure,
    kinds: &[faunus::topology::AtomKind],
) -> Vec<BeadRow> {
    beads
        .iter()
        .zip(&structure.atom_ids)
        .enumerate()
        .map(|(i, (bead, &id))| {
            let kind = &kinds[id];
            BeadRow {
                chain: bead.chain_id.clone(),
                residue: bead.res_seq,
                kind: kind.name().to_string(),
                charge: result.charges[i],
                charge_variance: bead
                    .bead_type
                    .is_titratable()
                    .then(|| result.charge_variance[i]),
                mass: structure.masses[i],
                sigma: kind.sigma(),
                epsilon: kind.epsilon(),
                lambda: kind.lambda(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{CgOptions, PairSpec, PairSystem, Source};
    use faunus::interatomic::coulomb::{Medium, Salt};
    use std::path::PathBuf;

    fn cif_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/calvados3/2CDS.cif")
    }

    fn cif() -> Vec<u8> {
        std::fs::read(cif_path()).unwrap()
    }

    fn finish(titrating: Titrating) -> Molecule {
        match titrating.advance(usize::MAX).unwrap() {
            Step::Done(molecule) => *molecule,
            Step::Running(_) => panic!("advance(usize::MAX) must finish"),
        }
    }

    fn hh_molecule() -> Molecule {
        finish(Titrating::start(&cif(), MoleculeSpec::default(), 1).unwrap())
    }

    #[test]
    fn hh_molecule_matches_a_scan() {
        let molecule = hh_molecule();
        let scan = PairSystem::load(PairSpec {
            a: Source::File(&cif_path()),
            b: None,
            top: None,
            cg: CgOptions::default(),
            medium: &Medium::salt_water(298.15, Salt::SodiumChloride, 0.1),
        })
        .unwrap();
        assert_eq!(molecule.beads.len(), scan.ref_a.pos.len());
        assert_eq!(molecule.body.positions.len(), molecule.beads.len());

        let z: f64 = molecule.beads.iter().map(|b| b.charge).sum();
        assert!((z - molecule.multipole.charge).abs() < 1e-9);

        // CALVADOS 3 lysine (cgkitten's table).
        let lys = molecule.beads.iter().find(|b| b.kind == "LYS").unwrap();
        assert_eq!(lys.sigma, Some(6.36));
        assert!((lys.lambda.unwrap() - 0.13806).abs() < 1e-4);
        assert!(molecule
            .beads
            .iter()
            .all(|b| b.sigma.is_some() && b.lambda.is_some()));
    }

    /// Fixed beads have no fluctuation to report; a zero would read as "measured, none".
    #[test]
    fn only_titratable_beads_have_charge_variance() {
        let molecule = hh_molecule();
        let variance = |kind: &str| {
            let bead = molecule.beads.iter().find(|b| b.kind == kind).unwrap();
            bead.charge_variance
        };
        assert_eq!(variance("ALA"), None);
        assert_eq!(variance("GLY"), None);
        // pKa 6.54 at pH 7: q ≈ 0.26, so q(1 − q) ≈ 0.19.
        assert!(variance("HIS").is_some_and(|v| v > 0.1));
        assert!(variance("LYS").is_some());
    }

    /// Protons leave as pH rises; pH 7 must agree with a single HH titration.
    #[test]
    fn titration_curve_falls_with_ph() {
        let ph = [3.0, 5.0, 7.0, 9.0, 11.0];
        let curve = titration_curve(&cif(), &MoleculeSpec::default(), &ph).unwrap();
        assert_eq!(curve.iter().map(|p| p.ph).collect::<Vec<_>>(), ph);
        assert!(curve.windows(2).all(|w| w[1].charge < w[0].charge));
        assert!(curve.iter().all(|p| p.capacitance > 0.0));
        assert!((curve[2].charge - hh_molecule().multipole.charge).abs() < 1e-12);
    }

    /// The browser advances in frame-sized chunks; that mustn't change the answer.
    #[test]
    fn chunked_mc_matches_one_advance() {
        let spec = MoleculeSpec {
            mc_sweeps: 200,
            ..Default::default()
        };
        let whole = finish(Titrating::start(&cif(), spec.clone(), 7).unwrap());
        let mut step = Step::Running(Titrating::start(&cif(), spec, 7).unwrap());
        let mut progress = vec![];
        let chunked = loop {
            match step {
                Step::Running(t) => {
                    progress.push(t.progress());
                    step = t.advance(30).unwrap();
                }
                Step::Done(m) => break *m,
            }
        };
        assert_eq!(progress.len(), 7, "200 sweeps in chunks of 30");
        assert!(progress.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(chunked.beads, whole.beads);
        assert_ne!(chunked.multipole.charge, hh_molecule().multipole.charge);
    }
}

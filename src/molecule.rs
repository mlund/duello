//! One structure, coarse-grained and titrated on its own, for per-molecule analysis.
//! Titration is incremental so a single-threaded front-end can stay responsive.
//! Kinds and parameters come from the same topology step as a scan; pKa overrides
//! and hydrophobic scaling aren't offered here.

use crate::archive::CgBody;
use crate::system::{cg_policy, keep_chains, Loaded, MERGE_TOLERANCE};
use anyhow::{Context, Result};
use cgkitten::forcefield::{self, HydrophobicScaling};
use cgkitten::{Bead, ChargeCalc, ChargeResult, Multipole, ResidueKey, Titration};
use std::collections::HashSet;

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
    /// Chain IDs to keep; empty keeps all.
    pub chains: Vec<String>,
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
            chains: Vec::new(),
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

/// Chains in file order with their residue counts, for choosing which to keep.
/// Counts what the parser keeps (no water), so ligands count but get no bead.
pub fn chains(contents: &[u8]) -> Vec<(String, usize)> {
    let mut seen = HashSet::new();
    let mut found: Vec<(String, usize)> = Vec::new();
    for atom in cgkitten::atoms_bytes(contents) {
        let residue = ResidueKey {
            chain_id: atom.chain_id,
            res_seq: atom.res_seq,
            i_code: atom.i_code,
        };
        if seen.contains(&residue) {
            continue;
        }
        match found.iter_mut().find(|(id, _)| *id == residue.chain_id) {
            Some((_, count)) => *count += 1,
            None => found.push((residue.chain_id.clone(), 1)),
        }
        seen.insert(residue);
    }
    found
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
    /// ⟨|μ|⟩, e·Å.
    pub dipole: f64,
    /// ⟨‖Θ‖⟩, e·Å².
    pub quadrupole: f64,
}

impl CurvePoint {
    fn new(ph: f64, multipole: &Multipole) -> Self {
        Self {
            ph,
            charge: multipole.charge,
            capacitance: multipole.capacitance(),
            dipole: multipole.dipole,
            quadrupole: multipole.quadrupole,
        }
    }
}

/// Henderson–Hasselbalch points per [`Curve::advance_ideal`]: each samples site
/// states (~0.5 ms for 2CDS), so a frame takes a few.
const IDEAL_PER_ADVANCE: usize = 4;

/// MC and ideal titration curves computed one pH at a time, so they fill in over
/// many frames; see [`Curve::advance`]. `spec.ph` is ignored.
pub struct Curve {
    beads: Vec<Bead>,
    spec: MoleculeSpec,
    ph: Vec<f64>,
    seed: u64,
    /// At `ph[points.len()]`.
    running: Option<Titration>,
    points: Vec<CurvePoint>,
    /// Henderson–Hasselbalch for the same beads, over `ideal_ph`.
    ideal: Vec<CurvePoint>,
    ideal_ph: Vec<f64>,
}

impl Curve {
    /// MC over `ph`, HH over `ideal_ph`.
    pub fn start(
        pdb: &[u8],
        spec: &MoleculeSpec,
        ph: &[f64],
        ideal_ph: &[f64],
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            beads: coarse_grain(pdb, spec)?,
            spec: spec.clone(),
            ph: ph.to_vec(),
            seed,
            running: None,
            points: Vec::new(),
            ideal: Vec::new(),
            ideal_ph: ideal_ph.to_vec(),
        })
    }

    /// A few more ideal points; cheap, so callers run it every frame.
    pub fn advance_ideal(&mut self) {
        let next = self.ideal.len();
        let end = (next + IDEAL_PER_ADVANCE).min(self.ideal_ph.len());
        for &ph in &self.ideal_ph[next..end] {
            let result = titrate(&self.beads, &self.spec, ph, 0, self.seed).finish();
            self.ideal.push(CurvePoint::new(ph, &result.multipole));
        }
    }

    /// At most `max_sweeps` more MC sweeps, across points; true once both curves
    /// are done.
    pub fn advance(&mut self, max_sweeps: usize) -> bool {
        let mut left = max_sweeps;
        while let Some(&ph) = self.ph.get(self.points.len()) {
            let titration = self.running.get_or_insert_with(|| {
                titrate(&self.beads, &self.spec, ph, self.spec.mc_sweeps, self.seed)
            });
            // A titration runs exactly `mc_sweeps`, so its progress counts them.
            let remaining = ((1.0 - titration.progress()) * self.spec.mc_sweeps as f64).round();
            if !titration.advance(left) {
                return false;
            }
            left = left.saturating_sub(remaining as usize);
            let result = self.running.take().unwrap().finish();
            self.points.push(CurvePoint::new(ph, &result.multipole));
            if left == 0 {
                break;
            }
        }
        self.is_done()
    }

    /// Finished points, in pH order.
    pub fn points(&self) -> &[CurvePoint] {
        &self.points
    }

    /// Fraction completed, 0–1.
    pub fn progress(&self) -> f64 {
        if self.ph.is_empty() {
            return 1.0;
        }
        let current = self.running.as_ref().map_or(0.0, Titration::progress);
        (self.points.len() as f64 + current) / self.ph.len() as f64
    }

    pub fn is_done(&self) -> bool {
        self.points.len() == self.ph.len() && self.ideal.len() == self.ideal_ph.len()
    }

    /// Finished Henderson–Hasselbalch points, in pH order.
    pub fn ideal(&self) -> &[CurvePoint] {
        &self.ideal
    }
}

fn titrate(beads: &[Bead], spec: &MoleculeSpec, ph: f64, mc_sweeps: usize, seed: u64) -> Titration {
    let at = MoleculeSpec {
        ph,
        mc_sweeps,
        ..spec.clone()
    };
    at.charge_calc().start(beads, seed)
}

fn coarse_grain(pdb: &[u8], spec: &MoleculeSpec) -> Result<Vec<Bead>> {
    let beads = cgkitten::coarse_grain_bytes(pdb, cg_policy(spec.single_bead));
    keep_chains(beads, &spec.chains)
}

/// Goes through the scan's YAML → faunus topology route, so σ, ε, λ and merged
/// charges are the ones a scan of this structure would use.
fn build(
    beads: &[Bead],
    result: &ChargeResult,
    ff: Option<&dyn forcefield::ForceField>,
) -> Result<Molecule> {
    let charged = result.apply(beads);
    let loaded = Loaded::from_generated(
        cgkitten::assemble_system(std::slice::from_ref(&charged), ff, MERGE_TOLERANCE)
            .map_err(anyhow::Error::msg)?,
    )?;
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
    fn chains_are_listed_in_file_order_with_residue_counts() {
        let found = chains(&crate::web_api::tests::two_chains());
        assert_eq!(found, [("A".to_string(), 3), ("B".to_string(), 2)]);
    }

    #[test]
    fn only_the_chosen_chains_are_coarse_grained() {
        let spec = MoleculeSpec {
            chains: vec!["B".into()],
            ..Default::default()
        };
        let molecule =
            finish(Titrating::start(&crate::web_api::tests::two_chains(), spec, 1).unwrap());
        let chains: Vec<_> = molecule.beads.iter().map(|b| b.chain.as_str()).collect();
        assert_eq!(chains, ["B"; 4]);
    }

    #[test]
    fn hh_molecule_matches_a_scan() {
        let molecule = hh_molecule();
        let scan = PairSystem::load(PairSpec {
            a: Source::File(&cif_path()),
            b: None,
            top: None,
            chains: [&[], &[]],
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
        let mut curve = Curve::start(&cif(), &MoleculeSpec::default(), &[], &ph, 0).unwrap();
        assert!(curve.ideal().is_empty(), "start leaves the work to advance");
        curve.advance_ideal();
        assert_eq!(curve.ideal().len(), 4, "a few points a frame");
        assert!(!curve.advance(0));
        curve.advance_ideal();
        assert!(curve.advance(0), "ideal points cost no sweeps");
        let curve = curve.ideal();
        assert_eq!(curve.iter().map(|p| p.ph).collect::<Vec<_>>(), ph);
        assert!(curve.windows(2).all(|w| w[1].charge < w[0].charge));
        assert!(curve.iter().all(|p| p.capacitance > 0.0));
        let hh = hh_molecule().multipole;
        assert!((curve[2].charge - hh.charge).abs() < 1e-12);
        // Sampled moments, other seed: equal within sampling noise.
        assert!((curve[2].dipole / hh.dipole - 1.0).abs() < 0.05);
    }

    /// The browser fills the MC curve a few sweeps a frame, a point at a time.
    #[test]
    fn mc_curve_fills_point_by_point() {
        let spec = MoleculeSpec {
            mc_sweeps: 100,
            ..Default::default()
        };
        let mut curve = Curve::start(&cif(), &spec, &[4.0, 7.0, 10.0], &[], 3).unwrap();
        let (mut counts, mut progress) = (vec![], vec![]);
        while !curve.advance(40) {
            counts.push(curve.points().len());
            progress.push(curve.progress());
        }
        assert_eq!(curve.points().len(), 3);
        assert!(counts.windows(2).all(|w| w[0] <= w[1]) && counts.contains(&1));
        assert!(progress.windows(2).all(|w| w[0] < w[1]));
        let points = curve.points();
        assert!(points.windows(2).all(|w| w[1].charge < w[0].charge));
        // A frame's budget carries past a finished point: 150 = 100 + 50 sweeps.
        let mut carried = Curve::start(&cif(), &spec, &[4.0, 7.0, 10.0], &[], 3).unwrap();
        assert!(!carried.advance(150));
        assert_eq!(carried.points().len(), 1);
        assert!((carried.progress() - 1.5 / 3.0).abs() < 1e-12);
        assert!(points.iter().all(|p| p.dipole > 0.0 && p.quadrupole > 0.0));
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

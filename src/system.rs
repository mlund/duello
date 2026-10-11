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

//! Everything a scan needs before the first energy evaluation, behind one call so the
//! CLI and web front-ends can't drift apart on loading, coarse-graining, or pairing.

use crate::energy::PairMatrix;
use crate::structure::Structure;
use anyhow::{Context, Result};
use faunus::energy::NonbondedMatrix;
use faunus::interatomic::coulomb::Medium;
use faunus::topology::{self, AtomKind, Topology};
use std::borrow::Cow;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};

/// Coarse-graining options for PDB/CIF input.
#[derive(Clone, Debug)]
pub struct CgOptions {
    /// pH for charge calculation (default 7.0).
    pub ph: f64,
    /// Force field model name (default "calvados3").
    pub model: String,
    /// Use single-bead CG policy (default true).
    pub single_bead: bool,
    /// Hydrophobic scaling (default: no scaling).
    pub scale_hydrophobic: cgkitten::forcefield::HydrophobicScaling,
    /// Chain IDs to keep (empty = all chains).
    pub chains: Vec<String>,
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            ph: 7.0,
            model: "calvados3".to_string(),
            single_bead: true,
            scale_hydrophobic: cgkitten::forcefield::HydrophobicScaling::NoScale,
            chains: Vec::new(),
        }
    }
}

/// Where a molecule comes from.
pub enum Source<'a> {
    /// XYZ (needs a topology file) or PDB/mmCIF, detected by extension.
    File(&'a Path),
    /// PDB contents.
    PdbBytes(&'a [u8]),
}

/// What to load.
pub struct PairSpec<'a> {
    pub a: Source<'a>,
    /// `None` reuses A: a homodimer then shares one titration outcome, which an
    /// independent second coarse-graining may not reproduce (gh #39).
    pub b: Option<Source<'a>>,
    /// Topology for XYZ input.
    pub top: Option<&'a Path>,
    pub cg: CgOptions,
    pub medium: &'a Medium,
}

/// Two rigid bodies sharing one atom-kind table, plus their pair energy.
pub struct PairSystem {
    pub ref_a: Structure,
    /// Molecule B with atom ids in `topology`'s table.
    pub ref_b: Structure,
    /// A's atom kinds with B's distinct kinds appended.
    pub topology: Topology,
    pub pair_matrix: PairMatrix,
    /// Coarse-graining output for A and B, if any. Nothing is written to disk here;
    /// persisting it is the caller's choice.
    pub generated: [Option<cgkitten::CgOutput>; 2],
}

impl PairSystem {
    pub fn load(spec: PairSpec) -> Result<Self> {
        let load = |source| Molecule::load(source, spec.top, &spec.cg);
        let Molecule {
            structure: ref_a,
            topology,
            origin,
        } = load(&spec.a)?;
        let (ref_b, topology, generated_b) = match &spec.b {
            None => (ref_a.clone(), topology, None),
            Some(b) => merge_b(&topology, load(b)?),
        };
        // Pair potentials come from A's input; B only contributes atom kinds.
        let nonbonded = origin.nonbonded(&topology, spec.medium)?;
        let pair_matrix = PairMatrix::new(nonbonded, topology.atomkinds(), spec.medium);
        Ok(Self {
            ref_a,
            ref_b,
            topology,
            pair_matrix,
            generated: [origin.into_generated(), generated_b],
        })
    }
}

/// Where a molecule's topology came from; decides how its pair potentials are read.
enum TopologyOrigin {
    /// User file: may `include:` force-field files relative to itself.
    File(PathBuf),
    /// Coarse-grained in memory; self-contained.
    Generated(cgkitten::CgOutput),
}

impl TopologyOrigin {
    fn nonbonded(&self, topology: &Topology, medium: &Medium) -> Result<NonbondedMatrix> {
        let medium = Some(medium.clone());
        match self {
            Self::File(path) => NonbondedMatrix::from_file(path, topology, medium),
            Self::Generated(cg) => NonbondedMatrix::from_str(&cg.topology, topology, medium),
        }
    }

    fn into_generated(self) -> Option<cgkitten::CgOutput> {
        match self {
            Self::Generated(cg) => Some(cg),
            Self::File(_) => None,
        }
    }
}

struct Molecule {
    structure: Structure,
    topology: Topology,
    origin: TopologyOrigin,
}

impl Molecule {
    fn load(source: &Source, top: Option<&Path>, cg: &CgOptions) -> Result<Self> {
        let (bytes, is_pdb) = match source {
            Source::PdbBytes(bytes) => (Cow::Borrowed(*bytes), true),
            Source::File(path) => match coarse_grain_format(path) {
                Some(is_pdb) => (Cow::Owned(read(path)?), is_pdb),
                None => return Self::from_xyz_file(path, top, cg),
            },
        };
        Self::from_generated(coarse_grain(&bytes, is_pdb, cg)?)
    }

    fn from_xyz_file(xyz: &Path, top: Option<&Path>, cg: &CgOptions) -> Result<Self> {
        anyhow::ensure!(
            cg.chains.is_empty(),
            "--chain is only supported for PDB/CIF input, not XYZ"
        );
        let top = top.context("--top is required when input is an XYZ file")?;
        let topology = finalized(Topology::from_file_partial(top)?, top)?;
        let structure = Structure::from_xyz(&xyz.to_path_buf(), topology.atomkinds())?;
        Ok(Self {
            structure,
            topology,
            origin: TopologyOrigin::File(top.to_path_buf()),
        })
    }

    fn from_generated(cg: cgkitten::CgOutput) -> Result<Self> {
        // Self-contained, so there is no file to resolve relative paths against.
        let topology = finalized(Topology::from_str_partial(&cg.topology)?, Path::new("."))?;
        let structure = Structure::from_xyz_str(&cg.xyz, topology.atomkinds())?;
        Ok(Self {
            structure,
            topology,
            origin: TopologyOrigin::Generated(cg),
        })
    }
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("Could not open {}", path.display()))
}

fn finalized(mut topology: Topology, base: &Path) -> Result<Topology> {
    topology.finalize_atoms()?;
    topology.finalize_molecules(base)?;
    topology::set_missing_epsilon(topology.atomkinds_mut(), 2.479);
    Ok(topology)
}

/// `Some(is_pdb)` for input to coarse-grain, `None` for XYZ.
fn coarse_grain_format(path: &Path) -> Option<bool> {
    let ext = path.extension()?.to_ascii_lowercase();
    match ext.to_str()? {
        "pdb" => Some(true),
        "cif" | "mmcif" => Some(false),
        _ => None,
    }
}

fn coarse_grain(bytes: &[u8], is_pdb: bool, cg: &CgOptions) -> Result<cgkitten::CgOutput> {
    let policy: &dyn cgkitten::CoarseGrain = if cg.single_bead {
        &cgkitten::SingleBead
    } else {
        &cgkitten::MultiBead
    };
    cgkitten::coarse_grain_to_strings(
        BufReader::new(Cursor::new(bytes)),
        is_pdb,
        &cgkitten::ChargeCalc::new().ph(cg.ph).mc(10000),
        &cg.model,
        policy,
        0.02,
        cg.scale_hydrophobic,
        &cg.chains,
    )
    .map_err(anyhow::Error::msg)
}

/// B's structure with ids remapped into a table merged from A and B. Needed because
/// each molecule may have been coarse-grained into its own id space (gh #39).
fn merge_b(
    topology_a: &Topology,
    b: Molecule,
) -> (Structure, Topology, Option<cgkitten::CgOutput>) {
    let (merged, b_remap) = combine_topologies(topology_a, &b.topology);
    let mut structure_b = b.structure;
    structure_b.remap_atom_ids(&b_remap);
    (structure_b, merged, b.origin.into_generated())
}

/// Merge molecule B's atom kinds into A's, returning the combined topology and a
/// map from B's old atom-kind id to its id in the merged table.
///
/// Kinds with identical names, interaction parameters, and custom atom properties share
/// an id, keeping the `n_types^2` pair table small. Distinct names are preserved because
/// topology YAML may contain name-specific pair overrides.
fn combine_topologies(topology_a: &Topology, topology_b: &Topology) -> (Topology, Vec<usize>) {
    let mut merged = topology_a.clone();
    let mut b_remap = Vec::with_capacity(topology_b.atomkinds().len());

    for kind_b in topology_b.atomkinds() {
        let merged_id = merged
            .atomkinds()
            .iter()
            .position(|kind| same_interaction(kind, kind_b))
            .unwrap_or_else(|| {
                let id = merged.atomkinds().len();
                merged.add_atomkind(kind_b.clone());
                id
            });
        b_remap.push(merged_id);
    }

    // Re-index the combined table without `finalize_atoms`, which rejects duplicate
    // names that can legitimately arise from independent titration outcomes.
    for (id, kind) in merged.atomkinds_mut().iter_mut().enumerate() {
        kind.set_id(id);
    }

    (merged, b_remap)
}

/// True if two atom kinds are interchangeable for scan energy evaluation.
fn same_interaction(a: &AtomKind, b: &AtomKind) -> bool {
    let opt_bits = |x: Option<f64>| x.map(f64::to_bits);
    a.name() == b.name()
        && a.charge().to_bits() == b.charge().to_bits()
        && a.mass().to_bits() == b.mass().to_bits()
        && opt_bits(a.sigma()) == opt_bits(b.sigma())
        && opt_bits(a.epsilon()) == opt_bits(b.epsilon())
        && opt_bits(a.lambda()) == opt_bits(b.lambda())
        && a.custom() == b.custom()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faunus::interatomic::coulomb::Salt;

    fn example(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join(rel)
    }

    /// Tracked examples must load against the current faunus schema.
    #[test]
    fn examples_load() {
        for (xyz, top) in [
            ("cppm/cppm-p00.xyz", "cppm/topology.yaml"),
            ("calvados3/2CDS.xyz", "calvados3/topology.yaml"),
        ] {
            let (xyz, top) = (example(xyz), example(top));
            let system = PairSystem::load(PairSpec {
                a: Source::File(&xyz),
                b: None,
                top: Some(&top),
                cg: CgOptions::default(),
                medium: &Medium::salt_water(298.15, Salt::SodiumChloride, 0.1),
            });
            assert!(
                system.is_ok(),
                "{}: {:#}",
                top.display(),
                system.err().unwrap()
            );
        }
    }

    // `combine_topologies` keeps internal-seam tests: its dedup rules only show up for
    // independently titrated molecules, which can't be set up deterministically from outside.

    fn atoms_only(yaml: &str) -> Topology {
        let mut top = Topology::from_str_partial(yaml).unwrap();
        top.finalize_atoms().unwrap();
        top
    }

    #[test]
    fn combine_topologies_dedups_shared_and_appends_distinct() {
        let top_a = atoms_only(
            "atoms:\n  - {name: GLY, mass: 57.0, charge: 0.0, sigma: 4.5, eps: 0.83}\n  \
             - {name: O1, mass: 0.0, charge: -0.9, sigma: 2.0, eps: 0.83}\n",
        );
        let top_b = atoms_only(
            "atoms:\n  - {name: GLY, mass: 57.0, charge: 0.0, sigma: 4.5, eps: 0.83}\n  \
             - {name: O1, mass: 0.0, charge: -0.4, sigma: 2.0, eps: 0.83}\n",
        );

        let (merged, b_remap) = combine_topologies(&top_a, &top_b);

        assert_eq!(merged.atomkinds().len(), 3);
        for (i, kind) in merged.atomkinds().iter().enumerate() {
            assert_eq!(kind.id(), i);
        }
        assert_eq!(b_remap, vec![0, 2]);
        assert_eq!(
            merged.atomkinds()[2].charge().to_bits(),
            (-0.4f64).to_bits()
        );
    }

    #[test]
    fn combine_identical_topologies_is_a_noop() {
        let top = atoms_only(
            "atoms:\n  - {name: GLY, mass: 57.0, charge: 0.0, sigma: 4.5, eps: 0.83}\n  \
             - {name: ASP, mass: 115.0, charge: -1.0, sigma: 5.6, eps: 0.83}\n",
        );
        let (merged, b_remap) = combine_topologies(&top, &top);
        assert_eq!(merged.atomkinds().len(), 2);
        assert_eq!(b_remap, vec![0, 1]);
    }

    #[test]
    fn combine_topologies_preserves_distinct_names_with_same_parameters() {
        let top_a =
            atoms_only("atoms:\n  - {name: A, mass: 10.0, charge: 0.0, sigma: 2.0, eps: 0.5}\n");
        let top_b =
            atoms_only("atoms:\n  - {name: B, mass: 10.0, charge: 0.0, sigma: 2.0, eps: 0.5}\n");

        let (merged, b_remap) = combine_topologies(&top_a, &top_b);

        assert_eq!(merged.atomkinds().len(), 2);
        assert_eq!(merged.atomkinds()[0].name(), "A");
        assert_eq!(merged.atomkinds()[1].name(), "B");
        assert_eq!(b_remap, vec![1]);
    }

    #[test]
    fn combine_topologies_preserves_distinct_custom_properties() {
        let top_a = atoms_only(
            "atoms:\n  - {name: A, mass: 10.0, charge: 1.0, sigma: 2.0, eps: 0.5, custom: {alpha: 0.0}}\n",
        );
        let top_b = atoms_only(
            "atoms:\n  - {name: A, mass: 10.0, charge: 1.0, sigma: 2.0, eps: 0.5, custom: {alpha: 1.0}}\n",
        );

        let (merged, b_remap) = combine_topologies(&top_a, &top_b);

        assert_eq!(merged.atomkinds().len(), 2);
        assert_eq!(
            merged.atomkinds()[0].custom(),
            top_a.atomkinds()[0].custom()
        );
        assert_eq!(
            merged.atomkinds()[1].custom(),
            top_b.atomkinds()[0].custom()
        );
        assert_eq!(b_remap, vec![1]);
    }
}

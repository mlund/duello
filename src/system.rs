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
use faunus::topology::{self, Topology};
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
    pub ref_b: Structure,
    /// Atom kinds of both molecules.
    pub topology: Topology,
    pub pair_matrix: PairMatrix,
    /// Coarse-graining output, if any. Nothing is written to disk here;
    /// persisting it is the caller's choice.
    pub generated: Option<cgkitten::CgSystem>,
}

impl PairSystem {
    pub fn load(spec: PairSpec) -> Result<Self> {
        let sources: Vec<&Source> = std::iter::once(&spec.a).chain(spec.b.as_ref()).collect();
        let Loaded {
            structures,
            topology,
            origin,
        } = Loaded::new(&sources, spec.top, &spec.cg)?;
        let mut structures = structures.into_iter();
        let ref_a = structures.next().context("no structure loaded")?;
        let ref_b = structures.next().unwrap_or_else(|| ref_a.clone());
        let nonbonded = origin.nonbonded(&topology, spec.medium)?;
        let pair_matrix = PairMatrix::new(nonbonded, topology.atomkinds(), spec.medium);
        Ok(Self {
            ref_a,
            ref_b,
            topology,
            pair_matrix,
            generated: origin.into_generated(),
        })
    }
}

impl Source<'_> {
    /// `Some(is_pdb)` for input to coarse-grain, `None` for XYZ.
    fn atomistic_format(&self) -> Option<bool> {
        match self {
            Source::PdbBytes(_) => Some(true),
            Source::File(path) => coarse_grain_format(path),
        }
    }

    fn bytes(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            Source::PdbBytes(bytes) => Ok(Cow::Borrowed(bytes)),
            Source::File(path) => read(path).map(Cow::Owned),
        }
    }
}

/// Where the topology came from; decides how pair potentials are read.
enum TopologyOrigin {
    /// User file: may `include:` force-field files relative to itself.
    File(PathBuf),
    /// Coarse-grained in memory; self-contained.
    Generated(cgkitten::CgSystem),
}

impl TopologyOrigin {
    fn nonbonded(&self, topology: &Topology, medium: &Medium) -> Result<NonbondedMatrix> {
        let medium = Some(medium.clone());
        match self {
            Self::File(path) => NonbondedMatrix::from_file(path, topology, medium),
            Self::Generated(cg) => NonbondedMatrix::from_str(&cg.topology, topology, medium),
        }
    }

    fn into_generated(self) -> Option<cgkitten::CgSystem> {
        match self {
            Self::Generated(cg) => Some(cg),
            Self::File(_) => None,
        }
    }
}

/// Structures in input order, all indexing one topology. A single topology is what
/// gives every cross pair a potential: one per molecule would cover only its own kinds.
struct Loaded {
    structures: Vec<Structure>,
    topology: Topology,
    origin: TopologyOrigin,
}

impl Loaded {
    /// Checks for mixed input before reading anything.
    fn new(sources: &[&Source], top: Option<&Path>, cg: &CgOptions) -> Result<Self> {
        let formats: Vec<_> = sources.iter().map(|s| s.atomistic_format()).collect();
        if formats.iter().all(Option::is_none) {
            // PDB bytes always have a format, so every source here is an XYZ file.
            let paths: Vec<&Path> = sources
                .iter()
                .filter_map(|s| match s {
                    Source::File(path) => Some(*path),
                    Source::PdbBytes(_) => None,
                })
                .collect();
            return Self::from_xyz_files(&paths, top, cg);
        }
        anyhow::ensure!(
            formats.iter().all(Option::is_some),
            "cannot mix XYZ and PDB/CIF input: they would not share a topology"
        );
        let inputs = sources
            .iter()
            .zip(formats.into_iter().flatten())
            .map(|(source, is_pdb)| Ok((source.bytes()?, is_pdb)))
            .collect::<Result<Vec<_>>>()?;
        Self::from_generated(coarse_grain(&inputs, cg)?)
    }

    fn from_xyz_files(xyz: &[&Path], top: Option<&Path>, cg: &CgOptions) -> Result<Self> {
        anyhow::ensure!(
            cg.chains.is_empty(),
            "--chain is only supported for PDB/CIF input, not XYZ"
        );
        let top = top.context("--top is required when input is an XYZ file")?;
        let topology = finalized(Topology::from_file_partial(top)?, top)?;
        let structures = xyz
            .iter()
            .map(|path| Structure::from_xyz(&path.to_path_buf(), topology.atomkinds()))
            .collect::<Result<_>>()?;
        Ok(Self {
            structures,
            topology,
            origin: TopologyOrigin::File(top.to_path_buf()),
        })
    }

    fn from_generated(cg: cgkitten::CgSystem) -> Result<Self> {
        // Self-contained, so there is no file to resolve relative paths against.
        let topology = finalized(Topology::from_str_partial(&cg.topology)?, Path::new("."))?;
        let structures = cg
            .xyz
            .iter()
            .map(|xyz| Structure::from_xyz_str(xyz, topology.atomkinds()))
            .collect::<Result<_>>()?;
        Ok(Self {
            structures,
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

fn coarse_grain(inputs: &[(Cow<[u8]>, bool)], cg: &CgOptions) -> Result<cgkitten::CgSystem> {
    let policy: &dyn cgkitten::CoarseGrain = if cg.single_bead {
        &cgkitten::SingleBead
    } else {
        &cgkitten::MultiBead
    };
    let readers = inputs
        .iter()
        .map(|(bytes, is_pdb)| (BufReader::new(Cursor::new(bytes.as_ref())), *is_pdb));
    cgkitten::coarse_grain_jointly(
        readers,
        &cgkitten::ChargeCalc::new().ph(cg.ph).mc(10000),
        &cg.model,
        policy,
        0.02,
        cg.scale_hydrophobic,
        &cg.chains,
    )
    .map_err(anyhow::Error::msg)
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

    /// Mixed input would need two topologies, which can't give cross pairs a potential.
    #[test]
    fn mixed_xyz_and_atomistic_input_is_rejected() {
        let (xyz, top) = (example("cppm/cppm-p00.xyz"), example("cppm/topology.yaml"));
        let result = PairSystem::load(PairSpec {
            a: Source::File(&xyz),
            b: Some(Source::PdbBytes(b"")),
            top: Some(&top),
            cg: CgOptions::default(),
            medium: &Medium::salt_water(298.15, Salt::SodiumChloride, 0.1),
        });
        assert!(result.is_err_and(|e| e.to_string().contains("cannot mix")));
    }
}
